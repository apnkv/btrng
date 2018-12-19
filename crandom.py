import numpy as np

from bitarray import bitarray
from utils import load_photos_from_folder, numpy_arrays_from_paths


GLOBAL_BRIGHTNESS_LO = 3
GLOBAL_BRIGHTNESS_HI = 255 - GLOBAL_BRIGHTNESS_LO


class CameraNoiseTRNG:
    def __init__(self, calibration_frames=None):
        self.height = None
        self.width = None
        self.mask = None
        self._mean_brightness = None
        self._std_brightness = None
        self._mean_last_bit = None
        self._calibration_frame_count = 0

        self._flip_bits = False
        self._entropy_buffer = bitarray()

        self._brightness_lo = GLOBAL_BRIGHTNESS_LO
        self._brightness_hi = GLOBAL_BRIGHTNESS_HI

        self._calibrated = False
        if calibration_frames:
            self.calibrate(calibration_frames)

    @staticmethod
    def _validate_and_load_frames(frames):
        assert len(frames) > 0

        if type(frames) == str:
            frames = load_photos_from_folder(frames)
        elif type(frames) in (list, tuple) and type(frames[0]) == str:
            frames = numpy_arrays_from_paths(frames, grayscale=True)
        elif type(frames) == np.array:
            pass
        else:
            raise TypeError

        return frames

    def calibrate(self, frames):
        """
        :param frames: can be a numpy array of 1 or 3 channel images, a folder name, or a list of file paths
        :return: nothing, but internal parameters are calibrated against given images
        """
        frames = self._validate_and_load_frames(frames)

        self.height, self.width = frames.shape[1:]

        self._mean_brightness = np.round(np.mean(frames, axis=0))
        self._std_brightness = np.std(frames, axis=0)
        self._mean_last_bit = np.mean(frames & 1, axis=0)

        conf_interval = 0.03  # TODO: put an accurate formula depending on N

        self.mask = (self._mean_last_bit >= 0.5 - conf_interval) & (self._mean_last_bit <= 0.5 + conf_interval)
        self._calibrated = True

    def capacity(self):
        return len(self._entropy_buffer)

    def feed_entropy(self, frames):
        if not self._calibrated:
            raise Exception('Cannot parse entropy before calibration.')

        frames = self._validate_and_load_frames(frames)
        total_bits_fed = 0

        for i, frame in enumerate(frames):
            l_bits = frame[self.mask]
            # strip over- and undersaturated pixels
            l_bits = l_bits[l_bits <= self._brightness_hi]
            l_bits = l_bits[l_bits >= self._brightness_lo]
            # take only last bits
            l_bits = l_bits & 1

            l_bits = l_bits.reshape(-1)
            length = len(l_bits)
            square_side = int(np.floor(np.sqrt(length)))

            sq = bitarray(l_bits[:square_side * square_side].reshape(square_side, square_side).T.reshape(-1).tolist())
            remainder = bitarray(l_bits[square_side * square_side:].tolist())

            if self._flip_bits:
                sq.invert()
                remainder.invert()

            self._entropy_buffer = sq + remainder + self._entropy_buffer

            total_bits_fed += length
            self._flip_bits = not self._flip_bits

    def mask(self):
        return self.mask

    def mean_brightness(self):
        return self._mean_brightness

    def std_brightness(self):
        return self._mean_last_bit

    def random_bits(self, nbits):
        if nbits > self.capacity():
            raise ValueError('Not enough entropy in the buffer.')

        bits = self._entropy_buffer[:nbits]
        self._entropy_buffer = self._entropy_buffer[nbits:]

        return bits

    def random_number(self, nbits):
        bits = self.random_bits(nbits)
        return int(bits.to01(), base=2)

    def random_numbers(self, size, nbits=32):
        raise NotImplementedError
