import numpy as np


def stats():
    pass


def calibrate():
    pass


LAST_BITS = 1
BRIGHTNESS_LO = 2
BRIGHTNESS_HI = 253


def random_bits(frames, nbits=64):
    matrix_size = np.floor(np.sqrt(nbits))
    matrix = np.zeros(matrix_size * matrix_size, dtype=np.uint8)

    frame_even = True
    insertion_pos = 0

    for frame in frames:
        frame = frame.reshape(-1)  # flatten
        frame = frame[BRIGHTNESS_LO <= matrix <= BRIGHTNESS_HI]  # choose pixels inside threshold
        frame &= 2 ** LAST_BITS - 1  # choose last bits

        if frame_even:
            frame = ~frame

        frame_even = not frame_even
