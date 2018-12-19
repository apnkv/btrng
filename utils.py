from PIL import Image
from glob import glob

import numpy as np
import rawpy


def numpy_arrays_from_paths(filepaths, grayscale=False):
    frames = []
    for filepath in filepaths:
        img = Image.open(filepath)
        if grayscale:
            img = img.convert('L')
        data = np.array(img).astype(np.uint8)
        frames.append(data)
    return frames


def load_photos_from_folder(folder, fmt='jpg', postprocess=True, grayscale=True):
    images = []

    glob_string_small = f'{folder}/*.{fmt}'
    glob_string_caps = f'{folder}/*.{fmt.upper()}'

    for path in glob(glob_string_small) + glob(glob_string_caps):
        if fmt in ('jpg', 'png', 'heic'):
            image = Image.open(path)
            if grayscale:
                image = image.convert('L')
            image = np.array(image, dtype=np.uint8)
        elif fmt in ('dng', 'cr2'):
            image = rawpy.imread(path)
            if postprocess:
                image = image.postprocess()
                if grayscale:
                    image = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
                image = image.astype(np.uint8)
        else:
            raise ValueError
        images.append(image)

    images = np.array(images)

    return images


def pixel_statistics(frames, n_last_bits=1):
    mb = np.round(np.mean(frames, axis=0))
    sb = np.std(frames, axis=0)
    mlb = np.mean(frames & (2 ** n_last_bits - 1), axis=0)

    mask = (mb > 2) & (mb < 253)

    return mb[mask], sb[mask], mlb[mask]
