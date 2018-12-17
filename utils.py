from PIL import Image
import numpy as np

def numpy_arrays_from_paths(filepaths):
    frames = []
    for filepath in filepaths:
        img = Image.open(filepath)
        data = np.array(img).astype(np.uint8)
        frames.append(data)
    return frames
