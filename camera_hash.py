import hashlib
import sha3
import numpy as np

def random_bits(frames, nbits=64):
    # RETURN: bytes object of length `nbits`/8
#     assert frames.dtype == np.uint8
    single_arr = np.concatenate([frame.flatten() for frame in frames])
    
    s = hashlib.shake_256()
    s.update(single_arr)
    return s.digest(nbits // 8)