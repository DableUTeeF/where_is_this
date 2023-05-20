import numpy as np
import hamming_distance

print(
    hamming_distance.hamming_distance(
        np.array([3]),
        np.array(['001'], dtype=np.ubyte),
        np.array(['101'], dtype=np.ubyte))
)

print(
    hamming_distance.hamming_distance(
        3,
        '011',
        '001'
    )
)
