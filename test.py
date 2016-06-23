import bsonnumpy
import bson
import numpy as np

document = {"0": 99, "1": 88, "2": 77}
#document = {"0": 1, "1": 2, "2": 3}
dtype = np.dtype('int32')
array = np.ndarray([3], dtype='int32')
array[0] = 1
array[1] = 2
array[2] = 3
# dtype = np.dtype(('x', np.float))
utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)

result = bsonnumpy.bson_to_ndarray(utf8, dtype, array)
print "result", result
