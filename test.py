import bsonnumpy
import bson
import numpy as np

document = bson.SON([("0", 99), ("1", 88), ("2", 77)])
utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)

def compare_types(test_dtype):

    dtype = np.dtype(test_dtype)
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    assert result.dtype == test_dtype
    print "result", result, "python type", type(result), "dtype", result.dtype

scalar_types = [
    np.bool_, np.int_, np.intc, np.intp,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float_, np.float16, np.float32, np.float64,
    np.complex_, np.complex64, np.complex128]

for t in scalar_types:
    compare_types(t)
