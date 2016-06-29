import bsonnumpy
import bson
import numpy as np
import random
import sys
import datetime
PY3 = sys.version_info[0] >= 3

document = bson.SON([("0", 99), ("1", 88), ("2", 77)])
utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)

# Integer types
def compare_types(test_dtype, check=None):
    dtype = np.dtype(test_dtype)
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    assert result.dtype == test_dtype
    print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])

scalar_types = [
    np.bool_, np.int_, np.intc, np.intp,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float_, np.float16, np.float32, np.float64,
    np.complex_, np.complex64, np.complex128]

# Complex types
def compare_oid():
    document = bson.SON([("0", bson.ObjectId()), ("1", bson.ObjectId()), ("2", bson.ObjectId())])
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype("<V12")
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])
    assert result.dtype == dtype
    for b in range(3):
        assert str(document[str(b)].binary) == str(result[b])

def compare_string():
    document = bson.SON([("0", "string_0"), ("1", "str1"), ("2", "utf8-2")])
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype("<S8")
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])
    for b in range(3):
        assert document[str(b)] == result[b]

def compare_binary():
    document = bson.SON([("0", bson.Binary("binary_0")), ("1", bson.Binary('bin1')), ("2", bson.Binary('utf8-2'))])
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype("<V15")
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])
    print result
    for b in range(3):
        # WHY WONT U COMPARE RIGHT????
        assert bytes(document[str(b)]) == result[b]

def compare_datetime():
    document = bson.SON([("0", datetime.datetime(1970, 1, 1)),
                         ("1", datetime.datetime(1980, 1, 1)),
                         ("2", datetime.datetime(1990, 1, 1))])
    print document
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype('datetime64[s]')
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])
    print result
    for b in range(3):
        print np.datetime64(document[str(b)]), result[b].tolist()

def compare_null():
    document = bson.SON([("0", None), ("1", None), ("2", None)])
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype('int32')
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])
    for b in range(3):
        assert not result[b]


# for t in scalar_types:
#     compare_types(t)
# compare_oid()
# compare_string()
# compare_binary()
# compare_datetime()
compare_null()
