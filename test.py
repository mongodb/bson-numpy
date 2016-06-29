import bsonnumpy
import bson
import numpy as np
import random
import sys
import datetime
PY3 = sys.version_info[0] >= 3


# Integer types
def compare_integer_types(test_dtype, check=None):
    print "dtype=", test_dtype
    document_numpy = bson.SON([("0", test_dtype(99)), ("1", test_dtype(88)), ("2", test_dtype(77))])
    document_python = bson.SON([("0", np.asscalar(test_dtype(99))), ("1", np.asscalar(test_dtype(88))), ("2", np.asscalar(test_dtype(77)))])
    print "test_dtype=", type(document_numpy["0"])
    print "asscalar type=", type(document_python["0"])
    utf8 = bson._dict_to_bson(document_python, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype(test_dtype)
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    assert result.dtype == test_dtype
    for i in range(3):
        print "comparing", document_numpy[str(i)], result[i]
        # assert document_numpy[str(i)] == result[i]


scalar_types = [
    np.bool_, #np.int_, np.intc, np.intp, np.uint64, np.int64,
    np.int8, np.int16, np.int32,
    np.uint8, np.uint16, np.uint32,
    np.float_, np.float16, np.float32, np.float64,
    #np.complex_, np.complex64, np.complex128
]

# Complex types
def compare_oid():
    document = bson.SON([("0", bson.ObjectId()), ("1", bson.ObjectId()), ("2", bson.ObjectId())])
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype("<V12")
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    # print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])
    assert result.dtype == dtype
    for b in range(3):
        assert str(document[str(b)].binary) == str(result[b])

def compare_string():
    document = bson.SON([("0", "string_0"), ("1", "str1"), ("2", "utf8-2")])
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    dtype = np.dtype("<S8")
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    # print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])
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

def compare_document():
    document = bson.SON([("0", bson.SON([("a", 1)])), ("1", bson.SON([("b", 2)])), ("2", bson.SON([("c", 3)]))])
    utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)
    sys.stdout.write("utf8:\t\t")
    for i in utf8[7:19]:
        sys.stdout.write(i.encode('hex'))
        sys.stdout.write('|')
    print
    a1 = bson._dict_to_bson(bson.SON([("a", 1)]), False, bson.DEFAULT_CODEC_OPTIONS)
    sys.stdout.write("a1:\t\t")
    for i in a1:
        sys.stdout.write(i.encode('hex'))
        sys.stdout.write('|')
    print
    # print "DECODED", bson.BSON(utf8[7:19]).decode()
    dtype = np.dtype('<V12')
    result = bsonnumpy.bson_to_ndarray(utf8, dtype)
    # print "result", result, "python type", type(result), "dtype", result.dtype, "type(result[0])", type(result[0])

    sys.stdout.write("result[0]:\t")
    for i in bytes(result[0]):
        sys.stdout.write(i.encode('hex'))
        sys.stdout.write('|')
    print

    for b in range(3):
        doc = bson.SON(bson.BSON(bytes(result[b])).decode())
        assert doc == document[str(b)]



# for t in scalar_types:
#     compare_integer_types(t)
# compare_oid()
# compare_string()
# compare_binary()
# compare_datetime()
# compare_null()
compare_document()
