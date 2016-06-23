import bsonnumpy
import bson

document = {"x": [1,2,3]}
utf8 = bson._dict_to_bson(document, False, bson.DEFAULT_CODEC_OPTIONS)

bsonnumpy.bson_to_ndarray(utf8)
