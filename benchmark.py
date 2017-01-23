import math
import timeit

import pymongo
import numpy as np
from bson import CodecOptions
from bson.raw_bson import RawBSONDocument

import bsonnumpy

N_DOCUMENTS = 10 * 1000
N_TRIALS = 50

db = None


def _setup():
    global db

    db = pymongo.MongoClient().bsonnumpy_test
    collection = db.collection
    collection.drop()
    collection.insert_many([
        {'_id': i, 'x': i * math.pi}
        for i in range(N_DOCUMENTS)])


def _teardown():
    db.collection.drop()


def traditional_func():
    collection = db.collection
    cursor = collection.find()
    dtype = np.dtype([('_id', np.int64), ('x', np.float64)])
    np.array([(doc['_id'], doc['x']) for doc in cursor], dtype=dtype)


def bson_numpy_func():
    raw_coll = db.get_collection(
            'collection',
            codec_options=CodecOptions(document_class=RawBSONDocument))

    cursor = raw_coll.find()
    dtype = np.dtype([('_id', np.int64), ('x', np.float64)])

    bsonnumpy.sequence_to_ndarray(
        (doc.raw for doc in cursor), dtype, raw_coll.count())

_setup()

for func in traditional_func, bson_numpy_func:
    duration = timeit.timeit(bson_numpy_func, number=N_TRIALS)
    print("%s: %.2f" % (func.__name__, duration / N_TRIALS))

_teardown()
