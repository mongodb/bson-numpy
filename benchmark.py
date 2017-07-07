import math
import sys
import timeit

import pymongo
import numpy as np
from bson import BSON, CodecOptions, Int64
from bson.raw_bson import RawBSONDocument

import bsonnumpy

N_DOCUMENTS = 10 * 1000
N_TRIALS = 50

db = None
raw_bson = None


def _setup():
    global db
    global raw_bson

    db = pymongo.MongoClient().bsonnumpy_test
    collection = db.collection
    collection.drop()
    collection.insert_many([
        {'_id': i, 'x': i * math.pi}
        for i in range(N_DOCUMENTS)])

    # Ignore for now that the first batch defaults to 101 documents.
    raw_bson_docs = [{'_id': i, 'x': float(i)} for i in range(1000)]
    raw_bson = BSON.encode({'ok': 1,
                            'cursor': {
                                'id': Int64(1234),
                                'ns': 'db.collection',
                                'firstBatch': raw_bson_docs}})


def _teardown():
    db.collection.drop()


bench_fns = {}


def bench(name):
    def assign_name(fn):
        bench_fns[name] = fn
        return fn

    return assign_name


@bench('conventional')
def conventional_func():
    collection = db.collection
    cursor = collection.find()
    dtype = np.dtype([('_id', np.int64), ('x', np.float64)])
    np.array([(doc['_id'], doc['x']) for doc in cursor], dtype=dtype)


@bench('bson-numpy')
def bson_numpy_func():
    raw_coll = db.get_collection(
        'collection',
        codec_options=CodecOptions(document_class=RawBSONDocument))

    cursor = raw_coll.find()
    dtype = np.dtype([('_id', np.int64), ('x', np.float64)])
    bsonnumpy.sequence_to_ndarray(
        (doc.raw for doc in cursor), dtype, raw_coll.count())


@bench('bson')
def bson_func():
    for _ in BSON(raw_bson).decode()['cursor']['firstBatch']:
        pass


@bench('raw-bson')
def raw_bson_func():
    options = CodecOptions(document_class=RawBSONDocument)
    for _ in BSON(raw_bson).decode(options)['cursor']['firstBatch']:
        pass


@bench('raw-batches')
def raw_bson_func():
    c = db.collection
    try:
        batches = list(c.find(raw_batches=True))
    except TypeError as exc:
        if "unexpected keyword argument 'raw_batches'" in str(exc):
            print("Wrong PyMongo: no 'raw_batches' feature")
            return
        else:
            raise

    dtype = np.dtype([('_id', np.int64), ('x', np.float64)])
    bsonnumpy.sequence_to_ndarray(batches, dtype, c.count())


_setup()

for name in sys.argv[1:]:
    if name not in bench_fns:
        sys.stderr.write("Unknown function \"%s\"\n" % name)
        sys.stderr.write("Available functions:\n%s\n" % ("\n".join(bench_fns)))
        sys.exit(1)

for name, fn in bench_fns.items():
    if not sys.argv[1:] or name in sys.argv[1:]:
        duration = min(timeit.Timer(fn).repeat(3, N_TRIALS))
        print("%s: %.2f" % (name, duration))

_teardown()
