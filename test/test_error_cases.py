import bson
import bsonnumpy
import numpy as np

from test import TestToNdarray

class TestSequenceToNdarray(TestToNdarray):

    def test_mismatched_dtype(self):
        docs = [{"x": i, "y": 10 - i} for i in range(10)]
        dtype = np.dtype([('x', np.int32), ('y', np.int32)])
        self.make_mixed_collection_test(docs, dtype)
