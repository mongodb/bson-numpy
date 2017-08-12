import datetime

try:
    import pandas as pd
    from pandas.testing import assert_index_equal
except ImportError:
    pd = None

import numpy as np

import bsonnumpy
from test import client_context, unittest


def to_dataframe(seq, dtype, n):
    data = bsonnumpy.sequence_to_ndarray(seq, dtype, n)
    if '_id' in dtype.fields:
        return pd.DataFrame(data, index=data['_id'])
    else:
        return pd.DataFrame(data)


@unittest.skipUnless(pd, "requires pandas")
class TestSequence2Pandas(unittest.TestCase):
    def dataframe_test(self, docs, dtype):
        db = client_context.client.bsonnumpy_test
        coll = db.coll
        coll.delete_many({})
        coll.insert_many(docs)
        return to_dataframe(coll.find_raw().sort('_id'), dtype, coll.count())

    @client_context.require_connected
    def test_one_value(self):
        docs = [{"_id": i} for i in range(10, 20)]
        df = self.dataframe_test(docs, np.dtype([('_id', np.int32)]))
        self.assertEqual(df.shape, (10, 1))
        self.assertEqual(df['_id'].name, '_id')
        self.assertEqual(df['_id'].dtype, np.int32)
        np.testing.assert_array_equal(df['_id'].as_matrix(), np.arange(10, 20))

    @client_context.require_connected
    def test_multi_values(self):
        now = datetime.datetime.now()
        docs = [{'d': 2.0,
                 'int32': 1,
                 'int64': 2 ** 40,
                 'b': False,
                 'dt': now}]

        df = self.dataframe_test(docs,
                                 np.dtype([
                                     ('_id', 'S12'),
                                     ('d', np.double),
                                     ('int32', '<i4'),
                                     ('int64', '<i8'),
                                     ('b', 'b'),
                                     ('dt', '<i8')]))

        self.assertEqual(df.shape, (1, 6))
        assert_index_equal(df.columns,
                           pd.Index(['_id', 'd', 'int32', 'int64', 'b', 'dt']))

        self.assertEqual(df['_id'].dtype, np.dtype('O'))  # TODO why not 'S12'?
        self.assertEqual(df['d'].dtype, np.double)
        self.assertEqual(df['int32'].dtype, np.dtype('<i4'))
        self.assertEqual(df['int64'].dtype, np.dtype('<i8'))
        self.assertEqual(df['b'].dtype, np.dtype('b'))
        self.assertEqual(df['dt'].dtype, np.dtype('<i8'))

if __name__ == '__main__':
    unittest.main()
