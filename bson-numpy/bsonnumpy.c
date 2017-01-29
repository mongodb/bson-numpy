#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
/* #include <numpy/arrayobject.h> */
/* #include <numpy/npy_common.h> */
#include <numpy/ndarrayobject.h>

#include "libbson-1.0/bson.h"

static PyObject *BsonNumpyError;

static int
_load_scalar_from_bson(bson_iter_t *bsonit, PyArrayObject *ndarray, long offset,
                       npy_intp *coordinates, int current_depth,
                       PyArray_Descr *dtype, bool debug);


static bool
is_debug_mode(void)
{
    return NULL != getenv("BSON_NUMPY_DEBUG");
}


/* Stub to test passing ndarrays. */
static PyObject *
ndarray_to_bson(PyObject *self, PyObject *args)
{
    PyObject *array_obj;
    PyArray_Descr *dtype;
    PyArrayObject *ndarray;
    bson_t document;
    bool debug;
    npy_intp i;

    if (!PyArg_ParseTuple(args, "O", &array_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }

    /* Convert array */
    if (!PyArray_Check(array_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_OutputConverter(array_obj, &ndarray)) {
        PyErr_SetString(BsonNumpyError, "bad array type");
        return NULL;
    }
    dtype = PyArray_DTYPE(ndarray);

    /* npy_intp num_dims = PyArray_NDIM(ndarray);
    npy_intp *shape = PyArray_SHAPE(ndarray); */
    npy_intp num_documents = PyArray_DIM(ndarray, 0);

    debug = is_debug_mode();

    bson_init(&document);

    /* TODO: could use array iterator API but potentially better to recur
     * ourselves */
    for (i = 0; i < num_documents; i++) {
        void *pointer = PyArray_GETPTR1(ndarray, i);
        PyObject *result = PyArray_GETITEM(ndarray, pointer);
        PyObject *type = PyObject_Type(result);

        if (debug) {
            printf("got item at %i =", (int) i);
            PyObject_Print(result, stdout, 0);
            printf(" type=");
            PyObject_Print(type, stdout, 0);
            printf("\n");
        }
    }

    if (debug) {
        printf("ndarray=");
        PyObject_Print((PyObject *) ndarray, stdout, 0);
        printf(" dtype=");
        PyObject_Print((PyObject *) dtype, stdout, 0);
        printf("\n");
    }
    return Py_BuildValue(""); /* TODO: return document instead */
}


static const char *
_bson_type_name(bson_type_t t)
{
    switch (t) {
        case BSON_TYPE_DOUBLE:
            return "Double";
        case BSON_TYPE_UTF8:
            return "UTF-8 string";
        case BSON_TYPE_DOCUMENT:
            return "Sub-document";
        case BSON_TYPE_ARRAY:
            return "Array";
        case BSON_TYPE_BINARY:
            return "Binary";
        case BSON_TYPE_UNDEFINED:
            return "Undefined";
        case BSON_TYPE_OID:
            return "ObjectId";
        case BSON_TYPE_BOOL:
            return "Bool";
        case BSON_TYPE_DATE_TIME:
            return "Datetime";
        case BSON_TYPE_NULL:
            return "Null";
        case BSON_TYPE_REGEX:
            return "Regular Expression";
        case BSON_TYPE_DBPOINTER:
            return "DBPointer";
        case BSON_TYPE_CODE:
            return "Code";
        case BSON_TYPE_SYMBOL:
            return "Symbol";
        case BSON_TYPE_CODEWSCOPE:
            return "Code with Scope";
        case BSON_TYPE_INT32:
            return "Int32";
        case BSON_TYPE_TIMESTAMP:
            return "Timestamp";
        case BSON_TYPE_INT64:
            return "Int64";
#ifdef BSON_TYPE_DECIMAL128
        case BSON_TYPE_DECIMAL128:
            return "Decimal128";
#endif
        case BSON_TYPE_MAXKEY:
            return "MaxKey";
        case BSON_TYPE_MINKEY:
            return "MinKey";
        default:
            return "unknown";
    }
}


static int
_load_array_from_bson(bson_iter_t *it, PyArrayObject *ndarray, long offset,
                      npy_intp *coordinates, int current_depth,
                      PyArray_Descr *dtype, bool debug)
{
    npy_intp dimensions = PyArray_NDIM(ndarray);
    bson_iter_t sub_it;

    /* Get length of array */
    bson_iter_recurse(it, &sub_it);
    int count = 0;
    while (bson_iter_next(&sub_it)) {
        count++;
    }
    bson_iter_recurse(it, &sub_it);
    if (count == 1) {
        bson_iter_next(&sub_it);

        /* Array is of length 1, therefore we treat it like a number */
        return _load_scalar_from_bson(&sub_it, ndarray, offset, coordinates,
                                      current_depth, dtype, debug);
    } else {

        int i = 0;
        while (bson_iter_next(&sub_it)) {
            PyArray_Descr *sub_dtype = dtype->subarray ? dtype->subarray->base
                                                       : dtype;

            /* If we're recurring after the end of dtype's dimensions, we have
             * a flexible type subarray */
            long new_offset = offset;
            if (current_depth + 1 < dimensions) {
                coordinates[current_depth + 1] = i;
            } else {
                PyErr_SetString(BsonNumpyError, "TODO: unhandled case");
                return 0;
            }
            int ret = _load_scalar_from_bson(&sub_it, ndarray, new_offset,
                                             coordinates, current_depth + 1,
                                             sub_dtype, debug);
            if (ret == 0) {
                return 0;
            };
            i++;
        }
        return 1;
    }
}


static void
_bsonnumpy_type_err(bson_type_t bson_type, const PyArray_Descr *dtype,
                    const char *msg)
{
    PyObject *repr = PyObject_Repr((PyObject *) dtype);
#if PY_MAJOR_VERSION >= 3
    PyObject *reprs = PyUnicode_AsASCIIString(repr);
#else
    PyObject *reprs = PyString_AsEncodedString(repr, "utf-8", "replace");
#endif
    PyBytes_AsString(reprs);

    PyErr_Format(BsonNumpyError, "cannot convert %s to %s: %s",
                 _bson_type_name(bson_type), PyBytes_AsString(reprs), msg);

    Py_DECREF(reprs);
    Py_DECREF(repr);
}


static int
_load_utf8_from_bson(const bson_value_t *value, void *dst, PyArray_Descr *dtype)
{
    npy_intp itemsize;
    npy_intp bson_item_len;

    if (dtype->kind != 'S' && dtype->kind != 'U' && dtype->kind != 'V') {
        _bsonnumpy_type_err(value->value_type, dtype, "use 'S' 'U' or 'V'");
        return 0;
    }

    bson_item_len = value->value.v_utf8.len;
    itemsize = dtype->elsize;

    if (bson_item_len > itemsize) {
        /* truncate data that's too long */
        bson_item_len = itemsize;
    }

    memcpy(dst, value->value.v_utf8.str, bson_item_len);
    /* zero-pad data that's too short */
    memset(dst + bson_item_len, '\0', itemsize - bson_item_len);

    return 1;
}


static int
_load_binary_from_bson(const bson_value_t *value, void *dst,
                       PyArray_Descr *dtype)
{
    npy_intp itemsize;
    npy_intp bson_item_len;

    if (dtype->kind != 'S' && dtype->kind != 'U' && dtype->kind != 'V') {
        _bsonnumpy_type_err(value->value_type, dtype, "use 'S' 'U' or 'V'");
        return 0;
    }

    bson_item_len = value->value.v_binary.data_len;
    itemsize = dtype->elsize;

    if (bson_item_len > itemsize) {
        /* truncate data that's too long */
        bson_item_len = itemsize;
    }

    memcpy(dst, value->value.v_binary.data, bson_item_len);
    /* zero-pad data that's too short */
    memset(dst + bson_item_len, '\0', itemsize - bson_item_len);

    return 1;
}


static int
_load_oid_from_bson(const bson_value_t *value, void *dst, PyArray_Descr *dtype)
{
    if ((dtype->kind != 'S' && dtype->kind != 'V') || dtype->elsize != 12) {
        _bsonnumpy_type_err(value->value_type, dtype, "use 'S12' or 'V12'");
        return 0;
    }

    memcpy(dst, value->value.v_oid.bytes, sizeof value->value.v_oid.bytes);
    return 1;
}


static int
_load_int32_from_bson(const bson_value_t *value, void *dst,
                      PyArray_Descr *dtype)
{
    if (dtype->kind != 'i' && dtype->kind != 'u') {
        _bsonnumpy_type_err(value->value_type, dtype, "use an integer type");
        return 0;
    }

    if ((size_t) dtype->elsize < sizeof(int32_t)) {
        _bsonnumpy_type_err(value->value_type, dtype,
                            "use at least a 32-bit integer type");
        return 0;
    }

    memcpy(dst, &value->value.v_int32, sizeof value->value.v_int32);
    return 1;
}


static int
_load_int64_from_bson(const bson_value_t *value, void *dst,
                      PyArray_Descr *dtype)
{
    if (dtype->kind != 'i' && dtype->kind != 'u') {
        _bsonnumpy_type_err(value->value_type, dtype, "use an integer type");
        return 0;
    }

    if ((size_t) dtype->elsize < sizeof(int64_t)) {
        _bsonnumpy_type_err(value->value_type, dtype,
                            "use at least a 64-bit integer type");
        return 0;
    }

    memcpy(dst, &value->value.v_int64, sizeof value->value.v_int64);
    return 1;
}


static int
_load_double_from_bson(const bson_value_t *value, void *dst,
                       PyArray_Descr *dtype)
{
    if (dtype->kind != 'f') {
        _bsonnumpy_type_err(value->value_type, dtype,
                            "use a floating point type");
        return 0;
    }

    if ((size_t) dtype->elsize < sizeof(int64_t)) {
        _bsonnumpy_type_err(value->value_type, dtype,
                            "use at least a 64-bit floating point type");
    }

    memcpy(dst, &value->value.v_double, sizeof value->value.v_double);
    return 1;
}


static int
_load_bool_from_bson(const bson_value_t *value, void *dst, PyArray_Descr *dtype)
{
    if (dtype->kind != 'b' && dtype->kind != 'i' && dtype->kind != 'u') {
        _bsonnumpy_type_err(value->value_type, dtype,
                            "use 'b' or an integer type");
    }

    memcpy(dst, &value->value.v_bool, sizeof value->value.v_bool);
    return 1;
}


static int
_load_scalar_from_bson(bson_iter_t *bsonit, PyArrayObject *ndarray, long offset,
                       npy_intp *coordinates, int current_depth,
                       PyArray_Descr *dtype, bool debug)
{
    void *pointer;
    const bson_value_t *value;

    value = bson_iter_value(bsonit);
    pointer = PyArray_GetPtr(ndarray, coordinates) + offset;

    switch (value->value_type) {
        case BSON_TYPE_ARRAY:
            return _load_array_from_bson(bsonit, ndarray, offset, coordinates,
                                         current_depth, dtype, debug);
        case BSON_TYPE_DOUBLE:
            return _load_double_from_bson(value, pointer, dtype);
        case BSON_TYPE_UTF8:
            return _load_utf8_from_bson(value, pointer, dtype);
        case BSON_TYPE_BINARY:
            return _load_binary_from_bson(value, pointer, dtype);
        case BSON_TYPE_OID:
            return _load_oid_from_bson(value, pointer, dtype);
        case BSON_TYPE_BOOL:
            return _load_bool_from_bson(value, pointer, dtype);
        case BSON_TYPE_DATE_TIME:
            return _load_int64_from_bson(value, pointer, dtype);
        case BSON_TYPE_INT32:
            return _load_int32_from_bson(value, pointer, dtype);
        case BSON_TYPE_INT64:
            return _load_int64_from_bson(value, pointer, dtype);

        default:
            PyErr_Format(BsonNumpyError, "unsupported BSON type: %s",
                         _bson_type_name(value->value_type));
            return false;
    }
}


static PyObject *
bson_to_ndarray(PyObject *self, PyObject *args)
{
    /* Takes in a BSON byte string and a dtype */
    PyObject *binary_doc;
    PyObject *dtype_obj;
    PyObject *array_obj;
    const char *bytestr;
    PyArray_Descr *dtype;
    PyArrayObject *ndarray;
    Py_ssize_t bytes_len;
    Py_ssize_t number_dimensions = -1;
    npy_intp *dimension_lengths;
    bson_iter_t bsonit;
    bson_t *document;
    size_t err_offset;
    bool debug;
    npy_intp *coordinates;
    npy_intp i;

    if (!PyArg_ParseTuple(args, "SO", &binary_doc, &dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    bytestr = PyBytes_AS_STRING(binary_doc);
    bytes_len = PyBytes_GET_SIZE(binary_doc);
    /* slower than what??? Also, is this a valid cast? */
    document = bson_new_from_data((uint8_t *) bytestr, bytes_len);
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
        /* TODO: validate in a reasonable way, now segfaults if bad */
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return NULL;
    }

    /* Convert dtype */
    if (!PyArray_DescrCheck(dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
        return NULL;
    }

    debug = is_debug_mode();

    bson_iter_init(&bsonit, document);
    dimension_lengths = malloc(1 * sizeof(npy_intp));
    dimension_lengths[0] = bson_count_keys(document);
    number_dimensions = 1;

    if (dtype->subarray != NULL) {
        PyObject *shape = dtype->subarray->shape;
        if (!PyTuple_Check(shape)) {
            PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
            return NULL;
        }
        number_dimensions = (int) PyTuple_Size(shape);
    }

    Py_INCREF(dtype);

    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0);
    if (!array_obj) {
        return NULL;
    }

    if (NPY_FAIL == PyArray_OutputConverter(array_obj, &ndarray)) {
        return NULL;
    }

    coordinates = calloc(number_dimensions + 1, sizeof(npy_intp));
    for (i = 0; i < dimension_lengths[0]; i++) {
        bson_iter_next(&bsonit);
        coordinates[0] = i;
        int success = _load_scalar_from_bson(&bsonit, ndarray, 0, coordinates,
                                             0, dtype, debug);
        if (success == 0) {
            return NULL;
        }
    }

    free(dimension_lengths);
    free(document);
    free(coordinates);

    return array_obj;
}


static int
_load_flexible_from_bson(bson_t *document, npy_intp *coordinates,
                         PyArrayObject *ndarray, PyArray_Descr *dtype,
                         int current_depth, char *key_str,
                         npy_intp *sub_coordinates,
                         npy_intp sub_coordinates_length, npy_intp offset,
                         bool debug)
{
    PyObject *fields, *key, *value = NULL;
    int number_dimensions = PyArray_NDIM(ndarray);
    Py_ssize_t pos;
    bson_iter_t bsonit;
    PyArray_Descr *sub_dtype;
    PyObject *sub_dtype_obj, *offset_obj;
    int success;
    Py_ssize_t i;

    if (dtype->fields != NULL && dtype->fields != Py_None) {
        PyObject *ordered_names = PyArray_FieldNames(dtype->fields);
        Py_ssize_t number_fields = PyTuple_Size(ordered_names);
        /* A field is described by a tuple composed of another data-type
         * descriptor and a byte offset. */
        fields = dtype->fields;
        if (!PyDict_Check(fields)) {
            PyErr_SetString(BsonNumpyError,
                            "in _load_flexible_from_bson: dtype.fields not a"
                            " dict");
            return 0;
        }

        pos = 0;

        for (i = 0; i < number_fields; i++) {
            long offset_long;
            int sub_depth;
            key = PyTuple_GetItem(ordered_names, i);
            value = PyDict_GetItem(fields, key);

            if (!PyTuple_Check(value)) {
                PyErr_SetString(BsonNumpyError,
                                "dtype in fields is not a tuple");
                return 0;
            }
            if (PyUnicode_Check(key)) {
                key = PyUnicode_AsASCIIString(key);
            }
            if (!PyBytes_Check(key)) {
                PyErr_SetString(BsonNumpyError,
                                "bson string error in key names");
                return 0;
            }
            offset_obj = PyTuple_GetItem(value, 1);
            offset_long = PyLong_AsLong(offset_obj);
            key_str = PyBytes_AsString(key);
            sub_dtype_obj = PyTuple_GetItem(value, 0);

            /* Convert from python object to numpy dtype object */
            if (!PyArray_DescrConverter(sub_dtype_obj, &sub_dtype)) {
                PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
                return 0;
            }

            sub_depth = current_depth - number_dimensions;
            if (sub_dtype->subarray) {
                sub_coordinates[sub_depth] = i;
                _load_flexible_from_bson(document, coordinates, ndarray,
                                         sub_dtype, current_depth + 1, key_str,
                                         sub_coordinates,
                                         sub_coordinates_length + 1, offset,
                                         debug);

            } else if (sub_dtype->fields && sub_dtype->fields != Py_None) {
                bson_iter_init(&bsonit, document);
                if (bson_iter_find(&bsonit, key_str)) {
                    bson_t *sub_document;
                    uint32_t document_len;
                    const uint8_t *document_buffer;

                    if (!BSON_ITER_HOLDS_DOCUMENT(&bsonit)) {
                        PyErr_SetString(BsonNumpyError,
                                        "Expected list from dtype, got other"
                                        " type");
                        return 0;
                    }

                    bson_iter_document(&bsonit, &document_len,
                                       &document_buffer);
                    sub_document = bson_new_from_data(document_buffer,
                                                      document_len);

                    sub_coordinates[sub_depth] = i;

                    _load_flexible_from_bson(sub_document, coordinates, ndarray,
                                             sub_dtype, current_depth + 1, NULL,
                                             sub_coordinates,
                                             sub_coordinates_length + 1,
                                             offset + offset_long, debug);

                } else {
                    PyErr_SetString(BsonNumpyError,
                                    "Error: expected key from dtype in document, not found");
                }
            } else {
                bson_iter_init(&bsonit, document);
                if (bson_iter_find(&bsonit, key_str)) {
                    /* TODO: if sub_dtype->elsize==0, then it is a flexible type */
                    success = _load_scalar_from_bson(&bsonit, ndarray,
                                                     offset + offset_long,
                                                     coordinates, current_depth,
                                                     sub_dtype, debug);
                    if (!success) {
                        return 0;
                    }
                } else {
                    /* TODO: nicer error message */
                    PyErr_SetString(BsonNumpyError,
                                    "document does not match dtype.");
                    return 0;
                }
            }
            pos++;
        }
    } else if (dtype->subarray) {
        PyObject *shape = dtype->subarray->shape;
        PyArray_Descr *sub_descr = dtype->subarray->base;
        void *ptr;
        PyObject *subndarray_tuple;
        PyObject *length_obj;
        long length_long;

        bson_iter_init(&bsonit, document);

        Py_ssize_t dims_subarray = PyTuple_Size(shape);
        if (bson_iter_find(&bsonit, key_str)) {
            npy_intp i;
            int sub_i;
            PyArrayObject *subndarray;
            npy_intp *subarray_coordinates;
            bson_iter_t sub_it;

            if (!BSON_ITER_HOLDS_ARRAY(&bsonit)) {
                PyErr_SetString(BsonNumpyError,
                                "Expected list from dtype, got other type");
                return 0;
            }

            /* Get subarray as ndarray */
            ptr = PyArray_GetPtr(ndarray, coordinates);
            subndarray_tuple = PyArray_GETITEM(ndarray, ptr);
            for (sub_i = 0;
                 sub_i < current_depth - number_dimensions; sub_i++) {
                npy_intp offset = sub_coordinates[sub_i];
                subndarray_tuple = PyTuple_GetItem(subndarray_tuple, offset);
            }
            if (!PyArray_OutputConverter(subndarray_tuple, &subndarray)) {
                PyErr_SetString(BsonNumpyError,
                                "Expected subarray, got other type");
                return 0;
            }

            /* Get length of top-level array */
            length_obj = PyTuple_GetItem(shape, 0);
            length_long = PyLong_AsLong(length_obj);

            /* Create coordinates for subtype */
            subarray_coordinates = calloc(dims_subarray + 1, sizeof(npy_intp));

            /* Loop through array and load sub-arrays */
            bson_iter_recurse(&bsonit, &sub_it);
            for (i = 0; i < length_long; i++) {
                int success;

                bson_iter_next(&sub_it);
                subarray_coordinates[0] = i;
                success = _load_scalar_from_bson(&sub_it, subndarray, 0,
                                                 subarray_coordinates, 0,
                                                 sub_descr, debug);
                if (success == 0) {
                    return 0;
                }
            }
        } else {
            PyErr_SetString(BsonNumpyError, "key from dtype not found");
            return 0;
        }

    } else {
        PyErr_SetString(BsonNumpyError,
                        "dtype must include field names, like"
                        " dtype([('fieldname', numpy.int)])");
        return 0;
    }
    return 1;
}


static PyObject *
sequence_to_ndarray(PyObject *self, PyObject *args)
{
    PyObject *array_obj = NULL;
    PyObject *iterable_obj;
    PyObject *iterator_obj;
    PyObject *dtype_obj;
    PyObject *binary_doc;

    PyArray_Descr *dtype;
    PyArrayObject *ndarray;

    int num_documents;
    int number_dimensions;
    npy_intp *dimension_lengths = NULL;
    npy_intp *coordinates = NULL;
    bool debug;

    size_t err_offset;
    int row = 0;

    if (!PyArg_ParseTuple(args, "OOi", &iterator_obj, &dtype_obj,
                          &num_documents)) {
        return NULL;
    }
    if (!PyIter_Check(iterator_obj)) {
        /* it's not an iterator, maybe it's iterable? */
        iterable_obj = iterator_obj;
        Py_INCREF(iterable_obj);
        iterator_obj = PyObject_GetIter (iterable_obj);
        if (!iterator_obj) {
            Py_DECREF(iterable_obj);
            PyErr_SetString(BsonNumpyError,
                            "sequence_to_ndarray expects an iterator");
            return NULL;
        }
    }
    if (!PyArray_DescrCheck(dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
        return NULL;
    }

    debug = is_debug_mode();

    dimension_lengths = malloc(1 * sizeof(npy_intp));
    dimension_lengths[0] = num_documents;

    number_dimensions = 1;

    if (dtype->subarray != NULL) {
        PyObject *shape = dtype->subarray->shape;
        if (!PyTuple_Check(shape)) {
            PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
            return NULL;
        }
        number_dimensions = (int) PyTuple_Size(shape);
    }

    /* printf("dimension_lengths=%i, number_dimensions=%i\n", num_documents, number_dimensions); */
    Py_INCREF(dtype);

    /* This function steals a reference to dtype? */
    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0);
    if (!array_obj) {
        goto done;
    }

    if (NPY_FAIL == PyArray_OutputConverter(array_obj, &ndarray)) {
        goto done;
    }

    coordinates = calloc(1 + number_dimensions, sizeof(npy_intp));

    /* For each row */
    while ((binary_doc = PyIter_Next(iterator_obj))) {
        npy_intp *sub_coordinates;
        int p;

        /* Get BSON document */
        const char *bytes_str = PyBytes_AS_STRING(binary_doc);
        Py_ssize_t bytes_len = PyBytes_GET_SIZE(binary_doc);
        bson_t *document = bson_new_from_data((uint8_t *) bytes_str, bytes_len);

        if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
            /* TODO: validate in a reasonable way, now segfaults if bad */
            PyErr_SetString(BsonNumpyError, "Document failed validation");
            return 0;
        }

        sub_coordinates = calloc(100, sizeof(npy_intp));

        /* Don't need to pass key to first layer */
        if (!_load_flexible_from_bson(document, coordinates, ndarray,
                                      PyArray_DTYPE(ndarray), 1, NULL,
                                      sub_coordinates, 0, 0, debug)) {
            /* error set by _load_flexible_from_bson */
            return NULL;
        }

        free(document);
        coordinates[0] = ++row;
        if (row >= num_documents) {
            break;
        }

        /* Reset coordinates to zero */
        for (p = 1; p < number_dimensions; p++) {
            coordinates[p] = 0;
        }
    }

    if (row < num_documents) {
        PyObject *none_obj;
        PyArray_Dims newshape = {dimension_lengths, number_dimensions};
        if (debug) {
            printf("resizing from %d to %d\n", num_documents, row);
        }

        dimension_lengths[0] = row;
        /* returns None or NULL */
        none_obj = PyArray_Resize((PyArrayObject *) array_obj, &newshape,
                                  false /* refcheck */, NPY_CORDER);

        Py_XDECREF(none_obj);
    }

done:
    free(dimension_lengths);
    free(coordinates);

    if (PyErr_Occurred()) {
        Py_XDECREF(array_obj);
        return NULL;
    }

    return array_obj;
}


static PyMethodDef BsonNumpyMethods[] = {{"ndarray_to_bson",     ndarray_to_bson,     METH_VARARGS, "Convert an ndarray into a BSON byte string"},
                                         {"bson_to_ndarray",     bson_to_ndarray,     METH_VARARGS, "Convert BSON byte string into an ndarray"},
                                         {"sequence_to_ndarray", sequence_to_ndarray, METH_VARARGS, "Convert an iterator containing BSON documents into an ndarray"},
                                         {NULL,                  NULL,                0,            NULL}        /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef bsonnumpymodule = {
   PyModuleDef_HEAD_INIT,
   "bsonnumpy",
   NULL,
   -1,
   BsonNumpyMethods
};

PyMODINIT_FUNC
PyInit_bsonnumpy(void) {
    PyObject* m;

    m = PyModule_Create(&bsonnumpymodule);
    if (m == NULL)
        return NULL;

    BsonNumpyError = PyErr_NewException("bsonnumpy.error", NULL, NULL);
    Py_INCREF(BsonNumpyError);
    PyModule_AddObject(m, "error", BsonNumpyError);

    import_array();

    return m;
}
#else  /* Python 2.x */


PyMODINIT_FUNC
initbsonnumpy(void)
{
    PyObject *m;

    m = Py_InitModule("bsonnumpy", BsonNumpyMethods);
    if (m == NULL) {
        return;
    }

    BsonNumpyError = PyErr_NewException("bsonnumpy.error", NULL, NULL);
    Py_INCREF(BsonNumpyError);
    PyModule_AddObject(m, "error", BsonNumpyError);

    import_array();
}


#endif
