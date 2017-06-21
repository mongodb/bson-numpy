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
_load_scalar_from_bson(
        bson_iter_t *bsonit, PyArrayObject *ndarray, PyArray_Descr *dtype,
        npy_intp *coordinates, int current_depth, long offset);

static int
_load_document_from_bson(
        bson_t *document, PyArrayObject *ndarray, PyArray_Descr *dtype,
        npy_intp *array_coordinates, int array_depth,
        npy_intp *doc_coordinates, int doc_depth, npy_intp offset);

static bool debug_mode = false;

static void
debug(char* message, PyObject* object, bson_t* doc)
{
    if (NULL != getenv("BSON_NUMPY_DEBUG")) {
        printf("%s", message);
        if (object) {
            printf(": ");
            PyObject_Print(object, stdout, 0);
        }
        if (doc) {
            printf(": %s", bson_as_json(doc, NULL));

        }
        printf("\n");
    }
}

/* Stub to test passing ndarrays. */
static PyObject *
ndarray_to_bson(PyObject *self, PyObject *args)
{
    PyObject *array_obj;
    PyArray_Descr *dtype;
    PyArrayObject *ndarray;
    bson_t document;
//    npy_intp i;

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
//    npy_intp num_documents = PyArray_DIM(ndarray, 0);

    bson_init(&document);

    /* TODO: could use array iterator API but potentially better to recur
     * ourselves */
//    for (i = 0; i < num_documents; i++) {
//        void *pointer = PyArray_GETPTR1(ndarray, i);
//        PyObject *result = PyArray_GETITEM(ndarray, pointer);
//        PyObject *type = PyObject_Type(result);

//        if (debug) {
//            printf("got item at %i =", (int) i);
//            PyObject_Print(result, stdout, 0);
//            printf(" type=");
//            PyObject_Print(type, stdout, 0);
//            printf("\n");
//        }
//    }

//    if (debug) {
//        printf("ndarray=");
//        PyObject_Print((PyObject *) ndarray, stdout, 0);
//        printf(" dtype=");
//        PyObject_Print((PyObject *) dtype, stdout, 0);
//        printf("\n");
//    }
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
_load_scalar_from_bson(
        bson_iter_t *bsonit, PyArrayObject *ndarray, PyArray_Descr *dtype,
        npy_intp *coordinates, int current_depth, long offset)
{
    void *pointer;
    const bson_value_t *value;

    value = bson_iter_value(bsonit);
    pointer = PyArray_GetPtr(ndarray, coordinates) + offset;

    switch (value->value_type) {
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
        int success = _load_scalar_from_bson(&bsonit, ndarray, dtype,
                                             coordinates, 0, 0);
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
_load_array_from_bson(bson_iter_t *bsonit, PyArrayObject *ndarray,
                      PyArray_Descr *dtype, npy_intp *coordinates,
                      int current_depth, long offset)
{

    /* Type and length checks */
    if (!BSON_ITER_HOLDS_ARRAY(bsonit)) {
        PyErr_SetString(BsonNumpyError,
                        "invalid document: expected list from dtype,"
                                " got other type");
        return 0;
    }
    if (dtype->subarray == NULL) {
        debug("_load_array_from_bson called with non-array dtype",
              (PyObject*)dtype, NULL);
        PyErr_SetString(BsonNumpyError,
                        "expected subarray, invalid dtype");
        return 0;
    }
    PyObject* shape = dtype->subarray->shape;
    if (!PyTuple_Check(shape)) {
        PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
        return 0;
    }
    PyObject* expected_length_obj = PyTuple_GetItem(shape, current_depth);
    long expected_length = PyLong_AsLong(expected_length_obj);
    int dimensions = (int) PyTuple_Size(shape);

    bson_iter_t sub_it;
    bson_iter_recurse(bsonit, &sub_it);
    int count = 0;
    while (bson_iter_next(&sub_it)) {
        count++;
    }
    if (expected_length != count) {
        char buffer[100];
        snprintf(buffer, 100,
                 "expected length %li but got list of length %i",
                 expected_length, count);
        debug(buffer, NULL, NULL);
        PyErr_SetString(BsonNumpyError,
                        "invalid document: list is of incorrect length");
        return 0;
    }

    /* Load data into array */
    bson_iter_recurse(bsonit, &sub_it);
    if (count == 1) {
        debug("converting array of length 1 to scalar\n", NULL, NULL);
        bson_iter_next(&sub_it);

        /* Array is of length 1, therefore we treat it like a number */
        return _load_scalar_from_bson(&sub_it, ndarray, dtype,
                                      coordinates, current_depth, offset);
    } else {
        PyArray_Descr* subdtype = dtype;
        int (*load_func)(bson_iter_t*, PyArrayObject*, PyArray_Descr*,
                         npy_intp*, int, long) = &_load_array_from_bson;
        if(current_depth == dimensions - 1) {
            subdtype = dtype->subarray->base;
            load_func = &_load_scalar_from_bson;
        }

        int i = 0;
        while (bson_iter_next(&sub_it)) {
            long new_offset = offset;
            if (current_depth < dimensions) {
                coordinates[current_depth] = i;
            } else {
                PyErr_SetString(BsonNumpyError, "TODO: unhandled case");
                return 0;
            }
            int ret = (*load_func)(&sub_it, ndarray, subdtype,
                                   coordinates, current_depth + 1, new_offset);
            if (ret == 0) {
                /* Error set by loading function */
                return 0;
            };
            i++;
        }
        /* Reset the rest of the coordinates to zero */
        for (i = current_depth; i < dimensions; i++) {
            coordinates[i] = 0;
        }
        return 1;
    }
}


static int
_load_document_from_bson(
        bson_t *document, PyArrayObject *ndarray, PyArray_Descr *dtype,
        npy_intp *array_coordinates, int array_depth,
        npy_intp *doc_coordinates, int doc_depth, npy_intp offset) {

    PyObject *fields, *key, *value = NULL;
    bson_iter_t bsonit;
    char* key_str;
    PyArray_Descr *sub_dtype;
    PyObject *sub_dtype_obj, *offset_obj;
    Py_ssize_t i;
    int sub_i;

    if (dtype->fields != NULL && dtype->fields != Py_None) {
        /* A field is described by a tuple composed of another data-type
         * descriptor and a byte offset. */
        fields = dtype->fields;

        PyObject *ordered_names = PyArray_FieldNames(fields);
        Py_ssize_t number_fields = PyTuple_Size(ordered_names);
        if (!PyDict_Check(fields)) {
            debug("dtype->fields is not a dict", fields, NULL);
            PyErr_SetString(BsonNumpyError,
                            "invalid dtype: dtype.fields is not a dict");
            return 0;
        }
        for (i = 0; i < number_fields; i++) {
            long offset_long;
            key = PyTuple_GetItem(ordered_names, i);
            value = PyDict_GetItem(fields, key);

            /* Have found another layer of nested document */
            doc_coordinates[doc_depth] = i;

            /* Get key, sub dtype, and offset for each field in dtype */
            if (!PyTuple_Check(value)) {
                PyErr_SetString(
                        BsonNumpyError,
                        "invalid dtype: sub dtype is invalid");
                return 0;
            }
            if (PyUnicode_Check(key)) {
                key = PyUnicode_AsASCIIString(key);
            }
            if (!PyBytes_Check(key)) {
                PyErr_SetString(
                        BsonNumpyError,
                        "invalid document: bson string error in key name");
                return 0;
            }
            offset_obj = PyTuple_GetItem(value, 1);
            offset_long = PyLong_AsLong(offset_obj);
            key_str = PyBytes_AsString(key);
            sub_dtype_obj = PyTuple_GetItem(value, 0);
            if (!PyArray_DescrConverter(sub_dtype_obj, &sub_dtype)) {
                debug("dtype->fields sub dtype is invalid",
                      sub_dtype_obj, NULL);
                PyErr_SetString(
                        BsonNumpyError,
                        "invalid dtype: sub dtype is invalid");
                return 0;
            }

            /* Get subdocument from BSON document */
            bson_iter_init(&bsonit, document);
            if (!bson_iter_find(&bsonit, key_str)) {
                /* BSON document is missing key specified by dtype */
                char buffer[100];
                snprintf(buffer, 100,
                         "could not find key \"%s\" in sub document",
                         key_str);
                debug(buffer, NULL, document);
                PyErr_SetString(
                        BsonNumpyError,
                        "document does not match dtype");
                return 0;
            }

            if (sub_dtype->subarray) {
                /* If the current key's value is a subarray */

                array_coordinates[array_depth] = i;
                PyObject* shape = sub_dtype->subarray->shape;
                if (!PyTuple_Check(shape)) {
                    debug("invalid dtype->subarray->shape", shape, NULL);
                    PyErr_SetString(BsonNumpyError,
                                    "dtype argument had invalid subarray");
                    return 0;
                }
                size_t number_dimensions = (size_t)PyTuple_Size(shape);

                /* Index into ndarray with array_coordinates */
                PyArrayObject* subndarray;
                void* subarray_ptr = PyArray_GetPtr(ndarray, array_coordinates);
                PyObject* subarray_tuple = PyArray_GETITEM(
                        ndarray, subarray_ptr);

                /* Indexing into ndarray with named fields returns a tuple
                 * that is nested as many levels as there are fields */
                for (sub_i = 0; sub_i < doc_depth + 1; sub_i++) {
                    npy_intp index = doc_coordinates[sub_i];
                    subarray_tuple = PyTuple_GetItem(subarray_tuple, index);
                }
                PyObject* subarray_obj = subarray_tuple;

                /* Get element of array */
                if (!subarray_obj) {
                    debug("PyArray_GetField failed with dtype",
                          (PyObject*)dtype, NULL);
                    PyErr_SetString(BsonNumpyError,
                                    "indexing failed on named field");
                    return 0;
                }
                if (NPY_FAIL == PyArray_OutputConverter(subarray_obj,
                                                        &subndarray)) {
                    debug("PyArray_OutputConverter failed with array object",
                          subarray_obj, NULL);
                    PyErr_SetString(BsonNumpyError,
                                    "indexing failed on named field");
                    return 0;
                }
                npy_intp* new_coordinates = calloc(
                        1 + number_dimensions, sizeof(npy_intp));

                if (!_load_array_from_bson(&bsonit, subndarray, sub_dtype,
                                           new_coordinates, 0, 0)) {
                    /* error set by load_array_from_bson */
                    return 0;
                }

            } else if (sub_dtype->fields && sub_dtype->fields != Py_None) {
                /* If the current key's value is a subdocument */
                bson_t *sub_document;
                uint32_t document_len;
                const uint8_t *document_buffer;
                if (!BSON_ITER_HOLDS_DOCUMENT(&bsonit)) {
                    debug("the document that does not match dtype is",
                          NULL, document);
                    PyErr_SetString(BsonNumpyError,
                                    "invalid document: expected subdoc "
                                            "from dtype, got other type");
                    return 0;
                }
                bson_iter_document(&bsonit, &document_len,
                                   &document_buffer);
                sub_document = bson_new_from_data(document_buffer,
                                                  document_len);

                if (!_load_document_from_bson(sub_document, ndarray, sub_dtype,
                                              array_coordinates, array_depth,
                                              doc_coordinates, doc_depth + 1,
                                              offset + offset_long)) {
                    /* error set by _load_document_from_bson */
                    return 0;
                }
            } else {
                /* If the current key's value is a leaf */
                if (!_load_scalar_from_bson(&bsonit, ndarray, sub_dtype,
                                            array_coordinates, array_depth,
                                            offset + offset_long)) {
                    /* Error set by _load_scalar_from_bson */
                    return 0;
                }
            }
        }
        return 1;
    }

    /* Top-level dtype did not have named fields */
    debug("_load_document_from_bson called with a non-document dtype",
          (PyObject*)dtype, NULL);
    PyErr_SetString(BsonNumpyError,
                    "dtype must include field names, like"
                            " dtype([('fieldname', numpy.int)])");
    return 0;
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
    int number_dimensions = 1;
    npy_intp *dimension_lengths = NULL;
    npy_intp *array_coordinates = NULL;

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
            PyErr_SetString(PyExc_TypeError,
                            "sequence_to_ndarray requires an iterator");
            return NULL;
        }
    }
    if (!PyArray_DescrCheck(dtype_obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "sequence_to_ndarray requires a numpy.dtype");
        return NULL;
    }
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        debug("invalid dtype object", dtype_obj, NULL);
        PyErr_SetString(BsonNumpyError,
                        "dtype argument was invalid");
        return NULL;
    }
    if (num_documents < 0) {
        PyErr_SetString(BsonNumpyError,
                        "count argument was negative");
        return NULL;
    }

    dimension_lengths = malloc(1 * sizeof(npy_intp));
    dimension_lengths[0] = num_documents;

    Py_INCREF(dtype);

    /* This function steals a reference to dtype? */
    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0);
    if (!array_obj) {
        debug("PyArray_Zeros failed with dtype", dtype_obj, NULL);
        PyErr_SetString(BsonNumpyError,
                        "ndarray initialization failed");
        goto done;
    }

    if (NPY_FAIL == PyArray_OutputConverter(array_obj, &ndarray)) {
        debug("PyArray_OutputConverter failed with array object",
              array_obj, NULL);
        PyErr_SetString(BsonNumpyError,
                        "ndarray initialization failed");
        goto done;
    }

    array_coordinates = calloc(1 + number_dimensions, sizeof(npy_intp));

    /* For each document in the collection, fill row of ndarray */
    while ((binary_doc = PyIter_Next(iterator_obj))) {
        /* Get BSON document */
        const char *bytes_str = PyBytes_AS_STRING(binary_doc);
        Py_ssize_t bytes_len = PyBytes_GET_SIZE(binary_doc);
        bson_t *document = bson_new_from_data((uint8_t *) bytes_str, bytes_len);
        if (!document) {
            debug("binary document failed bson_new_from_document",
                  binary_doc, NULL);
            PyErr_SetString(BsonNumpyError,
                            "document from sequence failed validation");
            return 0;
        }

        if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
            debug("binary document failed bson_validate", binary_doc, NULL);
            PyErr_SetString(BsonNumpyError,
                            "document from sequence failed validation");
            return 0;
        }

        /* Need two sets of coordinates, one for documents and one for arrays */
        npy_intp* doc_coordinates = calloc(100, sizeof(npy_intp));

        /* current_depth = 1 because layer 0 is the whole sequence */
        if (!_load_document_from_bson(document, ndarray, PyArray_DTYPE(ndarray),
                                      array_coordinates, 1,
                                      doc_coordinates, 0, 0)) {
            /* error set by _load_document_from_bson */
            return NULL;
        }

        free(document); // TODO: maybe we can avoid mallocing every document?
        array_coordinates[0] = ++row;
        if (row >= num_documents) {
            break;
        }
    }

    if (row < num_documents) {
        PyObject *none_obj;
        PyArray_Dims newshape = {dimension_lengths, number_dimensions};
        if (debug_mode) {
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
    free(array_coordinates);

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
    debug_mode = (NULL != getenv("BSON_NUMPY_DEBUG")); // TODO: never set

    import_array();
}


#endif
