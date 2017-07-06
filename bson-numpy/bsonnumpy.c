#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
/* #include <numpy/arrayobject.h> */
/* #include <numpy/npy_common.h> */
#include <numpy/ndarrayobject.h>

#include "bson/bson.h"

static PyObject *BsonNumpyError;

typedef enum {
    DTYPE_NESTED, /* like np.dtype([('a', np.int64), ('b', np.double)]) */
    DTYPE_SCALAR, /* like np.int64 */
    DTYPE_ARRAY,  /* like np.dtype('3i') */
} node_type_t;


typedef struct _parsed_dtype_t {
    node_type_t node_type;
    char *field_name;
    Py_ssize_t offset;

    /* for scalars */
    int type_num;
    int elsize;

    /* for nested types */
    struct _parsed_dtype_t **children;
    Py_ssize_t n_children;

    /* for sub-array types */
    Py_ssize_t n_dimensions;
    long *dims;
} parsed_dtype_t;


static parsed_dtype_t *
parse_array_dtype(PyArray_Descr *dtype, const char *field_name);

static parsed_dtype_t *
parse_nested_dtype(PyArray_Descr *dtype, const char *field_name);

static parsed_dtype_t *
parse_scalar_dtype(PyArray_Descr *dtype, const char *field_name);

static void
parsed_dtype_destroy(parsed_dtype_t *parsed);


static parsed_dtype_t *
parsed_dtype_new (PyArray_Descr *dtype, const char *field_name)
{
    PyObject *fields = dtype->fields;

    if (dtype->subarray) {
        return parse_array_dtype(dtype, field_name);
    }

    if (fields && fields != Py_None) {
        return parse_nested_dtype(dtype, field_name);
    }

    return parse_scalar_dtype(dtype, field_name);
}


#define PARSE_FAIL do {                  \
        if (parsed) {                    \
           parsed_dtype_destroy(parsed); \
           parsed = NULL;                \
        }                                \
        goto done;                       \
    } while (0)


static parsed_dtype_t *
parse_array_dtype(PyArray_Descr *dtype, const char *field_name)
{
    parsed_dtype_t *parsed;
    Py_ssize_t i;
    PyObject* shape;
    PyObject* dim;

    parsed = bson_malloc0(sizeof(parsed_dtype_t));
    parsed->node_type = DTYPE_ARRAY;
    parsed->field_name = bson_strdup(field_name);
    parsed->type_num = dtype->type_num;

    shape = dtype->subarray->shape;
    if (!PyTuple_Check(shape)) {
        PyErr_SetString(BsonNumpyError, "dtype argument had invalid subarray");
        PARSE_FAIL;
    }

    parsed->n_dimensions = PyTuple_Size(shape);
    parsed->dims = bson_malloc0(parsed->n_dimensions * sizeof(Py_ssize_t));
    for (i = 0; i < parsed->n_dimensions; i++) {
        dim = PyTuple_GET_ITEM(shape, i);
#if PY_MAJOR_VERSION >= 3
        parsed->dims[i] = PyLong_AsLong(dim);
#else
        parsed->dims[i] = PyInt_AsLong(dim);
#endif
    }

done:
    return parsed;
}


static parsed_dtype_t *
parse_nested_dtype(PyArray_Descr *dtype, const char *field_name)
{
    parsed_dtype_t *parsed;
    PyObject *fields = NULL;
    PyObject *key = NULL;
    PyObject *unicode_key = NULL;
    const char *key_str;
    PyObject *value = NULL;
    PyObject *ordered_names = NULL;
    PyObject *sub_dtype_obj = NULL;
    PyArray_Descr *sub_dtype;
    Py_ssize_t i;

    parsed = bson_malloc0(sizeof(parsed_dtype_t));
    parsed->node_type = DTYPE_NESTED;
    parsed->field_name = bson_strdup(field_name);

    fields = dtype->fields;
    ordered_names = PyArray_FieldNames(fields);
    parsed->n_children = PyTuple_Size(ordered_names);
    parsed->children = bson_malloc0(
        parsed->n_children * sizeof(parsed_dtype_t *));

    for (i = 0; i < parsed->n_children; i++) {
        key = PyTuple_GET_ITEM(ordered_names, i);
        value = PyDict_GetItem(fields, key);
        if (PyUnicode_Check(key)) {
            unicode_key = PyUnicode_AsASCIIString(key);
            if (!unicode_key) {
                PARSE_FAIL;
            }

            key_str = PyBytes_AsString(unicode_key);
            Py_DECREF(unicode_key);
        } else {
            key_str = PyBytes_AsString(key);
        }

        if (!key_str) {
            PARSE_FAIL;
        }

        sub_dtype_obj = PyTuple_GET_ITEM(value, 0);
        if (!PyArray_DescrConverter(sub_dtype_obj, &sub_dtype)) {
            PARSE_FAIL;
        }

        parsed->children[i] = parsed_dtype_new(sub_dtype, key_str);
        if (!parsed->children[i]) {
            PARSE_FAIL;
        }

        parsed->children[i]->offset = PyLong_AsLong(PyTuple_GET_ITEM(value, 1));
    }

done:
    Py_XDECREF(ordered_names);
    Py_XDECREF(sub_dtype_obj);

    return parsed;
}


static parsed_dtype_t *
parse_scalar_dtype(PyArray_Descr *dtype, const char *field_name)
{
    parsed_dtype_t *parsed;

    parsed = bson_malloc0(sizeof(parsed_dtype_t));
    parsed->node_type = DTYPE_SCALAR;
    parsed->field_name = bson_strdup(field_name);
    parsed->type_num = dtype->type_num;
    parsed->elsize = dtype->elsize;

    return parsed;
}


static void
parsed_dtype_destroy(parsed_dtype_t *parsed)
{
    Py_ssize_t  i;

    if (parsed) {
        if (parsed->children) {
            for (i = 0; i < parsed->n_children; i++) {
                parsed_dtype_destroy(parsed->children[i]);
            }

            bson_free(parsed->children);
        }

        if (parsed->dims) {
            bson_free(parsed->dims);
        }

        bson_free(parsed->field_name);
        bson_free(parsed);
    }
}


static int
_load_scalar_from_bson(
    bson_iter_t *bsonit, PyArrayObject *ndarray, PyArray_Descr *dtype, parsed_dtype_t *parsed,
    npy_intp *coordinates, int current_depth, long offset);


static int
_load_document_from_bson(
    bson_t *document, PyArrayObject *ndarray, PyArray_Descr *dtype, parsed_dtype_t *parsed,
    npy_intp *array_coordinates, int array_depth,
    npy_intp *doc_coordinates, int doc_depth, npy_intp offset);


static bool debug_mode = false;

static void
debug(char* message, PyObject* object, bson_t* doc)
{
    if (debug_mode) {
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


static void
init_debug_mode(void)
{
    debug_mode = (NULL != getenv("BSON_NUMPY_DEBUG"));
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
        bson_iter_t *bsonit, PyArrayObject *ndarray, PyArray_Descr *dtype, parsed_dtype_t *parsed,
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


static int
_load_array_from_bson(bson_iter_t *bsonit, PyArrayObject *ndarray,
                      PyArray_Descr *dtype, parsed_dtype_t *parsed, npy_intp *coordinates,
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
    PyArray_Descr* subdtype = dtype;
    int (*load_func)(bson_iter_t*, PyArrayObject*, PyArray_Descr*, parsed_dtype_t *,
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
        int ret = (*load_func)(&sub_it, ndarray, subdtype, parsed,
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


static int
_load_document_from_bson(
        bson_t *document, PyArrayObject *ndarray, PyArray_Descr *dtype, parsed_dtype_t *parsed,
        npy_intp *array_coordinates, int array_depth,
        npy_intp *doc_coordinates, int doc_depth, npy_intp offset) {

    parsed_dtype_t *parsed_child;
    bson_iter_t bsonit;
    PyArray_Descr *sub_dtype;  /* TODO: delete */
    PyObject *sub_dtype_obj, *fields, *value;
    Py_ssize_t i;
    int sub_i;

    if (parsed->node_type == DTYPE_NESTED) {
        for (i = 0; i < parsed->n_children; i++) {
            parsed_child = parsed->children[i];

            /* Have found another layer of nested document */
            doc_coordinates[doc_depth] = i;

            fields = dtype->fields;
            value = PyDict_GetItemString(fields, parsed_child->field_name);
            sub_dtype_obj = PyTuple_GET_ITEM(value, 0);
            PyArray_DescrConverter(sub_dtype_obj, &sub_dtype);

            /* Get subdocument from BSON document */
            bson_iter_init(&bsonit, document);
            if (!bson_iter_find(&bsonit, parsed_child->field_name)) {
                PyErr_Format(
                        BsonNumpyError,
                        "document does not match dtype, missing key \"%s\"",
                        parsed_child->field_name);
                return 0;
            }

            if (parsed_child->node_type == DTYPE_ARRAY) {
                Py_ssize_t number_dimensions = parsed_child->n_dimensions;

                /* If the current key's value is a subarray */
                array_coordinates[array_depth] = i;
                PyObject* shape = sub_dtype->subarray->shape;
                if (!PyTuple_Check(shape)) {
                    debug("invalid dtype->subarray->shape", shape, NULL);
                    PyErr_SetString(BsonNumpyError,
                                    "dtype argument had invalid subarray");
                    return 0;
                }

                /* Index into ndarray with array_coordinates */
                PyArrayObject* subndarray;
                void* subarray_ptr = PyArray_GetPtr(ndarray, array_coordinates);
                PyObject* subarray_tuple = PyArray_GETITEM(
                        ndarray, subarray_ptr);

                /* Indexing into ndarray with named fields returns a tuple
                 * that is nested as many levels as there are fields */
                for (sub_i = 0; sub_i < doc_depth + 1; sub_i++) {
                    npy_intp index = doc_coordinates[sub_i];
                    subarray_tuple = PyTuple_GET_ITEM(subarray_tuple, index);
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

                if (!_load_array_from_bson(&bsonit, subndarray, sub_dtype, parsed,
                                           new_coordinates, 0, 0)) {
                    /* error set by load_array_from_bson */
                    return 0;
                }

            } else if (sub_dtype->fields && sub_dtype->fields != Py_None) {
                /* If the current key's value is a subdocument */
                bson_t sub_document;
                uint32_t document_len;
                const uint8_t *document_buffer;
                bool r;

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

                r = bson_init_static(&sub_document,
                                     document_buffer,
                                     document_len);

                if (!r) {
                    PyErr_SetString(BsonNumpyError,
                                    "document from sequence failed validation");
                }

                if (!_load_document_from_bson(&sub_document, ndarray, sub_dtype, parsed_child,
                                              array_coordinates, array_depth,
                                              doc_coordinates, doc_depth + 1,
                                              offset + parsed_child->offset)) {
                    /* error set by _load_document_from_bson */
                    return 0;
                }
            } else {
                /* If the current key's value is a leaf */
                if (!_load_scalar_from_bson(&bsonit, ndarray, sub_dtype, parsed_child,
                                            array_coordinates, array_depth,
                                            offset + parsed_child->offset)) {
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
    PyObject *iterable_obj = NULL;
    PyObject *iterator_obj = NULL;
    PyObject *binary_doc = NULL;
    PyArray_Descr *dtype = NULL;
    PyArrayObject *ndarray = NULL;
    parsed_dtype_t *parsed_dtype = NULL;

    int num_documents;
    int number_dimensions = 1;
    npy_intp *array_coordinates = NULL;
    size_t err_offset;
    int row = 0;
    npy_intp dimension_lengths[100];
    npy_intp doc_coordinates[100];

    if (!PyArg_ParseTuple(args, "OO&i", &iterator_obj, PyArray_DescrConverter,
                          &dtype, &num_documents)) {
        return NULL;
    }

    if (PyIter_Check(iterator_obj)) {
        Py_INCREF(iterator_obj);
    } else {
        /* it's not an iterator, maybe it's iterable? */
        iterable_obj = iterator_obj;
        iterator_obj = PyObject_GetIter(iterable_obj);
        if (!iterator_obj) {
            PyErr_SetString(PyExc_TypeError,
                            "sequence_to_ndarray requires an iterator");
            Py_DECREF(dtype);
            goto done;
        }
    }

    if (num_documents < 0) {
        PyErr_SetString(BsonNumpyError,
                        "count argument was negative");
        Py_DECREF(dtype);
        goto done;
    }

    parsed_dtype = parsed_dtype_new(dtype, NULL);
    if (!parsed_dtype) {
        goto done;
    }

    dimension_lengths[0] = num_documents;

    /* This function steals a reference to dtype */
    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0);
    if (!array_obj) {
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
        if (!PyBytes_Check(binary_doc)) {
            PyErr_SetString(PyExc_TypeError,
                            "sequence_to_ndarray requires sequence of bytes"
                                " objects");
            goto done;
        }

        /* Get BSON document */
        const char *bytes_str = PyBytes_AS_STRING(binary_doc);
        Py_ssize_t bytes_len = PyBytes_GET_SIZE(binary_doc);
        bson_t document;

        bool r = bson_init_static(&document, (uint8_t *) bytes_str, bytes_len);
        if (!r) {
            PyErr_SetString(BsonNumpyError,
                            "document from sequence failed validation");
            goto done;
        }

        /* current_depth = 1 because layer 0 is the whole sequence */
        if (!_load_document_from_bson(&document, ndarray, PyArray_DTYPE(ndarray), parsed_dtype,
                                      array_coordinates, 1,
                                      doc_coordinates, 0, 0)) {
            /* error set by _load_document_from_bson */
            goto done;
        }

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
    Py_XDECREF(iterator_obj);
    free(array_coordinates);
    parsed_dtype_destroy(parsed_dtype);

    if (PyErr_Occurred()) {
        Py_XDECREF(array_obj);
        return NULL;
    }

    return array_obj;
}

/* Stub */
static PyObject *
ndarray_to_sequence(PyObject *self, PyObject *args)
{
    PyObject *array_obj;
    PyArrayObject *ndarray;

    if (!PyArg_ParseTuple(args, "O", &array_obj)) {
        return NULL;
    }

    /* Convert array */
    if (!PyArray_Check(array_obj)) {
        PyErr_SetString(BsonNumpyError, "sequence_to_ndarray requires a numpy.ndarray");
        return NULL;
    }
    if (!PyArray_OutputConverter(array_obj, &ndarray)) {
        PyErr_SetString(BsonNumpyError, "invalid ndarray passed into sequence_to_ndarray");
        return NULL;
    }

    return PyTuple_New(0);
}


static PyMethodDef BsonNumpyMethods[] = {{"ndarray_to_sequence", ndarray_to_sequence, METH_VARARGS, "Convert an ndarray into a iterator of BSON documents"},
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

    init_debug_mode();
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

    init_debug_mode();
    import_array();
}


#endif
