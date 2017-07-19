#include "bsonnumpy.h"
#include "bsonnumpy_hashtable.h"
#include "bsonnumpy_field_order.h"



static PyObject *BsonNumpyError;

typedef enum
{
    DTYPE_NESTED, /* like np.dtype([('a', np.int64), ('b', np.double)]) */
    DTYPE_SCALAR, /* like np.int64 */
    DTYPE_ARRAY,  /* like np.dtype('3i') */
} node_type_t;


typedef struct _parsed_dtype_t
{
    node_type_t node_type;
    char *field_name;
    size_t field_len;
    char *repr;
    Py_ssize_t offset;
    char kind;

    /* for scalars */
    int elsize;

    /* for nested types */
    struct _parsed_dtype_t **children;
    Py_ssize_t n_children;
    field_order_t field_order;
    bool field_order_valid;
    bool *child_fields_seen;
    hash_table_t table;

    /* for sub-array types */
    Py_ssize_t n_dimensions;
    long *dims;
} parsed_dtype_t;


static parsed_dtype_t *
parse_array_dtype(PyArray_Descr *dtype, char *field_name);

static parsed_dtype_t *
parse_nested_dtype(PyArray_Descr *dtype, char *field_name);

static parsed_dtype_t *
parse_scalar_dtype(PyArray_Descr *dtype, char *field_name);

static void
parsed_dtype_destroy(parsed_dtype_t *parsed);


static parsed_dtype_t *
parsed_dtype_new(node_type_t node_type, PyArray_Descr *dtype, char *field_name)
{
    PyObject *repr;
    parsed_dtype_t *parsed;
#if PY_MAJOR_VERSION >= 3
    PyObject *s;
#endif

    repr = PyObject_Repr((PyObject *) dtype);
    parsed = bson_malloc0(sizeof(parsed_dtype_t));

    parsed->node_type = node_type;
    parsed->field_name = field_name;  /* take ownership */
    parsed->field_len = field_name ? strlen(field_name) : 0;
#if PY_MAJOR_VERSION >= 3
    s = PyUnicode_AsEncodedString(repr, "utf-8", "ignore");
    parsed->repr = bson_strdup(PyBytes_AS_STRING(s));
    Py_DECREF(s);
#else
    parsed->repr = bson_strdup(PyString_AS_STRING(repr));
#endif
    Py_DECREF(repr);

    return parsed;
}


static parsed_dtype_t *
parse_dtype(PyArray_Descr *dtype, char *field_name)
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
parse_array_dtype(PyArray_Descr *dtype, char *field_name)
{
    parsed_dtype_t *parsed;
    Py_ssize_t i;
    PyObject *shape;
    PyObject *dim;

    parsed = parsed_dtype_new(DTYPE_ARRAY, dtype, field_name);
    parsed->elsize = dtype->subarray->base->elsize;
    parsed->kind = dtype->subarray->base->kind;
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
parse_nested_dtype(PyArray_Descr *dtype, char *field_name)
{
    parsed_dtype_t *parsed;
    PyObject *fields = NULL;
    PyObject *key = NULL;
    PyObject *unicode_key = NULL;
    char *key_str;
    PyObject *value = NULL;
    PyObject *ordered_names = NULL;
    PyObject *sub_dtype_obj = NULL;
    PyArray_Descr *sub_dtype;
    Py_ssize_t i;

    parsed = parsed_dtype_new(DTYPE_NESTED, dtype, field_name);
    fields = dtype->fields;
    ordered_names = PyArray_FieldNames(fields);
    parsed->n_children = PyTuple_Size(ordered_names);
    parsed->children = bson_malloc0(
        parsed->n_children * sizeof(parsed_dtype_t *));

    table_init(&parsed->table, parsed->n_children);
    field_order_init(&parsed->field_order, (size_t) parsed->n_children);
    parsed->field_order_valid = false;
    parsed->child_fields_seen = bson_malloc0(parsed->n_children * sizeof(bool));

    for (i = 0; i < parsed->n_children; i++) {
        key = PyTuple_GET_ITEM(ordered_names, i);
        value = PyDict_GetItem(fields, key);
        if (PyUnicode_Check(key)) {
            unicode_key = PyUnicode_AsASCIIString(key);
            if (!unicode_key) {
                PARSE_FAIL;
            }

            key_str = bson_strdup(PyBytes_AsString(unicode_key));
            Py_DECREF(unicode_key);
        } else {
            key_str = bson_strdup(PyBytes_AsString(key));
        }

        if (!key_str) {
            PARSE_FAIL;
        }

        sub_dtype_obj = PyTuple_GET_ITEM(value, 0);
        if (!PyArray_DescrConverter(sub_dtype_obj, &sub_dtype)) {
            PARSE_FAIL;
        }

        parsed->children[i] = parse_dtype(sub_dtype, key_str);
        if (!parsed->children[i]) {
            PARSE_FAIL;
        }

        parsed->children[i]->offset = PyLong_AsLong(PyTuple_GET_ITEM(value, 1));
        table_insert(&parsed->table, key_str, i);
    }

done:
    Py_XDECREF(ordered_names);
    Py_XDECREF(sub_dtype_obj);

    return parsed;
}


static parsed_dtype_t *
parse_scalar_dtype(PyArray_Descr *dtype, char *field_name)
{
    parsed_dtype_t *parsed;

    parsed = parsed_dtype_new(DTYPE_SCALAR, dtype, field_name);
    parsed->elsize = dtype->elsize;
    parsed->kind = dtype->kind;

    return parsed;
}


static void
parsed_dtype_destroy(parsed_dtype_t *parsed)
{
    Py_ssize_t i;

    if (parsed) {
        if (parsed->children) {
            for (i = 0; i < parsed->n_children; i++) {
                parsed_dtype_destroy(parsed->children[i]);
            }

            bson_free(parsed->children);
            bson_free(parsed->child_fields_seen);
        }

        if (parsed->dims) {
            bson_free(parsed->dims);
        }

        table_destroy(&parsed->table);
        field_order_destroy(&parsed->field_order);
        bson_free(parsed->field_name);
        bson_free(parsed->repr);
        bson_free(parsed);
    }
}


static int
_load_scalar_from_bson(
    bson_iter_t *bsonit, PyArrayObject *ndarray, parsed_dtype_t *parsed,
    npy_intp *coordinates, int current_depth, long offset);


static int
_load_document_from_bson(
    bson_t *document, PyArrayObject *ndarray, parsed_dtype_t *parsed,
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
_bsonnumpy_type_err(bson_type_t bson_type, parsed_dtype_t *parsed,
                    const char *msg)
{
    PyErr_Format(BsonNumpyError, "cannot convert %s to %s: %s",
                 _bson_type_name(bson_type), parsed->repr, msg);
}


static int
_load_utf8_from_bson(const bson_value_t *value, void *dst,
                     parsed_dtype_t *parsed)
{
    npy_intp itemsize;
    npy_intp bson_item_len;

    if (parsed->kind != 'S' && parsed->kind != 'U' && parsed->kind != 'V') {
        _bsonnumpy_type_err(value->value_type, parsed, "use 'S' 'U' or 'V'");
        return 0;
    }

    bson_item_len = value->value.v_utf8.len;
    itemsize = parsed->elsize;

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
                       parsed_dtype_t *parsed)
{
    npy_intp itemsize;
    npy_intp bson_item_len;

    if (parsed->kind != 'S' && parsed->kind != 'U' && parsed->kind != 'V') {
        _bsonnumpy_type_err(value->value_type, parsed, "use 'S' 'U' or 'V'");
        return 0;
    }

    bson_item_len = value->value.v_binary.data_len;
    itemsize = parsed->elsize;

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
_load_oid_from_bson(const bson_value_t *value, void *dst, parsed_dtype_t *parsed)
{
    if ((parsed->kind != 'S' && parsed->kind != 'V') || parsed->elsize != 12) {
        _bsonnumpy_type_err(value->value_type, parsed, "use 'S12' or 'V12'");
        return 0;
    }

    memcpy(dst, value->value.v_oid.bytes, sizeof value->value.v_oid.bytes);
    return 1;
}


static int
_load_int32_from_bson(const bson_value_t *value, void *dst,
                      parsed_dtype_t *parsed)
{
    if (parsed->kind != 'i' && parsed->kind != 'u') {
        _bsonnumpy_type_err(value->value_type, parsed, "use an integer type");
        return 0;
    }

    if ((size_t) parsed->elsize < sizeof(int32_t)) {
        _bsonnumpy_type_err(value->value_type, parsed,
                            "use at least a 32-bit integer type");
        return 0;
    }

    memcpy(dst, &value->value.v_int32, sizeof value->value.v_int32);
    return 1;
}


static int
_load_int64_from_bson(const bson_value_t *value, void *dst,
                      parsed_dtype_t *parsed)
{
    if (parsed->kind != 'i' && parsed->kind != 'u') {
        _bsonnumpy_type_err(value->value_type, parsed, "use an integer type");
        return 0;
    }

    if ((size_t) parsed->elsize < sizeof(int64_t)) {
        _bsonnumpy_type_err(value->value_type, parsed,
                            "use at least a 64-bit integer type");
        return 0;
    }

    memcpy(dst, &value->value.v_int64, sizeof value->value.v_int64);
    return 1;
}


static int
_load_double_from_bson(const bson_value_t *value, void *dst,
                       parsed_dtype_t *parsed)
{
    if (parsed->kind != 'f') {
        _bsonnumpy_type_err(value->value_type, parsed,
                            "use a floating point type");
        return 0;
    }

    if ((size_t) parsed->elsize < sizeof(int64_t)) {
        _bsonnumpy_type_err(value->value_type, parsed,
                            "use at least a 64-bit floating point type");
    }

    memcpy(dst, &value->value.v_double, sizeof value->value.v_double);
    return 1;
}


static int
_load_bool_from_bson(const bson_value_t *value, void *dst,
                     parsed_dtype_t *parsed)
{
    if (parsed->kind != 'b' && parsed->kind != 'i' && parsed->kind != 'u') {
        _bsonnumpy_type_err(value->value_type, parsed,
                            "use 'b' or an integer type");
    }

    memcpy(dst, &value->value.v_bool, sizeof value->value.v_bool);
    return 1;
}


static int
_load_scalar_from_bson(
        bson_iter_t *bsonit, PyArrayObject *ndarray, parsed_dtype_t *parsed,
        npy_intp *coordinates, int current_depth, long offset)
{
    void *pointer;
    const bson_value_t *value;

    value = bson_iter_value(bsonit);
    pointer = PyArray_GetPtr(ndarray, coordinates) + offset;

    switch (value->value_type) {
        case BSON_TYPE_DOUBLE:
            return _load_double_from_bson(value, pointer, parsed);
        case BSON_TYPE_UTF8:
            return _load_utf8_from_bson(value, pointer, parsed);
        case BSON_TYPE_BINARY:
            return _load_binary_from_bson(value, pointer, parsed);
        case BSON_TYPE_OID:
            return _load_oid_from_bson(value, pointer, parsed);
        case BSON_TYPE_BOOL:
            return _load_bool_from_bson(value, pointer, parsed);
        case BSON_TYPE_DATE_TIME:
            return _load_int64_from_bson(value, pointer, parsed);
        case BSON_TYPE_INT32:
            return _load_int32_from_bson(value, pointer, parsed);
        case BSON_TYPE_INT64:
            return _load_int64_from_bson(value, pointer, parsed);

        default:
            PyErr_Format(BsonNumpyError, "unsupported BSON type: %s",
                         _bson_type_name(value->value_type));
            return false;
    }
}


static int
_load_array_from_bson(bson_iter_t *bsonit, PyArrayObject *ndarray,
                      parsed_dtype_t *parsed, npy_intp *coordinates,
                      int current_depth, long offset)
{
    long expected_length;
    Py_ssize_t dimensions;
    bson_iter_t sub_it;
    long i;

    /* Type and length checks */
    if (!BSON_ITER_HOLDS_ARRAY(bsonit)) {
        PyErr_SetString(BsonNumpyError,
                        "invalid document: expected list from dtype,"
                                " got other type");
        return 0;
    }

    expected_length = parsed->dims[current_depth];
    dimensions = parsed->n_dimensions;

    bson_iter_recurse(bsonit, &sub_it);
    int count = 0;
    while (bson_iter_next(&sub_it)) {
        count++;
    }

    if (expected_length != count) {
        PyErr_SetString(BsonNumpyError,
                        "invalid document: array is of incorrect length");
        return 0;
    }

    /* Load data into array */
    bson_iter_recurse(bsonit, &sub_it);
    int (*load_func)(bson_iter_t*, PyArrayObject*, parsed_dtype_t *,
                     npy_intp*, int, long) = &_load_array_from_bson;
    if(current_depth == dimensions - 1) {
        load_func = &_load_scalar_from_bson;
    }

    i = 0;
    while (bson_iter_next(&sub_it)) {
        long new_offset = offset;
        if (i > expected_length) {
            PyErr_Format(BsonNumpyError,
                         "invalid document: array is longer than expected"
                         " length of %ld", expected_length);
            return 0;
        }

        if (current_depth < dimensions) {
            coordinates[current_depth] = i;
        } else {
            PyErr_SetString(BsonNumpyError, "TODO: unhandled case");
            return 0;
        }

        int ret = (*load_func)(&sub_it, ndarray, parsed, coordinates,
                               current_depth + 1, new_offset);
        if (ret == 0) {
            /* Error set by loading function */
            return 0;
        };

        i++;
    }

    if (i < expected_length) {
        PyErr_Format(BsonNumpyError,
                     "invalid document: array is shorter than expected"
                     " length of %ld, got %ld", expected_length, i);

    }

    /* Reset the rest of the coordinates to zero */
    for (i = current_depth; i < dimensions; i++) {
        coordinates[i] = 0;
    }
    return 1;
}


static int
_load_element_from_bson(
    bson_iter_t *bsonit, PyArrayObject *ndarray, parsed_dtype_t *parsed,
    npy_intp *array_coordinates, int array_depth,
    npy_intp *doc_coordinates, int doc_depth, npy_intp offset)
{
    int sub_i;

    if (parsed->node_type == DTYPE_ARRAY) {
        Py_ssize_t number_dimensions = parsed->n_dimensions;

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
            1 + (size_t) number_dimensions, sizeof(npy_intp));

        if (!_load_array_from_bson(bsonit, subndarray, parsed,
                                   new_coordinates, 0, 0)) {
            /* error set by load_array_from_bson */
            return 0;
        }
    } else if (parsed->node_type == DTYPE_NESTED) {
        /* If the current key's value is a subdocument */
        bson_t sub_document;
        uint32_t document_len;
        const uint8_t *document_buffer;
        bool r;

        if (!BSON_ITER_HOLDS_DOCUMENT(bsonit)) {
            PyErr_SetString(BsonNumpyError,
                            "invalid document: expected subdoc "
                                "from dtype, got other type");
            return 0;
        }
        bson_iter_document(bsonit, &document_len,
                           &document_buffer);

        r = bson_init_static(&sub_document,
                             document_buffer,
                             document_len);

        if (!r) {
            PyErr_SetString(BsonNumpyError,
                            "document from sequence failed validation");
        }

        if (!_load_document_from_bson(&sub_document, ndarray, parsed,
                                      array_coordinates, array_depth,
                                      doc_coordinates, doc_depth + 1,
                                      offset + parsed->offset)) {
            /* error set by _load_document_from_bson */
            return 0;
        }
    } else {
        /* If the current key's value is a leaf */
        if (!_load_scalar_from_bson(bsonit, ndarray, parsed,
                                    array_coordinates, array_depth,
                                    offset + parsed->offset)) {
            /* Error set by _load_scalar_from_bson */
            return 0;
        }
    }

    return 1;
}


static long invalidations = 0;


static int
_load_document_from_bson(
        bson_t *document, PyArrayObject *ndarray, parsed_dtype_t *parsed,
        npy_intp *array_coordinates, int array_depth,
        npy_intp *doc_coordinates, int doc_depth, npy_intp offset)
{
    parsed_dtype_t *parsed_child;
    bson_iter_t bsonit;
    size_t bson_index;
    const field_order_elem_t *elem = NULL;
    const char *next_key;
    const char *key;
    Py_ssize_t i;
    int sub_i;

    if (parsed->node_type != DTYPE_NESTED) {
        /* Top-level dtype did not have named fields */
        PyErr_SetString(BsonNumpyError, "dtype must include field names, like"
            " dtype([('fieldname', numpy.int)])");
        return 0;
    }

    if (!bson_iter_init(&bsonit, document)) {
        PyErr_SetString(BsonNumpyError, "bson_iter_init failed");
        return 0;
    }

    memset(parsed->child_fields_seen, 0, parsed->n_children * sizeof(bool));

    bson_index = 0;

    while (true) {
        if (parsed->field_order_valid) {
            if (!field_order_match(&parsed->field_order, bson_index, &bsonit,
                                   &elem)) {
                parsed->field_order_valid = false;
                invalidations++;
            } else {
                i = elem->dtype_index;
                if (!bson_iter_next_with_len(&bsonit,
                                             (uint32_t)elem->key.len)) {
                    /* document completed */
                    break;
                }

                if (i == EMPTY) {
                    goto next;
                }
            }
        }

        if (!parsed->field_order_valid) {
            if (!bson_iter_next(&bsonit)) {
                /* document completed */
                break;
            }

            key = bson_iter_key(&bsonit);
            i = table_lookup(&parsed->table, key);
            field_order_set(&parsed->field_order, bson_index, key, i);

            if (i == EMPTY) {
                /* ignore extra fields in the document not in the dtype */
                goto next;
            }
        }

        parsed_child = parsed->children[i];
        parsed->child_fields_seen[i] = true;

        /* Have found another layer of nested document */
        doc_coordinates[doc_depth] = i;

        if (!_load_element_from_bson(&bsonit, ndarray, parsed_child,
                                     array_coordinates, array_depth,
                                     doc_coordinates, doc_depth, offset)) {
            return 0;
        }

next:
        bson_index++;
    }

done:
    for (i = 0; i < parsed->n_children; i++) {
        if (!parsed->child_fields_seen[i]) {
            PyErr_Format(BsonNumpyError,
                         "document does not match dtype, missing key \"%s\"",
                         parsed->children[i]->field_name);
            return 0;
        }
    }

    parsed->field_order_valid = true;

    return 1;
}

#define INVALID(_msg) do {                              \
    PyErr_Format(                                       \
        BsonNumpyError,                                 \
        "document from sequence failed validation: %s", \
        _msg);                                          \
    goto done;                                          \
} while (0)


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

    parsed_dtype = parse_dtype(dtype, NULL);
    if (!parsed_dtype) {
        Py_DECREF(dtype);
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
        const char *bytes_str;
        Py_ssize_t bytes_len;
        Py_ssize_t pos = 0;

        if (!PyBytes_Check(binary_doc)) {
            PyErr_SetString(PyExc_TypeError,
                            "sequence_to_ndarray requires sequence of bytes"
                                " objects");
            goto done;
        }

        bytes_str = PyBytes_AS_STRING(binary_doc);
        bytes_len = PyBytes_GET_SIZE(binary_doc);

        if (bytes_len < 5) {
            INVALID("must be at least 5 bytes");
        }

        while (pos < bytes_len) {
            uint32_t len_le;
            uint32_t len;
            bson_t document;

            memcpy (&len_le, bytes_str + pos, sizeof(len_le));
            len = BSON_UINT32_FROM_LE (len_le);
            if (len > (uint32_t) bytes_len) {
                INVALID("incomplete batch");
            }

            bool r = bson_init_static(&document, (uint8_t *) (bytes_str + pos),
                                      len);
            if (!r) {
                INVALID("incorrect length");
            }

            /* current_depth = 1 because layer 0 is the whole sequence */
            if (!_load_document_from_bson(&document, ndarray, parsed_dtype,
                                          array_coordinates, 1, doc_coordinates,
                                          0, 0)) {
                /* error set by _load_document_from_bson */
                goto done;
            }

            array_coordinates[0] = ++row;
            if (row >= num_documents) {
                goto check_row_count;
            }

            pos += len;
        }
    }

check_row_count:
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

static PyObject *
invalidation_count(PyObject *self, PyObject *args)
{
    return PyLong_FromLong(invalidations);
}


static PyMethodDef BsonNumpyMethods[] = {{"ndarray_to_sequence", ndarray_to_sequence, METH_VARARGS, "Convert an ndarray into a iterator of BSON documents"},
                                         {"sequence_to_ndarray", sequence_to_ndarray, METH_VARARGS, "Convert an iterator containing BSON documents into an ndarray"},
                                         {"_invalidation_count", invalidation_count, METH_NOARGS, "Internal diagnostic"},
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
