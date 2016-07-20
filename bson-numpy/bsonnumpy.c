#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//#include <numpy/arrayobject.h>
//#include <numpy/npy_common.h>
#include <numpy/ndarrayobject.h>

#include "bson.h"

static PyObject* BsonNumpyError;

static PyObject*
ndarray_to_bson(PyObject* self, PyObject* args) // Stub to test passing ndarrays.
{
    PyObject* dtype_obj;
    PyObject* array_obj;
    PyArray_Descr* dtype;
    PyArrayObject* array;
    if (!PyArg_ParseTuple(args, "OO", &dtype_obj, &array_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    // Convert dtype
    if (!PyArray_DescrCheck(dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
        return NULL;
    }

    // Convert array
    if (!PyArray_Check(array_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_OutputConverter(array_obj, &array)) {
        PyErr_SetString(BsonNumpyError, "bad array type");
        return NULL;
    }
    return Py_BuildValue("");
}
/*
    |Straightforward
    case BSON_TYPE_INT64
    case BSON_TYPE_INT32
    case BSON_TYPE_DOUBLE
    case BSON_TYPE_BOOL

    case BSON_TYPE_OID
    case BSON_TYPE_UTF8
    case BSON_TYPE_BINARY
    case BSON_TYPE_SYMBOL
    case BSON_TYPE_CODE
    case BSON_TYPE_DATE_TIME

    case BSON_TYPE_DOCUMENT

    |Totally different case
    case BSON_TYPE_ARRAY:

    |With issues to work out:
    case BSON_TYPE_TIMESTAMP
    case BSON_TYPE_REGEX

    |Not clear what to do, maybe make a flexible type?
    case BSON_TYPE_DBPOINTER
    case BSON_TYPE_CODEWSCOPE

    Probably error, no bson_iter_ for
    case BSON_TYPE_UNDEFINED
    case BSON_TYPE_NULL
    case BSON_TYPE_MAXKEY
    case BSON_TYPE_MINKEY
    case BSON_TYPE_EOD
 */



static int _load_scalar(bson_iter_t* bsonit,
                       npy_intp* coordinates,
                       PyArrayObject* ndarray,
                       npy_intp depth,
                       npy_intp number_dimensions,
                       PyArray_Descr* dtype) {

        bson_iter_t sub_it;
        int itemsize = PyArray_DESCR(ndarray)->elsize;;
        int len = itemsize;
        int success = 0;
        int copy = 1;
        printf("in load_scalar, depth=%i, coordinates=[", (int)depth);
        for(int i=0;i<number_dimensions;i++) {
            printf("%i,", (int)coordinates[i]);
        }
        printf("] + DTYPE=");
        PyObject_Print((PyObject*)dtype, stdout, 0);
        printf("\n");

        if(BSON_ITER_HOLDS_ARRAY(bsonit)) {
            printf("GOT ARRAY\n");
            bson_iter_recurse(bsonit, &sub_it);

            int i = 0;
            while( bson_iter_next(&sub_it) ) { // TODO: loop on ndarray not on bson, goign to have to pass dimensions from tuple
                coordinates[depth + 1] = i;
                //TODO: need to get sub dtype, is base type enough?
                printf("SUBARRAY:");
                PyObject_Print((PyObject*)dtype->subarray->base, stdout, 0);
                printf("\n");
                _load_scalar(&sub_it, coordinates, ndarray, depth+1, number_dimensions, dtype->subarray->base);
                i++;
            }
            return 1; // TODO: check result of _load_scalar
        }
        void* pointer = PyArray_GetPtr(ndarray, coordinates);
        const bson_value_t* value = bson_iter_value(bsonit);
        void* data_ptr = (void*)&value->value;
        printf("BSON TYPE:%i\n", value->value_type);
        switch(value->value_type) {
        case BSON_TYPE_UTF8:
            data_ptr = value->value.v_utf8.str; // Unclear why using value->value doesn't work
            len = value->value.v_utf8.len;
            printf("\tGOT STRING=%s\n", data_ptr);
                printf("SUBARRAY_STRING:");
                PyObject_Print((PyObject*)dtype->subarray->base, stdout, 0);
                printf("\n");
            break;
        case BSON_TYPE_BINARY:
            data_ptr = value->value.v_binary.data;
            len = value->value.v_binary.data_len;
            break;
        case BSON_TYPE_SYMBOL: // deprecated
            data_ptr = value->value.v_symbol.symbol;
            len = value->value.v_symbol.len;
            break;
        case BSON_TYPE_CODE:
            data_ptr = value->value.v_code.code;
            len = value->value.v_code.code_len;
            break;
        case BSON_TYPE_DOCUMENT:
            // TODO: what about V lengths that are longer than the doc?
            data_ptr = value->value.v_doc.data;
            len = value->value.v_doc.data_len;
            break;

        // Have to special case for timestamp and regex bc there's no np equiv
        case BSON_TYPE_TIMESTAMP:
            memcpy(pointer, &value->value.v_timestamp.timestamp, sizeof(int32_t));
            memcpy((pointer+sizeof(int32_t)), &value->value.v_timestamp.increment, sizeof(int32_t));
            copy = 0;
            break;
        case BSON_TYPE_REGEX:
            len = (int)strlen(value->value.v_regex.regex);
            memcpy(pointer, value->value.v_regex.regex, len);
            memset(pointer + len, '\0', 1);
            memcpy(pointer + len + 1, value->value.v_regex.options, (int)strlen(value->value.v_regex.options));
            len = len + (int)strlen(value->value.v_regex.options) + 1;
            copy = 0;
            break;


        default:
        printf("GOT OTHER=%f\n", value->value.v_double);

        }

        if(copy && len == itemsize) {
            PyObject* data = PyArray_Scalar(data_ptr, dtype, NULL);
            printf("ITEM=");
            PyObject_Print(data, stdout, 0);
            printf("\n");
            success = PyArray_SETITEM(ndarray, pointer, data);
    //        Py_INCREF(data);
        }
        else if(copy) {
            // Dealing with data that's shorter than the array datatype, so we can't read using the macros.
            if(len > itemsize) {
                len = itemsize; // truncate data that's too big
            }
            memcpy(pointer, data_ptr, len);
            memset(pointer + len, '\0', itemsize - len);

        }
        printf("AT END OF LOAD_SCALAR:");
        PyObject_Print((PyObject*)ndarray, stdout, 0);
        printf("\n");
        return success;
}

////TODO: repeated code
//static PyObject* _init_ndarray(int length, PyArray_Descr* dtype, npy_intp* dimension_lengths)
//{
//    return NULL;
//}

static PyObject*
bson_to_ndarray(PyObject* self, PyObject* args)
{
    // Takes in a BSON byte string
    PyObject* binary_obj;
    PyObject* dtype_obj;
    PyObject *array_obj;
    const char* bytestr;
    PyArray_Descr* dtype;
    PyArrayObject* ndarray;
    Py_ssize_t bytes_len;
    Py_ssize_t number_dimensions = -1;
    npy_intp* dimension_lengths;
    bson_iter_t bsonit;
    bson_t* document;
    size_t err_offset;

    if (!PyArg_ParseTuple(args, "SO", &binary_obj, &dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    bytestr = PyBytes_AS_STRING(binary_obj);
    bytes_len = PyBytes_GET_SIZE(binary_obj);
    document = bson_new_from_data((uint8_t*)bytestr, bytes_len); // slower than what??? Also, is this a valid cast? TODO: free
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
     // TODO: validate in a reasonable way, now segfaults if bad
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return NULL;
    }
//    char* str = bson_as_json(document, (size_t*)&bytes_len);
//    printf("DOCUMENT: %s\n", str);

    // Convert dtype
    if (!PyArray_DescrCheck(dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
        return NULL;
    }

    bson_iter_init(&bsonit, document);
    dimension_lengths = malloc(1* sizeof(npy_intp));
    dimension_lengths[0] = bson_count_keys(document);
    number_dimensions = 1;

    if(dtype->subarray != NULL) {
        PyObject *shape = dtype->subarray->shape;
        if(!PyTuple_Check(shape)) {
            PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
            return NULL;
        }
        number_dimensions = (int)PyTuple_Size(shape);
    }

//    printf("dtype_obj=");
//    PyObject_Print(dtype_obj, stdout, 0);
//    printf("\n");

    Py_INCREF(dtype);

    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0); // This function steals a reference to dtype?
//    printf("array_obj=");
//    PyObject_Print(array_obj, stdout, 0);
//    printf("\n");

    PyArray_OutputConverter(array_obj, &ndarray);


    npy_intp* coordinates = calloc(number_dimensions, sizeof(npy_intp));
    for(npy_intp i=0;i<dimension_lengths[0];i++) {
        bson_iter_next(&bsonit);
        coordinates[0] = i;
        int success = _load_scalar(&bsonit, coordinates, ndarray, 0, number_dimensions, dtype);
        if(success == -1) {
            PyErr_SetString(BsonNumpyError, "item failed to load");
            return NULL;
        }
    }

    free(dimension_lengths);
    free(document);
    free(coordinates);
//    Py_INCREF(array_obj);
    return array_obj;
}


static int validate_field_type(PyObject* np_type, bson_iter_t* bsonit) {
    // Create dict in Python because easiest
    return 1;

}

static int load_document(PyObject* binary_obj,
                         PyArray_Descr* dtype,
                         npy_intp* coordinates,
                         PyArrayObject* ndarray,
                         npy_intp depth,
                         npy_intp number_dimensions) {
    PyObject* fields, *key, *value = NULL;
    Py_ssize_t pos, bytes_len = 0;
    bson_t* document;
    bson_iter_t bsonit;
    size_t err_offset;
    const char* bytes_str;
    const char* key_str;

    bytes_str = PyBytes_AS_STRING(binary_obj);
    bytes_len = PyBytes_GET_SIZE(binary_obj);
    document = bson_new_from_data((uint8_t*)bytes_str, bytes_len);
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
     // TODO: validate in a reasonable way, now segfaults if bad
        printf("FAILED BSON_VALIDATE\n");
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return 0;
    }


    char* str = bson_as_json(document, (size_t*)&bytes_len);
    printf("DOCUMENT: %s\n", str);
    bson_iter_init(&bsonit, document);
    fields = dtype->fields; //A field is described by a tuple composed of another data- type-descriptor and a byte offset.
    printf("DTYPE->FIELDS:");
    PyObject_Print(fields, stdout, 0);
    printf("\n");

    if(fields==NULL) {
        printf("FIELDS IS NONE!!!!!!\n");
        // simple type
    }
    else {
        // Need to validate fields.
        if(!PyDict_Check(fields)) {
            PyErr_SetString(BsonNumpyError, "dtype.fields was not a dictionary?");
            return 0;
        }
        pos = 0;
        int col = 0;
        int success;
        PyArray_Descr* sub_dtype;
        PyObject* sub_dtype_obj;

        while(PyDict_Next(fields, &pos, &key, &value)) { // for each column
            key_str = PyBytes_AS_STRING(key);
            if (!PyTuple_Check(value)) {
                PyErr_SetString(BsonNumpyError, "dtype in fields is not a tuple");
            }
            sub_dtype_obj = PyTuple_GetItem(value, 0);
            if (!PyArray_DescrConverter(sub_dtype_obj, &sub_dtype)) {
                PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
                return 0;
            }

            printf("FOUND KEY: ");
            PyObject_Print(key, stdout, 0);
            printf("|DTYPE: ");
            PyObject_Print(sub_dtype, stdout, 0);
            printf("\n");

            if(bson_iter_find(&bsonit, key_str)) {
                coordinates[1] = col;
                success = _load_scalar(&bsonit, coordinates, ndarray, depth, number_dimensions, sub_dtype);
                if(!success) {
                    PyErr_SetString(BsonNumpyError, "failed to load scalar");
                    return 0;
                }
            }
            else {
                printf("ERROR, KEY %s NT FOUND\n", key_str);
            }
            if(!validate_field_type(value, &bsonit)) {
                PyErr_SetString(BsonNumpyError, "field type was incorrect");
                return 0;
            }

        }

    }
    free(document);
    return 1;
}

static PyObject*
collection_to_ndarray(PyObject* self, PyObject* args) // Better name please! Collection/cursor both seem to specific to PyMongo.
{
    PyObject* iterator_obj;
    PyObject* dtype_obj;
    PyObject* array_obj;
    PyObject* binary_obj;

    PyArray_Descr* dtype;
    PyArrayObject* ndarray;

    int num_documents;
    Py_ssize_t number_dimensions;
    npy_intp* dimension_lengths;

    if (!PyArg_ParseTuple(args, "OOi", &iterator_obj, &dtype_obj, &num_documents)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if(!PyIter_Check(iterator_obj)) {
        PyErr_SetString(BsonNumpyError, "collection_to_ndarray expects an iterator");
        return NULL;
    }
    if (!PyArray_DescrCheck(dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
        return NULL;
    }
    printf("DTYPE:");
    PyObject_Print((PyObject*)dtype, stdout, 0);
    printf(" NUM_DOCS=%i\n", num_documents);

    dimension_lengths = malloc(1* sizeof(npy_intp));
    dimension_lengths[0] = num_documents;
    number_dimensions = 1;

    if(dtype->subarray != NULL) {
        PyObject *shape = dtype->subarray->shape;
        if(!PyTuple_Check(shape)) {
            PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
            return NULL;
        }
        number_dimensions = (int)PyTuple_Size(shape);
    }

    Py_INCREF(dtype);

    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0); // This function steals a reference to dtype?

    printf("ARRAY: ");
    PyObject_Print(array_obj, stdout, 0);
    printf("\n");

    PyArray_OutputConverter(array_obj, &ndarray);


    npy_intp* coordinates = calloc(1 + number_dimensions, sizeof(npy_intp));


    int row = 0;
    while((binary_obj = PyIter_Next(iterator_obj))) { // For each row
        if(load_document(binary_obj, dtype, coordinates,
                          ndarray, 0, 1 + number_dimensions) == 0) { // error set by load_document
            return NULL;
        }
        coordinates[0] = row++;
        // Insert into column
    }
    free(dimension_lengths);
    free(coordinates);
//    Py_INCREF(array_obj);
    return array_obj;

}


static PyMethodDef BsonNumpyMethods[] = {
    {"ndarray_to_bson", ndarray_to_bson, METH_VARARGS,
     "Convert an ndarray into a BSON byte string"},
    {"bson_to_ndarray", bson_to_ndarray, METH_VARARGS,
     "Convert BSON byte string into an ndarray"},
    {"collection_to_ndarray", collection_to_ndarray, METH_VARARGS,
     "Convert an iterator containing BSON documents into an ndarray"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initbsonnumpy(void)
{
    PyObject* m;

    m = Py_InitModule("bsonnumpy", BsonNumpyMethods);
    if (m == NULL)
        return;

    BsonNumpyError = PyErr_NewException("bsonnumpy.error", NULL, NULL);
    Py_INCREF(BsonNumpyError);
    PyModule_AddObject(m, "error", BsonNumpyError);

    import_array();
}


int
main(int argc, char* argv[])
{
    printf("RUNNING MAIN");
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initbsonnumpy();
}

