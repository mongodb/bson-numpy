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



static int _load_scalar(bson_iter_t* bsonit, // TODO: elsize won't work for flexible types
                       npy_intp* coordinates,
                       PyArrayObject* ndarray,
                       npy_intp depth,
                       npy_intp number_dimensions,
                       PyArray_Descr* dtype) {
        bson_iter_t sub_it;
        int itemsize = PyArray_DESCR(ndarray)->elsize;
        int bson_item_len = itemsize;
        int success = 0;
        int copy = 1;
        printf("\tin load_scalar, depth=%i, coordinates=[", (int)depth);
        for(int i=0;i<number_dimensions;i++) {
            printf("%i,", (int)coordinates[i]);
        }
        printf("] + DTYPE=");
        PyObject_Print((PyObject*)dtype, stdout, 0);
        printf("\n");

        void* pointer = PyArray_GetPtr(ndarray, coordinates);

        int dtype_num = dtype->type_num;
//        printf("DTYPE NUM=%i\n", dtype_num); // TODO: Use this for error checking

        if(BSON_ITER_HOLDS_ARRAY(bsonit)) {
            printf("\t\tBSON_ITER_HOLDS_ARRAY\n");
            // Get length of array
            bson_iter_recurse(bsonit, &sub_it);
            int count = 0;
            while( bson_iter_next(&sub_it)) {
                count++;
            }
            bson_iter_recurse(bsonit, &sub_it);
            if (count == 1) {
                bson_iter_next(&sub_it);
                printf("\t\tIGNORING ARRAY OF LEN 1\n");
                // Array is of length 1, therefore we treat it like a number
                return _load_scalar(&sub_it, coordinates, ndarray, depth,
                                    number_dimensions, dtype);
            } else {
                PyArray_Descr* base = dtype->subarray ? dtype->subarray->base :
                                      dtype; // TODO: I'm not sure why this is working for arrays of length 1 that are subarrays
                int i = 0;
                while( bson_iter_next(&sub_it) ) { // TODO: loop on ndarray not on bson, going to have to pass dimensions from tuple
                    coordinates[depth + 1] = i;
                    printf("\t\tSUBARRAY: new coordinates= [");
                    for(int i=0;i<number_dimensions;i++) {
                        printf("%i,", (int)coordinates[i]);
                    }
                    printf("], new dtype=");
                    PyObject_Print((PyObject*)base, stdout, 0);
                    printf("\n");
                    _load_scalar(&sub_it, coordinates, ndarray, depth+1, number_dimensions, base);
                    i++;
                }
                return 1; // TODO: check result of _load_scalar
          }
        }
        const bson_value_t* value = bson_iter_value(bsonit);
        void* data_ptr = (void*)&value->value;
//        printf("Switching on %i\n", value->value_type);
        switch(value->value_type) {
        case BSON_TYPE_UTF8:
            data_ptr = value->value.v_utf8.str; // Unclear why using value->value doesn't work
            bson_item_len = value->value.v_utf8.len;
            break;
        case BSON_TYPE_INT32:
            data_ptr = &value->value.v_int32;
            bson_item_len = sizeof (value->value.v_int32);
            break;
        case BSON_TYPE_INT64:
            data_ptr = &value->value.v_int64;
            bson_item_len = sizeof (value->value.v_int64);
            break;
        case BSON_TYPE_BINARY:
            data_ptr = value->value.v_binary.data;
            bson_item_len = value->value.v_binary.data_len;
            break;
        case BSON_TYPE_SYMBOL: // deprecated
            data_ptr = value->value.v_symbol.symbol;
            bson_item_len = value->value.v_symbol.len;
            break;
        case BSON_TYPE_CODE:
            data_ptr = value->value.v_code.code;
            bson_item_len = value->value.v_code.code_len;
            break;
        case BSON_TYPE_DOCUMENT:
            // TODO: what about V lengths that are longer than the doc?
            // TODO: check for flexible dtype with dtype.fields
            data_ptr = value->value.v_doc.data;
            bson_item_len = value->value.v_doc.data_len;
            break;

        // Have to special case for timestamp and regex bc there's no np equiv
        case BSON_TYPE_TIMESTAMP:
            memcpy(pointer, &value->value.v_timestamp.timestamp, sizeof(int32_t));
            memcpy((pointer+sizeof(int32_t)), &value->value.v_timestamp.increment, sizeof(int32_t));
            copy = 0;
            break;
        case BSON_TYPE_REGEX:
            bson_item_len = (int)strlen(value->value.v_regex.regex);
            memcpy(pointer, value->value.v_regex.regex, bson_item_len);
            memset(pointer + bson_item_len, '\0', 1);
            memcpy(pointer + bson_item_len + 1, value->value.v_regex.options, (int)strlen(value->value.v_regex.options));
            bson_item_len = bson_item_len + (int)strlen(value->value.v_regex.options) + 1;
            copy = 0;
            break;

        }

        if(copy && bson_item_len == itemsize) {
            PyObject* data = PyArray_Scalar(data_ptr, dtype, NULL);
            success = PyArray_SETITEM(ndarray, pointer, data);
    //        Py_INCREF(data);
        }
        else if(copy) {
            // Dealing with data that's shorter than the array datatype, so we can't read using the macros.
            if(bson_item_len > itemsize) {
                bson_item_len = itemsize; // truncate data that's too big
            }
            memcpy(pointer, data_ptr, bson_item_len);
            memset(pointer + bson_item_len, '\0', itemsize - bson_item_len);
            success = 1;

        }
//        printf("\t\tEND OF LOAD SCALAR:");
//        PyObject_Print((PyObject*)ndarray, stdout, 0);
//        printf("\n");
        return success;
}


static PyObject*
bson_to_ndarray(PyObject* self, PyObject* args)
{
    // Takes in a BSON byte string and a dtype
    PyObject* binary_doc;
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

    if (!PyArg_ParseTuple(args, "SO", &binary_doc, &dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    bytestr = PyBytes_AS_STRING(binary_doc);
    bytes_len = PyBytes_GET_SIZE(binary_doc);
    document = bson_new_from_data((uint8_t*)bytestr, bytes_len); // slower than what??? Also, is this a valid cast? TODO: free
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
     // TODO: validate in a reasonable way, now segfaults if bad
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return NULL;
    }
    char* str = bson_as_json(document, (size_t*)&bytes_len);
    printf("DOCUMENT: %s\n", str);

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
    dimension_lengths = malloc(1* sizeof(npy_intp)); // TODO: leaked?
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

    Py_INCREF(dtype);

    printf("PYARRAY_ZEROS(1, %i, ", (int)dimension_lengths[0]);
    PyObject_Print((PyObject*)dtype, stdout, 0);
    printf(", 0)\n");

    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0); // This function steals a reference to dtype?

    PyArray_OutputConverter(array_obj, &ndarray); //TODO: figure this out again


    npy_intp* coordinates = calloc(number_dimensions + 1, sizeof(npy_intp));
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

static int load_document(PyObject* binary_doc,
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

    bytes_str = PyBytes_AS_STRING(binary_doc);
    bytes_len = PyBytes_GET_SIZE(binary_doc);
    document = bson_new_from_data((uint8_t*)bytes_str, bytes_len);
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
     // TODO: validate in a reasonable way, now segfaults if bad
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return 0;
    }

    char* str = bson_as_json(document, (size_t*)&bytes_len);
    printf("\nDOCUMENT: %s, DTYPE:", str);
    PyObject_Print(dtype->fields, stdout, 0);
    printf("\n");

    fields = dtype->fields; //A field is described by a tuple composed of another data- type-descriptor and a byte offset.

    if(dtype->fields != NULL && dtype->fields != Py_None) {
        // Need to validate fields.
        if(!PyDict_Check(fields)) {
            PyErr_SetString(BsonNumpyError, "in load_document: dtype.fields was not a dictionary?");
            return 0;
        }
        pos = 0;
        PyArray_Descr* sub_dtype;
        PyObject* sub_dtype_obj, *offset;
        int success;

        PyObject* ordered_dict = PyDict_New();
        while(PyDict_Next(fields, &pos, &key, &value)) { // for each column --> this could be rewritten not use a dict
            if (!PyTuple_Check(value)) {
                PyErr_SetString(BsonNumpyError, "dtype in fields is not a tuple");
            }
            offset = PyTuple_GetItem(value, 1);
            PyObject* new_tuple = PyTuple_New(2);
            PyTuple_SetItem(new_tuple, 0, key);
            PyTuple_SetItem(new_tuple, 1, value);
            PyDict_SetItem(ordered_dict, offset, new_tuple);
        }
        PyObject* offsets = PyDict_Keys(ordered_dict);
        PyList_Sort(offsets); // TODO: this is probably unneccssary --> load_scalar calculates offset instead of being told
        Py_ssize_t total_length = PyList_Size(offsets);

//        printf("just finished creating sorted dict:");
//        PyObject_Print(ordered_dict, stdout, 0);
//        printf(" offsets=");
//        PyObject_Print(offsets, stdout, 0);
//        printf(" total_length=%i\n", (int)total_length);

        // for each key in document, matching order in array
        for (Py_ssize_t i=0; i<total_length; i++) {
            PyObject* curr_offset = PyList_GetItem(offsets, i);
            PyObject* key_value_tuple = PyDict_GetItem(ordered_dict, curr_offset);
            key = PyTuple_GetItem(key_value_tuple, 0);
            value = PyTuple_GetItem(key_value_tuple, 1);


            key_str = PyBytes_AS_STRING(key);
            if (!PyTuple_Check(value)) {
                PyErr_SetString(BsonNumpyError, "dtype in fields is not a tuple");
            }
            sub_dtype_obj = PyTuple_GetItem(value, 0);


            if (!PyArray_DescrConverter(sub_dtype_obj, &sub_dtype)) { // Convert from python object to numpy dtype object
                PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
                return 0;
            }
            coordinates[depth] = i;

            printf("-->looping through fields, key=");
            PyObject_Print(key, stdout, 0);
            printf(" dtype=");
            PyObject_Print((PyObject*)sub_dtype, stdout, 0);
            printf(" coordinates: [");
            for (int i=0;i<number_dimensions;i++) {
                printf("%i,", (int)coordinates[i]);
            }
            printf("]\n");



            bson_iter_init(&bsonit, document);
            if(bson_iter_find(&bsonit, key_str)) { //CURRENT TODO: coordinates aren't being set right
                success = _load_scalar(&bsonit, coordinates, ndarray, depth + 1, number_dimensions, sub_dtype);
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

static int _get_depth(int current_depth, PyArray_Descr* dtype) {
    if(dtype->subarray != NULL) {
        PyObject *shape = dtype->subarray->shape;
        if(!PyTuple_Check(shape)) {
            PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
            return -1;
        }
        return (int)PyTuple_Size(shape) + current_depth;
    }

    if(dtype->fields != NULL && dtype->fields != Py_None) { // flexible type
        PyObject* fields = dtype->fields;
        PyObject* key, *value, *sub_dtype_obj;
        PyArray_Descr* sub_dtype;
        Py_ssize_t pos = 0;

        if(!PyDict_Check(fields)) {
            PyErr_SetString(BsonNumpyError, "in get_depth: type.fields was not a dictionary?");
            return -1;
        }

        int max_depth = 0;
        int sub_depth = 1;

        while(PyDict_Next(fields, &pos, &key, &value)) { // for each column
            if (!PyTuple_Check(value)) {
                PyErr_SetString(BsonNumpyError, "in get_depth: dtype in fields is not a tuple");
                return -1;
            }
            sub_dtype_obj = PyTuple_GetItem(value, 0);
            if (!PyArray_DescrConverter(sub_dtype_obj, &sub_dtype)) {
                PyErr_SetString(BsonNumpyError, "in get_depth: dtype passed in was invalid");
                return -1;
            }
            sub_depth = _get_depth(current_depth + 1, sub_dtype);
            if(sub_depth > max_depth) {
                max_depth = sub_depth;
            }
        }
        return sub_depth + current_depth;
    }
    return current_depth + 1;
}


// TODO: RENAME TO SEQUENCE_
static PyObject*
collection_to_ndarray(PyObject* self, PyObject* args) // Better name please! Collection/cursor both seem to specific to PyMongo.
{
    PyObject* iterator_obj;
    PyObject* dtype_obj;
    PyObject* array_obj;
    PyObject* binary_doc;

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

    dimension_lengths = malloc(1* sizeof(npy_intp));
    dimension_lengths[0] = num_documents;

    // Get maximum depth for an arbitrarily complicated flexible array. dtype->subarray only works if it is a contiguous array :(

    number_dimensions = _get_depth(0, dtype);
    if(number_dimensions == -1) {
        return NULL;
    }
    printf("dimension_lengths=%i, number_dimensions=%i\n", (int)num_documents, (int)number_dimensions);

    Py_INCREF(dtype);

    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0); // This function steals a reference to dtype?

    PyArray_OutputConverter(array_obj, &ndarray); // why did i add this?


    npy_intp* coordinates = calloc(1 + number_dimensions, sizeof(npy_intp));


    int row = 0;
    while((binary_doc = PyIter_Next(iterator_obj))) { // For each row
        printf("START COORDINATES: [");
        for(int q = 0; q < number_dimensions; q++) {
            printf("%i,", (int)coordinates[q]);
        }
        printf("]\n");
        if(load_document(binary_doc, dtype, coordinates,
                          ndarray, 1, number_dimensions) == 0) { // error set by load_document
            return NULL;
        }
        coordinates[0] = ++row;
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
#else  // Python 2.x
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
#endif
