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
                        PyArrayObject* ndarray,
                        long offset,
                        npy_intp* coordinates,
                        int current_depth,
                        PyArray_Descr* dtype) {
    bson_iter_t sub_it;
    npy_intp dimensions = PyArray_NDIM(ndarray);
    npy_intp itemsize;

    if (current_depth < dimensions) { // If we are within a flexible type
        itemsize = PyArray_STRIDE(ndarray, current_depth);
        printf("\tsetting itemsize using strides: %li\n", itemsize);
    } else {
        itemsize = dtype->elsize; // Not sure about this
    }
    npy_intp bson_item_len = itemsize;
    int success = 0;
    int copy = 1;

    printf("\tin load_scalar, dimensions=%i, current_depth=%i, offset=%i, itemsize=%i ", (int)dimensions, current_depth, (int)offset, (int)itemsize);
    printf("coordinates=["); for(int i=0;i<dimensions;i++) { printf("%i,", (int)coordinates[i]); } printf("]\n");


    void* pointer = PyArray_GetPtr(ndarray, coordinates);
    pointer += offset;

    if(BSON_ITER_HOLDS_ARRAY(bsonit)) {

        printf("\t\tFound subarray\n");

        // Get length of array
        bson_iter_recurse(bsonit, &sub_it);
        int count = 0;
        while( bson_iter_next(&sub_it)) {
            count++;
        }
        bson_iter_recurse(bsonit, &sub_it);
        if (count == 1) {
            bson_iter_next(&sub_it);

            printf("\t\t-ignoring array of len 1\n");

            // Array is of length 1, therefore we treat it like a number
            return _load_scalar(&sub_it, ndarray,offset, coordinates, current_depth, dtype); // arrays of length 1 have the same dtype as element
        } else {

            int i = 0;
            while( bson_iter_next(&sub_it) ) {
                PyArray_Descr* sub_dtype = dtype->subarray ? dtype->subarray->base : dtype;

                // If we're recurring after the end of dtype's dimensions, we have a flexible type subarray
                long new_offset = offset;
                if (current_depth + 1 < dimensions) {
                    coordinates[current_depth + 1] = i;
                } else {
                    PyErr_SetString(BsonNumpyError, "TODO: unhandled case");
                    return 0;
                }
                printf("\t\t-->new dtype="); PyObject_Print((PyObject*)sub_dtype, stdout, 0);
                printf(" new depth=%i, dimensions=%i, index=%i\n", current_depth + 1, (int)dimensions, i);

                printf("\t\t-->recurring on load_scalar: new coordinates= ["); for(int i=0;i<dimensions;i++) { printf("%i,", (int)coordinates[i]); }printf("]\n");

                int ret = _load_scalar(&sub_it, ndarray, new_offset, coordinates, current_depth + 1, sub_dtype);
                if (ret == 0) {
                    return 0;
                };
                i++;
            }
            return 1;
      }
    }
    const bson_value_t* value = bson_iter_value(bsonit);
    void* data_ptr = (void*)&value->value;

    printf("\t- switching on %i\n", value->value_type);

    switch(value->value_type) {
    case BSON_TYPE_UTF8:
        data_ptr = value->value.v_utf8.str; // Unclear why using value->value doesn't work
        bson_item_len = value->value.v_utf8.len;
        break;
    case BSON_TYPE_INT32:
        data_ptr = (void*)&value->value.v_int32;
        bson_item_len = sizeof (value->value.v_int32);
        break;
    case BSON_TYPE_INT64:
        data_ptr = (void*)&value->value.v_int64;
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
        success = 1;
        break;
    case BSON_TYPE_REGEX:
        bson_item_len = (int)strlen(value->value.v_regex.regex);
        memcpy(pointer, value->value.v_regex.regex, bson_item_len);
        memset(pointer + bson_item_len, '\0', 1);
        memcpy(pointer + bson_item_len + 1, value->value.v_regex.options, (int)strlen(value->value.v_regex.options));
        bson_item_len = bson_item_len + (int)strlen(value->value.v_regex.options) + 1;
        copy = 0;
        success = 1;
        break;
    default:
        printf("TODO: bson type %i not handled\n", value->value_type);
    }

    /* Commented out because PyArray_SETITEM fails for flexible types, but memcpy works.
       TODO: use macros whenever possible, better to handle errors. Can check how far off by GETITEM address w coordinates vs. pointer
       PyObject* data = PyArray_Scalar(data_ptr, dtype, NULL);
       success = PyArray_SETITEM(ndarray, pointer, data);
     */
    if(copy) {
        if(bson_item_len > itemsize) {
            bson_item_len = itemsize; // truncate data that's too big
        }
        memcpy(pointer, data_ptr, bson_item_len);
        memset(pointer + bson_item_len, '\0', itemsize - bson_item_len);
        success = 1;
    }
//        printf("\t\tEND OF LOAD SCALAR:"); PyObject_Print((PyObject*)ndarray, stdout, 0); printf("\n");
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

    printf("PYARRAY_ZEROS(1, %i, ", (int)dimension_lengths[0]); PyObject_Print((PyObject*)dtype, stdout, 0); printf(", 0)\n");

    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0); // This function steals a reference to dtype?

    PyArray_OutputConverter(array_obj, &ndarray);


    npy_intp* coordinates = calloc(number_dimensions + 1, sizeof(npy_intp));
    for(npy_intp i=0;i<dimension_lengths[0];i++) {
        bson_iter_next(&bsonit);
        coordinates[0] = i;
        int success = _load_scalar(&bsonit, ndarray, 0, coordinates, 0, dtype);
        if(success == 0) {
            return NULL;
        }
    }

    free(dimension_lengths);
    free(document);
    free(coordinates);
//    Py_INCREF(array_obj);
    return array_obj;
}

static int _load_flexible(bson_t* document,
                          npy_intp* coordinates,
                          PyArrayObject* ndarray,
                          PyArray_Descr* dtype,
                          int current_depth,
                          char* key_str,
                          npy_intp* sub_coordinates,
                          npy_intp sub_coordinates_length,
                          npy_intp offset) {
    PyObject* fields, *key, *value = NULL;
    int number_dimensions = PyArray_NDIM(ndarray);
    Py_ssize_t pos = 0;
    bson_iter_t bsonit;

    printf("    in _load_flexible: KEY=%s, SUB_COORDINATES=[", key_str);for (int i = 0; i < sub_coordinates_length; i++) { printf("%i,", (int) sub_coordinates[i]); } printf("]\n");
    printf(" DTYPE IS ");

    if(dtype->fields != NULL && dtype->fields != Py_None) {
        printf("FLEXIBLE\n\tnames=");
        PyObject* ordered_names = PyArray_FieldNames(dtype->fields);
        PyObject_Print(ordered_names, stdout, 0); printf(" fields="); PyObject_Print(dtype->fields, stdout, 0);

        Py_ssize_t number_fields = PyTuple_Size(ordered_names);
        printf(" len(fields)=%li\n", number_fields);


        fields = dtype->fields; // A field is described by a tuple composed of another data- type-descriptor and a byte offset.
        if(!PyDict_Check(fields)) {
            PyErr_SetString(BsonNumpyError, "in _load_flexible: dtype.fields was not a dictionary?");
            return 0;
        }
        pos = 0;
        PyArray_Descr* sub_dtype;
        PyObject* sub_dtype_obj, *offset_obj;
        int success;

        for (Py_ssize_t i=0; i<number_fields; i++) {
            key = PyTuple_GetItem(ordered_names, i);
            value = PyDict_GetItem(fields, key);

            printf("    -->looping through fields: key="); PyObject_Print(key, stdout, 0); printf(" value="); PyObject_Print(value, stdout, 0); printf("\n");
            if (!PyTuple_Check(value)) {
                PyErr_SetString(BsonNumpyError, "dtype in fields is not a tuple");
                return 0;
            }
            if (PyUnicode_Check(key)) {
                key = PyUnicode_AsASCIIString(key);
            }
            if (!PyBytes_Check(key)) {
                PyErr_SetString(BsonNumpyError, "bson string error in key names");
                return 0;
            }
            offset_obj = PyTuple_GetItem(value, 1);
            long offset_long = PyLong_AsLong(offset_obj);
            key_str = PyBytes_AsString(key);
            sub_dtype_obj = PyTuple_GetItem(value, 0);

            if (!PyArray_DescrConverter(sub_dtype_obj,
                                        &sub_dtype)) { // Convert from python object to numpy dtype object
                PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
                return 0;
            }

            int sub_depth = current_depth - number_dimensions;
            if (sub_dtype->subarray) {
                printf("\t Recurring with SUBARRAY\n");


                sub_coordinates[sub_depth] = i;
                _load_flexible(document, coordinates, ndarray, sub_dtype, current_depth + 1, key_str, sub_coordinates, sub_coordinates_length+1, offset);

            } else if (sub_dtype->fields && sub_dtype->fields != Py_None) {
                printf("\t Recurring with FIELDS: sub_dtype="); PyObject_Print(sub_dtype->fields, stdout, 0); printf("\n");
                bson_iter_init(&bsonit, document);
                if (bson_iter_find(&bsonit, key_str)) {
                    if (!BSON_ITER_HOLDS_DOCUMENT(&bsonit)) {
                        PyErr_SetString(BsonNumpyError, "Expected list from dtype, got other type");
                        return 0;
                    }

                    bson_t* sub_document;
                    uint32_t document_len;
                    const uint8_t* document_buffer;
                    bson_iter_document(&bsonit, &document_len, &document_buffer);
                    sub_document = bson_new_from_data(document_buffer, document_len);

                    printf("\t setting sub_coordinates at %i to %i\n", sub_depth, i);
                    sub_coordinates[sub_depth] = i;


                    _load_flexible(sub_document, coordinates, ndarray, sub_dtype, current_depth + 1, NULL, sub_coordinates, sub_coordinates_length+1, offset + offset_long);


                } else {
                    PyErr_SetString(BsonNumpyError, "Error: expected key from dtype in document, not found");
                }
//                PyErr_SetString(BsonNumpyError, "TODO: not implemented");
//                return 0;
            }
            else {
                printf("\tLOADING VAL: key="); PyObject_Print(key, stdout, 0); printf(" dtype="); PyObject_Print((PyObject *) sub_dtype, stdout, 0);
                printf(" offset=%i, coordinates: [", (int) offset_long); for (int i = 0; i < number_dimensions; i++) { printf("%i,", (int) coordinates[i]); } printf("]\n");

                bson_iter_init(&bsonit, document);
                if (bson_iter_find(&bsonit, key_str)) {
                    //TODO: if sub_dtype->elsize==0, then it is a flexible type
                    success = _load_scalar(&bsonit, ndarray, offset + offset_long, coordinates, current_depth, sub_dtype);
                    if (!success) {
                        return 0;
                    }
                } else {
                    PyErr_SetString(BsonNumpyError, "document does not match dtype."); // TODO: nicer error message
                    return 0;
                }
            }
            pos++;
        }
    } else if (dtype->subarray) {
        printf("ARRAY\n");
        PyObject* shape = dtype->subarray->shape;
        PyArray_Descr* sub_descr = dtype->subarray->base;

        printf("\tdtype="); PyObject_Print((PyObject*)sub_descr, stdout, 0); printf(" shape="); PyObject_Print(shape, stdout, 0); printf("\n");

        bson_iter_init(&bsonit, document);


        Py_ssize_t dims_subarray = PyTuple_Size(shape);
        if (bson_iter_find(&bsonit, key_str)) {
            if (!BSON_ITER_HOLDS_ARRAY(&bsonit)) {
                PyErr_SetString(BsonNumpyError, "Expected list from dtype, got other type");
                return 0;
            }

            // Get subarray as ndarray
            void* ptr = PyArray_GetPtr(ndarray, coordinates);
            PyObject* subndarray_tuple = PyArray_GETITEM(ndarray, ptr);
//            printf("curr depth=%i, num_dims=%i: STARTING TUPLE=", current_depth, number_dimensions); PyObject_Print(subndarray_tuple, stdout, 0); printf("\n");
            for (int sub_i = 0; sub_i < current_depth - number_dimensions; sub_i++) {
                int offset = sub_coordinates[sub_i];
//                printf("\t offset=%i\n", offset);
                subndarray_tuple = PyTuple_GetItem(subndarray_tuple, offset);
//                printf("\t SUB TUPLE="); PyObject_Print(subndarray_tuple, stdout, 0); printf("\n");
            }
//            printf("OBJ="); PyObject_Print(subndarray_tuple, stdout, 0); printf("\n");
            PyArrayObject* subndarray;
            if (!PyArray_OutputConverter(subndarray_tuple, &subndarray)) {
                PyErr_SetString(BsonNumpyError, "Expected subarray, got other type");
                return 0;
            }
            // Get length of top-level array
            PyObject* length_obj = PyTuple_GetItem(shape, 0);
            long length_long = PyLong_AsLong(length_obj);

            // Create coordinates for subtype
            npy_intp* subarray_coordinates = calloc(dims_subarray + 1, sizeof(npy_intp));

            // Loop through array and load sub-arrays
            printf("looping through top-level array, len=%li\n", length_long);
            bson_iter_t sub_it;
            bson_iter_recurse(&bsonit, &sub_it);
            for(npy_intp i=0;i<length_long;i++) {
//                printf("\t(SUB)START COORDINATES="); for (int i = 0; i < dims_subarray; i++) { printf("%i,", (int) subarray_coordinates[i]); } printf("]\n");
                bson_iter_next(&sub_it);
                subarray_coordinates[0] = i;
                int success = _load_scalar(&sub_it, subndarray, 0, subarray_coordinates, 0, sub_descr);
                if(success == 0) {
                    return 0;
                }
            }
        }
        else {
            PyErr_SetString(BsonNumpyError, "key from dtype not found");
            return 0;
        }

    } else {
        PyErr_SetString(BsonNumpyError, "TODO: constant loaded with _load_dcument, shouldn't happen");
        return 0;
    }
    return 1;
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
    int number_dimensions;
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

    number_dimensions = 1;

    if(dtype->subarray != NULL) {
        PyObject *shape = dtype->subarray->shape;
        if(!PyTuple_Check(shape)) {
            PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
            return NULL;
        }
        number_dimensions = (int)PyTuple_Size(shape);
    }

    printf("dimension_lengths=%i, number_dimensions=%i\n", num_documents, number_dimensions);

    Py_INCREF(dtype);

    array_obj = PyArray_Zeros(1, dimension_lengths, dtype, 0); // This function steals a reference to dtype?

    PyArray_OutputConverter(array_obj, &ndarray);


    npy_intp* coordinates = calloc(1 + number_dimensions, sizeof(npy_intp));

    size_t err_offset;
    int row = 0;
    while((binary_doc = PyIter_Next(iterator_obj))) { // For each row

        printf("START COORDINATES: ["); for(int q = 0; q < number_dimensions; q++) { printf("%i,", (int)coordinates[q]);} printf("]\n");

        // Get BSON document
        const char* bytes_str = PyBytes_AS_STRING(binary_doc);
        Py_ssize_t bytes_len = PyBytes_GET_SIZE(binary_doc);
        bson_t* document = bson_new_from_data((uint8_t*)bytes_str, bytes_len);
        if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
            // TODO: validate in a reasonable way, now segfaults if bad
            PyErr_SetString(BsonNumpyError, "Document failed validation");
            return 0;
        }

        char* str = bson_as_json(document, (size_t*)&bytes_len);
        printf("\nDOCUMENT: %s, dtype->fields dict:", str); PyObject_Print(dtype->fields, stdout, 0); printf("\n");

        npy_intp* sub_coordinates = calloc(100, sizeof(npy_intp));

        if(_load_flexible(document, coordinates, ndarray, PyArray_DTYPE(ndarray), 1, NULL, sub_coordinates, 0, 0) == 0) { // Don't need to pass key to first layer
            return NULL; // error set by _load_flexible
        }
        free(document);
        coordinates[0] = ++row;
        // Reset coordinates to zero
        for(int p = 1; p < number_dimensions; p++) {
            coordinates[p] = 0;
        }
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
