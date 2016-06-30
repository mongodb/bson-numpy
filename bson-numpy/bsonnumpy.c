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

bson_type_t get_bson_type(enum NPY_TYPES numpy_type) {
    return BSON_TYPE_INT32;
}

//NPY_TYPES get_numpy_type(bson_type_t bson_type) {
//    switch(bson_type) {
//    }
//}



static PyObject*
run_command(PyObject* self, PyObject* args)
{
    const char* command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0) {
        PyErr_SetString(BsonNumpyError, "System command failed");
        return NULL;
    }
    return Py_BuildValue("i", sts);
}

static PyObject*
ndarray_to_bson(PyObject* self, PyObject* args)
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
                       char* pointer,
                       PyArrayObject* ndarray) {
        const bson_value_t* value = bson_iter_value(bsonit);
        int itemsize = PyArray_DESCR(ndarray)->elsize;;
        int len = itemsize;
        int success = 0;
        int copy = 1;
        int trailing_null = 1;

        void* data_ptr = (void*)&value->value;
        switch(value->value_type) {
        case BSON_TYPE_UTF8:
            data_ptr = value->value.v_utf8.str; // Unclear why using value->value doesn't work
            len = value->value.v_utf8.len;
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
        case BSON_TYPE_TIMESTAMP:
            // Have to special case here because there's no numpy equivalent to a timestamp.
            memcpy(pointer, &value->value.v_timestamp.timestamp, sizeof(int32_t));
            memcpy((pointer+sizeof(int32_t)), &value->value.v_timestamp.increment, sizeof(int32_t));
            copy = 0;
            trailing_null = 0;
            break;
        case BSON_TYPE_REGEX:
            len = strlen(value->value.v_regex.regex);
            memcpy(pointer, value->value.v_regex.regex, len);
            memset(pointer + len, '\0', 1);
            memcpy(pointer + len + 1, value->value.v_regex.options, strlen(value->value.v_regex.options));
            len = len + strlen(value->value.v_regex.options) + 1;
            printf("regex_len=%i, option_len=%i\n", len, strlen(value->value.v_regex.options));
            printf("regex=%s, options=%s\n", value->value.v_regex.regex, value->value.v_regex.options);
            copy = 0;
            break;

            // Have to special case here
        }
        printf("TYPE=%i\n", value->value_type);

        if(copy) {
            PyObject* data = PyArray_Scalar(data_ptr, PyArray_DESCR(ndarray), NULL);
            success = PyArray_SETITEM(ndarray, pointer, data);
            printf("item=");
            PyObject_Print(data, stdout, 0);
            printf("\n");
    //        Py_INCREF(data);
        }
        if(trailing_null) {
            if(len < itemsize) {
                memset(pointer + len, '\0', itemsize - len);
            }
        }
        return success;
}

static PyObject*
bson_to_ndarray(PyObject* self, PyObject* args)
{
    // Takes in a BSON byte string
    PyObject* bobj;
    PyObject* dtype_obj;
    PyObject *array_obj;
    const char* bytestr;
    PyArray_Descr* dtype;
    PyArrayObject* ndarray;
    Py_ssize_t len;
    bson_iter_t bsonit;
    bson_t* document;
    size_t err_offset;

    if (!PyArg_ParseTuple(args, "SO", &bobj, &dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    bytestr = PyBytes_AS_STRING(bobj);
    len = PyBytes_GET_SIZE(bobj);
    document = bson_new_from_data((uint8_t*)bytestr, len); // slower than what??? Also, is this a valid cast?
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
     // TODO: validate in a reasonable way, now segfaults if bad
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return NULL;
    }
    char* str = bson_as_json(document, (size_t*)&len);
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

    int keys = bson_count_keys(document);
    npy_intp* dims = malloc(sizeof(npy_intp)*1);
    dims[0] = keys;
    array_obj = PyArray_SimpleNewFromDescr(1, dims, dtype);
    free(dims);
    PyArray_OutputConverter(array_obj, &ndarray);

    for(npy_intp i=0;i<keys;i++) {
        bson_iter_next(&bsonit);
        char* pointer =  PyArray_GetPtr(ndarray, &i);
        int success = _load_scalar(&bsonit, pointer, ndarray);
        if(success == -1) {
            PyErr_SetString(BsonNumpyError, "item failed to load");
            return NULL;
        }

    }

    Py_INCREF(array_obj);
    return array_obj;
}

static PyMethodDef BsonNumpyMethods[] = {
    {"run_command",  run_command, METH_VARARGS,
     "Execute a shell command."},
    {"ndarray_to_bson", ndarray_to_bson, METH_VARARGS,
     "Convert an ndarray into a BSON byte string"},
    {"bson_to_ndarray", bson_to_ndarray, METH_VARARGS,
     "Convert BSON byte string into an ndarray"},
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
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initbsonnumpy();
}

