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
//    case BSON_TYPE_EOD: return
//    case BSON_TYPE_DOUBLE: return
//    case BSON_TYPE_UTF8: return
//    //case BSON_TYPE_DOCUMENT SHOULDN'T NEED
//    //case BSON_TYPE_ARRAY SHOULDN'T NEED
//    case BSON_TYPE_BINARY: return
//    case BSON_TYPE_UNDEFINED: return
//    case BSON_TYPE_OID: return
//    case BSON_TYPE_BOOL: return
//    case BSON_TYPE_DATE_TIME: return
//    case BSON_TYPE_NULL: return
//    case BSON_TYPE_REGEX: return
//    //case BSON_TYPE_DBPOINTER ???
//    case BSON_TYPE_CODE: return
//    case BSON_TYPE_SYMBOL: return
//    case BSON_TYPE_CODEWSCOPE: return
//    case BSON_TYPE_INT32: return
//    case BSON_TYPE_TIMESTAMP: return
//    case BSON_TYPE_INT64: return
//    case BSON_TYPE_MAXKEY: return
//    case BSON_TYPE_MINKEY return
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
    return Py_BuildValue("");
}

static PyObject*
bson_to_ndarray(PyObject* self, PyObject* args)
{
    // Takes in a BSON byte string
    PyObject* bobj;
    PyObject* dtype_obj;
    PyObject* array_obj;
    const char* bytestr;
    Py_ssize_t len;
    bson_iter_t bsonit;
    bson_t* document;
    size_t err_offset;
    PyArray_Descr* dtype;
    PyArrayObject* array;

    if (!PyArg_ParseTuple(args, "SOO", &bobj, &dtype_obj, &array_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    bytestr = PyBytes_AS_STRING(bobj);
    len = PyBytes_GET_SIZE(bobj);
    if (!PyArray_DescrCheck(dtype_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    if (!PyArray_Check(array_obj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }

    // Convert dtype
    if (!PyArray_DescrConverter(dtype_obj, &dtype)) {
        PyErr_SetString(BsonNumpyError, "dtype passed in was invalid");
        return NULL;
    }
    // Convert array
    if (!PyArray_OutputConverter(array_obj, &array)) {
        PyErr_SetString(BsonNumpyError, "bad array type");
        return NULL;
    }
     // TODO: validate in a reasonable way
    document = bson_new_from_data((uint8_t*)bytestr, len); // slower than what??? Also, is this a valid cast?
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return NULL;
    }
    char* str = bson_as_json(document, (size_t*)&len);
    printf("DOCUMENT: %s\n", str);

    bson_iter_init(&bsonit, document);
//    bson_iter_recurse(&bsonit, &sub_it);

    // Allocate ndarray --> any way of finding what array types/size from bson_it without iterating?
    // Could find type and buffer length, then create ndarray based on 1st element.

    int keys = bson_count_keys(document);
    PyObject *array2;
    npy_intp* dims = malloc(sizeof(npy_intp)*1);
    dims[0] = keys;
    array2 = PyArray_SimpleNewFromDescr(1, PyArray_DIMS(array), dtype);
    free(dims);
    PyArrayObject* ndarray;
    PyArray_OutputConverter(array2, &ndarray);

    for(npy_intp i=0;i<keys;i++) {
        bson_iter_next(&bsonit);
        bson_type_t elem1_type = bson_iter_type(&bsonit);
        int x = bson_iter_int32(&bsonit);
        char* pointer =  PyArray_GetPtr(ndarray, &i);
        PyArray_SETITEM(ndarray, pointer, PyInt_FromLong(x)); // TODO: use Numpy types
    }

    Py_INCREF(array2);
    return (PyObject*)array2;
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

