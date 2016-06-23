#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bson.h"

static PyObject* BsonNumpyError;


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
    const char* bytestr;
    Py_ssize_t len;
    bson_iter_t bsonit;
    bson_iter_t sub_it;
    bson_t* document;
    bson_error_t error;
    size_t err_offset;

    PyArg_ParseTuple(args, "O", &bobj);
    if (!PyBytes_Check(bobj)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }
    bytestr = PyBytes_AS_STRING(bobj);
    len = PyBytes_GET_SIZE(bobj);

     // TODO: validate in a reasonable way
    document = bson_new_from_data((uint8_t*)bytestr, len); // slower than what??? Also, is this a valid cast?
    if (!bson_validate(document, BSON_VALIDATE_NONE, &err_offset)) {
        PyErr_SetString(BsonNumpyError, "Document failed validation");
        return NULL;
    }
    char* str = bson_as_json(document, (size_t*)&len);
    printf("DOCUMENT: %s\n", str);

    bson_iter_init(&bsonit, document);
    bson_iter_find (&bsonit, "x");

    if (!BSON_ITER_HOLDS_ARRAY(&bsonit)) {
        PyErr_SetString(BsonNumpyError, "Document must contain an array");
        return NULL;
    }
    bson_iter_recurse(&bsonit, &sub_it);

    // Allocate ndarray --> any way of finding what array types/size from bson_it?
    while( bson_iter_next(&sub_it) ) {
        //fill array
    }



    return Py_BuildValue("");
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

