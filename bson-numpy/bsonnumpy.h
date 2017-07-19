#ifndef BSONNUMPY_H
#define BSONNUMPY_H

#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "bson/bson.h"

typedef struct
{
    const char *s;
    size_t len;
} bsnp_string_t;


ssize_t
bsnp_next_power_of_two(ssize_t v);

void
bsnp_string_init(bsnp_string_t *string, const char *s);


#endif //BSONNUMPY_H
