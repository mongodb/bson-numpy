#ifndef BSONNUMPY_FIELDorder_H
#define BSONNUMPY_FIELDorder_H

#include "bsonnumpy.h"


typedef struct
{
    bsnp_string_t key;
    ssize_t dtype_index;
} field_order_elem_t;


typedef struct
{
    size_t maxsize;
    field_order_elem_t *elems;
    size_t n_elems;
} field_order_t;


void
field_order_init(field_order_t *order, size_t size);


static inline void
field_order_set(field_order_t *order, size_t bson_index, const char *key,
                ssize_t dtype_index)
{
    if (bson_index >= order->maxsize) {
        order->maxsize *= 2;
        order->elems = bson_realloc(
            order->elems, order->maxsize * sizeof(field_order_elem_t));
    }

    bsnp_string_init(&order->elems[bson_index].key, key);
    order->elems[bson_index].dtype_index = dtype_index;
    order->n_elems = BSON_MAX(bson_index + 1, order->n_elems);
}


static inline size_t
field_order_elems(field_order_t *order)
{
    return order->n_elems;
}


static inline field_order_elem_t *
field_order_get(field_order_t *order, size_t bson_index)
{
    assert (bson_index <= order->maxsize);
    assert (bson_index <= order->n_elems);
    return &order->elems[bson_index];
}


void
field_order_destroy(field_order_t *order);


#endif //BSONNUMPY_FIELDorder_H
