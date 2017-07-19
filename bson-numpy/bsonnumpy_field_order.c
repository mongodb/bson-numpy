#include "bsonnumpy_field_order.h"


void
field_order_init(field_order_t *order, size_t size)
{
    order->maxsize = (size_t) bsnp_next_power_of_two(BSON_MIN(size, 8));
    order->elems = bson_malloc(order->maxsize * sizeof(field_order_elem_t));
    order->n_elems = 0;
}


void
field_order_destroy(field_order_t *order)
{
    bson_free(order->elems);
}
