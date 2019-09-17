/*
 * Copyright 2016-present MongoDB, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BSONNUMPY_FIELDorder_H
#define BSONNUMPY_FIELDorder_H

#include "bsonnumpy.h"


static const ssize_t NO_INDEX = -1;


typedef struct
{
    bsnp_string_t key;
    ssize_t dtype_index;
} field_order_elem_t;


static const field_order_elem_t final = {{"", 0}, -1};


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


static inline bool
field_order_match(field_order_t *order, size_t bson_index, bson_iter_t *it,
                  const field_order_elem_t **out)
{
    const field_order_elem_t *elem;
    const char *next_key;

    if (bson_index > order->n_elems) {
        return false;
    }

    if (bson_index == order->n_elems) {
        /* reached the end */
        *out = &final;
        return true;
    }

    elem = &order->elems[bson_index];

    if ((it->next_off + elem->key.len + 1) > it->len) {
        return false;
    }

    next_key = (const char *) (it->raw + it->next_off + 1);
    if (0 != memcmp(next_key, elem->key.s, elem->key.len)) {
        return false;
    }

    *out = elem;
    return true;
}


void
field_order_destroy(field_order_t *order);


#endif //BSONNUMPY_FIELDorder_H
