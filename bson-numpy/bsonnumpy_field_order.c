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
