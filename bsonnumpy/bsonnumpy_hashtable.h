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

#ifndef BSONNUMPY_HASHTABLE_H
#define BSONNUMPY_HASHTABLE_H

#include "bsonnumpy.h"


typedef struct
{
    bsnp_string_t key;
    ssize_t ideal_pos;
    ssize_t value;
} hash_table_entry_t;

typedef struct
{
    hash_table_entry_t *entries;
    ssize_t size;
    ssize_t used;
} hash_table_t;

static const ssize_t EMPTY = -1;

void
table_init(hash_table_t *table, ssize_t n_entries);

void
table_insert(hash_table_t *table, const char *key, ssize_t value);

ssize_t
table_lookup(hash_table_t *table, const char *key);

void
table_destroy(hash_table_t *table);

#endif //BSONNUMPY_HASHTABLE_H
