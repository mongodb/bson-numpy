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

#include "bsonnumpy_hashtable.h"


/* how much larger the table is than the number of entries */
const ssize_t TABLE_MULTIPLE = 4;


Py_ssize_t
string_hash(const char *string, size_t len)
{
    size_t hash = 5381;
    const unsigned char *p = (const unsigned char *) string;

    for (; len >= 8; len -= 8) {
        hash = ((hash << 5) + hash) + *p++;
        hash = ((hash << 5) + hash) + *p++;
        hash = ((hash << 5) + hash) + *p++;
        hash = ((hash << 5) + hash) + *p++;
        hash = ((hash << 5) + hash) + *p++;
        hash = ((hash << 5) + hash) + *p++;
        hash = ((hash << 5) + hash) + *p++;
        hash = ((hash << 5) + hash) + *p++;
    }

    switch (len) {
        case 7: hash = ((hash << 5) + hash) + *p++;
        case 6: hash = ((hash << 5) + hash) + *p++;
        case 5: hash = ((hash << 5) + hash) + *p++;
        case 4: hash = ((hash << 5) + hash) + *p++;
        case 3: hash = ((hash << 5) + hash) + *p++;
        case 2: hash = ((hash << 5) + hash) + *p++;
        case 1: hash = ((hash << 5) + hash) + *p++;
            break;
        case 0:
        default:
            break;
    }

    return hash;
}


void
table_init(hash_table_t *table, ssize_t n_entries)
{
    ssize_t i;

    table->size = bsnp_next_power_of_two(n_entries * TABLE_MULTIPLE);
    table->entries = bson_malloc0(table->size * sizeof(hash_table_entry_t));

    for (i = 0; i < table->size; i++) {
        table->entries[i].value = EMPTY;
    }
}


/* simple insertion w/ robin hood hashing. keys are always unique. no resize. */
void
table_insert(hash_table_t *table, const char *key, ssize_t value)
{
    ssize_t mask = table->size - 1;
    ssize_t dist_key = 0;
    Py_ssize_t hash;
    ssize_t i;

    hash_table_entry_t entry;

    bsnp_string_init(&entry.key, key);
    entry.value = value;

    hash = string_hash(key, entry.key.len);

    /* table size is power of 2, hash & (size-1) is faster than hash % size */
    i = entry.ideal_pos = hash & mask;

    while (true) {
        hash_table_entry_t *inplace;
        ssize_t dist_inplace;

        inplace = &table->entries[i];
        if (inplace->value == EMPTY) {
            memcpy(inplace, &entry, sizeof(hash_table_entry_t));
            table->used++;
            return;
        }

        /* this spot is taken. if this entry is closer to its ideal spot than
         * the input is, swap them and find a new place for this entry. */
        dist_inplace = (i - inplace->ideal_pos) & mask;
        if (dist_inplace < dist_key) {
            hash_table_entry_t tmp;

            /* swap with input, start searching for place for swapped entry */
            memcpy(&tmp, inplace, sizeof(hash_table_entry_t));
            memcpy(inplace, &entry, sizeof(hash_table_entry_t));
            memcpy(&entry, &tmp, sizeof(hash_table_entry_t));

            dist_key = dist_inplace;
        }

        dist_key++;
        i++;
        i &= mask;
    }
}


ssize_t
table_lookup(hash_table_t *table, const char *key)
{
    ssize_t mask = table->size - 1;
    Py_ssize_t hash;
    ssize_t i;
    ssize_t dist_key = 0;

    hash = string_hash(key, strlen(key));
    i = hash & mask;

    while (true) {
        hash_table_entry_t *entry = &table->entries[i];

        if (entry->value == EMPTY || !strcmp(entry->key.s, key)) {
            return entry->value;
        }

        /* we haven't yet found the key in the table, and this entry is farther
         * from its ideal spot than key would be if it were here, so we know
         * the key is absent */
        if (dist_key > ((i - entry->ideal_pos) & mask)) {
            return EMPTY;
        }

        dist_key++;
        i++;
        i &= mask;
    }
}


void
table_destroy(hash_table_t *table)
{
    bson_free(table->entries);
}
