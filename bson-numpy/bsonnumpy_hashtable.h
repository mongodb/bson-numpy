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

#endif //BSONNUMPY_HASHTABLE_H
