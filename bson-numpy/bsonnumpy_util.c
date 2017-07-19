#include "bsonnumpy.h"


void
bsnp_string_init(bsnp_string_t *string, const char *s)
{
    string->s = s;
    string->len = strlen(s);
}


ssize_t
bsnp_next_power_of_two(ssize_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if BSON_WORD_SIZE == 64
    v |= v >> 32;
#endif
    v++;

    return v;
}
