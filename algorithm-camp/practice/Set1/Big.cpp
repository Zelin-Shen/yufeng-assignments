/*
Fast A+B for huge input.

- Buffered fread input + buffered fwrite output
- Manual integer parse/print
*/

#include <cstdio>
#include <cstdint>

static const int IS = 1 << 22, OS = 1 << 22;
static unsigned char ib[IS], ob[OS];
static int ip = 0, il = 0, op = 0;

static inline int gc() {
    if (ip >= il) {
        il = (int)fread(ib, 1, IS, stdin);
        ip = 0;
        if (!il) return -1;
    }
    return ib[ip++];
}

static inline bool rd(uint32_t &x) {
    int c = gc();
    while (c <= 32) {
        if (c == -1) return false;
        c = gc();
    }
    uint32_t v = 0;
    while ((unsigned)(c - '0') < 10u) {
        v = v * 10u + (uint32_t)(c - '0');
        c = gc();
    }
    x = v;
    return true;
}

static inline void flush() {
    if (op) fwrite(ob, 1, op, stdout), op = 0;
}

static inline void pc(unsigned char c) {
    if (op >= OS) flush();
    ob[op++] = c;
}

static inline void wt(uint32_t x) {
    unsigned char s[10];
    int n = 0;
    do s[n++] = (unsigned char)('0' + x % 10u), x /= 10u;
    while (x);
    while (n--) pc(s[n]);
    pc('\n');
}

int main() {
    uint32_t n, a, b;
    if (!rd(n)) return 0;
    while (n--) rd(a), rd(b), wt(a + b);
    flush();
    return 0;
}