/*
 * Maximum Gap via Radix Sort
 *
 * k<=16: counting sort on [0, 2^k), linear scan for max gap
 * k=32 : 2-pass LSD radix sort with 16-bit radix (a->b by low16, b->a by high16)
 *        then linear scan for max gap
 *
 * Radix sort advantages over bucket-based max gap:
 *   - Only 65536 counters (fits in L1/L2 cache)
 *   - Structured scatter writes vs 67M random bucket accesses
 *
 * Time:  O(n)
 * Space: O(n) auxiliary for radix sort
 */

#include <cstdio>
#include <cstring>

typedef unsigned int u32;

u32 nextInt(u32 x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

static u32 a[1 << 26];
static u32 b[1 << 26];
static int cnt[1 << 16];

int main() {
    int n, k;
    u32 seed;
    scanf("%d%d%u", &n, &k, &seed);

    for (int i = 0; i < n; i++) {
        seed = nextInt(seed);
        a[i] = seed >> (32 - k);
    }

    if (k <= 16) {
        int vmax = 1 << k;
        memset(cnt, 0, sizeof(int) * vmax);
        for (int i = 0; i < n; i++) cnt[a[i]]++;

        u32 ans = 0;
        int prev = -1;
        for (int i = 0; i < vmax; i++) {
            if (cnt[i]) {
                if (prev >= 0) {
                    u32 gap = (u32)(i - prev);
                    if (gap > ans) ans = gap;
                }
                prev = i;
            }
        }
        printf("%u\n", ans);
    } else {
        // Pass 1: sort by lower 16 bits (a -> b)
        memset(cnt, 0, sizeof(cnt));
        for (int i = 0; i < n; i++) cnt[a[i] & 0xFFFF]++;
        for (int i = 1; i < 65536; i++) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--) b[--cnt[a[i] & 0xFFFF]] = a[i];

        // Pass 2: sort by upper 16 bits (b -> a)
        memset(cnt, 0, sizeof(cnt));
        for (int i = 0; i < n; i++) cnt[b[i] >> 16]++;
        for (int i = 1; i < 65536; i++) cnt[i] += cnt[i - 1];
        for (int i = n - 1; i >= 0; i--) a[--cnt[b[i] >> 16]] = b[i];

        // Linear scan for max gap
        u32 ans = 0;
        for (int i = 1; i < n; i++) {
            u32 gap = a[i] - a[i - 1];
            if (gap > ans) ans = gap;
        }
        printf("%u\n", ans);
    }

    return 0;
}