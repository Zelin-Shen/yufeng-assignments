/*
 * Radix Sort — optimized for n=1e8, k=16 or k=32
 *
 * k<=16: counting sort (single pass count + reconstruct)
 * k>16 : 4-pass LSD radix sort, 8-bit radix (256 buckets)
 *        - One combined counting pass for all 4 digits (saves 3 read passes)
 *        - 256 write-fronts fit in L1 cache => scatter is cache-friendly
 *        - Forward iteration for better hardware prefetch
 *
 * Total passes over data for k=32:
 *   1 (count) + 4 (scatter) + 1 (hash) = 6   (vs 8+1 naive)
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

static u32 a[100000000];
static u32 b[100000000];
static int c0[256], c1[256], c2[256], c3[256];

int main() {
    int n, k;
    u32 seed;
    scanf("%d%d%u", &n, &k, &seed);

    for (int i = 0; i < n; i++) {
        seed = nextInt(seed);
        a[i] = seed >> (32 - k);
    }

    if (k <= 16) {
        // Counting sort
        static int cnt[65536];
        int vmax = 1 << k;
        memset(cnt, 0, sizeof(int) * vmax);
        for (int i = 0; i < n; i++) cnt[a[i]]++;
        int pos = 0;
        for (int v = 0; v < vmax; v++) {
            int c = cnt[v];
            for (int j = 0; j < c; j++) a[pos++] = (u32)v;
        }
    } else {
        // Combined counting: one pass counts all 4 byte-digits
        memset(c0, 0, sizeof(c0));
        memset(c1, 0, sizeof(c1));
        memset(c2, 0, sizeof(c2));
        memset(c3, 0, sizeof(c3));
        for (int i = 0; i < n; i++) {
            u32 v = a[i];
            c0[v & 0xFF]++;
            c1[(v >> 8) & 0xFF]++;
            c2[(v >> 16) & 0xFF]++;
            c3[v >> 24]++;
        }

        // Exclusive prefix sums
        int s;
        s = 0; for (int i = 0; i < 256; i++) { int t = c0[i]; c0[i] = s; s += t; }
        s = 0; for (int i = 0; i < 256; i++) { int t = c1[i]; c1[i] = s; s += t; }
        s = 0; for (int i = 0; i < 256; i++) { int t = c2[i]; c2[i] = s; s += t; }
        s = 0; for (int i = 0; i < 256; i++) { int t = c3[i]; c3[i] = s; s += t; }

        // 4 forward-scatter passes (stable LSD)
        for (int i = 0; i < n; i++) b[c0[a[i] & 0xFF]++] = a[i];           // bits 0-7
        for (int i = 0; i < n; i++) a[c1[(b[i] >> 8) & 0xFF]++] = b[i];    // bits 8-15
        for (int i = 0; i < n; i++) b[c2[(a[i] >> 16) & 0xFF]++] = a[i];   // bits 16-23
        for (int i = 0; i < n; i++) a[c3[b[i] >> 24]++] = b[i];            // bits 24-31
    }

    // Hash output
    u32 x = 998244353, ret = 0;
    for (int i = 0; i < n; i++) {
        ret ^= (a[i] + x);
        x = nextInt(x);
    }
    printf("%u\n", ret);

    return 0;
}