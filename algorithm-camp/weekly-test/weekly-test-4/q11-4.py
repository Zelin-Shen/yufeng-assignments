#!/usr/bin/env pypy3
"""
Maximum XOR Query Solver

Overview:
    For each query y, compute max(x_i XOR y) over a static set of n numbers x_i.

Method:
    - Build a 32-level binary trie (bits 31..0) for all x_i.
    - Query greedily: at each bit, try opposite branch first to maximize XOR bit.

Complexity:
    - Build:  O(32 * n)
    - Query:  O(32 * q)
    - Total:  O(32 * (n + q))
    - Memory: O(32 * n)
"""

import sys
from array import array


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return

    p = 0
    n = int(data[p])
    p += 1
    q = int(data[p])
    p += 1

    # Upper bound: root + 32 nodes per inserted number
    max_nodes = (n << 5) + 1

    # Child pointers for bit 0 / bit 1
    ch0 = array("i", [-1]) * max_nodes
    ch1 = array("i", [-1]) * max_nodes

    c0 = ch0
    c1 = ch1
    cnt = 1  # node 0 is root

    # Insert numbers
    for _ in range(n):
        x = int(data[p])
        p += 1
        u = 0
        b = 31
        while b >= 0:
            if (x >> b) & 1:
                v = c1[u]
                if v == -1:
                    v = cnt
                    c1[u] = v
                    cnt += 1
                u = v
            else:
                v = c0[u]
                if v == -1:
                    v = cnt
                    c0[u] = v
                    cnt += 1
                u = v
            b -= 1

    out = []
    push = out.append

    # Answer queries
    for _ in range(q):
        y = int(data[p])
        p += 1
        u = 0
        ans = 0
        b = 31
        while b >= 0:
            bit = (y >> b) & 1
            if bit == 0:
                v = c1[u]
                if v != -1:
                    ans |= 1 << b
                    u = v
                else:
                    u = c0[u]
            else:
                v = c0[u]
                if v != -1:
                    ans |= 1 << b
                    u = v
                else:
                    u = c1[u]
            b -= 1
        push(str(ans))

    sys.stdout.write("\n".join(out))


solve()
