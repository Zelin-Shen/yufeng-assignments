#!/usr/bin/env pypy3
"""
Fast Power Modulo - Calculate last 3 digits of a^b

Algorithm: Fast exponentiation with modulo
- a^b mod 1000
- Time complexity: O(log b) per query
"""

import sys

inp = sys.stdin.buffer.read().decode().split()
T = int(inp[0])

res = []
for i in range(1, T * 2, 2):
    a = int(inp[i]) % 1000
    b = int(inp[i + 1])

    ans = 1
    while b:
        if b & 1:
            ans = ans * a % 1000
        a = a * a % 1000
        b >>= 1

    res.append(str(ans))

sys.stdout.write("\n".join(res) + "\n")
