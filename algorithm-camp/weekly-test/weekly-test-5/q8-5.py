#!/usr/bin/env pypy3
"""
Right Triangle Counting

Problem Description:
    Given n distinct points P_i = (x_i, y_i) on a plane, count the number of
    right triangles that can be formed using these points as vertices.

Algorithm:
    For each point O as the potential right-angle vertex:
    1. Compute direction vectors from O to all other points.
    2. Normalize each vector (dx, dy) by gcd to get unique direction keys.
    3. Count perpendicular direction pairs via hash map lookup.

Complexity:
    - Time: O(n^2 * log(max_coord))
    - Space: O(n)
"""

import sys
from math import gcd


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return

    p = 0
    n = int(data[p])
    p += 1

    if n < 3:
        sys.stdout.write("0")
        return

    xs = [0] * n
    ys = [0] * n
    for i in range(n):
        xs[i] = int(data[p])
        p += 1
        ys[i] = int(data[p])
        p += 1

    ans = 0

    for i in range(n):
        ox, oy = xs[i], ys[i]
        cnt = {}

        for j in range(n):
            if i == j:
                continue
            dx = xs[j] - ox
            dy = ys[j] - oy

            g = gcd(dx, dy) if dx != 0 or dy != 0 else 1
            dx //= g
            dy //= g

            # Canonical form: make unique
            if dx < 0 or (dx == 0 and dy < 0):
                dx = -dx
                dy = -dy

            key = (dx, dy)
            cnt[key] = cnt.get(key, 0) + 1

        for (dx, dy), c1 in cnt.items():
            # Perpendicular: rotate 90 degrees
            px = -dy
            py = dx

            if px < 0 or (px == 0 and py < 0):
                px = -px
                py = -py

            # Avoid double counting
            if (dx, dy) < (px, py):
                c2 = cnt.get((px, py))
                if c2:
                    ans += c1 * c2

    sys.stdout.write(str(ans))


solve()
