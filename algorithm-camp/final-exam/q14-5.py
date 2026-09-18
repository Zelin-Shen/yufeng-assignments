#!/usr/bin/env pypy3
"""
Farthest Pair of Points

Problem Description:
    Given n points on a plane, find the maximum distance between any two
    points. Output the distance rounded to 2 decimal places.

Algorithm:
    1. Build convex hull using Andrew's monotone chain: O(n log n).
    2. Rotating calipers for diameter: O(h) where h = hull size.
    3. Work with squared distances to avoid sqrt until output.

Complexity:
    O(n log n) time, O(n) space.
"""

import sys
from math import sqrt


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    ptr = 0
    n = int(data[ptr])
    ptr += 1

    px = [0] * n
    py = [0] * n
    for i in range(n):
        px[i] = int(data[ptr])
        ptr += 1
        py[i] = int(data[ptr])
        ptr += 1

    pts = sorted(set(zip(px, py)))
    n = len(pts)

    if n == 1:
        sys.stdout.write("0.00")
        return

    if n == 2:
        dx = pts[0][0] - pts[1][0]
        dy = pts[0][1] - pts[1][1]
        sys.stdout.write(f"{sqrt(dx * dx + dy * dy):.2f}")
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    def cross(oi, ai, bi):
        return (xs[ai] - xs[oi]) * (ys[bi] - ys[oi]) - (ys[ai] - ys[oi]) * (
            xs[bi] - xs[oi]
        )

    lower = []
    for i in range(n):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)

    upper = []
    for i in range(n - 1, -1, -1):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)

    hull = lower[:-1] + upper[:-1]
    h = len(hull)

    if h == 2:
        dx = xs[hull[0]] - xs[hull[1]]
        dy = ys[hull[0]] - ys[hull[1]]
        sys.stdout.write(f"{sqrt(dx * dx + dy * dy):.2f}")
        return

    hx = [xs[i] for i in hull]
    hy = [ys[i] for i in hull]

    max_d2 = 0
    j = 1
    for i in range(h):
        ni = (i + 1) if i + 1 < h else 0
        dxi = hx[ni] - hx[i]
        dyi = hy[ni] - hy[i]
        while True:
            nj = (j + 1) if j + 1 < h else 0
            if dxi * (hy[nj] - hy[j]) - dyi * (hx[nj] - hx[j]) > 0:
                j = nj
            else:
                break
        dx = hx[i] - hx[j]
        dy = hy[i] - hy[j]
        d2 = dx * dx + dy * dy
        if d2 > max_d2:
            max_d2 = d2
        nj = (j + 1) if j + 1 < h else 0
        dx = hx[i] - hx[nj]
        dy = hy[i] - hy[nj]
        d2 = dx * dx + dy * dy
        if d2 > max_d2:
            max_d2 = d2

    sys.stdout.write(f"{sqrt(max_d2):.2f}")


solve()
