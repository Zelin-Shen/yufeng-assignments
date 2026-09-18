#!/usr/bin/env pypy3
"""
Rounded Rectangle Convex Hull

Problem Description:
    Given n rounded rectangles with dimensions (a, b) and corner radius r,
    find the minimum rope length needed to enclose all rectangles.

Algorithm:
    1. For each rectangle, compute 4 corner arc centers (considering rotation).
    2. Build convex hull using Andrew's monotone chain algorithm.
    3. Calculate hull perimeter and add the rounded corner contribution (2*PI*r).

Complexity:
    - Time: O(n log n)
    - Space: O(n)
"""

import sys
from math import cos, pi, sin, sqrt

PI = pi


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return

    p = 0
    a = float(data[p])
    p += 1
    b = float(data[p])
    p += 1
    r = float(data[p])
    p += 1
    n = int(data[p])
    p += 1

    if n == 0:
        sys.stdout.write("0.00")
        return

    # Half dimensions (distance from center to corner arc center)
    ha = a / 2 - r
    hb = b / 2 - r

    # Collect all corner arc centers
    points = []

    for _ in range(n):
        cx = float(data[p])
        p += 1
        cy = float(data[p])
        p += 1
        theta = float(data[p])
        p += 1

        cos_t = cos(theta)
        sin_t = sin(theta)

        # 4 corners relative to center
        corners = [(ha, hb), (ha, -hb), (-ha, -hb), (-ha, hb)]

        for lx, ly in corners:
            # Rotate and translate
            rx = lx * cos_t - ly * sin_t + cx
            ry = lx * sin_t + ly * cos_t + cy
            points.append((rx, ry))

    # Remove duplicates and sort
    points = sorted(set(points))

    if len(points) == 1:
        sys.stdout.write(f"{2 * PI * r:.2f}")
        return

    # Build convex hull using Andrew's monotone chain
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Lower hull: left to right
    lower = []
    for pt in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)

    # Upper hull: right to left
    upper = []
    for pt in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)

    # Concatenate lower and upper hulls (remove duplicate endpoints)
    hull = lower[:-1] + upper[:-1]

    if len(hull) < 2:
        sys.stdout.write(f"{2 * PI * r:.2f}")
        return

    # Calculate perimeter
    perimeter = 0.0
    m = len(hull)
    for i in range(m):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % m]
        dx = x2 - x1
        dy = y2 - y1
        perimeter += sqrt(dx * dx + dy * dy)

    # Add rounded corners (full circle)
    perimeter += 2 * PI * r

    sys.stdout.write(f"{perimeter:.2f}")


solve()
