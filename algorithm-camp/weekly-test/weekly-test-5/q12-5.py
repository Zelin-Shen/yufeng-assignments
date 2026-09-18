#!/usr/bin/env pypy3
"""
Line Segment Intersection

Problem Description:
    Given two line segments in 2D plane, determine their intersection:
    - Output "-1" if no intersection
    - Output "inf" if infinitely many intersections (overlapping collinear segments)
    - Output the intersection point (x, y) with 4 decimal places if exactly one

Algorithm:
    1. Compute cross products to check straddling.
    2. Handle collinear case separately via bounding box overlap.
    3. Use parametric form for intersection computation.

Complexity:
    - Time: O(1) per test case
    - Space: O(1) per test case
"""

import sys


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return

    p = 0
    T = int(data[p])
    p += 1
    out = []

    for _ in range(T):
        x1 = int(data[p])
        p += 1
        y1 = int(data[p])
        p += 1
        x2 = int(data[p])
        p += 1
        y2 = int(data[p])
        p += 1
        x3 = int(data[p])
        p += 1
        y3 = int(data[p])
        p += 1
        x4 = int(data[p])
        p += 1
        y4 = int(data[p])
        p += 1

        # Direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        # Cross product helper
        def cross(ax, ay, bx, by):
            return ax * by - ay * bx

        # Compute cross products for straddling check
        d1 = cross(x1 - x3, y1 - y3, dx2, dy2)
        d2 = cross(x2 - x3, y2 - y3, dx2, dy2)
        d3 = cross(x3 - x1, y3 - y1, dx1, dy1)
        d4 = cross(x4 - x1, y4 - y1, dx1, dy1)

        # Collinear case: all cross products are 0
        if d1 == 0 and d2 == 0 and d3 == 0 and d4 == 0:
            # Check overlap using bounding boxes
            lx1, rx1 = (x1, x2) if x1 <= x2 else (x2, x1)
            lx2, rx2 = (x3, x4) if x3 <= x4 else (x4, x3)
            ly1, ry1 = (y1, y2) if y1 <= y2 else (y2, y1)
            ly2, ry2 = (y3, y4) if y3 <= y4 else (y4, y3)

            if rx1 < lx2 or rx2 < lx1 or ry1 < ly2 or ry2 < ly1:
                # No overlap
                out.append("-1")
            else:
                # Overlap exists
                ol_x1 = max(lx1, lx2)
                ol_x2 = min(rx1, rx2)
                ol_y1 = max(ly1, ly2)
                ol_y2 = min(ry1, ry2)

                if ol_x1 == ol_x2 and ol_y1 == ol_y2:
                    # Single point intersection
                    out.append(f"{ol_x1:.4f} {ol_y1:.4f}")
                else:
                    # Infinite intersections
                    out.append("inf")
            continue

        # General case: check if segments straddle each other
        if d1 * d2 < 0 and d3 * d4 < 0:
            # Intersection exists and is unique
            denom = cross(dx1, dy1, dx2, dy2)

            if denom == 0:
                out.append("-1")
                continue

            # Compute intersection using parametric form
            num_t = cross(x3 - x1, y3 - y1, dx2, dy2)

            px = x1 + num_t * dx1 / denom
            py = y1 + num_t * dy1 / denom

            out.append(f"{px:.4f} {py:.4f}")
        else:
            # Check endpoint touch cases
            def on_segment(px, py, ax, ay, bx, by):
                if cross(ax - px, ay - py, bx - px, by - py) != 0:
                    return False
                return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(
                    ay, by
                )

            touch = None
            if d1 == 0 and on_segment(x1, y1, x3, y3, x4, y4):
                touch = (x1, y1)
            elif d2 == 0 and on_segment(x2, y2, x3, y3, x4, y4):
                touch = (x2, y2)
            elif d3 == 0 and on_segment(x3, y3, x1, y1, x2, y2):
                touch = (x3, y3)
            elif d4 == 0 and on_segment(x4, y4, x1, y1, x2, y2):
                touch = (x4, y4)

            if touch:
                out.append(f"{touch[0]:.4f} {touch[1]:.4f}")
            else:
                out.append("-1")

    sys.stdout.write("\n".join(out))


solve()
