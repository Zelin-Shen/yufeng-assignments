#!/usr/bin/env pypy3
"""
Merge Materials (Minimum Total Cost)

Problem Description:
    Given n materials with masses a1..an, merge all into one through n-1
    operations. Each operation selects two materials (masses a and b) and
    merges them into one with mass a+b, at a cost of a*b.
    Find the minimum possible total cost.

Algorithm:
    The total cost is INVARIANT regardless of merge order.
    Total = sum of all pairwise products = (S^2 - Q) / 2
    where S = sum of all ai, Q = sum of ai^2.

Complexity:
    O(n) time, O(1) extra space.
"""

import sys


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    S = 0
    Q = 0
    for i in range(1, len(data)):
        a = int(data[i])
        S += a
        Q += a * a
    sys.stdout.write(str((S * S - Q) >> 1))


solve()
