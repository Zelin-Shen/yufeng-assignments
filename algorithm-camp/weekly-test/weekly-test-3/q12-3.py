#!/usr/bin/env pypy3
"""
Minimum Generating Set of Numerical Semigroup

Given set A of positive integers, <A> = all positive integers representable
as sum of one or more elements of A (with repetition).
Find the smallest B such that <B> = <A>.

Key insight: a in A is a generator iff it cannot be expressed as a sum of
two or more elements from A. Equivalently, for sorted unique A, element a
is a generator iff no c in A with c < a satisfies (a - c) in <A>.

Since <A> includes every element of A, (a-c) in <A> means (a-c) is
representable by elements of A. For c < a, a-c <= max(A)-min(A) = D.
Elements > D cannot appear in any representation of values in [1..D].
So we only knapsack on elements <= D, with bitset doubling trick.

Time:  O(|gen| * log(D) * D/64 + |A|^2)
Space: O(D / 8)
"""

import sys
from bisect import bisect_right
from functools import reduce
from math import gcd


def solve():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    A = sorted(set(int(data[i + 1]) for i in range(n)))
    k = len(A)

    if k == 1:
        sys.stdout.write("1\n")
        return

    g = reduce(gcd, A)
    if g > 1:
        A = [a // g for a in A]

    if A[0] == 1:
        sys.stdout.write("1\n")
        return

    D = A[-1] - A[0]

    # --- Phase 1: bitset knapsack on [0..D] for elements <= D ---
    reach = 1
    mask = (1 << (D + 1)) - 1
    ans = 0
    bd = bisect_right(A, D)  # A[j] <= D for j < bd

    for idx in range(bd):
        a = A[idx]
        if (reach >> a) & 1:
            continue
        ans += 1
        s = a
        while s <= D:
            reach |= reach << s
            s <<= 1
        reach &= mask

    # --- Phase 2: elements > D, check via byte-level bit queries ---
    rb = reach.to_bytes((D >> 3) + 1, "little")

    for i in range(bd, k):
        a = A[i]
        decomp = False
        for j in range(bd):
            v = a - A[j]
            if (rb[v >> 3] >> (v & 7)) & 1:
                decomp = True
                break
        if not decomp:
            ans += 1

    sys.stdout.write(str(ans) + "\n")


solve()
