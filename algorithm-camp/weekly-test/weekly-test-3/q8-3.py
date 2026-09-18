#!/usr/bin/env pypy3
"""
DNA Sequence Similarity via Global Alignment.

Scoring: match=4, mismatch=2, gap=1, double-gap=0.
Key insight: optimal similarity = n + m + 2 * LCS(A, B).

Proof: Let k = number of aligned non-gap pairs, matches = number of
exact matches among them. Alignment length = n + m - k.
Total = 1*(n+m-2k) + 2*(k-matches) + 4*matches = n + m + 2*matches.
Maximizing matches is exactly the LCS problem.

Uses bit-parallel LCS algorithm (Allison-Dill / Hyyro) for
O(n * m / w) time where w = 64 (machine word size).
"""

import sys


def solve():
    data = sys.stdin.buffer.read().split()
    A = data[1]
    m = int(data[2])
    B = data[3]

    # Precompute match bitmasks for each character in alphabet
    PM = [0] * 256
    for j in range(m):
        PM[B[j]] |= 1 << j

    # Bit-parallel LCS
    # M encodes horizontal differences of DP row: M[j]=1 iff dp[i][j]>dp[i][j-1]
    M = 0
    mask = (1 << m) - 1

    for i in range(len(A)):
        X = M | PM[A[i]]
        Y = ((M << 1) | 1) & mask
        M = X & ((X - Y) ^ X)

    lcs = bin(M).count("1")
    sys.stdout.write(str(len(A) + m + 2 * lcs) + "\n")


solve()
