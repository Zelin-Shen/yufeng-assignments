#!/usr/bin/env pypy3
"""
String Partition into T-Prefixes

Problem Description:
    Given two strings T and S, partition S into the minimum number of
    segments such that each segment is a non-empty prefix of T.
    Output the minimum number of segments, or "Fake" if impossible.

Algorithm:
    1. Compute Z-function of T (O(|T|)).
    2. Use Extended KMP to compute, for each position i in S, the longest
       common prefix of S[i..] and T[0..]. Call this lcp[i] (O(|S|)).
    3. Solve a Jump Game II problem: from position i we can jump 1..lcp[i]
       steps. Find the minimum number of jumps to reach position |S|.

Complexity:
    O(|S| + |T|) time, O(|T|) space.
"""

import sys
from array import array


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return

    T = data[2]
    S = data[3]
    tl = len(T)
    sl = len(S)

    if sl == 0:
        sys.stdout.write("0")
        return

    # Z-function of T
    zt = array("I", bytes(4 * tl))
    zt[0] = tl
    li = 0
    r = 0
    for i in range(1, tl):
        if i < r:
            v = zt[i - li]
            if v < r - i:
                zt[i] = v
                continue
            zt[i] = r - i
        while zt[i] < tl - i and T[zt[i]] == T[i + zt[i]]:
            zt[i] += 1
        if i + zt[i] > r:
            li = i
            r = i + zt[i]

    # Extended KMP + Jump Game II (online)
    li = 0
    r = 0
    jumps = 0
    cur_end = 0
    farthest = 0

    for i in range(sl):
        if i < r:
            lcp_i = zt[i - li]
            if lcp_i > r - i:
                lcp_i = r - i
        else:
            lcp_i = 0
        while lcp_i < tl and i + lcp_i < sl and S[i + lcp_i] == T[lcp_i]:
            lcp_i += 1
        if i + lcp_i > r:
            li = i
            r = i + lcp_i

        end = i + lcp_i
        if end > farthest:
            farthest = end
        if i == cur_end:
            if farthest <= i:
                sys.stdout.write("Fake")
                return
            jumps += 1
            cur_end = farthest
            if cur_end >= sl:
                break

    sys.stdout.write(str(jumps) if farthest >= sl else "Fake")


solve()
