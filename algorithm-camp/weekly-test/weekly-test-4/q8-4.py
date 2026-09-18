#!/usr/bin/env pypy3
"""
Longest Prefix Match in Dictionary Set - Trie + Greedy Traversal

Problem Description:
    Given a set S of n lowercase strings and q query strings t,
    for each query we need the maximum i such that prefix t[0:i] exists
    as a complete string in S.

Algorithm:
    1. Build Trie:
       Insert all strings in S into a 26-ary Trie (lowercase letters only).
       Each node stores:
       - 26 child pointers (for 'a'..'z')
       - an end marker indicating whether a complete dictionary string ends here

    2. Query Processing:
       For each query string t, traverse the Trie character by character:
       - If next child does not exist, traversal stops immediately.
       - If current node is an end marker, update answer with current depth.
       The final recorded depth is the longest prefix length that is in S.

Complexity:
    Let Ls = total length of all strings in S, Lt = total length of all query strings.
    - Build: O(Ls)
    - Queries: O(Lt)
    - Total: O(Ls + Lt)
    - Space: O(number of Trie nodes * 26), bounded by total inserted characters.
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

    # Trie:
    # node 0 = root
    # nxt[u*26 + c] = child index, -1 if absent
    nxt = array("i", [-1] * 26)
    is_end = bytearray(1)
    nodes = 1

    # Build
    for _ in range(n):
        s = data[p]
        p += 1  # bytes
        u = 0
        for ch in s:
            c = ch - 97
            idx = u * 26 + c
            v = nxt[idx]
            if v == -1:
                v = nodes
                nodes += 1
                nxt[idx] = v
                nxt.extend([-1] * 26)
                is_end.append(0)
            u = v
        is_end[u] = 1

    out = []
    push = out.append

    # Query
    for _ in range(q):
        t = data[p]
        p += 1
        u = 0
        best = 0
        depth = 0
        for ch in t:
            v = nxt[u * 26 + (ch - 97)]
            if v == -1:
                break
            u = v
            depth += 1
            if is_end[u]:
                best = depth
        push(str(best))

    sys.stdout.write("\n".join(out))


solve()
