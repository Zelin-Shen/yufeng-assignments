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

MOD = 1000000007

s = sys.stdin.buffer.readline().strip()
n = len(s)

if n == 0:
    sys.stdout.write("1")
    raise SystemExit

a = bytearray(n + 1)
a[1:] = s

pi = array("I", [0]) * (n + 1)
j = 0
for i in range(2, n + 1):
    c = a[i]
    while j and a[j + 1] != c:
        j = pi[j]
    if a[j + 1] == c:
        j += 1
    pi[i] = j

dep = array("I", [0]) * (n + 1)
for i in range(1, n + 1):
    dep[i] = dep[pi[i]] + 1

LOG = (n + 1).bit_length()
up = [array("I", [0]) * (n + 1) for _ in range(LOG)]
up[0] = pi
for b in range(1, LOG):
    prev = up[b - 1]
    cur = up[b]
    for v in range(n + 1):
        cur[v] = prev[prev[v]]

ans = 1
for i in range(1, n + 1):
    half = i >> 1
    t = i
    for b in range(LOG - 1, -1, -1):
        anc = up[b][t]
        if anc > half:
            t = anc
    c = dep[pi[t]]
    ans = (ans * (c + 1)) % MOD

sys.stdout.write(str(ans))
