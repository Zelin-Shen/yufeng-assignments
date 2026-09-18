#!/usr/bin/env pypy3
"""
Minimum Maximum Edge Weight with k Removals

Problem: Find path from 1 to n, remove at most k edges, minimize max remaining edge weight.

Algorithm:
1. Binary search on the answer (max edge weight)
2. For each candidate answer 'limit', check if there exists a path where:
        - Number of edges with weight > limit is at most k
3. Use Dijkstra to find path with minimum count of edges > limit

Time Complexity: O(log(m) * (m+n)log(n))
"""

import sys
from collections import deque

d = list(map(int, sys.stdin.read().split()))
n, m, k = d[0], d[1], d[2]

g = [[] for _ in range(n + 1)]
w = set([0])

i = 3
for _ in range(m):
    u, v, wt = d[i], d[i + 1], d[i + 2]
    i += 3
    g[u].append((v, wt))
    g[v].append((u, wt))
    w.add(wt)

wl = sorted(w)


def f(x):
    t = [k + 2] * (n + 1)
    t[1] = 0
    q = deque([1])

    while q:
        u = q.popleft()
        if u == n:
            return 1
        c = t[u]
        if c > k:
            continue
        for v, e in g[u]:
            nc = c + (e > x)
            if nc < t[v]:
                t[v] = nc
                if nc <= k:
                    (q.appendleft if e <= x else q.append)(v)
    return t[n] <= k


line, r = 0, len(wl) - 1
a = wl[-1]

while line <= r:
    m = (line + r) >> 1
    if f(wl[m]):
        a = wl[m]
        r = m - 1
    else:
        line = m + 1

print(a)
