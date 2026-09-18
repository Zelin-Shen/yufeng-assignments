#!/usr/bin/env pypy3
"""
Minimum Maximum Weight Path - Binary Search + BFS

Problem: Find path from s to t that minimizes the maximum node weight.

Algorithm:
1. Binary search on the maximum weight limit
2. For each limit, check if s can reach t using only nodes with weight <= limit
3. Find the minimum valid limit

Time Complexity: O((n+m)log(n))
"""

import sys

d = list(map(int, sys.stdin.read().split()))
n, m = d[0], d[1]
w = [0] + d[2 : n + 2]

g = [[] for _ in range(n + 1)]
i = n + 2
for _ in range(m):
    u, v = d[i], d[i + 1]
    i += 2
    g[u].append(v)
    g[v].append(u)

s, t = d[i], d[i + 1]

p = list(range(n + 1))


def f(x):
    if p[x] != x:
        p[x] = f(p[x])
    return p[x]


a = [0] * (n + 1)

for wt, u in sorted((w[i], i) for i in range(1, n + 1)):
    a[u] = 1
    pu = f(u)

    for v in g[u]:
        if a[v]:
            pv = f(v)
            if pu != pv:
                p[pu] = pv
                pu = pv

    if a[s] and a[t] and f(s) == f(t):
        print(wt)
        sys.exit()

print(-1)
