#!/usr/bin/env pypy3
"""
Delete At Most One Edge to Make DAG

Problem Description:
    Given a directed graph with n vertices and m edges, determine if
    deleting at most one edge can make it acyclic (a DAG).

Algorithm:
    1. Run Kahn's algorithm. If all n nodes processed -> already DAG -> YES.
    2. Otherwise, find a cycle via iterative DFS using coloring:
       0 = unvisited, 1 = in-progress, 2 = done.
    3. For each edge on the cycle, test removal by re-running Kahn's.
    4. The critical edge (if any) must lie on every cycle, so trying edges
       from one found cycle suffices.

Complexity:
    O(n * (n + m)) time, O(n + m) space.
"""

import sys


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    m = int(data[ptr])
    ptr += 1

    adj = [[] for _ in range(n)]
    in_deg = [0] * n
    for _ in range(m):
        u = int(data[ptr]) - 1
        ptr += 1
        v = int(data[ptr]) - 1
        ptr += 1
        adj[u].append(v)
        in_deg[v] += 1

    def is_dag_skip(su, sv):
        deg = list(in_deg)
        if su >= 0:
            deg[sv] -= 1
        q = [0] * n
        head = 0
        tail = 0
        for i in range(n):
            if deg[i] == 0:
                q[tail] = i
                tail += 1
        cnt = 0
        while head < tail:
            u = q[head]
            head += 1
            cnt += 1
            for v in adj[u]:
                if u == su and v == sv:
                    continue
                deg[v] -= 1
                if deg[v] == 0:
                    q[tail] = v
                    tail += 1
        return cnt == n

    if is_dag_skip(-1, -1):
        sys.stdout.write("YES")
        return

    # Find a cycle via iterative DFS
    color = [0] * n
    par = [-1] * n
    cycle = None

    for start in range(n):
        if color[start]:
            continue
        stack = [(start, 0)]
        color[start] = 1
        while stack:
            u, ei = stack[-1]
            if ei < len(adj[u]):
                stack[-1] = (u, ei + 1)
                v = adj[u][ei]
                if color[v] == 1:
                    cyc = [(u, v)]
                    cur = u
                    while cur != v:
                        cyc.append((par[cur], cur))
                        cur = par[cur]
                    cycle = cyc
                    break
                if color[v] == 0:
                    color[v] = 1
                    par[v] = u
                    stack.append((v, 0))
            else:
                color[u] = 2
                stack.pop()
        if cycle is not None:
            break

    if cycle is None:
        sys.stdout.write("YES")
        return

    for su, sv in cycle:
        if is_dag_skip(su, sv):
            sys.stdout.write("YES")
            return

    sys.stdout.write("NO")


solve()
