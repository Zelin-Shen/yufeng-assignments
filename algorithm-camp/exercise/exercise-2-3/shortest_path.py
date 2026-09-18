#!/usr/bin/env pypy3
"""
Shortest Path Problem - Dijkstra Algorithm with Heap Optimization

Problem Description:
Given a directed weighted graph with n nodes and m edges.
Find the shortest path from node S to node T.

Algorithm: Dijkstra with Priority Queue (Heap)
1. Use min-heap to always process the node with smallest distance
2. For each node, update distances to its neighbors
3. Mark nodes as visited to avoid reprocessing

Time Complexity: O((n + m) log n)
Space Complexity: O(n + m)
"""

import sys
from heapq import heappush, heappop

n, m, S, T = map(int, sys.stdin.readline().split())

graph = [[] for _ in range(n + 1)]
for _ in range(m):
	u, v, w = map(int, sys.stdin.readline().split())
	graph[u].append((v, w))
	graph[v].append((u, w))
	
INF = float('inf')
dist = [INF] * (n + 1)
dist[S] = 0
visited = [False] * (n + 1)

heap = [(0, S)]

while heap:
	d, u = heappop(heap)
	
	if visited[u]:
		continue
	
	visited[u] = True
	
	for v, w in graph[u]:
		if not visited[v]:
			nd = d + w
			if nd < dist[v]:
				dist[v] = nd
				heappush(heap, (nd, v))
				
if dist[T] == INF:
	print(-1)
else:
	print(dist[T])