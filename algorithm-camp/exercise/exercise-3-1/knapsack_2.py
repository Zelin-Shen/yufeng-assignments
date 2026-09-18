#!/usr/bin/env pypy3
"""
Knapsack Problem 2 - Multiple Queries with Item Exclusion

Problem Description:
Given n items with weight and value, answer q queries.
Each query asks: what's the max value if we exclude item x?

Algorithm: Prefix and Suffix Knapsack Preprocessing
1. Compute prefix[i]: knapsack with items 0 to i-1
2. Compute suffix[i]: knapsack with items i+1 to n-1
3. For query x: merge prefix[x] and suffix[x]

Time Complexity: O(n * MAXV + q * MAXV)
Space Complexity: O(nMAXV)
"""

import sys
input = sys.stdin.readline

MAXV = 5001

n = int(input())
items = [tuple(map(int, input().split())) for _ in range(n)]

prefix = [[0] * MAXV for _ in range(n + 1)]
for i in range(n):
	w, v = items[i]
	for j in range(MAXV):
		if j < w:
			prefix[i + 1][j] = prefix[i][j]
		else:
			prefix[i + 1][j] = max(prefix[i][j], prefix[i][j - w] + v)
			
suffix = [[0] * MAXV for _ in range(n + 1)]
for i in range(n - 1, -1, -1):
	w, v = items[i]
	for j in range(MAXV):
		if j < w:
			suffix[i][j] = suffix[i + 1][j]
		else:
			suffix[i][j] = max(suffix[i + 1][j], suffix[i + 1][j - w] + v)
			
q = int(input())
for _ in range(q):
	V, x = map(int, input().split())
	x -= 1
	
	ans = 0
	for j in range(V + 1):
		ans = max(ans, prefix[x][j] + suffix[x + 1][V - j])
		
	print(ans)