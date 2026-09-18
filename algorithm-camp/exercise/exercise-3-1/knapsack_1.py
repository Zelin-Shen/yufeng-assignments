#!/usr/bin/env pypy3
"""
0-1 Knapsack Problem - Space Optimized DP

Problem Description:
Given n items, each with kind (0=single, 1=multiple), value, and weight.
Find maximum value that can fit in a knapsack of capacity V.
- Kind 0: can only take once (0-1 knapsack)
- Kind 1: can take multiple times (unbounded knapsack)

Algorithm: Dynamic Programming with 1D array
- State: dp[j] = maximum value with capacity j
- For kind 0: iterate j from V to weight (reverse to avoid reusing)
- For kind 1: iterate j from weight to V (forward to allow reusing)

Time Complexity: O(n * V)
Space Complexity: O(V)
"""

n, V = map(int, input().split())
dp = [0] * (V + 1)

for _ in range(n):
	k, v, w = map(int, input().split())
	
	if k:
		# Complete knapsack
		for j in range(w, V + 1):
			new = dp[j - w] + v
			if new > dp[j]:
				dp[j] = new
	else:
		# 0-1 knapsack
		for j in range(V, w - 1, -1):
			new = dp[j - w] + v
			if new > dp[j]:
				dp[j] = new
				
print(dp[V])