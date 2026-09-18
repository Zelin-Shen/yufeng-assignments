#!/usr/bin/env pypy3
"""
Digital Triangle - Dynamic Programming Solution

Problem Description:
Given a digital triangle with n rows, find the maximum sum path from top to bottom.
At each position (i,j), you can move to (i+1,j) or (i+1,j+1).

Algorithm: Dynamic Programming
- State: dp[i][j] = maximum sum to reach position (i,j)
- Transition: dp[i][j] = triangle[i][j] + max(dp[i-1][j-1], dp[i-1][j])
- Base case: dp[0][0] = triangle[0][0]

Time Complexity: O(n^2)
Space Complexity: O(n)
"""

import sys

n = int(sys.stdin.readline())
prev = list(map(int, sys.stdin.readline().split()))

for i in range(1, n):
	curr = list(map(int, sys.stdin.readline().split()))
	
	curr[0] += prev[0]
	for j in range(1, i):
		curr[j] += max(prev[j-1], prev[j])
	curr[i] += prev[i-1]
	
	prev = curr
	
print(max(prev))