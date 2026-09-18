#!/usr/bin/env pypy3
"""
Longest Common Subsequence of Permutations - Optimized to LIS

Key insight: For permutations, LCS(A,B) = LIS of transformed sequence.

Algorithm:
1. Map each element in B to its position
2. Transform A by replacing each element with its position in B
3. Find LIS using binary search (O(n log n))

Time Complexity: O(n log n)
Space Complexity: O(n)
"""

import sys
from bisect import bisect_left

input = sys.stdin.readline

n = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

pos = [0] * (n + 1)
for i in range(n):
	pos[B[i]] = i
	
transformed = [pos[a] for a in A]

dp = []

for num in transformed:
	idx = bisect_left(dp, num)
	if idx == len(dp):
		dp.append(num)
	else:
		dp[idx] = num
		
print(len(dp))