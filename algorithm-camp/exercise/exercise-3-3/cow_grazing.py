#!/usr/bin/env pypy3
"""
Cow Eating Grass - Interval DP Solution

Key insights from hints:
1. Loss = time_interval * remaining_grass_count
2. Cow must traverse a continuous interval
3. After eating grass in [l,r], cow is at position l or r
4. Sort grass positions first

State: dp[l][r][j] = minimum loss when eating grass in [l,r]
		j=0: cow ends at position l
		j=1: cow ends at position r

Transition:
dp[l][r][0] = min(
	dp[l+1][r][0] + (n-r+l) * |x[l]-x[l+1]|,  # from l+1
	dp[l+1][r][1] + (n-r+l) * |x[l]-x[r]|     # from r
)

dp[l][r][1] = min(
	dp[l][r-1][1] + (n-r+l) * |x[r]-x[r-1]|,  # from r-1
	dp[l][r-1][0] + (n-r+l) * |x[r]-x[l]|     # from l
)

Base: dp[i][i][0] = dp[i][i][1] = |x[i]-k| * n
"""

import sys

data = list(map(int, sys.stdin.read().split()))
n, k = data[0], data[1]
x = sorted(data[2:n + 2])

INF = 10**18
dp = [[[INF, INF] for _ in range(n)] for _ in range(n)]

for i in range(n):
	v = abs(x[i] - k) * n
	dp[i][i][0] = dp[i][i][1] = v
	
for ln in range(2, n + 1):
	for l in range(n - ln + 1):
		r = l + ln - 1
		c = n - r + l
		l1, r1 = l + 1, r - 1
		
		v1 = dp[l1][r][0] + c * abs(x[l] - x[l1])
		v2 = dp[l1][r][1] + c * abs(x[l] - x[r])
		dp[l][r][0] = v1 if v1 < v2 else v2
		
		v1 = dp[l][r1][1] + c * abs(x[r] - x[r1])
		v2 = dp[l][r1][0] + c * abs(x[r] - x[l])
		dp[l][r][1] = v1 if v1 < v2 else v2
		
print(min(dp[0][n - 1]))