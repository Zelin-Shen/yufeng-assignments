"""
Assignment Problem (Workload) - Bitmask DP Solution

Given an n x n cost matrix where t[i][k] is the time for person i to complete job k,
find the minimum total time to assign exactly one job to each person.

Approach: Use bitmask dynamic programming where dp[mask] represents the minimum cost
to assign the jobs indicated by set bits in 'mask' to the first popcount(mask) persons.
For each state, we try assigning each unassigned job to the next person.

Time Complexity:  O(n * 2^n)
Space Complexity: O(2^n)
"""

import sys
input = sys.stdin.readline

def solve():
    n = int(input())
    cost = [list(map(int, input().split())) for _ in range(n)]

    # dp[mask] = min cost assigning jobs in mask to first popcount(mask) persons
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        i = bin(mask).count('1')  # current person index
        if i >= n:
            continue
        for j in range(n):        # try assigning job j to person i
            if mask >> j & 1:
                continue
            nxt = mask | (1 << j)
            val = dp[mask] + cost[i][j]
            if val < dp[nxt]:
                dp[nxt] = val

    print(dp[(1 << n) - 1])

solve()