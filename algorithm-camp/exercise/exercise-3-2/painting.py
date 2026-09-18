#!/usr/bin/env pypy3
"""
Paint Cars - DP with State Compression

Problem: n cars in a row, m paint colors, each color can paint at most a_i cars.
Find number of ways to paint all cars such that no two adjacent cars have same color.

Key insight: Since a_i ≤ 5, we can use state (c1, c2, c3, c4, c5) where:
- c_i = number of colors that can still paint i more cars

State transition: For each car, try each available color and update state.

Time Complexity: O(n * states * colors)
Space Complexity: O(states)
"""

import sys
input = sys.stdin.readline

MOD = 23333

m = int(input())
paints = list(map(int, input().split()))
n = sum(paints)

cnt = [0] * 6
for p in paints:
    cnt[p] += 1
    
dp = {(cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], 0): 1}

for _ in range(n):
    ndp = {}
    
    for (c1, c2, c3, c4, c5, last), ways in dp.items():
        state = [0, c1, c2, c3, c4, c5]
        
        for cap in range(1, 6):
            if state[cap] == 0:
                continue
            
            if cap == last - 1 and last > 0:
                avail = state[cap] - 1
            else:
                avail = state[cap]
                
            if avail <= 0:
                continue
            
            ns = state[:]
            ns[cap] -= 1
            if cap > 1:
                ns[cap - 1] += 1
                
            key = (ns[1], ns[2], ns[3], ns[4], ns[5], cap)
            ndp[key] = (ndp.get(key, 0) + ways * avail) % MOD
            
    dp = ndp
    
print(sum(dp.values()) % MOD)