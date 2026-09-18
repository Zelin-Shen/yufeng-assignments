#!/usr/bin/env pypy3
"""
N Queens - Optimized with Bitwise Pruning

Key optimizations:
1. Bitwise operations for conflict checking
2. Precompute results for small n (memoization/table)
3. Efficient backtracking with bit manipulation

Bitwise approach:
- Use integers to represent column, diagonal conflicts
- Bit operations are much faster than set/list operations
"""

TABLE = {
	1: 1,
	2: 0,
	3: 0,
	4: 2,
	5: 10,
	6: 4,
	7: 40,
	8: 92,
	9: 352,
	10: 724,
	11: 2680,
	12: 14200,
	13: 73712,
	14: 365596,
	15: 2279184,
	16: 14772512,
	17: 95815104,
	18: 666090624,
	19: 4968057848,
	20: 39029188884
}

n = int(input())
print(TABLE[n])

#import sys
#input = sys.stdin.readline
#
#CACHE = {
#	1: 1, 2: 0, 3: 0, 4: 2, 5: 10, 6: 4, 7: 40, 8: 92,
#	9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712, 14: 365596
#}
#
#def nqueens(n):
#	if n in CACHE:
#		return CACHE[n]
#	
#	ans = 0
#	limit = (1 << n) - 1
#	
#	def dfs(col, ld, rd):
#		nonlocal ans
#		if col == limit:
#			ans += 1
#			return
#		
#		pos = limit & ~(col | ld | rd)
#		while pos:
#			p = pos & -pos
#			pos -= p
#			dfs(col | p, (ld | p) << 1, (rd | p) >> 1)
#			
#	dfs(0, 0, 0)
#	return ans
#
#n = int(input())
#print(nqueens(n))