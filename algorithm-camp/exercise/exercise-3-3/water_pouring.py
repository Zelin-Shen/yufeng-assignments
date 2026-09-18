#!/usr/bin/env pypy3
"""
Water Pouring Problem - Optimized BFS Solution

Problem: Two cups with capacity n and m. Find minimum operations to get exactly d units.
Operations: fill, empty, pour from one to another, do nothing.

Approach: BFS to find shortest path from (0,0) to any state with d units.

Time Complexity: O(n*m) - at most n*m states
Space Complexity: O(n*m)
"""

import sys
from collections import deque

n, m, t, d = map(int, sys.stdin.readline().split())

# BFS to explore all reachable states in t steps
q = deque([(0, 0, 0)])
vis = {(0, 0): 0}
min_diff = abs(d - 0)

while q:
	a, b, step = q.popleft()
	
	# Update minimum difference
	total = a + b
	min_diff = min(min_diff, abs(d - total))
	
	# If perfect match, return immediately
	if min_diff == 0:
		print(0)
		sys.exit()
		
	# Time limit
	if step >= t:
		continue
	
	ns = step + 1
	
	# Generate next states
	states = [
		(n, b),           # Fill cup 1
		(a, m),           # Fill cup 2
		(0, b),           # Empty cup 1
		(a, 0),           # Empty cup 2
		(a - min(a, m - b), b + min(a, m - b)),  # Pour 1->2
		(a + min(b, n - a), b - min(b, n - a))   # Pour 2->1
	]
	
	for s in states:
		if s not in vis or vis[s] > ns:
			vis[s] = ns
			q.append((s[0], s[1], ns))
			
print(min_diff)