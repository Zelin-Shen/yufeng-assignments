#!/usr/bin/env pypy3
"""
KMP (Knuth-Morris-Pratt) string matching algorithm.
Operates on raw bytes for maximum speed in Python.
- Build failure/prefix function in O(m).
- Search in O(n) with amortized O(1) per character.
Overall time complexity: O(n + m).
"""

import sys

def solve():
	_read = sys.stdin.buffer.readline
	n = int(_read())
	A = _read().rstrip()
	m = int(_read())
	B = _read().rstrip()
	
	if m > n:
		return
	
	fail = [0] * m
	j = 0
	for i in range(1, m):
		while j and B[i] != B[j]:
			j = fail[j - 1]
		if B[i] == B[j]:
			j += 1
		fail[i] = j
	fail = tuple(fail)
	
	res = []
	ap = res.append
	j = 0
	mm1 = m - 1
	for i, c in enumerate(A):
		while j and c != B[j]:
			j = fail[j - 1]
		if c == B[j]:
			j += 1
			if j == m:
				ap(i - mm1)
				j = fail[mm1]
				
	if res:
		sys.stdout.write('\n'.join(map(str, res)) + '\n')
		
solve()