#!/usr/bin/env pypy3
"""
2D Matrix Pattern Matching using 2D Rolling Hash

Algorithm:
1. Row-wise rolling hash of length d for each row of the large matrix
2. Column-wise rolling hash of length c on the row-hash matrix
3. Same 2D hash for the pattern matrix to get target value
4. Scan the final hash matrix for matches

Uses dual moduli to minimize hash collision probability.
Time complexity: O(a * b), Space complexity: O(a * b)
"""

import sys

def solve():
	data = sys.stdin.buffer.read().split()
	idx = 0
	a = int(data[idx]); idx += 1
	b = int(data[idx]); idx += 1
	c = int(data[idx]); idx += 1
	d = int(data[idx]); idx += 1
	
	# Read a x b matrix as list of rows
	mat1 = [None] * a
	for i in range(a):
		mat1[i] = [int(data[idx + j]) for j in range(b)]
		idx += b
		
	# Read c x d pattern as list of rows
	mat2 = [None] * c
	for i in range(c):
		mat2[i] = [int(data[idx + j]) for j in range(d)]
		idx += d
		
	R = a - c + 1  # valid top-left row count
	C = b - d + 1  # valid top-left col count
	
	if R <= 0 or C <= 0:
		return
	
	M1 = 1000000007
	M2 = 998244353
	BR = 131
	BC = 137
	PRD1 = pow(BR, d, M1)
	PRD2 = pow(BR, d, M2)
	PCD1 = pow(BC, c, M1)
	PCD2 = pow(BC, c, M2)
	
	# Step 1: row-wise rolling hash for every row of mat1
	rh1 = [[0] * C for _ in range(a)]
	rh2 = [[0] * C for _ in range(a)]
	
	for i in range(a):
		row = mat1[i]
		h1 = h2 = 0
		for k in range(d):
			v = row[k]
			h1 = (h1 * BR + v) % M1
			h2 = (h2 * BR + v) % M2
		r1 = rh1[i]; r2 = rh2[i]
		r1[0] = h1; r2[0] = h2
		for j in range(1, C):
			o = row[j - 1]; n = row[j + d - 1]
			h1 = (h1 * BR - o * PRD1 + n) % M1
			h2 = (h2 * BR - o * PRD2 + n) % M2
			r1[j] = h1; r2[j] = h2
			
	# Step 2: target 2D hash for pattern
	t1 = t2 = 0
	for i in range(c):
		row = mat2[i]
		h1 = h2 = 0
		for k in range(d):
			v = row[k]
			h1 = (h1 * BR + v) % M1
			h2 = (h2 * BR + v) % M2
		t1 = (t1 * BC + h1) % M1
		t2 = (t2 * BC + h2) % M2
		
	# Step 3: column-wise rolling hash + match
	ch1 = [0] * C; ch2 = [0] * C
	out = []
	ap = out.append
	
	# Init first c rows
	for i in range(c):
		r1 = rh1[i]; r2 = rh2[i]
		for j in range(C):
			ch1[j] = (ch1[j] * BC + r1[j]) % M1
			ch2[j] = (ch2[j] * BC + r2[j]) % M2
			
	# Check row 0
	for j in range(C):
		if ch1[j] == t1 and ch2[j] == t2:
			ap("1 "); ap(str(j + 1)); ap("\n")
			
	# Roll remaining rows
	for i in range(1, R):
		o1 = rh1[i - 1]; o2 = rh2[i - 1]
		n1 = rh1[i + c - 1]; n2 = rh2[i + c - 1]
		ip = str(i + 1)
		for j in range(C):
			v1 = (ch1[j] * BC - o1[j] * PCD1 + n1[j]) % M1
			v2 = (ch2[j] * BC - o2[j] * PCD2 + n2[j]) % M2
			ch1[j] = v1; ch2[j] = v2
			if v1 == t1 and v2 == t2:
				ap(ip); ap(" "); ap(str(j + 1)); ap("\n")
				
	sys.stdout.write(''.join(out))
	
solve()