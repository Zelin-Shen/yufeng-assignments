#!/usr/bin/env pypy3
"""
Distinct Subsequences Count (mod 23333)

Let f(i) = number of distinct non-empty subsequences in s[0..i-1].
Let pre(i) = last position where s[i] appeared before (0 if first time).

Recurrence (1-indexed):
	f(i) = 2*f(i-1) + 1,                if pre(i) == 0
	f(i) = 2*f(i-1) - f(pre(i) - 1),    if pre(i) != 0

Answer = f(n) mod 23333.

Complexity: O(n) time, O(1) extra space (only track last[c] for 26 chars).
"""

import sys

def solve():
	s = sys.stdin.buffer.readline().strip()
	n = len(s)
	MOD = 23333
	last = [-1] * 256          # last f-value before previous occurrence of char c
	f = 0                      # f(0) = 0
	for i in range(n):
		c = s[i]
		if last[c] == -1:
			# first occurrence: f_new = 2*f + 1
			nf = (2 * f + 1) % MOD
		else:
			# repeated: f_new = 2*f - f(pre-1), where last[c] stored that f(pre-1)
			nf = (2 * f - last[c]) % MOD
		last[c] = f            # save current f as f(i-1) for future duplicates
		f = nf
	sys.stdout.write(str(f % MOD) + '\n')
	
solve()