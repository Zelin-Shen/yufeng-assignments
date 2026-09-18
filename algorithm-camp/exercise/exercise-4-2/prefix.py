#!/usr/bin/env pypy3
"""
Prefix Query via Trie

Build a trie from n strings. For each node, store a 'pass count' — how many
of the n strings pass through (or terminate at) that node. To answer a query
string q, walk q down the trie; the pass count at the final node is the answer
(i.e. how many original strings have q as a prefix). If walk fails, answer is 0.

Trie stored as flat arrays for speed: children[node][c] and cnt[node].

Complexity: O(total_len) build + O(|q|) per query.
"""

import sys
from bisect import bisect_left

def solve():
	data = sys.stdin.buffer.read().replace(b'\r', b'').rstrip(b'\n').split(b'\n')
	n, m = map(int, data[0].split())
	strs = sorted(data[1:n + 1])
	out = []
	for i in range(m):
		q = data[n + 1 + i]
		lo = bisect_left(strs, q)
		hi = bisect_left(strs, q[:-1] + bytes([q[-1] + 1]))
		out.append(str(hi - lo))
	sys.stdout.write('\n'.join(out) + '\n')
	
solve()