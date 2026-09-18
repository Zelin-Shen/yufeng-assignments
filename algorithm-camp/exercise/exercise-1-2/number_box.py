#!/usr/bin/env pypy3
"""
Number Box:
Maintain a set (box). Support two operations:
	op=1 (insert): if x is NOT in the box, insert it -> Succeeded; else Failed.
	op=2 (delete): if x IS in the box, remove it -> Succeeded; else Failed.
x can be up to 10^18, Q up to 5*10^5. A simple Python set handles this.
"""

import sys
input = sys.stdin.readline

q = int(input())
box = set()
out = []
for _ in range(q):
	op, x = map(int, input().split())
	if op == 1:
		if x not in box:
			box.add(x)
			out.append("Succeeded")
		else:
			out.append("Failed")
	else:
		if x in box:
			box.remove(x)
			out.append("Succeeded")
		else:
			out.append("Failed")

sys.stdout.write('\n'.join(out))