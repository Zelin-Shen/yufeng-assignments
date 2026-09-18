#!/usr/bin/env pypy3
"""
Stack Simulation: Implement a stack supporting three operations:
	1 X  - Push string X onto the stack.
	2    - Pop the top element and print it.
	3 Y  - Query the element at position Y (1-indexed from bottom) and print it.

Uses a simple list as the stack. Position 1 is the bottom (index 0).
"""

import sys
input = sys.stdin.readline

n = int(input())
stack = []
out = []
for _ in range(n):
	line = input().split()
	op = line[0]
	if op == '1':
		stack.append(line[1])
	elif op == '2':
		out.append(stack.pop())
	else:
		out.append(stack[int(line[1]) - 1])
			
sys.stdout.write('\n'.join(out))