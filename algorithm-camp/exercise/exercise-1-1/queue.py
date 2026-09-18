#!/usr/bin/env pypy3
"""
Queue Simulation: Implement a queue supporting three operations:
	1 X  - Enqueue string X to the back of the queue.
	2    - Dequeue the front element and print it.
	3 Y  - Query the element at position Y (1-indexed from front) and print it.

Uses a list as the queue with a front pointer to avoid O(n) pop(0).
Position 1 is the current front of the queue.
"""

import sys
input = sys.stdin.readline

n = int(input())
queue = []
head = 0
out = []
for _ in range(n):
	line = input().split()
	op = line[0]
	if op == '1':
		queue.append(line[1])
	elif op == '2':
		out.append(queue[head])
		head += 1
	else:
		out.append(queue[head + int(line[1]) - 1])
			
sys.stdout.write('\n'.join(out))