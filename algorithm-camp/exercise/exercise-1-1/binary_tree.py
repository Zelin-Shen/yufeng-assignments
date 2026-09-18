#!/usr/bin/env pypy3
"""
Binary Search Tree (BST) Construction from Permutation:
Given a permutation of 1..n, insert elements sequentially into a BST,
then output the preorder and postorder traversals.

Since tree height <= 50, recursive traversal is safe.
We use arrays for left/right children to avoid object overhead.
"""

import sys
from sys import setrecursionlimit
input = sys.stdin.readline
setrecursionlimit(200000)

n = int(input())
a = list(map(int, input().split()))

left = [0] * (n + 1)
right = [0] * (n + 1)

root = a[0]
for i in range(1, n):
	v = a[i]
	cur = root
	while True:
		if v < cur:
			if left[cur]:
				cur = left[cur]
			else:
				left[cur] = v
				break
		else:
			if right[cur]:
				cur = right[cur]
			else:
				right[cur] = v
				break
			
pre = []
post = []

stack = [root]
while stack:
	node = stack.pop()
	pre.append(node)
	if right[node]:
		stack.append(right[node])
	if left[node]:
		stack.append(left[node])
		
stack1 = [root]
while stack1:
	node = stack1.pop()
	post.append(node)
	if left[node]:
		stack1.append(left[node])
	if right[node]:
		stack1.append(right[node])
post.reverse()

sys.stdout.write(' '.join(map(str, pre)) + '\n')
sys.stdout.write(' '.join(map(str, post)) + '\n')