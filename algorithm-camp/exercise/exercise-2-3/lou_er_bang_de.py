#!/usr/bin/env pypy3
"""
Lower Bound Problem - Binary Search Solution

Problem Description:
Given a sequence A of n integers and Q queries.
For each query with value x, find how many numbers in A are >= x and output the smallest such number.
If no such number exists, output -1.

Algorithm:
1. Sort the sequence A
2. For each query x, use binary search to find the first element >= x
3. Count how many elements are >= x

Time Complexity: O(n log n + Q log n)
Space Complexity: O(n)
"""

"""
Lower Bound - Minimal Overhead Version
"""

from bisect import bisect_left
import sys

n = int(sys.stdin.readline())
A = sorted(map(int, sys.stdin.readline().split()))
Q = int(sys.stdin.readline())

for _ in range(Q):
	x = int(sys.stdin.readline())
	pos = bisect_left(A, x)
	print(A[pos] if pos < n else -1)