#!/usr/bin/env pypy3
"""
Sequence Counting Problem - Optimized Divide and Conquer

Problem Description:
Count contiguous subsequences where max - min ≤ d.

Algorithm:
Divide and conquer with optimized cal function using two pointers.
Maintain running min/max while extending ranges.

Time Complexity: O(n log n)
Space Complexity: O(n)
"""

def solve(arr, l, r, d):
	if l == r:
		return 0
	
	mid = (l + r) // 2
	count = solve(arr, l, mid, d) + solve(arr, mid + 1, r, d)
	count += cal(arr, l, r, mid, d)
	
	return count

def cal(arr, l, r, mid, d):
	"""
	Optimized crossing count with preprocessing.
	"""
	count = 0
	
	# Precompute min/max from each position to mid (going right)
	left_min = {}
	left_max = {}
	
	for i in range(mid, l - 1, -1):
		if i == mid:
			left_min[i] = arr[i]
			left_max[i] = arr[i]
		else:
			left_min[i] = min(arr[i], left_min[i + 1])
			left_max[i] = max(arr[i], left_max[i + 1])
			
	# Precompute min/max from mid+1 to each position (going right)
	right_min = {}
	right_max = {}
	
	for j in range(mid + 1, r + 1):
		if j == mid + 1:
			right_min[j] = arr[j]
			right_max[j] = arr[j]
		else:
			right_min[j] = min(arr[j], right_min[j - 1])
			right_max[j] = max(arr[j], right_max[j - 1])
			
	# Two pointers
	j = r + 1  # Right boundary (exclusive)
	
	for i in range(mid, l - 1, -1):
		# Move j left until [i, j-1] is valid
		while j > mid + 1:
			# Get min/max for [i, mid] and [mid+1, j-1]
			total_min = min(left_min[i], right_min[j - 1])
			total_max = max(left_max[i], right_max[j - 1])
			
			if total_max - total_min <= d:
				break
			j -= 1
			
		# Count valid endpoints from mid+1 to j-1
		count += max(0, j - mid - 1)
		
	return count

n, d = map(int, input().split())
arr = list(map(int, input().split()))
print(solve(arr, 0, n - 1, d))