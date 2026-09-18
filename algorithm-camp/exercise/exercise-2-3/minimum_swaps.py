#!/usr/bin/env pypy3
"""
Minimum Swaps Problem - Inversion Count using Merge Sort

Problem: Find minimum adjacent swaps to sort an array.
Solution: The answer equals the number of inversions.

An inversion is a pair (i, j) where i < j but arr[i] > arr[j].
Each adjacent swap reduces inversions by exactly 1.

Algorithm: Modified merge sort to count inversions
Time Complexity: O(n log n)
Space Complexity: O(n)
"""

def merge_sort_count(arr, temp, left, right):
	"""
	Merge sort that counts inversions.
	
	Args:
		arr: array to sort
		temp: temporary array for merging
		left, right: range boundaries
		
	Returns:
		Number of inversions in range [left, right]
	"""
	inv_count = 0
	
	if left < right:
		mid = (left + right) // 2
		
		# Count inversions in left half
		inv_count += merge_sort_count(arr, temp, left, mid)
		
		# Count inversions in right half
		inv_count += merge_sort_count(arr, temp, mid + 1, right)
		
		# Count inversions across mid and merge
		inv_count += merge(arr, temp, left, mid, right)
		
	return inv_count

def merge(arr, temp, left, mid, right):
	"""
	Merge two sorted halves and count inversions.
	
	When element from right half is smaller than element from left half,
	it forms inversions with all remaining elements in left half.
	"""
	i = left      # Index for left subarray
	j = mid + 1   # Index for right subarray
	k = left      # Index for merged array
	inv_count = 0
	
	# Merge and count inversions
	while i <= mid and j <= right:
		if arr[i] <= arr[j]:
			temp[k] = arr[i]
			i += 1
		else:
			# arr[i] > arr[j], so all elements from i to mid form inversions with arr[j]
			temp[k] = arr[j]
			inv_count += (mid - i + 1)
			j += 1
		k += 1
		
	# Copy remaining elements from left half
	while i <= mid:
		temp[k] = arr[i]
		i += 1
		k += 1
		
	# Copy remaining elements from right half
	while j <= right:
		temp[k] = arr[j]
		j += 1
		k += 1
		
	# Copy merged array back to original
	for i in range(left, right + 1):
		arr[i] = temp[i]
		
	return inv_count

def count_inversions(arr):
	"""
	Count inversions in array using merge sort.
	"""
	n = len(arr)
	temp = [0] * n
	return merge_sort_count(arr, temp, 0, n - 1)

# Read input
n = int(input())
arr = list(map(int, input().split()))

# Count inversions (minimum swaps needed)
result = count_inversions(arr)

# Output result
print(result)