#!/usr/bin/env pypy3
"""
Array Partition Problem - Minimize Maximum Subarray Sum

Problem Description:
Given n positive integers in a sequence, divide them into m consecutive parts

Algorithm:
1. Binary search on the answer (maximum subarray sum)
2. For each candidate max_sum, greedily check if we can partition into ≤ m parts
3. Greedy check: iterate through array, add elements to current part while sum ≤ max_sum,
   start new part when adding next element would exceed max_sum
4. Find the minimum max_sum that allows valid partitioning into ≤ m parts

Time Complexity: O(n * log(sum))
Space Complexity: O(n)
"""

def can_partition(arr, m, max_sum):
    """
    Check if we can partition array into at most m consecutive parts
    where each part's sum is at most max_sum.
    
    Args:
        arr: input array
        m: maximum number of parts allowed
        max_sum: maximum sum allowed for each part
    
    Returns:
        True if valid partitioning exists, False otherwise
    """
    parts_used = 1  # Start with one part
    current_sum = 0
    
    for num in arr:
        # If single number exceeds max_sum, impossible
        if num > max_sum:
            return False
        
        # Try to add current number to current part
        if current_sum + num <= max_sum:
            current_sum += num
        else:
            # Need a new part
            parts_used += 1
            current_sum = num
            
            # If we need more than m parts, fail
            if parts_used > m:
                return False
    
    return True

# Read n (array length) and m (number of parts)
n, m = map(int, input().split())

# Read n positive integers
arr = list(map(int, input().split()))

# Binary search on the answer (maximum part sum)
# Lower bound: maximum single element (cannot be smaller)
# Upper bound: sum of all elements (all in one part)
left = max(arr)
right = sum(arr)

# Find minimum max_sum that allows valid partitioning
result = right
while left <= right:
    mid = (left + right) // 2
    
    # Check if we can partition with max_sum = mid
    if can_partition(arr, m, mid):
        result = mid  # Valid, try smaller
        right = mid - 1
    else:
        left = mid + 1  # Invalid, need larger max_sum

# Output the minimum maximum part sum
print(result)