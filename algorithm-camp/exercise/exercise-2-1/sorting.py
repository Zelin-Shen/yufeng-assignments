#!/usr/bin/env pypy3
"""
Sorting Problem Solution

Problem Description:
Given n integers, sort them in ascending order (from smallest to largest).

Algorithm:
- Use Python's built-in sorted() function which implements Timsort
- Timsort has O(n log n) time complexity in worst case
- Space complexity: O(n) for storing the sorted array

Input Format:
- Line 1: A positive integer n (number of integers)
- Line 2: n space-separated integers

Output Format:
- One line containing n space-separated integers in ascending order

Time Complexity: O(n log n)
Space Complexity: O(n)

Constraints:
- n ≤ 500000
- Absolute value of integers ≤ 10^9
- Time limit: 2 seconds
- Memory limit: 256 MB
"""

n = int(input())

numbers = list(map(int, input().split()))

sorted_numbers = sorted(numbers)

print(' '.join(map(str, sorted_numbers)))