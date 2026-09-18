#!/usr/bin/env pypy3
"""
Grade Ranking with Inverse Pair Counting:
Given n students with algorithm and data structure training scores:
1. Sort by total score (descending), then by algorithm score (descending)
2. Maintain stable sort for ties
3. Output sorted ranking and count inverse pairs

An inverse pair (i,j) means student i appears before j in original order
but after j in sorted order.

Time: O(n log n) for sorting, O(n^2) for counting inverse pairs (acceptable for n≤5000)
"""

n = int(input())
students = []
for i in range(n):
	algo, ds = map(int, input().split())
	total = algo + ds
	students.append((i + 1, total, algo, ds))
	
sorted_students = sorted(students, key=lambda x: (-x[1], -x[2]))

for student in sorted_students:
	print(student[0], student[1], student[2], student[3])
	
inverse_count = 0
sorted_positions = {student[0]: idx for idx, student in enumerate(sorted_students)}

for i in range(n):
	for j in range(i + 1, n):
		orig_i = i + 1
		orig_j = j + 1
		if sorted_positions[orig_i] > sorted_positions[orig_j]:
			inverse_count += 1
			
print(inverse_count)