#!/usr/bin/env pypy3
"""
Huffman Coding - Minimum Weighted Path Length:
Given n words with their occurrence frequencies, construct an optimal binary 
encoding (Huffman tree) that minimizes the total encoding length.

The problem asks for the minimum sum of (frequency[i] * code_length[i]) for all words.

Algorithm:
1. Use a min-heap to repeatedly merge the two smallest frequency nodes
2. Each merge creates a parent node with combined frequency
3. The total cost is the sum of all internal node frequencies
4. This greedy approach guarantees the optimal Huffman encoding

Time: O(n log n), Space: O(n)
"""

import heapq

n = int(input())
freq = []
for _ in range(n):
	freq.append(int(input()))
	
heap = freq[:]
heapq.heapify(heap)

total_cost = 0

while len(heap) > 1:
	f1 = heapq.heappop(heap)
	f2 = heapq.heappop(heap)
	merged = f1 + f2
	total_cost += merged
	heapq.heappush(heap, merged)
	
print(total_cost)