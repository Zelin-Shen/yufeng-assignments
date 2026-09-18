"""
Maximal Red Rectangle (Tsinghua OJ).

Grid of '.' (red) and 'X' (green). Find the largest
sub-rectangle containing only '.' cells.

For each row, build histogram heights of consecutive '.'
from that row upward. Apply monotonic stack to get the
largest rectangle in each histogram.

Time Complexity : O(n * m)
Space Complexity: O(m)
"""

import sys
input = sys.stdin.readline

n, m = map(int, input().split())
h = [0] * m
ans = 0

for i in range(n):
    row = input().strip()
    for j in range(m):
        h[j] = h[j] + 1 if row[j] == '.' else 0

    stack = []
    for k in range(m + 1):
        cur = h[k] if k < m else 0
        while stack and h[stack[-1]] > cur:
            height = h[stack.pop()]
            width = k if not stack else k - stack[-1] - 1
            area = height * width
            if area > ans:
                ans = area
        stack.append(k)

print(ans)