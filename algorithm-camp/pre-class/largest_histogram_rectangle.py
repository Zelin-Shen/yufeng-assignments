"""
Largest Rectangle in Histogram using Monotonic Stack.

Maintain a stack of indices with strictly increasing heights.
When a shorter bar appears, pop and compute the area where
the popped bar is the limiting height.
Append a sentinel height 0 to flush all remaining bars.

Time Complexity : O(n)
Space Complexity: O(n)
"""

import sys
input = sys.stdin.readline

n = int(input())
h = list(map(int, input().split()))

stack = []
ans = 0

for i in range(n + 1):
    cur = h[i] if i < n else 0
    while stack and h[stack[-1]] > cur:
        height = h[stack.pop()]
        width = i if not stack else i - stack[-1] - 1
        area = height * width
        if area > ans:
            ans = area
    stack.append(i)

print(ans)