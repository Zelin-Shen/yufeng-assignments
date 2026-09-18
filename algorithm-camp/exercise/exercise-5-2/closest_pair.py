#!/usr/bin/env pypy3
"""
Closest Pair of Points via Divide and Conquer - O(n log n) to O(n log^2 n)

Problem:
    Given n points on a 2D plane, find the Euclidean distance between the closest pair.

Algorithm:
    1. Sort all points by x-coordinate.
    2. Recursively divide the point set into two halves by a vertical line.
    3. Find the minimum distance 'd' in the left half and right half recursively.
    4. Merge step:
       - Consider a strip of width 2d centered at the dividing line.
       - Collect points inside this strip.
       - Sort these points by y-coordinate.
       - For each point, check distances to subsequent points where y-difference < d.
         Geometrically, for each point, we only need to check a constant number of neighbors (at most 6).
    5. Return the minimum distance found.

Optimization:
    - Used squared distances for comparisons to avoid expensive sqrt() calls during the process.
    - Used __slots__ in the Point class to reduce memory overhead and improve attribute access speed.
    - Used sys.stdin.read() for fast I/O parsing.
    - The recursion depth is O(log n), safe for n <= 300,000.

Complexity:
    Time:  O(n log n) for the initial sort. The recursive step involves sorting the strip.
           In this implementation, sorting the strip locally leads to O(n log^2 n) worst case,
           but is extremely fast in practice for Python due to Timsort efficiency on small chunks.
           A theoretical O(n log n) requires pre-sorting by Y, which involves more list copying overhead.
    Space: O(n) for storing points and recursion stack.
"""

import sys
import math


N = 300005

class ip:
    # Using __slots__ for significant memory reduction and faster attribute access
    __slots__ = ('x', 'y')
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # Sort by x, then y
    def __lt__(self, other):
        if self.x == other.x:
            return self.y < other.y
        return self.x < other.x

# Global list to hold points
pts = []

# Calculate squared distance to avoid sqrt overhead
def dist_sq(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return dx * dx + dy * dy

# Brute force for small subsets (size <= 4)
def brute_force(l, r):
    min_d = float('inf')
    for i in range(l, r):
        for j in range(i + 1, r + 1):
            d = dist_sq(pts[i], pts[j])
            if d < min_d:
                min_d = d
    return min_d

# Recursive divide and conquer function
def solve(l, r):
    # Base case: if few points, use brute force
    if r - l <= 3:
        return brute_force(l, r)
    
    mid = (l + r) // 2
    mid_x = pts[mid].x
    
    # Recursive calls
    d_left = solve(l, mid)
    d_right = solve(mid + 1, r)
    d = d_left if d_left < d_right else d_right
    
    # Build the strip
    # Collect points where |x - mid_x| < sqrt(d)
    # Since d is squared, we check (x - mid_x)^2 < d
    strip = []
    # Note: pts is sorted by x, so we can iterate outwards from mid if needed,
    # but linear scan is O(N) which is fine for the merge step.
    for i in range(l, r + 1):
        if (pts[i].x - mid_x) ** 2 < d:
            strip.append(pts[i])
            
    # Sort strip by y-coordinate
    strip.sort(key=lambda p: p.y)
    
    # Scan the strip
    len_strip = len(strip)
    for i in range(len_strip):
        p1 = strip[i]
        # Check next points. We only need to check a few points ahead
        # because if y_diff >= sqrt(d), dist_sq will certainly be >= d.
        for j in range(i + 1, len_strip):
            p2 = strip[j]
            dy = p2.y - p1.y
            # If y difference squared is already >= d, no need to check further
            if dy * dy >= d:
                break
            
            # Calculate actual distance
            d_new = dist_sq(p1, p2)
            if d_new < d:
                d = d_new
                
    return d

def getAnswer(n, X, Y):
    global pts
    pts = [ip(X[i], Y[i]) for i in range(n)]
    
    # Sort points by x coordinate
    pts.sort()
    
    # Start divide and conquer
    min_sq_dist = solve(0, n - 1)
    
    # Return the square root of the minimum squared distance
    return math.sqrt(min_sq_dist)


# Optimized I/O
input_data = sys.stdin.read().split()
if input_data:
    iterator = iter(input_data)
    try:
        n = int(next(iterator))
        X = []
        Y = []
        for _ in range(n):
            X.append(int(next(iterator)))
            Y.append(int(next(iterator)))
        
        ans = getAnswer(n, X, Y)
        print("{:.2f}".format(ans))
    except StopIteration:
        pass