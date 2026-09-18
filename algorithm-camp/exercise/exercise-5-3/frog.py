#!/usr/bin/env pypy3
"""
Frog Jump - Dynamic Programming with Monotonic Queue Optimization

Problem:
    A frog jumps on n lotus leaves. Each jump must have distance >= previous distance
    and maintain direction. Find the maximum sum of values collected.

Algorithm:
    The problem can be decomposed into two independent directions: Left-to-Right and Right-to-Left.
    We solve for one direction and apply the same logic to the reversed list.
    
    For Left-to-Right (sorted by coordinate ascending):
    Let dp[i][j] be the maximum value obtained for a path ending with a jump from i to j (i < j).
    Transition: dp[i][j] = s[j] + max(s[i], max_{k < i, x[i]-x[k] <= x[j]-x[i]} dp[k][i]).
    
    Optimization:
    For a fixed 'i', we iterate 'j' from i+1 to n-1.
    The distance dist = x[j] - x[i] is increasing.
    The condition for 'k' is x[i] - x[k] <= dist, which is equivalent to x[k] >= x[i] - dist.
    As 'j' increases, 'dist' increases, so 'x[i] - dist' decreases.
    This means the lower bound for valid 'k' coordinates decreases, expanding the range of valid 'k' indices.
    Since we iterate 'j' outwards, we can maintain a pointer 'ptr' that scans leftwards from 'i-1'.
    This reduces the inner transition cost from O(N) to O(1) amortized.
    
    Total Complexity: O(N^2) for each direction.
    Space Complexity: O(N^2) for the DP table.

Data Structures:
    - points: List of (x, s) sorted by coordinate.
    - dp: A list of lists (or dictionary) storing dp[i][j].
"""

import sys

def calculate_max_score(points, n):
    """
    Calculates max score for moving in increasing x direction.
    points: list of (x, score) sorted by x.
    """
    
    dp = [[0] * n for _ in range(n)]
    max_ans = 0
    
    for i in range(n):
        current_max_dp = points[i][1]
        ptr = i - 1
        for j in range(i + 1, n):
            dist = points[j][0] - points[i][0]
            limit_coord = points[i][0] - dist
            while ptr >= 0 and points[ptr][0] >= limit_coord:
                val = dp[ptr][i]
                if val > current_max_dp:
                    current_max_dp = val
                ptr -= 1
            dp[i][j] = current_max_dp + points[j][1]
            if dp[i][j] > max_ans:
                max_ans = dp[i][j]
                
    return max_ans

def solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    iterator = iter(input_data)
    try:
        n = int(next(iterator))
    except StopIteration:
        return

    points = []
    for _ in range(n):
        x = int(next(iterator))
        s = int(next(iterator))
        points.append((x, s))
    
    # Sort by coordinate
    points.sort()
    
    # Direction 1: Left to Right
    ans1 = calculate_max_score(points, n)
    
    # Direction 2: Right to Left
    points_rev = [(-x, s) for x, s in points]
    points_rev.sort()
    
    ans2 = calculate_max_score(points_rev, n)
    
    print(max(ans1, ans2))

solve()