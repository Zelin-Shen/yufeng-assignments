#!/usr/bin/env pypy3
"""
Count All Palindromic Substrings via Manacher's Algorithm - O(n)

Problem:
    Given a string s of lowercase letters, count the number of substrings
    that are palindromes. A substring is a contiguous segment of s.

Algorithm:
    1. Build a transformed array t by inserting sentinel (0) between each
       character: t = [0, s[0], 0, s[1], 0, ..., s[n-1], 0].
       Length of t is m = 2*n + 1.

    2. Run Manacher's algorithm on t to compute p[i] for each position i,
       where p[i] is the maximum palindrome radius centered at i in t.
       p[i] also equals the length of the longest palindrome in the
       original string centered at that position.

    3. For each center i in t, the number of distinct palindromic substrings
       centered at i is ceil(p[i] / 2) = (p[i] + 1) // 2.
       - Odd positions (original chars): palindromes of length 1, 3, ..., p[i]
       - Even positions (separators):    palindromes of length 2, 4, ..., p[i]

    4. Answer = sum of (p[i] + 1) // 2 over all i in [0, m).

Complexity:
    Time:  O(n)  — Manacher is linear, summation is linear.
    Space: O(n)  — transformed array and radius array.
"""

import sys

def solve():
    s = sys.stdin.buffer.readline().strip()
    n = len(s)
    if n == 0:
        sys.stdout.write("0\n")
        return

    # Build transformed bytearray: [0, s[0], 0, s[1], 0, ..., s[n-1], 0]
    # Using 0 as separator — guaranteed no collision with lowercase a-z
    m = 2 * n + 1
    t = bytearray(m)
    for i in range(n):
        t[2 * i + 1] = s[i]
    # Even indices remain 0 (separator), odd indices hold original bytes

    # Manacher's algorithm
    p = [0] * m
    c = 0  # center of the rightmost palindrome
    r = 0  # right boundary of the rightmost palindrome

    for i in range(m):
        # Mirror index
        if i < r:
            p[i] = min(r - i, p[2 * c - i])
        # Attempt to expand palindrome centered at i
        lo = i - p[i] - 1
        hi = i + p[i] + 1
        while lo >= 0 and hi < m and t[lo] == t[hi]:
            p[i] += 1
            lo -= 1
            hi += 1
        # Update rightmost palindrome if needed
        if i + p[i] > r:
            c = i
            r = i + p[i]

    # Sum up palindrome counts from each center
    ans = 0
    for i in range(m):
        ans += (p[i] + 1) >> 1

    sys.stdout.write(str(ans) + "\n")

solve()