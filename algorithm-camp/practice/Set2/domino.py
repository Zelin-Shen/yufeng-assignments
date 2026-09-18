"""
Domino (Black/White) - Count of "Unpleasant" Arrangements

We have n dominoes, each can be Black or White.
An arrangement is called unpleasant if it contains at least one run of
three consecutive dominoes with the same color ("BBB" or "WWW").

Task:
Given n (1 <= n <= 10000), compute the number of unpleasant arrangements.

Key idea:
Total number of arrangements is 2^n.
So:
    unpleasant = total - pleasant
where "pleasant" means no three consecutive equal colors.

DP for pleasant arrangements:
Let:
    a[i] = number of pleasant binary strings of length i ending with run length 1
    b[i] = number of pleasant binary strings of length i ending with run length 2
Then:
    a[i] = a[i-1] + b[i-1]   (switch color, run resets to 1)
    b[i] = a[i-1]            (same color as previous, run goes 1 -> 2)
Thus:
    pleasant[i] = a[i] + b[i]
and it satisfies:
    pleasant[i] = pleasant[i-1] + pleasant[i-2], for i >= 3
with:
    pleasant[1] = 2
    pleasant[2] = 4

Because n can be 10000, the result is huge, but Python integers are arbitrary precision,
so no manual big-integer implementation is needed.

Complexity:
    Time:  O(n)
    Space: O(1) extra (excluding big integer storage)
"""

import sys

def solve():
    data = sys.stdin.readline().strip()
    n = int(data)

    # Handle very small lengths directly.
    if n == 1:
        print(0)  # all 2 strings are pleasant, so unpleasant = 0
        return
    if n == 2:
        print(0)  # still impossible to have 3 equal consecutive dominoes
        return

    # pleasant(1)=2, pleasant(2)=4
    p1, p2 = 2, 4
    for _ in range(3, n + 1):
        p1, p2 = p2, p1 + p2  # Fibonacci-like transition

    pleasant = p2
    total = 1 << n  # 2^n
    unpleasant = total - pleasant
    print(unpleasant)

solve()