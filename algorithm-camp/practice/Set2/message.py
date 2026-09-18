"""
Message Passing (NOIP 2008) - Maximum Total Favorability on Two Non-overlapping Paths

We have an m x n grid with non-negative values.
Two students are at (1,1) and (m,n), value at these two cells is 0.
We need:
1) one path from (1,1) to (m,n), moving only Right/Down
2) one path from (m,n) back to (1,1), moving only Left/Up

Equivalent transformation:
Both paths can be viewed as two simultaneous walkers moving from (1,1) to (m,n),
each step being Right/Down, with the same number of steps at each "time layer".
A cell's value can be collected at most once in total (except endpoints are 0 anyway).

DP state:
Let k be the total step layer (k = i + j), with 2 <= k <= m + n.
If walker A is at (i1, j1) and walker B at (i2, j2), then:
    j1 = k - i1
    j2 = k - i2
Define:
    dp[i1][i2] = maximum collected value up to layer k
Transition from previous layer k-1 with four move combinations:
    (down,down), (down,right), (right,down), (right,right)
which correspond to previous rows:
    (i1, i2), (i1-1, i2), (i1, i2-1), (i1-1, i2-1)

Value added at current layer:
    grid[i1][j1] + grid[i2][j2], but if (i1, j1) == (i2, j2), add only once.

Complexities:
    Time:  O((m+n) * m * m)  <= about 5e5 states/transitions for m,n<=50
    Space: O(m * m), rolling by layer k
"""

import sys

def solve():
    input = sys.stdin.readline
    m, n = map(int, input().split())
    a = [[0] * (n + 1)]
    for _ in range(m):
        a.append([0] + list(map(int, input().split())))

    NEG = -10**18

    # dp for current layer k, indexed by i1, i2 (1..m)
    prev = [[NEG] * (m + 1) for _ in range(m + 1)]
    prev[1][1] = 0  # at k=2, both at (1,1), value is 0 by statement

    for k in range(3, m + n + 1):
        cur = [[NEG] * (m + 1) for _ in range(m + 1)]

        # Valid row range so that column j = k - i is in [1, n]
        r_lo = max(1, k - n)
        r_hi = min(m, k - 1)

        for i1 in range(r_lo, r_hi + 1):
            j1 = k - i1
            for i2 in range(r_lo, r_hi + 1):
                j2 = k - i2

                best_prev = prev[i1][i2]  # both moved right
                if i1 > 1:
                    if prev[i1 - 1][i2] > best_prev:
                        best_prev = prev[i1 - 1][i2]
                if i2 > 1:
                    if prev[i1][i2 - 1] > best_prev:
                        best_prev = prev[i1][i2 - 1]
                if i1 > 1 and i2 > 1:
                    if prev[i1 - 1][i2 - 1] > best_prev:
                        best_prev = prev[i1 - 1][i2 - 1]

                if best_prev == NEG:
                    continue

                gain = a[i1][j1]
                if i1 != i2 or j1 != j2:
                    gain += a[i2][j2]

                cur[i1][i2] = best_prev + gain

        prev = cur

    print(prev[m][m])

solve()