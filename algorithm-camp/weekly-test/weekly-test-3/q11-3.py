#!/usr/bin/env pypy3
"""
Game Boss Fight with Energy Management and 0-1 Knapsack DP.

Given initial energy s (decreases by 1 per time unit), n timed items
(each usable for energy +e_i or health +h_i), find the minimum time
to accumulate health >= h (clear). If impossible, find max survival.

dp[j] = max energy when accumulated health = j (-1 = unreachable).
Items sorted by time; 0-1 knapsack per item with overflow at j = h.
Survival: greedily use every item for energy to maximize lifetime.

Time:  O(n * h) with max-health pruning
Space: O(h)
"""

import sys


def solve():
    data = sys.stdin.buffer.read().split()
    ptr = 0
    h = int(data[ptr])
    ptr += 1
    n = int(data[ptr])
    ptr += 1
    s = int(data[ptr])
    ptr += 1

    items = []
    for _ in range(n):
        t = int(data[ptr])
        ptr += 1
        e = int(data[ptr])
        ptr += 1
        hi = int(data[ptr])
        ptr += 1
        items.append((t, e, hi))

    items.sort()

    # dp[j]: max energy at health j; -1 means unreachable
    dp = [-1] * (h + 1)
    dp[0] = s
    prev_t = 0
    mxh = 0  # highest reachable health index so far

    for t_k, e_k, h_k in items:
        delta = t_k - prev_t
        prev_t = t_k

        # --- time elapses: energy drops, kill dead states ---
        if delta > 0:
            for j in range(mxh + 1):
                v = dp[j]
                if v >= 0:
                    v -= delta
                    dp[j] = v if v >= 0 else -1

        hk = min(h_k, h)

        if hk == 0:
            # no health value: energy option only
            for j in range(mxh + 1):
                if dp[j] >= 0:
                    dp[j] += e_k
        else:
            new_mxh = min(h, mxh + hk)

            # best reachable energy that can overflow into dp[h]
            # any j' with j' + hk >= h maps to h
            best = -1
            if new_mxh >= h:
                lo = h - hk
                if lo < 0:
                    lo = 0
                hi_b = min(mxh, h)
                for jj in range(lo, hi_b + 1):
                    if dp[jj] > best:
                        best = dp[jj]

            # 0-1 knapsack, high-to-low to prevent item reuse
            for j in range(new_mxh, hk - 1, -1):
                # energy option: keep health, gain energy
                ve = dp[j]
                if ve >= 0:
                    ve += e_k
                # health option: gain health from j-hk, energy unchanged
                vh = dp[j - hk]
                dp[j] = ve if ve >= vh else vh

            # overflow update for j = h
            if best > dp[h]:
                dp[h] = best

            # states below hk: only energy option possible
            top = min(hk - 1, mxh)
            for j in range(top, -1, -1):
                if dp[j] >= 0:
                    dp[j] += e_k

            mxh = new_mxh

        # check clear condition
        if dp[h] >= 0:
            sys.stdout.write(str(t_k) + "\n")
            return

    # --- cannot clear: max survival (all items used for energy) ---
    energy = s
    prev_t = 0
    survive = s
    for t_k, e_k, h_k in items:
        delta = t_k - prev_t
        if energy < delta:
            survive = prev_t + energy
            break
        energy -= delta
        energy += e_k
        prev_t = t_k
    else:
        survive = prev_t + energy

    sys.stdout.write("-1\n" + str(survive) + "\n")


solve()
