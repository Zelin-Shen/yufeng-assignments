#!/usr/bin/env pypy3
"""
2xn Grid Rotation Puzzle

Problem Description:
    A 2xn grid contains numbers 1..2n exactly once. An operation selects a
    2x2 block (columns i, i+1) and rotates it 180 degrees.
    Given initial and target states, find the minimum number of operations
    to transform initial into target, or -1 if impossible.

Algorithm:
    Each 180-degree rotation swaps two adjacent columns AND reverses both.
    The minimum operations equals inversions in the column permutation,
    subject to parity constraint: source k at target j requires
    k + flip(k,j) = j (mod 2). Greedy matching + BIT for inversions.

Complexity:
    O(n log n) time, O(n) space.
"""

import sys
from collections import defaultdict


def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    ptr = 0
    n = int(data[ptr])
    ptr += 1

    init0 = [0] * n
    init1 = [0] * n
    for i in range(n):
        init0[i] = int(data[ptr])
        ptr += 1
    for i in range(n):
        init1[i] = int(data[ptr])
        ptr += 1
    tgt0 = [0] * n
    tgt1 = [0] * n
    for i in range(n):
        tgt0[i] = int(data[ptr])
        ptr += 1
    for i in range(n):
        tgt1[i] = int(data[ptr])
        ptr += 1

    if n == 1:
        sys.stdout.write("-1" if init0[0] != tgt0[0] or init1[0] != tgt1[0] else "0")
        return

    # 4 bins per group: orientation(0/1) * 2 + 1-indexed parity(0/1)
    src_lists = defaultdict(lambda: [[], [], [], []])
    for j in range(n):
        a, b = init0[j], init1[j]
        key = (min(a, b), max(a, b))
        ori = 1 if a > b else 0
        src_lists[key][ori * 2 + ((j + 1) & 1)].append(j + 1)

    src_ptrs = defaultdict(lambda: [0, 0, 0, 0])
    perm = [0] * (n + 1)

    for j in range(1, n + 1):
        c, d = tgt0[j - 1], tgt1[j - 1]
        key = (min(c, d), max(c, d))
        if key not in src_lists:
            sys.stdout.write("-1")
            return

        tgt_ori = 1 if c > d else 0
        jpar = j & 1
        same_bin = tgt_ori * 2 + jpar
        diff_bin = (1 - tgt_ori) * 2 + (1 - jpar)

        INF = 0x7FFFFFFF
        sp = src_ptrs[key][same_bin]
        same_val = (
            src_lists[key][same_bin][sp] if sp < len(src_lists[key][same_bin]) else INF
        )
        dp = src_ptrs[key][diff_bin]
        diff_val = (
            src_lists[key][diff_bin][dp] if dp < len(src_lists[key][diff_bin]) else INF
        )

        if same_val == INF and diff_val == INF:
            sys.stdout.write("-1")
            return

        if same_val <= diff_val:
            perm[j] = same_val
            src_ptrs[key][same_bin] = sp + 1
        else:
            perm[j] = diff_val
            src_ptrs[key][diff_bin] = dp + 1

    # BIT inversion count
    tree = [0] * (n + 2)
    inv_count = 0
    for j in range(1, n + 1):
        k = perm[j]
        i = k
        prefix = 0
        while i > 0:
            prefix += tree[i]
            i -= i & (-i)
        i = n
        total = 0
        while i > 0:
            total += tree[i]
            i -= i & (-i)
        inv_count += total - prefix
        # update
        i = k
        while i <= n:
            tree[i] += 1
            i += i & (-i)

    sys.stdout.write(str(inv_count))


solve()
