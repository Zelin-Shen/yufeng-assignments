#!/usr/bin/env pypy3
"""
Unique Topological Sort Check via Kahn's Algorithm

Problem:
    Determine if a Directed Acyclic Graph (DAG) has a unique topological ordering.
    A "legal" sequence is one where no edge points from a later element to an earlier one.
    This corresponds to a topological sort. We need to check if exactly one such sort exists.

Algorithm:
    1. Calculate in-degrees for all vertices.
    2. Initialize a queue with all vertices having in-degree 0.
    3. Iterate while the queue is not empty:
       - If the queue size is greater than 1, multiple valid candidates exist for
         the next position in the sequence. Thus, the order is not unique. Return 0.
       - Pop a vertex u, increment processed count.
       - For each neighbor v of u, decrement in-degree[v]. If in-degree[v] becomes 0,
         push v into the queue.
    4. If the number of processed vertices equals n, the unique order exists. Return 1.

Optimization:
    - Replaced standard input() with sys.stdin.read() for O(1) I/O overhead.
    - Used collections.deque for efficient queue operations.
    - Local variable binding inside the function to reduce global lookup overhead.

Complexity:
    Time:  O(n + m) per test case.
    Space: O(n + m) for graph storage.
"""

import sys
from collections import deque


N = 10005

# Global variables for graph data
# n, m: vertex count and edge count
# In: In-degree array
# e: Adjacency list
n, m = 0, 0
In = [0 for i in range(N)]
e = [[] for i in range(N)]

def getAnswer():
    '''Determines if the graph has a unique topological order'''
    # Bind global variables to local for faster access
    in_deg = In
    adj = e
    lim = n + 1
    
    q = deque()
    
    # Initialize queue with 0 in-degree nodes
    for i in range(1, lim):
        if in_deg[i] == 0:
            q.append(i)
            
    cnt = 0
    while q:
        # If queue size > 1, we have multiple choices -> not unique
        if len(q) > 1:
            return 0
        
        u = q.popleft()
        cnt += 1
        
        # Process neighbors
        for v in adj[u]:
            in_deg[v] -= 1
            if in_deg[v] == 0:
                q.append(v)
                
    return 1 if cnt == n else 0


# High-efficiency I/O implementation
def solve():
    # Read all input from stdin at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    iterator = iter(input_data)
    
    try:
        # Number of test cases
        T = int(next(iterator))
        res = []
        
        for _ in range(T):
            # Update global n, m
            global n, m
            n = int(next(iterator))
            m = int(next(iterator))
            
            # Reset graph data for relevant range
            for i in range(1, n + 1):
                In[i] = 0
                e[i].clear()
            
            # Read edges
            for _ in range(m):
                x = int(next(iterator))
                y = int(next(iterator))
                e[x].append(y)
                In[y] += 1
            
            # Append result
            res.append(str(getAnswer()))
            
        # Output all results
        sys.stdout.write('\n'.join(res) + '\n')
        
    except StopIteration:
        pass

solve()