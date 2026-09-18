#!/usr/bin/env pypy3
"""
Chess Problem - Maximum Bipartite Matching

Problem Description:
Given an n×n chessboard where:
- 1 means the position CAN place a rook
- 0 means the position CANNOT place a rook
Place maximum number of rooks such that no two rooks attack each other.

Key Insight:
- Treat rows as X-set and columns as Y-set in bipartite graph
- Each valid position (i,j) where board[i][j]=1 represents an edge
- Find maximum matching in bipartite graph

Algorithm:
Hungarian algorithm (DFS-based augmenting path method)

Time Complexity: O(n^3)
Space Complexity: O(n^2)
"""

def dfs(row, graph, match_col, visited):
	"""
	Find augmenting path for given row using DFS.
	
	Args:
		row: current row trying to match
		graph: graph[row] = list of valid columns
		match_col: match_col[col] = matched row (-1 if unmatched)
		visited: visited columns in current DFS
		
	Returns:
		True if augmenting path found
	"""
	for col in graph[row]:
		if visited[col]:
			continue
		visited[col] = True
		
		# If column unmatched or can find augmenting path for matched row
		if match_col[col] == -1 or dfs(match_col[col], graph, match_col, visited):
			match_col[col] = row
			return True
		
	return False

def max_matching(n, graph):
	"""
	Find maximum matching using Hungarian algorithm.
	
	Args:
		n: board size
		graph: adjacency list
		
	Returns:
		Maximum number of non-attacking rooks
	"""
	match_col = [-1] * n
	matching = 0
	
	for row in range(n):
		visited = [False] * n
		if dfs(row, graph, match_col, visited):
			matching += 1
			
	return matching

# Read input
n = int(input())

# Build graph: graph[row] = list of valid columns
graph = [[] for _ in range(n)]

for row in range(n):
	line = list(map(int, input().split()))
	for col in range(n):
		if line[col] == 1:
			graph[row].append(col)
		
# Find and output maximum matching
result = max_matching(n, graph)
print(result)