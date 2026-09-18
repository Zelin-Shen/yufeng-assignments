"""
Road Upgrade - Maximum Spanning Tree Problem:

Given n cities and m bidirectional roads where road i has weight i, select minimum
roads to upgrade such that for any two cities, the maximum bottleneck path in the
upgraded graph equals the maximum bottleneck path in the original graph.

Solution: Build Maximum Spanning Tree using Kruskal's algorithm
- Sort edges by weight in descending order (by road number)
- Use union-find to select edges that don't form cycles
- Select exactly n-1 edges to form a spanning tree

Time Complexity: O(m log m)
Space Complexity: O(n + m)
"""

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n + 1))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y
            return True
        return False

n, m = map(int, input().split())

edges = []
for i in range(1, m + 1):
    u, v = map(int, input().split())
    edges.append((i, u, v))

edges.sort(reverse=True)

uf = UnionFind(n)
selected = []

for road_num, u, v in edges:
    if uf.union(u, v):
        selected.append(road_num)
        if len(selected) == n - 1:
            break

selected.sort()

print(len(selected))
for road in selected:
    print(road)