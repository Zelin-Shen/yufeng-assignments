"""
Equality Constraints - Union-Find Problem:

Given n variables and m constraints (equality or inequality), determine if there\nexists a valid assignment satisfying all constraints.

Algorithm:
1. Process all equality constraints (c=1) first using union-find to merge variables
2. Check all inequality constraints (c=0) for conflicts
3. If two variables that must be unequal are in the same set, output "No"

Time Complexity: O(m * α(n)) where α is inverse Ackermann (nearly O(1))
Space Complexity: O(n)
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

t = int(input())
for _ in range(t):
    n, m = map(int, input().split())
    
    equal = []
    unequal = []
    
    for _ in range(m):
        a, b, c = map(int, input().split())
        if c == 1:
            equal.append((a, b))
        else:
            unequal.append((a, b))
    
    uf = UnionFind(n)
    
    for a, b in equal:
        uf.union(a, b)
    
    valid = True
    for a, b in unequal:
        if uf.find(a) == uf.find(b):
            valid = False
            break
    
    print("Yes" if valid else "No")