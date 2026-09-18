#!/usr/bin/env pypy3
"""
Convex Hull Construction using Andrew's Monotone Chain Algorithm

Problem:
	Given N 2D points, find the convex hull.
	All points lying on the edges of the convex hull (collinear points) must be included.
	Compute a specific hash value based on the indices of the hull points.

Algorithm:
	1. Sort the points lexicographically (by x, then by y).
	2. Construct the Lower Hull:
		Iterate through sorted points. While the sequence makes a non-left turn
		(cross product < 0), pop the last point. This retains collinear points (cross product == 0).
	3. Construct the Upper Hull:
		Iterate through sorted points in reverse. Apply the same logic.
	4. Concatenate the lower and upper hulls.
	5. Remove duplicate points (the start and end points are shared, and strictly
		collinear points on the "ends" of the hull might be added by both halves).
		Using a set based on coordinates handles deduplication efficiently.
	6. Calculate the result: (Product of indices * m) % (n + 1).

Complexity:
	Time:  O(N log N) due to sorting. The hull construction is O(N).
	Space: O(N) to store points and the hull.
"""

import sys

# Define the point class as per the template
class ip:
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.i = 0 # 1-based index
		
	# Lexicographical comparison
	def __lt__(self, other):
		if self.x == other.x:
			return self.y < other.y
		return self.x < other.x
	
# Calculate the cross product of vectors OA and OB.
# Returns a positive value for counter-clockwise turn, negative for clockwise, 0 for collinear.
def cross(o, a, b):
	return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

# Compute the convex hull
# a: list of all input points (ip objects)
# b: list to store the convex hull points (pre-allocated)
# n: number of points
# Returns: the number of points on the convex hull (after deduplication)
def convex(a, b, n):
	# Sort points lexicographically
	a.sort()
	
	# Build lower hull
	lower = []
	for p in a:
		while len(lower) >= 2:
			# To include collinear points on the boundary, we only pop if the turn is strictly clockwise (right turn).
			# If cross == 0 (collinear), we keep the point.
			if cross(lower[-2], lower[-1], p) < 0:
				lower.pop()
			else:
				break
		lower.append(p)
		
	# Build upper hull
	upper = []
	for p in reversed(a):
		while len(upper) >= 2:
			if cross(upper[-2], upper[-1], p) < 0:
				upper.pop()
			else:
				break
		upper.append(p)
		
	# Concatenate lower and upper hulls to form the full hull.
	# lower[-1] is the same as upper[0] (rightmost point).
	# upper[-1] is the same as lower[0] (leftmost point).
	# So we take all of lower, and all of upper except the ends.
	# However, if all points are collinear, lower contains all points, upper contains all points reversed.
	# The concatenation logic below puts them together. Deduplication is handled next.
		
	hull_list = lower[:-1] + upper[:-1]
	
	# Deduplicate points.
	# Since points on the connecting line between the leftmost and rightmost points 
	# might be visited by both lower and upper hull construction (in the case of collinearity),
	# we use a set based on coordinates to ensure unique points.
	unique_map = {}
	for p in hull_list:
		# Use (x, y) tuple as key
		unique_map[(p.x, p.y)] = p
		
	# Fill the output array b
	idx = 0
	for p in unique_map.values():
		b[idx] = p
		idx += 1
		
	return idx

input_data = sys.stdin.read().split()
if input_data:
	iterator = iter(input_data)
	try:
		n = int(next(iterator))
		a = [ip(0, 0) for i in range(n)]
		b = [ip(0, 0) for i in range(n + 1)]
		
		for i in range(n):
			x = int(next(iterator))
			y = int(next(iterator))
			a[i] = ip(x, y)
			a[i].i = i + 1
			
		m = convex(a, b, n)
		ans = m
		for i in range(m):
			ans = (ans * b[i].i) % (n + 1)
		print(ans)
	except StopIteration:
		pass