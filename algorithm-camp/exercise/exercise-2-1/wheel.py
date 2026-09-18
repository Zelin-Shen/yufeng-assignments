#!/usr/bin/env pypy3
"""
Turntable Problem - De Bruijn Sequence via Eulerian Circuit

Problem Description:
Generate a De Bruijn sequence B(2,n) by finding an Eulerian circuit in a graph.

Graph Construction:
- Nodes: all (n-1)-bit binary numbers (2^(n-1) nodes)
- Edges: all n-bit binary numbers (2^n edges)
- For edge representing n-bit number x:
	- Source node: first (n-1) bits of x
	- Target node: last (n-1) bits of x
- Example: for n=3, edge "101" connects node "10" to node "01"

Algorithm:
1. Build directed graph as described above
2. Find Eulerian circuit using Hierholzer's algorithm
3. The sequence of edge labels forms the De Bruijn sequence
4. Output the last bit of each edge in the circuit

Time Complexity: O(2^n)
Space Complexity: O(2^n)
"""

def find_eulerian_circuit(n):
		"""
		Find Eulerian circuit in De Bruijn graph and generate sequence.
		
		Args:
				n: bit length
				
		Returns:
				De Bruijn sequence as string
		"""
		if n == 1:
				return "01"
	
		# Graph representation: adjacency list
		# Each node is an (n-1)-bit number (0 to 2^(n-1)-1)
		# Each node has edges labeled 0 and 1
		graph = {}
		in_degree = {}
		out_degree = {}
	
		max_node = 1 << (n - 1)  # 2^(n-1) nodes
	
		# Build graph
		for node in range(max_node):
				graph[node] = [0, 1]  # Two outgoing edges (append 0 or 1)
				out_degree[node] = 2
				in_degree[node] = 0
			
		# Count in-degrees
		for node in range(max_node):
				for bit in [0, 1]:
						# Next node: shift left, add bit, keep (n-1) bits
						next_node = ((node << 1) | bit) & (max_node - 1)
						in_degree[next_node] += 1
					
		# Find Eulerian circuit using Hierholzer's algorithm
		# Start from node 0
		stack = [0]
		circuit = []
		current_edges = {node: 0 for node in range(max_node)}  # Track which edge to use next
	
		while stack:
				node = stack[-1]
			
				# Check if node has unused edges
				if current_edges[node] < len(graph[node]):
						# Take next edge
						bit = graph[node][current_edges[node]]
						current_edges[node] += 1
					
						# Calculate next node
						next_node = ((node << 1) | bit) & (max_node - 1)
						stack.append(next_node)
				else:
						# No more edges, add to circuit
						circuit.append(stack.pop())
					
		# Reverse circuit to get correct order
		circuit.reverse()
	
		# Build De Bruijn sequence from circuit
		# Start with first node (n-1 bits)
		result = []
	
		# For each transition in circuit, record the bit that was added
		for i in range(len(circuit) - 1):
				curr_node = circuit[i]
				next_node = circuit[i + 1]
			
				# Determine which bit was added
				# next_node = ((curr_node << 1) | bit) & (max_node - 1)
				# Try bit = 0 and bit = 1
				for bit in [0, 1]:
						if ((curr_node << 1) | bit) & (max_node - 1) == next_node:
								result.append(str(bit))
								break
					
		# Prepend the starting node (n-1 zeros)
		sequence = "0" * (n - 1) + "".join(result)
	
		# Return first 2^n characters
		return sequence[:1 << n]

# Read input
n = int(input())

# Generate De Bruijn sequence via Eulerian circuit
result = find_eulerian_circuit(n)

# Output result
print(result)