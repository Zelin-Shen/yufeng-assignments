"""
Stack Sort: Sort a stack using only one auxiliary stack.

Algorithm:
- Pop the top element 'temp' from the input stack.
- While the auxiliary stack is not empty and its top > temp,
  move aux's top back to the input stack.
- Push temp onto the auxiliary stack.
- Repeat until the input stack is empty.
- The auxiliary stack is now sorted (smallest at bottom, largest on top).

Time Complexity : O(n^2) worst case
Space Complexity: O(n) for the auxiliary stack
"""

import sys

data = sys.stdin.read().split()
n = int(data[0])
stack = [int(data[i + 1]) for i in range(n)]

aux = []

while stack:
    temp = stack.pop()
    while aux and aux[-1] > temp:
        stack.append(aux.pop())
    aux.append(temp)

print('\n'.join(map(str, aux)))