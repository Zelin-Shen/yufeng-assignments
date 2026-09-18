#!/usr/bin/env pypy3
"""
Card Game Strategy - Greedy Approach

Problem Description:
    A game involves 2n cards (values 1 to 2n) split between two players.
    In each of n rounds, both players play one card. The player with the smaller value wins.
    The goal is to calculate the maximum number of wins for the first player.

Algorithm:
    1. Identification:
       Determine the opponent's cards by finding the missing numbers in the range [1, 2n]
       relative to the player's hand.
    2. Sorting:
       Sort both the player's cards (A) and opponent's cards (B) in ascending order.
       Note: In this game, smaller value = stronger card.
    3. Greedy Matching (Two Pointers):
       We maintain pointers to the strongest (left/smallest) and weakest (right/largest) cards
       for both players.
       Iterating n times:
       - If my strongest card (A[left]) beats opponent's strongest card (B[left]):
         This is a guaranteed win. Taking it immediately is optimal because if I can beat their
         strongest, I should use my strongest available to minimize "overkill" (saving stronger
         cards for later is not an issue as we process optimally).
         Action: Win++, left pointers move right.
       - Else (My strongest loses to opponent's strongest):
         Opponent's strongest card is unbeatable by any of my remaining cards.
         I must lose this round. To minimize resource loss, I sacrifice my weakest card
         (A[right]) to consume their strongest card (B[left]).
         Action: Loss, my right pointer moves left, opponent left pointer moves right.

Optimization Details:
    - Fast I/O: `sys.stdin.read().split()` is used for O(1) input reading relative to size.
    - Presence Array: A boolean list is used to identify opponent's cards in O(n), which is
      faster and more memory-efficient than a hash set for dense integer ranges.
    - Local Variables: Variables are localized inside the function to reduce global lookup overhead.
    - Output: `sys.stdout.write` is used for faster output.

Complexity:
    Time: O(n log n) due to sorting. The greedy logic is O(n).
    Space: O(n) for storing cards and presence flags.
"""

import sys

def solve():
    # Read all input data at once for maximum efficiency
    input_data = sys.stdin.read().split()
    
    if not input_data:
        return

    iterator = iter(input_data)
    
    # Read n
    n = int(next(iterator))
    
    # Initialize variables
    # Using a list for player cards
    my_cards = []
    # Efficient presence check using a boolean list
    # Indices 1 to 2n. Size 2n + 1 to allow direct indexing.
    limit = 2 * n + 1
    present = [False] * limit
    
    # Read my n cards
    for _ in range(n):
        x = int(next(iterator))
        my_cards.append(x)
        present[x] = True
        
    # Identify opponent's cards
    oppo_cards = []
    # Iterate through the total range to find missing cards
    for i in range(1, limit):
        if not present[i]:
            oppo_cards.append(i)
            
    # Sort both hands in ascending order
    # Smallest value is the "Strongest" card
    my_cards.sort()
    oppo_cards.sort()
    
    # Greedy strategy variables
    wins = 0
    l_a, l_b = 0, 0           # Pointers to the strongest (smallest) cards
    r_a, r_b = n - 1, n - 1   # Pointers to the weakest (largest) cards
    
    # Simulate n rounds
    for _ in range(n):
        # Compare strongest cards
        if my_cards[l_a] < oppo_cards[l_b]:
            # Win case: My strongest beats opponent's strongest
            wins += 1
            l_a += 1
            l_b += 1
        else:
            # Lose case: Opponent's strongest is unbeatable.
            # Sacrifice my weakest card to consume it.
            r_a -= 1
            l_b += 1
            
    # Output the result
    sys.stdout.write(str(wins) + '\n')

solve()