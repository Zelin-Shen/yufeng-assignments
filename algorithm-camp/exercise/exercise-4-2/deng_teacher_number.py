#!/usr/bin/env pypy3
#!/usr/bin/env pypy3
"""
Deng's Number Problem — Sieve of Eratosthenes + Smallest Prime Factor (SPF)

Definitions:
		- Prime: natural number > 1 with no divisors other than 1 and itself.
		- Deng's Number: a composite number whose ALL non-trivial divisors
			(excluding 1 and itself) are prime.
			Equivalently, a semiprime — product of exactly two primes (with
			multiplicity), i.e. p*q where p,q are primes (p <= q).

Key insight:
		A composite x is a Deng's Number iff x / spf(x) is prime,
		where spf(x) is the smallest prime factor of x.
		Proof: if x = p1 * p2 * ... * pk (k >= 2), then x has a composite
		divisor p1*p2 whenever k >= 3. So k must equal exactly 2.

Algorithm:
		1. Run Eratosthenes sieve to mark primes and record spf[i] for each i.
		2. k=0 → output all primes <= n.
				k=1 → for each composite i, output i if i // spf[i] is prime.

Complexity: O(n log log n) time, O(n) space.  n <= 2*10^5.
"""

import sys

def solve():
	data = sys.stdin.buffer.read().split()
	n = int(data[0])
	k = int(data[1])
	
	# Sieve of Eratosthenes — fast slice zeroing
	is_prime = bytearray(n + 1)
	if n >= 2:
		is_prime[2:] = b'\x01' * (n - 1)
		
	i = 2
	while i * i <= n:
		if is_prime[i]:
			is_prime[i * i::i] = bytearray((n - i * i) // i + 1)
		i += 1
		
	out = []
	if k == 0:
		# Output all primes <= n
		for i in range(2, n + 1):
			if is_prime[i]:
				out.append(str(i))
	else:
		# Generate all semiprimes via prime pairs
		primes = [i for i in range(2, n + 1) if is_prime[i]]
		is_deng = bytearray(n + 1)
		np_ = len(primes)
		for i in range(np_):
			p = primes[i]
			if p * p > n:
				break
			for j in range(i, np_):
				pq = p * primes[j]
				if pq > n:
					break
				is_deng[pq] = 1
		# Scan in ascending order
		for i in range(4, n + 1):
			if is_deng[i]:
				out.append(str(i))
				
	if out:
		sys.stdout.write('\n'.join(out) + '\n')
		
solve()