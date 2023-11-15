import numpy as np


def primes_less_than(n):
    """Returns sorted array of primes such that `2 <= prime < n`."""
    j = 3
    primes = np.ones((n + 1) // 2, dtype=bool)
  
    while j * j <= n:
        if primes[j//2]:
            primes[j*j//2::j] = False
        j += 2
  
    ret = 2 * np.where(primes)[0] + 1
    ret[0] = 2  # :(
    
    return ret
