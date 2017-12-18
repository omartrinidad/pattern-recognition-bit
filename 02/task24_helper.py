import numpy as np
from itertools import chain, combinations

def powerset(x):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(x)
  return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def phi(x):
  length = len(x)
  ps = powerset(np.arange(length))
  print map(lambda e: reduce(lambda xe, ye: xe * x[ye], e, 1), list(ps))

phi([3, 2, 1, 4])
