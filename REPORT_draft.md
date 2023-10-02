## no report template so we put it here for now :(

### Task 3.1 Convergence and Attractors
- When we applied the update rule repeatedly all the patterns reached a stable point right away. x1 and x3 are clearly attractors but x2d didn't converge towards a stored point but instead towards `[-1.  1. -1. -1. -1.  1. -1. -1.]`. 

- We found 14 attractors in the network of 256 possible patterns. 

- When we change half of the pattern we cannot recall the original matrix. E.g. when trying to find x1 from `[-1  1  1  1  1  1 -1 -1]` we get `[-1 1 1 -1 -1 1 -1 1]` so 3 bits are still wrong. However, when we only distort 3 bits we get the correct pattern, at least in the example when x1 is distorted to `[-1 -1  1  1  1  1 -1 -1]`.