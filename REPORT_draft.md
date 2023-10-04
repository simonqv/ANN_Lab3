## no report template so we put it here for now :(

### Task 3.1 Convergence and Attractors
- When we applied the update rule repeatedly all the patterns reached a stable point right away. x1 and x3 are clearly attractors but x2d didn't converge towards a stored point but instead towards `[-1.  1. -1. -1. -1.  1. -1. -1.]`. 

- We found 14 attractors in the network of 256 possible patterns. 

- When we change half of the pattern we cannot recall the original matrix. E.g. when trying to find x1 from `[-1  1  1  1  1  1 -1 -1]` we get `[-1 1 1 -1 -1 1 -1 1]` so 3 bits are still wrong. However, when we only distort 3 bits we get the correct pattern, at least in the example when x1 is distorted to `[-1 -1  1  1  1  1 -1 -1]`.


### Task 3.2 Sequential Update
- We found that the patterns are stable by having no mismatches when comparing the pattern we trained on with the output we get when trying to recall that same pattern. 

- For p10 we set 200 random elements to 1 to degrade the pattern p1. To make a mixture of p2 and p3 to create p11 we take first 512 elements from p2 and last 512 elements from p3 and combine them. This results in p11 finding p3. If we change the ration to e.g. 800 first from p2 we recall p2 instead. We also manage to recall p1 from p10.


### Task 3.3 Energy
Energy at p1     -1439.390625
Energy at p2     -1365.640625
Energy at p3     -1462.25

Energy at distorted patterns
Energy at p10    -612.73046875
Energy at p11    -198.54296875 

The energy gets lower each iteration when we do sequentail learning. When we train on p1, p2 and p3 and try to recall using p10 we find that the energy reaches the energy value at p1. see figure 3_3_point3.png We note the energy at every 200 step in the sequential update, total 20 energy-values recorded. 

The energy fluctuates heavily when we set the weights to random values. (We calculated the weight by creating a symmetric weight matrix where the input values followed N(0,1) and the final matrix was normalized. We then shuffled this matrix to get the random one.) We are not successful in finding p1 from these random weights. The energy never finds a minimum but instead jumps around close to 0. see fig 3_3rand

When we make the weights random but symmetric we can still not recall anything like p1 but the energy is much lower. Around -600. This indicates we find a local minima. see figure 3_3sym. this is because ????