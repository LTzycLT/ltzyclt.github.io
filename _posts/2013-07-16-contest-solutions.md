---
title: 'A Contest to Celebrate Girlfriend''s Birthday'
date: 2013-07-16
permalink: /posts/2013/07/cut-rope-ii/
tags:
  - competitive programming
  - dynamic programming
  - algorithm
---

A competitive programming contest problem solution set - named after celebrating my girlfriend's birthday!

## Problem 1: Cut the Rope II

**Problem Statement:** Given a rope of length L, count the number of ways to cut it into segments such that all segments have different lengths. (L ≤ 50000)

**Solution Approach:**

A clever DP approach. Since all segments must be different, we can observe that there are at most x segments where x*(x+1)/2 ≤ L, so x is at most around 320.

Define `dp[i][j][k]` where:
- i ≤ 320 (number of segments)
- j ≤ 50000 (total length used)
- k < 2 (state indicator)

State definitions:
- `k = 0`: Number of ways to use i segments with total length j, **without** any segment of length 1
- `k = 1`: Number of ways to use i segments with total length j, **with** at least one segment of length 1

**Transitions:**
```
dp[i][j][0] = dp[i][j-i][0] + dp[i][j-i][1]
// Increase all current segments by 1
dp[i][j][1] = dp[i-1][j-1][0]
// Add a new segment of length 1
```

**Complexity:** O(320 × 50000)

## Problem 2: Shortest Path

**Problem Statement:** Find the shortest path from node 1 to node n where edge lengths are strictly increasing along the path.

**Solution 1: Sorting + Batch Processing**

Sort all edges by length. Process edges with the same length together, updating distances accordingly.

**Solution 2: Graph Reconstruction**

1. Split each undirected edge into two directed edges
2. Create new nodes for each edge endpoint
3. Sort nodes by edge length, ensuring that nodes with smaller edge lengths come first
4. Add zero-length directed edges between adjacent nodes in the sorted order
5. Run shortest path algorithm on the new graph

This guarantees that we always traverse edges in strictly increasing order of length.

---

*These problems were solved during a fun contest in 2013!*