---
title: 'Suffix Automaton Tutorial'
date: 2013-08-06
permalink: /posts/2013/08/suffix-automaton-tutorial/
tags:
  - algorithm
  - data structures
  - suffix automaton
  - competitive programming
---

This is a comprehensive tutorial on Suffix Automaton (SAM) with detailed problem solutions from competitive programming contests.

## Overview

A suffix automaton is a powerful data structure for processing string problems efficiently. It represents all substrings of a string in O(n) space and can be built in O(n) time.

## Key Concepts

### What is Suffix Automaton?

A suffix automaton is a directed acyclic graph where:
- Each node represents a set of substrings (endpos equivalence class)
- Edges represent character transitions
- Parent tree (failure links) represents inclusion relationships

### Properties

1. **Linear size**: At most 2n-1 nodes and 3n-4 transitions
2. **Efficient construction**: O(n) time complexity
3. **Versatility**: Can solve many string problems

## Problem Solutions

### Problem 1: HDU 4622 - Reincarnation

Find the number of distinct substrings for each query substring.

**Approach**: Build suffix automaton and use DP to count distinct substrings.

### Problem 2: SPOJ NSUBSTR

Find maximum occurrence count for substrings of each length.

**Approach**: Use parent tree to aggregate occurrence counts, then compute answer per length.

### Problem 3: SPOJ SUBLEX

Find k-th lexicographically smallest distinct substring.

**Approach**: DP on DAG to count paths, then walk the automaton to find k-th string.

### Problem 4: SPOJ LCS

Find longest common substring of multiple strings.

**Approach**: Build SAM from first string, then walk other strings through it.

## Code Template

```cpp
const int MAXN = 100000;
const int ALPHABET = 26;

struct SuffixAutomaton {
    struct State {
        int len, link;
        int next[ALPHABET];
        State() : len(0), link(-1) {
            memset(next, -1, sizeof(next));
        }
    };
    
    vector<State> st;
    int last;
    
    SuffixAutomaton() {
        st.reserve(2 * MAXN);
        st.push_back(State());  // initial state
        last = 0;
    }
    
    void extend(char c) {
        int cur = st.size();
        st.push_back(State());
        st[cur].len = st[last].len + 1;
        
        int p = last;
        while (p != -1 && st[p].next[c] == -1) {
            st[p].next[c] = cur;
            p = st[p].link;
        }
        
        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[c];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = st.size();
                st.push_back(st[q]);
                st[clone].len = st[p].len + 1;
                
                while (p != -1 && st[p].next[c] == q) {
                    st[p].next[c] = clone;
                    p = st[p].link;
                }
                
                st[q].link = st[cur].link = clone;
            }
        }
        last = cur;
    }
};
```

## Applications

1. **Distinct substring counting**
2. **Longest common substring**
3. **K-th smallest substring**
4. **Pattern matching**
5. **Occurrence counting**

---

*This tutorial was originally written in 2013 as competitive programming notes. The full code with detailed solutions is available upon request.*