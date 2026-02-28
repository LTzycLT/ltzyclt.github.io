---
title: 'HDU 4622 - Reincarnation: Distinct Substrings'
date: 2013-08-10
permalink: /posts/2013/08/hdu-4622-reincarnation/
tags:
  - competitive programming
  - suffix automaton
  - string algorithms
---

**Problem**: Given a string, for each query, find the number of distinct substrings in a given substring [l, r].

**Constraints**: String length up to 2000, multiple test cases.

## Solution Approach

This problem can be efficiently solved using **Suffix Automaton (SAM)**.

### Key Insight

For each position `i` as the starting point, we build a suffix automaton for the substring `s[i...n-1]`. For each node in the automaton, we maintain the number of distinct paths (which equals the number of distinct substrings ending at that node).

### Algorithm

1. For each starting position `i`, initialize a new SAM
2. Extend the SAM with characters from `s[i]` to `s[n-1]`
3. While extending, accumulate the count of distinct substrings
4. Store the results in a 2D array `g[i][j]` for quick query response

### Implementation

```cpp
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <queue>
#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))
using namespace std;

const int ch = 26;
const int N = 100000;

class SAM {
public:
    int f[N], chd[N][ch+1], len[N], sw[200];
    int last, sz;
    
    void init() {
        clr(chd[0], 0);
        f[0] = -1;
        chd[0][ch] = 1;
        last = 0;
        sz = 0;
        for(int i = 0; i < 26; i++)
            sw[i+'a'] = i;
    }
    
    int add(char inc, int l) {
        int ans = 0;
        int c = sw[inc];
        int x = last;
        last = ++sz;
        clr(chd[sz], 0);
        len[sz] = l;
        
        for(; x != -1 && chd[x][c] == 0; x = f[x]) {
            chd[x][c] = sz;
            chd[last][ch] += chd[x][ch];
            ans += chd[x][ch];
        }
        
        if(x == -1) {
            f[sz] = 0;
        } else {
            int y = chd[x][c];
            if(len[y] == len[x] + 1) {
                f[sz] = y;
            } else {
                sz++;
                len[sz] = len[x] + 1;
                memcpy(chd[sz], chd[y], ch*4);
                chd[sz][ch] = 0;
                f[sz] = f[y];
                f[last] = f[y] = sz;
                
                for(; x != -1 && chd[x][c] == y; x = f[x]) {
                    chd[x][c] = sz;
                    chd[y][ch] -= chd[x][ch];
                    chd[sz][ch] += chd[x][ch];
                }
            }
        }
        return ans;
    }
} sam;

char s[100000];
int n, m;
int g[2005][2005];

int main() {
    int T;
    scanf("%d", &T);
    while(T--) {
        scanf("%s", s);
        n = strlen(s);
        for(int i = 0; i < n; i++) {
            sam.init();
            for(int j = i; j < n; j++) {
                if(j != i)
                    g[i][j] = g[i][j-1] + sam.add(s[j], j-i+1);
                else
                    g[i][j] = sam.add(s[j], j-i+1);
            }
        }
        int Q;
        scanf("%d", &Q);
        while(Q--) {
            int l, r;
            scanf("%d%d", &l, &r);
            l--; r--;
            printf("%d\n", g[l][r]);
        }
    }
    return 0;
}
```

### Complexity Analysis

- **Preprocessing**: O(n²) where n is string length
- **Query**: O(1) per query
- **Space**: O(n²) for storing the DP table

### Related Problems

- HDU 4641: Dynamic version with character additions
- SPOJ NSUBSTR: Maximum occurrences per length
- SPOJ SUBLEX: K-th smallest substring

---

*This solution was originally written in 2013 during competitive programming practice.*