---
title: 'Suffix Automaton Tutorial'
date: 2013-08-06 09:47:01
tags:
  - acm
  - algorithm
  - data structures
  - suffix automaton
---

This is a comprehensive tutorial on Suffix Automaton (SAM) with detailed problem solutions.

## Problem 1: HDU 4622 - Reincarnation

**题意:** 询问每个子串里包含的不同的子串的个数。

**思路:** 枚举每一个点作为起点做后缀自动机，维护自动机每个节点能有多少条路径到达。

```c++
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <queue>
#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))
#define pii pair<int,int>
#define mpr(a,b) make_pair(a,b)
using namespace std;

const int ch = 26;
const int N=100000;
class SAM{
    public:
    int f[N],chd[N][ch+1],len[N],sw[200];
    //ch+1 number of tempt suffix at status
    int last,sz;
    void init(){
        clr(chd[0],0);
        f[0] = -1;
        chd[0][ch] = 1;
        last = 0;
        sz = 0;
        for(int i=0;i<26;i++)
            sw[i+'a'] = i;
    }
    int add(char inc,int l){
        int ans = 0;
        int c = sw[inc];
        int x = last;
        last = ++sz;
        clr(chd[sz],0);
        len[sz] = l;
        for(;x!=-1&&chd[x][c]==0;x=f[x]){
            chd[x][c] = sz;
            chd[last][ch] += chd[x][ch];
            ans += chd[x][ch];
        }
        if(x == -1){
            f[sz]=0;
        }
        else{
            int y = chd[x][c];
            if(len[y] == len[x]+1){
                f[sz]=y;
            }
            else{
                sz++;
                len[sz] = len[x] + 1;
                
                memcpy(chd[sz],chd[y],ch*4);
//                for(int i=0;i<ch;i++) chd[sz][i] = chd[y][i];
                chd[sz][ch] = 0;

                f[sz] = f[y];
                f[last] = f[y] = sz;

                for(;x!=-1&&chd[x][c]==y;x=f[x]){
                    chd[x][c] = sz;
                    chd[y][ch] -= chd[x][ch];
                    chd[sz][ch] += chd[x][ch];
                }
            }
        }
        return ans;
    }
}sam;

char s[100000];
int n,m;


int g[2005][2005];
int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    int T;
    scanf("%d",&T);
    while(T--){
        scanf("%s",s);
        n = strlen(s);
        for(int i=0;i<n;i++){
            sam.init();
            for(int j=i;j<n;j++){
                if(j!=i)
                    g[i][j] = g[i][j-1] + sam.add(s[j],j-i+1);
                else
                    g[i][j] = sam.add(s[j],j-i+1);
            }
        }
        int Q;
        scanf("%d",&Q);
        while(Q--){
            int l,r;
            scanf("%d%d",&l,&r);
            l--;r--;
            printf("%d\n",g[l][r]);
        }
    }
    return 0;
}
```

## Problem 2: HDU 4641 - Dynamic Substring Count

**题意:** 给一个串在末尾动态增加字符，询问当前串中至少出现k次的子串个数。

**思路:** 与上题相比还需要维护增加字符时，以新加入字符为串尾的串的相应节点出现次数，只要更新parent树中当前last的祖先即可。

```c++
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <queue>
#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))
#define pii pair<int,int>
#define mpr(a,b) make_pair(a,b)
using namespace std;

const int ch = 26;
const int N=1000000;
int k;

class SAM{
    public:
    int f[N],chd[N][ch+2],len[N],sw[200];
    //ch+1 number of tempt suffix at status
    int last,sz;
    void init(){
        clr(chd[0],0);
        f[0] = -1;
        chd[0][ch] = 1;
        last = 0;
        sz = 0;
        for(int i=0;i<26;i++)
            sw[i+'a'] = i;
    }
    ll add(char inc,int l){
        int c = sw[inc];
        int x = last;
        last = ++sz;
        clr(chd[sz],0);
        len[sz] = l;
        for(;x!=-1&&chd[x][c]==0;x=f[x]){
            chd[x][c] = sz;
            chd[last][ch] += chd[x][ch];
        }
        if(x == -1){
            f[sz]=0;
        }
        else{
            int y = chd[x][c];
            if(len[y] == len[x]+1){
                f[sz]=y;
            }
            else{
                sz++;
                len[sz] = len[x] + 1;
                for(int i=0;i<ch;i++) chd[sz][i] = chd[y][i];
                chd[sz][ch] = 0;
                chd[sz][ch+1] = chd[y][ch+1];

                f[sz] = f[y];
                f[last] = f[y] = sz;

                for(;x!=-1&&chd[x][c]==y;x=f[x]){
                    chd[x][c] = sz;
                    chd[y][ch] -= chd[x][ch];
                    chd[sz][ch] += chd[x][ch];
                }

            }
        }
        ll ans = 0;
        x = last;
        for(x = last;x!=0;x=f[x]){
            if(chd[x][ch+1]>=k) break;
            chd[x][ch+1] ++;
            if(chd[x][ch+1]==k)
                ans += chd[x][ch];
        }
        return ans;
    }
}sam;

char s[100000];
int n,m;


int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    while(scanf("%d%d%d",&n,&m,&k)!=EOF){
        scanf("%s",s);
        sam.init();
        ll ans = 0;
        for(int i=0;i<n;i++)
            ans += sam.add(s[i],i+1);
        while(m--){
            int type;
            scanf("%d",&type);
            if(type==1){
                char c[2];
                scanf("%s",c);
                ans += sam.add(c[0],++n);
            }
            else{
                printf("%lld\n",ans);
            }
        }
    }
    return 0;
}
```

## Problem 3: HDU 2609 - 循环不同构

**题意:** 给你n个串，问有多少个是循环不同构。

**思路:** 可以使用最小表示法来求，后缀自动机的话先把串扩展成两倍，加进自动机，然后在自动机里求长度为原来长度的最小的串即可，需要遍历一遍。最后用trie树判重。

## Problem 4: SPOJ NSUBSTR - 最大出现次数

**题意:** 求一个串中长度为i的串最多出现的次数。

**思路:** 记录自动机每个节点到达的次数，利用parent树来做。然后利用自动机的性质得到每个节点最长到达的串，即构造时的len,然后从大到小更新一遍。

## Problem 5: SPOJ SUBLEX - 第k小子串

**题意:** 询问一个串中第k小的不同的子串。

**思路:** 关键要得到自动机每个节点后面能够有多少种不同走法，有向无环图做下dp。然后走一遍自动机。

## Problem 6: SPOJ LCS / LCS2 - 最长公共子串

**题意:** 求n个串的最长公共子串。

**思路:** 这两题比较类似。先以第一个串构造自动机，然后每个串都在自动机上走一遍，维护每个节点所能匹配的最长的串，对于所有串每个节点取个匹配的最小值，再对所有节点取个最大值即可。

## Problem 7: BZOJ 2806

这题比较综合。就说一下后缀自动机的部分：要得到一个串在某个位置往前与其他模版串所能匹配的最长长度。把模版串用'$'号串起来，建立后缀自动机。然后一开始的串走一遍就可以了。

## Problem 8: HDU 4436 - 子串数字和

**题意:** 给你多个由0~9组成的字符串，问其中所有子串代表的不同的数字的和。

**思路:** 位数高的在前面，所以先把串反过来，让位数高的后加进自动机。多个串要判重所以先用'$'连接起来，建立自动机。要得到每个节点后面的其他节点的值以及出现次数，然后当前节点的值就是 `dp[u] = dp[v] * 10 + i * p[v]` (i为转移的数字，p[v]为出现次数)，还是dag上的dp。注意要把i=0的情况处理好。答案就是dp[0]。

## Problem 9: SPOJ COT4

看了一眼别人的做法，只能说我没打过ACM。有空在写。。。。

---

*This tutorial was originally written in 2013 as competitive programming notes.*