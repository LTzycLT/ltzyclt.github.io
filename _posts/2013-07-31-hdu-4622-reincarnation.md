---
title: 'HDU 4622 Reincarnation - 三种解法对比'
date: 2013-07-31 13:02:00
tags:
  - acm
  - algorithm
  - suffix array
  - hash
  - suffix automaton
---

**题意:** 长度为2000的串，10000个询问：[l,r]这个子串包含多少不同的子串。

比赛时用的后缀数组 o(n*Q)，赛后用的 hash o(n*n*log(n)) 和后缀自动机，这个log的出现是因为hash写不来 = =。

# 解法一：后缀数组

先对原来的串做一遍后缀数组，对于每个询问，对sa数组进行扫描，如果当前前缀在[l,r]范围内，就找到之前和他公共前缀最长且也在[l,r]范围内的串，lcp最长公共前缀，答案加上串在当前询问中的长度减去最长公共前缀。

值得注意的是和当前串公共前缀最长的并不就是上一次在[l,r]范围内的串，也有可能是再之前，原因是询问的区间，相当于给每个串截断了一部分，所以区间里sa数组不再代表这区间里的串的排名，处理比较容易，见代码。

```c++
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <set>
#define ll long long
#define clr(a,b) memset(a,b,sizeof(a));
using namespace std;

const int N = 2010;
int ua[N],ub[N],us[N];
int cmp(int *r,int a,int b,int l){
    return r[a]==r[b]&&r[a+l]==r[b+l];
}
void da(int *r,int *sa,int n,int m){
    int i,j,p,*x=ua,*y=ub,*t;
    for(i=0;i<m;i++)us[i]=0;
    for(i=0;i<n;i++)us[x[i]=r[i]]++;
    for(i=1;i<m;i++)us[i]+=us[i-1];
    for(i=n-1;i>=0;i--)sa[--us[x[i]]]=i;
    for(j=1,p=1;p<n;j*=2,m=p){
        for(p=0,i=n-j;i<n;i++)y[p++]=i;
        for(i=0;i<n;i++)if(sa[i]>=j)y[p++]=sa[i]-j;
        for(i=0;i<m;i++)us[i]=0;
        for(i=0;i<n;i++)us[x[i]]++;
        for(i=1;i<m;i++)us[i]+=us[i-1];
        for(i=n-1;i>=0;i--)sa[--us[x[y[i]]]]=y[i];
        for(t=x,x=y,y=t,p=1,x[sa[0]]=0,i=1;i<n;i++)
            x[sa[i]]=cmp(y,sa[i-1],sa[i],j)?p-1:p++;
    }
}
int rank[N],height[N];
void calheight(int *r,int *sa,int n){
    int i,j,k=0;
    for(i=1;i<=n;i++)rank[sa[i]]=i;
    for(i=0;i<n;height[rank[i++]]=k)
        for(k?k--:0,j=sa[rank[i]-1];r[i+k]==r[j+k];k++);
}
int *RMQ=height;
int mm[N];
int best[20][N];
void initRMQ(int n){
    int i,j,a,b;
    for(mm[0]=-1,i=1;i<=n;i++)
        mm[i]=((i&(i-1))==0)?mm[i-1]+1:mm[i-1];
    for(i=1;i<=n;i++)best[0][i]=i;
    for(i=1;i<=mm[n];i++)for(j=1;j<=n+1-(1<<i);j++){
        a=best[i-1][j];
        b=best[i-1][j+(1<<(i-1))];
        if(RMQ[a]<RMQ[b])best[i][j]=a;
        else best[i][j]=b;
    }
}
int askRMQ(int a,int b){
    int t;
    t=mm[b-a+1];b-=(1<<t)-1;
    a=best[t][a];b=best[t][b];
    return RMQ[a]<RMQ[b]?a:b;
}
int lcp(int a,int b){
    int t;
    a=rank[a],b=rank[b];
    if(a>b){t=a;a=b;b=t;}
    return height[askRMQ(a+1,b)];
}
int t,r[N],m,len[N],sa[N];
char a[N];
int main(){
//    freopen("/home/axorb/in","r",stdin);
    scanf("%d",&t);
    while(t--){
        scanf("%s",a);
        int l=strlen(a);
        for(int i=0;i<l;i++){
            r[i]=a[i];
            len[i]=l-i;
        }
        r[l]=0;len[l]=0;
        da(r,sa,l+1,256);
        calheight(r,sa,l);
        initRMQ(l);
        scanf("%d",&m);
        while(m--){
            int q,w;
            scanf("%d%d",&q,&w);
            q--,w--;
            int o=w-q+1;
            int last=-1;
            int ans=0;
            for(int i=1,p=0;i<=l&&p<o;i++)if(sa[i]>=q&&sa[i]<=w){
                p++;
                if(last==-1)ans+=min(len[sa[i]],w-sa[i]+1);
                else{
                    int h=min(lcp(sa[i],sa[last]),w-sa[last]+1);
                    ans+=min(len[sa[i]],w-sa[i]+1)-min(h,w-sa[i]+1);
                }
                if(last!=-1&&w-sa[last]+1 > w-sa[i]+1){
                    int he = lcp(sa[last],sa[i]);
                    if(he >= w-sa[i]+1) last = last;
                    else last = i;
                }                                    //维护之前公共前缀最长的串，只要判断当前串是否是last的前缀，是last不变，不是last就为i
                else last = i;
            }
            printf("%d\n",ans);
        }
    }
    return 0;
}
```

# 解法二：Hash

用hash把所有相同的子串放在一起，[l1,r1] [l2,r2]...。现在对于每一个询问，只把这个串第一次出现的算是这个询问的，也就是询问[L,R]，满足 l1<L<=l2, R>=r2 时 [l2,r2] 算给他。

然后利用扫描线和部分和维护，对于子串[l2,r2]，g[r2][l1+1] 加1，g[r2][l2+1] 减1，对于所有r，从左到右求一遍部分和，再从上到下求一遍部分和，g[r][l]就是询问l,r的答案了。

# 解法三：后缀自动机

预处理，解法不多说（详见后缀自动机教程）。

---

*Originally written in 2013, comparing three different approaches to solve the same problem.*