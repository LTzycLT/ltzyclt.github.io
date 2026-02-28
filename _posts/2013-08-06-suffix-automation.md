---
title: suffix automation
date: 2013-08-06 09:47:01
tags: acm
---

hdu 4622
题意：询问每个子串里包含的不同的子串的个数。
枚举每一个点作为起点做后缀自动机，维护自动机每个节点能有多少条路径到达。
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

hdu 4641
题意：给一个串在末尾动态增加字符，询问当前串中至少出现k次的子串个数。
与上题相比还需要维护增加字符时，以新加入字符为串尾的串的相应节点出现次数，只要更新parent树中当前last的祖先即可
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

hdu 2609
题意：给你n个串，问有多少个是循环不同构。
可以使用最小表示法来求，后缀自动机的话先把串扩展成两倍，加进自动机，然后在自动机里求长度为原来长度的最小的串即可，需要遍历一遍。
最后用trie树判重。
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

char s[100000];
int n,m;

const int ch = 2;
const int N=1000000;

class SAM{
    public:
    int f[N],chd[N][ch],len[N],sw[200];
    int last,sz;
    void init(){
        clr(chd[0],0);
        last = 0;
        sz = 0;
        sw['0'] = 0;
        sw['1'] = 1;
    }
    void add(char inc,int l){
        int c = sw[inc];
        int x = last;
        last = ++sz;
        clr(chd[sz],0);
        len[sz] = l;
        for(;x&&chd[x][c]==0;x=f[x]) chd[x][c] = sz;
        int y = chd[x][c];
        if(y == 0){chd[x][c]=sz;f[sz]=0;}
        else if(len[y] == len[x]+1){f[sz]=y;}
        else{
            sz++;
            len[sz] = len[x] + 1;
            for(int i=0;i<ch;i++) chd[sz][i] = chd[y][i];
            f[sz] = f[y];
            f[last] = f[y] = sz;
            for(;x&&chd[x][c]==y;x=f[x]) chd[x][c] = sz;
            if(chd[x][c]==y) chd[x][c] = sz;

            for(;x;x=f[x]) sz = sz;
        }
    }
    void go(char s[],int dis){
        int rt = 0;
        for(int i=0;i<dis;i++){
            for(int j=0;j<ch;j++){
                if(chd[rt][j]){
                    s[i] = '0' + j;
                    rt = chd[rt][j];
                    break;
                }
            }
        }
    }
}sam;
class Trie{
    public:
    int trie[N][ch+1],sw[200];
    int top;
    void init(){
        top = 0;
        clr(trie[0],-1);
        sw['0'] = 0;
        sw['1'] = 1;
    }
    void insert(char s[],int len){
        int tmp = 0,nxt = 0;
        for(int i=0;i<len;i++,tmp=nxt){
            nxt = trie[tmp][sw[s[i]]];
            if(nxt==-1){
                top++;
                clr(trie[top],-1);
                trie[tmp][sw[s[i]]] = nxt =top;
            }
        }
        trie[tmp][ch] = 1;
    }
    int count(int rt){
        int sum = 0;
        for(int i=0;i<ch;i++){
            if(trie[rt][i]!=-1)
                sum += count(trie[rt][i]);
        }
        if(trie[rt][ch]!=-1) sum++;
        return sum ;
    }
}t;

int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    while(scanf("%d",&n)!=EOF){
        t.init();
        for(int i=0;i<n;i++){
            sam.init();
            scanf("%s",s);
            m = strlen(s);
            for(int i=0;i<m;i++) sam.add(s[i],i+1);
            for(int i=m;i<2*m;i++) sam.add(s[i-m],i+1);
            sam.go(s,m);
            t.insert(s,m);
        }
        printf("%d\n",t.count(0));
    }
    return 0;
}
```

spoj NSUBSTR
题意：求一个串中长度为i的串最多出现的次数。
记录自动机每个节点到达的次数，利用parent树来做。然后利用自动机的性质得到每个节点最长到达的串，即构造时的len,然后从大到小更新一遍。
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
const int N=510000;
int k;

class SAM{
    public:
    int f[N],chd[N][ch+1],len[N],sw[200];
    vector<int> p[N];

    int last,sz;
    void init(){
        clr(chd[0],0);
        f[0] = -1;
        last = 0;
        sz = 0;
        for(int i=0;i<26;i++)
            sw[i+'a'] = i;
    }
    void add(char inc,int l){
        int c = sw[inc];
        int x = last;
        last = ++sz;
        clr(chd[sz],0);
        len[sz] = l;

        chd[last][ch]++;

        for(;x!=-1&&chd[x][c]==0;x=f[x]){
            chd[x][c] = sz;
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

                f[sz] = f[y];
                f[last] = f[y] = sz;

                for(;x!=-1&&chd[x][c]==y;x=f[x]){
                    chd[x][c] = sz;
                }
            }
        }
        return ;
    }
    void parent_tree(){
        for(int i=0;i<=sz;i++) p[i].clear();
        for(int v=1;v<=sz;v++){
            int u = f[v];
            p[u].push_back(v);
        }
        dfs_tree(0);
    }
    void dfs_tree(int u){
        for(int i=0;i<p[u].size();i++){
            int v = p[u][i];
            dfs_tree(v);
            chd[u][ch] += chd[v][ch];
        }
    }
}sam;

char s[1000000];
int a[1000000];
int n,m;


int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    scanf("%s",s);
    sam.init();
    n = strlen(s);
    for(int i=0;i<n;i++)
        sam.add(s[i],i+1);
    sam.parent_tree();
    for(int i=1;i<=sam.sz;i++){
        int l = sam.len[i];
        int w = sam.chd[i][ch];
        a[l] = max(a[l],w);
    }
    for(int i=n;i>=1;i--) a[i] = max(a[i+1],a[i]);
    for(int i=1;i<=n;i++)
        printf("%d\n",a[i]);

    return 0;
}
```

spoj SUBLEX
题意：询问一个串中第k小的不同的子串。
关键要得到自动机每个节点后面能够有多少种不同走法，有向无环图做下dp。然后走一遍自动机。
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
const int N=200000;
char s[1000000];
int k;

class SAM{
    public:
    int f[N],chd[N][ch],len[N],sw[200];
    int q[N][ch],l[N];
    ll dp[N];
    bool vis[N];

    int last,sz;
    void init(){
        clr(vis,0);
        clr(dp,0);
        clr(chd[0],0);
        f[0] = -1;
        last = 0;
        sz = 0;
        for(int i=0;i<26;i++)
            sw[i+'a'] = i;
    }
    void add(char inc,int l){
        int c = sw[inc];
        int x = last;
        last = ++sz;
        clr(chd[sz],0);
        len[sz] = l;

        for(;x!=-1&&chd[x][c]==0;x=f[x]){
            chd[x][c] = sz;
        }
        if(x == -1) f[sz]=0;
        else{
            int y = chd[x][c];
            if(len[y] == len[x]+1) f[sz]=y;
            else{
                sz++;
                len[sz] = len[x] + 1;
                for(int i=0;i<ch;i++) chd[sz][i] = chd[y][i];

                f[sz] = f[y];
                f[last] = f[y] = sz;

                for(;x!=-1&&chd[x][c]==y;x=f[x]){
                    chd[x][c] = sz;
                }
            }
        }
        return ;
    }
    void dfs(int u){
        if(vis[u]) return ;
        vis[u] = 1;
        l[u] = 0;
        for(int i=0;i<ch;i++){
            int v = chd[u][i];
            if(v!=0){
                dfs(v);
                q[u][l[u]] = i;
                l[u] ++;
                dp[u] += dp[v];
            }
        }
        dp[u]++;
    }
    void go(ll k){
        int rt = 0,length = 0;
        while(k){
            if(rt!=0) k--;
            if(k<=0) break;
            for(int i=0;i<l[rt];i++){
                int nc = q[rt][i];
                int nxt = chd[rt][nc];
                if(dp[nxt]>=k){
                    rt = nxt ;
                    s[length++] = nc+'a';
                    break;
                }
                else
                    k -= dp[nxt];
            }
        }
        s[length] = 0;
        puts(s);
    }
}sam;

int n,m;


int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    scanf("%s",s);
    sam.init();
    n = strlen(s);
    for(int i=0;i<n;i++)
        sam.add(s[i],i+1);
    sam.dfs(0);
    int Q;
    scanf("%d",&Q);
    while(Q--){
        ll k;
        scanf("%lld",&k);
        sam.go(k);
    }

    return 0;
}



spoj LCS
spoj LCS2
这两题比较类似，求n个串的最长公共子串。
先以第一个串构造自动机，然后每个串都在自动机上走一遍，维护每个节点所能匹配的最长的串，对于所有串每个节点取个匹配的最小值，在对所有节点取个最大值即可。
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
using namespace std;

const int N = 1000000,ch = 26;
char s[1000000];
const int inf = 1e8;

struct SAM{
    public:
    int chd[N][ch],f[N],sw[200],len[N];
    int val[N],tmp[N];
    vector<int> p[N];
    int sz,last;
    void init(){
        clr(chd[0],0);
        sz = 0;
        last = 0;
        f[0] = -1;
        for(int i=0;i<ch;i++)
            sw[i+'a'] = i;
    }
    void add(char tc,int l){
        int c = sw[(int)tc];
        int x = last;
        last = ++sz;
        len[last] = l;
        clr(chd[sz],0);

        for(;x!=-1&&chd[x][c]==0;x=f[x]) chd[x][c] = last;

        if(x==-1) f[last] = 0;
        else{
            int y = chd[x][c];
            if(len[y] == len[x] + 1) f[last] = y;
            else{
                sz ++;
                memcpy(chd[sz],chd[y],ch*4);
//                for(int i=0;i<ch;i++) chd[sz][i] = chd[y][i];
                len[sz] = len[x] + 1;

                f[sz] = f[y];
                f[y] = f[last] =sz;

                for(;x!=-1 && chd[x][c]==y;x=f[x]) chd[x][c] = sz;
            }
        }
    }
    void build(){
        for(int i=0;i<=sz;i++) p[i].clear();
        for(int i=0;i<=sz;i++)
            p[f[i]].push_back(i);
    }

    void go(char s[],int n){
        for(int i=0;i<=sz;i++) tmp[i] = 0;
        int rt = 0,l=0;
        for(int i=0;i<n;i++){
            int c = sw[(int)s[i]];
            if(chd[rt][c]){
                l++;
                rt = chd[rt][c];
            }else{
                while(rt!=-1&&chd[rt][c]==0) rt = f[rt];
                if(rt==-1){
                    rt = 0;
                    l = 0;
                }
                else{
                    l = len[rt] + 1;
                    rt = chd[rt][c];
                }
            }
            if(l>tmp[rt]) tmp[rt] = l;
        }
        dfs(0);
        for(int i=0;i<=sz;i++){
            if(tmp[i] > len[i]) tmp[i] = len[i];
            if(tmp[i]<val[i])
                val[i] = tmp[i];
        }
    }

    __inline void dfs(int u){
        int n = p[u].size();
        for(int i=0;i<n;i++){
            int v = p[u][i];
            dfs(v);
            if(tmp[v]>tmp[u]) tmp[u] = tmp[v];
        }
    }

    int get(){
        int ans = 0;
        for(int i=0;i<=sz;i++) ans = max(ans,val[i]);
        return ans;
    }
}sam;


int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    int cnt = 0 ;

    sam.init();
    scanf("%s",s);
    int n = strlen(s);
    for(int i=0;i<n;i++) sam.add(s[i],i+1);

    sam.build();

    for(int i=0;i<=sam.sz;i++) sam.val[i] = inf;
    while(scanf("%s",s)!=EOF){
        int n = strlen(s);
        sam.go(s,n);
        cnt ++;
    }
    if(cnt==0){
        printf("%d\n",n);
        return 0;
    }
    printf("%d\n",sam.get());
    return 0;
}
```

bzoj 2806
这题比较综合
就说一下后缀自动机的部分：要得到一个串在某个位置往前与其他模版串所能匹配的最长长度。把模版串用'$'号串起来，建立后缀自动机。然后一开始的串走一遍就可以了。

/**************************************************************
    Problem: 2806
    User: _LT_zyc
    Language: C++
    Result: Accepted
    Time:1232 ms
    Memory:79788 kb
****************************************************************/
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
 
const int ch = 3;
const int N=3000000;
class SAM{
    public:
    int f[N],chd[N][ch],len[N],sw[200];
    //ch+1 number of tempt suffix at status
    int last,sz;
    void init(){
        clr(chd[0],0);
        f[0] = -1;
        last = 0;
        sz = 0;
        sw['0'] = 0;
        sw['1'] = 1;
        sw['2'] = 2;
    }
    void add(char inc,int l){
        int c = sw[(int)inc];
        int x = last;
        last = ++sz;
        clr(chd[sz],0);
        len[sz] = l;
        for(;~x&&chd[x][c]==0;x=f[x]){
            chd[x][c] = sz;
        }
        if(x == -1) f[sz]=0;
        else{
            int y = chd[x][c];
            if(len[y] == len[x]+1)  f[sz]=y;
            else{
                sz++;
                len[sz] = len[x] + 1;
 
                memcpy(chd[sz],chd[y],sizeof(chd[sz]));
 
                f[sz] = f[y];
                f[last] = f[y] = sz;
 
                for(;~x&&chd[x][c]==y;x=f[x]){
                    chd[x][c] = sz;
                }
            }
        }
    }
    void go(char s[],int n,int a[]){
        int rt = 0,l=0;
        for(int i=0;i<n;i++){
            int c = sw[(int)s[i]];
            if(chd[rt][c]){
                l++;
                rt=chd[rt][c];
            }else{
                while(~rt&&chd[rt][c]==0) rt=f[rt];
 
                if(rt==-1){
                    rt=0;
                    l=0;
                }
                else {
                    l=len[rt]+1;
                    rt=chd[rt][c];
                }
            }
            a[i] = l;
        }
    }
}sam;
 
char s[1200000];
int n,m,a[1200000];
int dp[1200000],len;
int q[1200000],qid[1200000];
 
bool get(int gap){
    int l=0,r=0;
    for(int i=0;i<len;i++){
        if(i!=0) dp[i] = dp[i-1];
 
        int x = i-a[i];
        while(r>l&&qid[l]<x) l++;
 
        if(a[i]>=gap){
            dp[i] = max(dp[i],a[i]);
            if(r>l) dp[i] = max(dp[i],q[l]+i);
        }
 
        x = i-gap+1;
        if(x>=0){
            int w= dp[x]-x;
            while(r>l&&w>=q[r-1]) r--;
            qid[r] = x;
            q[r++] = w;
        }
        if(dp[i]*10 >= len*9) return 1;
    }
    return 0;
}
int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    sam.init();
    scanf("%d%d",&n,&m);
    for(int i=0;i<m;i++){
        if(i!=0)
            sam.add('2',++len);
        scanf("%s",s);
        int l = strlen(s);
        for(int j=0;j<l;j++)
            sam.add(s[j],++len);
    }
    while(n--){
        scanf("%s",s);
        len = strlen(s);
        sam.go(s,len,a);
        int l=0,r=len;
        while(r!=l){
            int mid = (l+r)/2+1;
            if(get(mid)) l = mid;
            else r = mid-1;
        }
        printf("%d\n",r);
    }
    return 0;
}
```

hdu 4436
题意：给你多个由0~9组成的字符串，问其中所有子串代表的不同的数字的和。
位数高的在前面，所以先把串反过来，让位数高的后加进自动机。
多个串要判重所以先用'$'连接起来，建立自动机。
要得到每个节点后面的其他节点的值以及出现次数，然后当前节点的值就是 dp[u]  = dp[v] *10 + i*p[v];(i为转移的数字，p[v]为出现次数)，还是dag上的dp
注意要把i=0的情况处理好。
答案就是dp[0]。
```c++
#pragma comment(linker, "/STACK:65536000")
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

const int N = 500000;
const int ch = 11;
const int mod = 2012;
class SAM{
    public :
    int chd[N][ch],len[N],sw[200],f[N];
    int sz,last,dp[N],vis[N],p[N];

    void init(){
        clr(chd,0);
        sz = 0;
        last = 0;
        len[0] = 0;
        f[0] = -1;
        sw['$'] = 10;
        for(int i=0;i<10;i++)
            sw[i+'0'] = i;
    }

    void add(char tc,int l){
        int c = sw[(int)tc];
        int x = last;
        last = ++sz;
        clr(chd[sz],0);
        len[sz] = l;

        for(;~x&&chd[x][c]==0;x=f[x]) chd[x][c] = sz;
        if(x==-1) f[sz] = 0;
        else{
            int y = chd[x][c];
            if(len[y] == len[x] + 1) f[sz] = y;
            else{
                sz++;
                memcpy(chd[sz],chd[y],sizeof(chd[sz]));
                len[sz] = len[x] + 1;

                f[sz] = f[y];
                f[y] = f[last] =sz;
                for(;~x&&chd[x][c]==y;x=f[x]) chd[x][c] = sz;
            }
        }
    }
    void dfs(int u){
        if(vis[u]) return ;
        vis[u] = 1;
        for(int i=0;i<ch-1;i++){
            int v = chd[u][i];
            if(v){
                dfs(v);
                if(i==0){
                    p[u]+=p[v];
                    if(p[u]>=mod) p[u]%=mod;
                    dp[u] += dp[v] * 10 +i*p[v];
                }else{
                    p[u]+=p[v]+1;
                    if(p[u]>=mod) p[u]%=mod;
                    dp[u] += dp[v] * 10 +i*(p[v]+1);
                }
                if(dp[u]>=mod) dp[u] %= mod;
            }
        }
    }
    int go(){
        for(int i=0;i<=sz;i++) p[i] = dp[i] =vis[i]= 0;
        dfs(0);
        return dp[0];
    }
}sam;

int n;
char s[200000];
int main(){
//    freopen("/home/zyc/Documents/Code/cpp/in","r",stdin);
    while(scanf("%d",&n)!=EOF){
        int ans = 0;
        int len = 0;
        sam.init();
        for(int i=0;i<n;i++){
            scanf("%s",s);
            int m = strlen(s);
            for(int j=0;j<m/2;j++) swap(s[j],s[m-j-1]);
            if(i!=0) sam.add('$',++len);
            for(int j=0;j<m;j++) sam.add(s[j],++len);
        }
        ans += sam.go();
        printf("%d\n",ans%mod);
    }
    return 0;
}
```

spoj COT4
看了一眼别人的做法，只能说我没打过ACM。有空在写。。。。