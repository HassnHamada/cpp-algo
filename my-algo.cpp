// --------------------kmp-------------------
const int N = 1e6 + 10;
char str[N];
int pi[N];

int newL(int i, char c)
{
    while (i && str[i] != c)
    {
        i = pi[i - 1];
    }
    return i + (int)(str[i] == c);
}

void kmp()
{
    for (int i = 1; str[i]; i++)
    {
        pi[i] = newL(pi[i - 1], str[i]);
    }
}

// ---------------z algorithm-----------------
const int N = 1e6 + 10;
char str[N];
int zzz[N];

void computeZ()
{
    int &i = zzz[0];
    int l = 0, r = 0;
    for (i = 1; str[i]; i++)
    {
        if (i >= r)
        {
            l = r = i;
            zzz[i] = 0;
        }
        else
        {
            zzz[i] = zzz[i - l];
        }
        if (i + zzz[i] >= r)
        {
            l = i;
            while (str[r] == str[r - l])
            {
                r++;
            }
            zzz[i] = r - l;
        }
    }
}

// -----------------manacher-----------------
// get the length of every palindrome in O(n)

const int N = 1e6 + 10;
char str[N];
int ppp[N];

void extend()
{
    int n = strlen(str);
    for (int i = (n + 1) * 2; i; i -= 2)
    {
        str[i] = str[n--];
        str[i - 1] = '#';
    }
    str[0] = '$';
}

void shrink()
{
    int i = 0;
    for (int j = 2; str[j]; j += 2)
    {
        str[i++] = str[j];
    }
    str[i] = '\0';
}

void manacher()
{
    extend();
    int n = strlen(str), c = 1, r = 1;
    memset(ppp, 0, n * sizeof(ppp[0]));
    for (int i = 2; i < n; i++)
    {
        int m = 2 * c - i;
        if (i < r)
        {
            ppp[i] = min(ppp[m], r - i);
        }
        while (str[i + ppp[i] + 1] == str[i - ppp[i] - 1])
        {
            ppp[i]++;
        }
        if (i + ppp[i] > r)
        {
            c = i;
            r = i + ppp[i];
        }
    }
}

// -------------------lca--------------------

const int N = 1e6 + 10, M = 20;
int lvl[N], per[N][M];
vector<int> tre[N];

void dfs(int n = 1, int p = 0)
{
    lvl[n] = lvl[p] + 1;
    per[n][0] = p;
    for (auto &&i : tre[n])
    {
        if (i == p)
        {
            continue;
        }
        dfs(i, n);
    }
}

int lca(int a, int b)
{
    if (lvl[b] > lvl[a])
    {
        swap(a, b);
    }
    for (int i = M - 1; ~i && lvl[a] > lvl[b]; i--)
    {
        if (per[a][i] == -1 || lvl[per[a][i]] < lvl[b])
        {
            continue;
        }
        a = per[a][i];
    }
    assert(lvl[a] == lvl[b]);
    if (a == b)
    {
        return a;
    }
    for (int i = M - 1; ~i; i--)
    {
        if (per[a][i] == per[b][i])
        {
            continue;
        }
        a = per[a][i];
        b = per[b][i];
    }
    return per[a][0] == per[b][0] ? per[a][0] : -1;
}

void main()
{
    dfs();
    for (int i = 1; i < M; i++)
    {
        for (int j = 1; j <= n; j++)
        {
            per[j][i] = per[per[j][i - 1]][i - 1];
        }
    }
}

// ---------------sparse table---------------

const int N = 1e6 + 10, M = 20;
int st[N][M], lg[N], n;

// O(logn)
int getsum(int s, int e)
{
    int ret = 0;
    for (int i = M - 1; i >= 0; i--)
    {
        if ((1 << i) <= e - s + 1)
        {
            ret += st[s][i];
            s += 1 << i;
        }
    }
}

// O(1)
int getmax(int s, int e)
{
    int i = lg[e - s + 1];
    return max(st[s][i], st[e - (1 << i) + 1][i]);
}

void sparsetable()
{
    for (int i = 2; i <= n; i++)
    {
        lg[i] = lg[i / 2] + 1;
    }
    for (int i = 1; i <= lg[n]; i++)
    {
        for (int j = 0; j + (1 << (i - 1)) < n; j++)
        {

            // Do something
            // max
            // st[j][i] = max(st[j][i - 1], st[j + (1 << (i - 1))][i - 1]);
            // sum
            // st[j][i] = st[j][i - 1] + st[j + (1 << (i - 1))][i - 1];
        }
    }
}

// ---------------bit & 2d-bit---------------

const int N = 1e6 + 10, M = 20;
int tre[N], tree[M][N];

void add(int pos, int val)
{
    for (++pos; pos <= N; pos += (pos & (-pos)))
    {
        tre[pos - 1] += val;
    }
}

int get(int pos)
{
    int ret = 0;
    for (++pos; pos; pos -= (pos & (-pos)))
    {
        ret += tre[pos - 1];
    }
    return ret;
}

void add(int i, int j, ll val)
{
    for (i++; i <= M; i += (i & (-i)))
    {
        for (int jj = j + 1; jj <= N; jj += (jj & (-jj)))
        {
            tree[i - 1][jj - 1] += val;
        }
    }
}

ll get(int i, int j)
{
    ll ret = 0;
    for (i++; i; i -= (i & (-i)))
    {
        for (int jj = j + 1; jj; jj -= (jj & (-jj)))
        {
            ret += tree[i - 1][jj - 1];
        }
    }
    return ret;
}

ll get(int si, int sj, int ei, int ej)
{
    return get(ei, ej) - get(ei, sj - 1) - get(si - 1, ej) + get(si - 1, sj - 1);
}

// ----------------mo algorithm---------------

const int N = 1e6 + 10;
int arr[N];

void add(int val)
{
    // add element
}
void del(int val)
{
    // remove element
}

int s, e;

void updateSE(int qs, int qe)
{
    while (e < qe + 1)
    {
        add(arr[e++]);
    }
    while (s > qs)
    {
        add(arr[--s]);
    }
    while (s < qs)
    {
        del(arr[s++]);
    }
    while (e > qe + 1)
    {
        del(arr[--e]);
    }
}

// ----segment tree with lazy propagation----

const int N = 1 << 20;
int arr[N], tre[N << 1], lz[N << 1];
int n;

void build(int i = 0, int s = 0, int e = n - 1)
{
    if (s == e)
    {
        tre[i] = arr[s];
        return;
    }
    int l = 2 * i + 1, r = l + 1, m = (e - s) / 2 + s;
    build(l, s, m);
    build(r, m + 1, e);
    tre[i] = tre[l] + tre[r];
}

void pd(int i, int s, int e)
{
    assert(s != e);
    int l = 2 * i + 1, r = l + 1, m = (e - s) / 2 + s;
    tre[l] = lz[i] * (m - s + 1);
    tre[r] = lz[i] * (e - m);
    if (s < m)
    {
        lz[l] = lz[i];
    }
    if (m + 1 < e)
    {
        lz[r] = lz[i];
    }
    lz[i] = -1;
}

void add(int qs, int qe, int qv, int i = 0, int s = 0, int e = n - 1)
{
    if (s > qe || e < qs)
    {
        return;
    }
    if (s >= qs && e <= qe)
    {
        // update node
        tre[i] = qv * (e - s + 1);
        // add to lazy
        lz[i] = qv;
        return;
    }
    if (~lz[i])
    {
        pd(i, s, e);
    }
    int l = 2 * i + 1, r = l + 1, m = (e - s) / 2 + s;
    add(qs, qe, l, s, m);
    add(qs, qe, r, m + 1, e);
    tre[i] = tre[l] + tre[r];
}

int get(int qs, int qe, int i = 0, int s = 0, int e = n - 1)
{
    if (s > qe || e < qs)
    {
        return 0;
    }
    if (s >= qs && e <= qe)
    {
        return tre[i];
    }
    if (~lz[i])
    {
        pd(i, s, e);
    }
    int l = 2 * i + 1, r = l + 1, m = (e - s) / 2 + s;
    return get(qs, qe, l, s, m) + get(qs, qe, r, m + 1, e);
}

// -----------matrix exponentiation-----------

const int MOD = 1e9 + 7;

struct Mat
{
    int x, y;
    vector<vector<int>> mat;
    Mat(int _x, int _y)
    {
        x = _x;
        y = _y;
        mat = vector<vector<int>>(x, vector<int>(y, 0));
    }

    Mat(int _x, int _y, vector<int> vec)
    {
        assert(_x * _y == (int)vec.size());
        x = _x;
        y = _y;
        mat = vector<vector<int>>(x, vector<int>(y, 0));
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                mat[i][j] = vec[i * x + j];
            }
        }
    }
    Mat operator*(const Mat &other) const
    {
        assert(this->y == other.x);
        Mat ret(this->x, other.y);
        for (int i = 0; i < ret.x; i++)
        {
            for (int j = 0; j < ret.y; j++)
            {
                for (int k = 0; k < y; k++)
                {
                    ret.mat[i][j] = (ret.mat[i][j] + 1ll * mat[i][k] * other.mat[k][j] % MOD) % MOD;
                }
            }
        }
        return ret;
    }
    void identity()
    {
        assert(x == y);
        for (int i = 0; i < x; i++)
        {
            for (int j = 0; j < y; j++)
            {
                mat[i][j] = i == j;
            }
        }
    }
};

Mat mat_pow(Mat b, int p)
{
    assert(b.x == b.y);
    Mat ret(b.x, b.y);
    ret.identity();
    while (p)
    {
        if (p & 1)
        {
            ret = ret * b;
        }
        b = b * b;
        p /= 2;
    }
    return ret;
}

// fibonacci O(logn) using matrices
int fib(int a, int b, int p)
{
    Mat i(1, 2, {a, b});
    Mat t(2, 2, {0, 1, 1, 1});
    return (i * mat_pow(t, p)).mat[0][0];
}

// ---------------hashing--------------------

// HV 151, 157, 163, 167, 173

ll my_pow(ll b, ll p, ll m)
{
    ll r = 1;
    while (p)
    {
        if (p & 1)
        {
            r = r * b % m;
        }
        b = b * b % m;
        p = p / 2;
    }
    return r;
}

ll inv(ll a, ll mod)
{
    return my_pow(a, mod - 2, mod);
}

struct SHA
{
    ll v, b, m, s;
    void hashBack(char l)
    {
        v = (v * b + l) % m;
        s += 1;
    }
    void hashFront(char l)
    {
        v = (v + l * my_pow(b, s, m)) % m;
        s += 1;
    }
    void unhashBack(char l)
    {
        v = (v - l + m) * inv(b, m) % m;
        s -= 1;
    }
    void unhashFront(char l)
    {
        s -= 1;
        v = (v - l * my_pow(b, s, m) + m) % m;
    }
    void extend(SHA &other)
    {
        assert(this->m == other.m && this->b == other.b);
        v = (v * my_pow(b, other.s, m) + other.v) % m;
        s += other.s;
    }
};

// ---------------xor-trie-------------------

const int M = 2, L = 30;

struct Node
{
    int nxt[M];
    int lev, prv;
    Node()
    {
        memset(nxt, -1, sizeof(nxt));
        lev = 0;
        prv = -1;
    }
};
vector<Node> trie;

void add(int n)
{
    int c = 0;
    for (int i = L - 1; ~i; i--)
    {
        int v = (n >> i) & 1;
        if (trie[c].nxt[v] == -1)
        {
            trie[c].nxt[v] = trie.size();
            trie.emplace_back();
            trie.back().prv = c;
        }
        c = trie[c].nxt[v];
    }
    trie[c].lev += 1;
}

void del(int n)
{
    int c = 0;
    for (int i = L - 1; ~i; i--)
    {
        int v = (n >> i) & 1;
        c = trie[c].nxt[v];
    }
    trie[c].lev -= 1;
    if (trie[c].lev == 0)
    {
        int nxt = -1;
        while (trie[c].nxt[0] == -1 || trie[c].nxt[1] == -1)
        {
            nxt = c;
            c = trie[c].prv;
        }
        if (~trie[c].nxt[0] && trie[c].nxt[0] == nxt)
        {
            trie[c].nxt[0] = -1;
        }
        if (~trie[c].nxt[1] && trie[c].nxt[1] == nxt)
        {
            trie[c].nxt[1] = -1;
        }
    }
}

int get(int n)
{
    int c = 0, ret = 0;
    for (int i = L - 1; ~i; i--)
    {
        int v = (n >> i) & 1;
        ret <<= 1;
        if (trie[c].nxt[v] == -1)
        {
            ret |= (v ^ 1);
            c = trie[c].nxt[v ^ 1];
        }
        else
        {
            ret |= v;
            c = trie[c].nxt[v];
        }
    }
    return ret;
}

// -------------line intersection------------

struct Line
{
    ll x1, y1, x2, y2;
    bool inter(Line &l)
    {
        ll v1 = X(l.x1 - x1, l.y1 - y1, l.x2 - x1, l.y2 - y1),
           v2 = X(l.x1 - x2, l.y1 - y2, l.x2 - x2, l.y2 - y2),
           v3 = X(x1 - l.x1, y1 - l.y1, x2 - l.x1, y2 - l.y1),
           v4 = X(x1 - l.x2, y1 - l.y2, x2 - l.x2, y2 - l.y2);
        if (v1 && v2 && v3 && v4)
        {
            return ((v1 > 0 && v2 < 0) || (v1 < 0 && v2 > 0)) &&
                   ((v3 > 0 && v4 < 0) || (v3 < 0 && v4 > 0));
        }
        if (v1 == 0)
        {
            Line a{x1, y1, l.x1, l.y1}, b{x1, y1, l.x2, l.y2};
            if (max(a.len(), b.len()) < l.len())
            {
                return true;
            }
        }
        if (v2 == 0)
        {
            Line a{x2, y2, l.x1, l.y1}, b{x2, y2, l.x2, l.y2};
            if (max(a.len(), b.len()) < l.len())
            {
                return true;
            }
        }
        if (v3 == 0)
        {
            Line a{l.x1, l.y1, x1, y1}, b{l.x1, l.y1, x2, y2};
            if (max(a.len(), b.len()) < len())
            {
                return true;
            }
        }
        if (v4 == 0)
        {
            Line a{l.x2, l.y2, x1, y1}, b{l.x2, l.y2, x2, y2};
            if (max(a.len(), b.len()) < len())
            {
                return true;
            }
        }
        return false;
    }
    ll X(ll a, ll b, ll c, ll d)
    {
        return a * d - b * c;
    }
    ll len()
    {
        return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    }
};

// -----------------sieve--------------------

const int N = 1e6 + 10;

bitset<N> prime;
vector<int> vec;

void fastSieve(int n = N)
{
    prime.set();
    for (ll i = 3; i * i <= n; i += 2)
    {
        for (ll j = i * i; prime[i / 2] && j <= n; j += 2 * i)
        {
            prime[j / 2] = 0;
        }
    }
    vec.push_back(2);
    for (int i = 3; i <= n; i += 2)
    {
        if (prime[i / 2])
        {
            vec.push_back(i);
        }
    }
}

// -----------bipartite matching-------------

const int N = 1e2 + 10;

int nxtn[N], nxtm[N];
bool vis[N];
int n, m;

bool canCon(int ii)
{
    if (vis[ii])
    {
        return false;
    }
    vis[ii] = true;
    // loop over the graph
    for (int j = 0; j < m; j++)
    {
        if (isCon(ii, j) && (nxtm[j] == -1 || canCon(nxtm[j])))
        {
            nxtm[j] = ii;
            nxtn[ii] = j;
            return true;
        }
    }
    return false;
}

void main()
{
    memset(nxtn, -1, sizeof(nxtn[0]) * n);
    memset(nxtm, -1, sizeof(nxtm[0]) * m);
    vector<int> ans;
    for (int i = 0; i < n; i++)
    {
        memset(vis, 0, sizeof(vis[0]) * n);
        if (nxtn[i] == -1 && canCon(i))
        {
            ans.push_back(i);
        }
    }
}

// -----------Aho-Corasick algorithm---------

const int N = 1e5 + 10, M = 128;
char str[N], tem[N];
struct Node
{
    const int p;
    const char pch;
    int nxt[M], go[M], link = -1;
    int leaf = -1;
    Node(int p = 0, char pch = '$') : p(p), pch(pch)
    {
        memset(nxt, -1, sizeof(nxt));
        memset(go, -1, sizeof(go));
    }
};

vector<Node> tri;
bitset<N> bs;
int ids[N], nid;

int insert(char *ptr)
{
    int v = 0;
    for (int i = 0; *(ptr + i); i++)
    {
        int c = *(ptr + i);
        if (tri[v].nxt[c] == -1)
        {
            tri[v].nxt[c] = tri.size();
            tri.emplace_back(v, c);
        }
        v = tri[v].nxt[c];
    }
    return tri[v].leaf == -1 ? tri[v].leaf = nid++ : tri[v].leaf;
}

int go(int, int);

int get_link(int v)
{
    if (tri[v].link == -1)
    {
        if (v == 0 || tri[v].p == 0)
        {
            tri[v].link = 0;
        }
        else
        {
            tri[v].link = go(get_link(tri[v].p), tri[v].pch);
        }
    }
    return tri[v].link;
}

int go(int v, int c)
{
    if (tri[v].go[c] == -1)
    {
        if (tri[v].nxt[c] == -1)
        {
            tri[v].go[c] = v ? go(get_link(v), c) : 0;
        }
        else
        {
            tri[v].go[c] = tri[v].nxt[c];
        }
    }
    return tri[v].go[c];
}

void match(char *ptr)
{
    int v = 0;
    for (int i = 0; *(ptr + i); i++)
    {
        int c = *(ptr + i);
        while (v && tri[v].nxt[c] == -1)
        {
            v = get_link(v);
        }
        if (tri[v].nxt[c] != -1)
        {
            v = tri[v].nxt[c];
        }
        if (tri[v].leaf != -1)
        {
            bs[tri[v].leaf] = 1;
        }
    }
}

void main()
{
    tri.emplace_back();
    // add strings dictionary
    int q;
    scanf("%d", &q);
    for (int i = 0; i < q; i++)
    {
        scanf("%s", tem);
        ids[i] = insert(tem);
    }
    // match any of them
    scanf("%s", str);
    match(str);
    // print matched
    for (int i = 0; i < q; i++)
    {
        printf(bs[ids[i]] ? "y\n" : "n\n");
    }
}

// -----------------tarjan-------------------

const int N = 2e5 + 10, EMP = -1;

int n, m, id, cid, ids[N], low[N], cyc[N];
bool onStk[N];
vector<int> grf[N];
stack<int> stk;

void dfs(int c)
{
    low[c] = ids[c] = id++;
    onStk[c] = true;
    stk.push(c);
    for (auto &&i : grf[c])
    {
        if (ids[i] == EMP)
        {
            dfs(i);
        }
        if (onStk[i])
        {
            low[c] = min(low[c], low[i]);
        }
    }
    if (ids[c] == low[c])
    {
        cyc[c] = ++cid;
        while (!stk.empty())
        {
            int t = stk.top();
            stk.pop();
            cyc[t] = cyc[c];
            onStk[t] = false;
            low[t] = low[c];
            if (t == c)
            {
                break;
            }
        }
    }
}

void tarjan()
{
    for (int i = 1; i <= n; i++)
    {
        ids[i] = EMP;
    }
    for (int i = 1; i <= n; i++)
    {
        if (ids[i] == EMP)
        {
            dfs(i);
        }
    }
}

// ------bridges & articulation points-------

int n;
vector<vector<int>> adj;
vector<bool> visited;
vector<int> tin, low;
int timer;

void dfs(int v, int p = -1)
{
    visited[v] = true;
    tin[v] = low[v] = timer++;
    int children = 0;
    for (int to : adj[v])
    {
        if (to == p)
        {
            continue;
        }
        if (visited[to])
        {
            low[v] = min(low[v], tin[to]);
        }
        else
        {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] > tin[v])
            {
                // IS_BRIDGE(v, to);
            }
            if (low[to] >= tin[v] && p != -1)
            {
                // IS_CUTPOINT(v);
            }
            ++children;
        }
    }
    if (p == -1 && children > 1)
    {
        // IS_CUTPOINT(v);
    }
}

void find()
{
    timer = 0;
    visited.assign(n, false);
    tin.assign(n, -1);
    low.assign(n, -1);
    for (int i = 0; i < n; ++i)
    {
        if (!visited[i])
        {
            dfs(i);
        }
    }
}

// ---------------suffix array---------------

struct Suffix
{
    int i;
    pair<int, int> r;
    bool operator<(const Suffix &other) const
    {
        if (this->r.first == other.r.first)
        {
            return this->r.second < other.r.second;
        }
        return this->r.first < other.r.first;
    }
} arr[N];

char str[N];

void calcSuffix()
{
    int n = strlen(str);
    for (int i = 0; i < n; i++)
    {
        arr[i].i = i;
        arr[i].r.first = str[i] - 'a';
        arr[i].r.second = i + 1 < n ? str[i + 1] - 'a' : -1;
    }
    sort(arr, arr + n);
    vector<int> ind(n);
    for (int i = 2; i < n; i *= 2)
    {
        pair<int, int> prv = arr[0].r;
        arr[0].r.first = ind[arr[0].i] = 0;
        for (int j = 1; j < n; j++)
        {
            ind[arr[j].i] = j;
            if (arr[j].r == prv)
            {
                arr[j].r.first = arr[j - 1].r.first;
            }
            else
            {
                prv = arr[j].r;
                arr[j].r.first = arr[j - 1].r.first + 1;
            }
        }
        for (int j = 0; j < n; j++)
        {
            arr[j].r.second = arr[j].i + i < n ? arr[ind[arr[j].i + i]].r.first : -1;
        }
        sort(arr, arr + n);
    }
}

// ---------------factorial------------------

const int N = 1e5 + 10, MOD = 1e9 + 7;
ll fact[N], ifact[N], inv[N];

void factPre()
{
    fact[0] = fact[1] = ifact[0] = ifact[1] = inv[0] = inv[1] = 1;
    for (int i = 2; i < N; i++)
    {
        fact[i] = fact[i - 1] * i % MOD;
        inv[i] = (MOD - (MOD / i) * inv[MOD % i] % MOD) % MOD;
        ifact[i] = ifact[i - 1] * inv[i] % MOD;
    }
}

// ----------------DSU-----------------------

const int N = 1e5 + 10;

struct DSU
{
    int per[N];
    void init(int n)
    {
        iota(per, per + n, 0);
    }
    int find(int a)
    {
        return per[a] == a ? a : per[a] = find(per[a]);
    }
    void merge(int a, int b)
    {
        per[find(b)] = find(a);
    }
    bool isCon(int a, int b)
    {
        return find(a) == find(b);
    }
};