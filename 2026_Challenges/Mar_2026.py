###############################################
# 1536. Minimum Swaps to Arrange a Binary Grid
# 02MAR26
################################################
class Solution:
    def minSwaps(self, grid: List[List[int]]) -> int:
        '''
        grid is square
        we can only swap to adjacent rows
        for i in range(1,n):
            count_zeros_at_col(i) >= 1, there can be zeros below main daig, but need to make zeros above main diag
        count zeros for each row to the right, until we hit a wone
        '''
        n = len(grid)
        count_zeros = [0]*n

        for i in range(n):
            for j in range(n-1,-1,-1):
                if grid[i][j] == 0:
                    count_zeros[i] += 1
                else:
                    break
        
        #each row i, should have at least n-i zeros in ot
        #if it does, we are good, otherwise, find the the closest row to it with at least n-i zeros to the right
        ans = 0
        print(count_zeros)
        for i in range(n):
            j = i
            while j < n and count_zeros[j] < n - i - 1:
                j += 1
            if j == n:
                return -1
            ans += j - i
            #swap back up to i
            while j > i:
                count_zeros[j],count_zeros[j-1] = count_zeros[j-1],count_zeros[j]
                j -= 1
        
        return ans

################################################
# 3129. Find All Possible Stable Binary Arrays I
# 09MAR26
#################################################
#close
#MLE
#need to remove limit state
class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        '''
        length of the array cannot exceed zero + one
        there must be zero count_zeros and one count_ones
        each subaarray with size > limit must have both 0 and 1
        need to use dp
        for a stable array, we cannot have more than limit 0's or limit 1's consecutively
        states are
        Let dp[a][b][c = 0/1][d] be the number of stable arrays with exactly a 0s, b 1s and consecutive d value of c’s at the end.
        c is the last digit used before adding (0/1)
        '''
        memo = {}
        mod = 10**9 + 7

        def dp(a, b, c, d):
            if d > limit:
                return 0
            if a < 0 or b < 0:
                return 0
            if a == 0 and b == 0:
                return 1

            if (a, b, c, d) in memo:
                return memo[(a, b, c, d)]

            add_zero = dp(a-1, b, c, d+1) if c == 0 else dp(a-1, b, 0, 1)
            add_one = dp(a, b-1, c, d+1) if c == 1 else dp(a, b-1, 1, 1)

            ans = (add_zero + add_one) % mod
            memo[(a, b, c, d)] = ans
            return ans

        zeros = dp(zero-1, one, 0, 1)
        ones = dp(zero, one-1, 1, 1)

        return (zeros + ones) % mod
    
#cacheing works a little better
class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        '''
        length of the array cannot exceed zero + one
        there must be zero count_zeros and one count_ones
        each subaarray with size > limit must have both 0 and 1
        need to use dp
        for a stable array, we cannot have more than limit 0's or limit 1's consecutively
        states are
        Let dp[a][b][c = 0/1][d] be the number of stable arrays with exactly a 0s, b 1s and consecutive d value of c’s at the end.
        c is the last digit used before adding (0/1)
        '''
        memo = {}
        mod = 10**9 + 7
        
        @cache
        def dp(a, b, c, d):
            if d > limit:
                return 0
            if a < 0 or b < 0:
                return 0
            if a == 0 and b == 0:
                return 1

            #if (a, b, c, d) in memo:
            #    return memo[(a, b, c, d)]

            add_zero = dp(a-1, b, c, d+1) if c == 0 else dp(a-1, b, 0, 1)
            add_one = dp(a, b-1, c, d+1) if c == 1 else dp(a, b-1, 1, 1)

            ans = (add_zero + add_one) % mod
            #memo[(a, b, c, d)] = ans
            return ans

        zeros = dp(zero-1, one, 0, 1)
        ones = dp(zero, one-1, 1, 1)

        return (zeros + ones) % mod
            
class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        mod = 10**9 + 7
        memo = {}

        def dp(a, b, last):
            if a < 0 or b < 0:
                return 0

            if (a,b,last) in memo:
                return memo[(a,b,last)]

            if last == 0:
                if b == 0:
                    return 1 if a <= limit else 0

                ans = dp(a-1,b,0) + dp(a-1,b,1)
                if a > limit:
                    ans -= dp(a-limit-1,b,1)

            else:
                if a == 0:
                    return 1 if b <= limit else 0

                ans = dp(a,b-1,1) + dp(a,b-1,0)
                if b > limit:
                    ans -= dp(a,b-limit-1,0)

            memo[(a,b,last)] = ans % mod
            return memo[(a,b,last)]

        return (dp(zero,one,0) + dp(zero,one,1)) % mod
            

########################################################
# 3600. Maximize Spanning Tree Stability with Upgrades
# 12MAR26
########################################################
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1
        
        return True
    
class Solution:
    def maxStability(self, n: int, edges: List[List[int]], k: int) -> int:
        '''
        sort and binary search on answer
        need to use kruskal
        so we try to build an MST with a bigger and bigger stability
        '''
        #sort edges on decreasing weight
        edges.sort(key = lambda x: -x[2])

        def check(mid):
            dsu = DSU(n)
            used = 0
            extra = k

            # mandatory edges
            for u,v,w,must in edges:
                if must == 1:
                    if w < mid:
                        return False
                    if not dsu.union(u,v):
                        return False
                    used += 1

            # optional edges
            for u,v,w,must in edges:
                if must == 0:
                    if w >= mid:
                        if dsu.union(u,v):
                            used += 1
                    elif extra > 0 and 2*w >= mid:
                        if dsu.union(u,v):
                            used += 1
                            extra -= 1
                    #if we already have enogh edges to make MST
                    if used == n-1:
                        return True

            return used == n-1 #MST needs n-1 edges

        #boundaries
        left = 0
        right = max(2*w for _,_,w,_ in edges)
        ans = -1

        while left <= right:
            mid = (left + right) // 2
            #can we build an MST with mid stability
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans



