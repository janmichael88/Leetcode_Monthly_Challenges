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

##############################################################
# 3296. Minimum Number of Seconds to Make Mountain Height Zero
# 14MAR26
###############################################################
class Solution:
    def minNumberOfSeconds(self, mountainHeight: int, workerTimes: List[int]) -> int:
        '''
        define reduce(x) as the amount of time to reduce height by x
        worker_time[i]*(sum(1, to x + 1))
        to get that sum we can use gauss trick
        x*(x+1)/2
        i need to check(x), it returns the amount of time to decrease mountainHeight to 0, in x seconds
        if i can do check(x), i can do it in check(x+1) seconds
        so given a time x check
        for a worker[i] 
        time = work*((x*(x+1)/2))
        need to solve the qudratic
        and sum for all workers, was on the righty track, and check if bigger than mountain height
        x = (-1 + math.sqrt(1 + 8*time/work)) / 2
        '''
        #need positive root
        def solve_x(time, work):
            return (-1 + math.sqrt(1 + 8*time/work)) // 2 #floor division

        def check(x):
            total_height = 0
            for w in workerTimes:
                h = solve_x(x,w)
                total_height += h
            
            return total_height
        
        left = 0
        right = max(workerTimes) * mountainHeight * (mountainHeight + 1) // 2
        ans = right

        while left <= right:
            mid = left + (right - left) // 2

            if check(mid) >= mountainHeight:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans

#################################################
# 1878. Get Biggest Three Rhombus Sums in a Grid
# 16MAR25
###################################################
#need all the borders
class Solution:
    def getBiggestThree(self, grid: List[List[int]]) -> List[int]:
        '''
        enumerate all, say we have tip of rhombus at (i,j), 
        for some k in range(0,n), we can find left, and right at
        (i+k,j+k), (i+k,j-k) if in bounds
        we can find the bottom be picking left and doing
        (i+2k,j+2k)
        careful with boundary checking, grid is not square
        crap sum is the whole border
        '''
        rows,cols = len(grid),len(grid)
        sums = []
        for i in range(rows):
            for j in range(cols):
                sums.append(grid[i][j]) #single cell is rhombus
                for k in range(1,max(rows,cols)):
                    left = (i+k,j-k)
                    right = (i+k,j+k)
                    bottom = (left[0] + k, left[1] + k)
                    #check corners for within bounds
                    if 0 <= left[0] < rows and 0 <= left[1] < cols:
                        if 0 <= right[0] < rows and 0 <= right[1] < cols:
                            if 0 <= bottom[0] < rows and 0 <= bottom[1] < cols:
                                curr_sum = grid[left[0]][left[1]] + grid[right[0]][right[1]] + grid[bottom[0]][bottom[1]] + grid[i][j]
                                sums.append(curr_sum)
        sums.sort(reverse = True)
        return sums[:3]

#walk borders CCW
class Solution:
    def getBiggestThree(self, grid):
        '''
        just walk the borders for each length k
        '''
        rows, cols = len(grid), len(grid[0])
        s = set()

        for i in range(rows):
            for j in range(cols):

                # size 0 rhombus
                s.add(grid[i][j])

                k = 1
                while True:
                    #out of bounds
                    if not (0 <= i + 2*k < rows and 0 <= j - k and j + k < cols):
                        break

                    total = 0

                    # edge 1: top -> left
                    for t in range(k):
                        total += grid[i + t][j - t]

                    # edge 2: left -> bottom
                    for t in range(k):
                        total += grid[i + k + t][j - k + t]

                    # edge 3: bottom -> right
                    for t in range(k):
                        total += grid[i + 2*k - t][j + t]

                    # edge 4: right -> top
                    for t in range(k):
                        total += grid[i + k - t][j + k - t]

                    s.add(total)
                    k += 1

        return sorted(s, reverse=True)[:3]
    
###########################################
# 1727. Largest Submatrix With Rearrangements
# 17MAR26
###########################################
class Solution:
    def largestSubmatrix(self, matrix: List[List[int]]) -> int:
        '''
        count consec ones in cols at each positions for each col
        '''
        rows,cols = len(matrix), len(matrix[0])
        consec_ones = [[0]*cols for _ in range(rows)]

        for c in range(cols):
            consec_ones[0][c] = matrix[0][c]
        
        for i in range(1,rows):
            for j in range(cols):
                if matrix[i][j] == 1:
                    consec_ones[i][j] = consec_ones[i-1][j] + 1
                else:
                    consec_ones[i][j] = 0
        ans = 0
        for r in consec_ones:
            sorted_r = sorted(r,reverse = True)
            for i in range(cols):
                height = sorted_r[i]
                ans = max(ans, height*(i+1))
        return ans

###################################################################
# 3070. Count Submatrices with Top-Left Element and Sum Less Than k
# 18MAR26
####################################################################
class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        '''
        two do pref sum, then
        check all submatries locked in at (0,0) for top left
        '''
        rows,cols = len(grid), len(grid[0])
        pref_sum = [[0]*(cols + 1) for _ in range(rows+1)]
        #count on fly
        ans = 0
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                pref_sum[i][j] = grid[i-1][j-1]  \
                + pref_sum[i-1][j]  \
                + pref_sum[i][j-1]  \
                - pref_sum[i-1][j-1]
                if pref_sum[i][j] <= k:
                    ans += 1
        
        return ans
    
#we can also do linear space
#keep sum array that hold col sum up to current row
class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        '''
        two do pref sum, then
        check all submatries locked in at (0,0) for top left
        '''
        rows,cols = len(grid), len(grid[0])
        col_sums = [0]*cols

        ans = 0
        for i in range(rows):
            row_sum = 0
            for j in range(cols):
                col_sums[j] += grid[i][j]
                row_sum += col_sums[j]
                if row_sum <= k:
                    ans += 1
        
        return ans
    
##########################################################
# 3212. Count Submatrices With Equal Frequency of X and Y
# 19MAR26
##########################################################
class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        '''
        keep two seperate pref_sums showing counts for X and Y
        then check all of them with anchor (0,0)
        '''
        rows,cols = len(grid),len(grid[0])
        count_xs = [[0]*(cols + 1) for _ in range(rows+1)]
        count_ys = [[0]*(cols + 1) for _ in range(rows+1)]

        ans = 0
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                if grid[i-1][j-1] == "X":
                    count_xs[i][j] = 1 + count_xs[i-1][j] + count_xs[i][j-1] - count_xs[i-1][j-1]
                else:
                    count_xs[i][j] = count_xs[i-1][j] + count_xs[i][j-1] - count_xs[i-1][j-1]
        
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                if grid[i-1][j-1] == "Y":
                    count_ys[i][j] = 1 + count_ys[i-1][j] + count_ys[i][j-1] - count_ys[i-1][j-1]
                else:
                    count_ys[i][j] = count_ys[i-1][j] + count_ys[i][j-1] - count_ys[i-1][j-1]

                
                #validate on second pass:
                if count_xs[i][j] == count_ys[i][j] and count_xs[i][j] > 0:
                    ans += 1

        return ans
    
#single pass
class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        '''
        keep two seperate pref_sums showing counts for X and Y
        then check all of them with anchor (0,0)
        '''
        rows,cols = len(grid),len(grid[0])
        count_xs = [[0]*(cols + 1) for _ in range(rows+1)]
        count_ys = [[0]*(cols + 1) for _ in range(rows+1)]

        ans = 0
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                if grid[i-1][j-1] == "X":
                    count_xs[i][j] = 1 + count_xs[i-1][j] + count_xs[i][j-1] - count_xs[i-1][j-1]
                else:
                    count_xs[i][j] = count_xs[i-1][j] + count_xs[i][j-1] - count_xs[i-1][j-1]
                
                if grid[i-1][j-1] == "Y":
                    count_ys[i][j] = 1 + count_ys[i-1][j] + count_ys[i][j-1] - count_ys[i-1][j-1]
                else:
                    count_ys[i][j] = count_ys[i-1][j] + count_ys[i][j-1] - count_ys[i-1][j-1]

                
                #validate on second pass:
                if count_xs[i][j] == count_ys[i][j] and count_xs[i][j] > 0:
                    ans += 1
        return ans
    
###########################################
# 3643. Flip Square Submatrix Vertically
# 21MAR26
############################################
class Solution:
    def reverseSubmatrix(self, grid: List[List[int]], x: int, y: int, k: int) -> List[List[int]]:
        '''
        flip in place
        we need to go down vertically
        flip start row and end row one at a time, two poiters
        '''
        start_row, end_row = x, x + k - 1
        while start_row < end_row:
            for col in range(y,y+k):
                grid[start_row][col], grid[end_row][col] =  grid[end_row][col], grid[start_row][col]
            
            start_row += 1
            end_row -= 1
        
        return grid

   
class Solution:
    def reverseSubmatrix(self, grid: List[List[int]], x: int, y: int, k: int) -> List[List[int]]:
        '''
        flip in place
        we need to go down vertically
        '''

        for c in range(y, y + k):

            top = x
            bot = x + k - 1

            while top < bot:
                grid[top][c], grid[bot][c] = grid[bot][c], grid[top][c]
                top += 1
                bot -= 1

        return grid

#################################################
# 2470. Number of Subarrays With LCM Equal to K
# 22MAR26
#################################################
from math import gcd
class Solution:
    def subarrayLCM(self, nums: List[int], k: int) -> int:
        '''
        we can use dp here
        if we have, we can just try to extend each 
        oh remember these aren't subsequnces, these are subarrays
        lcm is associative
        say we have lcm(a,b,c,d) = lcm(lcm(lcm(a,b),c),d)
        lcm of single elemnt it itself
        '''
        n = len(nums)
        def lcm(a,b):
            return a*b//math.gcd(a,b)
        ans = 0
        for i in range(n):
            curr_lcm = 1
            for j in range(i,n):
                curr_lcm = lcm(curr_lcm,nums[j])
                if curr_lcm == k:
                    ans += 1
                
                if curr_lcm > k:
                    break
        return ans
    
#######################################################
# 1594. Maximum Non Negative Product in a Matrix
# 23MAR26
########################################################
class Solution:
    def maxProductPath(self, grid: List[List[int]]) -> int:
        '''
        this is just dp, can only move right or down
        cant treat it like summing, because a later negative could make it positive
        store min and max
        '''
        rows, cols = len(grid), len(grid[0])
        memo = {}
        MOD = 10**9 + 7

        def dp(i, j):
            if (i, j) == (rows - 1, cols - 1):
                return (grid[i][j], grid[i][j])
            
            if (i, j) in memo:
                return memo[(i, j)]

            curr = grid[i][j]
            cands = []
            if i + 1 < rows:
                down_min, down_max = dp(i + 1, j)
                cands.append(down_min)
                cands.append(down_max)
            if j + 1 < cols:
                right_min, right_max = dp(i, j + 1)
                cands.append(right_min)
                cands.append(right_max)

            new_min = curr*min(cands)
            new_max = curr*max(cands)

            memo[(i, j)] = (new_min, new_max)
            return memo[(i, j)]

        ans_min, ans_max = dp(0, 0)
        #print(ans_min,ans_max)
        if ans_max < 0:
            return -1
        return ans_max % MOD
    
#need to only go to valid states
#dont treat like sum
class Solution:
    def maxProductPath(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        memo = {}
        MOD = 10**9 + 7

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]

            curr = grid[i][j]

            # destination
            if i == rows - 1 and j == cols - 1:
                return (curr, curr)

            candidates = []

            # go down
            if i + 1 < rows:
                dmin, dmax = dp(i + 1, j)
                candidates.append(curr * dmin)
                candidates.append(curr * dmax)

            # go right
            if j + 1 < cols:
                rmin, rmax = dp(i, j + 1)
                candidates.append(curr * rmin)
                candidates.append(curr * rmax)

            new_min = min(candidates)
            new_max = max(candidates)

            memo[(i, j)] = (new_min, new_max)
            return memo[(i, j)]

        ans_min, ans_max = dp(0, 0)

        if ans_max < 0:
            return -1
        return ans_max % MOD
    
#bottom up
class Solution:
    def maxProductPath(self, grid: List[List[int]]) -> int:
        '''
        bottom up
        '''
        rows, cols = len(grid), len(grid[0])
        dp = dp = [[[None,None] for _ in range(cols)] for _ in range(rows)]
        dp[rows-1][cols-1] = [grid[rows-1][cols-1],grid[rows-1][cols-1]]
        MOD = 10**9 + 7

        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                if (i,j) == (rows-1,cols-1):
                    continue

                curr = grid[i][j]
                candidates = []

                # go down
                if i + 1 < rows:
                    dmin, dmax = dp[i + 1][j]
                    candidates.append(curr * dmin)
                    candidates.append(curr * dmax)

                # go right
                if j + 1 < cols:
                    rmin, rmax = dp[i][j+1]
                    candidates.append(curr * rmin)
                    candidates.append(curr * rmax)

                new_min = min(candidates)
                new_max = max(candidates)

                dp[i][j] = (new_min, new_max)


        ans_min, ans_max = dp[0][0]

        if ans_max < 0:
            return -1
        return ans_max % MOD

#########################################################
# 2906. Construct Product Matrix 
# 24MAR26
##########################################################
#damnit wtf
class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        '''
        finally an ez medirum
        get total product then divide out
        unless division operator takes too long
        if i had an 1d array [a,b,c,d,e]
        i can use pref and suff prod
        [b*c*d*e, a*c*d*e]
        its left pref_prod[i-1]*suff_prod[i+1]
        how to do using 2d array
        put all elements into single array, then conver (i,j) to an index
        '''
        rows,cols = len(grid),len(grid[0])
        pref_prd = [1]
        suff_prd = [1]
        for i in range(rows):
            for j in range(cols):
                pref_prd.append(grid[i][j]*pref_prd[-1])
        
        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                suff_prd.append(grid[i][j]*suff_prd[-1])
        suff_prd = suff_prd[::-1]

        for i in range(rows):
            for j in range(cols):
                k = i*cols + j
                left = pref_prd[k]
                right = suff_prd[k+1]
                #print(left,right)
                grid[i][j] = (left*right) % 12345
        

        return grid
    
#product except self
#just store pref before putting into ans matrix
#then update

#then on suffix, add in
class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        '''
        finally an ez medirum
        get total product then divide out
        unless division operator takes too long
        if i had an 1d array [a,b,c,d,e]
        i can use pref and suff prod
        [b*c*d*e, a*c*d*e]
        its left pref_prod[i-1]*suff_prod[i+1]
        how to do using 2d array
        put all elements into single array, then conver (i,j) to an index
        '''
        MOD = 12345
        rows, cols = len(grid), len(grid[0])

        # result matrix
        res = [[1] * cols for _ in range(rows)]

        # ----- pass 1: prefix -----
        pref = 1
        for i in range(rows):
            for j in range(cols):
                res[i][j] = pref
                pref = (pref * grid[i][j]) % MOD

        # ----- pass 2: suffix + combine -----
        suff = 1
        for i in range(rows - 1, -1, -1):
            for j in range(cols - 1, -1, -1):
                res[i][j] = (res[i][j] * suff) % MOD
                suff = (suff * grid[i][j]) % MOD

        return res
    
##########################################
# 3546. Equal Sum Grid Partition I
# 27MAR26
##########################################
class Solution:
    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        '''
        use 2d prefsum and try all cuts
        if there are r rows, then there are r-1 cuts
        if there are c cols, then there are c-1 cuts
        '''
        rows,cols = len(grid),len(grid[0])
        pref_sum = [[0]*(cols+1) for _ in range(rows+1)]
        for r in range(rows):
            for c in range(cols):
                pref_sum[r+1][c+1] = (grid[r][c] + pref_sum[r][c+1] + pref_sum[r+1][c] - pref_sum[r][c])
            
        total_sum = pref_sum[-1][-1]

        #try all row horiontal cuts
        for r in range(1,rows):
            top = pref_sum[r][cols]
            if (top*2 == total_sum):
                return True
        #try all vert cuts
        for c in range(1,cols):
            left = pref_sum[rows][c]
            if (left*2 == total_sum):
                return True
        return False
    
################################################################
# 3548. Equal Sum Grid Partition II
# 27MAR26
#################################################################
import collections

class Solution:
    def canPartitionGrid(self, grid: List[List[int]]) -> bool:
        rows, cols = len(grid), len(grid[0])
        pref_sum = [[0]*(cols+1) for _ in range(rows+1)]
        
        # 1. Build Prefix Sum
        for r in range(rows):
            for c in range(cols):
                pref_sum[r+1][c+1] = grid[r][c] + pref_sum[r+1][c] + pref_sum[r][c+1] - pref_sum[r][c]
        
        total_sum = pref_sum[rows][cols]

        # 2. Helper to check if a value exists in a "connected-safe" way
        def has_valid_discount(r_start, r_end, c_start, c_end, target_val):
            h = r_end - r_start
            w = c_end - c_start
            if h <= 0 or w <= 0 or target_val < 0: return False
            
            # Case A: Thin Row (1 row high)
            if h == 1:
                return grid[r_start][c_start] == target_val or grid[r_start][c_end-1] == target_val
            
            # Case B: Thin Column (1 col wide)
            if w == 1:
                return grid[r_start][c_start] == target_val or grid[r_end-1][c_start] == target_val
            
            # Case C: Thick Section (Check all elements)
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    if grid[r][c] == target_val:
                        return True
            return False

        # 3. Try Horizontal Cuts
        for r in range(1, rows):
            top_sum = pref_sum[r][cols]
            bottom_sum = total_sum - top_sum
            
            # Top needs to lose 'diff' OR Bottom needs to lose 'diff'
            if top_sum == bottom_sum: return True
            
            # If top is heavier: Top - X = Bottom  => X = top - bottom
            if has_valid_discount(0, r, 0, cols, top_sum - bottom_sum): return True
            # If bottom is heavier: Bottom - X = Top => X = bottom - top
            if has_valid_discount(r, rows, 0, cols, bottom_sum - top_sum): return True

        # 4. Try Vertical Cuts
        for c in range(1, cols):
            left_sum = pref_sum[rows][c]
            right_sum = total_sum - left_sum
            
            if left_sum == right_sum: return True
            
            if has_valid_discount(0, rows, 0, c, left_sum - right_sum): return True
            if has_valid_discount(0, rows, c, cols, right_sum - left_sum): return True

        return False
    
###############################################
# 2573. Find the String with LCP 
# 29MAR26
###############################################
class Solution:
    def findTheString(self, lcp: List[List[int]]) -> str:
        '''
        we have the lcp matrix, we just need to make it
        lcp(i,j) = length og longest common prefix between substrings word[i:n-1] and word[j:n-1]
        example lcp(0,0) = 4 means lcp from word[0:n-1] to word[0:n-1] is 4
        '''
        n = len(lcp)
        word = [""]*n
        curr = ord('a')

    
        #biuld
        for i in range(n):
            if not word[i]:
                if curr > ord('z'):
                    return ""
                word[i] = chr(curr)
                for j in range(i+1,n):
                    if lcp[i][j] > 0: 
                        word[j] = word[i]
                curr += 1
        #validate, rebuild lcp and check if ==
        new_lcp = [[0]*n for _ in range(n)]
        for i in range(n-1,-1,-1):
            for j in range(n-1,-1,-1):
                if word[i] == word[j]:
                    if i == n-1 or j == n-1:
                        new_lcp[i][j] = 1
                    else:
                        new_lcp[i][j] = 1 + new_lcp[i+1][j+1]
    
        if new_lcp == lcp:
            return "".join(word)
        return ""
    
################################################################
# 2839. Check if Strings Can be Made Equal With Operations I
# 29MAR26
################################################################
class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        '''
        our only options are (0,2) and (1,3)
        call this opA and opB
        i can do opA,apB, opA -> opB, opB ->o pA
        [a,b,c,d]
        [c,b,a,d]
        [c,d,a,b]
        [a,d,c,b]
        '''
        s1,s2 = list(s1),list(s2)
        def op(i,j,arr):
            arr[i],arr[j] = arr[j],arr[i]
            return arr

        s1_possibles = [s1]
        s2_possibles = [s2]
        a,b = [0,2],[1,3]
        trans = [[a],[b],[a,b],[b,a]]
        for t in trans:
            for i,j in t:
                s1_main,s2_main = s1[:],s2[:]
                s1_main = op(i,j,s1_main)
                s1_possibles.append(s1_main[:])
                s2_main = op(i,j,s2_main)
                s2_possibles.append(s2_main[:])
        
        for u in s1_possibles:
            for v in s2_possibles:
                if u == v:
                    return True
        return False

class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        '''
        our only options are (0,2) and (1,3)
        two chars must be same at (0,2) and two chars must be same at (1,3)
        '''
        return ({s1[0], s1[2]} == {s2[0], s2[2]} and
                {s1[1], s1[3]} == {s2[1], s2[3]})
