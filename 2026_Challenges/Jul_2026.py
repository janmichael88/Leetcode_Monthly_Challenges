#####################################
# 3286. Find a Safe Walk Through a Grid
# 02JUL26
###########################################
class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        '''
        this is just 01 bfs/djiktras
        we need to store the beast
        need to minimize damage taken along the apth for djikstras
        '''
        rows, cols = len(grid), len(grid[0])

        dirs = [(0,1),(0,-1),(1,0),(-1,0)]

        dist = [[float("inf")] * cols for _ in range(rows)]
        dist[0][0] = grid[0][0]

        pq = [(grid[0][0], 0, 0)]

        while pq:
            damage, i, j = heapq.heappop(pq)

            if damage > dist[i][j]:
                continue

            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    nd = damage + grid[ni][nj]
                    if nd < dist[ni][nj]:
                        dist[ni][nj] = nd
                        heapq.heappush(pq, (nd, ni, nj))

        #also do
        #return health - dist[rows-1][cols-1] >= 1
        return dist[rows - 1][cols - 1] < health
    
#0/1 BFS
from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        rows, cols = len(grid), len(grid[0])

        health -= grid[0][0]
        if health < 1:
            return False

        best = [[-1] * cols for _ in range(rows)]
        dq = deque([(0, 0, health)])
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while dq:
            i, j, hp = dq.popleft()

            if hp <= best[i][j]:
                continue
            best[i][j] = hp

            if (i, j) == (rows - 1, cols - 1):
                return True

            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    nhp = hp - grid[ni][nj]
                    if nhp < 1:
                        continue

                    if grid[ni][nj] == 0:
                        dq.appendleft((ni, nj, nhp))
                    else:
                        dq.append((ni, nj, nhp))

        return False
    
#################################################
# 3620. Network Recovery Pathways
# 04JUL26
################################################
class Solution:
    def findMaxPathScore(self, edges: List[List[int]], online: List[bool], k: int) -> int:
        '''
        we have a DAG
        need max path score
            score is min edge on path, and total path score does need exceed k
            can only use online nodes
        binary seach on ans using djikstras
        when doing dijsktrs only include edges >= mid
        '''
        def dijkstras(edges,online,k,min_edge):
            n = len(online)
            graph = defaultdict(list)
            for u,v,w in edges:
                #filter
                if online[u] and online[v] and w >= min_edge:
                    graph[u].append((v,w))
            
            dists = [float('inf')]*n
            dists[0] = 0
            pq = [(0,0)]

            while pq:
                d,u = heapq.heappop(pq)
                if d > dists[u]:
                    continue
                for v,w in graph[u]:
                    nd = d + w
                    if nd < dists[v]:
                        dists[v] = nd
                        heapq.heappush(pq, (nd,v))
                        
            if dists[n-1] == float('inf') or dists[n-1] > k:
                return False
            return True

        #edge case...

        left,right =  0, max([w for _,_,w in edges]) if edges else 0
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            can_do = dijkstras(edges,online,k,mid)
            if can_do:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans
    
######################################################
# 2492. Minimum Score of a Path Between Two Cities
# 04JUL24
########################################################
#revisited
#TLE
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        dfs from 1 to n
        there is a pathway always from 1 to n
        '''
        graph = defaultdict(list)
        for u,v,w in roads:
            graph[u].append((v,w))
            graph[v].append((u,w))
        
        ans = [float('inf')]
        def dfs(curr,parent,seen,d):
        
            seen.add(curr)
            for neigh,dist in graph[curr]:
                if neigh != parent and neigh not in seen:
                    ans[0] = min(ans[0],dist)
                    dfs(neigh,curr,seen,dist)
            seen.remove(curr)
        
        seen = set()
        dfs(1,-1,seen,float('inf'))
        return ans[0]
    
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        dfs from 1 to n
        there is a pathway always from 1 to n
        find nodes on path , then take the smallest edges
        '''
        graph = defaultdict(list)
        for u,v,w in roads:
            graph[u].append((v,w))
            graph[v].append((u,w))

        on_path = set()
        seen = set()

        def dfs(curr,parent,seen):
            on_path.add(curr)
            seen.add(curr)
            for neigh,dist in graph[curr]:
                if neigh != parent and neigh not in seen:
                    dfs(neigh,curr,seen)
            

        dfs(1,-1,seen)
        ans = float('inf')
        for u,v,w in roads:
            if u in on_path or v in on_path:
                ans = min(ans,w)
        return ans
    
#######################################################
# 1301. Number of Paths with Max Score
# 05JUL26
########################################################
#TLE, close though
class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        '''
        this is just dp
        we can only go left or up, or left up
        '''
        n = len(board)
        dirrs = [(0,-1),(-1,0),(-1,-1)]
        mod = 10**9 + 7

        #find max score
        def dp1(i, j):
            if board[i][j] == 'E':
                return 0

            if (i, j) in memo:
                return memo[(i, j)]

            ans = -float("inf")

            for di, dj in dirrs:
                ii, jj = i + di, j + dj

                if (0 <= ii < n and 0 <= jj < n and board[ii][jj] != 'X'):
                    score = dp1(ii, jj)
                    if score == -float("inf"):
                        continue

                    if board[ii][jj].isdigit():
                        score += int(board[ii][jj])

                    ans = max(ans, score)

            memo[(i, j)] = ans
            return ans

        def dp2(i, j, curr_sum):
            if board[i][j] == 'E':
                return 1 if curr_sum == max_score else 0

            if (i, j, curr_sum) in memo:
                return memo[(i, j, curr_sum)]

            ways = 0

            for di, dj in dirrs:
                ii, jj = i + di, j + dj

                if (0 <= ii < n and 0 <= jj < n and board[ii][jj] != 'X'):
                    nxt = curr_sum
                    if board[ii][jj].isdigit():
                        nxt += int(board[ii][jj])

                    ways += dp2(ii, jj, nxt)

            memo[(i, j, curr_sum)] = ways % mod
            return memo[(i, j, curr_sum)]
                
        memo = {}
        max_score = dp1(n-1,n-1)
        if max_score == float("-inf"):
            return [0,0]
        memo = {}
        num_ways = dp2(n-1,n-1,0)
        return [max_score,num_ways]
    
#need to make into one function
class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        '''
        this is just dp
        we can only go left or up, or left up
        '''
        n = len(board)
        dirrs = [(0,-1),(-1,0),(-1,-1)]
        mod = 10**9 + 7

        #find max score
        def dp(i, j):
            if board[i][j] == 'E':
                return [0,1]

            if (i, j) in memo:
                return memo[(i, j)]

            best = -float("inf")
            ways = 0

            for di, dj in dirrs:
                ii, jj = i + di, j + dj

                if (0 <= ii < n and 0 <= jj < n and board[ii][jj] != 'X'):
                    child_score,child_ways = dp(ii, jj)
                    if child_score == -float("inf"):
                        continue
                    candidate = child_score
                    if board[ii][jj].isdigit():
                        candidate += int(board[ii][jj])

                    if candidate > best:
                        best = candidate
                        ways = child_ways
                    elif candidate == best:
                        ways += child_ways

            entry = [best, ways % mod]
            memo[(i, j)] = entry
            return entry
        
        memo = {}
        score,ways = dp(n-1,n-1)
        if score == float('-inf'):
            return [0,0]
        
        return [score,ways]

####################################################
# Concatenate Non-Zero Digits and Multiply by Sum I
# 06JUL26
####################################################
class Solution:
    def sumAndMultiply(self, n: int) -> int:
        '''
        pull the digits
        '''
        digits = [int(ch) for ch in  str(n)]
        x = 0
        sum_ = 0
        for ch in digits:
            if ch != 0:
                x = x*10 + ch
                sum_ += ch
        
        return x*sum_
    
############################################################
# 3756. Concatenate Non-Zero Digits and Multiply by Sum II
# 08JUL26
##############################################################
#TLE
class Solution:
    def sumAndMultiply(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        use prefix sum for x of digits
        and prefix concantention but its a little different
            this is a little different, we cant do it like normal pref_sum because the zeros break it
            similar to hashing, but the standard trick is to remove the lft pary by diving by the appropriate power of 10
        
        pref_concat stors concat up to i of all non zero digits
        pref_count number of non-zero digits before i
        '''
        pref_sum = [0]
        for ch in s:
            pref_sum.append(int(ch) + pref_sum[-1])

        pref_concat = [0]
        pref_cnt = [0]

        for ch in s:
            d = int(ch)
            if d:
                pref_concat.append(pref_concat[-1] * 10 + d)
                pref_cnt.append(pref_cnt[-1] + 1)
            else:
                pref_concat.append(pref_concat[-1])
                pref_cnt.append(pref_cnt[-1])

        
        ans = []
        mod = 10**9 + 7
        for l,r in queries:
            x = pref_sum[r+1] - pref_sum[l]
            k = pref_cnt[r+1] % mod - pref_cnt[l] % mod
            concat = (pref_concat[r+1] % (10**k)) % mod
            ans.append(x*concat % mod)

        return ans
    
#finally
class Solution:
    def sumAndMultiply(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        use prefix sum for x of digits
        and prefix concantention but its a little different
            this is a little different, we cant do it like normal pref_sum because the zeros break it
            similar to hashing, but the standard trick is to remove the lft pary by diving by the appropriate power of 10
        
        pref_concat stors concat up to i of all non zero digits
        pref_count number of non-zero digits before i
        recall rolling  hash
        hash(l, r) = pref[r] - pref[l] * base^(r-l)

        left_part = 123
        answer    = 456
        k = 3

        pref_concat[r+1] = 123456

        answer = pref_concat[r+1] - left_part * 10^k
        answer = (pref_concat[r+1] - pref_concat[l] * pow10[k]) % MOD
        '''
        mod = 10**9 + 7

        pref_sum = [0]
        pref_concat = [0]
        pref_cnt = [0]

        for ch in s:
            d = int(ch)
            pref_sum.append(pref_sum[-1] + d)

            if d:
                pref_concat.append((pref_concat[-1] * 10 + d) % mod)
                pref_cnt.append(pref_cnt[-1] + 1)
            else:
                pref_concat.append(pref_concat[-1])
                pref_cnt.append(pref_cnt[-1])

        pow10 = [1] * (pref_cnt[-1] + 1)
        for i in range(1, len(pow10)):
            pow10[i] = (pow10[i - 1] * 10) % mod


        ans = []
        for l, r in queries:
            x = pref_sum[r + 1] - pref_sum[l]
            k = pref_cnt[r + 1] - pref_cnt[l]
            concat = (pref_concat[r + 1] - pref_concat[l] * pow10[k]) % mod
            ans.append(x * concat % mod)

        return ans

##################################################
# 3534. Path Existence Queries in a Graph II
# 11JUL26
###################################################
class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[int]:
        '''
        lets comb through this problem so we can finally get over binary lifiting
        first thing to note is that the graph is implicit, so we have numbers ordered like 
        1 --3 --5 --8 --10 --12, with maxDiff = 3
        the firstt comp is 1,3,5 then 8,10,12

        so if we have nums[i] <= nums[j] <= nums[k] amd nums[k] - nums[i] > maxDiff, i cannot reach anything after k
        but we need the minimum distatance, so we need to track jumps, but we can't just loop through all the jumps for each query
        that would take too long, O(n), 

        so for each node, we need to record the furthest distance, going back to our first exaplem wed have something like
        1 -> {3,5}
        3 -> {5,6}
        so its kinda like jump game, and we can ask
        Starting at index i, repeatedly replace, where we have i = next[i], how many jumps is that

        notes on binary lifiting, at its cores its like jumping by powers of two, its very standard, either know it or dont
        it creates a sprase table, and use the sparse table to answer queries 
        suppose we have a linked list like : 0 -> 3 -> 5 -> 8 -> 10 -> None
        and we want to ask, where are you after 13 jumps from node 0
        the dumb way is just do the jumps itertiavely from 0, and on, but that requires O(jump) time ~ O(n)

        a better way would be to compure where i am after 2 jumps, 4, jumps, 8.... all the way,
        jump1[x] = next node
        jump2[x] = next[next[x]]
        jump4[x] = jump2[jump2[x]]
        jump8[x] = jump4[jump4[x]]

        this becomes sparse table
                        1 jump   2 jumps   4 jumps   8 jumps

            0            3         5         10       None
            3            5         8        None      None
            5            8        10        None      None
            8           10       None       None      None

        notice each column is twice the previous one

        so now if i ask, for 13 jumps
        13 in binary is 13 = 8 + 4 + 1
        get jumps for each from each column, thats only 3 ops, we've reduced to O(jump) to O(log(jump))

        the general formula
        up[k][v] menas where are you after 2**k jumps
        so we have, up[0][v] = next[v]
        ans we have, up[1][v] = up[0][ up[0][v] ] 
        and we have  up[2][v] = up[1][ up[1][v] ]
        generally we have up[k][v] = up[k-1][ up[k-1][v] ], which is bottom up dp

        then answering a query becomes like this,
        wherre am i after 19 jumps? first convert to binary 10011, which is 16 + 2 + 1
        and we get this:
                node = start

        for k in reversed(range(LOG)):
            if (steps >> k) & 1: #check bit is set
                node = up[k][node]
        '''
        # --------------------------------------------------
        # Step 1. Sort nodes by value
        # --------------------------------------------------
        arr = sorted((nums[i], i) for i in range(n))

        # pos[original node] = position in sorted order
        pos = [0] * n
        values = [0] * n

        for i, (val, node) in enumerate(arr):
            pos[node] = i
            values[i] = val

        # --------------------------------------------------
        # Step 2. Compute next[i]
        # next[i] = furthest index reachable in ONE edge
        # --------------------------------------------------
        next_ = [0] * n

        j = 0
        for i in range(n):
            if j < i:
                j = i

            while j + 1 < n and values[j + 1] - values[i] <= maxDiff:
                j += 1

            next_[i] = j

        # --------------------------------------------------
        # Step 3. Connected components
        # Every gap > maxDiff starts a new component.
        # --------------------------------------------------
        comp = [0] * n

        cid = 0
        comp[0] = 0

        for i in range(1, n):
            if values[i] - values[i - 1] > maxDiff:
                cid += 1
            comp[i] = cid

        # --------------------------------------------------
        # Step 4. Binary lifting
        # up[k][i] = position after 2^k greedy jumps
        # --------------------------------------------------
        LOG = n.bit_length()

        up = [[0] * n for _ in range(LOG)]
        up[0] = next_[:]

        for k in range(1, LOG):
            for i in range(n):
                up[k][i] = up[k - 1][up[k - 1][i]]

        # --------------------------------------------------
        # Step 5. Answer queries
        # --------------------------------------------------
        ans = []

        for u, v in queries:

            a = pos[u]
            b = pos[v]

            if a > b:
                a, b = b, a

            # Different connected components
            if comp[a] != comp[b]:
                ans.append(-1)
                continue

            if a == b:
                ans.append(0)
                continue

            cur = a
            steps = 0

            # Greedily take largest binary jumps
            for k in range(LOG - 1, -1, -1):
                nxt = up[k][cur]
                if nxt < b:
                    cur = nxt
                    steps += 1 << k

            ans.append(steps + 1)

        return ans
    
###################################################
# 2572. Count the Number of Square-Free Subsets
# 13JUL26
###################################################
class Solution:
    def squareFreeSubsets(self, nums: List[int]) -> int:
        '''
        subset of the array is square free if product of all its elements is a zqure free integer
        sqaure free integer is an integer that is divisible by no square other than 1
        '''
        MOD = 10 ** 9 + 7
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] # 10
        freq = Counter(nums)
        # consider only unique elements from nums
        arr = list(set(nums))
        N = len(arr) # N <= 30
        
        @cache
        def rec(i, prod, empty):
            if i == N: 
                return not empty
            
            # not pick
            cnt = rec(i + 1, prod, empty)

            # check and pick
            x = arr[i]
            next_prod = prod * x
            is_distinct_prime_product = all(next_prod % (prime * prime) for prime in primes)
            if is_distinct_prime_product:
                multiplier = freq[x] if x != 1 else (pow(2, freq[1], MOD) - 1)
                cnt += multiplier * rec(i + 1, next_prod, 0)
            # print(i, prod, next_prod, is_distinct_prime_product)

            return cnt % MOD

        return rec(0, 1, 1)
    
#############################################################
# 3336. Find the Number of Subsequences With Equal GCD
# 14JUL26
##############################################################
#ezzzz
import math
class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        '''
        is dp to stor number of subsequnces up to i with gcd1 and gcd2
        '''
        n = len(nums)
        memo = {}
        mod = 10**9 + 7

        def dp(i,gcd1,gcd2):
            if i == n:
                return gcd1 == gcd2
            if (i,gcd1,gcd2) in memo:
                return memo[(i,gcd1,gcd2)]
            no_take_ways = dp(i+1, gcd1, gcd2)
            take1 = dp(i+1, math.gcd(gcd1,nums[i]), gcd2)
            take2 = dp(i+1, gcd1, math.gcd(gcd2,nums[i]))
            ans = no_take_ways + take1 + take2
            ans %= mod
            memo[(i,gcd1,gcd2)] = ans
            return ans
        
        return dp(0,0,0) - 1 #the null set
    
###################################################
# 3658. GCD of Odd and Even Sums
# 14JUL26
####################################################
import math
class Solution:
    def gcdOfOddEvenSums(self, n: int) -> int:
        '''
        use sum of arithmetic series
        for odd its
        n*(1 + 2n - 1) // 2
        n + 2n^2 - n
        2n^2 // 2 = n

        for even its

        (2n + 2n^2) // 2
        n + n^2 = n(1 + n)


        euclidean gcd is
        gcd(a,b) = gcd(b, a % b)
        if b == 0 return a

        trick for remembering, we form pairs
        say we want sum a + b + c + d
        we can do (a + b) + (b + c) + (c + b) + (d + a)
        then divide by two 

        properties of gcd
        gcd(n*a,n*b) = n*gcd(a,b)
        gcd(a,b) = gcd(a, b-a) = gcd(b-k*a) 
        '''
        #recursive gcd
        def gcd(a,b):
            if b == 0:
                return a
            return gcd(b, a % b)
        sumOdd = n*(1 +2*n-1) // 2
        sumEven = n*(2 + 2*n) // 2

        return math.gcd(sumOdd,sumEven)
    
#################################################
# 2521. Distinct Prime Factors of Product of Array
# 15JUL26
#################################################
class Solution:
    def distinctPrimeFactors(self, nums: List[int]) -> int:
        '''
        for each number get the prime factorizaion and put into set
        '''
        prime_factors = Counter()

        def get_pf(n):
            factors = Counter()

            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors[d] += 1
                    n //= d
                d += 1

            if n > 1:
                factors[n] += 1

            return factors

        for num in nums:
            facts = get_pf(num)
            prime_factors |= facts

        return len(prime_factors)
            
###########################################
# 3867. Sum of GCD of Formed Pairs
# 15JUL26
##########################################
#cheese is realllll
class Solution:
    def gcdSum(self, nums: list[int]) -> int:
        '''
        follow the rules
        '''
        def gcd(a,b):
            if b == 0:
                return a
            
            return gcd(b, a % b)
        
        n = len(nums)
        max_i = [0]*n
        max_i[0] = nums[0]
        for i in range(n):
            max_i[i] = max(max_i[i-1],nums[i])
        
        prefGcd = [0]*n
        for i in range(n):
            prefGcd[i] = gcd(nums[i],max_i[i])
        
        #sort
        prefGcd.sort()
        ans = 0
        left,right = 0,n-1
        
        while left < right:
            ans += gcd(prefGcd[left],prefGcd[right])
            left += 1
            right -= 1
        
        return ans
    
#################################################
# 3312. Sorted GCD Pair Queries
# 17JUL26
#################################################
class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        '''
        If there are n numbers, there are

            n * (n - 1) // 2

        unordered pairs (i < j).
        '''

        counts = Counter(nums)
        m = max(nums)

        # -------------------------------------------------------------
        # Step 1: Count how many numbers are divisible by every g.
        #
        # counts[x] = frequency of the value x.
        #
        # gcd_counts[g] = number of elements in nums divisible by g.
        #
        # Example:
        # nums = [2,4,6,8]
        #
        # gcd_counts[2] = 4
        # because every number is divisible by 2.
        #
        # This DOES NOT count pairs yet.
        #
        # We do this because if gcd(a,b)=g, then both a and b must be
        # divisible by g.
        # -------------------------------------------------------------
        gcd_counts = Counter()
        for g in range(1, m + 1):
            for multiple in range(g, m + 1, g):
                gcd_counts[g] += counts[multiple]

        # -------------------------------------------------------------
        # Step 2: Compute the number of pairs whose gcd is EXACTLY g.
        #
        # If gcd_counts[g] = k, then
        #
        #     C(k,2)
        #
        # is the number of pairs whose gcd is a MULTIPLE of g.
        #
        # Those pairs are mixed together:
        #
        # gcd = g
        # gcd = 2g
        # gcd = 3g
        # ...
        #
        # Therefore,
        #
        # exact[g] = C(k,2)
        #
        # initially overcounts.
        #
        # Process g from largest to smallest. Since larger multiples
        # have already been computed, subtract them away:
        #
        # exact[g]
        #     = pairs divisible by g
        #       - pairs with gcd = 2g
        #       - pairs with gcd = 3g
        #       - ...
        #
        # After subtracting every multiple, exact[g] is the number of
        # pairs whose gcd is exactly g.
        # -------------------------------------------------------------
        exact = [0] * (m + 1)

        for g in range(m, 0, -1):
            exact[g] = gcd_counts[g] * (gcd_counts[g] - 1) // 2

            for multiple in range(2 * g, m + 1, g):
                exact[g] -= exact[multiple]

        # -------------------------------------------------------------
        # Step 3:
        # Convert exact[] into a prefix sum.
        #
        # exact[g] now stores the number of pairs with gcd <= g.
        #
        # This makes it possible to binary search for the gcd
        # corresponding to the q-th pair in sorted order.
        # -------------------------------------------------------------
        for i in range(1, m + 1):
            exact[i] += exact[i - 1]

        ans = []
        for q in queries:
            idx = bisect.bisect_left(exact, q + 1)
            ans.append(idx)

        return ans
    
#############################################
# 3499. Maximize Active Section with Trade I
# 20JUL26
#############################################
class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        '''
        in one operation we can
            convert a contiguous block of 1s (surrouned by 0s to all 0s)
            then conver the blocks os that is surrounded by 1s to all 1s
        
        problem is misleading, its telling us to remove the middles ones, convert to zeros
        then covnert all back to ones
        we really only go up by the left and right parts' lengths if they are zero
        '''
        counts = Counter(s)
        ans = counts['1']

        #aguemtn
        s = "1" + s + "1"
        #parition into blocks of ones
        parts = []
        for k,g in groupby(s):
            parts.append(list(g))
        n = len(parts)
        for i in range(1,n-1):
            p = parts[i]
            if p[0] == "1":
                left,right = parts[i-1],parts[i+1]
                gain = len(left) + len(right)
                #the ones in the center block are already included in the original count
                #so we just go up by the left and right blocks, which should be zeros
                ans = max(ans,counts['1'] + gain)
                
        
        return ans