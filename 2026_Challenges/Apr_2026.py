######################################################
# 3418. Maximum Amount of Money Robot Can Earn
# 02APR26
######################################################
#top down gets MLE
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        '''
        dp, states are (i,j,k)
        netrualize or don't netrualize
        '''
        memo = {}
        rows, cols = len(coins), len(coins[0])

        def dp(i, j, k):
            if (i, j) == (rows - 1, cols - 1):
                if coins[i][j] >= 0:
                    return coins[i][j]
                else:
                    if k > 0:
                        return 0
                    else:
                        return coins[i][j]

            if (i, j, k) in memo:
                return memo[(i, j, k)]

            ans = float('-inf')

            if coins[i][j] >= 0:
                if i + 1 < rows:
                    ans = max(ans, coins[i][j] + dp(i + 1, j, k))
                if j + 1 < cols:
                    ans = max(ans, coins[i][j] + dp(i, j + 1, k))
            else:
                # dont neutralize
                if i + 1 < rows:
                    ans = max(ans, coins[i][j] + dp(i + 1, j, k))
                if j + 1 < cols:
                    ans = max(ans, coins[i][j] + dp(i, j + 1, k))

                # option 2: neutralize if we can
                if k > 0:
                    if i + 1 < rows:
                        ans = max(ans, dp(i + 1, j, k - 1))
                    if j + 1 < cols:
                        ans = max(ans, dp(i, j + 1, k - 1))

            memo[(i, j, k)] = ans
            return ans

        return dp(0, 0, 2)
    
#yesssss
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        '''
        dp, states are (i,j,k)
        netrualize or don't netrualize
        '''
        rows, cols = len(coins), len(coins[0])
        dp = [[[float('-inf')]*3 for _ in range(cols)] for _ in range(rows)]

        #base case fill
        for i in range(rows):
            for j in range(cols):
                for k in range(3):
                    if (i, j) == (rows - 1, cols - 1):
                        if coins[i][j] >= 0:
                            dp[i][j][k] = coins[i][j]
                        else:
                            if k > 0:
                                dp[i][j][k] = 0
                            else:
                                dp[i][j][k] = coins[i][j]
        
        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                if (i, j) == (rows - 1, cols - 1):
                    continue
                for k in range(3):
                    ans = float('-inf')
                    if coins[i][j] >= 0:
                        if i + 1 < rows:
                            ans = max(ans, coins[i][j] + dp[i + 1][j][k])
                        if j + 1 < cols:
                            ans = max(ans, coins[i][j] + dp[i][j + 1][k])
                    else:
                        # dont neutralize
                        if i + 1 < rows:
                            ans = max(ans, coins[i][j] + dp[i + 1][j][k])
                        if j + 1 < cols:
                            ans = max(ans, coins[i][j] + dp[i][j + 1][k])

                        # option 2: neutralize if we can
                        if k > 0:
                            if i + 1 < rows:
                                ans = max(ans, dp[i + 1][j][k - 1])
                            if j + 1 < cols:
                                ans = max(ans, dp[i][j + 1][k - 1])

                    dp[i][j][k] = ans

     
        return dp[0][0][2]
    

########################################
# 3661. Maximum Walls Destroyed by Robots
# 03APR26
#########################################
#almost...
#i need unique walls
#in this case id need to track which walls get hit to prevent double counting
import bisect
class Solution:
    def maxWalls(self, robots: List[int], distance: List[int], walls: List[int]) -> int:
        '''
        if we have a robot at robots[i]
            its bullet can span (robots[i] - distance[i] or robots[i] + distance[i]) and destorys all walls
            if another robot is between those two (it stops)
        
        this is dp with binary search
        for each robot, try shooting left or right and take the most walls you can hit
        binary search for the wall that it can hit, and binary search for the closest robot
        
        '''
        walls.sort()
        arr = [(r,d) for r,d in zip(robots,distance)]
        arr.sort(key = lambda x: x[0])
        robots = [r for r,_ in arr] #we now robots sorted ascending, walls sorted asceding, and the bullet dist for each robot
        memo = {}

        def dp(i):
            n = len(robots)
            if i >= len(robots):
                return 0
            if i in memo:
                return memo[i]
            r,d = arr[i]
            
            #left shoot
            left_stop = robots[i-1] if i > 0 else float('-inf')
            left_bound = max(r-d,left_stop)

            left_count = bisect.bisect_left(walls,r) - bisect.bisect_left(walls,left_bound)
            #right shoot
            right_stop = robots[i+1] if i < n-1 else float('inf')
            right_bound = min(r+d,right_stop)
            
            right_count = bisect.bisect_right(walls, right_bound) -bisect.bisect_right(walls, r)
            ans = max(left_count,right_count) + dp(i+1)
            memo[i] = ans
            return ans

        return dp(0)
    
##############################################
# 2069. Walking Robot Simulation II
# 06APR26
##############################################
#im off by one somwehre
class Robot:

    def __init__(self, width: int, height: int):
        '''
        #robot only moves along permieter
        #if on bottom edge, its facing East
        #if on right edge, its facing North
        #if on top edge, its facing West
        #if on left edge, its facing South
        #but there's a small transition period, where if its on the corner, it coudl still be facing the same previous direction
        think like this BBBBLLLLLLTTTTTRRR, and then this will repeat
        put this into a coornidate set and we can advance by one
        it turns and retires the step
        '''
        self.width,self.height = width,height
        self.coords = []
        #fill bottom
        for x in range(width):
            self.coords.append((x,0))
        #fill right
        for y in range(1,height):
            self.coords.append((width-1,y))
        #fill top
        for x in range(width-2,-1,-1):
            self.coords.append((x,height-1))
        #fill left
        for y in range(height-2,0,-1):
            self.coords.append((y,0))
        self.idx = 0
        self.n = len(self.coords)


    def step(self, num: int) -> None:
        #the problem is the num can be very big, and it might be called alot
        self.idx = (self.idx + num) % self.n

    def getPos(self) -> List[int]:

        return self.coords[self.idx]

    def getDir(self) -> str:
        x,y = self.coords[self.idx]
        if y == 0:
            return "East"
        if x == self.width-1:
            return "North"
        if y == self.height-1:
            return "West"
        return "South"


# Your Robot object will be instantiated and called as such:
# obj = Robot(width, height)
# obj.step(num)
# param_2 = obj.getPos()
# param_3 = obj.getDir()

class Robot:

    def __init__(self, width: int, height: int):
        '''
        #robot only moves along permieter
        #if on bottom edge, its facing East
        #if on right edge, its facing North
        #if on top edge, its facing West
        #if on left edge, its facing South
        #but there's a small transition period, where if its on the corner, it coudl still be facing the same previous direction
        think like this BBBBLLLLLLTTTTTRRR, and then this will repeat
        put this into a coornidate set and we can advance by one
        it turns and retires the step
        '''
        self.width,self.height = width,height
        self.coords = []
        self.dirs = []
        #fill bottom
        for x in range(width):
            self.coords.append((x,0))
            self.dirs.append("East")
        #fill right
        for y in range(1,height):
            self.coords.append((width-1,y))
            self.dirs.append("North")
        #fill top
        for x in range(width-2,-1,-1):
            self.coords.append((x,height-1))
            self.dirs.append("West")
        #fill left
        for y in range(height-2,0,-1):
            self.coords.append((0,y))
            self.dirs.append("South")
        self.idx = 0
        self.moved = False
        self.n = len(self.coords)
        self.dirs[0] = "South"


    def step(self, num: int) -> None:
        self.moved = True
        #the problem is the num can be very big, and it might be called alot
        self.idx = (self.idx + num) % self.n

    def getPos(self) -> List[int]:

        return self.coords[self.idx]

    def getDir(self) -> str:
        if not self.moved:
            return "East"
        return self.dirs[self.idx]

###################################################
# 1433. Check If a String Can Break Another String 
# 07APR26
###################################################
class Solution:
    def checkIfCanBreak(self, s1: str, s2: str) -> bool:
        '''
        we need to find a permutation of s1 or s2, wich that perm(s1) breaks s2 or perm(s2) breaks s1
        '''
        s1 = sorted(list(s1))
        s2 = sorted(list(s2))

        if all([u >= v for (u,v) in zip(s1,s2)]):
            return True
        if all([u >= v for (u,v) in zip(s2,s1)]):
            return True
        
        return False
    
##################################################
# 3653. XOR After Range Multiplication Queries I
# 08APR2026
###################################################
class Solution:
    def xorAfterQueries(self, nums: List[int], queries: List[List[int]]) -> int:
        '''
        brute force first
        '''
        mod = 10**9 + 7
        for l,r,k,v in queries:
            for i in range(l,r+1,k):
                nums[i] = (nums[i] * v) % mod

        ans = nums[0]
        for n in nums[1:]:
            ans = ans ^ n
        
        return ans

######################################################
# Minimum Distance Between Three Equal Elements II
# 11APR26
######################################################
class Solution:
    def minimumDistance(self, nums: List[int]) -> int:
        '''
        put into hashmap the nums to a list of their indices
        say we have (i,j,k)
        evidently we have:
            abs(i - j) + abs(j - k) + abs(k - i) simplifies to 2 * (max(i, j, k) - min(i, j, k)), but why?
            if we order them its just
            (j - i) + (k-j) +(k-i), after sorting of course

            but (j-i) + (k-j) = (k-i)
            so we have (k-i) + (k-i)
            and this is just 2*max(i, j, k) - min(i, j, k)
        '''
        mapp = defaultdict(list)
        ans = float('inf')
        for i, num in enumerate(nums):
            mapp[num].append(i)
        
        for k,v in mapp.items():
            if len(v) >= 3:
                v = sorted(v)
                n = len(v)
                for i in range(n-2):
                    ans = min(ans, 2*(v[i+2] - v[i]))
        
        return ans if ans != float('inf') else -1
    
##########################################################
# 1320. Minimum Distance to Type a Word Using Two Fingers
# 12APR26
##########################################################
class Solution:
    def minimumDistance(self, word: str) -> int:
        '''
        we have two fingers,
        for each finger we can 
        * press with no cost
        * move from some letter with cost
        states are this: 
        (i,j,k)
        smallest movements when you have one finger on i-th char and the other one on j-th char already having written k first characters from word.
        if i have fingers on some (i,j)
        then find min cost to new ii jj from (i,j)
        the thing is how do i start since the first move doesn't cost
        '''
        #first generate dists matrix
        cols = 6
        rows = 5
        chars_to_cell = {}
        for i in range(26):
            ch1 = chr(ord('A') + i)
            i1,j1 = i // 6, i % 6
            for j in range(26):
                ch2 = chr(ord('A') + j)
                i2,j2 = j // 6, j % 6
                dist = abs(i1 - i2) + abs(j1 - j2)
                chars_to_cell[(ch1,ch2)] = dist
        
        memo = {}

        def dp(i, j, k):
            if k == len(word):
                return 0
            if (i, j, k) in memo:
                return memo[(i, j, k)]

            target = word[k]

            #first fingerr
            cost1 = chars_to_cell.get((i, target),0) + dp(target, j, k + 1)
            #second
            cost2 = chars_to_cell.get((j, target),0) + dp(i, target, k + 1)

            ans = min(cost1, cost2)
            memo[(i, j, k)] = ans
            return ans

        return dp(None, None, 0)

##############################################
# 1559. Detect Cycles in 2D Grid
# 13APR26
###############################################
class Solution:
    def containsCycle(self, grid):
        '''
        dfs from each cell (i,j) and maintai its the same char
        '''
        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(i, j, pi, pj, ch):
            if (i, j) in visited:
                return True

            visited.add((i, j))

            for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == ch:
                    if (ni, nj) == (pi, pj):
                        continue
                    if dfs(ni, nj, i, j, ch):
                        return True

            return False

        for i in range(rows):
            for j in range(cols):
                if (i, j) not in visited:
                    if dfs(i, j, -1, -1, grid[i][j]):
                        return True

        return False
    
###############################################
# 2585. Number of Ways to Earn Points
# 14APR26
################################################
class Solution:
    def waysToReachTarget(self, target: int, types: List[List[int]]) -> int:
        '''
        compress types array then knapsack
        '''
        memo = {}
        mod = 10**9 + 7

        def dp(i,score):
            if score == target:
                return 1
            if i >= len(types):
                if score == target:
                    return 1
                return 0
            if score > target:
                return 0
            if (i,score) in memo:
                return memo[(i,score)]
            ways = 0
            count,mark = types[i] 
            for solved in range(count+1):
                ways += dp(i+1,score + solved*mark)
                ways %= mod
            ways %= mod
            memo[(i,score)] = ways
            return ways

        return dp(0,0)
    
###############################################################
# 2515. Shortest Distance to Target String in a Circular Array
# 15APR25
###############################################################
class Solution:
    def closestTarget(self, words: List[str], target: str, startIndex: int) -> int:
        '''
        its either the distance going clockwise or counter clockwise
        go left or go right
        '''
        n = len(words)
        ans = float('inf')
        for i,w in enumerate(words):
            if w == target:
                cw = (i - startIndex + n) % n
                ccw = (startIndex - i + n) % n
                ans = min(ans,cw,ccw)
        
        return ans if ans != float('inf') else -1

#################################################
# 3488. Closest Equal Element Queries
# 16APR26
#################################################
class Solution:
    def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        '''
        for each num, map to a list of indices
        then for the query look into the bucket of indices and find the min dist
        if i have a negative index, i just return it back to an index between 0 and n-1
            its just + n then % n
        '''
        n = len(nums)
        mapp = defaultdict(list)
        
        for i,num in enumerate(nums):
            mapp[num].append(i)
        
        for num,idxs in mapp.items():
            #add in aritifcal indices
            idxs.insert(0, idxs[-1] - n)
            idxs.append(idxs[1] + n)

        ans = []
        for i in range(len(queries)):
            x = nums[queries[i]]
            pos_list = mapp[x]
            if len(pos_list) == 3:
                ans.append(-1)
                continue
            pos = bisect.bisect_left(pos_list, queries[i])
            ans.append(min(
                pos_list[pos + 1] - pos_list[pos],
                pos_list[pos] - pos_list[pos - 1],
            ))

        return ans
    
#######################################################
# 3761. Minimum Absolute Distance Between Mirror Pairs
# 17APR26
#######################################################
class Solution:
    def minMirrorPairDistance(self, nums: List[int]) -> int:
        '''
        scan left to right and record last index in hashmap
        we need to store the index of the reverse number, not the actual number
            if a number was seen in this hashmap, it means its reverse was present!
        trickyyyyyy
        '''
        def reverse(x):
            revd = 0
            while x:
                revd *= 10
                revd += x % 10
                x //= 10
            return revd
        
        ans = float('inf')
        n = len(nums)
        last_seen = defaultdict()
        #left to right
        for i,num in enumerate(nums):
            revd = reverse(num)
            if num in last_seen:
                ans = min(ans,abs(i - last_seen[num]))
            last_seen[revd] = i

        return ans if ans != float('inf') else -1
            
##############################################
# 3783. Mirror Distance of an Integer
# 18APR26
###############################################
class Solution:
    def mirrorDistance(self, n: int) -> int:
        '''
        blahhhh
        '''
        def reverse(x):
            revd = 0
            while x:
                revd *= 10
                revd += x % 10
                x //= 10
            return revd
        
        return abs(n - reverse(n))
    
#############################################
# 1855. Maximum Distance Between a Pair of Values
# 19APR26
##############################################
class Solution:
    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        fix and i in nums1, then binary search in nums2 from j, where j == i to the end
        if we fix an i, we want the rightmost j, such that nums1[i] <= nums2[j]
        '''
        ans = 0
        for i in range(len(nums1)):
            cand_ans = -1
            left = i
            right = len(nums2) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if nums2[mid] >= nums1[i]:
                    cand_ans = mid
                    left = mid + 1
                else:
                    right = mid - 1
            if cand_ans != -1:
                ans = max(ans, cand_ans - i)
        
        return ans
    
class Solution:
    def maxDistance(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can use two pointers
        '''
        ans = 0
        i,j = 0,0

        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                ans = max(ans, j-i)
                j += 1
            else:
                i += 1
        
        return ans
    
#########################################################
# 1722. Minimize Hamming Distance After Swap Operations
# 21APR26
###########################################################
class DSU:
    def __init__(self, n):
        self.parents = [i for i in range(n)]
        self.rank = [1 for _ in range(n)]

    def find(self, x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])  # path compression
        return self.parents[x]
    
    def union(self, x, y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        if x_par == y_par:
            return False
        
        if self.rank[x_par] >= self.rank[y_par]:
            self.parents[y_par] = x_par
            self.rank[x_par] += self.rank[y_par]
        else:
            self.parents[x_par] = y_par
            self.rank[y_par] += self.rank[x_par]
        
        return True

class Solution:
    def minimumHammingDistance(self, source: List[int], target: List[int], allowedSwaps: List[List[int]]) -> int:
        '''
        we can only swap in the source array
        if node indices are part of the same component, they can be freely swapped in any order
        find connected components on indices
        then for each component, find commone elements in taget
        elements not in common will contribute to total hamming distance
        can use dfs/bfs/union find
        call find(i) to find parent root
        parents array are just intermediate parents
        '''
        n = len(source)
        dsu = DSU(n)
        for u,v in allowedSwaps:
            dsu.union(u,v)
        
        comps = defaultdict(list)
        for i,v in enumerate(dsu.parents):
            comps[dsu.find(i)].append(i)
        #find common
        uncommon = 0
        for g in comps.values():
            left,right = Counter(),Counter()
            for i in g:
                left[source[i]] += 1
                right[target[i]] += 1
            diff = left - right
            if len(diff) == 0:
                continue
            for v in diff.values():
                uncommon += v

        return uncommon
    
#############################################
# 2452. Words Within Two Edits of Dictionary
# 22APR26
##############################################
#TLE
class Solution:
    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        '''
        try one edit, and then two edits
        then for each i,j pair change letter
        '''
        dictionary = set(dictionary)
        ans = []

        def helper(word):
            n = len(word)
            w = list(word)
            for i in range(n):
                for idx in range(26):
                    ch = chr(ord('a') + idx)
                    old = w[i]
                    w[i] = ch
                    yield "".join(w)
                    w[i] = old
            for i in range(n):
                for idx1 in range(26):
                    ch1 = chr(ord('a') + idx1)
                    old1 = w[i]
                    for j in range(i+1,n):
                        for idx2 in range(26):
                            ch2 = chr(ord('a') + idx2)
                            old2 = w[j]
                            w[i] = ch1
                            w[j] = ch2
                            yield "".join(w)
                            w[i] = old1
                            w[j] = old2

        for word in queries:
            #one edit
            for neigh in helper(word):
                if neigh in dictionary:
                    ans.append(word)
                    break            
        return ans
    
class Solution:
    def twoEditWords(self, queries: List[str], dictionary: List[str]) -> List[str]:
        '''
        compre each word in queries with each word in dictionary
        if we are at <= 2 diff chagnes away, we can make it
        '''
        ans = []
        for q in queries:
            for d in dictionary:
                diff = 0
                #they are all euqal length
                for ch1,ch2 in zip(q,d):
                    if ch1 != ch2:
                        diff += 1
                        if diff > 2:
                            break

                if diff <= 2:
                    ans.append(q)
                    break
        
        return ans

#############################################
# 2615. Sum of Distances
# 23APR26
#############################################
class Solution:
    def distance(self, nums: List[int]) -> List[int]:
        '''
        we can store all indices for some num in nums
        say we have indics a,b,c for some num x at a
        what we want is
        sum(abs(a-b),abs(b-c),abs(a-c))
        for indices j < i (before i)
        its: (arr[i] - arr[0]) + (arr[i] - arr[1]) + ... + (arr[i] - arr[i-1])
        there are i elements on the left, and each contributes arr[i]
        each contributes arr[i] i times
        So total becomes:
        i * arr[i] − (sum of all left elements)

        That’s why:
        left = i * arr[i] - pref[i-1]

        pref[i-1] already stores:
        (arr[0] + arr[1] + ... + arr[i-1])

        for indices j > i, on the right
        (arr[i+1] - arr[i]) + (arr[i+2] - arr[i]) + ...

        Here:

        Sum of right elements = (arr[i+1] + arr[i+2] + ...)
        Number of elements = (siz - i - 1)
        So:
        total = (sum of right elements) − (count * arr[i])

        That’s why:
        right = (pref[last] - pref[i]) - (siz-i-1) * arr[i]

        We just add both contributions:
        res[arr[i]] = left + right

        '''
        mapp = defaultdict(list)
        n = len(nums)
        ans = [0]*n

        for i,num in enumerate(nums):
            mapp[num].append(i)
        
        for arr in mapp.values():
            size = len(arr)
            pref_sum = [0]*size
            pref_sum[0] = arr[0]

            for i in range(1,size):
                pref_sum[i] = pref_sum[i-1] + arr[i]
            
            for i in range(size):
                left,right = 0,0
                
                if i > 0:
                    left = i*arr[i] - pref_sum[i-1]
                if i < size - 1:
                    right = (pref_sum[size-1] - pref_sum[i]) - (size - i -1)*arr[i]

                ans[arr[i]] = left + right        
        return ans

#############################################
# 2833. Furthest Point From Origin
# 23APR26
###############################################
class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        '''
        cant spaces and apply directions
        '''
        start = 0
        spaces = 0
        for ch in moves:
            if ch == 'L':
                start -= 1
            elif ch == 'R':
                start += 1
            else:
                spaces += 1
        
        return abs(start) + spaces

####################################################
# 3464. Maximize the Distance Between Points on a Square
# 25APR26
####################################################
#close
# :(
class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        '''
        points are all on the boundary of the square with length side and bottom left is (0,0)
        pick k points such that the min manhat distance between anytwo points is maximum
        binary search on for the manhat dist by seeing if can pick k points, imagine points are in order
        if i pick a point i, i need to find another j such that manhat_dist(i,j) is a large is possible but i need k points
        furthest two points will be axis unaligned
        the set of k points, if there is one
        '''
        left = []
        #x = 0, and increasing y
        for x,y in points:
            if x == 0:
                left.append((x,y))
        left.sort(key = lambda x: x[1])
        #y = size and increasing x
        top = []
        for x,y in points:
            if y == side:
                top.append((x,y))
        top.sort(key = lambda x: x[0])
        #x = side, y going down
        right = []
        for x,y in points:
            if x == side:
                right.append((x,y))
        right.sort(key = lambda x: -x[1])

        #y = 0, x going down
        bottom = []
        for x,y in points:
            if y == 0:
                bottom.append((x,y))
        bottom.sort(key = lambda x: -x[0])
        combined = left + top + right + bottom
        clockwise = [list(combined[0])]
        for i in range(1,len(combined)):
            x,y = combined[i]
            if [x,y] != clockwise[-1]:
                clockwise.append([x,y])
        if clockwise[0] == clockwise[-1]:
            clockwise.pop(-1)
        n = len(clockwise)
        arr = clockwise + clockwise
        ans = float('-inf')
        for i in range(n):
            local_ans = float('inf')
            for j in range(i+1,i+k):
                p1,p2 = arr[j],arr[j-1]
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                local_ans = min(local_ans,dist)
            print(local_ans)
            ans = max(ans,local_ans)
            #sub = arr[i:i+k]
            #print(sub)
        return ans
        
class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        arr = []
        
        for x, y in points:
            if x == 0:
                arr.append(y)
            elif y == side:
                arr.append(side + x)
            elif x == side:
                arr.append(side * 3 - y)
            else:
                arr.append(side * 4 - x)
        
        arr.sort()
        
        def check(limit: int) -> bool:
            perimeter = side * 4
            for start in arr:
                end = start + perimeter - limit
                cur = start
                for _ in range(k - 1):
                    idx = bisect_left(arr, cur + limit)
                    if idx == len(arr) or arr[idx] > end:
                        cur = -1
                        break
                    cur = arr[idx]
                if cur >= 0:
                    return True
            return False
        
        lo, hi = 1, side
        ans = 0
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                lo = mid + 1
                ans = mid
            else:
                hi = mid - 1
                
        return ans

###############################################
# 1391. Check if There is a Valid Path in a Grid
# 27APR26
#################################################
#almost :(
class Solution:
    def hasValidPath(self, grid: List[List[int]]) -> bool:
        '''
        each type can only go into certain types
        and each can only do into a direction
        '''
        dirrs = {
            1: [(0, -1), (0, 1)],      # left, right
            2: [(-1, 0), (1, 0)],      # up, down
            3: [(0, -1), (1, 0)],      # left, down
            4: [(0, 1), (1, 0)],       # right, down
            5: [(0, -1), (-1, 0)],     # left, up
            6: [(0, 1), (-1, 0)]       # right, up
        }
                
        allowed = {
            1: [1,3,4,5,6],
            2: [2,3,4,5,6],
            3: [1,2,4,5,6],
            4: [1,2,3,5,6],
            5: [1,2,3,4,6],
            6: [1,2,3,4,5]
        }

        memo = {}
        seen = set()
        rows,cols = len(grid),len(grid[0])
        def check(i,j,rows,cols,seen):
            if 0 <= i < rows and 0 <= j < cols and (i,j) not in seen:
                return True
            return False

        def can_do(i,j,seen):
            rows,cols = len(grid),len(grid[0])
            if (i,j) == (rows-1,cols-1):
                return True
            if (i,j) in memo:
                return memo[(i,j)]
            seen.add((i,j))
            curr = grid[i][j]
            possible_dirrs = dirrs[curr]
            possible_neighs = allowed[curr]
            for di,dj in possible_dirrs:
                ii,jj = i + di, j + dj
                if check(ii,jj,rows,cols,seen):
                    if can_do(ii,jj,seen) and grid[ii][jj] in possible_neighs:
                        memo[(ii,jj)] = True
                        return True

            seen.remove((i,j))
            memo[(ii,jj)] = False
            return False


        return can_do(0,0,seen)
    

class Solution:
    def hasValidPath(self, grid: List[List[int]]) -> bool:
        '''
        if i go from A to B, then B to A must be allowed in the reverse direction
        its a valid neighbor if both are reachable from one another
        i could have made a another graph (i,j) and check if all 4 direction neighbors are reachable
        '''
        dirrs = {
            1: [(0, -1), (0, 1)],      # left, right
            2: [(-1, 0), (1, 0)],      # up, down
            3: [(0, -1), (1, 0)],      # left, down
            4: [(0, 1), (1, 0)],       # right, down
            5: [(0, -1), (-1, 0)],     # left, up
            6: [(0, 1), (-1, 0)]       # right, up
        }

        rows, cols = len(grid), len(grid[0])
        seen = set()

        def dfs(i, j):
            if (i, j) == (rows - 1, cols - 1):
                return True

            seen.add((i, j))

            for di, dj in dirrs[grid[i][j]]:
                ni, nj = i + di, j + dj

                if 0 <= ni < rows and 0 <= nj < cols and (ni, nj) not in seen:
                    if (-di, -dj) in dirrs[grid[ni][nj]]:
                        if dfs(ni, nj):
                            return True

            return False

        return dfs(0, 0)
    
class Solution:
    def hasValidPath(self, grid: List[List[int]]) -> bool:
        '''
        if i go from A to B, then B to A must be allowed in the reverse direction
        its a valid neighbor if both are reachable from one another
        i could have made a another graph (i,j) and check if all 4 direction neighbors are reachable
        '''
        dirrs = {
            1: [(0, -1), (0, 1)],      # left, right
            2: [(-1, 0), (1, 0)],      # up, down
            3: [(0, -1), (1, 0)],      # left, down
            4: [(0, 1), (1, 0)],       # right, down
            5: [(0, -1), (-1, 0)],     # left, up
            6: [(0, 1), (-1, 0)]       # right, up
        }

        rows, cols = len(grid), len(grid[0])
        seen = set()
        q = deque([(0,0)])

        while q:
            i,j = q.popleft()

            if (i, j) == (rows - 1, cols - 1):
                return True

            seen.add((i, j))

            for di, dj in dirrs[grid[i][j]]:
                ni, nj = i + di, j + dj

                if 0 <= ni < rows and 0 <= nj < cols and (ni, nj) not in seen:
                    if (-di, -dj) in dirrs[grid[ni][nj]]:
                        q.append((ni,nj))
            
        return False
    
###############################################
# 811. Subdomain Visit Count
# 28APR26
###############################################
class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        '''
        hash map and add counts
        split on empty space, then split on .
        then we need suffixes of each
        then add up counts
        '''
        hits = Counter()

        for cp in cpdomains:
            count,domain = cp.split(" ")
            subs = domain.split(".")
            n = len(subs)
            for i in range(1,n+1):
                temp = ".".join(subs[-i:])
                hits[temp] += int(count)
        ans = []
        for k,v in hits.items():
            temp = str(v)+" "+k
            ans.append(temp)
        return ans