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