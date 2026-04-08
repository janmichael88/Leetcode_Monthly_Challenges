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