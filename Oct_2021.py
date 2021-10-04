####################################
#01_OCT_2021
#1143. Longest Common Subsequence
###################################
#recursive knapsack solution
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        '''
        i can use recursion, two pointers i and j,
        if i and j match, advance them both, if they dont, there are two options
        stay at i, move j
        stat at j, move i
        if either of these pointers got the end, we have a subsequence, which is just
        '''
        
        memo = {}
        def rec(i,j):
            #ending case
            if i == len(text1) or j == len(text2):
                return 0
            #retrieve
            if (i,j) in memo:
                return memo[(i,j)]

            #matching, 1 + advance both
            if text1[i] == text2[j]:
                res = 1 + rec(i+1,j+1)
                memo[(i,j)] = res
                return res
            
            #two options
            first = rec(i+1,j)
            second = rec(i,j+1)
            res = max(first,second)
            memo[(i,j)] = res
            return res
        
        return rec(0,0)

#dp solution, 2d array
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        '''
        dp solution, just translate from the top down recursive approach
        remember we work backwards
        '''
        M = len(text1)
        N = len(text2)
        
        dp = [[0]*(N+1) for _ in range(M+1)]
        #remember we padded the dp array with one extra col and row with zeros
        for i in range(M-1,-1,-1):
            for j in range(N-1,-1,-1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i+1][j+1]
                else:
                    dp[i][j] = max(dp[i+1][j],dp[i][j+1])
        
        return dp[0][0]

#dp solution space optimized
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        '''
        we can also optimize space 
        instead of using the whole dp array, just keep current and previous rows
        '''
        #first check if text1 doest not reference the shoortest string, swap them
        if len(text2) < len(text1):
            #swap
            text1,text2 = text2,text1
        
        M = len(text1)
        N = len(text2)
        
        prev = [0]*(N+1)
        curr = [0]*(N+1)
        #remember we padded the dp array with one extra col and row with zeros
        for i in range(M-1,-1,-1):
            for j in range(N-1,-1,-1):
                if text1[i] == text2[j]:
                    curr[j] = 1 + prev[j+1]
                else:
                    curr[j] = max(prev[j],curr[j+1])
            #update rwos
            prev,curr = curr,prev
        
        return prev[0]
                
##################################
# 01_OCT_2021
# 1428. Leftmost Column with at Least a One
#############################
#exhausted calls, 
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        '''
        we are given a row sorted binary matrix, so its in a non decreasing order along the rows
        we want the index of the leftmost columnwith a 1 in it
        can only access matrix using the API
        brute force way would be to access all elements in the object
        we want to go down columns starting with leftmost column, if when going down this row, we hit a 1, return that column index
        otherwise -1
        '''
        rows,cols = binaryMatrix.dimensions()
        for col in range(cols):
            for row in range(rows):
                if binaryMatrix.get(row,col) == 1:
                    return col
        return -1

#yessss
#note we could also limit the row searching by keeping track of what col index a 1 was foud in
#then use that index as the right most bound
#keep updating right bound if we have a more left one
class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        '''
        repeatedly making calls to API won't work, cheeky way, cache calls in the constructor first
        then check matrix, but that's stupid
        i could binary search along each row for all rows
        '''
        rows,cols = binaryMatrix.dimensions()
        smallest_i = cols
        for row in range(rows):
            #binary seach along a row
            left = 0
            right = cols - 1
            # right = smallest_i - 1
            while left < right:
                mid = left + (right - left) // 2
                val = binaryMatrix.get(row,mid)
                #if its a 1, i can still look in the lower half
                if val == 1:
                    right = mid
                else:
                    left = mid + 1
            if binaryMatrix.get(row,left) == 1:
                smallest_i = min(smallest_i,left)
        
        return smallest_i if smallest_i != cols else -1

#Linear in M+N time
class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        '''
        this is similar to searching a 2d matrix
        the most optimal approach is saddle back search
        we need to start from a point in the matrix, where we can find a non increasing element
        and a non decreasing element
        
        '''
        rows,cols = binaryMatrix.dimensions()
        curr_row = 0
        curr_col =  cols - 1
        
        #while we are in bounds
        while curr_row < rows and curr_col >= 0:
            #if we find a 1, stay in row, and go down column
            if binaryMatrix.get(curr_row,curr_col) == 1:
                curr_col -= 1
            else:
                curr_row += 1
        
        return curr_col + 1 if curr_col != cols - 1 else -1

#########################################
# 02_OCT_21
# 174. Dugeon Game
##########################################
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        '''
        we cam use dp to solve this problem
        intution:
        we the knigh were to start at the bottomr right cell of the dungeon, he would need to have at leat 1 more than that negative value, lets call this value curr
        from curr, the knight could have come from above, UP
        or from the left, LEFT (remember the knight can only move down and to the left)
        if at UP, we gained some health, we would only need curr - health there
        if at LEFT, we also gain health, and the difference beteween what we gain > what we would need, well would need at least 1
        so, dp[i][j] answers the question: what is the min health needed to get to the bottom right, if i start at i,j
        now, we need to examine from UPLEFT (UL)
        if negative we add from the next down or right cells, and we take the max,
        each cell will be marked with two minimal healths points (one for each actions)
        algo:
            define dp array, dp[row][col] answers question min health needed to reach des
            if dp[row][col] is negative, i need at least  1 - dp[row][col] (think really fucking hard about why)
            if dpr[row][col] is postive, i gain from this, and would only need at least 1 to get here
            the rules for filling out the dp:
                if possible by taking right step from curr cell, the knight might need right health points
                if possible by taking the don step, the knight might need down_health points
                if either of the options exists, we take the min of the two opetions
                if non of the alternatives exsits, we are at the desitation cell
                    * if curr cell is postives, then 1 health is enough
                    if curr cell is negative, then know should possese one 1 - health points
            declare dp array values with inf
        '''
        rows = len(dungeon)
        cols = len(dungeon[0])
        
        dp = [[float('inf')]*cols for _ in range(rows)]
        
        #get min health function
        def get_min_health(currCell,nextRow,nextCol):
            #outside of rows or cols:
            if nextRow >= rows or nextCol >= cols:
                return float('inf')
            return max(1,dp[nextRow][nextCol] - currCell)
        
        #starting from end
        for row in range(rows-1,-1,-1):
            for col in range(cols-1,-1,-1):
                currCell = dungeon[row][col]
                
                #get possible valies
                right_health = get_min_health(currCell,row,col+1)
                down_health = get_min_health(currCell, row+1,col)
                next_health = min(right_health,down_health)
                
                if next_health != float('inf'):
                    min_health = next_health
                else:
                    min_health = 1 if currCell >= 0 else (1 - currCell)
                dp[row][col] = min_health
        
        return dp[0][0]

#another way using inplace for dungeon
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        '''
        using in place dungeon array for dp
        
        '''
        rows = len(dungeon)
        cols = len(dungeon[0])
        
        #first set the destinaion cell to the minimum necesary
        dungeon[rows-1][cols-1] = max(1,1-dungeon[rows-1][cols-1])
        
        #on the last column, we can only go down, fill this up first
        for row in range(rows-2,-1,-1):
            dungeon[row][cols-1] = max(1,dungeon[row+1][cols-1] - dungeon[row][cols-1])
            
        #for the last row, we can only go left
        for col in range(cols-2,-1,-1):
            dungeon[rows-1][col] = max(1,dungeon[rows-1][col+1] - dungeon[rows-1][col])
            
        #now fill in the reamin
        for row in range(rows-2,-1,-1):
            for col in range(cols-2,-1,-1):
                min_left_right = min(dungeon[row+1][col],dungeon[row][col+1])
                res = max(1,min_left_right - dungeon[row][col])
                dungeon[row][col] = res
        
        return dungeon[0][0]

################################
# 03_OCT_21
# 55. Jump Game
################################
#BFS gives TLE
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        '''
        i can try using BFS and for each jump length, try making that jump length
        return true if i hit the the last index
        '''
        N = len(nums)
        seen = set([0])
        q = deque([0])
        
        while q:
            curr = q.popleft()
            if curr == N - 1:
                return True
            for step in range(nums[curr]+1):
                neigh = curr + step
                if neigh < N:
                    if neigh not in seen:
                        seen.add(neigh)
                        q.append(neigh)
        
        return False

#naive backtracking, i.e backtracking with memo
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        '''
        recursive solution, without memo
        using inefficient back tracking
        '''
        N = len(nums)
        memo = {}
        
        def dfs(i):
            if i == N - 1:
                return True
            if i in memo:
                return memo[i]
            
            max_jump = min(i + nums[i], N - 1)
            #could also switch lines to
            '''
            for j in range(max_jump,i,-1):
           	'''
            for j in range(i+1,max_jump+1):
                if dfs(j):
                    return True
            memo[i] = False
            return memo[i]
        
        return dfs(0)


#efficient recusrive
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        '''
        recursive solution, without memo
        using inefficient back tracking
        to make the memo more efficient, we mark each position as good or bad
        then call only from good positions, this prunes the call tree
        bad jumps leads us to spot where we have zero jumps, FUCKING STUCK
        
        '''
        N = len(nums)
        memo = [0]*N
        #states are 0:unknown, 1: Good, 2: bad
        memo[N-1] = 1
        
        def dfs(i):
            if memo[i] != 0:
                if memo[i] == 1:
                    return True
                else:
                    return False
            
            max_jump = min(i + nums[i],N-1)
            for j in range(i+1,max_jump+1):
                if dfs(j):
                    memo[j] == 1
                    return True
            
            memo[i] = 2
            return False
        
        return dfs(0)

#bottom up dp
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        '''
        now lets convert this effecient recursive to top down dp
        the key observation is that we only make a jump to the right
        this means that if we start from the right of the array, every time we will query a position to our right
        that position has laready been determined as being GOOD or BAD
        
        '''
        N = len(nums)
        memo = [0]*N
        #states are 0:unknown, 1: Good, 2: bad
        memo[N-1] = 1
        
        #now we start from the seond to last position
        for i in range(N-2,-1,-1):
            max_jump = min(i + nums[i],N-1)
            for j in range(i+1,max_jump+1):
                if memo[j] == 1:
                    memo[i] == 1
                    break
        
        return memo[0] == 1

#greedy
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        '''
        greedy:
        if when we hit a good poistion, coming from a previous good positino
        we only need to start from the first one hit!, we don't care any good positions after the first good one
        LINEAR TIME INTUTION COMES FROM THIS!
        keep track of the left most good poistion
        we also stop from using an array to store our results
        from right to left, for each position, check if there is a potential to jump that reaches a good index
        i.e currPos + nums[currPos] >= lefmostGood index
        if we can reach a good, than current posiiton is good, and this new good must be the next left most good
        '''
        N = len(nums)
        lastGood = N - 1
        for i in range(N-1,-1,-1):
            #if we can get to the last good, update last good
            if i + nums[i] >= lastGood:
                lastGood = i
        
        return lastGood == 0
        
################################
# 04_OCT_21
# 463. Island Perimeter
################################
#close but not really correct
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        '''
        there is exactly one island, and the island doest not have any lakes
        i can use dfs the mark the cells that make up the eldn
        scan the grid to find the first 1, which should contribute 4 edges
        after than whenever i dfs, i can decrease by 1 and add 4 whenver i can dfs
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        self.perim = 0
        
        def dfs(r,c,seen):
            seen.add((r,c))
            self.perim += 4
            for dx,dy in dirrs:
                neigh_x = r + dx
                neigh_y = c + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #is a one
                    if grid[neigh_x][neigh_y] == 1 and (neigh_x,neigh_y) not in seen:
                        seen.add((neigh_x,neigh_y))
                        self.perim -= 1
                        dfs(neigh_x,neigh_y,seen)

                        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    dfs(i,j,set())
                    return self.perim

#zeros surrounding a 1 increase perim by 1
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        '''
        while traversing the grid, if we hit a 1, find the zeros surroundint it
        for each 0 around 1, add 1
        if its a 1 mark as visitied and dfs from there
        and up all the edges surrding that 1
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(r,c):
            #out of bounds or zero, adds 1 to this perimiter
            if (r < 0) or (r >= rows) or (c < 0) or (c >= cols) or grid[r][c] == 0:
                return 1
            #if its a 1, mark as visited and dfs
            if grid[r][c] == 1:
                grid[r][c] = -1
                ans = 0
                for dx,dy in dirrs:
                    neigh_r = r + dx
                    neigh_c = c + dy
                    ans += dfs(neigh_r,neigh_c)
                return ans
            else:
                return 0
            
        
        perim = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    perim += dfs(i,j)
                    return perim

#simple counting
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        '''
        one way is to count the land and water cells
        if we are at a land cell, we know we can add a max of 4
        but if we are bounded by up,down,left,right
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        perim = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    perim += 4
                    #now lets decrease if we can
                    for dx,dy in dirrs:
                        neigh_x = i + dx
                        neigh_y = j + dy
                        if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                            if grid[neigh_x][neigh_y] == 1:
                                perim -= 1
        
        return perim

#efficient counting
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        '''
        we can optimize the counting, by realzing that we scan left to right, top to bottom
        so we only ever need to look to the left and up
        if we hit an island we know we can add 4
        if that island has a left nieghbor, decrement by 2
        if island has up neighbor decreament by 2
        WHY? becasue we share and edge along up edge and left edge
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        perim = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    perim += 4
                    
                    #check up
                    if i > 0 and grid[i-1][j] == 1:
                        perim -= 2
                    if j > 0 and grid[i][j-1] == 1:
                        perim -= 2
        
        return perim
        