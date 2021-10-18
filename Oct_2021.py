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
      
#extra problem
#################################
# 04OCT21
# 27. Remove Element
#################################
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        '''
        i can use two pointers on this guy, and send the element to be remvoed to the end of the array
        so long as that last element is not the element to be removed
        just move the elements, the online judger will take nums[:k]
        then just set nums = nums[:k]
        '''
        N = len(nums)
        i = 0
        for j in range(N):
            if nums[j] != val:
                nums[i] = nums[j]
                i += 1
        return i

class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        '''
        two pointers from the end
        swap and decrease size
        note if when we swap, we don't move up left, we stay and check on the next interation
        '''
        N = len(nums)
        left,right = 0, N-1
        while left < N:
            if nums[left] == val:
                nums[left] = nums[right]
                N -= 1
                right -= 1
            else:
                left += 1
        
        return N
                
############################
# 05OCT21
# 70. Climbing Stairs
############################
#recursion
class Solution:
    def climbStairs(self, n: int) -> int:
        '''
        if there is 1 step, theres only 1 way
        if there are 2 steps, there are two ways
        if there are n steps, its just the sum of the previous steps
        F_n = F_{n-1} + F_{n-2}
        '''
        memo = {}
        
        def rec(i):
            if i == 1:
                return 1
            if i == 2:
                return 2
            
            if i in memo:
                return memo[i]
            ans = rec(i-1) + rec(i-2)
            memo[i] = ans
            return ans
        
        return rec(n)

#bottom up 2
class Solution:
    def climbStairs(self, n: int) -> int:
        '''
        bottom up 2, with array.
        return last element
        '''
        
        if n == 1:
            return 1
        
        if n == 2:
            return 2
        dp = [0]*n
        
        dp[0] = 1
        dp[1] = 2
        
        for i in range(2,n):
            dp[i] = dp[i-1] + dp[i-2]
        
        return dp[n-1]

#dp, constant space
class Solution:
    def climbStairs(self, n: int) -> int:
        '''
        bottom up dp, constant space
        '''
        
        if n == 1:
            return 1
        
        first = 1
        second = 2
        
        for i in range(3,n+1):
            third = first + second
            first = second
            second = third
        
        return second

#using golden ratio
class Solution:
    def climbStairs(self, n: int) -> int:
        '''
        golden ratio, to get log(N) times
        F_n = \frac{1}{sqrt(5)}[(\frac{1 + sqrt(5)}{2})^n - (\frac{1-sqrt(5)}{2})^n]
        
        '''
        sqrt = 5**(.5)
        fibn = (((1+sqrt)/2)**(n+1)) - (((1 - sqrt)/2)**(n+1))
        return int(fibn/sqrt)

#binets method, logN multiply
class Solution:
    def climbStairs(self, n: int) -> int:
        '''
        we can use binets method
        first define transition matrix: 
        [
        F{n+1} F{n}
        F{n}   F{n-1}
        ]
        
        [
        1 1
        1 0
        ]
        then we we just use matrix expoenntial algo to raise this matrix to a power, power n
        done in logN times
        we return the ans the upper left
        '''
        q = [[1,1],[1,0]]
        
        def multi(a,b):
            c = [[0,0],[0,0]]
            for i in range(2):
                for j in range(2):
                    c[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j]
            return c
        
        def fastpow(a,n):
            ans = [[1,0],[0,1]]
            while n > 0:
                if (n & 1) == 1:
                    ans = multi(ans,a)
                n >>= 1
                a = multi(a,a)
            return ans
        
        return fastpow(q,n)[0][0]

#############################
# 05_OCT_2021
# 217. Contains Duplicate
#############################
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        '''
        use hash seen and terminate
        '''
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        
        return False

#######################################
# 06OCT21
# 442. Find All Duplicates in an Array
#######################################
#naive way
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        '''
        naive way is just to use seen set
        '''
        ans = []
        seen = set()
        for num in nums:
            if num in seen:
                ans.append(num)
            else:
                seen.add(num)
        
        return ans

#using set
class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        '''
        we have legnth N, and each number is in range[1,n]
        we can use gauss trick to get the sum
        if there are duplicates, it means we are missing numbers in th range [1,n]
        if we decrement a value in nums by 1, it must go back to a valid index
        if we do this op for each element in the first pass, negate decremented index value
        second pass record negative posiionts
        '''
        ans = set()
        #negate each value to be marked as seen
        #we need to take the abs, otherwise we would be negatively indexing
        for num in nums:
            nums[abs(num)-1] *= -1
        
        #now pass again checking for positive values, negated twice
        #we are negatinge the positions
        for num in nums:
            if nums[abs(num)-1] > 0:
                ans.add(abs(num))
        return(ans)

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        '''
        we have legnth N, and each number is in range[1,n]
        we can use gauss trick to get the sum
        if there are duplicates, it means we are missing numbers in th range [1,n]
        if we decrement a value in nums by 1, it must go back to a valid index
        if we do this op for each element in the first pass, negate decremented index value
        second pass record negative posiionts
        '''
        ans = set()
        #negate each value to be marked as seen
        #we need to take the abs, otherwise we would be negatively indexing
        for num in nums:
            if nums[abs(num)-1] < 0:
                ans.add(abs(num))
            nums[abs(num)-1] *= -1
        
        
        return ans

class Solution:
    def findDuplicates(self, nums: List[int]) -> List[int]:
        '''
        we have legnth N, and each number is in range[1,n]
        we can use gauss trick to get the sum
        if there are duplicates, it means we are missing numbers in th range [1,n]
        if we decrement a value in nums by 1, it must go back to a valid index
        if we do this op for each element in the first pass, negate decremented index value
        second pass record negative posiionts
        '''
        ans = []
        #negate each value to be marked as seen
        #we need to take the abs, otherwise we would be negatively indexing
        for num in nums:
            nums[abs(num)-1] *= -1
        
        #now pass again checking for positive values, negated twice
        #we are negatinge the positions
        for num in nums:
            if nums[abs(num)-1] > 0:
                ans.append(abs(num))
                #correct the second occurence
                nums[abs(num)-1] *= -1
        return(ans)

############################################
# 06OCT21
# 863. All Nodes Distance K in Binary Tree
############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        '''
        i can make the graph for the tree
        then use bfs to find all nodes that are k away from target
        '''
        #build graph
        graph = defaultdict(list)
        def dfs(node,parent):
            if not node:
                return
            if node.val not in graph:
                if parent:
                    graph[node.val].append(parent.val)
                if node.left:
                    graph[node.val].append(node.left.val)
                if node.right:
                    graph[node.val].append(node.right.val)
            
            dfs(node.left,node)
            dfs(node.right,node)
        
        dfs(root,None)
        
        print(graph)
        
        #bfs
        seen = set([target.val])
        q = deque([(target.val,0)])
        ans = []
        
        while q:
            curr, steps = q.popleft()
            if steps == k:
                ans.append(curr)
            
            for neigh in graph[curr]:
                if neigh not in seen:
                    seen.add(neigh)
                    q.append((neigh,steps+1))
        
        return ans
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        '''
        we also could use recursion twice,
        first time we mapp parents to node
        then dfs using this new mapping
        '''
        
        mapp = {}
        
        def build_map(node,parent):
            if not node:
                return
            mapp[node] = parent
            build_map(node.left,node)
            build_map(node.right,node)
        
        build_map(root,None)
        
        seen = set()
        ans = []
        
        def dfs(node,distance):
            if not node or node in seen:
                return
            
            seen.add(node)
            
            if distance == k:
                ans.append(node.val)
            
            elif distance < k:
                dfs(node.left,distance+1)
                dfs(node.right,distance+1)
                dfs(mapp[node],distance+1)
        #call from target to explore
        #if we had parent, we would not need to do the first part
        dfs(target,0)
        return ans

#####################
# 07OCT21
# 79. Word Search
#####################
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        '''
        dfs from each i,j cell to find the word
        advance index only if there is matching characer
        dfs only if we can form here
        '''
        rows = len(board)
        cols = len(board[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(i,j,idx):
            if idx >= len(word):
                return True #word found
            #bounds 
            if 0 <= i < rows and 0 <= j < cols and board[i][j] == word[idx]:
                #get the char and mark in board
                temp = board[i][j]
                board[i][j] = None
                
                for dx,dy in dirrs:
                    if dfs(i+dx,j+dy,idx+1):
                        return True
                
                #return the value
                board[i][j] = temp
            
            return False

        for i in range(rows):
            for j in range(cols):
                if dfs(i,j,0):
                    return True
        
        return False

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        '''
        dfs from each i,j cell to find the word
        advance index only if there is matching characer
        dfs only if we can form here
        we could also use a seen set to add in the cell
        then dfs from this cell
        at the end we back track
        '''
        rows = len(board)
        cols = len(board[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(i,j,idx,seen):
            if idx >= len(word):
                return True #word found
            #bounds 
            if 0 <= i < rows and 0 <= j < cols and board[i][j] == word[idx] and (i,j) not in seen:
                seen.add((i,j))
                
                for dx,dy in dirrs:
                    if dfs(i+dx,j+dy,idx+1,seen):
                        return True
                
                #return the value
                seen.remove((i,j))
            
            return False

        for i in range(rows):
            for j in range(cols):
                if dfs(i,j,0,set()):
                    return True
        
        return False

######################################
# 08_OCT_21
# 208. Implement Trie (Prefix Tree)
######################################
class Trie:

    def __init__(self):
        self.T = {}
        

    def insert(self, word: str) -> None:
        root = self.T
        for ch in word:
            if ch not in root:
                root[ch] = {}
            root = root[ch]
        #mark end as special char
        root['#'] = True
        
    def find(self, pref: str) -> bool:
        root = self.T
        for ch in pref:
            if ch not in root:
                return None
            root = root[ch]
        
        return root

    def search(self, word: str) -> bool:
        node = self.find(word)
        return node is not None and "#" in node

        

    def startsWith(self, prefix: str) -> bool:
        return self.find(prefix) is not None


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

#more OOP like
#define node object as well
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEnd = False

class Trie:

    def __init__(self):
        self.root = TrieNode()   

    def insert(self, word: str) -> None:
        root = self.root
        for ch in word:
            if ch not in root.children:
                root.children[ch] = TrieNode()
            root = root.children[ch]
        #mark end as special char
        root.isEnd = True
        

    def search(self, word: str) -> bool:
        root = self.root
        for ch in word:
            if ch not in root.children:
                return False
            root = root.children[ch]
        
        return root.isEnd
        

    def startsWith(self, prefix: str) -> bool:
        root = self.root
        for ch in prefix:
            if ch not in root.children:
                return False
            root = root.children[ch]
        
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

############################
# 08COT21
# 1216. Valid Palindrome III
############################
#longest palindromic subsequence, find it, and check that it is at least N-k
class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        '''
        this is similar to finding the longest palindromic subsequence
        if we find a plaindromic subsequence of len(s) - k, then k s is a valid k palindric sub
        algo:
            find longest plaindromic subsequence
            return if ans at least N - k
            
        '''
        memo = {}
        
        def dp(l,r):
            #left pointer passer right pointer empty string
            if l > r:
                return 0
            if l == r:
                #at least one string
                return 1
            if (l,r) in memo:
                return memo[(l,r)]
            #if they match
            if s[l] == s[r]:
                #its just + 2 from inside
                ans = dp(l+1,r-1) + 2
                memo[(l,r)] = ans
                return ans
            #knapsack
            ans = max(dp(l+1,r),dp(l,r-1))
            memo[(l,r)] = ans
            return ans
            
        
        return dp(0,len(s)-1) >= len(s) - k

#dp solution
#O(N^2 space)
class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        ''' 
        we can translate this to 2d dp from the recursive solution
        using full dp array
        dp[i][j] represents number of valid subsequences between i and j
        wants dp[0][len(s)-1] as our numbber
            
        '''
        N = len(s)
        dp = [[0]*N for _ in range(N)]
        
        for i in range(N-1,-1,-1):
            #base case we always have length 1 as palindrome
            dp[i][i] = 1
            for j in range(i+1,N):
                #matching
                if s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] + 2
                else:
                    dp[i][j] = max(dp[i+1][j],dp[i][j-1])
                
        return dp[0][N-1] >= N - k

#bottom up dp constance space
class Solution:
    def isValidPalindrome(self, s: str, k: int) -> bool:
        ''' 
        we can translate this to 2d dp from the recursive solution
        using full dp array
        dp[i][j] represents number of valid subsequences between i and j
        wants dp[0][len(s)-1] as our numbber
            
        '''
        N = len(s)
        curr = [0]*N
        prev = [0]*N
        
        for i in range(N-1,-1,-1):
            #base case we always have length 1 as palindrome
            curr[i] = 1
            for j in range(i+1,N):
                #matching
                if s[i] == s[j]:
                    curr[j] = prev[j-1] + 2
                else:
                    curr[j] = max(prev[j],curr[j-1])
            
            curr,prev = prev,curr
        
        return prev[N-1] >= N - k

#######################
# 09_OCT_21
# 212. Word Search II
#######################
#dfs from each cell for reach word, TLE
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        '''
        i could dfs the board for each word in words
        when a word is found, return at that word to ans
        return ans
        '''
        rows = len(board)
        cols = len(board[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        ans = set()
        
        def dfs(i,j,pos,word):
            if pos >= len(word):
                return True
            #bounds 
            if 0 <= i < rows and 0 <= j < cols and board[i][j] == word[pos] and (i,j) not in seen:
                seen.add((i,j))
                
                for dx,dy in dirrs:
                    if dfs(i+dx,j+dy,pos+1,word):
                        return True
                
                #return the value
                seen.remove((i,j))
            
            return False
        
        for word in words:
            seen = set()
            for i in range(rows):
                for j in range(cols):
                    if word not in seen:
                        if dfs(i,j,0,word):
                            ans.add(word)
                        
        
        return ans
        
#hmmm, this still takes too long
#first lets define our Trie
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEnd = False

class Trie:

    def __init__(self):
        self.root = TrieNode()   

    def insert(self, word: str) -> None:
        root = self.root
        for ch in word:
            if ch not in root.children:
                root.children[ch] = TrieNode()
            root = root.children[ch]
        #mark end as special char
        root.isEnd = True
        

    def search(self, word: str) -> bool:
        root = self.root
        for ch in word:
            if ch not in root.children:
                return False
            root = root.children[ch]
        
        return root.isEnd
        

    def startsWith(self, prefix: str) -> bool:
        root = self.root
        for ch in prefix:
            if ch not in root.children:
                return False
            root = root.children[ch]
        
        return True
        
        
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        '''
        first make Trie structure for easy lookup with word
        then use dfs to explore each cell
        backtrack once we are done using dfs
        on our backtrack we can mark a cell as being explored with 0
        once we are done backtracking, return that to original value
        build path along,
        and if path is in tree we are good
        only dfs i pref of path is in trie
        '''
        t = Trie()
        for word in words:
            t.insert(word)
        
        rows = len(board)
        cols = len(board[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        
        ans = set()
        
        def backtrack(r,c,path):
            #bounds and currently being visited, terminate search
            if not ((0 <= r < rows) and  (0 <= c < cols)) or board[r][c] == 0:
                return
            path += board[r][c]
            
            #if we can't find this path in t, stop our search
            if not t.startsWith(path):
                return
            #if the current string is in t we are good
            if t.search(path) and string not in ans:
                ans.add(path)
            
            #backtrack
            currChar = board[r][c]
            board[r][c] = 0
            
            for dx,dy in dirrs:
                backtrack(r+dx,c+dy,path)
            
            #backtrack
            board[r][c] = currChar
            
        for i in range(rows):
            for j in range(cols):
                backtrack(i,j,"")
        
        return ans

#official solution
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        '''
        offical LC solution
        #build trie
            #then mark each leaf of the Trie, with a special char, and its word
            #once we have found this word, remmove word from Trie
        #optimizations
            #back track along the nodes in Trie, dfs using the Trie and progress through the leaves
        #prunes nodees in Trie during backtracking
            #once we have seen a word, removed from Trie
        '''
        ending_char = '$'
        
        trie = {}
        for word in words:
            node = trie
            for ch in word:
                #retive next node
                node = node.setdefault(ch,{})
            #mark ending
            node[ending_char]  = word
        
        rows = len(board)
        cols = len(board[0])
        dirrs = dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        ans = []
        def backtrack(row,col,parent):
            #get letter
            letter = board[row][col]
            #get root
            curr_root = parent[letter]
            #check if we found word, by remving special char
            word_match = curr_root.pop(ending_char,False)
            
            if word_match:
                #removed from trie, and avoid using set for results
                ans.append(word_match)
                
            #being backtrackig by marking as visited
            board[row][col] = '#'
            
            #inner recursion
            for dx,dy in dirrs:
                new_row = row + dx
                new_col = col + dy
                #bounds
                if not (0 <= new_row < rows) or not (0 <= new_col < cols):
                    continue
                #if we can't find from this curr node
                if board[new_row][new_col] not in curr_root:
                    continue
                #otherwise recurse
                backtrack(new_row,new_col,curr_root)
            
            #backtrack
            board[row][col] = letter
            
            #prine nodes
            if not curr_root:
                parent.pop(letter)
                
        for i in range(rows):
            for j in range(cols):
                if board[i][j] in trie:
                    backtrack(i,j,trie)
        
        return ans

############################
# 09_OCT_21
# 7. Reverse Integer
############################
#close one
class Solution:
    def reverse(self, x: int) -> int:
        '''
        i can use the divmod function to get qutoient and remainder
        then do the base 10 incrmeent multiply trick
        '''
        sign = -1 if x < 0 else 1
        x = abs(x)
        
        ans = 0
        
        while x>0:
            x,r = divmod(x,10)
            ans += r
            ans *= 10
        
        ans = (ans*sign) // 10 
        return 0 if ans > 2**31 else ans

#make sure to check overflow
class Solution:
    def reverse(self, x: int) -> int:
        '''
        i can use the divmod function to get qutoient and remainder
        then do the base 10 incrmeent multiply trick
        '''
        result = 0

        if x < 0:
            symbol = -1
            x = -x
        else:
            symbol = 1

        while x:
            result = result * 10 + x % 10
            x //= 10

        return 0 if result > pow(2, 31) else result * symbol

####################################
# 10_OCT_21
# 201. Bitwise AND of Numbers Range
####################################
#TLE, the range can be really big
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        '''
        just apply and operator to all number in range
        '''
        ans = left
        for num in range(left+1,right+1):
            ans = ans & num
        return ans

#bit shifts
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        '''
        recall if we and a series of 0 and 1 bits, if there is a 0 present at all
        the expression resolves to zero
        take [9,12]
        if we align all bits up
        00001001
        00001010
        00001011
        00001100
        
        we see they share common prefix
        
        we can reformulate the question, as given two integers, return common prevfix
        idea:
            use bit shit until we can to a common prefix
        algo:
            while m < n:
                shift both digits one at atime
                count up shifts
            return the number m shifted by the number of shifts in the opposite direction
        
        '''
        shifts = 0
        while left < right:
            left >>= 1
            right >>= 1
            shifts += 1
        
        return left << shifts

#using Brian Kernighan
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        '''
        there is a way to turn off the right most bit of a number
        it is called Brian Kernighan algo
        IDEA:
            When we do an AND bit operation between num num-1, the right most bit of the original number
            would be turned off (from zero to one)
            n & (n-1)
        the idea is that for a given range [m,n] and m < n, we could iterativle apply the trick
        on the number n to turn off the right most bit of on until it becomess <= than the beginning range
        finally we do AND bettwen n' and m to get res
        '''
        while left < right:
            #reduce n
            right = (right) & (right-1)
        
        return left & right

############################
# 11_OCT_21
# 543. Diameter of Binary Tree
############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        '''
        if im at a node, a node to its child has length 1, but to get from the other child
        it would be 1 + 1
        this is the recurrsce
        if left:
            path so far
        if right:
            path so far
        return max(left,right) + 1
        '''
        diameter = 0
        
            
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.ans = max(self.ans,left+right)
            return max(left,right) + 1
            
            
        
        dfs(root)
        return self.ans
            
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        '''
        https://leetcode.com/problems/diameter-of-binary-tree/discuss/112275/Python-Simple-and-Logical-Idea
        we can do this withou the use of a global
        pass diameters and heights as a return
        the answer is the max of left diameter,right diamter, or sum of the two height of the trees
        calculate max_diam and curr height as we go along
        '''
        
        def dfs(node):
            if not node:
                return 0,0
            left_diam,left_height = dfs(node.left)
            right_diam,right_height = dfs(node.right)
            #currheight at this node
            curr_height = max(left_height,right_height) + 1
            #max diam at this curr node is max of left,right or some of heights
            max_diam = max(left_diam,right_diam,left_height + right_height)
            return max_diam,curr_height
        
        return dfs(root)[0]
        
#print longest path
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        '''
        if i wanted to print the nodes along the longest diameter
        '''
        if not root:
            return None
        
        self.max_length = 0
        self.node_list = []
        self.traverse(root)
        
        for node in self.node_list:
            print(node.val)
        
        return self.max_length
    
    def traverse(self, root):
        if not root:
            return (0, [])
        
        left_max, left_node_list = self.traverse(root.left)
        right_max, right_node_list = self.traverse(root.right)
        
        if left_max + right_max > self.max_length:
            self.max_length = left_max + right_max
            self.node_list = left_node_list + right_node_list
        
        if left_max > right_max:
            return (left_max+1, left_node_list + [root])
        else:
            return (right_max+1, right_node_list + [root])


##############################
# 12_OCT_21
# 374. Guess Number Higher or Lower
##############################
class Solution:
    def guessNumber(self, n: int) -> int:
        '''
        this is just binary search
        '''
        lo,hi = 1,n
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            res = guess(mid)
            #found it
            if res == 0:
                return mid
            #pick < mid
            elif res == -1:
                hi = mid - 1
            #pick > mid
            else:
                lo = mid + 1
        return -1

#############################################################
# 13_OCT_21
# 1008. Construct Binary Search Tree from Preorder Traversal
#############################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        '''
        i can sort the preorder to get inorder as well
        we puck elements from the preorder array, then look up this element for the in order array
        elements to left of this prointer in inorder are in left subtree, to the right are in right sub tree
        if not possible use nulll as pointer
        '''
        N = len(preorder)
        if N < 1:
            return None
        #set pointer to pre
        pre_idx = [0]
        inorder = sorted(preorder)
        
        #need fast lookup for inorder
        inorder_map = {val:idx for idx,val in enumerate(inorder)}
        
        def make_tree(inorder_left,inorder_right):
            #base case
            if inorder_left == inorder_right:
                return None
            
            #first get peroder root
            root_val = preorder[pre_idx[0]]
            #make root
            root = TreeNode(root_val)
            #now forom this root, split using inorder
            inorder_idx = inorder_map[root_val]
            #move up pointer before recursing
            pre_idx[0] += 1
            #recurse
            root.left = make_tree(inorder_left,inorder_idx)
            root.right = make_tree(inorder_idx+1,inorder_right)
            return root
        
        return make_tree(0,N)

#recursive solution
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        '''
        sorting takes up times, we can do this in O(N) times
        the inorder traversal above was used to check if the element could be place in this subtree
        since this is a valid BST, we can use lower and upper limit tricks to see if an element belongs in this subtreee
        similar to validate BST
        algo:
            initiate lower and upper limits
            start from frist element
            helper(lower,upper)
                if preorder array is used up, i.e is N, return None
                if curr val is smaller than lower limit, or larger than upper limit, return None
                if curr val is in limits, place and recurse with changed limits left and right
        '''
        N = len(preorder)
        if N < 1:
            return None
        idx = [0]
        def make_tree(lower,upper):
            #base case, end up array
            if idx[0] == N:
                return None
            #get current root val
            val = preorder[idx[0]]
            #check bounds
            if val < lower or val > upper:
                return None
            
            idx[0] += 1
            root = TreeNode(val)
            root.left = make_tree(lower,val)
            root.right = make_tree(val,upper)
            return root
        
        return make_tree(float('-inf'),float('inf'))

#iterative
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        '''
        we can turn the recursive solution into an interative one using the stack
        algo:
            set the first preorder element as the root
            loop over all indices in preorder array
            adjust the parent node
                pop out of stack all elements with the value smaller than the childe value
                change parent node at each pop
            id node.val < child.val:
                set its right
            else:
                set its left
            push node back on to stack
        '''
        N = len(preorder)
        if N < 1:
            return None
        
        root = TreeNode(preorder[0])
        stack = [root]
        
        for i in range(1,N):
            node,child = stack[-1], TreeNode(preorder[i])
            #adjust parent
            while stack and stack[-1].val < child.val:
                node = stack.pop()
            
            #adjust chilren
            if node.val < child.val:
                node.right = child
            else:
                node.left = child
            stack.append(child)
        
        return root

#################################
# 14OCT21
# 279. Perfect Squares
#################################
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        we can use recursion to find the minimum number of primes
        so we have a number n, and the last prime number to be used was k
        if we knew the min numbers used to find n-k, then the ans would be min(n-k) + 1
        rec(n) = rec(n-k) for all k in square nums
        '''
        #we only need to generate squares less than sqrt(n)
        squares = set()
        start = 1
        while start*start <= n:
            squares.add(start*start)
            start += 1

        memo = {}
        
        def rec(curr):
            #found at least 1 square number
            if curr in squares:
                return 1
            elif curr in memo:
                return memo[curr]
            min_squares = float('inf')
            for square in squares:
                #if the current square is > than curr, can't ujse
                if square > curr:
                    break
                #otherwsie minmize
                new_num = rec(curr-square) + 1
                min_squares = min(min_squares, new_num)
                
            memo[curr] = min_squares
            return min_squares
        
        return rec(n)

#TLE, the squares need to be in order
class Solution:
    def numSquares(self, n: int) -> int:
        square_nums = [i**2 for i in range(1, int(math.sqrt(n))+1)]
        memo = {}

        def minNumSquares(k):
            """ recursive solution """
            # bottom cases: find a square number
            if k in square_nums:
                return 1
            if k in memo:
                return memo[k]
            min_num = float('inf')

            # Find the minimal value among all possible solutions
            for square in square_nums:
                if k < square:
                    break
                new_num = minNumSquares(k-square) + 1
                min_num = min(min_num, new_num)
            memo[k] = min_num
            return min_num

        return minNumSquares(n)

#https://leetcode.com/problems/perfect-squares/discuss/1513258/VERY-EASY-TO-UNDERSTAND-WITH-PICTURE-PYTHON-RECURSION-%2B-MEMOIZATION
#recusive solution
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        we can use recursion, but for each invocation check perfect squares up that number in the call
        when we reduce a number to zero, we have found a possibel way
        at most we can use only n 1's to get n
        then we just check all primes up to that curr and find the minimum
        '''
        memo = [-1]*(n+1)
        
        def rec(curr):
            if curr == 0:
                return 0
            if curr < 0:
                return n
            if memo[curr] != -1:
                return memo[curr]
            min_squares = n
            start = 1
            while start*start <= n:
                curr_count = rec(curr-(start*start))
                min_squares = min(min_squares,curr_count)
                start += 1
            
            memo[curr] = min_squares + 1
            return memo[curr]
        
        return rec(n)
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        we can translate the recursive solution using a dp 
        and go bottom up
        genereate all square nums
        dp array, and dp[i] represnets the minimum number of perfect squares need to det i
        we want whats currentlty there are the the solution from the last dp less than the current square
        '''
        square_nums = [i**2 for i in range(1, int(math.sqrt(n))+1)]
        dp = [float('inf')]*(n+1)
        #bottom case, zero
        dp[0] = 0
        for i in range(1,n+1):
            #fheck on each square
            for square in square_nums:
                #cannot go negative
                if i < square:
                    break
                dp[i] = min(dp[i],dp[i-square]+1)
                
        print(dp)
        return dp[-1]

#greedy emumeration
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        we can use recursion and implement it greddily
        this helps prune the call tree
        intuition:
            starting from the combination of one single number to multiple numbers, once we find a combination that can sum up to the givne number n, then we can say that we msut have found the smalleat combination
            since we are looking greedily for combindation sizes from small to large
        we can define a function that check wheter n can be divided by the count
        so numSquares(n) = min of all count from [1,n] is_divided_by(n,count)
        algo:
            precompute all sqaures from [1,sqrt(n)]
            in main loop, iteratize from smallest size to largest size, check if n can be divided by the sum of combination (num elements)
            bottom case, when count == 1, we jsut need to check if n is a square number! OMFG!!!, then its juss 1
            
        '''
        squares = set([i * i for i in range(1, int(n**0.5)+1)])
        def can_make_with_count(n,count):
            #if we got to 1, check whter this n is prime
            if count == 1:
                return n in squares
            
            #recurse
            for k in squares:
                #use up a prime, and decrease
                if can_make_with_count(n-k,count-1):
                    return True
            return False
        
        for count in range(1,n+1):
            if can_make_with_count(n,count):
                return count

#greedy and BFS
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        we notice that in the greedy method, its a call tree using cam_make_with_count
        this makes an N ary tree with sqrt(n) nodes
        intuition:
            given an N ary tree, where each node represents a remainder of the number n, subtracting a combination
            of square numbers, our task is to find the node in the tree:
                1. value of the nodes should be a sqaure number
                2. closes to the root
        in the greedy method, we actually make the call tree similar to a graph in BFS
        algo:
            generate all possible squares
            make q, which could keep all the remainders to enumerate at each level
            iterate over q
                at each iteration checkk if remainder is one of square numbers
                if not, subtract it with on of the squars numbers to get remainder and add remainder to nextq
            break out of loop once we get a remainder that is square
        
        NOTE: our q is a set, to help eliminate redundancy
        '''
        squares = []
        start = 1
        while start*start <= n:
            squares.append(start*start)
            start += 1
        
        levels = 0
        q = {n}
        while q:
            #go down in level
            levels += 1
            next_q = set()
            for rem in q:
                for sq in squares:
                    if rem == sq:
                        return levels
                    elif rem < sq:
                        break
                    else:
                        next_q.add(rem - sq)
            #get next level
            q = next_q
        
        return levels

#mathy way
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        bachets conjecture sets upper boud for number of sqaures to make real numbers
        p = a^2 + b^2 + c^2 + d^2
        largest numsquares can b <= 4
        however lagrange four square theorem does not tell us directly the least numbers of squares to decompose
        adrien-maries legendr:
            three square theorem
            4^k (8m+7) = n = a^2 + b^2 + c^2
            allows us to check if number can only be decomposed into 4 squares
        Case 3.1). if the number is a square number itself, which is easy to check e.g. n == int(sqrt(n)) ^ 2.
        Case 3.2). if the number can be decomposed into the sum of two squares. Unfortunately, there is no mathematical weapon that can help us to check this case in one shot. We need to resort to the enumeration approach.
        '''
        def isSquare(n:int) -> bool:
            sq = int(math.sqrt(n))
            return sq*sq == n

        while (n & 3) == 0:
            n >>= 2      # reducing the 4^k factor from number
        if (n & 7) == 7: # mod 8
            return 4

        if isSquare(n):
            return 1
        # check if the number can be decomposed into sum of two squares
        for i in range(1, int(n**(0.5)) + 1):
            if isSquare(n - i*i):
                return 2
        # bottom case from the three-square theorem
        return 3

#########################################
# 15OCT21
# 309. Best Time to Buy and Sell Stock with Cooldown
#########################################
#recursie solution
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        we are given an array of stock prices
        we have an unlimited number of transactions, i.e we can buy or sell as many times
        but after selling stock, we have to wait on more day
        return max profits
        at any one day, we have two choices, buy or sellf
        for each recursive call, maintain a buy state
        
        if we cannot afford a transaction today just adavance
        ans1 = rec(day+1,buy)
        
        if we can afforcd a transaction today
        if buy:
            ans2 = -prices[i] + rec(i+1,false)
        else:
            ans2 = prices[i] + rec(i+2,true)
        
        if we go past the index, i.e past the days, return 0, no profit CAN be made
        '''
        memo = {}
        N = len(prices)
        
        def rec(day,buy):
            if day >= N:
                return 0
            if (day,buy) in memo:
                return memo[(day,buy)]
            #no transaction, we always have this choice
            ans1 = rec(day+1,buy)
            
            #second choice
            ans2 = 0
            if buy:
                ans2 = -prices[day] + rec(day+1,0)
            else:
                ans2 = prices[day] + rec(day+2,1)
                
            memo[(day,buy)] = max(ans1,ans2)
            return memo[(day,buy)]
        #we start off with buying
        return rec(0,1)

#translating to dp
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        we can turn the top down recursive solution into a bottom up dp solution
        at any one time we a have of just staying, i.e not gettin profit
        if we can buy, going down in profit
        if we can sell, going up in profit
        for every day, we have two states of buy or not buy
        '''
        if not prices:
            return 0
        
        #make dp arrays, add 2 more spots for edge cases
        N = len(prices)
        dp = [[0]*(2) for _ in range(N+3)]
        
        for day in range(N-1,-1,-1):
            for buy in range(2):
                #edge cases
                if day >= N:
                    dp[day][buy] = 0
                else:
                    ans1 = dp[day+1][buy]
                    if buy:
                        ans2 = -prices[day] + dp[day+1][0]
                    else:
                        ans2 = prices[day] + dp[day+2][1]
                
                #update
                dp[day][buy] = max(ans1,ans2)
        
        return dp[0][1]

#using state machines, offical solutions
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/solution/
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        we can define states:
            held: agent holds a stock that it bought some point before
            sold: agent just sold stock right before entering this state
            reset: starting point; holding no stock and did not sell stock before; gives us cooldown after selling
        
        actions:
            sell: agent sells stock at curr moment, after selling agent goes to sold state
            but: buy, goes to hold state:
            rest: no transaction, stay
            
        go down prices, and look at each state, calculate states for each price!
        each node in graph, we hold max profits so far
        
        algo:
            define three state arrays
            (i.e. held[i], sold[i] and reset[i]) which correspond to the three states that we defined before.
            elements in state arrays repsent max profits so far
            example sold[2] is max profits we gain if we sell sock at price point[2]
        
        transitions:
            sold[i] = hold[i-1] + price[i]
            held[i] = max(held[i-1],reset[i-1] - price[i])
            reset[i] = max(reset[i-1],sold[i-1])
        
        max profits is max(sold[n],reset[n]), either sell or hold at n
        base case:
            kicked off from the reset state, since we don't start off holding any stock
            and so we assign inital values of sold and helad as Integer.MIN_VALUE
        '''
        N = len(prices)
        sold = [0]*(N+1)
        held = [0]*(N+1)
        reset = [0]*(N+1)
        
        #bases cases
        sold[0] = held[0] = float('-inf')
        
        for i in range(N):
            sold[i+1] = held[i] + prices[i]
            held[i+1] = max(held[i],reset[i] - prices[i])
            reset[i+1] = max(reset[i],sold[i])
        
        return max(sold[-1],reset[-1])

#reducing state space to O(1)
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        sold, held, reset = float('-inf'), float('-inf'), 0

        for price in prices:
            # Alternative: the calculation is done in parallel.
            # Therefore no need to keep temporary variables
            #sold, held, reset = held + price, max(held, reset-price), max(reset, sold)

            pre_sold = sold
            sold = held + price
            held = max(held, reset - price)
            reset = max(reset, pre_sold)

        return max(sold, reset)

#######################################
# 16OCT21
# 123: Best Time to Buy and Sell Stock III
#######################################
#recursive
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        lets try to do this recursifely first
        we have day, buy, and 2 transactions, once 2 are down we are left to keep advancing in the array
        we need to know if we are in own or sell state
        we need to be holding stock to get it
        '''
        memo = {}
        N = len(prices)
        
        def rec(day,own,k):
            if day >= N or k == 0:
                return 0
            if (day,own,k) in memo:
                return memo[(day,own,k)]
            #if i had own stack, i can sell or stay, and if sell use up transation
            #take means i do an action
            if own:
                #i can sell and go up in prices
                take = prices[day] + rec(day+1,0,k-1)
                #or just hold and go on to the next day
                no_take = rec(day+1,1,k)
            else:
                #i dont own, so i have to buy, but keep k
                take = -prices[day] + rec(day+1,1,k)
                #stay but keep k
                no_take = rec(day+1,0,k)
            
            memo[(day,own,k)] = max(take,no_take)
            return memo[(day,own,k)]

        
        return rec(0,0,2)

#now translate to dp
#this doesn't really work, assumes we own and have k transactions done
#need to treat transactions differently
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        now just try translating recursiion to dp
        3 d dp array, but since we only have 2 own states, and k transctions
        its just O(days*states*transations)
        which the last two are a constant factor
        '''
        if not prices:
            return 0
        
        N = len(prices)
        dp = [[[0]*(3)]*(2) for _ in range(N+1)] #to include for base case +1, but upper bound must be inclusive
        for day in range(N,-1,-1):
            for own in range(2):
                for trans in range(3):
                    #base case,out of bounds or out of trans
                    if trans == 0 or day >= N:
                        dp[day][own][trans] = 0
                    else:
                        if own:
                            take = prices[day] + dp[day+1][0][trans-1]
                            no_take = dp[day+1][1][trans]
                        else:
                            take = -prices[day] + dp[day+1][1][trans]
                            no_take = dp[day+1][0][trans]
                        
                        dp[day][own][trans] = max(take,no_take)
                        
        
        print(dp)
#another recursive way
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/1523723/C%2B%2B-or-Four-Solutions-%3A-Recursion-Memoization-DP-with-O(N)-space-DP-with-O(1)-Space
        #just another way to think recursively
        if we have k transactions with no cool down, we would have buy,sell.....buy,sell k times
        if have at least one transction we have choices:
            no transaction = rec(day+1,trans left)
        if we buy:
            -prices[day] + rec(day+1,trans-1)
        or sell
            prices[day] + rec(day+1,trans-1)
        base cases, gone past days, 0 or no trans left, which is zero
        
        '''
        memo = {}
        N = len(prices)
        def rec(day,transactionsLeft):
            if day >= N or transactionsLeft == 0:
                return 0
            if (day,transactionsLeft) in memo:
                return memo[(day,transactionsLeft)]

            #we can always choose to stay
            ans1 = rec(day+1,transactionsLeft)
            ans2 = 0
            #if we can buy
            buy = (transactionsLeft % 2 == 0)
            if buy:
                ans2 = -prices[day] + rec(day+1, transactionsLeft -1)
            else:
                ans2 = prices[day] + rec(day+1, transactionsLeft -1)

            memo[(day,transactionsLeft)] = max(ans1,ans2)
            return memo[(day,transactionsLeft)] 
        
        return rec(0,4)

#translating to dp
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        # https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/discuss/1523723/C%2B%2B-or-Four-Solutions-%3A-Recursion-Memoization-DP-with-O(N)-space-DP-with-O(1)-Space
        #just another way to think recursively
        if we have k transactions with no cool down, we would have buy,sell.....buy,sell k times
        if have at least one transction we have choices:
            no transaction = rec(day+1,trans left)
        if we buy:
            -prices[day] + rec(day+1,trans-1)
        or sell
            prices[day] + rec(day+1,trans-1)
        base cases, gone past days, 0 or no trans left, which is zero
        
        '''
        N = len(prices)
        k = 2
        dp = [[0]*(2*k+1) for _ in range(N+1)]
        
        for day in range(N, -1,-1):
            for trans in range(2*k+1):
                if day >= N or trans == 0:
                    dp[day][trans] = 0
                else:
                    ans1 = dp[day+1][trans]
                    ans2 = 0
                    buy = (trans % 2 == 0)
                    if buy:
                        ans2 = -prices[day] + dp[day+1][trans -1]
                    else:
                        ans2 = prices[day] + dp[day+1][trans-1]
                    
                    dp[day][trans] = max(ans1,ans2)
        
        return dp[0][-1]

##############################
# 17OCT21
# 437. Path Sum III
##############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        '''
        i can do down a path each bracnh
        and for each path, check if the sumSoFar == targetSum, it is, then increment count by 1
        dfs twice
        define helper funcion, and invoke at each node in the tree
        '''
        if not root:
            return 0
        
        self.total = 0
        def dfs_from_node(node,sumSoFar):
            if not node:
                return 0
            if sumSoFar + node.val == targetSum:
                self.total += 1
            dfs_from_node(node.left,sumSoFar + node.val)
            dfs_from_node(node.right,sumSoFar + node.val)
        
        def dfs_from_root(node):
            if not node:
                return 
            dfs_from_node(node,0)
            dfs_from_root(node.left)
            dfs_from_root(node.right)
        
        dfs_from_root(root)
        return self.total
            
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        '''
        we can use the concept of prefix sum, and check if a complement sum resides in our hasmap or not
        recall the problem, Find Number of continuous subarray that sum to target
        algo:
            init global variable to hold counts
            default to preoder if we want to span the tree
            update sumSoFar with curr node val
            check if it euqls target
            but we also could have found this sum compleement, in our hasmap
        '''
        countSumsSeen = defaultdict(int)
        self.total = 0
        
        def dfs(node,sumSoFar):
            if not node:
                return 
            sumSoFar += node.val
            
            #case 1, its a match
            if sumSoFar == targetSum:
                self.total += 1
            
            #case 2, find comp
            comp = sumSoFar - targetSum
            if comp in countSumsSeen:
                self.total += countSumsSeen[comp]
            
            #add to seen
            countSumsSeen[sumSoFar] += 1
            
            #recruse
            dfs(node.left,sumSoFar)
            dfs(node.right,sumSoFar)
            
            #back track
            countSumsSeen[sumSoFar] -= 1
        
        dfs(root,0)
        return self.total

################################################
# 17OCT21
# 302. Smallest Rectangle Enclosing Black Pixels
################################################
#i had this one! just the fucking cornere cases
# not we could have minimized and maximized on the fly
class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        '''
        if im given the position of a black pixel, and there is only on region
        i can bfs from this posiiton to find all positions of black
        then i can the smallest bounding box for this compoenent
        and subtract that from the total area of the sqaure
        '''
        rows = len(image)
        cols = len(image[0])
        
        if rows == 0 or cols == 0:
            return 1
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        seen = set([(x,y)])
        q = deque([(x,y)])
        
        while q:
            curr_x,curr_y = q.popleft()
            #find nieghs
            for dx,dy in dirrs:
                neigh_x = curr_x + dx
                neigh_y = curr_y + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #not seen
                    if (neigh_x,neigh_y) not in seen and image[neigh_x][neigh_y] == '1':
                        seen.add((neigh_x,neigh_y))
                        q.append((neigh_x,neigh_y))
                        
        #now find min x and max y
        min_x = rows
        max_x = 0
        
        min_y = cols
        max_y = 0
        
        for i,j in seen:
            min_x = min(min_x,i)
            max_x = max(max_x,i)
            
            min_y = min(min_y,j)
            max_y = max(max_y,j)
        
        size_x = max_x - min_x + 1
        size_y = max_y - min_y + 1
        
        return (1+ max_x - min_x)*(1 + max_y - min_y)

#naive linear
class Solution:
    def minArea(self, image: List[List[str]], x: int, y: int) -> int:
        '''
        naive way, which still passes would be to scan each row and find upper and lower bounds for x and y
        
        '''
        if not image:
            return 0
        
        rows = len(image)
        cols = len(image[0])
        
        min_x = rows
        min_y = cols
        
        max_x = 0
        max_y = 0
        
        has_black = False
        
        for i in range(rows):
            for j in range(cols):
                if image[i][j] == '1':
                    has_black = True
                    
                    min_x = min(min_x,i)
                    min_y = min(min_y,j)

                    max_x = max(max_x,i)
                    max_y = max(max_y,j)
                    
        if not has_black:
            return 0
        
        return (1 + max_x - min_x)*(1+max_y-min_y)
                