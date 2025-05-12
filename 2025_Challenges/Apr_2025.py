###########################################
# 3073. Maximum Increasing Triplet Value
# 01APR25
##########################################
#phewww
from sortedcontainers import SortedList
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        '''
        triplet needs to be increasing
        with order being preserved
        '''
        n = len(nums)
        right = [0]*n
        right[-1] = nums[-1]
        #for each element nums[i], find its largest to the right
        for i in range(n-2,-1,-1):
            right[i] = max(nums[i],right[i+1])
        #when at i for i in range(1,n-1), we know the maximum to the right at nums[i] is just right[i]
        #now how we we find the best left for some nums[i]
        ordered_left = SortedList([])
        ordered_left.add(nums[0])
        ans = 0 #smallest valie could be zero
        for i in range(1,n-1):
            #binary serach for the greatest left
            middle = nums[i]
            greatest_left = ordered_left.bisect_left(middle)
            if greatest_left - 1 >= 0:
                #print(ordered_left[greatest_left-1],middle,right[i])
                if ordered_left[greatest_left-1] < middle < right[i]:
                    ans = max(ans, ordered_left[greatest_left-1] - middle + right[i] )
            ordered_left.add(middle)
        
        return ans

##############################################
# 2873. Maximum Value of an Ordered Triplet I
# 02APR25
##############################################
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        '''
        for each index i from 1 to n-2, we need the largest number just greater than nums[i] on its left
        the triplet does not need to be increasing we just need the maximum on the left
        '''
        n = len(nums)
        right = [0]*n
        right[-1] = nums[-1]
        #for each element nums[i], find its largest to the right
        for i in range(n-2,-1,-1):
            right[i] = max(nums[i],right[i+1])
        left = [0]*n
        left[0] = nums[0]
        for i in range(1,n):
            left[i] = max(nums[i],left[i-1])
        #print(left)
        #print(right)
        ans = 0
        for i in range(1,n-1):
            cand = (left[i-1] - nums[i])*right[i+1]
            #print(left[i-1],nums[i],right[i+1])
            ans = max(ans,cand)
        
        return ans
    
#two pass is just ptr maniuplation to get i from start and i from end
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        left = [0] * n
        right = [0] * n
        for i in range(1, n):
            left[i] = max(left[i - 1], nums[i - 1])
            right[n - 1 - i] = max(right[n - i], nums[n - i])
        res = 0
        for j in range(1, n - 1):
            res = max(res, (left[j] - nums[j]) * right[j])
        return res
    
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        '''
        constant space, just maximize previous left and right
        '''
        ans = 0
        left = 0
        middle = 0
        n = len(nums)
        for i in range(n):
            ans = max(ans,left*nums[i])
            left = max(left,middle - nums[i])
            middle = max(middle,nums[i])
        
        return ans

###############################################
# 2874. Maximum Value of an Ordered Triplet II
# 03APR25
###############################################
#dp
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        '''
        need to keep max to the left
        and max to the right
        '''
        n = len(nums)
        left = [0]*n
        right = [0]*n
        for i in range(1,n):
            left[i] = max(nums[i-1],left[i-1])
            right[n-i-1] = max(right[n-i],nums[n-i])
        
        ans = 0
        for i in range(1,n-1):
            ans = max(ans, (left[i] - nums[i])*right[i])
        
        return ans

#constant dp
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        ans = 0
        left = 0
        middle = 0
        n = len(nums)
        for i in range(n):
            ans = max(ans,left*nums[i])
            left = max(left,middle - nums[i])
            middle = max(middle,nums[i])
        
        return ans
    
#############################################
# 2012. Sum of Beauty in the Array
# 03APR25
#############################################
#brute force TLE
class Solution:
    def sumOfBeauties(self, nums: List[int]) -> int:
        '''
        when checking for the 2 condition at index i, nums[i] needs to be bigger then all nums[:i-1]
        but less than all nums[i+1:]
        '''
        #brute force
        beauty = 0
        n = len(nums)
        for i in range(1,n-1):
            left = max(nums[:i])
            right = min(nums[i+1:])
            if left < nums[i] < right:
                beauty += 2
            elif nums[i-1] < nums[i] < nums[i+1]:
                beauty += 1
            #print(left,nums[i],right)
        
        return beauty
    
#three pass dp
class Solution:
    def sumOfBeauties(self, nums: List[int]) -> int:
        '''
        when checking for the 2 condition at index i, nums[i] needs to be bigger then all nums[:i-1]
        but less than all nums[i+1:]

        need to precmopute left maxes and right mins
        '''
        #brute force
        beauty = 0
        n = len(nums)
        left_max = [0]*n
        left_max[0] = nums[0]
        for i in range(1,n):
            left_max[i] = max(nums[i],left_max[i-1])
        
        right_min = [0]*n
        right_min[-1] = nums[-1]
        for i in range(n-2,-1,-1):
            right_min[i] = min(right_min[i+1],nums[i])
        
        for i in range(1,n-1):
            left = left_max[i-1]
            right = right_min[i+1]
            if left < nums[i] < right:
                beauty += 2
            elif nums[i-1] < nums[i] < nums[i+1]:
                beauty += 1
            #print(left,nums[i],right)
        
        return beauty
    
#two pass
class Solution:
    def sumOfBeauties(self, nums: List[int]) -> int:
        '''
        when checking for the 2 condition at index i, nums[i] needs to be bigger then all nums[:i-1]
        but less than all nums[i+1:]

        need to precmopute left maxes and right mins
        '''
        #brute force
        beauty = 0
        n = len(nums)
        left_max = [0]*n
        left_max[0] = nums[0]
        right_min = [0]*n
        right_min[-1] = nums[-1]
        for i in range(1,n):
            #need to exclude index at left_max[i] and right_min[i]
            #needs to be to the left and right of i, not including i
            left_max[i] = max(nums[i-1],left_max[i-1])
            right_min[n-i-1] = min(right_min[n-i],nums[n-i])
        
        for i in range(1,n-1):
            left = left_max[i]
            right = right_min[i]
            if left < nums[i] < right:
                beauty += 2
            elif nums[i-1] < nums[i] < nums[i+1]:
                beauty += 1
            #print(left,nums[i],right)
        
        return beauty

#################################################
# 1123. Lowest Common Ancestor of Deepest Leaves
# 04APR25
#################################################
#meh works but i don't like it
#Definition for a binary tree node.
class TreeNode:
     def __init__(self, val=0, left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
class Solution:
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        find deepest leaves, then just use lca, two passes, no we can't just do this because we could have
        mutiple nodes that aare the deepest like
        lca could also be itself
                1
            2       3
          4    5  6   7
        
        first find the deepest leaves
        '''
        depths = defaultdict(set)
        self.dfs(root,0,depths)
        max_depth = max(depths.keys())
        deepest_leaves = depths[max_depth]
        return self.lca(root,deepest_leaves)
    
    def dfs(self,node,depth,depths):
        if not node:
            return
        if not node.left and not node.right:
            depths[depth].add(node.val)
        self.dfs(node.left,depth+1,depths)
        self.dfs(node.right,depth+1,depths)
    
    def lca(self,node,deepest_leaves):
        if not node:
            return None
        if not node.left and not node.right:
            if node.val in deepest_leaves:
                return node
        left = self.lca(node.left,deepest_leaves)
        right = self.lca(node.right,deepest_leaves)
        if left != None and right != None:
            return node
        if left == None and right == None:
            return None
        return left if left else right

#dp,
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we can use recurion and return two args
            depth, and this nodes lca
            if the left subtree has a deeper depth, return its depth, and the deepest node in the left subtree
        recall max depth of a tree
        '''
        def rec(node):
            if not node:
                return [0,None]
            
            deepest_depth_left, left_lca = rec(node.left)
            deepest_depth_right, right_lca = rec(node.right)
            #if left subtree has a deeper leave, return it and the lca on on the left
            if deepest_depth_left > deepest_depth_right:
                return [deepest_depth_left + 1, left_lca]
            elif deepest_depth_left < deepest_depth_right:
                return [deepest_depth_right + 1, right_lca]
            #equal, return node
            return [deepest_depth_left + 1,node]

        return rec(root)[1]
    
############################################
# 368. Largest Divisible Subset (REVISTED)
# 06APR25
############################################
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        '''
        iterative
        corollary 1:
            in an array [E,F,G] and [E < F < G]
            for any value that can be divided by the largest number in a divisible subset, we can add that number to the subset
        corollary 2:
            for any value that can divide the smallest elmeent in the subset, we can form another divisible subset
            we only need to check the the largest or the smallest when of the current subset whent rying to extend it out
        '''
        n = len(nums)
        if n == 0:
            return []
        
        EDS = [[] for _ in range(n)] #largest subset adding at nums[i]
        nums.sort()
        for i in range(n):
            max_subset = []
            for k in range(i):
                if (nums[i] % nums[k] == 0):
                    if len(EDS[k]) > len(max_subset):
                        max_subset = EDS[k][:]
            EDS[i].extend(max_subset)
            EDS[i].append(nums[i])
        
        return max(EDS,key = len)
    
######################################################
# 2052. Minimum Cost to Separate Sentence Into Rows
# 06APR25
#####################################################
class Solution:
    def minimumCost(self, sentence: str, k: int) -> int:
        '''
        split or dont split,
        if we split here we need to add a cost
        the issue is that the last row does not contribute the cost, so we can substract the cost out
        need to use dp(i,j)
        we actually donn't need the words, just their lenghts
        oh woops, we can only split such that the number of characters is at mopst k row is at most k
        and we need to split
        base case is when we are within k for the last row, return 0
        note, the index i may be different, unlike other base cases where we are outside the bounds
        in this case, the base case could be one of many different i positions, so long as we are within k
        wierd base case tidbit here....
        '''
        words = sentence.split(" ")
        words = [len(w) for w in words]
        memo = {}
        n = len(words)
        m = len(sentence)
        def dp(i):
            #if we are in the last row return nothing
            if sum(words[i:]) + (n - i - 1) <= k:
                return 0
            if i in memo:
                return memo[i]
            
            ans = float('inf')
            curr_chars = words[i]
            curr_words = 1
            ii = i #need to move to the next i, but need other i for caching
            #jeeze this problem sucks...
            while i < n and curr_chars + (curr_words - 1) <= k:
                cost = (k - (curr_chars + (curr_words - 1)))**2
                ans = min(ans, cost + dp(ii+1))
                ii += 1
                curr_chars += words[ii]
                curr_words += 1
            memo[i] = ans
            return ans
        
        return dp(0)

#instead of getting word lengths, just act on the string
#adding cache decorator with memo doesn't change anything
class Solution:
    def minimumCost(self, sentence: str, k: int) -> int:
        '''
        just iterate on the string
        '''
        n = len(sentence)
        memo = {}
        @lru_cache(None)
        def dp(i):
            if n - i <= k:
                return 0
            if i in memo:
                return memo[i]
            ans = float('inf')
            #advance
            if sentence[i] == ' ':
                i += 1
            start = i
            while i < n and i - start <= k:
                if sentence[i] == ' ':
                    ans = min(ans, (k-(i-start))**2 + dp(i+1))
                i += 1
            memo[i] = ans
            return ans
        return dp(0)

#############################################
# 416. Partition Equal Subset Sum (REVISTED)
# 07APR25
#############################################
#top down
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        '''
        if i find the sum of the array, the two paritions must be equal
        i can try building a subset until the subset becomes SUM // 2
        if it any point the subset grows beyond that i need to abandon
        it would help to sort the array first
        '''
        N = len(nums)
        SUM = sum(nums)
        
        #cannot be partitioned evenly
        if SUM % 2 == 1:
            return False
        
        #sort
        nums = sorted(nums)
        
        memo = {}
        
        def rec(i,curr_sum):
            #got to end
            if i == N:
                if curr_sum == SUM // 2:
                    return True
                else:
                    return False
            
            if curr_sum > SUM // 2:
                return False
            
            #got here
            if curr_sum == SUM // 2:
                return True
            
            if (i,curr_sum) in memo:
                return memo[(i,curr_sum)]
            take = rec(i+1,curr_sum+nums[i])
            no_take = rec(i+1,curr_sum)
            res = take or no_take
            memo[(i,curr_sum)] = res
            return res
        
        return rec(0,0)
            
#bottom up
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        '''
        if i find the sum of the array, the two paritions must be equal
        i can try building a subset until the subset becomes SUM // 2
        if it any point the subset grows beyond that i need to abandon
        it would help to sort the array first
        '''
        N = len(nums)
        SUM = sum(nums)
        
        #cannot be partitioned evenly
        if SUM % 2 == 1:
            return False
        
        #sort
        nums = sorted(nums)
        
        dp = [[False]*((SUM // 2) + 1) for i in range(N+1)]

        #base case fill
        for curr_sum in range((SUM // 2) + 1):
            if curr_sum == (SUM // 2):
                dp[N][curr_sum] = True

        for i in range(N-1,-1,-1):
            for curr_sum in range((SUM // 2) -1,-1,-1):
                if curr_sum + nums[i] > SUM // 2:
                    continue
                take = dp[i+1][curr_sum+nums[i]]
                no_take = dp[i+1][curr_sum]
                res = take or no_take
                dp[i][curr_sum] = res
        
        return dp[0][0]
        
#####################################################################
# 1981. Minimize the Difference Between Target and Chosen Elements
# 07APR25
#####################################################################
#kida like advent of code!
class Solution:
    def minimizeTheDifference(self, mat: List[List[int]], target: int) -> int:
        '''
        want to minimize abs((sum of elemnts for all nums chose) - target)
        we want to minize the dist from the chosen nums
        evaluating all would result in 70**70, this is too big
        biggest sum would be 70*70, smallest sum would be 1
        '''
        sums = set([0])
        for row in mat:
            next_sums = set()
            for num in row:
                for curr_sum in sums:
                    next_sums.add(curr_sum + num)
            sums = next_sums
        
        ans = float('inf')
        for num in sums:
            ans = min(ans, abs(num - target))
        
        return ans
    
#dp
class Solution:
    def minimizeTheDifference(self, mat: List[List[int]], target: int) -> int:
        '''
        you can also use dp, but for each row remove duplicates and sort
        dp states (index i as row, and curr_sum)
        '''
        new_mat = [sorted(list(set(row))) for row in mat]
        n = len(mat)
        memo = {}
        
        @lru_cache(None)
        def dp(i,curr_sum):
            if i >= n:
                return abs(curr_sum - target) 
            if (i,curr_sum) in memo:
                return memo[(i,curr_sum)]
            
            ans = float('inf')
            for num in mat[i]:
                ans = min(ans, dp(i+1,num + curr_sum))

            
            memo[(i,curr_sum)] = ans
            return ans

        return dp(0,0)
    
    
########################################################################
# 3396. Minimum Number of Operations to Make Elements in Array Distinct
# 08APR25
########################################################################
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        '''
        brute force it
        '''
        count = 0
        while len(nums) > 0:
            if len(nums) == len(set(nums)):
                return count
            nums = nums[3:]
            count += 1
        return count
    
#linear time
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        '''
        reduce problem to finding suffix where all elements are distinct
        and since we remove three at a time, we need to do 
        i // 3 + 1 ops
        '''
        seen = set()
        n = len(nums)
        for i in range(n-1,-1,-1):
            if nums[i] in seen:
                return i//3 + 1
            seen.add(nums[i])
        
        return 0
    
#############################################################
# 3375. Minimum Operations to Make Array Values Equal to K
# 09APR25
############################################################
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        '''
        the description of valid is confusing
            an integer h is valid os all the values in nums that are > h are identical
        
        operations are:
            * select integerh, that is valid for nums
            * for each index i, where nums[i] > h, set nums[i] t0 k
        
        return min number of operations to make every element in nums equal to k
        we need to get to an array like [k,k,k...,k,k]
        try using all possible h from 0 to the second highest h
        we need to pick and h, then set all those greater than h, to h
        so the largest get reduced to h
        if k is bigger than the minimum we can never do it
        we can only choose a valid h if all the numers that are greater than h are identical
        '''
        if min(nums) < k:
            return -1
        #no need for duplicates
        nums = set(nums)
        nums.add(k)
        return len(nums) - 1
    
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        '''
        if the max element in nums is x, and the second largest is y
        then we can choose, h such that y <= h < x, and repalce all occruences of x in the array with h
        if there is a number smaller than k, we can't do it
        otherwise count the number of different integers greater than k
        '''
        seen = set()
        for num in nums:
            if num < k:
                return -1
            elif num > k:
                seen.add(num)
        
        return len(seen)
    
########################################################
# 317. Shortest Distance from All Buildings (REVISTED)
# 09APR25
#######################################################
from collections import deque

class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        # Function for BFS
        def bfs(grid, row, col, totalHouses):
            # Next four directions.
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            
            rows = len(grid)
            cols = len(grid[0])
            distanceSum = 0
            housesReached = 0
            
            # Queue to do a bfs, starting from (row,col) cell
            q = deque([(row, col)])
            
            # Keep track of visited cells
            vis = [[False] * cols for _ in range(rows)]
            vis[row][col] = True
            
            steps = 0
            
            while q and housesReached != totalHouses:
                for _ in range(len(q)):
                    curr = q.popleft()
                    r, c = curr
                    
                    # If this cell is a house, then add the distance from the source to this cell
                    # and we go past from this cell.
                    if grid[r][c] == 1:
                        distanceSum += steps
                        housesReached += 1
                        continue
                    
                    # This cell was an empty cell, hence traverse the next cells which is not a blockage.
                    for dr, dc in dirs:
                        nextRow = r + dr
                        nextCol = c + dc
                        
                        if 0 <= nextRow < rows and 0 <= nextCol < cols:
                            if not vis[nextRow][nextCol] and grid[nextRow][nextCol] != 2:
                                vis[nextRow][nextCol] = True
                                q.append((nextRow, nextCol))
                
                # After traversing one level cells, increment the steps by 1 to reach the next level.
                steps += 1
            
            # If we did not reach all houses, then any cell visited also cannot reach all houses.
            # Set all cells visited to 2 so we do not check them again and return INF.
            if housesReached != totalHouses:
                for r in range(rows):
                    for c in range(cols):
                        if grid[r][c] == 0 and vis[r][c]:
                            grid[r][c] = 2
                return float('inf')
            
            # If we have reached all houses then return the total distance calculated.
            return distanceSum
        
        # Main function to find the shortest distance
        minDistance = float('inf')
        rows = len(grid)
        cols = len(grid[0])
        totalHouses = sum(1 for r in range(rows) for c in range(cols) if grid[r][c] == 1)
        
        # Find the min distance sum for each empty cell.
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    minDistance = min(minDistance, bfs(grid, r, c, totalHouses))
        
        # If it is impossible to reach all houses from any empty cell, then return -1.
        if minDistance == float('inf'):
            return -1
        return minDistance

from typing import List   
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        '''
        we can bfs fomr houses to empty lands
        if we can reach a house from an empty land, then we can do the other way
        this is better if there are fewer houses than empty lands
        during bfs, we need to store 2 values for each (i,j) of empty cells
            the totaldist sum from all house to this empty land
            number of houses that can reach this emtpy land
        '''
        rows, cols = len(grid), len(grid[0])
        minDistance = float('inf')
        totalHouses = 0

        # distances[row][col] = [total_distance, houses_reached]
        distances = [[[0, 0] for _ in range(cols)] for _ in range(rows)]

        # Count houses and run BFS from each one
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] == 1:
                    totalHouses += 1
                    self.bfs(grid, distances, row, col)

        # Look for the minimum distance among valid empty lands
        for row in range(rows):
            for col in range(cols):
                if distances[row][col][1] == totalHouses:
                    minDistance = min(minDistance, distances[row][col][0])

        return -1 if minDistance == float('inf') else minDistance
    def bfs(self, grid: List[List[int]], distances: List[List[List[int]]], row: int, col: int) -> None:
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        rows, cols = len(grid), len(grid[0])

        q = deque([(row, col)])
        vis = [[False] * cols for _ in range(rows)]
        vis[row][col] = True

        steps = 0

        while q:
            for _ in range(len(q)):
                r, c = q.popleft()

                if grid[r][c] == 0:
                    distances[r][c][0] += steps
                    distances[r][c][1] += 1

                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not vis[nr][nc] and grid[nr][nc] == 0:
                            vis[nr][nc] = True
                            q.append((nr, nc))
            steps += 1

##############################################
# 2999. Count the Number of Powerful Integers
# 10APR25
#############################################
#TLE, phew
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        '''
        start with the suffix s, then try prepending it with a digit
        note that limit cannot be smaller than any digit in s; removed edge case
        because it can be leading zeros fuck
        but it must be unique
        '''
        powerful = set()
        def rec(s,start,finish):
            if len(s) > len(str(finish)):
                return
            if int(s) > finish:
                return
            if start <= int(s) <= finish:
                powerful.add(int(s))
            
            for prepend in range(limit+1):
                rec(str(prepend)+s,start,finish)

        
        rec(s,start,finish)
        return len(powerful)
            
#almost there
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        '''
        start with the suffix s, then try prepending it with a digit
        note that limit cannot be smaller than any digit in s; removed edge case
        because it can be leading zeros fuck
        but it must be unique
        us digit dp to count integers in range 1 to x
        so it would be dp(finish) - dp(start-1)
        i can still use strings on s, just change the bounds
        two subproblems
            1. use dp to find the number of powerful intergers in the range(1,x)
            its the fucking zeros that are messing shit up
        '''
        @lru_cache(None)
        def dp(s,x):
            if len(s) > len(str(x)):
                return 0
            if int(s) > x:
                return 0
            if int(s) <= x:
                if s.startswith("0"):
                    count = 0
                    for prepend in range(1,limit+1):
                        next_s = str(prepend)+s
                        count += dp(next_s,x)

                    return count

                else:
                    count = 1
                    #issue is with zeros, although the next s can start with zero, we can still prepend to it
                    for prepend in range(limit+1):
                        next_s = str(prepend)+s
                        count += dp(next_s,x)

                    return count
                
        return dp(s,finish) - dp(s,start-1)


######################################
# 2843. Count Symmetric Integers
# 11APR25
######################################
class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        '''
        check all from low to high
        then get left and right digit sum parts
        '''
        count = 0
        for num in range(low,high+1):
            count += self.check_sum(num)
        
        return count
    
    def check_sum(self,num):
        digits = []
        while num:
            digits.append(num % 10)
            num = num // 10
        
        if len(digits) % 2 == 1:
            return False
        return sum(digits[:len(digits)//2]) == sum(digits[len(digits)//2:])
    
class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        '''
        if a number is a multiple of 11 and in between [1,100)
        it is symmetric
        if its four digit number, just part the thoussands, hundres, tens and ones part and check
        '''
        count = 0
        for num in range(low,high+1):
            if num < 100 and num % 11 == 0:
                count += 1
            if 1000 <= num < 10000:
                left_sum = (num // 1000) + (num % 1000) // 100
                right_sum = (num % 100) // 10 + (num % 10)
                if left_sum == right_sum:
                    count += 1
        
        return count

###########################################
# 3272. Find the Count of Good Integers
# 12APR25
###########################################
class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        '''
        k-palindromic means the number x is a palindrome and is divible by k
        its good if it can be arranged to k-palindromic
            any integer cannot have leading zeros before or after
            can't arrange 1010 to 101 or 0110
        
        we can count number of integers that can be palindromic
        for example if n is odd, take n = 3 for example
        the middle can be anything YXY, but can permute this anyway because we allowed to rerrange
        could be YXY, XYY, XYY, for any collection of (Y,X,Y) the are n! perms of it
        and for each we can choose digits 0 to 9, except the first digits cannot be zero
        we can also count the number of digits that are divivible by k up to a limit
        can we actually enumerate all integers? up to 10**5
        almost,
        note counting rule for unique permutations:
            count = factorial(n) / (product(freq(char)) for char in char_counts)!
        '''
        #odd length
        if n % 2 == 1:
            possible = set()
            ans = 0
            for center in range(10):
                if n-2 <= 0:
                    cand = str(center)
                    if int(cand) != 0 and int(cand) % k == 0:
                        possible.add(int(cand))
                else:
                    for left in range(int(10**(n//2))):
                        cand = str(left)+str(center)+str(left)[::-1]
                        if len(cand) == n and int(cand) % k == 0 and cand == str(int(cand)):
                            #sort for uniqueness
                            cand = "".join(sorted(cand))
                            possible.add(cand)
            for num in possible:
                ans += self.count_perms(str(num))
            return ans
    
    #for each sorted can in possible we need to find its contributions
    #this is with leading zeros
    def count_perms(self, num):
        counts = Counter(num)
        n = factorial(len(num))
        denom = 1
        for count in counts.values():
            denom *= factorial(count)
        #if there are zeros, remove the counts with leading or ending
        ans = n // denom
        if '0' in counts:
            leading_zero_count = counts['0']
            temp = len(num) - leading_zero_count
            ans -= factorial(temp)
        return ans

#it works!
class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        '''
        k-palindromic means the number x is a palindrome and is divible by k
        its good if it can be arranged to k-palindromic
            any integer cannot have leading zeros before or after
            can't arrange 1010 to 101 or 0110
        
        we can count number of integers that can be palindromic
        for example if n is odd, take n = 3 for example
        the middle can be anything YXY, but can permute this anyway because we allowed to rerrange
        could be YXY, XYY, XYY, for any collection of (Y,X,Y) the are n! perms of it
        and for each we can choose digits 0 to 9, except the first digits cannot be zero
        we can also count the number of digits that are divivible by k up to a limit
        can we actually enumerate all integers? up to 10**5
        almost,
        note counting rule for unique permutations:
            count = factorial(n) / (product(freq(char)) for char in char_counts)!
        '''
        #odd length
        if n % 2 == 1:
            possible = set()
            ans = 0
            for center in range(10):
                if n-2 <= 0:
                    cand = str(center)
                    if int(cand) != 0 and int(cand) % k == 0:
                        possible.add(int(cand))
                else:
                    for left in range(int(10**(n//2))):
                        cand = str(left)+str(center)+str(left)[::-1]
                        if len(cand) == n and int(cand) % k == 0 and cand == str(int(cand)):
                            #sort for uniqueness
                            cand = "".join(sorted(cand))
                            possible.add(cand)
            return self.count_perms(possible)
        #case for length 2
        else:
            possible = set()
            for left in range(int(10**(n//2))):
                cand = str(left)+str(left)[::-1]
                if len(cand) == n and int(cand) % k == 0 and cand == str(int(cand)):
                    #sort for uniqueness
                    cand = "".join(sorted(cand))
                    possible.add(cand)
            return self.count_perms(possible)
    #for each sorted can in possible we need to find its contributions
    #this is with leading zeros
    #this part taken out of solution, but pretty much got the idea for the first part!
    def count_perms(self, possible):
        ans = 0
        for num in possible:
            num = str(num)
            n = len(num)
            counts = Counter(num)
            #cannot place zero in the first n - count('0') spots
            total = (n - counts['0'])*factorial(n-1)
            #for uniquness
            for x in counts.values():
                total = total // factorial(x)
            ans += total
        
        return ans
    
########################################
# 1922. Count Good Numbers
# 13APR25
#########################################
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        '''
        if we are even indices, we can only use even digits
        if we are odd indices, we can only use odd digits
        n is 10**15, so we need something in logrithmic time
        if n == 1, its 5
        if n == 2, is 5*4
        if n == 3,its 5*4*5
        answer alternates 5,4,5
        count number of 4 and 5's used the use fast exponentaion
        '''
        count_5s = n // 2
        count_4s = n // 2
        if n % 2 == 1:
            count_5s += 1
        
        mod = 10**9 + 7
        
        #return (5**count_5s)*(4**count_4s) % mod
        return self.fast_pow(5,count_5s, mod)*self.fast_pow(4,count_4s, mod) % mod
    
    def fast_pow(self,base,power,mod):
        if power == 0:
            return 1
        half_pow = self.fast_pow(base,power//2,mod)
        if power % 2 == 0:
            return (half_pow % mod)*(half_pow % mod) % mod
        return base*(half_pow % mod)*(half_pow % mod) % mod
    
##########################################
# 1534. Count Good Triplets(REVISITED)
# 14APR25
###########################################
class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        '''
        what if i sort and fix the center?
        '''
        ans = 0
        N = len(arr)
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    x,y,z = arr[i],arr[j],arr[k]
                    if abs(x - y) <= a and abs(y-z) <= b and abs(x-z) <= c:
                        ans += 1
        return ans
    
#O(N*N)
class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        '''
        we can do it in O(N*N) time
        first find pairs (j,k) that satisfy abs(arr[j] - arr[k]) <= b
        then count the pairs with i using pref sum, since numbers are only up to 1000
        if we have this pair, then our possible i's could be in the range
            abs(arr[j] - a) to arr[j] + a
            abs(arr[k] - c) to arr[k] + c
            we can count these pairs using prefsum
        
        updating on the fly
        after we find each (j,k) pair
        need to maintain the condition that the indices ofr the numbers stored so far in the counts satisfy i < j,
        we just need to increment from arr[j] to 1000, 
            becasue for each pair, another i becomes available
        '''
        ans = 0
        n = len(arr)
        counts = [0]*1001
        #all (j,k) pairs
        for j in range(n):
            for k in range(j+1,n):
                if abs(arr[j] - arr[k]) <= b:
                    #find lower bounds for i
                    lower_ij, higher_ij = arr[j] - a, arr[j] + a
                    lower_ik, higher_ik = arr[k] - c, arr[k] + c
                    i_lower = max(0, lower_ij, lower_ik)
                    i_higher = min(1000, higher_ij, higher_ik)
                    if i_lower <= i_higher:
                        ans += counts[i_higher] if i_lower == 0 else counts[i_higher] - counts[i_lower]
            #incrent for each pair
            for k in range(arr[j],1001):
                counts[k] += 1
        
        return ans

#####################################
# 1995. Count Special Quadruplets
# 14APR25
#####################################
#brute force
class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        count = 0
        N = len(nums)
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    for l in range(k+1,N):
                        if nums[i] + nums[j] + nums[k] == nums[l]:
                            count += 1
        return count

#gahh
class Solution:
    def countQuadruplets(self, nums: List[int]) -> int:
        '''
        we can rewrite as 
        nums[a] + nums[b] == nums[d] - nums[c], or any other substtion with (a,b,c)
        ans a < b < c < d
        so we can sort
        '''
        nums.sort()
        n = len(nums)
        counts = Counter(nums)
        ab_pairs = defaultdict(list)
        for i in range(n):
            for j in range(i+1,n):
                ab_pairs[nums[i] + nums[j]].append((i,j))
        
        ans = 0
        for i in range(n-1,-1,-1):
            for j in range(i-1,-1,-1):
                curr_sum = nums[i] - nums[j]
                if curr_sum in ab_pairs:
                    for a,b in ab_pairs[curr_sum]:
                        if a < b < j < i:
                            #need to make sure counts are allowed
                            cand = (nums[a],nums[b],nums[j],nums[i])
                            cand_count = Counter(cand)
                            if all([counts[k] >= v for k,v in cand_count.items()]):
                                ans += 1
        return ans

#########################################
# 2179. Count Good Triplets in an Array
# 15APR25
########################################
#almost
class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        inital thoughts, check for valid triplets in nums1 against them in nums2
            but there could be too many to check
        hints are actually good on this problem
        for each value y in nums1 find the number of values of x that appear before y in both arrays
        same thing for fiding values greater than y
        then for every value of y, count the number of good triplets that can be performed if y is the middle
        you we can just check a pair in nums1, then check again in nums2
        [2,0,1,3] unless is position in the array
        (2,0,1) -> (0,1,2)
        '''
        #try brute force checking
        mapp1 = {num : i for (i,num) in enumerate(nums1)}
        mapp2 = {num : i for (i,num) in enumerate(nums2)}
        pairs = 0
        n = len(nums1)

        for num in range(n):
            #check nums1, smaller
            i = mapp1[num]
            smaller_than_nums1 = set()
            for j in range(i):
                if mapp2[nums1[j]] < num:
                    smaller_than_nums1.add(nums1[j])
            #check larger
            larger_than_nums1 = set()
            i = mapp2[num]
            for j in range(i+1,n):
                if mapp2[nums1[j]] > num:
                    larger_than_nums1.add(nums1[j])
            #check all against num2
            pairs += len(smaller_than_nums1)*len(larger_than_nums1)

        
        return pairs
    
from sortedcontainers import SortedList 
class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we need to count the number of elements smaller than than some number y on both nums1 and nums2
        same for the number of elements after
        answer is the product of thise counts for each num
        #check out this solution
        https://leetcode.com/problems/count-good-triplets-in-an-array/submissions/1608122566/?envType=daily-question&envId=2025-04-15
        '''
        #while traversing nums1 with a num num, we need to know its poistion in b
        n = len(nums1)
        position_nums1_to_nums2 = [0]*n
        for i,num in enumerate(nums2):
            position_nums1_to_nums2[num] = i
        
        #for each number in nums1, find number of elements to the left in both nums1 and nums2
        pos_in_nums2 = SortedList([position_nums1_to_nums2[nums1[0]]]) #sorted indices for nums2, that have seens elements from nums1
        left = [0]
        for i in range(1,n):
            num = nums1[i]
            pos_in_nums2.add(position_nums1_to_nums2[num]) #these give me the indices sorted in nums2
            #now to find the elements smaller in nums1 and nums2, binary search on pos_in_nums2
            left.append(pos_in_nums2.bisect_left(position_nums1_to_nums2[num]))
        
        #now do same, but going right to left
        pos_in_nums2 = SortedList([position_nums1_to_nums2[nums1[-1]]])
        right = [0]
        for i in range(n-2,-1,-1):
            num = nums1[i]
            pos_in_nums2.add(position_nums1_to_nums2[num])
            idx = pos_in_nums2.bisect_left(position_nums1_to_nums2[num])
            right.append(len(pos_in_nums2) - (idx+1))
        
        #count them up
        triplets = 0
        for i in range(n):
            triplets += left[i]*right[n-i-1]
        
        return triplets

#same thing as above, but brute force
class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        brute force version
        '''
        #while traversing nums1 with a num num, we need to know its poistion in b
        n = len(nums1)
        position_nums1_to_nums2 = [0]*n
        for i,num in enumerate(nums2):
            position_nums1_to_nums2[num] = i
        
        #for each number in nums1, find number of elements to the left in both nums1 and nums2
        left = [0]
        seen_from_nums1 = [position_nums1_to_nums2[nums1[0]]]
        for i in range(1,n):
            num = nums1[i]
            seen_from_nums1.append(position_nums1_to_nums2[num])
            count_smaller = 0
            for seen in seen_from_nums1:
                if seen < position_nums1_to_nums2[num]:
                    count_smaller += 1
            left.append(count_smaller)
        
        right = [0]
        seen_from_nums1 = [position_nums1_to_nums2[nums1[-1]]]
        for i in range(n-2,-1,-1):
            num = nums1[i]
            seen_from_nums1.append(position_nums1_to_nums2[num])
            count_larger = 0
            for seen in seen_from_nums1:
                if seen > position_nums1_to_nums2[num]:
                    count_larger += 1
            right.append(count_larger)

        #count them up
        triplets = 0
        for i in range(n):
            triplets += left[i]*right[n-i-1]
        
        return triplets

from sortedcontainers import SortedList
class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        '''
        mapp = {num : i for i,num in enumerate(nums1)}
        triplets = 0
        n = len(nums1)
        sl = SortedList([])
        for num in nums2:
            #idex stores the current index of the element we are currenlty on in nums2 that's in nums1
            idx = mapp[num]
            #find number of element smaller than our current idx, that we've seen from nums1
            #what we have processed so far
            left = sl.bisect_left(idx)
            #(n-1-idx) are what we haven't processed
            #len(sl) - left are the ones we've seen so far
            right = (n - 1 - idx) - (len(sl) - left)
            triplets += left*right
            sl.add(idx) #add indices
        
        return triplets
    
#two pass
from sortedcontainers import SortedList
class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        '''
        mapp = {num : i for i,num in enumerate(nums1)}
        triplets = 0
        n = len(nums1)
        sl = SortedList([])
        count_left = []
        for num in nums2:
            idx = mapp[num]
            left = sl.bisect_left(idx)
            count_left.append(left)
            sl.add(idx)
        
        sl = SortedList([])
        count_right = []
        for num in nums2[::-1]:
            idx = mapp[num]
            right = sl.bisect_left(idx)
            count_right.append(len(sl) - (right ))
            sl.add(idx)
        
        #count them up
        triplets = 0
        for i in range(n):
            triplets += count_left[i]*count_right[n-i-1]
        
        return triplets
    
#segment tree
class SegmentTree:
    def __init__(self, arr):
        #make three
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 0, 0, self.n - 1)

    def build(self, arr, node, start, end):
        if start == end:
            self.tree[node] = arr[start]
        else:
            mid = (start + end) // 2
            self.build(arr, 2 * node + 1, start, mid)
            self.build(arr, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def update(self, idx, val, node=0, start=None, end=None):
        if start is None:
            start = 0
            end = self.n - 1
        if start == end:
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(idx, val, 2 * node + 1, start, mid)
            else:
                self.update(idx, val, 2 * node + 2, mid + 1, end)
            self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def query(self, l, r, node=0, start=None, end=None):
        if start is None:
            start = 0
            end = self.n - 1
        if l > end or r < start:
            return 0
        if l <= start and end <= r:
            return self.tree[node]
        mid = (start + end) // 2
        leftSum = self.query(l, r, 2 * node + 1, start, mid)
        rightSum = self.query(l, r, 2 * node + 2, mid + 1, end)
        return leftSum + rightSum


class Solution:
    def goodTriplets(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        trick is that this is a frequency count based segment tree
        '''
        n = len(nums1)
        mapp = {num: i for i, num in enumerate(nums1)}

        seg_left = SegmentTree([0] * n)
        count_left = []
        for num in nums2:
            idx = mapp[num]
            #queries how many times we've seen an index from nums1 in nums2
            left = seg_left.query(0, idx - 1)
            count_left.append(left)
            #give is the count at this idx
            curr_val = seg_left.query(idx, idx)
            #increment the count for this index by 1
            seg_left.update(idx, curr_val + 1)

        seg_right = SegmentTree([0] * n)
        count_right = []
        for num in reversed(nums2):
            idx = mapp[num]
            right = seg_right.query(idx + 1, n - 1)
            count_right.append(right)
            curr_val = seg_right.query(idx, idx)
            seg_right.update(idx, curr_val + 1)
        
        triplets = 0
        for i in range(n):
            triplets += count_left[i] * count_right[n - i - 1]
        
        return triplets


###########################################
# 2537. Count the Number of Good Subarrays
# 16APR25
###########################################
#sliding window, counting ending
class Solution:
    def countGood(self, nums: List[int], k: int) -> int:
        '''
        need at least length 2 subarray to have a pair
        if we have a good subarray, we could keep adding to it so that way it continues to have pairs
        if we have m counts of some number, then we can make m*(m-1)/2 pairs
            but this would require checking over the count map, and numbers could get very big
            so we overcount and then remove counts
        if we are at index i, and have met the conition, then there are n-i good subarrays
            adding any more just increases the counts and we are alreadt at least k pairs
        '''
        left = 0
        counts = Counter()
        good_subs = 0
        curr_pairs = 0
        n = len(nums)

        for right in range(n):
            #add in
            num = nums[right]
            #for every new num, we add in the prev count of pairs
            curr_pairs += counts[num]
            counts[num] += 1
            while left < right and curr_pairs >= k:
                good_subs += n - right
                num = nums[left]
                #when we remove from subarray, we need to decrement and remove pair count
                counts[num] -= 1
                curr_pairs -= counts[num]
                left += 1
            
        return good_subs

#######################################################
# 2176. Count Equal and Divisible Pairs in an Array
# 17APR25
########################################################
class Solution:
    def countPairs(self, nums: List[int], k: int) -> int:
        '''
        brute force
        '''
        pairs = 0
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                if nums[i] == nums[j] and i*j % k == 0:
                    pairs += 1
        
        return pairs
    
##########################################
# 2145. Count the Hidden Sequences
# 21APR25
##########################################
class Solution:
    def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
        '''
        say we have the hidden array [a,b,c,d]
        and diffs [x,y,z], then have
        b - a = x
        c - b = y
        d - c = z
        make a sequence, then see how many values we can fit in this sequence that are in between upper and lower
        if we fix the starting as zero, then we can shift all the values in the array by k
        but the shift (k + num) is limited to upper and lower
        for each num in the new array lower <= (num + k) <= upper
        if it's in between upper and lower, we are free to use (upper - lower) + 1 elements
        i.e the shift is limited by the range of any hidden array
        another way to think about it is that you are given a time series, and you want to fit the whole
        time series in between lower and upper we can do (upper - max_vlaue) + (min_val - lower)

        '''
        start = 0
        seq = [start]
        for d in differences:
            seq.append(seq[-1] + d)
        
        bounds = max(seq) - min(seq)
        #can't have negative count
        return max(0,(upper - lower + 1) - bounds)
    
class Solution:
    def numberOfArrays(self, differences: List[int], lower: int, upper: int) -> int:
        '''
        one pass,
        just maintain min and max
        '''
        curr = 0
        min_ = 0
        max_ = 0

        for d in differences:
            curr += d
            min_ = min(min_, curr)
            max_ = max(max_, curr)
        
        return max(0, (upper - lower + 1) - (max_ - min_))
    
#############################################################
# 2311. Longest Binary Subsequence Less Than or Equal to K
# 21APR25
#############################################################
#dp, with states (i,path) MLE
class Solution:
    def longestSubsequence(self, s: str, k: int) -> int:
        '''
        let dp(i) be the length of the longest subsequence <= k
        i need to keep string paths 2 though
        '''
        n = len(s)
        memo = {}
        def dp(i,path):
            if i >= n:
                if path and int(path,2) <= k:
                    return len(path)
                
                return 0
            if path and int(path,2) > k:
                return 0
            if (i,path) in memo:
                return memo[(i,path)]
            #take
            take = dp(i+1,path+s[i])
            no_take = dp(i+1,path)
            ans = max(take,no_take)
            memo[(i,path)] = ans
            return ans

        return dp(0,"")

#converting to strings to int, still MLE
class Solution:
    def longestSubsequence(self, s: str, k: int) -> int:
        '''
        let dp(i) be the length of the longest subsequence <= k
        i need to keep string paths 2 though
        '''
        n = len(s)

        @lru_cache(None)
        def dp(i, val):
            if val > k:
                return float('-inf')  # invalid
            if i == n:
                return 0

            # Option 1: skip s[i]
            not_take = dp(i + 1, val)

            # Option 2: take s[i]
            take = float('-inf')
            new_val = (val << 1) | int(s[i])
            if new_val <= k:
                take = 1 + dp(i + 1, new_val)

            return max(take, not_take)

        return dp(0, 0)
    
class Solution:
    def longestSubsequence(self, s: str, k: int) -> int:
        '''
        let dp(i) be the length of the longest subsequence <= k
        i need to keep string paths 2 though
        we are essentially building up the smallest int(subseqs) <= k
        then we need to minimize each dp(i), such that adding another bit make it smaller than k
        then we are left with subseeqneces with values <= k, that are of length i
        '''
        dp = [0]
        n = len(s)
        for i in range(n):
            v = int(s[i])
            if dp[-1]*2 + v <= k:
                dp.append(dp[-1]*2 + v)
            #minimize and see if adding this to any previous subsequencce, but maintain length i
            for j in range(len(dp) - 1, 0,-1):
                dp[j] = min(dp[j], dp[j-1]*2 + v)
        
        return len(dp) - 1


#greedily take zeros
class Solution:
    def longestSubsequence(self, s: str, k: int) -> int:
        '''
        greedily take all zeros, then try to take the right most ones
        '''
        curr_num = 0
        length = 0
        for ch in s[::-1]:
            #zero adds to length, but doen'st increase num
            if ch == '0':
                length += 1
            #otherwise its a 1, so we need to check if we can take it, so we shift by the length (2**power)
            #and if its <= k, we can add it
            elif curr_num + (1 << length) <= k:
                curr_num = curr_num + (1 << length)
                length += 1
        
        return length


############################################
# 2338. Count the Number of Ideal Arrays
# 21APR25
############################################
#TLE
class Solution:
    def idealArrays(self, n: int, maxValue: int) -> int:
        '''
        an array is ideal of every num in arr is in between 1 and maxValue
        and arr[i] % arr[i-1] == 0 for all i
        say we are building an array, and up to this point we [...,k] and we are ideal so far
        we can add to this another number j, such that j % k == 0
        states are index i, and currvalue
        '''
        memo = {}
        mod = 10**9 + 7
        def dp(i,curr_num):
            if i >= n:
                return 1
            if (i,curr_num) in memo:
                return memo[(i,curr_num)]
            ways = 0
            for next_num in range(curr_num,maxValue+1):
                if next_num % curr_num == 0:
                    ways += dp(i+1,next_num)
                    ways %= mod
            memo[(i,curr_num)] = ways % mod
            return ways % mod
        
        ans = 0
        for i in range(1,maxValue+1):
            ans += dp(1,i)
            ans %= mod
        return ans % mod
    
#bottle neck is cacluting ways in dp
#use combinatorics
import math
class Solution:
    def idealArrays(self, n: int, maxValue: int) -> int:
        '''
        need to some linearish algo to complete
        first notice that [1,1,1..1] to [maxValue,maxValue,...] are always ideal, so we have at least maxValue arrays
        make more ideal arrays for each of these starting ones
        record all starting values 1 to maxvalue as 1
        in counts array, num, mean [num,num..num] array 
        if we start with 2, and n= 5, then we could have [1,x,x,x,x], and we are free to place 2 in any of the n-1 spots
            this contributes comb(n-1,2)
        so we try this for all k, ie. all k comb(n-1,k)
        it could be at position 1: [1,2,2,2,2], 2: [1,1,2,2,2], 3: [1,1,1,2,2], 4: [1,1,1,1,2] but no more. We cannot have [1,1,1,1,1].
        code, k means number of transitions (from if the whole array only has 1 and 2, k = 1 transition)
        If the array is like this: [1,2,4,4,4]: k = 2 there are two transitions
            k is the transistion point where we get new values
        we are choosing k transition point from (n-1) possibilities, therefore (n-1)Ck ways
        progressively increment the number of elements to put into the array and keep counting the number of ways
        count map is for counting how many ways the last biggest number in the array can be reached. 
            For example, numbers like 32 can be 2*16 and  4*8

        '''
        ways = maxValue #initalize to maxValue,
        mod = 10**9 + 7
        #i.e for each value v to maxValue we can have the array [v,v,v...v] as a possile ideal array
        counts = {num : 1 for num in range(1,maxValue + 1)}
        #number of changes in the array
        for k in range(1,n):
            temp = Counter()
            for last_int in counts.keys():
                for mult in range(2,maxValue // last_int + 1):
                    new_last_int = last_int*mult
                    ways += math.comb(n-1,k)*counts[last_int]
                    ways %= mod
                    temp[new_last_int] += counts[last_int]

            counts = temp #dp swap

        return ways % mod
    
#combinatorics paradigm known as 'stars and bars'
#https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
class Solution:
    def idealArrays(self, n: int, maxValue: int) -> int:
        '''
        crux of the problem is getting number of ways we can generate an ideal string
        if all the elements in the array are increasing and num[i+1] % [i] == 0
        Computes the number of ways to distribute (n-1) gaps into (k-1) spaces. In other words:

        From an array of length k, if you fix the values, 
        you can place them in any of the n positions, choosing k positions out of n, preserving order.

        This is equivalent to C(n-1, k-1) — standard "stars and bars" combinatorics.
        say we have the array [1,1] this contributes 1 way, and is given by comb(2-1,2-1), same for all [1,2],[1,3]
        so we really just need to keep length and the last number
        '''
        mod = 10**9+7
        @cache
        def getcomb(k):
            # put n identical balls into k boxes, ensuring each box have >=1 balls
            return comb(n-1, k-1)
        @cache
        def dfs(v, l):
            # opt 1: followed by no new value, composing array ended with v
            res = comb(n-1, l-1) if l else 0
            if l==n:
                return res
            
            # opt 2: followed by multiplier of v
            for nv in range(2*v, maxValue+1, v):
                res += dfs(nv, l+1)
                res %= mod
            return res % mod
        
        ways = 0
        for i in range(1,maxValue + 1):
            ways += dfs(i,1)
            ways %= mod
        
        return ways % mod
        

#BFS, TLE
import math
class Solution:
    def idealArrays(self, n: int, maxValue: int) -> int:
        '''
        BFS TLE, we can represent a valid sequence of elements [last_num,lengt]
        [1,3] is [3,2], [1,2,4,8] is [8,4]
        inital states are [i,1] for i in range(maxValue+1)
        then we use combinatorics
        need comb(n-1,k-1), computes number of valid arrays from k numbers (stars and bars)
        question is number of ways to put n unique balls in k unique bines
        number of k length tuples among n items
        '''
        mod = 10**9 + 7
        q = deque([])
        for i in range(1,maxValue + 1):
            q.append([i,1])
        
        count = 0
        while q:
            last_num,curr_length = q.popleft()
            count += math.comb(n-1,curr_length-1) #stars and bars, but why
            #this contributes a partial count since curr_length could be < n, but we need to get them all
            count %= mod
            next_num = 2*last_num
            if curr_length == n or next_num > maxValue:
                continue #this can no longer contribute a count
            while next_num <= maxValue:
                q.append([next_num,curr_length + 1])
                next_num += last_num
        
        return count % mod

class Solution:
    def idealArrays(self, n: int, maxValue: int) -> int:
        '''
        dp states are (value,length), where value is the length of the last number added to the array
        and length is the number, the number of ways to generate this is comb(n-1,length-1), stars and bars
            ways to permute array but presevering order
        we can also precompute the binomial coeffcient table:
            C(n, k) = C(n-1, k) + C(n-1, k-1)
        '''
        # Comb table creation using dynamic programming to calculate combinations
        comb_table = [[0] * 16 for _ in range(n)]
        mod = 10**9 + 7  # Module for the problem, to prevent integer overflow
        for i in range(n):
            for j in range(min(16, i + 1)):
                # Base case: There is one way to choose 0 items
                if j == 0:
                    comb_table[i][j] = 1
                else:
                    # Use the recurrence relation C(n, k) = C(n-1, k) + C(n-1, k-1)
                    comb_table[i][j] = (comb_table[i - 1][j] + comb_table[i - 1][j - 1]) % mod
        
        #dp with caching
        @lru_cache(maxsize=None)
        def dp(value, length):
            """ 
            Performs a depth-first search starting with a given value
            and length of the sequence, and returns the number of ideal arrays.
            """
            # Initialize result with the combination C(n-1, length-1)
            result = comb_table[-1][length - 1]
            # If we haven't reached the desired length, continue building the sequence
            if length < n:
                multiple = 2
                # Iterate through the possible multiples of 'value' within the maxValue
                while multiple * value <= maxValue:
                    # Recursively call dfs and update the result
                    result = (result + dp(multiple * value, length + 1)) % mod
                    multiple += 1
            return result
        
        ways = 0
        for i in range(1,maxValue+1):
            ways += dp(i,1) % mod
        
        return ways % mod

############################################
# 2799. Count Complete Subarrays in an Array
# 24MAY25
############################################
#brute force
class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        '''
        we need count of distint elements == number of distinct elemetns in the whole array
        if we have a subarry array that already meets the criteria, we could add to it, if there are other eleements 
        so we can't shrink when we meet the criteria
        but if we go over the criterie, adding will not help
        so we need to shrink when we go over
        '''
        distinct = len(set(nums))
        ans = 0
        n = len(nums)
        for i in range(n):
            window = set()
            for j in range(i,n):
                window.add(nums[j])
                if len(window) == distinct:
                    ans += 1
        
        return ans
    
#indireclty counting when we have len(window) == distinct
class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        '''
        we need count of distint elements == number of distinct elemetns in the whole array
        if we have a subarry array that already meets the criteria, we could add to it, if there are other eleements 
        so we can't shrink when we meet the criteria
        but if we go over the criterie, adding will not help
        so we need to shrink when we go over
        indirectly counting, if we have a valid window, len(window) == distinct, we can shrink
        and it means any subarray before i and ending at j, has distint values
        '''
        distinct = len(set(nums))
        window = Counter()
        ans = 0
        left = 0
        n = len(nums)
        for right in range(n):
            num = nums[right]
            window[num] += 1
            while len(window) == distinct:
                num = nums[left]
                window[num] -= 1
                if window[num] == 0:
                    del window[num]
                left += 1
            ans += left
        
        return ans

class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        '''
        we can't have mnore than disctinct elements
        so once a window == distinct, anything ending must be good
        '''
        distinct = len(set(nums))
        window = Counter()
        ans = 0
        left = 0
        n = len(nums)
        for right in range(n):
            num = nums[right]
            window[num] += 1
            while len(window) == distinct:
                ans += n - right
                num = nums[left]
                window[num] -= 1
                if window[num] == 0:
                    del window[num]
                left += 1
        
        return ans
    
#######################################
# 1138. Alphabet Board Path
# 25APR25
########################################
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        '''
        make a hashmap for each letter in the board, then get the moves to that positions
        shit the problem is getting to z, we can't walk outside the board
        i can use dfs/bfs to find the next letter so that way i dont walk off the board
        '''
        board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z****"] #padd z with special chars
        mapp = {}
        for i,row in enumerate(board):
            for j,ch in enumerate(row):
                mapp[ch] = (i,j)
        
        curr = [0,0]
        ans = ""
        for letter in target:
            i,j,path = self.find_neigh(board,curr,letter)
            ans += path
            ans += '!'
            curr = [i,j]
        
        return ans
    
    def find_neigh(self,board,start,next_char):
        rows,cols = len(board),len(board[0])
        dirrs = [(1,0,'D'),(-1,0,'U'),(0,1,'R'),(0,-1,'L')]
        i,j = start
        seen = set()
        q = deque([(i,j,"")])
        while q:
            i,j,path = q.popleft()
            if board[i][j] == next_char:
                return [i,j,path]
            seen.add((i,j))
            for di,dj,letter in dirrs:
                ii = i + di
                jj = j + dj
                if 0 <= ii < rows and 0 <= jj < cols and board[ii][jj] != '*' and (ii,jj) not in seen:
                    q.append((ii,jj,path+letter))
            
class Solution:
    def alphabetBoardPath(self, target: str) -> str:
        '''
        same as first approach, but check L U before R D
        since R,D might leave outside the board
        '''
        board = ["abcde", "fghij", "klmno", "pqrst", "uvwxy", "z"]
        mapp = {}
        for i,row in enumerate(board):
            for j,ch in enumerate(row):
                mapp[ch] = (i,j)
        
        x0, y0 = 0, 0
        res = []
        for c in target:
            x, y = mapp[c]
            if y < y0: res.append('L' * (y0 - y))
            if x < x0: res.append('U' * (x0 - x))
            if x > x0: res.append('D' * (x - x0))
            if y > y0: res.append('R' * (y - y0))
            res.append('!')
            x0, y0 = x, y
        return "".join(res)
    
############################################
# 2845. Count of Interesting Subarrays (brute force)
# 25APR25
#############################################
class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        '''
        a subarry is interesting if all nums in subarray % modulo == k
        then length % modulo = k
        [a,b,c,d] is interseting if 
            all(a % modulo == k)
        convert each num to num % modulo
        brute force problem is to decode the array into a pref sum array, where counts[i] is the number of indices
        where nums[i] % modulo = k
        i.e turn nums into pref_sum, where the pref_sum is nums[i] % modulo = k
        then we just need to check all subarrays where (counts[i] - counts[j]) % modulo == k
        '''
        #convert each num to % modulo
        n = len(nums)
        counts = [0]
        for num in nums:
            temp = (num % modulo) == k
            counts.append(counts[-1] + temp)
        
        interesting = 0
        for i in range(n+1):
            for j in range(0,i):
                if (counts[i] - counts[j]) % modulo == k:
                    interesting += 1
        return interesting

#using hashmap for complement search
class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        '''
        a subarry is interesting if all nums in subarray % modulo == k
        then length % modulo = k
        [a,b,c,d] is interseting if 
            all(a % modulo == k)
        convert each num to num % modulo
        brute force problem is to decode the array into a pref sum array, where counts[i] is the number of indices
        where nums[i] % modulo = k
        i.e turn nums into pref_sum, where the pref_sum is nums[i] % modulo = k
        then we just need to check all subarrays where (counts[i] - counts[j]) % modulo == k
        '''
        #convert each num to % modulo
        n = len(nums)
        counts = [0]
        for num in nums:
            temp = (num % modulo) == k
            counts.append(counts[-1] + temp)
        
        interesting = 0
        mapp = Counter()

        for count in counts:
            interesting += mapp[(count + modulo - k) % modulo]
            mapp[count % modulo] += 1

        return interesting

###############################################################
# 2137. Pour Water Between Buckets to Make Water Levels Equal
# 25APR25
################################################################
class Solution:
    def equalizeWater(self, buckets: List[int], loss: int) -> float:
        '''
        we need to equalize the buckets, but every time we pour, we lose ((loss)/100)*poured
        if we didn't lose any water, we could just bring them all the the average
        binary serach on answer
        try to pour for a certain amount, with each pour, make sure we have water
        '''
        left = 0
        right = max(buckets)
        delta = 1e-5
        ans = 0

        while right - left >= delta:
            mid = left + (right - left)/2
            if self.f(buckets,mid,loss):
                ans = mid
                left = mid
            else:
                right = mid

        return ans
    
    def f(self,buckets,target,loss):
        excess = 0
        for b in buckets:
            if b >= target:
                #this bucket is more than target, so we have extra water
                excess += (b-target)*(1 - loss/100)
            else:
                #need to use excess water to fill
                excess -= target - b
        return excess >= 0
    
#####################################################
# 2444. Count Subarrays With Fixed Bounds (REVISTED)
# 26APR25
#####################################################
#not quite
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        '''
        we need the min of subarray and max of subarray == k
        what if all the numbers in a subarray are in between minK and maxK?
            then i can only count subarrays that include minK and maK
            if we count m counts of minK and n counts of maxK in a subarray
        
        divide subarrays so that each subarray is in between minK and maxK
        inclusion exclusioon
            if a subarray is of length k, then there are k*(k+1) // 2 subarrays
            of these, how many subarrays don't have min and max as minK and maxK?
            use counts, say we have this subarray and at i is minK and at j is maxK
            we can fix these (i,j), and extend the bounds to the left and right, the answer is the product
            but then we would have to this for each subarray
        '''
        subs = []
        curr_sub = []
        for num in nums:
            if minK <= num <= maxK:
                curr_sub.append(num)
            else:
                subs.append(curr_sub)
                curr_sub = []
        if curr_sub:
            subs.append(curr_sub)
        
        #solve for each subarray
        ways = 0
        for sub in subs:
            ways += self.solve(sub,minK,maxK)
        
        return ways
    
    def solve(self,arr,minK,maxK):
        #need to store the leftmost and rightmost min
        #also need to store the leftmost and rightmost max
        #store the indices for min and max
        n = len(arr)
        min_idxs = []
        max_idxs = []
        for i,num in enumerate(arr):
            if num == minK:
                min_idxs.append(i)
            if num == maxK:
                max_idxs.append(i)
        if minK == maxK:
            k = len(min_idxs)
            return k*(k+1) //2
        count = 0
        for i in min_idxs:
            for j in max_idxs:
                left = min(i,j)
                right = max(i,j)
                count += (left+1)*(n - right)
        return count
        
#n*n
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        '''
        for N sqaured, keep track if we;ve seen minK and maxK
        if we are outsdide, we can't include this array
        fix i
        '''
        N = len(nums)
        count = 0
        for i in range(N):
            min_found = max_found = False
            for j in range(i, N):
                if nums[j] < minK or nums[j] > maxK:
                    break
                if nums[j] == minK:
                    min_found = True
                if nums[j] == maxK:
                    max_found = True
                if min_found and max_found:
                    count += 1
        return count
    
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        N = len(nums)
        count = 0
        min_pos = max_pos = last_oob = -1
        for i in range(N):
            if nums[i] == minK: # find left/right bound
                min_pos = i
            if nums[i] == maxK: # find left/right bound
                max_pos = i
            if minK > nums[i] or nums[i] > maxK: # find last out of bounds num pos
                last_oob = i
                min_pos = max_pos = -1 # reset bounds
            if min_pos != -1 and max_pos != -1: # if in bounds -> count between leftmost bound and last_oob
                count += min(min_pos, max_pos) - last_oob
        return count
    
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        '''
        record last out of bound index
        '''
        N = len(nums)
        count = 0
        min_pos = max_pos = last_oob = -1
        for i,num in enumerate(nums):
            if num == minK:
                min_pos = i
            if num == maxK:
                max_pos = i
            #out of range
            if num > maxK or num < minK:
                last_oob = i
                min_pos = max_pos = last_oob = -1 #reset
            if min_pos != -1 and max_pos != -1: # if in bounds -> count between leftmost 
                count += max(0, min(min_pos,max_pos)) - last_oob
        
        return count
    
#########################################################
# 3392. Count Subarrays of Length Three With a Condition
# 27APR25
#########################################################
#ez sunday, phew
class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        '''
        just check every three
        better to use multiplication instead of divsion, so you don't get floating numbers
        '''
        count = 0
        n = len(nums)

        for i in range(0,n-2):
            if (nums[i] + nums[i+2])*2 == nums[i+1]:
                count += 1
        
        return count
    
##############################################
# 2302. Count Subarrays With Score Less Than K
# 28APR25
##############################################
#ezzzz
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        '''
        see how score grows:
        [a,b,c,d]
        1 -> a
        2 -> 2a + 2b
        3 -> 3a + 3b + 3c
                test = [1,2,3,4,5]
        size = 0
        curr_sum = 0
        for num in test:
            curr_sum += num
            curr_score = curr_sum*(size+1)
            print(curr_score)
            size += 1
        
        return 0
        
        if curr_score is x, then adding some number y increases by y*(l+1)
        now what  happens when we shrink?

        addinf only increases the score, so once we are at score k, we don't need to keep adding
        keep track of sum and length in window, when we shrink we incrment  count

        fucking ez bro
        '''
        curr_sum = 0
        count = 0
        left = 0
        for right,num in enumerate(nums):
            curr_sum += num
            #curr_score = curr_sum*(right - left + 1)
            #shrink while we are too big
            while left <= right and curr_sum*(right - left + 1) >= k:
                curr_sum -= nums[left]
                left += 1
            #we have a valid window, so anything ending at right from left is valie
            count += right - left + 1

        return count

#############################################################################
# 2962. Count Subarrays Where Max Element Appears at Least K Times (REVISTED)
# 29APR25
#############################################################################
#variant 1, adding n - right for valid windows
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        '''
        just keep track of max(nums)
        '''
        count = 0
        window_count = 0
        left = 0
        n = len(nums)
        max_num = max(nums)

        for right in range(n):
            #only add if num is max nums
            num = nums[right]
            window_count += num == max_num
            while left <= right and window_count >= k:
                count += n - right
                num = nums[left]
                window_count -= nums[left] == max_num
                left += 1
        
        return count
    
#variant 2, addin in left, 
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        '''
        maintain window with count
        when counts == k, we know this subarray from left to right is good
        so we shrink until we are no longer good, then this mean that the number of good windows is the number of times we moved left
        '''
        count = 0
        window_count = 0
        left = 0
        n = len(nums)
        max_num = max(nums)

        for right in range(n):
            #only add if num is max nums
            num = nums[right]
            window_count += num == max_num
            while window_count == k:
                num = nums[left]
                window_count -= nums[left] == max_num
                left += 1
            count += left
        
        return count