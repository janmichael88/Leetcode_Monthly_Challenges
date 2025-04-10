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
