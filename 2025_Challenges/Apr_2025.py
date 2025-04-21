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

        '''
        start = 0
        seq = [start]
        for d in differences:
            seq.append(seq[-1] + d)
        
        bounds = max(seq) - min(seq)
        #can't have negative count
        return max(0,(upper - lower + 1) - bounds)