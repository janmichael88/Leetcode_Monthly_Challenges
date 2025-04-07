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
        
            
        