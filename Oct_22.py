############################################
# 1155. Number of Dice Rolls With Target Sum
# 02COT22
##########################################
#bactracking TLE
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        '''
        i can solve this using backtracking, but it might time out
        '''
        self.num_ways = 0
        
        def backtrack(num_die,curr_score):
            if num_die > n:
                return
            if curr_score > target:
                return
            if curr_score == target:
                self.num_ways += 1
                self.num_ways %= 10**9 + 7
                return
            for side in range(1,k+1):
                backtrack(num_die+1,curr_score + side)
    

        backtrack(0,0)
        return self.num_ways

#combine subproblems at a subroot
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        '''
        top down
        '''
        memo = {}
        
        def backtrack(num_die,curr_score):
            if num_die == 0:
                if curr_score == 0:
                    return 1
                else:
                    return 0
            if (num_die,curr_score) in memo:
                return memo[(num_die,curr_score)]
            ans = 0
            for side in range(1,k+1):
                ans += backtrack(num_die-1,curr_score - side)
            ans %= 10**9 + 7
            #print(num_die,curr_score)
            memo[(num_die,curr_score)] = ans
            return ans
    

        return backtrack(n,target)

class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        '''
        bottom up dp
        '''
        memo = {}
        
        dp = [[0]*(target+1) for _ in range(n+1)]
        
        #base cases
        dp[0][0] = 1
        
        for num_die in range(1,n+1):
            for curr_score in range(1,target+1):
                ans = 0
                for side in range(1,k+1):
                    if num_die - 1 >= 0 and curr_score - side >= 0:
                        ans += dp[num_die-1][curr_score - side]
                        
                ans %= 10**9 + 7
                dp[num_die][curr_score] = ans

        return dp[n][target]

##########################################
# 1578. Minimum Time to Make Rope Colorful 
# 03OCT22
##########################################
#FUCKING EASSSSSY boyyyyy XDDD
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        i can scan to find the groupings of balloons
        then in each grouping, delete the ballones that take up the smallest needed time
        i can use two pointers and sliding window to find the groups
        '''
        N = len(colors)
        left,right = 0,0
        min_time = 0
        
        while left < N:
            #if we can expand the current grouping
            while right + 1 < N and colors[right+1] == colors[left]:
                right += 1
            
            #if we have a window
            if right > left:
                #get the max in this window
                max_ = max(neededTime[left:right+1])
                sum_ = sum(neededTime[left:right+1])
                min_time += sum_ - max_
            
            right += 1
            left = right
        
        return min_time
            
#actual solutions from write up
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        the intuions are that we need to only keep at least one balloon in the group and that we keep the balloon with the maximuim
        algo:
            init totaltime, left, right and 0
            pass over balloons and for each group, record the total removal time
        '''
        total_time = 0
        i,j = 0,0
        
        N = len(colors)
        
        while i < N and j < N:
            curr_total = 0
            curr_max = 0
            
            #final all balongs have same color, and update max and totals
            while j < N and colors[i] == colors[j]:
                curr_total += neededTime[j]
                curr_max = max(curr_max,neededTime[j])
                j += 1
            
            #first pas is zero anyway
            total_time += curr_total - curr_max
            i = j
        
        return total_time

class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        we can do this in one pass using one pointer by adding the smaller removal times directly to the answer
        intuition:
            for each group, we always record the largest removal time (lets call it currMaxTime for convience)
            and add other smaller removal times to totalTime
            when we have another newly added removal time (t[i]) that belongs to the curret group, we compare t[i]
             with currMaxTime, add the samller one totoalTime, and leave the largerone as the currmaxtime
        '''
        total_time = 0
        curr_max_time = 0 #maxtime for current group
        N = len(colors)
        
        for i in range(N):
            #if this ballong is the first baollong of a new group, rest
            if i > 0 and colors[i] != colors[i-1]:
                curr_max_time = 0
                
            total_time += min(curr_max_time,neededTime[i])
            curr_max_time = max(curr_max_time,neededTime[i])
        
        return total_time

############################
# 112. Path Sum (REVISTED)
# 04OCT22
############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        '''
        just dfs carrying sum along the way
        '''
        self.ans = False
        
        def dfs(node,curr_sum):
            if not node:
                return
            dfs(node.left,curr_sum + node.val)
            if not node.left and not node.right and curr_sum + node.val == targetSum:
                self.ans = True
            dfs(node.right,curr_sum + node.val)
        
        
        dfs(root,0)
        return self.ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        '''
        let dp(node,sum) return whether or not we have a valid root to leaf == targetsum
        then dp(node,sum) = dp(node.left,sum+node.val) or dp(node.right,sum_node.val)
        base case, empty not cannot be an answer
        '''
        def dp(node,sum_):
            if not node:
                return False
            if not node.left and not node.right and sum_ + node.val == targetSum:
                return True
            left = dp(node.left,sum_ + node.val)
            right = dp(node.right,sum_ + node.val)
            return left or right
        
        return dp(root,0)


