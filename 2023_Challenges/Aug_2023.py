##################################
# 77. Combinations (REVISTED)
# 01AUG23
##################################
#recurse, backtrack with taken array
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        '''
        number of combinations is given as nCk, i.e n choose k
        if we have some indices (i,j,k), any permuation of (i,j,k) is considered the same combination
        i.e combinations are unordered
        backtracking with taken array, and keep track if items taken, 
        when k == 0, we have a pair, then just bactrack
        '''
        ans = []
        taken = [False]*n
        nums = [i for i in range(1,n+1)]
        def backtrack(i,taken):
            if i == n:
                return
            if sum(taken) == k:
                curr_comb = []
                for j in range(n):
                    if taken[j] == True:
                        curr_comb.append(nums[j])
                
                ans.append(curr_comb)
                return
            
            for j in range(i,n):
                if not taken[j]:
                    taken[j] = True
                    backtrack(j,taken)
                    taken[j] = False
        
        
        backtrack(0,taken)
        return ans
    
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        '''
        back track, no taken array, but prune 
        '''
        ans = []
        def backtrack(first_num,path):
            if len(path) == k:
                ans.append(path[:])
                return
            
            nums_needed = k - len(path)
            remaining = n - first_num + 1
            nums_available = remaining - nums_needed
            #If we moved to a child outside of this range, like firstNum + available + 1, then we will run out of numbers to use before reaching a length of k.
            
            for next_num in range(first_num,first_num + nums_available + 1):
                path.append(next_num)
                backtrack(next_num+1,path)
                path.pop()
        
        backtrack(1,[])
        return ans
    
