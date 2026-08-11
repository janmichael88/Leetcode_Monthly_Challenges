######################################################
# 3731. Find Missing Elements
# 04AUG26
######################################################
class Solution:
    def findMissingElements(self, nums: List[int]) -> List[int]:
        '''
        sort and record
        '''
        missing = []
        nums.sort()
        n = len(nums)
        i = 0
        while i < n:
            if i + 1 < n and nums[i+1] - nums[i] != 1:
                diff = nums[i+1] - nums[i]
                for d in range(1,diff):
                    missing.append(nums[i] + d)
            i += 1
        
        return missing
    
##################################################
# 3310. Remove Methods From Project
# 05AUG26
##################################################
class Solution:
    def remainingMethods(self, n: int, k: int, invocations: List[List[int]]) -> List[int]:
        '''
        we need to remove the suspicous projects
        a group of methods can be removed only iff no method outside the group invokes any methods within it
        if there is a cycle, those nodes need to be removed
            cycle detection in DAG
        
        invocation implies direction
        '''
        graph = defaultdict(list)
        for u,v in invocations:
            graph[u].append(v)
        
        suspicious = [False]*n
        seen = set()

        #dfs from k and mark them
        def dfs1(node,seen,possible):
            suspicious[node] = True
            seen.add(node)
            for neigh in graph[node]:
                if neigh not in seen:
                    dfs1(neigh,seen,suspicious)
        
        dfs1(k,seen,suspicious)
        
        # If any non-suspicious method invokes a suspicious one,the all nodes are not suspicious
        for u, v in invocations:
            if not suspicious[u] and suspicious[v]:
                print(u,v)
                #then the whole thing is good
                return list(range(n))

        # otherwise, we can't tell, so just take all the non suspicious ones, or remove the suspicious ones
        return [i for i in range(n) if not suspicious[i]]
    
##############################################################
# 3345. Smallest Divisible Digit Product I
# 06AUG26
################################################################
class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        '''
        '''
        def get_prod(num):
            prod = 1
            while num:
                prod = prod*(num % 10)
                num = num // 10

            return prod
        
        while get_prod(n) % t != 0:
            n += 1
        
        return n
    
####################################################################
# 2996. Smallest Missing Integer Greater Than Sequential Prefix Sum
# 10AUG26
####################################################################
class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        '''
        find the longest sequenctial pref sum. it must start at zero
        '''
        curr_sum = nums[0]
        n = len(nums)
        for i in range(1,n):
            if nums[i] - nums[i-1] == 1:
                curr_sum += nums[i]
            else:
                break
            
        #we need the smallest x missing fomr nums such x is >= sum of the longest sequential prefix
        nums = set(nums)
        print(curr_sum)
        while curr_sum in nums:
            curr_sum += 1
        return curr_sum