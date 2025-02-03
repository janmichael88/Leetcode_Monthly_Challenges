###################################
# 3151. Special Array I
# 01FEB25
##################################
class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        '''
        just check left and right
        '''
        n = len(nums)
        for i in range(n):
            if i == 0:
                if i + 1 < n and nums[i] % 2 == nums[i+1] % 2:
                    return False
            elif i == n-1:
                if i - 1 >= 0 and nums[i-1] % 2 == nums[i] % 2:
                    return False
            else:
                if (nums[i-1] % 2 == nums[i] % 2) or (nums[i] % 2 == nums[i+1] % 2):
                    return False
        return True
    
class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        '''
        only need to check i to i+1 pairs
        i to i-1 pair is the same
        can also use bitwize with XOR
        '''
        n = len(nums)
        for i in range(n-1):
            if (nums[i] & 1) ^ (nums[i+1] & 1) == 0:
                return False
        return True
    
###############################################
# 1852. Distinct Numbers in Each Subarray
# 01FEB25
################################################
class Solution:
    def distinctNumbers(self, nums: List[int], k: int) -> List[int]:
        '''
        sliding window of counts
        '''
        left = 0
        ans = []
        window = Counter()
        
        for right,num in enumerate(nums):
            window[num] += 1
            #shrink first if we have to
            if right - left + 1 > k:
                window[nums[left]] -= 1
                if window[nums[left]] == 0:
                    del window[nums[left]]
                left += 1
            #valid window
            if right - left + 1 == k:
                ans.append(len(window))
        
        return ans
    
####################################################
# 2940. Find Building Where Alice and Bob Can Meet
# 02FEB25
###################################################
class Solution:
    def leftmostBuildingQueries(self, heights: List[int], queries: List[List[int]]) -> List[int]:
        '''
        is person is staning on index i, they can move to index j if i < j and heights[i] < heights[j]
        for each query find leftmost index where they can jump too
        for an index i, we need to find the left most index j, where heights[j] > heights[i]
        for each query (l,r) we can take max(heights[l],heights[r]), then find the left most index that is just greater
        left most index is the same as closes to the right
        
        imagine the queries were single indices
        we could traverse the heights in reverse (monostack) and for each query find the largest height in the stack
        we maintain a stack of indices in decreasing order of heights
            i.e build stack, and binary search in the stack
        for the current building any shorter or equal buildings in the stack, cannot be the anser, so pop them
        if the stack is not empty, the top if our answer, if empty its -1 -> similar to next greater eleemnt ii
        
        now for pairs of indices, the task is to find the first height to ther ight
        intutino, we look for the height that is just greater than the max of the height indices
        while processing the a queyr, the stack already contains all element greater than the current hegith
        idea is that the stack already contains all the elements > than the current height
        '''
        mono_stack = []
        result = [-1 for _ in range(len(queries))]
        new_queries = [[] for _ in range(len(heights))]
        #each index stores the list of queries that require this index as the maximum index of the query pair
        #each query is stored as a pair contain the required height (heighs[a]) and the query index
        for i in range(len(queries)):
            a = queries[i][0]
            b = queries[i][1]
            if a > b:
                a, b = b, a
            #b is bigger than a, so the query so far b is just heights[b]
            #if they are equal, set as b, since a <= b
            if heights[b] > heights[a] or a == b:
                result[i] = b
            else:
                #otherwise we need to look for all heights from index a, for this query
                new_queries[b].append((heights[a], i))

        for i in range(len(heights) - 1, -1, -1):
            mono_stack_size = len(mono_stack)
            for a, b in new_queries[i]:
                #we are looking for the first building with a height > than the query's required height
                position = self.search(a, mono_stack)
                if position < mono_stack_size and position >= 0:
                    #we're looking for the index, not the height
                    result[b] = mono_stack[position][1]
            #monostack to keep track of building heights and their indices in decreasing order of height
            while mono_stack and mono_stack[-1][0] <= heights[i]:
                mono_stack.pop()
            mono_stack.append((heights[i], i))
            print(mono_stack)
        return result

    def search(self, height, mono_stack):
        left = 0
        right = len(mono_stack) - 1
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if mono_stack[mid][0] > height:
                ans = max(ans, mid)
                left = mid + 1
            else:
                right = mid - 1
        return ans