#############################################
# 3010. Divide an Array Into Subarrays With Minimum Cost I
# 01FEB26
#############################################
class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        '''
        cost of subarray is value of first element 
        need min possible sum of these subarray
        if we partition nums into 3 subarrays, there are 2 break points
        we can ge the first elements as
            (index0,i+1,j+1)
        enumerate all break proints (i,j) and take minimum sum
        '''
        ans = float('inf')
        n = len(nums)
        for i in range(1,n-1):
            for j in range(i+1,n):
                ans = min(ans, nums[0] + nums[i] + nums[j])
        
        return ans
    

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        '''
        single pass with variable swapping
        nums[0] is fixed
        we need to find to indices (i,j), where i < j and nums[i],nums[j] are the smallest
        '''
        min1 = float('inf')
        min2 = float('inf')
        n = len(nums)
        for i in range(1,n):
            if nums[i] < min1:
                min2 = min1
                min1 = nums[i]
            
            elif nums[i] < min2:
                min2 = nums[i]
        
        return nums[0] + min1 + min2
    
############################################################
# 3013. Divide an Array Into Subarrays With Minimum Cost II
# 02FEB26
############################################################
#TLE, yes!
class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        '''
        we need k subarrays where the stating index of the second subarray and the starting index of the kth subarray is <= dist
        first subarray will alwasy have cost of nums[0]
        if we have subarray with index 0,i1,i2...ik
        we need ik - i1 to be <= dist
        '''
        ans = float('inf')
        n = len(nums)

        for i in range(1, n):
            left = i + 1
            right = min(i + dist, n - 1)

            # Need at least k-2 elements to choose from
            if right - left + 1 < k - 2:
                continue

            vals = nums[left:right + 1]
            vals.sort()

            cost = nums[0] + nums[i] + sum(vals[:k - 2])
            ans = min(ans, cost)

        return ans
    
from sortedcontainers import SortedList
from typing import List

class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        '''
        instead of sorting on the fly, treat like sliding window with two balance structures
        we need to maintain them, similar to sliding window median and sliding windoo k smallest sum
        '''
        n = len(nums)
        ans = float('inf')

        small = SortedList()   # smallest k-2 elements
        large = SortedList()   # rest
        small_sum = 0

        def add(x):
            nonlocal small_sum
            #if we can take in the smaller
            if len(small) < k - 2:
                small.add(x)
                small_sum += x
            else:
                #if its bigger than smaller, we need to rebalance
                if x < small[-1]:
                    large.add(small.pop())
                    small.add(x)
                    small_sum += x - large[0] #update the sum
                else:
                    #add to the rest
                    large.add(x)

        def remove(x):
            nonlocal small_sum
            if x in small:
                small.remove(x)
                small_sum -= x
                if large:
                    y = large.pop(0)
                    small.add(y)
                    small_sum += y
            else:
                large.remove(x)

        # Initial window for i = 1
        for j in range(2, min(n, 1 + dist + 1)):
            add(nums[j])

        for i in range(1, n):
            if len(small) == k - 2:
                ans = min(ans, nums[0] + nums[i] + small_sum)

            # Slide window
            if i + 1 < n:
                remove(nums[i + 1])
            if i + dist + 1 < n:
                add(nums[i + dist + 1])

        return ans

from sortedcontainers import SortedList
from typing import List
class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        '''
        we need to keep track of the sum of the smallest k-2 elements at anyone time
        we can use two sorted containers and maintain sums in the range nums[i+1:i+dist+1]
        as we increase i, lose one element on left and gain one element on right
        invariant:
            len(smaller) == k - 2, we must rebalance here
        '''
        n = len(nums)
        ans = float('inf')

        smaller = SortedList([]) #smallest k-2
        larger = SortedList([])
        smallest_sum = 0

        def add(x):
            nonlocal smallest_sum
            #if we can add
            if len(smaller) < k - 2:
                smaller.add(x)
                smallest_sum += x
            else:
                if x < smaller[-1]: #must put into smaller, extrude from larger
                    larger.add(smaller.pop())
                    smaller.add(x)
                    smallest_sum += x - larger[0] #change in sum, need to update on the fly
                else:
                    larger.add(x)
        
        def remove(x):
            nonlocal smallest_sum
            if x in smaller:
                smaller.remove(x)
                smallest_sum -= x
                if larger:
                    temp = larger.pop()
                    smaller.add(temp)
                    smallest_sum += temp
            else:
                larger.remove(x)
            
        #initial window for i = 1
        for j in range(2,min(n, 1 + dist + 1)):
            add(nums[j])
        
        for i in range(1,n):
            #check we have a sum
            if len(smaller) == k - 2:
                ans = min(ans, nums[0] + nums[i] + smallest_sum)
            
            #sliding window
            if i + 1 < n:
                remove(nums[i+1])
            if i + dist + 1 < n:
                add(nums[i + dist + 1])
        
        return ans

#######################################
# 3637. Trionic Array I
# 02FEB26
########################################
class Solution:
    def isTrionic(self, nums: List[int]) -> bool:
        '''
        so its 1 for inc and 0 for dec, else -1
        if its trionic then it should be 
        [1,1,1,0,0,1,1,1]
        if the array is trionic, the diff array can be reduced to the form [1,0,1]
        '''
        diffs = []
        n = len(nums)
        for i in range(1,n):
            diff = nums[i] - nums[i-1]
            if diff > 0:
                if diffs and diffs[-1] == 1:
                    continue
                else:
                    diffs.append(1)
            elif diff < 0:
                if diffs and diffs[-1] == 0:
                    continue
                else:
                    diffs.append(0)
            else: #if equal we can immediately escape
                return False
        
        return diffs == [1,0,1]
    
###################################################
# 3640. Trionic Array II
# 04DEC26
###################################################
#hints gave it away :(
#finite state machine
class Solution:
    def maxSumTrionic(self, nums: List[int]) -> int:
        '''
        is my solution from before, say for example we have this up/down signature
        1 for inc and 0 for dec, else -1
        if its trionic then it should be 
        [1,1,1,0,0,1,1,1]
        we need the max sum in this range left to right
        but it could be any subarray, we can solve that using kadanes
        hmmmm, just follow the hints and the transitions
        '''
        n = len(nums)
        NEG = float('-inf')

        dp0 = [NEG] * n
        dp1 = [NEG] * n
        dp2 = [NEG] * n
        dp3 = [NEG] * n

        dp0[0] = nums[0]

        for i in range(1, n):
            diff = nums[i] - nums[i - 1]
            # Hint 6: dp0 carries on increasing
            if diff > 0:
                dp0[i] = max(dp0[i - 1] + nums[i], nums[i])
            else:
                dp0[i] = nums[i]

            # Hint 4: inc phase
            if diff > 0:
                dp1[i] = max(dp1[i - 1] + nums[i],dp0[i - 1] + nums[i])
                dp3[i] = max(dp3[i - 1] + nums[i],dp2[i - 1] + nums[i])

            # Hint 5: dec phase
            if diff < 0:
                dp2[i] = max(dp2[i - 1] + nums[i],dp1[i - 1] + nums[i])

        return max(dp3)
    
######################################################
# 3379. Transformed Array
# 05FEB26
#####################################################
class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        '''
        allocate array, then calc using mod length
        '''
        n = len(nums)
        ans = [-1]*n

        for i in range(n):
            num = nums[i]
            if num > 0:
                j = (i + num) % n
                ans[i] = nums[j]
            elif num < 0:
                j = (i - abs(num)) % n
                ans[i] = nums[j]
            else:
                ans[i] = num
        
        return ans
    
class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        '''
        can just do i + nums[i]
        but that could be negative
        so add in n
        (i + num[i] + n) % n
        '''
        n = len(nums)
        ans = [-1]*n

        for i in range(n):
            num = nums[i]
            j = (i + nums[i] + n) % n
            ans[i] = nums[j]
        
        return ans
    
####################################################
# 3634. Minimum Removals to Balance Array
# 06FEB26
#####################################################
class Solution:
    def minRemoval(self, nums: List[int], k: int) -> int:
        '''
        an array is balanced if its max element is at most k times min
        i.e max(nums) <= k*min(nums)
        what if i sort the array
        [1,2,6,9]
        and use two pointers, problem is i dont know whether to remove the min or the max
        '''
        #uhhh, i dont know why it works lolol
        ans = float('inf')
        n = len(nums)
        nums.sort()
        right = 0
        for left in range(n):
            while right < n and nums[right] <= k*nums[left]:
                right += 1
            ans = min(ans, n - (right - left))
        
        return ans
            