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