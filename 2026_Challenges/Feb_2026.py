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
        
        return anslee