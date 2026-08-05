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