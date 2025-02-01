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