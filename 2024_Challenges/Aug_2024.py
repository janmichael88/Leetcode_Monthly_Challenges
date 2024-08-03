##################################
# 2678. Number of Senior Citizens
# 01AUG24
###################################
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        '''
        first ten chars are phone numbers
        next char is geneder
        next to chars are age
        10+1 
        ''' 
        age_idx = 10 + 1
        ans = 0
        for d in details:
            if int(d[age_idx:age_idx + 2]) > 60:
                ans += 1
        
        return ans
    
#doing char by char insteaf of converting the whole thing
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        '''
        first ten chars are phone numbers
        next char is geneder
        next to chars are age
        10+1 
        ''' 
        age_idx = 10 + 1
        ans = 0
        for d in details:
            tens = ord(d[age_idx]) - ord('0')
            ones = ord(d[age_idx + 1]) - ord('0')
            
            age = tens*10 + ones
            if age > 60:
                ans += 1
        
        return ans
    
###################################################
# 2134. Minimum Swaps to Group All 1's Together II
# 02AUG24
##################################################
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        '''
        there could be multiple arrangments for which a ciruclar array is divided accordigly, we want the minimum
        if there are k's, then each subarray of length k should all 1
        we want the the subarray wit the smallest number of zeros
        we dont want to keep computing the number of zeros, so we can use sliding window
        '''
        ones = sum([num == 1 for num in nums])
        if ones == 0:
            return 0
        nums_doubled = nums + nums
        ans = float('inf')
        left = 0
        count_zeros = 0
        
        for right in range(len(nums_doubled)):
            count_zeros += nums_doubled[right] == 0
            if right - left + 1 == ones:
                ans = min(ans,count_zeros)
                count_zeros -= nums_doubled[left] == 0
                left += 1
            
        return ans
    
#we dont need to concat, jsut use mod N
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        '''
        no need to concat, just use mod N
        '''
        ones = sum([num == 1 for num in nums])
        if ones == 0:
            return 0
        ans = float('inf')
        left = 0
        count_zeros = 0
        N = len(nums)
        
        for right in range(2*N):
            count_zeros += nums[right % N] == 0
            if right - left + 1 == ones:
                ans = min(ans,count_zeros)
                count_zeros -= nums[left % N] == 0
                left += 1

        return ans