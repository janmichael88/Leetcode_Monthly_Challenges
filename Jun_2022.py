#################################
# 357. Count Numbers with Unique Digits
# 01JUN22
##################################
#good solution write up with other backtracking solutions
#https://leetcode.com/problems/count-numbers-with-unique-digits/discuss/83054/Backtracking-solution

class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        '''
        backtracking approachn involves trying create a digit in the range 0 to 10**n
        we want to append a new digit only if we haven't added that digit yet
        
        we need to call the function for each value of n than add them up
        since the the answer of n, should be the sum for all for n = 1 to n
        '''
        
        ans = 1
        self.MAX = 10**n
        
        used = [False]*10 #digits 0 to 9
        
        def rec(prev):
            count = 0
            if prev < self.MAX:
                count += 1
            else:
                return count
            
            for i in range(10):
                if not used[i]:
                    used[i] = True
                    curr = 10*prev + i
                    count += rec(curr)
                    used[i] = False
            
            return count
        
        for i in range(1,10):
            used[i] = True
            ans += rec(i)
            used[i] = False
        
        return ans


#################################
# 643. Maximum Average Subarray I
# 01JUN22
##################################
#brute force, check all subarray sums
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        '''
        brute force would be to examin all sub array sums of size k
        then just take the max
        '''
        
        ans = float('-inf')
        N = len(nums)
        #edge case
        if N == k:
            return sum(nums)/k
        
        
        for i in range(N-k+1):
            ans = max(ans,sum(nums[i:i+k])/k)
        return ans


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        '''
        the problem is summing acorss a subarray, which takes N*k time in total
        we can use the cumsum array in just get the running sum in constant time
        '''
        cum_sum = [0]
        for num in nums:
            cum_sum.append(num+cum_sum[-1])
            
        ans = float('-inf')
        N = len(nums)
        #edge case
        if N == k:
            return sum(nums)/k
        
        for i in range(k,N+1):
            sub_sum = cum_sum[i] - cum_sum[i-k]
            curr_avg = sub_sum /k
            ans = max(ans,curr_avg)
        
        return ans

#sliding window
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        '''
        we can just mainain a cum sum of size k
        then just add in the new element and remove the i-k eleemtn
        '''
            
        N = len(nums)
        #edge case
        if N == k:
            return sum(nums)/k
        
        curr_sum = 0
        for i in range(k):
            curr_sum += nums[i]
        
        
        res = float('-inf')
        for i in range(k,N):
            curr_sum += nums[i] - nums[i-k]
            res = max(res,curr_sum)
        
        return res / k