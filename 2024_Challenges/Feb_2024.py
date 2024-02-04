######################################################
# 2966. Divide Array Into Arrays With Max Difference
# 01FEB24
######################################################
#inteligently build the array
class Solution:
    def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
        '''
        need to divide array into one or more arrays of size 3,
        need to satisfy:
            1. each element of nums should be in exactly one array (basically enforcing multiplicty)
            2. difference between any to elements in on array is <= k
        well first of all it needs to be a multiple of 3
        sort array and group by 3
        '''
        N = len(nums)
        if N % 3 != 0:
            return []
        
        nums.sort()
        ans = []
        for i in range(0,N,3):
            candidate = nums[i:i+3]
            #all should be less than k
            is_valid = True
            for i in range(3):
                for j in range(3):
                    if i != j and candidate[i] - candidate[j] > k:
                        is_valid = False
                        return []
            if is_valid:
                ans.append(candidate)
        
        return ans
    
class Solution:
    def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
        '''
        we can just sort and check groups of three, for each group of three, we check last with the first,
        since they are sorted, this would be the largest difference between and two elements and any pair in between would be smaller
        examine an array like:
        [a, a+1, a+2, a+3, a+4, a+5], where k = 2
        we can partition like: [[a, a+1, a+2], [a+3, a+4, a+5]
        if we were to partition like: [a+2, a+3, a+4], the other part would have to be [a, a+1, a+5]
        and this lart part would not be valid
        intuition is to keep values close together
        the greedy proof is actually tricky, since we sort, if we can't amke a group with this current parttion of three, its impossible to make a group after that,
        since the numbers are sorted
        '''
        N = len(nums)
        if N % 3 != 0:
            return []
        
        nums.sort()
        ans = []
        for i in range(0,N,3):
            candidate = nums[i:i+3]
            if candidate[-1] - candidate[0] > k:
                return []
            ans.append(candidate)
        
        return ans
    
##########################################
# 1291. Sequential Digits (REVISTED)
#  02FEB24
##########################################
#ez peez recursion
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        '''
        for it to be sequential it must be a substring of '123456789'
        build number recursively and check if in bounds
        '''
        ans = []
        digits = [1,2,3,4,5,6,7,8,9]
        N = len(digits)
        
        def rec(i,num,low,high):
            if low <= num <= high:
                ans.append(num)
            if i >= N:
                return
            if num > high:
                return            
            rec(i+1, num*10 + digits[i],low,high)
        
        
        for i in range(N):
            rec(i,0,low,high)
        return sorted(ans)
    
###########################################
# 1043. Partition Array for Maximum Sum
# 03FEB24
###########################################
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        '''
        for sum index i, i can make a subarray using i to i+k
        try all subarrays? and find max
        maintain dp array dp(i,j) is max valu starting with i and up to j
        then just grab max from the columns
        no we can just look to the right
        for some index i, we need to find the max value of a subarray it would be a part of
        
        states is just some index i
        for this index i treat as max and increment ans by arr[i]*(length of subarray) + some rec function

        treat as knapsack
        for some index i, pick as the max element, then increase the contribution by the maximum element times the size of the subarray
        then add this to the next dp(j+1) state, and maximize along the way
        or we choose to end the current subarray
        '''
        N = len(arr)
        memo = {}
        
        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            curr_max = 0
            ans = 0
            end = min(N,i+k)
            for j in range(i,end):
                curr_max = max(curr_max, arr[j])
                ans = max(ans, curr_max*(j - i + 1) + dp(j+1))
            
            memo[i] = ans
            return ans
        
        return dp(0)

#bottom up
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        '''
        '''
        N = len(arr)
        dp = [0]*(N+1)
        
        for i in range(N-1,-1,-1):
            curr_max = 0
            end = min(N,i+k)
            for j in range(i,end):
                curr_max = max(curr_max, arr[j])
                dp[i] = max(dp[i], curr_max*(j-i+1) + dp[j+1])
            
        return dp[0]
    
#note: wouldn't it work if we wanted to find the maximum i for each arr[i] for each i
    

#####################################################
# 2800. Shortest String That Contains Three Strings
# 02FEB24
####################################################
    