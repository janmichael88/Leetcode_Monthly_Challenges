############################################
# 634. Find the Derangement of An Array
# 02NOV23
############################################
#top down fails??
class Solution:
    def findDerangement(self, n: int) -> int:
        '''
        no element should appear in its original posistion
        array is sorted in ascending order from numbers [1 to n]
        input is too big for brute force
        n = 1 = [1]
            0 ways
        n = 2 = [1,2]
            1 way
        n = 3 = [1,2,3]
            2 ways, 
            
        if i knew the number of dearrangements for some n, and i want to add in a new number n+1,
        i cant place i cann't place it at it index, but i can put at at all other spots (n-1) times somthing ....
        '''
        memo = {}
        mod = 10**9 + 7
        
        def dp(n):
            if n == 1:
                return 0
            if n <= 0:
                return 1
            if n in memo:
                return memo[n]
            ans = (n-1)*(dp(n-1) + dp(n-2))
            ans %= mod
            memo[n] = ans
            return ans
        
        return dp(n)
    
class Solution:
    def findDerangement(self, n: int) -> int:
        '''
        no element should appear in its original posistion
        array is sorted in ascending order from numbers [1 to n]
        input is too big for brute force
        n = 1 = [1]
            0 ways
        n = 2 = [1,2]
            1 way
        n = 3 = [1,2,3]
            2 ways, 
            
        if i knew the number of dearrangements for some n, and i want to add in a new number n+1,
        i cant place i cann't place it at it index, but i can put at at all other spots (n-1) times somthing ....
        dp(n) = (n-1)*dp(n-1) + dp(n-2)*(n-1)
            = (n-1)*(dp(n-1) + dp(n-2))
            
        we either place i with the number we are swapping:
            (n-1)*dp(n-2)
        or we don't place it
            (n-1)*dp(n-1)
        '''
        mod = 10**9 + 7
        if n == 1:
            return 0
        if n == 0:
            return 1
        
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 0
        
        for i in range(2,n+1):
            a = (i-1)*dp[i-1] % mod
            b = (i-1)*dp[i-2] % mod
            dp[i] = (a % mod) + (b % mod) % mod
        
        return dp[n] % mod
            