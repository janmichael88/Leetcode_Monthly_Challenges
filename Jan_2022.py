#########################
# 312. Burst Balloons
# 01JAN22
#########################
#naive recursion
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        '''
        we are given n ballones indexed 0 to n-1, each represents a number
        i i burst the ith ballong, i get nums[i-1]*num[i]*nums[i+1] coins
        if i-1 or i+1 goes out of bounds, then treat it as if there is a ballong with 1 painted on it
        rather
        if i - 1 < 0 or i+1 >= N
        coins += nums[i]
        this is just dp
        
        we can define dp as
        dp(nums):
            max_coins = 0
            for i in ragne(1,len(nums)-2):
                gain = nums[i-1]*nums[i]*nums[i+1]
                remain = dp(nums iwthout nums[i])
                max_coins = max(max_couns,gain+remaining)
            return max_couns
        we need to be careful of the cases where i is at the beignning or end
        in this case just fix to both ends of the array (simlar to sentinel nodes in linked lists)
        the following below is naive recursion O(N*2^N)
        '''
        memo = {}
        
        def dp(nums):
            if tuple(nums) in memo:
                return memo[tuple(nums)]
            
            if len(nums) == 0:
                return 0
            max_coins = 0
            for i in range(1,len(nums)-1):
                gain = nums[i-1]*nums[i]*nums[i+1]
                remain = dp(nums[:i]+nums[i+1:])
                max_coins = max(max_coins,gain+remain)
            memo[tuple(nums)] = max_coins
            return max_coins
        
        nums = [1] + nums + [1]
        return dp(nums)

#recursion
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        '''
        we can adopt the substring (i,j approach) for dp with this probalme
        from the first approach, we need to decrease the number of states 
        as well as decrease the time spent on each state
        typically we strive for reduction in O(2^N) to O(N^2)
        we can use left and right sub pointers to reduce input sizes
        if we pick a middle balloong to pop, then we just need the answers from the left and right sub arrays
        so far we have:
        
        function dp(left, right) {
            // base case is ignored
            max_coins = 0
            for i in 1 ... nums.length - 2:
                // burst nums[i]
                gain = nums[i - 1] * nums[i] * nums[i + 1]
                // burst remaining
                remaining = dp(left, i - 1) + dp(i + 1, right)
                max_coins = max(result, gain + remaining)
            return max_coins
        
        be careful of when to include start and end points for reduced array sized
        but the problem lies in 
        remaining = dp(left, i - 1) + dp(i + 1, right)
        example, if the size of the left subarray has been reduced to 2, and right is >=3
        the coins gained from popping the last ballong in the left subarray depends on whetther or not we 
        popping the left ballong in the right array!
        in other words dp(left,i-1) and dp(i+1,right) are not indepdenent
        so we are stuck and our divide and conquer fails heres BUT if we keep nums[i] alive all the time
        then nums[i-2]*nums[i-1]*nums[i] always refer to the correct balloons and the left part and right
        are independent
        if we just mark nums[i] as the last burst ballong among [left,right]
        
        special properties for edge cases
            if we have N length array of all a;s
            then we always have a*a*a, except last tows
            last two would yeild a*a*1 and another 1*a*1
            
        Therefore, we have N-2 a * a * a, one a * a * 1, and one 1 * a * 1. Adding together, we have (N - 2) * a * a * a + a * a + a.
        this is actually a variant of matrix chain multiplciation
        '''
        #special case
        if len(nums) > 1 and len(set(nums)) == 1:
            return (nums[0] ** 3) * (len(nums) - 2) + nums[0] ** 2 + nums[0]
        
        # handle edge case
        nums = [1] + nums + [1]
        
        memo = {}
        
        def dp(left,right):
            if right - left < 0:
                return 0
            if (left,right) in memo:
                return memo[(left,right)]
            res = 0
            #find last burst ith ballong between left and right
            for i in range(left,right+1):
                #nums[i] is the last burst ballong
                gain = nums[left-1]*nums[i]*nums[right+1]
                #divide
                rem = dp(left,i -1) + dp(i+1,right)
                res = max(res, rem + gain)
            memo[(left,right)] = res
            return res
        
        return dp(1,len(nums)-2)

#dp
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        '''
        we can translate the top down to bottom up
        notes:
            while iterating we need to be careful on the order of iteration
            such that dp[left][i-1] and dp[i+1][right] are iterated before dp[left][right], duh
            this is imporant because in order to calculate dp[left][right] we nee the results of
            dp[left][i-1] and dp[i+1][right] for i in betten [left,right]
            if we added fake balloons to the start end ends of the array, we only need the top right
            (i.e we don't need the whole dp table)
            the order we want to go in is down to up and left to right
        algo:
            1. handle special cases
            2. add dummy balloons
            3. init dp table where dp[left][right] represents max coins obtains if we buirts all ballonons on itnerval [left,right]
            4. iterate over dp array such trhat dp[left][i-1] and dp[i+1][right] are calcualted before dp[left][right]
            5. we iterate for all i in between left and right
            6. first:   dp[left][i - 1] + dp[i + 1][right]
            7. second: nums[left - 1] * nums[i] * nums[right + 1]
            8. return dp[1][len(nums)-2], 
            9. not dp[0][len(nums)-1] since we added the ones ourselves
        '''
        if len(nums) > 1 and len(set(nums)) == 1:
            return (nums[0] ** 3) * (len(nums) - 2) + nums[0] ** 2 + nums[0]

        # handle edge case
        nums = [1] + nums + [1]
        n = len(nums)
        # dp[i][j] represents
        # maximum if we burst all nums[left]...nums[right], inclusive
        dp = [[0] * n for _ in range(n)]

        # do not include the first one and the last one
        # since they are both fake balloons added by ourselves and we can not
        # burst them
        for left in range(n - 2, 0, -1):
            for right in range(left, n - 1):
                # find the last burst one in nums[left]...nums[right]
                for i in range(left, right + 1):
                    # nums[i] is the last burst one
                    gain = nums[left - 1] * nums[i] * nums[right + 1]
                    # recursively call left side and right side
                    remaining = dp[left][i - 1] + dp[i + 1][right]
                    # update
                    dp[left][right] = max(remaining + gain, dp[left][right])
        # burst nums[1]...nums[n-2], excluding the first one and the last one
        return dp[1][n - 2]