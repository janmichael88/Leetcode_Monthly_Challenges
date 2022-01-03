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

#######################################################
# Pairs of Songs With Total Durations Divisible by 60
# 02DEC21
#######################################################
class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        '''
        return the number of songs, such that their sum is divisble by 60
        (a+b) % 60 == 0
        (a%60) + (b%60) == 0
        from the hint, we only need to consider each song length mod 60
        count the number of songs with length % 60 and store that in array of size 60
        dont forget to check complemetbs
        we are loooking for two songs a and b such that
        a & 60 == 0, b % 60 or (a % 60) + (b % 60) == 60 
        
        '''
        mapp = collections.defaultdict(int)
        pairs = 0
        for t in time:
            #get mod 60 if this song
            mod_60 = t % 60
            #if its mod 60 already, find songs where mod 60 is 0
            if mod_60 == 0:
                pairs += mapp[mod_60]
            #if its not check for compelemnt
            else:
                complement = 60 - mod_60
                pairs += mapp[complement]
            #now put mod 60 into mapp
            mapp[mod_60] += 1
            
            
        return pairs

###############################
# 568. Maximum Vacation Days
# 02DEC21
###############################
class Solution:
    def maxVacationDays(self, flights: List[List[int]], days: List[List[int]]) -> int:
        '''
        my job is to schedule the traveling to maximize the number of vacation days i can take following the rules
        return max vacation days i can take during k weeks
        rules:
        1. i can travel to n cities, indexced 0 to n - 1, initallty on city 0
        2. cities connected by flights there exists a flight i,j if flights[i][j] == 1
        3. we have k weeks, 7 days, we can only take one flight on one days, and we can only take a flight on each week's monday morning
        4. for each city, i am restricted in days, given matrisx days which is n by k, i can only stay in city i for days[i][k] for the k'th week
        5. i could stay in a city beyong the number of vacations days, but you should work on the extra days,
        which will not be counted as vacation days
        6. if i fly from city A to cyt B and take the vacation on that day, the deduction is towards the vacation days of city B in that week
        7. do not consider the impact of flight hours on vacation days
        return max vacation days i can take during k weeks
        
        for each week, i need to play and work such that it's sum is seven days
        
        inution:
            for every function call, we traverse over all the cities and find out  all cities connected to current one
            for the current day we can either travel to a new city of stay (0/1 knapsack)
            if we decide to switch lets call it j
            after chaing the city we need to find the number of vacationa which we can take from the new city as current city
            days[j][weekno] + dfs(j,weekno + 1)
        '''
        memo = {}
        n = len(flights)
        k = len(days[0])
        def dp(curr_city,week_num):
            #got to last weekno, no vacation
            if week_num == k:
                return 0
            if (curr_city,week_num) in memo:
                return memo[(curr_city,week_num)]
            max_days = 0
            for i in range(n):
                #can take flight or stay at same cit
                if flights[curr_city][i] == 1 or i == curr_city:
                    #get new days
                    vac = days[i][week_num] + dp(i,week_num + 1)
                    max_days = max(max_days,vac)
            memo[(curr_city,week_num)] = max_days
            return max_days
        
        return dp(0,0)

class Solution:
    def maxVacationDays(self, flights: List[List[int]], days: List[List[int]]) -> int:
        '''
        now just translate to bottom up
        the maximum number of vacationss that cant be taken given that we start from the ith city in the kth week is not dependent on the vacations that can be taken in earlier weeks
        it only depends in the number of vacations that can be taken in the upcoming weeks and also on the connetions in flights
        dp[i][k] represent the maximum number of vacations which can be taken from the ith city in the kth week
        which would give us our anser
        to dill each dp[i][k]:
            1. start from the ith city in the kth week and staty in the same city for the k+1 week
            so dp[i][k] = days[i][k] + dp[i,k+1]
            
            2. we start from the ith city in the kth week and move the the jth city in the k+1 week
            only if flights[i][j] is 1
        
        in order to maximize the number of vacastions that can be taken from the ith city in the kth week,
        we need to choose the desitinatino city that leads to a max number of vacatinos
        so the factor to be considered here is
        maxdays[j][k] + days[j][k+1] for all i,j,k satsifying flights[i][j] = 0
            
        '''
        n = len(flights)
        k = len(days[0])
        
        dp = [[0]*(k+1) for _ in range(n)]
        
        #start backwards in days to get the first entry
        for week in range(k-1,-1,-1):
            #check all curr cities
            for curr_city in range(n):
                #first update by stating in city
                dp[curr_city][week] = days[curr_city][week] + dp[curr_city][week+1]
                #now check for flights
                for dest_city in range(n):
                    if flights[curr_city][dest_city] == 1:
                        take = days[dest_city][week] + dp[dest_city][week+1]
                        no_take = dp[curr_city][week]
                        dp[curr_city][week] = max(take,no_take)
        
        return dp[0][0]