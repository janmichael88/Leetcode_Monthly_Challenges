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
# 02JAN22
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
# 02JAN22
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

###########################
# 997. Find the Town Judge
# 03JAN22
###########################
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        '''
        we have n people 1 to n
        if the town judge exists then:
            1. town judge tructs nobody
            2. everybody (except town judge) trust the judge
            3. there is eacatly one person that statfies 1 and 2
        
        return label of judge if it exits, else -1
        for this to be true, all nodes except judge must point
        make the adj list
        then scan from 1 to n if if its not there
        if its check that everyone trusts it
        '''
        
        adj_list = defaultdict(int)
        in_degree = defaultdict(int)
        
        for a,b in trust:
            adj_list[a] = b
            #incrment in degree
            in_degree[b] += 1
        
        #pass from 1 to n
        for i in range(1,n+1):
            if i not in adj_list:
                if in_degree[i] == n - 1:
                    return i
        
        return -1

class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        '''
        another way is to just check the in and out degrees
        also for there to be a judge, there must be n-1 edges
        otherwise we cannot do it
        '''
        if len(trust) < n -1:
            return - 1
        
        indegree = [0]*(n+1)
        outdegree = [0]*(n+1)
        
        for a,b in trust:
            outdegree[a] += 1
            indegree[b] += 1
        
        for i in range(1,n+1):
            if indegree[i] == n - 1 and outdegree[i] == 0:
                return i
            
        return -1

class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        '''
        we can use trust scrcore for each person
        for an out going edge, -1, incoming edge +1
        the max indegree is N-1
        the town judge must be the person with score N-1 exactly
        '''
        if len(trust) < n -1:
            return - 1
        
        trust_scores = [0]*(n+1)
        
        for a,b in trust:
            trust_scores[a] -= 1
            trust_scores[b] += 1
        
        for i in range(1,n+1):
            if trust_scores[i] == n - 1:
                return i
        
        return -1

#########################
# 401. Binary Watch
# 03JAN22
#########################
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        #cheeky way
        watch = [1,2,4,8,1,2,4,8,16,32]
        times = []
        for leds in itertools.combinations(range(len(watch)), turnedOn):
            h = sum(watch[i] for i in leds if i < 4)
            m = sum(watch[i] for i in leds if i >= 4)
            if h > 11 or m > 59: continue
            times.append("{}:{:02d}".format(h, m))
        return times

class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        '''
        for a given turned on number i can grab the indices that belong to it
        the watch ash 4 digits for hours [1,,2,4,8]
        the watche has [1,2,4,8,16,32]
        ive i'm given a number say 4 lighted turned on
        i want a combination if 4 indices taken from the total
        '''
        watch = [1,2,4,8,1,2,4,8,16,32]
        N = len(watch)
        ans = []
        for lights in itertools.combinations(range(N),turnedOn):
            #now check hours and in mins
            hours = 0
            mins = 0
            for i in lights:
                if i < 4:
                    hours += watch[i]
                if i >= 4:
                    mins += watch[i]
            
            if hours > 11 or mins > 59:
                continue
            ans.append("{}:{:02d}".format(hours,mins))
        
        return ans

#dfs solution
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        '''
        we can use recursion to find out the number of solutions
        for each call keep track of n,hours,mins,and current index to take
        then for each call 
            decrement n, but add to hours anad/or mins if we can
            since the numbers are in binary we can use bit shifts to icnrement
            also we can prune when hours > 11 or mins > 59
        '''
        results = []
        
        def dfs(n,hours,mins,idx):
            #prune
            if hours >= 12 or mins > 59:
                return
            if not n:
                temp = str(hours)+":"+"0"*(mins < 10) + str(mins)
                results.append(temp)
                return
            #check all
            for i in range(idx,10):
                #first 4 positions are for hours
                if i < 4:
                    dfs(n-1,hours | (1 << i),mins,i+1)
                #the rest are for minuts
                else:
                    k = i - 4
                    dfs(n-1,hours,mins | (1 << k),i+1)
        
        dfs(turnedOn,0,0,0)
        return results

#not using bit manip
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        '''
        we can use recursion to find out the number of solutions
        for each call keep track of n,hours,mins,and current index to take
        then for each call 
            decrement n, but add to hours anad/or mins if we can
            since the numbers are in binary we can use bit shifts to icnrement
            also we can prune when hours > 11 or mins > 59
        '''
        def dfs(LEDS, idx, hrs, mins, n):
            # base cases
            if hrs >= 12 or mins >= 60:
                return
            if n == 0:
                time = str(hrs) + ":" + "0"*(mins<10) + str(mins)
                result.append(time)
                return

            if idx < len(LEDS):
                if idx <= 3:  # handle hours
                    dfs(LEDS, idx+1, hrs + LEDS[idx], mins, n-1)
                else:  # handle minutes
                    dfs(LEDS, idx+1, hrs, mins + LEDS[idx], n-1)
                # next hour / min state
                dfs(LEDS, idx+1, hrs, mins, n)
        result = []
        LEDS = [
            8, 4, 2, 1,  # top row of watch
            32, 16, 8, 4, 2, 1 # bottom row of watch
        ]
        dfs(LEDS, 0, 0, 0, turnedOn)
        
        return result

#######################################
# 1009. Complement of Base 10 Integer
# 04JAN22
######################################
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        '''
        just flip bit by bit
        '''
        if n == 0:
            return 1
        temp = n
        pos = 1
        while temp:
            n = n ^ pos
            temp = temp >> 1
            pos = pos << 1
        
        return n

class Solution:
    def bitwiseComplement(self, num: int) -> int:
        '''
        count bits and use ones mask
        '''
        if num == 0:
            return 1
        temp = num
        bits = 0
        while temp:
            bits += 1
            temp = temp >> 1
        #get ones mask
        ones = (1 << bits) - 1
        return num ^ ones

################################
# 131. Palindrome Partitioning
# 05JAN22
################################
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''
        the input is small enough for an exponenital solution
        generation all partitions and check each paritions is a palindrome
        for adding to a built path, check if its a palindrome, then add it
        '''
        N = len(s)
        results = []
        
        def isPal(left,right):
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        def dfs(i,path):
            #if i got to the end of the string, its a valid path
            if i == N:
                results.append(path[:])
                return
            
            for j in range(i,N):
                if isPal(i,j):
                    path.append(s[i:j+1])
                    dfs(j+1,path)
                    path.pop()
                    
        dfs(0,[])
        return(results)

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''
        the input is small enough for an exponenital solution
        generation all partitions and check each paritions is a palindrome
        for adding to a built path, check if its a palindrome, then add it
        
        we can also determine if a string a palindrom in 0(1) time by using overlapping subproblems
        '''
        N = len(s)
        results = []
        dp = [[False]*N for _ in range(N)]
        
        
        def dfs(i,path):
            #if i got to the end of the string, its a valid path
            if i == N:
                results.append(path[:])
                return
            
            for j in range(i,N):
                if s[i] == s[j]:
                    if j - i <= 2 or dp[i+1][j-1] == True:
                        dp[i][j] = True
                        path.append(s[i:j+1])
                        dfs(j+1,path)
                        path.pop()
                    
        dfs(0,[])
        return(results)

##########################
# 1094. Car Pooling
# 06JAN22
##########################
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        '''
        the car can only go east, i.e to the right
        so we cannot go back, this eliminates backtracking
        we could evluate all paths going forward, but that would be exponenital in lenght lists
        for each start, the capcity goes up of numb passengers
        for each end, the capacity goes down
        i can proccess these in order of drop off and record the the state of my car, after sorting of course
        note similarity to meeting rooms II problem
        '''
        times = []
        for people,start,end in trips:
            times.append((start,people))
            times.append((end,-people))
        
        #sort
        times.sort()
        
        #now go in order and pick up people and check\
        car = 0
        for curr,people in times:
            car += people
            if car > capacity:
                return False
        return True

class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        '''
        there is also a bucket sort solution
        notice that there max distance is no more than 1000
        initlize 1001 buckets and put number passenger change in each bucket
        '''
        times = [0]*1001
        for people, start, end in trips:
            times[start] += people
            times[end] -= people
        
        car = 0
        for peep in times:
            car += peep
            if car > capacity:
                return False
        return True

'''
this is cool solution using cum sums

Let us start with example and understand what exactly we are asked to do:
trips = [[3,2,7],[3,7,9],[8,3,9]], capacity = 11. Let us represent it in the following way:

# # 3 3 3 3 3 # # #
# # # # # # # 3 3 3
# # # 8 8 8 8 8 8 8

Now, the question is we sum all these lists, where we deal # as 0 (0 0 3 11 11 11 11 11 11 11) will some number be more than capacity = 11 or not. Let us instead each list construct another list, such that its cumulative sum is our list:

0 0 3 0 0 0 0 -3 0 0 0
0 0 0 0 0 0 0 3 0 0 -3
0 0 0 8 0 0 0 0 0 0 -8

Then if we sum these lists and evaluate cumulative sums, we will have exactly what is needed:

0 0 3 8 0 0 0 0 0 0 -11 -> cumulative sums -> 0 0 3 11 11 11 11 11 11 11 0
'''
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        m = max([i for _,_,i in trips])
        times = [0]*(m+2)
        for i,j,k in trips:
            times[j+1] += i
            times[k+1] -= i
        
        car = 0
        for change in times:
            car += change
            if car > capacity:
                return False
        return True
        
###############################
# 382. Linked List Random Node
# 07JAN22
###############################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def __init__(self, head: Optional[ListNode]):
        self.nums = []
        temp = head
        while temp:
            self.nums.append(temp.val)
            temp = temp.next

    def getRandom(self) -> int:
        #another way, geneteat randome number from 1 to 0
        #pick = int(random.random() * len(self.nums))
        #use pick as index
        return random.choice(self.nums)


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def __init__(self, head: Optional[ListNode]):
        '''
        good reivist of a problem
        we can use reservoir sampling to sample over a populatin of unknown size with constnace space
        the algo is intended to sample k elements from a population of unknonw size, in this case k
        reservior sampling is a family, but the most popular one is known as algoirthm R
        in pseudo code:
        
        def algoR(S,R):
            #S has items to sample from of unknonw size K
            #creat R
            for i in range(1,K):
                R[i] = S[i]
            #replace elements with gradually increasing prob
            for i in range(k+1,n):
                j = random(1,i)
                if j <= k:
                    R[j] = S[i]
        
        the important thing to remember is that at ane one time, each element has equal probability of being chosen into the R array
        '''
        self.head = head
        

    def getRandom(self) -> int:
        scope = 1
        chosen = 0
        curr = self.head
        while curr:
            #decide whther to include the element in reservoir
            if random.random() < 1/scope:
                chosen = curr.val
            #move on to next and update
            curr = curr.next
            scope += 1
        
        return chosen


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()

##########################
# 1463. Cherry Pickup II
# 08JAN22
##########################
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        '''
        two robots, one starting at (0,0) and the other (0,cols-1)
        find max cherries we can take using both robots
            * from (i,j) we can go (i+1,j), (i+1,j-1), (i+1,j+1)
            * can only go down really
            * passing through i,j we go up grid[i][j]
            * if both go through same cell, only one takes cherries
            * both need to his the bottom
            
        in the first problem we only had one robot, now have 2
        if we move robot1, then try to figure out robot 2, thats too many subproblems
        if we move boths robots together, we get (r1,r2,c1,c2)
        which represents the rows and cols for each robot, but we know we have to move them together
        and that they only go down
        so r1 == r2
        then we have (r,c1,c2), since its dp, we try all and find the max
        dp(i,j,k) = max(dp(new_i,new_j,new_k) for all possible (new_i,new_j,new_k) + dp(i,j,k)
        there are 9 possible subprblems from and i,j,k
            ROBOT1 | ROBOT2
        ------------------------
         LEFT DOWN |  LEFT DOWN
         LEFT DOWN |       DOWN
         LEFT DOWN | RIGHT DOWN
              DOWN |  LEFT DOWN
              DOWN |       DOWN
              DOWN | RIGHT DOWN
        RIGHT DOWN |  LEFT DOWN
        RIGHT DOWN |       DOWN  
        RIGHT DOWN | RIGHT DOWN 
        
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        memo = {}
        
        def dp(i,j,k):
            #if out of bounds, no cherries to be taken
            if i < 0 or i >= rows or j < 0 or j >= cols or k < 0 or k >=cols:
                return 0
            if (i,j,k) in memo:
                return memo[(i,j,k)]
            #current cell
            res = 0
            res += grid[i][j]
            #if robots are not on the same scell
            if j != k:
                res += grid[i][k]
            #solve subproblems if i haven't hit the last row
            if i != rows -1:
                max_of_all_subproblems = 0
                for dj in [-1,0,1]:
                    for dk in [-1,0,1]:
                        max_of_all_subproblems = max(max_of_all_subproblems,dp(i+1,j+dj,k+dk))
                #add to res
                res += max_of_all_subproblems
            
            memo[(i,j,k)] = res
            return res
        
        return dp(0,0,cols -1)

class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        '''
        also bottom up
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dp = [[[0]*cols]*cols]*rows
        
        #starting from the end
        for i in reversed(range(rows)):
            for j in range(cols):
                for k in range(cols):
                    #current cell
                    res = 0
                    res += grid[i][j]
                    #if robots are not on the same scell
                    if j != k:
                        res += grid[i][k]
                    #solve subproblems if i haven't hit the last row
                    if i != rows -1:
                        max_of_all_subproblems = 0
                        for dj in [-1,0,1]:
                            for dk in [-1,0,1]:
                                #must be in bounds
                                if (0 <= i + 1 < rows) and (0 <= j + dj < cols) and (0 <= k + dk < cols):
                                    max_of_all_subproblems = max(max_of_all_subproblems,dp[i+1][j +dj][k+dk])
                        res += max_of_all_subproblems
                    dp[i][j][k] = res
                        
        return dp[0][0][cols-1]

################################
# 1041. Robot Bounded In Circle
# 09JAN22
###############################
class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        '''
        there has to be a pattern, in order to return true, the robot has to come back
        the problem with simulating is that it would take a long time before coming back to center
        from the hint
            if the final vector isn't point north it has to come back
            so how do a keep track of the vector state
            the final vector consits of change in direction plus change in position
        
        this is a good example of limit cycle trajectory
        inital proog:
            after at most 4 cycles, the limit cycle trajectory return to the inital points, it is bounded
            ideally if we ran through in intrusction set 4 times and ended up at 0,0, the robot is bounded
        '''
        #four pass check
        start = [0,0]
        #must go in clockwise direction for modular arithmetic to work
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        #initally we are facing north
        idx = 0
        for _ in range(4):
            for step in instructions:
                if step == 'L':
                    idx = (idx+3) % 4
                elif step == 'R':
                    idx = (idx+1) % 4
                else:
                    start[0] += dirs[idx][0]
                    start[1] += dirs[idx][1]
        return start == [0,0]

class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        '''
        we dont have to check 4 times, just check if at start or not facing north
        '''
        #four pass check
        start = [0,0]
        #must go in clockwise direction for modular arithmetic to work
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        #initally we are facing north
        idx = 0
        for step in instructions:
            if step == 'L':
                idx = (idx+3) % 4
            elif step == 'R':
                idx = (idx+1) % 4
            else:
                start[0] += dirs[idx][0]
                start[1] += dirs[idx][1]
        return start == [0,0] or idx != 0

##################################
# 4. Median of Two Sorted Arrays
# 09JAN22
##################################
#log(M+N)

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        '''
        we can solve this in log(m+n) using kth smallest in two sorted arrays
        a few notes:
            * when the total length is pdd, the median is in the middle
            * when the total length is even, the median is the average of the middle 2
        for getkth recursive:
            if sum of two medians indices is smaller than k
                if nums1 median value bigger than nums2, then nums2's first half will always be positioned before nums1's median, so k would never be in num2's first half
            if sum of two median's indicies is bigger than k
                if nums1 median value bigger than nums2, then nums2's first half would be merged before nums1's first half, thus k always come before nums1's median, then nums1's second half would never include k


        '''
        len1 = len(nums1)
        len2 = len(nums2)
        # when total length is odd, the median is the middle
        if (len1 + len2) % 2 != 0:
            return self.get_kth(nums1, nums2, 0, len1-1, 0, len2-1, (len1+len2)//2)
        else:
        # when total length is even, the median is the average of the middle 2
            middle1 = self.get_kth(nums1, nums2, 0, len1-1, 0, len2-1, (len1+len2)//2)
            middle2 = self.get_kth(nums1, nums2, 0, len1-1, 0, len2-1, (len1+len2)//2-1)
            return (middle1 + middle2) / 2
        
    def get_kth(self, nums1, nums2, start1, end1, start2, end2, k):
        if start1 > end1:
            return nums2[k-start1]
        if start2 > end2:
            return nums1[k-start2]

        middle1 = (start1 + end1) // 2
        middle2 = (start2 + end2) // 2
        middle1_value = nums1[middle1]
        middle2_value = nums2[middle2]

        # if sum of two median's indicies is smaller than k
        # i dont have at least k elements
        if (middle1 + middle2) < k:
                # if nums1 median value bigger than nums2, then nums2's first half will always be positioned before nums1's median, so k would never be in num2's first half
            if middle1_value > middle2_value:
                #move up the smaller
                return self.get_kth(nums1, nums2, start1, end1, middle2+1, end2, k)
            else:
                return self.get_kth(nums1, nums2, middle1+1, end1, start2, end2, k)
                # if sum of two median's indicies is bigger than k
        else:
            # if nums1 median value bigger than nums2, then nums2's first half would be merged before nums1's first half, thus k always come before nums1's median, then nums1's second half would never include k
            if middle1_value > middle2_value:
                return self.get_kth(nums1, nums2, start1, middle1-1, start2, end2, k)
            else:
                return self.get_kth(nums1, nums2, start1, end1, start2, middle2-1, k)


#from tushar roy
#min log(m,n)
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        m, n = len(nums1), len(nums2)
        
        left_size = (m + n + 1) // 2
        start = 0
        end = m
        is_even = ((m + n) % 2) == 0
        while start <= end:
            a_part = (start + end) // 2
            b_part = left_size - a_part
            
            aleftmax = float("-inf") if a_part == 0 else nums1[a_part - 1]
            arightmin = float("inf") if a_part == m else nums1[a_part]
            bleftmax = float("-inf") if b_part == 0 else nums2[b_part - 1]
            brightmin = float("inf") if b_part == n else nums2[b_part]
            
            if aleftmax <= brightmin and bleftmax <= arightmin:
                if not is_even:
                    return max(aleftmax, bleftmax)
                else:
                    return (max(aleftmax, bleftmax) + min(arightmin, brightmin))/ 2
            elif aleftmax > brightmin:
                end = a_part - 1
            elif bleftmax > arightmin:
                start = a_part + 1
            
###################
# 67. Add Binary
# 10JAN22
###################
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        '''
        start off by reversing the two strings
        create string and
        while loop for both twoo pointers
        add and carry
        finish remaing
        returned reverse string as answwer
        '''
        a = a[::-1]
        b = a[::-1]
        
        res = ""
        carry = 0
        i = j = 0
        
        while i < len(a) and j < len(b):
            #get initial value
            val = (int(a[i]) + int(b[i])) % 2
            #include carry
            val += carry
            #update carry
            carry = (int(a[i]) + int(b[i])) // 2
            #add to res
            res += str(val)
            i += 1
            j += 1
        
        #remainders
        while i < len(a):
            val = (int(a[i])) % 2
            val += carry
            carry = (int(a[i]) + carry) //2
            
            res += str(val)
            i += 1
            
        while j < len(b):
            val = (int(b[j])) % 2
            val += carry
            carry = (int(b[j]) + carry) //2
            res += str(val)
            j += 1
        
        if carry:
            res += '1'
        
        return res[::-1]

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        '''
        we can use the fill string operation in python
        '''
        N = max(len(a),len(b))
        #this fills left
        a = a.zfill(N)
        b = b.zfill(N)
        
        carry = 0
        ans = ""
        
        #now start backwards
        for i in range(N-1,-1,-1):
            #we will add to carry and use that as our answer
            if a[i] == '1':
                carry += 1
            if b[i] == '1':
                carry += 1
            #check
            if carry % 2 == 1:
                ans += '1'
            else:
                ans += '0'
            
            carry = carry // 2
        
        if carry:
            ans += '1'
        
        return "".join(ans[::-1])

class Solution:
    def addBinary(self, a: str, b: str) -> str:
        '''
        we can xor if we can't use addition
        turns out a ^ b is answer without carry
        and to find current carry we just shift this number to the left once
        now the problem is reduced:
            find the sum of answer without carry and carry
        its the same problem - to sum two numbers
        and once could solve it in a loop while carry it not == o
        algo:
            convert a an b in to ints - x will be used to keep ans and y for carry
            while carry i non zero:
                current answer w/o carry ix ans = x^y
                current carry is left shift - carry = (x & y) << 1
                swap
        '''
        x = int(a,2)
        y = int(b,2)
        
        while y:
            #get answer without carry
            answer = x ^ y
            #find the current carry
            carry = (x & y) << 1
            x,y = answer,carry
        
        return bin(x)[2:]

#cool way of controlling loop invariant
class Solution:
    def addBinary(self, a, b):
        i, j, summ, carry = len(a) - 1, len(b) - 1, [], 0
        while i >= 0 or j >= 0 or carry:
            d1 = int(a[i]) if i >= 0 else 0
            d2 = int(b[j]) if j >= 0 else 0
            summ += [str((d1 + d2 + carry) % 2)]
            carry = (d1 + d2 + carry) // 2
            i, j = i-1, j-1 
        return "".join(summ[::-1])

#########################################
# 1022. Sum of Root To Leaf Binary Numbers
# 11JAN22
#########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        '''
        this is really just building all root to leaf paths and adding
        i can use a global variable and add to this when i hit a leaf node
        i can use the shift operator to move and add
        then use | to add
        '''
        self.res = 0
        
        def dfs(node,curr):
            if not node:
                return
            #if there is leaf, add
            curr = (curr << 1) | node.val
            if not node.left and not node.right:
                self.res += curr

            dfs(node.left,curr)
            dfs(node.right,curr)

        dfs(root,0)
        return self.res

class Solution:
    def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
        '''
        this is really just building all root to leaf paths and adding
        i can use a global variable and add to this when i hit a leaf node
        i can use the shift operator to move and add
        then use | to add
        '''
        res = 0
        
        stack = [(root,0)]
        
        while stack:
            node,curr = stack.pop()
            if not node:
                continue
            curr = (curr << 1) | node.val

            if not node.left and not node.right:
                res += curr
            
            stack.append((node.left,curr))
            stack.append((node.right,curr))


        return res

###############################
# 422. Valid Word Square
# 11JAN22
###############################
class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        '''
        for it to be a valid word square all i,j elements must be the same as j,i
        
        '''
        for i in range(len(words)):
            #matching word
            matched_word = words[i]
            #check of this word is bigger than the square
            if len(matched_word) > len(words):
                return False
            #now check chars
            for j in range(len(matched_word)):
                #if the word to be checked is smaller than current word it is not valid
                #i is the current row index, and if the j'th word to be checked has smaller length than the current row it must be an invalid square
                if len(words[j]) <= i or words[j][i] != matched_word[j]:
                    return False
        return True

'''
zip(*words) is a commonly used Python trick to transpose a matrix. zip_longest is used in place of zip for stirngs of different lengths, e.g. words = ["abcd","bnrt","crm","dt"].

>>> list(zip(*words))
[('a', 'b', 'c', 'd'), ('b', 'n', 'r', 't')]
>>> list(zip_longest(*words))
[('a', 'b', 'c', 'd'), ('b', 'n', 'r', 't'), ('c', 'r', 'm', None), ('d', 't', None, None)]

'''
class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        return words == ["".join(x) for x in zip_longest(*words, fillvalue="")]

class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        '''
        using zip and zip longest
        
        '''
        zipped_words = []
        for x in zip_longest(*words,fillvalue=""):
            zipped_words.append("".join(x))
        
        return words == zipped_words

class Solution:
    def validWordSquare(self, words: List[str]) -> bool:
        '''
        using zip and zip longest
        
        '''
        for i in range(len(words)):
            build_word = ""
            for j in range(len(words)):
                if i < len(words[j]):
                    build_word += words[j][i]
            
            if words[i] != build_word:
                return False
        
        return True

####################################
# 588. Design In-Memory File System
# 11JAN22
####################################
#almost!
class Dir:
    def __init__(self):
        self.dirs = defaultdict(Dir)
        self.files = defaultdict(str)

class FileSystem:
    '''
    we can use a next dicts for each
    each node contains two hashmaps files and dirs
    files contains names of the file and its contents
    dirs is another name with subdirectory as key and a structure as value
    '''

    def __init__(self):
        self.root = Dir()
        

    def ls(self, path: str) -> List[str]:
        #give reference to root
        t = self.root
        #hold files
        files = []
        #not empty back slash
        if path != "/":
            d = path.split("/")
            #don't start at begnning and end
            for i in range(1,len(d)-1):
                #move the root
                if d[i] in t.dirs:
                    t = t.dirs[d[i]]
            if d[len(d)-1] in t.files:
                files.append(d[len(d)-1])
                return files
            else:
                if d[len(d)-1] in t.dirs:
                    t = t.dirs[d(len(d))-1]
        #add the rest
        for foo in t.dirs.keys():
            files.append(foo)
        for foo in t.files.keys():
            files.append(foo)
            
        return sorted(files)

    def mkdir(self, path: str) -> None:
        t = self.root
        d = path.split("/")
        for i in range(1,len(d)):
            if d[i] in t.dirs:
                #make new
                t.dirs[d[i]] = Dir()
            #always
            if d[i] in t.dirs:
                t = t.dirs[d[i]]
        

    def addContentToFile(self, filePath: str, content: str) -> None:
        t = self.root
        d = filePath.split("/")
        for i in range(1,len(d)-1):
            if d[i] in t.dirs:
                t = t.dirs[d[i]]
        #add
        if d[len(d)-1] in t.files:
            temp = t.files[d[len(d)-1]]
            t.files[d[len(d)-1]] = temp+content
        else:
            t.files[d[len(d)-1]] = content

    def readContentFromFile(self, filePath: str) -> str:
        t = self.root
        d = filePath.split("/")
        for i in range(1,len(d)-1):
            if d[i] in t.dirs:
                t = t.dirs[d[i]]
        
        if d[len(d)-1] in t.files:
            return t.files[d(len(d))-1]
        else:
            return ""
        

class TrieNode:
    def __init__(self):
        self.content = ""
        self.children = defaultdict(TrieNode)
        self.isfile = False

class FileSystem:

    def __init__(self):
        self.root = TrieNode()
        
    def ls(self, path: str) -> List[str]:
        node = self.root
        path_list = path.split("/")
        for p in path_list:
            if not p:
                continue
            node = node.children.get(p)
        if node.isfile:
            return [p]
        ans = [i for i in node.children.keys()]
        if not ans:
            return ans
        ans.sort()
        return ans

    def mkdir(self, path: str) -> None:
        path_list = path.split("/")
        node = self.root
        for p in path_list:
            if not p:
                continue
            node = node.children[p]
        

    def addContentToFile(self, filePath: str, content: str) -> None:
        path_list = filePath.split("/")
        node = self.root
        for p in path_list:
            if not p:
                continue
            node = node.children[p]
        node.content += content
        node.isfile = True

    def readContentFromFile(self, filePath: str) -> str:
        path_list = filePath.split("/")
        node = self.root
        for p in path_list:
            if not p:
                continue
            node = node.children.get(p)
        return node.content
        


# Your FileSystem object will be instantiated and called as such:
# obj = FileSystem()
# param_1 = obj.ls(path)
# obj.mkdir(path)
# obj.addContentToFile(filePath,content)
# param_4 = obj.readContentFromFile(filePath)

##########################################
# 701. Insert into a Binary Search Tree
# 12JAN22
######################################
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        '''
        just use recursion
        if val is greater than node go right
        else go left
        '''
        def dfs(node,val):
            if not node:
                return TreeNode(val)
            if val > node.val:
                node.right = dfs(node.right,val)
            else:
                node.left = dfs(node.left,val)
            return node
        
        return dfs(root,val)

class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        '''
        another way is to just use two pointers and keep connection
        '''
        if not root:
            return TreeNode(val)
        
        prev = None
        curr = root
        
        while curr:
            prev = curr
            if curr.val > val:
                curr = curr.left
            else:
                curr = curr.right
                
        #now im at the point with prev because curr is none
        if prev.val > val:
            prev.left = TreeNode(val)
        else:
            prev.right = TreeNode(val)
        
        return root

#################################################
# 452. Minimum Number of Arrows to Burst Balloons
# 13JAN22
#################################################
#sorting on start
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        '''
        this is kinda like an interval problem
        if the intervals intersect somewhere, than using one arrow at the intersection pops all the balloons
        if i sort on their starting x's, keep merging until i can't
        once i can't i use up an arrow
        '''
        N = len(points)
        if N == 0:
            return 0
        #sort on start
        points.sort(key = lambda x: x[0])
        #keep extending otherwise use up an arrow
        arrows = 1
        end = points[0][1]
        for curr_start,curr_end in points:
            if curr_start <= end:
                #update end
                end = min(end,curr_end)
            else:
                end = curr_end
                arrows += 1
        
        return arrows

#sorting on end, not proof of greedy algos is hard
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        '''
        another way is to sort on their end index
        for a balloon there are two possibilites:
            to have a start coord smaller than the curr ending balloon, means thats these balloons could have been popped togetehr
            to have a start coor larger than the curr ending, means we woul need an extra arrow
        
        this means that could always track the end of the curr ballon and ignore all balloons which end before it
        once the current ballon has ended, we increase the number of arraows by one
        '''
        N = len(points)
        if N == 0:
            return 0
        #sort on start
        points.sort(key = lambda x: x[1])
        arrows = 1
        starting_end = points[0][1]
        for start,end in points:
            if start > starting_end:
                arrows += 1
                starting_end = end
        
        return arrows


######################################
# 166. Fraction to Recurring Decimal
# 13JAN22
######################################
class Solution:
    def fractionToDecimal(self, num: int, denom: int) -> str:
        '''
        a stupid way would be to divide num by num than find the repeating unit
        you need to try a couple of test cases with long division
        the key insight here is the once the remainder starts repeating, so does the divied resutl
        algo:
            need to keep a hashtable that maps from the remainder to is ppositoin in the fractional part
            once that has been found, you can enclose the recurring fractinoal part with ()
        '''
        #first check if it goes evenly
        if num % denom == 0:
            return str(num//denom)
        
        #otherwise we gotta get the integral part and decimal part
        sign = '' if num*denom >=0 else '-'
        #easier to start with positive values
        num = abs(num)
        denom = abs(denom)
        
        #init rest with sign and intergral part
        res = sign+str(num//denom)+"."
        
        #start off with remainder
        num = num % denom
        i = 0
        part = ''
        
        m = {num:i} #the remainder and position
        
        while num % denom != 0: #while there is a reaminder
            #add zeros digit
            num *= 10
            i += 1
            rem = num % denom
            part += str(num // denom)
            #if we have seen this remainder, build it part and return
            if rem in m:
                #we've repeated up this part so first find the non repreating part
                non_repeating = part[:m[rem]]
                repeating = part[m[rem]:]
                return res + non_repeating + '('+repeating+')'
            #other mark as new
            m[rem] = i
            num = rem
        
        #must bs non repeating in remainder
        return res + part

############################
# 13JAN22
# 156. Binary Tree Upside Down
###########################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        if i'm at a node,
        the left child if there is one becomes the new root
        the root becomes right
        the original right becomes left
        it is guranteed that every right node has a sibling a left node with the same parent
        and has no children
        
        #another way of reframing
        1. the root's right node becomes the left node of the left node of root
            root.left.left = root.right
        2. root becomes the right node of root's left node:
            root.leftright = root
        3. above rules apply on the left edge and return left node along path
        note this is bottom up
        '''
        def dfs(root):
            if not root or (not root.left and not root.right):
                return root
            
            left = dfs(root.left)
            root.left.left = root.right
            root.left.right = root
            root.left = None
            root.right = None
            return left
        
        return dfs(root)

class Solution:
    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we can use top down, but since a tree is DAG we need to pass information along
        specifically we need the parent from where we came from and the right sibling
        '''
        self.ans = None
        
        def dfs(root,parent,right):
            if not root:
                return 
            left_child = root.left
            right_child = root.right
            root.left = right
            root.right = parent

            if left_child:
                dfs(left_child,root,right_child)
            else:
                self.ans = root
        
        dfs(root,None,None)
        return self.ans

#iterative
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def upsideDownBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we can also do this iteratively, but we need to kepe track of the old referemces
        '''
        old_left_leaf = None
        old_parent = None
        old_right_sibling = None
        
        while root:
            #change the current level
            left_child = root.left
            right_child = root.right
            root.left = old_right_sibling
            root.right = old_parent
            
            #check if we need to keep going
            if not left_child:
                old_left_leaf = root
                break
            #otherwise go down a level
            old_parent = root
            root = left_child
            old_right_sibling = right_child
        
        return old_left_leaf
        

#helpful https://leetcode.com/problems/binary-tree-upside-down/discuss/502244/Graph-Explanation-with-3-Python-Solutions

############################
# 8. String to Integer (atoi)
# 14JAN22
############################
class Solution:
    def myAtoi(self, s: str) -> int:
        '''
        keep advanving left pointer if its not an ord
        then keep decrementing right pointer if is not order
        * discard all leading whitespace
        * get sign of number
        * overflow
        * invalid input
        '''
        INT_MIN = -(2**31)
        INT_MAX = 2**31 - 1
        
        s = s.strip()
        
        N = len(s)
        sign = 1
        num = 0
        i = 0
        
        if N == 0:
            return 0
        
        
        #sign correction
        if s[0] == '-':
            sign *= -1
            i += 1
        elif s[0] == '+':
            i += 1
        
        #go through string
        while i < N and '0' <= s[i] <= '9':
            curr_digit = ord(s[i]) - ord('0')
            #abandon if we know its oveflown
            if num > INT_MAX // 10 or (num == INT_MAX // 10 and curr_digit >7):
                return INT_MAX if sign == 1 else INT_MIN
            num = num*10 + curr_digit
            i += 1
        
        return num*sign
            
#offical solution
#abbreviating 0-7 and 0-8 check
class Solution:
    def myAtoi(self, s: str) -> int:
        '''
        just follow the rules, make sure we strip ending and leading whitespaces
        make sure we get rid of any leading zeros
        if we had access to a longs vars, we could just simply check for overflow and underflow
        but it's better to assume
        the last section really about checking for underflow and overflow
        
        overflow:
            max is 2**31 - 1 = 2147483647
            case 1: curr number is less than int_max / 10(214748364) we can add in any number
            case 2: curr number is greater than int_max / 10(214748364) adding any number will go over
            case 3: if curr number is equal to int_max / 10 (214748364) we are limited to the digits between 0 and 7
        
        underflow:
            the min value is -2**31 = -2147483648
            case 1: if the curr numer is greater than int_min / 10 (-214748364) then we can add any new digits
            case 2: of the curr number is less than int_mina / 10 (-214748364), appending any digits will make it less
            case 3: if curr numer is == tp int_min / 10, we are limited to anydigits between 0 and 8
            
        now notice:
        Notice that cases 1 and 2 are similar for overflow and underflow. The only difference is case 3: for overflow, we can append any digit between 0 and 7, but for underflow, we can append any digit between 0 and 8.
        
        we can combine both cases and if examine only the asbolute values
        
        iniitally store the sign
        if curr number is less than int_max / 10, we can append next digitd
        if curr number is greater than int_max / 10:
            and sign is +, reutnr int_max
            and sign is negative, return int_min
            
        uf curr number is euqal to int_max / 10:
            we can add any digit in between 0 and 7
                if next digit is 8, return int max
                if sign is negtaigve, wetrun int mina
            if greater than 8
                if + return int max
                if _ return int min
                
        Note: We do not need to handle 0-7 for positive and 0-8 for negative integers separately. If the sign is negative and the current number is 214748364, then appending the digit 8, which is more than 7, will also lead to the same result, i.e., INT_MIN.
        '''
        sign = 1
        res = 0
        index = 0
        N = len(s)
        
        INT_MAX = 2**31 - 1
        INT_MIN = -(2**31)
        
        #discards all white spaces from beignning
        while index < N and s[index] == ' ':
            index += 1
        
        #get sign
        if index < N and s[index] == '+':
            sign = 1
            index += 1
        elif index < N and s[index] == '-':
            sign = -1
            index += 1
            
        #travere next digits of input and stop if it is not a digit
        #end of string is also non digit character
        while index < N and s[index].isdigit():
            #get digit
            digit = int(s[index])
            
            #check over and underflow
            if (res > INT_MAX // 10) or (res == INT_MAX // 10 and digit > INT_MAX % 10):
                return INT_MAX if sign == 1 else INT_MIN
            
            res *= 10
            res += digit
            index += 1
        
        return sign*res

#using a state machine
class StateMachine:
    def __init__(self):
        self.State = { "q0": 1, "q1": 2, "q2": 3, "qd": 4 }
        self.INT_MAX, self.INT_MIN = pow(2, 31) - 1, -pow(2, 31)
        
        # Store current state value.
        self.__current_state = self.State["q0"]
        # Store result formed and its sign.
        self.__result = 0
        self.__sign = 1

    def to_state_q1(self, ch: chr) -> None:
        """Transition to state q1."""
        self.__sign = -1 if (ch == '-') else 1
        self.__current_state = self.State["q1"]
    
    def to_state_q2(self, digit: int) -> None:
        """Transition to state q2."""
        self.__current_state = self.State["q2"]
        self.append_digit(digit)
    
    def to_state_qd(self) -> None:
        """Transition to dead state qd."""
        self.__current_state = self.State["qd"]
    
    def append_digit(self, digit: int) -> None:
        """Append digit to result, if out of range return clamped value."""
        if ((self.__result > self.INT_MAX // 10) or 
            (self.__result == self.INT_MAX // 10 and digit > self.INT_MAX % 10)):
            if self.__sign == 1:
                # If sign is 1, clamp result to INT_MAX.
                self.__result = self.INT_MAX
            else:
                # If sign is -1, clamp result to INT_MIN.
                self.__result = self.INT_MIN
                self.__sign = 1
            
            # When the 32-bit int range is exceeded, a dead state is reached.
            self.to_state_qd()
        else:
            # Append current digit to the result. 
            self.__result = (self.__result * 10) + digit

    def transition(self, ch: chr) -> None:
        """Change state based on current input character."""
        if self.__current_state == self.State["q0"]:
            # Beginning state of the string (or some whitespaces are skipped).
            if ch == ' ':
                # Current character is a whitespaces.
                # We stay in same state. 
                return
            elif ch == '-' or ch == '+':
                # Current character is a sign.
                self.to_state_q1(ch)
            elif ch.isdigit():
                # Current character is a digit.
                self.to_state_q2(int(ch))
            else:
                # Current character is not a space/sign/digit.
                # Reached a dead state.
                self.to_state_qd()
        
        elif self.__current_state == self.State["q1"] or self.__current_state == self.State["q2"]:
            # Previous character was a sign or digit.
            if ch.isdigit():
                # Current character is a digit.
                self.to_state_q2(int(ch))
            else:
                # Current character is not a digit.
                # Reached a dead state.
                self.to_state_qd()
    
    def get_integer(self) -> None:
        """Return the final result formed with it's sign."""
        return self.__sign * self.__result
    
    def get_state(self) -> None:
        """Get current state."""
        return self.__current_state

class Solution:
    def myAtoi(self, input: str) -> int:
        q = StateMachine()
        
        for ch in input:
            q.transition(ch)
            if q.get_state() == q.State["qd"]:
                break

        return q.get_integer()

###########################
# 1345. Jump Game IV
# 15JAN22
###########################
#close one
#don't forget to clear to prevent a reduanct search

class Solution:
    def minJumps(self, arr: List[int]) -> int:
        '''
        from an index i, i can jump to i+1,i-1 or to j where arr[i] == arr[j]
        i can used bfs to find the min number of steps for this one
        but first i need to mapp
        the mapp will mark values to a list of indices
        '''
        neighbors = defaultdict(list)
        for i,num in enumerate(arr):
            neighbors[num].append(i)
            
        #mark beignning index as seen
        seen = set()
        seen.add(0)
        N = len(arr)
        q = deque([(0,0)])
        
        while q:
            curr,steps = q.popleft()
            if curr == N - 1:
                return steps
            #now find valid neighbors,adjacnet first
            neighs = [curr-1,curr+1]
            #now find neighbors with matching values
            for cand in neighbors[arr[curr]]:
                neighs.append(cand)
            #clear the list to prevent
            neighbors[arr[curr]].clear()
            #now check if we can visite
            for n in neighs:
                #in bounds
                if 0 <= n < N:
                    if n not in seen:
                        q.append((n,steps+1))
                        seen.add(n)
        
        return -1

class Solution:
    def minJumps(self, arr: List[int]) -> int:
        '''
        can also do it an additoinal way using next layer and return global step
        '''
        N = len(arr)
        
        if N <= 1:
            return 0
        
        neighbors = defaultdict(list)
        for i,num in enumerate(arr):
            neighbors[num].append(i)
            
        curr = [0]
        seen = {0}
        steps = 0
        
        while curr:
            nex = []
            #check
            for node in curr:
                if node == N-1:
                    return steps
                #check same value
                for child in neighbors[arr[node]]:
                    if child not in seen:
                        seen.add(child)
                        nex.append(child)
                
                #clear to prevent redudant seraches
                neighbors[arr[node]].clear()
                
                #check neighbaord
                for child in [node-1, node+1]:
                    if 0 <= child < len(arr) and child not in seen:
                        seen.add(child)
                        nex.append(child)
            
            curr = nex
            steps += 1
        
###########################
# 253. Meeting Rooms II
# 15JAN22
###########################
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        '''
        this is similar to the bursting ballonws problem
        if i sort in start times
        if the start time of meeting overlaps with the start and end, we need a new room
        i can use a heap and use it to keep track of the earliest ending time
        then we can just add a room meeting into the q and pop if we have to
        we need to use a min heap
        '''
        if not intervals:
            return 0
        
        intervals.sort(key = lambda x: x[0])
        
        rooms = []
        #add the first room ending time
        heappush(rooms,intervals[0][1])
        
        #now check starting with second meeting
        for start,end in intervals[1:]:
            #if the start time i'm on is less than the smallest end time so far, we have a free room
            if rooms[0] <= start:
                heappop(rooms)
            #add in the new rooms end time
            heappush(rooms,end)
        
        return len(rooms)
            
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        '''
        we can also use a two pointer trick after sorting starts end ends invdidually
        we keep advancing our start pointer if we are less than the smallest ending time
        ounce we get to a point where our start time is less than the smallest end, we know that a room has been freed
        '''
        if not intervals:
            return 0
        
        starts = []
        ends = []
        for start,end in intervals:
            starts.append(start)
            ends.append(end)
        
        #sort both
        starts.sort()
        ends.sort()
        
        rooms = 0
        start_pointer = 0
        end_pointer = 0
        
        while start_pointer < len(intervals):
                        # If there is a meeting that has ended by the time the meeting at `start_pointer` starts
            if starts[start_pointer] >= ends[end_pointer]:
                # Free up a room and increment the end_pointer.
                rooms -= 1
                end_pointer += 1

            # We do this irrespective of whether a room frees up or not.
            # If a room got free, then this used_rooms += 1 wouldn't have any effect. used_rooms would
            # remain the same in that case. If no room was free, then this would increase used_rooms
            rooms += 1    
            start_pointer += 1 

        
        return rooms

###########################################
# 849. Maximize Distance to Closest Person
# 16JAN22
###########################################
class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        '''
        we have at least 1 person sitting
        and at least on open seat
        we want to place Alex in an empty seat such that the distance between him and the next seat is maximized
        [1,0,0,0,1,0,1]
        nearest to the left:    [0,1,2,3,0,1,0]
        neartest to the right   [0,3,2,1,0,1,0]
        we want the max distance from both sides at the ith position
        then check in both arrays and update only if left max and right max are greater
        
        For approach 1, the 'right' array is constructed in incorrectly. For cases like [1,0,0,0], the left array is correct and becomes [0,1,2,3]. However, the right array becomes [0,6,5,4]. I'm sure this is an oversight, but in the case where there is no one in the right side, we are free to take the last seat.
        '''
        N = len(seats)
        
        #find lefts arrays
        #in the case where there is only one seat
        left = [N]*N
        for i in range(N):
            if seats[i] == 1:
                left[i] = 0
            elif i > 0:
                left[i] = left[i-1] + 1
            #remember this loop control if i'm the i+1 or i-1 falls out of index
                
        #find the rights array
        right = [N]*N
        for i in range(N-1,-1,-1):
            if seats[i] == 1:
                right[i] = 0
            elif i < N-1:
                right[i] = right[i+1] + 1
                
        ans = 0
        for i in range(N):
            #find the smallest if left and right
            smallest_left_and_right = min(left[i],right[i])
            #find the max
            ans = max(ans,smallest_left_and_right)
        print(left,right)
        return ans

class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        '''
        we can also use group by
        note that in groups of size K adjacent empty seats, the answer must be (K+1) // 2
        i can find this group and find the largest suze
        '''
        N = len(seats)
        K = 0
        ans = 0
        
        #first find longest adjacent 0s
        for i in range(N):
            if seats[i] == 1:
                K = 0
            else:
                K += 1
                ans = max(ans, (K + 1) // 2)
        
        #now check for cases in 1,0,0,0,0
        for i in range(N):
            if seats[i] == 1:
                ans = max(ans,i)
                break
        
        for i in range(N-1,-1,-1):
            if seats[i] == 1:
                ans = max(ans,N-1-i)
                break
        
        return ans

######################################
# 405. Convert a Number to Hexadecimal
# 16JAN22
######################################
class Solution:
    def toHex(self, num: int) -> str:
        '''
        first, if a number is greater than zero, we can convert
        is less than zero first find its complement by shifting it to the largest val
        for a number we can find its compelmeent by inverting bits and adding 1
        
        now that we have its compelement we need to find the hexa
        we can use digits 0123456789abcdef
        then just usse the mod and reduce trick matching indices
        '''
        if num < 0:
            num += 2**32
        
        hex_alpha = "0123456789abcdef"
        res = ""
        
        while num:
            res = hex_alpha[num % 16] + res
            num //= 16
        
        if not res:
            return "0"
        
        return res

############################
# 290. Word Pattern
# 17JAN22
############################
#two maps
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        '''
        we can make mapping of each letter to word
        if we haven't made the bijection, i.e char to word, make it
        if we have made, check that the mapping is right
        then return false
        '''
        N = len(pattern)
        words = s.split(" ")
        
        if N != len(words):
            return False
        
        mapp = {}
        mapp_2 = {}
        
        for char, word in zip(pattern, words):
            if char in mapp:
                if mapp[char] != word:
                    return False
            elif word in mapp_2:
                if mapp_2[word] != char:
                    return False
            else:
                mapp[char] = word
                mapp_2[word] = char
        
        return True

class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        '''
        we can avoid using two hashmaps, if we check that the len of unique chars in pattern does not equal length of the unique set of words
        '''
        N = len(pattern)
        words = s.split(" ")
        
        if N != len(words):
            return False
        if len(set(pattern)) != len(set(words)):
            return False
        
        mapp = {}
        for char, word in zip(pattern, words):
            if char in mapp:
                if mapp[char] != word:
                    return False
            else:
                mapp[char] = word
                
        return True

############################
# 455. Assign Cookies
# 17JAN22
#############################
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        '''
        we are given len(g) children
        and len(s) cookies
        we want to try to give each child at most 1 cookie
        i can assigne s[j] to g[i] if s[j] >= g[i]
        we want to maximuze the content
        greediyl try to satisfy the content using the largest cookies
        sort and use two pointers
        '''
        g.sort()
        s.sort()
        i,j = len(g)-1,len(s)-1
        count = 0
        while (i>=0 and j>=0):
            if g[i] <= s[j]:
                count += 1
                j-=1
            i -= 1
        return count
        
############################
# 605. Can Place Flowers
# 18JAN22
############################
#its the stupid ifs
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        '''
        well it was easier than i thought, just sscan and place
        for each position we just left and right, only if we can
        '''
        N = len(flowerbed)
        i = 0
        count = 0
        
        while i < N:
            if i == 0:
                if flowerbed[i] == 0 and flowerbed[i+1] == 0:
                    flowerbed[i] += 1
                    count += 1
            elif i == N-1:
                if flowerbed[i] == 0 and flowerbed[i-1] == 0:
                    flowerbed[i] += 1
                    count += 1
            elif flowerbed[i] == 0:
                if flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
                    flowerbed[i] += 1
                    count += 1
            i += 1
        
        return count >= n

class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        '''
        well it was easier than i thought, just sscan and place
        for each position we just left and right, only if we can
        '''
        N = len(flowerbed)
        i = 0
        count = 0
        
        while i < N:
            if flowerbed[i] == 0:
                if i == 0 or flowerbed[i-1] == 0:
                    if i == N -1 or flowerbed[i+1] == 0:
                        flowerbed[i] += 1
                        count += 1
            i += 1
        
        return count >= n

#terminate early
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        '''
        well it was easier than i thought, just sscan and place
        for each position we just left and right, only if we can
        '''
        N = len(flowerbed)
        i = 0
        count = 0
        
        while i < N:
            if flowerbed[i] == 0:
                if i == 0 or flowerbed[i-1] == 0:
                    if i == N -1 or flowerbed[i+1] == 0:
                        flowerbed[i] += 1
                        count += 1
            if count >= n:
                return True
            i += 1
        
        return False

#############################
# 482. License Key Formatting
# 18JAN22
#############################
class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        '''
        we have string s, with n dashes and n+1 groups
        we want to re format the string such that the groups have at least k chars
        except the first group
        if the number of chars goes into k evenly, we know that each group can have k chars
        if it does not then it is the first group that must have < k chars
        if num_chars % k != 0, then the first group should have num_chars % k chars
        then the rest evens out
        '''
        #get usubale chars
        s = "".join(s.split("-"))
        #convet lowercases chars
        s = [ch.upper() for ch in s]
        
        groups = []
        if len(s) % k == 0:
            groups.append("".join(s[:k]))
            s = s[k:]
        else:
            groups.append("".join(s[:len(s) % k]))
            s = s[len(s) % k:]
            
        #now take from s in lenghts of 2
        for i in range(0,len(s),k):
            groups.append("".join(s[i:i+k]))
            
        return "-".join(groups)

class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        '''
        just another way, first find the length of the first group
        then just advance in K steps
        '''
        s = "".join(s.split('-')).upper()
        N = len(s)
        #find remaidner
        rem = k if N % k == 0 else N % k
        #set curr pointer to end of rem
        i = rem
        res = s[:rem]
        
        while i < N:
            res += '-' + s[i:i+k]
            i += k
        
        return res

###########################
# 142. Linked List Cycle II
# 19JAN22
############################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i could dfs for each node
        '''
        curr = head
        seen = set()
        
        while curr:
            if curr in seen:
                return curr
            else:
                seen.add(curr)
                curr = curr.next
        
        return None

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        we can use the tortise and the hare trick and use two phases
        phase 1:
            slow and fast pointers until they intersection
            if they don't intersection we know there isn't a cycle so return null
            if we have a linked list with a cycle of length C:
                let us label nodes in cycle from 0 to C - 1
                let us lable node out of cycle from F to -1 (where F is the length not in cycle)
                after F iterations, slow has gone to 0 and fast has down to fast % C iterations
                fast traverses 2*F nodes in F iterations, all of which are in the cycle
                after C-h iterations, the slow points to C-h in the cycle, but also fast should come back
                fast + 2(C-fast) = 2C-fast
                                = (C- fast) % C
                so after n interations, both slow and fast will at some point a in the cycle
        phase 2:
            if we have found a cycle, we must find the entrance,
            we know that fast moves twice as fast as slow, 2d(slow) = d(fast)
            two pointers from head and interseciton and advance until they meet
            2(F + a) = F + nC + a
            Hence the coordinate of the intersection point is F + a = nCF+a=nC
        '''
        if not head:
            return None
        intersect = self.getIntersect(head)
        if not intersect:
            return None
        
        p1 = head
        p2 = intersect
        
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
        return p1
    
    def getIntersect(self,head):
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return slow
        
        return None

#################################
# 209. Minimum Size Subarray Sum
# 19JAN22
################################
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        brute force would be to generate all contig subrray and find min sum at least target
        we want the legnth not the actual sum
        '''
        ans = float('inf')
        N = len(nums)
        for i in range(N):
            for j in range(i,N):
                SUM = sum(nums[i:j+1])
                if SUM >= target:
                    ans = min(ans, j-i+1)
                    break #we can break here since we are looking for the smallest
        
        return ans if ans != float('inf') else 0

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        brute force would be to generate all contig subrray and find min sum at least target
        we want the legnth not the actual sum
        we can reduce to n^2  by storing cumsums and finding sum in 0(1) time
        '''
        ans = float('inf')
        N = len(nums)
        if N == 0:
            return 0
        
        pref_sum = [0]*N
        pref_sum[0] = nums[0]
        
        for i in range(1,N):
            pref_sum[i] = pref_sum[i-1] + nums[i]
        
        for i in range(N):
            for j in range(i,N):
                curr_sum = pref_sum[j] - pref_sum[i] + nums[i]
                if curr_sum >= target:
                    ans = min(ans, j-i+1)
                    break
        
        return ans if ans != float('inf') else 0

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        we can imporve from N^2 by using binary search
        but to make this easier we need to prefix the pref_sum with a zero
        recall for the N^2 approach, we were looking for a subarray staring with i and ending with j
        such that it was greater than s
        '''
        ans = float('inf')
        N = len(nums)
        if N == 0:
            return 0
        
        pref_sum = [0]
        
        for num in nums:
            pref_sum.append(num + pref_sum[-1])
        
        #we can index uding pref_sum for an i change to i+1
        #sum of nums[i:j] == pref_sum[j+1] - pref_sum[i]
        '''
        for i in range(N):
            for j in range(1,N+1):
                substring = nums[i:j]
                print(sum(substring),pref_sum[j] - pref_sum[i])
        '''
        for i in range(N):
            #find lower bound, smallest pref_sum[j] at least target
            to_find = target + pref_sum[i]
            left = i
            right = len(pref_sum) - 1
            while left < right:
                mid = left + (right - left) // 2
                if pref_sum[mid] <= to_find:
                    left = mid + 1
                else:
                    right = mid
            #print(left,right,mid)
            if mid != len(pref_sum) - 1:
                ans = min(ans, (left - i + 1))
        
        return ans if ans != float('inf') else 0

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        two pointerrs sliding window
        we can move the starting index of the current subarray as soon as we know that no better 
        could be done with this index as starting index
        then we just move pointers so that we are never above target until we hit target
        '''
        N = len(nums)
        ans = float('inf')
        left = 0
        curr_sum = 0
        for i in range(N):
            curr_sum += nums[i]
            while curr_sum >= target:
                ans = min(ans, i + 1 - left)
                curr_sum -= nums[left]
                left += 1
        
        return ans if ans != float('inf') else 0

#################################
# 875. Koko Eating Bananas
# 20JAN22
#################################
#brute force
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        '''
        '''
        #start with the smallest k and go uo
        k = 1
        
        while True:
            hours = 0
            for b in piles:
                hours += math.ceil(b/k)
            
            if hours <= h:
                return k
            else:
                k += 1

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        '''
        we want to find the minimum k such that she can eat all the bannans
        we can pick any k, for each hour she chooses some pile of bannans and eats k from that pile
        if pile < k bannase she eats all of them and will not east any more
        binary seach for all k between min(piles) and max(piles)
        if she can eat at this value of k, then she can do for all values less than k
        if a pile < k, eats them all and cannot eat during this hour
        if pile > k, i need to calculate how many hours it will take
        we need to return the left or right bound int this case since we want the minimum workable k
        
        '''
        left = 1
        right = max(piles)
        
        while left < right:
            mid = left + (right - left) // 2
            
            hours = 0
            for p in piles:
                hours += math.ceil(p/mid)
                
            #check if we can do it
            if hours <= h:
                right = mid
            else:
                left = mid + 1
        
        return left

##########################
# 134. Gas Station
# 21JAN22
##########################
#brute force
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        '''
        we have care with unlimited gas tank, and we can increment by gas[i] when we are on i
        to go from i to i + 1 costs cost[i]
        we wwant to to find the starting index if we can traverse the whole circuit
        
        brute force would be to try all indices and see if i can return back to i
        start with that
        '''
        N = len(gas)
        
        for i in range(N):
            tank = 0
            curr = i
            #increase by this amount
            tank += gas[curr]
            while tank > 0:
                #now check if i can reach
                next_station = (curr + 1) % N
                if next_station == i and tank - cost[curr] >= 0:
                    return i+1
                else:
                    #update tank
                    tank -= cost[curr]
                    tank += gas[next_station]
                    curr = next_station
        
        return -1

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        '''
        1. first note that if sum(gas) < sum(cost) we can't do it 
        2. its impossible to start at a station i if gas[i] - cost[i] < 0, or gas[i] < cost[i]
        because there wouldn't be enough in the tank to travel i + 1
        
        we can generalize this by just checking the curr tank at each station, and if it goes < 0 it cannot be the start
        
        algo: 
            1. init total tank and curr tank as zero and choose station 0 as starting
            2. iterate over all stations:
                update totla and curr
                if curr < 0 at i + 1, make i+1 the new starting point and reset curr to 0
            3. return -1 if total < 0 and starting station otherwise
            
        why does this work?
        imagine the situation when the total tank >= 0 and the algoreturns Ns at starting station
        then this ensures its possible to go from Ns to 0, but what  about 0 to Ns
        we can use proof by contradiction and assume 0 < k < Ns such that one couldn't reach this station starting from Ns
        we can right the cost at each stationo as
        \sum_{i=0}^{N} gas[i] - cost[i]
        we can split this as:
        \sum_{i=0}^{k} gas[i] - cost[i] + \sum_{i=k+1}^{N-1} gas[i] - cost[i]
        second part is negative, otherwise the starting stationo would be before Ns
        it couold be equal to zero only in the case k = Ns -1
        
        '''
        n = len(gas)
        
        total_tank, curr_tank = 0, 0
        starting_station = 0
        for i in range(n):
            total_tank += gas[i] - cost[i]
            curr_tank += gas[i] - cost[i]
            # If one couldn't get here,
            if curr_tank < 0:
                # Pick up the next station as the starting one.
                starting_station = i + 1
                # Start with an empty tank.
                curr_tank = 0
        
        return starting_station if total_tank >= 0 else -1

########################
# 1510. Stone Game IV
# 22JAN22
#######################
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        '''
        i can just keep taking stones recursviely until i get the  case on alices turn
        if on alice's turn we have 1 left, return true
        per zermelos' theorem, if the game cannot end in a draw in any way,
        then there exists a winning strategy for either player
        if it is found that a player can get to a winning position from their current position, then that persno must win
        rather, if we start with alice, and we  find that alice can get to a winning position
        than Alice must win,
        if alice can win from that position,  then she must win!
        base case  is fired when alice can get to one stone, which is True
        if 0 stones in her turn, she cant wint
        note that if we start in  reverse of sqrt to 1 its so much faster
        '''
        memo = {}
        
        def dfs(stones):
            if stones == 0:
                return False
            if stones == 1:
                return True
            if stones in memo:
                return memo[stones]
            root = int(stones**0.5)
            for take in range(1,root+1)[::-1]:
                #remoee
                remove = stones - take*take
                #if at any pooint it's false on this turn, its' true on next
                if dfs(remove) == False:
                    memo[stones] = True
                    return True
            memo[stones] = False
            return False
        
        return dfs(n)
                    
class Solution:
    def winnerSquareGame(self, n: int) -> bool:
        '''
        in dfs, we started from n and went down to no stones
        here we need to build up to the number n
        dp[i] reprsents the winning state we are in
        dp[0] = False
        dp[1] = True
        '''
        dp = [False]*(n+1)
        dp[1] = True
        for stone in range(1,n+1):
            for take in range(1, int(stone**0.5)+1)[::-1]:
                remove = stone - take*take
                #there are stoones  left
                if remove >= 0:
                    if dp[remove] == False:
                        dp[stone] = True
                        break
        
        return dp[n]
        
############################
# 1291. Sequential Digits
# 22JAN22
############################
#close one
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        '''
        the largest sequential digit is
        123 456 789
        the smallest sequential digit is 
        12
        i can use recursion to generate a sequential digit using the times 1 plus next digit trick
        '''
        all_sequential = []
        
        def rec(last_digit,path):
            if last_digit > 10:
                return
            if low <= path <= high:
                all_sequential.append(path)
                return
            path = path*10 + last_digit
            rec(last_digit+1,path)
        
        for i in range(1,10):
            rec(i,0)
        return(all_sequential)

class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        '''
        we can use a sliding window apporach
        we notice that all the sequential numbers are substrings of 1234456789
        we can generate all subtrings and check they are inboounds
        '''
        nums = '123456789'
        N = len(nums)
        smallest = len(str(low))
        largest = len(str(high))
        ans = []
        for l in range(smallest,largest+1):
            for start in range(N-l+1):
                num = nums[start:start+l]
                if low <= int(num) <= high:
                    ans.append(num)
        
        return ans

#############################################
#1602. Find Nearest Right Node in Binary Tree
# 23JAN22
##############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findNearestRightNode(self, root: TreeNode, u: TreeNode) -> Optional[TreeNode]:
        '''
        we can just do level order bfs
        and for each level check if the current node is u,
        if it is and have less than the number of nodes, return the next one
        otherwise return None
        '''
        q = deque([root])
        
        while q:
            #get size of current level
            N = len(q)
            for i in range(N):
                curr =  q.popleft()
                if curr == u:
                    if i < N -1:
                        return q.popleft()
                    else:
                        return None
                else:
                    if curr.left:
                        q.append(curr.left)
                    if curr.right:
                        q.append(curr.right)
        return None
        
########################
# 223. Rectangle Area
# 23JAN22
########################
class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        '''
        first find the area of both rects
        then find intersectino
        area1 = abs(C-A)*abs(B-D)
        area2 = abs(E-G)*abs(F-H)
        w = min(C,G)-max(A,E)
        h = min(D, H)-max(B,F)
        if w<=0 or h<=0:
            return area1 + area2
        else:
            return area1 + area2 - w*h
        '''
        area_a = abs(ax2 - ax1)*abs(ay2-ay1)
        area_b = abs(bx2 - bx1)*abs(by2-by1)
        #now we need to find intersection, lets call it w and h
        w_intersection = max(min(ax2,bx2) - max(ax1,bx1),0)
        h_intersection = max(min(ay2,by2) - max(ay1,by1),0)
        
        return area_a + area_b - abs(w_intersection)*abs(h_intersection)

#######################
# 520. Detect Capital
# 24JAN22
#######################
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        '''
        usage of capitals is correct if any of three are true:
            1. all letters are capitals
            2. all leters are not
            3. only the first letter in this is capital
        '''
        def case1(w):
            N = len(w)
            upper_cases = 0
            for ch in w:
                if ch.isupper():
                    upper_cases += 1
            return upper_cases == N
        
        def case2(w):
            N = len(w)
            lower_cases = 0
            for ch in w:
                if ch.islower():
                    lower_cases += 1
            return lower_cases == N
        
        def case3(w):
            N = len(w)
            upper_cases = 0
            for ch in w[1:]:
                if ch.isupper():
                    upper_cases += 1
            return w[0].isupper() and upper_cases == 0
        
        return case1(word) or case2(word) or case3(word)

#cheeky way
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word.islower() or word.isupper() or word.istitle()


#improved
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        '''
        we can improve this if we combine cases
        note that the biggest differeence between cases 2 and 3 are is the condition of that first char
        for 2 and 3, no matter what the frist char is, the rest should be lower case 
        '''
        n = len(word)
        
        if n == 1:
            return True
        
        #case 1, all capital
        if word[0].isupper() and word[1].isupper():
            for i in range(2,n):
                if not word[i].isupper():
                    return False
        else:
            for i in range(1,n):
                if word[i].isupper():
                    return False
        
        return True

import re
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        '''
        we can use regix
        [A-Z]*, matches one char between A-Z followed by any number
        [a-z]*. matches one char between a-z any number
        [A-Z][a-z]*
        we can pipe these three 
        [A−Z]∗∣[a−z]∗∣[A−Z][a−z]∗
        '''
        pattern = r'[A−Z]*|[a−z]*|[A−Z][a−z]*'
        return re.fullmatch(pattern,word)
        

###############################
# 492. Construct the Rectangle
# 24JAN22
###############################
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        '''
        we need to return an L,W == area
        L >= W
        abs(L-W) should be as small as possible
        
        if the webpage is a sqaure, then the L and W values that give the best ans are sqrt(area)
        but we can only go in integer amounts
        i can start with width 1
        keep decrementing by 1 until i find it
        '''
        start = int(area**.5)
        while start >= 1:
            if area % start == 0:
                return [area // start,start]
            else:
                start -= 1

###########################
# 941. Valid Mountain Array
# 25JAN22
###########################
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        '''
        we can just check for strictly increasing and decreasing
        can't check every i
        we need to walk up then walk down
        
        '''
        N = len(arr)
        
        #size requirement
        if N < 3:
            return False
        
        #walk up
        i = 0
        
        while i < N-1 and arr[i+1] > arr[i]: #note the order of the logicals
            i += 1
            
        #if i got to the end or didn't move at all
        if i == 0 or i == N-1:
            return False

        #walk down
        while i < N-1 and arr[i+1] < arr[i]:
            i += 1
        
        #i must be at the end of the array
        return i == N-1

############################
# 500. Keyboard Row
# 25JAN22
############################
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        '''
        for each word check if its chars are in the row
        make sure to lowercase ach one
        we only want to make sure its on only one row!
        '''
        rows = ['qwertyuiop','asdfghjkl','zxcvbnm']
        
        ans = []
        
        for word in words:
            count = 0
            for row in rows:
                word_lower = word.lower()
                #now check each char is in this row
                if all([ch in row for ch in word_lower]):
                    count += 1
            if count == 1:
                ans.append(word)
        
        return ans

#this was kind of cool way to compare sets in python
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        '''
        for each word check if its chars are in the row
        make sure to lowercase ach one
        we only want to make sure its on only one row!
        '''
        rows = [set('qwertyuiop'),set('asdfghjkl'),set('zxcvbnm')]
        
        ans = []
        
        for word in words:
            w = set(word.lower())
            for row in rows:
                if w <= row:
                    ans.append(word)
                    break
        
        return ans

############################
# Gmail Label Strings (Extra Problem https://leetcode.com/discuss/interview-question/1717614/google-phone-screen-swe-l3-gmail-label-strings)
# 25JAN22
############################
'''
given a container of folders, return its labels
folders = [
    {id: 27, parentId: 15, name: 'projects'},
    {id: 81, parentId: 27, name: 'novel'},
    {id: 15, parentId: 0, name: personal'}, // a parentId of 0 means root
    {id: 35, parentId: 27, name: 'blog'},
]

// Sample output:

labels = [
    'personal/projects',
    'personal/projects/novel',
    'personal',
    'personal/projects/blog',
]

i need to make a graph
each vertex is an edge with out direction going 
keep two maps one for graph and one for name
then dfs
'''
import collections
folders = folders = [
    {'id': 27, 'parentId': 15, 'name': 'projects'},
    {'id': 81, 'parentId': 27, 'name': 'novel'},
    {'id': 15, 'parentId': 0, 'name': 'personal'}, #a parentId of 0 means root
    {'id': 35, 'parentId': 27, 'name': 'blog'},
]

graph = collections.defaultdict(list)
id_to_name = collections.defaultdict()

for folder in folders:
    #make edge
    graph[folder['parentId']].append(folder['id'])
    #add to name
    id_to_name[folder['id']] = folder['name']

res = []


def dfs(curr,path):
    res.append(path)
    if curr in graph:
        for neigh in graph[curr]:
            #update path
            path = path + '/' + id_to_name[neigh]
            dfs(neigh,path)

dfs(0,'')
print(res)

#could also do bfs

import collections
folders = folders = [
    {'id': 27, 'parentId': 15, 'name': 'projects'},
    {'id': 81, 'parentId': 27, 'name': 'novel'},
    {'id': 15, 'parentId': 0, 'name': 'personal'}, #a parentId of 0 means root
    {'id': 35, 'parentId': 27, 'name': 'blog'},
]

graph = collections.defaultdict(list)
id_to_name = collections.defaultdict()

for folder in folders:
    #make edge
    graph[folder['parentId']].append(folder['id'])
    #add to name
    id_to_name[folder['id']] = folder['name']

res = []
q = collections.deque([[0,'']]) #store ad list with curr and path
while q:
    curr,path = q.popleft()
    res.append(path)
    if curr in graph:
        for neigh in graph[curr]:
            path = path + '/' + id_to_name[neigh]
            q.append([neigh,path])

print(res)

#############################################
#1305. All Elements in Two Binary Search Trees
# 26JAN22
############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        '''
        dumn way would be to just travers both trees and return sorted
        '''
        ans = []
        def dfs(node):
            if not node:
                return
            ans.append(node.val)
            dfs(node.left)
            dfs(node.right)
            
        dfs(root1)
        dfs(root2)
        
        return sorted(ans)
        
#withour sorting
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        '''
        we can inorder borth trees then use two point merge trick
        '''
        def inorder(root, lst):
            if not root: return
            inorder(root.left, lst)
            lst.append(root.val)
            inorder(root.right, lst)
        
        lst1, lst2 = [], []
        inorder(root1, lst1)
        inorder(root2, lst2)
        
        i1, i2, res = 0, 0, []
        s1, s2 = len(lst1), len(lst2)
        
        while i1 < s1 and i2 < s2:
            if lst1[i1] < lst2[i2]:
                res += [lst1[i1]]
                i1 += 1
            else:
                res += [lst2[i2]]
                i2 += 1
                
        return res + lst1[i1:] + lst2[i2:]
            
class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        '''
        we can save space by doing in order in parallel
        here we can use a stack to arrest the momemnts in recursion
        '''
        stack1 = []
        stack2 = []
        
        merged = []
        
        while root1 or root2 or stack1 or stack2:
            #go left in both
            while root1:
                stack1.append(root1)
                root1 = root1.left
            while root2:
                stack2.append(root2)
                root2 = root2.left
                
            #add the smallest value to merged, pop from stack, and go right
            if not stack2 or stack1 and stack1[-1].val <= stack2[-1].val:
                root1 = stack1.pop()
                merged.append(root1.val)
                root1 = root1.right
            else:
                root2 = stack2.pop()
                merged.append(root2.val)   
                root2 = root2.right

        return merged

#################################
# 244. Shortest Word Distance II
# 26JAN22
#################################
class WordDistance:
    '''
    i can make a mapping for words dict to index position
    then just return the mapped values
    '''

    def __init__(self, wordsDict: List[str]):
        self.mapp = collections.defaultdict(list)
        for i,word in enumerate(wordsDict):
            self.mapp[word].append(i)

    def shortest(self, word1: str, word2: str) -> int:
        #get their posisitions
        pos1 = self.mapp[word1]
        pos2 = self.mapp[word2]
        ans = float('inf')
        for i in pos1:
            for j in pos2:
                ans = min(ans, abs(i-j))
        
        return ans


# Your WordDistance object will be instantiated and called as such:
# obj = WordDistance(wordsDict)
# param_1 = obj.shortest(word1,word2)

class WordDistance:
    '''
    we can improve the shortest call to linear time instead of the double for loop
    we adavance the smallest pointer first to minimize the diff
    '''

    def __init__(self, wordsDict: List[str]):
        self.mapp = collections.defaultdict(list)
        for i,word in enumerate(wordsDict):
            self.mapp[word].append(i)

    def shortest(self, word1: str, word2: str) -> int:
        #get their posisitions
        pos1 = self.mapp[word1]
        pos2 = self.mapp[word2]
        ans = float('inf')
        i,j = 0,0
        #we don't need to traverse the entirey of both lists, just stop when the we done with the shortes one
        #why? if we got to the end of the shorter list, and the second list must be increasing, we 
        #would only every increase the the diff
        while i < len(pos1) and j < len(pos2):
            ans = min(ans, abs(pos1[i]-pos2[j]))
            if pos1[i] < pos2[j]:
                i += 1
            else:
                j += 1
        
        return ans

