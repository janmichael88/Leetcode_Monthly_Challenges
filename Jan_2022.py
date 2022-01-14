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