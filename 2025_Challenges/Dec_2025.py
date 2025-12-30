############################################
# 3623. Count Number of Trapezoids I
# 02DEC25
#############################################
#TLE
#problem is picking two distinct groups
class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        '''
        group by y coordinate
        then chose two distinct gruos and from each group select two points to make a trapezoid
        remember the trapezoid coords need to be unique
        '''
        mapp = defaultdict(list)
        for x,y in points:
            mapp[y].append((x,y))
        
        groups = [k for k,v in mapp.items()]
        ans = 0
        mod = 10**9 + 7
        n = len(groups)
        for i in range(n):
            for j in range(i+1,n):
                #how many ways can i pick 2 points from groups[i], nCr
                ways1 = math.comb(len(mapp[groups[i]]),2)
                ways2 = math.comb(len(mapp[groups[j]]),2)
                ans += ways1*ways2
                ans %= mod
        
        return ans
    
class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        '''
        group by y coordinate
        then chose two distinct gruos and from each group select two points to make a trapezoid
        remember the trapezoid coords need to be unique
        can't pick two distinct groups, take too long, use sums
        oh whoops you can't just combine the groups into one

        this is a cool trick for computing pairwise products (which would be n*n)
        but in linear time
        suppose you have A: [x0, x1, x2]
res = x0 * x1 + x0 * x2 + x1 * x2 (with 2 nested for loops, O(n^2))

now in O(n)
init: res = 0, tot = 0
for i in 0...n
i = 0: res += tot * x0 ==> res = 0; tot += x0 ==> tot = x0;
i = 1: res += tot * x1 ==> res = x0 * x1; tot += x1 ==> tot = x0+x1;
i = 2: res += tot * x2 ==> res = x0 * x1 + (x0+x1) * x2 ==> res = x0 * x1 + x0 * x2 + x1 * x2
tot += x2 ==> tot = x0 + x1 + x2

I hope it's a little bit clear now

And for this problem, you can also use the suffix sum of number of paires within the same group if you don't know how to compute res in O(n)
        better way to see it
        res += x[i]*(sum of all previous values)
        
        '''
        mapp = Counter()
        n = len(points)
        for x,y in points:
            mapp[y] += 1
        
        ans = 0
        total_sum = 0
        mod = 10**9 + 7
        for k,v in mapp.items():
            first_group = v
            ways1 = math.comb(first_group, 2)
            ans += ways1*total_sum
            ans %= mod
            total_sum += ways1 % mod
        
        return ans % mod
    
class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        '''
        group by y coordinate
        then chose two distinct gruos and from each group select two points to make a trapezoid
        remember the trapezoid coords need to be unique
        can't pick two distinct groups, take too long, use sums
        oh whoops you can't just combine the groups into one

        for linear trick
        say we have [a,b,c,d] and we want [a*b + a*c + a*d + b*c + b*d + c*d]
        so it just a*sum(b,c,d) + b*sum(c,d) * c*sum(d)
        the sums are just suffix sums!, we can also so it that way too
        '''
        mapp = Counter()
        n = len(points)
        for x,y in points:
            mapp[y] += 1
        
        ans = 0
        total_sum = 0
        mod = 10**9 + 7
        for k,v in mapp.items():
            first_group = v
            ways1 = math.comb(first_group, 2)
            ans += ways1*total_sum
            ans %= mod
            total_sum += ways1 % mod
        
        return ans % mod
    
#############################################
# 2211. Count Collisions on a Road
# 04DEC25
##############################################
#almost
class Solution:
    def countCollisions(self, directions: str) -> int:
        '''
        if R -> <- L
            thats two collisions, and they become S at the point of collision
            collisions += 1
        
        if R -> S
            thats one, and the stay there
        
        if S <- L
            thats one, and they stay there
        
        use stack
        '''
        ans = 0
        stack = []
        for ch in directions:
            if ch == 'R':
                stack.append('R')
            elif ch == 'L':
                if stack:
                    if stack[-1] == 'R':
                        ans += 2
                        stack.pop()
                        stack.append('S')
                    elif stack[-1] == 'S':
                        ans += 1
                        stack.pop()
                        stack.append('S')
                else:
                    stack.append('L')
            elif ch == 'S':
                if stack:
                    if stack[-1] == 'R':
                        ans += 1
                        stack.pop()
                        stack.append('S')
                else:
                    stack.append('S')
        
        #since we started going right, we might have SRRSS, we need to tally up these
        new_stack = []
        for ch in stack:
            if ch == 'S':
                while new_stack and new_stack[-1] == 'R':
                    ans += 1
                    new_stack.pop()
                new_stack.append(ch)
            else:
                new_stack.append(ch)
        print(new_stack)
        return ans
    
class Solution:
    def countCollisions(self, directions: str) -> int:
        '''
        travere left to right and use flag to record status of vehicls on left
        if there are no vehicles on left side or all vehilces on left are moving left, set to 1
        if a collision occuret on the left and vehicales stop flag is 0
        if there are consecutive vehiblces on left moving right, flag counts vehicles moving left
        '''
        res = 0
        flag = -1

        for c in directions:
            if c == "L":
                if flag >= 0:
                    res += flag + 1
                    flag = 0
            elif c == "S":
                if flag > 0:
                    res += flag
                flag = 0
            else:
                if flag >= 0:
                    flag += 1
                else:
                    flag = 1
        return res
    
class Solution:
    def countCollisions(self, directions: str) -> int:
        '''
        remove left moving on left side and right moving in rihgt side
        everything inside should collide exactly one
        '''
        directions = directions.lstrip("L").rstrip("R")
        return len(directions) - directions.count("S")
    
#####################################################
# 3432. Count Partitions with Even Sum Difference
# 05DEC25
#######################################################
class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        '''
        '''
        left_sum = 0
        right_sum = sum(nums)
        ans = 0
        for num in nums[:-1]:
            left_sum += num
            right_sum -= num
            print(left_sum,right_sum)
            if (left_sum - right_sum) % 2 == 0:
                ans += 1
        
        return ans

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        '''
        just check total sum
        if total sum is even, then any partition works, which there are n - 1
        '''
        sum_ = sum(nums)
        n = len(nums)
        if sum_ % 2 == 0:
            return n-1
        return 0
    
##############################################################
# 3578. Count Partitions With Max-Min Difference at Most K
# 06NOV25
#############################################################
#TLE, n*n, recursively
class Solution:
    def countPartitions(self, nums: List[int], k: int) -> int:
        '''
        need to partition nums into on or more contiguous segments
        such that in each segment, the dif between its min and max is at most k 
        oh each segment it must be <= k, not among all segments
        [9,4,1,3,7], k = 4
        if we did the whole array, it would not work
        9-1 > 4
        if there are n nums, there arr n-1 possible partition spots
        there's always at least 1 possible partition scheme
        try n*n recursive solution first
        '''
        n = len(nums)
        memo = {}
        mod = 10**9 + 7
        def dp(i):
            if i == n:
                return 1   # one valid partitioning completed
            if i in memo:
                return memo[i]

            ways = 0
            curr_min = curr_max = nums[i]

            for j in range(i, n):
                curr_min = min(curr_min, nums[j])
                curr_max = max(curr_max, nums[j])

                if curr_max - curr_min <= k:
                    ways += dp(j + 1)
                    ways %= mod
                else:
                    break
            ways %= mod
            memo[i] = ways
            return ways
        
        return dp(0)

################################################################
# 3577. Count the Number of Computer Unlocking Permutations
# 10DEC25
###############################################################
import math
class Solution:
    def countPermutations(self, complexity: List[int]) -> int:
        '''
        count number of valid ways such that i can unlock computers
        each way is a permutation
        if im at j, i can go to any i, such that j < i and complexity[j] < complexity[i]
        but j must have been unlocked first
        0 starts unlocked
        '''
        #ensure we can start at 0
        start = complexity[0]
        for n in complexity[1:]:
            if n <= start:
                return 0
        
        n = len(complexity)
        return math.factorial(n-1) % (10**9 + 7)
    
################################################
# 3531. Count Covered Buildings
# 12DEC25
################################################
class Solution:
    def countCoveredBuildings(self, n: int, buildings: List[List[int]]) -> int:
        '''
        sort by x and y then check
        for each y store the xvalues sorted
        for each x, store the y values sorted
        another way is just to store the mins and maxes without sorting
        '''
        x_map = defaultdict(list)
        y_map = defaultdict(list)
        for x,y in buildings:
            y_map[y].append(x)
            x_map[x].append(y)
        
        #sort
        for k in x_map:
            x_map[k] = sorted(x_map[k])
        for k in y_map:
            y_map[k] = sorted(y_map[k])
        
        ans = 0
        for x,y in buildings:
            x_axis = y_map[y]
            y_axis = x_map[x]
            if (x_axis[0] < x < x_axis[-1]) and (y_axis[0] < y < y_axis[-1]): 
                ans += 1
        
        return ans
    
###################################################
# 3433. Count Mentions Per User
# 12DEC25
##################################################
#close one
class Solution:
    def countMentions(self, n: int, events: List[List[str]]) -> List[int]:
        '''
        MESSAGE, can be one where a mentions b, could message all, here mentions all online users
        OFFLINE, perons goes away at timestamp and becomes available at timestamp + 60
        '''
        counts = [0]*n
        #need to check offline before messaging
        offline = deque([]) #store as (time,id)
        online = set()
        #all online
        for i in range(n):
            online.add(i)
        sorted_events = []
        for mess,time,string in events:
            entry = [mess,int(time),string]
            sorted_events.append(entry)
        
        sorted_events.sort(key = lambda x: x[1])

        for mess,time,string in sorted_events:
            #return all offline back online if possible
            while offline and offline[0][0] + 60 >= time:
                a,b = offline.popleft()
                online.add(b)
            if mess == "MESSAGE":
                if string == "ALL":
                    for i in range(n):
                        counts[i] += 1
                elif string == "HERE":
                    for i in online:
                        counts[i] += 1
                else:
                    string = string.split(" ")
                    for m in string:
                        ID = int(m[2:])
                        counts[ID] += 1
            #user goes offline
            else:
                offline.append((time,int(string)))
                online.remove(int(string))
        return counts
    
#just mark next on line time for user
class Solution:
    def countMentions(
        self, numberOfUsers: int, events: List[List[str]]) -> List[int]:
        '''
        '''
        events.sort(key=lambda e: (int(e[1]), e[0] == "MESSAGE"))
        count = [0] * numberOfUsers
        next_online_time = [0] * numberOfUsers
        for event in events:
            cur_time = int(event[1])
            if event[0] == "MESSAGE":
                if event[2] == "ALL":
                    for i in range(numberOfUsers):
                        count[i] += 1
                elif event[2] == "HERE":
                    for i, t in enumerate(next_online_time):
                        if t <= cur_time:
                            count[i] += 1
                else:
                    for idx in event[2].split():
                        count[int(idx[2:])] += 1
            else:
                next_online_time[int(event[2])] = cur_time + 60
        return count
    
############################################
# 3606. Coupon Code Validator
# 13DEC25
############################################
import re
class Solution:
    def validateCoupons(self, code: List[str], businessLine: List[str], isActive: List[bool]) -> List[str]:
        '''
        follow the rules
        '''

        def is_valid(s):
            return bool(re.fullmatch(r"[A-Za-z0-9_]+", s))
        
        mapp = defaultdict(list)
        n = len(code)
        for i in range(n):
            c = code[i]
            if is_valid(c) and isActive[i]:
                mapp[businessLine[i]].append(c)
        
        print(mapp)
        needed = ["electronics", "grocery", "pharmacy", "restaurant"]
        ans = []
        for n in needed:
            ans.extend(sorted(mapp[n]))
        
        return ans

###################################################
# 2110. Number of Smooth Descent Periods of a Stock
# 15DEC25
###################################################
class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        '''
        if we have a decreasing array streak, keep going
        if its length is k, we can can k! to it
        '''
        ans = 0
        streak = prices[0]
        size = 1
        for p in prices[1:]:
            #extend streak
            if streak - p == 1:
                size += 1
                streak = p
            else:
                ans += (size*(size + 1)) //2
                streak = p
                size = 1

        ans += (size*(size + 1)) //2
        return ans
    
class Solution:
    def getDescentPeriods(self, prices: List[int]) -> int:
        '''
        if we have a decreasing array streak, keep going
        if its length is k, we can can k! to it
        '''
        ans = 1
        streak = prices[0]
        size = 1
        for p in prices[1:]:
            #extend streak
            if streak - p == 1:
                size += 1
                streak = p
            else:
                streak = p
                size = 1
            ans += size

        return ans
    

##########################################################
# 3562. Maximum Profit from Trading Stocks with Discounts
# 16DEC25
########################################################
#ehhhh
import math
class Solution:
    def maxProfit(self, n: int, present: List[int], future: List[int], hierarchy: List[List[int]], budget: int) -> int:
        '''
        if there isn't a discount
        the profixt is future[i] - present[i], if we didn't go over budget
        if a boss purchases stock, then all their employees get the discount
        there are no duplicate edges, and its DAG, no cycles
        discount is for employee's direct boss
        keep in mind, index 1 is the CEO
        if im at index i, he gets discount if boss j has bought
        states (boss_bought,curr,budget)
        given the test conditions, if n nodes, there can be at most n-1 edges
        '''
        graph = defaultdict(list)
        for boss,emp in hierarchy:
            graph[boss].append(emp)
        
        memo = {}
        def dp(boss_bought,curr,budget):
            if budget < 0:
                return float('-inf')
            key = (boss_bought,curr,budget)
            if key in memo:
                return memo[key]

            ans = 0
            for emp in graph[curr]:
                if boss_bought:
                    cost_to_buy = present[curr-1] // 2
                else:
                    cost_to_buy = present[curr-1]
                
                profit = future[curr-1] - cost_to_buy
                buy = profit + dp(True,emp,budget - profit)
                no_buy = dp(False,emp,budget)
                ans = max(ans,buy,no_buy)
            memo[key] = ans
            return ans
        
        return dp(False,1,budget)

#almost
import math
class Solution:
    def maxProfit(self, n: int, present: List[int], future: List[int], hierarchy: List[List[int]], budget: int) -> int:
        '''
        if there isn't a discount
        the profixt is future[i] - present[i], if we didn't go over budget
        if a boss purchases stock, then all their employees get the discount
        there are no duplicate edges, and its DAG, no cycles
        discount is for employee's direct boss
        keep in mind, index 1 is the CEO
        if im at index i, he gets discount if boss j has bought
        states (boss_bought,curr,budget)
        given the test conditions, if n nodes, there can be at most n-1 edges
        '''
        graph = defaultdict(list)
        for boss, emp in hierarchy:
            graph[boss].append(emp)

        memo = {}

        def dp(curr: int, boss_bought: bool, budget: int) -> int:
            if budget < 0:
                return float('-inf')

            key = (curr, boss_bought, budget)
            if key in memo:
                return memo[key]

            # ---------- OPTION 1: do NOT buy curr ----------
            profit_no_buy = 0
            for emp in graph[curr]:
                profit_no_buy += dp(emp, False, budget)

            # ---------- OPTION 2: buy curr ----------
            cost = present[curr - 1] // 2 if boss_bought else present[curr - 1]

            profit_buy = float('-inf')
            if budget >= cost:
                profit_buy = future[curr - 1] - cost
                for emp in graph[curr]:
                    profit_buy += dp(emp, True, budget - cost)

            ans = max(profit_no_buy, profit_buy)
            memo[key] = ans
            return ans

        return dp(1, False, budget)
    

################################################
# 3573. Best Time to Buy and Sell Stock V
# 17DEC25
###############################################
#aww man
#MLE
class Solution:
    def maximumProfit(self, prices: List[int], k: int) -> int:
        '''
        states should be (i,k,holding)
        then we have options 
        0: not holding
        1: holding
        2: in shortselling
        '''
        memo = {}
        n = len(prices)

        def dp(i, k, holding):
            if k < 0:
                return float('-inf')
            if i == n:
                return 0 if holding == 0 else float('-inf')

            key = (i, k, holding)
            if key in memo:
                return memo[key]

            if holding == 0:
                buy = -prices[i] + dp(i + 1, k, 1)
                skip = dp(i + 1, k, 0)

                #FIX: short consumes a transaction on OPEN
                short = prices[i] + dp(i + 1, k - 1, 2)

                ans = max(buy, skip, short)

            elif holding == 1:
                sell = prices[i] + dp(i + 1, k - 1, 0)
                skip = dp(i + 1, k, 1)
                ans = max(sell, skip)

            else:  # holding == 2 (short)
                cover = -prices[i] + dp(i + 1, k, 0)
                skip = dp(i + 1, k, 2)
                ans = max(cover, skip)

            memo[key] = ans
            return ans

        return dp(0, k, 0)

#need to use 2d array for memo or just cache decorator
#nopeee
class Solution:
    def maximumProfit(self, prices: List[int], k: int) -> int:
        '''
        states should be (i,k,holding)
        then we have options 
        0: not holding
        1: holding
        2: in shortselling
        '''
        memo = {}
        n = len(prices)

        @cache
        def dp(i, k, holding):
            if k < 0:
                return float('-inf')
            if i == n:
                return 0 if holding == 0 else float('-inf') #need to complete a transaction
            
            #key = (i, k, holding)
            #if key in memo:
            #    return memo[key]
            
            if holding == 0:
                buy = -prices[i] + dp(i + 1, k, 1)
                skip = dp(i + 1, k, 0)

                #FIX: short consumes a transaction on OPEN
                short = prices[i] + dp(i + 1, k - 1, 2)

                ans = max(buy, skip, short)

            elif holding == 1:
                sell = prices[i] + dp(i + 1, k - 1, 0)
                skip = dp(i + 1, k, 1)
                ans = max(sell, skip)

            else:  # holding == 2 (short)
                cover = -prices[i] + dp(i + 1, k, 0)
                skip = dp(i + 1, k, 2)
                ans = max(cover, skip)

            #memo[key] = ans
            return ans

        return dp(0, k, 0)

#bottom up
class Solution:
    def maximumProfit(self, prices: List[int], k: int) -> int:
        n = len(prices)
        NEG_INF = float('-inf')

        # dp[i][t][h]
        dp = [[[NEG_INF] * 3 for _ in range(k + 1)] for _ in range(n + 1)]

        # ---- base case ----
        for t in range(k + 1):
            dp[n][t][0] = 0
            dp[n][t][1] = NEG_INF
            dp[n][t][2] = NEG_INF

        # ---- fill table ----
        for i in range(n - 1, -1, -1):
            for t in range(k + 1):

                # flat
                dp[i][t][0] = max(
                    dp[i + 1][t][0],                    # skip
                    -prices[i] + dp[i + 1][t][1],       # buy
                    prices[i] + dp[i + 1][t][2]         # short
                )

                # long
                dp[i][t][1] = dp[i + 1][t][1]           # hold
                if t > 0:
                    dp[i][t][1] = max(
                        dp[i][t][1],
                        prices[i] + dp[i + 1][t - 1][0] # sell
                    )

                # short
                dp[i][t][2] = dp[i + 1][t][2]           # hold short
                if t > 0:
                    dp[i][t][2] = max(
                        dp[i][t][2],
                        -prices[i] + dp[i + 1][t - 1][0] # cover
                    )

        return dp[0][k][0]

##############################################
# 955. Delete Columns to Make Sorted II
# 21DEC25
##############################################
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        '''
        need the minimum possible deletion indexes to delete from strs to make them equal
        we can delete alll indices to make them ordered (null strings)
        so at most n deletions, maybe at least 0
        ca
        bb
        ac
        go in order, if col is ordered we keep it, otherwise we need to delete it
        '''
        def inorder(arr):
            n = len(arr)
            for i in range(1,n):
                if arr[i-1] > arr[i]:
                    return False
            return True
        transposed = list(zip(*strs))
        deletions = 0
        n = len(transposed)
        needed = [""]*len(strs)
        for col in transposed:
            curr = needed[:]
            for i,ch in enumerate(col):
                curr[i] += ch
            if inorder(curr):
                needed = curr
            else:
                deletions += 1
        
        return deletions

##########################################
# 960. Delete Columns to Make Sorted III
# 23DEC25
#########################################
class Solution:
    def minDeletionSize(self, A: List[str]) -> int:
        '''
        treat like maximum increasing subsequence
        '''
        n = len(A[0])
        dp = [1] * n
        for j in range(1, n):
            for i in range(j):
                #for each str in strs, ensure strs[:i] is orderd
                if all(a[i] <= a[j] for a in A):
                    #if it is, we can extend the sequence (among all words)
                    dp[j] = max(dp[j], dp[i] + 1)
        #negative answer
        return n - max(dp)
    
##########################################
# 3074. Apple Redistribution into Boxes
# 24DEC25
##########################################
class Solution:
    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        '''
        just use the largst boxes
        '''
        apples = sum(apple)
        capacity.sort(reverse = True)
        boxes = 0
        for c in capacity:
            apples -= c
            boxes += 1
            if apples <= 0:
                return boxes
        return boxes 
    
##############################################
# 939. Minimum Area Rectangle
# 26DEC25
##############################################
class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        '''
        just pick two points and use as bottom left and upper right
        make sure they are not colinear
        '''
        pointset = set(map(tuple, points))
        ans = float('inf')

        for i in range(len(points)):
            x1, y1 = points[i]
            for j in range(i + 1, len(points)):
                x2, y2 = points[j]

                # must form a diagonal
                if x1 == x2 or y1 == y2:
                    continue

                # check the other 2 corners are in the pointSet
                if (x1, y2) in pointset and (x2, y1) in pointset:
                    area = abs(x2 - x1) * abs(y2 - y1)
                    ans = min(ans, area)

        return 0 if ans == float('inf') else ans

###############################################
# 756. Pyramid Transition Matrix
# 29DEC25
###############################################
class Solution:
    def pyramidTransition(self, bottom: str, allowed: List[str]) -> bool:
        '''
        should be a hardo problem, 
        just call a peice three length tuple, (x,y,z)
        bottom level only goes up to 6, so there are 6 levels
        1
        22
        333
        4444
        55555
        666666
        which means there are 1 + 2 + 3 + 4 + 5 = 15, 3 block pieces
        (i,j), start at (0,0) the top piece, every piece i use advances i + 1, and j + 2
        if i get to the second to last row, i need to make sure the peices that i use match
        i would also need the tip of the block
        would be easier to start off with mapp children map to back to parent
        '''
        graph = defaultdict(list)  # (left,right) -> list of possible tops
        for left, right, top in allowed:
            graph[(left, right)].append(top)

        def build(row, path, i, results):
            # finished building row above
            if i == len(row) - 1:
                results.append("".join(path))
                return

            pair = (row[i], row[i+1])
            if pair not in graph:
                return

            for top in graph[pair]:
                path.append(top)
                build(row, path, i + 1, results)
                path.pop()

        def all_rows_above(row):
            results = []
            build(row, [], 0, results)
            return results

        @lru_cache(None)
        def rec(row):
            if len(row) == 1:
                return True

            for cand in all_rows_above(row):
                if rec(cand):
                    return True
            return False

        return rec(bottom)
    
########################################
# 1895. Largest Magic Square
# 30DEC25
#########################################
#barely passes
class Solution:
    def largestMagicSquare(self, grid: List[List[int]]) -> int:
        '''
        check all k sqaures
        '''
        rows,cols = len(grid),len(grid[0])
        ans = 1
        for k in range(min(rows,cols),1,-1):
            for i in range(rows-k+1):
                for j in range(cols-k+1):
                    square = self.get_square(grid,i,j,k)
                    if self.check_magic(square,k):
                        return k
        return ans

    
    def get_square(self,grid,i,j,k):
        square = []
        for ii in range(k):
            row = []
            for jj in range(k):
                row.append(grid[i+ii][j+jj])

            square.append(row)
        
        return square
    def check_magic(self,square,k):
        row_sums = [0]*k
        col_sums = [0]*k
        diag_sum = 0
        anti_diag_sum = 0
        
        for i in range(k):
            for j in range(k):
                row_sums[i] += square[i][j]
                col_sums[j] += square[i][j]
                if i == j:
                    diag_sum += square[i][j]
                if i + j == k-1:
                    anti_diag_sum += square[i][j]
        check_sums = row_sums + col_sums + [diag_sum] + [anti_diag_sum]
        return len(set(check_sums)) == 1
    
class Solution:
    def largestMagicSquare(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        ans = 1

        # try largest k first
        for k in range(min(rows, cols), 1, -1):
            for i in range(rows - k + 1):
                for j in range(cols - k + 1):
                    if self.check_magic(grid, i, j, k):
                        return k
        return ans


    def check_magic(self, grid, si, sj, k):
        # target sum = first row sum
        target = sum(grid[si][sj + x] for x in range(k))

        # check all rows
        for r in range(k):
            s = 0
            for c in range(k):
                s += grid[si + r][sj + c]
            if s != target:
                return False

        # check all cols
        for c in range(k):
            s = 0
            for r in range(k):
                s += grid[si + r][sj + c]
            if s != target:
                return False

        # main diagonal
        diag = 0
        for d in range(k):
            diag += grid[si + d][sj + d]
        if diag != target:
            return False

        # anti-diagonal
        anti = 0
        for d in range(k):
            anti += grid[si + d][sj + (k - 1 - d)]
        if anti != target:
            return False

        return True
