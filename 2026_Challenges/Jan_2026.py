############################################
# 1411. Number of Ways to Paint N × 3 Grid
# 03JAN25
############################################
#this is actually a very beautfil problem
class Solution:
    def numOfWays(self, n: int) -> int:
        '''
        need to use dp
        colors can be, red, yellow, green, (0,1,2)
        if im at cell (i,j), i need to make sure its color isn't the same at (i-1,j) and (i,j-1)
        we will fill going left to right, and up to down
        enumerate all possible transitions from one row to the next
        '''
        mod = 10**9 + 7
        colors = [0,1,2]
        patterns = []
        
        #get patterns
        for a in colors:
            for b in colors:
                for c in colors:
                    if a != b and b != c:
                        patterns.append((a,b,c))
        
        graph = defaultdict(list)
        for p1 in patterns:
            for p2 in patterns:
                #make sure cols aren't the same
                if (p1[0] != p2[0] and p1[1] != p2[1] and p1[2] != p2[2]):
                    graph[p1].append(p2)
        
        memo = {}
        def dp(i,state):
            if i >= n:
                return 1
            if (i,state) in memo:
                return memo[(i,state)]
            
            #no pattern yet
            ways = 0
            for neigh in graph[state]:
                ways += dp(i+1,neigh)
                ways %= mod
            ways %= mod
            memo[(i,state)] = ways
            return ways

        ans = 0
        for p in graph:
            ans += dp(1,p)
            ans %= mod
        return ans
    
    
class Solution:
    def numOfWays(self, n: int) -> int:
        '''
        need to use dp
        colors can be, red, yellow, green, (0,1,2)
        if im at cell (i,j), i need to make sure its color isn't the same at (i-1,j) and (i,j-1)
        we will fill going left to right, and up to down
        enumerate all possible transitions from one row to the next
        '''
        mod = 10**9 + 7
        colors = [0, 1, 2]
        patterns = []

        # generate all valid row patterns (no adjacent equal colors)
        for a in colors:
            for b in colors:
                for c in colors:
                    if a != b and b != c:
                        patterns.append((a, b, c))

        # build compatibility graph between rows
        graph = defaultdict(list)
        for p1 in patterns:
            for p2 in patterns:
                if (p1[0] != p2[0] and
                    p1[1] != p2[1] and
                    p1[2] != p2[2]):
                    graph[p1].append(p2)

        # dp[p] = number of ways where current row has pattern p
        dp = {p: 1 for p in patterns}   # first row

        # build rows 2..n
        for _ in range(1, n):
            new_dp = {p: 0 for p in patterns}
            for p in patterns:
                ways = dp[p]
                if ways == 0:
                    continue
                for nxt in graph[p]:
                    new_dp[nxt] = (new_dp[nxt] + ways) % mod
            dp = new_dp

        return sum(dp.values()) % mod

############################################
# 1390. Four Divisors
# 04JAN25
############################################
class Solution:
    def sumFourDivisors(self, nums: List[int]) -> int:
        '''
        for each number get the divsors in logrithmic time
        if there are 4, sum them up
        21 -> 1,3,4,7
        '''
        def get_divisors(num):
            divs = set()
            curr_div = 1
            while curr_div*curr_div <= num:
                if num % curr_div == 0:
                    divs.add(curr_div)
                    divs.add(num // curr_div)
                curr_div += 1
            if len(divs) == 4:
                return sum(divs)
            return 0
        
        ans = 0
        for num in nums:
            ans += get_divisors(num)
        return ans
    
################################################################
# 2237. Count Positions on Street With Required Brightness
# 05JAN26
################################################################
class Solution:
    def meetRequirement(self, n: int, lights: List[List[int]], requirement: List[int]) -> int:
        '''
        this is line sweep
        count at each position i
        '''
        counts = [0]*(n+1)
        for idx,r in lights:
            left = max(0,idx-r)
            right = min(n-1,idx+r)
            counts[left] += 1
            counts[right+1] -= 1
        
        for i in range(1,n):
            counts[i] += counts[i-1]
        
        ans = 0
        for i in range(n):
            ans += counts[i] >= requirement[i]
        
        return ans