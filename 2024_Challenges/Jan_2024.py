#############################################
# 455. Assign Cookies (REVISTED)
# 01JAN24
#############################################
#two pointer
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        '''
        we are given len(g) children and len(s) cookies need to maximize the number of contenchildren
        sort greed and cookies increasingly
        '''
        g.sort()
        s.sort()
        ans = 0
        
        i,j = 0,0
        
        while i < len(g) and j < len(s):
            #can assigned
            if g[i] <= s[j]:
                ans += 1
                i += 1
                j += 1
            else:
                j += 1
        
        return ans

#########################################################
# 2610. Convert an Array Into a 2D Array With Conditions
# 02JAN24
########################################################
class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        '''
        need to make 2d, each row should be countain distinct numbers
        and be minimal as possible
        
        count up the elements and greddily use the largest ones until we can't make repeats
        order doesnt matter
        i can use a waiting queue to put elements back in, or just keep track of history
        '''
        N = len(nums)
        used = [False]*N
        
        matrix = []
        
        while sum(used) != N:
            curr_row = []
            for i in range(N):
                if not used[i] and nums[i] not in curr_row:
                    curr_row.append(nums[i])
                    used[i] = True
            
            matrix.append(curr_row)
            curr_row = []
        
        return matrix
    
class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        '''
        we can count up the nums
        the number of rows needed will be the maximum frequency
        we can place 1 of each element at this number of rows
        '''
        counts = Counter(nums)
        rows = max(counts.values())
        ans = [[] for _ in range(rows)]


        for num,count in counts.items():
            for row in range(count):
                ans[row].append(num)
        
        return ans
    
class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        '''
        single pass, but updat whenever we need a new row
        '''
        frequency = [0] * (len(nums) + 1)
        res = []

        for i in nums:
            if frequency[i] >= len(res):
                res.append([])
            res[frequency[i]].append(i)
            frequency[i] += 1

        return res

########################################
# 2125. Number of Laser Beams in a Bank
# 03JAN24
########################################
class Solution:
    def numberOfBeams(self, bank: List[str]) -> int:
        '''
        keep track of the curr row and next row,
        if the nextrow has devices then there are count(ones curr_row)*count(ones next_row)
        then update the rows
        '''
        ans = 0
        N = len(bank)
        curr_row = bank[0]

        for next_row in bank[1:]:
            if next_row.count('1') > 0:
                ans += curr_row.count('1')*next_row.count('1')
                curr_row = next_row
        
        return ans
    
class Solution:
    def numberOfBeams(self, bank: List[str]) -> int:
        '''
        not storing rows just prev count and ans
        '''
        prev = ans = 0
        
        for r in bank:
            count = r.count('1')
            ans += prev*count
            if count:
                prev = count
        
        return ans

########################################################
# 2870. Minimum Number of Operations to Make Array Empty
# 04DEC23
########################################################
#easssssy
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        its always more optimum to delete three eleements if we can
        heap with counts
        if top of heap only ahs counts 1, return -1
        other wise take three if we can
        or take two
        wait, we cant always take three first, if counts are multiple of three, then ops is just count // 3
        if its 4, we need to take it twice or use three until we can get remainder of two, then is plus count//3 
        say we have count like 7
        we cant do 3 3 1,
        we need to do 3 2 2
        what about like 14
        could do 14/2 = 7 times or
        3 3 3 3 3
        
        need to be careful if count % 3 == 2 or count % 3 == 1
        '''
        counts = Counter(nums)
        
        ops = 0
        for num, count in counts.items():
            if count == 1:
                return -1
            #goes into three
            elif count % 3 == 0:
                ops += count // 3
            #if remainder by three is 2, we optimize the use of three
            elif count % 3 == 2:
                ops += (count // 3) + 1
            #if its one, we need to leave at least 4
            else:
                ops += ((count // 3) - 1) + 2
        
        return ops
                
#looking back we can redidce the last one to the count//3 - 1 +2 to count//3 + 1
#now this is just ceiling
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        just ceiling now
        '''
        counts = Counter(nums)
        
        ops = 0
        for num, count in counts.items():
            if count == 1:
                return -1
            else:
                ops += math.ceil(count / 3)
        
        return ops
                
###########################################
# 1066. Campus Bikes II (REVISTED)
# 05DEC23
###########################################
#bottom up direct
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        '''
        set cover problem?
        state is (bike_index, mask of available workers)
        if we get to the last bike and not all the workers have been assigned, invalid state return float('inf')
        otherwise all are assigned return 0
        dp with backtracking
        if its backtrackng with dp, translation isn't always clear
        if we went bottom up right with backtracking, we dont actually need to implement backtracking
        why? because of the order in which we calculate the sub problems
        by the time we calcule dp(i,mask), we could have already computed, but in top down, we keep going until
        we haven computed the subproblem or can access the state through memoization
        '''
        def manhattan(worker, bike):
            return abs(worker[0] - bike[0]) + abs(worker[1] - bike[1])

        numWorkers, numBikes = len(workers), len(bikes)
        
        #dp table is (numWorkers + 1 by (1 << numBikes))
        dp = [[float('inf')] * (1 << numBikes) for _ in range(numWorkers + 1)]

        # Base case: no workers left, distance is 0
        for bikeState in range(1 << numBikes):
            dp[numWorkers][bikeState] = 0


        for workerIndex in range(numWorkers - 1, -1, -1):
            for bikeState in range(1 << numBikes):
                smallestSoFar = float('inf')
                for bikeIndex in range(numBikes):
                    if (bikeState & (1 << bikeIndex)) == 0:
                        toAdd = manhattan(workers[workerIndex], bikes[bikeIndex])
                        potentialSmallest = toAdd + dp[workerIndex + 1][bikeState | (1 << bikeIndex)]
                        smallestSoFar = min(smallestSoFar, potentialSmallest)

                dp[workerIndex][bikeState] = smallestSoFar

        return dp[0][0]
    
#####################################
# 672. Bulb Switcher II
# 05JAN24
#######################################
class Solution:
    def flipLights(self, n: int, m: int) -> int:
        '''
        we have n bulbs, all initally turned on, and 4 buttons
        1. flip all bulbs
        2. flip even bulbs
        3. flip odd bulbs
        4. flip every third bulb
        
        need to do exactly n presses, can do any press
        return number of different possible statuses
        
        intutions:
            1. order of opertions does not matter, if i do 1 then 2, it is the same as 2 then 1
            2. two operations done in succession does nothing i.e 1 then 1, 2 then 2
            3. using button 1 and 2 always gives the affect of button 3, 
            4. using button 1 and 3 gives 2
            5. using button 2 and 3 gives 1
            6. state of all bubls only depends on the first 2 or 3 builbs
        so there are only 8 cases
        All_on, 1, 2, 3, 4, 1+4, 2+4, 3+4
        we can get all these cases when  n > 2 and m >= 3
        so we just chek cases for all n < 3 and m < 3
        '''
        if n == 0:
            return 1
        if n == 1:
            return 2
        if n == 2 and m == 1:
            return 3
        if n == 2:
            return 4
        if m == 1:
            return 4
        if m == 2:
            return 7
        if m >= 3:
            return 8
        return 8

class Solution:
    def flipLights(self, n: int, m: int) -> int:
        '''
        there are not many states
            1. operations are reversible
            2. operations can be reproduced by other operations
        
        beyond 3, n and m become irrelevant because there are at most 8 states all of which become achievable when m and n are large enough;
        below 3, fn(n, m) = fn(n-1, m-1) + fn(n-1, m).
        
        n/m 0 1 2 3 4 5		
        0   1 1 1 1 1 1	 
        1   1 2 2 2 2 2
        2   1 3 4 4 4 4
        3   1 4 7 8 8 8
        4   1 4 7 8 8 8
        5   1 4 7 8 8 8


        '''
        def fn(n,m):
            if m*n == 0:
                return 1
            return fn(n-1,m-1) + fn(n-1,m)
        
        return fn(min(n,3), min(m,3))