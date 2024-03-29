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
    
###################################################
# 1235. Maximum Profit in Job Scheduling (REVISTED)
# 06DEC24
###################################################
#good review
class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        '''
        sort and use dp(i) as the maximum profift using using jobs[i:]
        want dp(0)
        sort by starting time
        '''
        entries = []
        for s,e,p in zip(startTime,endTime,profit):
            entries.append((s,e,p))
        
        entries.sort(key = lambda x: x[0])
        #best resplit back into arrays for the search
        starts = []
        ends = []
        profs = []
        
        for s,e,p in entries:
            starts.append(s)
            ends.append(e)
            profs.append(p)
        
        N = len(starts)
        memo = {}
        
        def bin_search(arr,next_end): #arr will be the ends array
            left = 0
            right = len(arr)
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] < next_end:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        
        #test = bin_search(ends,3)
        #print(ends[test])
            
        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            
            #knapsack take job starting at i, or move to the next i+1 job
            #ans is the max of these two options
            next_end = bin_search(starts,ends[i])
            take = profs[i] + dp(next_end)
            no_take = dp(i+1)
            ans = max(take,no_take)
            memo[i] = ans
            return ans
        
        return dp(0)
            
#bottom up
class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        '''
        sort and use dp(i) as the maximum profift using using jobs[i:]
        want dp(0)
        sort by starting time
        '''
        entries = []
        for s,e,p in zip(startTime,endTime,profit):
            entries.append((s,e,p))
        
        entries.sort(key = lambda x: x[0])
        #best resplit back into arrays for the search
        starts = []
        ends = []
        profs = []
        
        for s,e,p in entries:
            starts.append(s)
            ends.append(e)
            profs.append(p)
        
        N = len(starts)
        dp = [0]*(N+1)
        
        def bin_search(arr,next_end): #arr will be the ends array
            left = 0
            right = len(arr)
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] < next_end:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        
        for i in range(N-1,-1,-1):
            next_end = bin_search(starts,ends[i])
            take = profs[i] + dp[next_end]
            no_take = dp[i+1]
            ans = max(take,no_take)
            dp[i] = ans
        
        return dp[0]
    
############################################
# 2008. Maximum Earnings From Taxi
# 06JAN24
############################################
class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        '''
        sort in starts
        n doesn't fucking matter lmaooooo
        '''
        rides.sort(key = lambda x: x[0])
        N = len(rides)
        memo = {}
        
        def bin_search(arr,next_start): #arre is rides array
            left = 0
            right = len(arr)
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid][0] < next_start:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        def dp(i):
            if i >= N:
                #make surew we didn't go past n
                return 0
            
            if i in memo:
                return memo[i]
            
            next_start = bin_search(rides,rides[i][1])
            prof = rides[i][1] - rides[i][0] + rides[i][2]
            take = prof + dp(next_start)
            no_take = dp(i+1)
            ans = max(take,no_take)
            memo[i] = ans
            return ans
        
        return dp(0)
    
#bottom up
class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        '''
        sort in starts
        n doesn't fucking matter lmaooooo
        '''
        rides.sort(key = lambda x: x[0])
        N = len(rides)
        memo = {}
        
        def bin_search(arr,next_start): #arre is rides array
            left = 0
            right = len(arr)
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid][0] < next_start:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        dp = [0]*(N+1)
        
        for i in range(N-1,-1,-1):
            next_start = bin_search(rides,rides[i][1])
            prof = rides[i][1] - rides[i][0] + rides[i][2]
            take = prof + dp[next_start]
            no_take = dp[i+1]
            ans = max(take,no_take)
            dp[i] = ans
        
        return dp[0]
    
############################################
# 301. Remove Invalid Parentheses
# 06JAN24
#############################################
#gotta love AC brute force solutions
class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        '''
        try all removals and check balance
        need to check balance at the end
        '''
        valid = []
        N = len(s)
        mapp = {'(': 1, ')':-1}
        self.min = N
        
        def is_valid(s):
            balance = 0
            for ch in s:
                if ch in mapp:
                    balance += mapp[ch]
                if balance < 0:
                    return False
            
            return balance == 0
        def rec(i,path):
            if i == N:
                if is_valid(path):
                    valid.append((path, N - len(path)))
                    self.min = min(self.min,N - len(path))
                return
            if s[i] in mapp:
                rec(i+1, path+s[i])
                rec(i+1,path)
            else:
                rec(i+1, path+s[i])
        
        rec(0,"")
        ans = set()
        for p,d in valid:
            if d == self.min:
                ans.add(p)
        
        return ans
    
#pruning
class Solution:
    
    def __init__(self):
        self.valid_expressions = None
        self.min_removed = None
        
    def reset(self):
        self.valid_expressions = set()
        self.min_removed = float('inf')
    
    def backtrack(self,string, i, left, right, path, removals):
        #end of string
        if i == len(string):
            #validate
            if left == right:
                #next smallest removels
                if removals <= self.min_removed:
                    possible = "".join(path)
                    #new min
                    if removals < self.min_removed:
                        self.valid_expressions = set()
                        self.min_removed = removals
                    #add in
                    self.valid_expressions.add(possible)
        
        else:
            curr_char = string[i]
            #not a bracket
            if curr_char not in '()':
                path.append(curr_char)
                self.backtrack(string,i+1,left,right,path,removals)
                path.pop()
            else:
                #deletion
                self.backtrack(string,i+1,left,right,path,removals+1)
                #add char to path
                path.append(curr_char)
                if curr_char == '(':
                    self.backtrack(string,i+1,left+1,right,path,removals)
                elif right < left:
                    #consinde onle when we dont have enough closing
                    self.backtrack(string,i+1,left,right+1,path,removals)
                path.pop()
        
        
    def removeInvalidParentheses(self, s: str) -> List[str]:
        '''
        we can prune base on closing bracket
        we can't early termiante on opening, because there could be closing to make it valid
        but closing, we can
            if we have too many closing then this path would eventually become invalid
        
        states are index, left_count, right_count, expre, rem_count
        left_count and right_counts are counts of parenthese
        rem_count is the number of removals
        keep expression in [] and pop when we are done
        use getter and setter methods
        '''
        self.reset()
        self.backtrack(s,0,0,0,[],0)
        return self.valid_expressions
        
###################################################
# 446. Arithmetic Slices II - Subsequence (REVISTED)
# 07JAN24
###################################################
#doesn't pass in python but barely passes in JAVA
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        count all arithmetic subequeces, then subtract out the weak ones
        state should be index and common difference d, call it (i,diff)
        if we are some index i, we can pick and num at index j with j being [i+1,N-1]
            so long as nums[j] == diff
        
        the we just check all (i,j) and sum up dp(i, nums[j] - nums[i])
            
        
        '''
        N = len(nums)
        memo = {}
        
        def dp(i,diff):
            if i == N:
                return 0
            
            if (i,diff) in memo:
                return memo[(i,diff)]
            
            curr_ways = 0
            for j in range(i+1,N):
                next_diff = nums[j] - nums[i]
                if next_diff == diff:
                    curr_ways += dp(j,next_diff) + 1
            
            memo[(i,diff)] = curr_ways
            return curr_ways
        
        ways = 0
        for i in range(N):
            for j in range(i+1,N):
                ways += dp(j,nums[j] - nums[i])
        
        return ways
    
#bottom up keep dp array of Counter objects and keep count of 2 lenght arith sequences, which is just any pair
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        N = len(nums)
        dp = [Counter() for _ in range(N)]
        
        for i in range(N):
            for j in range(i):
                diff = nums[i] - nums[j]
                dp[i][diff] += dp[j][diff] + 1
        
        
        return sum(sum(count.values()) for count in dp) - (N*(N-1)//2)
    
#without subtracint out week arithmetic sequences
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        N = len(nums)
        dp = [Counter() for _ in range(N)]
        
        ans = 0
        for i in range(N):
            for j in range(i):
                diff = nums[i] - nums[j]
                dp[i][diff] += dp[j][diff] + 1
                #add the ones ending at j
                ans += dp[j][diff]
        
        return ans
    
##############################################
# 872. Leaf-Similar Trees (REVISTED)
# 09JAN24
##############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        '''
        leaves should read left node right
        '''
        def getLeaves(node):
            if not node:
                return []
            if not node.left and not node.right:
                return [node.val]
            left = getLeaves(node.left)
            right = getLeaves(node.right)
            return left + right
        
        
        return getLeaves(root1) == getLeaves(root2)
            
#####################################################
# 2385. Amount of Time for Binary Tree to Be Infected
# 10JAN24
####################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        '''
        turn the tree into a directed graph
        then just do BFS
        but we need to keep track of the longest path, not the number of layers we explore
        '''
        graph = defaultdict(list)
        def dfs(node,parent):
            if not node:
                return
            if parent:
                graph[node.val].append(parent.val)
                graph[parent.val].append(node.val)
            dfs(node.left,node)
            dfs(node.right,node)
        
        dfs(root,None)
        time = 0
        q = [(start,0)]
        visited = set()
        
        
        while len(q) > 0:
            N = len(q)
            next_q = []
            for i in range(N):
                curr,curr_time = q[i]
                time = max(time,curr_time)
                visited.add(curr)
                for neigh in graph[curr]:
                    if neigh not in visited:
                        next_q.append((neigh,curr_time + 1))
            
            q = next_q
    
        return time
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        '''
        can also just count layers
        '''
        graph = defaultdict(list)
        def dfs(node,parent):
            if not node:
                return
            if parent:
                graph[node.val].append(parent.val)
                graph[parent.val].append(node.val)
            dfs(node.left,node)
            dfs(node.right,node)
        
        dfs(root,None)
        time = 0
        q = [start]
        visited = set()
        
        
        while len(q) > 0:
            N = len(q)
            next_q = []
            for i in range(N):
                curr = q[i]
                visited.add(curr)
                for neigh in graph[curr]:
                    if neigh not in visited:
                        next_q.append(neigh)
            
            q = next_q
            time += 1
        return time - 1
    
#one pass
class Solution:
    def __init__(self):
        self.max_dist = 0
        
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        '''
        if start node were the root of three, the answer would just be the depth
        need to find the maximum dista from the start node using the depths
        say the start node is at some depth k (From the root)
        which means from the root, it is k away, do take depth from one of the subtrees and it to k (this would be for the subtree above the start node if the
        start node were not the root)
        to find the subtree below this node, answer if just the depth from this node, or (Depth for while tree) - k
        ans is the max
        
        need to find the max distance of the start node using the depths of the subtrees
        if we have found the start note in this subtree return a negative depth
        wwehn we encounter a negative depth, we know this subtree contains the start node
        
        we might be in the case that we have caluclated max depth before finding the start node, so we need to save the max distnace and continue searching
        '''
        a = self.dfs(root,start)
        print(a)
        return self.max_dist
        
    def dfs(self,root,start): #this function should return the max depth with start node start
        #no depth
        depth = 0
        if not root:
            return depth
        
        left = self.dfs(root.left,start)
        right = self.dfs(root.right,start)
        
        if root.val == start:
            self.max_dist = max(left,right)
            depth -= 1
        elif left >= 0 and right >= 0:
            #just the max depth calcualtion
            depth = max(left,right) + 1
        else:
            #when root is not start node, but subttree contains startnode, this is just  the ditance from the root to start
            dist = abs(left) + abs(right) #dist from this node to the root
            self.max_dist = max(self.max_dist,dist)
            #depth for this start node
            depth = min(left,right) - 1
        
        return depth
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def amountOfTime(self, root: Optional[TreeNode], start: int) -> int:
        '''
        dfs with two returns
        1. whther we find number or not, and max distance to that node
        '''
        self.ans = 0
        def dfs(node,start):
            if not node:
                return [False,0] #[found start, max depth ot that cell]
            
            found_left, depth_left = dfs(node.left,start)
            found_right, depth_right = dfs(node.right,start)
            
            if node.val == start:
                curr_max = max(depth_left,depth_right)
                self.ans = max(self.ans,curr_max)
                return [True,0] #pass max disatnce to that cell, weve found it, so return true and we are 0 away
            
            #if we found it in the left subtree
            if found_left:
                #we need left and right + 1
                self.ans = max(self.ans, depth_left + depth_right + 1)
                #pass down going left
                return [True,depth_left+1]
            
            #same thing with right
            elif found_right:
                self.ans = max(self.ans, depth_left + depth_right + 1)
                return [True, depth_right+1]
            
            #just find max depth
            return [False, max(depth_left,depth_right) + 1]
        
        
    
        dfs(root,start)
        return self.ans
    
####################################################################
# 1026. Maximum Difference Between Node and Ancestor (REVISTED)
# 11JAN24
#####################################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        '''
        looks like binry tree week but with multiple returns and global update answer
        pass min and max in both directions and calculate??
        '''
        self.ans = float('-inf')
        
        
        def dfs(node,curr_min,curr_max):
            if not node:
                return
            min_compare = abs(curr_min - node.val)
            max_compare = abs(curr_max - node.val)
            self.ans = max(self.ans, min_compare, max_compare)
            dfs(node.left, min(curr_min, node.val), max(curr_max,node.val))
            dfs(node.right, min(curr_min, node.val), max(curr_max,node.val))
            
            
        dfs(root,root.val,root.val)
        return self.ans
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        '''
        make this dp
        
        '''
        if not root:
            return 0
        
        def dp(node,curr_max,curr_min):
            if not node:
                return curr_max - curr_min
            left = dp(node.left, min(curr_min, node.val), max(curr_max,node.val))
            right = dp(node.right, min(curr_min, node.val), max(curr_max,node.val))
            return max(left,right)
        
        
        return dp(root,root.val,root.val)
    
################################################
# 676. Implement Magic Dictionary
# 11JAN24
################################################
#brute force works
class MagicDictionary:

    def __init__(self):
        '''
        hash everything ans check
        '''
    def buildDict(self, dictionary: List[str]) -> None:
        self.D = set(dictionary)
        

    def search(self, searchWord: str) -> bool:
        #for each letter change it to all lower cases and check
        #make surew we change!
        N = len(searchWord)
        for i in range(N):
            for j in range(26):
                sub_char = chr(ord('a') + j)
                temp = searchWord[:i] + sub_char + searchWord[i+1:]
                if sub_char != searchWord[i] and temp in self.D:
                    return True
        
        return False


# Your MagicDictionary object will be instantiated and called as such:
# obj = MagicDictionary()
# obj.buildDict(dictionary)
# param_2 = obj.search(searchWord)
    
#now do Trie solution
class Node:
    def __init__(self):
        self.children = defaultdict()
        self.end = False
        
class Trie:
    def __init__(self):
        self.root = Node()
    
    def insert(self,word):
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = Node()
            curr = curr.children[ch]
        
        #mark
        curr.end = True
        
    #check function, but keep track of replacements, we cant replace more the
    #problem is that if we have a replacement, when should we replace
    def check(self,word,k,node):
        if k < 0:
            return False
        if not word:
            if k == 0 and node.end == True:
                return True
            return False
        
        for char, next_node in node.children.items():
            if char == word[0]:
                if self.check(word[1:],k,next_node):
                    return True
                else:
                    if self.check(word[1:],k - 1,next_node):
                        return True
        
        return False
                

class MagicDictionary:

    def __init__(self):
        self.trie = Trie()
        

    def buildDict(self, dictionary: List[str]) -> None:
        for word in dictionary:
            self.trie.insert(word)
        
        

    def search(self, searchWord: str) -> bool:
        return self.trie.check(searchWord,1,self.trie.root)
        
# Your MagicDictionary object will be instantiated and called as such:
# obj = MagicDictionary()
# obj.buildDict(dictionary)
# param_2 = obj.search(searchWord)
    
#another way
class TrieNode(object):
    #don't forget about nesting clases
    def __init__(self):
        self.isAend = False
        self.contains = defaultdict(TrieNode)
        
class MagicDictionary(object):
    
    def __init__(self):
        self.root = TrieNode()
        
    def addWord(self, word):
        r = self.root
        for ch in word:
            if ch not in r.contains:
                r.contains[ch] = TrieNode()
            r = r.contains[ch]
        r.isAend = True

    def findWord(self, remain, r, word):
        if not word:
            return True if remain == 0 and r.isAend else False
        for key,next_ in r.contains.items():
            if key == word[0]:
                if self.findWord(remain, next_, word[1:]):
                    return True
            elif remain == 1:
                if self.findWord(0, next_, word[1:]):
                    return True
        return False
    
    def buildDict(self, dict):
        for word in dict:
            self.addWord(word)
    
    def search(self, word):
        return self.findWord(1, self.root, word)

# Your MagicDictionary object will be instantiated and called as such:
# obj = MagicDictionary()
# obj.buildDict(dictionary)
# param_2 = obj.search(searchWord)
    
#############################################################
# 1347. Minimum Number of Steps to Make Two Strings Anagram
# 13JAN24
#############################################################
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        '''
        strings are same length
        strings are anagrams if sorted orders equal or freq chars are equal
        count up freq chars in s
        then for through t, and increment the counts of there aren't any
        find all the matching chars with counts in each
        ans is just the difference
        '''
        N = len(s)
        counts_s = Counter(s)
        counts_t = Counter(t)
        matched = counts_s & counts_t
        
        return N - sum(matched.values())
    
class Solution:
    def minSteps(self, s: str, t: str) -> int:
        '''
        we can just record the differnces of characters on the fly
        '''
        counts = [0]*26
        for s_char,t_char in zip(s,t):
            counts[ord(s_char) - ord('a')] += 1
            counts[ord(t_char) - ord('a')] -= 1
        
        
        ans = 0
        for c in counts:
            if c >= 0:
                ans += c
        
        return ans
    
##############################################
# 681. Next Closest Time
# 15JAN24
###############################################
#meh too many edge cases
class Solution:
    def nextClosestTime(self, time: str) -> str:
        '''
        input is small enough to try all enumerations
        generate all enumerations
        modul 24*60, convert each time to seconds after 0000
        its next time, ans can be itself
        
        '''
        digits = list(time.replace(":",""))
        possible = []
        
        def convertSeconds(time):
            h = time // 60
            m = time % 60
            if h < 10:
                h = "0"+str(h)
            else:
                h = str(h)
            return h+":"+str(m)
        
        def backtrack(i,digits,path):
            if i == len(digits):
                #validate
                temp = "".join(path)
                if 0 <= int(temp[:2]) <= 24 and 0 <= int(temp[2:]) <= 59:
                    possible.append(temp)
                return
            
            for d in digits:
                path.append(d)
                backtrack(i+1,digits,path)
                path.pop()
        
        backtrack(0,digits,[])
        #convert each to seconds
        seconds = []
        for t in possible:
            s = int(t[:2])*60 + int(t[2:])
            seconds.append(s)

        print(seconds)
        #sort
        seconds.sort()
        h,m = time.split(":")
        search_time = int(h)*60 + int(m)
        idx = bisect.bisect_left(seconds,search_time)
        if idx == len(seconds)-1:
            return convertSeconds(seconds[0])
        
        return convertSeconds(seconds[idx+1])
        
#step by step
class Solution:
    def nextClosestTime(self, time: str) -> str:
        '''
        move one minute a time until we get to a time that has all the digits
        '''
        digits = set(list(time.replace(":","")))
        h,m = time.split(":")
        curr_time = int(h)*60 + int(m)
        
        while True:
            curr_time = (curr_time + 1) % (24*60)
            #check that curr_time uses all the digits
            curr_h,curr_m = divmod(curr_time, 60)
            test = '%02d:%02d' % divmod(curr_time, 60)
            test_digits = list(test.replace(":",""))
            if all([d in digits for d in test_digits]):
                return test
        
        return time
    
#cool way with min comparison of tuples size 2
class Solution:
    def nextClosestTime(self, time):
        """Time O(1) Space O(1)
        Easy to read and understand although not
        necessarily fastest-executing algorithm."""
        
        solutions = []

        # it is easiest to think of all possible times in terms of minues. Also, 24 * 60 means
        # we will never go over the total possible number of minutes (or hours, when we convert
        # back in next step) in a day.
        for i in range(24 * 60):

            # using divmod returns (number of whole divisions, remainder). This is a helpful
            # trick for making sure that minutes are valid. Hours will not exceed 24 due to
            # above loop.
            for t in ['%02d:%02d' % divmod(i, 60)]:

                # in python set intersection/union/subset can be done with standard boolean
                # operators. This is saying that all elements of t must be a subset or equivalent
                # set to all elements in time
                if set(t) <= set(time):

                    # creates tuple where first element is bool. min() will then select the lowest False
                    # element (meaning the new time is in the same day as the input time), and only if that
                    # doesn't exist will it select the lowest True tuple (aka closest time is next day).
                    solutions.append((t<=time, t))

        # elements in solutions will look like
        # (True, '11:12'),
        # (False, '11:21'), etc
        return min(solutions)[1]

###############################################
# 2484. Count Palindromic Subsequences
# 15JAN24
###############################################
#MLE, need to use cache decorator
class Solution:
    def countPalindromes(self, s: str) -> int:
        '''
        we can use dp with 4 states, first digit pair and second digit pair
        then its just knap sack
        '''
        N = len(s)
        memo = {}
        
        def dp(ind,first,second,i):
            mod = 10**9 + 7
            #found a subsequence
            if i == 5:
                return 1
            if ind >= N:
                return 0
            #retreive
            if (ind,first,second,i) in memo:
                return memo[(ind,first,second,i)]
            #dont take
            ways = dp(ind+1,first,second,i)
            #first digit in length 5 subsequence
            if i == 0:
                ways += dp(ind+1,int(s[ind]),second,i+1)
                ways %= mod
            #second digit in sequence
            elif i == 1:
                ways += dp(ind+1,first,int(s[ind]),i+1)
                ways %= mod
            #third digit, this is the center, it doesn't need to match
            elif i == 2:
                ways += dp(ind+1,first,second,i+1)
                ways %= mod
            #fourth digit, must match second
            elif i == 3:
                if int(s[ind]) == second:
                    ways += dp(ind+1,first,second,i+1)
                    ways %= mod
            #fifth digit, must macht first
            elif i == 4:
                if int(s[ind]) == first:
                    ways += dp(ind + 1, first,second,i+1)
                    ways %= mod
            
            memo[(ind,first,second,i)] = ways % mod
            return ways % mod
            
        
        return dp(0,0,0,0)
    
class Solution:
    def countPalindromes(self, s: str) -> int:
        '''
        we can use dp with 4 states, first digit pair and second digit pair
        then its just knap sack
        '''
        N = len(s)

        @cache
        def dp(ind,first,second,i):
            mod = 10**9 + 7
            #found a subsequence
            if i == 5:
                return 1
            if ind >= N:
                return 0

            #dont take
            ways = dp(ind+1,first,second,i)
            #first digit in length 5 subsequence
            if i == 0:
                ways += dp(ind+1,int(s[ind]),second,i+1)
                ways %= mod
            #second digit in sequence
            elif i == 1:
                ways += dp(ind+1,first,int(s[ind]),i+1)
                ways %= mod
            #third digit, this is the center, it doesn't need to match
            elif i == 2:
                ways += dp(ind+1,first,second,i+1)
                ways %= mod
            #fourth digit, must match second
            elif i == 3:
                if int(s[ind]) == second:
                    ways += dp(ind+1,first,second,i+1)
                    ways %= mod
            #fifth digit, must macht first
            elif i == 4:
                if int(s[ind]) == first:
                    ways += dp(ind + 1, first,second,i+1)
                    ways %= mod
            

            return ways % mod
            
        
        return dp(0,0,0,0)
                
class Solution:
    def countPalindromes(self, s: str) -> int:
        '''
        we can use dp with 4 states, first digit pair and second digit pair
        then its just knap sack
        '''
        N = len(s)
        mod = 10 ** 9 + 7
        @cache
        #states
        #i is position in string, curr is current 2 length subsequence, mid_after means we are looking for the part after the center
        def solve(i, curr, mid_after):
            #we have finished looking for the right part
            if mid_after and curr == '':  
                return 1
            #no matches
            if i >= N:  
                return 0
            #dont include this char
            res = solve(i + 1, curr, mid_after)  # skip current char
            #trying to build up left part
            if len(curr) < 2 and not mid_after: # adding chars to pref
                return res + solve(i + 1, curr + s[i], 0)
            
            #build up left part, now at center
            if len(curr) == 2 and not mid_after: 
                return res + solve(i + 1, curr, 1)
            
            #we have a mid after, check we can extend
            if mid_after and curr[-1] == s[i]: 
                #match and look at net suffix, i.e the last char in the left side
                return res + solve(i + 1, curr[:-1], mid_after)
            return res 
              
        ans = solve(0, '', 0) % mod
        return ans
    
#counting
class Solution:
    def countPalindromes(self, s: str) -> int:
        '''
        this is more of a counting problem
        for some middle at index i, count number of pairs (2 length subsequences) to the left of i and to the right of i
        intuition:
            count number of ordered pairs that mirror before and after each middle index (2,len(s) -2)
        
        * if length of string is less than 5 cant be done
        * get number of ordered pairs before and after
        * count number of matching pais before and after each index
        count pairs before and after can be done in N^2 time
        we need an O(N) to count pairs!
        '''
        if len(s) < 5:
            return 0
        
        #Returns the running pair counts for each index
        def get_pairs(s):
            seen_cnt = {str(num):0 for num in range(10)}
            seen_cnt[s[0]] = 1 #We have seen the first character since we start the loop at 1
            pair_cnts = defaultdict(int)
            res = [defaultdict(int)] #Filler empty dict (index = 0 / end index)
            
            #Getting running pairs
            for idx in range(1, len(s) - 1):
                res.append(pair_cnts.copy()) #Append running pair counts
                for num in seen_cnt.keys():
                    pair_cnts[(num, s[idx])] += seen_cnt[num] #this is a runnig sum of pairs
                seen_cnt[s[idx]] += 1
                
            #the res array is read like this, for some index i, we can retrieve pair counts by accessing res[i]
            #its going to be pairs to the left os this index
            #becareful when going in reverse
            #Filler empty dict (index = 0 / end index)
            res.append(defaultdict(int)) 
            #we need to pad the arrays with None
            return res
        
        ans = 0 
        mod = 10**9 + 7
        #get pre and post pair counts
        pre = get_pairs(s)
        post = get_pairs(s[::-1])[::-1]
        
        for i in range(2,len(s) - 2):
            #check all pairs before i and pairs after i
            #number of palindromic subsequences is just the product
            for k,v in pre[i].items():
                if k in post[i]:
                    ans += post[i][k]*v
                    ans %= mod
        
        return ans
    
###############################################
# 380. Insert Delete GetRandom O(1) (REVISTED)
# 16JAN24
###############################################
class RandomizedSet:
    '''
    for the get random method, just use random choice module
    use hashmap to store index to value pair
    and use array to keep values
    
    
    for the delete operation, we just move the last element to the deleted index
    we can get the last element using an array
    '''
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dict = {}
        self.list = []

        
    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True
        

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dict:
            # move the last element to the place idx of the element to delete
            last_element, idx = self.list[-1], self.dict[val]
            self.list[idx], self.dict[last_element] = last_element, idx
            # delete the last element
            self.list.pop()
            del self.dict[val]
            return True
        return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return choice(self.list)
    
#################################################
# 686. Repeated String Match
# 16JAN24
#################################################
#try all k
class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        '''
        need to repeat a some number of times so that b is a substring of a
        if i have string a, concatenating it k times, i can have only so many substrings
        
        '''
        temp = ""
        k = 0
        while len(temp) < len(b):
            temp += a
            k += 1
            if b in temp:
                return k

        #one more
        temp += a
        k += 1
        if b in temp:
            return k
        
        return -1

class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        '''
        need to repeat a some number of times so that b is a substring of a
        if i have string a, concatenating it k times, i can have only so many substrings
        
        abcabc
        
        i concat a twice, and for this new string, i can hve any prefix repeated some number of times
        if we were to wrtie S = A+A+A ... (i.e concatenate some number of times)
        we only need to check all suffixes, as long as S is long enough to contain B and S has period at most len(a)
        
        suppose q is the least number for which len(b) <= len(A*q)
        we only need to check whther B is a subtring of A*q or A*(q+1)
        if we try k < q, then B has a larger length than A*q and therefore can't be a substring
        if we do one more (i.e q+1), then the concatenated a will be long enough to contain b
        rather:
            check that a[i:i+len(b)] = B for i in range(len(A))
            
        https://leetcode.com/problems/repeated-string-match/discuss/108090/Intuitive-Python-2-liner
        better explanation
        
        Let n be the answer, the minimum number of times A has to be repeated.

        For B to be inside A, A has to be repeated sufficient times such that it is at least as long as B (or one more), hence we can conclude that the theoretical lower bound for the answer would be length of B / length of A.

        Let x be the theoretical lower bound, which is ceil(len(B)/len(A)).

        The answer n can only be x or x + 1 (in the case where len(B) is a multiple of len(A) like in A = "abcd" and B = "cdabcdab") and not more. Because if B is already in A * n, B is definitely in A * (n + 1).

        Hence we only need to check whether B in A * x or B in A * (x + 1), and if both are not possible return -1.
        
        #note ceiling can also be written this way
        -(-len(B) // len(A)) # Equal to ceil(len(b) / len(a)
        '''
        q = math.ceil(len(b) / len(a))
        for i in [0,1]:
            a_concat = a*(q+i)
            if b in a_concat:
                return q + i
        
        return -1
    
#rabin karp
#same intition as above, but insteaf of checking in string, we use rabin karp
class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        '''
        insteaf of checking (b in a_concat) we can use rabin karp, reviet on rabin karp
        for some string s, s[i] is some integer ascii rode, then for some prime p:
        hash(S) = \sum{0 <= i < len(s)}p^{i}*S[i]
        rather
        hash[S[1:] + x], where x is the next character we want to add
        '''
        def RabinKarp(s, t):
            #we want to find a in t
            p, m = 31, 10**9 + 9
            S, T = len(s), len(t)
            
            #calculate powers mod m for each position starting after 1
            power = [1] * max(S, T)
            for i in range(1, len(power)):
                power[i] = (power[i - 1] * p) % m
            
            #hasing the text string
            #this is hash for prefix string
            H = [0] * (T + 1)
            for i in range(T):
                H[i + 1] = (H[i] + (ord(t[i]) - ord('a') + 1) * power[i]) % m
            
            #get hash for pattern s
            HS = 0
            for i in range(S):
                HS = (HS + (ord(s[i]) - ord('a') + 1) * power[i]) % m

            #sliding window comparisons, slide s through t, and get hash for each window
            currHS = 0
            for i in range(T - S + 1):
                currHS = (H[i + S] - H[i] + m) % m
                if currHS == HS * power[i] % m:
                    return True

            return False
        
        q = math.ceil(len(b) / len(a))
        for i in [0,1]:
            a_concat = a*(q+i)
            if RabinKarp(b,a_concat):
                return q + i
        
        return -1
        
##############################
# 1417. Reformat The String
# 17JAN24
##############################
class Solution:
    def reformat(self, s: str) -> str:
        '''
        intelligently build the string
        take chars and digits and place alternatingly
        '''
        letters = []
        digits = []
        for ch in s:
            #is digit
            if '0' <= ch <= '9':
                digits.append(ch)
            else:
                letters.append(ch)
        
        #the can be unequal, but cant be more than 1
        if abs(len(letters) - len(digits)) > 1:
            return ""
        
        ans = ""
        first = None
        second = None
        if len(letters) >= len(digits):
            first = letters
            second = digits
        else:
            first = digits
            second = letters
            
        for i in range(max(len(first),len(second))):
            ans += first[i] if i < len(first) else ""
            ans += second[i] if i < len(second) else ""
        
        return ans

class Solution:    
    def reformat(self, s: str) -> str:
        a, b = [], []
        for c in s:
            if 'a' <= c <= 'z':
                a.append(c)
            else:
                b.append(c)
        #a get priority over b
        if len(a) < len(b):
            a, b = b, a
        if len(a) - len(b) >= 2:
            return ''
        ans = ''
        for i in range(len(a)+len(b)):
            #even is first
            if i % 2 == 0:
                ans += a[i//2]
            else:
                ans += b[i//2]
        return ans

#####################################
# 1427. Perform String Shifts
# 18JAN24
######################################
#brute force
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        '''
        rebuild sting for each operation
        '''
        for d,k in shift:
            k = k % len(s)
            #left shift
            if d == 0:
                s = s[k:] + s[:k]
            else:
                s = s[len(s) - k:] + s[:len(s) - k]
        
        return s

#using q
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        '''
        we can use deque to simulate shifting
        shifts are cumulative, a shift of left by k is cancelled by shift of right by k
        accumlate shifts and apply the final shift
        i can apply the reaminder of abs(shifts_accum)
        '''
        shifts_accum = 0
        for d,k in shift:
            #if left shift
            if d == 0:
                shifts_accum -= k
            else:
                shifts_accum += k
        
        sign = -1 if shifts_accum < 0 else 1
        size = abs(shifts_accum) % len(s)
        if sign == -1:
            #shift left
            q = deque(list(s))
            while size:
                size -= 1
                q.append(q.popleft())
            
            return "".join(q)
        
        else:
            q = deque(list(s))
            while size:
                size -= 1
                q.appendleft(q.pop())
            
            return "".join(q)
        
#using index after getting next shift
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        '''
        rebuild sting for each operation
        '''
        shifts_accum = 0
        for d,k in shift:
            #if left shift
            if d == 0:
                shifts_accum -= k
            else:
                shifts_accum += k
        
        sign = -1 if shifts_accum < 0 else 1
        size = abs(shifts_accum) % len(s)
        
        if sign == -1:
            s = s[size:] + s[:size]
        else:
            s = s[len(s) - size:] + s[:len(s) - size]
            #could also so
            #s = s[-size:] + s[:-size]
        
        return s
    
#treat right shifts as negative left shifts
#apply final shift as left shift
#dont forget to mod to make it positive again
class Solution:
    def stringShift(self, s: str, shift: List[List[int]]) -> str:
        '''
        just use left shifts
        '''
        left_shifts = 0
        for d,k in shift:
            #if left shift
            if d == 1:
                left_shifts -= k
            else:
                left_shifts += k
        
        left_shifts %= len(s)
        
        s = s[left_shifts:] + s[:left_shifts]
        
        return s
    
############################################
# 931. Minimum Falling Path Sum (REVISTED)
# 19JAN24
#############################################
#top down
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        '''
        let dp(i,j) be the minimum falling path starting with (i,j)
        if we are beyond last row, return 0
        dont forget to check DOWN!!! mofo!
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        memo = {}

        def dp(i,j):
            if i >= rows:
                return 0
            #out of bounds
            if j < 0 or j >= cols:
                return float('inf')
            if (i,j) in memo:
                return memo[(i,j)]
            #then we just check down and left, and down and right
            left = dp(i+1,j-1)
            right = dp(i+1,j+1)
            down = dp(i+1,j)
            ans = min(left,right,down) + matrix[i][j]
            memo[(i,j)] = ans
            return ans

        ans = float('inf')
        for col in range(cols):
            ans = min(ans, dp(0,col))
        
        return ans
    
#bottom up
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        '''
        for bottom up, dont do it in place, just make a copy
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        dp = [[0]*(cols+1) for _ in range(rows+1)]

        #fill start from just before last row
        for row in range(rows-1,-1,-1):
            for col in range(cols):
                #check if first col
                if col == 0:
                    dp[row][col] = matrix[row][col] + min(dp[row+1][col], dp[row+1][col+1])
                #last col
                elif col == cols - 1:
                    dp[row][col] = matrix[row][col] + min(dp[row+1][col], dp[row+1][col-1])
                else:
                    dp[row][col] = matrix[row][col] + min(dp[row+1][col], dp[row+1][col-1], dp[row+1][col+1])

        print(dp)
        return min(dp[0][:-1])
    
############################################
# 907. Sum of Subarray Minimums (REVISTED)
# 20JAN24
############################################
#monostack
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        '''
        find given range for which arr[i] is smallest
        then find number of subarrays using arr[i]
        contribution is (count sub_arrays)*arr[i]
        to find which range element is smallest, we can use monostack
        when we come to a point where we didn't maintain an icnreasing stack, it means the num at the top of stack was
        the smallest we ecounter so far
        so we need to find the contributomp
        while findinf the boundary elements for a range, we look for elements that are strictly less than then current elemetn on the left
        to decide the right bonudary, we look for elements <= to the current eleemetn
        previous smaller index is whwat is at the top
        curr index is top
        right is the current index i, rememebr are mainting range where the current elementat the top is smallest
        '''
        mod = 10**9 + 7
        stack = []
        sum_of_mins = 0
        
        for i in range(len(arr)): 
            '''
            think of [1,2,3,4,1]
            4 is smalest in the range [4] (left index would be 3, curr index would be 4, right would be 5)
            3 is smallestt i the range [3,4] (left index owuld be 2, curr index owuld be 3, right is still 5)
            2 is smallest in tange [2,3,4]
            1 is smallest in the range [1,2,3,4]
            one we're don, we go back to add 1
            '''
            while stack and arr[stack[-1]] >= arr[i]:
                
                 # Notice the sign ">=", This ensures that no contribution
                # is counted twice. right_boundary takes equal or smaller 
                # elements into account while left_boundary takes only the
                # strictly smaller elements into account
                
                #do this every time, since the array is montonic increasing, we have found a place where nums[i] can be a minimum
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                right = i
                
                #contribution to min sum for the current element at the top
                count = (mid - left)*(right - mid)
                sum_of_mins += count*arr[mid]
                sum_of_mins %= mod
            
            stack.append(i)
        #stack left over
        while stack:
            mid = stack.pop()
            left = -1 if not stack else stack[-1]
            right = i
            
            #contribution to min sum but with the end of the array
            count = (mid - left)*(len(arr) - mid)
            sum_of_mins += count*arr[mid]

            
        return sum_of_mins % mod

########################################
# 2104. Sum of Subarray Ranges
# 20JAN24
#########################################
#brute force passes
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        '''
        find sum of all subarray ranges
        similar to 907. Sum of Subarray Minimums
        build up icnreasing monostack, we dont need to keep indices, just the actual nums
        what doest it mean to maintain an increasing subarray?
            [1,2,3]
        
        bruter force should pass
        '''
        ans = 0
        N = len(nums)
        for left in range(N):
            curr_min,curr_max = float('inf'),float('-inf')
            for right in range(left,N):
                curr_min = min(curr_min, nums[right])
                curr_max = max(curr_max, nums[right])
                ans += curr_max - curr_min
        
        return ans
    
#mono stack is hard
#basically just two monostack passes
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        '''
        find sum of all subarray ranges
        similar to 907. Sum of Subarray Minimums
        build up icnreasing monostack, we dont need to keep indices, just the actual nums
        what doest it mean to maintain an increasing subarray?
            [1,2,3]
        
        bruter force should pass
        if we knew the number of subarrys with some minimum k
        and if we knew the number of subarray with some maximum l
        then the answer would just num_subarrays*l - num_subarrays*k
        meaning we can calculate partial sums seperately
        so we just re-use solution from 907 twice!
        '''
        #find sum of mins
        stack = []
        sum_mins = 0
        N = len(nums)
        for i in range(N):
            #montonic increasing
            while stack and nums[stack[-1]] >= nums[i]:
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                right = i
                count = (mid - left)*(right - mid)
                sum_mins += count*nums[mid]
            stack.append(i)
    
        #remainder of stack
        while stack:
            mid = stack.pop()
            left = -1 if not stack else stack[-1]
            count = (mid - left)*(N - mid)
            sum_mins += count*nums[mid]

        #now find sum of maxs
        sum_maxs = 0
        stack = []
        for i in range(N):
            #montonic decreasing this time
            while stack and nums[stack[-1]] <= nums[i]:
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                right = i
                count = (mid - left)*(right - mid)
                sum_maxs += count*nums[mid]
            stack.append(i)
    
        #remainder of stack
        while stack:
            mid = stack.pop()
            left = -1 if not stack else stack[-1]
            count = (mid - left)*(N - mid)
            sum_maxs += count*nums[mid]


        return sum_maxs - sum_mins
    
#insteaf of adding in extra while loop for the rest of stack
#go to N+1, which forces use to pop from the remaainder of the stack
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        '''
        we can skip the extra while loop, 
        go to N+1, and if we are at the end, we HAVE to clear it
        '''
        #find sum of mins
        stack = []
        sum_mins = 0
        N = len(nums)
        for i in range(N+1):
            #montonic increasing
            while stack and (i == N or nums[stack[-1]] >= nums[i]):
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                right = i
                count = (mid - left)*(right - mid)
                sum_mins += count*nums[mid]
            stack.append(i)
    

        #now find sum of maxs
        sum_maxs = 0
        stack = []
        for i in range(N+1):
            #montonic decreasing
            while stack and (i == N or nums[stack[-1]] <= nums[i]):
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                right = i
                count = (mid - left)*(right - mid)
                sum_maxs += count*nums[mid]
            stack.append(i)
    

        return sum_maxs - sum_mins
    
###########################################
# 687. Longest Univalue Path
# 20JAN24
###########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestUnivaluePath(self, root: Optional[TreeNode]) -> int:
        """
        issue is that it might not just be root to leaf path
        if im at node, and value to a child is same value, the path can be exteneded by one
        if both at the same, i can same them up
        really its just the number of nodes in a subtree if all the nodes are the same, this would be brute force

        say we are at a node, if we have our functino that returns number of node vals == to node.val (as left and right)
        then the answer is just left + right
        really its just the diamtere of the tree max(dp(node.left) + dp(node.right)) + 1
        if its not the same, we just return 0
        """
        self.ans = 0

        def dp(node, parent):
            if not node:
                return 0
            left = dp(node.left, node.val)
            right = dp(node.right, node.val)
            self.ans = max(self.ans, left + right)
            if node.val == parent:
                return max(left, right) + 1
            else:
                return 0

        dp(root, -1)
        return self.ans

#########################################
# 720. Longest Word in Dictionary
# 21NOV24
#########################################
class Solution:
    def longestWord(self, words: List[str]) -> str:
        '''
        hash all prefixes of each word
        then check for each word if we can build it up
        as we check validate size and lexographical order
        '''
        prefs = set()
        words = set(words)
        for w in words:
            for i in range(len(w)):
                p = w[:i+1]
                if p in words:
                    prefs.add(p)
        
        
        #now check each word
        def check(word,prefs):
            for i in range(len(word)):
                p = word[:i+1]
                if p not in prefs:
                    return False
            return True
        
        ans = ""
        for w in words:
            if check(w,prefs):
                #min on negative length and then lexo on sting
                ans = min([ans,w], key = lambda x : (-len(x),x))
        
        return ans

#trie solution
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False
    
    def insert(self,word):
        curr = self
        for ch in word:
            curr = curr.children[ch] #willd default to a new TrieNode
        
        curr.is_word = True
    
    def contains(self,word):
        curr = self
        for ch in word:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        
        return curr.is_word == True
    
class Solution:
    def longestWord(self, words: List[str]) -> str:
        '''
        using Trie, dont foreget recursive implementaion
        '''
        trie = TrieNode()
        for word in words:
            trie.insert(word)
            
        #now check each word
        def check(word,trie):
            for i in range(len(word)):
                p = word[:i+1]
                if not trie.contains(p):
                    return False
            return True
        
        ans = ""
        for w in words:
            if check(w,trie):
                #min on negative length and then lexo on sting
                ans = min([ans,w], key = lambda x : (-len(x),x))
        
        return ans
            
class Solution:
    def longestWord(self, words: List[str]) -> str:
        '''
        sort words alphabetically so shorter words alays come first
        initally seen is just the null string, so the first prefix of string length 1 is just empty
        try building up the word one prefix at a time if we can
        '''
        words.sort()
        seen = set([""])
        ans = ""
        for word in words:
            if word[:-1] in seen:
                seen.add(word)
                if len(word) > len(ans):
                    ans = word
        return ans
    
#############################################
# 722. Remove Comments
# 22JAN24
##############################################
class Solution:
    def removeComments(self, source: List[str]) -> List[str]:
        '''
        from hints we need to parse line by line
        rules:
            * If we start a block comment and we aren't in a block, then we will skip over the next two characters and change our state to be in a block.
            * If we end a block comment and we are in a block, then we will skip over the next two characters and change our state to be not in a block.
            * If we start a line comment and we aren't in a block, then we will ignore the rest of the line.
            * If we aren't in a block comment (and it wasn't the start of a comment), we will record the character we are at.
            * At the end of each line, if we aren't in a block, we will record the line.
        '''
        in_block = False
        ans = []
        for line in source:
            i = 0
            if not in_block:
                new_line = []
                #idea is the we can span multiple lines until we hit a block comment
            while i < len(line):
                #not in block and start block
                if line[i:i+2] == '/*' and not in_block:
                    in_block = True
                    i += 1
                #in block and end of block
                elif line[i:i+2] == '*/' and in_block:
                    in_block = False
                    i += 1
                #single line comment
                elif not in_block and line[i:i+2] == "//":
                    break #skip line
                #no block, no comment, we need this line
                elif not in_block:
                    new_line.append(line[i])
                i += 1
            
            if new_line and not in_block:
                ans.append("".join(new_line))
        
        return ans
    
#another solution
#https://leetcode.com/problems/remove-comments/discuss/109210/Simple-Python-one-pass-with-clear-inline-explanation!!!
class Solution(object):
    def removeComments(self, source):
        """
        :type source: List[str]
        :rtype: List[str]
        """
        res, buffer, block_comment_open = [], '', False
        for line in source:
            i = 0
            while i < len(line):
                char = line[i]
                # "//" -> Line comment.
                if char == '/' and (i + 1) < len(line) and line[i + 1] == '/' and not block_comment_open:
                    i = len(line) # Advance pointer to end of current line.
                # "/*" -> Start of block comment.
                elif char == '/' and (i + 1) < len(line) and line[i + 1] == '*' and not block_comment_open:
                    block_comment_open = True
                    i += 1
                # "*/" -> End of block comment.
                elif char == '*' and (i + 1) < len(line) and line[i + 1] == '/' and block_comment_open:
                    block_comment_open = False
                    i += 1
                # Normal character. Append to buffer if not in block comment.
                elif not block_comment_open:
                    buffer += char
                i += 1
            if buffer and not block_comment_open:
                res.append(buffer)
                buffer = ''
        return res
    
##################################################################################
# 1239. Maximum Length of a Concatenated String with Unique Characters (REVISTED)
# 23JAN24
##################################################################################
#jeeezus
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        states are index into arr and mask for chars used in the subsequence
        fiter our seqs that aren't uniqe
        '''

        
        #filter unique sequences
        f = lambda x: len(x) == len(set(x))
        arr = list(filter(f,arr))
        N = len(arr)
        memo = {}
        
        def dp(i,mask):
            if i >= N:
                return 0
            if (i,mask) in memo:
                return memo[(i,mask)]
            
            ans = dp(i+1,mask)
            for j in range(i,N):
                #take j only if we the chars in this current mask haven't been used
                next_seq = arr[j]
                next_mask = 0
                for ch in next_seq:
                    pos = ord(ch) - ord('a')
                    next_mask |= (1 << pos)
                #check positions all 26
                is_uniq = True
                for i in range(26):
                    check_mask = 1 << i
                    if mask & next_mask & check_mask > 0:
                        is_uniq = False
                        break
                #check no overlapping
                if is_uniq:
                    ans = max(ans, len(next_seq) + dp(j+1,mask | next_mask))
            
            memo[(i,mask)] = ans
            return ans
                              
        
        ans = 0
        for i in range(N):
            #validate
            if len(arr[i]) == len(set(arr[i])):
                ans = max(ans, dp(i,0))
        
        return ans

#cheese way
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        
        N = len(arr)
        memo = {}
        
        def dp(i,path):
            if len(path) != len(set(path)):
                return 0
            if (i,path) in memo:
                return memo[(i,path)]
            
            ans = len(path)
            for j in range(i,N):
                ans = max(ans, dp(j+1, path + arr[j]))
            
            memo[(i,path)] = ans
            return ans
        
        return dp(0,"")

#cheese iteraive way
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        build all possible solutions iteratively
        start with empty string and concat only if it results in one that doesn't have repeated chars
        '''
        possible = [""]
        ans = 0
        for word in arr:
            N = len(possible)
            for i in range(N):
                next_seq = possible[i] + word
                if len(next_seq) != len(set(next_seq)):
                    continue
                possible.append(next_seq)
                ans = max(ans, len(next_seq))
        
        return ans
        
#iterative with mask
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        in the bit mask, include taken chars at bit positions as well as the length
        length will be capped at 26 anway
        read_lenghth:
            length = mask >> 26
        read_mask:
            (1 << 26) - 1, mask of 1s in each of the 26 spots
            bitset = mask & ((1 << 26) - 1)
        
        store seens results at set of masks intead of their string representations
        '''
        seen = set([0])
        
        def addWord(word,seen):
            char_mask = 0
            for ch in word:
                mask = 1 << ord(ch) - ord('a')
                #if weve seen char return 0
                if char_mask & mask:
                    return 0
                #add to mask
                char_mask += mask
            
            #check to see if wen've seen this mask in seen
            if char_mask + (len(word) << 26) in seen:
                return 0
            
            #check all results in seen
            ans = 0
            for res in list(seen):
                if res & char_mask:
                    continue
                
                #new entry
                new_len = (res >> 26) + len(word)
                new_mask = char_mask + res & ((1 << 26) - 1)
                
                seen.add((new_len << 26) + new_mask)
                ans = max(ans, new_len)
            
            return ans
        
        ans = 0
        for word in arr:
            ans = max(ans,addWord(word,seen))
        
        return ans
            
#backtracking
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        backtracking
        '''
        #convert arr to character bitset
        arr_mask = set()
        for w in arr:
            if self.getMask(w):
                arr_mask.add(self.getMask(w))
        #convert back to list
        arr_mask = list(arr_mask)
        return self.backtrack(arr_mask, 0,0,0)
        
        
    
    def backtrack(self, arr_mask, pos, res_chars, res_len):
        best = res_len
        for i in range(pos,len(arr_mask)):
            #get mask and length
            new_chars = arr_mask[i] & ((1 << 26) - 1)
            new_len = arr_mask[i] >> 26
            
            #intserction
            if new_chars & res_chars:
                continue
            
            #add in
            res_chars += new_chars
            res_len += new_len
            best = max(best, self.backtrack(arr_mask, i+1, res_chars,res_len))
            
            #backtrack
            res_chars -= new_chars
            res_len -= new_len
        
        return best
        
    def getMask(self,word):
        mask = 0
        for ch in word:
            pos = 1 << ord(ch) - ord('a')
            if mask & pos:
                return None
            mask += pos

        return mask + (len(word) << 26)
    
#############################################################
# 1457. Pseudo-Palindromic Paths in a Binary Tree (REVISTED)
# 24JAN24
##############################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        '''
        node values in path form a palindrome if at most one digit has odd frequence (parity)
        only need to check root to leaves
        keep counter object, then validate at the leaves
        issue is copying count object i have to copy before!
        '''
        self.ans = 0
        
        def dfs(node,counts):
            if not node:
                return
            #leaf node
            new_counts = counts.copy()
            new_counts[node.val] += 1
            if not node.left and not node.right:
                odd = 0
                for k,v in new_counts.items():
                    if v % 2 == 1:
                        odd += 1
                if odd <= 1:
                    self.ans += 1
                return
            dfs(node.left,new_counts)
            dfs(node.right,new_counts)
        
        
        dfs(root,Counter())
        return self.ans
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        '''
        just mark whether or not weve seen this node in the path
        even occurencces will set bit position to 0
        odd occruence will set position to 1
        essentially keep a mask, and in the mask only one position can have a set bit
        rather power of two
        '''
        ans = [0]
        
        def dfs(node,mask):
            if not node:
                return
            #flip bit
            mask ^= (1 << node.val)
            if not node.left and not node.right:
                if mask & (mask -1) == 0:
                    ans[0] += 1
            
            dfs(node.left,mask)
            dfs(node.right,mask)
        
        dfs(root,0)
        return ans[0]


##################################################
# 2573. Find the String with LCP
# 23JAN24
##################################################
class Solution:
    def findTheString(self, lcp: List[List[int]]) -> str:
        '''
        lcp[i][j] gives length of longest common prefix between substrings  word[i,n-1] amd word[j,n-1]
        hint 1; use lcp array to determine which groups of elements must be equal
        hint 2: match smallest letter to the group that contains the smallest unassigned idnex
        hint 3: build lcp matrix of resulting string then check if == to lcp
        
        intutions:
            if lcp[i][j] > 0, then A[i] == A[j]
            if lcp[i][j] == 0, then A[i] != A[j], where A is the canddiate string
        transistions
            lcp[i][j] = lcp[i+1][j+1] + 1
        
        build candidate dp lcp, and compare to lcp
        return ans otherwise impossible
        
        we try to build smallest string the follows lcp array, start with a
        if lcp[i][j] == 0, and the char do not equal, the we need to use the next smallest
        if char is beyong 0, then it can't be donw
        '''
        N = len(lcp)
        cand = [0]*N #convert back to stirng using ord('a') offset
        
        for i in range(N):
            for j in range(N):
                if (lcp[i][j] == 0 and cand[i] == cand[j]):
                    cand[j] += 1
        
        #check if we are grater than 26
        for num in cand:
            if num > 26:
                return ""
        #build candidate lcp array using transition
        cand_lcp = [[0]*(N+1) for _ in range(N+1)]
        for i in range(N-1,-1,-1):
            for j in range(N-1,-1,-1):
                if cand[i] == cand[j]:
                    cand_lcp[i][j] = 1 + cand_lcp[i+1][j+1]
        
        #compare
        for i in range(N):
            for j in range(N):
                if lcp[i][j] != cand_lcp[i][j]:
                    return ""
        
        ans = ""
        for num in cand:
            ans += chr(ord('a') + num)
        
        return ans
    
#################################################
# 737. Sentence Similarity II
# 24JAN24
################################################
#dfs
class Solution:
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        '''
        make mapp of similar pairs and check all are similar
        if there's a connected path between words they are similar need to use union find and see if they are in the same group
        or we can traverse do both
        '''
        if len(sentence1) != len(sentence2):
            return False
        
        graph = defaultdict(set)
        for u,v in similarPairs:
            graph[u].add(v)
            graph[v].add(u)
            
            
        #function to check if similar, basically a path from start to end
        def isSimilar(start,end,seen,graph):
            if start == end:
                return True
            seen.add(start)
            for neigh in graph[start]:
                if neigh not in seen:
                    if isSimilar(neigh,end,seen,graph):
                        return True
            
            return False
        
        for u,v in zip(sentence1,sentence2):
            seen = set()
            if not isSimilar(u,v,seen,graph):
                return False
        
        return True

#bfs
class Solution:
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:

        if len(sentence1) != len(sentence2):
            return False
        
        graph = defaultdict(set)
        for u,v in similarPairs:
            graph[u].add(v)
            graph[v].add(u)
            
            
        #function to check if similar, basically a path from start to end
        def isSimilar(start,end,seen,graph):
            q = deque([start])
            while q:
                start = q.popleft()
                if start == end:
                    return True
                seen.add(start)
                for neigh in graph[start]:
                    if neigh not in seen:
                        q.append(neigh)

            return False
        
        for u,v in zip(sentence1,sentence2):
            seen = set()
            if not isSimilar(u,v,seen,graph):
                return False
        
        return True
            
#union find
class DSU:
    def __init__(self,words):
        self.parent = {w:w for w in words}
        self.size = {w:1 for w in words}
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        #no union
        if x_par == y_par:
            return
        
        #swap groups
        if self.size[x_par] >= self.size[y_par]:
            self.parent[y_par] = x_par
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
        else:
            self.parent[x_par] = y_par
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
        
class Solution:
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        '''
        new way to do union find, insteaf of array we need words pointing to themselves
        once we get union find, we just check that each word pair points to the same parent
        need all the workds
        '''
        if len(sentence1) != len(sentence2):
            return False
        
        words = set(chain(*similarPairs))
        for u,v in zip(sentence1,sentence2):
            words.add(u)
            words.add(v)
            
        dsu = DSU(words)
        for u,v in similarPairs:
            dsu.union(u,v)
        
        for u,v in zip(sentence1,sentence2):
            if dsu.find(u) != dsu.find(v):
                return False
        return True

##################################
# 738. Monotone Increasing Digits
# 25JAN24
##################################
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        '''
        n is too big to try all permutations of some number b
        build digit one by one, adding the largest possible that makes is less than or equal to n
        i can use stack to build the digit
        intuition:
            if a number is not montone increasing, we will need to convert to 9 as much as we can
            but such that we dont make the number more than n
        
        55627 -> 55599
        55427 -> 49999
        99996 -> 89999
        100 -> 99
        
        travesre digits right to left and whenever a digit to the left is amller than its right, we decrease 1, then change to 9 to rhr gith
        '''
        #single digit case
        if n < 10:
            return n
        
        digits = [int(ch) for ch in str(n)]
        N = len(digits)
        
        for i in range(N-1,0,-1):
            if digits[i] < digits[i-1]:
                digits[i-1] -= 1
                for j in range(i,N):
                    digits[j] = 9
        
        ans = 0
        for num in digits:
            ans = ans*10 + num
        
        return ans
    
class Solution:
    def monotoneIncreasingDigits(self, n: int) -> int:
        '''
        instead of changing all numbers to the right of i to 9, we mark the last i
        '''
        #single digit case
        if n < 10:
            return n
        
        digits = [int(ch) for ch in str(n)]
        N = len(digits)
        last = n
        
        for i in range(N-1,0,-1):
            if digits[i] < digits[i-1]:
                digits[i-1] -= 1
                last = i
        
        for j in range(last,N):
            digits[j] = 9
        
        ans = 0
        for num in digits:
            ans = ans*10 + num
        
        return ans
    
########################################
# 576. Out of Boundary Paths (REVISTED)
# 26JAN24
#########################################
#top down
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        states should be (i,j,moves) 
        then we check all moves, out of bounds return 1, otherwise add them up

        '''
        memo = {}
        
        def dp(i,j,moves):
            #out of bounds 
            mod = 10**9 + 7
            if i < 0 or i >= m or j < 0 or j >= n:
                if moves >= 0:
                    return 1
                return 0
            
            #careful with the moves condition
            if moves < 0:
                return 0
            
            if (i,j, moves) in memo:
                return memo[(i,j,moves)]
            
            ans = 0
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                ans += dp(i + dx, j + dy,moves-1) % mod
            
            ans %= mod
            memo[(i,j,moves)] = ans
            return ans
        
        
        return dp(startRow,startColumn,maxMove)
    
#bottom up
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        states should be (i,j,moves) 
        then we check all moves, out of bounds return 1, otherwise add them up

        '''
        mod = 10**9 + 7
        dp = [[[0]*(maxMove + 1) for _ in range(n)] for _ in range(m)]
        
        #one away from base case
        for moves in range(1,maxMove+1):
            for i in range(m):
                for j in range(n):
                    ans = 0
                    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        neigh_x = i + dx
                        neigh_y = j + dy
                        #in bounds
                        if 0 <= neigh_x < m and 0 <= neigh_y < n:
                            ans += dp[neigh_x][neigh_y][moves-1]
                            dp[i][j][moves] = ans % mod
                        else:
                            ans += 1
                            ans %= mod
                    dp[i][j][moves] = ans % mod
                    
        
        return dp[startRow][startColumn][maxMove] % mod
    
#################################################################
# 2120. Execution of All Suffix Instructions Staying in a Grid
# 26JAN24
#################################################################
class Solution:
    def executeInstructions(self, n: int, startPos: List[int], s: str) -> List[int]:
        '''
        extension of 576. Out of Boundary Paths 
        for each suffix i, do the path, and if we end up walking off the path, we coudn't do that last instsruction
        careful when doing down, which mean index goes up
        when going up, index goes down!
        '''
        dirrs = {'U':[-1,0],
                'D':[1,0],
                'L':[0,-1],
                'R':[0,1]}
        
        def move(startPos,suffix):
            x,y = startPos
            k = 0
            for ch in suffix:
                x += dirrs[ch][0]
                y += dirrs[ch][1]
                if not (0 <= x < n) or not (0 <= y < n):
                    return k
                k += 1
            return k
        
        start_x,start_y = startPos
        N = len(s)
        
        ans = []
        for i in range(N):
            ans.append(move(startPos,s[i:]))
        
        return ans
    

##############################################
# 629. K Inverse Pairs Array (REVISTED)
# 28JAN24
##############################################
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        '''
        for an inverse pair
        i < j and both i and j are indexable and nums[i] > nums[j]
        for some number n, from 1 to n, return number of different arrays where there  are exactly k inverse pairs
        if k = 1
            only increasing array
        needs to be O(n*k)
        say we have the previous (n-1) array and we are placing n, we can place n somewhere in the array that it generates additional inverse pairs
        call array, a_0 the beigning array like [1,2,3,4]
        if we want to have k inverser pairs, we need to shift some number in a_0 to the left k times
        for more than on number we can combine shifts such that there sum == k
        example
        [1,2,4,3] 1 inver pairs, shifted 4 to the left 1 time
        [2,4,1,3], 3 inverse pairs, shifted 2 to the left 1 time, and shifted 4 to the left 1 time, ans is 1 + 2 = 3
        add ways from n-1 and k - shift
        dp(n,k) =
            sum ways for dp(n-1,k-1) + dp(n-1,k-2) ... dp(0,0) 
            but we cannot go more than n shifts and we cannot for more than k inverse pairs
        if we alread have the desired arrangement n with exactly k shifts, we jusst add the next and to the end
        '''
        memo = {}
        mod = 10**9 + 7
        
        def dp(n,k):
            if n == 0:
                return 0
            if k == 0:
                return 1
            if (n,k) in memo:
                return memo[(n,k)]
            ways = 0
            position = 0
            #upper limit on position, k - positions < 0, no arrangment exists with engtiave inverse pairs
            
            while position < n and position <= k:
                ways += dp(n-1,k-position) % mod
                ways %= mod
                position += 1
            
            ways %= mod
            memo[(n,k)] = ways
            return ways
        
        
        return dp(n,k)
    
#########################################################
# 1567. Maximum Length of Subarray With Positive Product
# 29JAN24
#########################################################
class Solution:
    def getMaxLen(self, nums: List[int]) -> int:
        '''
        if subarray contains even number of 0's product will be positive
        any subarray with a zero in it will be 0
        split array into subarray by zeros
        '''
        subarrays = []
        curr_sub = []
        for num in nums:
            if num == 0:
                subarrays.append(curr_sub)
                curr_sub = []
            else:
                curr_sub.append(num)
        
        if curr_sub:
            subarrays.append(curr_sub)
        
        ans = 0
        #test each subarray
        for arr in subarrays:
            #check count negatives
            count_negatives = 0
            for num in arr:
                if num < 0:
                    count_negatives += 1
            #if even
            if count_negatives % 2 == 0:
                ans = max(ans, len(arr))
            else:
                #find first negative from front
                i = 0
                while i < len(arr) and arr[i] > 0:
                    i += 1
                ans = max(ans, len(arr) - (i+1))
                
                #find first negative from end
                i = len(arr) - 1
                while i >= 0 and arr[i] > 0:
                    i -= 1
                
                ans = max(ans, i)
        return ans

######################################################
# 1265. Print Immutable Linked List in Reverse
# 31JAN24
######################################################
class Solution:
    def printLinkedListInReverse(self, head: 'ImmutableListNode') -> None:
        def recurse(curr):
            if (curr == None):
                return
            recurse(curr.getNext())
            curr.printValue()
        
        recurse(head)

class Solution:
    def printLinkedListInReverse(self, head: 'ImmutableListNode') -> None:
        '''
        #just need to traverse, we can use recursion
        curr = head
        while curr:
            curr.printValue()
            curr = curr.getNext()

        '''
        stack = []
        curr = head
        
        while curr:
            stack.append(curr)
            curr = curr.getNext()
        
        while stack:
            curr = stack.pop()
            curr.printValue()

#Square root decomposition


#######################################################
# 1552. Magnetic Force Between Two Balls
# 31JAN24
#######################################################
class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        '''
        balls need to be placed in baskets, we want the minimum magnetic force between any two balls to be maximum
        if we have more ballls than we do baskets, we can't do it
        brute force woul be to try all pairings
        binary search for workable solution
        need funcions thats runs in O(N) time
        if we can do with some some number k, check if we can do k+1
        we will always have more than 2 baskets and more than 2 balls
        will we always need to place a ball on the first basket?
        there will be multiple solutions for some min force
        min force would be 1, what about the max force?
        low = 1
        high = max(position) - min(position)
        '''
        #sort buckets
        position.sort()
        def minForce(cand, arr, balls):
            #neeed to maintain cand distance between prev basket and currbasket
            prev = arr[0]
            balls -= 1
            for next_ in arr[1:]:
                #if we can place
                if abs(next_ - prev) >= cand:
                    balls -= 1
                    prev = next_
                if balls == 0:
                    return True
            
            return balls == 0
        
        
        low = 1
        high = max(position) - min(position)
        while low <= high:
            mid = low + (high - low) // 2
            if minForce(mid,position,m):
                low = mid + 1
            else:
                high = mid - 1
        
        return low - 1
    
###############################################################
# 2556. Disconnect Path in a Binary Matrix by at Most One Flip
# 29JAN24
##############################################################
#doenst work, cant just count paths because the pathts intserct
class Solution:
    def isPossibleToCutPath(self, grid: List[List[int]]) -> bool:
        '''
        count number of paths from (0,0) to (m-1,n-1)
        return true of there is only one path
        cant flip cells (0,0) or (rows-1,cols-1)
        can't just count paths
        '''
        rows = len(grid)
        cols = len(grid[0])
        memo = {}
        
        def dp(i,j):
            if (i,j) == (rows-1,cols-1):
                return 1
            if (i,j) in memo:
                return memo[(i,j)]
            down = 0
            if i + 1 < rows and grid[i+1][j] == 1:
                down = dp(i+1,j)
            
            right = 0
            if j + 1 < cols and grid[i][j+1] == 1:
                right = dp(i,j+1)
            
            ans = right + down
            memo[(i,j)] = ans
            return ans
        
        
        paths = dp(0,0)
        if paths == 0:
            return True
        return paths == 1

class Solution:
    def isPossibleToCutPath(self, grid):
        '''
        see if we can reach the ending cell from the start cell
        we need to do thiw twice, remember we need to make it disconnected using at most one flip, (we dont have to flip, i.e they may already be disconnected)
        omg we can flip a 0 to 1 or a 1 to 0, but we cannot fli[]
        
        '''
        m, n = len(grid), len(grid[0])
    
        def dfs(i,j):
            #if we can reach (m-1,n-1) from (0,0)
            if i == m-1 and j == n-1:
                return True

            #invalid move
            if (i == m or j == n or grid[i][j] == 0):
                return False
            
            #not the first cell,  (0,0) will change to 0 for anything else, and only if its a 1
            if (i+j): 
                grid[i][j] = 0

            return dfs(i+1,j) or dfs(i,j+1)

        #if we can do it, return true
        if dfs(0,0) == False:
            return True
        for r in grid:
            print(r)

        #try one more time
        if dfs(0,0) == False:
            return True
        
        for r in grid:
            print(r)
        
        #otherwise we can't do it
        return False

class Solution:
    def isPossibleToCutPath(self, grid: List[List[int]]) -> bool:
        '''
        start at (0,1), to the right and (1,0)  down
        we search for two non-intersecting paths from these points
        for the first search we go right first, and only go down if we cannot reach the last cell by going right
        for the second search, we flip the direction
        that ay the first path is closes to the top-right borders of the grid, and the second path to the bottom left border
        if we find both paths, then we cannot make the matrix disconnected
            when we traverse a path, we change the values in the matrix to zero, so we canot reuse them a second time
        since we flip values for the first path, we do not need to flip directions for the second one
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        def dfs(i,j):
            #out of bounds or zero, false
            if (i == rows) or (j == cols) or (grid[i][j] == 0):
                return False
            #reached end
            if (i,j) == (rows -1, cols - 1):
                return True
            #mark
            if (i != 0) or (j != 0):
                grid[i][j] = 0
            
            return dfs(i+1,j) or dfs(i,j+1)
        
        return not (dfs(0,0) and dfs(0,0))