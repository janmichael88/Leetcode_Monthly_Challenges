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
        def solve(i, curr, mid_after):
            if mid_after and curr == '':  
                return 1
            if i >= N:  
                return 0	
            res = solve(i + 1, curr, mid_after)  # skip current char
            if len(curr) < 2 and not mid_after: # adding chars to pref
                return res + solve(i + 1, curr + s[i], 0)
            
            if len(curr) == 2 and not mid_after: 
                return res + solve(i + 1, curr, 1)
            
            if mid_after and curr[-1] == s[i]: 
                return res + solve(i + 1, curr[:-1], mid_after)
            return res 
              
        ans = solve(0, '', 0) % mod
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