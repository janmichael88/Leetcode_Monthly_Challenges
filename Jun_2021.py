###########################
#   Max Area of Island
###########################
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        '''
        dfs, but we can modify the board in place by flipping 1 back to 0
        '''
        self.max_area = 0
        rows = len(grid)
        cols = len(grid[0])
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(row,col):
            #out of bounds and not 1, just return 0
            if row < 0 or col <  0 or row >= rows or col >= cols or grid[row][col] == 0:
                return 0
            #mark
            grid[row][col] = 0
            
            #image we are at a cell surrouned by ones, the area is just the sum of all these plus the sqaure itselt
            area = 0
            for dx,dy in dirs:
                area += dfs(row+dx,col+dy)
            return area + 1
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    self.max_area = max(self.max_area,dfs(i,j))
        
        return self.max_area

#iterative with stack
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        '''
        now make this iterative
        '''
        max_area = 0
        rows = len(grid)
        cols = len(grid[0])
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    stack = [(i,j)]
                    grid[i][j] = 0
                    area = 0
                    while stack:
                        x,y = stack.pop()
                        area += 1
                        for dx,dy in dirs:
                            new_x = x + dx
                            new_y = y + dy
                            if 0 <= new_x < rows and 0 <= new_y < cols:
                                if grid[new_x][new_y] == 1:
                                    stack.append((new_x,new_y))
                                    #mark
                                    grid[new_x][new_y] = 0
                    max_area = max(max_area,area)
        return max_area
        
#################################
# Interleaving String
##################################
#nope, its dp/recursion
#fuck the first week
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        interleaving does not need to be alternating
        so long as in the new string s1[i] is before s1[i+1] and t1[i] is before t1[i+1]
        what if i kept three pointers, and adavance the pointers only when then match
        then return whether or not we moved through s3
        '''
        s1_ptr = 0
        s2_ptr = 0
        s3_ptr = 0
        
        while s1_ptr < len(s1) and s2_ptr < len(s2):
            if s1[s1_ptr] == s3[s3_ptr]:
                s1_ptr += 1
                s3_ptr += 1
            elif s2[s2_ptr] == s3[s3_ptr]:
                s2_ptr += 1
                s3_ptr += 1
        #if i finished both strings
        if s1_ptr == len(s1)-1 and s2_ptr == len(s2)-1:
            return s3_ptr == len(s3) -1
        
        #now use the rest of either string
        if s1_ptr < len(s1):
            while s1_ptr < len(s1):
                if s1[s1_ptr] == s3[s3_ptr]:
                    s1_ptr += 1
                    s3_ptr += 1
        
        if s2_ptr < len(s2):
            while s1_ptr < len(s1):
                if s2[s2_ptr] == s3[s3_ptr]:
                    s2_ptr += 1
                    s3_ptr += 1
        
        return s3_ptr == len(s3) -1

#brute force recursion
#turns out the brute force was also a recursive solution
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        brute force would be to generate all interlavings and check against s3
        to generate all interleavings we use recurion
        '''
        #edge case
        if len(s1) + len(s2) != len(s3):
            return False
        
        def rec(s1,i,s2,j,res,s3):
            #check only after making string
            if i == len(s1) and j == len(s2) and res == s3:
                return True
            #start building if we can use s1
            first = None
            second = None
            if i < len(s1):
                first = rec(s1,i+1,s2,j,res+s1[i],s3)
            #start building s2
            if j < len(s2):
                second = rec(s1,i,s2,j+1,res+s2[j],s3)
            
            return first or second
        
        return rec(s1,0,s2,0,"",s3)

#recursive with memo
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        brute force recursion had overlapping subproblems
        we can use three pointers and a memo and store whether the current portions of strings have already been evaluated
        if when we include a new char it matched the new char at s2 we include it inthe processed string
        recursion ends when either of the two string have been full procseed
        '''
        #edge case
        if len(s1) + len(s2) != len(s3):
            return False
        
        memo = {}
        
        def recurse(s1,i,s2,j,s3,k,current):
            #matching 
            if current == s3 and i == len(s1) and j == len(s2):
                return True
            if current and current[-1] != s3[k-1]:
                return False
            first = None
            second = None
            if (i,j) in memo:
                return memo[(i,j)]
            if i < len(s1):
                first = recurse(s1,i+1,s2,j,s3,k+1,current+s1[i])
            if j < len(s2):
                second = recurse(s1,i,s2,j+1,s3,k+1,current+s2[j])
            
            res = first or second
            memo[(i,j)] = res
            return res
        
        return recurse(s1,0,s2,0,s3,0,"")

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        if len(s1) + len(s2) != len(s3):
            return False
        
        memo ={}
        
        def recurse(i,j,k):
            #memo retreive
            if (i,j,k) in memo:
                return memo[(i,j,k)]
            #base case
            if i == len(s1) and j == len(s2) and k == len(s3):
                return True
            first = None
            second = None
            if i < len(s1) and s1[i] == s3[k]:
                first = recurse(i+1,j,k+1)
            if j < len(s2) and s2[j] == s3[k]:
                second = recurse(i,j+1,k+1)
            memo[(i,j,k)] = first or second
            return first or second
        
        return recurse(0,0,0)

#dp
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        we can use dp,
        frame the question if we are at an index k in s3, is this prefix an interleaving of s1 and s2 up to i and j respectively
        dp[i][j] solves the problem, if it is possible to obtain a substring if lengths i+j+2 which is a prefix up to s3 at k
        to fill in the dp array:
            1. if when adding the character i or j, if it doesn't match the kth index of 3, that cell is False
            2. if it does match:
                if s1 at i matches, we need to keep x at the last position in the resultant interleaved string
            to get the current index at s3 from i and j, we check at i+j+1
        '''
        if len(s1) + len(s2) != len(s3):
            return False
        
        #dp array is len(s1) + 1 by len(s2) + 1
        dp = [[0]*(len(s2)+1) for _ in range(len(s1)+1)]
        for i in range(len(s1)+1):
            for j in range(len(s2)+1):
                #empty strings is trivially true
                if i == 0 and j == 0:
                    dp[i][j] =  True
                #s1 is empty and taking only from s2, should be true if the next char matches i+j+1 and previous was true
                elif i == 0:
                    dp[i][j] = dp[i][j-1] and s2[j-1] == s3[i+j-1]
                #s2 is empty and taking only from s2, 
                elif j == 0:
                    dp[i][j] = dp[i-1][j] and s1[i-1] == s3[i+j-1]
                #main case. check either previos i or j matches 
                else:
                    dp[i][j] = (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1])
        
        return dp[len(s1)][len(s2)]

#dp 1d array
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        1 d array dp
        '''
        if len(s1) + len(s2) != len(s3):
            return False
        
        #dp array is len(s1) + 1 by len(s2) + 1
        dp = [0]*(len(s2)+1)
        for i in range(len(s1)+1):
            for j in range(len(s2)+1):
                #empty strings is trivially true
                if i == 0 and j == 0:
                    dp[j] =  True
                #s1 is empty and taking only from s2, should be true if the next char matches i+j+1 and previous was true
                elif i == 0:
                    dp[j] = dp[j-1] and s2[j-1] == s3[i+j-1]
                #s2 is empty and taking only from s2, 
                elif j == 0:
                    dp[j] = dp[j] and s1[i-1] == s3[i+j-1]
                #main case. check either previos i or j matches 
                else:
                    dp[j] = (dp[j-1] and s2[j-1] == s3[i+j-1]) or (dp[j] and s1[i-1] == s3[i+j-1])
        
        return dp[-1]

##################################
#  Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts
##################################
class Solution:
    def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
        '''
        warm up, brute force would be to examine all gaps
        and then get the areas
        '''
        #sort arrays
        horizontalCuts.sort()
        verticalCuts.sort()
        
        #find the gaps, don't fort to include gaps from zero
        #prepend teh cuts with zeros
        horizontalCuts = [0] + horizontalCuts
        for i in range(len(horizontalCuts)-1):
            horizontalCuts[i] = horizontalCuts[i+1] - horizontalCuts[i]
        horizontalCuts[-1] = h - horizontalCuts[-1]

        #vertical cuts
        verticalCuts = [0] + verticalCuts
        for i in range(len(verticalCuts)-1):
            verticalCuts[i] = verticalCuts[i+1] - verticalCuts[i]
        verticalCuts[-1] = w - verticalCuts[-1]
        
        return max(verticalCuts)*max(horizontalCuts)  % (10**9 +7)

#############
# Paint House
#############
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        '''
        brute force would be to enumerate all paths (all arrays of length n where each element
        contains (0,1,2), remove ones that do not have adjacent 0s 1s or 2s
        then get the costs for each, and return the min
        time complexity owuld be:
            generating k 3s .... O(3^k = 2^{n-k})
        
        we need to use a recursion
        imagine the recursion tree, and at each node, we add a house to our path
        at the root, we are free to choose any house,
        but after depth 1, we only have 2 choices for each house
        total permutations is 3*(2^(n-1))
        now think, if we are at the nth house, and all n-1 houses are valid, how should we choose the next house?
            well we want the house that is not of the same color of the n-1 th house, and also is the minimum of our two choices
            if we start at the n-1th house, we chose the next house that is the minimum
            then we go to the n-2th house after making the desicions, oly after we have chosen the nth house
            *be sure to look at the animation for this one
        algo:
            we invoke three recursive calls starting with the first decision of painting the first house red,blue,green
            we then return the minimum of these three
        '''
        memo = {}
        def paint(n,color):
            if (n,color) in memo:
                return memo[(n,color)]
            total_cost = costs[n][color]
            if n == len(costs)-1: 
                pass
            #red color at nth house, the next house can only be blue or green
            elif color == 0:
                total_cost += min(paint(n+1,1),paint(n+1,2))
            #now do the same with the other two colors
            elif color == 1:
                total_cost += min(paint(n+1,0),paint(n+1,2))
            else:
                total_cost += min(paint(n+1,0),paint(n+1,1))
            memo[(n,color)] = total_cost
            return total_cost
        if len(costs) == 0:
            return 0
        
        red = paint(0,0)
        green = paint(0,1)
        blue = paint(0,2)
        
        return min(red,green,blue)

#dp
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        '''
        note: if we connected the recursion tree, i.e making a directed graph,
        this would be similar to dynamic programming
        dp approach, start backwards and keep taking the minimum
        we first define the subproblem to be calculting the total cost for a particular house position and color
        we can use the costs array to calculate the cost of painting the house that color and the min cost to paint all houses after it
        note we are writing to the inout array,
        just make sure how to use another dp array, in 1d if we did not want to write
        algo:
            start at the last house,
            don't touch the last row
            and add the minimum to each cell
            
        '''
        if len(costs) == 0:
            return 0
        for n in range(len(costs)-2,-1,-1):
            #total cost of painting the nth house red
            costs[n][0] += min(costs[n+1][1],costs[n+1][2])
            #n-1th green
            costs[n][1] += min(costs[n+1][0],costs[n+1][2])
            #n-1th blue
            costs[n][2] += min(costs[n+1][0],costs[n+1][1])
            
        #return min of first row
        return min(costs[0])    

#dp space optimzied
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        '''
        note: if we connected the recursion tree, i.e making a directed graph,
        this would be similar to dynamic programming
        dp approach, start backwards and keep taking the minimum
        we first define the subproblem to be calculting the total cost for a particular house position and color
        we can use the costs array to calculate the cost of painting the house that color and the min cost to paint all houses after it
        note we are writing to the inout array,
        just make sure how to use another dp array, in 1d if we did not want to write
        algo:
            start at the last house,
            don't touch the last row
            and add the minimum to each cell
            
        '''
        if len(costs) == 0: return 0

        previous_row = costs[-1]
        for n in reversed(range(len(costs) - 1)):

            current_row = copy.deepcopy(costs[n])
            # Total cost of painting nth house red?
            current_row[0] += min(previous_row[1], previous_row[2])
            # Total cost of painting nth house green?
            current_row[1] += min(previous_row[0], previous_row[2])
            # Total cost of painting nth house blue?
            current_row[2] += min(previous_row[0], previous_row[1])
            previous_row = current_row

        return min(previous_row)

#############################
# Open The Lock
#############################
#idk why this getgs hung up
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        '''
        this is just bfs
        we can think of the state of the number as a node, and we can go to any from the state in one turn
        if when we visit a node, we hit a dead end return -1
        otherwise we need to keep going until target
        the trick is to how to generate the adjaceny matrix
        
        termination conditions
            if ive hit that target
            or if  ive hit all the dead ends
        '''
        
        seen = set()
        deadends = set(deadends)
        
        def get_neighbors(node):
            neighs = []
            for i in range(4):
                for dirr in (1,-1):
                    list_node = list(node)
                    digit = int(list_node[i])
                    digit += dirr
                    digit %=10
                    #make  new digit
                    list_node[i] = str(digit)
                    neighs.append("".join(list_node))
            return neighs
        #add the first one
        
        seen.add('0000')
        q = deque([('0000',0)]) #node, moves
        while q:
            node,moves = q.popleft()
            if node == target:
                return moves
            #check deadends
            if node in deadends:
                continue
            #generate neighbors
            neighs = get_neighbors(node)
            print(node)
            for n in neighs:
                if n not in seen:
                    seen.add(node)
                    q.append((n,moves+1))
        return -1

#here's the actual way, but mine works just fine
#ist the list operator, generated neighbors during bfs
class Solution(object):
    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """
        '''
        from the hint, this is a graph traversal problem, in fact find the shortest path problem
        there are 100000 different states i.e nodes,
        each node conected by an edge which is the turn of a dial
        i can't go into the state if its in dead ends
        dfs or bfs, BFS obvs! would find the shortes path...dfs would enumerate all possoble paths
        
        to solve the shortest path problem we need to use bfs
        useing a q and a seen set
        
        we define a neighbors functions, for each position in the lock (0,1,2,3) for each of the turns d (-1,1) we determine the vlaues of the lock after the ith wheel has been turned in the d direction
        
        edge cases:
            make sure we do no traverse and edge that leads to dead end, and we must also add, '0000' in the beginning
        '''
        def neighbors(node):
            #get nodes differing by one turn on the dial
            for i in range(4):
                x = int(node[i])
                for d in (-1,1):
                    y = (x + d) % 10
                    yield node[:i] + str(y) + node[i+1:]
                    
        dead = set(deadends)
        seen = {'0000'}
        q = deque([('0000',0)])
        
        while q:
            node,depth = q.popleft()
            if node == target:
                return depth
            if node in dead:
                continue
            for n in neighbors(node):
                if n not in seen:
                    seen.add(n)
                    q.append((n,depth+1))
                    
        return -1

#####################
# Maximum Performance of a Team
#####################
#not quite...
class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        '''
        we need to pick k engineers to get max performance
        permformance is  = \sum_{i in k}^{k} speed[i]*efficieny[i]
        brute force would obvie be to enumerate all combinatinos of k enigneers and find max effecient
        hint says sort by effce
        then we need to build a team, such that the next engineer has >= curr eff and >= speed 
        '''
        #pair efficiencies with (eff,idx) then sort
        effs = [(eff,idx) for idx,eff in enumerate(efficiency)]
        effs.sort(reverse=True)

        #start with the first eng and see if speed and eff >= curr, if so add
        maxPerformance = 0
        
        currSpeed, currEff = 1,1
        
        #ptr into eff array
        ptr = 0
        while k > 0 and ptr < len(effs):
            #examine curr enginerr
            curr_eng = effs[ptr]
            #gett eff and speed
            eng_eff = curr_eng[0]
            eng_speed = speed[curr_eng[1]]
            if eng_eff >= currEff and eng_speed >= currSpeed:
                maxPerformance += eng_eff*eng_speed
                #update
                currSpeed = min(currSpeed,eng_speed)
                currEff = min(currEff,eng_eff)
                k -= 1
            ptr += 1
        
        return maxPerformance
            
class Solution:
    def maxPerformance(self, n: int, speed: List[int], efficiency: List[int], k: int) -> int:
        '''
        so greedy won't work here, i can keep a heap of fixed size, k
        then when a new engineer with high eff*speed is bigger than the minium, replace that enigeer
        we would want a min heap to have acess to the least efficient engineer
        greedy algorithm:
            make locally optimal choices, solve the problem globalls
            when we examine a candidate, we take the current candidates eff as the min
            then when examining other cands, we need to amke sure they have >= min eff
        
        algo:
            key: for a fixed memebr, the next member should have higher eff and higher speed
            sort effs in descending order
            initally we keep pushing on to the heap (the current speed)
            increment speed sum, and update perf
            when we have too many memebrs on the team we start popping off reduing speed sum
        take away, we initally sort by effs, but the heap sorts by speeds
         woops, peformance is the summ of speeds times min efficients
        '''
        engineers = [(eff,sp) for eff,sp in zip(efficiency,speed)]
        #sort by effs,deault is first element
        engineers.sort(reverse=True)
        
        speed_sum = 0
        max_perf = 0
        
        team = [] #heap holding current members of team
        for eff,sp in engineers:
            #maintin team size
            if len(team) >= k:
                #remove the slowest memebr
                speed_sum -= heappop(team)
            #other wise to team
            heappush(team,sp)
            
            #update
            speed_sum += sp
            max_perf = max(max_perf,speed_sum*eff)
            
        return max_perf % (10**9 + 7)

#########################
#  128. Longest Consecutive Sequence
#########################
#close one
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        '''
        best way is to sort and check consective diffs are increasin then add to  count
        [1,2,3,4,100,200]
        '''
        nums.sort()
        size = 0
        N = len(nums)
        
        size = 0
        left,right = 0,0
        while right + 1 < N:
            if nums[right+1] - nums[right] == 1:
                right += 1
                if nums[right] != nums[left]:
                    size = max(size,right-left+1)
            else:
                left = right
                right +=1
                
        
        return size

#count streaks
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums: 
            return 0
        
        nums.sort()
        
        longest_streak = 1
        curr_streak = 1
        
        #check streaks
        for i in range(1,len(nums)):
            if nums[i] != nums[i-1]:
                if nums[i] == nums[i-1] + 1: 
                    #part of streak
                    curr_streak += 1
                else:
                    #not  part of  streak reset
                    longest_streak = max(longest_streak,curr_streak)
                    curr_streak = 1
        return max(longest_streak,curr_streak)

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        #we can use a hash to check if nums + 1 is seen
        #we then keep building up the lonesr consecutive sequence
        longest = 0
        nums = set(nums)
        
        for num in nums:
            #prev not seen
            if num-1 not in nums:
                curr_num = num
                curr_streak = 1
                
                while curr_num + 1 in nums:
                    curr_streak += 1
                    curr_num += 1
                #update 
                longest = max(longest,curr_streak)
        return longest

#############################
#  Min Cost Climbing Stairs
##############################
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        '''
        we can define  the  recyr relation,sub problem,
        as the min cost  to reach the ith step as
        minimumCost[i] = min(minimumCost[i - 1] + cost[i - 1], minimumCost[i - 2] + cost[i - 2])
        we then return the answer foor the final step
        dp array of length + 1, we want too reach the top floor

        '''
        dp = [0]*(len(cost)+1)
        
        for i in range(2,len(dp)):
            #start froom secoond step since base cases for steps 1 and have already been defined
            one_step = dp[i-1]+cost[i-1]
            two_step = dp[i-2]+cost[i-2]
            dp[i] = min(one_step,two_step)
        
        return dp[-1]

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        '''
        once we have defined the recurence relation we can use recursion
        we ca

        '''
        memo = {}
        def recurse(step):
            #base case
            if step <= 1:
                return 0
            if step in memo:
                return memo[step]
            one_step = cost[step-1] + recurse(step-1)
            two_step = cost[step-2] + recurse(step-2)
            memo[step] = min(one_step,two_step)
            return memo[step]
    
        #call on final step
        return recurse(len(cost))

####################
# Jump Game VI
####################
#using dq
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        '''
        wuthout using heap
        maintain monotonically decresing q, so that largest score is always on top
        '''
        N = len(nums)
        score = [0]*N
        score[0] = nums[0]
        
        q = deque()
        #load first 
        q.append(0)
        for i in range(1,N):
            while q and q[0] < i - k:
                q.popleft()
            score[i] = score[q[0]] + nums[i]
            #pop smaller value
            while q and score[i] >= score[q[-1]]: 
                q.pop()
            q.append(i)
        
        return score[-1]

class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        '''
        dp[i] answers the questiono what is the max score at the end startubg from index i
        score[i] = max(score[i-k], ..., score[i-1]) + nums[i]
        but we take k time  trying to figure oout the  max
        too much, but this problemb is similar to the monootonic que probkem sliding window  max
        For example, if k == 3 and [score[i-3], score[i-2], score[i-1]] = [2, 4, 10], their maximum is score[i-1] = 10
        maintain q as monotonically decreasing and  max is always at top
        algo:
        Step 1: Initialize a dp array score, where score[i] represents the max score starting at nums[0] and ending at nums[i].

Step 2: Initialize a max-heap priority_queue.

Step 3: Iterate over nums. For each element nums[i]:

If the index of top of priority_queue is less than i-k, pop the top. Repeat.
Update score[i] to the sum of corresponding score of the index of top of priority_queue and nums[i] (i.e., score[priorityQueue.peek()[1]] + nums[i]).
Push pair (score[i], i) into priority_queue.
Step 4: Return the last element of score.
        '''
        N = len(nums)
        score = [0]*N
        score[0] = nums[0]
        max_heap = []
        #push first number  in
        heappush(max_heap,(-nums[0],0))
        for  i in range(1,N):
            #only want  indices within k steps
            while max_heap[0][1] < i - k:
                heappop(max_heap)
            score[i] =  nums[i]+score[max_heap[0][1]]
            heappush(max_heap,(-score[i],i))
        return score[-1]

#####################
# My Calendar I
######################
class MyCalendar:
    '''
    store as list of sorted intervals
    then use binary search to find the start times
    well brute force is accepted
    
    We will maintain a list of interval events (not necessarily sorted). Evidently, two events [s1, e1) and [s2, e2) do not conflict if and only if one of them starts after the other one ends: either e1 <= s2 OR e2 <= s1. By De Morgan's laws, this means the events conflict when s1 < e2 AND s2 < e1.
    '''

    def __init__(self):
        self.intervals = []
        

    def book(self, start: int, end: int) -> bool:
        for a,b in self.intervals:
            if a < end and start < b:
                return False
        self.intervals.append([start,end])
        return True
        


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)

#####################
#   Stone Game VII
#######################
#not quite
class Solution:
    def stoneGameVII(self, stones: List[int]) -> int:
        '''
        both play optimally
        just simulate and return the final difference
        [5,3,1,4,2]
        [5,3,1,4] or [3,1,4,2]
        sum(stones[:-1]) or sum(stones[1:])
        '''
        alice = 0
        bob = 0
        turn = 0 #determine who gets points
        
        while stones:
            op1 = stones[:-1]
            op2 = stones[1:]
            
            if turn % 2 == 0:
                #alice gets points
                alice += max(sum(op1),sum(op2))
            else:
                bob += max(sum(op1),sum(op2))
                
            turn += 1
            
            if sum(op1) > sum(op2):
                stones = op1
            else:
                stones = op2
        
        return alice - bob
            
class Solution:
    def stoneGameVII(self, stones: List[int]) -> int:
        '''
        alice wants to maximize her score at the end
        bob wants to minimize the difference
        scoreRemoveFirst = sum(stones[start + 1] to stones[end])
        scoreRemoveLast = sum(stones[start] to stones[end - 1])
        Fun fact: Alice had already predicated what choice Bob is going to make.
        both are playing  optimally
        prefix sum, pad zero start 
        stones = [5,4,3,2,1]
        pref = [0,5,9,12,14,15]
        if i want sum of first to third stones, [start,end]
        pref[end+1] - pref[start]
        For Bob, he will try to return the maximum negative value. So that the difference between his and Alice's score is minimum.
        For Alice, she will try to return the maximum positive value. So that the difference between her and Bob's score is maximum.
        note we cannot be greedy here and just take max of scores
        We must know the difference in score with the opponent player for both choices
        writing to A MEMO does not work for this problem
        '''
        N = len(stones)
        pref = [0]
        memo = {}
        for stone in stones:
            pref.append(pref[-1]+stone)
        def rec(i,j):
            if i > j:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            total = pref[j+1] - pref[i]
            ans = total - min(stones[i]+rec(i+1,j),stones[j]+rec(i,j-1))
            memo[(i,j)] = ans
            return memo[(i,j)]
        
        return rec(0,N-1)
        
#dp solution
class Solution:
    def stoneGameVII(self, s: List[int]) -> int:
        dp = [[0] * len(s) for _ in range(len(s))]
        p_sum = [0] + list(accumulate(s))
        for i in range(len(s) - 2, -1, -1):
            for j in range(i + 1, len(s)):
                dp[i][j] = max(p_sum[j + 1] - p_sum[i + 1] - dp[i + 1][j], 
                               p_sum[j] - p_sum[i] - dp[i][j - 1]);
        return dp[0][len(s) - 1]

###################################
# Minimum Number of Refueling Stops
#################################
class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        '''
        we start at 0 but we are 0 target miles away
        we are given a list of gas stations and stations[i][0] gives distance and stations[i][1]
        give liters availavble
        tank can hold infinite volume and has startfuel 
        we have the choice to stop and refuel the tank or not
        what is the least number of stops to reach target
        Note that if the car reaches a gas station with 0 fuel left, the car can still refuel there.  If the car reaches the destination with 0 fuel left, it is still considered to have arrived.
        stations are in order
        
        this is a 0/1 knap sack problem, with the size of the bag being liminted to the fewest number of stops
        dp[t] answers the question what is the fuetherst we can get to refeuling t times
        we want the smallest i for which dp[i] >=- target
        so for every station if the curr distance dp[t] >= s[i][0] we can refuel
        dp[t + 1] = max(dp[t + 1], dp[t] + s[i][1])
        
        In the end, we'll return the first t with dp[t] >= target,
        otherwise we'll return -1.
        
        Now let's look at the update step. When adding a station station[i] = (location, capacity), any time we could reach this station with t refueling stops, we can now reach capacity further with t+1 refueling stops.

For example, if we could reach a distance of 15 with 1 refueling stop, and now we added a station at location 10 with 30 liters of fuel, then we could potentially reach a distance of 45 with 2 refueling stops
        '''
        dp = [startFuel]+[0]*len(stations)
        #examine all stops
        for i,(station,fuel) in enumerate(stations):
            #all stops up to i
            for stop in range(i,-1,-1):
                if dp[stop] >= station:
                    dp[stop+1] = max(dp[stop+1], dp[stop]+fuel)
        for i,d in enumerate(dp):
            if d >= target:
                return i
        return -1

class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        '''
        using a heap, keep track of the previous gas stations weve seen
        When we run out of fuel before reaching the next station, we'll retroactively fuel up: greedily choosing the largest gas stations first.
        This is guaranteed to succeed because we drive the largest distance possible before each refueling stop, and therefore have the largest choice of gas stations to (retroactively) stop at.
        '''
        pq = []
        stations.append([target,float('inf')])
        
        ans = 0
        prev = 0
        for station,cap in stations:
            startFuel -= station - prev
            while pq and startFuel < 0:
                startFuel += -heappop(pq)
                ans += 1
            if startFuel < 0:
                return -1
            heappush(pq, -cap)
            prev = station
        return ans
        
##########################
#   Palindrome Pairs
#########################
#TLE
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        '''
        we can try all ij pairs and check if concatenation yields palindrome
        '''
        def ispal(word):
            N = len(word)
            l,r = 0,N-1
            while l <= r:
                if word[l] != word[r]:
                    return False
                l += 1
                r -= 1
            return True
        
        pairs = []
        for i in range(len(words)):
            for j in range(len(words)):
                if  i != j and ispal(words[i]+words[j]):
                    pairs.append([i,j])
        return pairs

class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        '''
        hashing
        The simplest way to make a palindrome is to take 2 words that are the reverse of each other and put them together. In this case, we get 2 different palindromes, as we can put either word first.
        cat tac
        tac cat
        we know that there are alwasy 2 unique palindromes that can be formed by 2 words that are the revsrese of each other
        we know there are no duplicates in the lits
        case 1:
        cat tac
        case 2: 
        cat ..... tac
        word, some palindrome, reverse word1
        case 3:
        reverse word 2, some palindrome, word 2
        empty strin with already palindrome is palindrome
        noteL when 2 words of the same lengthj form a palindrome, it must be because word 1 is the revrse of word 2
        in pseudo code:
            Check if the reverse of word is present. If it is, then we have a case 1 pair by appending the reverse onto the end of word.
For each suffix of word, check if the suffix is a palindrome. if it is a palindrome, then reverse the remaining prefix and check if it's in the list. If it is, then this is an example of case 2.
For each prefix of word, check if the prefix is a palindrome. if it is a palindrome, then reverse the remaining suffix and check if it's in the list. If it is, then this is an example of case 3.

We'll call a suffix a "valid suffix" of a word if the remainder (prefix) of the word forms a palindrome. The function allValidSuffixes finds all such suffixes. For example, the "valid suffixes" of the word "exempt" are "xempt" (remove "e") and "mpt" (remove 'exe').

same idea with prefixes

Examples of case 1 can be found by reversing the current word and looking it up. One edge case to be careful of is that if a word is a palindrome by itself, then we don't want to add a pair that includes that same word twice. This case only comes up in case 1, because case 1 is the only case that deals with pairs where the words are of equal length.

Examples of case 2 can be found by calling allValidSuffixes and then reversing each of the suffixes found and looking them up.

Examples of case 3 can be found by calling allValidPrefixes and then reversing each of the prefixes found and looking them up.

It would be possible to simplify further (not done here) by recognizing that case 1 is really just a special case of case 2 and case 3. This is because the empty string is a palindrome prefix/ suffix of any word.
        '''
        def all_valid_prefixes(word):
            valid_prefixes = []
            for i in range(len(word)):
                if word[i:] == word[i:][::-1]:
                    valid_prefixes.append(word[:i])
            return valid_prefixes

        def all_valid_suffixes(word):
            valid_suffixes = []
            for i in range(len(word)):
                if word[:i+1] == word[:i+1][::-1]:
                    valid_suffixes.append(word[i + 1:])
            return valid_suffixes

        word_lookup = {word: i for i, word in enumerate(words)}
        solutions = []

        for word_index, word in enumerate(words):
            reversed_word = word[::-1]

            # Build solutions of case #1. This word will be word 1.
            if reversed_word in word_lookup and word_index != word_lookup[reversed_word]:
                solutions.append([word_index, word_lookup[reversed_word]])

            # Build solutions of case #2. This word will be word 2.
            for suffix in all_valid_suffixes(word):
                reversed_suffix = suffix[::-1]
                if reversed_suffix in word_lookup:
                    solutions.append([word_lookup[reversed_suffix], word_index])

            # Build solutions of case #3. This word will be word 1.
            for prefix in all_valid_prefixes(word):
                reversed_prefix = prefix[::-1]
                if reversed_prefix in word_lookup:
                    solutions.append([word_index, word_lookup[reversed_prefix]])

        return solutions

##############################
# Maximum Units on a Truck
###############################
class Solution:
    def maximumUnits(self, boxTypes: List[List[int]], truckSize: int) -> int:
        '''
        this is just a math problem
        if we had infinitley many boxes of type i
        sort on on number of units
        keep taking as much as we can
        key inisight, as at one times the boxes to add to the truck
        boxcount = min(truckSize,boxTypes[i][0])
        '''
        boxTypes.sort(key = lambda x: x[1],reverse=True)
        totalUnits = 0
        for box,unit in boxTypes:
            boxCount = min(truckSize,box)
            totalUnits += boxCount*unit
            truckSize -= boxCount
            if truckSize == 0:
                break
        return totalUnits

###################
# The Maze II
###################
#dfs, TLE
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        '''
        the ball keeps rolling until hitting a wall
        only then can we change direction
        want the shortest path, this implies bfs
        borders of the maze are walls
        note the ball call roll past the destinattion if it doesn't hit a wall
        dfs first
        keep distance array, dist[][], which reprsents the shortest path to this position from the start
        when we reach wall or boundary, keep track of number of steps
        Suppose, we reach the position (i,j)(i,j) starting from the last position (k,l)(k,l). Now, for this position, we need to determine the minimum number of steps taken to reach this position starting from the startstart position. 
         For this, we check if the current path takes lesser steps to reach (i,j)(i,j) than any other previous path taken to reach the same position i.e. we check if distance[k][l] + countdistance[k][l]+count is lesser than distance[i][j]distance[i][j]. If not, we continue the process of traversal from the position (k,l)(k,l) in the next direction.
         If distance[k][l] + countdistance[k][l]+count is lesser than distance[i][j]distance[i][j], we can reach the position (i,j)(i,j) from the current route in lesser number of steps. Thus, we need to update the value of distance[i][j]distance[i][j] as distance[k][l] + countdistance[k][l]+count. 
         Further, now we need to try to reach the destination, destdest, from the end position (i,j)(i,j), since this could lead to a shorter path to destdest. Thus, we again call the same function dfs but with the position (i,j)(i,j) acting as the current position.
         At the end, the entry in distance array corresponding to the destination, destdest's coordinates gives the required minimum distance to reach the destination. If the destination can't be reached, the corresponding entry will contain \text{Integer.MAX_VALUE}.
        '''
        rows = len(maze)
        cols = len(maze[0])
        dist = [[float('inf')]*cols for _ in range(rows)]
        dist[start[0]][start[1]] = 0
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(start):
            for dx,dy in dirs:
                x = start[0] + dx
                y = start[1] + dy
                count = 0
                while 0 <= x < rows and 0 <= y < cols and maze[x][y] == 0 :
                    x += dx
                    y += dy
                    count += 1
                
                if dist[start[0]][start[1]] + count < dist[x - dx][y - dy]:
                    dist[x - dx][y - dy] = dist[start[0]][start[1]] + count
                    dfs((x-dx,y-dy))
        dfs(start)
        
        if dist[destination[0]][destination[1]] != float('inf'):
            return dist[destination[0]][destination[1]]
        else:
            return -1

#bfs
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
    	        rows = len(maze)
        cols = len(maze[0])
        dist = [[float('inf')]*cols for _ in range(rows)]
        dist[start[0]][start[1]] = 0
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        q = deque([(start)])
        
        while q:
            curr = q.popleft()
            for dx,dy in dirs:
                x = curr[0] + dx
                y = curr[1] + dy
                count = 0
                while 0 <= x < rows and 0 <= y < cols and maze[x][y] == 0 :
                    x += dx
                    y += dy
                    count += 1
                
                if dist[curr[0]][curr[1]] + count < dist[x - dx][y - dy]:
                    dist[x - dx][y - dy] = dist[curr[0]][curr[1]] + count
                    q.append((x-dx,y-dy))
        
        if dist[destination[0]][destination[1]] != float('inf'):
            return dist[destination[0]][destination[1]]
        else:
            return -1

##########################
# Matchsticks to Square
##########################
#close one...
class Solution:
    def makesquare(self, matchsticks: List[int]) -> bool:
        '''
        recursivley generate all the sides
        once ive exhausted all of them and we dont find the sum of the matches on each of the
        four sides are equal, return false
        this is just brute force recursion, look how big the size of the matchsticks array is
        
        '''
        N = len(matchsticks)
        def recurse(idx,build):
            if idx == N: #gone through all matches
                #if we have square
                if sum(build[0]) == sum(build[1]) == sum(build[2]) == sum(build[3]):
                    return True
            
            for i in range(4):
                build[i].append(matchsticks[idx])
                recurse(idx+1,build)
            
            return False
        
        return recurse(0,[[],[],[],[]])

class Solution:
    def makesquare(self, matchsticks: List[int]) -> bool:
        '''
        we can first find the perimeter of the square
        the find the length of the target side
        first check right away if possible
        then recuse
        '''
        if not matchsticks:
            return False
        
        N = len(matchsticks)
        perim = sum(matchsticks)
        possible_side = perim // 4
        
        #check
        if possible_side*4 != perim:
            return False
        #sort, because we want to consider the longest one first
        matchsticks.sort(reverse = True)
        sides = [0]*4
        
        def recurse(idx):
            if idx == N:
                #check
                return sides[0] == sides[1] == sides[2] == possible_side
            for i in range(4):
                #if we can still fit into ith side
                if sides[i] + matchsticks[idx] <= possible_side:
                    sides[i] += matchsticks[idx]
                    #if we can continue recursin
                    if recurse(idx+1):
                        return True
                    #othewise back track
                    sides[i] -= matchsticks[idx]
            return False
        
        return recurse(0)

##################
# Generate Parentheses
##################
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        '''
        brute force generation of all possible
        at most it would be 2**16 possible string permutations
        checkingg, each permutaion would be times 16
        (2**16)*16
        then check if valid
        '''
        possible = '()'*n
        results = []
        
        #messy but it still works
        def isvalid(string):
            stack = []
            for ch in string:
                if ch == ')':
                    if stack and stack[-1] == '(':
                        stack.pop()
                    else:
                        stack.append(ch)
                else:
                    stack.append(ch)
            return len(stack) == 0
        
        #recusrively generate candidates and check
        def rec_check(build):
            if len(build) == 2*n:
                if isvalid("".join(build)):
                    results.append("".join(build))
            else:
                build.append('(')
                rec_check(build)
                build.pop()
                build.append(')')
                rec_check(build)
                build.pop()
            
        rec_check([])
        return results
    
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        '''
        instead of trying to generate all possible permutations, we try to control when we add an ( or )
        we can start with an opening bracket if we still have (of n)
        and we can start a closing brackeet if it would not exceed ther number of opening brackets
        
        '''
        res = []
        
        def backtrack(build, left,right):
            if len(build) == 2*n:
                res.append("".join(build))
                return
            if left < n:
                build.append('(')
                backtrack(build,left+1,right)
                build.pop()
            if right < left:
                build.append(")")
                backtrack(build,left, right+1)
                build.pop()
        backtrack([],0,0)
        return res

############################
# Number of Subarrays with Bounded Maximum
#############################
#brute force TLE
class Solution:
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        '''
        brute force
        examine all possible contiguous sub arrays
        and check that  left <= max <= right
        '''
        count = 0
        N = len(nums)
        for size in range(N):
            for start in range(N-size):
                sub = nums[start:start+size+1]
                if left <= max(sub) <= right:
                    count += 1
        return count

class Solution:
    def numSubarrayBoundedMax(self, nums: List[int], left: int, right: int) -> int:
        '''
        we can use dp
        dp[i] is the answer to the sub problem: what is the max number of valid sub arrays ending with A[i]
        
        if nums[i] < left we can only include nums[i] to a valid sub array ending with A[i-1]
        dp[i] = dp[i-1] for i > 0
        
        if nums[i] > R: we cannot include nums[i] in the current subarray
        
        of l <= nums <= right this can be included
        
        then we just return the sum of dp
        
        
        '''
        N = len(nums)
        dp = [0]*N
        prev = -1 #break in sub array
        
        for i in range(N):
            if nums[i] < left:
                dp[i] = dp[i-1]
            if nums[i] > right:
                dp[i] = 0
                prev = i
            if left <= nums[i] <= right:
                dp[i] = i - prev
        return sum(dp)


##################
#  Shortest Distance to Target Color
##################
#for the most part this was solved
#i think there is some overheap inolved with using a dict
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        '''
        precalculate all distances for each index for each color 3*N
        then lookup in O(1) time
        '''
        N = len(colors)
        cache = defaultdict(list)
        
        def search_left(idx,color):
            ptr = idx
            while ptr >= 0:
                if colors[ptr] == color:
                    return idx - ptr
                ptr -= 1
            return -1
        
        def search_right(idx,color):
            ptr = idx
            while ptr < N:
                if colors[ptr] == color:
                    return ptr - idx
                ptr += 1
            return -1
        
        for i in range(N):
            for c in (1,2,3):
                left = search_left(i,c)
                right = search_right(i,c)
                if left == -1 and right == -1:
                    cache[i].append(-1)
                elif left == -1 and right != -1:
                    cache[i].append(right)
                elif left != -1 and right == -1:
                    cache[i].append(left)
                else:
                    cache[i].append(min(left,right))
        
        res = []
        for idx,col in queries:
            res.append(cache[idx][col-1])
        return res

class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        '''
        we need to relaize the if colors[i] and colors[j] are c when i < j and theres no c between i and j, then for each index k, between i and j
        shortest distance between k and c on its left is k - i
        shortest distance between k and c on right is j-k

        KEY:  
        an imporant fact is that if colors[i] and colors[j] are both c, with i > j and tghere are no c's betweeen them
        then for each k between i and j
        	the shortest distance between k and c on the left is k i
        	and j-k on the right
        '''
        N = len(colors)
        rights = [0,0,0]
        lefts = [N-1]*3
        
        distance = [[-1]*N for _ in range(3)]
        
        #looking forward
        for i in range(N):
            color = colors[i]-1
            for j in range(rights[color],i+1):
                distance[color][j] = i - j
            rights[color] = i+ 1
            
        #looking backward\
        for i in range(N-1,-1,-1):
            color = colors[i] - 1
            for j in range(lefts[color],i-1,-1):
                #if we did not find a target color on its right 
                #or we find out that a targetg color on its left is close to the one onts right
                if distance[color][j] == -1 or distance[color][j] > j - i:
                    distance[color][j] = j - i
            lefts[color] = i -1
            
        return [distance[color-1][index] for index,color in queries]

###############################
#  Range Sum Query - Mutable
##############################
class NumArray:

    def __init__(self, nums: List[int]):
        '''
        tricky part is how to handle the update in O(1) time
        if i can't find away, then generating the pref sum array every time would be o(N)
        well lets just try this first
        '''
        self.nums = nums
        self.pref_sum = [0]
        for num in nums:
            self.pref_sum.append(self.pref_sum[-1] + num)
        

    def update(self, index: int, val: int) -> None:
        #reupdate
        self.nums[index] = val
        self.pref_sum = [0]
        for num in self.nums:
            self.pref_sum.append(self.pref_sum[-1] + num)
            
    def sumRange(self, left: int, right: int) -> int:
        return self.pref_sum[right + 1] - self.pref_sum[left]
        


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(index,val)
# param_2 = obj.sumRange(left,right)

#update O(1)
#still too slow
class NumArray:

    def __init__(self, nums: List[int]):
        '''
        tricky part is how to handle the update in O(1) time
        if i can't find away, then generating the pref sum array every time would be o(N)
        well lets just try this first
        '''
        self.nums = nums
        self.pref_sum = [0]
        for num in nums:
            self.pref_sum.append(self.pref_sum[-1] + num)
        

    def update(self, index: int, val: int) -> None:
        #reupdate
        curr_num = self.nums[index]
        delta = val - curr_num
        self.nums[index] = val
        #updatew pref_sum with delta
        for i in range(index+1,len(self.pref_sum)):
            self.pref_sum[i] = self.pref_sum[i] + delta
            
    def sumRange(self, left: int, right: int) -> int:
        return self.pref_sum[right + 1] - self.pref_sum[left]

class NumArray:
    '''
    sqrt decomposotion
    divid the nums array into sqrt(len(nums)) get sums in each block
    careful of overlapping sums
    '''

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.l = len(nums)**0.5
        self.length = int(ceil(len(nums)/self.l))
        self.b = [0]*self.length
        for i in range(len(nums)):
            self.b[i//self.length] += nums[i]
        

    def update(self, index: int, val: int) -> None:
        b_l = index // self.length
        self.b[b_l] = self.b[b_l] - self.nums[index] + val
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        res = 0
        startblock = left // self.length
        endblock = right // self.length
        if startblock == endblock:
            for i in range(right+1):
                res += self.nums[i]
        else:
            for i in range((startblock+1)*self.length - 1):
                res += self.nums[i]
            for i in range(startblock+1,endblock):
                res += self.b[i]
            for i in range(endblock*self.length, right+1):
                res += self.nums[i]
        return res

#https://www.youtube.com/watch?v=CWDQJGaN1gY&ab_channel=TusharRoy-CodingMadeSimpleTusharRoy-CodingMadeSimple
class NumArray:
    '''
    binary indexed tree typically used to solve range dum problems in log n time
    to get the next index in the tree:
    * get twos complement
    * AND with oriignal number
    * add to original number
    
    '''

    def __init__(self, nums: List[int]):
        self.n  = len(nums)
        self.nums = nums
        self.BIT = [0]*(self.n + 1)
        for i in range(self.n):
            idx = i + 1
            while idx <= self.n:
                self.BIT[idx] += nums[i]
                #get next
                idx += (idx & -idx)
        
    def update(self, index: int, val: int) -> None:
        delta = val - self.nums[index]
        self.nums[index] = val
        index += 1
        while index <= self.n:
            self.BIT[index] += delta
            index += (index & -index)

    #for this part accumlate the right pointers in BIT and takeaway the left
    def sumRange(self, left: int, right: int) -> int:
        res = 0
        right += 1
        while right:
            res += self.BIT[right]
            right -= (right & -right)
        while left:
            res -= self.BIT[left]
            left -= (left & - left)
        return res

#####################
#  K Inverse Pairs Array
#####################
#even recursion gets TLE
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        '''
        recall an inverse in an array (inversion)
        i < j and num[i] > nums[j]
        we are given n, the size of the array and k, the number of inversions
        we want the number of arrays of size n such that there are k inversions
        the array can only have distinct elemnts [1,n]
        
        if k is zero, this is just the increasing array from to 1 N
        lets start off with recursion and memoization
        start with array [1,2,4,3] there is one inversion cause my moving 4 to the left one time
        now examine [2,4,1,3] we shifted 2 to the left one time and 4 2 times, 3 shifts 3 inversions
        
        key:
        shifting a number to the left y times increases inversions by y
        since there are y numbers smaller than the shifted number laying to the right
        now takew the array [2,4,1,3,5] we added a 5 but it did not increase the number of inversions
        
        so adding a 5 y stepes from the right adds a total of y inverse pairs 
        examine [5,2,4,1,3], we have added 5 more inversions
        
        key 2:
        if we know the number of inverse pairs, call it x, in any abritary array b with some n, we can add a new element n+1 to array b at a positions p steps from the right 
        such that x + p = k to generate a total of k inverse pairs
        
        we can find out the number of inversions for arrays up to size n-1 each array having k inversions
        key 3:
        to generate the arrangements with exactyl k inverse pairs and n elements, we can add this new number n to all arrangments with k inverse pairs to the last spot
        for arrangements with k - 1 inverse pairs, we can add n at a positinos 1 step from the right
        
        Similarly, for an element with k-ik−i number of inverse pairs, we can add this new number nn at a position ii steps from the right. Each of these updations to the arrays leads to a new arrangement, each with the number of inverse pairs being equal to kk.
        
        so for a given n and k, we can add up the number of arrnangments of size n+1 for all k
        
        recurrence: count(i,j) represents the number of arrangemnts with i elements and j inverse pairs
        if n == 0, no paird exist,
        if k == 0, only one arrangemnt
        otherwise \sum_{i=0}^{min(k,n-1)} count(n-1,k-i)

        notes on the recurrense:
        	asssume we have alreafy calculated counts for all k arrangements of size n - 1
        	we can get the the kth arrangamenty for n by adding the newlement to the right at k - i positions
        	then we accumlated the recursive calls


        '''
        if k == 0:
            return 1 #just the increasing array
        mod = 10**9 + 7
        memo = [[0]*1001 for _ in range(1001)]
        
        def rec(i,j):
            if i == 0:
                return 0
            if j == 0:
                return 1
            if memo[i][j] != 0:
                return memo[i][j]
            count = 0
            for shift in range(min(j,i-1)+1):
                count = (count + rec(i-1,j-shift)) % mod
            memo[i][j] = count
            return count
        
        return rec(n,k)

#dp but still TLE
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        '''
        recursion is still too slow and results in O(n**2 * k)
        we can try dp in the same O(time) but faster due to less overhap from memo/cache lookup
        if we know the solutions for:
            count(n-1,0), count(n-1,1),count(n-1,2)......count(n-1,k)
        we can get:
            count(n,k) = \sum_{i=0}^{min(k,n-1)} count(n-1,k-i)
        dp[i][j] represents the number of arrangements with i elements and j inversions
        dp updates are similar to the recurrence relations:
            1. if n== 0, dp[0][k] = 0
            2. if k == 0, dp[n][0] = 1
            3. dp[i][j] = \sum_{p=0}^{min(i,j-1)} count(i-1,j-p)
        notes on upper limit:
            the limit \text{min}(j, i-1)min(j,i−1) is used to account for the cases where the number of inverse pairs needed becomes negative(p>jp>j) 
            or the case where the new inverse pairs needed by adding the n^{th}n th number is more than n-1n−1 which isn't possible, 
            since the new number can be added at (n-1)^{th}(n−1) th position at most from the right.
        '''
        mod = 10**9 + 7
        dp = [[0]*(k+1) for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(k+1):
                if j == 0:
                    dp[i][j] = 1
                else:
                    for p in range(min(j,i-1)+1):
                        dp[i][j] = (dp[i][j] + dp[i-1][j-p]) % mod
        return dp[n][k]

#finally!
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        '''
        the last approach involved a summation, which result in a linear time operation for a cell
        in the dp array
        in this case we can fill each cell with the cumsum up to the current element in a row in any dp entry
        dp[i][j] = count(i,j) + \sum_{k=0}^{j-1} dp[i][k] and count(i,j) refers to the number of arrangements with i elements and exactly  j inverse pairs
        thus each entry contains the sum of all the previous elemnts in the same row along its own result
        to obtain the sum of elementd from dp[i-1][j-i+1] to dp[i-1][j] we can use dp[i-1][j] - dp[i-1][j-i]
        to reflect the condition(min,j,i-1) in the last appaorach we ned to take the sum o only the i elements in the previous row, if i elements exists until we reach the end of the array going back
        only i elements are ocnsidered because we are generating j inverse pairs by adding i as the new number at the jth position
        at the end we return dp[n][k] - dp[n]pk-1
        '''
        mod = 10**9 + 7
        dp = [[0]*(k+1) for _ in range(n+1)]
        for i in range(1,n+1):
            for j in range(k+1):
                if j == 0:
                    dp[i][j] = 1
                else:
                    if j - i >= 0:
                        val = dp[i-1][j] + mod - dp[i-1][j-i]
                    else:
                        val = dp[i-1][j] + mod
                    dp[i][j] = (dp[i][j-1] + val) % mod
        if k > 0:
            return (dp[n][k] + mod - dp[n][k-1]) % mod
        else:
            return (dp[n][k] + mod) % mod

#finally got the recursion to work
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        '''
        recurion using memoization but instead of caching an actual result
        we cache the function call
        for it to be a little faster, lets use a 2d array instead of the dict
        '''
        memo = [[0]*1001 for _ in range(1001)]
        mod = 10**9 + 7
        
        def inversions(n,k):
            if n== 0:
                return 0
            if k == 0:
                return 1
            if memo[n][k] != 0:
                return memo[n][k]
            if k - n >= 0:
                val = (inversions(n-1,k) + mod - inversions(n-1,k-n)) % mod
            else:
                val = (inversions(n-1,k) + mod) % mod
            memo[n][k] = (inversions(n,k-1) + val) % mod
            return memo[n][k]
        if k > 0:
            return (inversions(n,k) + mod - inversions(n,k-1)) % mod
        else:
            return (inversions(n,k) + mod) % mod

#######################
# Swim in Rising Water
#######################
#close one...
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        '''
        we have n by n grid with each element being grid[i][j] elevation
        we want the smallest time to reach N-1 N-1
        at anytime t, i can swim so long as t <= elevation
        t increments by one
        '''
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        seen = set((0,0))
        rows = len(grid)
        cols = len(grid[0])
        
        t = 0
        q = deque([(0,0)])
        
        while q:
            x,y = q.popleft()
            if x == rows - 1 and y == cols - 1:
                return t
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and grid[neigh_x][neigh_y] <= t and (neigh_x,neigh_y) not in seen:
                    q.append((neigh_x,neigh_y))
                    seen.add((neigh_x,neigh_y))
            #we might not be able to go anywhere yet
            if len(q) == 0:
                q.append((x,y))
            t += 1

class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        '''
        we have n by n grid with each element being grid[i][j] elevation
        we want the smallest time to reach N-1 N-1
        at anytime t, i can swim so long as t <= elevation
        t increments by one
        we can actually just uyse a heap and swim to the direction if the smallest elevation
        
        '''
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        seen = set((0,0))
        rows = len(grid)
        cols = len(grid[0])
        
        
        heap = [(grid[0][0],0,0)]
        time = 0
        
        while heap:
            elevation,x,y = heappop(heap)
            time = max(time,elevation)
            if x == rows - 1 and y == cols - 1:
                return time
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and (neigh_x,neigh_y) not in seen:
                    heappush(heap,(grid[neigh_x][neigh_y],neigh_x,neigh_y))
                    seen.add((neigh_x,neigh_y))
                    
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        '''
        we can also interpret this is finding the minimum spannin tree (min edges touching all nodes)
        for this problem we want the shortes path with smallest elevation
        kruskal's algo anfd DSU
        we start from a list of sets where each set contains only a single node
        then greeduly merge together untyil there is only one set left 
        this reamining set turns out to be the MST
        algo:
            consider each cell in the grid as a dnoe
            sort the cells based on elevation increasinlgy
            typical bfs but if we find that the startiung and ending points are connected thanks to the newly added nodes, we can exit the loop
            and the wight of the curren cell wouldf be the minimal waiting time
        '''
        def parent(x):
            while root[x]!=x:
                root[x] = root[root[x]]
                x = root[x]
            return x
        
        
        #finding rank and using path compression
        def union(x,y):
            px = parent(x)
            py = parent(y)
            if px != py:
                if size[px] > size[py]:
                    px,py = py,px
                size[py] += size[px]
                root[px] = py
        
        N = len(grid)
        size = [1]*(N*N)
        root = list(range(N*N))
        seen = [[False]*N for _ in range(N)]
        positions=sorted([(i,j) for i in range(N) for j in range(N)],key=lambda x:grid[x[0]][x[1]])
        
        for i,j in positions:
            seen[i][j]=True
            # explore the neighbors to grow the disjoint sets
            for x,y in (i+1,j),(i-1,j),(i,j-1),(i,j+1):
                if 0<=x<N and 0<=y<N and seen[x][y]:
                    union(i*N+j,x*N+y)

            # the start and end points are joined together
            if parent(0)==parent(N*N-1):
                return grid[i][j]

####################
# Pascal's Triangle
####################
#meh it works
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        '''
        idk how to do it recursively, but i can do it iteratively
        '''
        res = []
        first = [1]
        for i in range(numRows+1):
            #add in the first one
            temp = [1]
            for j in range(i-2):
                temp.append(first[j]+first[j+1])
            temp.append(1)
            first= temp
            res.append(temp)
        return([[1]]+res[2:])
            
#another way controlling for edge cases
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = []
        for row_num in range(numRows):
            #get the row
            row = [None for _ in range(row_num+1)]
            row[0] = 1
            row[-1] = 1
            
            for j in range(1,len(row)-1):
                row[j] = res[row_num-1][j-1] + res[row_num-1][j]
            res.append(row)
        
        return res

#####################################
# Number of Matching Subsequences
####################################
#$TLE
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        '''
        we can just check if each words[i] is a subsequencve of s
        helper function and two pointers
        check if we got to the end
        '''
        def is_subseq(w,s):
            len_w = len(w)
            len_s = len(s)
            p1,p2 = 0,0
            while p1 < len_w and p2  < len_s:
                if w[p1] == s[p2]:
                    p1 += 1
                    p2 += 1
                else:
                    p1 += 1
            return p2 == len(s)
        
        count = 0
        for w in words:
            if is_subseq(s,w):
                count += 1
        return count
        #print(is_subseq('abcde','a'))

#really cool way
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        '''
        brute force solution takes up too much time checcking if sub sequence
        we first make a dictionary for all the words where initally the key is the first word of each word
        we map them to a list
        then we can go through each char in s
        retereve the words
        then for each word check if it is single char, meaning we were able to get a subsequence
        if not go to the next char in the word, and get the next chars
        https://leetcode.com/problems/number-of-matching-subsequences/discuss/329381/Python-Solution-With-Detailed-Explanation
        '''
        mapp = defaultdict(list)
        count = 0
        for w in words:
            mapp[w[0]].append(w)
        
        for char in s:
            next_words = mapp[char]
            #clear
            mapp[char] = []
            for w in next_words:
                if len(w) == 1:
                    count += 1
                else:
                    mapp[w[1]].append(w[1:])
        return count

#binary search
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        '''
        we can use binary search to check if all the indices for each lertte are increasding
        then for each char in eacx word, we'll see if if we can insieffg all the way to the right
        if we have gotten all the way to the right, we cannot make a subsequence
        '''
        lookup = defaultdict(list)
        count = 0
        
        for i,char in enumerate(s):
            lookup[char].append(i)
        
        def bs(lst,idx):
            #bisect left
            l,r = 0,len(lst)
            while l < r:
                mid = l + (r-l) // 2
                if idx < lst[mid]:
                    r = mid
                else:
                    l = mid+1
            return l
        
        for w in words:
            prev = -1 # keeps track of right most index
            found = True
            for char in w:
                right = bs(lookup[char],prev)
                #got to the end, cannot do it
                if right == len(lookup[char]):
                    found = False
                    break
                else:
                    prev = lookup[char][right]
            
            if found == True:
                count += 1
        
        return count

############################
#   All Paths from Source Lead to Destination
#########################
#close one! 48/51
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        '''
        this is just a cycle detection
        if when traversing we hit an already visisted node
        return True
        
        '''
        adj_list = defaultdict(list)
        for start,end in edges:
            adj_list[start].append(end)
        
        def dfs(node,seen):
            if node in seen:
                return False
            if node == destination:
                return True
            seen.add(node)
            for neigh in adj_list[node]:
                if dfs(neigh,seen):
                    return True
                return False
        
        return dfs(source,set())

class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        '''
        this is a good review for cycle detection
        we need to check if all pathd from source get to destination
        destination as no edges coming out of it, i.e has an out directino of zero
        if a node has no neighbors, check that this current node we are in is such
        number of paths is a finite number! there can be no cycles,
        so this is cycle detection
        key: if there are any cycles in our graph, retun false
        and we need to ensure that a node with zero out direction is the destination
        notes on cycle detection, variant of node coloring
        assign vertices to one of three colors
        we need to mark a node a being visted along a path
        if after we see that visiting this node along the path to desitnation, no cycle and mark as visited completely!
        0 means not vivisted before
        1 means visited along a path 
        2 means visited already
        '''
        adj_list = defaultdict(list)
        for start,end in edges:
            adj_list[start].append(end)
        visited = [0 for _ in range(n)]
        
        def dfs(node):
            #outdirection == 0 and is destination
            if len(adj_list[node]) == 0:
                return node == destination
            #if visited along path, cycle
            elif adj_list[node] == 1:
                return False
            elif adj_list[node] == 2:
                return True
            else:
                #visit and mark
                visited[node] = 1
                for neigh in adj_list[node]:
                    if not dfs(neigh):
                        return False
                visited[node] = 2
                return True
        
        return dfs(source)

######################
# Reverse Linked List II
#######################
#fucking edge cases
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        '''
        keep going until i hit the left
        reverse up to left
        reconnect
        '''
        # go up to left to the node just befor left
        temp = head
        ptr = 1
        while ptr < left-1:
            temp = temp.next
            ptr += 1
        #temp is now at left, reverse temp.next up to right
        next_temp = temp.next
        copy_temp = temp
        prev = None
        while left <= right:
            nextt = next_temp.next
            next_temp.next = prev
            prev = next_temp
            next_temp = nextt
            left += 1
        temp.next = prev
        ptr = 1
        temp = head
        while ptr < right:
            temp = temp.next
            ptr += 1
        temp.next = nextt
        return head

class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        '''
        you had the right idea, just the implementation wasn't right
        advance left - 1 steps
        reverse from left to right
        fix connections using cached pre pointers
        '''
        if left == right:
            return head
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        #move up to left
        for i in range(left-1):
            pre = pre.next
        
        curr = pre.next
        nxt = curr.next
        
        #reverse in range left to right
        for i in range(right- left):
            tmp = nxt.next
            nxt.next = curr
            curr = nxt
            nxt = tmp
            
        #reconnect left and right accordingly
        #the order in which we do this part is important
        pre.next.next = nxt
        pre.next = curr
        return dummy.next

#################
#   Out of Boundary Paths
#################
#TLE
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        i could try all possible paths using dfs
        if when we go out of bounds, increment a counter
        terminate when out of bounds but also when moves > maxMoves
        '''
        self.paths = 0
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(i,j,moves):
            if i == m or j == n or i < 0 or j < 0:
                self.paths += 1
                return
            if moves == 0:
                return
            for dx,dy in dirrs:
                new_x = i + dx
                new_y = j + dy
                dfs(new_x,new_y,moves-1)
        
        dfs(startRow,startColumn,maxMove)
        return self.paths

class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        i could try all possible paths using dfs
        if when we go out of bounds, increment a counter
        terminate when out of bounds but also when moves > maxMoves
        '''
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(i,j,moves):
            if i == m or j == n or i < 0 or j < 0:
                return 1
            if moves == 0:
                return 0
            paths = 0
            for dx,dy in dirrs:
                new_x = i + dx
                new_y = j + dy
                paths += dfs(new_x,new_y,moves-1)
            return paths
        
        return dfs(startRow,startColumn,maxMove)

#we just had to add a memo in, that's all
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        i could try all possible paths using dfs
        if when we go out of bounds, increment a counter
        terminate when out of bounds but also when moves > maxMoves
        '''
        mod = 10**9 + 7
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        memo = {}
        def dfs(i,j,moves):
            if i == m or j == n or i < 0 or j < 0:
                return 1
            if moves == 0:
                return 0
            if (i,j,moves) in memo:
                return memo[(i,j,moves)]
            paths = 0
            for dx,dy in dirrs:
                new_x = i + dx
                new_y = j + dy
                paths += dfs(new_x,new_y,moves-1) % mod
            paths %= mod
            memo[(i,j,moves)] = paths
            return paths
        
        return dfs(startRow,startColumn,maxMove)

#dp
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        we can turn this into bottom up dp if we can reach and  i j posisition
        we can reach each adjacent cells in x+1 moves
        dp[i][j] refers to the number of ways the position correspoinding to the indicies (i,j) can be reached given some particular number of moves
        at the current state, the dp array stores the number of ways various positions can be reached, by making use of x-1 moves, 
        so in order to determine the number of ways (i,j) can be reached we sum of the moves from adjacent poditions dp[i][j] = dp[i-1][j] + dp[i+1][j] + dp[i][j-1] + dp[i][j+1] being sure to take care of boundary conditions
         instead of updating the dp array for the current(x) moves, we make use of a temporary 2-D array temp to store the updated results for x moves,we make use of a temporary 2-D array temptemp to store the updated results for xx moves, making use of the results obtained for dpdp array corresponding to x-1x−1 moves
         we then update dp based on temp and now dp repredent the entried corresponding to xc moves
         update count at a boundary
         count += dp[i][j] when at boundary
        '''
        MOD = 10**9 + 7
        nxt = [[0] * n for _ in range(m)]
        nxt[startRow][startColumn] = 1

        ans = 0
        for time in range(maxMove):
            cur = nxt
            nxt = [[0] * n for _ in range(m)]
            for r, row in enumerate(cur):
                for c, val in enumerate(row):
                    for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                        if 0 <= nr < m and 0 <= nc < n:
                            nxt[nr][nc] += val
                            nxt[nr][nc] %= MOD
                        else:
                            ans += val
                            ans %= MOD

        return ans

#########################
# Redundant Connection
#########################
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        '''
        i need to remove an edge that still connectects all the nodes
        the edge removed should appear be the edge that is the most last in the array
        brute force would be to remove an edge and check with dfs for cycle
        if cycle is present it cannot be that edge and check that we touch all nodes
        i.e remove edge one by one
        '''
        nodes = len(edges)
        
        def dfs(node,seen):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen)
            return seen
        
        for i in range(nodes-1,-1,-1):
            temp = copy.deepcopy(edges)
            #delete an edge
            del temp[i]
            #build adj list
            adj_list = defaultdict(list)
            for a,b in temp:
                adj_list[a].append(b)
                adj_list[b].append(a)
            #dfs on 1, we need to span it anyway
            curr_state = dfs(1,set())
            #check we have all nodes
            if len(curr_state) == nodes:
                return edges[i]
        
        return -1

#using DSU
class DSU:
    def __init__(self,N):
        self.parent = [0]*N
        self.rank = [0]*N
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        xr = self.find(x)
        yr = self.find(y)
        if xr == yr:
            return False
        elif self.rank[xr] < self.rank[yr]:
            self.parent[xr] = yr
        elif self.rank[xr] > self.rank[yr]:
            self.parent[yr] = xr
        else:
            self.parent[yr] = xr
            self.rank[xr] += 1
        return True

        

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        '''
        we can also try DFU by rank using path compression
        dsu is a data sturcutre that maintains knowlegde of conneted componenets and allows for quick query
        find: gets member node of sets
        unions: draws an edge connecting compoenenst of find(x) and find(y)
        '''
        N = len(edges)
        dsu = DSU(N)
        for v,u in edges:
            if not dsu.union(v,u):
                return v,u

#####################################
# Count of Smaller Numbers After Self
#####################################
#brute forc
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        N = len(nums)
        counts = [0]*N
        for i in range(N):
            for j in range(i+1,N):
                if nums[j] < nums[i]:
                    counts[i] += 1
        return counts

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        '''
        segment tree solution
        notice that for any element nums[i], we want nums to the right and in range (-int,nums[i]-1)
        therefore, for each index x, we need a query to find the sum of those counsts
        we would need counts of values, bucketsort
        idea, we need to perform a min range query in logarithmic time
        notice the constant constraint, we can just make buckets in that range
        we also need to stor negative values for the buckets array 
        we can jusst shiuft all number to non-nwgtiave
        nums[i] = nums[i] + offset
        constant is 10**4
        we need to make sure that when we query an index, say i, only elements from index i+1 to the end of the array are present in the buckets
        algo:
            implemetn segment tree, since all values in tree are zero, we only need update an query
            each node in the segment tree is a buckey ofr counts for that nm
        '''
        #segmen tree
        def update(index,value,tree,size):
            #shift index to lead
            index += size
            #update from leaf to root
            tree[index] += value
            while index > 1:
                index //= 2
                tree[index] = tree[index*2] + tree[index*2 + 1]
        
        def query(left,right, tree,size):
            result = 0
            left += size
            right += size
            while left < right:
                #if left is a right node, bring value and move to parent's right
                if left % 2 == 1:
                    result += tree[left]
                    left += 1
                #move parewnt
                left //= 2
                #if rriht, is right node, bring vlaye of the left and move parent
                if right % 2 == 1:
                    right -= 1
                    result += tree[right]
                right //=2
            
            return result
        
        offset = 10**4
        size = 2 * 10**4 + 1
        tree = [0]*(2*size)
        result = []
        for num in reversed(nums):
            smaller_count = query(0,num+offset,tree,size)
            result.append(smaller_count)
            update(num+offset,1,tree,size)
        
        return reversed(result)

class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        '''
        using binary indexed tree, can we turn this into querying prefix sum
        yes, we can turn the number of smaller elements into a preficx sum for the range [0,num+offset-1]
        algo:
            implement BIT with offset 10**4
            pass over the numbers in reverse anf for each num:
                shift num to the appropriate offset
                query number of elements in the BIT smaller than nums
                update count
        '''
        #implement BIT
        def update(index,value,tree,size):
            index += 1
            while index < size:
                tree[index] += value
                index += index & -index
        
        def query(index,tree):
            #retutn sum, of counts nums less than, we start backwartds
            result = 0
            while index >= 1:
                result += tree[index]
                index -= index & -index
            return result
        
        offset = 10**4  # offset negative to non-negative
        size = 2 * 10**4 + 2  # total possible values in nums plus one dummy
        tree = [0] * size
        result = []
        for num in reversed(nums):
            smaller_count = query(num + offset, tree)
            result.append(smaller_count)
            update(num + offset, 1, tree, size)
        return reversed(result)

#using sorted list
from sortedcontainers import SortedList

class Solution:
    def countSmaller(self, nums):
        SList, ans = SortedList(), []
        for num in nums[::-1]:
            ind = SortedList.bisect_left(SList, num)
            ans.append(ind)
            SList.add(num)
            
        return ans[::-1]
      
####################
# Candy
#######################
#TLE
class Solution:
    def candy(self, ratings: List[int]) -> int:
        '''
        brute force would be to keep track of candies given to students
        initially each student gets one, and we scan from left to right
        if ratings[i] > ratings[i-1] and candies[i] <= candies[i-1], we update candies[i] to
        candies[i] = candies[i-1] + 1
        also if ratings[i] > ratings[i+1]
        we need to update again: candies[i] = candies[i] + candies[i+1]
        '''
        N = len(ratings)
        candies = [1]*N
        change = True
        while change:
            change = False
            for i in range(N):
                if (i != N-1) and (ratings[i] > ratings[i+1]) and (candies[i] <= candies[i+1]):
                    candies[i] = candies[i+1] + 1
                    change = True
                if (i > 0) and (ratings[i] > ratings[i-1]) and (candies[i] <= candies[i-1]):
                    candies[i] = candies[i-1] + 1
                    change = True
        
        return sum(candies)

#two pass,two arrays
class Solution:
    def candy(self, ratings: List[int]) -> int:
        '''
        we can use two arrays for this problem with two passes
        going left to right, we check:
        the student with a higher rating than the neighbor to their left, should get more cnadies
        going right o left:
        the student with a higher rating than the neighbor to their right, should get more candies
        then finally we pass both array and take the max, using running sum
        inution:
            minimize neighbors locally
            to get minimum local answer, you want at least, so max
        #for one pass, go left to right,
        then on the reverse pass update and sum along the way
        '''
        N = len(ratings)
        rightwards = [1]*N
        leftwards = [1]*N
        
        #left to right
        for i in range(1,N):
            if ratings[i] > ratings[i-1]:
                rightwards[i] = rightwards[i-1] + 1
        
        #right to left
        for i in range(N-2,-1,-1):
            if ratings[i] > ratings[i+1]:
                leftwards[i] = leftwards[i+1] + 1
                
        res = 0
        for i in range(N):
            res += max(leftwards[i],rightwards[i])
        
        return res

#########################
# Remove All Adjacent Duplicates In String
##########################
class Solution:
    def removeDuplicates(self, s: str) -> str:
        '''
        this is just a stack problem
        if there is something on the stack and it matches the letter im, clear it
        otherwise add the letter
        '''
        stack = []
        for ch in s:
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                stack.append(ch)
        return "".join(stack)
       
############################
#  Max Consecutive Ones III
############################
#almost had it!
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        '''
        brute force would be to identify where the zeros are,flip one a time and count ones
        i can use two pointers and keep finding consecutive ones
        when i reach the end of streak try extending it
        '''
        N = len(nums)
        longest_streak = 0
        left,right = 0,0
        curr_streak = 0
        curr_flips = 0
        
        while right < N:
            if nums[right] == 1:
                curr_streak += 1
                right += 1
            #not a one, but we can extend
            elif nums[right] == 0 and curr_flips < k:
                curr_streak += 1
                curr_flips += 1
                right += 1
            #not a one, but we cannot extend
            elif nums[right] == 0 and curr_flips == k:
                #reset
                longest_streak = max(longest_streak,curr_streak)
                curr_streak = 0
                curr_flips = 0
                right += 1
                left = right
        return longest_streak

#counter zeros in stead
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        '''
        brute force would be to identify where the zeros are,flip one a time and count ones
        i can use two pointers and keep finding consecutive ones
        when i reach the end of streak try extending it
        '''
        N = len(nums)
        left = 0
        
        for right in range(N):
            #if zero in window, reduc it
            if nums[right] == 0:
                k -= 1
            #if we have icluded more than k zeros, contract
            if k < 0:
                if nums[left] == 0:
                    #we get a flip back
                    k += 1
                #otherwise just move up left
                left += 1
        return right - left + 1

#reduinc if stastments
class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        left = 0
        for right in range(len(A)):
            # If we included a zero in the window we reduce the value of K.
            # Since K is the maximum zeros allowed in a window.
            K -= 1 - A[right]
            # A negative K denotes we have consumed all allowed flips and window has
            # more than allowed zeros, thus increment left pointer by 1 to keep the window size same.
            if K < 0:
                # If the left element to be thrown out is zero we increase K.
                K += 1 - A[left]
                left += 1
        return right - left + 1

################
# Armstrong Number
#################
class Solution:
    def isArmstrong(self, n: int) -> bool:
        '''
        a d-digit number is an armstsrong number of \sum_{i=0}^{digits} digits[i]**digit = the number
        '''
        str_n = str(n)
        d = len(str_n)
        SUM = 0
        for i in range(d):
            SUM += int(str_n[i])**d
        return SUM == n
        
class Solution:
    def isArmstrong(self, n: int) -> bool:
        '''
        a d-digit number is an armstsrong number of \sum_{i=0}^{digits} digits[i]**digit = the number
        we can pop offdigits using the mod and // 10
        '''
        #first get powr
        temp = n
        power = 0
        while temp != 0:
            power += 1
            temp //= 10
        SUM = 0
        temp = n
        while temp != 0:
            SUM += (temp % 10)**power
            temp //= 10
        
        return SUM == n
        

###########################################
# Lowest Common Ancestor of a Binary Tree
###########################################

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        brute force would be to generate all paths, find p and q, then find earliest common parent
        https://www.youtube.com/watch?v=13m9ZCB8gjw&t=4s&ab_channel=TusharRoy-CodingMadeSimple
        '''
        def dfs(node):
            if node == None:
                return None
            if node == p or node == q:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            #if there something to return on left or right, this must be the LCA
            if left != None and right != None:
                return node
            #if we returned nothing, retuing nothing
            if left == None and right == None:
                return None
            return left if left else right
        
        return dfs(root)
        
