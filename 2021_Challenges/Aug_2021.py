################
# Optimize Water Distribution in a Village
###############

class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        '''
        we want to span all nodes with minimum cost
        either making well at node or connecting using min cost from pipes
        minimum spanning tree? Primms?
        the duh brute force would be to generate all possible types and find min
        '''
                #build graph
        graph = defaultdict(list)
        
        #add in virtual edges touching all other edges
        for i,cost in enumerate(wells):
            graph[0].append((cost,i+1))
        
        #edges, two ways
        for h1,h2,cost in pipes:
            graph[h1].append((cost,h2))
            graph[h2].append((cost,h1))
        
        #keep set to maintain MST
        mst_set = set([0])
        
        #keep heap starting from frist edge, note we don't need to start from the first node
        heapify(graph[0])
        edges_heap = graph[0]
        
        total_cost = 0
        while len(mst_set) < n+1:
            cost,next_house = heappop(edges_heap)
            if next_house not in mst_set:
                #add it in
                mst_set.add(next_house)
                total_cost += cost
                #neighbors
                for new_cost,neighbor_house in graph[next_house]:
                    if neighbor_house not in mst_set:
                        heappush(edges_heap,(new_cost,neighbor_house))
        return total_cost

class UnionFind:
    def __init__(self,size) -> None:
        self.group = [i for i in range(size+1)]
        self.rank = [0]*(size+1)
    
    def find(self,person: int) -> int:
        if self.group[person] != person:
            self.group[person] = self.find(self.group[person])
        return self.group[person]
    
    def union(self, person_1:int, person_2:int) -> bool:
        group_1 = self.find(person_1)
        group_2 = self.find(person_2)
        if group_1 == group_2:
            return False

        # attach the group of lower rank to the group with higher rank
        if self.rank[group_1] > self.rank[group_2]:
            self.group[group_2] = group_1
        elif self.rank[group_1] < self.rank[group_2]:
            self.group[group_1] = group_2
        else:
            self.group[group_1] = group_2
            self.rank[group_2] += 1

        return True
        
        
class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        '''
        we can use kruskals algorithm using union find by rank with path compression
        kruskal incrementally adds disjoin sets
        in kruskal we consider all edges at once, ranked by their cosses
        idea:
            iterate through all the edges orderd by their costs and for each edge, we decide
            whether or not we include it in the MST, i.e we only add an edge if it connects componets
        DSU:    
            find, returns id of group, to which a belongs
            union, joins a and b, if they are already in the same group we do nothing
        '''
        ordered_edges = []
        #add in virutal index
        for i,weight in enumerate(wells):
            ordered_edges.append((weight,0,i+1))
        
        #add in existing edges
        for h1,h2,weight in pipes:
            ordered_edges.append((weight,h1,h2))
        
        #sort increasingly on edges weight
        ordered_edges.sort(key = lambda x: x[0])
        
        #union find
        uf = UnionFind(n)
        total_cost = 0
        for cost,h1,h2 in ordered_edges:
            #do we need to add this edge
            if uf.union(h1,h2):
                total_cost += cost
        return total_cost


#################
# 827. Making A Large Island
##################
class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        '''
        first solution times out, ends up being O(N^4), we can do better
        recall we check every zero, but really we can just check if a zero belongs to the same group
         For example, consider grid = [[0,1],[1,1]]. 
         The answer is 4, not 1 + 3 + 3, since the right neighbor and the bottom neighbor of the 0 belong to the same group.
         we can keep track of a group id for each unique group of zeros 
         i.e, each group of conected zeros
         algo:
         for each group, mark with unique index, and dfs to finds its area
         then for each 0, look at the neighboring group ids seen and add the area of rthose groups plus 1 (for the current zero)
         
        '''
        N = len(grid)
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def neighbors(r,c):
            for dx,dy in dirrs:
                new_x = r + dx
                new_y = c + dy
                if 0 <= new_x < N and 0 <= new_y < N:
                    yield new_x,new_y
        
        #recall we dfs for each group
        def dfs(r,c,idx):
            ans = 1
            grid[r][c] = idx
            for new_x,new_y in neighbors(r,c):
                if grid[new_x][new_y] == 1:
                    ans += dfs(new_x,new_y,idx)
            return ans
        
        #hold areas for groups
        areas = {}
        idx = 2
        for i in range(N):
            for j in range(N):
                if grid[i][j] == 1:
                    areas[idx] = dfs(i,j,idx)
                    idx += 1
        
        #print(areas)
        ans = max(areas.values() or [0])
        for i in range(N):
            for j in range(N):
                if grid[i][j] == 0:
                    #make seen set
                    seen = set()
                    for new_x,new_y in neighbors(i,j):
                        if grid[new_x][new_y] > 1:
                            seen.add(grid[new_x][new_y])
                    #get area of current group
                    curr_area = sum(areas[idx] for idx in seen)
                    ans = max(ans,1+curr_area)
        return ans
                    
################
# Two Sum
#################
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        '''
        two pass hash,
        dump nums:idx into a mapp
        then when retraversing, check for comp in mapp
        '''
        mapp = {}
        for i,num in enumerate(nums):
            mapp[num] = i
        
        for i,num in enumerate(nums):
            if target - num in mapp:
                if mapp[(target-num)] != i:
                    return [i,mapp[(target-num)]]
        
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        mapp = {}
        for i,num in enumerate(nums):
            comp = target - num
            if comp in mapp:
                return [i, mapp[comp]]
            mapp[num] = i

##################
# Subsets II
##################
#https://leetcode.com/problems/subsets-ii/discuss/1380237/C%2B%2BPython-Bitmasking-Backtracking-Iterative-Solutions-with-Picture-Clean-and-Concise
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        '''
        we can use recursion with backtracking
        we sort nums, and when making a subset check is nums[i] == nums[i-1]
        this would result in a duplicate
        '''
        results = []
        nums.sort()
        N = len(nums)
        
        def backtrack(idx,curr):
            results.append(curr[:])
            for i in range(idx,len(nums)):
                if i > idx and nums[i] == nums[i-1]:
                    continue
                curr.append(nums[i])
                backtrack(i+1,curr)
                curr.pop()
        
        backtrack(0,[])
        return results

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        seen = set()

        for mask in range(1 << n):
            subset = []
            for i in range(n):
                bit = (mask >> i) & 1  # Get i_th bit of mask
                if bit == 1:
                    subset.append(nums[i])

            seen.add(tuple(subset))

        return seen

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = set()
        ans.add(tuple())
        
        for num in nums:
            newSubsets = set()
            for subset in ans:  # Iterate previous subsets from ans
                newSubsets.add(tuple(list(subset) + [num]))
            ans.update(newSubsets)  # Add new subsets to ans
            
        return ans

####################
# Path Sum II
####################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        '''
        what if i generated all paths first, then check sums, and only return paths
        who's sum is targetsum
        '''
        if not root:
            return root
        self.paths = []
        
        def dfs(node,path):
            if not node:
                return
            #leaf
            if not node.left and not node.right:
                self.paths.append(path+[node.val])
            
            dfs(node.left,path+[node.val])
            dfs(node.right,path+[node.val])
        
        dfs(root,[])
        res = []
        for path in self.paths:
            if sum(path) == targetSum:
                res.append(path)
        return res

class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        '''
        keep exploring all paths decremeting a currSum by the node val
        when it hits zero and is a leaf its path
        '''
        valid_paths = []
        def dfs(node,currSum,currPath):
            if not node:
                return None
            currSum -= node.val
            currPath.append(node.val)
            #check if leaf
            if not node.left and not node.right:
                if currSum == 0:
                    valid_paths.append(currPath[:])
            else:
                dfs(node.left, currSum,currPath)
                dfs(node.right,currSum,currPath)
            currPath.pop()
        
        dfs(root,targetSum,[])
        return valid_paths
        

#########################
# Stone Game
#########################
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        '''
        even number of piles, odd number of stones, so there's always a winner
        if alex starts first, lee will always ends the game
        both play optimally, and at any one time, you can only take stones from start or end
        alex's score is just max(start,end) + current score
        if i had a recusrive that returned the previous score so far, alex would just take the max
        then just see if alex's score is greater than SUM - score

        '''
        N = len(piles)
        all_stones = sum(piles)
        memo = {}
        
        def max_score(i):
            if i == (N//2) -1:
                return max(piles[i],piles[i+1])
            if i in memo:
                return memo[i]
            
            score = max_score(i+1) + max(piles[i],piles[-i-1])
            memo[i] = score
            return score
        
        alex = max_score(0)
        return alex > all_stones - alex

#the actual way
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        '''
        https://leetcode.com/problems/stone-game/discuss/1384652/cpython-dp-no-math-easy-to-understand-clean-concise/1037584
        we need both players to play optimally
        if we had a recusrive function that returns two scores, alex and lee
        and we each call we move two pointers i += 1, j -=1
        and we take from the beginning or end
        we increase each score by taking the greater 1
        '''
        memo = {}
        def rec(i,j):
            if i > j:
                return (0,0)
            
            if (i,j) in memo:
                return memo[(i,j)]
            
            take_first = rec(i+1,j)
            take_end = rec(i,j-1)
            
            ans = None
            #if start is higher, have alex take it, and lee doesnt
            if piles[i] + take_first[1] > piles[j] + take_end[1]:
                ans = piles[i] + take_first[1],take_first[0]
                memo[(i,j)] = ans
                return ans
            #otherwise the end is larger
            ans = piles[j] + take_end[1], take_end[0]
            memo[(i,j)] = ans
            return ans
        
        alex,lee = rec(0,len(piles)-1)
        return alex > lee

#iterative dp 
class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        n = len(piles)
        dp = [[(0, 0)] * n for _ in range(n)]
        
        for left in range(n - 1, -1, -1):
            for right in range(left, n):
                if left == right:  # Base case
                    dp[left][right] = (piles[left], 0)
                    continue
                    
                pickLeft = dp[left + 1][right]
                pickRight = dp[left][right - 1]
                if piles[left] + pickLeft[1] > piles[right] + pickRight[1]:  # If the left choice has higher score than the right choice
                    dp[left][right] = (piles[left] + pickLeft[1], pickLeft[0])  # then pick left
                else:
                    dp[left][right] = (piles[right] + pickRight[1], pickRight[0])  # else pick right

        aliceScore, leeScore = dp[0][n - 1]
        return aliceScore > leeScore

#####################################
#  N-ary Tree Level Order Traversal
#####################################
#bfs
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        '''
        bfs
        '''
        if not root:
            return []
        ans = []
        q = deque([root])
        while q:
            curr_level = []
            N = len(q)
            for i in range(N):
                curr_node = q.popleft()
                curr_level.append(curr_node.val)
                for child in curr_node.children:
                    q.append(child)
            ans.append(curr_level)
        
        return ans
                
#dfs global hash
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        '''
        using global hash to rebuild for dfs
        '''
        if not root:
            return []
        levels = defaultdict(list)
        self.max_lv = 0
        def dfs(node,depth):
            if not node:
                return
            self.max_lv = max(self.max_lv,depth)
            levels[depth].append(node.val)
            for child in node.children:
                dfs(child,depth+1)
        dfs(root,0)
        res = []
        for i in range(self.max_lv+1):
            res.append(levels[i])
        return res

#dfs not global
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        '''
        using global hash to rebuild for dfs
        '''
        if not root:
            return []
        levels = []
        def dfs(node,depth):
            if not node:
                return
            if len(levels) == depth:
                levels.append([])
            levels[depth].append(node.val)
            for child in node.children:
                dfs(child,depth+1)
                
        dfs(root,0)
        return levels

######################
# Palindrome Partitioning II
######################
#TLE
class Solution:
    def minCut(self, s: str) -> int:
        '''
        we keep cutting at string s if the two pieces are palindromes
        we abandon a cut if it does not
        backtrack to get the min cuts
        min cuts = min(min so far, 1+minimum cuts for the reamining substrintg)
        #naive backtracking
        algo:
            1. define cusrsive method that finds min cuts for a subtring starting at index start and ending at index end
            2. to find cuts, we must also know the min cuts seens so far for the other palindrom partiions
            3. inital cuts would be set to len(s) - 1, i.e each char is trivially a palindrome
            4. we must generate all possible substrings with starts and edn
        '''
        memo = {}
        #function to check if string is palindrom
        def palindrome(string):
            start,end = 0,len(string)-1
            while start <= end:
                if string[start] != string[end]:
                    return False
                start += 1
                end -= 1
            return True


        def findMinCuts(s,start,end,minCuts):
            #base condition, end of string, or peice is a palindrom, no more cuts needed
            if start == end or palindrome(s[start:end+1]):
                return 0
            if (start,end) in memo:
                return memo[(start,end)]
            for i in range(start,end+1):
                #find result for substring (start,i) if its is a palindrome
                if palindrome(s[start:i+1]):
                    minCuts = min(minCuts,1+findMinCuts(s,i+1,end,minCuts))
            memo[(start,end)] = minCuts
            return minCuts
        
        return findMinCuts(s,0,len(s)-1,len(s)-1)

#we need to add a memo for both palin and mincuts
class Solution:
    def minCut(self, s: str) -> int:
        memo = {}
        memo_palin = {}
        #function to check if string is palindrom
        def palindrome(start,end):
            if start >= end:
                return True
            if (start,end) in memo_palin:
                return memo_palin[(start,end)]
            res = (s[start] == s[end] and palindrome(start+1,end-1))
            memo[(start,end)] = res
            return res



        def findMinCuts(s,start,end,minCuts):
            #base condition, end of string, or peice is a palindrom, no more cuts needed
            if start == end or palindrome(start,end):
                return 0
            if (start,end) in memo:
                return memo[(start,end)]
            for i in range(start,end+1):
                #find result for substring (start,i) if its is a palindrome
                if palindrome(start,i):
                    minCuts = min(minCuts,1+findMinCuts(s,i+1,end,minCuts))
            memo[(start,end)] = minCuts
            return minCuts
        
        return findMinCuts(s,0,len(s)-1,len(s)-1)

#written more sucunitl and moving only start indices
class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        memo_palin = {}
        memo_cuts = {}
        
        def isPalindrome(l, r):  # l, r inclusive
            if (l,r) in memo_palin:
                return memo_palin[(l,r)]
            if l >= r: 
                return True
            if s[l] != s[r]: 
                return False
            ans = isPalindrome(l+1, r-1)
            memo_palin[(l,r)] = ans
            return ans
        
        def dp(i):  # s[i..n-1]
            if i == n:
                return 0
            if i in memo_cuts:
                return memo_cuts[i]
            ans = math.inf
            for j in range(i, n):
                if (isPalindrome(i, j)):
                    ans = min(ans, dp(j+1) + 1)
            memo_cuts[i] = ans
            return ans
        
        return dp(0) - 1

#iterative dp
class Solution:
    def minCut(self, s: str) -> int:
        '''
        bottom up dp
        algo:
            1. two nested loops, starting out with ending index
            2. inner starts with all starting indcies
            3. build 1D cuts dp array, and dp[i] stored the min number of cuts for substring enidng at i
            4. initally m in cut is equal to max cuts so far, and so for a substring ending at index end,
            the min cut would be equal to the value of index end
            5. we can calculate min sut for s[start:end] as
            minimum(minimumCut, Minimum cuts for substring s(start, end))
Minimum cuts for substring s(start, end) = 1 + Minimum cuts for substring s(0, start - 1)
            Minimum cuts for substring s.substring(0, start - 1) is equivalent to finding the result for substring ending at index start - 1 which can be given by cutsDp[start - 1]. So, we can say that,
        
        https://leetcode.com/problems/palindrome-partitioning-ii/discuss/777729/Python%3A-Easy!-Simple-DP-Solution-oror-Time-O(n2)-oror-Explanation
        '''
        if not s : return 0
        
        dp = [[True] * len(s) for i in range(len(s))]        # DP Matrix of booleans -> dp[i][j] - TRUE if 's[i: j + 1]' is palindrome, else FALSE
        cuts = [float("inf")] * len(s)                       # DP cuts array -> indicates min cuts require till ith entry
        
        # We first find all palindromic substrings
        for r in range(1, len(s)):
            for c in range(len(s) - r):
                if not (s[c] == s[c + r] and dp[c + 1][c + r - 1]):
                    dp[c][c + r] = False

        # For ith column, we check every entry till diagonal element
        # If dp[j][i] is true, implies 's[j: i + 1]'' is palindrome and
        # we check if we get minimum cuts considering this substring or not  
        for i in range(len(s)):
            for j in range(i + 1):
                if dp[j][i]:
                    cuts[i] = min(cuts[i], (cuts[j - 1] + 1) if j - 1 >= 0 else 0)
                    
        return cuts[-1]

##########################
# Rank Transform of a Matrix
##########################
#welp, its a true hard, don't feel bad if you can't get it
class Solution:
    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        '''
        https://leetcode.com/problems/rank-transform-of-a-matrix/solution/
        key takeawasy from BFS approach:
            the rank at (i,j) can just be the max of the rank in row i or max of the rank col j and add 1
            we could do this for all i,j sorted, but we can cache for all rows and cols
            BUT, ranks of the same value connected by a row or col, SHOULD have the same rank
            fiind whold connecte parts max rank first, then updat each point in that part to that rank
        HOW DO WE FIND CONNECTED PARTS?
            The idea of BFS is simple: from a starting point, add all the directly connected points (i.e., with the same row or same column) into a waiting queue, 
            pop points from the waiting queue, add new directly connected points into the queue, and repeat until the queue is empty.
            CONNECT POINTRS row to column and column to row
            BUT HOW DO WE STORE? We could have two graphs, graphCol and graphRow
            WE CAN COMBINE THESE GRAPHS....usng complenemt indices ~i or ~j
            ~col = -col - 1
            Therefore, we can use a single graph to store the connections between row and column: if i >= 0, graph[i] 
            represents i-th row's neighbors (the complement of indexes of linked columns), and if i < 0, graph[~i] gives ~i-th column's adjacent points (the indexes of linked rows).
            Now, only MM points (represent rows) and NN points (represent columns) are in the graph. 
        
        algo:
            1. init graph for different values, iterate matrix and links rows and columns to correspond graphh
            2. init value2index map to store connected comps
                *map reads at value: [list of poins with same value]
            3. fill in value2index again
                * for each point BFS to find connected components
            4. sort keys in value2Index
            5.init our answer matrix and fill in values from mapp
        '''
        m = len(matrix)
        n = len(matrix[0])

        # link row to col, and link col to row
        graphs = {}  # graphs[v]: the connection graph of value v
        for i in range(m):
            for j in range(n):
                v = matrix[i][j]
                # if not initialized, initial it
                if v not in graphs:
                    graphs[v] = {}
                if i not in graphs[v]:
                    graphs[v][i] = []
                if ~j not in graphs[v]:
                    graphs[v][~j] = []
                # link i to j, and link j to i
                graphs[v][i].append(~j)
                graphs[v][~j].append(i)

        # put points into `value2index` dict, grouped by connection
        value2index = {}  # {v: [[points1], [points2], ...], ...}
        seen = set()  # mark whether put into `value2index` or not
        for i in range(m):
            for j in range(n):
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                v = matrix[i][j]
                graph = graphs[v]
                # start bfs
                q = [i, ~j]
                rowcols = {i, ~j}  # store visited row and col
                while q:
                    node = q.pop(0)
                    for rowcol in graph[node]:
                        if rowcol not in rowcols:
                            q.append(rowcol)
                            rowcols.add(rowcol)
                # transform rowcols into points
                points = set()
                for rowcol in rowcols:
                    for k in graph[rowcol]:
                        if k >= 0:
                            points.add((k, ~rowcol))
                            seen.add((k, ~rowcol))
                        else:
                            points.add((rowcol, ~k))
                            seen.add((rowcol, ~k))
                if v not in value2index:
                    value2index[v] = []
                value2index[v].append(points)

        answer = [[0]*n for _ in range(m)]  # the required rank matrix
        rowmax = [0] * m  # rowmax[i]: the max rank in i row
        colmax = [0] * n  # colmax[j]: the max rank in j col
        for v in sorted(value2index.keys()):
            # update by connected points with same value
            for points in value2index[v]:
                rank = 1
                for i, j in points:
                    rank = max(rank, max(rowmax[i], colmax[j]) + 1)
                for i, j in points:
                    answer[i][j] = rank
                    # update rowmax and colmax
                    rowmax[i] = max(rowmax[i], rank)
                    colmax[j] = max(colmax[j], rank)

        return answer

#union find
class Solution:
    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        '''
        we can use union find with path compression
        recall for the connected compoentns of the saeme value, we connect row to col and col to row
        what union needs to do is assue that find(Row2) and find(Col1) yeild same value
        add path compression to facilite find
        union by rank: when we merge two trees, and we make the rank with the higher rank the parent
        and the node with the lower rank the child
        algo:
            1. implenet union find,
            2. initn UF's for for different values
            3. get value2index to store connectred parts
                value2indes if of the form {v1: {root1:....,root2 }
            4. fill in value2indx again by traversing the matrix:
                for a point, us find to calcute its root
            5. sort the keys in value2index
            6. find max rank for row,col, and conected parts
        '''
        m = len(matrix)
        n = len(matrix[0])

        # implement find and union
        def find(UF, x):
            if x != UF[x]:
                UF[x] = find(UF, UF[x])
            return UF[x]

        def union(UF, x, y):
            UF.setdefault(x, x)
            UF.setdefault(y, y)
            UF[find(UF, x)] = find(UF, y)

        # link row and col together
        UFs = {}  # UFs[v]: the Union-Find of value v
        for i in range(m):
            for j in range(n):
                v = matrix[i][j]
                if v not in UFs:
                    UFs[v] = {}
                # union i to j
                union(UFs[v], i, ~j)

        # put points into `value2index` dict, grouped by connection
        value2index = {}
        for i in range(m):
            for j in range(n):
                v = matrix[i][j]
                if v not in value2index:
                    value2index[v] = {}
                f = find(UFs[v], i)
                if f not in value2index[v]:
                    value2index[v][f] = []
                value2index[v][f].append((i, j))

        answer = [[0]*n for _ in range(m)]  # the required rank matrix
        rowmax = [0] * m  # rowmax[i]: the max rank in i row
        colmax = [0] * n  # colmax[j]: the max rank in j col
        for v in sorted(value2index.keys()):
            # update by connected points with same value
            for points in value2index[v].values():
                rank = 1
                for i, j in points:
                    rank = max(rank, max(rowmax[i], colmax[j]) + 1)
                for i, j in points:
                    answer[i][j] = rank
                    # update rowmax and colmax
                    rowmax[i] = max(rowmax[i], rank)
                    colmax[j] = max(colmax[j], rank)

        return answer
################
#Add Strings
##################
#its the fucking edge cases...
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        '''
        we can use two pointers starting from the beginning
        keep carry
        carry //
        ans % 10
        '''
        ans = ""
        #we need to prepend ans
        ptr1 = len(num1)
        ptr2 = len(num2)
        carry = 0
        
        while ptr1 > 0 and ptr2 > 0:
            first = int(num1[ptr1-1])
            second = int(num2[ptr2-1])
            res = first + second
            if carry:
                res += carry
                carry = 0
            carry += res // 10
            res %= 10
            ans = str(res)+ans
            ptr1 -= 1
            ptr2 -= 1
        
        #if we still have to move ptr1
        while ptr1 > 0:
            first = int(num1[ptr1-1])
            if carry:
                first += carry
                carry = 0
            carry += first // 10
            first %= 10
            ans = str(first)+ans
            ptr1 -= 1
        
        #if ptr2
        while ptr2 > 0:
            first = int(num1[ptr2-1])
            if carry:
                first += carry
                carry = 0
            carry += first // 10
            first %= 10
            ans = str(first)+ans
            ptr2 -= 1
        
        #if carry
        if carry:
            ans = str(carry)+ans
        
        return ans

class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        res = []
        carry = 0
        ptr1 = len(num1)-1
        ptr2 = len(num2)-1
        
        #we start adding from the end
        while ptr1 >= 0 or ptr2 >=0:
            first = ord(num1[ptr1]) - ord('0') if ptr1 >= 0 else 0
            second = ord(num2[ptr2]) - ord('0') if ptr2 >= 0 else 0
            
            val = (first + second + carry) % 10
            carry = (first + second + carry) // 10
            res.append(str(val))
            ptr1 -= 1
            ptr2 -= 1
        
        #final carry
        if carry:
            res.append(str(carry))
        
        return "".join(res)[::-1]

#####################
#  Flip String to Monotone Increasing
#####################
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        '''
        there are multiple answerrs, we just want min flips
        there must be a recurrence of finding the minimum
        must end in a 1 or all zeros
        for each s[i] we can either flip or not flip
        if we flip we add 1, if we don't we don't add 1
        and we just return the min number of flips
        https://leetcode.com/problems/flip-string-to-monotone-increasing/discuss/1394646/Python-Super-easy-Top-down-DP-Clean-and-Concise
        '''
        memo = {}
        N = len(s)
        
        #function that returns min flips up to i
        def recurse(i,prev_digit):
            if i == N:
                return 0
            if (i,prev_digit) in memo:
                return memo[(i,prev_digit)]
            curr = ord(s[i]) - ord('0') #either 1 or 0
            flipped_curr = 1 - curr
            minflips = float('inf')
            #if flippinf curr digit makes it montonic consider it
            if flipped_curr >= prev_digit:
                minflips = 1 + recurse(i+1,flipped_curr)
            #otherwise do not flip
            if curr >= prev_digit:
                minflips = min(minflips,recurse(i+1,curr))
            memo[(i,prev_digit)] = minflips
            return minflips
        
        return recurse(0,0)

#dp, translation from recursion
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        n = len(s)
        INF = n  # There are maxmimum N flips we can use to make the string monotone increasing!
        
        dp = [[INF] * 2 for _ in range(n + 1)]
        dp[n][0] = dp[n][1] = 0  # Base case
        for i in range(n - 1, -1, -1):
            for j in range(2):
                d = ord(s[i]) - ord('0')
                if d >= j:
                    dp[i][j] = min(dp[i][j], dp[i + 1][d])  # Don't flip
                if 1 - d >= j:
                    dp[i][j] = min(dp[i][j], dp[i + 1][1 - d] + 1)  # Flip
        return dp[0][0]

#iterative dp but a different way
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        '''
        imagine we are at s[i] and we know up to s[i] it costs count flips
        and there were also count ones up to s[i]
        no when we enoucnter a new one, how sohould count flip be updates
        when there is a 1, we do not need to flip
        when zero comes,:
            flip newly appened 0 to 1
            or flip count 1 to 0
        tkae away, if there's a one, we don't need to flip it, but count as ones
        otherwise flip a zero and take the min of count ones or count flips
        '''
        count_ones = 0
        count_flips = 0
        for ch in s:
            if ch == '1':
                count_ones += 1
            else:
                count_flips += 1
            count_flips = min(count_ones,count_flips)
        
        return count_flips

##################
# Array of Doubled Pairs
##################
#gahhh
class Solution:
    def canReorderDoubled(self, arr: List[int]) -> bool:
        '''
        we are given an array of even length
        return true iff it is possible to reorder such that arr[2*i+1] = 2*arr[2*i]
        for i in range the range [0,len(arr)/2]
        run this to see the index pattern
        for i in range(5):
            print(2*i + 1,2*i)
        we want to see if we turn the array into pairs (arr[i],2*arr[i])
        buid pairs?
        '''
        counts = Counter(arr)
        N = len(arr)
        
        for num in counts:
            #if i cant find its double of halved
            if num*2 not in counts or num//2 not in counts:
                return False
            #if i can
            elif num*2 in counts or num//2 in counts:
                doubled = counts[num*2]
                halved = counts[num/2]

#greedy sort
class Solution:
    def canReorderDoubled(self, arr: List[int]) -> bool:
        '''
        we can be greedy
        if x is currently the array elemtn with the least asbsolute value, it must pair with 2*x
        check elements in order of absolute value
        when we check an element x and it is not used, we pair it with 2*x,
        then decrement the counts of x and 2*x
        if we can't pair it wih 2*x after sorting, we can't do it
        '''
        counts = Counter(arr)
        for num in sorted(arr,key = abs):
            #if there;s nothing to pair
            if counts[num] == 0:
                continue
            if counts[2*num] == 0:
                return False
            counts[num] -= 1
            counts[2*num] -= 1
        return True

#######################
# Paint Fence
#######################
class Solution:
    def numWays(self, n: int, k: int) -> int:
        '''
        top down recursion
        there are a few cases we know right off the bat, if n = 1, return k
        if n = 2, then return k*k
        so far we have totalways(i), give the num ways to paint i posts
        totalways(1) = k, and totalways(2) = k*k
        now we just need the num ways for 3 <= i <= n
        how to paint the ith post:
            1. use a different color than the previous posts, so we have k-1 colors left
            meaning there are (k-1)*totalways(i-1) ways to paint the ith post a different color
            2. use same color s previous post only if the (i-1) post is a different color than the (i-2 post)
        so how many ways are the paint (i-1)th post a different color of (i-2)th post?
        well as stated in the first option, there are (k-1)*totalways(i-1) to paoint i th post a different color
        and so that means there are 1*(k-1)*totalways(i-2)
        recurrence is:
        totalWays(i) = (k - 1) * totalWays(i - 1) + (k - 1) * totalWays(i - 2)
        using distributive property
        totalWays(i) = (k - 1) * (totalWays(i - 1) + totalWays(i - 2))
        '''
        memo = {}
        def totalWays(i):
            if i == 1:
                return k
            if i == 2:
                return k*k
            if i in memo:
                return memo[i]
            ith_minus_one = (k-1)*totalWays(i-1)
            ith_minus_two = (k-1)*totalWays(i-2)
            res = ith_minus_one + ith_minus_two
            memo[i] = res
            return res
        return totalWays(n)
            
#bottom up dp
class Solution:
    def numWays(self, n: int, k: int) -> int:
        '''
        bottom up dp
        '''
        if n == 1:
            return k
        if n == 2:
            return k*k
        
        dp = [0]*(n+1)
        dp[1] = k
        dp[2] = k*k
        
        for i in range(3,n+1):
            dp[i] = (k-1)*(dp[i-1]+dp[i-2])
        return dp[-1]

#################
# Group Anagrams
#################
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        '''
        just use tuple as of zeros size 26
        and ord with a
        '''
        mapp = defaultdict(list)
        for w in strs:
            counts = [0]*26
            for ch in w:
                idx = ord(ch) - ord('a')
                counts[idx] += 1
            mapp[tuple(counts)].append(w)
        return mapp.values()

# could also just do sroted string
class Solution(object):
    def groupAnagrams(self, strs):
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()

####################
# Set Matrix Zeroes
####################
#in place but O(rows+cols) space
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        '''
        dumb way is to make a copy of the matrix
        then in the matrix, find the rows and cols have zeros
        then adjust copied matrix
        then adjust pointer of original matrix
        '''
        m = len(matrix)
        n = len(matrix[0])
        zero_rows = set()
        zero_cols = set()
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    zero_rows.add(i)
                    zero_cols.add(j)
        
        for r in zero_rows:
            for c in range(n):
                matrix[r][c] = 0
        for c in zero_cols:
            for r in range(m):
                matrix[r][c] = 0

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        '''
        we need to watch for the first row,col i.e matrix[0][0]
        since they share the same idx
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        
        first_col = False
        
        for i in range(rows):
            #check first col
            if matrix[i][0] == 0:
                first_col = True
            #if element if zero, set first row and col to zero
            for j in range(1,cols):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0
        
        #pass array again using first row and col
        for i in range(1,rows):
            for j in range(1,cols):
                if not matrix[0][j] or not matrix[i][0]:
                    matrix[i][j] = 0
        
        #check if first row needs changing
        if matrix[0][0] == 0:
            for j in range(cols):
                matrix[0][j] = 0
        
        #see if first col needs to be zero
        if first_col:
            for i in range(rows):
                matrix[i][0] = 0


######################
# Remove Boxes
######################
#nice try
class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        '''
        dp, we want to maximize the number of points we can get
        for a count,k, of a certain box color, we can get k*k points
        dynamic programming
        if i had a function dp(i) that returned the max score so far
        then i would need to take the box's whose count would maximize my score
        '''
        self.score = 0
        
        def dp(curr_score,counts):
            if len(counts) == 0:
                self.score = max(self.score,curr_score)
                
            for k,v in list(counts.items()):
                del counts[k]
                dp(curr_score+v,counts)
                counts[k] = v
        dp(0,Counter(boxes))
        return self.score

#note this only passes if we use lru cache
class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        '''
        hardest problem all year on leetcode
        https://leetcode.com/problems/remove-boxes/discuss/101310/Java-top-down-and-bottom-up-DP-solutions
        JUST try to copy the solution first after reading the writeup, then look back at the wrtie up
        '''
        N = len(boxes)
        N += 1
        memo = [[[0]*N for _ in range(N)] for _ in range(N)]
        
        def dp(i,j,k):
            if memo[i][k][k] > 0:
                return memo[i][j][k]
            if i > j:
                return 0
            if i == j:
                return (k+1)*(k+1) #if we have taken k boxes of the same color, we can keep taking of the same color from i+1 and from j-1
            #starting with how many continutous of same color
            while i < j and boxes[i+1] == boxes[i]:
                k += 1
                i += 1
            
            max_val = (k+1)*(k+1) + dp(i+1,j,0)
            
            for m in range(i+1,j+1):
                if boxes[m] == boxes[i]:
                    points = dp(i+1,m-1,0) + dp(m,j,k+1)
                    max_val = max(max_val,points)
            memo[i][j][k] = max_val
            return max_val
        
        return dp(0,len(boxes)-1,0) 

class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        '''
        hardest problem all year on leetcode
        https://leetcode.com/problems/remove-boxes/discuss/101310/Java-top-down-and-bottom-up-DP-solutions
        JUST try to copy the solution first after reading the writeup, then look back at the wrtie up
        going over artiel
        we can define T(i,j) as the max points we can get by removing boxes on the sub array boxes[i,j]
        if we take the first boxes the recurrence becomes 1 + T(i+1,j)
        or taking form the end 1 + T(i,j-1)
        but we can also keep removing boxes of the same color in one round and our score goes up k*k,
        k being the number of boxes we removed
        NOTE: for this problem, the sub problem is not 'entirely self contained' (i.e we need more than just left and right pointers, for this case at least)
        we need to add more info for the sub problem to be self contained
        currently if we have the remaining subarray[m,j], we don't know how many boxes of the same color as boxes[m] on its left
        we redefine T(i,j,k) as the max number of points possible removing tghe boxes in [i,j], WITH k boxes to the left of boxes[i] and having same color as boxes[i]
        1. Inovocation: 
            T(0,n-1,0)
        2. termination:
            T(i,i-1,k) = 0, no boxes so no points
            T(i,i,k) = (k+1)*(k+1), we've taking k boxes, to the left of boxes[i]
        3. recurrence:
            a.  If we remove boxes[i] first, we get (k + 1) * (k + 1) + T(i + 1, j, 0)
            b.  If we decide to attach boxes[i] to some other box of the same color, say boxes[m], then from our analyses above, the total points will be T(i + 1, m - 1, 0) + T(m, j, k + 1), where for the first term, since there is no attached boxes for subarray boxes[i + 1, m - 1], we have k = 0 
            c. But we are not done yet. What if there are multiple boxes of the same color as boxes[i] within subarray boxes[i + 1, j]? We have to try each of them and choose the one that yields the maximum points. Therefore the final answer for this case will be: max(T(i + 1, m - 1, 0) + T(m, j, k + 1)) where i < m <= j && boxes[i] == boxes[m]
        '''
        
        @lru_cache(maxsize=None)
        def dp(i,j,k):
            if i > j:
                return 0
            if i == j:
                return (k+1)*(k+1) #if we have taken k boxes of the same color, we can keep taking of the same color from i+1 and from j-1
            #starting with how many continutous of same color
            while i < j and boxes[i+1] == boxes[i]:
                k += 1
                i += 1
            
            max_val = (k+1)*(k+1) + dp(i+1,j,0)
            
            for m in range(i+1,j+1):
                if boxes[m] == boxes[i]:
                    points = dp(i+1,m-1,0) + dp(m,j,k+1)
                    max_val = max(max_val,points)
            return max_val
        
        return dp(0,len(boxes)-1,0) 

#bottom up iterative
class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        '''
        iterative, bottom up dp
        '''
        N = len(boxes)
        dp = [[[0]*N for _ in range(N)] for _ in range(N)]
        
        for j in range(N):
            for k in range(j+1):
                dp[j][j][k] = (k+1)*(k+1)
                
        for l in range(1,N):
            for j in range(l,N):
                i = j - l
                for k in range(0,i+1):
                    res = (k+1)*(k+1) + dp[i+1][j][0]
                    for m in range(i+1,j+1):
                        if boxes[m] == boxes[i]:
                            res = max(res,dp[i+1][m-1][0]+dp[m][j][k+1])
                    dp[i][j][k] = res
                    
        return 0 if N == 0 else dp[0][N-1][0]

#############################
#   Minimum Window Substring
#############################
#TLE if i just compare counts
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        '''
        we want the minimum size substring in s, that contains all chars in t
        brute force would be to check all substrings s, check if they contain all chars in t
        and return the min
        the counts of chars must also be correct
        '''
        count_t = Counter(t)
        N = len(s)
        min_length = float('inf')
        output = ""
        
        def compare_counts(count):
            for k,v in count_t.items():
                if k not in count or count[k] < v:
                    return False
            return True
        for size in range(N):
            for i in range(N-size):
                substring = s[i:i+size+1]
                count_sub = Counter(substring)
                if compare_counts(count_sub) == True:
                    if len(substring) < min_length:
                        min_length = len(substring)
                        output = substring
        
        return output


###########################
# Minimum Window Substring
###########################
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        '''
        we are looking for the smallest substring in s that contains all chars in t
        starting off i know that all of s should contain t, if it doens't return""
        now can we make s smaller?
        we can examine all substrings for lengths 1 to len(s)
        and just update as we go
        N = len(s)
        M = windows of length N
        O(N*M)
        '''
        count_t = Counter(t)
        N = len(s)
        min_length = float('inf')
        output = ""
        
        def compare_counts(count):
            for k,v in count_t.items():
                if k not in count or count[k] < v:
                    return False
            return True
        for size in range(N):
            for i in range(N-size):
                substring = s[i:i+size+1]
                count_sub = Counter(substring)
                if compare_counts(count_sub) == True:
                    if len(substring) < min_length:
                        min_length = len(substring)
                        output = substring
        
        return output


class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        '''
        we can use a sliding window approach to solve the problem
        use left and right pointers whose jobs is to expains the current window 
        and then we ahve the left pointer who's job is to make it smaller
        at any point in time only one of these pointers move and the other remains fixed
        algo:
            1. we start with two pointers, left and right on first element
            2. use right to exapand the window until we get a desiable window
            3. once we have window with all chars windo in t, we move left to see if we can contract it
            4. if window is still deriable we keep on updatin the minimum
            5. after conracting to as small as we can, we move the right until we satisfy the criteria
            6. move left again
            7. right keeps going until it reaches the end
        '''
        if not t or not s:
            return ""
        N = len(s)
        count_t = Counter(t)
        #need to keep track of items made for count_t
        needed = len(count_t)
        created = 0
        window_counts = {}
        #pointers
        left,right = 0,0
        #result objects
        output = ""
        min_length = float('inf')
        
        #sliding 
        while right < N:
            char = s[right]
            if char in window_counts:
                window_counts[char] += 1
            else:
                window_counts[char] = 1
            if char in count_t and window_counts[char] == count_t[char]:
                created += 1

            #as we ar expaning we need to see if we can contract
            while left <= right and created == needed:
                char = s[left]
                #result update
                if right - left + 1 < min_length:
                    min_length = right - left + 1
                    output = s[left:right+1]
                window_counts[char] -= 1
                if char in count_t and window_counts[char] < count_t[char]:
                    created -= 1
                left += 1
            
            right += 1
        return output
        
#optimized
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        '''
        we can opitmize, in the case where len(s) >>> len(t)
        or, filtered(s) <<< s
        we create a filter list s, which has all (index,char) for all char in t
        '''
        if not t or not s:
            return ""

        dict_t = Counter(t)

        required = len(dict_t)

        # Filter all the characters from s into a new list along with their index.
        # The filtering criteria is that the character should be present in t.
        filtered_s = []
        for i, char in enumerate(s):
            if char in dict_t:
                filtered_s.append((i, char))

        l, r = 0, 0
        formed = 0
        window_counts = {}

        ans = float("inf"), None, None

        # Look for the characters only in the filtered list instead of entire s. This helps to reduce our search.
        # Hence, we follow the sliding window approach on as small list.
        while r < len(filtered_s):
            character = filtered_s[r][1]
            window_counts[character] = window_counts.get(character, 0) + 1

            if window_counts[character] == dict_t[character]:
                formed += 1

            # If the current window has all the characters in desired frequencies i.e. t is present in the window
            while l <= r and formed == required:
                character = filtered_s[l][1]

                # Save the smallest window until now.
                end = filtered_s[r][0]
                start = filtered_s[l][0]
                if end - start + 1 < ans[0]:
                    ans = (end - start + 1, start, end)

                window_counts[character] -= 1
                if window_counts[character] < dict_t[character]:
                    formed -= 1
                l += 1    

            r += 1    
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]

#################################
# Range Sum Query - Immutable
##############################
class NumArray:
    '''
    do as the problem says and sum
    '''

    def __init__(self, nums: List[int]):
        self.nums = nums
        

    def sumRange(self, left: int, right: int) -> int:
        res = 0
        for i in range(left,right+1):
            res += self.nums[i]
        return res
        


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)

#now using prefix sum
class NumArray:
    '''
    using pref sum
    '''

    def __init__(self, nums: List[int]):
        N = len(nums)
        self.pref = [0]*(N+1)
        for i in range(1,len(self.pref)):
            self.pref[i] = self.pref[i-1] + nums[i-1]

    def sumRange(self, left: int, right: int) -> int:
        return self.pref[right+1] - self.pref[left]

##################################
# Count Good Nodes in Binary Tree
##################################
#YASSS!
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        '''
        a node is a good node if in the path from root to node, there are no nodes greater than X
        we can use dfs to keep track of max, and if a node is a least a max, its good
        
        '''
        self.good = 0
        self.max = root.val
        
        def dfs(node,curr_max):
            if not node:
                return
            if node.val >= curr_max:
                self.good += 1
                curr_max = max(self.max,node.val)
            dfs(node.left,curr_max)
            dfs(node.right,curr_max)
            
        dfs(root,self.max)
        return self.good

#iterative
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        '''
        using stack
        '''
        good_nodes = 0
        stack = [(root,root.val)]
        while stack:
            curr_node,curr_max = stack.pop()
            if curr_node.val >= curr_max:
                good_nodes += 1
                curr_max = max(curr_max,curr_node.val)
            if curr_node.left:
                stack.append((curr_node.left,curr_max))
            if curr_node.right:
                stack.append((curr_node.right,curr_max))
        
        return good_nodes

#NOTE, the order in which we traverse the nodes does not matter, BFS implementation is similar to DFS

########################
# Decode Ways
########################
class Solution:
    def numDecodings(self, s: str) -> int:
        '''
        neeed to use recursion
        i can either take a single digit and it can be decoded
        or i can take two digits and at can be decoded if in between 0 and 26
        we can return a valid count if after getting to end or end - 1
        if we had a function rec(i) that gave us the number of of wasys at i
        then we would need do add to that the number of ways i+2
        rec(i) = rec(i+1) + rec(i+2) if i:i+2 is valid
        we either take 1 or take 2
        '''
        memo = {}
        
        def rec(i):
            #got to the end
            if i == len(s):
                return 1
            #terminate since we cant decode a zero
            if s[i] == '0':
                return 0
            #now check if second from end
            if i == len(s)-1:
                return 1
            if i in memo:
                return memo[i]
            
            ans = rec(i+1)
            if int(s[i:i+2]) <= 26:
                ans += rec(i+2)
            memo[i] = ans
            return ans
        
        return rec(0)

#iterative dp, bottom up
class Solution:
    def numDecodings(self, s: str) -> int:
        '''
        iterartive dp
        dp[i] answers the question, how many valid ways can i decode up to i
        base cases empty strin is just zero
        string length 1, is 1, if not zero
        then use transision
        dp[i] = dp[i-1] + dp[i-2] is i:i-2 is valid
        '''
        N = len(s) 
        dp = [0]*(N+1)
        dp[0] = 1
        #length 1 string
        dp[1] = 0 if s[0] == '0' else 1
        for i in range(2,len(dp)):
            #check if digit is possible 
            if s[i-1] != '0':
                dp[i] = dp[i-1]
            #now check the two digit case
            two = int(s[i-2:i])
            if 10 <= two <= 26:
                dp[i] += dp[i-2]
        return dp[-1]

###################
#Paint House II
####################
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        first think of some edge cases, and what inputs are allowed, always be mindful of what is being brought in
        as input, example, if k= 1, all n houses must be the same color, which does not meet the criteria
        if k = 2, then we kno the prbolem is alwats soleable, we can always avoid adjacent houses being painted the same color
        base case, is when we hit the last house, which can just be pulled from the costs table
        recursive case:
        
        '''
        memo = {}
        N = len(costs)
        #base case
        if N == 0:
            return 0
        k = len(costs[0])
       
        def paint(n,color):
            if (n,color) in memo:
                return memo[(n,color)]
            if n == N - 1:
                return costs[n][color]
            #we want to find the min costs
            total_cost = float('inf')
            for next_color in range(k):
                if next_color == color: #remember cannot be
                    continue
                total_cost = min(total_cost,paint(n+1,next_color))
            #add to the previous cost   
            total_cost += costs[n][color]
            memo[(n,color)] = total_cost
            return total_cost
        
        min_cost = float('inf')
        for color in range(k):
            min_cost = min(min_cost,paint(0,color))
        return min_cost

#dp, in place, similar to last problem
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        now imagine looking at the costs, the rows are the costs to paint ith house, the jth color
        pick one from each row, such that the sum of those numbers is minized,
        becasue 2 adjacent houses cannot the same color, adjacent rows, must be pick from different columrns
        The way that we solve it is to iterate over the cells and determine what the cheapest way of getting to that cell is. We'll work from top to bottom.
        for each i,j cell, find the minimum costs
        '''
        #special cases
        N = len(costs)
        if N == 0:
            return 0
        k = len(costs[0])
        
        #starting froms second
        for house in range(1,N):
            for color in range(k):
                min_cost = float('inf')
                for next_color in range(k):
                    if next_color != color:
                        min_cost = min(min_cost,costs[house-1][next_color])
                #add this the the one we are one
                costs[house][color] += min_cost
        
        #we want the minimum in the last rows
        return min(costs[-1])

#dp additional space
#in case we were not allowed to modify input
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        using up sapce
        '''
        #special cases
        N = len(costs)
        if N == 0:
            return 0
        k = len(costs[0])
        prev_row = costs[0]
        
        #starting froms second
        for house in range(1,N):
            curr_row = [0]*k
            for color in range(k):
                min_cost = float('inf')
                for next_color in range(k):
                    if next_color != color:
                        min_cost = min(min_cost,prev_row[next_color])
                #add this the the one we are one
                curr_row[color] += costs[house][color] + min_cost
            prev_row = curr_row
        
        #we want the minimum in the last rows
        return min(prev_row)

#optimized O(nk)
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        we cam optimize from O(k*n^2) to O(n*k)
        this is just one of those things you either know or don't
        we don;t need to looks at the entires previous row for every cell!
        we really only need to add the first min and then the second min
        we can modify the input in place
        '''
        n = len(costs)
        if n == 0: return 0
        k = len(costs[0])

        for house in range(1, n):
            # Find the colors with the minimum and second to minimum
            # in the previous row.
            min_color = second_min_color = None
            for color in range(k):
                cost = costs[house - 1][color]
                if min_color is None or cost < costs[house - 1][min_color]:
                    second_min_color = min_color
                    min_color = color
                elif second_min_color is None or cost < costs[house - 1][second_min_color]:
                    second_min_color = color
            # And now update the costs for the current row.
            for color in range(k):
                if color == min_color:
                    costs[house][color] += costs[house - 1][second_min_color]
                else:
                    costs[house][color] += costs[house - 1][min_color]

        #The answer will now be the minimum of the last row.
        return min(costs[-1])


#########################################
#Maximum Product of Splitted Binary Tree
###########################################
#nice try
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        '''
        we want to split the tree in two, such that the product of the sums of the tree is maximized
        if i had a recursive function that returns the sum of the subtree
        the answer is just max((total_sum - sub_tree_sum)*sub_tree_sum)
        pass tree once to get total sum
        '''
        MOD = 10**9 + 7
        self.total_sum = 0
        self.max_product = 0
        #function to get sum
        def total_sum(node):
            if not node:
                return
            self.total_sum += node.val
            total_sum(node.left)
            total_sum(node.right)
            
        #now we get sums for each subtree
        def split_tree(node,curr_sum):
            if not node:
                return
            curr_sum += node.val
            self.max_product = max(self.max_product, (self.total_sum - curr_sum)*curr_sum)
            self.max_product %= MOD
            split_tree(node.left,0)
            split_tree(node.right,0)
        
        total_sum(root)
        split_tree(root,0)
        return self.max_product

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        '''
        we want to split the tree in two, such that the product of the sums of the tree is maximized
        if i had a recursive function that returns the sum of the subtree
        the answer is just max((total_sum - sub_tree_sum)*sub_tree_sum)
        pass tree once to get total sum
        we can define the sum of a subtree as
        rec_sum(node) = rec_sum(node.left) + rec_sum(node.right) + node.val
        for each node, find the subtree sums
        also use the function ti find the total sum at the end
        '''
        MOD = 10**9 + 7
        all_sum = []
        max_product = 0
        
        def subTreeSum(node):
            if not node:
                return 0
            left_sum = subTreeSum(node.left)
            right_sum = subTreeSum(node.right)
            ans = left_sum + right_sum + node.val
            all_sum.append(ans)
            return ans
        
        totalSum = subTreeSum(root)
        for SUM in all_sum:
            max_product = max(max_product, (totalSum - SUM)*SUM)
        return max_product % MOD

#two pass
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        '''
        we can also do this in two passes, first find the sum of the whole tree
        then find the max productrs at each node
        '''
        self.total_sum = 0
        self.max_product = 0
        MOD = 10**9 + 7
        
        def treeSum(node):
            if not node:
                return 0
            left = treeSum(node.left)
            right = treeSum(node.right)
            return left + right + node.val
        
        def max_product(node):
            if not node:
                return 0
            left = max_product(node.left)
            right = max_product(node.right)
            subTreeSum = left + right + node.val
            self.max_product = max(self.max_product,(self.total_sum-subTreeSum)*subTreeSum)
            return subTreeSum
        
        self.total_sum = treeSum(root)
        max_product(root)
        return self.max_product % MOD

#######################
# Valid Sudoku
#######################
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        '''
        not really sure how to solve this problem, lets go over some of the solutions
        we can hash each cell by row and col (i,j) indexing and check if the number already exists in this row or colr
        we can also get the box hash by taking (i//3)*3+3(j//3), turns out all (i,j) in a box mapp to q unique number
        or we could use a tuple (i//3,i//3) and do a hash lookup correspdoing to a box
        note, we are only checking if the board is valid, not solveale 
        so we just need to check duplicates along row,col, and box
        algo:
            use hash sets row each row,col,and box key, then check if the current val resides in any of these
            false if it does, otherwise add
            if we get to the end return true
        '''
        N = 9
        
        rows = [set() for _ in range(N)]
        cols = [set() for _ in range(N)]
        boxes = [set() for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                val = board[i][j]
                if val == '.':
                    continue
                #check row
                if val in rows[i]:
                    return False
                rows[i].add(val)
                
                #check col
                if val in cols[j]:
                    return False
                cols[j].add(val)
                
                #check box
                box_idx = (i//3)*3 + (j//3)
                if val in boxes[box_idx]:
                    return False
                boxes[box_idx].add(val)
            
        return True

class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        '''
        instead of using a hash set for each row,column,and box
        we can represent it each as an array of 0's and 1's of lenght N
        and we just check if this number has been taking up
        '''
        N = 9
        rows = [[0]*N for _ in range(N)]
        cols = [[0]*N for _ in range(N)]
        boxes = [[0]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                if board[i][j] == ".":
                    continue
                position = int(board[i][j]) - 1
                
                if rows[i][position] == 1:
                    return False
                rows[i][position] = 1
                
                if cols[j][position] == 1:
                    return False
                cols[j][position ] = 1
                
                box_idx = (i//3)*3 + (j//3)
                if boxes[box_idx][position] == 1:
                    return False
                boxes[box_idx][position] = 1
            
        return True

#bit masking to reduce space
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        '''
        was can use bit masking to solve this problem to save space
        use binary number to repersent state or row,col,box
        '''
        N = 9
        rows = [0]*N
        cols = [0]*N
        boxes = [0]*N
        
        for i in range(N):
            for j in range(N):
                if board[i][j] == ".":
                    continue
                position = int(board[i][j]) - 1
                
                if rows[i] & (1 << position) == 1:
                    return False
                rows[i] = rows[i] | (1 << position)
                
                if cols[j] & (1 << position) == 1:
                    return False
                cols[j] = cols[j] | (1 << position)
                
                box_idx = (i//3)*3 + (j//3)
                if boxes[box_idx] & (1 << position) == 1:
                    return False
                boxes[box_idx] = boxes[box_idx] | (1 << position)
            
        return True

#########################
# Sudoku Solver
#########################
#using bit shifts
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        we can backtrack, otherwise called contraint programming
        algo:
            start from upper left and place numbers
            make sure to check we can place a number
            when we can' get to lower left, we backtrack
            we can return true when we have reached the end of the board (the lower left)
        use bitwise trick to reset rows,cols,and boxes placement
        '''
        #box size
        n = 3
        #board size 
        N = n*n
        #anon function to get box idx on flow
        box_index = lambda row,col: (row // n)*n + (col//n)
        rows = [0]*N
        cols = [0]*N
        boxes = [0]*N
        self.sudoku_solved = False
        
        def could_place(d,row,col):
            #remember to offset when checking using bitwise
            can_rows = rows[row] & (1 << d-1)
            can_cols = cols[col] & (1 << d-1)
            can_boxes = boxes[box_index(row,col)] & (1 << d -1)
            return not (can_rows or can_cols or can_boxes)
        
        def place_number(d,row,col):
            rows[row] |= (1 << d - 1)
            cols[col] |= (1 << d - 1)
            boxes[box_index(row,col)] |= (1 << d - 1)
            board[row][col] = str(d)
        
        def remove_number(d,row,col):
            rows[row] &= ~(1 << d - 1)
            cols[col] &= ~(1 << d - 1)
            boxes[box_index(row,col)] &= ~(1 << d - 1)
            
        def place_next_numbers(row,col):
            #at the end of the board
            if row == N -1 and col == N -1:
                self.sudoku_solved = True
            else:
                #end of row, on to next col
                if col == N - 1:
                    backtrack(row+1,0)
                else:
                    backtrack(row,col+1)
                    
        def backtrack(row,col):
            #if we need to place
            if board[row][col] == ".":
                for d in range(1,10):
                    if could_place(d,row,col):
                        place_number(d,row,col)
                        place_next_numbers(row,col)
                        #if we have solved, we can terminate the backtrack, otherwise remove number
                        if not self.sudoku_solved:
                            remove_number(d,row,col)
            else:
                place_next_numbers(row,col)
                
        for i in range(N):
            for j in range(N):
                if board[i][j] != '.':
                    d = int(board[i][j])
                    place_number(d,i,j)
        backtrack(0,0)
                    
#using hash set
class Solution:
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        def could_place(d, row, col):
            """
            Check if one could place a number d in (row, col) cell
            """
            return not (d in rows[row] or d in columns[col] or \
                    d in boxes[box_index(row, col)])
        
        def place_number(d, row, col):
            """
            Place a number d in (row, col) cell
            """
            rows[row][d] += 1
            columns[col][d] += 1
            boxes[box_index(row, col)][d] += 1
            board[row][col] = str(d)
            
        def remove_number(d, row, col):
            """
            Remove a number which didn't lead 
            to a solution
            """
            del rows[row][d]
            del columns[col][d]
            del boxes[box_index(row, col)][d]
            board[row][col] = '.'    
            
        def place_next_numbers(row, col):
            """
            Call backtrack function in recursion
            to continue to place numbers
            till the moment we have a solution
            """
            # if we're in the last cell
            # that means we have the solution
            if col == N - 1 and row == N - 1:
                nonlocal sudoku_solved
                sudoku_solved = True
            #if not yet    
            else:
                # if we're in the end of the row
                # go to the next row
                if col == N - 1:
                    backtrack(row + 1, 0)
                # go to the next column
                else:
                    backtrack(row, col + 1)
                
                
        def backtrack(row = 0, col = 0):
            """
            Backtracking
            """
            # if the cell is empty
            if board[row][col] == '.':
                # iterate over all numbers from 1 to 9
                for d in range(1, 10):
                    if could_place(d, row, col):
                        place_number(d, row, col)
                        place_next_numbers(row, col)
                        # if sudoku is solved, there is no need to backtrack
                        # since the single unique solution is promised
                        if not sudoku_solved:
                            remove_number(d, row, col)
            else:
                place_next_numbers(row, col)
                    
        # box size
        n = 3
        # row size
        N = n * n
        # lambda function to compute box index
        box_index = lambda row, col: (row // n ) * n + col // n
        
        # init rows, columns and boxes
        rows = [defaultdict(int) for i in range(N)]
        columns = [defaultdict(int) for i in range(N)]
        boxes = [defaultdict(int) for i in range(N)]
        for i in range(N):
            for j in range(N):
                if board[i][j] != '.': 
                    d = int(board[i][j])
                    place_number(d, i, j)
        
        sudoku_solved = False
        backtrack()

#just another way
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        we can backtrack, otherwise called contraint programming
        algo:
            start from upper left and place numbers
            make sure to check we can place a number
            when we can' get to lower left, we backtrack
            we can return true when we have reached the end of the board (the lower left)
        use bitwise trick to reset rows,cols,and boxes placement
        '''
        n = len(board)
        rows = [set() for _ in range(n)]
        cols = [set() for _ in range(n)]
        boxes = [set() for _ in range(n)]
        
        for r in range(n):
            for c in range(n):
                if board[r][c] == '.':
                    continue
                v = int(board[r][c])
                rows[r].add(v)
                cols[c].add(v)
                boxes[(r//3)*3 + c//3].add(v)
        
        def is_valid(r, c, v):
            box_id = (r // 3) * 3 + c // 3
            return v not in rows[r] and v not in cols[c] and v not in boxes[box_id]

        
        def backtrack(r, c):
            if r == n - 1 and c == n:
                return True
            elif c == n:
                c = 0
                r += 1

            # current grid has been filled
            if board[r][c] != '.':
                return backtrack(r, c + 1)

            box_id = (r // 3) * 3 + c // 3
            for v in range(1, n + 1):
                if not is_valid(r, c, v):
                    continue

                board[r][c] = str(v)
                rows[r].add(v)
                cols[c].add(v)
                boxes[box_id].add(v)

                if backtrack(r, c + 1):
                    return True

                # backtrack
                board[r][c] = '.'
                rows[r].remove(v)
                cols[c].remove(v)
                boxes[box_id].remove(v)

            return False


        backtrack(0, 0)

##########################
# Rectangle Area II
##########################
#brute force n**3
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        '''
        https://leetcode.com/problems/rectangle-area-ii/discuss/1419181/Python-4-solutions-n3-greater-n2-log-n-greater-n2-greater-n-log-n-explained
        we can use coordintate compression,
        we assign a 'rank' for each x and y val
        then we map these back to a grid to show if that rectanlge is part of the figure
        then we can find its width and its area, if it is part of the figure
        '''
        xs = set()
        ys = set()
        for x1,y1,x2,y2 in rectangles:
            xs.add(x1)
            xs.add(x2)
            ys.add(y1)
            ys.add(y2)
        
        #sort
        xs = sorted(xs)
        ys = sorted(ys)
        
        #assign ranks
        x_ranks = {x:i for i,x in enumerate(xs)}
        y_ranks = {y:i for i,y in enumerate(ys)}
        
        #make grid size of ranks*ranks
        m = len(y_ranks)
        n = len(x_ranks)
        grid = [[0]*m for _ in range(n)]
        
        #now include in figure
        for x1,y1,x2,y2 in rectangles:
            for x in range(x_ranks[x1],x_ranks[x2]):
                for y in range(y_ranks[y1],y_ranks[y2]):
                    grid[x][y] = 1
        area = 0
        for x in range(n-1):
            for y in range(m-1):
                width = xs[x+1] - xs[x]
                height = ys[y+1] - ys[y]
                area += grid[x][y]*width*height
        
        return area % (10**9 + 7)

#O(N^2 log N)
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        '''
        we can use the sweep line algorithm, we sort rectable by their x coord
        the move vertial line left to right
        then it becomes similar to merge intervals, we also need to keep active set of sides and remove them or add to it
        https://leetcode.com/problems/rectangle-area-ii/discuss/1419181/Python-4-solutions-n3-greater-n2-log-n-greater-n2-greater-n-log-n-explained
        '''
        def merge(intervals):
            #return sum of intervals
            ans = []
            for beg,end in sorted(intervals):
                if not ans or ans[-1][1] < beg:
                    ans += [[beg,end]]
                else:
                    ans[-1][1] = max(ans[-1][1],end)
            return sum(j-i for i,j in ans)
        
        sides_left = [(x1,0,y1,y2) for x1,y1,x2,y2 in rectangles]
        sides_right = [(x2,1,y1,y2) for x1,y1,x2,y2 in rectangles]
        #we can get all the events
        sides = sorted(sides_left + sides_right)
        
        intervals = []
        ans = 0
        prev_x = sides[0][0]
        
        
        for x, op_cl, y1, y2 in sides:
            ans += merge(intervals) * (x - prev_x)
            
            if op_cl == 0:
                intervals.append((y1,y2))
            else:
                intervals.remove((y1,y2))     
            prev_x = x
            
        return ans % (10**9 + 7)

#optimizing merge to N
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        '''
        we can do merge interal in N time instead of N^2
        create all unique y coords first, then we create corrrespondes between indexes and these y croods
        finallyt count how many times each segment is covered
        we again creates dies like in last approach
        then we go through sides and update area, but when we meet a new segment, we update count
        +1 for rect -1 for end
        then we update our y cur sum and check if count is zero, and if it is more than 0 add lengto this segment
        '''
        ys = set()
        for x1,y1,x2,y2 in rectangles:
            ys.add(y1)
            ys.add(y2)
        ys = sorted(ys)
        y_i = {v:i for i,v in enumerate(ys)}
        count = [0]*(len(y_i))
        
        #same as last problem
        sides_lft = [(x1,-1,y1,y2) for x1,y1,x2,y2 in rectangles]
        sides_rgh = [(x2,1,y1,y2) for x1,y1,x2,y2 in rectangles]
        sides = sorted(sides_lft + sides_rgh)
        
        cur_x = cur_y_sum = area = 0
        for x, op_cl, y1, y2 in sides:
            area += (x - cur_x) * cur_y_sum
            cur_x = x
            for i in range(y_i[y1], y_i[y2]):
                count[i] += op_cl
            cur_y_sum = sum(y2 - y1 if c else 0 for y1, y2, c in zip(ys, ys[1:], count))
        return area % (10 ** 9 + 7)

#segment tree solution
'''
There is also O(n log n) solution, using segment trees. 
The idea is to use segment tree where we need to deal only with updates of ranges. 
If we need only range updats, and not queries, not need to do lazy updates - see https://cp-algorithms.com/data_structures/segment_tree.html, (Addition on segments) chapter.
Also we need to need to keep self.total dictionary, which for each segment will calculate lenght of all active segments we have so far.
'''
class SegmentTree:
    def __init__(self, xs):
        self.cnts = defaultdict(int)
        self.total = defaultdict(int)
        self.xs = xs

    def update(self, v, tl, tr, l, r, h):
        if l > r: return
        if l == tl and r == tr:
            self.cnts[v] += h
        else:
            tm = (tl + tr)//2
            self.update(v*2, tl, tm, l, min(r, tm), h)
            self.update(v*2+1, tm+1, tr, max(l, tm+1), r, h)
          
        if self.cnts[v] > 0:
            self.total[v] = self.xs[tr + 1] - self.xs[tl]
        else:
            self.total[v] = self.total[v*2] + self.total[v*2+1]
        return self.total[v]
    
class Solution:
    def rectangleArea(self, rectangles):
        xs = sorted(set([x for x1, y1, x2, y2 in rectangles for x in [x1, x2]]))
        xs_i = {x:i for i, x in enumerate(xs)}

        STree = SegmentTree(xs)
        L = []
        for x1, y1, x2, y2 in rectangles:
            L.append([y1, 1, x1, x2])
            L.append([y2, -1, x1, x2])
        L.sort()

        cur_y = cur_x_sum = area = 0
        
        for y, op_cl, x1, x2 in L:
            area += (y - cur_y) * cur_x_sum
            cur_y = y
            STree.update(1, 0,  len(xs) - 1, xs_i[x1], xs_i[x2]-1, op_cl)
            cur_x_sum = STree.total[1]
            
        return area % (10 ** 9 + 7)

####################
# Two Sum IV - Input is a BST
#####################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        '''
        naive way would be to traverse the tree once and get all node values
        then tree it like two sum
        warm up with this
        '''
        nodes = set()
        
        def dfs(node):
            if not node:
                return
            nodes.add(node.val)
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        nodes = {k:i for i,k in enumerate(nodes)}
        for n in nodes:
            if k - n in nodes:
                if nodes[k-n] != nodes[n]:
                    return True
        return False

class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        '''
        single pass, traverse tree in both directions and keep track of elements we have seen so far
        for every node, check if k - node.val exists in hashset
        
        '''
        seen = set()
        
        def dfs(node):
            if not node:
                return False
            if k - node.val in seen:
                return True
            seen.add(node.val)
            return dfs(node.left) or dfs(node.right)
        
        return dfs(root)

class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        '''
        single pass, traverse tree in both directions and keep track of elements we have seen so far
        for every node, check if k - node.val exists in hashset
        
        '''
        seen = set()
        stack = [root]
        
        while stack:
            if stack[-1] is not None:
                node = stack.pop()
                if k - node.val in seen:
                    return True
                seen.add(node.val)
                stack.append(node.right)
                stack.append(node.left)
            #pop off the empty node
            else:
                stack.pop()
        
        return False

class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        '''
        we  could just do inorder traversal and two pointer technique with 2sum
        '''
        nodes = []
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            nodes.append(node.val)
            inorder(node.right)
        inorder(root)
        
        l,r = 0,len(nodes) - 1
        while l < r:
            curr_sum = nodes[l] + nodes[r]
            if curr_sum == k:
                return True
            if curr_sum < k:
                l += 1
            else:
                r -= 1
        return False

#######################
# Graph Valid Tree
#######################
#close one...
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        '''
        this is just a cycle detection problem
        if there is a cycle, it cannot be a valid tree
        0 means not visited
        1 means visited along a path
        2 means already visited
        '''
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        
        def dfs(node,visited):
            if visited[node] == 1:
                return False
            elif visited[node] == 2:
                return True
            else:
                visited[node] = 1
                for neigh in adj_list[node]:
                    if not dfs(neigh,visited):
                        return False
                visited[node] = 2
                return True
        for i in range(n):
            if dfs(i,[0]*n):
                return False
        return True

#the problem is that this is undirected
#just see if one of the neighbors came back from the node we were currentl visiting
#and also record the number of nodes in seen
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        '''
        this is just a cycle detection problem
        if there is a cycle, it cannot be a valid tree
        we need to becare to allow for trivial cycles A to B back A
        so we just need to make sure, we go over an EDGE only once
        when we go along an edge, we should do something to ensyre we dont go back along this edge in the opposiut direciton
        or we can can keep track of the parent node, where we initally came from
        finally we also need to check that the components are connectd, since it must be a valid tree
        '''
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        seen = set()
        def dfs(node,parent):
            if node in seen:
                return
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh == parent:
                    continue
                if neigh in seen:
                    return False
                result = dfs(neigh,node)
                if not result:
                    return False
            return True
        
        return dfs(0,-1) and len(seen) == n

#iterative version
#note, bfs is the same thing here with dfs
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        parent = {0:-1}
        stack = [0]
        
        while stack:
            node = stack.pop()
            for neigh in adj_list[node]:
                if neigh == parent[node]:
                    continue
                if neigh in parent:
                    return False
                parent[neigh] = node
                stack.append(neigh)


        return len(parent) == n

#using graph property of tree
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        '''
        for a graph to be a valid tree, it must have n-1 edges
        any less it would have unconnected components, any more it would haave a cycle
        iff graphy is fully connect and has n-1 edges it cannot contain a cycle
        algo:
            check whether or not there are n-1 edges, not not return false
            check if cullconnected
        really we are just checking if we can touch all nodes, only after we have seen that there are
        n-1 edges
        '''
        if len(edges) != n - 1:
            return False
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        seen = set()
        
        def dfs(node):
            if node in seen:
                return
            seen.add(node)
            for neigh in adj_list[node]:
                dfs(neigh)
        dfs(0)
        return len(seen) == n

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        '''
        for a graph to be a valid tree, it must have n-1 edges
        any less it would have unconnected components, any more it would haave a cycle
        iff graphy is fully connect and has n-1 edges it cannot contain a cycle
        algo:
            check whether or not there are n-1 edges, not not return false
            check if cullconnected
        really we are just checking if we can touch all nodes, only after we have seen that there are
        n-1 edges
        '''
        if len(edges) != n - 1:
            return False
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        seen = set()
        stack = [0]
        
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for neigh in adj_list[node]:
                stack.append(neigh)
                
        return len(seen) == n

#using UF
class UF:
    def __init__(self,n):
        self.parent = [node for node in range(n)]
        self.size = [1]*n
    def find(self,A):
        root = A
        while root != self.parent[root]:
            root = self.parent[root]
        while A != root:
            old_root = self.parent[A]
            self.parent[A] = root
            A = old_root
        return root
    def union(self,A,B):
        #returns true if we can merge or false if cannot
        root_A = self.find(A)
        root_B = self.find(B)
        #if in the same set
        if root_A == root_B:
            return False
        #sent parents to larger guys
        # We want to ensure the larger set remains the root.
        if self.size[root_A] < self.size[root_B]:
            # Make root_B the overall root.
            self.parent[root_A] = root_B
            # The size of the set rooted at B is the sum of the 2.
            self.size[root_B] += self.size[root_A]
        else:
            # Make root_A the overall root.
            self.parent[root_B] = root_A
            # The size of the set rooted at A is the sum of the 2.
            self.size[root_A] += self.size[root_B]
        return True

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        '''
        we can use union find on the edges, and try to connect the whole thing
        if we can't, must return false, why?
        because we were adding and edge to compoenents that were already connected
        we can use path compression to make it faster
        '''
        if len(edges) != n-1:
            return False
        uf = UF(n)
        for a,b in edges:
            if not uf.union(a,b):
                return False
        return True

##################################
# Complex Number Multiplication
################################
#close one
class Solution:
    def complexNumberMultiply(self, num1: str, num2: str) -> str:
        '''
        define function to realy and imigiary parts of numbers
        define function to carrry out multiple
        img part will alwasy have contacnt >= 0
        '''
        def getParts(num):
            num = num.split("+")
            return [num[0],num[1]]
        
        def multiply(a,b):
            #i can always multiply the real parts
            real = int(a[0])*int(b[0])
            #now imaginary part
            imag = None
            if '-' in a[1] and '-' in b[1]:
                #its going to be negative
                imag = -1*int(a[1][1])*int(b[1][1])
            elif '-' in a[1]
                imag = int(a[1][1])*int(b[1][1])
            return [real,imag]
        
        num1 = getParts(num1)
        num2 = getParts(num2)
        print(multiply(num1,num2))

class Solution:
    def complexNumberMultiply(self, num1: str, num2: str) -> str:
        '''
        define function to realy and imigiary parts of numbers
        define function to carrry out multiple
        the real part is just:
            (real part num1)*(real part num2) - (imag part num1)*(imag part num2)
        imag part is just:
            (real part num1)*(imag part num2) + (real part num2)*(imag part num1)
        '''
        def get_parts(num):
            num = num.split("+")
            return int(num[0]),int(num[1][:-1])
        
        num1_real,num1_imag = get_parts(num1)
        num2_real,num2_imag = get_parts(num2)
        real = num1_real*num2_real - num1_imag*num2_imag
        imag = num1_real*num2_imag + num2_real*num1_imag
        return "{}+{}i".format(real,imag)

######################
# Sum of Square Numbers
##################### 
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''
        generate all square numbers less than sqrt(2**31 - 1) into set
        then loop up to sqrt(2**31 -1), for each num in the loop c - num
        '''
        squares = set()
        i = 0
        while i <= int((c**0.5) + 1):
            squares.add(i*i)
            i += 1
            
        for num in range(int((c**.5)+1)):
            if c - num*num in squares:
                return True
        return False

class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''
        tunrs out n^2 can be written out as the sum of the first n odd numbers
        n^2 = \sum_{i=1}^{n} (2*i-1)
        '''
        a = 0
        while a*a <= c:
            b = c - a*a
            i = 1
            SUM = 0
            while SUM < b:
                SUM += i
                i += 2
            if SUM == b:
                return True
            
            a += 1
        
        return False

class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''        
        instead of checking for all square numbers less then sqrt(c)
        generate all possible candiate up at a*a
        find the sqrt(comp)
        and check if this matches comp
        
        '''
        a = 0
        while a*a <= c:
            b = (c - a*a)**.5
            if b == int(b):
                return True
            a += 1
        
        return False

class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''        
        we can also use binary search to find a numer in the range [0,c-a*a]
        where the mid == c-a*a
        
        '''
        def bin_search(lo,hi,num):
            if lo > hi:
                return False
            mid = lo + (hi - lo) // 2
            if mid*mid == num:
                return True
            if mid*mid > num:
                return bin_search(lo,mid-1,num)
            else:
                return bin_search(mid+1,hi,num)
            
        a = 0
        while a*a <= c:
            b = c - a*a
            if bin_search(0,b,b):
                return True
            a += 1
        return False
        
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''        
        we can also use two pointers the same way we solve two sum
        for nums in range [0,sqrt(c)]
        set left and right pointers to find left*left + right*right == c
        move up left if less 
        else move down right
        
        '''
        left,right = 0, int(c**.5)
        while left <= right:
            curr = left*left + right*right
            if curr == c:
                return True
            if curr < c:
                left += 1
            else:
                right -= 1
        
        return False

#################################################
#  Verify Preorder Serialization of a Binary Tree
#################################################
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        '''
        we can think of a binary tree containing slots, 
        so we just need to fill in the slots as we go along
        initally there is one slot
        coming to a number we need to occupy the slot
        adding a number also consumes a slot and adds two nodes
        # takes up a slot and does not increease nodes
        return if slots == 0
        '''
        preorder = preorder.split(",")
        slots = 1
        for num in preorder:
            #always ue up a slot
            slots -= 1
            #if we even for below 0, its not a valid tree
            if slots < 0:
                return False
            if num != "#":
                slots += 2
        
        return slots == 0

#we can also just keep track of null and non null
#and only fire onf # or not and not a comma
class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        '''
        preorder is node,left,right
        if we are given an inorder traversal, we get the nums in order increasing
        if we are given post order traversal, we get the nums in order decreasing
        if either of these are true return false
        otherwise return true
        '''
        nodes = preorder.split(',')
        slots = 1
        for node in nodes:
            if slots <= 0:
                return False
            if node == "#":
                slots -= 1
            else:
                slots += 1

class Solution:
    def isValidSerialization(self, preorder: str) -> bool:
        '''
        https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/discuss/1427004/Python-simple-stack-explained
        we can tree num # # as being a leaf node, and when we encounter this we can just pop off the leaf and replace with #
        at the end, if this is a valid pre-order, we should only have a # in our stack
        '''
        stack = []
        for num in preorder.split(","):
            stack.append(num)
            #only invoke when we have more than 2 items in stack
            while len(stack) > 2 and stack[-2:] == ["#"]*2 and stack[-3] != "#":
                for _ in range(3):
                    stack.pop()
                stack.append("#")
        
        return stack == ["#"]

#####################################
#  Longest Uncommon Subsequence II
#####################################
class Solution:
    def findLUSlength(self, strs: List[str]) -> int:
        '''
        looks like the inputs are small enough
        do all pairwise comparisions in array checking if word1 is a subsequenc of word2
        if its not update the max length
        i need to compar each word with every other word!
        too slow though
        '''
        def is_sub(w1,w2):
            i = 0
            j = 0
            while i < len(w1) and j < len(w2):
                if w1[i] == w2[j]:
                    i += 1
                j += 1
            return i == len(w1)
        
        mapp = {i:0 for i in range(len(strs))}
        
        for i in range(len(strs)):
            for j in range(len(strs)):
                if i != j:
                    if not is_sub(strs[i],strs[j]):
                        mapp[i] += 1
        longest = -1
        for k,v in mapp.items():
            if v == len(strs) -1:
                longest = max(longest,len(strs[k]))
        return longest
        
class Solution:
    def findLUSlength(self, strs: List[str]) -> int:
        '''
        we can just generate all possible subsequences for all the words
        O((num words)*2^(average length of the word))
        we can use the subsets trick to generate the sequences
        put each subsequence into a map, and if this is a unique subsequence, it must be uncomoon to all N-1 strings
        '''
        mapp = {}
        
        for s in strs:
            for i in range(1 << len(s)):
                subseq = ""
                for j in range(len(s)):
                    if ((i >> j) & 1):
                        subseq += s[j]
                if subseq in mapp:
                    mapp[subseq] += 1
                else:
                    mapp[subseq] = 1
                print(subseq)
        
        res = -1
        for k,v in mapp.items():
            if v == 1:
                res = max(res,len(k))
        
        return res

class Solution:
    def findLUSlength(self, strs: List[str]) -> int:
        '''
        you almost had it the first time, 
        we don't need a mapp to check, just check if we've gotten to the end without isubseq firiing
        we can also save some time by ordering the strings decreasingly
        then we just compare the longest string with every smaller string
        '''
        def is_sub(w1,w2):
            i = 0
            j = 0
            while i < len(w1) and j < len(w2):
                if w1[i] == w2[j]:
                    i += 1
                j += 1
            return i == len(w1)
        
        strs.sort(key = lambda x: len(x),reverse = True)
        
        for i in range(len(strs)):
            is_uncommon = True
            for j in range(len(strs)):
                if i == j:
                    continue
                if is_sub(strs[i],strs[j]):
                    is_uncommon = False
                    break
            if is_uncommon:
                return len(strs[i])
            
        return -1

###########################
# Maximum Profit in Job Scheduling 
###########################
#recursion
class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        '''
        we need to schedule jobs so that no two jobs are conlficting
        if we have jobA and jobB, then they are conflicting if end of jobA < start of job B
        or end of jobB < start of job A
        brute force would be to enumerate all possible options to schedule or not, and find max profit
        generating all sequences would be 2^N, then pass to check conflicting
        we can sort by start time, and shedule or take jobs
        if we schedule job[i] then we can only start another job j if start job j > end job i
        look for the next job to schedule only if we can, eith we skip it
        rec(i) = max(rec[i-1]+ profit[i],rec(i))
        https://leetcode.com/problems/maximum-profit-in-job-scheduling/discuss/918804/Python-Top-Down-and-Bottom-Up-DP-7-lines-each
        '''
        #define binsearch helper function to find the left bound of the start time
        def bin_search(arr,target):
            left,right = 0,len(arr)
            while left < right:
                mid = (left + right) // 2
                if arr[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
            return left
        
        #sort the the inputs by start times
        start,end,profits = zip(*sorted(zip(startTime,endTime,profit)))
        
        #for each start time, mapp them to next endtime using binseach
        jump = {i: bisect.bisect_left(start, end[i]) for i in range(len(start))}

        #rec(i) = max(profit[i] + rec(i), rec(i+1))
        memo = {}
        
        def rec(i):
            if i == len(start):
                return 0 #we are done, no more profit
            if i in memo:
                return memo[i]
            take = profits[i] + rec(jump[i])
            no_take = rec(i+1)
            res = max(take,no_take)
            memo[i] = res
            return res
        
        return rec(0)

#dp solution
class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        #sort the the inputs by start times
        start,end,profits = zip(*sorted(zip(startTime,endTime,profit)))
        
        #for each start time, mapp them to next endtime using binseach
        jump = {i: bisect.bisect_left(start, end[i]) for i in range(len(start))}

        #rec(i) = max(profit[i] + rec(i), rec(i+1))
        dp = [0]*(len(start)+1)
        
        for i in range(len(start)-1,-1,-1):
            dp[i] = max(dp[i+1], profits[i] + dp[jump[i]])
        
        return dp[0]

class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        '''
        if i wanted to put binary search into the rec function
        '''
        n = len(startTime)
        jobs = sorted(list(zip(startTime,endTime,profit)))
        startTime = [jobs[i][0] for i in range(n)]
        
        memo = {}
        
        def dp(i):
            if i == n:
                return 0
            if i in memo:
                return memo[i]
            
            no_take = dp(i+1)
            #if we want to take the job, find the next job who's start is just after this current end
            next_job_idx = bisect_left(startTime,jobs[i][1])
            take = jobs[i][2] + dp(next_job_idx)
            res = max(take,no_take)
            memo[i] = res
            return res
        
        return dp(0)

###########################
# Patching Array
###########################
class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        '''
        for any missing number in range [1,n] we need to add a number smaller than or equal to get its sum
        if we have a number miss, the current smallest missing number, then we know [1,miss) must be covered
        example, suppose we have nums [1,2,3,8] and n = 16, can cover [1,6] and [8,14]
        NOTE: increasing sequence 1,2,3, we tack on max sum to 8 to get [8,14]
        we are missing 7,15,16, and if we add numbers > 7, 7 will still be missing
        SUPPOSE the number we want to add is x, then we can cofer [1,miss), and [x,x+miss]
        and since x <= miss, the two ranges should coge [1,x+miss)
        WE want to choose x as large as possible so that the range can cover as larges as possible values
        SO, the best options is x == miss
        after we covered missed, we can recalculate the coverage and see whats the new smallest missing number, we than patche that number
        do repeatedly until there is no missing number
        algo:
            init range [1,miss) = [1,1) = empty
            while n is not covered:
                if the current element nums[i] <= miss:
                    extend range to [1+nums[i] + miss]
                    then increase i by 1
                else:
                    patch the array with miss, i.e range becomes [1,miss+miss]
                    increase patches by 1
        return patches
        '''
        patches = 0
        i = 0
        miss = 1
        while miss <= n:
            if i < len(nums) and nums[i] <= miss: #miss is coered
                miss += nums[i]
                i += 1
            else:
                #we need to patch a number, in this case we extend by miss
                miss += miss
                patches += 1
        return patches

class Solution:
    def minPatches(self, nums: List[int], n: int) -> int:
        '''
        #https://leetcode.com/problems/patching-array/discuss/338621/Python-O(n)-with-detailed-explanation
        init and empty list and keep adding new numbers from nums into this list
        keep updating the coverage range and ensure a continuous coverage range
        * you only need to care about wheterh the newly added number will break the coverage or not
        example:
            * suppose we have coverage for nums [1,10] and next number if 11, we get [1,21]
            which is just 2*coverage+1
            * supppose next number of smaller than 11, like 3, then we just add 3 to coverage [1,13]
            which is just coverage+num
            * if it equals 11, we cannot cover 11, so we add it to get by manually patching it
        '''
        covered = 0
        patches = 0
        i = 0
        while covered < n:
            num = nums[i] if i < len(nums) else float('inf')
            if num > covered + 1: #we need to path, either we miss
                patches += 1
                covered = covered*2 + 1
            else:
                #extend by covered, since we can cover the smallest miss
                covered += num
                i += 1
        return patches
        
#######################
# Equal Tree Partition
#######################
#i think you had this one to be honest, this would be fine in an interview
#make sure to not include the final sum in the list
#otherwise you would have 0 == 0, because we do not want the whole tree sum as a potential node sum
class Solution:
    def checkEqualTree(self, root: Optional[TreeNode]) -> bool:
        '''
        what if i find all the the sums for each node
        then all i would have to do is find the sum of the whole tree
        and then for each node num, call it curr_sum, check if curr_sum == total sum - curr_sum
        return true if this fires
        otherwise return false
        '''
        node_sums = []
        
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            node_sum = left + right + node.val
            node_sums.append(node_sum)
            return node_sum
        
        tree_sum = dfs(root)
        node_sums.pop()
        
        for s in node_sums:
            if tree_sum - s == s:
                return True
        return False
            
class Solution:
    def checkEqualTree(self, root: Optional[TreeNode]) -> bool:
        '''
        we can just check if half the sum of the whole tree is present as any of the subtree sums
        since both parts need to be equal
        '''
        seen = set()
        
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            curr_sum = left + right + node.val
            seen.add(curr_sum)
            return curr_sum
        
        total = dfs(root)
        seen.remove(total)
        return total / 2.0 in seen
        
########################
# Range Addition II
########################
#this gets MLE
class Solution:
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        '''
        increase each cell in in the ops array
        then count them up
        '''
        M = [[0]*n for _ in range(m)]
        for op in ops:
            for i in range(op[0]):
                for j in range(op[1]):
                    M[i][j] += 1
        
        counts = Counter()
        for i in range(m):
            for j in range(n):
                counts[M[i][j]] += 1
        return counts[max(counts)]

class Solution:
    def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
        '''
        we increase the cells by 1 for every operation
        so the cell that gets updated the most would be the maximim one
        the intserction of all ops is given by the box m*n
        where m and n are the min(op_{i}[0]) and min (op_{i}[1]) for all ops
        m*n gives us the multiplicty of the largest number n
        '''
        for x,y in ops:
            m = min(m,x)
            n = min(n,y)
        return m*n

######################################
# Find Minimum in Rotated Sorted Array
######################################
class Solution:
    def findMin(self, nums: List[int]) -> int:
        '''
        the array is sorted, but there is a break now
        if the array were sorted, you would need to returnt he first element
        example
         [4, 5, 6, 7, 0, 1, 2]
         mid is at 7, but we see that mid is greater than left, so the min is on the right
         [0,1,2]
         we go to 1 and see that 0 is less than 1, so it must be on the left
         [0,1], but at this point left has crossed right so we return left
        '''
        l,r = 0,len(nums) -1
        while l < r:
            mid = l + (r - l) // 2
            #min lies on the right
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
        return nums[l]

class Solution:
    def findMin(self, nums: List[int]) -> int:
        '''
        official solution
        notes:
            if last element is greater than  first element, no rotaion, so return first element
        we need to find the inflection point
        the inflection point has the properties
            all element ti left of inflectino point > first element in array
            all elements to righ of inflectino point < first elment in array
        '''
        #special cases
        if len(nums) == 1:
            return nums[0]
        l, r = 0, len(nums)-1
        if nums[r] > nums[0]:
            return nums[0]
        
        #binary search
        while r >= l:
            mid = l + (r - l) // 2
            #if we found inflection point
            if nums[mid] > nums[mid+1]:
                return nums[mid+1]
            if nums[mid-1] > nums[mid]:
                return nums[mid]
            #now we need to find the inflectino point
            if nums[mid] > nums[0]:
                l = mid + 1
            else:
                r = mid - 1
