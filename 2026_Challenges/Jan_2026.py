############################################
# 1411. Number of Ways to Paint N × 3 Grid
# 03JAN26
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
# 04JAN26
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
    
####################################################
# 1458. Max Dot Product of Two Subsequences (REVISTED)
# 08JAN26
####################################################
class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        i can use i,j as a state, but the problem is trying to make sure i have the same length
        '''
        memo = {}

        def dp(i, j):
            if i == len(nums1) or j == len(nums2):
                return float('-inf')

            if (i, j) in memo:
                return memo[(i, j)]

            # skip options
            op1 = dp(i + 1, j)
            op2 = dp(i, j + 1)

            # take both: either start here, or extend
            take = nums1[i] * nums2[j] #could be by itself without taking another part of a subsequence
            op3 = max(
                take,
                take + dp(i + 1, j + 1)
            )

            ans = max(op1, op2, op3)
            memo[(i, j)] = ans
            return ans

        return dp(0, 0)
    
class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        i can use i,j as a state, but the problem is trying to make sure i have the same length
        '''
        memo = {}
        dp = [[0]*(len(nums2) + 1) for _ in range(len(nums1) + 1)]
        #base case fill
        for i in range(len(nums1) + 1):
            for j in range(len(nums2) + 1):
                if i == len(nums1) or j == len(nums2):
                    dp[i][j] = float('-inf')
        
        for i in range(len(nums1)-1,-1,-1):
            for j in range(len(nums2)-1,-1,-1):

                # skip options
                op1 = dp[i + 1][j]
                op2 = dp[i][j + 1]

                # take both: either start here, or extend
                take = nums1[i] * nums2[j] #could be by itself without taking another part of a subsequence
                op3 = max(
                    take,
                    take + dp[i + 1][j + 1]
                )

                ans = max(op1, op2, op3)
                dp[i][j] = ans
        return dp[0][0]
    
############################################
# 2049. Count Nodes With the Highest Score
# 08JAN26
############################################
class Solution:
    def countHighestScoreNodes(self, parents: List[int]) -> int:
        '''
        make the graph
        left is dfs(neigh)
        right is dfs(neigh)
        up is n - 1 - left - right
        '''
        graph = collections.defaultdict(list)
        for node, parent in enumerate(parents):  
            graph[parent].append(node)
        n = len(parents)                         
        d = Counter()
        def count_nodes(node):                   
            # number of children node + self
            p, s = 1, 1                          
            # p: product, s: size
            for child in graph[node]:            
                # for each child (only 2 at maximum)
                res = count_nodes(child)         
                # get its nodes count of each subtree sizes for the product
                p *= res                         
                #size increment
                s += res 
            #we need count from above (subtree from it parent)            
            p *= max(1, n - s)              
            d[p] += 1                           
            return s                        
        count_nodes(0)                              
        return d[max(d.keys())]                 

#######################################
# 1666. Change the Root of a Binary Tree
# 09JAN26
#######################################
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def flipBinaryTree(self, root: 'Node', leaf: 'Node') -> 'Node':
        '''
        recursion, we have access to parent via the node class
        keep track of node, and where we came from
        '''
        def flip_recu(node, from_node):
            # set and break pointers between node and from_node
            p = node.parent
            node.parent = from_node #came from
            if node.left == from_node: 
                node.left = None
            if node.right == from_node: 
                node.right = None
                
            # stopping condition
            if node == root:
                return node
            
            # set right child
            if node.left: 
                node.right = node.left
            # set left child
            node.left = flip_recu(p, node)
            return node
        
        return flip_recu(leaf, None)
    
#########################################
# 2826. Sorting Three Groups
# 10JAN26
########################################
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        '''
        if we have the longest increasing subsequence, then we have the minimum operations
        find LIS, then its just len(nums) - LIS
        '''
        memo = {}
        n = len(nums)
        def dp(i):
            if i >= len(nums):
                return 0
            if i in memo:
                return memo[i]
            ans = 1
            for j in range(i+1,n):
                if nums[j] >= nums[i]:
                    ans = max(ans, 1 + dp(j))
            memo[i] = ans
            return ans
        lis = 1
        for i in range(n):
            lis = max(lis,dp(i))

        return n - lis
    
###########################################
# 3453. Separate Squares I
# 12JAN26
###########################################
class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        '''
        if total area of all squares = A
        then we need to find a line, where the areas == A/2
        binary search on answer, given a line, check if the area's are equal
        they are all square
        '''
        left = 0
        right = 0
        for x, y, l in squares:
            right = max(right, y + l)

        def calc(line):
            area_under = 0
            total_area = 0
            for x, y, l in squares:
                total_area += l * l
                height = max(0, min(l, line - y))
                area_under += height * l
            return area_under - total_area / 2

        while right - left > 1e-5:
            mid = (left + right) / 2
            if calc(mid) >= 0:
                right = mid
            else:
                left = mid

        return left