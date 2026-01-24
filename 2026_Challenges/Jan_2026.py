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
        ans = 0
        while right - left > 1e-5:
            mid = (left + right) / 2
            if calc(mid) >= 0:
                right = mid
            else:
                ans = mid
                left = mid

        return ans
    
############################################################
# 2975. Maximum Square Area by Removing Fences From a Field
# 16JAN26
############################################################
class Solution:
    def maximizeSquareArea(self, m: int, n: int, hFences: List[int], vFences: List[int]) -> int:
        '''
        remove fences, either horizontal or vertical anre get maximum area
        we want to make the maximum square

        '''
        #add in bounds to hfences
        hs = hFences + [1] + [m]
        vs = vFences + [1] + [n]
        ans = -1
        mod = 10**9 + 7
        hs.sort()
        vs.sort()
        h_diffs = set()
        for i in range(len(hs)):
            for j in range(i+1,len(hs)):
                h_diffs.add(hs[j] - hs[i])
        
        v_diffs = set()
        for i in range(len(vs)):
            for j in range(i+1,len(vs)):
                v_diffs.add(vs[j] - vs[i])
        
        sides = h_diffs & v_diffs
        max_edge = max(sides,default = 0)
        return (max_edge**2 % mod) if max_edge else -1
    

################################################################
# 3047. Find the Largest Area of Square Inside Two Rectangles
# 18JAN26
#################################################################
class Solution:
    def largestSquareArea(self, bottomLeft: List[List[int]], topRight: List[List[int]]) -> int:
        '''
        we have n rectangles id by there bottomLeft and topRight
        need to find square of max area that can fit inside two rectnables
        hint says to brute force area of each pair of rectangles
        if there is an overlap between two rectangles
        the square intersection is simpyll just the min of the intersecting areas width and height
        '''
        def rectangles_intersect(r1_bl, r1_tr, r2_bl, r2_tr):
            x1_min, y1_min = r1_bl
            x1_max, y1_max = r1_tr
            x2_min, y2_min = r2_bl
            x2_max, y2_max = r2_tr

            # Check for separation
            if x1_max <= x2_min:  # r1 is left of r2
                return False
            if x2_max <= x1_min:  # r2 is left of r1
                return False
            if y1_max <= y2_min:  # r1 is below r2
                return False
            if y2_max <= y1_min:  # r2 is below r1
                return False

            return True
        
        def intersection_square_area(r1_bl, r1_tr, r2_bl, r2_tr):
            x1_min, y1_min = r1_bl
            x1_max, y1_max = r1_tr
            x2_min, y2_min = r2_bl
            x2_max, y2_max = r2_tr

            width = min(x1_max, x2_max) - max(x1_min, x2_min)
            height = min(y1_max, y2_max) - max(y1_min, y2_min)

            if width <= 0 or height <= 0:
                return 0  # no intersection

            side = min(width, height)
            return side

        n = len(bottomLeft)
        ans = 0

        for i in range(n):
            for j in range(i + 1, n):
                x1_min, y1_min = bottomLeft[i]
                x1_max, y1_max = topRight[i]
                x2_min, y2_min = bottomLeft[j]
                x2_max, y2_max = topRight[j]

                width = min(x1_max, x2_max) - max(x1_min, x2_min)
                height = min(y1_max, y2_max) - max(y1_min, y2_min)

                if width <= 0 or height <= 0:
                    continue

                side = min(width, height)
                ans = max(ans, side * side)

        return ans
    
###################################################################################
# 1292. Maximum Side Length of a Square with Sum Less than or Equal to Threshold
# 19JAN26
#####################################################################################
class Solution:
    def maxSideLength(self, mat: List[List[int]], threshold: int) -> int:
        '''
        need to use 2d presum
        even if i the the 2d pref_sum, i still need to check all (i,j) starts
        and then all lengths, which is still n^3 anyway
        try all sums, don't forget pref sum
        2d pref_sum
        pref_sum[i][j] = mat[i-1][j-1] + pref_sum[i-1] + pref_sum[i][j-1] - pref_sum[i-1][j-1]
        from 1 to rows and 1 to cols
        '''
        m, n = len(mat), len(mat[0])
        pref = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                pref[i][j] = (
                    mat[i-1][j-1]
                    + pref[i-1][j]
                    + pref[i][j-1]
                    - pref[i-1][j-1]
                )
        
        def square_sum(pref, r, c, k):
            r2 = r + k
            c2 = c + k
            return pref[r2][c2] - pref[r][c2] - pref[r2][c] + pref[r][c]

        ans = 0
        for k in range(1, min(m, n) + 1):          # square size
            for i in range(m - k + 1):             # row
                for j in range(n - k + 1):         # col
                    sq_sum = square_sum(pref, i, j, k)
                    if sq_sum <= threshold:
                        ans = max(ans,k)
        
        return ans

##############################################
# 3314. Construct the Minimum Bitwise Array I
# 20JAN26
##############################################
class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        '''
        need to make ans array such that
        ans[i] | (ans[i] + 1) = nums[i]
        and it must be a small as possible
        '''
        n = len(nums)
        ans = [-1]*n

        #do i only need to check up to nums[i]
        for i in range(n):
            for num in range(1,nums[i] + 1):
                if num | (num + 1) == nums[i]:
                    ans[i] = num
                    break
        
        return ans
    
###############################################
# 3315. Construct the Minimum Bitwise Array II
# 20JAN26
################################################
class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        '''
        flip the lowest set bit before the frist zero bit, producing the smallest possible number
        less than x that differes by exactly one bit
        need the position of the first zero in nums[i], we need it to be as smallas possible,
        so check bit positions to the left
        ans + 1, sets the first zero bit to 1, and changes all lower bits from 1 to 0
        so the effect of ans | (ans + 1) is taht the first 0 bit in ans becomes 1
        '''
        ans = []
        for i in range(len(nums)):
            res = -1
            d = 1
            while (nums[i] & d) != 0:
                res = nums[i] ^ d #turn bit to zero
                #flip each bit position as a result of adding + 1
                #need nums[i] & d to be zero
                d <<= 1
            #when we encounter a 0 bit, it means there is no smaller number than the current ans
            #shifting would only lead to a larger number
            ans.append(res)
        return ans
    
############################################
# 3507. Minimum Pair Removal to Sort Array I
# 22JAN26
############################################
class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        '''
        simulate, find min sum and check
        '''

        def check(arr):
            n = len(arr)
            for i in range(1,n):
                if arr[i] < arr[i-1]:
                    return False
            
            return True
        
        def find_min_sum(arr):
            min_sum = float('inf')
            idx = -1
            n = len(arr)
            for i in range(1,n):
                curr_sum = arr[i] + arr[i-1]
                if curr_sum < min_sum:
                    min_sum = curr_sum
                    idx = i - 1
            return (min_sum,idx)

        temp = nums[:]
        ans = 0
        while check(temp) == False:
            min_sum,idx = find_min_sum(temp)
            temp[idx] = min_sum
            temp.pop(idx+1)
            ans += 1
        
        return ans

##################################################
# 3510. Minimum Pair Removal to Sort Array II
# 24JAN26
###################################################
class Node:
    def __init__(self,value,left):
        self.value = value
        self.left = left #???? #index position
        self.prev = None
        self.next = None

class PQItem:
    def __init__(self,first,second,cost):
        self.first = first
        self.second = second
        self.cost = cost
    
    #less than invariant
    def __lt__(self,other):
        if self.cost == other.cost:
            return self.first.left < other.first.left
        return self.cost < other.cost

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        '''
        need to simulate effeciently
        i can used heap to store min nums with index
        min_heap -> (min_sum,index)
        when we remove the min_sum,
        garbage heap/lazy deletion, remov entries when they become stale
        store as double linked list for easy deletion
        when merging adjacent elements in array this double linked list
        important part here -> after removing pair (i,j), the original eleemnts located at i-1 and j+1 
        will form two new adjcanet number pairs with the new merged element
        at the same time, the other pairs, originally, (i-1,i) and (j,j+1) if they exsist become dirty data
        we can delete lazily
            i.e keep in heap, and remove stale indices when we can
        
        how to determine dirty data
            assume we merge left, we check whetehre an index position has already been merged
            only when both elements are not merger are their reference considered valid
        
        then we just need to check the motoncity of the array
            we can just count decreasing states at each adjacent pair
            when decreaseCount = 0, the array is nondecerasing
        
        merging elements in constant time can be done with a double linked list
        '''
        pq = []
        head = Node(nums[0],0)
        curr = head
        n = len(nums)
        merged = [False]*n
        dec_count = 0
        count = 0

        for i in range(1,n):
            new_node = Node(nums[i],i)
            curr.next = new_node
            new_node.prev = curr
            pq_item = PQItem(curr,new_node,curr.value + new_node.value)
            heapq.heappush(pq,pq_item)

            #check
            if nums[i-1] > nums[i]:
                dec_count += 1
            curr = new_node
        
        while dec_count > 0:
            item = heapq.heappop(pq)
            first, second, cost = item.first, item.second, item.cost

            #check if we can merge, check if sum is unchanged
            #this is the stale state check (dirty data)
            if merged[first.left] or merged[second.left] or first.value + second.value != cost:
                continue
            count += 1

            if first.value > second.value:
                dec_count -= 1
            
            #deletion in DLL
            prev_node = first.prev
            next_node = second.next
            first.next = next_node
            if next_node:
                next_node.prev = first

            #motonicity check
            if prev_node:
                if prev_node.value > first.value and prev_node.value <= cost:
                    dec_count -= 1
                elif prev_node.value <= first.value and prev_node.value > cost:
                    dec_count += 1

                heapq.heappush(
                    pq, PQItem(prev_node, first, prev_node.value + cost)
                )

            if next_node:
                if second.value > next_node.value and cost <= next_node.value:
                    dec_count -= 1
                elif second.value <= next_node.value and cost > next_node.value:
                    dec_count += 1
                heapq.heappush(
                    pq, PQItem(first, next_node, cost + next_node.value)
                )

            first.value = cost
            merged[second.left] = True
        
        return count
