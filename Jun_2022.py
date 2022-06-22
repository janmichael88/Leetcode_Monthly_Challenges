#################################
# 357. Count Numbers with Unique Digits
# 01JUN22
##################################
#good solution write up with other backtracking solutions
#https://leetcode.com/problems/count-numbers-with-unique-digits/discuss/83054/Backtracking-solution

class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        '''
        backtracking approachn involves trying create a digit in the range 0 to 10**n
        we want to append a new digit only if we haven't added that digit yet
        
        we need to call the function for each value of n than add them up
        since the the answer of n, should be the sum for all for n = 1 to n
        '''
        
        ans = 1
        self.MAX = 10**n
        
        used = [False]*10 #digits 0 to 9
        
        def rec(prev):
            count = 0
            if prev < self.MAX:
                count += 1
            else:
                return count
            
            for i in range(10):
                if not used[i]:
                    used[i] = True
                    curr = 10*prev + i
                    count += rec(curr)
                    used[i] = False
            
            return count
        
        for i in range(1,10):
            used[i] = True
            ans += rec(i)
            used[i] = False
        
        return ans

#another way
class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        '''
        just another way
        
        '''
        MAX = 10**n
        used = [False]*10
        
        def dfs(current):
            #bottom case, cannott make any digits
            if current >= MAX:
                return 0
            #otherwise we have at least 1 digit
            count = 1
            #recursive case
            for i in range(10):
                if current == 0 and i == 0: #ignore leading zero
                    continue
                #take
                if not used[i]:
                    used[i] = True
                    count += dfs(current*10 + i)
                    used[i] = False
            
            return count
        
        return dfs(0)

class Solution:
    def countNumbersWithUniqueDigits(self, n: int) -> int:
        '''
        there is also a recurrence
        f(1) = 10 digits, just the digits [0 to 9]
        f(2) = 81 + 10
        we know ther are 100 possible digits in the range [0 <= x < 100]
        numbers with at least two digits are:
         100 - 10 = 90
        of these, we cannot chose 11,22,33...99, of which there are 9
        90 - 9 = 81
            9*(two digits same number)
        we can write this number as 
        f(3) = f(2)*8

        dp(n) = dp(n-1)*(10-n-1)
        base case dp(1) = 10
        '''
        if n == 0:
            return 1
        
        memo = {}
        
        def dp(n):
            if n == 1:
                return 10
            if n == 2:
                return 81
            if n in memo:
                return memo[n]
            ans = dp(n-1)*(10-n+1)
            memo[n] = ans
            return ans
        
        ans = 0
        for i in range(1,n+1):
            ans += dp(i)
        
        return ans
        

#################################
# 643. Maximum Average Subarray I
# 01JUN22
##################################
#brute force, check all subarray sums
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        '''
        brute force would be to examin all sub array sums of size k
        then just take the max
        '''
        
        ans = float('-inf')
        N = len(nums)
        #edge case
        if N == k:
            return sum(nums)/k
        
        
        for i in range(N-k+1):
            ans = max(ans,sum(nums[i:i+k])/k)
        return ans


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        '''
        the problem is summing acorss a subarray, which takes N*k time in total
        we can use the cumsum array in just get the running sum in constant time
        '''
        cum_sum = [0]
        for num in nums:
            cum_sum.append(num+cum_sum[-1])
            
        ans = float('-inf')
        N = len(nums)
        #edge case
        if N == k:
            return sum(nums)/k
        
        for i in range(k,N+1):
            sub_sum = cum_sum[i] - cum_sum[i-k]
            curr_avg = sub_sum /k
            ans = max(ans,curr_avg)
        
        return ans

#sliding window
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        '''
        we can just mainain a cum sum of size k
        then just add in the new element and remove the i-k eleemtn
        '''
            
        N = len(nums)
        #edge case
        if N == k:
            return sum(nums)/k
        
        curr_sum = 0
        for i in range(k):
            curr_sum += nums[i]
        
        
        res = float('-inf')
        for i in range(k,N):
            curr_sum += nums[i] - nums[i-k]
            res = max(res,curr_sum)
        
        return res / k

#########################
# 867. Transpose Matrix
# 02JUN22
#########################
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        '''
        index (i,j) becomes (j,i) if i !+ j
        which works if rows == cols
       
        i can allocate a new matrix with the inverted dimensions
        then pass the patrix again putting elements in their proper spot
        '''
        rows = len(matrix)
        cols = len(matrix[0])
       
        new_matrix = [[0]*rows for _ in range(cols)]
       
        for i in range(rows):
            for j in range(cols):
                new_matrix[j][i] = matrix[i][j]
       
        return new_matrix

###########################
# 51. N-Queens (Revisited)
# 04JUN22
##########################
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        q_col_idx = []
        def dfs(queens,diags,anti_diags):
            row = len(queens)
            if row == n:
                q_col_idx.append(queens[:]) #these are list of indices (col) for each row in order
                return
            for col in range(n):
                if col not in queens and (row - col) not in diags and (row + col) not in anti_diags:
                    queens.append(col)
                    diags.append(row-col)
                    anti_diags.append(row+col)
                    dfs(queens,diags+[row-col],anti_diags+ [row+col])
                    queens.pop()
                    diags.pop()
                    anti_diags.pop()
                    
        dfs([],[],[])
        return [ ["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in q_col_idx]


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        output = []
        
        def dfs(solution,excluded):
            #solution is a list of strings
            row = len(solution)
            #we can still go down the board
            if row < n:
                #dfs
                #we are always goind down a row, so go across cols
                for col in range(n):
                    if (row,col) in excluded:
                        continue
                    #we can place a queen at this col
                    new_excluded = set()
                    curr_row_solution = "."*col+"Q"+"."*(n-col-1) #place a queen
                    #now go down rows
                    for r in range(row,n):
                        new_excluded.add((r,col))
                    #down daig right
                    row_diag = row_anti_diag = row
                    col_diag = col_anti_diag = col
                    
                    while col_diag < n:
                        row_diag += 1
                        col_diag += 1
                        new_excluded.add((row_diag,col_diag))
                        
                    #anti diag
                    while col_anti_diag > 0:
                        row_anti_diag += 1
                        col_anti_diag -= 1
                        new_excluded.add((row_anti_diag,col_anti_diag))
                        
                    solution.append(curr_row_solution)
                    dfs(solution,excluded | new_excluded)
                    excluded.discard(new_excluded)
                    solution.pop()
                    
            else:
                output.append(solution[:])
        dfs([],set())
        return output

##############################
# 52. N-Queens II (REVISITED)
# 05JUN22
##############################
class Solution:
    def totalNQueens(self, n: int) -> int:
        
        def dfs(rows,excluded):
            #we can still go down the board
            if rows < n:
                #dfs
                solutions = 0
                #we are always goind down a row, so go across cols
                for col in range(n):
                    if (rows,col) in excluded:
                        continue
                    #we can place a queen at this col
                    new_excluded = set()
                    #now go down rows
                    for r in range(rows,n):
                        new_excluded.add((r,col))
                    #down daig right
                    row_diag = row_anti_diag = rows
                    col_diag = col_anti_diag = col
                    
                    while col_diag < n:
                        row_diag += 1
                        col_diag += 1
                        new_excluded.add((row_diag,col_diag))
                        
                    #anti diag
                    while col_anti_diag > 0:
                        row_anti_diag += 1
                        col_anti_diag -= 1
                        new_excluded.add((row_anti_diag,col_anti_diag))
                        
                    rows += 1
                    solutions += dfs(rows,excluded | new_excluded)
                    rows -= 1
                
                return solutions
                    
            else:
                return 1
            
        return dfs(0,set())

###############################
# 360. Sort Transformed Array
# 04JUN22
################################
class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        '''
        the stupid way is to apply the the transformation to each element thens sort
        '''
        def f(num):
            return a*num*num + b*num + c
        
        for i in range(len(nums)):
            nums[i] = f(nums[i])
        
        nums.sort()
        return nums

#two pointer
class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        '''
        now can we do this in O(N) time?
        the function is parabolic, if we find the vertex of the parabola, then to the right can be stricly increasing
        and to the left it can be striclty decreasing
        if a > 0, the function is concave uo
        if a < 0, then the function is concave down
        
        things to note,
            1. the vertex of the parabola is at the point, x = -b/2a, but we don't really need that point
            if a > 0
                the points to the right of the vertex are increasing
                the points to the left are decreasing
                the ends of the parabola have the largest values
            if a < 0:
                the vertex is the largest value
                the points to the right are decreaing
                the points to the left are decreasing
            
            we can use a two pointer trick (since the array is already sorted)
            and take the greater of the two elements (or the smaller of the two, depending wheter or a > 0 or a < 0)
            
        
        three scenarios:
            nums[-1] <= vertex, meaning all values in nums will be on the left side of the center line of the quadratic function graph. (Decreasing side)
nums[0] >= vertex, meaning all values in nums will be on the right side of the center line of the quadratic function graph. (Increasing side)
nums[0] <= nums[i] <= vertex <= nums[j] <= nums[-1], meaning some values are on the left and some are on the right.

        intuion:
            we don't really care what b and c are, because the above two cases catch everything
        
        '''
        def f(x):
            return a*x*x + b*x + c
        
        N = len(nums)
        
        index = 0 if a < 0 else N-1 #largest values on the ends, if a > 0, else smallest values on the ends
        left = 0
        right = N-1
        ans = [0]*N
        
        while left <= right:
            l_val, r_val = f(nums[left]), f(nums[right])
            
            #concave up, take lrageststart adding answers from the end of the array
            if a >= 0:
                if l_val > r_val:
                    ans[index] = l_val
                    left += 1
                else:
                    ans[index] = r_val
                    right -= 1
                index -= 1
            #concave down, take minimum
            else:
                if l_val > r_val:
                    ans[index] = r_val
                    right -= 1
                else:
                    ans[index] = l_val
                    left += 1
                
                index += 1
        
        return ans

########################################
# 1197. Minimum Knight Moves (REVISITED)
# 06JUN22
#########################################
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        '''
        note that solutions are symmetric, rather the targer point (x,y) is symmetric about vertical, horizontal, and diagonal axis
        (x,y) same as (-x,y), (x,-y), (-x,-y)
        so we an really take abs(x) and abs(y)
        
        if we start from the target, there are only two moves that bring us closer to the origin (-1,-2) or (-2,-1)
        
        we define dp(x,y) as the min steps to get to the point (x,y)
        
        any (x,y) pair who's sum is two, is exactly two moves away from the origin
        
        dp(x,y)=min(dp(∣x−2∣,∣y−1∣),dp(∣x−1∣,∣y−2∣))+1
        
        '''
        memo = {}
        
        def dp(x,y):
            if (x,y) == (0,0):
                return 0
            if x+y == 2:
                return 2
            if (x,y) in memo:
                return memo[(x,y)]
            
            first_move = dp(abs(x-1),abs(y-2))
            second_move = dp(abs(x-2),abs(y-1))
            ans = min(first_move,second_move) + 1
            memo[(x,y)] = ans
            return ans
        
        return dp(abs(x),abs(y))

#there is an O(1) solution
#https://math.stackexchange.com/questions/1135683/minimum-number-of-steps-for-knight-in-chess
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        x, y = abs(x), abs(y)
        if (x < y): 
            x, y = y, x
        if (x == 1 and y == 0): 
            return 3        
        if (x == 2 and y == 2): 
            return 4        
        delta = x - y
        if (y > delta): 
            return delta - 2 * int((delta - y) // 3);
        else: 
            return delta - 2 * int((delta - y) // 4);

############################
# 88. Merge Sorted Array (Revisited)
# 07JUN22
############################
#close one..
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        '''
        we want to modify nums1 in place
        usual approach involves two points, adavanving both while taking the smaller
        note,
        nums1 has been modified to have length m+nn
        
        what is put the last n elements of nums2 in the last n spots in nums1
        then move the two pointers and just swap?
        '''
        #put elements ofr nums2 into nums1 at the end
        p1 = m
        p2 = 0
        while p1 < m+n:
            nums1[p1] = nums2[p2]
            p1 += 1
            p2 += 1
        
        #two pointers again, but swap
        p1 = 0
        p2 = m
        
        while p1 < m + n:
            print(p1,p2)
            if p1 == p2 and p2 < m+n-1:
                p2 += 1
            if  p2 < m+ n and nums1[p1] > nums1[p2]:
                #swap
                print('swapped')
                nums1[p1],nums1[p2] = nums1[p2],nums1[p1]
                p1 += 1
                p2 += 1
            elif nums1[p1] <= nums1[p2]:
                p1 += 1

#three pointers, copy over nums1
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        '''
        we can make a copy of nums1, then use two pointers to read and another point to write
        being careful to watch for boundary conditions
        '''
        nums1_copy = nums1[:m] #grabbing only first
        
        #read pointers
        p1 = 0
        p2 = 0
        
        for p in range(m+n):
            if p2 >= n or (p1 < m and nums1_copy[p1] <= nums2[p2]):
                nums1[p] = nums1_copy[p1]
                p1 += 1
            else:
                nums1[p] = nums2[p2]
                p2 += 1

#the pointers, start backwards
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        '''
        the idea is to work backwards, this is a good tip for trying to solve in place problems
        algo:
            ptr1 starts at m-1
            ptr2 starts at n-1
            and ptr3 starts at m+n-1
            this way it is guaranteed that once we start overwirting the first m values in nums1, we will have alredy written each into 
            its new positions
        '''
        p1 = m - 1
        p2 = n - 1
        
        for p in range(m+n-1,-1,-1):
            #n < m, so if we exhaust p2, which is smaller, we should b done
            if p2 < 0:
                break
            if p1 >= 0 and nums1[p1] > nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1

#another way
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        '''
        recall that we can fit n elements into the remaing spots in the nums1 array
        either we fit all n elements into nums1 from nums2
        or we use up m elements from nums1 and have remainings elements from nums2
        '''
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        
        while n > 0:
            nums1[m+n-1] = nums2[n-1]
            n -= 1


##########################
# 372. Super Pow
# 05JUN22
##########################class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        '''
        do as the problem says
        '''
        return pow(a, int(''.join(map(str, b))), 1337)

class Solution:
    def superPow(self, a: int, b: List[int]) -> int:
        '''
        we can take advantage of two properties
        1. x*y % 1337 == (x % 1331)*(y % 1337) % 1337
        2. if b = m*10 + d, then a**b == (a**d)*(a**10)**m
        
        note that for the pow funciton pow(x,y,z) = (x^y) % z
        
        so if we want to find:
            a**b % 1337
            
        we can define a recursive function
        rec(a,b) that returns a**b % 1337
        rec(a,b) = 
            (a^(b % 10) % 1337)*rec(a^10 % 1337,b)% 1337 
        '''
        def rec(a,b):
            #base case
            if not b:
                return 1
            first_part = pow(a,b.pop(),1337)
            prev = rec(pow(a,10,1337),b)
            return first_part*prev % 1337
        
        return rec(a,b)

#iterative version
class Solution(object):
    def superPow(self, a, b):        
        acc = 1
        while b: 
            a, acc = pow(a, 10, 1337), pow(a, b.pop(), 1337) * acc % 1337
        return acc

#######################
# 657. Robot Return to Origin
# 07JUN22
#######################
class Solution:
    def judgeCircle(self, moves: str) -> bool:
        '''
        a move should cancel another move
        for example UP +1
        DOWN - 1
        answer is zero
        cancel in both directions
        
        '''
        balance_horiz = 0
        balance_vert = 0
        for m in moves:
            if m == 'U':
                balance_vert += 1
            elif m == 'D':
                balance_vert -= 1
            elif m == 'R':
                balance_horiz += 1
            else:
                balance_horiz -= 1
        
        return balance_horiz == 0  and balance_vert == 0


############################################
# 671. Second Minimum Node In a Binary Tree
# 08JUN22
############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
        '''
        easiet way is to traverse and sort, and grab the second 
        without using the special property
        '''
        nodes = set()
        def inorder(node):
            if not node:
                return 
            inorder(node.left)
            nodes.add(node.val)
            inorder(node.right)
        
        inorder(root)
        nodes = list(nodes)
        nodes.sort()
        
        if not root:
            return -1
        
        if len(nodes) == 1:
            return -1
        
        return nodes[1]

#using special property
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
        '''
        we can assign first min to root.val
        if when we get to a node in our traversal and wee see that node.val > first_min, it cannot be in that subtree
        so we do not need to recurse
        
        we also only care about the second minimum, so we do not need to record any values > our current canddiate
        '''
        self.first_min = root.val
        self.ans = float('inf')
        
        def dfs(node):
            if node:
                if (self.first_min < node.val) and (node.val < self.ans):
                    self.ans = node.val
                elif self.first_min == node.val:
                    dfs(node.left)
                    dfs(node.right)
        
        dfs(root)
        return self.ans if self.ans < float('inf') else -1

#the answer must be greater than the root and smaller than everything else
class Solution(object):
    def findSecondMinimumValue(self, root):
        res = [float('inf')]
        def traverse(node):
            if not node:
                return
            if root.val < node.val < res[0]:
                res[0] = node.val
            traverse(node.left)
            traverse(node.right)
        traverse(root)
        return -1 if res[0] == float('inf') else res[0]

##############################################################
# 1151. Minimum Swaps to Group All 1's Together (REVISTED)
# 09JUN22
##############################################################
class Solution:
    def minSwaps(self, data: List[int]) -> int:
        '''
        we want the the ones grouped together
        sum(data) gives us the nuber of ones, but also gives us the the size of the necesary aubarray of ones
        in the subarray, the number of zeros would be the number of swaps
        we can use a sliding window to find the min number of zeros in a subarray of size sum(data)
        '''
        ones = sum(data)
        
        if ones == 0:
            return 0
        N = len(data)
        
        min_swaps = float('inf')
        
        left = 0
        right = 0
        
        curr_swaps = 0
        
        while right < N:
            while right - left < ones:
                curr_swaps += data[right] == 0
                right += 1
            min_swaps = min(min_swaps,curr_swaps)
            curr_swaps -= data[left] == 0
            left += 1
        
        return min_swaps

######################################
# 373. Find K Pairs with Smallest Sums
# 09JUN22
######################################
#i guess we can't use two pointers here
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        '''
        the arrays are sorted in ascenidng order
        two pointer and keep advance the smaller of the two
        '''
        i = j = 0
        
        ans = []
        
        while i < len(nums1) or j < len(nums2) or k > 0:
            first = nums1[i] if i < len(nums1) else nums1[i]
            second = nums2[j] if j < len(nums2) else nums2[-1]
            
            pair = [first,second]
            ans.append(pair)
            k -= 1
            

            if (i < len(nums1) and j < len(nums2)):
                if nums1[i] < nums2[j]:
                    j += 1

                else:
                    i += 1
            else:
                break
        
        return ans
            
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        '''
        the idea is to keep track if the next min sum pair into a min heap
        since the arrays are sorted we can start by pairing each num in nums1 to nums2[0]
        onto a heap push entry (sum pair,index into i, and 0)
        when extract the entry, write pair to answer, but push back the next number in nums2 paired with an entry for nums1
        
        phew!
        '''
        heap = []
        ans = []
        
        if len(nums1) == 0 or len(nums2) == 0 or k == 0:
            return res
        for i in range(len(nums1)):
            entry = (nums1[i] + nums2[0],i,0)
            heap.append(entry)
        
        heapq.heapify(heap)
        
        while heap and k > 0:
            pair_sum, i,j = heapq.heappop(heap)
            ans.append([nums1[i],nums2[j]])
            k -= 1
            #edge case to stay within num2
            if j == len(nums2) - 1:
                continue
            heapq.heappush(heap,(nums1[i] + nums2[j+1],i,j+1))
        
        return ans
                         
###############################################################
# 3. Longest Substring Without Repeating Characters (Revisited)
# 10JUN22
###############################################################
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        '''
        we can just keep a sliding window, and keep expanding it until we find a repeated char
        if when we see a repeated char, its time to update the disatance
        but we also need to mapp the char to an index to advance that point in the window
        
        '''
        N = len(s)
        seen = set()
        ans = 0
        
        left = right = 0
        
        while right < N:
            while right < N and s[right] not in seen:
                seen.add(s[right])
                right += 1
            ans = max(ans,right - left)
            seen.remove(s[left])
            left += 1
        
        return ans

#############################
# 1695. Maximum Erasure Value (Revisited)
#  12JUN22
##############################
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        '''
        this is just finding largest subarray with unique values but obtain sum of window
        '''
        N = len(nums)
        max_erasure_value = float('-inf')
        curr_sum = 0
        unique = set()
        left = right = 0
        
        while right < N:
            while right < N and nums[right] not in unique:
                unique.add(nums[right])
                curr_sum += nums[right]
                right += 1
            
            max_erasure_value = max(max_erasure_value,curr_sum)
            unique.remove(nums[left])
            curr_sum -= nums[left]
            left += 1
        
        return max_erasure_value
            
#########################
# 120. Triangle (REVISTED)
# 13JUN22
##########################
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        '''
        if we define dp(i,j) this represents the the minimum sum to get to dp(i,j)
        
        we need to minize this for the current ij we are on
        
        dp(i,j) = {
            triangle(i,j) + min(dp(i-1,j-1),dp(i-1j+1))
            if i can go left and right
        }
        dp(0,0) = triangle[0][0]
        
        '''
        rows = len(triangle)
        #i would need to call dp for each number on the last row of triangle
        res = float('inf')
        memo = {}
        
        def dp(i,j):
            if (i,j) == (0,0):
                return triangle[0][0]
            if (i,j) in memo:
                return memo[(i,j)]
            #staring cell in row
            if j == 0:
                ans = triangle[i][j] + dp(i-1,j)
                memo[(i,j)] = ans
                return ans
            #ending cell in row
            if j == i:
                ans = triangle[i][j] + dp(i-1,j-1)
                memo[(i,j)] = ans
                return ans
            else:
                ans = triangle[i][j] + min(dp(i-1,j),dp(i-1,j-1))
                memo[(i,j)] = ans
                return ans
        
        
        for j in range(len(triangle[-1])):
            res = min(res,dp(rows-1,j))
            #print(triangle[rows-1][j])
        
        return res

class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        '''
        another way of defining the recurrence is if we let
        dp(i,j) be the min sum path down to the base
        then dp(i,j) = triangle[i][j] + min(dp(i+1,j-1),dp(i+1.j))
        '''
        memo = {}
        N = len(triangle)
        def dp(i,j):
            if i == N:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            curr_ans = triangle[i][j]
            right = dp(i+1,j)
            left = dp(i+1,j+1)
            curr_ans += min(left,right)
            memo[(i,j)] = curr_ans
            return curr_ans
        
        return dp(0,0)


################################
# 931. Minimum Falling Path Sum
# 13JUN22
################################
#top down
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        rows = len(matrix)
        #i would need to call dp for each number on the last row of triangle
        res = float('inf')
        memo = {}
        
        def dp(i,j):
            if i == 0:
                return matrix[i][j]
            if (i,j) in memo:
                return memo[(i,j)]
            #staring cell in row
            if j == 0:
                ans = matrix[i][j] + min(dp(i-1,j),dp(i-1,j+1))
                memo[(i,j)] = ans
                return ans
            #ending cell in row
            if j == rows-1:
                ans = matrix[i][j] + min(dp(i-1,j-1),dp(i-1,j))
                memo[(i,j)] = ans
                return ans
            else:
                ans = matrix[i][j] + min(dp(i-1,j),dp(i-1,j-1),dp(i-1,j+1))
                memo[(i,j)] = ans
                return ans
        
        
        for j in range(rows):
            res = min(res,dp(rows-1,j))
            #print(triangle[rows-1][j])
        
        return res

#bottom up in place
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        '''
        we can just overwite the rows with the minimum going down
        '''
        rows = len(matrix)
        for row in range(1,rows):
            for col in range(rows):
                #if starting cell in row, i can only take directly above
                if col == 0:
                    matrix[row][col] += min(matrix[row-1][col],matrix[row-1][col+1])
                elif col == rows-1:
                    matrix[row][col] +=  min(matrix[row-1][col],matrix[row-1][col-1])
                else:
                    matrix[row][col] +=  min(matrix[row-1][col],matrix[row-1][col-1],matrix[row-1][col+1])
        
        return min(matrix[-1])   

####################################################
# 583. Delete Operation for Two Strings (REVISITED)
# 14MAY22
####################################################
#indirectly using LCS, top down
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        '''
        we can just turn this problem into finding the longest common subsequence problem
        if there exists an LCS between two strings word1 and word2
        we need to delete everything but the LCS in the two strings
        so it would be len(word1) + len(word2) - 2*lcs, we double them because they are present in both
        
        lcs function
        
        let dp(i,j) return the lenght of the lcs for strings word1[:i] and word2[:j]
        dp(i,j) ={
            if word1[i] == word2[j]:
              return 1 + dp(i-1,j-1)
             else:
                return max(dp(i-1,j),dp(i,j-1))
        }
        
        base cases, when we we fall out of the index, there isn't an LCS at all, return 0
        '''
        memo = {}
        M = len(word1)
        N = len(word2)
        
        def dp(i,j):
            if i == 0 or j == 0:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            if word1[i-1] == word2[j-1]:
                ans = 1 + dp(i-1,j-1)
                memo[(i,j)] = ans
                return ans
            else:
                ans =  max(dp(i-1,j),dp(i,j-1))
                memo[(i,j)] = ans
                return ans
        
        
        LCS = dp(M,N)
        return M + N - 2*LCS

#indirectly using LCS bottom up
#just know that we can reduce space by taking prev row, updating curr row, then swap/reassign
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        '''
        '''
        M = len(word1)
        N = len(word2)
        
        dp = [[0]*(N+1) for _ in range(M+1)]
        
        for i in range(1,M+1):
            for j in range(1,N+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        
        
        LCS = dp[M][N]
        return M + N - 2*LCS


#directly w/o using LCS
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        '''
        if we define dp(i,j) as the NUMBER of deletions required to make the strings word1[:i-1] == word2[:j-1]
        dp(i,j) = {
            
            if word1[i-1] == word2[j-1]
                return dp(i-1,j-1)
            else:
                return 1 + min(dp(i-1,j),dp(i,j-1))
                
            
        }
        base case, if we have gotten down to both i == 0 and j == 0, it means they weren't equal, so we need to delte all
        '''
        memo = {}
        M = len(word1)
        N = len(word2)
        
        def dp(i,j):
            if i == 0 or j == 0:
                return i + j
            if (i,j) in memo:
                return memo[(i,j)]
            if word1[i-1] == word2[j-1]:
                ans = dp(i-1,j-1)
                memo[(i,j)] = ans
                return ans
            else:
                ans = 1 + min(dp(i-1,j),dp(i,j-1))
                memo[(i,j)] = ans
                return ans
        
        return dp(M,N)

################################################
# 712. Minimum ASCII Delete Sum for Two Strings
# 14JUN22
#################################################
#close one...
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        '''
        if i let dp(i,j) be the lowest ascii sum of delted chars to make s1[:i] == s2[:j]
        then dp(i,j) = {
            if s1[i] == s2[j]:
                return ord(s1[i]) + dp(i-1,j-1)
            else:
                min(s1[i],s2[j]) + min(dp(i-1,j),dp(i,j-1))
        }
        
        call dp(len(s1),len(s2)) to check all i,j, i may have to scan the memo again
        
        if i == 0 or j == 0
            return the asci sum of the other string, i.e we have to delete all of them to get the other onr
        '''
        memo = {}
        M = len(s1)
        N = len(s2)
        
        
        def dp(i,j):
            if i < 0 or j < 0:
                return sum([ord(c) for c in s1[:i]]) or sum([ord(c) for c in s2[:j]])
            if (i,j) in memo:
                return memo[(i,j)]
            if s1[i-1] == s2[j-1]:
                ans = dp(i-1,j-1)
                memo[(i,j)] = ans
                return ans
            else:
                ans = min(dp(i-1,j) + ord(s1[i-1]) ,dp(i,j-1) + ord(s2[j-1]) )
                memo[(i,j)] = ans
                return ans
        
        return dp(M,N)

#actrual 
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        '''
        we let dp(i,j) be the min sum delete for make strings s1[i:] == s2[:j]
        
        dp(i,j) = {
            if s1[i] == s2[j], there is no cost to delte:
                return dp(i+1,j+1)
            else:
                the letters are different so we take the min
                first = dp(i+1,j) + ord(s1[i])
                second = dp(i,j+1) + ord(s2[j])
                return min(first,second)
        }
        
        base cases:
            if we have moved them all the way to end, there as no cost anymore
            i == len(s1) and j == len(s2):
                return 0
            if we have the empty string
                return sum needed for rest of the asci strring
        '''
        M = len(s1)
        N = len(s2)
        #store answer to subproblems
        memo = {}
        #fast loop for base cases
        s1_deletions = [0]*(M+1)
        s2_deletions = [0]*(N+1)
        
        for i in range(M-1,-1,-1):
            s1_deletions[i] = s1_deletions[i+1] + ord(s1[i])
        
        for i in range(N-1,-1,-1):
            s2_deletions[i] = s2_deletions[i+1] + ord(s2[i])
        
        
        
        def dp(i,j):
            if i == M and j == N:
                return 0
            if i == M or j == N:
                return s1_deletions[i] or s2_deletions[j]
            if (i,j) in memo:
                return memo[(i,j)]
            #mathcing, no cost
            if s1[i] == s2[j]:
                ans = dp(i+1,j+1)
                memo[(i,j)] = ans
                return ans
            else:
                first = dp(i+1,j) + ord(s1[i])
                second = dp(i,j+1) + ord(s2[j])
                ans = min(first,second)
                memo[(i,j)] = ans
                return ans
        
        return dp(0,0)

#bottom up dp
class Solution(object):
    def minimumDeleteSum(self, s1, s2):
        dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

        for i in range(len(s1) - 1, -1, -1):
            dp[i][len(s2)] = dp[i+1][len(s2)] + ord(s1[i])
        for j in range(len(s2) - 1, -1, -1):
            dp[len(s1)][j] = dp[len(s1)][j+1] + ord(s2[j])

        for i in range(len(s1) - 1, -1, -1):
            for j in range(len(s2) - 1, -1, -1):
                if s1[i] == s2[j]:
                    dp[i][j] = dp[i+1][j+1]
                else:
                    dp[i][j] = min(dp[i+1][j] + ord(s1[i]),
                                   dp[i][j+1] + ord(s2[j]))

        return dp[0][0]


#########################################
# 1048. Longest String Chain (REVISITED)
# 15JUN22
#########################################
#top down dp
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        '''
        we define a wordA predecessor as to wordB if:
            len(wordA) < len(wordB) and wordA is a subsequence of wordB
        
        return the longest possible word chain in words
        then sort the words alphabetically and by increasing length
        
        if we let dp(i) be the longest chaining using words[:i]
        
        then dp(i) = 1 + dp(i-1) if word[i] is a successor 
        
        rather we can use dfs and for each call genearte all its possible subseqences of one less char
        the answer to each node returns the longest chain maid so far
        we when dfs on each word and maximize
        
        dp(word) = 1 + max(dp(for all words with one char deleted from word))
        dont forget to backtrack
        '''
        words = set(words)
        memo = {}
        
        def dp(word):
            if word in memo:
                return memo[(word)]
            max_length = 1 #at least size 1 for chain with single word
            #generate neighboring words
            for i in range(len(word)):
                neigh_word = word[:i]+word[i+1:]
                if neigh_word in words:
                    curr_length = 1 + dp(neigh_word)
                    max_length = max(max_length,curr_length)
                    
            memo[word] = max_length
            return max_length
        
        ans = 0
        for word in words:
            ans = max(ans,dp(word))
        
        return ans

#translate bottom up
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        '''
        we can go botom up by sorting and start buulindg up a chain
        if we knew the previous length, then if we can reach this current word its just prevLength + 1
        we still use hashmap as memo
        '''
        #dp array
        dp = defaultdict(int)
        words.sort(key = lambda x: len(x))
        
        ans = 0
        for word in words:
            max_length = 1
            for i in range(len(word)):
                neigh_word = word[:i]+word[i+1:]
                if neigh_word in words:
                    curr_length = 1 + dp[neigh_word]
                    max_length = max(max_length,curr_length)
                
            dp[word] = max_length
            ans = max(ans,dp[word])
            
        
        return ans
            

#LIS
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        '''
        we can also treat this as the longest increasing subsequence
        if we define dp(i) is the longest chain created using the words[:i]
        then we would have to check for all words before the ith word
        
        first sort
        dp(i) = {
            min_length = 1
            for j in range(i-1):
                if word[j] is pred of word[i]:
                    min_lenght = min(min_length,dp(j))
        }
        
        to check if word1 is pred of word2:
            check len(word1) + 1 == len(word2)
            word1 is subsequence of word2
        '''
        def isPred(u,v):
            if len(u) + 1 != len(v):
                return False
            i = 0
            for ch in v:
                if i == len(u):
                    return True
                if u[i] == ch:
                    i += 1
            
            return i == len(u)

        
        words.sort(key=len)
        memo = {}
        
        def dp(i):
            if i < 0:
                return 1
            if i in memo:
                return memo[i]
            min_length = 1
            for j in range(i):
                if isPred(words[j],words[i]):
                    min_length = max(min_length,dp(j)+1)
            
            memo[i] = min_length
            return min_length
        
        ans = 1
        N = len(words)
        for i in range(N):
            ans = max(ans,dp(i))
        
        return ans

###############################################
# 5. Longest Palindromic Substring (REVISITED)
# 16JUN22
#############################################
#close one
class Solution:
    def longestPalindrome(self, s: str) -> str:
        '''
        a mistake is to think to find the longest common substring between s and s[::-1]
        but the longest common substring may not be palindromic
        recall dp transition for longest common substring
        dp(i,j) = {
            if s[i] == t[j]:
                return 1 + dp(i-1,j-1)
            else:
                return 0
        }
        base case is empty string, no longest common susbtring, return 0
        
        now for the actual algo:
            
        if we define dp(i,j) as the answer to whether s[i:j] is a palindrome
        then dp(i,j) = {
            if s[i] == s[j]:
                return dp(i+1,j-1)
            else:
                return False
        }
        base cases
            single char:
                i == j:
                    return True
                j - i == 1:
                    return s[i] == s[j]
        
        then we call this dp function of all i,j substrings and maximize
        '''
        if len(s) == 1:
            return s
        N = len(s)
        memo = {}
        
        
        def dp(i,j):
            if i == j:
                return 1
            if j - i == 1:
                return s[j] == s[i]
            if (i,j) in memo:
                return memo[(i,j)]
            if s[i] == s[j]:
                ans = dp(i+1,j-1)
                memo[(i,j)] = ans
                return ans
            else:
                memo[(i,j)] = False
                #dp(i+1,j)
                #dp(i,j-1)
                return False
            
        ans = ""
        for i in range(N-1,-1,-1):
            for j in range(i,N):
                if dp(i,j) and len(s[i:j+1]) > len(ans):
                    ans = s[i:j+1]
        
        return ans
        
                    
#i don't think traditional dp works here...
class Solution:
    def longestPalindrome(self, s: str) -> str:
        #matain largest to get it, as well as pointers
        self.start = 0
        self.end = 0
        self.largest = 1
        memo = {}
        N = len(s)
        
        def dp(i,j):
            if (i,j) in memo:
                return memo[(i,j)]
            if i == j:
                return True
            if i > j:
                return True
            if s[i] == s[j] and dp(i+1,j-1):
                ans = True
                memo[(i,j)] = ans
                
                if self.largest < j - i + 1:
                    self.largest = j -i + 1
                    self.start = i
                    self.end = j
                
                return ans
            
            else:
                ans = False
                memo[(i,j)] = ans
                dp(i+1,j)
                dp(i,j-1)
                return ans
        
        dp(0,N-1)
        return s[self.start:self.end+1]


###############################
# 968. Binary Tree Cameras (REVISITED)
# 19JUN22
###############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        '''
        when we place a camera, the camera can cover the node, the node's parent, and both of its children
        so really if we new the minimum on the left and right subtrees, we could take the min the answer from both subtrees, assuming they were already minimal, then add to that the coverage of the node we are on
        but this is hard, because we need to pass informatino about a node, its parent and it's children
        
        DP:
            lets try to cover every node, starting from the top and going down
            every node considered must be covered by a camera, or an immediate neighbor
            intuion: since cameras only care about local state, we want to try to take advantage of this
        
        we let dp(node) return some information about how many cameras it takes to cover the subtree for this node in differetns states, but what are the states
        
            state 1; strict subtree, all the nodes below this node are covered except this node
            state 2 normal subtree, all nodes below and included this node are covered, but not camera at this node
            state 3, placed a camera, all the nodes below and including this node are covered, AND there is a camera here
        
        transition:
            to cover a strict subtree, children of this node must be normal
            to cover a normal subtree without placing a camera here, the children of this node must be in states 1 or 2, and at least one of those chidlren must be in state 2
            to cover the subtree when placing a camera here, the children can be in any state
        '''
        def dp(node):
            #return statments, return the number of cameras it takes to cover each of the 3 states
            # 0: Strict ST; All nodes below this are covered, but not this one
            # 1: Normal ST; All nodes below and incl this are covered - no camera
            # 2: Placed camera; All nodes below this are covered, plus camera here
            #base case, no node, 
            if not node:
                return 0,0,float('inf')
            left = dp(node.left)
            right = dp(node.right)
            
            dp0 = left[1] + right[1]
            dp1 = min(left[2] + min(right[1:]),right[2] + min(left[1:]))
            dp2 = 1 + min(left) + min(right)
            
            return dp0,dp1,dp2
        
        return min(dp(root)[1:])

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        '''
        instead of trying to cover every node from the top down, lets try to cover it from the bottom up
        considering placing a cmcer with the deepest nodes first and working out way up
        if a node has its children covered and has a parent, then it is stricly better to place a camera at this nodes' parent
        
        algo:
            if a node has chidlren that are not covered by a camcera, then we must place a camera here
            additionally if a node has no parent and it is not covered, we place a camera here
        '''
        self.ans = 0 
        covered = {None}
        
        def dfs(node,parent = None):
            if node:
                dfs(node.left,node)
                dfs(node.right,node)
                
                if parent == None and node not in covered or node.left not in covered or node.right not in covered:
                    self.ans += 1
                    covered.add(node)
                    covered.add(parent)
                    covered.add(node.left)
                    covered.add(node.right)
        
        
        
        dfs(root)
        return self.ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        '''
        one last way, good review
        #https://leetcode.com/problems/binary-tree-cameras/discuss/1211695/JS-Python-Java-C%2B%2B-or-Easy-Recursive-DFS-Solution-w-Explanation
        intutions:
            we shouldn't need to place a camera on the leaf nodes, if they are covered from avoer
            so we can solve this bottom up, dfs
            dfs on to each node, but have it return SOME kind of information back to the parent
        
        needs;
            nothing below needs monitoring
            a camera was placed below and can monitor the parent
            an unmonitored node below needs a camera placed above
        returns;
            if no child needs monitoring, we hold off on placing a camera, and indicate that this node is in this state
            one or more of the children need moniotring, so place here and return that this parent/node will be monitored
            onfe of hte children has a camera and the toerh child either has a camera or oden'st need moniotring
            (this tree is fully monitored but has not monitoring to the parent)
        '''
        self.ans = 0
        
        def _dfs(node: TreeNode):
            '''
            Returns:
                0: Ignore (M)
                1: Placed camera in child (C)
                3: Need camera in parent (U)
            '''
            if node is None: 
                return 0
            
            val = _dfs(node.left) + _dfs(node.right)
            if val == 0: 
                return 3
            if val < 3:
                # 0 or 1 or 2
                return 0
            
            # val >= 3 and therefore we need a camera in current node.
            self.ans += 1
            return 1
        
        return self.ans + 1 if _dfs(root) >= 3 else self.ans

#############################
# 361. Bomb Enemy
# 21JUN22
#############################
#TLE brute force, note JAVA passes
class Solution:
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        '''
        we can just check how many soldier we can kill if we place a bomb on this cell (i,j)
        can only place on a '0' cell
        '''
        if len(grid) == 0:
            return 0
        
        rows = len(grid)
        cols = len(grid[0])
        
        def get_kill_count(row,col):
            enemy_count = 0
            #look to the left of the cell
            for c in range(col-1,-1,-1):
                if grid[row][c] == 'W':
                    break
                elif grid[row][c] == 'E':
                    enemy_count += 1
            
            #look to the right
            for c in range(col+1,cols):
                if grid[row][c] == 'W':
                    break
                elif grid[row][c] == 'E':
                    enemy_count += 1
            
            #look up
            for r in range(row-1,-1,-1):
                if grid[r][col] == 'W':
                    break
                elif grid[r][col] == 'E':
                    enemy_count += 1
            
            #look down
            for r in range(row+1,rows):
                if grid[r][col] == 'W':
                    break
                elif grid[r][col] == 'E':
                    enemy_count += 1
            
            return enemy_count
        
        ans = 0
        for i in range(rows):
            for j in range(cols):
                ans = max(ans,get_kill_count(i,j))
        
        return ans

#consolidate range iterators
class Solution:
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        if len(grid) == 0:
            return 0

        rows, cols = len(grid), len(grid[0])

        def killEnemies(row, col):
            enemy_count = 0
            row_ranges = [range(row - 1, -1, -1), range(row + 1, rows, 1)]
            for row_range in row_ranges:
                for r in row_range:
                    if grid[r][col] == 'W':
                        break
                    elif grid[r][col] == 'E':
                        enemy_count += 1

            col_ranges = [range(col - 1, -1, -1), range(col + 1, cols, 1)]
            for col_range in col_ranges:
                for c in col_range:
                    if grid[row][c] == 'W':
                        break
                    elif grid[row][c] == 'E':
                        enemy_count += 1

            return enemy_count

        max_count = 0
        for row in range(0, rows):
            for col in range(0, cols):
                if grid[row][col] == '0':
                    max_count = max(max_count, killEnemies(row, col))

        return max_count

class Solution:
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        '''
        dp is actually pretty tricky, if we define dp(i,j) as the number of hits we get from this cell
        then it would just be:
        dp(i,j) = {
            dp(i,j-1) + dp(i+1,j) + dp(i.j-1) + dp(i,j+1)
        }
        since the bomb shoots acrros allrows and columns, we only need two of these answers from previous subproblems, the other two are redundant
        dp(i,j) = {
            dp(i-1,j) + dp(i,j+1)
        }
        
        if we knew the number of row hits and col hits obtain from placing a bomb at this cell, we could just sum them up
        but how do we build up the subproblems to get row hits and col hits
        
        case 1:
            if cell is at beginning, just scan row to get hits until we run into a wall, otherwise reset row hits
        case 2:
            if cell is right after a wall, we need to recalculate hits gain starting from this cell
            
        we only need to maintain hits along a row, but we need hits for each col
        '''
        if len(grid) == 0:
            return 0
        
        rows = len(grid)
        cols = len(grid[0])
        
        row_hits = 0
        col_hits = [0]*cols
        
        ans = 0
        
        for row in range(rows):
            for col in range(cols):
                #for these two loops, also could have done while loops
                #getting hit counter for this row
                if col == 0 or grid[row][col-1] == 'W':
                    row_hits = 0
                    for k in range(col,cols):
                        if grid[row][k] == 'W':
                            break
                        elif grid[row][k] == 'E':
                            row_hits += 1
                #getting col hits
                if row == 0 or grid[row-1][col] == 'W':
                    col_hits[col] = 0
                    for k in range(row,rows):
                        if grid[k][col] == 'W':
                            break
                        elif grid[k][col] == 'E':
                            col_hits[col] += 1
                
                #get total hits for this cell
                if grid[row][col] == '0':
                    total_hits = row_hits + col_hits[col]
                    ans = max(ans,total_hits)
        

def maxKilledEnemies(self, grid):
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    result = 0
    colhits = [0] * n
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if j == 0 or row[j-1] == 'W':
                rowhits = 0
                k = j
                while k < n and row[k] != 'W':
                    rowhits += row[k] == 'E'
                    k += 1
            if i == 0 or grid[i-1][j] == 'W':
                colhits[j] = 0
                k = i
                while k < m and grid[k][j] != 'W':
                    colhits[j] += grid[k][j] == 'E'
                    k += 1
            if cell == '0':
                result = max(result, rowhits + colhits[j])
    return 

#storing individual dp entries
class Solution:
    def maxKilledEnemies(self, grid: List[List[str]]) -> int:
        m,n=len(grid),len(grid[0])
        dp=[[0 for _ in range(n)] for _ in range(m)]
        #left->right
        for i in range(m):
            cur=0
            for j in range(n):
                if grid[i][j]=='W': cur=0
                if grid[i][j]=='E': cur+=1
                else: dp[i][j]=cur
        #right->left
        for i in range(m):
            cur=0
            for j in range(n-1,-1,-1):
                if grid[i][j]=='W': cur=0
                if grid[i][j]=='E': cur+=1
                else: dp[i][j]+=cur
        #up->down
        for j in range(n):
            cur=0
            for i in range(m):
                if grid[i][j]=='W': cur=0
                if grid[i][j]=='E': cur+=1
                else: dp[i][j]+=cur
        #down->up
        for j in range(n):
            cur=0
            for i in range(m-1,-1,-1):
                if grid[i][j]=='W': cur=0
                if grid[i][j]=='E': cur+=1
                else: dp[i][j]+=cur
                    
        return max(dp[i][j] for i in range(m) for j in range(n)) 