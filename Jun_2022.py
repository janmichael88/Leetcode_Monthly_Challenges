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