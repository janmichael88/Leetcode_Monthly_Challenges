#################################################
# 2071. Maximum Number of Tasks You Can Assign
# 01MAY25
#################################################
class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        '''
        in order to complete task[i], workers[j] >= task[i]
        we can use pills only once per worker, and it increease this workes strength by strength
        we wou;ndt want to use a stronger work to finish a task, if there is a weaker worker that can do it
        so save the strongest workes, 
        sort workers and tasks increasinly?
        '''
        tasks.sort()
        workers.sort()
        i,j = 0,0
        print(tasks)
        print(workers)
        
        while i < len(tasks) and j < len(workers):
            #worker is strong enough
            if workers[j] >= tasks[i]:
                i += 1
                j += 1
            #not strong enough, but we have pill to use to make it strong enough
            elif pills > 0 and workers[j] + strength >= tasks[i]:
                pills -= 1
                i += 1
                j += 1
            #try using stonger worker
            else:
                j += 1
        
        return i
    
#gahhh
class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        '''
        can't just pair smallest taks with smallest works, the reason is becaue of pill strength
        we can't priortize using a pill if a worker isn't strong enough
        so we shold save pills after using our strongest workers
        '''
        tasks.sort()
        workers.sort()
        i,j = 0,0

        #first pass, dont use pulls, then keep track of unised workers that can use pill
        unused_workers = [] #by index
        while i < len(tasks) and j < len(workers):
            #worker is strong enough
            if workers[j] >= tasks[i]:
                i += 1
                j += 1
            #try using stonger worker
            else:
                unused_workers.append(workers[j])
                j += 1
        unused_workers.sort(reverse = True)
        print(unused_workers)
        print(tasks[i:])
        #second pass?
        j = 0
        
        while i < len(tasks) and j < len(unused_workers):
            if pills > 0 and unused_workers[j] + strength >= tasks[i]:
                pills -= 1
                i += 1
                j += 1
            #try using stonger worker
            else:
                j += 1
        
        return i

from sortedcontainers import SortedList
class Solution:
    def maxTaskAssign(
        self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        '''
        binarry search, but maintain sortedset (list) when searching
        sortedlist with workers
        case 1:
            the worke with the highest available value is greater than or equal to the task value
            in this case, we do not need to use a pill
            we can assign this worker (with the maximum value) to task and remove from pool or workers
            why? since this is the most difficult task, any worker who can complete it, can comlete any otehr task
            so assign strongest worker available to the hargest task
        
        case 2:
            no woker can complete the task without a pill
            in this case we must use a pill
            look for the weakest worker who can complete the task with the pill (i.e worker with value >= t - strength)
            and remove them from pool
        
        iterate tasks and worrkers in increasing order
        '''
        n, m = len(tasks), len(workers)
        tasks.sort()
        workers.sort()

        def check(mid: int) -> bool:
            p = pills
            # Ordered set of workers
            #we need the m-mid worrkes from the end, for a given min
            ws = SortedList(workers[m - mid :])
            # Enumerate each task from largest to smallest
            for i in range(mid - 1, -1, -1):
                # If the largest element in the ordered set is greater than or equal to tasks[i]
                if ws[-1] >= tasks[i]:
                    ws.pop()
                else:
                    #no pills, cant doo
                    if p == 0:
                        return False
                    #use pill to find the worker that can do it
                    #ie look for tasks[i] - strength
                    rep = ws.bisect_left(tasks[i] - strength)
                    #can't do
                    if rep == len(ws):
                        return False
                    #use pill
                    p -= 1
                    #remove this worker
                    ws.pop(rep)
            #can do
            return True

        left, right, ans = 1, min(m, n), 0
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans

#################################################
# 3343. Count Number of Balanced Permutations
# 12MAY25
################################################
import math
class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        '''
        count number and then distribute them between odd and even positions usinng dp
            need to check all numbes from 0 to 9, and add them up
        given number 000111122, 
        we have 4 even positions and 5 odd positions
        to have it balance, we need 8/2 == 4 for sum of each spot

        0/3:
            There are comb(5, 3) = 10 ways to put 3 zeros into 5 odd positions.
            The result is 10 * dp(0 + 1, 4 - 0, 5 - 3, 4 - 0).
        1/2:
            There are comb(4, 1) = 4 ways to put 1 zero into 4 even positions.
            There are comb(5, 2) = 10 ways to put 2 zeros into 5 odd positions.
            The result is 4 * 10 * dp(1, 3, 3, 4).
        2/1:
            There are comb(4, 2) = 6 ways to put 2 zeros into 4 even positions.
            There are comb(5, 1) = 5 ways to put 1 zero into 5 odd positions.
            The result is 6 * 5 * dp(1, 2, 4, 4).
        3/0:
            There are comb(4, 3) = 4 ways to put 3 zeros into 4 even positions.
            The result is 4 * dp(1, 1, 5, 4).
            We continue the recursion to distribute 1 and 2. After distributing all numbers, we return 1 if the balanced sum is zero, and 0 otherwise.


        '''
        cnt = Counter(int(ch) for ch in num)
        total = sum(int(ch) for ch in num)

        @cache
        def dfs(curr_digit, odd, even, balance):
            if odd == 0 and even == 0 and balance == 0:
                return 1
            #possbile pruning herre
            if curr_digit > 9 or odd < 0 or even < 0 or balance < 0:
                return 0
            res = 0
            for num_times in range(0, cnt[curr_digit] + 1):
                ways_odd = math.comb(odd,num_times)
                ways_even = math.comb(even, cnt[curr_digit] - num_times)
                #if we place odd with j occurence, we place cnt[i] - j occurrences
                #for balace, we using the current digit j times
                #go up to the next digit, reduce odd positions, reduce even positionss, update balance
                #state definition is (odd spots, even sports, number, and balance)
                prev_ways = dfs(curr_digit + 1, odd - num_times, even - (cnt[curr_digit] - num_times), balance - curr_digit * num_times)
                res += ways_even * ways_odd * prev_ways
            return res % 1000000007

        return 0 if total % 2 else dfs(0, len(num) - len(num) // 2, len(num) // 2, total // 2)
    
############################################################
# 3335. Total Characters in String After Transformations I
# 13MAYY25
#############################################################
#ez
class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        '''
        we can't simulate, that would take to long,
        so calculate final transformation
        the presence of z, wil increase the string by size by 1
        for each char, count how many times that would move to z given t transformations
        if we have a z, it goes to ab, but ab, got also go to z again, and it can keep going if t is large enough
        we can do this independently
        z -> ab 
        a -> z after 26 transformations
        b -> z after 25 transformations
        but if there's enough t to get to z for every char wed get even more
        since there are only 26 letters, we can check!
        '''
        counts = [0]*26
        for ch in s:
            idx = ord(ch) - ord('a')
            counts[idx] += 1
        mod = 10**9 + 7
        #do this t times
        for _ in range(t):
            next_counts = [0]*26
            #if there's a z, we need increment a and b
            next_counts[0] += counts[-1] % mod
            next_counts[1] = (counts[0] + counts[-1]) % mod
            #promote everything else
            for i in range(2,26):
                next_counts[i] = counts[i-1]
            #swap
            counts = next_counts[:]
        
        return sum(counts) % mod


#count map gets TLE
class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        '''
        using count map
        '''
        counts = Counter(s)
        for _ in range(t):
            next_counts = Counter()
            #first add up z
            z_counts = counts['z']
            next_counts['a'] += z_counts
            next_counts['b'] = z_counts + counts['a']
            #promote the rest
            for i in range(2,26):
                curr_letter = chr(ord('a') + i)
                prev_letter = chr(ord('a') + i - 1)
                next_counts[curr_letter] = counts[prev_letter]
            
            counts = next_counts
        
        ans = 0
        mod = 10**9 + 7
        for k,v in counts.items():
            ans += v % mod
        
        return ans % mod

class Solution(object):
    def lengthAfterTransformations(self, s, t):
        mod = 10**9 + 7
        nums = [0]*26
        for ch in s:
            nums[ord(ch) - 97] += 1

        for _ in range(t):
            cur = [0]*26
            z = nums[25]
            if z:
                # 'z' → 'a' and 'b'
                cur[0] = (cur[0] + z) % mod
                cur[1] = (cur[1] + z) % mod
            for j in range(25):
                v = nums[j]
                if v:
                    cur[j+1] = (cur[j+1] + v) % mod
            nums = cur

        res = 0
        for v in nums:
            res = (res + v) % mod
        return res
    
#############################################################
# 3337. Total Characters in String After Transformations II 
# 14MAY25
############################################################
#numpy doesn't do modulo operations
import numpy as np
class Solution:
    def lengthAfterTransformations(self, s: str, t: int, nums: List[int]) -> int:
        '''
        for transfomrations, each char in s at i, for s[i] is replaced with the next nums[s[i] -'a'] in front of it
        with wrap around
        return length after t transformations
        model problem as matrix mulitplcation,
        use fast exponention to do it
        we are multiplying two matrices, (1 by 26)*(26 by 26) and do this t times
        then sum the final vector
        the hard part is determining the transformation matrix, we then exponentiate that t times and multiply it by the initial vector from s
        transformation matrix M[i][j] = 1
        means the ith char can generate to one j
        note numpy mat mult doesn't use module 10**0 + 7
        '''
       # initial freq vector
        u = [0] * 26
        for ch in s:
            u[ord(ch) - ord('a')] += 1
        
        # transformation matrix, the ith character and make an addditinoal j character with shifts away with wrap around
        M = [[0] * 26 for _ in range(26)]
        for i, shift in enumerate(nums):
            for j in range(i + 1, i + shift + 1):
                M[i][j % 26] = 1
        
        # cast as array
        M = np.array(M, dtype=np.int64)
        u = np.array(u, dtype=np.int64)

        # exponentiate
        M_to_t = np.linalg.matrix_power(M, t)

        # sum the vector
        final = u @ M_to_t

        return int(np.sum(final)) % (10**9 + 7)
    
import numpy as np
class Solution:
    def lengthAfterTransformations(self, s: str, t: int, nums: List[int]) -> int:
        '''
        for transfomrations, each char in s at i, for s[i] is replaced with the next nums[s[i] -'a'] in front of it
        with wrap around
        return length after t transformations
        model problem as matrix mulitplcation,
        use fast exponention to do it
        we are multiplying two matrices, (1 by 26)*(26 by 26) and do this t times
        then sum the final vector
        the hard part is determining the transformation matrix, we then exponentiate that t times and multiply it by the initial vector from s
        transformation matrix M[i][j] = 1
        means the ith char can generate to one j
        note numpy mat mult doesn't use module 10**0 + 7
        '''
        MOD = 10**9 + 7
        # initial counts
        u = [0] * 26
        for ch in s:
            u[ord(ch) - ord('a')] += 1

        # transformation matrix (something about i to something about j) in one step
        # remember this
        M = [[0] * 26 for _ in range(26)]
        for i, shift in enumerate(nums):
            for j in range(i + 1, i + shift + 1):
                M[i][j % 26] = 1

        # matrix multiplication
        def mat_mult(mat1, mat2, mod):
            res = [[0] * len(mat2[0]) for _ in range(len(mat1))]
            for i in range(len(mat1)):
                for j in range(len(mat2[0])):
                    for k in range(len(mat2)):
                        res[i][j] += mat1[i][k] * mat2[k][j]
                    res[i][j] %= mod
            return res


        def mat_pow(mat, exp):
            result = [[int(i == j) for j in range(26)] for i in range(26)]  # identity matrix
            while exp > 0:
                if exp % 2 == 1:
                    result = mat_mult(result, mat,MOD)
                mat = mat_mult(mat, mat, MOD)
                exp //= 2
            return result

        M_to_t = mat_pow(M, t)

        # Multiply vector u with matrix M_to_t
        final = mat_mult([u],M_to_t,MOD)

        return sum(final[0]) % MOD

##############################################
# 2999. Count the Number of Powerful Integers
# 14MAY25
##############################################
#digit dp 
#states are (pos,and limit at pos)
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        '''
        its limit at each position, we are bounded by it or not going from left to right
        recall each digit in a powerful integer cannot exceed leimit
        '''
        @cache
        def dfs(pos: int, lim: int,t):
            if len(t) < n:
                return 0
            if len(t) - pos == n:
                if lim:
                    #suffix ending
                    return int(s <= t[pos:])
                else:
                    return 1
            up = 9
            #if we have a limit at this position, we can go up to t hat
            if lim:
                up = min(int(t[pos]),limit)
            #if we don't we could go up to 9 or the limit
            else:
                up = min(9,limit)
            #up = min(int(t[pos]) if lim else 9, limit)
            ans = 0 #add up ways
            for i in range(up + 1):
                #move up position, and check limit
                next_limit = lim and i == int(t[pos])
                ans += dfs(pos + 1, next_limit,t)
            return ans

        n = len(s)
        return dfs(0,True,str(finish)) - dfs(0,True,str(start - 1))
    
######################################################
# 2900. Longest Unequal Adjacent Groups Subsequence I
# 15MAY25
######################################################
class Solution:
    def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        '''
        we need to return the subsequence, not just its length!
        '''
        ans = [0]
        for i in range(1,len(groups)):
            if groups[i] != groups[ans[-1]]:
                ans.append(i)
        return [words[i] for i in ans]

#one pass
class Solution:
    def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        '''
        we need to return the subsequence, not just its length!
        '''
        ans = [words[0]]
        for i in range(1,len(groups)):
            if groups[i] != groups[i-1]:
                ans.append(words[i])
        return ans
    
class Solution:
    def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        '''
        dp solution
        examine from each index i and groups, but keep track of indices
        '''
        best_idxs = []
        n = len(groups)
        for i in range(n):
            curr_idxs = [i]
            for j in range(i+1,n):
                if groups[j] != groups[curr_idxs[-1]]:
                    curr_idxs.append(j)
            if len(curr_idxs) > len(best_idxs):
                best_idxs = curr_idxs
            
            print(best_idxs)
        
        return [words[i] for i in best_idxs]
    
#dp, but returning path, 
class Solution:
    def getLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        '''
        dp with path tracking, 
        whenver we make an update to a new optimuma, we need record the previous state where we were at optimum
        to the next state that is the new optimum, in this the states are idxs
        we are adding so it needs to be ending
        this is the main point of the problem
        
        n = len(words)
        dp = [1]*n
        for i in range(n):
            for j in range(i-1,-1,-1):
                if groups[i] != groups[j]:
                    dp[i] = max(dp[i],dp[j] + 1) 
        
        print(dp)
        now how can we track paths though?
        '''
        n = len(words)
        dp = [1]*n
        prev = [-1]*n
        longest = 1
        end_idx = 0
        for i in range(n):
            #store best optimum so far before the dp update!
            curr_best = dp[i]
            prev_best = prev[i]
            for j in range(i-1,-1,-1):
                if groups[i] != groups[j] and dp[j] + 1 > curr_best:
                    curr_best = dp[j] + 1
                    prev_best = j
            dp[i] = curr_best
            prev[i] = prev_best
            if dp[i] > longest:
                longest = dp[i]
                end_idx = i
            
        #follow pointers back
        print(prev)
        ans = []
        curr = end_idx
        while curr != -1:
            ans.append(words[curr])
            curr = prev[curr]
        
        return ans[::-1]

#######################################################
# 2901. Longest Unequal Adjacent Groups Subsequence II
# 16MAY25
#######################################################
#series questiosn this week!
class Solution:
    def getWordsInLongestSubsequence(self, words: List[str], groups: List[int]) -> List[str]:
        '''
        hamming distance between two strings, if two strings are equal in length, its the number of unequal spots
        needs longest subsequence such that
            adjacent strings are unequal and adjacent words are equal in length
            and hamming distance between them is == 1
        
        same as the previous one, but add in contraint of hamming distance
        '''
        n = len(words)
        dp = [1]*n
        prev = [-1]*n
        longest = 1
        end_idx = 0
        for i in range(n):
            #store best optimum so far before the dp update!
            curr_best = dp[i]
            prev_best = prev[i]
            for j in range(i-1,-1,-1):
                w1 = words[i]
                w2 = words[j]
                if groups[i] != groups[j] and len(w1) == len(w2) and self.hamming(w1,w2) == 1 and dp[j] + 1 > curr_best:
                    curr_best = dp[j] + 1
                    prev_best = j
            dp[i] = curr_best
            prev[i] = prev_best
            if dp[i] > longest:
                longest = dp[i]
                end_idx = i
        #follow pointers back
        print(prev)
        ans = []
        curr = end_idx
        while curr != -1:
            ans.append(words[curr])
            curr = prev[curr]
        
        return ans[::-1]
    
    def hamming(self,w1,w2):
        h = 0
        for x,y in zip(w1,w2):
            h += x != y
        return h
        
###################################
# 3024. Type of Triangle
# 19MAY25
###################################
class Solution:
    def triangleType(self, nums: List[int]) -> str:
        '''
        count and check
        you first need to check if it can make a valid triangle
        '''
        #check valid triangle first
        if not self.is_valid_triangle(nums):
            return "none"
        counts = Counter(nums)
        for k,v in counts.items():
            if v == 3:
                return "equilateral"
            elif v == 2:
                return "isosceles"
        return "scalene"
    
    def is_valid_triangle(self,nums) -> bool:
        a,b,c = nums
        return a + b > c and a + c > b and b + c > a

class Solution:
    def triangleType(self, nums: List[int]) -> str:
        '''
        if we sort, we can just check the two smallest sides to the larger side, 
            if its <= the third, a trianlge can't be made
        '''
        nums.sort()
        if nums[0] + nums[1] <= nums[2]:
            return "none"
        elif nums[0] == nums[2]:
            return "equilateral"
        elif nums[0] == nums[1] or nums[1] == nums[2]:
            return "isosceles"
        else:
            return "scalene"
        
###################################################
# 1931. Painting a Grid With Three Different Colors
# 19MAY25
####################################################
#top down
#states are (col,and prev bit mask)
class Solution:
    def colorTheGrid(self, m: int, n: int) -> int:
        '''
        we have m rows by n cols
        there will be at most 5 rows and 1000 columns
        if we encode each column using a mask and if there are 3 colors
            1,2,3 -> {01,10,11}
        we can space of these using two bit positions, 
        there can be at most 10 bit positions, 2**10 = 1024, so we at least 1024*1000
        for example, if there are 5 rows, we can color column using this bitmask
        (11 01 11 01 11), which is 887
        curr = colors for the current jth column
        prev = colors for the previous columns
        when we reach the end of the column, we start a new one, prev = curr, and curr = 0
        for each position, we try each color, provided its not the same as the one on the left and up
            we are doing down to right, so we only need to check left and up
        '''
        memo = {}
        mod = 10**9 + 7
        def dp(i,j,curr,prev):
            #reached end of row, color next col, so increase j by 1, start from top
            #swap prev with curr and new bit mask here
            if i == m:
                return dp(0,j+1,0,curr)
            #reached end, so we have a valid coloring
            if j == n:
                return 1 #reached the last column
            if i == 0 and (j,prev) in memo: #coloring the first row, we don't need to check up and left
            #just check curr col and previous for answer, idk why though
                return memo[(j,prev)]
            ways = 0
            #if its first cell assumes previous cell is white (we can take any color for the next)
            # up =(cur >>  ((i - 1) * 2))  -> get the last cell color. if you are are 4th cell . 
            # means cur store color of last three cell which is of 6 bit. you shift it by 4 bit to get the color of only
            #last cell. AND  it with 3(11)  to clear other bits(in case any).
            # Same logic applied to find left cell color. We shift prev pattern to get the color of left cell in prev pattern.
            if i == 0:
                up = 0
            else:
                up = (curr >> ((i - 1) * 2)) & 3
            left = (prev >> (i*2)) & 3
            for k in range(1,3+1): #try all three colors
                #try differnt color, make sure its not left and up
                if k != left and k != up:
                    #add up ways, push to next row, but stay in col jj
                    #color this position, we can't use | since each cell takes of two bit position
                    #so we choose the color k shifted i*2
                    ways = (ways + dp(i + 1, j, curr + (k << (i * 2)), prev)) % mod
            
            memo[(j,prev)] = ways
            return ways
        
        return dp(0,0,0,0)

#using arrays instead of bitmasks
class Solution:
    def colorTheGrid(self, m: int, n: int) -> int:
        '''
        try using array, instead of mask
        this is really just push dp
        '''
        memo = {}
        mod = 10**9 + 7
        def dp(i,j,curr_col,prev_col):
            #last row, move on
            if i == m:
                return dp(0,j+1,[0]*m,curr_col)
            #valid column, only gotten here if we've gone through the whole grid
            if j == n:
                return 1
            #state compression, memo check
            #idk why it works though
            if i == 0 and (j,tuple(prev_col)) in memo:
                return memo[(j,tuple(prev_col))]
            
            ways = 0
            up = curr_col[i - 1] if i > 0 else 0
            left = prev_col[i]
            for other_color in range(1,3+1):
                #need to be able to look up and left
                if other_color != up and other_color != left:
                    next_col = curr_col[:]
                    next_col[i] = other_color
                    ways += dp(i+1,j,next_col,prev_col)
                    ways %= mod
            ways %= mod
            memo[(j,tuple(prev_col))] = ways
            return ways
        
        return dp(0,0,[0]*m,[0]*m)
    
############################################# 
# 3362. Zero Array Transformation III
# 24MAY25
#############################################
class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        '''
        we can only decrement a value in nums by at most 1 for a given query
        we can choose any element in the range for a given query
            so if a number is already zero, don't touch it, which is the same as decremeting by 0
        
        sort the queries and greedily apply them
        says greedily choose for furthest ending point, but i thought it would be queries that have the largest range
        if we can use i queries, then we can remove len(queries)- i

        binary search on the sorted queries, nah nice try though
        '''
        queries.sort(key = lambda x: x[0])
        left = 0
        right = len(queries) - 1
        ans = -1

        while left <= right:
            mid = left + (right - left) // 2
            if self.can_do(nums,queries,mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        if ans == -1:
            return ans
        
        return len(queries) - ans - 1
    
    def can_do(self,nums,queries,k):
        n = len(nums)
        indices_touched = [0]*(n+1)
        for l,r in queries[:k+1]:
            indices_touched[l] += 1
            indices_touched[r+1] -= 1
        for i in range(1,n+1):
            indices_touched[i] += indices_touched[i-1]
        
        #try bringin down indices
        for i in range(n):
            if nums[i] == 0:
                continue
            if nums[i] > indices_touched[i]:
                return False
        
        return True
    
#two heaps
class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        '''
        hint says to sort the queries and greedily pick
        essentially we need to cover each index i with at least nums[i] of the given intervals
        if we are at some index i, we want the largest ending, so that way we can cover more indices
            greedily picking the one with the furthest end minimizes the total picks
        
        two heaps,
            one is like a garbage heap
        '''
        n = len(nums)
        q = len(queries)
        starts = [[] for _ in range(n)] #for each start,their ends
        for l,r in queries:
            starts[l].append(r)
        
        available = [] #max heap of ends
        active = [] #min heap of ends
        chosen = 0

        for i in range(n):
            #for an index i, load up all possible ends, remember for an index, pick the furthest one
            for end in starts[i]:
                heapq.heappush(available, - end)
            #pop from min-heap any intervals whose end < i (since we can't cover them)
            while active and active[0] < i:
                heapq.heappop(active)
            #compute how many more intervals we need
            #this is how many times nums[i] can be brought to zero
            need = nums[i] - len(active)
            for _ in range(need):
                while available and -available[0] < i:
                    heapq.heappop(available)
                #can't do
                if not available:
                    return -1
                r = -heapq.heappop(available)
                heapq.heappush(active,r)
                chosen += 1
        
        return q - chosen
    
class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        queries.sort(key=lambda x: x[0])
        heap = []
        deltaArray = [0] * (len(nums) + 1)
        operations = 0
        j = 0
        for i, num in enumerate(nums):
            operations += deltaArray[i]
            while j < len(queries) and queries[j][0] == i:
                heappush(heap, -queries[j][1])
                j += 1
            while operations < num and heap and -heap[0] >= i:
                operations += 1
                deltaArray[-heappop(heap) + 1] -= 1
            if operations < num:
                return -1
        return len(heap)
    
#########################################################################
# 2131. Longest Palindrome by Concatenating Two Letter Words (REVISTED)
# 25MAY25
#########################################################################
class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        '''
        words are only two chars long
        we can pair x and x' if x == reversed(x')
        the extra additions are min(count(x),count(x'))
        now for x's thare are the same,
            like aa,bb,cc...
            if there are an even number, we can take all of them
            if there are an odd number, we can take all but one
            we can still do take only one char of ther repeated
        '''
        counts = Counter(words)
        ans = 0
        center = False
        for w in counts:
            #same
            if w[0] == w[1]:
                if counts[w] % 2 == 0:
                    ans += 2*counts[w]
                else:
                    ans += 2*(counts[w] - 1) #need to include one less
                    center = True
            elif w[0] != w[1]:
                ans += 2*min(counts[w],counts[w[::-1]])
        if center:
            ans += 2 #put back in
        return ans

################################################
# 3068. Find the Maximum Sum of Node Values
# 25MAY25
#################################################
#close one.....
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        '''
        anytime we take some edge (u,v), we need to update nums[u] nums[u] = nums[u] % k
        and nums[v] = nums[v] % k
        we want to this any number of times that gives us the max sum
        does repeating an edge make a difference?
        we are always doing nums[i] = nums[i] XOR k
        for an edge, we either do it or we dont
            try it and take max
        '''
        memo = {}
        n = len(nums)
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        def dp(i,prev):
            if i in memo:
                return memo[i]
            ans = 0
            for neigh in graph[i]:
                if neigh != prev:
                #take edge or not
                    take = (nums[i] ^ k) + (nums[neigh] ^ k)
                    no_take = nums[i] + nums[neigh]
                    temp = max(take,no_take)
                    ans = max(ans,temp + dp(neigh,i))
            memo[i] = ans  
            return ans
        
        ans = 0
        for i in range(n):
            ans = max(ans,dp(i,-1))
        
        return ans


class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        '''
        its a tree, its completely connected, undirected, ans has no cycles
        so every node is reachable from another node
            i.e for all nodes u, every other node v is reachable via some path l
        recall properties fro XOR
        aXORb = bXORa
        aXORa = 0
        0XORb = b
        so we can have; aXORbXORb = aXOR0 =a
        going back to the path with l, if we apply the operations on each node in L
        intermediate nodes will remain unchanged, but the ending nodes in path (u,v) end up getting udated
        we can update their values in nums as if they were already connected
        for an pair of nodes (u,v) along some path l
            if an operation has been done an even number of times, nums[u] and nums[v] end up getting updated
        so states are (node,parity) #pairty of ops (i.e) number of times a node has been updated
            we only apply the ops, if done and even number of times
            otherwise state is invalid
        
        essentially for any pair of nodes (u,v), nums[u] ans nums[v] get updated to nums[u] ^ k
        if we have hit this node an even number of times
        if we don't, it never gets updated
        '''
        memo = {}
        def dp(i,parity):
            if i == len(nums):
                if parity % 2 == 0:
                    return 0
                return float('-inf')
            if (i,parity) in memo:
                return memo[(i,parity)]
            
            no_take = nums[i] + dp(i+1,parity)
            take = (nums[i] ^ k) + dp(i+1,parity ^ 1)
            ans = max(no_take,take)
            memo[(i,parity)] = ans
            return ans
        
        return dp(0,0)
    
#bottom up
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        '''
        bottom up
        '''
        n = len(nums)
        dp = [[0]*2 for _ in range(n+1)]
        dp[n][0] = 0
        dp[n][1] = float('-inf')

        for i in range(n-1,-1,-1):
            for parity in range(2):
                no_take = nums[i] + dp[i+1][parity]
                take = (nums[i] ^ k) + dp[i+1][parity ^ 1]
                ans = max(no_take,take)
                dp[i][parity] = ans
        
        return dp[0][0]
    
###############################################
# 1857. Largest Color Value in a Directed Graph
# 26MAY25
###############################################
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        '''
        could there be a cycle and a valid path?
        of is it the case if there's a cycle immediately return -1
        need path with largest color value
            color value is number of nodes in path that have most frequent color
        top sort and dp
        dp[u][c] = max count of vertices with color c of any path starting from u
        for top sort, start at the leaves
        '''
        graph = defaultdict(list)
        n = len(colors)
        degree = [0]*n
        for u,v in edges:
            graph[u].append(v) #its directed no back edge
            degree[v] += 1
        
        q = deque([])
        for i in range(n):
            if degree[i] == 0:
                q.append(i)
    
        dp = [[0]*26 for _ in range(n)]
        #to check for cycle, make sure we visit all nodes
        nodes_seen = 0
        ans = 0
        #top sort
        while q:
            curr = q.popleft()
            nodes_seen += 1 #mark seen
            color_idx = ord(colors[curr]) - ord('a')
            #add in color count
            dp[curr][color_idx] += 1
            ans = max(ans, dp[curr][color_idx]) #maximize, arriving at this node with this color
            for neigh in graph[curr]:
                #update for all previous colors
                for i in range(26):
                    #for each neighbor, update the max
                    dp[neigh][i] = max(dp[neigh][i],dp[curr][i])
                #top sort addition
                degree[neigh] -= 1
                if degree[neigh] == 0:
                    q.append(neigh)
        if nodes_seen < n:
            return -1
        return ans

#######################################################
# 2894. Divisible and Non-divisible Sums Difference
# 27MAY24
#########################################################
class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        '''
        for numbers divible by m
        we have num1 = m + 2m + 3m .... 
        we need to find the number of nums from 1 to n
            k = n//m
        so num1 is (k*(k+1)//2) *m, i.e sum of all nums from 1 to n divisble by m
        to find ones not divisible by m, its just:
        n*(n+1)//2 - (k*(k+1)//2) *m
        the ans is just nums2 subtracted again
        n*(n+1)//2 - (k*(k+1)//2) *m - (k*(k+1)//2) *m
        n*(n+1)//2 - (k*k(+1))*m
        '''
        k = n//m
        return (n*(n+1)) // 2 - (k*(k+1))*m