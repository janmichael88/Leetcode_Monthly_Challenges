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
