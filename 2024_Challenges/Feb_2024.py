######################################################
# 2966. Divide Array Into Arrays With Max Difference
# 01FEB24
######################################################
#inteligently build the array
class Solution:
    def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
        '''
        need to divide array into one or more arrays of size 3,
        need to satisfy:
            1. each element of nums should be in exactly one array (basically enforcing multiplicty)
            2. difference between any to elements in on array is <= k
        well first of all it needs to be a multiple of 3
        sort array and group by 3
        '''
        N = len(nums)
        if N % 3 != 0:
            return []
        
        nums.sort()
        ans = []
        for i in range(0,N,3):
            candidate = nums[i:i+3]
            #all should be less than k
            is_valid = True
            for i in range(3):
                for j in range(3):
                    if i != j and candidate[i] - candidate[j] > k:
                        is_valid = False
                        return []
            if is_valid:
                ans.append(candidate)
        
        return ans
    
class Solution:
    def divideArray(self, nums: List[int], k: int) -> List[List[int]]:
        '''
        we can just sort and check groups of three, for each group of three, we check last with the first,
        since they are sorted, this would be the largest difference between and two elements and any pair in between would be smaller
        examine an array like:
        [a, a+1, a+2, a+3, a+4, a+5], where k = 2
        we can partition like: [[a, a+1, a+2], [a+3, a+4, a+5]
        if we were to partition like: [a+2, a+3, a+4], the other part would have to be [a, a+1, a+5]
        and this lart part would not be valid
        intuition is to keep values close together
        the greedy proof is actually tricky, since we sort, if we can't amke a group with this current parttion of three, its impossible to make a group after that,
        since the numbers are sorted
        '''
        N = len(nums)
        if N % 3 != 0:
            return []
        
        nums.sort()
        ans = []
        for i in range(0,N,3):
            candidate = nums[i:i+3]
            if candidate[-1] - candidate[0] > k:
                return []
            ans.append(candidate)
        
        return ans
    
##########################################
# 1291. Sequential Digits (REVISTED)
#  02FEB24
##########################################
#ez peez recursion
class Solution:
    def sequentialDigits(self, low: int, high: int) -> List[int]:
        '''
        for it to be sequential it must be a substring of '123456789'
        build number recursively and check if in bounds
        '''
        ans = []
        digits = [1,2,3,4,5,6,7,8,9]
        N = len(digits)
        
        def rec(i,num,low,high):
            if low <= num <= high:
                ans.append(num)
            if i >= N:
                return
            if num > high:
                return            
            rec(i+1, num*10 + digits[i],low,high)
        
        
        for i in range(N):
            rec(i,0,low,high)
        return sorted(ans)
    
###########################################
# 1043. Partition Array for Maximum Sum
# 03FEB24
###########################################
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        '''
        for sum index i, i can make a subarray using i to i+k
        try all subarrays? and find max
        maintain dp array dp(i,j) is max valu starting with i and up to j
        then just grab max from the columns
        no we can just look to the right
        for some index i, we need to find the max value of a subarray it would be a part of
        
        states is just some index i
        for this index i treat as max and increment ans by arr[i]*(length of subarray) + some rec function

        treat as knapsack
        for some index i, pick as the max element, then increase the contribution by the maximum element times the size of the subarray
        then add this to the next dp(j+1) state, and maximize along the way
        or we choose to end the current subarray
        '''
        N = len(arr)
        memo = {}
        
        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            curr_max = 0
            ans = 0
            end = min(N,i+k)
            for j in range(i,end):
                curr_max = max(curr_max, arr[j])
                ans = max(ans, curr_max*(j - i + 1) + dp(j+1))
            
            memo[i] = ans
            return ans
        
        return dp(0)

#bottom up
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        '''
        '''
        N = len(arr)
        dp = [0]*(N+1)
        
        for i in range(N-1,-1,-1):
            curr_max = 0
            end = min(N,i+k)
            for j in range(i,end):
                curr_max = max(curr_max, arr[j])
                dp[i] = max(dp[i], curr_max*(j-i+1) + dp[j+1])
            
        return dp[0]
    
#note: wouldn't it work if we wanted to find the maximum i for each arr[i] for each i
    
####################################################
# 76. Minimum Window Substring (REVISTED)
# 04FEB24
####################################################
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        '''
        need to find min substring of s, such that every char in t (including counts) is in the window
        maintain sliding window, we have a valid substring if the counts in t are all zero
        '''
        count_t = Counter(t)
        count_s = Counter()
        
        def isValid(count_s,count_t):
            #should be eqaul in cont_s
            for k,v in count_t.items():
                #we dont have enough
                if count_s[k] < v:
                    return False
            return True
        
        N = len(s)
        left = right = 0
        ans = float('inf')
        smallest_left, smallest_right = -1,-1
        
        while right < N:
            #expand
            curr_char = s[right]
            count_s[curr_char] += 1
            
            #try to shrink
            while left <= right and isValid(count_s,count_t):
                curr_char = s[left]
                count_s[curr_char] -= 1
                
                curr_size = right - left + 1
                if curr_size < ans:
                    ans = curr_size
                    smallest_left = left
                    smallest_right = right
                left += 1
            
            right += 1
        
        if ans == float('inf'):
            return ""
        return s[smallest_left:smallest_right+1]

#we can filter s, for only the chars that are in t
#insteaf of advancing 1 by 1, we skip right to the next position
#so insteaf of going through the entire string s, we move through an array of letters and their indices
#only if chars are in t
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        '''
        need to find min substring of s, such that every char in t (including counts) is in the window
        maintain sliding window, we have a valid substring if the counts in t are all zero
        '''
        count_t = Counter(t)
        count_s = Counter()
        
        def isValid(count_s,count_t):
            #should be eqaul in cont_s
            for k,v in count_t.items():
                #we dont have enough
                if count_s[k] < v:
                    return False
            return True
        
        filtered_s = []
        for i,ch in enumerate(s):
            if ch in count_t:
                filtered_s.append((ch,i))
        
        N = len(filtered_s)
        left = right = 0
        ans = float('inf')
        smallest_left, smallest_right = -1,-1
        
        while right < N:
            #expand
            curr_char,right_ptr = filtered_s[right]
            count_s[curr_char] += 1
            
            #try to shrink
            while left <= right and isValid(count_s,count_t):
                curr_char,left_ptr = filtered_s[left]
                count_s[curr_char] -= 1
                
                curr_size = right_ptr - left_ptr + 1
                if curr_size < ans:
                    ans = curr_size
                    smallest_left = left_ptr
                    smallest_right = right_ptr
                left += 1
            
            right += 1
        
        if ans == float('inf'):
            return ""
        return s[smallest_left:smallest_right+1]

#####################################################
# 2800. Shortest String That Contains Three Strings
# 02FEB24
####################################################
class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        '''
        strings a,b,c must exists
        i can concatenate a,b,c together this would be the maximum
        we also need the lexographically smallest one
        try all permutations that can generate a,b,c 
            a,b,c
            a,c,b
            b,c,a ...
            there are 6 of them
        
        '''
        def merge(s1,s2):
            #if s2 in s1, best we can so is s1
            if s2 in s1:
                return s1
            #otherwise find prefix that matches suffix
            #we are checking suffixes of s1
            #we can speed this up with kmp
            for i in range(len(s1)):
                if s2.startswith(s1[i:]): #longest shared segment between s1 and s2
                    return s1[:i] + s2 #if s2 starts with this suffix of s1, we can concatenate prefix with s2
            
            #otherwise concat both
            return s1+s2
        
        res = ""
        size = float('inf')
        for s1,s2,s3 in itertools.permutations([a,b,c]):
            s = merge(merge(s1,s2),s3)
            if len(s) < size:
                #smallest size
                res = s
                size = len(s)
            elif len(s) == size:
                #equal size, want lexographically smallest one
                res = min(res,s)
        
        return res
    
#another solution
#https://leetcode.com/problems/shortest-string-that-contains-three-strings/discuss/3836344/Python-Check-all-Permuations
class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def f(a, b):
            if b in a: return a
            for k in range(len(a), -1, -1):
                if a.endswith(b[:k]):
                    return a + b[k:]
        return min((f(f(a,b), c) for a,b,c in permutations((a,b,c))), key=lambda a: (len(a), a))
    
#############################################################
# 2982. Find Longest Special Substring That Occurs Thrice II
# 05FEB24
#############################################################
class Solution:
    def maximumLength(self, s: str) -> int:
        '''
        a special string is a string that contains only one character
        want the length of the longest special substring which occurs at least thrice
        sliding window, and for each speical substring, store its number of occurences, 
        find the ones that occur three times, and take the lognest by length
        use dp array to keep track of the longest special string ending at i
        then it becomes a conting problem
        if dp[i] = 3
        then for speicial string length 1, we have 3
        speicial string length 2 of we have 2
        special string length 3 we have 1
        so its 3 - size + 1
        
        need to store counts by char
        '''
        N = len(s)
        counts = defaultdict(Counter)
        prev = s[0]
        length = 1
        counts[prev][length] += 1
        
        for i in range(1,N):
            curr = s[i]
            #extend
            if curr == prev:
                length += 1
                counts[prev][length] += 1
            else:
                length = 1
                counts[curr][length] += 1
                prev = curr
        
        #search for one that occurs thrices
        ans = -1
        
        for i in range(26):
            char = chr(ord('a') + i)
            sum_ = 0
            #try all sizes from N to 1,
            #this is a cunting problem
            for j in range(N,0,-1):
                sum_ += counts[char][j]
                if sum_ >= 3:
                    ans = max(ans,j)
                    break
        return ans
    
#####################################################
# 2273. Find Resultant Array After Removing Anagrams
# 06FEB24
#####################################################
class Solution:
    def removeAnagrams(self, words: List[str]) -> List[str]:
        '''
        just keep checking anagrams, use delete 
        '''
        def encode(word):
            counts = [0]*26
            for ch in word:
                counts[ord(ch) - ord('a')] += 1
            
            return tuple(counts)
        
        
        while True:
            to_delete = set()
            N = len(words)
            for i in range(N-1):
                if encode(words[i]) == encode(words[i+1]):
                    to_delete.add(i+1)
            
            if len(to_delete) == 0:
                break
            
            #remove words
            new_words = []
            for i in range(N):
                if i not in to_delete:
                    new_words.append(words[i])
            
            words = new_words
        
        return words
            
class Solution:
    def removeAnagrams(self, words: List[str]) -> List[str]:
        '''
        #groupby?
        for a,b in groupby(words,sorted):
            print(a,b)
            print("----")
            print(next(b))
        '''
        return [next(g) for _,g in groupby(words,sorted)]

#################################################
# 1463. Cherry Pickup II (REVISTED)
# 11FEB24
################################################
#4 state
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        '''
        three state dp; cell (i,j) and k (robot 0 or robot 1)
        robot 1 is initially on (0,0) and robot 2 is one (0,cols-1)
        robots need to reach the bottom row
        we can start iwht 4 states (x_1,y_1,x_2,y_2), where we store the max cherries for each state
        the we move
        notice how we can only move down
        '''
        rows,cols = len(grid),len(grid[0])
        memo = {}
        
        def dp(x1,y1,x2,y2):
            #reached the bottomw row and are in bounds
            if (x1 == rows - 1) and (x2 == rows - 1):
                if (0 <= y1 < cols) and (0 <= y2 < cols):
                    if y1 != y2:
                        return grid[x1][y1] + grid[x2][y2]
                    else:
                        return grid[x1][y1]
            #out of bounds robot 1
            if (x1 < 0) or (x1 == rows) or (y1 < 0) or (y1 == cols):
                return float('-inf')
            if (x2 < 0) or (x2 == rows) or (y2 < 0) or (y2 == cols):
                return float('-inf')
            if (x1,y1,x2,y2) in memo:
                return memo[(x1,y1,x2,y2)]
            ans = float('-inf')
            
            #find next states, such that they are not overlapping
            next_states = set()
            for a in [-1,0,-1]:
                for b in [-1,0,1]:
                    temp = (x1 + 1, y1 + a, x2 + 1, y2 + b)
                    next_states.add(temp)
            for state in next_states:
                #can only take one
                if y1 == y2:
                    ans = max(ans, grid[x1][y1] + dp(*state))
                #take both
                else:
                    ans = max(ans, grid[x1][y1] + grid[x2][y2] + dp(*state))
            memo[(x1,y1,x2,y2)] = ans
            return ans
        
        
        return dp(0,0,0,cols-1)

class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        rows,cols = len(grid),len(grid[0])
        memo = {}
        
        def dp(x1,y1,x2,y2):
            #reached the bottomw row and are in bounds
            if (x1 == rows - 1) and (x2 == rows - 1):
                if (0 <= y1 < cols) and (0 <= y2 < cols):
                    return 0
                else:
                    return float('-inf')
            #out of bounds robot 1
            if (x1 < 0) or (x1 == rows) or (y1 < 0) or (y1 == cols):
                return float('-inf')
            if (x2 < 0) or (x2 == rows) or (y2 < 0) or (y2 == cols):
                return float('-inf')
            if (x1,y1,x2,y2) in memo:
                return memo[(x1,y1,x2,y2)]
            cherries = 0
            cherries += grid[x1][y1]
            ans = 0
            if y1 != y2:
                cherries += grid[x2][y2]
            
            #find next states, such that they are not overlapping
            next_states = set()
            for a in [-1,0,-1]:
                for b in [-1,0,1]:
                    temp = (x1 + 1, y1 + a, x2 + 1, y2 + b)
                    next_states.add(temp)
            for state in next_states:
                ans = max(ans, dp(*state))
            memo[(x1,y1,x2,y2)] = ans + cherries
            return ans +cherries
        
        
        return dp(0,0,0,cols-1)

#reduction to 3,becuase we alwyas move down 1
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        @lru_cache(None)
        def dfs(r, c1, c2):
            if r == m: return 0
            cherries = grid[r][c1] if c1 == c2 else grid[r][c1] + grid[r][c2]
            ans = 0
            for nc1 in range(c1 - 1, c1 + 2):
                for nc2 in range(c2 - 1, c2 + 2):
                    if 0 <= nc1 < n and 0 <= nc2 < n:
                        ans = max(ans, dfs(r + 1, nc1, nc2))
            return ans + cherries

        return dfs(0, 0, n - 1)
    
    
#####################################################
# 2787. Ways to Express an Integer as Sum of Powers
# 08FEB24
######################################################
#MLE, too many states
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        '''
        generate powers of x, then dp
        oooo it must be unique1
        i can keep a mask of what number ive taken
        no i can use another index into the powers array
        so it becomes dp(i,j)
        
        
        '''
        powers = []
        
        i = 1
        while i**x <= n:
            powers.append(i**x)
            i += 1
        N = len(powers)
        
        
        memo = {}
        mod = 10**9 + 7
        
        def dp(i,j):
            if i == 0:
                return 1
            if j >= N:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            #knapsack, take this jth one and increment
            #or skip ip
            take = 0
            if i - powers[j] >= 0:
                take = dp(i - powers[j],  j+1)
            no_take = dp(i,j+1)
            ways = take + no_take
            ways %= mod
            memo[(i,j)] = ways
            return ways
        
        
        return dp(n,0)
    

class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        '''
        i dont need to generate all powers
        state is (num,largest power)
        
        '''
        MOD = 10 ** 9 + 7
        memo = {}
        def dfs(n, x, m):
            r = n - m ** x
            if r == 0:
                return 1
            if (n,m) in memo:
                return memo[(n,m)]
            elif r > 0:
                ans = dfs(n, x, m + 1) + dfs(r, x, m + 1)
                ans %= MOD
                memo[(n,m)] = ans
                return ans
            else:
                return 0
        return dfs(n, x, 1) % MOD
    
###################################################
# 2943. Maximize Area of Square Hole in Grid
# 11FEB24
###################################################
#FML, the bars are already there, we just need to chose from hBars and vBars
class Solution:
    def maximizeSquareHoleArea(self, n: int, m: int, hBars: List[int], vBars: List[int]) -> int:
        '''
        the hold needs to be square, 
        the initial grid however does not need to be square
        need to sort hBars and vBars consider them seperately
        compute the longest sequnece of sonectuive itereges values in each array
        notice the dimensinos of the grid are (n + 2) x (m + 2)
        if i have an array like 3,4,5,6,7,  i can remove 4,5,6 and keeep 7 as the boundaries
        fuck the bars could be repeated, need to remove duplicatees
        
        omg all the bars are already there, we just need to 
        '''
        #need longest consecutive streak
        def getLongest(barset):
            longest = 1
            for num in barset:
                size = 1
                curr = num
                while curr + 1 in barset:
                    curr += 1
                    size += 1
                
                longest = max(longest,size)
            
            return longest
        
        if len(hBars) == 0 or len(vBars) == 0:
            return 1
        hBars = set(hBars)
        vBars = set(vBars)
        
        longest_h = getLongest(hBars)
        longest_v = getLongest(vBars)
        
        edge = min(longest_h,longest_v) + 1
        return edge*edge

###########################################
# 2149. Rearrange Array Elements by Sign
# 14FEB2
###########################################
class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        '''
        need to preserve order
        every consecutive pair of integers must have opposite signs
        start with positive
        i can pull apart with two lists and rebuidlg
        '''
        pos = []
        neg = []
        for num in nums:
            if num > 0:
                pos.append(num)
            else:
                neg.append(num)
        
        ans = []
        for p,n in zip(pos,neg):
            ans.append(p)
            ans.append(n)
        
        return ans
    
#now do in place
class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:
        '''
        kinda like bubble sort,
        seach for the first number that isn't the right sign
        '''
        N = len(nums)
        sign = 1
        for i in range(N):
            #not at right sign
            if nums[i]*sign < 0:
                j = i + 1
                #seach for next number with correct sign
                while j < N and nums[j]*sign < 0:
                    j += 1
                #bubble back
                while j > i:
                    nums[j], nums[j - 1] = nums[j - 1],nums[j]
                    j -= 1
            
            #flip sign
            sign *= -1
        
        return nums
            