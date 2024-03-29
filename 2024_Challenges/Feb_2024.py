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
            
##############################################
# 2971. Find Polygon With the Largest Perimeter
# 15FEB24
###############################################
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        '''
        polygon needs at least three sides, which we will always have
        given sides a1,a2,a3...ak
        and we have a1 <= a2 <= a3 ... <= ak
        and a1 + a2 + a3 + ak-1 > a_k, then were always exists a polygon with k sides
        sort the sides
        [5,5,5]
        [0,5,10,15]
        use the first side, 5, 
        pref_sum and check how far we can go?
        if i take the largest side, call it at position i, i need to ensure that the sume of th previous sides is bigger
        after soriting, pick side i, and check that sum[0 to i-1] is greater than i
        greddily start with the largest side and work down
        
        '''
        nums.sort()
        pref_sum = [nums[0]]
        for num in nums[1:]:
            pref_sum.append(pref_sum[-1] + num)
        
        
        #pointer to largst side
        ptr = len(nums) - 1
        while ptr >= 3 and pref_sum[ptr-1] <= nums[ptr]:
            ptr -= 1
            
        
        #validate
        if pref_sum[ptr-1] > nums[ptr]:
            return pref_sum[ptr]
        
        return -1
            
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        '''
        we can just sort and accumulate along the way, and stop when we can't use the enxt longest side
        the intuition is that the longest side must be greater than the sum of the other remaining sides
        if the current side is smaller than the sum of the previous sides, we can still make a polygon due to the inequality
        
        turns out the check for i >= 2 is uncessary
        when i = 0, prev__sum initally is zero and so largest wouyld still never updates
        when i = 1, prev_sum takes first element, and since the array is sorted, the first element cannot be < prev_sum when i == 1
        
        '''
        nums.sort()
        largest = -1
        prev_sum = 0
        for i in range(len(nums)):
            
            if (i >= 2 and nums[i] < prev_sum):
                largest = prev_sum + nums[i]
            
            prev_sum += nums[i]
        
        return largest
    
################################################
# 2171. Removing Minimum Number of Magic Beans
# 16FEB24
#################################################
class Solution:
    def minimumRemoval(self, beans: List[int]) -> int:
        '''
        we need the minimum number of beans to remove in order to make the bags equal, some bags can be zero
        after sorting
        [1,4,5,6]
        if we select beans[i] as the number of beans remaning in each bag, then the number of beans to be removed is just:
            sum(beans) - (len(beans) - i)*beans[i]
        
        for some j < i; the are all removed, which is just sum of beas[0:i-1]
        for some j >= i, they become A[i], contributing A[i:len(beans)]
        summing them both gives sum(beans) - (len(beans) - i)*beans[i]
        '''
        beans.sort()
        ans = float('inf')
        N = len(beans)
        SUM = sum(beans)
        
        for i in range(N):
            #choose this as the bag we want to bring everything down to
            ans = min(ans,SUM - (N - i)*beans[i] )
        
        return ans
    
#brute force inutition
class Solution:
    def minimumRemoval(self, beans: List[int]) -> int:
        '''
        pref_sum, sort beans and choose beans[i] as the one we want to bring down to
        then count beans needed for removal to the left and to the right and minimize
        '''
        beans.sort()
        N = len(beans)
        
        pref_sum = [0]
        for b in beans:
            pref_sum.append(pref_sum[-1] + b)
        
        ans = float('inf')
        
        for i in range(N):
            left = beans[:i]
            right = beans[i+1:]
            removals_right = sum(right) - beans[i]*(N-i-1) #maintin beans[i]
            removals_left = sum(left)
            ans = min(ans,removals_left + removals_right)
        
        return ans
    

# optimize with pref _sum
class Solution:
    def minimumRemoval(self, beans: List[int]) -> int:
        '''
        pref_sum, sort beans and choose beans[i] as the one we want to bring down to
        then count beans needed for removal to the left and to the right and minimize
        '''
        beans.sort()
        N = len(beans)
        
        pref_sum = [0]
        for b in beans:
            pref_sum.append(pref_sum[-1] + b)
        
        ans = float('inf')
        
      # brute force 
      #  for i in range(N):
      #      left = beans[:i]
      #      right = beans[i+1:]
      #      removals_right = sum(right) - beans[i]*(N-i-1) #maintin beans[i]
      #      removals_left = sum(left)
      #      ans = min(ans,removals_left + removals_right)
        
        for i in range(N):
            removals_right = (pref_sum[-1] - pref_sum[i+1]) - beans[i]*(N-i-1)
            removals_left = pref_sum[i]
            ans = min(ans, removals_left + removals_right)
        return ans
    
#we dont need pref sum, if we have sum of left removals, we can find the some of the right removals
class Solution:
    def minimumRemoval(self, beans: List[int]) -> int:
        '''
        pref_sum, sort beans and choose beans[i] as the one we want to bring down to
        then count beans needed for removal to the left and to the right and minimize
        '''
        beans.sort()
        N = len(beans)
        ans = float('inf')
        SUM = sum(beans)
        removals_left = 0
        
        for i in range(N):
            removals_left += beans[i]
            removals_right = (SUM - removals_left - beans[i]) - beans[i]*(N-i-1)
            ans = min(ans, removals_left + removals_right)
        return ans
    
######################################################
# 1642. Furthest Building You Can Reach (REVISTED)
# 17FEB24
#######################################################
#doesn't quite work
#71/78
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        '''
        can either use ladders or brickets to climb the buildings
        intution is to use bricks to clear the smallest gaps first, then use ladders
        ladders on the biggest jumps
        '''
        jumps = [] #need to keep track of the largest jumps, max heap
        N = len(heights)
        for i in range(N-1):
            if heights[i+1] > heights[i]:
                jumps.append(-(heights[i+1] - heights[i]))
        
        heapq.heapify(jumps)
        
        i = 0
        while i < N-1:
            #can move
            if heights[i+1] <= heights[i]:
                i += 1
            else:
                #we have a gap
                jump = heights[i+1] - heights[i]
                #if this jump is the biggest and correspons to the biggest jump, use a ladder if we have it
                if len(jumps) > 0 and -jumps[0] == jump:
                    if ladders > 0:
                        ladders -= 1
                        heapq.heappop(jumps)
                        i += 1
                    #no ladders check if we can use bricks
                    elif bricks >= -jumps[0]:
                        bricks += heapq.heappop(jumps)
                        i += 1
                    #no bricks or ladders
                    else:
                        return i
                #current gap is not the largest, then use bricks
                else:
                    if bricks >= jump:
                        #i need to clear this jump from the heap! but this would take even longer
                        bricks -= jump
                        i += 1
                    elif ladders > 0:
                        ladders -= 1
                        i += 1
                    else:
                        return i
        
        return i
    
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        '''
        ladder variant
        '''
        ladder_allocations = [] #used ladders
        N = len(heights)
        for i in range(N-1):
            #find gap
            gap = heights[i+1] - heights[i]
            if gap <= 0:
                continue
            
            #if ladders remaning
            if ladders > 0:
                heapq.heappush(ladder_allocations, gap)
                ladders -= 1
            else:
                #try relcaming ladder by using bricks
                if not ladder_allocations or gap < ladder_allocations[0]:
                    bricks -= gap
                else:
                    #swap ladderw with bricks
                    bricks -= heapq.heappop(ladder_allocations)
                    heapq.heappush(ladder_allocations, gap)
                if bricks < 0:
                    return i
        
        return N - 1
    
#brick variant
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        '''
        ladder variant
        insteaf of reclaiming a ladder, we reclaim bricks
        when we are out of bricks, we relace the largest climb with a ladder and get bricks bacj
        '''
        brick_allocations = [] #max heap
        N = len(heights)
        for i in range(N-1):
            #find gap
            gap = heights[i+1] - heights[i]
            if gap <= 0:
                continue
            #push into max heap and use bricks
            bricks -= gap
            heapq.heappush(brick_allocations, -gap)
            
            #still good to conitue
            if bricks >= 0:
                continue
            
            #we cant go forward
            if ladders <= 0:
                return i
            
            #relcaim bricks and user ladder
            bricks += -heapq.heappop(brick_allocations)
            ladders -= 1
        
        return N - 1
    
#binary search workable solution paradigm
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        '''
        try and see if we can reach some index i, if we can reach index i, then we can reach everything less than i
        for each index i, min sort the climbs and allocate bricks and ladders
        '''
        def can_reach(building_idx,heights,bricks,ladders):
            climbs = []
            for i in range(1,building_idx+1):
                if heights[i] > heights[i-1]:
                    climbs.append(heights[i] - heights[i-1])
            
            #sort
            climbs.sort()
            for c in climbs:
                if bricks >= c:
                    bricks -= c
                elif ladders > 0:
                    ladders -= 1
                else:
                    return False
            
            return True
        
        
        #binary search
        left = 0
        right = len(heights)-1
        while left < right:
            mid = left + (right - left +1) // 2 #need upper middle for this one
            if can_reach(mid,heights,bricks,ladders):
                left = mid
            else:
                right = mid - 1
        
        return left
            
################################################
# 2599. Make the Prefix Sum Non-negative
# 18FEB24
################################################
class Solution:
    def makePrefSumNonNegative(self, nums: List[int]) -> int:
        '''
        if pref sum suddenly became negative, then the numbe we just added to the rolling sum is the issue
        we can replace it with another number, but we are only allowed to move a number to the end of nums
        ending sum will always be the same
        '''
        moves = 0
        pref_sum = 0
        min_heap = []
        
        for num in nums:
            pref_sum += num
            heapq.heappush(min_heap, num)
            if pref_sum < 0:
                #undo to minimum
                moves += 1 
                pref_sum -= heapq.heappop(min_heap)
        
        return moves
    
class Solution:
    def makePrefSumNonNegative(self, nums: List[int]) -> int:
        '''
        the test cases are written always so that such a pref sum exsits
        if pre sum becomes negative, we just keep taking it back until it becomes positive again
        '''
        min_heap = []
        moves = 0
        pref_sum = 0
        
        for num in nums:
            pref_sum += num
            heapq.heappush(min_heap,num)
            while len(min_heap) > 0 and pref_sum < 0:
                moves += 1
                pref_sum -= heapq.heappop(min_heap)
        
        return moves

##############################################
# 2402. Meeting Rooms III
# 18FEB24
##############################################
class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        '''
        need to return the number of the room that held the most meetings, tie break with lowest room number
        n rooms, labeled n-1, meetings time is half closd interval, all start times are unique
        rules:
        1. each meeting will takc eplace in unused room with lowesst number
        2. if there are no available rooms, meeting will be delayed until room becomes free, meeting time duration should be the same
        3. when a room becomes unused, meetings that have an ealier original start time should be given the room
        
        sort on starts
        min heap for avialbel roomms, 
        '''
        meetings.sort(key = lambda x : x[0])
        room_counts = [0]*n
        
        rooms_available = [i for i in range(n)] #min heap
        heapq.heapify(rooms_available)
        
        end_times = [] #min heap
        
        for start,end in meetings:
            duration = end - start
            #take from used rooms and put back into available
            while end_times and end_times[0][0] <= start:
                next_end, next_room = heapq.heappop(end_times)
                heapq.heappush(rooms_available, next_room)
            #we have avialable roomrs
            if len(rooms_available) > 0:
                next_room = heapq.heappop(rooms_available)
                #move to used rooms
                heapq.heappush(end_times, [end,next_room])
            #otherwise wait for the next available room
            else:
                #in the case we can't find a room, look for the earliest available end time and update it byt adding duration to end time
                next_end,next_room = heapq.heappop(end_times)
                heapq.heappush(end_times, [next_end + duration,next_room])
            
            room_counts[next_room] += 1
        
        #print(room_counts)
        max_count = max(room_counts)
        return room_counts.index(max_count)
    
#brute force
class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        '''
        brute force would be to sort and scan rooms from smallest to largest
        allocate first smallest index room if available, if there are none available, delay maeeting until room becomes free
        in this case grab the first open room
        room_availability_time and counts
        1. if we find available room, assign meeting to that room, and upate availability tome
        2. no room available, serach for room the becomes available soongest, take and update duration
        '''
        counts = [0]*n
        earliest_available_time = [0]*n
        meetings.sort(key = lambda x : x[0])
        
        for start,end in meetings:
            duration = end - start
            min_available_room = 0
            min_available_time = float('inf')
            found_unused_room = False
            #search rooms increasninly
            for i in range(n):
                #if we can use this room take and update
                if earliest_available_time[i] <= start:
                    found_unused_room = True
                    counts[i] += 1
                    earliest_available_time[i] = end
                    break
                #need to updated the next avilable time
                if min_available_time > earliest_available_time[i]:
                    min_available_time = earliest_available_time[i]
                    min_available_room = i
            
            #if we havent found one, yes the one that will become next available
            if not found_unused_room:
                earliest_available_time[min_available_room] += duration
                counts[min_available_room] += 1
        
        return counts.index(max(counts))
                
    
##############################################
# 2946. Matrix Similarity After Cyclic Shifts
# 19FEB24
###############################################
class Solution:
    def areSimilar(self, mat: List[List[int]], k: int) -> bool:
        '''
        reduce k mod cols
        '''
        k = k % len(mat[0])
        for i,row in enumerate(mat):
            if i % 2 == 0: #left shift
                l,r = row[:k],row[k:]
                temp = r + l
            else:
                l,r = row[:len(row) - k], row[len(row) - k:]
                temp = r + l
            
            if temp != row:
                return False
        
        return True
    
#################
# 751. IP to CIDR
# 20FEB24
#################
class Solution:
    def ipToCIDR(self, ip: str, n: int) -> List[str]:
        '''
        this is a cool problem, buts its bitch
        need to return the shortest list of CIDR blocks that covers the range of IP addresses
        we need minimum intervals that cover the necessary ip addresses from start ip to ip + n
        hint: 1. convert the ip addresses to and from long integer
        want to know the most addresses we can put in this block, starting from "start" ip to n, 
        it is the smallest between the lowest bit of start and the highest bit of n, then repeat this process with new start and n
        '''
        def ip_to_bin(ip): #convert ip to len 32 bit mask
            vals = []
            for num in map(int, ip.split(".")):
                vals.append("{:08b}".format(num)) #format zeros, padding left
            
            return "".join(vals)
        
        def bin_to_ip(b):
            vals = []
            for i in range(0,32,8):
                part = str(int(b[i:i+8],2))
                vals.append(part)
            
            return ".".join(vals)
        
        
        #print(ip_to_bin(ip))
        #print(bin_to_ip(ip_to_bin(ip)))
        ans = []
        cur_ip = ip_to_bin(ip)
        while n > 0:
            last_one = 31
            while last_one> -1 and cur_ip[last_one] != '1':
                last_one -= 1
            trailing_zero = 31 - last_one
            while n - (1 << trailing_zero) < 0:
                trailing_zero -= 1
                
            cidr = "{0}/{1}".format(bin_to_ip(cur_ip), 32 - trailing_zero)
            ans.append(cidr)
            n -= 2 ** trailing_zero
            cur_ip = "{:032b}".format(int(cur_ip, 2) + (1 << trailing_zero))
        return ans
    
class Solution:
    def ipToCIDR(self, ip: str, n: int) -> List[str]:
        '''
        idea is to find the minimum cover for the n ip addresses starting from ip
        rather, we can want the the minmum number of CIDR blocks that cover ip + n
        n can't be more than 1000
        if we have an ip 255.0.0.7/x, then 2^(32-x) can be covereed
        mask is the first x bits from the front
        utility to convert ip to num and num to up
        check low bit, given position if first set bit starting frm the right
            say we have 255.0.0.8 -  11111111 00000000 00000000 00001000
            and         255.0.0.15 - 11111111 00000000 00000000 00001111
            both have bits set at position 4
        
        countdown from the given ip address
        number of the slash can't be more than 32, if /32, theres only one ip adress it can cover
        '''
        num = self.ip_to_num(ip)
        ans = []
        
        #while we have ips to cover
        while n > 0:
            #find first bit position and get number of ips in this range
            upper_bound = self.lowbit(num)
            #if we covered too many
            while upper_bound > n:
                upper_bound = upper_bound >> 1
            
            #we can cover up tp upper_bound ips, so reduce n by such
            n = n - upper_bound
            
            #get this CIDR block using upperbound IPs to cover
            cidr_block = self.num_to_ip(num) + "/" + str(32 - self.ilowbit(upper_bound))
            ans.append(cidr_block)
            #increase by the number of ips we just covered
            num = num + upper_bound
        
        return ans
                
    
    def ip_to_num(self,ip):
        nums = list(map(int, ip.split(".")))
        ans = 0
        for i,num in enumerate(nums):
            ans += (num << (8*(4-i-1)))
        
        return ans
    
    def num_to_ip(self,num):
        parts = []
        for i in range(4):
            parts.append(str((num >> (8*(4-i-1)) & 255)))
        
        return ".".join(parts)
    
    def ilowbit(self,num):
        for i in range(32):
            if num & (1 << i):
                return i
        
        return 32
    
    def lowbit(self, num):
        return 1 << self.ilowbit(num) #2^i
    
###############################################
# 2401. Longest Nice Subarray
# 21FEB24
###############################################
#had the right idea, but we cant' just & numbers, we need to set them
#note this is wrong
class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        '''
        for the largest number 10**9, we can use 30 bits to represent it
        in order for a subarray to be nice, bitwise AND for all numbers must be zero, which there cant be a one in the same 
        bit position for any two numbers
        idk why length 1 is a nice subarray
        
        
        '''
        ans = 1
        left = 0
        right = left + 1
        
        curr_bitwise_and = nums[left]
        N = len(nums)
        while right < N:
            if curr_bitwise_and & nums[right] != 0:
                left = right
                right = left + 1
                curr_bitwise_and = nums[left-1]
            else:
                while right < N and curr_bitwise_and & nums[right] == 0:
                    curr_bitwise_and = curr_bitwise_and & nums[right]
                    right += 1
                ans = max(ans, right - left + 1)
                left = right
                right = left + 1
                curr_bitwise_and = nums[left-1]
        
        return ans
    
class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        '''
        need to treat it like we're storing the bit positions of each number
        idea, collect 1 indices,
            for example, 10 = (1010), so we have (1,3)
        
        say we have temp as the current AND of all numbers, if temp & next num == 0, its ok to add the right bound
        otherwise we on take
        
        take is |
        untake is XOR, ^
        '''
        ans = 1
        left  = 0
        curr_state = nums[left]
        
        for right in range(1,len(nums)):
            #if we aren't in a nice subarray move left
            while left < right and (curr_state & nums[right]):
                #untake 
                curr_state = curr_state ^ nums[left]
                left += 1
            
            #its ok to take the number
            curr_state = curr_state | nums[right]
            ans = max(ans, right - left + 1)
        
        return ans

################################################
# 2768. Number of Black Blocks
# 22FEB24
################################################
class Solution:
    def countBlackBlocks(self, m: int, n: int, coordinates: List[List[int]]) -> List[int]:
        '''
        input is too big to traverse the whole grid
        block is defined as 2 x 2 submatrix,
            rather a block with a cell [x,y] as its top left corner contains
            [x, y], [x + 1, y], [x, y + 1], and [x + 1, y + 1]
        arr array can lony have values 0,1,2,3,4
        arr[i] is number of blocks that contains exactly one black cells
        there can be (m-2)*(n-2) blocks
        notice m and n are at least 2
        iterate on cooridantes, and see what block it is a part of
        for every cell, it could be a part of 4 blocks
        say we have cell (i,j) need to find to top lefts
        (i,j) -> (i-1,j-1), (i-1,j), (i,j-1), and (i,j)
        '''
        #this will only count if black cell belong to block
        #what about count not a back cell (white, cell) belonging to a block?
        #get total blocks and when we assign block mark as used
        #then get difference
        total_blocks = (m-1)*(n-1)
        counts = Counter()
        used_blocks = set()
        for i,j in coordinates:
            #find blocks this (x,y) could belong too
            for dx,dy in [(i-1,j-1), (i-1,j), (i,j-1),(i,j)]:
                #in bounds
                #this wont record the number of zero blocks
                if 0 <= dx < m-1 and 0 <= dy < n-1:
                    used_blocks.add((dx,dy))
                    counts[(dx,dy)] += 1
        
        ans = [0]*5
        ans[0] = total_blocks - len(used_blocks)
        
        #print(counts)
        #print(used_blocks)
        #print(ans)
        
        #count of counts
        for k,v in Counter(counts.values()).items():
            ans[k] = v
        
        return ans
    
##########################################
# 765. Couples Holding Hands
# 22FEB24
##########################################
#if pairs are good, keep them
#otherwise swap with the one we want to find
#idk why this greedy swapping works?
class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        '''
        question is given an array of len 2*n, with numbers [0 to 2*n-1]
        find minimum number of swaps to make them ordered, such that (i and i+1) are next to each other
        rather for all indices from 0 to n//2
        we want 2*i and 2*i+1 next to each other in the array
        hint1, say there are N two seat couches, for each couple draw an edge from the cout of one partner to the couch of another partner
        N = len(row)
        for i in range(N//2):
            print(2*i,2*i+1)
        [0,2,1,3]
        dist from 0 to 1 is 1
        dist from 2 to 3 is 1
        we need them to be 0
        if there are an even number of couples that are the same dist away, then number of swaps is just count(couples) // 2
        '''
        #greedy just swap wit the next value if we can't
        N = len(row)
        num_to_id = {num:i for i,num in enumerate(row)}
        swaps = 0
        
        #step on even pairs, i.e left couple of pair
        for i in range(0,N,2):
            #dont touch good pairs
            if row[i] % 2 == 0 and row[i+1] == row[i] + 1:
                continue
            if row[i+1] % 2 == 0 and row[i] == row[i+1] + 1:
                continue
            #invalid pair
            if row[i] % 2 == 0:
                num_to_move = row[i+1]
                next_pos = num_to_id[row[i] + 1] #find matching pair
            elif row[i+1] % 2 == 0:
                num_to_move = row[i]
                next_pos = num_to_id[row[i+1] + 1] #find matching pair
            elif row[i] % 2 == 1:
                num_to_move = row[i+1]
                next_pos = num_to_id[row[i] - 1] #instead of left to right, its right to left
            
            swaps += 1
            #move them
            row[next_pos] = num_to_move
            num_to_id[num_to_move] = next_pos
        
        return swaps
    
class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        '''
        if there are N couples, we treat each couple as node
        now if in positions 2*i and 2*i + 1 we have a person from couple u and a person from couple v,
        then the permutations are going to involve u and v
        so there is an edge between u and v
        the min number of swpas is just N - number of connected componenets
        
        comes from theory of permutations
            an perm can be decompsose into a composition of cyclic swaps
            if a cylic perm has k elements, we need k -1 swaps
        
        could also use dfs to find the connected compoenents
        '''
        #convert poeple to couple indicies
        #(0,1) -> 1, (2,3) -> 2
        
        N = len(row)
        for i in range(N):
            row[i] = row[i] // 2
        
        #make adj_list, for each i connect to i+1
        adj_list = defaultdict(set)
        for i in range(0,N,2):
            adj_list[row[i]].add(row[i+1])
            adj_list[row[i+1]].add(row[i])
        
        #use dfs to count up number of components
        comps = 0
        seen = set()
        
        def dfs(node,seen):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen)
        
        for i in row:
            if i not in seen:
                dfs(i,seen)
                comps += 1
        
        return N//2 - comps
    
#union find solution
class DSU:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.comps = n
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        if x_par != y_par:
            self.comps -= 1
            self.parent[x_par] = y_par

class Solution:
    def minSwapsCouples(self, row: List[int]) -> int:
        '''
        we can also do union find, when we join a group, just decrement the component size
        '''
        #convert poeple to couple indicies
        #(0,1) -> 0, (2,3) -> 1
        
        N = len(row)
        for i in range(N):
            row[i] = row[i] // 2
        
        #make adj_list, for each i connect to i+1
        adj_list = defaultdict(set)
        for i in range(0,N,2):
            adj_list[row[i]].add(row[i+1])
            adj_list[row[i+1]].add(row[i])
            
        
        UF = DSU(N//2)
        for i in range(0,N//2):
            a = row[2*i]
            b = row[2*i+1]
            UF.union(a,b)
        
        return N//2 - UF.comps
    
###################################################
# 787. Cheapest Flights Within K Stops (REVISTED)
# 23FEB24
#################################################
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        '''
        
        dp:
            states are (curr_stop,k)
        '''
        graph = defaultdict(list)
        for u,v,weight in flights:
            graph[u].append((v,weight))
            
        memo = {}
        
        def dp(curr,k):
            if curr == dst:
                return 0
            if k < 0:
                return float('inf')
            if (curr,k) in memo:
                return memo[(curr,k)]
            
            ans = float('inf')
            for neigh,weight in graph[curr]:
                ans = min(ans, weight + dp(neigh,k-1))
            
            memo[(curr,k)] = ans
            return ans
        
        
        ans = dp(src,k)
        if ans != float('inf'):
            return ans
        return -1
    
#bfs level by level
#but keep dists array from djikstra
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        '''
        bfs, but with levels limited by k
        works because in one step we get the minimum
        we just keep distances array and minimze the profits
        '''
        adj_list = defaultdict(list)
        for u,v,w in flights:
            adj_list[u].append((v,w))
        price = [float(inf)] * n
        next_level = deque([(src,0)]) # node,cost_so_far
        #need to store next_level for the current k
        for _ in range(k + 1):
            level = next_level
            next_level = deque()
            while level:                
                node,cost = level.popleft()
                for nei,w in adj_list[node]:
                    if price[nei] > cost + w:
                        price[nei] = cost + w
                        next_level.append((nei, price[nei]))

        return -1 if price[dst] == float(inf) else price[dst]
    
#bfs, store level in entries and
#keep dist array
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        '''
        bfs, but with levels limited by k
        works because in one step we get the minimum
        we just keep distances array and minimze the profits
        '''
        adj_list = defaultdict(list)
        for u,v,w in flights:
            adj_list[u].append((v,w))
        price = [float(inf)] * n
        q = deque([(src,0,k)]) # (node,cost_so_far,stops)
        while q:                
            node,cost,stops = q.popleft()
            for nei,w in adj_list[node]:
                if price[nei] > cost + w:
                    price[nei] = cost + w
                    #only add if we have enough stops
                    if stops > 0:
                        q.append((nei, price[nei],stops-1))
        return -1 if price[dst] == float(inf) else price[dst]
                

#greedy usingg djikstras, need to first entry to be (cost_so_far)
#and instead of prices, we keep track of stops
class Solution:
    def findCheapestPrice(self, N: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        adj_list = defaultdict(list)
        for u,v,w in flights: adj_list[u].append((v,w))
        heap = [(0, 0, src)]
        stops = [float(inf)] * N
        while heap:            
            cost, cur_stops, node = heapq.heappop(heap)
            if node == dst:
                return cost
            if stops[node] > cur_stops and cur_stops <= k:
                stops[node] = cur_stops
                for nei,w in adj_list[node]:
                    heapq.heappush(heap, (cost + w, cur_stops + 1, nei))        
        return -1

##################################################
# 2093. Minimum Cost to Reach City With Discounts
# 23FEB24
##################################################
class Solution:
    def minimumCost(self, n: int, highways: List[List[int]], discounts: int) -> int:
        '''
        the problem is that the graph is undirected
        treat this like dp
        djikstras, make sure to keep track of discounts used
        need to used djikstras with two states insteaf of one
        '''
        pq = [(0, 0, discounts)]
        graph = collections.defaultdict(list)
        visited = dict()
        cost = dict()

        for a, b, dist in highways:
            graph[a].append(b)
            graph[b].append(a)
            cost[(a, b)] = dist
            cost[(b, a)] = dist

        while pq:
            curr_cost, node, curr_disc = heapq.heappop(pq)

            # Because of how djikstra works, when we reach this node the first time,
            # we will reach there with the lowest cost.  However, we may reach this node
            # again with a highest cost, but more discount tickets, which can lead to a 
            # more optimal soln at the end.  If we ever come back to this node with the same or 
            # fewer discounts, the soln is not optimal.
            if node in visited and curr_disc <= visited[node]: #only update if we can use more discounts, not less
                continue
            visited[node] = curr_disc

            if node == n - 1:
                return curr_cost
            for neigh in graph[node]:
                if curr_disc > 0:
                    heapq.heappush(pq, (cost[(node, neigh)] // 2 + curr_cost, neigh, curr_disc - 1))
                heapq.heappush(pq, (cost[(node, neigh)] + curr_cost, neigh, curr_disc))
        # no soln
        return -1

#djikstras butt used visited set
class Solution:
    def minimumCost(self, n: int, highways: List[List[int]], discounts: int) -> int:
        '''
        the problem is that the graph is undirected
        treat this like dp
        djikstras, make sure to keep track of discounts used
        need to used djikstras with two states insteaf of one
        we are LOOKING for the cheapest costs at diffferent states
            states are (distcounts and cuty)
        '''
        pq = [(0, 0, discounts)]
        graph = collections.defaultdict(list)
        visited = set()

        for a, b, cost in highways:
            graph[a].append((b,cost))
            graph[b].append((a,cost))
        
        #this is just pruning (can do with BFS too)
        while pq:
            curr_cost, curr_city, curr_discounts = heapq.heappop(pq)
            #if weve seen this hear, we can skip, since its already minimum
            if (curr_city,curr_discounts) in visited:
                continue
            
            #found destination
            if curr_city == n-1:
                return curr_cost
            
            visited.add((curr_city,curr_discounts))
            
            for neigh,toll in graph[curr_city]:
                #no discount
                if (neigh,curr_discounts) not in visited:
                    heapq.heappush(pq,(curr_cost + toll, neigh,curr_discounts))
                
                #disctounts to be applied we haven't seen this state
                if curr_discounts > 0 and (neigh,curr_discounts -1) not in visited:
                    heapq.heappush(pq, (curr_cost + toll // 2,neigh,curr_discounts-1))
        
        return -1
            
    
#######################################################
# 2092. Find All People With Secret 
# 24FEB24
#######################################################
#close but no cigar
class DSU:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        #i dont think we need size array here
        self.size = [1 for i in range(n)]
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        if x_par == y_par:
            return
        
        #swap groups
        if self.size[x_par] >= self.size[y_par]:
            self.parent[y_par] = x_par
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
        else:
            self.parent[x_par] = y_par
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
            
#bfs with djikstras dist array
class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        '''
        notes, a person can attend multiple meetings at the same time, meetings are transitive in nature
        person 0 tells firstPerson
        meetings can also be disjoint
        in order for a person to know a secret, the previous person they met with must know the secret at or before the time they meet
        say we have [0, 1, 3], [0, 2, 5], [0, 3, 6]
        then we have [1,4,2], [1,9,4], [1,0,3], note last has been processed already
        only 9 knows the secret since, 1 and 4 met a time 2, but 1 didn't know the secret until time 3
        start with 0 and firstPerson, since they know the secret at time 0
        process people whome they meet after the time at which they learned the secret
        important
        we may come to process a node again IF we can reach it at an earlier time -> Djikstras
        its really just multi source BFS
        
        BFS with distances array, but we we time as the contraint we wish to minimize
        if we can reach this person at an earlier time we update,
        then we just check for each person, if the earliest time != float('inf')
        '''
        #make graph
        graph = defaultdict(list)
        for u,v,time in meetings:
            graph[u].append((v,time))
            graph[v].append((u,time))
            
        earliest_times = [float('inf')]*n #earliest times at which a person learned the secret
        #mark earliest
        earliest_times[0] = 0
        earliest_times[firstPerson] = 0
        
        q = deque([])
        q.append((0,0))
        q.append((firstPerson,0)) #entries start (person, and earliest time they knew the secret)
        
        while q:
            curr_person, learned_secret_time = q.popleft()
            for neigh,next_time in graph[curr_person]:
                #the time for this neigh must be after the time curr person leanerd the secret
                if next_time >= learned_secret_time:
                    #if we can make an improvment
                    if earliest_times[neigh] > next_time:
                        earliest_times[neigh] = next_time
                        q.append((neigh, next_time ))
                        
        ans = []
        for i in range(n):
            if earliest_times[i] != float('inf'):
                ans.append(i)
        
        return ans
    
#note, its not always distances, sometimes it can be an improvment on time
#rather, an improvment on an dist/time hueristic
#we are allowed to revisit a state if its an improvment
    
#we can do dfs instead of bfs
#we just do dfs for 0 and first person
class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        '''
        dfs variant
        '''
        #make graph
        graph = defaultdict(list)
        for u,v,time in meetings:
            graph[u].append((v,time))
            graph[v].append((u,time))
            
        earliest_times = [float('inf')]*n #earliest times at which a person learned the secret
        #mark earliest
        earliest_times[0] = 0
        earliest_times[firstPerson] = 0
        
        
        def dfs(curr_person,learned_secret_time):
            for neigh,next_time in graph[curr_person]:
                #the time for this neigh must be after the time curr person leanerd the secret
                if next_time >= learned_secret_time:
                    #if we can make an improvment
                    if earliest_times[neigh] > next_time:
                        earliest_times[neigh] = next_time
                        dfs(neigh, next_time)
                        
        dfs(0,0)
        dfs(firstPerson,0)
                        
        ans = []
        for i in range(n):
            if earliest_times[i] != float('inf'):
                ans.append(i)
        
        return ans
    
class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        '''
        for djikstras we need to process them in order of learned times first
        because of djikstras, the time we meet them, will be the earliest time they learned the secret
        bfs we are allowed to revisit if we can make an improvement
        djikstra's we visit only once, and is guarnteeed to be the minimum
        so we need to djikstras on earliest time a person learned the secret
        '''
        #make graph
        graph = defaultdict(list)
        for u,v,time in meetings:
            graph[u].append((v,time))
            graph[v].append((u,time))
            
        
        seen = [False]*n
        pq = []
        #entries are (time and person)
        heapq.heappush(pq, (0,0))
        heapq.heappush(pq, (0,firstPerson))
        
        while pq:
            learned_secret_time, curr_person = heapq.heappop(pq)
            #state is guaranteed to be minimum
            if seen[curr_person]:
                continue
            
            seen[curr_person] = True
            for neigh,neigh_time in graph[curr_person]:
                #if an improvment and times comes after
                if not seen[neigh] and neigh_time >= learned_secret_time:
                    heapq.heappush(pq, (neigh_time, neigh))
        
        ans = []
        for i in range(n):
            if seen[i]:
                ans.append(i)
        
        return ans

#uf solution
class DSU:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.size = [1]*n
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        if x_par == y_par:
            return
        #updates
        if self.size[x_par] >= self.size[y_par]:
            self.parent[y_par] = x_par
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 1
        else:
            self.parent[x_par] = y_par
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 1
            
    def disunite(self,x):
        self.parent[x] = x
        self.size[x] = 1
        
    def connected(self,x,y):
        return self.find(x) == self.find(y)
    
class Solution:
    def findAllPeople(self, n: int, meetings: List[List[int]], firstPerson: int) -> List[int]:
        '''
        for union find, we need to sort the meetings by time, and group meetings by time
        then for each meeting we try to unite them
        if the meetins were connected to zero, then they know the secret, 
        otherwise we can disunite them
            this is a cool addon to DSU
        why? do we need to check to zero after uniting
        because they have to know the secret to propogate it to others
        so for after uniting all the meetings for time t, we check for zero
            disunite those that aren't pointing to zero
        keeping a flag array won't work either
            why? if a person is connected to 0, then he/she knows the secret
        
        inuition 1
            suppose that one participant of a transitive meeting gets to know secret after time t, if they come to
            know the secret at anyt time after t, it will not affect meetings happening at time t
        intuition 2
            if none of the (x,y) people knew the secret ebfore or at time t, and assume one of them
            get to konw the secret after time t, it would not matter becasue they didnt know the secret at time t anyway!
        
        so we disunite if they didn't konw the meeting
        if we left them united, then when we check for 0, it would incorreclt imply that (x,y) new a secret then in fact they did not
        disunite prevent them from being incorrectly propogated
        for diunite, we just have them point to themselves
        '''
        meetings_mapp = defaultdict(list)
        for x,y,t in meetings:
            meetings_mapp[t].append((x,y))
        
        uf = DSU(n)
        #0 and first
        uf.union(0,firstPerson)
        
        for t in sorted(meetings_mapp):
            for x,y in meetings_mapp[t]:
                uf.union(x,y)
            
            #disunite all those not connected to zero
            for x,y in meetings_mapp[t]:
                #we just joined x and y, so check that either of them are not pointing to zero
                if not uf.connected(x,0):
                    #disunite
                    uf.disunite(x)
                    uf.disunite(y)
        
        #check all those point to 0
        return [i for i in range(n) if uf.connected(i,0)]
        
#################################################
# 2709. Greatest Common Divisor Traversal
# 25FEB24
##################################################
#need back edge
#TLE
class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        '''
        we can only traverse indices if: given index i and j and != j
            gcd(nums[i], gcd(nums[j])) > 1
        
        need to determine if every pair of indices i and j, where i < j, there exsists a traversal
        input is too big to try all pairs
        create prime factors list for all indices
        add edge between the neighbors of prime factors
        then just check all indicies are connected
        '''
        if len(nums) == 1:
            return True


        def getPrimeFactors(num):
            factors = set()
            curr_factor = 2
            while num > 1:
                while num % curr_factor == 0:
                    num = num // curr_factor
                    factors.add(curr_factor)
                curr_factor += 1
            
            return factors
        
        graph = defaultdict(set)
        
        for num in set(nums):
            neighs = getPrimeFactors(num)
            for neigh in neighs:
                graph[num].add(neigh)
                graph[neigh].add(num)
                #need back edge
                
        if not graph:
            return False
        
        def dfs(node,seen,graph):
            seen.add(node)
            for neigh in graph[node]:
                if neigh not in seen:
                    dfs(neigh,seen,graph)
                    
        seen = set()
        comps = 0
        for num in nums:
            if num not in seen:
                dfs(num,seen,graph)
                comps += 1
        
        return comps == 1
    
#need to use visited index and visited prime
class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        '''
        we can only traverse indices if: given index i and j and != j
            gcd(nums[i], gcd(nums[j])) > 1
        
        need to determine if every pair of indices i and j, where i < j, there exsists a traversal
        input is too big to try all pairs
        create prime factors list for all indices
        add edge between the neighbors of prime factors
        then just check all indicies are connected
        
        we need to map each primefactor to the indices
        and we need to map index to primefactor
        
        then we just do dfs and check that there is only one component
        '''
        #edge cases
        if len(nums) == 1:
            return True
        if 1 in set(nums):
            return False

        prime_to_index = defaultdict(list)
        index_to_prime = defaultdict(list)
        
        for i, num in enumerate(nums):
            curr_num = num
            #get all prime factors for num
            for j in range(2,int(num**0.5) + 1):
                if curr_num % j == 0:
                    prime_to_index[j].append(i)
                    index_to_prime[i].append(j)
                    while curr_num % j == 0:
                        curr_num = curr_num // j
            #final factor
            if curr_num > 1:
                prime_to_index[curr_num].append(i)
                index_to_prime[i].append(curr_num)
        visited_prime = set()
        visited_index = set()
        comps = 0
        for i in range(len(nums)):
            if i not in visited_index:
                self.dfs(i,visited_index,visited_prime,prime_to_index,index_to_prime)
                comps += 1
        
        return comps == 1
       
    #need to seen sets, for index and prime
    def dfs(self, index, visited_index, visited_prime, prime_to_index, index_to_prime):
        visited_index.add(index)
        for next_prime in index_to_prime[index]:
            if next_prime in visited_prime:
                continue
            visited_prime.add(next_prime)
            for neigh_index in prime_to_index[next_prime]:
                if neigh_index not in visited_index:
                    self.dfs(neigh_index, visited_index,visited_prime,prime_to_index,index_to_prime)

#union find solution
class DSU:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.size = [1]*n
        self.comps = n
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        if x_par == y_par:
            return
        #updates
        if self.size[x_par] >= self.size[y_par]:
            self.parent[y_par] = x_par
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 1
        else:
            self.parent[x_par] = y_par
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 1
        self.comps -= 1
    
class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        '''
        union find solution
        we can go from index i to index j, if there is an edge, an edge being gcd(nums[i], nums[j]) > 1
        edges are undirected, gcd is transitive, so all nodes must be connected,
        the problem is that we can't check all (i,j) pairs, its justt too many, imagine we have array with all evenes
        every node is connected - complete graph
        
        note, using union find after checking all pairs to for gcd > 1 would work, but we could have too many pairs
        we need to precompute all primes first! inteaf of checking for all prime factors for each number
        we need to compute prime factors with seive,
        https://leetcode.com/problems/greatest-common-divisor-traversal/discuss/4778362/Detailed-Intuition-and-Explanation-(Python-Solution)
        consider going throual all primes, there are only about 1000, but we would need to check all primes are divisible 
        for each number n, which is 1000*10**5 which is still to big
        thankfully we can decompose the largest number in only 5 primes
            2*3*5*7*11 > 10**5
        basically we need access to all primes for each number up to n, we can get this using seive
        '''
        n = len(nums)
        largest = max(nums)
        prime_divisors = [[] for _ in range(largest+1)]
        for p in range(2,largest+1):
            if len(prime_divisors[p]) == 0:
                i = p
                while i <= largest:
                    prime_divisors[i].append(p)
                    i += p
        
        uf = DSU(n)
        multiple_idx = {} #need to mapp a prive divisor to an index
        for i,num in enumerate(nums):
            for p_div in prime_divisors[num]:
                multiple_idx[p_div] = multiple_idx.get(p_div,i)
                uf.union(i,multiple_idx[p_div])
        
#######################
# 1245. Tree Diameter
# 26FEB24
########################
#the bfs passes
class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        '''
        this is just longest path in undirect graph,
        but we have a tree, the root of three has zero indegree, so we can use dp
        if this were longest path in undirected graph, thats NP hard
        hints
            1. start at any node, and find furthest node from it
            2. then found furtherest node from this
            3. dimeter is dis between the two nodes
        '''
        if not edges:
            return 0
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def bfs(start,graph):
            N = len(graph)
            dist = [float('-inf')]*N
            dist[start] = 0
            q = deque([start])
            seen = set()
            
            while q:
                curr = q.popleft()
                seen.add(curr)
                for neigh in graph[curr]:
                    if neigh not in seen:
                        if dist[neigh] < dist[curr] + 1:
                            dist[neigh] = dist[curr] + 1
                            q.append(neigh)
            return dist
        
        dist = bfs(0,graph)
        #find furthest node, call it a
        B = dist.index(max(dist))
        #bfs
        dist = bfs(B,graph)
        C = dist.index(max(dist))
        #one more time
        dist = bfs(B,graph)
        return dist[C]

class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        '''
        we dont need to keep the distances array in the BFS approach
        we also can do it with two passes of BFS
        
        for bfs, the last node we see is the extreme distance
        so we walk as far a possible to this node, then walk as far as possible again
        '''
        if not edges:
            return 0
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        def bfs(start,graph):
            N = len(graph)
            q = deque([(start,0)])
            seen = set()
            dist = -1
            last_node = start
            
            while q:
                curr,curr_dist = q.popleft()
                dist = max(dist,curr_dist)
                seen.add(curr)
                for neigh in graph[curr]:
                    if neigh not in seen:
                        q.append((neigh,curr_dist + 1))
                        last_node = neigh
            return last_node,dist
        
        a,dist1 = bfs(0,graph)
        b,dist2 = bfs(a,graph)
        return dist2
    
class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        '''
        this is similart to the diameter of N-ary tree
        we just need to find the the nodes with the two longest diamterers and add them
        also since the graph is not specifically a tree, we just make sure we dont go back to a parent in each call
        '''
        if not edges:
            return 0
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        ans = [0]
        
        def dp(node,parent):
            first_longest, second_longest = 0,0
            for neigh in graph[node]:
                if neigh == parent:
                    continue
                    
                child_ans = dp(neigh,node)
                if first_longest < second_longest:
                    first_longest = max(child_ans,first_longest)
                else:
                    second_longest = max(child_ans,second_longest)
                
            
            ans[0] = max(ans[0], first_longest + second_longest)
            return max(first_longest,second_longest) + 1
        
        dp(0,None)
        return ans[0]
    
#################################################
# 2689. Extract Kth Character From The Rope Tree
# 27FEB24
#################################################
# Definition for a rope tree node.
# class RopeTreeNode(object):
#     def __init__(self, len=0, val="", left=None, right=None):
#         self.len = len
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getKthCharacter(self, root: Optional[object], k: int) -> str:
        """
        :type root: Optional[RopeTreeNode]
        """
        '''
        use recursino to concat the string, then return the kth char
        '''
        def rec(node):
            if not node:
                return ""
            if node.len == 0:
                return node.val
            left = rec(node.left)
            right = rec(node.right)
            return left + right
        
        return rec(root)[k-1]
            
#top down, divide and conquer
# Definition for a rope tree node.
# class RopeTreeNode(object):
#     def __init__(self, len=0, val="", left=None, right=None):
#         self.len = len
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getKthCharacter(self, root: Optional[object], k: int) -> str:
        """
        :type root: Optional[RopeTreeNode]
        """
        '''
        another way would be to pass in k and check
        if the length of the left node is >= k we can go left
        '''
        def rec(node,k):
            if node.len > 0:
                left_length = 0
                if node.left:
                    left_length = max(node.left.len, len(node.left.val))
                
                if left_length >= k:
                    return rec(node.left,k)
                else:
                    return rec(node.right, k - left_length)
            
            return node.val[k-1]
        
        return rec(root,k)
                
# Definition for a rope tree node.
# class RopeTreeNode(object):
#     def __init__(self, len=0, val="", left=None, right=None):
#         self.len = len
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getKthCharacter(self, root: Optional[object], k: int) -> str:
        """
        :type root: Optional[RopeTreeNode]
        """
        '''
        another way would be to pass in k and check
        if the length of the left node is >= k we can go left
        '''
        def rec(node,k):
            #if leaf, we can get this char
            if node.len == 0:
                return node.val[k-1]
            
            #need to check if we can go left
            else:
                left_length = 0
                if node.left:
                    left_length = max(node.left.len, len(node.left.val))
                if left_length >= k:
                    return rec(node.left,k)
                else:
                    return rec(node.right, k - left_length)
        
        return rec(root,k)
            
#######################################
# 1740. Find Distance in a Binary Tree
# 28FEB24
#######################################
#bfs with dist array
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDistance(self, root: Optional[TreeNode], p: int, q: int) -> int:
        '''
        easy way is to convert tree to graph and then do bfs from p and check dist q
        '''
        graph = defaultdict(list)
        
        def dfs(node,parent,graph):
            if not node:
                return
            if parent:
                graph[parent.val].append(node.val)
                graph[node.val].append(parent.val)
            
            dfs(node.left,node,graph)
            dfs(node.right,node,graph)
        
        dfs(root,None,graph)
        dist = defaultdict(lambda: float('inf'))
        #start with p
        dist[p] = 0
        queue = deque([p])
        
        while queue:
            curr = queue.popleft()
            for neigh in graph[curr]:
                if dist[neigh] > dist[curr] + 1:
                    dist[neigh] = dist[curr] + 1
                    queue.append(neigh)
        
        return dist[q]
    
#dfs using lca
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDistance(self, root: Optional[TreeNode], p: int, q: int) -> int:
        '''
        trick is to find the LCA of p and q
        then get dists from p to lca and q to lca
        the ans is the sum
        '''
        def lca(node,p,q):
            if not node or node.val == p or node.val == q:
                return node
            left = lca(node.left,p,q)
            right = lca(node.right,p,q)
            if left and right:
                return node
            else:
                return left or right #can do (None or 5)
        
        #dist function
        def dist(node,target):
            if not node:
                return float('inf')
            if node.val == target:
                return 0
            
            left = dist(node.left,target)
            right = dist(node.right,target)
            return 1 + min(left,right)
        
        LCA = lca(root,p,q)
        return dist(LCA,p) + dist(LCA,q)
        
#####################
# 1609. Even Odd Tree
# 29FEB24
#####################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        '''
        bfs and check level requirments, either on the fly or after the fact
        even indexed level : odd values strictly increasing
        odd indes level : even values strictly decreasing
        the problem is that i need to validate a row at a time
        '''
        def validate(q,row):
            N = len(q)
            #even
            if row % 2 == 0:
                if N == 1:
                    if q[0].val % 2 == 0:
                        return False
                else:
                    for i in range(N-1):
                        if q[i+1].val % 2 == 0:
                            return False
                        elif q[i+1].val - q[i].val <= 0:
                            return False
            #odd
            else:
                if N == 1:
                    if q[0].val % 2 == 1:
                        return False
                else:
                    for i in range(N-1):
                        if q[i+1].val % 2 == 1:
                            return False
                        elif q[i+1].val - q[i].val >= 0:
                            return False
            return True
            
        curr_level = 0
        q = deque([root])
        
        while q:
            N = len(q)
            if not validate(q,curr_level):
                #row = []
                #for t in q:
                #    row.append(t.val)
                #print(row)
                
                return False
            for i in range(N):
                curr = q.popleft()
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
            
            curr_level += 1
        
        return True

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isEvenOddTree(self, root: Optional[TreeNode]) -> bool:
        '''
        dfs, 
        but store previous level last nodes in temp array
        and we check that:
            nodes on even levels must have odd values
            nodes on odd levels must have even value, so parities must be different
        
        for increasing odd condition we check
            node.val <= prev[level]
        
        for decreasing even condition we check
            node.val >= prev[level]
        
        empty nodes are triviallly odd/even
        '''
        prev_values = [] #store last values by level
        def dfs(node,curr_level):
            if not node:
                return True
            #if same parity, return falsse
            if node.val % 2 == curr_level % 2:
                return False
            #prepare future levels to store last values
            while len(prev_values) <= curr_level:
                prev_values.append(-1)
            
            #if we have a previous value, we need to validate
            if prev_values[curr_level] != -1:
                if curr_level % 2 == 0 and node.val <= prev_values[curr_level]:
                    return False
                if curr_level % 2 == 1 and node.val >= prev_values[curr_level]:
                    return False
            
            #set new prev_value
            prev_values[curr_level] = node.val
            
            return dfs(node.left,curr_level + 1) and dfs(node.right, curr_level + 1)
        
        return dfs(root,0)
                