################################################
# 3568. Minimum Moves to Clean the Classroom
# 02SEP26
################################################
class Solution:
    def minMoves(self, classroom: List[str], energy: int) -> int:
        '''
        notice there are most 10 L cell in thr grid
        bfs with states (x,y,mask,e,steps)
             initializing with (sx, sy, 0, energy, 0), and for each move update e (–1 per step), update mask on 'L', reset e=energy on 'R', and return steps when mask == fullMask.
        maintain 3d array: bestEnergy[x][y][mask] and skip any new state with e <= bestEnergy[x][y][mask]
            since a smaller enery wouldn't allow for exploraton
        '''
        rows, cols = len(classroom), len(classroom[0])

        sx, sy = -1, -1
        litter = []

        for i in range(rows):
            for j in range(cols):
                if classroom[i][j] == "S":
                    sx, sy = i, j
                elif classroom[i][j] == "L":
                    litter.append((i, j))

        litter = {coord: i for i, coord in enumerate(litter)}

        #full_mask = (1 << len(litter)) - 1

        # (row, col, mask) -> maximum energy we've reached this state with
        best_energy = defaultdict(lambda: float('-inf'))

        starting_state = (sx, sy, set(), energy, 0)
        q = deque([starting_state])

        #init first state
        best_energy[(sx, sy, 0)] = energy

        while q:
            x, y, curr_mask, curr_energy, steps = q.popleft()

            if len(curr_mask) == len(litter):
                return steps

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < rows and 0 <= ny < cols):
                    continue

                if classroom[nx][ny] == "X":
                    continue

                # Moving costs one energy
                neigh_energy = curr_energy - 1

                if neigh_energy < 0:
                    continue

                neigh_mask = curr_mask.copy()

                # Recharge
                if classroom[nx][ny] == "R":
                    neigh_energy = energy

                # Collect litter
                elif classroom[nx][ny] == "L":
                    neigh_mask.add((nx,ny))

                state_key = (nx, ny, frozenset(neigh_mask))

                # Only explore if we arrive with more energy
                if neigh_energy > best_energy[state_key]:
                    best_energy[state_key] = neigh_energy
                    q.append((nx, ny, neigh_mask, neigh_energy, steps + 1))

        return -1
    
#########################################
# 3875. Construct Uniform Parity Array I
# 03SEP26
##########################################
class Solution:
    def uniformArray(self, nums1: list[int]) -> bool:
        '''
        i can either do
        nums2[i] = nums[i]
        nums2[i] = nums1[i] - nums1[j] for j != i

        forr each index i in nums1, see possile transofmrations we can get
        we can always make it even - even = even
        odd - even = odd
        odd - odd = even
        we will always be able to do it
        '''
        return True
    
############################################
# 3876. Construct Uniform Parity Array II
# 03SEP26
#############################################
#closeee :(
class Solution:
    def uniformArray(self, nums1: list[int]) -> bool:
        '''
        similat to the first problem, but now the numbers have to be positive
        not negeative
        try setting parity to all even, or all odd
        '''
        def can_odd(arr):
            # Already all odd
            if all(num % 2 == 1 for num in arr):
                return True

            # Get the actual odd numbers
            odds = [num for num in arr if num % 2 == 1]

            # No odd number available to subtract
            if len(odds) == 0:
                return False

            smallest_odd = min(odds)

            # Every even number must be able to subtract an odd
            # and remain positive
            for num in arr:
                if num % 2 == 0:
                    if num - smallest_odd <= 0:
                        return False

            return True

        def can_even(arr):
            # Already all even
            if all(num % 2 == 0 for num in arr):
                return True

            # Get the actual even numbers
            evens = [num for num in arr if num % 2 == 0]

            # No even number available to subtract
            if len(evens) == 0:
                return False

            smallest_even = min(evens)

            # Every odd number must be able to subtract an even
            # and remain positive
            for num in arr:
                if num % 2 == 1:
                    if num - smallest_even <= 0:
                        return False

            return True

        return can_odd(nums1) or can_even(nums1)
    
class Solution:
    def uniformArray(self, nums1: list[int]) -> bool:
        '''
        similat to the first problem, but now the numbers have to be positive
        not negeative
        try setting parity to all even, or all odd
        even - even = even
        even - odd  = odd
        odd  - even = odd
        odd  - odd  = even
        '''
        def can_odd(arr):
            if all(num % 2 == 1 for num in arr):
                return True

            odds = [num for num in arr if num % 2 == 1]

            if not odds:
                return False

            smallest_odd = min(odds)

            return all(num - smallest_odd > 0 for num in arr if num % 2 == 0)

        def can_even(arr):
            if all(num % 2 == 0 for num in arr):
                return True

            odds = [num for num in arr if num % 2 == 1]

            if not odds:
                return False

            # Need an odd number to turn odd numbers into even:
            # odd - odd = even
            #
            # But because we can't use the same index, this requires
            # more careful reasoning about the two smallest odds.

            return False

        return can_odd(nums1) or can_even(nums1)
    
#################################################
# 3903. Smallest Stable Index I
# 04SEP26
##################################################
class Solution:
    def firstStableIndex(self, nums: list[int], k: int) -> int:
        '''
        need leftmost stable index
        brute force first
        '''
        n = len(nums)
        for i in range(n):
            left,right = nums[:i+1],nums[i:]
            instability = max(left) - min(right)
            if instability <= k:
                return i
        return -1

class Solution:
    def firstStableIndex(self, nums: list[int], k: int) -> int:
        '''
        pref max and suff min
        '''
        n = len(nums)
        pref_max = nums[:]
        for i in range(1,n):
            pref_max[i] = max(pref_max[i-1],nums[i])
        
        suff_min = nums[:]
        for i in range(n-2,-1,-1):
            suff_min[i] = min(suff_min[i+1],nums[i])
        
        for i in range(n):
            instability = pref_max[i] - suff_min[i]
            if instability <= k:
                return i
        return -1
    
###############################################
# 3904. Smallest Stable Index II
# 04SEP26
################################################
class Solution:
    def firstStableIndex(self, nums: list[int], k: int) -> int:
        n = len(nums)
        pref_max = nums[:]
        for i in range(1,n):
            pref_max[i] = max(pref_max[i-1],nums[i])
        
        suff_min = nums[:]
        for i in range(n-2,-1,-1):
            suff_min[i] = min(suff_min[i+1],nums[i])
        
        for i in range(n):
            instability = pref_max[i] - suff_min[i]
            if instability <= k:
                return i
        return -1
    
##################################################
# 115. Distinct Subsequences
# 07SEP26
##################################################
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        '''
        dp on i and j
        if j gets to the end we've found a way
        options
        if s[i] == t[j]
            1 + dp(i+j,j+1)

        '''
        memo = {}

        def dp(i, j):
            if j == len(t):
                return 1
            if i == len(s):
                return 0

            if (i, j) in memo:
                return memo[(i, j)]

            if s[i] == t[j]:
                ways = dp(i + 1, j + 1) + dp(i + 1, j)
            else:
                ways = dp(i + 1, j)

            memo[(i, j)] = ways
            return ways

        return dp(0, 0)
    
################################################
# 3870. Count Commas in Range
# 08SEP26
#################################################
class Solution:
    def countCommas(self, n: int) -> int:
        '''
        every three positions need a comma
        '''
        def count_commas(n):
            if n < 1000:
                return 0

            return (len(str(n)) - 1) // 3
        
        ans = 0
        for i in range(1,n+1):
            ans += count_commas(i)
        
        return ans
    
class Solution:
    def countCommas(self, n: int) -> int:
        '''
        n is only between 1 and 10000
        '''
        ans = 0
        for i in range(1,n+1):
            if i > 999:
                ans += 1
        
        return ans
    
#################################################
# 3871. Count Commas in Range II
# 10SEP24
##################################################
class Solution:
    def countCommas(self, n: int) -> int:
        '''
        for 1-3 digits, 0 commas, 1 - 999
        for 4-6 digits, 1 commas, 1000 - 999,999
        for 7-9 digits, 2 commas 
        for 10-12 digits, 3 commas
        forr 13-15 digits, 4 commas
        i can slide this interval and advance it, the count the numbers in aht range by doing right - left + 1
        '''
        ans = 0
        commas = 0
        left = 1_000

        while left <= n:
            #find next right before checking n
            right = left * 1000 - 1
            #in between
            count = min(n, right) - left + 1
            print(count)
            ans += count * (commas + 1)

            left *= 1000
            commas += 1

        return ans

################################################
# 3483. Unique 3-Digit Even Numbers
# 11SEP26
#################################################
class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        '''
        i can use recursion to build them
        '''
        ans = set()
        n = len(digits)
        def rec(path,used):
            if len(path) == 3:
                num = "".join(path)
                if int(num) % 2 == 0 and num[0] != "0":
                    ans.add(num)
                return
            
            for i in range(n):
                if i in used:
                    continue
                used.add(i)
                path.append(str(digits[i]))
                rec(path,used)
                path.pop()
                used.remove(i)
        rec([],set())

        return len(ans)

class Solution:
    def totalNumbers(self, digits: List[int]) -> int:
        '''
        just try all indicex (i,j,k) you dum dum
        '''
        n = len(digits)
        seen = set()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if digits[i] == 0 or digits[k] % 2 == 1:
                        continue
                    num = digits[i]*100 + digits[j]*10 + digits[k]
                    seen.add(num)
        
        return len(seen)

#####################################################
# 3414. Maximum Score of Non-overlapping Intervals
# 11SEP26
####################################################
class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        '''
        dp, but we need to return the indices rather than finding the max score
        and the indice must be lexographically smallest
        '''

        # Keep the original index
        events = [(start, end, value, i) for i, (start, end, value) in enumerate(intervals)]

        events.sort()
        n = len(events)
        memo = {}

        def dp(i, k):
            if i == n or k == 0:
                return 0, ()

            if (i, k) in memo:
                return memo[(i, k)]

            # Option 1: skip
            no_take_value, no_take_indices = dp(i + 1, k)

            # Option 2: take
            curr_end = events[i][1]
            curr_value = events[i][2]
            curr_index = events[i][3]

            next_index = self.binary_search(events, curr_end)

            take_value, take_indices = dp(next_index, k - 1)
            take_value += curr_value

            # Add current original index and keep indices sorted
            take_indices = tuple(sorted((curr_index,) + take_indices))

            # Pick the better solution
            if take_value > no_take_value:
                ans = (take_value, take_indices)
            elif take_value < no_take_value:
                ans = (no_take_value, no_take_indices)
            else:
                # Same value -> lexicographically smaller indices
                ans = min((take_value, take_indices),(no_take_value, no_take_indices))

            memo[(i, k)] = ans
            return ans


        return dp(0, 4)[1]

    def binary_search(self, arr, target_end):
        """
        Find the first index where event's start time > target_end
        """
        left, right = 0, len(arr) - 1
        ans = len(arr)

        while left <= right:
            mid = (left + right) // 2

            if arr[mid][0] > target_end:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans

###################################################################
# 2472. Maximum Number of Non-overlapping Palindrome Substrings
# 15SEP26
###########################################################
#need to precompute whetehr s[i:j] is palindrom
class Solution:
    def maxPalindromes(self, s: str, k: int) -> int:
        '''
        it needs to be at least k
        if we pick a substring of lenth l, which is >= k, that works, but can we also pick a longer one longer than k?
            picking a longer gone reduces the chance of finding another one
            so if we don't have at least k, keep looking
        dp i, then check all k lengths from there
        '''
        memo = {}
        n = len(s)

        def dp(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            #check all >= k length palindromes
            ans = dp(i+1)
            for j in range(i + k, n + 1):
                substring = s[i:j]
                if substring == substring[::-1]:
                    ans = max(ans, 1 + dp(j))
            memo[i] = ans
            return ans
        
        return dp(0)

class Solution:
    def maxPalindromes(self, s: str, k: int) -> int:
        '''
        it needs to be at least k
        if we pick a substring of lenth l, which is >= k, that works, but can we also pick a longer one longer than k?
            picking a longer gone reduces the chance of finding another one
            so if we don't have at least k, keep looking
        dp i, then check all k lengths from there
        '''
        memo = {}
        n = len(s)
        is_pal = [[False] * n for _ in range(n)]

        for i in range(n):
            is_pal[i][i] = True

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1

                if s[i] == s[j]:
                    if length == 2:
                        is_pal[i][j] = True
                    else:
                        is_pal[i][j] = is_pal[i + 1][j - 1]

        def dp(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            #check all >= k length palindromes
            ans = dp(i+1)
            for j in range(i + k, n + 1):
                if is_pal[i][j-1]: #one less index spot
                    ans = max(ans, 1 + dp(j))
            memo[i] = ans
            return ans
        
        return dp(0)

#########################################################
# 1621. Number of Sets of K Non-Overlapping Line Segments
# 16SEP26
########################################################
#TLE, similar to rod cutting
class Solution:
    def numberOfSets(self, n: int, k: int) -> int:
        '''
        tricky dp.... :(
        states are (i,remainng segments)
        also need a flag variable indicating whether or not we are in the iddle of placing a line (placed but not stared)
        so (i,segs,flag)
        the flag is important because i don't need to be placing a line segment
        its kinda of like stars and bars
        imagine you are given an array [0...n-1], i need to pick k tuples (l,r), where l and r must be in the range [0,n-1]
        '''
        memo = {}
        mod = 10**9 + 7

        def dp(i, seg):
            if seg == 0:
                return 1

            if i >= n:
                return 0

            if (i,seg) in memo:
                return memo[(i,seg)]

            skip = dp(i+1,seg) #extending

            #new starts!
            take = 0
            for r in range(i+1,n):
                take += dp(r,seg - 1)

            ways = (skip + take) % mod
            memo[(i,seg)] = ways
            return ways
        
        return dp(0,k)

#bottom up no suff sum optimization
class Solution:
    def numberOfSets(self, n: int, k: int) -> int:
        '''
        tricky dp.... :(
        states are (i,remainng segments)
        also need a flag variable indicating whether or not we are in the iddle of placing a line (placed but not stared)
        so (i,segs,flag)
        the flag is important because i don't need to be placing a line segment
        its kinda of like stars and bars
        imagine you are given an array [0...n-1], i need to pick k tuples (l,r), where l and r must be in the range [0,n-1]
        '''
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        mod = 10**9 + 7

        for i in range(n + 1):
            dp[i][0] = 1

        for i in range(n - 1, -1, -1):
            for seg in range(1, k + 1):

                # Don't use i as a left endpoint
                skip = dp[i + 1][seg]

                # Use i as the left endpoint
                take = 0
                for r in range(i + 1, n):
                    take += dp[r][seg - 1]

                dp[i][seg] = (skip + take) % mod

        return dp[0][k]

class Solution:
    def numberOfSets(self, n: int, k: int) -> int:
        '''
        final optimzation
        '''
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        mod = 10**9 + 7

        for i in range(n + 1):
            dp[i][0] = 1

        suffix = [0] * (k + 1)

        for i in range(n - 1, -1, -1):

            for seg in range(1, k + 1):
                skip = dp[i + 1][seg]
                take = suffix[seg]

                dp[i][seg] = (skip + take) % mod

            # Now prepare suffixes for the next row (i - 1)
            for seg in range(1, k + 1):
                suffix[seg] += dp[i][seg - 1]
                suffix[seg] %= mod

        return dp[0][k]

####################################################################
# 1477. Find Two Non-overlapping Sub-arrays Each With Target Sum
# 18SEP26
#################################################################
class Solution:
    def minSumOfLengths(self, arr: list[int], target: int) -> int:
        n = len(arr)

        pref = [float('inf')] * n
        suff = [float('inf')] * n

        # build pref
        left = 0
        curr_sum = 0
        best = float('inf')

        for right in range(n):
            curr_sum += arr[right]

            while curr_sum > target:
                curr_sum -= arr[left]
                left += 1

            if curr_sum == target:
                length = right - left + 1
                best = min(best, length)

            pref[right] = best

        # build suff
        right = n - 1
        curr_sum = 0
        best = float('inf')

        for left in range(n - 1, -1, -1):
            curr_sum += arr[left]

            while curr_sum > target:
                curr_sum -= arr[right]
                right -= 1

            if curr_sum == target:
                length = right - left + 1
                best = min(best, length)

            suff[left] = best

        ans = float('inf')

        # try every split
        for i in range(n - 1):
            if pref[i] != float('inf') and suff[i + 1] != float('inf'):
                ans = min(ans, pref[i] + suff[i + 1])

        return -1 if ans == float('inf') else ans

#######################################################
# 1520. Maximum Number of Non-Overlapping Substrings
# 18SEP26
########################################################
class Solution:
    def maxNumOfSubstrings(self, s: str) -> list[str]:
        '''
        substrings should not overlap and
        note that its impossible for any two valid substrings to overlap unless one is inside the other
        a subsring must contain all occurences of c
            so it must go up to the first and last index of the char
        '''
        #find first and last indices of each char
        first = {}
        last = {}
        for i,ch in enumerate(s):
            if ch not in first:
                first[ch] = i
            if ch not in last:
                last[ch] = i
            first[ch] = min(first[ch],i)
            last[ch] = max(last[ch],i)

        #expand range on each char
        def get_range(ch):
            l,r = first[ch],last[ch]
            i = l
            while i <= r:
                curr_ch = s[i]
                #if this char occurs before we can't make a valid substring
                if first[curr_ch] < l:
                    return None
                r = max(r,last[curr_ch])
                i += 1
            
            return (l,r)

        intervals = []
        for ch in first:
            interval = get_range(ch)
            if interval:
                intervals.append(interval)
        #sort in earliest finish
        intervals.sort(key = lambda x: x[1])
        ans = []
        end = -1
        for l,r in intervals:
            if l > end:
                ans.append(s[l:r+1])
                end = r
        return ans

#####################################################
# 1401. Circle and Rectangle Overlapping
# 20SEP26
######################################################
class Solution:
    def checkOverlap(self, radius: int, xCenter: int, yCenter: int, x1: int, y1: int, x2: int, y2: int) -> bool:
        '''
        locate closest point of square to center of the circle?, then calc distane and check <= radius
        if the closest point to the perimeter of a circle from a square also the closest to its center?
            it must be!
        find an axis aligned point from center to square, then check dist
        remember clamping for range!
        if wa have range [2,8] and a point p
        p_clamped = max(2,min(p,8))
        '''
        #find axis aligned point from center to square
        #let center point be called O
        #we have to clamp the nearest axis aligned point on the square
        closest_x = max(x1,min(xCenter,x2))
        closest_y = max(y1,min(yCenter,y2))
        sq_dist = (xCenter - closest_x)**2 + (yCenter - closest_y)**2
        return sq_dist <= radius**2
        
##########################################
# 3498. Reverse Degree of a String
# 20SEP26
##########################################
class Solution:
    def reverseDegree(self, s: str) -> int:
        '''
        cheese
        '''
        ans = 0
        for i,ch in enumerate(s):
            left = ord(ch) - ord('a')
            right = 26 - left
            ans += right*(i+1)
        
        return ans

###########################################
# 3524. Find X Value of Array I
# 21SEP26
###########################################
class Solution:
    def resultArray(self, nums: List[int], k: int) -> List[int]:
        '''
        we are allowed to remove any non-overlappng pref and suffix from nums such that it remains non-empty
        x-value is number of way to perform operation so that product of remaining elements leaves a remainder of x
        when divided by k
        so how many subarrays exists such for each product % k in range(1,k)
        '''
        memo = {}
        n = len(nums)
        #dp function, index i and current modulo
        def dp(i,r):
            if i == n:
                return [0]*k
            if (i,r) in memo:
                return memo[(i,r)]
            
            ans = [0]*k
            next_r = r*nums[i] % k
            ans[next_r] += 1
            next_counts = dp(i+1,next_r)
            for x in range(k):
                ans[x] += next_counts[x]
            
            memo[(i,r)] = ans
            return ans
        
        ans = [0]*k
        for i in range(n):
            counts = dp(i,1)
            for x in range(k):
                ans[x] += counts[x]
        
        return ans

#iterative, roll up the counts
class Solution:
    def resultArray(self, nums: List[int], k: int) -> List[int]:
        '''
        iterative
        '''
        n = len(nums)
        dp = [0]*k
        ans = [0]*k
        for i in range(n):
            next_dp = [0]*k
            next_r = nums[i] % k
            next_dp[next_r] += 1
            for x in range(k):
                next_dp[(next_r)*x % k] += dp[x]
            
            dp = next_dp[:]
            for x in range(k):
                ans[x] += dp[x]
        
        return ans

################################################
# 3550. Smallest Index With Digit Sum Equal to Index
# 24SEP26
#################################################
class Solution:
    def smallestIndex(self, nums: List[int]) -> int:
        '''

        '''
        for i,num in enumerate(nums):
            digit_sum = sum([int(ch) for ch in str(num)])
            if digit_sum == i:
                return i
        
        return -1

#####################################################
# 1807. Evaluate the Bracket Pairs of a String
# 26SEP26
#######################################################
class Solution:
    def evaluate(self, s: str, knowledge: list[list[str]]) -> str:
        '''
        no nested brackets in s
        '''
        mapp = {}
        for k, v in knowledge:
            mapp[k] = v
        
        ans = []
        n = len(s)
        i = 0

        while i < n:
            # process opening
            if s[i] == "(":
                i += 1
                curr = ""

                while i < n and s[i] != ")":
                    curr += s[i]
                    i += 1

                if curr in mapp:
                    ans.append(mapp[curr])
                else:
                    ans.append("?")

                i += 1   # move past ')'

            else:
                curr = ""

                while i < n and s[i].isalpha():
                    curr += s[i]
                    i += 1

                ans.append(curr)

        return "".join(ans)

####################################################
# 1096. Brace Expansion II 
# 28SEP26
#####################################################
class Solution:
    def braceExpansionII(self, expression: str) -> list[str]:
        '''
        whenever we have {}{}, we need to multiply them out
        when we have {},{}, its just the union
        need to process in chunks
        R("a{b,c}{d,e}f{g,h}")
        {ab,ac}{d,e}f{g,h}
        {abd,abe,acd,ace}f{g,h}
        {abdf,abef,acdf,acef}{g,h}
         {"abdfg", "abdfh", "abefg", "abefh", "acdfg", "acdfh", "acefg", "acefh"}
         we can just process left to right
         can we do it recursively?
        '''
        def multiply(A, B):
            return {a + b for a in A for b in B}

        def parse(i):
            current = {""}
            result = set()

            while i < len(expression):

                if expression[i] == '{':
                    inside, i = parse(i + 1)
                    current = multiply(current, inside)

                elif expression[i] == ',':
                    result.update(current)
                    current = {""}
                    i += 1

                elif expression[i] == '}':
                    result.update(current)
                    return result, i + 1

                else:
                    current = multiply(current,{expression[i]})
                    i += 1

            result.update(current)

            return result, i

        return sorted(parse(0)[0])

########################################################
# 2267. Check if There Is a Valid Parentheses String Path
# 28SEP26
########################################################
class Solution:
    def hasValidPath(self, grid: list[list[str]]) -> bool:
        '''
        states should be (i,j,balance)
        '''
        rows,cols = len(grid),len(grid[0])
        memo = {}
        dirrs = [(0,1),(1,0)]
        def dp(i,j,bal):
            if bal < 0:
                return False
            if (i,j) == (rows-1,cols-1):
                if bal == 0:
                    return True
                return False
            
            if (i,j,bal) in memo:
                return memo[(i,j,bal)]
            
            ans = False
            for di,dj in dirrs:
                ii,jj = i + di, j + dj
                if 0 <= ii < rows and 0 <= jj < cols:
                    next_bal = bal
                    if grid[ii][jj] == '(':
                        next_bal += 1
                    else:
                        next_bal -= 1
                    
                    if dp(ii,jj,next_bal) == True:
                        memo[(i,j,bal)] = True
                        return True
            memo[(i,j,bal)] = ans
            return ans
        
        curr_bal = 0
        if grid[0][0] == '(':
            curr_bal += 1
        else:
            curr_bal -= 1
        
        return dp(0,0,curr_bal)
