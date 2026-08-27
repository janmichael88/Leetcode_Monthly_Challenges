######################################################
# 3731. Find Missing Elements
# 04AUG26
######################################################
class Solution:
    def findMissingElements(self, nums: List[int]) -> List[int]:
        '''
        sort and record
        '''
        missing = []
        nums.sort()
        n = len(nums)
        i = 0
        while i < n:
            if i + 1 < n and nums[i+1] - nums[i] != 1:
                diff = nums[i+1] - nums[i]
                for d in range(1,diff):
                    missing.append(nums[i] + d)
            i += 1
        
        return missing
    
##################################################
# 3310. Remove Methods From Project
# 05AUG26
##################################################
class Solution:
    def remainingMethods(self, n: int, k: int, invocations: List[List[int]]) -> List[int]:
        '''
        we need to remove the suspicous projects
        a group of methods can be removed only iff no method outside the group invokes any methods within it
        if there is a cycle, those nodes need to be removed
            cycle detection in DAG
        
        invocation implies direction
        '''
        graph = defaultdict(list)
        for u,v in invocations:
            graph[u].append(v)
        
        suspicious = [False]*n
        seen = set()

        #dfs from k and mark them
        def dfs1(node,seen,possible):
            suspicious[node] = True
            seen.add(node)
            for neigh in graph[node]:
                if neigh not in seen:
                    dfs1(neigh,seen,suspicious)
        
        dfs1(k,seen,suspicious)
        
        # If any non-suspicious method invokes a suspicious one,the all nodes are not suspicious
        for u, v in invocations:
            if not suspicious[u] and suspicious[v]:
                print(u,v)
                #then the whole thing is good
                return list(range(n))

        # otherwise, we can't tell, so just take all the non suspicious ones, or remove the suspicious ones
        return [i for i in range(n) if not suspicious[i]]
    
##############################################################
# 3345. Smallest Divisible Digit Product I
# 06AUG26
################################################################
class Solution:
    def smallestNumber(self, n: int, t: int) -> int:
        '''
        '''
        def get_prod(num):
            prod = 1
            while num:
                prod = prod*(num % 10)
                num = num // 10

            return prod
        
        while get_prod(n) % t != 0:
            n += 1
        
        return n
    
####################################################################
# 2996. Smallest Missing Integer Greater Than Sequential Prefix Sum
# 10AUG26
####################################################################
class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        '''
        find the longest sequenctial pref sum. it must start at zero
        '''
        curr_sum = nums[0]
        n = len(nums)
        for i in range(1,n):
            if nums[i] - nums[i-1] == 1:
                curr_sum += nums[i]
            else:
                break
            
        #we need the smallest x missing fomr nums such x is >= sum of the longest sequential prefix
        nums = set(nums)
        print(curr_sum)
        while curr_sum in nums:
            curr_sum += 1
        return curr_sum
    
#####################################################
# 3941. Password Strength
# 12AUG26
#####################################################
class Solution:
    def passwordStrength(self, password: str) -> int:
        '''
        de uplicate and just apply point totals
        '''
        password = set(list(password))

        ans = 0
        for ch in password:
            if "a" <= ch <= "z":
                ans += 1
            elif "A" <= ch <= "Z":
                ans += 2
            elif "0" <= ch <= "9":
                ans += 3
            else:
                ans += 5
        return ans
    
##################################################
# 2213. Longest Substring of One Repeating Character
# 13AUG26
####################################################
class SegmentTree:
    def __init__(self, s):
        self.n = len(s)
        self.tree = [None] * (4 * self.n)

        self.build(1, 0, self.n - 1, s)

    def build(self, node, l, r, s):
        if l == r:
            c = s[l]
            #leaf # (left_char, right_char, left_len, right_len, best,length)
            self.tree[node] = (c, c, 1, 1, 1, 1)
            return

        mid = (l + r) // 2

        self.build(node * 2, l, mid, s)
        self.build(node * 2 + 1, mid + 1, r, s)

        self.tree[node] = self.merge(self.tree[node * 2],self.tree[node * 2 + 1])

    def merge(self, left, right):
        # (left_char, right_char, left_len, right_len, best, length)

        new_left_char = left[0]
        new_right_char = right[1]

        # Best run crossing the boundary
        merge_middle_len = 0

        if left[1] == right[0]:
            merge_middle_len = left[3] + right[2]

        # Prefix, matches and extends
        new_left_len = left[2]

        if left[1] == right[0] and left[2] == left[5]:
            new_left_len += right[2]

        # Suffix, matches and extends
        new_right_len = right[3]

        if left[1] == right[0] and right[3] == right[5]:
            new_right_len += left[3]

        # Best anywhere in the combined segment
        new_best = max(left[4],right[4],merge_middle_len)
        new_total_length = left[5] + right[5]

        return (new_left_char,new_right_char,new_left_len,new_right_len,new_best,new_total_length)
            
    def update(self, node, l, r, idx, c):
        if l == r:
            self.tree[node] = (c, c, 1, 1, 1, 1)
            return

        mid = (l + r) // 2

        if idx <= mid:
            self.update(node * 2, l, mid, idx, c)
        else:
            self.update(node * 2 + 1, mid + 1, r, idx, c)

        self.tree[node] = self.merge(self.tree[node * 2],self.tree[node * 2 + 1])

    def get_answer(self):
        return self.tree[1][4]
        
class Solution:
    def longestRepeating(self, s: str, queryCharacters: str, queryIndices: List[int]) -> List[int]:
        '''
        remember pattern for segment tree
        build:
            if leaf:
                create leaf
            else:
                build left
                build right
                merge

        update:
            if leaf
                update leaf
            else
                update left or right
                merge
        merge:
            combine from left + right nodes
        '''
        seg = SegmentTree(s)
        n = len(s)
        ans = []
        for idx,c in zip(queryIndices,queryCharacters):
            seg.update(1,0,n-1,idx,c)
            ans.append(seg.get_answer())
        return ans
    
#######################################################
# 3090. Maximum Length Substring With Two Occurrences
# 13AUG26
#######################################################
class Solution:
    def maximumLengthSubstring(self, s: str) -> int:
        '''
        substring can only contain at most two occurences of each character
        sliding window
        '''
        curr = Counter()
        ans = 0
        left = 0
        for right,ch in enumerate(s):
            curr[ch] += 1
            while left < right and curr[ch] > 2:
                curr[s[left]] -= 1
                left += 1
            
            ans = max(ans,right - left + 1)
        
        return ans
    

#####################################################
# 3702. Longest Subsequence With Non-Zero Bitwise XOR
# 16AUG26
##########################################################
class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        '''
        need longest subequence who's bitwize xor is no zero
        '''
        xor = 0
        n = len(nums)

        if nums.count(0) == n:
            return 0
        for num in nums:
            xor = xor ^ num
        
        if xor == 0:
            return n - 1
        return n

##################################################
# 2029. Stone Game IX
# 16AUG26
##################################################
class Solution:
    def stoneGameIX(self, stones: List[int]) -> bool:
        '''
        alice starts first
        the player who removes a stone loses if the sum of all the value of all removed stones is divisble by 3
        bob will win if there are on remaining stones (even if its alices turn)
        is there not a subsequence that leads to a sum divisible by 3?
        [1] Alice loses
        [1, 1] Alice loses
        [1, 2] Alice wins
        [1, 1, 2] Alice wins
        [1, 2, 2] Alice wins
        [1, 1, 1, 2, 2] Alice wins
        after considering stones of %3 = 0, the moves are of the form
        1121212121...
        and
        2212121212...
        stones of type 0, just switch the current remainder sum % 3 to to other peron's sturn
        '''
        #for each number in nums, convert to % 3
        count_rems = Counter()
        for stone in stones:
            count_rems[stone % 3] += 1

        #if we have an even occurnece of % 3 == 0
        #then alice will win if we have at least some number of %3 = 1 and %3 = 2
        if count_rems[0] % 2 == 0:
            return count_rems[1] > 0 and count_rems[2] > 0
        #odd occurence of % 3 == 0
        return abs(count_rems[1] - count_rems[2]) > 2      
    
#####################################################
# 3800. Minimum Cost to Make Two Binary Strings Equal
# 16AUG26
#######################################################
class Solution:
    def minimumCost(self, s: str, t: str, flipCost: int, swapCost: int, crossCost: int) -> int:
        '''
        we can choose any index i in s or t and flip it from 0 to 1 -> flipCost
        we can chose two distinct indices (i,j) and swap (s[i],s[j]) or (t[i],t[j]) -> swapCost
        wen chose an idnex i and swap s[i] with t[i] -> crossCost
        i say greedy solution 
        example s = "01"
                t = "10"
                we can flip at each position, 2*flipCosst
                or we can swap indices (0,1) in s or in t

        '''
        ca = 0 # 01
        cb = 0 # 10
        ans = 0
        for i,j in zip(s,t):
            if i=='0' and j == '1':
                ca+=1
            elif i=='1' and j=='0':
                cb+=1
        
        fswap = swapCost if swapCost < 2*flipCost else 2*flipCost
        sswap = crossCost + swapCost if crossCost + swapCost < 2*flipCost else 2*flipCost

        takeout = min(ca,cb)
        ans += takeout*fswap

        takeout_rem = abs(ca-cb)//2
        ans += takeout_rem*sswap

        #last one
        rem = abs(ca-cb)
        if rem % 2:
            ans += flipCost
        return ans
    
########################################
# 1563. Stone Game V
# 16AUG26
########################################
#close one
class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:
        '''
        we want to maximize the Alices score,
        at each step, split rows into two [left],[right]
        bob throws are the partitions with max value and alice goes up by minimum
        [arr] -> [left],[right]
        alice += min(left,right)
        arr = min(left,right)
        dp(i,j), then try splitting on all k between i and j paradigm
            gives the max score i can get for that range
        '''
        #try recursion on the array first
        #then think of states
        def rec(arr):
            if not arr:
                return 0
            n = len(arr)
            ans = 0
            for i in range(n):
                left,right = arr[:i+1],arr[i+1:]
                sum_left,sum_right = sum(left),sum(right)
                if sum_left > sum_right:
                    ans = max(ans, sum_right + rec(right))
                elif sum_right > sum_left:
                    ans = max(ans, sum_left + rec(left))
                #equal case, try both
                else:
                    both = max(sum_right + rec(right),sum_left + rec(left))
                    ans = max(ans,both)
            
            return ans
        
        return rec(stoneValue)
                    

#dp(i,j) but with pref_sums now
#TLE
class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:
        '''
        we want to maximize the Alices score,
        at each step, split rows into two [left],[right]
        bob throws are the partitions with max value and alice goes up by minimum
        [arr] -> [left],[right]
        alice += min(left,right)
        arr = min(left,right)
        dp(i,j), then try splitting on all k between i and j paradigm
            gives the max score i can get for that range
        prefsum
        '''
        n = len(stoneValue)
        pref_sum = [0]
        for s in stoneValue:
            pref_sum.append(pref_sum[-1] + s)

        #memo = {}

        @cache
        def dp(i,j):
            if i >= j:
                return 0
            #if (i,j) in memo:
            #    return memo[(i,j)]
            
            ans = 0
            for k in range(i,j):
                sum_left = pref_sum[k+1] - pref_sum[i]
                sum_right = pref_sum[j+1] - pref_sum[k+1]
                if sum_left > sum_right:
                    ans = max(ans, sum_right + dp(k+1,j))
                elif sum_right > sum_left:
                    ans = max(ans, sum_left + dp(i,k))
                #equal case, try both
                else:
                    both = max(sum_right + dp(k+1,j),sum_left + dp(i,k))
                    ans = max(ans,both)
            
            #memo[(i,j)] = ans
            return ans

        return dp(0,n-1)

#make it pass, just take sum one before searching on all k ietween (i,j)
class Solution:
    def stoneGameV(self, stoneValue: List[int]) -> int:
        '''
        we want to maximize the Alices score,
        at each step, split rows into two [left],[right]
        bob throws are the partitions with max value and alice goes up by minimum
        [arr] -> [left],[right]
        alice += min(left,right)
        arr = min(left,right)
        dp(i,j), then try splitting on all k between i and j paradigm
            gives the max score i can get for that range
        prefsum
        '''
        memo = {}
        n = len(stoneValue)
        @lru_cache(None)
        def dp(i,j):
            if i >= j:
                return 0
            #if (i,j) in memo:
            #    return memo[(i,j)]
            
            ans = 0
            total_sum = sum(stoneValue[i:j+1])
            sum_left = 0
            sum_right = 0
            for k in range(i,j):
                sum_left += stoneValue[k]
                sum_right = total_sum - sum_left
                if sum_left > sum_right:
                    ans = max(ans, sum_right + dp(k+1,j))
                elif sum_right > sum_left:
                    ans = max(ans, sum_left + dp(i,k))
                #equal case, try both
                else:
                    both = max(sum_right + dp(k+1,j),sum_left + dp(i,k))
                    ans = max(ans,both)
            
            #memo[(i,j)] = ans
            return ans

        return dp(0,n-1)

################################################
# 3471. Find the Largest Almost Missing Integer
# 18AUG26
################################################
class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        '''
        brute force it
        it can appear only in one subarray
        but can it have multiple repeats of it, yes its allowed
        '''
        ans = -1
        n = len(nums)
        counts = Counter()

        for i in range(0,n-k+1):
            arr = set(nums[i:i+k])
            for num in arr:
                counts[num] += 1
        
        for k,v in counts.items():
            if v == 1:
                ans = max(ans,k)
        
        return ans
    
#########################################
# 1386. Cinema Seat Allocation
# 20AUG26
##########################################
class Solution:
    def maxNumberOfFamilies(self, n: int, reservedSeats: List[List[int]]) -> int:
        '''
        n rows and 10 columns
        brute force would be to count the gaps in each of the rows, but n can be very big
        oh check onl rows the appear in 
        no matter what i can fit at most two 4 groups in a row
        we can add the reminain later
        '''
        mapp = defaultdict(set)

        for r, c in reservedSeats:
            mapp[r].add(c)

        ans = 2 * (n - len(mapp))

        for seats in mapp.values():
            left = all(c not in seats for c in range(2, 6))   # 2-5
            middle = all(c not in seats for c in range(4, 8)) # 4-7
            right = all(c not in seats for c in range(6, 10))  # 6-9

            if left and right:
                ans += 2
            elif left or middle or right:
                ans += 1

        return ans
    
###############################################
# 3069. Distribute Elements Into Two Arrays I
# 20AUG26
###############################################
class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        '''
        follow the rules
        '''
        arr1,arr2 = [nums[0]],[nums[1]]
        n = len(nums)

        op = 2
        while op < n:
            if arr1[-1] > arr2[-1]:
                arr1.append(nums[op])
            else:
                arr2.append(nums[op])
            
            op += 1
        
        return arr1 + arr2
    
#####################################################
# 3622. Check Divisibility by Digit Sum and Product
# 23AUG26
####################################################
class Solution:
    def checkDivisibility(self, n: int) -> bool:
        '''
        check rules
        '''
        curr_sum = 0
        curr_prod = 1
        for d in str(n):
            curr_sum += int(d)
            curr_prod *= int(d)
        
        return (n % (curr_sum + curr_prod) == 0)
    

#################################################
# 1927. Sum Game
# 24AUG26
###################################################
class Solution:
    def sumGame(self, num: str) -> bool:
        '''
        alice/bob, alice stars first
        on each turn, chose index i where num[i] == "?"
        replace with any digit
        game ends when there are no more '?'
        for bob to win, first half of num == second half of num
        for alice to win first half !? second half
        look at this example 25??
        left sum is 7, and right sum is zero
        if alice picks a number less than the diff, which is 7, bob and pick (1 - wtv_number_alice_picks) to make it equal
        in fact any number greater than 7, would allow alice to win
        if i pair a ? on the left with a ? on the right
        then a player can chose x, then the opposing player can pick 9 - x, and we can get x + (9-x) = 0
        if left qs == righ qs, bob can always respond to alice, and of sums are equal, bob can win
        if there are unequal qs, alice can chose a digit that does not allow bob to make the sums equal

        '''
        #first find left sum and right sum, and left qs and right qs
        n = len(num)
        left_sum, left_qs = 0,0
        for l in num[:n//2]:
            if l == "?":
                left_qs += 1
            else:
                left_sum += int(l)

        right_sum, right_qs = 0,0
        for r in num[n//2:]:
            if r == "?":
                right_qs += 1
            else:
                right_sum += int(r)
        
        # Difference in existing sums
        diff = left_sum - right_sum

        # Difference in number of '?'s
        qdiff = left_qs - right_qs

        # Alice wins iff the existing difference cannot be
        # compensated by the unmatched '?'s.
        return diff * 2 + 9 * qdiff != 0
    
###################################################
# 1872. Stone Game VIII
# 24AUG26
###################################################
class Solution:
    def stoneGameVIII(self, stones: List[int]) -> int:
        '''
        alice and bob again, alice goes first
        on each turn choose an integer x > 1, and remove the leftmost x stones
        increment the players score by their sum, add new stone back to left which == sum
        game stops when only one stone left
        need max diff between alice and bob score
        n is to big to do states on (i,j) must be on i only
        if a player takes a stone at i, their score went up by sum(stones[:i]) + prev_sum_before
        if they didn't it must have gone up by prev_sum_before
        '''
        n = len(stones)

        pref = [0] * n
        pref[0] = stones[0]

        for i in range(1, n):
            pref[i] = pref[i - 1] + stones[i]

        @lru_cache(None)
        def dp(i):
            #the only move here is to take the whole array
            if i == n - 1:
                return pref[i]
            take = pref[i] - dp(i+1)
            no_take = dp(i+1)
            return max(take,no_take)
        return dp(1)
    
#############################################
# 3718. Smallest Missing Multiple of K
# 25AUG26
##############################################
class Solution:
    def missingMultiple(self, nums: List[int], k: int) -> int:
        '''
        turn nums into set and skip by k
        '''
        nums = set(nums)
        ans = k
        while ans in nums:
            ans += k
        
        return ans
    
#################################################################
# 2904. Shortest and Lexicographically Smallest Beautiful String
# 26AUG26
#################################################################
class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        '''
        try brute force first
        '''
        smallest_length = float('inf')
        ans = ""
        n = len(s)
        for i in range(n):
            for j in range(i+1,n+1):
                sub = s[i:j]
                if sub.count("1") == k:
                    if len(sub) < smallest_length:
                        smallest_length = len(sub)
                        ans = sub
                    elif len(sub) == smallest_length:
                        ans = min(ans,sub)
        return ans
    
#sliding window
class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        '''
        we can also do sliding window
        '''
        smallest_length = float('inf')
        ans = ""
        n = len(s)
        window = Counter()
        left = 0
        for right,ch in enumerate(s):
            window[ch] += 1
            while window["1"] > k:
                window[s[left]] -= 1
                left += 1
            
            while window["1"] == k:
                sub = s[left:right + 1]
                if len(sub) < smallest_length:
                    smallest_length = len(sub)
                    ans = sub
                elif len(sub) == smallest_length:
                    ans = min(ans, sub)

                window[s[left]] -= 1
                left += 1
        
        return ans