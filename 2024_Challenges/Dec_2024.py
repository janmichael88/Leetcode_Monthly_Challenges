############################################
# 2109. Adding Spaces to a String
# 03DEC24
#############################################
class Solution:
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        '''
        slice the string at left and right
        prepend 0 and just slice
        '''
        spaces = [0] + spaces + [len(s)]
        words = []
        for i in range(len(spaces) - 1):
            temp = s[spaces[i]: spaces[i+1]]
            words.append(temp)
        
        return " ".join(words)

class Solution:
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        '''
        we can also just keep two pointers 
        one into spaces array meaning we can get to a space index
        and another into string
        '''
        ans = []
        space_idx = 0
        
        for i in range(len(s)):
            #in bounds and is the index we need a space for
            if space_idx < len(spaces) and i == spaces[space_idx]:
                ans.append(" ")
                space_idx += 1
            
            ans.append(s[i])
        
        return "".join(ans)

##########################################################
# 2825. Make String a Subsequence Using Cyclic Increments
# 04DEC24
#########################################################
class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        '''
        we select any set of indices in str1 and promot the char at that index (cyclically)
        check if we can make str2 a subsequence of str1 using the operation at most once
        checking if str2 isSubseq takes linear time
        
        each char in s can either be s[i] or (s[i] + 1) % 26 so check both
        '''
        idx_2 = 0
        
        for ch in str1:
            promoted_idx = (ord(ch) - ord('a') + 1) % 26
            promoted_ch = chr(ord('a') + promoted_idx)
            if ch == str2[idx_2] or promoted_ch == str2[idx_2]:
                idx_2 += 1
                if idx_2 == len(str2):
                    return True
        
        return idx_2 == len(str2)
            

    def is_sub(self,s,t):
        #check if t is subseq of s
        t_idx = 0
        for s_idx in range(len(s)):
            if t[t_idx] == s[s_idx]:
                t_idx += 1
        
        return t_idx == len(t)

##############################################
# 2337. Move Pieces to Obtain a String
# 05DEC24
##############################################
class Solution:
    def canChange(self, start: str, target: str) -> bool:
        '''
        we can move L to the left if there's a blank space
        we can move R to the right if there's a blank space
        using any number of moves, return true if we can reach target
        if we have target like:
        "L______RR", L could be in any of the positions "_"
        
        for both start and target, try to move L's left as possible and R's right as possible
        then check if equal
        relative order or L and R must stay the same
        omg store indices and L's and R's for each start and target
        then compare the indices
        L pieces are the start must be to the left of the L pieces in target
        R pieces at the start must be to the right of R pieces in target
        '''
        start_q = deque([])
        end_q = deque([])
        n = len(start)
        #lengthg of start and targer are equal
        for i in range(n):
            if start[i] != '_':
                start_q.append((start[i],i))
            if target[i] != '_':
                end_q.append((target[i],i))
        
        if len(start_q) != len(end_q):
            return False
        while start_q:
            start_char,start_idx = start_q.popleft()
            end_char, end_idx = end_q.popleft()
            
            if start_char != end_char:
                return False
            if start_char == 'L' and start_idx < end_idx:
                return False
            if start_char == 'R' and start_idx > end_idx:
                return False
        
            
        return True
                
#using array
class Solution:
    def canChange(self, start: str, target: str) -> bool:
        '''
        not using queue, just build as array
        '''
        start_q = []
        end_q = []
        n = len(start)
        #lengthg of start and targer are equal
        for i in range(n):
            if start[i] != '_':
                start_q.append((start[i],i))
            if target[i] != '_':
                end_q.append((target[i],i))
        
        if len(start_q) != len(end_q):
            return False
        
        for s,e in zip(start_q,end_q):
            
            start_char,start_idx = s
            end_char, end_idx = e
            
            if start_char != end_char:
                return False
            if start_char == 'L' and start_idx < end_idx:
                return False
            if start_char == 'R' and start_idx > end_idx:
                return False
        
            
        return True
    
############################################################
# 2554. Maximum Number of Integers to Choose From a Range I
# 06DEC24
###########################################################
class Solution:
    def maxCount(self, banned: List[int], n: int, maxSum: int) -> int:
        '''
        can only pick numbers from [1,n] but they cannot be banned
        can only use once, and their sum shouldn't exceed maxSum
        n is small
        there can be multiple
        sort banned incrreasing, then talk the smallest

        you need to use set, since they might not be unique!
        '''
        banned = set(banned)
        count = 0
        curr_sum = 0
        b_idx = 0
        
        for num in range(1,n+1):
            #is banned
            if num in banned:
                continue
            if curr_sum + num > maxSum:
                return count
            #otherwise we an just take it
            curr_sum += num
            count += 1

        return count
        
#binary search variant is just that instead of hashing to look for a banned number
#you is binary search in sorted(banned) to find it

##########################################
# 1760. Minimum Limit of Balls in a Bag
# 07DEC24
##########################################
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        '''
        i need to perform maxoperations
        an operation is the result if splitting a bag into two marbles
        penalty is the maximum of balls in bag
        minimize this penalty
        if we have some number k, we can make this is small as possible by doing
        say we have a number 7, if we split to [1,6], the penalty is just 6
        but if we split to [4,3], the penalty is 4, if we want to minimize penalty for one bag, we should just [k//2,k//2-1]
        we can further reduce 4 -> [2,2] and 3 -> [1,2]
        
        if we have a bag with k balls, and maxoperations
        we can make, 2**k bags if k is large enough, provided we use it on that kth bag, 
        intuition:
            say we a limit k_balls
            can we determine if its possible to split the balss so that no bag contains more than k_balls
        
        given nums[i] balls, how man ops would it take to make it so that in any of the bags, there at most k_balls
        we need to find the number of splits to ensure no bag exceeds this k_balls
        trick -> ceil(a/b) == (a + b - 1) // b
        we cant have partial balls, so we round up, if we round down, we lose a ball
        '''
        left = 1
        right = max(nums)
        ans = -1
        
        while left <= right:
            mid = left + ((right - left) // 2)
            if self.at_most(nums,mid,maxOperations):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
    
    def at_most(self, nums, k_balls, maxOperations):
        total_operations = 0
        for num in nums:
            #if bigger than k_balls, we need to split into bags so that its <= k_balls
            if num > k_balls:
                needed_ops = (num - 1) // k_balls
                total_operations += needed_ops
                
        return total_operations <= maxOperations
            
############################################
# 2054. Two Best Non-Overlapping Events
# 08DEC24
#############################################
class Solution:
    def maxTwoEvents(self, events: List[List[int]]) -> int:
        '''
        choose at most two events that are non-overlapping and maximize their sum
        start,end times are inclusive, if event ends at time t, must choose another event starting at t + 1
        if i sorted the events, i can pick an event, then binary search  for the next event that ends after i[1] 
        but then i would need to find the event that has the max value, max to the right
        need to binary search for the start time just greater than the current end
        '''
        n = len(events)
        events.sort(key = lambda x : x[0])
        right_maxs = [0]*n
        right_maxs[n-1] = events[n-1][2] #get value
        for i in range(n-2,-1,-1):
            right_maxs[i] = max(right_maxs[i+1],events[i][2] )
        
        ans = 0
        for i in range(n):
            curr_start,curr_end,curr_value = events[i]
            ans = max(ans,curr_value) #i dont have to take two events
            #binary seach for curr_end + 1
            left = i + 1
            right = n - 1
            best_idx = -1
            while left <= right:
                mid = left + (right - left) // 2
                if events[mid][0] <= curr_end:
                    left = mid + 1
                else:
                    best_idx = mid
                    right = mid - 1
            if best_idx != -1:
                ans = max(ans,curr_value, curr_value + right_maxs[best_idx])
        
        return ans
                
#can also do dp, 0/1 knapsack, but for the 1 case, binary search for it
class Solution:
    def maxTwoEvents(self, events: List[List[int]]) -> int:
        '''
        we can also do dp,
        keep track of number of events taken and and index
        dp(event,index)
        if events == 2 or index >= len(events) base cae
        '''
        n = len(events)
        events.sort(key = lambda x : x[0])
        memo = {}
        
        def dp(i,count):
            if i >= len(events) or count == 2:
                return 0
            if (i,count) in memo:
                return memo[(i, count)]
            left = i + 1
            right = n - 1
            best_idx = -1
            while left <= right:
                mid = left + (right - left) // 2
                if events[mid][0] <= events[i][1]:
                    left = mid + 1
                else:
                    best_idx = mid
                    right = mid - 1
            
            if best_idx != -1:
                take = events[i][2] + dp(best_idx,count+1)
            else:
                #take at least one event
                take = events[i][2]
            no_take = dp(i+1,count)
                
            ans = max(take,no_take)
            memo[(i,count)] = ans
            return ans
        
        return dp(0,0)
                
#########################
# 3152. Special Array II
# 10DEC24
#########################
class Solution:
    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        '''
        speical array -> every pair of its adjacent elements contains two number with different parity
        for each query q, we need to see if the subarray is speical or not
        split array into some non intersected continupus speical subarrays
        for each query check that the frist and last elements are in same subarray
        '''
        intervals = []
        curr_interval = [0]
        n = len(nums)
        for i in range(1,n):
            if (nums[i-1] % 2) != (nums[i] % 2):
                curr_interval.append(i)
            else:
                intervals.append(curr_interval)
                curr_interval = [i]
        
        if curr_interval:
            intervals.append(curr_interval)
        
        #hash each index to a group
        group = 0
        mapp = {}
        for sub in intervals:
            for i in sub:
                mapp[i] = group
            
            group += 1
        
        
        ans = []
        for l,r in queries:
            ans.append(mapp[l] == mapp[r])
        
        return ans
    
#############################################################
# 2981. Find Longest Special Substring That Occurs Thrice I
# 12DEC24
###############################################################
class Solution:
    def maximumLength(self, s: str) -> int:
        '''
        sliding window
        valid criteria is iff winow only has 1 elements and occurs at least three times
        oh whoops we need the the acutal substring
        brute force should work
        '''
        counts = Counter()
        ans = -1
        n = len(s)
        for i in range(n):
            for j in range(i+1,n+1):
                substring = s[i:j]
                if len(set(substring)) == 1:
                    counts[substring] += 1
                    if counts[substring] >= 3:
                        ans = max(ans,j-i)
        
        return ans
    
class Solution:
    def maximumLength(self, s: str) -> int:
        '''
        sliding window
        valid criteria is iff winow only has 1 elements and occurs at least three times
        oh whoops we need the the acutal substring
        brute force should work
        
        we can optimize in the inner loop using hashmaep
        just jeep track the the last character added if its not break out of the loop
        hashing strings takes too long, so instead of the key being the string
        intuition:
            since all special substrings consist of equal chars, the key signature will be (char, count_chars)
            
        #make sure to do part 2
        
            
        '''
        counts = Counter()
        ans = -1
        n = len(s)
        for i in range(n):
            last_char = ""
            curr_length = 0
            for j in range(i,n):
                if s[j] == last_char or last_char == "":
                    last_char = s[j]
                    curr_length += 1
                    counts[(last_char,curr_length)] += 1
                    if counts[(last_char,curr_length)] >= 3:
                        ans = max(ans,j-i+1)
                else:
                    break
                
                    
        
        return ans

############################################################
# 2779. Maximum Beauty of an Array After Applying Operation
# 12DEC24
#############################################################
class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        '''
        in one operation, we can pick an index i
        replace nums[i] with in integer in range [nums[i] -k, nums[i] + 1]
        beauty is longet subsequence conssiting of equal elements
        we can use any number of operations
        [a,b,c,d]
        if we sort the array, then we can just compare the left and right bounds of the subarray
        if nums[rigt] - nums[left] > 2*k, it cannot be a valid array
        '''
        nums.sort()
        n = len(nums)
        left = 0
        ans = 0
        for right,num in enumerate(nums):
            while nums[right] - nums[left] > 2*k:
                left += 1
            
            ans = max(ans, right - left + 1)
        
        return ans

#########################################################
# 2593. Find Score of an Array After Marking All Elements
# 13DEC24
#########################################################
class Solution:
    def findScore(self, nums: List[int]) -> int:
        '''
        map nums to list of indices,
        then use heap or sort to go in order
        use the mapp to retrieve indices
        '''
        n = len(nums)
        marked = [False]*n
        min_heap = [(num,i) for i,num in enumerate(nums)]
        heapq.heapify(min_heap)
        
        score = 0
        while min_heap:
            curr_num,curr_idx = heapq.heappop(min_heap)
            if not marked[curr_idx]:
                score += curr_num
                #its adjacents
                if curr_idx - 1 >= 0 and not marked[curr_idx - 1]:
                    marked[curr_idx - 1] = True
                if curr_idx + 1 < n and not marked[curr_idx + 1]:
                    marked[curr_idx + 1] = True
        
        return score
                    
#######################################
# 2762. Continuous Subarrays
# 14DEC24
#######################################
class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        '''
        a continuous subarray is some subarray[i:j]
        where abs(nums[i] - nums[i_2]) <= 2 for all pairs in the subarray
        if we have a continuous subarray of size k, then it will contribute k*(k+1) subarrays
        we only need to know if the max of subarray - min of a subarray > 2
        we can track the maximmum and minmum for a current window
        if the current window is not continueuous, keep shrinking until it becomes
        if we have a valid window, then any subarray ending with right is valid so its right - left + 1
        we can either say ending at right or starting with left
        '''
        left = 0
        counts = Counter()
        ans = 0
        
        for right,num in enumerate(nums):
            counts[num] += 1
            #this part can be optimized many differwant ways
            while left < right and max(counts) - min(counts) > 2:
                counts[nums[left]] -= 1
                if counts[nums[left]] == 0:
                    del counts[nums[left]]
                left += 1
            
            #count valid continuous subarrays
            ans += right - left + 1
        
        return ans
            