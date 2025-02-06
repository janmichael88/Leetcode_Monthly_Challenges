###################################
# 3151. Special Array I
# 01FEB25
##################################
class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        '''
        just check left and right
        '''
        n = len(nums)
        for i in range(n):
            if i == 0:
                if i + 1 < n and nums[i] % 2 == nums[i+1] % 2:
                    return False
            elif i == n-1:
                if i - 1 >= 0 and nums[i-1] % 2 == nums[i] % 2:
                    return False
            else:
                if (nums[i-1] % 2 == nums[i] % 2) or (nums[i] % 2 == nums[i+1] % 2):
                    return False
        return True
    
class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        '''
        only need to check i to i+1 pairs
        i to i-1 pair is the same
        can also use bitwize with XOR
        '''
        n = len(nums)
        for i in range(n-1):
            if (nums[i] & 1) ^ (nums[i+1] & 1) == 0:
                return False
        return True
    
###############################################
# 1852. Distinct Numbers in Each Subarray
# 01FEB25
################################################
class Solution:
    def distinctNumbers(self, nums: List[int], k: int) -> List[int]:
        '''
        sliding window of counts
        '''
        left = 0
        ans = []
        window = Counter()
        
        for right,num in enumerate(nums):
            window[num] += 1
            #shrink first if we have to
            if right - left + 1 > k:
                window[nums[left]] -= 1
                if window[nums[left]] == 0:
                    del window[nums[left]]
                left += 1
            #valid window
            if right - left + 1 == k:
                ans.append(len(window))
        
        return ans
    
####################################################
# 2940. Find Building Where Alice and Bob Can Meet
# 02FEB25
###################################################
class Solution:
    def leftmostBuildingQueries(self, heights: List[int], queries: List[List[int]]) -> List[int]:
        '''
        is person is staning on index i, they can move to index j if i < j and heights[i] < heights[j]
        for each query find leftmost index where they can jump too
        for an index i, we need to find the left most index j, where heights[j] > heights[i]
        for each query (l,r) we can take max(heights[l],heights[r]), then find the left most index that is just greater
        left most index is the same as closes to the right
        
        imagine the queries were single indices
        we could traverse the heights in reverse (monostack) and for each query find the largest height in the stack
        we maintain a stack of indices in decreasing order of heights
            i.e build stack, and binary search in the stack
        for the current building any shorter or equal buildings in the stack, cannot be the anser, so pop them
        if the stack is not empty, the top if our answer, if empty its -1 -> similar to next greater eleemnt ii
        
        now for pairs of indices, the task is to find the first height to ther ight
        intutino, we look for the height that is just greater than the max of the height indices
        while processing the a queyr, the stack already contains all element greater than the current hegith
        idea is that the stack already contains all the elements > than the current height
        '''
        mono_stack = []
        result = [-1 for _ in range(len(queries))]
        new_queries = [[] for _ in range(len(heights))]
        #each index stores the list of queries that require this index as the maximum index of the query pair
        #each query is stored as a pair contain the required height (heighs[a]) and the query index
        for i in range(len(queries)):
            a = queries[i][0]
            b = queries[i][1]
            if a > b:
                a, b = b, a
            #b is bigger than a, so the query so far b is just heights[b]
            #if they are equal, set as b, since a <= b
            if heights[b] > heights[a] or a == b:
                result[i] = b
            else:
                #otherwise we need to look for all heights from index a, for this query
                new_queries[b].append((heights[a], i))

        for i in range(len(heights) - 1, -1, -1):
            mono_stack_size = len(mono_stack)
            for a, b in new_queries[i]:
                #we are looking for the first building with a height > than the query's required height
                position = self.search(a, mono_stack)
                if position < mono_stack_size and position >= 0:
                    #we're looking for the index, not the height
                    result[b] = mono_stack[position][1]
            #monostack to keep track of building heights and their indices in decreasing order of height
            while mono_stack and mono_stack[-1][0] <= heights[i]:
                mono_stack.pop()
            mono_stack.append((heights[i], i))
            print(mono_stack)
        return result

    def search(self, height, mono_stack):
        left = 0
        right = len(mono_stack) - 1
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if mono_stack[mid][0] > height:
                ans = max(ans, mid)
                left = mid + 1
            else:
                right = mid - 1
        return ans
    
######################################################################
# 3105. Longest Strictly Increasing or Strictly Decreasing Subarray
# 03JAN25
#######################################################################
class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        '''
        count streaks for both inc and dec, then max
        can either do it seperately or do both on the fly
        need to maintain of strek was broken or not
        '''
        #do separetley
        lis = 1
        lis_streak = 1
        n = len(nums)
        for i in range(1,n):
            if nums[i] > nums[i-1]:
                lis_streak += 1
            else:
                lis_streak = 1
            lis = max(lis,lis_streak)
        
        dis = 1
        dis_streak = 1
        n = len(nums)
        for i in range(1,n):
            if nums[i] < nums[i-1]:
                dis_streak += 1
            else:
                dis_streak = 1
            dis = max(dis,dis_streak)
        
        return max(lis,dis)
        
class Solution:
    def longestMonotonicSubarray(self, nums: List[int]) -> int:
        '''
        count streaks for both inc and dec, then max
        can either do it seperately or do both on the fly
        need to maintain of strek was broken or not
        '''
        ans = 1
        lis_streak = 1
        dis_streak = 1
        n = len(nums)

        for i in range(1,n):
            if nums[i] > nums[i-1]:
                lis_streak += 1
                dis_streak = 1
            elif nums[i] < nums[i-1]:
                dis_streak += 1
                lis_streak = 1
            else:
                dis_streak = 1
                lis_streak = 1
            
            ans = max(ans,lis_streak,dis_streak)
        
        return ans
        
#########################################
# 1316. Distinct Echo Substrings
# 03FEB25
#########################################
#brute force
#check each substring, and compare left half to right half and check if echo
class Solution:
    def distinctEchoSubstrings(self, text: str) -> int:
        '''
        for each substring, just check if left half == right half
        if it is, the substring is an echo
        '''
        n = len(text)
        ans = set()

        for i in range(n):
            for j in range(i+1,n+1):
                substring = text[i:j]
                l = len(substring)
                if substring[:l//2] == substring[l//2:]:
                    ans.add(substring)
        
        return len(ans)
    
class Solution:
    def distinctEchoSubstrings(self, text: str) -> int:
        '''
        Precompute prefix hash for text
        Use prefix hash comparison to detect echo substrings
        '''
        n = len(text)
        ans = set()
        p = 31  # Alphabet size prime
        m = 10**9 + 7  

        # Precompute powers of p
        p_pow = [1] * (n + 1)
        for i in range(1, n + 1):
            p_pow[i] = (p_pow[i - 1] * p) % m

        # Compute prefix hashes
        pref_hash = [0] * (n + 1)
        for i in range(n):
            idx = ord(text[i]) - ord('a') + 1  # Convert char to int
            pref_hash[i + 1] = (pref_hash[i] * p + idx) % m

        # Check all possible echo substrings
        for length in range(1, n // 2 + 1):  # Iterate over possible lengths
            for i in range(n - 2 * length + 1):  # Ensure enough room for both halves
                j = i + length
                hash1 = self.get_hash(i, j, pref_hash, p_pow, m)
                hash2 = self.get_hash(j, j + length, pref_hash, p_pow, m)

                if hash1 == hash2:
                    ans.add(text[i:j + length])  # Store the unique substring

        return len(ans)  # Count distinct echo substrings

    def get_hash(self, i, j, pref_hash, p_pow, m):
        hash_value = (pref_hash[j] - pref_hash[i] * p_pow[j - i]) % m
        return hash_value if hash_value >= 0 else hash_value + m

##########################################
# 2100. Find Good Days to Rob the Bank
# 04FEB25
##########################################
class Solution:
    def goodDaysToRobBank(self, security: List[int], time: int) -> List[int]:
        '''
        day i is a good day to rob iff
        security[i - time] >= security[i - time + 1] >= ... >= security[i] <= ... <= security[i + time - 1] <= security[i + time]
        return array of all days (0 indexed that are good days to rob the bank)
        basically i is in a valley, brute force would be to check all i and its left and right
        need to store number of days that are increasing up to i and days that decreasing
        shuld be decreasing from left to right
        and increasing from right to left
        count decreasing days up to i from left to right, since that's the first part
        then count increasing days from right to left, the second part
        the the array is decreasing from security[i - time + 1] to security[i]
        then there must be at least time days that are decreasing!
        samv
        left array and right array paradigm
        pref or suff too

        '''
        n = len(security)
        inc_days = [0]*(n)
        dec_days = [0]*(n)

        for i in range(1,n):
            if security[i] <= security[i-1]:
                dec_days[i] = dec_days[i-1] + 1
        
        for i in range(n-2,-1,-1):
            if security[i] <= security[i+1]:
                inc_days[i] = inc_days[i+1] + 1

        ans = []
        for i in range(time,n-time):
            print(i)
            if inc_days[i] >= time and dec_days[i] >= time:
                ans.append(i)
        
        return ans
        
##################################################
# 2531. Make Number of Distinct Characters Equal
# 05FEB25
#################################################
#nice try
class Solution:
    def isItPossible(self, word1: str, word2: str) -> bool:
        '''
        hashmap of counts for both words
        check if equal length
        then try swapping and see if it could equalize them
        need to swap at their positions, not just any position
        '''
        counts1 = Counter(word1)
        counts2 = Counter(word2)

        if len(counts1) == len(counts2):
            return True
        
        for ch in word1:
            counts1[ch] -= 1
            if counts1[ch] == 0:
                del counts1[ch]
            counts2[ch] += 1
            #check
            if len(counts1) == len(counts2):
                return True
            #swap back
            counts2[ch] -= 1
            if counts2[ch] == 0:
                del counts2[ch]
            counts1[ch] += 1
        
        #try the other way
        for ch in word2:
            counts2[ch] -= 1
            if counts2[ch] == 0:
                del counts2[ch]
            counts1[ch] += 1
            if len(counts1) == len(counts2):
                return True
            #swap back
            counts1[ch] -= 1
            if counts1[ch] == 0:
                del counts1[ch]
            counts2[ch] += 1
        
        return len(counts1) == len(counts2)

class Solution:
    def isItPossible(self, word1: str, word2: str) -> bool:
        '''
        hashmap of counts for both words
        check if equal length
        then try swapping and see if it could equalize them
        need to swap at their positions, not just any position
        with a swap we can at most gain 1
        '''
        counts1 = Counter(word1)
        counts2 = Counter(word2)

        unique1 = len(counts1)
        unique2 = len(counts2)

        for ch1,count1 in counts1.items():
            for ch2,count2 in counts2.items():
                unique1 = len(counts1)
                unique2 = len(counts2)
                if count1 == 1 and ch1 != ch2:
                    unique1 -= 1
                if count2 == 1 and ch1 != ch2:
                    unique2 -= 1
                if ch1 not in counts2:
                    unique2 += 1
                if ch2 not in counts1:
                    unique1 += 1
                
                if unique1 == unique2:
                    return True
        
        return False

#mutate on fly
class Solution:
    def isItPossible(self, word1: str, word2: str) -> bool:
        '''
        hashmap of counts for both words
        check if equal length
        then try swapping and see if it could equalize them
        need to swap at their positions, not just any position
        with a swap we can at most gain 1
        '''
        counts1 = Counter(word1)
        counts2 = Counter(word2)

        for i in range(26):
            for j in range(26):
                ch1 = chr(ord('a') + i)
                ch2 = chr(ord('a') + j)
                #they both need to be in for me to swap
                if ch1 not in counts1 or ch2 not in counts2:
                    continue
                #insert
                self.insert_remove(counts1,ch2,ch1)
                self.insert_remove(counts2,ch1,ch2)

                if len(counts1) == len(counts2):
                    return True
                #swap back
                self.insert_remove(counts1,ch1,ch2)
                self.insert_remove(counts2,ch2,ch1)
        
        return False

    def insert_remove(self,mapp,toInsert,toRemove):
        mapp[toInsert] += 1
        mapp[toRemove] -= 1

        if mapp[toRemove] == 0:
            del mapp[toRemove]

