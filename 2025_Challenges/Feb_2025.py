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

#########################################
# 1726. Tuple with Same Product
# 06FEB25
#########################################
class Solution:
    def tupleSameProduct(self, nums: List[int]) -> int:
        '''
        we can fix (a,b)
        and look for another (c,d)
        tuples must be distinct
        i can use a hashmap and mapp tuple pairs (a,b) to some product
        all elements are distinct
        double count then divide by two
        its just going to be (num_digits!) / num_digits
        count permuataions formed
        there are 8 ways to permute (a,b,c,d), such that a*b = c*d
        its that (a,b) and (c,d) must stay next to each other
        we can swap a*b and c*d, and we can swap their order
        2*2*2 = 8
        count the total number of pairs and multiply by 8
        fun counting problem!
        '''
        mapp = Counter()
        n = len(nums)
        for i in range(n):
            for j in range(i+1,n):
                a,b = nums[i],nums[j]
                mapp[a*b] += 1
        
        ans = 0
        for k,v in mapp.items():
            #count must be greater than 1 to have a possible pairing
            if v > 1:
                pairs = (v*(v-1)) // 2 #this is unique pairs
                ans += 8*pairs
        
        return ans
    
##########################################################
# 3160. Find the Number of Distinct Colors Among the Balls
# 06FEB25
###########################################################
class Solution:
    def queryResults(self, limit: int, queries: List[List[int]]) -> List[int]:
        '''
        hashmap of balls to colors, initally balls arent colored so theres nothing in hashmap
        color balls and add size of mapp to ans
        '''
        ans = []
        mapp = {}
        count_colors = Counter()
        for i,color in queries:
            if i not in mapp:
                mapp[i] = color
                count_colors[color] += 1
            else:
                #we made this ball a different color
                prev_color = mapp[i]
                count_colors[prev_color] -= 1
                if count_colors[prev_color] == 0:
                    del count_colors[prev_color]
                mapp[i] = color
                count_colors[color] += 1
            
            ans.append(len(count_colors))
        
        return ans

###############################################
# 3153. Sum of Digit Differences of All Pairs
# 07FEB25
################################################
class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        '''
        all integers in nums have same number of digits
        digit difference is the count of diff digits between two pairs of numbers that are in different at the same position
        '''
        n = len(nums)
        k = len(str(nums[0]))
        counts = [Counter() for _ in range(k)]
        for num in nums:
            for i,digit in enumerate(str(num)):
                counts[i][digit] += 1
        
        ans = 0
        for c in counts:
            for k,v in c.items():
                #count how many diigit different pairs it can make
                #its just (n-v)*v
                #if i have k occreunces of digit at this position, it will make pairs that are different with n-k
                #divide by two for double counting
                ans += v*(n-v) 
        #this would have counted all pairs, but doubled
        return ans //2

##########################################
# 3332. Maximum Points Tourist Can Earn
# 07FEB25
##########################################
#caching TLE'S with actual hasmap
class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        '''
        this is dp
        each city is connected to every other city
        on the ith day, the tourist can
            stay += stayScore[i][curr]
            move from curr to dest
            and gain travel_score[i][curr][dest]
        
        notice how on the digaonlgs, its zero
        '''
        memo = {}

        def dp(curr_day,curr_city):
            if curr_day >= k:
                return 0
            if (curr_day,curr_city) in memo:
                return memo[(curr_day,curr_city)]
            #chose to stay
            ans = stayScore[curr_day][curr_city] + dp(curr_day + 1, curr_city)
            #try traveling
            for neigh_city in range(n):
                travel = travelScore[curr_city][neigh_city] + dp(curr_day + 1, neigh_city)
                ans = max(ans,travel)
            memo[(curr_day,curr_city)] = ans
            return ans
        
        ans = 0
        for i in range(n):
            ans = max(ans, dp(0,i))
        
        return ans

class Solution:
    def maxScore(self, n: int, k: int, stayScore: List[List[int]], travelScore: List[List[int]]) -> int:
        '''
        this is dp
        each city is connected to every other city
        on the ith day, the tourist can
            stay += stayScore[i][curr]
            move from curr to dest
            and gain travel_score[i][curr][dest]
        
        notice how on the digaonlgs, its zero
        '''
        @cache
        def dp(curr_day,curr_city):
            if curr_day >= k:
                return 0
            #chose to stay
            ans = stayScore[curr_day][curr_city] + dp(curr_day + 1, curr_city)
            #try traveling
            for neigh_city in range(n):
                travel = travelScore[curr_city][neigh_city] + dp(curr_day + 1, neigh_city)
                ans = max(ans,travel)
            return ans
        
        ans = 0
        for i in range(n):
            ans = max(ans, dp(0,i))
        
        return ans

##########################################
# 2349. Design a Number Container System
# 08MAY25
###########################################
from sortedcontainers import SortedList
class NumberContainers:

    def __init__(self):
        '''
        two hash maps
        first one mapps index to number 
        second ones mapps number to heap of indices for retrieval
        also need to keep track of index
        the problem is when we change and index already there, we need to remove it from theheap
        need to use sorted container or sorted set for this
        '''
        self.index_to_num = {}
        self.num_to_idxs = defaultdict(SortedList)

    def change(self, index: int, number: int) -> None:
        if index not in self.index_to_num:
            self.index_to_num[index] = number
            self.num_to_idxs[number].add(index)
        else:
            #delete from the other mapp
            prev = self.index_to_num[index]
            #for this prev number remove the current index
            self.num_to_idxs[prev].remove(index)
            self.index_to_num[index] = number
            self.num_to_idxs[number].add(index)

    def find(self, number: int) -> int:
        if len(self.num_to_idxs[number]) == 0:
            return -1
        return self.num_to_idxs[number][0]
        
# Your NumberContainers object will be instantiated and called as such:
# obj = NumberContainers()
# obj.change(index,number)
# param_2 = obj.find(number)

#heap, and update lazily (i.e only when we need to) also called garbage heap too
#deferred handling of index validity during the find operation
#rather than cleaning up after a change

class NumberContainers:
    def __init__(self):
        # Map to store number -> min heap of indices
        self.number_to_indices = defaultdict(list)
        # Map to store index -> number
        self.index_to_numbers = {}

    def change(self, index: int, number: int) -> None:
        # Update index to number mapping
        self.index_to_numbers[index] = number

        # Add index to the min heap for this number
        heapq.heappush(self.number_to_indices[number], index)

    def find(self, number: int) -> int:
        # If number doesn't exist in our map
        if not self.number_to_indices[number]:
            return -1

        # Keep checking top element until we find valid index
        while self.number_to_indices[number]:
            index = self.number_to_indices[number][0]

            # If index still maps to our target number, return it
            if self.index_to_numbers.get(index) == number:
                return index

            # Otherwise remove this stale index
            heapq.heappop(self.number_to_indices[number])
        return -1


# Your NumberContainers object will be instantiated and called as such:
# obj = NumberContainers()
# obj.change(index,number)
# param_2 = obj.find(number)

########################################
# 2364. Count Number of Bad Pairs
# 09FEB24
########################################
class Solution:
    def countBadPairs(self, nums: List[int]) -> int:
        '''
        hashmap problem
        we need j - i != nums[j] - nums[i]
        this is the same as
        nums[j] - j != nums[i] - [i]
        count the good pairs 
        i.e where nums[j] - j == nums[i] - [i]
        then substract from the totla number of pairs
        '''
        counts = Counter()
        n = len(nums)
        good_pairs = 0
        for i,num in enumerate(nums):
            temp = counts[num - i]
            good_pairs += temp
            counts[num - i] += 1
        
        total_pairs = (n*(n-1)) // 2
        return total_pairs - good_pairs
        
##########################################################
# 1375. Number of Times Binary String Is Prefix-Aligned
# 10FEB25
##########################################################
class Solution:
    def numTimesAllBlue(self, flips: List[int]) -> int:
        '''
        yikes there can be 5*10**4 bits
        we can check them all

        pre work
        curr = 0
        for f in flips:
            curr = curr | (1 << (f-1))
            print(bin(curr))

        omg i dont need to do that, just check if we're covered up to the ith flip
        if we are at  flip[i] we need to make sure we are covered from 1 to i
        line sweep?
        note flipes is a permutation of 1 to n [1,n]
        now how to check we cover from 1 to flips[i]
        keep track of the right most bulb
        and if the right most builb == i + 1, we are covered from 1 to rightmost
        '''
        right_most = 0
        count = 0
        for i,num in enumerate(flips):
            right_most = max(right_most, num)
            if right_most == i + 1:
                count += 1
        
        return count
    
class Solution:
    def numTimesAllBlue(self, flips: List[int]) -> int:
        '''
        another way to check if we are covered from 1 to i
        is check the sums from 1 to i,
        keep track of sum and check if current num == i*(i+1) // 2 
        '''
        count = 0
        curr_sum = 0
        for i, num in enumerate(flips):
            curr_sum += num
            i += 1
            if curr_sum == (i*(i+1)) // 2:
                count += 1
        
        return count
    
###############################################
# 1910. Remove All Occurrences of a Substring
# 11FEB25
###############################################
class Solution:
    def removeOccurrences(self, s: str, part: str) -> str:
        '''
        what matters is if i have the final part[:len(part)-1] chars in the stack
        that means i can clear it
        but part can be very long, and checking very time would increaces TC
        i say just use two pointers
        stupid ass problem
        '''
        stack = []
        part = list(part)
        for ch in s:
            stack.append(ch)
            if len(stack)>= len(part) and stack[-len(part):] == part:
                del stack[-len(part):]

        return "".join(stack)

class Solution:
    def removeOccurrences(self, s: str, part: str) -> str:
        '''
        what matters is if i have the final part[:len(part)-1] chars in the stack
        that means i can clear it
        but part can be very long, and checking very time would increaces TC
        i say just use two pointers
        stupid ass problem
        '''
        stack = []
        part = list(part)
        for ch in s:
            stack.append(ch)
            if len(stack)>= len(part) and stack[-len(part):] == part:
                #using pop
                for _ in range(len(part)):
                    stack.pop()


        return "".join(stack)
        
#KMP???

##################################################
# 2342. Max Sum of a Pair With Equal Sum of Digits
# 12FEB24
##################################################
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        '''
        we can choose two indices (i,j), where i != j
        and sum_digits(nums[i]) == sum(digits[j])
        for these we need the maximum_sum
        i.e for all max_sum(nums[i] + nums[j] for all (i,j) where sum_digits(nums[i]) == sum(digits[j]) )

        i can map sum digits for each num in nums, mapp to list of nums
        we only want the largest two, so take the largest
        how many sum digits are possible
        10**5
        9**5, not that many at all
        '''
        mapp = defaultdict(list)
        for num in nums:
            curr_sum = self.sum_digits(num)
            heapq.heappush(mapp[curr_sum],num)
            if len(mapp[curr_sum]) > 2:
                heapq.heappop(mapp[curr_sum])
        
        ans = -1
        for v in mapp.values():
            if len(v) == 2:
                ans = max(ans,sum(v))
        return ans
    
    def sum_digits(self,num):
        sum_ = 0
        while num:
            sum_ += num % 10
            num = num // 10
        
        return sum_

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        '''
        we can choose two indices (i,j), where i != j
        and sum_digits(nums[i]) == sum(digits[j])
        for these we need the maximum_sum
        i.e for all max_sum(nums[i] + nums[j] for all (i,j) where sum_digits(nums[i]) == sum(digits[j]) )

        i can map sum digits for each num in nums, mapp to list of nums
        we only want the largest two, so take the largest
        how many sum digits are possible
        10**5
        999999999, which is sum 81
        '''
        mapp = defaultdict(list)
        for num in nums:
            curr_sum = self.sum_digits(num)
            heapq.heappush(mapp[curr_sum],num)
            if len(mapp[curr_sum]) > 2:
                heapq.heappop(mapp[curr_sum])
        
        ans = -1
        for v in mapp.values():
            if len(v) == 2:
                ans = max(ans,sum(v))
        return ans
    
    def sum_digits(self,num):
        sum_ = 0
        while num:
            sum_ += num % 10
            num = num // 10
        
        return sum_

#on the fly, no precompute
class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        '''
        insteaf of using heap, just store the max
        '''
        mapp = defaultdict(lambda : 0)
        ans = -1
        for num in nums:
            curr_sum = self.sum_digits(num)
            if curr_sum > 0:
                if curr_sum in mapp:
                    ans = max(ans, mapp[curr_sum] + num)
            mapp[curr_sum] = max(mapp[curr_sum],num)
        
        return ans
                
    
    def sum_digits(self,num):
        sum_ = 0
        while num:
            sum_ += num % 10
            num = num // 10
        
        return sum_
