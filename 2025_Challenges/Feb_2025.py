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

##########################################################
# 3066. Minimum Operations to Exceed Threshold Value II
# 13FEB25
###########################################################
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        '''
        heap, 
        invariant conditions it that the min value in heap is smaller than k and that there are at least two in nums
        '''
        heapq.heapify(nums)
        steps = 0

        while len(nums) >= 2 and nums[0] < k:
            first = heapq.heappop(nums)
            second = heapq.heappop(nums)
            next_ = min(first,second)*2 + max(first,second)
            heapq.heappush(nums,next_)
            steps += 1
        
        return steps
    
###########################################
# 1352. Product of the Last K Numbers
# 14FEB25
###########################################
class ProductOfNumbers:
    '''
    does prefix product work here still
    say we have [a,b,c,d]
    [a, a*b, a*b*c, a*b*c*d]
    if we wanted c*d
    a*b*c*d / a*b = c*d
    it will only ever be the last k numbers, a suffix essentially
    if a zero is added, then any product after will be zero, so just clear the array
    '''

    def __init__(self):
        self.pref_prod = [1]

    def add(self, num: int) -> None:
        if num == 0:
            self.pref_prod = [0]*len(self.pref_prod)
            self.pref_prod.append(1)
        else:
            self.pref_prod.append(self.pref_prod[-1]*num)

    def getProduct(self, k: int) -> int:
        if self.pref_prod[-(k+1)] == 0:
            return 0
        return self.pref_prod[-1] // self.pref_prod[-(k+1)]
        

# Your ProductOfNumbers object will be instantiated and called as such:
# obj = ProductOfNumbers()
# obj.add(num)
# param_2 = obj.getProduct(k)

class ProductOfNumbers:

    def __init__(self):
        '''
        another way instead of erraing the pref product array is to keep a size variable
        when we hit a zero, we reset size to 0, the last k eleements' products will be zero anyway
        '''
        self.pref_prod = [1]
        self.size = 0

    def add(self, num: int) -> None:
        #if zero, reset
        if num == 0:
            self.pref_prod = [1]
            self.size = 0
        else:
            self.pref_prod.append(self.pref_prod[self.size]*num)
            self.size += 1
        
    def getProduct(self, k: int) -> int:
        #check if k from the end i larger
        if k > self.size:
            return 0
        return self.pref_prod[-1] // self.pref_prod[self.size - k]


# Your ProductOfNumbers object will be instantiated and called as such:
# obj = ProductOfNumbers()
# obj.add(num)
# param_2 = obj.getProduct(k)

###############################################
# 2698. Find the Punishment Number of an Integer
# 15FEB25
################################################
class Solution:
    def punishmentNumber(self, n: int) -> int:
        '''
        punishment number is defined as the sum of the squals of all integers i such that:
        * 1 <= i <= n
        * the decimal representatino of i*i can be partitioned into contiguous substrings
            such that the sum of the integer values of these substrings equals i
        
        looks like we can generate all paritions of a number and check the sum using backtracking
        if split at i, i need to take that numbers left part and add it
        if i don't just adance it
        states are index and curr_sum
        '''
        ans = 0
        for i in range(1,n+1):
            curr = i*i
            #check parititions of curr recursively
            memo = {}
            if self.rec(str(curr),0,0,i,memo):
                ans += curr
        return ans
    
    def rec(self,num,idx,curr_sum,target,memo):
        if idx >= len(num):
            return curr_sum == target
        
        if curr_sum > target:
            return False
        
        if (idx,curr_sum) in memo:
            return memo[(idx,curr_sum)]
        
        ans = False
        #try all paritions
        for next_idx in range(idx,len(num) + 1):
            #split
            split_num = int(num[idx:next_idx+1])
            split = self.rec(num,next_idx + 1, curr_sum + split_num,target,memo)
            ans = ans or split
        
        memo[(idx,curr_sum)] = ans
        return ans

##################################################################
# 1718. Construct the Lexicographically Largest Valid Sequence
# 16FEB25
###################################################################
#TLE, right idea, but need to add terminating
class Solution:
    def constructDistancedSequence(self, n: int) -> List[int]:
        '''
        1 occrus once in the sequence,
        every integer between 2 and n occurs twice int he sequence
        for every integer between 2 and n, the dist between the two occurences of i ix exactly i
        rather, if k is in between 2 and n, the distance between the two occurences of k is k
        if n, ans will be of size 2*n + 1
        try placing each number from 1 to n in the array if we can't backtrack
        assumes we have ans = [0]*(2*n + 1)
        if i'm at some index i, and placing some number num, that isn't a 1
        then i need to place num at ans[i] and at ans[i + num - 1]
        what i don't worry about placing a one, i can just place it in the last spot,
        then i can just worry about placing the numbers [2,n]
        need to try all permuations of 2 to n
        '''
        ans = [0]*(2*n - 1)
        final_ans = [-1]*(2*n-1)
        used_nums = set()
        self.backtrack(ans,0,n,used_nums,final_ans)
        #find the zero
        for i in range(len(final_ans)):
            if final_ans[i] == 0:
                final_ans[i] = 1
                break
        return final_ans

    def backtrack(self,arr,curr_idx,n,used_nums,final_ans):
        if curr_idx >= len(arr):
            if len(used_nums) == n - 1:
                final_ans[:] = max(final_ans,arr[:])

            return
        #want largest
        for num in range(n,1,-1):
            #place it
            if num not in used_nums and curr_idx + num < len(arr) and arr[curr_idx] == 0 and arr[curr_idx + num] == 0:
                arr[curr_idx] = num
                arr[curr_idx + num] = num
                used_nums.add(num)
                #recurse
                self.backtrack(arr,curr_idx+1,n,used_nums,final_ans)
                arr[curr_idx] = 0
                arr[curr_idx + num] = 0
                used_nums.remove(num)
            else:
                self.backtrack(arr,curr_idx + 1,n,used_nums,final_ans)
            
class Solution:
    def constructDistancedSequence(self, n: int) -> List[int]:
        '''
        1 occrus once in the sequence,
        every integer between 2 and n occurs twice int he sequence
        for every integer between 2 and n, the dist between the two occurences of i ix exactly i
        rather, if k is in between 2 and n, the distance between the two occurences of k is k
        if n, ans will be of size 2*n + 1
        try placing each number from 1 to n in the array if we can't backtrack
        assumes we have ans = [0]*(2*n + 1)
        if i'm at some index i, and placing some number num, that isn't a 1
        then i need to place num at ans[i] and at ans[i + num - 1]
        what i don't worry about placing a one, i can just place it in the last spot,
        then i can just worry about placing the numbers [2,n]
        need to try all permuations of 2 to n
        '''
        ans = [-1]*(2*n - 1)
        used_nums = set()
        self.backtrack(ans,0,n,used_nums)
        return ans


    def backtrack(self,arr,curr_idx,n,used_nums):
        #if we get to the end of the array
        if curr_idx == len(arr):
            return True
        #if alread filled, advance
        if arr[curr_idx] != -1:
            return self.backtrack(arr,curr_idx + 1, n,used_nums)
        for num in range(n,0,-1):
            #not used
            if num not in used_nums:
                #if its 1, just place
                if num == 1:
                    arr[curr_idx] = 1
                    used_nums.add(1)
                    if self.backtrack(arr,curr_idx+1,n,used_nums):
                        return True
                    arr[curr_idx] = -1
                    used_nums.remove(1)
                elif curr_idx + num < len(arr) and arr[curr_idx + num] == -1:
                    arr[curr_idx] = num
                    arr[curr_idx + num] = num
                    used_nums.add(num)
                    if self.backtrack(arr,curr_idx+1,n,used_nums):
                        return True
                    arr[curr_idx + num] = -1
                    arr[curr_idx] = -1
                    used_nums.remove(num)

        return False

########################################
# 1756. Design Most Recently Used Queue
# 16FEB25
########################################
class MRUQueue:

    def __init__(self, n: int):
        self.arr = [i for i in range(1,n+1)]

    def fetch(self, k: int) -> int:
        ans = self.arr.pop(k-1)
        self.arr.append(ans)
        return ans

# Your MRUQueue object will be instantiated and called as such:
# obj = MRUQueue(n)
# param_1 = obj.fetch(k)

##################################
# 1079. Letter Tile Possibilities
# 17FEB25
##################################
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        '''
        should be unique possible sequences
        use backtracking and dump into a set, choose to add this character or don't
        '''
        possible = set()
        n = len(tiles)
        used = [False]*n
        self.rec(0,tiles,used,[],possible)
        return len(possible)
    
    def rec(self,i,tiles,used,path,possible):
        if path:
            possible.add("".join(path))
        for j in range(len(tiles)):
            if not used[j]:
                used[j] = True
                path = path + [tiles[j]]
                self.rec(i+1,tiles,used,path,possible)
                path.pop()
                used[j] = False

#note, is using string concat, and with string, we don't need to pop?
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        '''
        should be unique possible sequences
        use backtracking and dump into a set, choose to add this character or don't
        '''
        possible = set()
        n = len(tiles)
        used = [False]*n
        self.rec(0,tiles,used,"",possible)
        return len(possible)
    
    def rec(self,i,tiles,used,path,possible):
        if path:
            possible.add("".join(path))
        for j in range(len(tiles)):
            if not used[j]:
                used[j] = True
                self.rec(i+1,tiles,used,path + tiles[j],possible)
                used[j] = False
    
#backtracking on counts
class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        '''
        we can do backtracking on counts
        count of the chars in tiles
        then try adding a letter if we can, then answer is just the sum of the leaves
        '''
        counts = Counter(tiles)
        return self.rec(counts)

    def rec(self,counts):
        count = 0
        for i in range(26):
            letter = chr(ord('A') + i)
            if counts[letter] > 0:
                counts[letter] -= 1
                #addind a valid letter means this is a valid sequence
                count += 1
                #roll up the counts
                count += self.rec(counts)
                counts[letter] += 1
        return count
    
##################################################
# 2375. Construct Smallest Number From DI String
# 18FEB25
##################################################
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        '''
        the constraints are small, so we can try all of them
        welp i pruned LMAOOO
        '''
        used = set()
        self.ans = "9"*(len(pattern) + 1)
        self.found = False
        self.backtrack("",used,pattern)
        return self.ans
        
    def backtrack(self,path,used,pattern):
        if self.found:
            return
        if len(path) == len(pattern) + 1:
            if self.check(path,pattern):
                self.ans = min(self.ans,path)
                self.found = True
            return
        for i in range(1,len(pattern) + 2):
            if i not in used:
                used.add(i)
                self.backtrack(path + str(i),used,pattern)
                used.remove(i)
    
    def check(self,candidate,pattern):
        for i in range(len(pattern)):
            if pattern[i] == 'I' and candidate[i+1] < candidate[i]:
                return False
            elif pattern[i] == 'D' and candidate[i+1] > candidate[i]:
                return False
        return True
    
#optimized backtracking
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        return str(self.find_smallest_number(pattern, 0, set(), 0))

    # Recursively find the smallest number that satisfies the pattern
    def find_smallest_number(
        self,
        pattern: str,
        current_position: int,
        used_nums: int,
        current_num: int,
    ) -> int:
        # Base case: return the current number when the whole pattern is processed
        if current_position > len(pattern):
            return current_num

        result = float("inf")
        last_digit = current_num % 10

        #thie part is important, otherwise we need to code for the decrement part sepeartely
        should_increment = (
            current_position == 0 or pattern[current_position - 1] == "I"
        )

        # Try all possible digits (1 to 9) that are not yet used and follow the pattern
        for current_digit in range(1, 10):
            if current_digit not in used_nums and (current_digit > last_digit) == should_increment:
                used_nums.add(current_digit)
                result = min(
                    result,
                    self.find_smallest_number(
                        pattern,
                        current_position + 1,
                        used_nums,
                        current_num * 10 + current_digit,
                    ),
                )
                used_nums.remove(current_digit)

        return result
    
#another way but coding out rules explicitly for I and D
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        return str(self.find_smallest_number(pattern, 0, set(), 0))

    def find_smallest_number(
        self,
        pattern: str,
        current_position: int,
        used_nums: int,
        current_num: int,
    ) -> int:
        # Base case: return the current number when the whole pattern is processed
        if current_position > len(pattern):
            return current_num

        result = float("inf")
        last_digit = current_num % 10

        for current_digit in range(1, 10):
            if current_digit not in used_nums:
                if (current_digit > last_digit and pattern[current_position - 1] == 'I') or \
                (current_digit < last_digit and pattern[current_position-1] == 'D') or \
                (current_position == 0):
                    used_nums.add(current_digit)
                    result = min(
                        result,
                        self.find_smallest_number(
                            pattern,
                            current_position + 1,
                            used_nums,
                            current_num * 10 + current_digit,
                        ),
                    )
                    used_nums.remove(current_digit)

        return result
    
#instead of backtracking, just pass in the mask
#this only works because the mask states are small
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        return str(self.find_smallest_number(pattern, 0, 0, 0))

    def find_smallest_number(
        self,
        pattern: str,
        current_position: int,
        used_nums: int,
        current_num: int,
    ) -> int:
        # Base case: return the current number when the whole pattern is processed
        if current_position > len(pattern):
            return current_num

        result = float("inf")
        last_digit = current_num % 10

        for current_digit in range(1, 10):
            if used_nums & (1 << current_digit) == 0:
                if (current_digit > last_digit and pattern[current_position - 1] == 'I') or \
                (current_digit < last_digit and pattern[current_position-1] == 'D') or \
                (current_position == 0):
                    result = min(
                        result,
                        self.find_smallest_number(
                            pattern,
                            current_position + 1,
                            used_nums | (1 << current_digit),
                            current_num * 10 + current_digit,
                        ),
                    )
        return result
    
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        '''
        if we have III, then its just 1234
        the problem is when we hit a D, we dont know how many D's happen after it
        instead of placing a number at the position, we delay it and go to the next
        we keep going until we hit an I or the end of the pattern
        keep track of the current number of positinos we assign to a digit
        if its I we call rec(i+1,count + 1)
        if its D, we call rec(u_1,count)
        '''
        ans = []
        self.rec(0,0,pattern,ans)
        return "".join(ans[::-1])
    
    def rec(self,i : int, curr_count : int, pattern : str, ans : List[int]):
        if i != len(pattern):
            if pattern[i] == 'I':
                self.rec(i+1,curr_count+1,pattern,ans)
            else:
                curr_count = self.rec(i+1,curr_count,pattern,ans)

        ans.append(str(curr_count + 1))
        return curr_count + 1
    
#now we can do stack implementation
#bascially after D, we need to reverse 
class Solution:
    def smallestNumber(self, pattern: str) -> str:
        '''
        stack implementation
        starting at 1, add to stack 
        if we hit the end of the pattern or another 'I' pop the number from the stack and add to ans
        '''
        stack = []
        ans = []
        curr_count = 0

        for i in range(len(pattern) + 1):
            stack.append(curr_count + 1)
            if i == len(pattern) or pattern[i] == 'I':
                while stack:
                    ans.append(str(stack.pop()))
            curr_count += 1
        
        return "".join(ans)

#########################################################################
# 1415. The k-th Lexicographical String of All Happy Strings of Length n
# 19FEB25
##########################################################################
#ezzzz
class Solution:
    def getHappyString(self, n: int, k: int) -> str:
        '''
        will brute force work
        if there are 10 spots, and each spot can take, a letter abc
        then there are less than 3**10
        i can generate all in order and return the kth one
        '''
        possible = []
        self.rec(n,"",possible)
        if k - 1 >= len(possible):
            return ""
        
        return possible[k-1]
    def rec(self,n,path,possible):
        if n == 0:
            possible.append(path)
            return
        for letter in 'abc':
            if not path or (path and path[-1] != letter):
                self.rec(n-1,path+letter,possible)

#rudimentary pruning
class Solution:
    def getHappyString(self, n: int, k: int) -> str:
        '''
        will brute force work
        if there are 10 spots, and each spot can take, a letter abc
        then there are less than 3**10
        i can generate all in order and return the kth one
        '''
        possible = []
        self.rec(n,"",possible,k)
        if k - 1 >= len(possible):
            return ""
        
        return possible[k-1]
    def rec(self,n,path,possible,k):
        if n == 0:
            possible.append(path)
            return
        if len(possible) == k:
            return
        for letter in 'abc':
            if not path or (path and path[-1] != letter):
                self.rec(n-1,path+letter,possible,k)

#######################################
# 1980. Find Unique Binary String
# 20FEB25
########################################
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        uh wtf, just check all strings from 1 to 2**n
        and make sure they're atleast of length n
        '''
        nums = set(nums)
        n = len(nums)
        for i in range(2**len(nums)):
            cand = self.get_bin(i,n)
            if cand not in nums:
                return cand
        
        return ""
    
    def get_bin(self,num,n):
        ans = []
        while num:
            ans.append(str(num & 1))
            num = num >> 1
        return '0'*(n-len(ans))+"".join(ans)

#recursion
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        nums = set(nums)
        n = len(nums)
        for cand in self.generate("",n):
            if cand not in nums:
                return cand
    
    def generate(self,path,n):
        if len(path) == n:
            yield path
        else:
            for num in ['0','1']:
                yield from self.generate(path+num,n)

#returning from
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        nums = set(nums)
        n = len(nums)
        return self.generate("",n,nums)

    def generate(self,path,n,nums):
        if len(path) == n:
            if path not in nums:
                return path
            return None
        ans = None
        for num in ['0','1']:
            ans = self.generate(path+num,n,nums)
            if ans:
                return ans
        return ans

class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        evidently this is just cantor's digonal argument, in fact the example is Cantor's!
        we need need to generate a binary string from nums
        we flip each digit in nums
        '''
        ans = []
        n = len(nums)
        for i in range(n):
            digit = int(nums[i][i])
            ans.append(str(1 - digit))
        
        return "".join(ans)

####################################################
# 1261. Find Elements in a Contaminated Binary Tree
# 21FEB25
###################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class FindElements:

    def __init__(self, root: Optional[TreeNode]):
        '''
        we can modify the tree in the constructor, as we decend the tree keep track of values in hashset
        '''
        self.values = set()
        def dfs(node,new_val):
            if not node:
                return
            self.values.add(new_val)
            node.val = new_val
            if node.left:
                dfs(node.left, new_val*2 + 1)
            if node.right:
                dfs(node.right, new_val*2 + 2)
        dfs(root,0)

    def find(self, target: int) -> bool:
        return target in self.values
        


# Your FindElements object will be instantiated and called as such:
# obj = FindElements(root)
# param_1 = obj.find(target)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class FindElements:

    def __init__(self, root: Optional[TreeNode]):
        '''
        we can also do BFS
        '''
        self.values = set()
        self.bfs(root)

    def find(self, target: int) -> bool:
        return target in self.values

    def bfs(self,root):
        q = deque([(root,0)])
        while q:
            node,curr_val = q.popleft()
            self.values.add(curr_val)
            if node.left:
                q.append((node.left,curr_val*2 + 1))
            if node.right:
                q.append((node.right,curr_val*2 + 2))


# Your FindElements object will be instantiated and called as such:
# obj = FindElements(root)
# param_1 = obj.find(target)

###############################################
# 1028. Recover a Tree From Preorder Traversal
# 22FEB25
##############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
        '''
        after each number we have D dashes where D is the depth of the node,
        if a node is at depth D, then its left child will be D + 1
        if a node has a child
        example:  "1-2--3--4-5--6--7"
        we can read as
        depth 0: [1]
        depth 1: [2.5]
        depth 2: [3,4,6,7]
         "1-2--3---4-5--6---7"
        depth 0: [1]
        depth 1: [2,5]
        depth 2: [3,6]
        depth 3: [4,7]
        '''
        #parse string and get levels
        parsed = []
        i = 0
        curr_num = 0
        curr_depth = 0
        while i < len(traversal):
            while i < len(traversal) and traversal[i].isdigit():
                curr_num = curr_num*10 + int(traversal[i])
                i += 1
            parsed.append((curr_num,curr_depth))
            curr_num = 0
            curr_depth = 0
            while i < len(traversal) and traversal[i] == '-':
                curr_depth += 1
                i += 1
        
        stack = []
        i = 0
        while i < len(parsed):
            curr_num,depth = parsed[i]
            #make node
            node = TreeNode(curr_num)
            while len(stack) > depth:
                stack.pop()
            if stack:
                if not stack[-1].left:
                    stack[-1].left = node
                else:
                    stack[-1].right = node
            stack.append(node)
            i += 1
        
        return stack[0]
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverFromPreorder(self, traversal: str) -> Optional[TreeNode]:
        '''
        after each number we have D dashes where D is the depth of the node,
        if a node is at depth D, then its left child will be D + 1
        if a node has a child
        example:  "1-2--3--4-5--6--7"
        we can read as
        depth 0: [1]
        depth 1: [2.5]
        depth 2: [3,4,6,7]
         "1-2--3---4-5--6---7"
        depth 0: [1]
        depth 1: [2,5]
        depth 2: [3,6]
        depth 3: [4,7]
        '''
        #parse string and get levels
        parsed = []
        i = 0
        curr_num = 0
        curr_depth = 0
        while i < len(traversal):
            while i < len(traversal) and traversal[i].isdigit():
                curr_num = curr_num*10 + int(traversal[i])
                i += 1
            parsed.append((curr_num,curr_depth))
            curr_num = 0
            curr_depth = 0
            while i < len(traversal) and traversal[i] == '-':
                curr_depth += 1
                i += 1
        idx = [0]
        return self.build(parsed,idx,0)
    
    def build(self, arr, idx,depth):
        if idx[0] >= len(arr):
            return None
        val,curr_depth = arr[idx[0]]
        if curr_depth != depth:
            return None
        node = TreeNode(val)
        idx[0] += 1
        node.left = self.build(arr,idx,depth+1)
        node.right = self.build(arr,idx,depth+1)
        return node

###################################################################
# 889. Construct Binary Tree from Preorder and Postorder Traversal
# 24FEB25
####################################################################
class Solution:
    def constructFromPrePost(
        self, preorder: List[int], postorder: List[int]
    ) -> Optional[TreeNode]:
        num_of_nodes = len(preorder)

        # Create the index list for `postorder`
        index_in_post_order = [0] * (num_of_nodes + 1)
        for index in range(num_of_nodes):
            # Store the index of the current element
            index_in_post_order[postorder[index]] = index

        return self._construct_tree(0, num_of_nodes - 1, 0, preorder, index_in_post_order)

    # Helper function to construct the tree recursively
    def _construct_tree(
        self,
        pre_start: int,
        pre_end: int,
        post_start: int,
        preorder: List[int],
        index_in_post_order: List[int],
    ) -> Optional[TreeNode]:
        # Base case: If there are no nodes to process, return None
        if pre_start > pre_end:
            return None

        # Base case: If only one node is left, return that node
        if pre_start == pre_end:
            return TreeNode(preorder[pre_start])

        # The left child root in preorder traversal (next element after root)
        left_root = preorder[pre_start + 1]

        # Calculate the number of nodes in the left subtree by searching in postorder
        num_of_nodes_in_left = index_in_post_order[left_root] - post_start + 1

        root = TreeNode(preorder[pre_start])

        # Recursively construct the left subtree
        root.left = self._construct_tree(
            pre_start + 1,
            pre_start + num_of_nodes_in_left,
            post_start,
            preorder,
            index_in_post_order,
        )

        # Recursively construct the right subtree
        root.right = self._construct_tree(
            pre_start + num_of_nodes_in_left + 1,
            pre_end,
            post_start + num_of_nodes_in_left,
            preorder,
            index_in_post_order,
        )

        return root
    
#########################################
# 2467. Most Profitable Path in a Tree
# 24FEB25
##########################################
class Solution:
    def mostProfitablePath(self, edges: List[List[int]], bob: int, amount: List[int]) -> int:
        '''
        i don't see how bob's path is fixed
            because its a tree! my fucking god, there can only be one path from bob to zero
            if we have the path, then we have the times
        the problem is tha they move at the same time....
        i can use bfs to find the nodes bob would touch, at each time point, then check if alice reaches that node at that same time point
        use this to check if they reach the node at the same time
        as we dfs for alice, we need to check bob times to see if:
            they arrived at the same time
            of if the gate has already been opened up
        '''
        graph = defaultdict(list)
        #need to know what nodes are leaves
        indegree = Counter()
        for u,v in edges:
            indegree[u] += 1
            indegree[v] += 1
            graph[u].append(v)
            graph[v].append(u)
        
        leaves = set()
        for node in indegree:
            if node != 0 and indegree[node] == 1:
                leaves.add(node)
        
        bob_path = [bob]
        bob_ans = []
        bob_seen = set()
        bob_seen.add(bob)
        self.bob_moves(graph,bob,bob_path,bob_seen,bob_ans)
        #need to mapp nodes to times, then bfs
        times = {}
        for i,node in enumerate(bob_ans):
            times[node] = i

        #bfs for alice to find nodes
        max_income = float('-inf')
        seen = set()
        q = deque([(0,0,0)]) #entries are (alice_node,curr_time,curr_income)
        
        #need to make sure we maximize only on leaves
        while q:
            alice,curr_time,curr_income = q.popleft()
            seen.add(alice)
            if (alice not in times or curr_time < times[alice]):
                curr_income += amount[alice]
            #same time
            elif curr_time == times[alice]:
                curr_income += amount[alice] // 2
            #leaf update
            if alice in leaves:
                max_income = max(max_income,curr_income)
            for neigh in graph[alice]:
                if neigh not in seen:
                    q.append((neigh,curr_time + 1, curr_income))
        
        return max_income

    #find path for bob
    def bob_moves(self,graph,bob,path,seen,bob_ans):
        if bob == 0:
            bob_ans[:] = path[:]
            return
        for neigh in graph[bob]:
            if neigh not in seen:
                path.append(neigh)
                seen.add(neigh)
                self.bob_moves(graph,neigh,path,seen,bob_ans)
                path.pop()
                seen.remove(neigh)

#instead of bfs for alice, we can do dfs
class Solution:
    def mostProfitablePath(self, edges: List[List[int]], bob: int, amount: List[int]) -> int:
        '''
        i don't see how bob's path is fixed
            because its a tree! my fucking god, there can only be one path from bob to zero
            if we have the path, then we have the times
        the problem is tha they move at the same time....
        i can use bfs to find the nodes bob would touch, at each time point, then check if alice reaches that node at that same time point
        use this to check if they reach the node at the same time
        as we dfs for alice, we need to check bob times to see if:
            they arrived at the same time
            of if the gate has already been opened up
        '''
        graph = defaultdict(list)
        #need to know what nodes are leaves
        indegree = Counter()
        for u,v in edges:
            indegree[u] += 1
            indegree[v] += 1
            graph[u].append(v)
            graph[v].append(u)
        
        leaves = set()
        for node in indegree:
            if node != 0 and indegree[node] == 1:
                leaves.add(node)
        
        bob_path = [bob]
        bob_ans = []
        bob_seen = set()
        bob_seen.add(bob)
        self.bob_moves(graph,bob,bob_path,bob_seen,bob_ans)
        #need to mapp nodes to times, then bfs
        times = {}
        for i,node in enumerate(bob_ans):
            times[node] = i

        #we can wrap this is a dfs function
        max_income = [float('-inf')]
        seen = set()
        self.alice_path(0,0,0,max_income,seen,leaves,times,amount,graph)
        return max_income[0]
    
    def alice_path(self,alice,curr_time,curr_income,max_income,seen,leaves,times,amount,graph):
        seen.add(alice)
        if (alice not in times or curr_time < times[alice]):
            curr_income += amount[alice]
        #same time
        elif curr_time == times[alice]:
            curr_income += amount[alice] // 2
        #leaf update
        if alice in leaves:
            max_income[0] = max(max_income[0],curr_income)
        for neigh in graph[alice]:
            if neigh not in seen:
                self.alice_path(neigh,curr_time + 1, curr_income,max_income,seen,leaves,times,amount,graph)

    #find path for bob
    def bob_moves(self,graph,bob,path,seen,bob_ans):
        if bob == 0:
            bob_ans[:] = path[:]
            return
        for neigh in graph[bob]:
            if neigh not in seen:
                path.append(neigh)
                seen.add(neigh)
                self.bob_moves(graph,neigh,path,seen,bob_ans)
                path.pop()
                seen.remove(neigh)

###########################################
# 1524. Number of Sub-arrays With Odd Sum
# 25FEB25
###########################################
class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        '''
        i can use pref_sum and check all sums for each (i,j) array
        if we have a pref_sum that is odd, we contribute odd_count subarrays

        '''
        mod = 10**9 + 7
        ans = 0
        pref_sum = 0
        even_count = 1
        odd_count = 0

        for num in arr:
            pref_sum += num
            #if its even, add the odd count and increment even
            if pref_sum % 2 == 0:
                ans += odd_count
                even_count += 1
            else:
                ans += even_count
                odd_count += 1

            ans %= mod
        
        return ans
    
class Solution:
    def numOfSubarrays(self, arr: List[int]) -> int:
        '''
        if we knew the count of subarrays who's sum is odd ending at i, then we just add them all up
        let dp(i) be the number of subarrays who's sum is odd and ends at index i
        if a current subarry ending at i has odd sum, them odding another odd would make it even,
            but adding an even number, would make it even
        if a current subarray ending at i had even num, then adding an odd number would make it odd
            but adding an even number would keep it even
        
        so we only need to keep track of how many subaraayrs up to a given index i have odd and even sums
        '''
        mod = 10**9 + 7
        n = len(arr)

        ending_odd = [0]*n
        ending_even = [0]*n

        #init, the last element parities
        if arr[-1] % 2 == 1:
            ending_odd[-1] = 1
        else:
            ending_even[-1] == 1

        for i in range(n-2,-1,-1):
            #if its odd
            if arr[i] % 2 == 1:
                #then we can at more odd counts from even
                ending_odd[i] = (1 + ending_even[i+1]) % mod
                ending_even[i] = (ending_odd[i+1]) % mod
            else:
                ending_even[i] = (1 + ending_even[i+1] % mod)
                ending_odd[i] = (ending_odd[i+1]) % mod
        
        return sum(ending_odd) % mod

#############################################
# 1749. Maximum Absolute Sum of Any Subarray
# 26FEB25
#############################################
class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        '''
        if all were positive, it would just be the same of the whole array
        compare with maximum sum subarray with minimum sum subarray
        '''
        if len(nums) == 1:
            return abs(nums[0])
        #find max
        current_subarray = max_subarray = nums[0]

        for num in nums[1:]:
            current_subarray = max(num, current_subarray + num)
            max_subarray = max(max_subarray, current_subarray)
        
        #find min
        current_subarray  = nums[0]
        min_subarray = float('inf')

        for num in nums[1:]:
            current_subarray = min(num, current_subarray + num)
            min_subarray = min(min_subarray, current_subarray)
        
        return max(abs(max_subarray),abs(min_subarray))

class Solution:
    def maxAbsoluteSum(self, nums: List[int]) -> int:
        '''
        if all were positive, it would just be the same of the whole array
        compare with maximum sum subarray with minimum sum subarray
        '''
        max_sub = self.kadane_max(nums)
        min_sub = self.kadane_min(nums)
        return max(max_sub,min_sub)
    
    def kadane_max(self,nums):
        n = len(nums)
        dp = [float('-inf')]*n
        dp[0] = nums[0]
        for i in range(1,n):
            dp[i] = max(dp[i-1] + nums[i],nums[i])
        
        return abs(max(dp))
    
    def kadane_min(self,nums):
        n = len(nums)
        dp = [float('inf')]*n
        dp[0] = nums[0]
        for i in range(1,n):
            dp[i] = min(dp[i-1] + nums[i],nums[i])
        
        return abs(min(dp))

############################################
# 873. Length of Longest Fibonacci Subsequence
# 27FEB25
#############################################
#TLE with memo
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        '''
        similar to LIS, but need to followe the fibonacci sequence
        or convultion filter maybe? bleaghhh
        need to keep track of index, and last two numbers in the sequence
        if we have the last two numbers, we can generate the next
        '''
        n = len(arr)
        memo = {}

        def dp(i,first,second):
            if i >= n:
                return 0
            if (i,first,second) in memo:
                return memo[(i,first,second)]
            
            ans = 2
            for j in range(i+1,n):
                if arr[j] == first + second:
                    ans = max(ans, 1 + dp(j,second,arr[j]))
            
            memo[(i,first,second)] = ans
            return ans

        ans = 0
        for i in range(n):
            for j in range(i+1,n):
                ans = max(ans,dp(j,arr[i],arr[j]))
        
        if ans == 2:
            return 0
        return ans
    
#still TLE with cache
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        '''
        similar to LIS, but need to followe the fibonacci sequence
        or convultion filter maybe? bleaghhh
        need to keep track of index, and last two numbers in the sequence
        if we have the last two numbers, we can generate the next
        '''
        n = len(arr)
        @lru_cache
        def dp(i,first,second):
            if i >= n:
                return 0
            
            ans = 2
            for j in range(i+1,n):
                if arr[j] == first + second:
                    ans = max(ans, 1 + dp(j,second,arr[j]))
            
            return ans

        ans = 0
        for i in range(n):
            for j in range(i+1,n):
                ans = max(ans,dp(j,arr[i],arr[j]))
        
        if ans == 2:
            return 0
        return ans
    
#binary search but MLE
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        '''
        similar to LIS, but need to followe the fibonacci sequence
        or convultion filter maybe? bleaghhh
        need to keep track of index, and last two numbers in the sequence
        if we have the last two numbers, we can generate the next

        if they're stricclyt increasing we can use binary search
        '''
        n = len(arr)

        @cache
        def dp(i,first,second):
            if i >= n:
                return 0
            ans = 2
            #binary search
            j = bisect.bisect_left(arr,first+second)
            if j < len(arr) and arr[j] == first + second:
                ans = max(ans, 1 + dp(j,second,arr[j]))

            return ans

        ans = 0
        for i in range(n):
            for j in range(i+1,n):
                ans = max(ans,dp(j,arr[i],arr[j]))
        
        if ans == 2:
            return 0
        return ans
    
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        '''
        similar to LIS, but need to followe the fibonacci sequence
        or convultion filter maybe? bleaghhh
        need to keep track of index, and last two numbers in the sequence
        if we have the last two numbers, we can generate the next

        if they're stricclyt increasing we can use binary search
        we don't need to fix i and j starts
        if we start with arr[i], then we can just check if arr[i] + some_next_num where this i in ther range(arr[i])

        there's too much overhead with hashmap as memo
        need to use arrays
         '''
        d = {val: i for i, val in enumerate(arr)}
        n = len(arr)
        memo = [[-1]*n for _ in range(n)]
        def dp(i, j):
            if memo[i][j] != -1:
                return memo[i][j]
            s = arr[i] + arr[j]
            if s in d:
                memo[i][j] = 1 + dp(j, d[s])
            else:
                memo[i][j] = 0
            return memo[i][j]
        
        ans = 0
        for i in range(len(arr) - 2):
            for j in range(i + 1, len(arr) - 1):
                ans = max(ans, dp(i,j))
        return ans + 2 if ans else 0

class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        '''
        need to use arrays as memo
        '''
        d = {val: i for i, val in enumerate(arr)}
        n = len(arr)
        memo = [[-1]*n for _ in range(n)]
        def dp(i, j):
            if memo[i][j] != -1:
                return memo[i][j]
            s = arr[i] + arr[j]
            if s in d:
                memo[i][j] = 1 + dp(j, d[s])
            else:
                memo[i][j] = 0
            return memo[i][j]
        
        ans = 0
        for i in range(len(arr) - 2):
            for j in range(i + 1, len(arr) - 1):
                local_ans = dp(i,j)
                if local_ans:
                    ans = max(ans,local_ans + 2)
        if ans:
            return ans
        return 0

#brute force actually works just find
class Solution:
    def lenLongestFibSubseq(self, arr: List[int]) -> int:
        '''
        we can just intelligent build a fib sequence,
        fix first and second numbers and keep advancing
        use hashset for 
        '''
        arr_set = set(arr)
        ans = 0
        n = len(arr)

        for i in range(n):
            for j in range(i+1,n):
                second = arr[j]
                third = arr[i] + arr[j]
                curr_length = 2
                while third in arr_set:
                    second,third = third, second + third
                    curr_length += 1
                    ans = max(ans,curr_length)
        
        return ans
    
#we can use dp
class Solution:
    def lenLongestFibSubseq(self, arr: list[int]) -> int:
        n = len(arr)
        max_len = 0
        # dp[prev][curr] stores length of Fibonacci sequence ending at indexes prev,curr
        dp = [[0] * n for _ in range(n)]

        # Map each value to its index for O(1) lookup
        val_to_idx = {num: idx for idx, num in enumerate(arr)}

        # Fill dp array
        for curr in range(n):
            for prev in range(curr):
                # Find if there exists a previous number to form Fibonacci sequence
                diff = arr[curr] - arr[prev]
                prev_idx = val_to_idx.get(diff, -1)

                # Update dp if valid Fibonacci sequence possible
                # diff < arr[prev] ensures strictly increasing sequence
                dp[prev][curr] = (
                    dp[prev_idx][prev] + 1
                    if diff < arr[prev] and prev_idx >= 0
                    else 2
                )
                max_len = max(max_len, dp[prev][curr])

        # Return 0 if no sequence of length > 2 found
        return max_len if max_len > 2 else 0


###########################################
# 1092. Shortest Common Supersequence 
# 28FEB25
############################################
#bleaghhh
class Solution:
    def shortestCommonSupersequence(self, str1: str, str2: str) -> str:
        '''
        if we were to concat str1+str2 and str2+str1
        then we just need to find the shortest sequence in each of them that has both
        'abac' and 'cab'
        try 'abaccab', we have to use the whole thing
        try 'cababac'
        then we can try looking for subsequence starting at i
        and try generating a sequence while we have chars to take
        '''
        concat1 = str1+str2
        concat2 = str2+str1
        ans_size = float('inf')
        ans = ""

        for i in range(len(concat1)):
            cand = self.make(i,concat1,str1,str2)
            if cand and len(cand) < ans_size:
                ans = "".join(cand)
                ans_size = len(cand)
        
        #try other one
        for i in range(len(concat2)):
            cand = self.make(i,concat2,str1,str2)
            if cand and len(cand) < ans_size:
                ans = "".join(cand)
                ans_size = len(cand)

        return ans
    
    def make(self,start,larger,str1,str2):
        candidate = []
        i,j = 0,0
        while start < len(larger) and (i < len(str1) or j < len(str2)):
            #if both match
            if (i < len(str1) and j < len(str2)) and larger[start] == str1[i] and larger[start] == str2[j]:
                candidate.append(larger[start])
                i += 1
                j += 1
            #do we match on str1 first or str2
            elif i < len(str1) and larger[start] == str1[i]:
                candidate.append(larger[start])
                i += 1
            elif j < len(str2) and larger[start] == str2[j]:
                candidate.append(larger[start])
                j += 1
            else:
                start += 1
        
        if i == len(str1) and j == len(str2):
            return candidate
        return ""
