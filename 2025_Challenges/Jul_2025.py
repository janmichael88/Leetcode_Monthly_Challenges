###########################################
# 3330. Find the Original Typed String I
# 01JUL25
###########################################
#using stack
class Solution:
    def possibleStringCount(self, word: str) -> int:
        '''
        if the display is aaaa
        the original could have been a,aa,aaa,or aaaa
        the key may have been held, more than once
        abbcccc
        (1-1) + (2-1) + (4-1)
        0 + 1 + 4

        count consecutive chars, call it size k, increment by k-1
        '''
        ans = 1
        stack = []
        for ch in word:
            if stack and stack[-1] != ch:
                k = len(stack)
                ans += (k-1)
                stack = []
            stack.append(ch)
        if stack:
            ans += len(stack) - 1
        return ans

#constant, just use pointers
class Solution:
    def possibleStringCount(self, word: str) -> int:
        '''
        if the display is aaaa
        the original could have been a,aa,aaa,or aaaa
        the key may have been held, more than once
        abbcccc
        (1-1) + (2-1) + (4-1)
        0 + 1 + 4

        count consecutive chars, call it size k, increment by k-1
        '''
        ans = 1
        i = 0
        while i < len(word):
            j = i+1
            while j < len(word) and word[i] == word[j]:
                j += 1
            
            ans += (j-i - 1)
            i = j
        
        return ans
            
##########################################
# 3333. Find the Original Typed String II
# 02JUL25
#########################################
#MLE
class Solution:
    def possibleStringCount(self, word: str, k: int) -> int:
        '''
        need number of possble strings of at least size k, any string with length >= k
        notice we don't have the "once" criteria as in the first problem
        dp starting from the beginnning, we can take the first character
            if we have taken this character is the next i+1 matches it, we can choose to take it or not
        
        then we'd need to keep track of index i and length
        which is O(n*k), not gonna work, anythin n*k is not going to work
        i can break up the string into consectuive groups by chars
        aabbccdd -> [2,2,2,2],
        then i need to build an arrayy such that for each num i need to use range(1,num+1) and have such that the sum is >= k
        you can just enumerate them, use bfs
        '''
        arr = []
        #make consectuive lengths array
        curr_length = 0
        prev = -1
        for ch in word:
            if prev == -1:
                prev = ch
                curr_length = 1
            elif ch == prev:
                curr_length += 1
            else:
                arr.append(curr_length)
                prev = ch
                curr_length = 1
        arr.append(curr_length)

        curr = list(range(1,arr[0]+1))
        for next_char_count in arr[1:]:
            #print(curr,next_char_count)
            next_level = []
            for curr_count in curr:
                for i in range(1,next_char_count+1):
                    next_level.append(curr_count+i)
            curr = next_level
        ans = 0
        for c in curr:
            if c >= k:
                ans += 1
        return ans
    
#MLE
class Solution:
    def possibleStringCount(self, word: str, k: int) -> int:
        '''
        need number of possble strings of at least size k, any string with length >= k
        notice we don't have the "once" criteria as in the first problem
        dp starting from the beginnning, we can take the first character
            if we have taken this character is the next i+1 matches it, we can choose to take it or not
        
        then we'd need to keep track of index i and length
        which is O(n*k), not gonna work, anythin n*k is not going to work
        i can break up the string into consectuive groups by chars
        aabbccdd -> [2,2,2,2],
        then i need to build an arrayy such that for each num i need to use range(1,num+1) and have such that the sum is >= k

        counting, inclusion/exclusion
        count the total number of straings we can have, which is just the prodcut of all the run lenghts
        then we substract the count of run lenghts that are 1 to k-1
        crux of the problem:
            so how can we caluclate the number of possible works for each length from 1 to k-1

        '''
        #just a cool way to deal with mod
        mod_mul = lambda a,b : (a*b) % 10**9 + 7
        mod_add = lambda a,b : (a+b) % 10**9 + 7
        mod_sub = lambda a,b : (a-b) % 10**9 + 7

        arr = []
        #make consectuive lengths array
        curr_length = 0
        prev = -1
        for ch in word:
            if prev == -1:
                prev = ch
                curr_length = 1
            elif ch == prev:
                curr_length += 1
            else:
                arr.append(curr_length)
                prev = ch
                curr_length = 1
        arr.append(curr_length)
        mod = 10**9 + 7
        #first compute total possible of any length k
        total = 1
        for l in arr:
            total *= l % mod
        total %= mod

        pref_sum = [0]
        for num in arr:
            pref_sum.append(pref_sum[-1] + num)
        
        memo = {}
        #calculate number of ways to make string <= j, using the first i run lenthgs
        #we would need dp(0,k-1) then substract that from total
        #can't do it top down, need to do it bottom up
        n = len(arr)
        memo = {}

        #start at i = 0, and with k-1
        def dp(i, remaining):
            mod = 10**9 + 7
            if remaining < 0:
                return 0
            if i >= n:
                if remaining == 0:
                    return 1
                return 0
            if (i,remaining) in memo:
                return memo[(i,remaining)]
            count = 0
            max_take = min(arr[i],remaining)
            for take in range(1,max_take + 1):
                count += dp(i+1,remaining-take) % mod
            memo[(i,remaining)] = count
            return count
        
        exclude = 0
        for i in range(1,k):
            exclude += dp(0,i) % mod
        return (total - exclude) % mod

#need to use prefsums, but carry each pref sum to next state
class Solution:
    def possibleStringCount(self, word: str, k: int) -> int:
        '''
        need number of possble strings of at least size k, any string with length >= k
        notice we don't have the "once" criteria as in the first problem
        dp starting from the beginnning, we can take the first character
            if we have taken this character is the next i+1 matches it, we can choose to take it or not
        
        then we'd need to keep track of index i and length
        which is O(n*k), not gonna work, anythin n*k is not going to work
        i can break up the string into consectuive groups by chars
        aabbccdd -> [2,2,2,2],
        then i need to build an arrayy such that for each num i need to use range(1,num+1) and have such that the sum is >= k

        counting, inclusion/exclusion
        count the total number of straings we can have, which is just the prodcut of all the run lenghts
        then we substract the count of run lenghts that are 1 to k-1
        crux of the problem:
            so how can we caluclate the number of possible works for each length from 1 to k-1

        '''
        #just a cool way to deal with mod
        mod_mul = lambda a,b : (a*b) % 10**9 + 7
        mod_add = lambda a,b : (a+b) % 10**9 + 7
        mod_sub = lambda a,b : (a-b) % 10**9 + 7

        arr = []
        #make consectuive lengths array
        curr_length = 0
        prev = -1
        for ch in word:
            if prev == -1:
                prev = ch
                curr_length = 1
            elif ch == prev:
                curr_length += 1
            else:
                arr.append(curr_length)
                prev = ch
                curr_length = 1
        arr.append(curr_length)
        mod = 10**9 + 7
        #first compute total possible of any length k
        total = 1
        n = len(arr)
        for l in arr:
            total *= l % mod
        total %= mod

        # dp[i][j]: number of ways to use first i groups to make length j
        dp = [[0] * (k) for _ in range(n + 1)]
        dp[0][0] = 1  # Base: 0 groups to make length 0

        for i in range(1, n + 1):
            prefix = [0] * (k + 1)
            for j in range(k):
                prefix[j + 1] = (prefix[j] + dp[i - 1][j]) % mod

            for j in range(i, k):
                min_take = 1
                max_take = min(arr[i - 1], j - (i - 1))
                if max_take < min_take:
                    continue
                dp[i][j] = (prefix[j - min_take + 1] - prefix[j - max_take]) % mod

        exclude = sum(dp[n][length] for length in range(1, k)) % mod

        return (total - exclude + mod) % mod

#this on actually passes though... fuck
mod_mul = lambda a, b: (a * b) % 1_000_000_007
mod_add = lambda a, b: (a + b) % 1_000_000_007
mod_sub = lambda a, b: (a - b) % 1_000_000_007

class Solution:
    def possibleStringCount(self, word: str, k: int) -> int: 
        segs = [1]
        for i in range(1, len(word)):
            if word[i] != word[i-1]:
                segs.append(1)
            else:
                segs[-1] += 1
        total = reduce(mod_mul, segs)
        if k <= len(segs):
            return total
        
        dp = [1] + ([0] * (k-1))
        for i in range(1, len(segs)+1):
            prefix = list(accumulate(dp, mod_add, initial=0))
            dp = [0] * k
            for j in range(i, k):
                dp[j] = mod_sub(prefix[j], prefix[j - min(segs[i-1], j-i+1)])
        less_than_k = reduce(mod_add, dp)
        return mod_sub(total, less_than_k)
    
#################################################
# 3304. Find the K-th Character in String Game I
# 03JUL25
################################################
class Solution:
    def kthCharacter(self, k: int) -> str:
        '''
        constrains are small enough to simulate
        '''
        word = ['a']
        for _ in range(k):
            next_chars = []
            for ch in word:
                delta = ord(ch) - ord('a')
                delta += 1 % 26
                temp = chr(ord('a') + delta)
                next_chars.append(temp)
            
            word.extend(next_chars)
            if len(word) >= k:
                return word[k-1]
        return word[k-1]
    
#log(k) time??? >.>
class Solution:
    def kthCharacter(self, k: int) -> str:
        '''
        logK solution
        observe that the string doubles every time, and k is <= 500
        even at 2**26, this is still way less than the limit of 500, so we don't need to worrk
        next notice, that the new half appended to the string, is just the old half, but every char is +1
        after k operations, the length is now 2**k
        to get the character at index i, depends on a previous character before it
            its the result of adding 1 to a character in the previous string
        since we double the length every time, that previous character has index i-p
            where p is the previous length of worst and the largest power of 2 <= i
        we have getchar(i) = getchar(i-p) + 1
            where p islargest power of <= i
        recally base case is getChar(0) == 'a'
        so we must continously divide i by the largest power of 2 <= i, and keep track of how man divisions
        '''
        index = k-1
        increments = 0
        while index > 0:
            p = 1
            while p*2 <= index:
                p = p*2
            increments += 1
            index -= p

        return chr(ord('a') + (increments % 26))  
        
class Solution:
    def kthCharacter(self, k: int) -> str:
        '''
        there's a pattern
        0 a 1
        1 ab 2
        2 abbc 4
        3 abbcbccd 8
        4 abbcbccdbccddeed 16

        now in the final string after 4 iterations notice, or when length is 2**4
        1 a
        2 b
        3 b
        4 c
        5 b
        6 c
        7 c
        8 d
        9 b
        10 c
        11 c
        12 d
        13 c
        14 d
        15 d
        16 e
        if you track the numer of times a character shifted from a
            its the number of 1's in k-1
        in the binary representation of (k-1) tells us how many times a position has been carried forward and promoted by 1
            for that kth character
        '''
        return chr(ord('a') + (k - 1).bit_count())
    

##########################################
# 1865. Finding Pairs With a Certain Sum
# 06JUL25
##########################################
class FindSumPairs:

    def __init__(self, nums1: List[int], nums2: List[int]):
        '''
        nums1 is small, and is never mutated
        iterate nums1 each time in count method

        '''
        self.nums1 = nums1
        self.nums2 = nums2
        self.counts2 = Counter(nums2)

    def add(self, index: int, val: int) -> None:
        prev = self.nums2[index]
        self.counts2[prev] -= 1
        if self.counts2[prev] == 0:
            del self.counts2[prev]
        self.nums2[index] += val
        self.counts2[prev + val] += 1

    def count(self, tot: int) -> int:
        ans = 0
        for num in self.nums1:
            ans += self.counts2[tot - num]
        return ans 

# Your FindSumPairs object will be instantiated and called as such:
# obj = FindSumPairs(nums1, nums2)
# obj.add(index,val)
# param_2 = obj.count(tot)

######################################################
# 1353. Maximum Number of Events That Can Be Attended
# 07JUL25
######################################################
class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        '''
        we only need tp be able to attend, we don't need to stay for the whole thing
        sort on stats and tie break by end time
        hints are mis leading, it even suggest min heap as a ds but not even saying we need to use minheap
        the idea is to iterate on days, rather than events, look at the constrains, its feabile to to in 10^5 times some constant
        we can only take one evevnt on any given day d,
            so we should take the one with the earlier end time
        
        use min_heap of endtimes
        '''
        events.sort()
        max_day = max([e for _,e in events])
        min_heap_end_times = []
        n = len(events)
        ans = 0
        j = 0

        for d in range(1,max_day + 1):
            #adding in candadate events that can be taken
            while j < n and events[j][0] <= d:
                end_time = events[j][1]
                heapq.heappush(min_heap_end_times, end_time)
                j += 1
            #clear events who's end is bigger than the current d
            while min_heap_end_times and min_heap_end_times[0] < d:
                heapq.heappop(min_heap_end_times)
            #we can take ane ven on this day, and use the even wih the earliest end time
            if min_heap_end_times:
                heapq.heappop(min_heap_end_times)
                ans += 1
        
        return ans

######################################################
# 3439. Reschedule Meetings for Maximum Free Time I
# 09JUL25
######################################################
#dammit, come back to this,
#i dont think the logic is quite right
class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        '''
        slide k blocks to make them that maximizes ungrouped area
        sliding window of k+1 and store max gaps in window
        problem is now how to to calucate gaps when expanding right and remove when shrinking left
        inital start time is is zero
        for right expansion
            gaps are between curr start and previous end
        for left contraction
            when we advance by 1, check next start and current end
        time intervals are increasing
        don't forget we have eventTime variable, marking the end
        '''
        ans = 0
        left = 0
        n = len(startTime)
        prev_end = 0
        curr_gap = 0
        for right in range(n+1):
            if right == n:
                gap_to_add = eventTime - prev_end
                curr_gap += gap_to_add
                prev_end = eventTime
            else:
                #expand right and
                gap_to_add = startTime[right] - prev_end
                curr_gap += gap_to_add
                prev_end = endTime[right]
            #if its too big
            if right - left + 1 > k and left + 1 < right:
                gap_to_remove = startTime[left + 1] - endTime[left]
                curr_gap -= gap_to_remove
                left += 1
            ans = max(ans,curr_gap)
            print(curr_gap)
        return ans

#finally
class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        '''
        slide k blocks to make them that maximizes ungrouped area
        sliding window of k+1 and store max gaps in window
        problem is now how to to calucate gaps when expanding right and remove when shrinking left
        inital start time is is zero
        for right expansion
            gaps are between curr start and previous end
        for left contraction
            when we advance by 1, check next start and current end
        time intervals are increasing
        don't forget we have eventTime variable, marking the end
        merge k+1 gaps
        '''
        n = len(startTime)
        ans = 0
        curr_gap = 0
        left = 0
        prev_end = 0
        gaps = [] #need to store gaps

        for right in range(n + 1):
            if right == n:
                gap = eventTime - prev_end
            else:
                gap = startTime[right] - prev_end

            curr_gap += gap
            gaps.append(gap)
            prev_end = eventTime if right == n else endTime[right]

            # Maintain window size of at most k+1 elements, if we have k meeetings, then we have k+1 gaps
            if right - left + 1 > k + 1:
                #need to remove the leftmost gap contribution! 
                curr_gap -= gaps[left]
                left += 1

            ans = max(ans, curr_gap)

        return ans

#prefix sum
class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        '''
        prefix sum,
        '''
        pref_sum = [0]
        n = len(startTime)
        prev_end = 0
        for i in range(n):
            gap = startTime[i] - prev_end
            pref_sum.append(pref_sum[-1] + gap)
            prev_end = endTime[i]
        
        #last gap time
        pref_sum.append(pref_sum[-1] + (eventTime - prev_end))
        ans = 0
        for right in range(1,len(pref_sum)):
            if right - (k+1) >= 0:
                ans = max(ans,pref_sum[right] - pref_sum[right - (k+1)])
            
        return ans
    
#####################################################
# 3440. Reschedule Meetings for Maximum Free Time II
# 11JUL25
#####################################################
class Solution:
    def maxFreeTime(self, eventTime: int, startTime: List[int], endTime: List[int]) -> int:
        '''
        we can reschedule at most 1 meeting, still need to remain non overlapping
        relative order of meetings can change
        we can only move one meeting, if we have too
            but we should move a meeting such that we have the largest gap
        startTime and endTime are already sorted 
        when we move an event to another sport, the gaps to the left of the event and to the right become joined
        try to fill each gap
        if we use an event to do so, see what happens to the change in the current gap (look left and right)
        but we need to be able place it some where, so we need to effeciently check where we can place this block
        we can use mapp or (pref max and suff max)
        hashmap approach need to validate index with biscet
        constant time with prefmax or suff max
            pref max to store largest block beforoe index i
            suff max to starge larvest black after index i
        then check
        '''
        n = len(startTime)
        pref_max = [0]*(n+1)
        ans = 0
        prev_end = 0
        for i in range(n):
            gap = startTime[i] - prev_end
            ans = max(ans,gap)
            prev_end = endTime[i]
            pref_max[i+1] = max(pref_max[i],gap)
        
        suff_max = [0]*(n+1)
        prev_start = eventTime
        for i in range(n-1,-1,-1):
            gap = prev_start - endTime[i]
            suff_max[i] = max(suff_max[i+1],gap)
            prev_start = startTime[i]
        
        for i in range(n):
            #remove the event and find gap
            left = 0 if i == 0 else endTime[i-1]
            right = eventTime if i == n-1 else startTime[i+1]
            gap_removal = right - left
            event_duration = endTime[i] - startTime[i]
            #if we can fix this event somehwere
            if (pref_max[i] >= event_duration) or (suff_max[i+1] >= event_duration):
                ans = max(ans,gap_removal)
            #we cant fit it anywhwere, we just slide it
            else:
                ans = max(ans, gap_removal - event_duration)
        
        return ans

#hashmap and effcient search through hashmap
from collections import Counter
import bisect

class Solution:
    def calculateGaps(self, eventTime: int, start: list[int], end: list[int]) -> list[int]:
        gaps = []
        prev = 0
        for i in range(len(start)):
            diff = start[i] - prev
            gaps.append(diff)
            prev = end[i]
        if prev != eventTime:
            gaps.append(eventTime - prev)
        return gaps

    def maxFreeTime(self, eventTime: int, start: list[int], end: list[int]) -> int:
        n = len(start)
        gaps = self.calculateGaps(eventTime, start, end)

        # Frequency counter and sorted list for lower_bound behavior
        intervalCnt = Counter(gaps)
        sorted_gaps = sorted(intervalCnt.keys())

        def remove_gap(gap):
            intervalCnt[gap] -= 1
            if intervalCnt[gap] == 0:
                del intervalCnt[gap]
                idx = bisect.bisect_left(sorted_gaps, gap)
                if idx < len(sorted_gaps) and sorted_gaps[idx] == gap:
                    sorted_gaps.pop(idx)

        def add_gap(gap):
            if gap not in intervalCnt:
                bisect.insort(sorted_gaps, gap)
            intervalCnt[gap] += 1

        ans = 0
        for i in range(n):
            prev = 0 if i == 0 else end[i - 1]
            next = eventTime if i == n - 1 else start[i + 1]

            prevInterval = start[i] - prev
            nextInterval = next - end[i]
            currInterval = end[i] - start[i]

            remove_gap(prevInterval)
            remove_gap(nextInterval)

            # Check if there is any gap >= current interval
            idx = bisect.bisect_left(sorted_gaps, currInterval)
            if idx < len(sorted_gaps):
                ans = max(ans, next - prev)
            else:
                ans = max(ans, next - prev - currInterval)

            add_gap(prevInterval)
            add_gap(nextInterval)

        return ans

#############################################################
# 1900. The Earliest and Latest Rounds Where Players Compete
# 12JUL25
##############################################################
#TLE
class Solution:
    def __init__(self):
        self.min_r = float('inf')
        self.max_r = float('-inf')

    def dfs(self, mask: int, round: int, i: int, j: int, first: int, second: int):
        if i >= j:
            self.dfs(mask, round + 1, 0, 27, first, second)
        elif (mask & (1 << i)) == 0:
            self.dfs(mask, round, i + 1, j, first, second)
        elif (mask & (1 << j)) == 0:
            self.dfs(mask, round, i, j - 1, first, second)
        elif i == first and j == second:
            self.min_r = min(self.min_r, round)
            self.max_r = max(self.max_r, round)
        else:
            if i != first and i != second:
                self.dfs(mask ^ (1 << i), round, i + 1, j - 1, first, second)
            if j != first and j != second:
                self.dfs(mask ^ (1 << j), round, i + 1, j - 1, first, second)

    def earliestAndLatest(self, n: int, first: int, second: int) -> list[int]:
        self.dfs((1 << n) - 1, 1, 0, 27, first - 1, second - 1)
        return [self.min_r, self.max_r]

class Solution:
    def earliestAndLatest(self, n: int, first: int, second: int) -> List[int]:
        '''
        this part is important:
            when num players is odd for a run, middle goes on
            after a round, the players are lined up in original ordering
        we are given the two best players who can beat anyone
        for any two players not competing against each other, choose a winner of this round
        bit masks on each state, then try each state
        n is at most 28, positions should fit in 32 bit integer
        and round so states are (row as bit, and num_rounds), then call dp function twice, one for min and one for max
        base cases are:
            bits left are firstPlayer pos and secondPlayer pos, and there are only two
            bits left are firstPlayer pos and secondPlayer pos, and is only one player in the middle
        the problem is we need to finish the round
        bin(num)[2:]

        '''
        first -= 1  # convert to 0-based
        second -= 1

        @lru_cache(maxsize=None)
        def dfs(mask: int, round: int, i: int, j: int) -> tuple[int, int]:
            if i >= j:
                return dfs(mask, round + 1, 0, n - 1)

            if not (mask & (1 << i)):
                return dfs(mask, round, i + 1, j)
            if not (mask & (1 << j)):
                return dfs(mask, round, i, j - 1)

            if i == first and j == second:
                return round, round

            res = []
            if i != first and i != second:
                res.append(dfs(mask ^ (1 << i), round, i + 1, j - 1))
            if j != first and j != second:
                res.append(dfs(mask ^ (1 << j), round, i + 1, j - 1))

            min_r = float('inf')
            max_r = float('-inf')
            for r1, r2 in res:
                min_r = min(min_r, r1)
                max_r = max(max_r, r2)

            return min_r, max_r

        result = dfs((1 << n) - 1, 1, 0, n - 1)
        return list(result)
    
class Solution:
    def earliestAndLatest(self, n: int, first: int, second: int) -> List[int]:
        '''
        this part is important:
            when num players is odd for a run, middle goes on
            after a round, the players are lined up in original ordering
        we are given the two best players who can beat anyone
        for any two players not competing against each other, choose a winner of this round
        bit masks on each state, then try each state
        states are (mask,round,i,j)  advance i and j and push states

        '''
        #convert to 0 base
        first -= 1
        second -= 1
        memo = {}

        def dp(mask,round,i,j):
            key = (mask,round,i,j)
            if key in memo:
                return memo[key]
            #round completed
            if i >= j:
                return dp(mask,round + 1,0,n-1)
            if not mask & (1 << i):
                return dp(mask,round,i+1,j)
            if not mask & (1 << j):
                return dp(mask,round,i,j-1)
            if i == first and j == second:
                return round, round
            
            #possible round answers
            res = []
            if i not in (first,second):
                lose_i = mask ^ (1 << i) #or win_j
                res.append(dp(lose_i,round,i+1,j-1))
            if j not in (first,second):
                lose_j = mask ^ (1 << j) #or win_i
                res.append(dp(lose_j,round,i+1,j-1))
            
            min_round = float('inf')
            max_round = float('-inf')
            for r1,r2 in res:
                min_round = min(min_round,r1)
                max_round = max(max_round,r2)
            ans = [min_round,max_round]
            memo[key] = ans
            return ans
        
        start_mask = 2**n - 1
        return dp(start_mask,1,0,n-1)
    
###########################################################
# 3178. Find the Child Who Has the Ball After K Seconds
# 14JUL25
###########################################################
class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        '''
        simulation
        '''
        dirr = -1
        curr = 0

        while k:
            if curr == 0 or curr == n-1:
                dirr *= -1
            curr += dirr
            k -= 1
        
        return curr

class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        '''
        if there are even rounds, its just num steps from the start
        otherwise its num steps from the right
        '''
        n -= 1
        rounds = k // n

        if rounds % 2 == 0:
            return k % n
        return n - (k % n)
    
###########################################
# 3136. Valid Word
# 15JUL25
############################################
class Solution:
    def isValid(self, word: str) -> bool:
        '''
        rulezzz
        '''
        if len(word) < 3:
            return False
        vowel_count = 0
        consonant_count = 0
        for ch in word:
            if not (ch.isalnum() and ch.isascii()):
                return False
            if ch in 'aeiouAEIOU':
                vowel_count += 1
            if ch.isalpha() and ch.lower() not in 'aeiou':
                consonant_count += 1
        return vowel_count >= 1 and consonant_count >= 1

#cleaner
class Solution:
    def isValid(self, word: str) -> bool:
        if len(word) < 3:
            return False

        has_vowel = False
        has_consonant = False

        for c in word:
            if c.isalpha():
                if c.lower() in "aeiou":
                    has_vowel = True
                else:
                    has_consonant = True
            elif not c.isdigit():
                return False

        return has_vowel and has_consonant
    
#############################################
# 893. Groups of Special-Equivalent Strings
# 15JUL25
#############################################
class Solution:
    def numSpecialEquivGroups(self, words: List[str]) -> int:
        '''
        move is:
            swap any two even indexed chars or any two odd index chars
        example:
            zzxy is equivalent to xyzz
        we can do
            zzxy -> (0,2) -> xzzy -> (1,3) -> xyzz

        we can make a group such that the group is largest as possible  
        for each word, try all pair swapping, the mapp, 
        kinda like an advent of code problem!

        say we have word
        [a,b,c,d]
        even indices can be permuted, and odd indices can be permuted
        get chars at even indices and sort them, then chars at odd indices and sort them, then weave
        this is the signature string
        '''
        mapp = Counter()
        for word in words:
            sig = self.get_sig(word)
            mapp[sig] += 1
        
        return len(mapp)
    
    def get_sig(self,word):
        even_chars = []
        odd_chars = []
        n = len(word)
        for i in range(n):
            if i % 2 == 0:
                even_chars.append(word[i])
            else:
                odd_chars.append(word[i])
        #sort
        even_chars.sort()
        odd_chars.sort()
        #weave
        sig = ['']*n
        idx = 0
        for ch in even_chars:
            sig[idx] = ch
            idx += 2
        idx = 1
        for ch in odd_chars:
            sig[idx] = ch
            idx += 2
        return "".join(sig)

class Solution:
    def numSpecialEquivGroups(self, words: List[str]) -> int:
        '''
        instead of sorting, use count arrays for odd,even inidces
        '''
        mapp = Counter()
        for word in words:
            sig = self.get_sig(word)
            mapp[sig] += 1
        
        return len(mapp)
    
    def get_sig(self,word):
        even_chars = [0]*26
        odd_chars = [0]*26
        n = len(word)
        for i,ch in enumerate(word):
            idx = ord(ch) - ord('a')
            if i % 2 == 0:
                even_chars[idx] += 1
            else:
                odd_chars[idx] += 1
        
        return (tuple(even_chars),tuple(odd_chars))

#cheeky one array using modtrick
class Solution(object):
    def numSpecialEquivGroups(self, A):
        def count(A):
            ans = [0] * 52
            for i, letter in enumerate(A):
                ans[ord(letter) - ord('a') + 26 * (i%2)] += 1
            return tuple(ans)

        return len({count(word) for word in A})
    
######################################################
# 3201. Find the Maximum Length of Valid Subsequence I
# 16JJUL25
######################################################
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        '''
        for a subsequence, we need pairwise sum parities to be equal
        odd + odd = even
        even + even = even
        odd + even = odd
        even + odd = odd
        try all,
            all odd, all even, alternave even odd, alternate odd even
        '''
        ans = 0
        curr = 0
        #try all odds
        for num in nums:
            if num % 2 == 1:
                curr += 1
        ans = max(ans,curr)
        #try all evens
        curr = 0
        for num in nums:
            if num % 2 == 0:
                curr += 1
        ans = max(ans,curr)
        
        #alternate odd,even
        curr_parity = 0
        curr = 0
        for num in nums:
            if num % 2 != curr_parity:
                curr += 1
                curr_parity = 1 - curr_parity
        ans = max(ans,curr)
        #alternate even odd
        curr_parity = 1
        curr = 0
        for num in nums:
            if num % 2 != curr_parity:
                curr += 1
                curr_parity = 1 - curr_parity
        ans = max(ans,curr)
        return ans
    
#just check all patterns
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        res = 0
        for pattern in [[0, 0], [0, 1], [1, 0], [1, 1]]:
            cnt = 0
            for num in nums:
                #count of element mod 2 gives parity at position (odd,evenlee)
                if num % 2 == pattern[cnt % 2]:
                    cnt += 1
            res = max(res, cnt)
        return res
    
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        '''
        another way is to track count of all odds, evens, or longest alternating chain
        priortize starting with left most
        '''
        evens = 0
        odds = 0
        alternating = 0
        parity = -1

        for num in nums:
            if num % 2 == 0:
                evens += 1
                if parity == -1 or parity == 1:
                    alternating += 1
            else:
                odds += 1
                if parity == -1 or parity == 0:
                    alternating += 1
            
            parity = num % 2
        
        return max(evens,odds,alternating)
    
########################################################
# 3202. Find the Maximum Length of Valid Subsequence II
# 17JUL25
#########################################################
#fuck it....
class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        '''
        every pairwise sum in the longest subsequence must have the same % k
        keep states i and last % k value
        say we have subsequence [a,b] and we want to add in c
        it must be that (a + b) % k = (b + c) % k
        we can do this:
        (a + b - b) % k = (b + c - b) % k
        so a %k = c % k
        so we can have (a,b) and add c only if c % k == a % k
        fix start and end, then check every c after that
        '''
        n = len(nums)
        left = 0
        ans = 0
        for right in range(1,n):
            first = nums[left]
            second = nums[right]
            dp = defaultdict()
            last_k = (first + second) % k
            dp[(right,last_k)] = 2
            for c in range(right+1,n):
                if c % k == start_k:
                    dp[c] = max(dp[right] + 1,dp[right])
                else:
                    dp[c] = dp[right]
            print(dp)

        return ans

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        dp = [[0] * k for _ in range(k)]
        res = 0
        for num in nums:
            num %= k
            for prev in range(k):
                dp[prev][num] = dp[num][prev] + 1
                res = max(res, dp[prev][num])
        return res

#############################################################
# 2163. Minimum Difference in Sums After Removal of Elements
# 18JUL25
#############################################################
class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        '''
        we are given 3 n elements
        need to remove subseqence of n elements
        such that the sum(first n element) - sum(second n element) is as small as possible
        to make small is possible make second sum large and first sum small
        nums in array are always positive
        for each index i consider how can we find the min possible sum of n elements with indices <= i
        '''
        n = len(nums)
        def dp_less(i,count,memo):
            if i >= n:
                if count == 0:
                    return 0
                return float('inf')
            if count < 0:
                return float('inf')
            if (i,count) in memo:
                return memo[(i,count)]
            take = nums[i] + dp_less(i+1,count-1,memo)
            no_take = dp_less(i+1,count,memo)
            ans = min(take,no_take)
            memo[(i,count)] = ans
            return ans

        def dp_greater(i,count,memo):
            if i >= n:
                if count == 0:
                    return 0
                return float('-inf')
            if count < 0:
                return float('-inf')
            if (i,count) in memo:
                return memo[(i,count)]
            take = nums[i] + dp_greater(i+1,count-1,memo)
            no_take = dp_greater(i+1,count,memo)
            ans = max(take,no_take)
            memo[(i,count)] = ans
            return ans

        
        memo_less = {}
        memo_greater = {}
        SUM = sum(nums)
        dp_less(0,n//3,memo_less)
        dp_greater(0,n//3,memo_greater)

        for i in range(n):
            min_ = dp_less(0,n//3,memo_less)
            max_ = dp_greater(0,n//3,memo_greater)
            print(SUM - min_,SUM - max_)


#finally
class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        '''
        need to space n indices to remove within nums
        such that the first n nums are as small as possible
        and the last n nums are as large as possible
        hint says for every index i, find min possible sum of n elements with indices <= i
        let k be n//3
        since we need to remove n//3 numbers
        we check k to k*2
        going left to right we find smallest sum for each index i from (k to k*2) - max heap to store the n smallest
        then going right to left we find the largest sum for each index i from k*2 to k - min heap to store the n largest
        then find smallest differ

        Where i ranges over the valid middle section (k - 1 ≤ i < 2k), and i + 1 marks the start of the right portion — ensuring no overlap between the k elements chosen on the left and the k on the right.
        '''
        n = len(nums)
        k = n // 3 #need to pick k nums to remove, so that first part is as small as possible and second part is as large as possible
        # min-heap (by negating values) to keep track of k largest nums on left
        max_heap = []
        left_sums = [0] * n
        curr_sum = sum(nums[:k])
        for i in range(k):
            heapq.heappush(max_heap, -nums[i])
        left_sums[k - 1] = curr_sum
        
        for i in range(k, 2 * k):
            heapq.heappush(max_heap, -nums[i])
            curr_sum += nums[i]
            curr_sum += heapq.heappop(max_heap)  # subtract largest negative = remove smallest actual value
            left_sums[i] = curr_sum
        
        # min-heap to keep track of k smallest nums on right
        min_heap = []
        right_sums = [0] * n
        curr_sum = sum(nums[-k:])
        for i in range(n - 1, n - k - 1, -1):
            heapq.heappush(min_heap, nums[i])
        right_sums[2 * k] = curr_sum
        
        for i in range(2 * k - 1, k - 1, -1):
            heapq.heappush(min_heap, nums[i])
            curr_sum += nums[i]
            curr_sum -= heapq.heappop(min_heap)
            right_sums[i] = curr_sum
        
        # now compute the minimum difference
        res = float('inf')
        for i in range(k - 1, 2 * k):
            #its not left_sums[i] - right_sums[i]
            #we need to make sure we dont intersect indices
            res = min(res, left_sums[i] - right_sums[i + 1])
        
        return res

###############################################
# 2210. Count Hills and Valleys in an Array
# 27JUL25
###############################################
class Solution:
    def countHillValley(self, nums: List[int]) -> int:
        '''
        start and end points can neither be a hill nor valley
        for each index, finds its closes non-equal neighbor
        cannot reduse index omg, you need to keep going 
        keep track of the last guard
        need to keep to ensure that we haven't use these indices yet,
        since we go from left to right we can check the rightmodt guard
        cool trick, keepig track of the rightmost edges
        '''
        ans = 0
        n = len(nums)
        last_right = -1  # rightmost index of previous counted region

        for i in range(1, n - 1):
            if i <= last_right:
                continue

            left = i - 1
            right = i + 1

            # find closest different on the left
            while left >= 0 and nums[left] == nums[i]:
                left -= 1

            # find closest different on the right
            while right < n and nums[right] == nums[i]:
                right += 1

            if left >= 0 and right < n:
                if nums[i] > nums[left] and nums[i] > nums[right]:
                    ans += 1
                    last_right = right - 1  # mark the range used
                elif nums[i] < nums[left] and nums[i] < nums[right]:
                    ans += 1
                    last_right = right - 1

        return ans
    
class Solution:
    def countHillValley(self, nums: List[int]) -> int:
        '''
        preporcess to clear dupliacates
        '''
        # remove duplicates
        cleaned = [nums[0]]
        for num in nums[1:]:
            if num != cleaned[-1]:
                cleaned.append(num)

        # check hills and valleys
        ans = 0
        for i in range(1, len(cleaned) - 1):
            if cleaned[i] > cleaned[i - 1] and cleaned[i] > cleaned[i + 1]:
                ans += 1  # hill
            elif cleaned[i] < cleaned[i - 1] and cleaned[i] < cleaned[i + 1]:
                ans += 1  # valley

        return ans
    
##############################################
# 2322. Minimum Score After Removals on a Tree
# 28JUL25
##############################################
#close one, gahhhh
class Solution:
    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
        '''
        unidrected tree, we are given node values and edges
        we need to remove a pair of edges to come three connected components
        for each comopoenents, XOR all nodes in each comp
        score = largest_xor = smallest_xor
        need min score
        brute force is to try all pairs of edges, then get componenents and find xor
        xoring the whole nums array gives us something
        its a tree, so dropping an edge will make 2 comps, dropping 2 edges will make three comps
        a tree with n nodes has n-1 eddges, removing n-1 edges makes n comps
        find the xor for each subtree, first assume we know the xor of each subtree

        '''
        graph = defaultdict(list)
        n = len(nums)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        for node in graph:
            graph[node] = sorted(graph[node])
        
        subtree_xors = [0]*n
        #compute subtree xors, assumption that tree is rooted at 0
        def dp(curr,parent):
            xor = nums[curr]
            for neigh in graph[curr]:
                if neigh != parent:
                    xor = xor ^ dp(neigh,curr)
            subtree_xors[curr] = xor
            return xor
        
        k = len(edges)
        for i in range(k):
            for j in range(i+1,k):
                u1,v1 = edges[i]
                u2,v2 = edges[j]
                
#dfs inside dfs N*N
class Solution:
    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
        '''
        first we start be doing dfs on the whole tree
        now imagine we have arrived to ode x,
            after deleting the edge between x and its parent node f, the subtree at x becaomse one of the three final parts
            we treat f as root run another DFS on the remaining part
            in this remaining part, we can try deleting another edges and obtrain three parts
        
        1. we compute the first part of x during the first DFS
        2. on the second dfs (with f as root), we traverse down to the node x'
            the xor of the subtree rooted at x' can also be compute while backtracking
        3. thir parts is just total_xor ^ part1 ^ part2
        '''
        #graph as usual
        graph = defaultdict(list)
        n = len(nums)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        total_xor = 0
        for num in nums:
            total_xor ^= num
        
        self.ans = float('inf')

        def dfs_second(curr,parent,first_part,anc):
            curr_xor = nums[curr]
            for neigh in graph[curr]:
                if neigh == parent:
                    continue
                curr_xor ^= dfs_second(neigh,curr,first_part,anc)
            
            if curr == anc:
                return curr_xor
            #its not really backtracking, more like global update for recursion
            self.ans = min(self.ans,self.calc(first_part, total_xor ^ first_part ^ curr_xor,curr_xor))
            return curr_xor
        def dfs_first(curr,parent):
            #this part gets the curr xor for rooting at x
            curr_xor = nums[curr]
            for neigh in graph[curr]:
                if neigh == parent:
                    continue
                curr_xor ^= dfs_first(neigh,curr)
            
            #dfs again from curr, to get the second part
            for neigh in graph[curr]:
                if neigh == parent:
                    continue
                dfs_second(neigh,curr,curr_xor,curr)
            
            return curr_xor
        
        dfs_first(0,-1)
        return self.ans
    
    def calc(self,p1,p2,p3):
        return max(p1,p2,p3) - min(p1,p2,p3)
    
#wtf is this times shit....

####################################################
# 2411. Smallest Subarrays With Maximum Bitwise OR
# 29JUL25
###################################################


class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        '''
        property of OR, is that it sets a bit if either 0,1 
        or can only increase as move through the array
        if we have a subarray, adding to it using an XOR can only increase it
        try consindering for each bit position
        for the subarray it needs to include the numbers at nums[i]
        since xoring a number only increases it, once we get to the max number we are done
        '''
        #mapp to store bit position from 0 to 31
        #to set this bit use this number
        n = len(nums)
        mapp = defaultdict(list)
        for i in range(32):
            for j,num in enumerate(nums):
                if num & (1 << i):
                    mapp[i].append(j)
        
        ans = []
        #why do we need the right mo
        for i,num in enumerate(nums):
            #init to i
            largest_index = i
            for bit in range(32):
                #check if bit is present in array
                if bit in mapp:
                    indices = mapp[bit]
                    #look for the index just after i to keep it minimum
                    idx = self.bin_search(indices,i)
                    if idx != -1:
                        #but we need the largest for all of them
                        largest_index = max(largest_index, indices[idx])
            ans.append(largest_index - i + 1)
        
        return ans
    def bin_search(self,arr,target):
        left = 0
        right = len(arr) - 1
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] >= target:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
    
class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        '''
        the idea is to traverse the array in reverse in descending order of index
        and for each bit position record the last index we have seen it
        '''
        last_seen = [-1]*32
        n = len(nums)
        ans = [1]*n

        for i in range(n-1,-1,-1):
            #check all bits for each number
            for bit in range(32):
                if nums[i] & (1 << bit):
                    last_seen[bit] = i
            max_idx = i
            for pos in last_seen:
                if pos != -1:
                    max_idx = max(max_idx,pos)
            
            ans[i] = max_idx - i + 1
        
        return ans
    
#############################################
# 898. Bitwise ORs of Subarrays
# 31JUL25
#############################################
#close one
class Solution:
    def subarrayBitwiseORs(self, arr: List[int]) -> int:
        '''
        return number of distict bitwise ORs for all non emty subarrays
        OR of a a subarray would result in an integer
        just check if we can set a bit
        if we have a subarray with bits already set, then OR'ing another number with it where it has 1's won't make a difference
        but OR'ing with a number that has 1s where the subarray bits aren't set will make it distinct
        for each bit position store if there's a number where that position is set
        then we can use dp on bits, if there's a bit set it contributes a count, if not skip it
        '''
        bits = [False]*32
        for i in range(32):
            for num in arr:
                if num & (1 << i):
                    bits[i] = True
        
        memo = {}
        def dp(i):
            if i >= len(bits):
                return 0
            if i in memo:
                return memo[i]
            if bits[i] == True:
                take = 1 + dp(i+1)
            else:
                take = 0
            no_take = dp(i+1)
            ways = take + no_take
            memo[i] = ways
            return ways
        
        return max(dp(0),1)
    
#enumeration, bfs
class Solution:
    def subarrayBitwiseORs(self, arr: List[int]) -> int:
        '''
        we need to use enumeraton
        for integer in nums, OR that num with all the previous integers in that set
        '''
        ans = set()
        prev = set()
        for num in arr:
            curr = {num}
            for p in prev:
                curr.add(p | num)
            ans.update(curr)
            prev = curr
        
        return len(ans)
    
class Solution:
    def subarrayBitwiseORs(self, arr: List[int]) -> int:
        '''
        using comprehension
        the number of unque values in this set is at most 32
        O(Nlog(max(arr)))
        '''
        ans = set()
        curr = {0}
        for num in arr:
            curr = {num | prev for prev in curr} | {num}
            ans = ans | curr
        
        return len(ans)
        