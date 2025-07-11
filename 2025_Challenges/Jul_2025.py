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
