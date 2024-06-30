###########################################################
# 1940. Longest Common Subsequence Between Sorted Arrays
# 01JUN24
##########################################################
class Solution:
    def longestCommonSubsequence(self, arrays: List[List[int]]) -> List[int]:
        '''
        the arrays are sorted, if they weren't i woulud have to use thee state dp
        fix one array, choose next array and get the common elmenets
        use common elements
        
        '''
        #find commone elements between first and second
        ans = []
        first = arrays[0]
        for next_array in arrays[1:]:
            curr_ans = []
            i,j = 0,0
            while i < len(first) and j < len(next_array):
                if first[i] == next_array[j]:
                    curr_ans.append(first[i])
                if first[i] < next_array[j]:
                    i += 1
                else:
                    j += 1
            
            ans = curr_ans
            first = curr_ans
        
        return ans
    
#hashamp with counts
class Solution:
    def longestCommonSubsequence(self, arrays: List[List[int]]) -> List[int]:
        '''
        we can also use a hasmap
        since each array is strictly increasing, there will be no duplicates
        so if we have N arrays and some number X appears N times, it will be a part of the
        longest_common_subsequence
        
        we could count all nums in arrays and go in order, or we can count in the fly
        we can count on the fly because when a common number == N, it needs to be added in order of the arrays anyway
        '''
        counts = defaultdict(int)
        N = len(arrays)
        longest_common = []
        for arr in arrays:
            for num in arr:
                counts[num] += 1
                if counts[num] == N:
                    longest_common.append(num)
        
        return longest_common
    
#binary search solution
#since the arrays are sorted we can find the next point to advance to without advancing one by one

###########################################################
# 2486. Append Characters to String to Make Subsequence
# 03JUN24
#########################################################
class Solution:
    def appendCharacters(self, s: str, t: str) -> int:
        '''
        need to add letters to s so that t becomes a subsequence of s, need minimum
        we can only add letters
        we need to keep advancing in s, if we can, we kno we have to add in the remaining
        find longest prefix of t that is also a subequence of s, we need to add in the remaing chars to s
        '''
        i = 0
        for ch in s:
            if i >= len(t):
                break
            if ch == t[i]:
                i += 1
        
        return len(t) - i

#################################
# 409. Longest Palindrome (REVISITED)
# 04JUN24
#####################################
class Solution:
    def longestPalindrome(self, s: str) -> int:
        '''
        for even length palindrome, letters must exsist in pairs
        for odd lenght, all letters except the center must exsist in pairs
        heap with pairs and take by two, until top is 1
        '''
        counts = Counter(s)
        max_heap = [-count for _,count in counts.items()]
        heapq.heapify(max_heap)
        
        ans = 0
        while max_heap:
            if max_heap[0] == -1:
                break
            curr_count = heapq.heappop(max_heap)
            ans += 2
            curr_count += 2
            if curr_count < 0:
                heapq.heappush(max_heap, curr_count)
        
        if max_heap and max_heap[0] == -1:
            ans += 1
        return ans
    
class Solution:
    def longestPalindrome(self, s: str) -> int:
        '''
        sort and use pairs, if i have a left over one add it
        '''
        counts = Counter(s)
        counts = [count for _,count in counts.items()]
        counts.sort(reverse = True)
        ans = 0
        last_one = False
        
        for num in counts:
            if num == 1:
                last_one = True
                break
            else:
                if num % 2 == 0:
                    ans += num
                else:
                    ans += num - 1
                    last_one = True
        
        if last_one:
            ans += 1
        
        return ans

class Solution:
    def longestPalindrome(self, s: str) -> int:
        '''
        you dont need to sort
        '''
        counts = Counter(s)
        counts = [count for _,count in counts.items()]
        ans = 0
        last_one = False
        
        for num in counts:
            if num % 2 == 0:
                ans += num
            else:
                ans += num - 1
                last_one = True
        
        if last_one:
            ans += 1
        
        return ans
                 
class Solution:
    def longestPalindrome(self, s: str) -> int:
        '''
        we can count can build on the fly
        odd counts of characters will have 1 unused letter to pair, unless its 1, then its the center
        we use this intution to keep track of the charactes that have an odd frequency
        then just subtract from len(s)
        '''
        counts = defaultdict(int)
        leftover_odds = 0
        
        for ch in s:
            counts[ch] += 1
            if counts[ch] % 2 == 1:
                leftover_odds += 1
            else:
                leftover_odds -= 1
        
        if leftover_odds > 0:
            return len(s) - leftover_odds + 1
        return len(s)
            
###########################################
# 1668. Maximum Repeating Substring
# 04JUN24
###########################################
#doesn't quite work
class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        '''
        just slide and count
        crap its k-repeating
        its just count streak but with word
        '''
        L = len(sequence)
        M = len(word)
        
        if word not in sequence:
            return 0
        max_streak = 1
        curr_streak = 0
        i = 0
        while i < L-M+1:
            if sequence[i:i+M] == word:
                curr_streak += 1
                i += M
            else:
                max_streak = max(max_streak,curr_streak)
                curr_streak = 0
                i += 1
        
        return max(max_streak,curr_streak)
    
class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        '''
        brute force would be to try all k
        '''
        L = len(sequence)
        M = len(word)
        
        for k in range(L-M+1,-1,-1):
            kth_repeating = word*k
            if kth_repeating in sequence:
                return k
        
        return 0
    
#binary search workable
class Solution:
    def maxRepeating(self, sequence: str, word: str) -> int:
        if word not in sequence:
            return 0

        left = 1
        ans = 1
        right = len(sequence) - len(word) + 1
        while left <= right:
            mid = (left + right) // 2
            if word * mid in sequence:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1 
                
        return ans

#########################################
# 1002. Find Common Characters (REVISTED)
# 05JUN24
########################################
#kinda tricky actually
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        '''
        count each char for each word
        contribution for this letter will be the minimum of its count
        '''
        count_words = []
        for w in words:
            count_words.append(Counter(w))
        
        ans = []
        for i in range(26):
            curr_char = chr(ord('a') + i)
            min_ans = float('inf')
            for w in count_words:
                min_ans = min(min_ans,w[curr_char])
            
            ans += [curr_char]*min_ans
        
        return ans
    
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        '''
        we can abuse counters and
        '''
        first = Counter(words[0])
        
        for w in words[1:]:
            first = first & Counter(w)
        
        return [v for v in first.elements()]
    
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        '''
        using map and reduce
        '''
        return reduce(lambda x,y : x & y, map(collections.Counter,words)).elements()

######################################
# 1763. Longest Nice Substring
# 05JUN24
#####################################
#brute force works
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        '''
        brute force works
        '''
        N = len(s)
        ans = ""
        for start in range(N):
            for end in range(start+1,N+1):
                sub = s[start:end]
                if self.isNice(sub):
                    if len(sub) > len(ans):
                        ans = sub
        
        return ans
    
    def isNice(self, s : str) -> bool:
        chars = set(s)
        for ch in chars:
            if ch.islower() and ch.upper() not in chars:
                return False
            if ch.isupper() and ch.lower() not in chars:
                return False
        
        return True
    
#divide and conquer
class Solution:
    def longestNiceSubstring(self, s: str) -> str:
        '''
        divide and conquer
        introducing swapcase
        split on all prefixes and suffices of s
        need earliest occurence
        '''
        def rec(s):
            if not s:
                return ""
            print(s)
            seen = set(s)
            for i,c in enumerate(s):
                if c.swapcase() not in seen:
                    left = rec(s[:i])
                    right = rec(s[i+1:])
                    if len(right) > len(left):
                        return right
                    return left
            
            return s
        
        return rec(s)
    
########################################
# 846. Hand of Straights
# 06JUN24
#########################################
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        '''
        if the numbers were in order, i could just keep creating a groups
        but there can be dupliates
        i count count them up and keep creating
        hints 1. if smallest number in a partition is V, then V+1,V+2,..V+k must also be in the array
        '''
        N = len(hand)
        
        #cannot divide evenly
        if N % groupSize != 0:
            return False
        counts = Counter(hand)
        ordered_nums = sorted(list(set(hand)))
        
        for num in ordered_nums:
            while counts[num] > 0:
                for k in range(groupSize):
                    if counts[num+k] == 0:
                        return False
                    counts[num+k] -= 1
        
        return True
        
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        '''
        we dont need to start with the smallest card
        we cant just pick any card, and check for streaks, 
        but we can pick a card, then decremnet to safe state,
        the trick is to identify where to start the streaks
        process is called reverse decrement
        '''
        N = len(hand)
        
        #cannot divide evenly
        if N % groupSize != 0:
            return False
        
        counts = Counter(hand)
        for num in hand:
            start = num
            #find start
            while counts[start-1] > 0:
                start -= 1
                
            while counts[start] > 0:
                for k in range(groupSize):
                    if counts[start + k] == 0:
                        return False
                    counts[start + k] -= 1
        
        return True

############################################
# 648. Replace Words (REVISITED)
# 07JUN24
############################################
class Node:
    def __init__(self):
        self.children = defaultdict()
        self.end = False
        
class Trie:
    def __init__(self,):
        self.root = Node()
        
    def insert(self,word):
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = Node()
            
            #move 
            curr = curr.children[ch]
        
        #mark
        curr.end = True
        
    def search(self,word):
        curr = self.root
        pref = ""
        for ch in word:
            if ch not in curr.children:
                if curr.end:
                    return pref
                return word
            #check before ending
            if curr.end:
                return pref
            pref += ch
            curr = curr.children[ch]
            
        
        if curr.end:
            return pref
        return word
            
class Solution:
    def replaceWords(self, dictionary: List[str], sentence: str) -> str:
        '''
        build trie of roots, and also add state for ending word
        need to check all possible roots!
        '''
        trie = Trie()
        for w in dictionary:
            trie.insert(w)
        
        sentence = sentence.split(" ") 
        ans = []
        for w in sentence:
            ans.append(trie.search(w))
        
        return " ".join(ans)
    
#########################################
# 826. Most Profit Assigning Work
# 07JUN24
#########################################
class Solution:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        '''
        ideally we'd like to perform the least difficult job with the greatest ammount of profit
        we'd like to perform this as many times as a worker can
        its not always the case that the most difficult job will have the highest profit
        assign each work the most profitable job it can do, given the difficulty
        say for example worker has j difficulty
        we want to chose the largest profitable job such that thae difficulty <= j
        
        edge cases obvie, but a generalize algo might not need for that
        
        need fast way to find maximum in certain range -> dp
        search will always be bounded left, starting 0
        
        sort by difficulty, then build prefix max
        then binary search on each worker
        '''
        pairs = [(diff,prof) for diff,prof in zip(difficulty,profit)]
        pairs.sort(key = lambda x: x[0])
        
        sorted_diffs = []
        ordered_profs = []
        for diff,prof in pairs:
            sorted_diffs.append(diff)
            ordered_profs.append(prof)
        
        pref_max = [0]
        for p in ordered_profs:
            pref_max.append(max(pref_max[-1],p))
        
        max_prof = 0
        for w in worker:
            left = 0
            right = len(sorted_diffs) - 1
            ans = left
            while left <= right:
                mid = left + (right  - left) // 2
                if w >= sorted_diffs[mid]:
                    ans = mid
                    left = mid + 1
                else:
                    right = mid - 1
            
            #print(w,ans,pref_max[ans])
            if sorted_diffs[ans] <= w:
                max_prof += pref_max[ans+1]
        
        return max_prof

#we can also just simply sort the events and find the max profitable one w
class Solution:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        '''
        we can pair them and use two pointers
        '''
        jobs = zip(difficulty,profit)
        jobs = sorted(list(jobs))
        ans = 0
        i = 0
        best_prof = 0
        
        for w in sorted(worker):
            while i < len(jobs) and w >= jobs[i][0]:
                best_prof = max(best_prof, jobs[i][1])
                i += 1
            
            ans += best_prof
        
        return ans
    
#revisited on 18JUN24
#diificulty variant
class Solution:
    def maxProfitAssignment(self, difficulty: List[int], profit: List[int], worker: List[int]) -> int:
        '''
        since the constraint on difficulty[i] and worker[i] are small, 
        can can just do it O(max(dificulty,worker))
        we dont need to know profits for jobs are higher than a worker can handle -> max_so_far
        use an array of max worker ability and store the max profit for this workers ability
        '''
        max_ability = max(worker)
        max_profits = [0]*(max_ability + 1)
        
        for i,diff in enumerate(difficulty):
            #if there is a job difficulty smaller than the max, update it
            #i.e there is job who's dififculty can be done but with a high profit
            if diff <= max_ability:
                max_profits[diff] = max(max_profits[diff], profit[i])
        
        #find max for each difficulrt
        for i in range(1,len(max_profits)):
            max_profits[i] = max(max_profits[i], max_profits[i-1])
        
        ans = 0
        for w in worker:
            ans += max_profits[w]
        
        return ans
    
#########################################
# 523. Continuous Subarray Sum (REVISITED)
# 08JUN24
##########################################
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        '''
        what we want to look for are pref_mods % k
        say we have some array
        [a,b,c,d,e,f,g] and says some array [b + c + d + e] % k == 0
        
        say we have some some pref_sum with [i+1,j]
        and in this interval its % k, rather
        pref_sum[j] - pref_sum[i] % k = 0
        we can express in divisor, dividend, and remainder
        pref_sum[i] = Qi*k + Ri
        pref_sum[j] = Qj*k + Rj
        
        we have -> Qi*k + Ri - (Qj*k + Rj_) % k == 0
        (Qi - Qj)*k + (Ri - Rj) % k == 0
        (Qi - Qk) is % k and
        both remainders like in range [0,k-1]
        (Ri - Rj) is % k if
        Ri - Rj = 0 or Ri = Rj
        so we want pref_mods that are equal to each other
        pref_sums  up to index j, % k should == pref_sum up to index i mode k
        
        '''
        curr_sum = 0
        pref_mods_mapp = {0:-1}
        
        for i,num in enumerate(nums):
            curr_sum += num
            if (curr_sum % k) in pref_mods_mapp:
                if i - pref_mods_mapp[curr_sum % k] > 1:
                    return True
            else:
                pref_mods_mapp[curr_sum % k] = i
        
        return False
    
#############################################
# 1588. Sum of All Odd Length Subarrays
# 09JUN24
#############################################
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        '''
        i can use pref_sum and check all add length subarrays
        '''
        N = len(arr)
        pref_sum = [0]*(N+1)
        curr_sum = 0
        for i in range(N):
            curr_sum += arr[i]
            pref_sum[i+1] = curr_sum
        

        ans = 0
        for start in range(N):
            for end in range(start,N,2):
                #print(arr[start:end+1], pref_sum[end+1]-pref_sum[start])
                ans += pref_sum[end+1] - pref_sum[start]
        
        return ans
    
#without pref_sum, just keep track of sums serpataltey
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        
        N = len(arr)
        ans = 0
        
        for start in range(N):
            curr_sum = 0
            for end in range(start,N):
                curr_sum += arr[end]
                if (end - start + 1) % 2 == 1:
                    ans += curr_sum
        
        return ans
    
#O(N)
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        '''
        count the number of times and index occurs
        then for each index its contribution is (num_times index occured)*value_at_index
        we can use dp to figure out how many times an index occurs in a subarray
        say we are given the array
        https://www.youtube.com/watch?v=J5IIH35EBVE&t=13s
        given index, find how many subarrays start with i and end with i, total number is the product
        for index i
            count start is N - i
            count end is i + 1
            total is (N-i)*(i+1)
        
        now half will be even, and half will be odd
        but there could be extra odd
        if N is odd there is an extra one
        need to pay attention to parity of the total number of subarrays
        but we need odd length for each of these
        '''
        N = len(arr)
        count_inidices = [1]*N
        ans = 0
        for i in range(N):
            #count possible subarrays using this index on left and right
            count_start = N-i
            count_end = i+1
            #print(i,count_start,count_end)
            total_subarrays = count_start*count_end
            odd_length = total_subarrays // 2
            odd_length += total_subarrays % 2
            ans += arr[i]*odd_length
        
        return ans
    
################################################
# 974. Subarray Sums Divisible by K (REVISITED)
# 09JUN24
################################################
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        '''
        just keep track of pref_mods
        '''
        curr_sum = 0
        pref_mods_mapp = defaultdict(int)
        pref_mods_mapp[0] = 1
        count = 0
        
        for i,num in enumerate(nums):
            curr_sum += num
            if (curr_sum % k) in pref_mods_mapp:
                count += pref_mods_mapp[curr_sum % k]
            pref_mods_mapp[curr_sum % k] += 1
            
        
        return count
        
#############################################
# 1590. Make Sum Divisible by P
# 09JUN24
#############################################
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        '''
        we need to remove a subarray to make sum divivibsle by p
        say we have subarray [a,b,c,d,e]
        we want (a+b+c+d+e) % p == 0
        
        if (a+b+c+d+e) % p == k, where k != 0
        then we need to find the smallest subarray == k where sum(subarray) % p == k
        '''
        SUM = sum(nums)
        if SUM % p == 0:
            return 0
        N = len(nums)
        k = SUM % p
        ans = N
        
        mapp = {}
        mapp[0] = -1
        curr_sum = 0
        for i,num in enumerate(nums):
            curr_sum += num
            #need to find its complement mod p
            if (curr_sum - k) % p in mapp:
                ans = min(ans, i - mapp[(curr_sum - k) % p ] )
    
            mapp[curr_sum % p] = i
        
        if ans < N:
            return ans
        return -1

#################################################
# 1636. Sort Array by Increasing Frequency
# 10JUN24
#################################################
class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        '''
        just use counting sort on count of counts
        if there are ties do decreasing order
        '''
        counts = Counter(nums)
        entries = [(count,num) for num,count in counts.items()]
        entries.sort(key = lambda x: (x[0],-x[1]))
        ans = []
        for count,num in entries:
            for _ in range(count):
                ans.append(num)
        
        return ans
    
################################################
# 2575. Find the Divisibility Array of a String
# 10JUN24
################################################
class Solution:
    def divisibilityArray(self, word: str, m: int) -> List[int]:
        '''
        just keep track of number going left to right
        the issue is that the number might be really big
        say we have some digit abc
        we can right as a*100 + b*10 + c*1
        (a*100 + b*10 + c*1) % m
        ((a*100 % m) + (b*10 % m) + (c*1 % m)) % m
        need to multiply by 10 each time but take mod m so it doesnt grow
        '''
        N = len(word)
        arr = [0]*N
        
        curr_rem = 0
        for i,ch in enumerate(word):
            curr_rem = (curr_rem*10 % m + int(ch) % m) % m
            if curr_rem == 0 :
                arr[i] = 1
        
        return arr
            
######################################
# 1122. Relative Sort Array (REVISTED)
# 11JUN24
######################################
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        '''
        we can use comparator
        '''
        mapp = defaultdict()
        MAX = max(arr1)
        for i,num in enumerate(arr1):
            mapp[num] = MAX + num
        
        for i,num in enumerate(arr2):
            mapp[num] = i
        arr1.sort(key = lambda x : mapp[x])
        return arr1
    
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        '''
        we can use counting sort
        '''
        max_ = max(arr1)
        counts = [0]*(max_ + 1)
        N = len(arr1)
        
        for num in arr1:
            counts[num] += 1
        
        ans = [-1]*N
        i = 0
        for num in arr2:
            while counts[num] > 0:
                ans[i] = num
                counts[num] -= 1
                i += 1
                
        #remaning
        for num in range(max_ + 1):
            while counts[num] > 0:
                ans[i] = num
                counts[num] -= 1
                i += 1
        
        return ans

#############################################
# 2845. Count of Interesting Subarrays
# 11JUN24
#############################################
#TLE
class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        '''
        for some subarray [l,r]
        i need count(nums[i] % module == kin range(l,r+1)
        this array is intereating
        
        say an interesting subarray is [a,b,c,d]
        we can decode the array as nums[i] % mod == k, which will be 0 or 1
        then we can just sum,
        the idea now involves finding subarrays where curr_sum % modulo == 0
        warm up with this first because i can't fucking think tonight
        '''
        N = len(nums)
        count = [0]*N
        for i in range(N):
            count[i] = 1 if nums[i] % modulo == k else 0
        
        
        #fill in sums
        for i in range(1,N):
            count[i] += count[i-1]
        
        count = [0] + count
        ans = 0
        for start in range(N):
            for end in range(start,N):
                cnt = count[end+1] - count[start]
                if cnt % modulo == k:
                    ans += 1
        
        return ans
    
class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        '''
        for some subarray [l,r]
        i need count(nums[i] % module == kin range(l,r+1)
        this array is intereating
        
        say an interesting subarray is [a,b,c,d]
        we can decode the array as nums[i] % mod == k, which will be 0 or 1
        then we can just sum,
        the idea now involves finding subarrays where curr_sum % modulo == 0
        warm up with this first because i can't fucking think tonight
        how many indicies statify
        (count[i] - count[j]) % modulo == k
        can rewrite as
        ((count[i] % modulo) - (count[j] % modulo)) % modulo == k
        
        say we have x = sum[0,i-1] and y = [0,r]
        y-x = sum[i,j]
        need (y - x) % modulo == l
        ((y % modulo) - (x % modulo)) % modulo == k
        k < modulo so k % modulo == k
         ((y % modulo) - (x % modulo)) % modulo == k % modulo
         (y % modulo) - (x % modulo) == k
         (y % modulo) - (x % modulo) == k % modulo
         (y % modulo) - (k % modulo) = (x % modulo)
         distrubutive property
         (y - k) % modulo == (x % modulo)
         so we know some prefix sum y, we can find the other
        https://leetcode.com/problems/count-of-interesting-subarrays/discuss/3994985/JavaC%2B%2BPython-Prefix-O(n)
        '''
        N = len(nums)
        mapp = defaultdict(int)
        mapp[0] = 1
        curr_count = 0
        ans = 0
        for num in nums:
            curr_count += num % modulo == k
            #looks for its complement count
            #subarray sum == k
            ans += mapp[(curr_count + modulo - k) % modulo]
            #this will also work
            #ans += mapp[(curr_count  - k) % modulo]
            #modular arithmetic in python is different than in javo C++
            mapp[curr_count % modulo] += 1
    
        return ans
    
#####################################
# 781. Rabbits in Forest
# 12JUN24
#####################################
#tricky counting
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        '''
        if im rabbit of color x, and my ans i,
        then for color x, there is at least 2 of x
        use index of rabbit as color and check previous answers
        there can't be more than len(ans) diffiernt colors
        each unique answer means there are at least ans + 1 rabbits
        each unique answer can repeate at most ans + 1 times
        [2,2,2], all are same color
        but [2,2,2,2] will have 6 raabits
        [2,2,2] make 3 raabits, but the next 2, cant be the same color at the first 3, so its 3 + (2+1)
        i.e the last two needs two rabbits of a different color
        so we need to find how many groups [2,2,2] -> math.ceil(v/(k+1))
        
        another way
        if x + 1 rabbits have the same color, then we get x + 1 rabbits who all answer x
        now if n rabbits answer x
        If n % (x + 1) == 0, we need n / (x + 1) groups of x + 1 rabbits.
        If n % (x + 1) != 0, we need n / (x + 1) + 1 groups of x + 1 rabbits.

        the number of groups is math.ceil(n / (x + 1)) and it equals to (n + x) / (x + 1) , which is more elegant.
        
        '''
        counts = Counter(answers)
        ans = 0
        for k,v in counts.items():
            #groups of raabits for k unique answers
            ans += math.ceil(v/(k+1))*(k+1)
        
        return ans
        
#one pass, count on the fly
class Solution:
    def numRabbits(self, answers: List[int]) -> int:
        '''
        ans rabbit says 0, there is one unique count for this colored rabbit
        otherwise check if we've already seen thie colorwe need to keep rabbits to a minimum. so if answer, 
        what the author calls i, for a rabbit has i == d[i], then we need to make a new group, and increment by i+1. 
        when making a new group, the counts are set to 0 again.
        '''
        min_rabbits = 0
        counts = defaultdict(int)
        
        for ans in answers:
            if ans == 0:
                min_rabbits += 1
            else:
                if ans not in counts or ans == counts[ans]:
                    #for a group of colore rabbits with ith color, the minimum can't be more than ans
                    #For example [2,2,2] and [2,2] has the same result (i.e) 3 but [2,2,2,2] should    
                    counts[ans] = 0
                    min_rabbits += 1 + ans
                else:
                    counts[ans] += 1
        
        return min_rabbits
        
        
        
################################################
# 2037. Minimum Number of Moves to Seat Everyone
# 13JUN24
################################################
class Solution:
    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
        '''
        just sort both
        '''
        seats.sort()
        students.sort()
        
        ans = 0
        for i,j in zip(seats,students):
            ans += abs(i-j)
        
        return ans
    
#counting sort
class Solution:
    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
        '''
        we can use counting sort
        then we record the differecnes
        +1 seat for seat in seats
        -1 student for student in students
        
        keep unmatche variable to keep track of number of unseated studenets or empty seats that have not been matched yet
        its positive if there are extra seats and engative if there are extra students
        if unmathced is -1, it means there is an unmatched student
        if its +1, it means there is an unmatched seat, each postition without a seat requires a move!
        so increment ans by abs(unmatched)
        
        then we just accumlate unmatched
        '''
        max_num = max(max(seats),max(students))
        
        diffs = [0]*(max_num+1)
        
        for s in seats:
            diffs[s] += 1
        
        for s in students:
            diffs[s] -= 1
        
        moves = 0
        unmatched = 0
        for d in diffs:
            moves += abs(unmatched)
            unmatched += d
        
        return moves

################################################
# 945. Minimum Increment to Make Array Unique
# 14Jun24
###############################################
class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        '''
        what if i sort the array
        [3,2,1,2,1,7]
        after sorting
        [1, 1, 2, 2, 3, 7] + 1
        [1, 2, 2, 2, 3, 7] + 1
        [1, 2, 3, 2, 3, 7] + 1
        [1, 2, 3, 4, 3, 7]
        
        sort and rais each one
        doing thus with one hand too while on plane lmaoooo fyck yeh
        '''
        N = len(nums)
        nums.sort()
        ans = 0
        curr_smallest = nums[0]
        
        
        for i in range(1,N):
            if nums[i] == nums[i-1]:
                ans += 1
                nums[i] += 1
                curr_smallest = nums[i]
            elif nums[i] < nums[i-1]:
                curr_smallest += 1
                ans += abs(nums[i] - curr_smallest)
                nums[i] = curr_smallest
        
        return ans
                
class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        '''
        sort and check sonsecutie elements
        since we  need them to be uniqe, we always try to make the ith elemet just greatser
        we also need to make them increasing
        '''
        min_moves = 0
        nums.sort()
        N = len(nums)
        
        for i in range(1,N):
            if nums[i] <= nums[i-1]:
                steps = nums[i-1] - nums[i] + 1
                min_moves += steps
                nums[i] = nums[i-1] + 1
            
        
        return min_moves
    
class Solution:
    def minIncrementForUnique(self, nums: List[int]) -> int:
        '''
        counting sort
        count the dupliates
        if there is an occurence of 1 for a num, we are good, otherwise we need to promote the 
        count of duplicate to the next value i.e carrythem
        but first we need to find the range for the counts array
        its just max(nums) + n + 1
        we could have an array of size n, where all the nums are max, in this case we'd be carrying over all n
        '''
        N = len(nums)
        MAX = max(nums)
        min_moves = 0
        
        counts = [0]*(N + MAX + 1)
        
        for num in nums:
            counts[num] += 1
            
        for i in range(len(counts)):
            if counts[i] <= 1:
                continue
            
            #find duplicates and carry over
            duplicates = counts[i] - 1
            counts[i+1] += duplicates
            #we dont need to do this part
            #counts[i] = 1
            min_moves += duplicates
        
        return min_moves
    
########################################################
# 2168. Unique Substrings With Equal Digit Frequency
# 14JAN24
########################################################
class Solution:
    def equalDigitFrequency(self, s: str) -> int:
        '''
        we need to count unique substrings
        inputs are small enough to find all unique
        we just need a fast way to compute the digit counts -> precompute
        need pref_counts[i] for all digits 1 to 9
        then 
        '''
        unique = set()
        N = len(s)
        for start in range(N):
            substring = ""
            counts = Counter()
            for end in range(start,N):
                substring += s[end]
                counts[s[end]] += 1
                #need all values in counts to be the same
                if len(set(counts.values())) == 1:
                    unique.add(substring)
        
        return len(unique)
    
#rolling hash
class Solution:
    def equalDigitFrequency(self, s: str) -> int:
        '''
        we can use rolling rash, compute hash of string in constant time from existing hash
        need prime number larger than alphabet
        need fast way to determine uniqe counts for string
        '''
        N = len(s)
        seen = set()
        mod = 10**9 + 7
        base = 11
        
        for i in range(N):
            counts = [0]*10
            s_hash = 0
            max_count = 0
            unique = 0
            for j in range(i,N):
                dig = ord(s[j]) - ord('0')
                #count unique digits
                unique += 1 if counts[dig] == 0 else 0
                counts[dig] += 1
                max_count = max(max_count,counts[dig])
                #rolling hahs O(1)
                s_hash = (s_hash*base + dig + 1) % mod
                #if substring has digit with equal frequency
                #we counted the unique number of digits
                #its length == unique_count times max_count
                if max_count*unique == j - i + 1:
                    seen.add(s_hash)
        
        return len(seen)
        
#####################################################
# 2813. Maximum Elegance of a K-Length Subsequence
# 16JUN24
#####################################################
#not quite
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        '''
        elegance of a subsequence is total_profit + distinct_categories**2
        where total_profit is sum of all profits and distinct_categories is number of distinct cats
        find max elegance  from all subseqeucnes with size k
        total profit will be increasing, so it makes sense to take the most profitibale items
            but we can get mor elegane if we use unique categories
        
        select the first k items,
        then for the remaing n-k items, try replacing the candidate set using the current tiem
        keep the k items in a heap, then if we can go up in score swap them
        when we add in a item, we can keep track of the cateogires in a set, in a hashmap, counting the categories
        elegance goes up by len(mapp)**2
        '''
        items.sort(key = lambda x: -x[0])
        klargest = [] #sum at the end
        mapp = defaultdict(int)
        N = len(items)
        
        for prof,cat in items[:k]:
            heapq.heappush(klargest, (prof,cat))
            mapp[cat] += 1
            
        for prof,cat in items[k:]:
            #if we can add this new cat in
            if cat not in mapp:
                #check count of minim
                if mapp[klargest[0][1]] > 1:
                    mapp[klargest[0][1]] -= 1
                    mapp[cat] += 1
                    heapq.heappop(klargest)
                    heapq.heappush(klargest, (prof,cat))
                    
        
        max_profit = 0
        for prof,cat in klargest:
            max_profit += prof
        
        return max_profit + len(mapp)**2
            
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        '''
        elegance of a subsequence is total_profit + distinct_categories**2
        where total_profit is sum of all profits and distinct_categories is number of distinct cats
        find max elegance  from all subseqeucnes with size k
        total profit will be increasing, so it makes sense to take the most profitibale items
            but we can get mor elegane if we use unique categories
        
        select the first k items,
        then for the remaing n-k items, try replacing the candidate set using the current tiem
        keep the k items in a heap, then if we can go up in score swap them
        when we add in a item, we can keep track of the cateogires in a set, in a hashmap, counting the categories
        elegance goes up by len(mapp)**2
        '''
        items.sort(reverse=True, key=lambda x: x[0])
        
        # Use a min-heap to store the minimum elements at the top
        minHeap = []
        s = set()
        total_sum = 0
        
        # Process the first k elements
        for i in range(k):
            total_sum += items[i][0]
            if items[i][1] in s:
                heapq.heappush(minHeap, items[i][0])
            else:
                s.add(items[i][1])
        
        # Calculate the initial elegance score
        n = len(s)
        max_elegance = total_sum + n * n
        
        # Process the remaining elements
        for i in range(k, len(items)):
            if items[i][1] not in s and minHeap:
                s.add(items[i][1])
                total_sum -= heapq.heappop(minHeap)
                total_sum += items[i][0]
                n = len(s)
                max_elegance = max(max_elegance, total_sum + n * n)
        
        return max_elegance
    
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        '''
        use set and heap to track duplicates
        '''
        items.sort(key = lambda x: -x[0])
        
        # Use a min-heap to store the minimum elements at the top
        minHeap = []
        s = set()
        total_sum = 0
        
        # Process the first k elements
        for i in range(k):
            total_sum += items[i][0]
            #if ive already seen this, ad dto heap so we know it can be elemenint
            if items[i][1] in s:
                heapq.heappush(minHeap, items[i][0])
            #otherwise first occurence
            else:
                s.add(items[i][1])
        
        # Calculate the initial elegance score
        n = len(s)
        max_elegance = total_sum + n * n
        
        # Process the remaining elements
        for i in range(k, len(items)):
            if items[i][1] not in s and minHeap:
                s.add(items[i][1])
                total_sum -= heapq.heappop(minHeap)
                total_sum += items[i][0]
                n = len(s)
                max_elegance = max(max_elegance, total_sum + n * n)
        
        return max_elegance
                    
#usings stack
class Solution:
    def findMaximumElegance(self, items: List[List[int]], k: int) -> int:
        '''
        use set and heap to track duplicates
        '''
        items.sort(key = lambda x: -x[0])
        
        # Use a min-heap to store the minimum elements at the top
        stack = []
        s = set()
        total_sum = 0
        
        # Process the first k elements
        for i in range(k):
            total_sum += items[i][0]
            #if ive already seen this, ad dto heap so we know it can be elemenint
            if items[i][1] in s:
                stack.append(items[i][0])
            #otherwise first occurence
            else:
                s.add(items[i][1])
        
        # Calculate the initial elegance score
        n = len(s)
        max_elegance = total_sum + n * n
        
        # Process the remaining elements
        for i in range(k, len(items)):
            if items[i][1] not in s and stack:
                s.add(items[i][1])
                total_sum -= stack.pop()
                total_sum += items[i][0]
                n = len(s)
                max_elegance = max(max_elegance, total_sum + n * n)
        
        return max_elegance
                    
#############################################
# 633. Sum of Square Numbers (REVISTED)
# 17JUN24
#############################################
#using square root function, 
#cheeky way to check if float == int for some number x is x == int(x)
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''
        two pointers starting at 0 and 2**31 - 1, then advance 
        if we fix a, then we can look for b
        a^2 + b ^2 = c
        b = (c - a^2)^0.5
        now we can just check if b == int(b)
        '''
        for a in range(0,int(c**.5)+1):
            b = (c - a*a)**.5
            if b == int(b):
                return True
        
        return False
    
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''
        two pointers starting at 0 and 2**31 - 1, then advance 
        if we fix a, then we can look for b
        a^2 + b ^2 = c
        b = (c - a^2)^0.5
        
        check if perfect square
        the sqaure of nth positive integer can be represented as a sum of the frist odd positives
        n**2 = \sum_{i=1}^{n} (2*i-1)
        '''
        a = 0
        while a*a <= c:
            b = c - a*a
            i = 1
            sum_odds = 0
            while (sum_odds < b):
                sum_odds += i
                i += 2
            if sum_odds == b:
                return True
            a += 1
        
        return False
    
#binary search
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        '''
        instead of usinr sqrt funtion we can also use binary search
        '''
        for a in range(0,int(c**.5)+1):
            b = (c - a*a)**.5
            if self.bin_search(0,int(b),int(b)):
                return True
        
        return False
    
    def bin_search(self, left, right,n):
        if left > right:
            return False
        mid = left + (right - left) // 2
        if mid*mid == n:
            return True
        if mid*mid > n:
            return self.bin_search(left,mid-1,n)
        return self.bin_search(left + 1, right,n)
    
###########################################
# 330. Patching Array (REVISITED)
# 17JUN24
###########################################
class Solution:
    '''
    convert nums into intervals
     0, 1, 2, 4, 5, 12 is [[0, 2], [4, 5], [12, 12]]
     the find smallest numbers to add
    '''
    def merge(self, intervals):
        intervals.sort(key=lambda x: x[0])
        merged = []
        
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0] - 1:
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged


    def minPatches(self, nums, n):
        ints, patches = [[0,0]], 0
        for num in nums:
            ints = self.merge(ints + [[i+num, j+num] for i,j in ints])

        while ints[0][1] < n:
            ints = self.merge(ints + [[i+ints[0][1]+1, j+ints[0][1]+1] for i,j in ints])
            patches += 1

        return patches

############################################################
# 1798. Maximum Number of Consecutive Values You Can Make
# 17JUN24
##########################################################
class Solution:
    def getMaximumConsecutive(self, coins: List[int]) -> int:
        '''
        sort and count what sums i can make
        sums needs to start at 0
        '''
        coins.sort()
        curr_sum = 0
        for c in coins:
            if curr_sum + 1 >= c:
                curr_sum += c
            else:
                return curr_sum + 1
        
        return curr_sum + 1
    
##################################################
# 2410. Maximum Matching of Players With Trainers
# 18JUN24
##################################################
class Solution:
    def matchPlayersAndTrainers(self, players: List[int], trainers: List[int]) -> int:
        '''
        sort trainers, then use binary search on each player
        check if we can find a trainer to matches
        sort both players and trainers, then use two pointers
        after sorting
        [4,7,9]
        [2,5,8,8]
        '''
        players.sort()
        trainers.sort()
        pairs = 0
        i,j = 0,0
        while i < len(players) and j < len(trainers):
            if players[i] <= trainers[j]:
                pairs += 1
                i += 1
                j += 1
            elif players[i] > trainers[j]:
                j += 1
        
        return pairs
    
##################################################
# 1482. Minimum Number of Days to Make m Bouquets
# 19JUN24
##################################################
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        '''
        this binary search on workable solution paraigm
        need linear time function to check if we wan make m bouqets with k flowers given some day
        '''
        left = min(bloomDay)
        right = max(bloomDay)
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            #try to make m boquets
            candidate_boquets = self.func(bloomDay,m,k,mid)
            if candidate_boquets >= m:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
    
    def func(self, flowers, m,k,day):
        boquets_made = 0
        curr_boquet = 0
        
        for f in flowers:
            if f <= day:
                #increment
                curr_boquet += 1
                if curr_boquet == k:
                    boquets_made += 1
                    curr_boquet = 0
            #cant uese
            else:
                curr_boquet = 0
        
        return boquets_made
    
####################################################
# 1552. Magnetic Force Between Two Balls (REVISTED)
# 20JUN24
###################################################
class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        '''
        binary search for a workable force
        need function to place at least m balls
        '''
        left = 0
        right = max(position) - min(position)
        position.sort()
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            candidate_balls_used = self.func(position,m,mid)
            #if we can do, save ans and look for better onw
            if candidate_balls_used >= m:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans
    
    def func(self, baskets, m, min_force):
        balls_used = 1
        last_ball_position = baskets[0]
        for b in baskets:
            mag_force = abs(b - last_ball_position)
            if mag_force >= min_force:
                balls_used += 1
                last_ball_position = b
        
        return balls_used

#############################################
# 2594. Minimum Time to Repair Cars
# 20JUN24
#############################################
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        '''
        a mechanic an repair n cars in r*n^2 minutes
        use all mechanisn and check minutes
        n = sqrt(minutes/r)
        int((mins / r)**.5) , cant do partial cars per mechanic, they need to fix it
        now what are the bounds?
            lower bound is 1 minute
            uppower bound just max out each mechanic, i.e find max time each mecanic needs to fix cars
        '''
        left = 1
        right = sum(ranks)*cars*cars
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            cand_cars_repaired = self.func(ranks,cars,mid)
            if cand_cars_repaired >= cars:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
        #for m in range(1,20):
        #    print(m,self.func(ranks,cars,m))
    
    #func to check all cars repaired given minutes
    def func(self, ranks, cars,mins):
        cars_repaired = 0
        for r in ranks:
            cars_repaired += int((mins / r)**.5)
        
        return cars_repaired

############################################
# 1052. Grumpy Bookstore Owner
# 21JUN24
############################################
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        '''
        1 is grumpy, 0 is not grumpy
        n minutes, where n == len(customres) == len(grumpy)
        first compute the overall satisfaction as is,
        then second pass to see if we can got more satisfactiion using sliding window
            i.e find additional max satifaction
        '''
        n = len(customers)
        max_satisfaction = 0
        for i in range(n):
            if grumpy[i] == 0:
                max_satisfaction += customers[i]
            
        curr_additional = 0
        for i in range(minutes):
            if grumpy[i] == 1:
                curr_additional += customers[i]
        
        max_additional = curr_additional
        for i in range(minutes,n):
            if grumpy[i] == 1:
                curr_additional += customers[i]
            if grumpy[i-minutes] == 1:
                curr_additional -= customers[i-minutes]
            
            max_additional = max(max_additional,curr_additional)
        
        return max_satisfaction + max_additional
                
#one pass
class Solution:
    def maxSatisfied(self, customers: List[int], grumpy: List[int], minutes: int) -> int:
        '''
        we can do this one pass just track both max_satisfaction and max_additional
        '''
        n = len(customers)
        max_satisfaction = 0
        max_additional = 0
        curr_additional = 0
        
        
        for i in range(n):
            if i >= minutes and grumpy[i-minutes] == 1:
                curr_additional -= customers[i-minutes]
            if grumpy[i] == 1:
                curr_additional += customers[i]
            if grumpy[i] == 0:
                max_satisfaction += customers[i]
            
            max_additional = max(max_additional, curr_additional)
        
        
        return max_satisfaction + max_additional

########################################
# 1248. Count Number of Nice Subarrays
# 22JUN24
#########################################
#brute force
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        '''
        i can use a count array to keep track of of the number of odd integers up to i
        brute force would be to check all (i,j)
        better would be to find complement count 
        '''
        N = len(nums)
        counts = [0]
        for num in nums:
            if num % 2 == 1:
                counts.append(counts[-1] + 1)
            else:
                counts.append(counts[-1])
        
        ans = 0
        for end in range(1,N+1):
            for start in range(end):
                if counts[end] - counts[start] == k:
                    ans += 1
        return ans
    
#sub array sum == k
#idea really stems from equation of prefsum for all start and ends
class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        '''
        i can use a count array to keep track of of the number of odd integers up to i
        brute force would be to check all (i,j)
        better would be to find complement count like subarray sum == k
        '''
        N = len(nums)
        counts = defaultdict(int)
        counts[0] = 1
        
        ans = 0
        curr_count = 0
        for num in nums:
            curr_count += num % 2 == 1
            if (curr_count - k) in counts:
                ans += counts[curr_count - k]
            
            counts[curr_count] += 1
        
        return ans

############################################
# 1580. Put Boxes Into the Warehouse II
# 18JUN24
############################################
#nice try
class Solution:
    def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
        '''
        keep mins on left and mins on right, 
        then sort boxes and check like in warehouse 1
        '''
        N = len(warehouse)
        mins_left = warehouse[:]
        for i in range(1,N):
            mins_left[i] = min(mins_left[i-1], warehouse[i])
        
        mins_right = warehouse[:]
        for i in range(N-2,-1,-1):
            mins_right[i] = min(mins_right[i+1],warehouse[i])
        
        boxes.sort()
        count = 0
        for i in range(N):
            if count < len(boxes):
                if boxes[count] <= mins_left[i]:
                    count += 1
                
        for i in range(N):
            if count < len(boxes):
                if boxes[count] <= mins_right[N-i-1]:
                    count += 1
        
        return count + 1

#try adding largest from both ends
class Solution:
    def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
        '''
        we can oush the biggest box from eather left or right
        intution, idea is that if we fit the largest in first, we certainly could have fit it something smaller before hand
        since we can insert from both ends, we use two pointers for leftmost and rightmost rooms
            we try to fit the lefmots room or the rightmost room with the larest box
        
        idea:
            by iterating over boxes from largest to smallest, we ensure that smaller boxes have a chance to be placed
            even if largesr boxes cannot fit in the remaining rooms
        '''
        boxes.sort(reverse = True)
        left = 0
        right = len(warehouse) - 1
        
        ans = 0
        for i in range(len(boxes)):
            if left <= right:
                if boxes[i] <= warehouse[left]:
                    left += 1
                    ans += 1
                elif boxes[i] <= warehouse[right]:
                    right -= 1
                    ans += 1
        
        return ans
    
#like warehouse I
class Solution:
    def maxBoxesInWarehouse(self, boxes: List[int], warehouse: List[int]) -> int:
        '''
        like warehouse1, we need to find the limiting heights from boths sides
        only then we can slide and try addin in smallest
        we first find the min heights going left to right
        '''
        n = len(warehouse)
        left_mins = warehouse[:]
        for i in range(1,n):
            left_mins[i]= min(left_mins[i-1],warehouse[i])
        
        right_mins = warehouse[:]
        for i in range(n-2,-1,-1):
            right_mins[i] = min(right_mins[i+1], warehouse[i])
        
        useable_heights = [0]*n
        for i in range(n):
            useable_heights[i] = max(left_mins[i],right_mins[i])
        
        #sort effetive heights!, because we can add from left or right
        #note we didn't do this in part 1
        useable_heights.sort()
        #treat like warehouse 1
        boxes.sort()
        count = 0
        for h in useable_heights:
            if count < len(boxes) and boxes[count] <= h:
                count += 1
        
        return count

###################################################################################
# 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
# 23JUN24
###################################################################################
#use sortedList
from sortedcontainers import SortedList
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        '''
        two pointer technique, we need to make it as long as possible need to keep expanding
        need fast way to find min and maximum between some interval (i,j) inclusive
        then we can use two poiniter effeciently
        need to use Sorted List to represent the subarray, then we can check
        use .remove
        '''
        sl = SortedList([])
        ans = 1
        left = 0
        N = len(nums)
        for right in range(N):
            sl.add(nums[right])
            while left < right and sl[-1] - sl[0] > limit:
                sl.remove(nums[left])
                left += 1
            
            if sl[-1] - sl[0] <= limit:
                ans = max(ans,len(sl))
        
        return ans
    
#two heap solution
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        '''
        instead of using multiset paradigm, or sorted container, wee need to use two heaps
        one heap stores max in current window and the other stores min in current window
        in addition, we store the indices in the heaps
        we need to keep the heaps updates by deleting the elements outside the new window after moving the left pointers
        '''
        ans = 1
        N = len(nums)
        
        max_heap = []
        min_heap = []
        
        left = 0
        
        for right,num in enumerate(nums):
            heapq.heappush(min_heap, (num,right))
            heapq.heappush(max_heap, (-num,right))
            
            while -max_heap[0][0] - min_heap[0][0] > limit:
                #move left
                #we need the left most one
                #all we know if the number's value in the array, we dont know where
                left = min(max_heap[0][1], min_heap[0][1]) + 1
                
                #we need to exclude eleemnts oustide the range, or less than left
                while max_heap[0][1] < left:
                    heapq.heappop(max_heap)
                
                while min_heap[0][1] < left:
                    heapq.heappop(min_heap)
            
            ans = max(ans, right - left + 1)
        
        return ans
    
#montonic deque
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        '''
        instead of two heaps we can can uses two deques
            one is motonic decreasing, so the largest it always first
            the other is montonic increasing, so the smallest is always first
        
        adding to the deques at each step,
        how do we shrink?
            popleft 
        
        before adding we need to maintain the invariants in the dequest
        '''
        max_q = deque([])
        min_q = deque([])
        left = 0
        ans = 1
        n = len(nums)
        
        for right,num in enumerate(nums):
            #maintain invariants
            while max_q and max_q[-1] < num:
                max_q.pop()
            
            max_q.append(num)
            
            while min_q and min_q[-1] > num:
                min_q.pop()
            
            min_q.append(num)
            
            #srhink
            while max_q[0] - min_q[0] > limit:
                #remove the eleemnts that are out of the current window, tha are not the max of the minimum
                if max_q[0] == nums[left]:
                    max_q.popleft()
                if min_q[0] == nums[left]:
                    min_q.popleft()
                left += 1
            
            ans = max(ans, right - left + 1)
        
        return ans

##################################################
# 995. Minimum Number of K Consecutive Bit Flips
# 24JUN24
#################################################
#brute force
class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        '''
        we need to flip so that nums has no zeros, or sum(nums) == len(nums)
        we can flip a lenght k subarray, and flips bits
        return min number of flips
        greedy?
        in order of this to work, on the last flip, there had to b a subarray of [0]s of length k
        there could be multiple ways
        if we could use any k, then we just flip the streaks of zeros
        
        intution
            order of flips does not mater
            find what indices to flip regardless of sequence
                does not matter if we flip at indices [0,4,5] or [4,0,5]
            number of times index is flipped, determine its values
            if its a 1, then even flips, sends it back to 1, odd flips, leaves it at 1
            opposite if oringially 0
                i.e if flipped an odd number of times, its value is flipped, remains same if even number of flips

        #brute force, for each index i, flip i to k-1, then check all are 1s, then reduce the indices
        '''
        zero_idxs = []
        n = len(nums)
        for i in range(n):
            if nums[i] == 0:
                zero_idxs.append(i)
                if i + k - 1 < n:
                    for j in range(i,i+k):
                        nums[j] = 1 - nums[j]
        
        #check if we can do
        if sum(nums) != n:
            return -1
        
        #reduce indices
        return len(zero_idxs)
        
class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        '''
        we need to flip so that nums has no zeros, or sum(nums) == len(nums)
        we can flip a lenght k subarray, and flips bits
        return min number of flips
        greedy?
        in order of this to work, on the last flip, there had to b a subarray of [0]s of length k
        there could be multiple ways
        if we could use any k, then we just flip the streaks of zeros
        
        intution
            order of flips does not mater
            find what indices to flip regardless of sequence
                does not matter if we flip at indices [0,4,5] or [4,0,5]
            number of times index is flipped, determine its values
            if its a 1, then even flips, sends it back to 1, odd flips, leaves it at 1
            opposite if oringially 0
                i.e if flipped an odd number of times, its value is flipped, remains same if even number of flips
            
        we can sort the sequence by increasing index, once sorted we minimize the size using parity
        say for example we flip at indidces
        [0,1,2,4,5,6,5,6,7], and k = 3
        its relly just [0..2] -> [4..6] -> [5..7],
        we can write as [0,4,5]
        indices are sorted, so subsequent flips with larger indices cannot alter the value at prior indices
        
        if nums[0] = 0 and is not in the flip sequence, it should remain 0 in the final result
        if nums[0] = 1 and 0 is in the flip sequence, nums[0] -> 0 in final result
        
        if nums[i] = 0, then i must be present in t he flip sequence [i to i+k-1]
        if nums[i] == 1, the i must not be in the sequence, and we do not flip [0 to k-1]
        
        we can use isFlipped array to keep trakc of the indices where kth bit flip needs to jappen
        conditions
            if flipped is false and nums[i] = 0, flip is required
            if flippes is true and nums[i] = 1, flip is required
        '''
        n = len(nums)
        isFlipped = [False]*n
        flips_needed = 0
        prev_flips = 0
        
        for i in range(n):
            if i >= k:
                if isFlipped[i-k]:
                    prev_flips -= 1
            
            if prev_flips % 2 == nums[i]:
                if i + k > n:
                    return -1
                
                prev_flips += 1
                flips_needed += 1
                isFlipped[i] = True
        
        return flips_needed
    
class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        '''
        https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/discuss/238609/JavaC%2B%2BPython-One-Pass-and-O(1)-Space
        theres only one way to flip A[0] and A[0] will tell us if we need to flip from A[0] ~A[k-1]
        really its just an understaning the mechanics if k bit flips
        isFlipped array 1 or 0 if we have flipped from i-k
        maintain flipped variable iff current bit is flipped
        '''
        N = len(nums)
        flipped = 0
        res = 0
        isFlipped = [False]*N
        
        for i in range(N):
            if i >= k:
                #flip it back
                flipped ^= isFlipped[i-k]
            
            if flipped == nums[i]:
                if i + k > N:
                    return -1
                
                isFlipped[i] = True
                flipped ^= 1
                res += 1
        
        return res
    
#deque solution
class Solution:
    def minKBitFlips(self, nums: List[int], k: int) -> int:
        '''
        instead of using isFlipped array if size len(nums)
        we maintin q of size k
        '''
        N = len(nums)
        dq = deque([])
        flipped = 0
        res = 0
        
        for i,num in enumerate(nums):
            if i >= k:
                flipped ^= dq[0]
            
            if flipped == num:
                if i + k > N:
                    return -1
                dq.append(1)
                flipped ^= 1 #could also do 1 - flipped
                res += 1
            else:
                dq.append(0)
            if len(dq) > k:
                dq.popleft()
        
        return res
    
#################################################
# 1038. Binary Search Tree to Greater Sum Tree
# 25JUN24
#################################################
#three pass
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        '''
        make nodes value == to the some of nodes >= to node.val
        to visit nodes in in sorted order we can use inorder
        i can precompute the sum in order
        thens econd pass, subtracting pref_sum in order
        its original key + sum vals greater
        '''
        nums = self.inorder(root)
        total = sum(nums)
        sum_less_than = 0
        for i in range(len(nums)):
            sum_less_than += nums[i]
            nums[i] = total - sum_less_than + nums[i]
            
        #now do in order2
        idx = [0]
        self.inorder2(root,nums,idx)
        return root
        
    
    def inorder(self, node):
        if not node:
            return []
        
        left = self.inorder(node.left)
        right = self.inorder(node.right)
        return left + [node.val] + right
    
    def inorder2(self,node,nums,idx):
        if not node:
            return
        self.inorder2(node.left,nums,idx)
        node.val = nums[idx[0]]
        idx[0] += 1
        self.inorder2(node.right,nums,idx)

#two pass
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        '''
        prescompute the sum then keep track of sumless than
        '''
        total_sum = self.sumTree(root)
        sum_less_than = [0]
        self.inorder(root,sum_less_than,total_sum)
        return root
    
    def sumTree(self, node):
        if not node:
            return 0
        return node.val + self.sumTree(node.left) + self.sumTree(node.right)
    
    def inorder(self,node,sum_less_than,total_sum):
        if not node:
            return
        self.inorder(node.left,sum_less_than,total_sum)
        sum_less_than[0] += node.val
        node.val = total_sum - sum_less_than[0] + node.val
        self.inorder(node.right,sum_less_than,total_sum)

#one pass?
#need to reverse inorder the get values larger
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        '''
        one pass, need to use reverse in order traversal and get larger values on the fly
        
        '''
        node_sum_greater = [0]
        self.postorder(root,node_sum_greater)
        return root
    
    def postorder(self,node, node_sum_greater):
        if not node:
            return
        self.postorder(node.right, node_sum_greater)
        node_sum_greater[0] += node.val
        node.val = node_sum_greater[0]
        self.postorder(node.left,node_sum_greater)

#iterative postorder
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        '''
        iterative
        keep stack and process while we have stuff in stack or node isnt null
        '''
        stack = []
        curr = root
        greater_sum = 0
        
        while len(stack) > 0 or curr != None:
            while curr != None:
                stack.append(curr)
                curr = curr.right
            
            curr = stack.pop()
            greater_sum += curr.val
            curr.val = greater_sum
            curr = curr.left
        
        return root

#constant space, morris traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        '''
        we can do morris traversal in reverse -> we just swap the left ans the rights
        in original morris traversal, we use inorder predecessor and thread when we can and sever when already threded
        for reverse inorder we need inorder successor

        thread back to succ, when we can,
        sever when its already threaded
        '''
        total = 0
        curr = root
        while curr != None:
            #no right subtree, process current node
            if curr.right == None:
                total += curr.val
                curr.val = total
                curr = curr.left
            else:
                #find inorder succ
                succ = self.get_successor(curr)
                #if there is no left subtree (or right subtree) we thread back
                if succ.left == None:
                    succ.left = curr
                    curr = curr.right
                #otherwise its already threaded
                #i.e if there is a left subtree, it was connect, so we need to sever
                else:
                    succ.left = None
                    total += curr.val
                    curr.val = total
                    curr = curr.left
        
        return root
        
    def get_successor(self,node):
        succ = node.right
        #left as far as we can and as long as its not threaded
        while succ.left != None and succ.left != node:
            succ = succ.left
        
        return succ
    
###########################################
# 1382. Balance a Binary Search Tree
# 26JUN24
############################################
#two pass
#Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        
        self.right = right
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        '''
        get the inorder and rebuild it
        '''
        vals = self.inorder(root)
        return self.build(vals,0,len(vals)-1)
    
    def inorder(self,root : TreeNode) -> List[int]:
        if not root:
            return []
        left = self.inorder(root.left)
        right = self.inorder(root.right)
        return left + [root.val] + right
    
    def build(self, vals : List[int], left : int, right : int) -> TreeNode:
        if left > right:
            return None
        
        mid = left + (right - left) // 2
        node = TreeNode(val = vals[mid])
        node.left = self.build(vals,left,mid-1)
        node.right = self.build(vals,mid+1,right)
        return node
        
#balacing in place
class Solution:
    def balanceBST(self, root: TreeNode) -> TreeNode:
        if not root:
            return None

        # Step 1: Create the backbone (vine)
        # Temporary dummy node
        vine_head = TreeNode(0)
        vine_head.right = root
        current = vine_head
        while current.right:
            if current.right.left:
                self.right_rotate(current, current.right)
            else:
                current = current.right

        # Step 2: Count the nodes
        node_count = 0
        current = vine_head.right
        while current:
            node_count += 1
            current = current.right

        # Step 3: Create a balanced BST
        m = 2 ** math.floor(math.log2(node_count + 1)) - 1
        self.make_rotations(vine_head, node_count - m)
        while m > 1:
            m //= 2
            self.make_rotations(vine_head, m)

        balanced_root = vine_head.right
        # Delete the temporary dummy node
        vine_head = None
        return balanced_root

    # Function to perform a right rotation
    def right_rotate(self, parent: TreeNode, node: TreeNode):
        tmp = node.left
        node.left = tmp.right
        tmp.right = node
        parent.right = tmp

    # Function to perform a left rotation
    def left_rotate(self, parent: TreeNode, node: TreeNode):
        tmp = node.right
        node.right = tmp.left
        tmp.left = node
        parent.right = tmp

    # Function to perform a series of left rotations to balance the vine
    def make_rotations(self, vine_head: TreeNode, count: int):
        current = vine_head
        for _ in range(count):
            tmp = current.right
            self.left_rotate(current, tmp)
            current = current.right

######################################################
# 2743. Count Substrings Without Repeating Character
# 26JUN24
#####################################################
#nice try
class Solution:
    def numberOfSpecialSubstrings(self, s: str) -> int:
        '''
        if we find a special substring of length k, then there are k*(k+1) // 2 substrings that are also special
        '''
        ans = 0
        left = 0
        
        chars = set()
        
        for right,ch in enumerate(s):
            if ch not in chars:
                chars.add(ch)
                ans += right - left + 1
            else:
                while left < right and s[left] == ch:
                    chars.remove(s[left])
                    left += 1
                chars.add(ch)
                ans += right - left + 1
        return ans
    
class Solution:
    def numberOfSpecialSubstrings(self, s: str) -> int:
        '''
        keep count mapp and add
        '''
        counts = Counter()
        left = 0
        ans = 0
        
        for right,ch in enumerate(s):
            counts[ch] += 1
            
            #shrink back until the number we added is 1 again
            
            while counts[ch] > 1:
                counts[s[left]] -= 1
                left += 1
            
            ans += right - left + 1
    
        return ans

class Solution:
    def numberOfSpecialSubstrings(self, s: str) -> int:
        '''
        keep count mapp and add
        keep track of last seen char indexx and get right most last seen
        '''
        last_seen_char = {}
        left = 0
        ans = 0
        
        for right,ch in enumerate(s):
            if ch in last_seen_char:
                left = max(left,last_seen_char[ch] + 1)
            
            last_seen_char[ch] = right
            ans += right - left + 1
    
        return ans
    
###################################
# 1791. Find Center of Star Graph
# 27JUN24
###################################
class Solution:
    def findCenter(self, edges: List[List[int]]) -> int:
        '''
        constant time,
        a center must exsist, and if it exsists the center must be a common node 
        pick any two edges and return the node thats in both of them
        '''
        a,b = edges[0],edges[1]
        if a[0] in b:
            return a[0]
        return a[1]
    
##########################################################
# 1790. Check if One String Swap Can Make Strings Equal
# 27JUN24
##########################################################
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        '''
        if chars are already equal we dont need to swap them
        we are only allowed one swap, so there can only be 2 positions where chars dont match
        '''
        unequal_spots = 0
        a = set()
        b = set()
        
        for u,v in zip(s1,s2):
            if u != v:
                unequal_spots += 1
                a.add(u)
                b.add(v)
        
        if unequal_spots not in (0,2):
            return False
        return a == b

class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        '''
        need to check that frequencies are the same
        use sum difference check
        then check uneuqal spots
        '''
        
        counts = Counter()
        unequal_spots = 0
        
        for u,v in zip(s1,s2):
            counts[u] += 1
            counts[v] -= 1
            if u != v:
                unequal_spots += 1
        
        for k,v in counts.items():
            if v != 0:
                return False
        
        return unequal_spots  in (0,2)
    
################################################
# 2285. Maximum Total Importance of Roads
# 28JUN24
################################################
#lucky guess lmaoo
class Solution:
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        '''
        graph is undirected
        we need maximum total imporance of all roads
        the cities that appear larger in frequency should be given a higher weight from to 1 to n
        count indegree? 
        but how to break ties
        '''
        indegree = [0]*n
        for u,v in roads:
            indegree[u] += 1
            indegree[v] += 1
        
        #pair with indices
        pairs = [(i,v) for i,v in enumerate(indegree)]
        pairs.sort(key = lambda x : -x[1])
        importances = [0]*n
        for rank,(i,v) in enumerate(pairs):
            importances[i] = n - rank
        
        ans = 0
        for u,v in roads:
            ans += importances[u] + importances[v]
        
        return ans
    
class Solution:
    def maximumImportance(self, n: int, roads: List[List[int]]) -> int:
        '''
        we dont need to to do three passes, we can just multiply by indegree
        say we have roads [a,b], [c,d], [a,c], [a,d]
        we need a + b + c + d + a + c + a + d =
        (a + a + a) + (b) + (c + c) + (d + d) to be maximum
        order by indegree then use largest ranks
        '''
        indegree = [0]*n
        for u,v in roads:
            indegree[u] += 1
            indegree[v] += 1
            
        max_importance = 0
        indegree.sort()
        rank = 1
        
        for d in indegree:
            max_importance += rank*d
            rank += 1
        
        return max_importance
    
##################################################
# 1315. Sum of Nodes with Even-Valued Grandparent
# 28JUN24
##################################################
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        '''
        grand parent is two steps away
        keep two variables parent val and grand parent val
        '''
        ans = [0]
        self.dfs(root,None,None,ans)
        return ans[0]
    
    def dfs(self, node, parentVal, grandParentVal,ans):
        if not node:
            return
        if grandParentVal and grandParentVal % 2 == 0:
            ans[0] += node.val
        self.dfs(node.left,node.val,parentVal,ans)
        self.dfs(node.right,node.val,parentVal,ans)

#bottom up
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        # you had it before
        def dp(node, parent, grandParent):
            if not node:
                return 0
            
            left = dp(node.left,node.val,parent)
            right = dp(node.right,node.val,parent)
            if grandParent % 2 == 0:
                return left + right + node.val
            return left + right
        
        return dp(root, -1,-1)
    
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        #two down
        def dp(node, is_parent_even):
            if not node:
                return 0
            
            left = dp(node.left, node.val % 2 == 0)
            right = dp(node.right, node.val % 2 == 0)
            sum_so_far = left + right
            #need sum for this tree as well as additional  sum if granparent is even
            if is_parent_even:
                if node.left:
                    sum_so_far += node.left.val
                if node.right:
                    sum_so_far += node.right.val
            
            return sum_so_far
        
        return dp(root,False)
        
###########################################################
# 2192. All Ancestors of a Node in a Directed Acyclic Graph
# 29JUN24
############################################################
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        '''
        build graph and start on nodes with zero indegree
        take too long to dfs/bfs on each one,
        top sort!
        '''
        edges.sort()
        graph = defaultdict(list)
        indegree = [0]*n
        
        for u,v in edges:
            graph[u].append(v)
            indegree[v] += 1
        
        ans = [set() for _ in range(n)]
        
        def dfs(starting_node,node,graph,seen,ans):
            seen.add(node)
            for neigh in graph[node]:
                ans[neigh].add(starting_node)
                if neigh not in seen:
                    dfs(starting_node,neigh,graph,seen,ans)
        
        for i in range(n):
            seen = set()
            dfs(i,i,graph,seen,ans)
        
        for i in range(n):
            ans[i] = sorted(list(ans[i]))
        
        return ans
            
#without set
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        '''
        build graph and start on nodes with zero indegree
        take too long to dfs/bfs on each one,
        top sort!
        '''
        edges.sort()
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
        
        ans = [[] for _ in range(n)]
        
        def dfs(starting_node,node,graph,ans):
            for neigh in graph[node]:
                #no ancestor yet for this child node
                #or we have a new ancestor to add
                if not ans[neigh] or ans[neigh][-1] != starting_node:
                    ans[neigh].append(starting_node)
                    dfs(starting_node,neigh,graph,ans)
        
        for i in range(n):
            dfs(i,i,graph,ans)
        
        return ans
    
#top sort
class Solution:
    def getAncestors(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        '''
        we can first use top sort to get the order, and then for each node add its ancestor
        need to keep ans array and ancestor set list
        we need track each nodes ancestor
        '''
        edges.sort()
        graph = defaultdict(list)
        indegree = [0]*n
        
        for u,v in edges:
            graph[u].append(v)
            indegree[v] += 1
        

        ancestors_set = [set() for _ in range(n)]
        top_order = []
        
        q = deque([])
        for i in range(n):
            if indegree[i] == 0:
                q.append(i)
        

        while q:
            curr = q.popleft()
            top_order.append(curr)
            for neigh in graph[curr]:
                indegree[neigh] -= 1
                if indegree[neigh] == 0:
                    q.append(neigh)
        
        for node in top_order:
            for neigh in graph[node]:
                ancestors_set[neigh].add(node)
                ancestors_set[neigh].update(ancestors_set[node])
        
        for i in range(n):
            ancestors_set[i] = sorted(list(ancestors_set[i]))
        
        return ancestors_set
                
        
