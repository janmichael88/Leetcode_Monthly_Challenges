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
