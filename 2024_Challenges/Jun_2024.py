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
        