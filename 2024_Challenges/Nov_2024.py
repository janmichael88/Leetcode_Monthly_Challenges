###############################################
# 1957. Delete Characters to Make Fancy String
# 03NOV24
###############################################
class Solution:
    def makeFancyString(self, s: str) -> str:
        '''
        build on the fly
        '''
        ans = ""
        
        for ch in s:
            if len(ans) >= 2 and ans[-1] == ans[-2] == ch:
                continue
            ans += ch
        
        return ans
    
##############################################
# 2490. Circular Sentence
# 03NOV24
###############################################
class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        '''
        check i and i + 1
        for the last word, we need to check n - 1 with zero
        '''
        words = sentence.split(" ")
        N = len(words)
        for i in range(N):
            if i == N  - 1:
                if words[i][-1] != words[0][0]:
                    return False
            else:
                if words[i][-1] != words[i+1][0]:
                    return False
        
        return True
    
#cheese way on space
class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        '''
        we know the sentence is valid, so insteaf of splitting, we check on spaces
        and if index i is a space, we check i - 1 and i + 1
        '''
        N = len(sentence)
        for i in range(N):
            if sentence[i] == ' ' and sentence[i-1] != sentence[i+1]:
                return False
        
        return sentence[-1] == sentence[0]
    

########################################
# 2955. Number of Same-End Substrings
# 03NOV24
########################################
class Solution:
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        same ending substrings are the number of substrings that have the same chars at start and ends
        if there are t chars in a substrin, then there are t*(t-1)/2 same ending substrings
        for each index i, keep track of the frequence count
        then use partial sums for all 26 chars
        '''
        n = len(s)
        pref_counts = [Counter()]
        
        for i in range(n):
            curr_count = Counter()
            curr_count[s[i]] += 1
            for k,v in pref_counts[-1].items():
                curr_count[k] += v
            
            pref_counts.append(curr_count)
        
        
        ans = []
        for l,r in queries:
            temp = 0
            for i in range(26):
                char = chr(ord('a') + i)
                count = pref_counts[r+1][char] - pref_counts[l][char]
                temp += count*(count + 1) // 2
            
            ans.append(temp)
        
        return ans

#ussing array instead of count object
class Solution:
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        the counting idea comes from the combindation formula
        if we have k occruences of some character i
        we can place these any number of these occurence at some start and some end
        so its just
        k choose 2
        k*(k-1)/2 + k
        k*(k+1)/2
        
        insteaf of using counter objects we can just an array of size 26
        '''
        n = len(s)
        pref_counts = [[0]*26 for _ in range(n+1)]
        
        for i in range(n):
            pref_counts[i+1][ord(s[i]) - ord('a')] += 1
        
        for i in range(1,n+1):
            for j,count in enumerate(pref_counts[i-1]):
                pref_counts[i][j] += count
        
        ans = []
        for l,r in queries:
            temp = 0
            for i in range(26):
                count = pref_counts[r+1][i] - pref_counts[l][i]
                temp += count*(count + 1) // 2
            
            ans.append(temp)
        
        return ans
                
#binary search???
class Solution:
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        for each char in s, store its indices in a hashmap
        then for each query, look for its left and right positions for each character
        we want to find the first position of the character that is at or after the starting index
        the first poistion of the character that is beyong the ending range
        '''
        mapp = defaultdict(list)
        for i,s in enumerate(s):
            mapp[s].append(i)
        
        
        ans = []
        for l,r in queries:
            count = 0
            #do this for each character
            for indices in mapp.values():
                #find leftmost and right most
                #for left <=
                left = bisect.bisect_left(indices,l)
                #for right just greater than
                right = bisect.bisect_right(indices,r)
                num = right - left
                count += (num)*(num + 1) // 2
            
            ans.append(count)
        
        return ans

##########################################
# 1513. Number of Substrings With Only 1s
# 04NOV24
###########################################
class Solution:
    def numSub(self, s: str) -> int:
        '''
        count the streaks
        if we have a streak of ones of length k, then there are k*(k+1) / 2 of subtrings
        find the streaks and compute
        '''
        ans = 0
        curr_streak = 0
        
        for ch in s:
            if ch == '0':
                ans += (curr_streak)*(curr_streak + 1) // 2
                curr_streak = 0
            else:
                curr_streak += 1
        
        ans += (curr_streak)*(curr_streak + 1) // 2
        return ans % (10**9 + 7)
    
#accumulating variant
class Solution:
    def numSub(self, s: str) -> int:
        '''
        count the streaks
        if we have a streak of ones of length k, then there are k*(k+1) / 2 of subtrings
        find the streaks and compute
        '''
        ans = 0
        curr_streak = 0
        
        for ch in s:
            if ch == '1':
                curr_streak += 1
            else:
                curr_streak = 0
            
            ans = (ans + curr_streak) % (10**9 + 7)
        
        return ans % (10**9 + 7)
    
#################################################################
# 2914. Minimum Number of Changes to Make Binary String Beautiful
# 05NOV24
##################################################################
class Solution:
    def minChanges(self, s: str) -> int:
        '''
        we can loop through each cahr in the string and keep track of the current sequence'ss length
        if we reeach the end of a sequence and its length is even, we can move on to the next sequence,
            basically count all streaks of zero's or ones
        
        if its an odd length, we will flip the last bit of that sequence to make it even
        flipping the last bit will an additional bit to the next sequcne, so count here is 1 when flip
        
        prrof by contradiction
        assume there exists a better solution that requires few flips by flipping some bit other than the lasst
        call this S1, which is of size k and is odd
        call another sequence S2
        S1 = b1 b2 b3 ... bk
        S2 is at b_{k+1}
        case 1 
            flip b_{k}
            it then S1 = b1 .. b_{k-1}
            and b_{k} becomes part of S2
            cosst is 1 flip
        
        case 2
            flip any number of bits where i < k
            this cost more than one flip
            but we already set S1 was optimal
        
        so assumption that there exists a better solution is false
        '''
        curr_char = s[0]
        streak_count = 0
        flips = 0
        
        for ch in s:
            if ch == curr_char:
                streak_count += 1
                continue
            
            if streak_count % 2 == 0:
                streak_count = 1
            else:
                streak_count = 0
                flips += 1
            
            curr_char = ch
        
        return flips
    
class Solution:
    def minChanges(self, s: str) -> int:
        '''
        the string is even length
        why does it need to be even though?
        whoops its not splitting, i can change any character in s to a 0 or a 1
        its beautiful if we can partition it into on or more substrings such that
        each had substring has even length and it only contains 1's or 0's
        check each block of size 2
        since each part consists of an even number of the same chars, we just check each block of size 2

        atomic unit of beautiful string is size 2
        if size 2 isn't beautfil, we need to make change to eithe the left or right pair

        even length means and possibilty of making it beautiful means that any substring that is beauitfil must be 00 or 11
        so we just count disimlar pairs
        '''
        n = len(s)
        ans = 0
        for i in range(0,n,2):
            if s[i] != s[i+1]:
                ans += 1
        
        return ans
    
#####################################################
# 3011. Find if Array Can Be Sorted
# 06NOV24
#####################################################
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        '''
        we are only allowed to swap two numbers if they have the same number of bits
        N^2 is allowed given the inputs
        for each number compute the number of set bits
        split the array into segments where each segment has the same number of bits
        largest element in prev segment should be smaller than current
        '''
        segments = []
        #from left to right largest num in prev must be smaller than the smallest num in next
        i = 0
        n = len(nums)
        
        
        while i < n:
            curr_segment = []
            while len(curr_segment) == 0 or (i < n and self.count_bits(curr_segment[-1]) == self.count_bits(nums[i])):
                curr_segment.append(nums[i])
                i += 1
            
            segments.append(curr_segment)
        
        m = len(segments)
        i = 0
        #print(segments)
        while i + 1 < m:
            prev = segments[i]
            next_ = segments[i+1]
            if max(prev) > min(next_):
                return False
            i += 1
        
        return True
        
    def count_bits(self,num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        
        return count
    

#bubble sort!, dont forget this
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        '''
        we can use bubble sort to try and sort
        we want to bubble left, i.e smaller eleements to the left
        if need to swap, but can't swap, return false
        otherwise swap
        '''
        #in place
        n = len(nums)
        
        for i in range(n):
            for j in range(n-i-1):
                if nums[j] <= nums[j+1]:
                    continue
                else:
                    if self.count_bits(nums[j]) == self.count_bits(nums[j+1]):
                        nums[j],nums[j+1] = nums[j+1],nums[j]
                    else:
                        return False
                    
        return True
        
        
    def count_bits(self,num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        
        return count
    
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        '''
        maintain min and max of sorted segments
        if we encountered a new bit counts, check that max is smaller than min
        '''
        curr_bits = 0
        curr_max = 0
        prev_max = 0
        
        for num in nums:
            #negment update set bit count and swap, prev with curr
            if self.count_bits(num) != curr_bits:
                curr_bits = self.count_bits(num)
                prev_max = curr_max
            #check for violation i.e canot be sorted
            if prev_max > num:
                return False
            curr_max = max(curr_max,num)
        
        return True
                    
    def count_bits(self,num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        
        return count

###########################################
# 1191. K-Concatenation Maximum Sum
# 07NOV24
############################################
class Solution:
    def kConcatenationMaxSum(self, arr: List[int], k: int) -> int:
        '''
        it's too expensive to concat k times and do max sum subarray
        for k == 1, i can just do kadanes
        the answer is the max of:
            1. answer for k == 1
            2. sum of whole array multiplie by k
            3. max suffix sum plus max pref_num + (k-2)*(whole array sum for k > 1)
        '''
        #kadane'ss
        mod = 10**9 + 7
        
        if k > 1:
            return ((k - 2) * max(sum(arr), 0) + self.kadane(arr * 2)) % mod 
        
        return self.kadane(arr)

    
    def kadane(self, arr):
        N = len(arr)
        dp = [0]*(N+1)
        
        for i in range(N):
            dp[i+1] = max(arr[i],dp[i] + arr[i])

        return max(dp)
        
############################################
# 1829. Maximum XOR for Each Query
# 08NOV24
############################################
class Solution:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        '''
        we need to find the k, that maximizes XOR, k is the answer to each query
        i can use pref xor to precompute the pref xor for all prefixes
        max xor is 2**maximumBit - 1
        just xor each prefix with 2**maximumBit - 1
        stupid ass problem
        
        examine some pref_xor and check this
        for i in range(20):
            print(pref_xor ^ 2**i)
        
        its an increasing function, maximized at the end,
        so just xor with 2**maxBit - 1
        '''
        pref_xor = [0]
        for num in nums:
            pref_xor.append(pref_xor[-1] ^ num)
        
        ans = []
        for pref in pref_xor[1:][::-1]:
            ans.append(pref ^ (2**maximumBit) - 1)
        
        return ans
        
#without going in reverse
class Solution:
    def getMaximumXor(self, nums: List[int], maximumBit: int) -> List[int]:
        '''
        we need to find the k, that maximizes XOR, k is the answer to each query
        i can use pref xor to precompute the pref xor for all prefixes
        max xor is 2**maximumBit - 1
        just xor each prefix with 2**maximumBit - 1
        stupid ass problem
        
        examine some pref_xor and check this
        for i in range(20):
            print(pref_xor ^ 2**i)
        
        its an increasing function, maximized at the end,
        so just xor with 2**maxBit - 1
        '''
        pref_xor = [0]
        for num in nums:
            pref_xor.append(pref_xor[-1] ^ num)
        
        ans = []
        for i in range(len(pref_xor) - 1, 0,-1):
            ans.append(pref_xor[i] ^ (2**maximumBit) - 1)
        
        return ans
        
##########################################
# 3133. Minimum Array End
# 10NOV24
###########################################
#no idea on this problem, just go through the solution :(
#note, this will TLE/fail in python
class Solution:
    def minEnd(self, n: int, x: int) -> int:
        '''
        the array must be increasing,
        and number in the array must have the initial set bits in x
        so we start with x as the first number and increment by 1
        but we need to include the set bits, so we xor
        '''
        ans = x
        while n > 1:
            ans = (ans + 1) | x
            n -= 1
        
        return ans
    

class Solution:
    def minEnd(self, n: int, x: int) -> int:
        '''
        from the hints, we can get the binary forms for both x and n-1
        the difference between the frist and last elements in their binary forms
        x's set bits need to be presevered in all numbers in the array
        n-1 gives us the flexibilty to fill in the gaps bettwen consecutive numbers
        intuition:
            merge the bit structures of x and n-1 in a way the allows us to build the smallest valid number
        
        example:
            n = 3, x = 4
            x -> (100)
            n -> (010)
            
            at position 2, x = 1 so we need to maintin set bit
            at position 1, x = 0, use bit from n-1
            as pos 0, x and n-1 == 0, so keep bit unset
        
        after getting the binary forms of x and n-1, copy bit from binN into binX, then move both
        '''
        res = 0
        #reduce n by 1
        n -= 1
        binX = [0]*64
        binN = [0]*64
        
        #checking set bits for x and n
        for i in range(64):
            mask = 1 << i
            binX[i] = 1 if (x & (mask) != 0) else 0
            binN[i] = 1 if (n & (mask) != 0) else 0
            
        
        #merge x with n-1
        X_ptr = 0
        N_ptr = 0
        
        while X_ptr < 63:
            #traverse binX until we hit 0
            while binX[X_ptr] != 0 and X_ptr < 63:
                X_ptr += 1
            
            #copy from binN into X
            binX[X_ptr] = binN[N_ptr]
            X_ptr += 1
            N_ptr += 1
        
        #compye ans
        for i in range(64):
            if binX[i]:
                res += 1 << i
        
            
