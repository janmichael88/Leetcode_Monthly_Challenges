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
        to make the number as small a possible, we fill from least significant bit the the most signicant bit
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
        
##################################################
# 3097. Shortest Subarray With OR at Least K II
# 10NOV24
#################################################
class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        '''
        is or increasing? only when the array is increasing, duh!
        is it always increasing, yes because it includes bits
        sliding window if we get more than k, untake it
        OR only sets bits, and never unsets bits
        so the number of different results for each nums[i] is at most the number of bits, 32
        keep expanding until we get to an xor that is at least k (i.e >= k)
        sliding window with set bit counts, we keep expanding until we get a subrray with OR >= k
        for each num, there are only 32 bit positions, so check all 32
        keep counter of bits in each of the positions
        '''
        ans = float('inf')
        bit_counts = [0]*32
        left = 0
        right = 0
        
        while right < len(nums):
            #for the current nunber increment bit count
            curr_num = nums[right]
            for i in range(32):
                if curr_num & (1 << i):
                    bit_counts[i] += 1
            
            #minimze ans while we have at least k
            while left <= right and self.set_bits_to_num(bit_counts) >= k:
                ans = min(ans, right - left + 1)
                #remove from window
                leftmost_num = nums[left]
                for i in range(32):
                    if leftmost_num & (1 << i):
                        bit_counts[i] -= 1
                left += 1
            
            right += 1
        
        if ans == float('inf'):
            return -1
        return ans
        
    def set_bits_to_num(self,counts):
        ans = 0
        for i in range(32):
            if counts[i]:
                ans += 1 << i
        
        return ans
    
#binary search solution??
class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        '''
        for the binary search solution we search for all possible lenghts
        if we find that no subarrya of a certain length is at least k, then we can disregard anything shorter
        if we find a valid length we save the best and look for a potentiall better way
        
        binary search on answer paradigm
        we can use sliding window approach to find subarray with at least k
        '''
        left = 1
        right = len(nums)
        ans = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            if self.at_least_k(nums,k,mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
    def at_least_k(self, nums, k, subarray_size):
        
        bit_counts = [0]*32
        
        for right in range(len(nums)):
            #for the current nunber increment bit count
            curr_num = nums[right]
            for i in range(32):
                if curr_num & (1 << i):
                    bit_counts[i] += 1
            
            #remove left most if bigger than size
            #we move one at time if we are bigger
            if right >= subarray_size:
                leftmost_num = nums[right - subarray_size]
                for i in range(32):
                    if leftmost_num & (1 << i):
                        bit_counts[i] -= 1
            
            #if valid
            if (right >= subarray_size - 1) and (self.set_bits_to_num(bit_counts) >= k):
                return True
        
        return False
        

    def set_bits_to_num(self,counts):
        ans = 0
        for i in range(32):
            if counts[i]:
                ans += 1 << i
        
        return ans
    
#########################################
# 2601. Prime Subtraction Operation
# 11NOV24
#########################################
class Solution:
    def primeSubOperation(self, nums: List[int]) -> bool:
        '''
        i can perform the following operation any number of times
        pick and index i that i havent picked before and pick prim p < nums[i] then do nums[i] - p
        return treu if we can make nums strictly increasing
        for number i can only pick a prime < nums[i], is there a best prime to pick?
        we can repeatedly substract p from nums[i]
        the most optimal prime is to pick the one that makes nums[i] the smallest as possible and greater than nums[i-1]
        generate primes
        for the first element make as small as possible by picking the correct prime
        i can use binary search
        '''
        #generate primes, onlyl up 10 1001
        MAX = 1001
        primes = [True]*(MAX + 1)
        p = 2
        while p*p <= MAX:
            if primes[p] == True:
                #mark every multiple
                for i in range(p*p,MAX + 1, p):
                    primes[i] = False
            
            p += 1
        
        primes = [i for i in range(2,MAX + 1) if primes[i] == True]
        prev = -1
        n = len(nums)
        #compare prev with curr
        
        for i in range(n):
            curr_num = nums[i]
            if curr_num <= prev:
                return False
            #find best prime
            best_prime = self.binary_search(primes,prev,curr_num)
            if best_prime > 0:
                curr_num -= best_prime
            prev = curr_num
        
        return True
            
    
    def binary_search(self,arr,prev,curr_num):
        #target shuold be nums[i] - prev
        #look for the prime that will make nums[i] - p as small as possible, but still greater than prev
        left = 0
        right = len(arr) - 1
        ans = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < curr_num and (curr_num - arr[mid] > prev):
                #try to make it smaller, by picking a larger prime
                ans = arr[mid]
                left = mid + 1
            else:
                right = mid - 1
        
        return ans
            

#brute force, check each prime on each nums
#O(len(nums))*(number or primes up to 1001)
#save best prime and advance left
class Solution:
    def primeSubOperation(self, nums: List[int]) -> bool:
        '''
        lets try doing the brute force solution
        for each nums[i] we need to find the prime that would make it as small as possible
        what we want is the largest p, such that nums[i] - p > prev
        '''
        #generate primes, onlyl up 10 1001
        MAX = 1001
        primes = [True]*(MAX + 1)
        p = 2
        while p*p <= MAX:
            if primes[p] == True:
                #mark every multiple
                for i in range(p*p,MAX + 1, p):
                    primes[i] = False
            
            p += 1
        
        primes = [i for i in range(2,MAX + 1) if primes[i] == True]
        prev = 0
        
        for num in nums:
            if num <= prev:
                return False
            #check all primes
            i = 0
            best_prime = -1
            #cool kind of invariant here
            while i < len(primes) and primes[i] < num and (num - primes[i] > prev):
                best_prime = primes[i]
                i += 1
            
            if best_prime > 0:
                num -= best_prime
            prev = num
        
        return True
            
###########################################
# 2070. Most Beautiful Item for Each Query
# 12NOV24
##########################################
class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        '''
        items are given as [price,beauty]
        for each query, determine the max beauty of an item whose price is <= to queries[j]
        sort on increasing price, tie breaker we we want maximum beauty to the right
        binary seach, but how do sort the items?
        say we pick an index i during binary search, how do we know this is the maximum beauty for that price
        if there are repeated prices, then for that price, keep the maximum beauty, we only care about the max beauty
        preprocess input
        
        looking for the right most price doesn't always guarantee the maximum beauty!, there might have been a smaller price with maximum beauty!
        storing only max beauty at each price is fine
        during binary search just store the max beauty
        after precomputing just pref max on the sorted_items array
        
        '''
        price_to_beauty = {}
        for p,b in items:
            price_to_beauty[p] = max(b,price_to_beauty.get(p,0))
        
        sorted_items = [[p,b] for p,b in price_to_beauty.items()]
        sorted_items.sort(key = lambda x : x[0])
        #pref max
        for i in range(1,len(sorted_items)):
            sorted_items[i][1] = max(sorted_items[i][1],sorted_items[i-1][1])
        ans = []
        for q in queries:
            beauty = self.bin_search(sorted_items,q)
            ans.append(beauty)
        
        return ans
    
    def bin_search(self,arr,target):
        left = 0
        right = len(arr) - 1
        ans = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid][0] <= target:
                ans = max(ans,arr[mid][1])
                left = mid + 1
            else:
                right = mid - 1
            
        return ans
    
#instead of using mapp, just pref max beauty on the items array itselt
class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        '''
        using pref_max
        '''
        items.sort()
        for i in range(1,len(items)):
            items[i][1] = max(items[i][1], items[i-1][1])
        ans = []
        for q in queries:
            beauty = self.bin_search(items,q)
            ans.append(beauty)
        
        return ans
    
    def bin_search(self,arr,target):
        left = 0
        right = len(arr) - 1
        ans = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid][0] <= target:
                ans = max(ans,arr[mid][1])
                left = mid + 1
            else:
                right = mid - 1
            
        return ans
    
#keeping index instead
class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        '''
        using pref_max
        '''
        items.sort()
        for i in range(1,len(items)):
            items[i][1] = max(items[i][1], items[i-1][1])
        ans = []
        for q in queries:
            beauty_idx = self.bin_search(items,q)
            if beauty_idx != -1:
                ans.append(items[beauty_idx][1])
            else:
                ans.append(0)
        
        return ans
    
    def bin_search(self,arr,target):
        left = 0
        right = len(arr) - 1
        ans = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid][0] <= target:
                #ans = max(ans,arr[mid][1])
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
            
        return ans
    
#old school pointer approach
class Solution:
    def maximumBeauty(self, items: List[List[int]], queries: List[int]) -> List[int]:
        '''
        instead of binary searching for each query i can just use two pointers
        sort items and queries
        same thing as before, with pre max array
        since queries are icnreasing, we can just advance one at time until we hit the price
        we also need to maintin the original index of queires for the answer!
        since we store the maximum beauty along the way, we no longer nee pref max
        '''
        queries = [[p,i] for i,p in enumerate(queries)]
        queries.sort(key = lambda x : x[0])
        items.sort()
        #for i in range(1,len(items)):
        #    items[i][1] = max(items[i][1], items[i-1][1])
            
        ans = [0]*len(queries)
        
        item_ptr = 0
        q_ptr = 0
        curr_max_beauty = 0
        while q_ptr < len(queries):
            while item_ptr < len(items) and items[item_ptr][0] <= queries[q_ptr][0]:
                curr_max_beauty = max(curr_max_beauty,items[item_ptr][1])
                item_ptr += 1
            
            ans[queries[q_ptr][1]] = curr_max_beauty
            q_ptr += 1
        
        return ans

#####################################################
# 2824. Count Pairs Whose Sum is Less than Target
# 13NOV24
#####################################################
#good precursor for 2563
class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        '''
        brute force is trivial
        sort the array, then fix nums[i]
        and find the best j
        if nums[i] + nums[j] > target, then every index after j wont work
        '''
        nums.sort()
        pairs = 0
        for i,num in enumerate(nums):
            left = i + 1
            right = len(nums) - 1
            while left <= right:
                mid = left + (right - left) // 2
                if num + nums[mid] >= target:
                    right = mid - 1
                else:
                    left = mid + 1
            
            pairs += left - i - 1
        
        return pairs


##########################################
# 2563. Count the Number of Fair Pairs
# 13NOV24
##########################################
class Solution:
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        '''
        we can sort the array first
        then search for the upper and lower bound indices
        the answer is just the number of indices between
        the number of pairs for this element is just how - ligh
        we need to look for the best number on the left of i, call it left_bound
        and we need to look for the best number on the right of i, call it right_bound
        the number of pairs we can form with i is just right_bound - left_bound
        
        intution,
        count pairs that have sums less than lower, then count pairs that have sums less than upper = 1
        its still lower bound, but we change the target by 1 
        lower - nums[i] <= nums[j] <= upper - nums[i]
        look for left and right bounds to the right of i, assuming we pick as as the first
        '''
        nums.sort()
        pairs = 0
        
        for i, num in enumerate(nums):
            left = bisect.bisect_left(nums, lower - nums[i], i + 1)
            #this works too
            #right = bisect.bisect_left(nums, upper - nums[i] + 1, i + 1)
            right = bisect.bisect_right(nums, upper - nums[i], i + 1)
            pairs += right - left
        
        return pairs
    
#summary
#bisect_left(target) == bisect_right(target + 1)
class Solution:
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        '''
        instead or calling bisect utility, just make the lower bound function
        we have lower <= nums[i] + nums[j] <= upper
        if we fix i
        lower - nums[i] <= nums[j]
        need to find the smallest nums[j]
        bisect_left
        
        upper - nums[i] >= nums[j]
        need to find the largest nums[j]
        bisect_right
        '''
        nums.sort()
        pairs = 0
        
        for i, num in enumerate(nums):
            left = self.bin_search(nums, i + 1, len(nums) - 1, lower - nums[i])
            right = self.bin_search(nums, i + 1, len(nums) - 1, upper - nums[i] + 1)
            #if we can find no such bounds, the pointers just return the number itself
            #and doesn't increment the pair count
            pairs += right - left
        
        return pairs
    
    def bin_search(self,arr,left,right,target):
        ans = left
        while left <= right:
            mid = left + (right - left) // 2
            #we are not looking for a value, we are looking for the insertion point
            #there is a difference
            if arr[mid] < target:
                ans = mid + 1
                left = mid + 1
            else:
                right = mid - 1
        
        return ans
    
#two pointers
class Solution:
    def countFairPairs(self, nums: List[int], lower: int, upper: int) -> int:
        '''
        we can also use two pointers!
        we can sort and count the number of valid windows!
        but we count in directly
        first count the number of valid windows less than lower
        then count the number of valid windows less than upper + 1
        
        for the actual counting we can use two pointers since the the array is sorted
        when we have a valid window increment by right - left
        if its too big, decrease right
        '''
        nums.sort()
        return self.count(nums,upper + 1) - self.count(nums,lower)
    
    def count(self,nums,target):
        left = 0
        right = len(nums) - 1
        pairs = 0
        
        while left < right:
            if nums[left] + nums[right] < target:
                pairs += right - left
                left += 1
            else:
                right -= 1
        
        return pairs
    
#################################################################
# 2064. Minimized Maximum of Products Distributed to Any Store
# 14NOV24
###############################################################
class Solution:
    def minimizedMaximum(self, n: int, quantities: List[int]) -> int:
        '''
        we have n stores, and m = len(quantities), product types
        q[i] = quantity if ith product
        distribute to n stores, and can only have at most one type
        a store can be given 0 products
        let x be the max number of product i can give to any store, want x to be a small a possible
        minimize the max number of products  that you are given to any store
        binary search on answer paradigm!
        search for the best k
        if we can reach k, then we can do do k+1, k+2, etc,
        we want the smallest k, that just works
        so we try from 0 all the way to max(quantities) until we get the firsst one that works, that is minimum
        because wecan just leave a store with the max, and the rest can be zero
        if we can't then we cant do anything more than k
        need linear time function to see if we can distribute k, such that any store will not be given more than k products
        '''
        def f(k,n,quantities):
            ans = 0
            for q in quantities:
                ans += ceil(q/k)
            
            return ans <= n
        
        left, right = 1, max(quantities)
        ans = 1
        while left <= right:
            mid = left + (right - left) // 2
            if f(mid,n,quantities):
                right = mid - 1
            else:
                ans = mid + 1
                left = mid + 1
        
        return ans
                
class Solution:
    def minimizedMaximum(self, n: int, quantities: List[int]) -> int:
        '''
        we have n stores, and m = len(quantities), product types
        q[i] = quantity if ith product
        distribute to n stores, and can only have at most one type
        a store can be given 0 products
        let x be the max number of product i can give to any store, want x to be a small a possible
        minimize the max number of products  that you are given to any store
        binary search on answer paradigm!
        search for the best k
        if we can reach k, then we can do do k+1, k+2, etc,
        we want the smallest k, that just works
        so we try from 0 all the way to max(quantities) until we get the firsst one that works, that is minimum
        because wecan just leave a store with the max, and the rest can be zero
        if we can't then we cant do anything more than k
        need linear time function to see if we can distribute k, such that any store will not be given more than k products
        
        insteaf of using the celing trick, lets try writing a linear functino to see if we can distrbute products with quantites not more than k
        we actually dont need to assign products to every store, we can leave stores as 0
        so check if we can go through all quantities
        '''
        def f(k,n,quantities):
            #we're not actually assigning k, its just that a store cannot have more than k
            q_ptr = 0
            curr_quant = quantities[q_ptr]
            for store in range(n):
                #use up k
                if curr_quant <= k:
                    q_ptr += 1
                    if q_ptr == len(quantities):
                        return True
                    else:
                        curr_quant = quantities[q_ptr]
                else:
                    #distribute maximum quantity to this ith store
                    curr_quant -= k
            
            return False
                    
        
        left, right = 1, max(quantities)
        ans = 1
        while left <= right:
            mid = left + (right - left) // 2
            if f(mid,n,quantities):
                right = mid - 1
            else:
                ans = mid + 1
                left = mid + 1
        
        return ans
                
'''
notes on feasibilty function,
we want to make sure that each store doesn't exceed a maximum of k, this is just
       def condition(k):
            return sum(ceil(q / k) for q in quantities) <= n
        
'''

###############################################################
# 1574. Shortest Subarray to be Removed to Make Array Sorted
# 15NOV24
################################################################
#this week sucks....
class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        '''
        the idea is to think of the array as two segments
        a prefix that is increasing and a suffix that is increasing
        the elements that need to be deleted are the middle elements
        or rather, some partition of the pref and middle and/or some partition of the middle ans suffix
        if the array is already increasing, there is nothing to remove
        if the array is decreasing then remove everying but the the first or everything by the left, i.e len(arr) - 1
        '''
        #find bound for longest increasing from start
        left = 0
        while left + 1 < len(arr) and arr[left] <= arr[left+1]:
            left += 1
        
        #increasing
        if left == len(arr) - 1:
            return 0
        
        first = arr[:left+1]
        #find portion on right, but dont go past left
        right = len(arr) - 1
        while right > left and arr[right] >= arr[right - 1]:
            right -= 1
        
        second = arr[right:]
        
        #build longest array
        longest = max(len(first),len(second))
        i,j = 0,0
        #print(first)
        #print(second)
        while i < len(first) and j < len(second):
            if first[i] <= second[j]:
                temp = first[:i+1] + second[j:]
                longest = max(longest,len(temp))
                #print(temp)
                i += 1
            else:
                j += 1
        
        return len(arr) - longest
    
#maths
class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        '''
        the idea is to think of the array as two segments
        a prefix that is increasing and a suffix that is increasing
        the elements that need to be deleted are the middle elements
        or rather, some partition of the pref and middle and/or some partition of the middle ans suffix
        if the array is already increasing, there is nothing to remove
        if the array is decreasing then remove everying but the the first or everything by the left, i.e len(arr) - 1
        '''
        #find bound for longest increasing from start
        left = 0
        while left + 1 < len(arr) and arr[left] <= arr[left+1]:
            left += 1
        
        #increasing
        if left == len(arr) - 1:
            return 0
        
        first = arr[:left+1]
        #find portion on right, but dont go past left
        right = len(arr) - 1
        while right > left and arr[right] >= arr[right - 1]:
            right -= 1
        
        second = arr[right:]
        
        #build longest array
        longest = max(len(first),len(second))
        i,j = 0,0
        #print(first)
        #print(second)
        while i < len(first) and j < len(second):
            if first[i] <= second[j]:
                #temp = first[:i+1] + second[j:]
                #compute size
                size = (i + 1) + (len(second) - j)
                #print(temp,size)
                longest = max(longest,size)
                #print(temp)
                i += 1
            else:
                j += 1
        
        return len(arr) - longest
        
############################################
# 3254. Find the Power of K-Size Subarrays I
# 16NOV24
############################################
#brute force works just fine
class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        '''
        for each subarray of size k, we need its power 
        power is the max of eleemnt if all of its eleemnts are consective and sorted in ascednig order
        otherwise its -1
        sliding window but keep track of prev min
        if the subarray isn't increasing, the ans is -1
        for each subarray we need to make sure nums[i] <= nums[i+1]
        
        '''
        n = len(nums)
        ans = []
        for i in range(n-k+1):
            subarray = nums[i:i+k]
            #check increasing
            if self.check_increasing(subarray):
                ans.append(max(subarray))
            else:
                ans.append(-1)
        
        return ans
    def check_increasing(self,arr):
        for i in range(1,len(arr)):
            if arr[i] - arr[i-1] != 1:
                return False
        return True
    
#clsoe one
class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        '''
        for each subarray of size k, we need its power 
        power is the max of eleemnt if all of its eleemnts are consective and sorted in ascednig order
        otherwise its -1
        sliding window but keep track of prev min
        if the subarray isn't increasing, the ans is -1
        for each subarray we need to make sure nums[i] <= nums[i+1]
        we can use slidigin window/deque of size k
        '''
        n = len(nums)
        ans = []
        curr_window = deque([])
        #make first k window
        for i in range(k):
            if not curr_window:
                curr_window.append(nums[i])
            elif curr_window and nums[i] - curr_window[-1] == 1:
                curr_window.append(nums[i])
            else:
                continue
        
        if len(curr_window) != k:
            ans.append(-1)
        else:
            ans.append(curr_window[-1])
        
        for i in range(k,n):
            #remove leftmost
            if curr_window:
                curr_window.popleft()
            if curr_window:
                if nums[i] - curr_window[-1] == 1:
                    curr_window.append(nums[i])
                    ans.append(nums[i])
                else:
                    curr_window.append(nums[i])
                    ans.append(-1)

        return ans
    
#using q, its really tricky
class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        '''
        use deque to store the indicies of the elements in valid sequence
        maintwain windeo of size k, if its valid, largest will be at the end
        if in valid, its just -1
        need to keep queue of indices
        '''
        n = len(nums)
        ans = []
        q = deque([])
        
        for i in range(n):
            #if the left index is out of bounds
            if q and i - q[0] >= k:
                q.popleft()
            
            #not consecutive
            if q and nums[i] - nums[i-1] != 1:
                q.clear()
            
            q.append(i)
            #past the first subarray
            if i >= k - 1:
                if len(q) == k:
                    #take the maximum
                    ans.append(nums[q[-1]])
                else:
                    #not a valid subarray sorted and consecutive
                    ans.append(-1)
        
        return ans
            
class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        '''
        we can use sliding window and expand on consecutive subarrays
        '''
        if k == 1:
            return nums
        
        n = len(nums)
        ans = []
        consecutive_count = 1
        
        left = 0
        for right in range(n):
            if right > 0 and nums[right] - nums[right - 1] == 1:
                consecutive_count += 1
            
            #if its too big
            if right - left + 1 > k:
                if nums[left + 1] - nums[left] == 1:
                    #reduce the consecutive count
                    consecutive_count -= 1
                left += 1
            
            #valid window
            if right - left + 1 == k:
                if consecutive_count == k:
                    ans.append(nums[right])
                else:
                    ans.append(-1)
        
        return ans
        

##############################################
# 862. Shortest Subarray with Sum at Least K
# 17NOV24
###############################################
#TLE, good start
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        '''
        generate prefix sum array and binary search on the answer
        brute force try all subarrays
        '''
        n = len(nums)
        pref_sum = [0]*(n+1)
        for i in range(n):
            pref_sum[i+1] = nums[i] + pref_sum[i]
        
        ans = float('inf')
        for left in range(n+1):
            for right in range(left+1,n+1):
                subarray_sum = pref_sum[right] - pref_sum[left]
                if subarray_sum >= k:
                    ans = min(ans,right - left)
        
        if ans == float('inf'):
            return -1
        return ans

#this wont work because negative values make it an non increasing array  
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        '''
        generate prefix sum array and binary search on the answer
        brute force try all subarrays
        left_sum + right_sum >= k
        look for right_sum >= k - left_sum
        
        aw shoot, its no longer and incerasing array because of the the negative values, so we cant use binary search
        
        '''
        n = len(nums)
        pref_sum = [0]*(n+1)
        for i in range(n):
            pref_sum[i+1] = nums[i] + pref_sum[i]
        
        ans = float('inf')
        for num in nums:
            if num >= k:
                ans = 1
                break

        for left in range(n+1):
            left_sum = pref_sum[left]
            #we need to look for right_sum >= k - left_sum
            lo = left
            hi = len(pref_sum) - 1
            best_idx = -1
            while lo <= hi:
                mid = lo + (hi - lo) // 2
                if pref_sum[mid] - pref_sum[left] < k:
                    #we need a bigger right
                    lo = mid + 1
                else:
                    best_idx = mid
                    hi = mid - 1
            if best_idx != -1:
                ans = min(ans, best_idx - left)
            
        
        if ans == float('inf'):
            return -1
        return ans
    
#heapq q for best previous pref_sums!
class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        '''
        the idea is store previouse pref_sums with their indices up to that point
        then we can just retrieve from the heap
        say we are at some index i, with pref_sum, ps, if prev_sum >= k, this is a valid subarray
        then we need to look back and check if their are other previous pref_sums, that cant satisfy 
        ps >= k
        '''
        n = len(nums)
        ans = float('inf')
        curr_sum = 0
        
        prev_pref_sums = []
        
        for i, num in enumerate(nums):
            curr_sum += num
            if curr_sum >= k:
                ans = min(ans, i + 1)
                
            #check if there were other previous pref_sums that satisfy >= k
            while prev_pref_sums and (curr_sum - prev_pref_sums[0][0] >= k):
                prev_ps,best_idx = heapq.heappop(prev_pref_sums)
                ans = min(ans, i - best_idx)
            
            #push
            heapq.heappush(prev_pref_sums, (curr_sum,i))
        
        if ans == float('inf'):
            return -1
        
        return ans
    
#######################################
# 1652. Defuse the Bomb (REVISTED)
# 19NOV24
#######################################
class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        '''
        if k == 0, its the zeros array
        if k > 0 take sum of numberrs to the left
            if just take some index (i + 1) % len(code)
        if k < 0, sum of k numbers to the right
            
        '''
        n = len(code)
        
        if k == 0:
            return [0]*n
        elif k > 0:
            ans = []
            for i in range(n):
                sum_next = 0
                start = i
                copy_k = k
                while copy_k > 0:
                    start += 1
                    copy_k -= 1
                    if start >= n:
                        start = 0
                    sum_next += code[start]
                
                ans.append(sum_next)
            
            return ans
        elif k < 0:
            k = abs(k)
            ans = []
            for i in range(n):
                sum_next = 0
                start = i
                copy_k = k
                while copy_k > 0:
                    start -= 1
                    copy_k -= 1
                    if start < 0:
                        start = n - 1
                    sum_next += code[start]
                ans.append(sum_next)
            
            return ans
        
class Solution:
    def decrypt(self, code: List[int], k: int) -> List[int]:
        '''
        we can also treat this like a sliding window
        [a,b,c,d,e] and k = 3
        [b + c + d, c + d + e, d + e + a, e + a + b, a + b + c]
        is we precinoute the starting sum, the next sum is just the left most removed and the right most added
        [5,7,1,4]
        first sum is 7 + 1 + 4 = 12
        second sum is just 1 + 4 + 5
            which what just (7+1+4) - 7 + 5
            
        first calculate the starting sum, of the first k elements
        as we shift the window to a each new index, we update sum by subtracting the element that's leaving the window and adding the new elment coming in
        
        for negative k, we just do it in reverse
        '''
        n = len(code)
        if k == 0:
            return [0]*n
        elif k > 0:
            starting_sum = 0
            ans = []
            for i in range(1,k+1):
                starting_sum += code[(0 + i) % n]
            ans.append(starting_sum)
            for i in range(1,n):
                starting_sum -= code[i % n]
                starting_sum += code[(i + k) % n]
                ans.append(starting_sum)
            return ans
            
        else:
            #now its in reverse
            k = abs(k)
            starting_sum = 0
            ans = []
            for i in range(1,k+1):
                starting_sum += code[((n-1) - i) % n]
            ans.append(starting_sum)
            for i in range(n-2,-1,-1):
                starting_sum -= code[i % n]
                starting_sum += code[(i + k) % n]
                ans.append(starting_sum)
            return ans[::-1]
            

########################################################
# 2461. Maximum Sum of Distinct Subarrays With Length K
# 19NOV24
########################################################
class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        '''
        sliding window with count map
        if an array has distinct elements and size k, take the maximum sum
        subarray will never be empty
        the issue is with shrinking the window
        we need to shrink the window if its bigger than k or if there is a repeated element
        '''
        counts = Counter()
        ans = 0
        curr_sum = 0
        left = 0
        N = len(nums)
        
        for right in range(N):
            #if we are trying to add in nums[right] need to make sure we can
            curr_num = nums[right]
            while counts[curr_num] >= 1 or right - left + 1 > k:
                counts[nums[left]] -= 1
                curr_sum -= nums[left]
                left += 1

            curr_sum += curr_num
            counts[curr_num] += 1
            
            if right - left + 1 == k:
                ans = max(ans,curr_sum)

                        
        return ans