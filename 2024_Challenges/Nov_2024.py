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
                

