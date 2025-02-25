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
    
#########################################################
# 2107. Number of Unique Flavors After Sharing K Candies
# 19NOV24
#########################################################
class Solution:
    def shareCandies(self, candies: List[int], k: int) -> int:
        ''' 
        need subarray of size k, where len(set(subarray)) is maximum
        sliding window, add in the first k, then pop and add back in
        i can only add to the window if next one is consecutive
        if when adding we break the consecutive variant, we must shrink the window again
        
        questions is asking count unique in candies - count unique in k length subarray
        problem is computing difference in hashmap takes O(N)
        we can track the difference as we add to the window and remove
        
        if im tracking the difference, remove from the counts hashmap
        YESSSSS
        '''

        curr_window = Counter()
        counts = Counter(candies)
        left = 0
        ans = 0
        N = len(candies)
        for right,c in enumerate(candies):
            #window to big
            if right - left + 1 > k:
                curr_window[candies[left]] -= 1
                counts[candies[left]] += 1
                if curr_window[candies[left]] == 0:
                    del curr_window[candies[left]]
                left += 1
            
            curr_window[c] += 1
            counts[c] -=1
            if counts[c] == 0:
                del counts[c]
            
            if right - left + 1 == k:
                #ans = max(ans, len(set(candies)) - len(curr_window))
                #this part, is right, but need faste way to compute
                ans = max(ans, len(counts))
        
        return ans


######################################################
# 2516. Take K of Each Character From Left and Right
# 21NOV24
#####################################################
#binaryy search for a valid array to remove
class Solution:
    def takeCharacters(self, s: str, k: int) -> int:
        '''
        first i can check of this is possible
        this is an excotic one
        if i take x chars from the left, what is the min number of chars i take from the right
        
        we can use binary search on the window size to remove
        if we can revmove a subarry of some size, then we can also remove any subarray less than size
        we want to maxmize the length of this subarray
        ans is just max(len(subrray) for valid subarrays)
        sliding window O(N) function, then maximize
        '''
        if k == 0:
            return 0
        if len(s) < 3*k:
            return -1
        
        d = Counter(s)

        if 'a' not in d or 'b' not in d or 'c' not in d:
            return -1
        
        if d['a'] < k or d['b'] < k or d['c'] < k:
            return -1
        n = len(s)
        
        #find the largest subarray we can remove
        left = 0
        right = n - 1
        ans = 0
        
        while left <= right:
            mid = left + (right - left)// 2
            if self.can_remove(s,k,mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        if ans == -1:
            return ans
        return len(s) - ans
            
    
    def can_remove(self, s, k,size):
        #rerturne true if we can remove a subarray of size size, and still staisfy the counstraint
        counts = Counter(s)
        left = 0
        for right,ch in enumerate(s):
            #remove the leftmost
            counts[ch] -= 1
            if right - left + 1 > size:
                counts[s[left]] += 1
                left += 1
            
            #check
            if right - left + 1 == size and counts['a'] >= k and counts['b'] >= k and counts['c'] >= k:
                return True
        
        return False
        
#sliding window
class Solution:
    def takeCharacters(self, s: str, k: int) -> int:
        '''
        we can reframe the problem as
        find the longest subarry in s, such that the counts of elements a,b,c are all at least k in the parts that are not part of the subaarray
        if we take some lefts x from left and some elements left from y
        the subarray between can be thouhgt of as s[x:len(s) - y]
        '''
        #first we check boundary conditions
        if k == 0:
            return 0
        if len(s) < 3*k:
            return -1
        counts = Counter(s)
        if any([char not in counts for char in ['a','b','c']]):
            return -1
        if any([counts[char] < k for char in ['a','b','c']]):
            return -1
        
        left = 0
        n = len(s)
        longest = 0
        
        for right,char in enumerate(s):
            #decrement count
            counts[char] -= 1
            
            #put back in window if we have to
            while counts['a'] < k or counts['b'] < k or counts['c'] < k:
                counts[s[left]] += 1
                left += 1
            
            longest = max(longest, right - left + 1)
        
        return len(s) - longest
                
#recursion
class Solution:
    def __init__(self):
        self.min_minutes = float("inf")

    def takeCharacters(self, s: str, k: int) -> int:
        if k == 0:
            return 0
        count = [0, 0, 0]
        self._solve(s, k, 0, len(s) - 1, count, 0)
        return -1 if self.min_minutes == float("inf") else self.min_minutes

    def _solve(self, s, k, left, right, count, minutes):
        # Base case: check if we have k of each character
        if count[0] >= k and count[1] >= k and count[2] >= k:
            self.min_minutes = min(self.min_minutes, minutes)
            return
        # If we can't take more characters
        if left > right:
            return

        # Take from left
        left_count = count.copy()
        left_count[ord(s[left]) - ord("a")] += 1
        self._solve(s, k, left + 1, right, left_count, minutes + 1)

        # Take from right
        right_count = count.copy()
        right_count[ord(s[right]) - ord("a")] += 1
        self._solve(s, k, left, right - 1, right_count, minutes + 1)

###############################################
# 2257. Count Unguarded Cells in the Grid
# 22NOV24
##############################################
class Solution:
    def countUnguarded(self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]) -> int:
        '''
        cells need to be without a wall to be considered unguarded
        gurads can see any cell in all 4 cardinal directions, vision can be blocked by another guard or wall
        guards can see in all 4 directions
        
        brute force would be to just mark each cell from a guard in all directions
        i can check each cell (i,j), its 10**5
        if i'm at a cell (i,j) that is not a wall, check (i,j-1) and (i,j+1)
        if any of those cells are guarded, then (i,j), must be guraded
        nope that doenst work
        
        walk the guards on the array!
        
        they can't see through an existing gurd already
        place the guards first, i need states for guard and guarded!
        '''
        matrix = [[0]*n for _ in range(m)]
        self.unguarded = 0
        self.guarded = 1
        self.guard = 2
        self.wall = 3
        
        #mark walls as -1
        for wall in walls:
            x,y = wall
            matrix[x][y] = self.wall
        
        for x,y in guards:
            matrix[x][y] = self.guard
            
        #for each gaurd walk them
        for x,y in guards:
            self.walk_guard(matrix, x,y,m,n)
        
        #count zeros
        unguarded = 0
        for i in range(m):
            for j in range(n):
                unguarded += matrix[i][j] == 0
        
        return unguarded
    
    def walk_guard(self,matrix,i,j,rows,cols):
        #walk up
        ii = i - 1
        while ii >= 0 and matrix[ii][j] != self.wall and matrix[ii][j] != self.guard:
            matrix[ii][j] = self.guarded
            ii -= 1
        #walk down
        ii = i + 1
        while ii < rows and matrix[ii][j] != self.wall and matrix[ii][j] != self.guard:
            matrix[ii][j] = self.guarded
            ii += 1
        
        #walk left
        jj = j - 1
        while jj >= 0 and matrix[i][jj] != self.wall and matrix[i][jj] != self.guard:
            matrix[i][jj] = 1
            jj -= 1
            
        #walk right
        jj = j + 1
        while jj < cols and matrix[i][jj] != self.wall and matrix[i][jj] != self.guard:
            matrix[i][jj] = 1
            jj += 1

#we can optimize bu looking only in the lines of sight of a guard
class Solution:
    def countUnguarded(self, m: int, n: int, guards: List[List[int]], walls: List[List[int]]) -> int:
        '''
        insteaf of walking the guards, we mark their lines of sight
        walking leading to revisting guarded cells
        example if a guard can see (2,3), we mark as guarded, 
        if another guard sees (2,3), we dont mark it again since its already seen
        
        then we just walk the directions,
        check rows, and then go left and right
            update accordingly
            
        check cols, then go up and down
            update accoridingly
            
        a cell cannot be considered guarded if there is an exsiting guard, fucking stupid lol
        '''
        matrix = [[0]*n for _ in range(m)]
        self.unguarded = 0
        self.guarded = 1
        self.guard = 2
        self.wall = 3
        
        #mark walls as -1
        for wall in walls:
            x,y = wall
            matrix[x][y] = self.wall
        
        for x,y in guards:
            matrix[x][y] = self.guard
            
        #row passes
        for row in range(m):
            active_line = matrix[row][0] == self.guard
            #left to right
            for col in range(1,n):
                active_line = self.update_cell_visibility(row,col,matrix,active_line)
            #right to left
            active_line = matrix[row][n-1] == self.guard
            for col in range(n-2,-1,-1):
                active_line = self.update_cell_visibility(row,col,matrix,active_line)
        
        #col passes
        for col in range(n):
            active_line = matrix[0][col] == self.guard
            for row in range(1,m):
                active_line = self.update_cell_visibility(row,col,matrix,active_line)
            active_line = matrix[m - 1][col] == self.guard
            for row in range(m-2,-1,-1):
                active_line = self.update_cell_visibility(row,col,matrix,active_line)
        
        ans = 0
        for row in range(m):
            for col in range(n):
                if matrix[row][col] == self.unguarded:
                    ans += 1
        
        return ans
                
            
    
    def update_cell_visibility(self,row,col,matrix,active_line):
        #activate guardline for current direction
        if matrix[row][col] == self.guard:
            return True
        if matrix[row][col] == self.wall:
            return False
        if active_line:
            matrix[row][col] = self.guarded
        return active_line
        

######################################################
# 1072. Flip Columns For Maximum Number of Equal Rows
# 23NOV24
######################################################
class Solution:
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        '''
        we need the maximum number of rows that have the same values ater some number of flips
        we need rows to be either all zeros or all ones
        we can only flip along columns
        
        [0,1]
        [1,0]
        
        can either flip for first or second column
        
        [0,1]
        [1,1]
        
        [0,0,0]
        [0,0,1]
        [1,1,0]
        
        if any two columns are inverses of each other, we can flip either one of them to make them equal
        hint says count rows that have inversed bit sets

        intution,
        the rwos that can be made uniform (all values in row are same) after flipping will be the combined total of rows
        that are identical and inversed
        '''
        rows, cols = len(matrix),len(matrix[0])
        #first count up rows
        counts = Counter()
        for r in matrix:
            counts[tuple(r)] += 1
        
        ans = 0
        for r in matrix:
            inversed_row = []
            for b in r:
                inversed_row.append(1-b)
            
            count_same = counts[tuple(r)]
            count_inverse = counts[tuple(inversed_row)]
            ans = max(ans, count_same + count_inverse)
        
        return ans
    
##############################################
# 1861. Rotating the Box
# 24NOV24
##############################################
class Solution:
    def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
        '''
        advent of code like question!
        we can just simulate pushing to the the right, once we are donce with that we can just tranpose the box
        first we can rotate, then just drop
        (i,j) -> (m - 1 - j,i)
        implementing drop is tricky
        how to i know when to finish dropping
        
        it might be easier to just push to the right bases on rows then rotate
        ["#",".","#"]
        
        drop to pattern, look for a stone, drop to drop to and replace with empty space!
        
        FOR DROP!
        idea is to swap to the spot and move up
        if ROCK, the inc up
        '''
        m,n = len(box),len(box[0])
        rotated = [[""]*m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                rotated[i][j] = box[m - 1 - j][i]
        
        rows,cols = len(rotated),len(rotated[0])
        #implement drop
        for col in range(cols):
            drop_to = n - 1
            curr_row = rows - 1
            while curr_row >= 0:
                #if its a stone, drop and replcae with empty spaces
                if rotated[curr_row][col] == "#":
                    rotated[curr_row][col] = "."
                    rotated[drop_to][col] = "#"
                    drop_to -= 1
                #otherwise new dropping spot
                if rotated[curr_row][col] == "*":
                    drop_to = curr_row - 1
                
                curr_row -= 1

        return rotated
                
###########################################
# 1975. Maximum Matrix Sum
# 24NOV24
###########################################
class Solution:
    def maxMatrixSum(self, matrix: List[List[int]]) -> int:
        '''
        we can use any number of operations
        an operation consists of multiplying adjacent eleemnts by -1, notice that prev negatives will increase
        we can only do a cell (i,j) and its neighboring cell (i,j+1), (i,j-1), (i-1,j), (i,j+1)
        use operation so that each row only has one negative number
        if you only have one negative element, you cannot convert to positive
        
        if we can make a row that has only one negative in it, we can line all the negatives up, and use operations on just a single column
        we need to transform some matrix to it canonical form, that bull shit again
        because we do ops in pairs, we can get negatives together
        [1,-1,1,-1,-1,1,1]
        
        if there's an even count of negativ numbers, we can flip them all to positive
        if the count is odd, one number has to stay negative
        
        the fact that we can use on adjacent means we can flip all numbers if even
        '''

        count_negatives = 0
        for r in matrix:
            for num in r:
                if num < 0:
                    count_negatives += 1
        
        if count_negatives % 2 == 0:
            ans = 0
            for r in matrix:
                for num in r:
                    ans += abs(num)
            return ans
        
        else:
            smallest_positive = float('inf')
            ans = 0
            for r in matrix:
                for num in r:
                    ans += abs(num)
                    smallest_positive = min(smallest_positive, abs(num))
            return ans - 2*smallest_positive
        
#reduce to one pass, count on the fly
class Solution:
    def maxMatrixSum(self, matrix: List[List[int]]) -> int:
        '''
        sum on fly and minimize on fly
        '''

        count_negatives = 0
        smallest_positive = float('inf')
        total_positive_sum = 0
        for r in matrix:
            for num in r:
                if num < 0:
                    count_negatives += 1
                smallest_positive = min(smallest_positive, abs(num))
                total_positive_sum += abs(num)
                
        
        if count_negatives % 2 == 1:
            total_positive_sum -= 2*smallest_positive
        
        return total_positive_sum
    
################################################################
# 1784. Check if Binary String Has at Most One Segment of Ones
# 24NOV24
################################################################
#kinda tricky
class Solution:
    def checkOnesSegment(self, s: str) -> bool:
        '''
        there can only be one at most one continguous segment
        count the streaks
        '''
        count_streaks = 0
        on_streak = False
        
        for ch in s:
            if ch == '1' and on_streak:
                continue
            elif ch == '1' and not on_streak:
                on_streak = True
            elif ch == '0' and on_streak:
                count_streaks += 1
                on_streak = False
            else:
                on_streak = False
        
        if on_streak:
            count_streaks += 1
    
        return count_streaks <= 1
        


###################################################
# 2371. Minimize Maximum Value in a Grid
# 24NOV24
###################################################
#close one       
class Solution:
    def minScore(self, grid: List[List[int]]) -> List[List[int]]:
        '''
        realtive order of every two elements in the same row and col should stay the same
        say for a row we have
        [4,5,6,7] -> [1,2,3,4]
        the maximum number in the matrix should be a smalle as possible
        replace in order from smallest to largest
        bet before putting check relative ordering, since we are going in increasing order, we know this current (i,j) is the smallest so far
        othweise we would have picked a larger number
        '''
        rows, cols = len(grid), len(grid[0])
        ans = [[0]*cols for _ in range(rows)]
        
        cells = []
        for i in range(rows):
            for j in range(cols):
                cells.append((grid[i][j],i,j))
        #sort increasingly
        cells.sort(key = lambda x : x[0])
        smallest = 0
        
        for num,i,j in cells:
            up = grid[i-1][j] if i - 1 >= 0 else float('inf')
            down = grid[i+1][j] if i + 1 < rows else float('inf')
            left = grid[i][j-1] if j - 1 >= cols else float('inf')
            right = grid[i][j+1] if j + 1 < cols else float('inf')
            if up <= num <= down or left <= num <= right:
                ans[i][j] = smallest
            else:
                smallest += 1
                ans[i][j] = smallest
        
        return ans
    
class Solution:
    def minScore(self, grid: List[List[int]]) -> List[List[int]]:
        '''
        to make the matrix as smallest as possible, we should start with the smallest number 1
        and we traverse the grid in order of smallest numbers first
        (grid[i][j], i,i)
        initially store smalles values in rows and cols as 1
        the smallest number in row,col can be determined by the largest value (up to now)
        
        we can prove by contradiction
        base case, single element is 1
        for the next k eleements, the first k elements are alrady increasing
        to find the next k+1, look for max in rows and cols, the add 1
        
        if we assume there was another v for k + 1 that wasn't max of rows and cols + 1, the previous asigned values up to k
        must have been smaller than v
        
        proof by contraction weak assumption
        '''
        rows, cols = len(grid), len(grid[0])
        row_maxs = [1]*rows
        col_maxs = [1]*cols
        
        cells = []
        for i in range(rows):
            for j in range(cols):
                cells.append((grid[i][j],i,j))
        #sort increasingly
        cells.sort(key = lambda x : x[0])
        
        for num,i,j in cells:
            new_val = max(row_maxs[i],col_maxs[j])
            grid[i][j] = new_val
            row_maxs[i] = new_val + 1
            col_maxs[j] = new_val + 1
        
        return grid
            
            
###################################
# 773. Sliding Puzzle
# 25NOV24
###################################
#BFS works just fine
#YEAHHH
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        '''
        use bfs an exapnd the states at each step
        the board is small so there won't be any steps
        dont precompute the graph, build states on the fly
        hash previously visited states as ((row1),(row2))
        
        so annoying
        '''
        first_zero = self.find_zero(board)
        seen = set()
        q = deque([])
        q.append([0,first_zero[0],first_zero[1],board]) #store as [moves,zero_i,zero_j,board]
        final_state = [[1,2,3],[4,5,0]]
        
        while q:
            moves,zero_i,zero_j,curr_board = q.popleft()
            if curr_board == final_state:
                return moves
            seen.add(self.get_state_sig(curr_board))
            #neigh search
            for neigh in self.get_neighs(zero_i,zero_j,curr_board):
                ii,jj,neigh_board = neigh
                #get sig
                board_sig = self.get_state_sig(neigh_board)
                if board_sig not in seen:
                    q.append([moves + 1, ii, jj, neigh_board])
        
        return -1
            
    #for each state pass in location of zero, instead of searching
    def get_neighs(self,zero_i, zero_j, matrix):
        rows,cols = 2,3
        for di,dj in [[1,0],[-1,0],[0,-1],[0,1]]:
            neigh_x = zero_i + di
            neigh_y = zero_j + dj
            if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                #swap
                matrix[zero_i][zero_j],matrix[neigh_x][neigh_y] = matrix[neigh_x][neigh_y],matrix[zero_i][zero_j]
                yield [neigh_x,neigh_y, [row[:] for row in matrix]]
                #swap back
                matrix[zero_i][zero_j],matrix[neigh_x][neigh_y] = matrix[neigh_x][neigh_y],matrix[zero_i][zero_j]
                
    #find first zero
    def find_zero(self,matrix):
        rows,cols = 2,3
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    return (i,j)
                
    def get_state_sig(self,matrix):
        sig = ""
        for r in matrix:
            for num in r:
                sig += str(num)
        
        return sig
    
#dfs, convert 2d to 1d and make transistion matrix
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        '''
        dfs solution, decompose 2d to 1d, then find the indices to where we can look up to switch to
        for example if we have the array [[1,2,3],[4,0,5]]
        [1,2,3,4,0,5]
        then we can mapp the locatino of the indices to the swap spots
        0 : [1, 3],
        1 : [0, 2, 4],
        2 : [1, 5],
        3: [0, 4],
        4: [3, 5, 1],
        5: [4, 2],
        '''
        # Direction map for zero's possible moves in a flattened 1D array (2x3 board)
    directions = [
        [1, 3],
        [0, 2, 4],
        [1, 5],
        [0, 4],
        [3, 5, 1],
        [4, 2],
    ]

    def slidingPuzzle(self, board: List[List[int]]) -> int:
        # Helper method to swap characters at indices i and j in the string
        def _swap(s, i, j):
            s = list(s)
            s[i], s[j] = s[j], s[i]
            return "".join(s)

        # Convert the 2D board into a string representation to use as state
        start_state = "".join(str(num) for row in board for num in row)

        # Dictionary to store the minimum moves for each visited state
        visited = {}

        def _dfs(state, zero_pos, moves):
            # Skip if this state has been visited with fewer or equal moves
            if state in visited and visited[state] <= moves:
                return
            visited[state] = moves

            # Try moving zero to each possible adjacent position
            for next_pos in self.directions[zero_pos]:
                new_state = _swap(
                    state, zero_pos, next_pos
                )  # Swap to generate new state
                _dfs(
                    new_state, next_pos, moves + 1
                )  # Recursive DFS with updated state and move count

        # Start DFS traversal from initial board state
        _dfs(start_state, start_state.index("0"), 0)

        # Return the minimum moves required to reach the target state, or -1 if unreachable
        return visited.get("123450", -1)
    
#################################
# 2924. Find Champion II
# 26NOV24
#################################
class Solution:
    def findChampion(self, n: int, edges: List[List[int]]) -> int:
        '''
        edges is respresentation of DAG
        u to v means u is stronger than b, i.e the path is decreasing
        team u will be champion if no tebam v is stronger than u
        nodes that have indegree >= 1 cannot be a champsionship team
        is there is a champion, there will only be "ONE" node with indegree 0
        
        '''
        indegree = {node : 0 for node in range(n)}
        for u,v in edges:
            indegree[v] += 1
        
        ans = -1
        
        for node,count in indegree.items():
            if count == 0:
                if ans == -1:
                    ans = node
                else:
                    return -1
        
        return ans
    
#we can also just use array
class Solution:
    def findChampion(self, n: int, edges: List[List[int]]) -> int:
        '''
        we can use an array
        
        '''
        indegree = [0]*n
        
        for u,v in edges:
            indegree[v] += 1
        
        champion = -1
        count = 0
        
        for i in range(n):
            if indegree[i] == 0:
                champion = 0
                count += 1
        
        return champion if count == 1 else -1

##########################################################
# 3243. Shortest Distance After Road Addition Queries I
# 27NOV24
#######################################################
class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        '''
        initially the dist marix for all (i,j) is just abs(j-i)
        queries add a road from (u to v)
        for each query find the shortest distance from 0 to n-1 ater adding in the i+1 roads
        bfs after each addition?
        '''
        #create initial graph
        graph = defaultdict(list)
        for i in range(n-1):
            graph[i].append(i+1)
            
        ans = []
        for u,v in queries:
            graph[u].append(v)
            min_dist = self.dijkstras(graph,n)
            ans.append(min_dist)
        
        return ans
    
    #good review on djikstras
    def dijkstras(self,graph,n):
        
        dists = [float('inf')]*n
        dists[0] = 0
        seen = set()
        
        pq = [(0,0)]
        
        while pq:
            min_dist, curr_node = heapq.heappop(pq)
            if dists[curr_node] < min_dist:
                continue
            seen.add(curr_node)
            for neigh in graph[curr_node]:
                if neigh in seen:
                    continue
                neigh_dist = dists[curr_node] + 1
                if dists[neigh] > neigh_dist:
                    dists[neigh] = neigh_dist
                    heapq.heappush(pq, (neigh_dist, neigh))
        
        return dists[n-1]
        
#dp SSP
class Solution:
    def shortestDistanceAfterQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        '''
        like with any SSP we can solve using dp
        '''
        #create initial graph
        graph = defaultdict(list)
        for i in range(n-1):
            graph[i].append(i+1)
            
        ans = []
        memo = {}
        for u,v in queries:
            graph[u].append(v)
            min_dist = self.dp(0,graph,memo,n)
            ans.append(min_dist)
            memo = {}
        
        return ans
    
    #we can also use dp
    def dp(self,node ,graph, memo,n):
        if node == n-1:
            return 0
        if node in memo:
            return memo[node]
        ans = float('inf')
        
        for neigh in graph[node]:
            ans = min(ans, 1 + self.dp(neigh,graph,memo,n))
        
        memo[node] = ans
        return ans
        
#################################################
# 2290. Minimum Obstacle Removal to Reach Corner
# 28NOV24
#################################################
#cant use dp
class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        '''
        rows*cols < 10**5, so i can use dp of the form (i,j)
        if i'm at (i,j) i can move, to some neigh cell
        if neigh cell is empty, move to it with no removals, otherwise move to it and reove
        '''
        memo = {}
        seen = set()
        
        rows,cols = len(grid), len(grid[0])
        seen.add((0,0))
        def dp(i,j):
            if (i,j) in memo:
                return memo[(i,j)]
            if (i,j) == (rows - 1, cols - 1):
                return 0           
            ans = float('inf')
            for di,dj in [[1,0],[-1,0],[0,1],[0,-1]]:
                neigh_i = i + di
                neigh_j = j + dj
                if 0 <= neigh_i < rows and 0 <= neigh_j < cols:
                    if (neigh_i,neigh_j) not in seen:
                        seen.add((neigh_i,neigh_j))
                        #if empty, just move to it
                        if grid[neigh_i][neigh_j] == 0:
                            ans = min(ans, dp(neigh_i,neigh_j))
                        else:
                            ans = min(ans, 1 + dp(neigh_i,neigh_j))
                        seen.remove((neigh_i,neigh_j))
                        
            
            memo[(i,j)] = ans
            return ans
        
        return dp(0,0)
    
#djikstra
class Solution:
    def minimumObstacles(self, grid: List[List[int]]) -> int:
        '''
        need to use dijkstras
        '''
        rows,cols = len(grid), len(grid[0])
        dists = [[float('inf')]*cols for _ in range(rows)]
        dists[0][0] = grid[0][0]
        seen = set()
        
        min_heap = [(grid[0][0], 0 ,0)]
        
        while min_heap:
            min_dist, i,j = heapq.heappop(min_heap)
            if dists[i][j] < min_dist:
                continue
            seen.add((i,j))
            
            for di,dj in [[1,0],[-1,0],[0,1],[0,-1]]:
                neigh_i = i + di
                neigh_j = j + dj
                if 0 <= neigh_i < rows and 0 <= neigh_j < cols:
                    if (neigh_i,neigh_j) in seen:
                        continue
                    neigh_dist = min_dist + grid[neigh_i][neigh_j]
                    if neigh_dist < dists[neigh_i][neigh_j]:
                        dists[neigh_i][neigh_j] = neigh_dist
                        heapq.heappush(min_heap, (neigh_dist, neigh_i, neigh_j))
        
        return dists[rows -1][cols - 1]

################################################
# 2577. Minimum Time to Visit a Cell In a Grid
# 29NOV24
################################################
class Solution:
    def minimumTime(self, grid: List[List[int]]) -> int:
        '''
        i can only visit a cell (i,j) if i visit it at >= grid[i][j]
        we can only visit a cell up down left right
        
        return minimum time required to rech -1
        if i'm at cell (i,j) with time t, i can only visit neigh cells is a neigh cell (ii,jj)
        has grid[ii][jj] >= t
        consider case where we have to go back and forth between two cells of matrix to unlock other cells
        each step takes 1 unit time
        we can walk back and forth repeatedly between a current cell and neigh cell
        '''
        #edge case if we can't make the first move
        if grid[0][1] > 1 and grid[1][0] > 1:
            return -1
        
        rows, cols = len(grid), len(grid[0])
        dists = [[float('inf')]*cols for _ in range(rows)]
        dists[0][0] = 0 
        
        min_heap = [(0,0,0)] #mintime (i,j)
        seen = set()
        
        while min_heap:
            min_time, i,j = heapq.heappop(min_heap)
            if min_time < dists[i][j]:
                continue
            seen.add((i,j))
            
            for di,dj in [[1,0],[-1,0],[0,1],[0,-1]]:
                neigh_i = di + i
                neigh_j = dj + j
                if 0 <= neigh_i < rows and 0 <= neigh_j < cols:
                    if (neigh_i,neigh_j) in seen:
                        continue
                    #if they neighboring cell is 1 away, we can walk to it or get waiting time to get to cell
                    neigh_time = min_time + 1
                    if min_time + 1 < grid[neigh_i][neigh_j]:
                        neigh_time =  grid[neigh_i][neigh_j] + ((grid[neigh_i][neigh_j] - min_time) % 2 == 0)
                    if dists[neigh_i][neigh_j] > neigh_time:
                        dists[neigh_i][neigh_j] = neigh_time
                        heapq.heappush(min_heap, (neigh_time, neigh_i, neigh_j))


        if dists[rows - 1][cols - 1] == float('inf'):
            return -1
        return dists[rows - 1][cols - 1]
                        
#anoher way
class Solution:
    def minimumTime(self, grid: List[List[int]]) -> int:
        '''
        i can only visit a cell (i,j) if i visit it at >= grid[i][j]
        we can only visit a cell up down left right
        
        return minimum time required to rech -1
        if i'm at cell (i,j) with time t, i can only visit neigh cells is a neigh cell (ii,jj)
        has grid[ii][jj] >= t
        consider case where we have to go back and forth between two cells of matrix to unlock other cells
        each step takes 1 unit time
        we can walk back and forth repeatedly between a current cell and neigh cell
        '''
        #edge case if we can't make the first move
        if grid[0][1] > 1 and grid[1][0] > 1:
            return -1
        
        rows, cols = len(grid), len(grid[0])
        dists = [[float('inf')]*cols for _ in range(rows)]
        dists[0][0] = 0 
        
        min_heap = [(0,0,0)] #mintime (i,j)
        seen = set()
        
        while min_heap:
            min_time, i,j = heapq.heappop(min_heap)
            if min_time < dists[i][j]:
                continue
            seen.add((i,j))
            
            for di,dj in [[1,0],[-1,0],[0,1],[0,-1]]:
                neigh_i = di + i
                neigh_j = dj + j
                if 0 <= neigh_i < rows and 0 <= neigh_j < cols:
                    if (neigh_i,neigh_j) in seen:
                        continue
                    #if they neighboring cell is 1 away, we can walk to it or get waiting time to get to cell
                    #if we can immediately visit neighor cell
                    if grid[neigh_i][neigh_j] <= min_time + 1:
                        dists[neigh_i][neigh_j] = min_time + 1
                        heapq.heappush(min_heap, (min_time + 1, neigh_i, neigh_j))
                    else:
                        neigh_time = min_time + 1
                        if min_time + 1 < grid[neigh_i][neigh_j]:
                            neigh_time =  grid[neigh_i][neigh_j] + ((grid[neigh_i][neigh_j] - min_time) % 2 == 0)
                        if dists[neigh_i][neigh_j] > neigh_time:
                            dists[neigh_i][neigh_j] = neigh_time
                            heapq.heappush(min_heap, (neigh_time, neigh_i, neigh_j))


        if dists[rows - 1][cols - 1] == float('inf'):
            return -1
        return dists[rows - 1][cols - 1]
                        
###############################################
# 2204. Distance to a Cycle in Undirected Graph
# 30DEC24
###############################################
class Solution:
    def distanceToCycle(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        we have n nodes with bi-directional edge list
        distance between two nodes (u,v) is the minimum number of edges that are need to go from (u to v)
        return ans array where ans[i] is the min distance between the i'th node and the any node in the cycle
        
        first find cycle and detect nodes in the cycle
        multipoint bfs (i.e queue up each node in the cycle)
        using dfs for contains cycle is is easy, its finding node in the cycle
        if we dfs and keep track of the order of visited nodes
        
        i can use kahns's algo, first start with nodes with 0 indegree
        keep going until we can continue, any nodes left must be part of cycle
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        cycle_nodes = self.find_cycle_node(graph, n)
        #mult point bfs
        dists = [float('inf')]*n
        q = deque([])
        seen = set()
        for c in cycle_nodes:
            dists[c] = 0
            q.append((0,c))
        
        while q:
            min_dist, curr_node = q.popleft()
            if dists[curr_node] < min_dist:
                continue
            seen.add(curr_node)
            for neigh in graph[curr_node]:
                if neigh in seen:
                    continue
                
                neigh_dist = dists[curr_node] + 1
                if dists[neigh] > min_dist:
                    dists[neigh] = neigh_dist
                    q.append((neigh_dist, neigh))
                    
        return dists
                    
    
    def find_cycle_node(self,graph,n):
        indegree = [0]*n
        
        for node,child in graph.items():
            for c in child:
                indegree[c] += 1
        
        q = deque([])
        for node,count in enumerate(indegree):
            if count == 1:
                q.append(node)
        
        while q:
            curr = q.popleft()
            for neigh in graph[curr]:
                indegree[neigh] -= 1
                if indegree[neigh] == 1:
                    q.append(neigh)
        
        #nodes with two edges are part of cycle
        cycle_nodes = []
        for i,count in enumerate(indegree):
            if count == 2:
                cycle_nodes.append(i)
        
        return cycle_nodes
        
class Solution:
    def distanceToCycle(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        we can use dfs to find the nodes in the cycle
        once we find the nodes in the cycle. we treat all nodes in this cycle as the root,
        then we can just dfs for each node, and store the minimum distances from the root
        
        the key for using dfs to find the cycles is to keep a seen set (for backtracking)
        and another array to follow node to parents back up
        very similar to the contains cycles, exceapt the once we find the cycle follow parents back up
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = [False]*n
        in_cycle = [False]*n
        parent = [False]*n
        
        self.contains_cycle(0,parent,graph,visited, in_cycle)
        #clear visited
        visited = [False]*n
        dists = [0]*n
        
        for i in range(n):
            if in_cycle[i]:
                self.calc_distances(i,0, graph,dists,visited,in_cycle)
                break
        
        return dists
    
    
    def contains_cycle(self, curr, parent, graph, visited, in_cycle):
        #mark
        visited[curr] = True
        for neigh in graph[curr]:
            #i could come back to curr from another 
            if not visited[neigh]:
                parent[neigh] = curr
                if self.contains_cycle(neigh,parent, graph, visited, in_cycle):
                    return True
            #i still dont get why this is the trigger for sart of cycle
            elif parent[curr] != neigh:
                in_cycle[neigh] = True
                temp = curr
                while temp != neigh:
                    in_cycle[temp] = True
                    temp = parent[temp]
        
        return True
    
    def calc_distances(self, curr_node, curr_dist, graph, dists, visited, in_cycle):
        visited[curr_node] = True
        dists[curr_node] = curr_dist
        for neigh in graph[curr_node]:
            if visited[neigh]:
                continue
            neigh_dist = 0 if in_cycle[neigh] else curr_dist + 1
            self.calc_distances(neigh, neigh_dist,graph, dists, visited, in_cycle)
        
##############################################
# 2097. Valid Arrangement of Pairs (REVISTED)
# 02DEC24
#############################################
class Solution:
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        '''
        graph problem
        we need to find an eulerian path of this graph
        i need to mapp end of i interval to start of another interval and start of interval to another end
        i cant look through every other jth pair if im on the ith pair, too long
        start or end nodes will have indegree zero
        
        eulerian path touches each edge once
        
        for eulerian paths, keep track of indegree and outdegree, and look for diffence between them
        in an undirected graph, all nodes hav even degree or exatcly two have oad degree
        in directed graph,
            each nodes out == in
            or exactly one node has one more out going edge
                i.e out = in + 1, which is out starting point
            
            if there is more than one such node, we can use any to find eulerian path
            tracking difference just means starting node's degree diff == 1
        
        post order dfs first, neighs before node (or children before node)
        intuition on why postorder
            if we are at u, with unvisited nodes (v), we need to visit them first because we need to eventually come back to u
            in the eulerian path, but not all edges might come back to u
            
        iterative version is kwown as Hierholzer
        '''
        graph = defaultdict(list)
        degree = Counter()
        #degree diff
        for u,v in pairs:
            graph[u].append(v)
            degree[u] += 1
            degree[v] -= 1
            
        for start in graph:
            if degree[start] == 1:
                break
        
        self.ans = []
        
        def dfs(node):
            neighs = graph[node]
            while neighs:
                neigh = neighs.pop()
                dfs(neigh)
                self.ans.append([node,neigh])
            
        dfs(start)
        return self.ans[::-1]
                