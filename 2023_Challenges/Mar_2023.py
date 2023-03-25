####################################
# 912. Sort an Array (REVISTED) 
# 01MAR23
###################################
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''
        merge sort using extra space
        '''
        return self.merge_sort(nums)
    
    def merge(self, left_list: List[int], right_list: List[int]) -> List[int]:
        merged = []
        
        i,j = 0,0
        while i < len(left_list) and j < len(right_list):
            if left_list[i] < right_list[j]:
                merged.append(left_list[i])
                i += 1
            else:
                merged.append(right_list[j])
                j += 1
        
        #the rest
        while i < len(left_list):
            merged.append(left_list[i])
            i += 1
        
        while j < len(right_list):
            merged.append(right_list[j])
            j += 1
        
        #could also have done
        #merged.extend(left_list[i:])
        #merged.extend(right_list[j:])
        
        return merged
        
    def merge_sort(self, array : List[int]) -> List[int]:
        if len(array) == 1:
            #single element, just return the array
            return array
        
        mid = len(array) // 2
        left = self.merge_sort(array[0:mid])
        right = self.merge_sort(array[mid:])
        return self.merge(left,right)


#inplace merge sort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''
        if we want in place merger sort, we need to use a temp array of len(nums)
        then we ferry elements from nums into temp
        then merge back into nums in sorted order
        '''
        self.temp = [0]*len(nums)
        self.nums = nums

        #invoke
        self.merge_sort(0,len(nums)-1)
        return self.nums
    def merge_sort(self,left:int,right:int):
        if left >= right:
            return
        mid = left + (right - left) // 2
        self.merge_sort(left,mid)
        self.merge_sort(mid+1,right)
        self.merge(left,mid,right)
    
    def merge(self,left:int, mid:int, right:int):
        #find starts and ends
        #we need access to the indices back into nums
        
        start1 = left
        start2 = mid + 1
        end1 = mid - left + 1
        end2 = right - mid
        
        #copy elements for both halves into temp array
        for i in range(end1):
            self.temp[start1 + i] = self.nums[start1 + i]
        
        for i in range(end2):
            self.temp[start2 + i] = self.nums[start2 + i]
            
        #merge in sorted order back inot numes
        i,j,k = 0,0,left #k is the insert posittion where we start at this current recursive call
        while i < end1 and j < end2:
            if self.temp[start1 + i] < self.temp[start2 + j]:
                self.nums[k] = self.temp[start1 + i]
                i += 1
            else:
                self.nums[k] = self.temp[start2 + j]
                j += 1
            
            k += 1
        
        #the rest
        while i < end1:
            self.nums[k] = self.temp[start1 + i]
            i += 1
            k += 1
        
        while j < end2:
            self.nums[k] = self.temp[start2 + j]
            j += 1
            k += 1
        
#heap sort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''
        notes on heap sort
        for an array of size N
        there are (N//2) -1 leaf nodes, indexed by [N//2 -1 to N - 1]
        we first call heappify on all the nodes that are not leaves in reverse
        then when soriting, swap with the largest element, then heapify at the index
        
        check out this article
        https://www.geeksforgeeks.org/heap-sort/
        
        
        
        '''
        
        def heapify(n,i):
            largest = i
            left = 2*i + 1
            right = 2*i + 2
            
            #if left is larger, set it
            if left < n and nums[left] > nums[largest]:
                largest = left
            #if right is larget
            if right < n and nums[right] > nums[largest]:
                largest = right
            #if largest is not root, swap root with the largest element
            #recursively heapify the affected subtree 
            if largest != i:
                nums[i],nums[largest] = nums[largest], nums[i]
                heapify(n,largest)
                
        N = len(nums)
        #build heap, top down
        #we only need to heapify on the non leave eleemnets in reverse, should give is the property that the largest element is on the top
        #print(nums)
        for i in range(N//2-1,-1,-1):
            #print(nums[i])
            heapify(N,i)
            #print("after ", nums)
        
        #travesre elements one by one, by swapping the root to the end
        #after teh swao, call heapify on the root
        #print(nums)
        for i in range(N-1,-1,-1):
            nums[0],nums[i] = nums[i],nums[0]
            heapify(i,0) #heappify only from 0 index to the current i
            #print(nums)
        
        return nums
        
#counting sort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''
        keep count mapp and pointer
        '''
        counts = Counter(nums)
        
        index = 0
        
        for i in range(min(nums),max(nums)+1):
            while counts[i] > 0:
                nums[index] = i
                index += 1
                counts[i] -= 1
        
        return nums

#quick sort, extra space
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''
        we can use quick sort with extra space
        pick parition point, then divide into three parts
            less than, equal to, greater then
            then recurse
        
        we can pick random pivot for average O(nlgn), worst case would be O(N^2)
        or use median of medians for deterministic O(nlgn)
        '''
        
        def quicksort(nums):
            if len(nums) <= 1:
                return nums
            
            pivot = random.choice(nums)
            less_than = [num for num in nums if num < pivot]
            equal = [num for num in nums if num == pivot]
            greater_than = [num for num in nums if num > pivot]
            
            left = quicksort(less_than)
            right = quicksort(greater_than)
            return left + equal + right
        
        return quicksort(nums)

#quicksort inplace

###################################
# 443. String Compression (REVISTED)
# 02MAR23
###################################
class Solution:
    def compress(self, chars: List[str]) -> int:
        '''
        i can use two pointers with a sliding window, once i'm done with the window just over write them with another pointer
        it wont matter anyway since we adavnce pointers
        '''
        N = len(chars)
        ptr = 0 #pointer to modify array
        curr_count = 0 #count of curr char
        left,right = 0,0
        
        while right < N:
            #expand
            while right < N and chars[right] == chars[left]:
                right += 1
                curr_count += 1
            
            #i need to modify the input array now
            #print(chars[left],curr_count)
            #curr_count = 0
            #left = right
        
            if curr_count == 1:
                chars[ptr] = chars[left]
                ptr += 1
            else:
                #get the digits to be splace in the array
                chars[ptr] = chars[left]
                ptr += 1
                for digit in str(curr_count):
                    chars[ptr] = digit
                    ptr += 1
            
            curr_count = 0
            left = right
        
        return ptr

####################################################################
# 28. Find the Index of the First Occurrence in a String  (REVISTED)
# 03MAR23
####################################################################
#sliding window
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        '''
        just check each character in needle 1 by one
        '''
        n = len(haystack)
        m = len(needle)
        
        for i in range(n-m+1):
            for j in range(m):
                if needle[j] != haystack[i+j]:
                    break
                #got to the end
                if j == m - 1:
                    return i
        
        return -1

#rabin karp
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        '''
        rabin karp, collisions also called spurious hits in rabin-karp
        recall idea is to use rolling hash, to compute next hash in constant time, only compare character by character when we have a matching hash
        rather spurious hit, is a consequence of a weak hasing function
        
        hashing function:
            hash(S[0...(m-1)]) = h(S[0])*26^m-1 + h(S[1])*26 ^ m -2 + ... + h(S[m-1])*26^0
        
        we only need to compute the hash of the first window of size m once, then we can use rolling hash to get the next hash in constant time
            * multiply the hash value of the previs m ubstring by raidx 26
            * subttract th vale of the first char in the previous m substring
            * add value of the last char in the current m-substring
            
        proof:
            H[i] = S[i]*b^(m-1) + S[i+1]*b^(m-2) + ... + S[i+m-1]*b^0
            adding 1
            H[i+1] = S[i+1]*b^(m-1) + S[i+2]*b^(m-2) + ... + S[i+m]*b^0
            
            induction:
            H[i+1] = H[i]*b + S[i]*b^m + S[i+m]
            
            * H[i]b is the left shifting one radix
            * S[i]*b is the left most char with modiified weightage, which is now out of the window
            * S[i+m] is the hashmapping of the right most character to be added to the window
            
        taking modulos:
        H[i+1]=((H[i]b mod MOD)−(S[i]b^mm mod MOD) + (S[i+m] mod MOD)) mod MOD
        where MOD is a large primer number 
        
        algo:
            1. store lengths of haystack and needle (m and n)
            2. if n < m, it means needle cant fit in haystack, return - 1
            3. constants:
                RADIX = base which is 26
                MOD = 10**9 + 7
                MAX_WEIGHT = largest value of the weight of any character, which is 26^m
            4. define hash_value() function:
                iterate string from right to left, multiplying and applying radix
                ans and factor are modded
            5. iterate over the haystack from 0 to n-m
                compute hash of first window
                then check if hashses match
                slide windwo and compute hash in O(1) time
        additional notes on integer overflow
        In Python3, we have used ord to convert a character to its ASCII value. Now, the ASCII Value of a is 97, and we want to scale it down to 0, hence, we have used ord(string[i]) - 97 in the code. Since Python3 can handle large integers, we need not use MOD. 
        In that case, the hash value will be unique, and we can simply return window_start if the hash value matches. But operations (addition, multiplication, and subtraction) on large integers are slow. 
        Since MAX_WEIGHT is MOD 2 6 � 26 m , therefore, we have calculated it iteratively, instead of using the pow() function, for a few reasons: pow() function can overflow, and we don't want that. In Iteration, we are MODing the value at each step, so we are safe. 
        In Java, the Math.pow() function returns a double, and it has a precision error. Thus, iterating is a better option. MODing at each step of iteration is better than first computing the entire large 2 6 � 26 m and then MODing it. In Java, we have added MOD in the � ( 1 ) O(1) formula to avoid downflowing to a negative value. 
        When we subtract two MODed integers, we need to add MOD. In We can write (a - b)%MOD = (a%MOD - b%MOD)%MOD. In actual case a > b but after MODing, a%MOD can be smaller than b%MOD. In that case, we need to add MOD. Thus (a - b)%MOD = (a%MOD - b%MOD + MOD)%MOD.
        '''
        m = len(needle)
        n = len(haystack)
        if m < m:
            return -1
        
        #constants
        RADIX = 26
        MOD = 1_000_000_033
        MAX_WEIGHT = 1 #need this for the shift right
        
        for _ in range(m):
            MAX_WEIGHT = (MAX_WEIGHT*RADIX) % MOD #insteaf of using power
            
        def hash_value(string):
            ans = 0
            factor = 1
            
            for i in range(m-1,-1,-1):
                ans += ((ord(string[i]) - ord('a'))*factor) % MOD
                factor = (factor * RADIX) % MOD
            
            return ans % MOD
        
        #get hash of needle
        hash_needle = hash_value(needle)
        
        # Check for each m-substring of haystack, starting at window_start
        for window_start in range(n - m + 1):
            if window_start == 0:
                # Compute hash of the First Substring
                hash_hay = hash_value(haystack[:m])
            else:
                # Update Hash using Previous Hash Value in O(1)
                hash_hay = ((hash_hay * RADIX) % MOD
                            - ((ord(haystack[window_start - 1]) - 97)
                            * MAX_WEIGHT) % MOD
                            + (ord(haystack[window_start + m - 1]) - 97)
                            + MOD) % MOD

            # If hash matches, Check Character by Character. 
            # Because of Mod, spurious hits can be there.
            if hash_needle == hash_hay:
                for i in range(m):
                    if needle[i] != haystack[i + window_start]:
                        break
                if i == m - 1:
                    return window_start

        return -1

#we can also use two different hashes to avoid spurious hits even more
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        m = len(needle)
        n = len(haystack)

        if n < m:
            return -1

        # CONSTANTS
        RADIX_1 = 26
        MOD_1 = 10**9+33
        MAX_WEIGHT_1 = 1
        RADIX_2 = 27
        MOD_2 = 2**31-1
        MAX_WEIGHT_2 = 1

        for _ in range(m):
            MAX_WEIGHT_1 = (MAX_WEIGHT_1 * RADIX_1) % MOD_1
            MAX_WEIGHT_2 = (MAX_WEIGHT_2 * RADIX_2) % MOD_2

        # Function to compute hash_pair of m-String
        def hash_pair(string):
            hash_1 = hash_2 = 0
            factor_1 = factor_2 = 1
            for i in range(m - 1, -1, -1):
                hash_1 += ((ord(string[i]) - 97) * (factor_1)) % MOD_1
                factor_1 = (factor_1 * RADIX_1) % MOD_1
                hash_2 += ((ord(string[i]) - 97) * (factor_2)) % MOD_2
                factor_2 = (factor_2 * RADIX_2) % MOD_2

            return [hash_1 % MOD_1, hash_2 % MOD_2]

        # Compute hash pairs of needle
        hash_needle = hash_pair(needle)

        # Check for each m-substring of haystack, starting at window_start
        for window_start in range(n - m + 1):
            if window_start == 0:
                # Compute hash pairs of the First Substring
                hash_hay = hash_pair(haystack)
            else:
                # Update Hash pairs using Previous using O(1) Value
                hash_hay[0] = (((hash_hay[0] * RADIX_1) % MOD_1
                               - ((ord(haystack[window_start - 1]) - 97)
                                  * (MAX_WEIGHT_1)) % MOD_1
                               + (ord(haystack[window_start + m - 1]) - 97))
                               % MOD_1)
                hash_hay[1] = (((hash_hay[1] * RADIX_2) % MOD_2
                               - ((ord(haystack[window_start - 1]) - 97)
                                  * (MAX_WEIGHT_2)) % MOD_2
                               + (ord(haystack[window_start + m - 1]) - 97))
                               % MOD_2)

            # If the hash matches, return immediately.
            # Probability of Spurious Hit tends to zero
            if hash_needle == hash_hay:
                return window_start
        return -1
    
#kmp
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        '''
        the problem with rabin-karp, is that we do a lot of repeated comparisions if we had already advanced quite far in needle
        if there is a mismatch, we only move by 1, rather we would want to advance to the last index in the mismatch
        but this doesn't always work!!!! good point here
        
        recall concept of proper prefix and proper suffix, substrings that can be prefix and suffix of string but != string
        border, substring of a string that is both proper prefix and proper suffix
        longest prefix also suffix (pi table, also defined as lps)
        or somtimes called the n table
        lps[i] is the lenght of the longest border of the string[0...i]
        lps[0] = 0 becuase for a single character, the only border is "" whose length is 0
        lps array naivly takes (N^3) time
        needed for needle
        for i = 1 to m-1
          for k = 0 to i
             if needle[0..k-1] == needle[i-(k-1)..i]
               longest_border[i] = k
                temp = "abcdabeeabf"
        N = len(temp)
        for i in range(1,N):
            for k in range(i+1):
                pref = temp[0:k]
                suff = temp[i- (k):i]
                print("pref is..", pref,"suff is.." ,suff)
        '''
        m = len(needle)
        n = len(haystack)
        
        if n < m:
            return -1
        
        #preprocesing, lps array
        lps = [0]*m
        #store length of previous border string
        prev = 0
        i = 1
        while i < m:
            if needle[i] == needle[prev]:
                #increase length of previous border
                prev += 1
                lps[i] = prev
                i += 1
            else:
                #only empty border exists, which has length zero
                if prev == 0:
                    lps[i] = 0
                    i += 1
                else:
                    #take previous
                    prev = lps[prev-1]
        
        #searching
        haystack_ptr = 0
        needle_ptr = 0
        
        while haystack_ptr < n:
            if haystack[haystack_ptr] == needle[needle_ptr]:
                #increment both
                haystack_ptr += 1
                needle_ptr += 1
                #if all matche in needle, return
                if needle_ptr == m:
                    return haystack_ptr - m
            else:
                #no matching yet
                if needle_ptr == 0:
                    haystack_ptr += 1
                else:
                    #optimally shift left needle , and dont change haystack
                    #because have the longest border pre calculate
                    needle_ptr = lps[needle_ptr - 1]
        
        return -1

##########################################
# 2444. Count Subarrays With Fixed Bounds
# 03MAR22
###########################################
class Solution:
    def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
        '''
        return the number of subarrays where the minimum == minK
        and the maximum == maxK
        in the topics, i see sliding window and montonic queue
        
        if i have a subarray, nums[i:j] such that all numbers in the array are between minK and maxK
        the answer must be j-i + 1
        let dp(i,j) be the number of subarrays in 
        
        focus on each subarray ending at i
        ask, how many valid subarrays ending at (i), more so, every elemenet in the subarray must conatain both minK and maxK
        and must also be in the range [minK,maxK]
        
        so we need to record three indices
            1. leftBound: the most recent value out of ther ange [minK,maxK]
            2. maxPosition: most recent index with value == maxK
            3. minPosition: most recent index wit value == minL
            
        we need to record the most rent value out of the range 
            because we are fixing the right end of the subarray (by considering how many valid subarrays end at current index), we need to know the farthest left we can start considering a subarray from
        we need to record the most recent minK and maxK
            becasue a valid subarray needs to contains at least one minK and maxK
            once we find the indices of the most recent minK and maxK, we can take the samller one (call it smaller)
            then the range [smaller,i] contains at least one minK and maxK
        
        let smaller = min(minPosition,maxPosition) so the range [smaller,i] contains at least on minK and maxK
        no we try to extend the subarray [smaller,i] from the left side
            (smaller-1,i) which we can do as long as we havent met a value out of the range
            
        if leftBound is to the right, smaller,smaler -leftBound is not a valid subarray
        so we can just treat as zero, to avoid negative results
        therefore the number of valid subarrays is max(0,min(minPositions,maxPostiion) - leftBound)
        '''
        count = 0
        min_pos, max_pos, left_bound, = -1,-1,-1,
        
        for i,num in enumerate(nums):
            #outside the interval mark the index
            if num < minK or num > maxK:
                left_bound = i
            
            #update most recent min+pos and max_pos
            if num == minK:
                min_pos = i
            if num == maxK:
                max_pos = i
                
            #count up valid subarrays ending at i
            #i.e the number of valid subarrays is the number of eleements
            count += max(0,min(min_pos,max_pos) - left_bound)
            
        return count

######################################
# 819. Most Common Word
# 08MAR23
######################################
#bleagh
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        '''
        parse string letter by letter to get words
        put into counter in the fly
        '''
        N = len(paragraph)
        counts = Counter()
        
        banned = set(banned)
        curr_word = ""
        
        i = 0
        max_counts = 0
        
        while i < N:
            #white spaces
            while i < N and not paragraph[i].isalpha():
                i += 1
            
            #we must be at a word
            while i < N and paragraph[i].isalpha():
                curr_word += paragraph[i]
                i += 1
            
            #check banned words
            if curr_word not in banned:
                counts[curr_word.lower()] += 1
                max_counts = max(max_counts,counts[curr_word.lower()]  )
            
            curr_word = ""
            i += 1
        
        for k,v in counts.items():
            if v == max_counts:
                return k
        
    class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        '''
        eaiest to replace special chars with spsace,
        reduce to lower, split then count
        '''
        banned = set(banned)
        new_string = ""
        
        for ch in paragraph:
            if ch.isalnum():
                new_string += ch.lower()
            else:
                new_string += " "
        
        words = new_string.split()
        counts = Counter()
        
        for w in words:
            if w not in banned:
                counts[w] += 1
        
        return max(counts.items(), key = operator.itemgetter(1))[0]
        
#single pass, using buffer
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:

        banned_words = set(banned)
        ans = ""
        max_count = 0
        word_count = defaultdict(int)
        word_buffer = []

        for p, char in enumerate(paragraph):
            #1). consume the characters in a word
            if char.isalnum():
                word_buffer.append(char.lower())
                if p != len(paragraph)-1:
                    continue

            #2). at the end of one word or at the end of paragraph
            if len(word_buffer) > 0:
                word = "".join(word_buffer)
                if word not in banned_words:
                    word_count[word] +=1
                    if word_count[word] > max_count:
                        max_count = word_count[word]
                        ans = word
                # reset the buffer for the next word
                word_buffer = []

        return ans

#####################################
# 1055. Shortest Way to Form String
# 15MAR23
#####################################
class Solution:
    def shortestWay(self, source: str, target: str) -> int:
        '''
        return minimum number of subsequences from source such that their concatenation == target
        source = "abc"
        target = "abcbc"
        
        from source we can get
            a, b, c
            ab, ac
            bc
            abc
        ans = 2
        
        we can rephrase the problem as:
            return the minimum number of times we need to concat source to get target as a subsequence
            if we concat source len(target) times, we are sure to get an answer
            
        first we try string concat of 1, then 2, then 3...
        keep going until we cant
        '''
        def is_subseq(original,subseq):
            i,j = 0,0
            while i < len(original) and j < len(subseq):
                #same
                if original[i] == subseq[j]:
                    i += 1
                    j += 1
                else:
                    i += 1
            
            return j == len(subseq)
        
        
        #first check that we can't make targer
        set_source = set(source)
        for ch in target:
            if ch not in set_source:
                return -1
        
        num_concats = 1
        curr_original = source
        
        while not is_subseq(curr_original,target):
            curr_original += source
            num_concats += 1
        
        return num_concats
    
#we can use binary search too
class Solution:
    def shortestWay(self, source: str, target: str) -> int:
        '''
        return minimum number of subsequences from source such that their concatenation == target
        source = "abc"
        target = "abcbc"
        
        from source we can get
            a, b, c
            ab, ac
            bc
            abc
        ans = 2
        
        we can rephrase the problem as:
            return the minimum number of times we need to concat source to get target as a subsequence
            if we concat source len(target) times, we are sure to get an answer
            
        first we try string concat of 1, then 2, then 3...
        keep going until we cant
        '''
        def is_subseq(original,subseq):
            i,j = 0,0
            while i < len(original) and j < len(subseq):
                #same
                if original[i] == subseq[j]:
                    i += 1
                    j += 1
                else:
                    i += 1
            
            return j == len(subseq)
        
        
        #first check that we can't make targer
        set_source = set(source)
        for ch in target:
            if ch not in set_source:
                return -1
        
        low = 1
        high = len(target) #cannot be more than len(target times)
        
        
        while low < high:
            mid = low + (high - low) // 2
            can_do = is_subseq(source*mid,target)
            if can_do:
                high = mid
            else:
                low = mid + 1
        
        return low
    
#actual answer is two pointers
class Solution:
    def shortestWay(self, source: str, target: str) -> int:
        '''
        we actually don't need to keep concating source, we just need to keep looping over it
        but intead of trying all concatentations, we want to find the immediate next occurence of the next char of target in source
        two points, but one pointer into source just gets modded len(source)
        if the pointer hits zer0, we have to icnreament the count of concats
        why?
            because we have gone through soruce already
        
        proof of optmiality
            we start with the smallest, if we find a new solution, it has to be the smallest solution
            proof by contradiction
        '''
        #first check that we can't make targer
        set_source = set(source)
        for ch in target:
            if ch not in set_source:
                return -1
            
        
        count = 0
        source_ptr = 0
        
        #loop through target to find the immediate ocurrence
        for ch in target:
            #if we have gotten to the beginning of source again
            if source_ptr == 0:
                count += 1
            
            #find first occruecnce of source in target
            while source[source_ptr] != ch:
                source_ptr = (source_ptr + 1) % len(source)
                
                #gone aroung again
                if source_ptr == 0:
                    count += 1
            
            #advance to next char
            source_ptr = (source_ptr + 1) % len(source)
        
        return count

##########################################
# 958. Check Completeness of a Binary Tree
# 15MAR23
##########################################
#close one...
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        '''
        complete binary tree is one where every level is filled as far left as possible (except the last level)
        and for each level, it can have between 1 and 2^h nodes inclusive
        
        first find the depth of tree
        then level order checking number of nodes between 1 and 2**curr_level
        '''
        def get_depth(node):
            if not node:
                return 0
            left = get_depth(node.left)
            right = get_depth(node.right)
            return max(left,right) + 1
        
        
        depth = get_depth(root) 
        curr_level = 0
        q = deque([root])
        
        while curr_level != depth - 1:
            N = len(q)
            #check
            if N > 2**curr_level:
                return False
            for _ in range(N):
                curr = q.popleft()
                q.append(curr.left)
                q.append(curr.right)
            curr_level += 1
        
        #we are at the second to last level here
        for _ in range(len(q)):
            curr = q.popleft()
            if not curr.left:
                return False
        
        return True
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        '''
        we can define a complete binary tree as a tree that  has no node to the right of the first null node
        and no node at a greater level than the first null node
        
        intuitino:
             The level-order traversal array of a complete binary tree will never have a null node in between non-null nodes.
        while doing level order, just check if have already scene an empty node before hand
        the only empty nodes should be on the last level
        '''
        if not root:
            return True
        
        found_empty = False
        
        q = deque([root])
        
        while q:
            curr = q.popleft()
            
            if not curr:
                found_empty = True
            else:
                if found_empty:
                    return False

                q.append(curr.left)
                q.append(curr.right)
        
        return True
                    
#dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isCompleteTree(self, root: Optional[TreeNode]) -> bool:
        '''
        recall that in a binary tree
        given a node at index i
            we can find its children at 2*i + 1 and 2*i + 2
        if there n nodes in the tree
        then an index cannot ever be mofre than n because ti would imply that we have a missing node somewhere
        
        intution
            a nodes index in the binary tree cannot be more than the number of nodes in tree
        '''
        def count_nodes(node):
            if not node:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        
        N = count_nodes(root)
        
        def dp(node,index,N):
            if not node:
                return True
            if index >= N:
                return False
            
            left = dp(node.left,2*index + 1,N)
            right = dp(node.right,2*index + 2,N)
            
            #inorder for this to be true, both left and right must be true
            return left and right
        
        return dp(root,0,N)
    
#######################################
# 1472. Design Browser History
# 18MAR23
#######################################
#nice idea
class BrowserHistory:
    '''
    after visiting, forward history is cleared
    try:
        hashmap:
            key is current step
            simple add and increment
            but when visiting again, i'd have to delete everything greater than the current step if i went back
    '''

    def __init__(self, homepage: str):
        self.cache = {}
        self.curr_step = 0
        self.max_step = 0

    def visit(self, url: str) -> None:
        self.cache[self.curr_step] = url
        self.curr_step += 1
        self.max_step = max(self.max_step,self.curr_step)
        #clear everythihng up to max
        for i in range(self.curr_step,self.max_step):
            del self.cache[i]

    def back(self, steps: int) -> str:
        #find go to
        go_to = self.curr_step - steps
        if go_to in self.cache:
            self.curr_step = go_to
            return 

    def forward(self, steps: int) -> str:
        


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)

#two stacks
class BrowserHistory:
    '''
    the idea is to use two stack
        one for future and one for past
        and variable holding current webpage
        
    visit:
        make new url be the currnet one
        and store current in history stack, then just clear the future stack
        
    back:
        need to go back steps
        while we have steps and there is stuff in the history stack, pop and puch current on to furture stafk
    
    forward:
        we need to go forward by ursls
        while we have steps and while there is forward, push current in the history stack and pop most recenlt visited from forward
    '''

    def __init__(self, homepage: str):
        self.history = []
        self.future = []
        self.curr_page = homepage

    def visit(self, url: str) -> None:
        self.history.append(self.curr_page)
        self.curr_page = url
        self.future = []
        

    def back(self, steps: int) -> str:
        while steps > 0 and self.history:
            #push current
            self.future.append(self.curr_page)
            self.curr_page = self.history.pop()
            steps -= 1
        
        return self.curr_page

    def forward(self, steps: int) -> str:
        while steps > 0 and self.future:
            self.history.append(self.curr_page)
            self.curr_page = self.future.pop()
            steps -= 1
        
        return self.curr_page
        


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)

#using doubly linked list
class Node:
    def __init__(self,data):
        self.data = data
        self.prev = None
        self.next = None

class BrowserHistory:
    '''
    we can use a double linked list 
    node obkcets:
        url
        prev
        none
    
    visit we just insert at curr,
    inserting cuases the previous nodes to disappear
    careful of the memory leak in C++ implementatino, we need to deallocate
    
    '''
    def __init__(self, homepage: str):
        self.head = Node(homepage)
        self.curr = self.head
        

    def visit(self, url: str) -> None:
        #prepare new node
        new_node = Node(url)
        self.curr.next = new_node
        new_node.prev = self.curr
        #advance
        self.curr = new_node

    def back(self, steps: int) -> str:
        while steps and self.curr.prev:
            self.curr = self.curr.prev
            steps -= 1
        
        return self.curr.data
        

    def forward(self, steps: int) -> str:
        while steps and self.curr.next:
            self.curr = self.curr.next
            steps -= 1
        
        return self.curr.data
        


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)

#############################################
# 1485. Clone Binary Tree With Random Pointer
# 18MAR23
#############################################
#i liked the idea
# Definition for Node.
# class Node:
#     def __init__(self, val=0, left=None, right=None, random=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.random = random

class Solution:
    def copyRandomBinaryTree(self, root: 'Optional[Node]') -> 'Optional[NodeCopy]':
        '''
        LMAO! they figured out how to remove the cheaters by making us return a new class
        insteaf of copy.deepcopy(class)
        we need to build the tree using recursion
        randome pointer points to a node


        '''
        def make_copy(node):
            if not node:
                return None
            
            new_copy = NodeCopy()
            new_copy.left = make_copy(node.left)
            new_copy.right = make_copy(node.right)
            new_copy.random = make_copy(node.random)
            return new_copy
        
        
        return make_copy(root)
    
# Definition for Node.
# class Node:
#     def __init__(self, val=0, left=None, right=None, random=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.random = random

class Solution:
    def copyRandomBinaryTree(self, root: 'Optional[Node]') -> 'Optional[NodeCopy]':
        '''
        LMAO! they figured out how to remove the cheaters by making us return a new class
        insteaf of copy.deepcopy(class)
        we need to build the tree using recursion
        randome pointer points to a node

        hashamp stores node in input and makes a NodeCopy
        recurse on each node and fetch its copy

        we need to use extra space to save the copies, then swap over
        this is similar to dfs with a seen set, we don't want to recopy a node we have already copied
        just fetch the copy from the hashamp

        similar to the way we stop dfs'ing on an already seen node in a graph

        '''
        clones = {}
        
        def make_copy(node):
            if not node:
                return 
            if node in clones:
                return clones[node]
            
            new_node = NodeCopy(node.val)
            clones[node] = new_node
            clones[node].left = make_copy(node.left)
            clones[node].right = make_copy(node.right)
            clones[node].random = make_copy(node.random)
            return new_node
        
        
        return make_copy(root)
    
# Definition for Node.
# class Node:
#     def __init__(self, val=0, left=None, right=None, random=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.random = random

class Solution:
    def copyRandomBinaryTree(self, root: 'Optional[Node]') -> 'Optional[NodeCopy]':
        '''
        think of the easier problem
        if we only had left and right, we could dfs, then copy left and right on first pass as nodecopies
        then on the second pass, traverse the randome but make the connections in the copied tree
        idea is to keep hashmap of old nodes to copied nodes
        
        intuition:
            we know which node the random pointer of any current node (node of the given tree) points to
            and if we store what would be the respective new node (the copied node) for any old node (use hashmap)
            then we can easily tell which node the radnom point of the new should point to
        '''
        orig_to_copy = {None:None} #interseting this only works if we init the dictinoat to have none values
        
        def copy_left_right(node):
            if not node:
                return None
            new_node = NodeCopy(node.val)
            new_node.left = copy_left_right(node.left)
            new_node.right = copy_left_right(node.right)
            #put into hashsmape
            orig_to_copy[node] = new_node
            return new_node
        
        #return copy_left_right(root)
        #notice that we just get the nodes with empty random pointers
        #now we need to traverse this tree to make the random connections
        def copy_random_ptrs(node):
            if not node:
                return None
            #first get the copty
            copied_node = orig_to_copy[node]
            #get the copy of the random node
            copied_random_node = orig_to_copy[node.random]
            #make new connection
            copied_node.random = copied_random_node
            copy_random_ptrs(node.left)
            copy_random_ptrs(node.right)
            
        copy = copy_left_right(root)
        copy_random_ptrs(root) #call this on the original to make connections into copy
        return copy
    
# Definition for Node.
# class Node:
#     def __init__(self, val=0, left=None, right=None, random=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.random = random

class Solution:
    def __init__(self,):
        self.already_copied: dict[Node,NodeCopy] = {None:None}
            
    #function to make copy
    def make_copy(self,node):
        if not node:
            return None
        
        if node in self.already_copied:
            return self.already_copied.get(node)
        
        copied = NodeCopy(node.val)
        #i have to make a copy first before recusring
        #but not only make a copy, mark it as already copied
        #otherwise it will just keep coping beforing finally caching
        self.already_copied[node] = copied
        copied.left = self.make_copy(node.left)
        copied.right = self.make_copy(node.right)
        copied.random = self.make_copy(node.random)
        return copied
        
    
    def copyRandomBinaryTree(self, root: 'Optional[Node]') -> 'Optional[NodeCopy]':
        '''
        mark node is already seen, rather already copied
        if node is already coped, fetch it and return it
        '''
        return self.make_copy(root)
    
############################################################
# 211. Design Add and Search Words Data Structure (REVISTED)
# 19MAR23
#############################################################
#close one
class Node:
    def __init__(self,):
        self.children = {}
        self.is_end = False

class WordDictionary:

    def __init__(self):
        '''
        the issue is that the dots can be anywhere
        i can use recursion on the word trie and check all children if at a dot
        '''
        self.trie = Node()

    def addWord(self, word: str) -> None:
        curr = self.trie
        for ch in word:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                new_node = Node()
                curr.children[ch] = new_node
                curr = new_node
        #mark
        curr.is_end = True

    def search(self, word: str) -> bool:
        def dfs(word,curr_node):
            if word[0] not in curr_node.children:
                return False
            elif word[0] in curr_node.children:
                curr_node = curr_node.children[word[0]]
            #if it's a dot, try all children
            elif word[0] == '.':
                for child in curr_node.children:
                    if dfs(word[1:],curr_node.children[child]):
                        return True
                return False
            return curr_node.is_end
        
        return dfs(word,self.trie)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)

#the actual solution
class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}


    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        node = self.trie

        for ch in word:
            if not ch in node:
                node[ch] = {}
            node = node[ch]
        node['$'] = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any letter.
        """
        def search_in_node(word, node) -> bool:
            for i, ch in enumerate(word):
                if not ch in node:
                    # if the current character is '.'
                    # check all possible nodes at this level
                    if ch == '.':
                        for x in node:
                            if x != '$' and search_in_node(word[i + 1:], node[x]):
                                return True
                    # if no nodes lead to answer
                    # or the current character != '.'
                    return False
                # if the character is found
                # go down to the next level in trie
                else:
                    node = node[ch]
            return '$' in node

        return search_in_node(word, self.trie)
    
##################################
# 830. Positions of Large Groups
# 21MAR23
###################################
#works
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        '''
        this is just 
        '''
        curr_char = s[0]
        curr_count = 1
        ans = []
        curr_group = [0]
        
        for i in range(1,len(s)):
            if s[i] == curr_char:
                curr_count += 1
            else:
                if curr_count >= 3:
                    curr_group.append(i-1)
                    ans.append(curr_group)
                    curr_group = [i]
                    curr_count = 1
                else:
                    curr_count = 1
                    curr_group = [i]
                curr_char  = s[i]
        #for the last one        
        if curr_count >= 3:
            curr_group.append(i)
            ans.append(curr_group)  
        return ans
    
#phew
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        '''
        we can maintain two poiters, left and right
        keep advancing right until we are no longer in the group
        once we are out of the group, see is the pointers are 3 away (inclusive)
        '''
        N = len(s)
        indicies = []
        left, right = 0,0
        while right < N:
            while right < N-1 and s[right] == s[right+1]:
                right += 1
            if (right - left + 1) >= 3:
                indicies.append([left,right])
            
            right += 1
            left = right
        
        return indicies
    

#another way using for loop
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        N = len(s)
        left = 0
        ans = []
        
        for right in range(N):
            #got to the end or mismtach with adjacent chars
            if right == N -1 or s[right] != s[right+1]:
                if right - left + 1 >= 3:
                    ans.append([left,right])
                
                #new group
                left = right + 1
        
        return ans
    

#######################################
# 2348. Number of Zero-Filled Subarrays
# 21MAR23
######################################
class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        '''
        sliding window?
        keep advancing right pointers until we have zeros, for every advancment of right, increment count
        when come to a non zero, move left
        
        if we have a sub array [0,0,0]
        we can have  3 [0]
        we can have 2 [0,0]
        we can have 1 [0,0,0]
        
        total subarrays = 3 + 2 + 1
        which is just sum of series from 1 to 3
        
        so given a lenghth N all zeros, total number of contributinos of subarrays would be
            count = N*(N+1) / 2
            
        now the problem just becomes find all subarray of [0]
        get the lenght, and return count
        '''
        count = 0
        N = len(nums)
        zeros = []
        
        for num in nums:
            if num == 0:
                zeros.append(num)
            else:
                if zeros:
                    count += (len(zeros)*(len(zeros) + 1)) // 2
                zeros = []
        
        if zeros:
            count += (len(zeros)*(len(zeros) + 1)) // 2
        return count
    
#turn out we just count the streaks
#but every time we have a zero increment the an by the current streak size
class Solution:
    def zeroFilledSubarray(self, nums: List[int]) -> int:
        '''
        keep track of curr streak size and incremtn count by the streak size
        if we have a subarrya of size k and we were to extend this to k + 1,
        we effectivley incrmenet the count by k + 1 subarrays
        [0,0,0]
        we already have
        [0] 3 times
        [0,0] 2 times
        [0,0,0] 1 time
        
        going to 4 
        [0,0,0,0]
        num arrays increase by 4
        '''
        count = 0
        size = 0
        
        for num in nums:
            if num == 0:
                size += 1
            else:
                size = 0
            print(size)
            count += size
        
        return count

###########################################
# 1504. Count Submatrices With All Ones
# 21MAR23
############################################
class Solution:
    def numSubmat(self, mat: List[List[int]]) -> int:
        '''
        i can store in each i,j the number of submatrices who'se left corner ends at i,j
        then just sum across the whole dp array
        
        similar to numer of squares in histogram
        
        follow the hints, i dont know exactly why
        1. for each row i, create an array nums where:
            if mat[i][j] == 0, then nums[j] = 0 else, nums[j] = nums[j-1] + 1
        2. in the ith row, the number of rectangles between column j and k (inclusive) and ends in row i is:
            sum(min(nums[j..idx])) where idx goes from j to k
            
        for each row, increment the count by the smallest width
        on the dp matrix
            keep going down a column, recording the smallest width as the number of rectangles starting from the left
            
        For each row, determine the number of submatrices of all 1's
        that terminate at each location and begin on the same row.
        
        Process the above matrix one row at a time. Moving down each
        column, determine the number of submatrices of all 1's that
        start on that row looking left.  Repeat for each row and return
        the total number of submatrices.

        For each 1 within a row, count the submatrices that contain
        the 1 and start on the same row either at the 1 or to the
        left of the 1. Proceed down the column that contains the 1
        until reaching a 0 or the bottom row of the matrix. While
        proceeding down a column, the width of the submatrices stays
        the same or gets thinner.
        '''
        
        rows = len(mat)
        cols = len(mat[0])
        
        dp = [[0]*cols for _ in range(rows)]
        #dp(i,j) gives the number of rectables in a row ending at [i,j] who's start is at i
        #and are consective

        
        #base cases
        for i in range(rows):
            if mat[i][0] == 1:
                dp[i][0] = 1
        
        for i in range(rows):
            for j in range(1,cols):
                if mat[i][j] == 1:
                    dp[i][j] = dp[i][j-1] + 1
        
        submatrices = 0
        for i in range(rows):
            for j in range(cols):
                if dp[i][j] != 0:
                    curr_row = i
                    submatrix_width = dp[i][j]
                    #keep going down
                    while curr_row < rows and dp[i][j] != 0:
                        submatrix_width = min(submatrix_width,dp[curr_row][j])
                        submatrices += submatrix_width
                        curr_row += 1
        
        return submatrices
    
####################################################
# 2492. Minimum Score of a Path Between Two Cities
# 22MAR23
#####################################################
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        from the hints, if all the cities are connect, i can use any road
        so the answer is just the minimum dist of all the roads (for that connected compoenent)

        we want the minimum score of all possible paths from 1 to n, we don't necessarily need to vsit all the nodes
        dfs on each node to find the minimum path

        i have to start at 1 anyway, 
        addind a new road, may reduce the minimum score

        from the hint, remove nodes not connected to 1
        then solve the problem on the connected graph
        '''
        adj_list = defaultdict(list)

        for u,v,dist in roads:
            #enetyr is going to be node: (neigh,dist)
            adj_list[u].append((v,dist))
            adj_list[v].append((u,dist))


        def dfs_connected_to_one(city,seen):
            seen.add(city)
            for neigh,dist in adj_list[city]:
                if neigh not in seen:
                    dfs_connected_to_one(neigh,seen)

        connected_to_one = set()
        dfs_connected_to_one(1,connected_to_one)

        #all these are connected to 1 somehow
        #dfs on these and get the minimum raod for the minimum score
        self.ans = float('inf')

        def dfs_find_min(city,seen,connected):
            seen.add(city)
            for neigh,dist in adj_list[city]:
                self.ans = min(self.ans,dist)
                #upaate ans no matter what, and then if its connected dfs on it, because we can go back on the edge
                if neigh in connected_to_one and neigh not in seen:
                    dfs_find_min(neigh,seen,connected_to_one)

        seen = set()
        dfs_find_min(1,seen,connected_to_one)
        return self.ans
    
#single source dfs
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        we can also just dfs starting from 1
        '''
        adj_list = defaultdict(list)
        seen = [False]*(n+1)
        self.ans = float('inf')
        
        for u,v,dist in roads:
            #enetyr is going to be node: (neigh,dist)
            adj_list[u].append((v,dist))
            adj_list[v].append((u,dist))
            
        
        def dfs(city):
            seen[city] = True
            
            for neigh,dist in adj_list[city]:
                self.ans = min(self.ans,dist)
                if not seen[neigh]:
                    dfs(neigh)
                    
        dfs(1)
        return self.ans
    
#bfs
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        bfs
        '''
        adj_list = defaultdict(list)
        seen = [False]*(n+1)
        self.ans = float('inf')
        
        for u,v,dist in roads:
            #enetyr is going to be node: (neigh,dist)
            adj_list[u].append((v,dist))
            adj_list[v].append((u,dist))
            
        
        q = deque([1])
        
        while q:
            city = q.popleft()
            seen[city] = True
            
            for neigh,dist in adj_list[city]:
                self.ans = min(self.ans,dist)
                if not seen[neigh]:
                    q.append(neigh)
                    
        return self.ans

#using union find
class UF:
    def __init__(self,n):
        self.size = [1]*(n+1)
        self.parent = [i for i in range(n+1)] #cities are from 1 to n
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def join(self,x,y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        
        if parent_x == parent_y:
            return
        elif self.size[parent_x] > self.size[parent_y]:
            self.parent[parent_y] = parent_x
            self.size[parent_x] += self.size[parent_y]
            self.size[parent_y] = 1
        
        elif self.size[parent_y] > self.size[parent_x]:
            self.parent[parent_x] = parent_y 
            self.size[parent_y] += self.size[parent_x]
            self.size[parent_x] = 1
        
        else:
            self.parent[parent_y] = parent_x
            self.size[parent_x] += self.size[parent_y]
            self.size[parent_y] = 1

class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        how could we use union find?
        we need to connect each city that is part of the connected component
        when ever we join an edge and that connecting edge points 1, we can minize this current score by taking the min dist
        call find on each (a,b)
        if any of their parents points to 1, they must be connected
        
        on the first pass, join each city along an edge (i.e make them connected)
        the on the second pass, find the representative of city 1, and see if we can get to that through each edge in roads (we only need to check one)
        minimze along the way
        '''
        uf = UF(n)
        
        #first pass join
        for u,v,dist in roads:
            uf.join(u,v)
        
        ans = float('inf')
        ones_parent = uf.find(1)
        
        for u,v,dist in roads:
            if uf.find(u) == ones_parent:
                ans = min(ans,dist)
        
        return ans

##########################################
# 1236. Web Crawler
# 22MAR23
###########################################
# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
#class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """

class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        '''
        jesus fuck, 20/21, meh i think i get it :)
        '''
        all_urls = set()
        def get_all(url,parser):
            #nothing to parse
            all_urls.add(url)
            if len(parser.getUrls(url)) == 0:
                return
            else:
                for neigh in parser.getUrls(url):
                    if neigh not in all_urls:
                        get_all(neigh,parser)
                    
        get_all(startUrl, htmlParser)
        #find host name from start url, they ust http protocol
        host = startUrl[:7]
        i = 7
        while i < len(startUrl) and startUrl[i] != '/':
            host += startUrl[i]
            i += 1
            
        ans = []
        for url in all_urls:
            if url.startswith(host):
                ans.append(url)
        
        return ans

# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
#class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """

class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        '''
        jesus fuck, 20/21, meh i think i get it :)
        '''
        
        def get_hostname(url):
            return url.split('/')[2]
        
        start_host = get_hostname(startUrl)
        seen = set()
        
        def get_all(url,parser):
            seen.add(url)
            #base case
            if len(parser.getUrls(url)) == 0:
                return
            else:
                for neigh in parser.getUrls(url):
                    if get_hostname(neigh) == start_host and neigh not in seen:
                        get_all(neigh,parser)
                    
        get_all(startUrl, htmlParser)

        
        return seen
    
#can use bfs
# """
# This is HtmlParser's API interface.
# You should not implement it, or speculate about its implementation
# """
#class HtmlParser(object):
#    def getUrls(self, url):
#        """
#        :type url: str
#        :rtype List[str]
#        """

class Solution:
    def crawl(self, startUrl: str, htmlParser: 'HtmlParser') -> List[str]:
        '''
        jesus fuck, 20/21, meh i think i get it :)
        '''
        
        def get_hostname(url):
            return url.split('/')[2]
        
        start_host = get_hostname(startUrl)
        seen = set()
        
        q = deque([startUrl])
        
        while q:
            url = q.popleft()
            seen.add(url)
            for neigh in htmlParser.getUrls(url):
                if get_hostname(neigh) == start_host and neigh not in seen:
                    q.append(neigh)
        
        return seen

########################################################
# 1319. Number of Operations to Make Network Connected
# 23MAR23
#########################################################
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        '''
        well union find scream out obvie for connected reasons
        use union find to get the connected components
        
        first off, 
            if i have n computers that need to be connected
            i need at leats n-1 cables
            
        count total connected components using dfs
        LMAOOOO this fucking worked

        imagine that each connected component is simplified to just a single node
        to connec this grah, we need a minimum of c - 1 edges
        if we have at least n-1 edges
        and we have c compoenents
        there can be at most n compoenents
        we need to connect the reamining (n-c) componnents, which use (n-c-1) edges
        (n-1) - (n-c)
        -1 + c
        c - 1 edges

        well if we have 1 compoenent, we don't need to use an edge, so it 1 - 1 = 0
        0 edges



        im still not sure its just num components less 1
        '''
        cables = len(connections)
        if cables < n - 1:
            return -1
        
        adj_list = defaultdict(list)
        for u,v in connections:
            adj_list[u].append(v)
            adj_list[v].append(u)
            
        num_components = 0
        
        def dfs(node,seen):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen)
        
        seen = set()
        for i in range(n):
            if i not in seen:
                dfs(i,seen)
                num_components += 1
        
        return num_components - 1
    

#bfs
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        cables = len(connections)
        if cables < n - 1:
            return -1
        
        adj_list = defaultdict(list)
        for u,v in connections:
            adj_list[u].append(v)
            adj_list[v].append(u)
            
        num_components = 0
        
        def bfs(node,seen):
            seen.add(node)
            q = deque([node])
            while q:
                curr = q.popleft()
                for neigh in adj_list[curr]:
                    if neigh not in seen:
                        seen.add(neigh)
                        q.append(neigh)

        seen = set()
        for i in range(n):
            if i not in seen:
                bfs(i,seen)
                num_components += 1
        
        return num_components - 1
    
#union find
class UF:
    def __init__(self,n):
        self.size = [1]*(n)
        self.parent = [i for i in range(n)] #cities are from 1 to n
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def join(self,x,y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        
        if parent_x == parent_y:
            return
        elif self.size[parent_x] > self.size[parent_y]:
            self.parent[parent_y] = parent_x
            self.size[parent_x] += self.size[parent_y]
            self.size[parent_y] = 1
        
        elif self.size[parent_y] > self.size[parent_x]:
            self.parent[parent_x] = parent_y 
            self.size[parent_y] += self.size[parent_x]
            self.size[parent_x] = 1
        
        else:
            self.parent[parent_y] = parent_x
            self.size[parent_x] += self.size[parent_y]
            self.size[parent_y] = 1

class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        '''
        union on edges,
        then check the number of disconnected componets
        for the uf approach, we initally start with n comoennets
        then traverse the edges, and if the nodes belong to different parents, join them and reduce the components by 1
        '''
        
        cables = len(connections)
        if cables < n - 1:
            return -1
        
        uf = UF(n)
        c = n
        for u,v in connections:
            if uf.find(u) != uf.find(v):
                uf.join(u,v)
                c -= 1
        
        return c - 1
    
##############################################################
# 1466. Reorder Routes to Make All Paths Lead to the City Zero
# 24MAR23
###############################################################
#bleagh
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        '''
        we have n cities, labeled 0 to n - 1 and we have n -1 road
        connections is a directed edge
        we want to reorient the roads such that any city can travel to city 0 (re orient means flip the direction)
        return the minimum number of edges changed
        guarnteed to reach city 0, 
            but people may not be able to get back, keep original config
        
        at most we have to reverse all the edges, so our answer can be more than len(connections)
        if i start going down a path from 0, and couldn't come back, that would mean i need to reverse some edges to get back
        
        i can dfs from 0
        then whil dfsing, if my neighbor's interger is > the current one i'm on, i need to erverse it
        dfs then count reversals
        this only worls if for each connection[i]
            connections[i][0] < connections[i][1]
        '''
        adj_list = defaultdict(list)
        for u,v in connections:
            adj_list[u].append(v)
            adj_list[v].append(u)
        
        

        seen = set()
        reversals = [0]
        
        def dfs(node,seen):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen)
                    reversals[0] += node < neigh
                    
        dfs(0,seen)
        return reversals[0]
    
#keep state of direction along an edge
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        '''
        we have n cities, labeled 0 to n - 1 and we have n -1 road
        connections is a directed edge
        we want to reorient the roads such that any city can travel to city 0 (re orient means flip the direction)
        return the minimum number of edges changed
        guarnteed to reach city 0, 
            but people may not be able to get back, keep original config
        
        at most we have to reverse all the edges, so our answer can be more than len(connections)
        if i start going down a path from 0, and couldn't come back, that would mean i need to reverse some edges to get back
        
        i can dfs from 0
        then whil dfsing, if my neighbor's interger is > the current one i'm on, i need to erverse it
        dfs then count reversals
        this only worls if for each connection[i]
            connections[i][0] < connections[i][1]
            
        we need to keep record of the direction 
        for each edge (1 as the forward direction) and (0 as the reverse)
        if child != parent, do down it and increment by the direction
        for each forward direction we revrse, we don't need to reverse
        '''
        adj_list = defaultdict(list)
        for u,v in connections:
            adj_list[u].append((v,1))
            adj_list[v].append((u,0))
        
        

        seen = set()
        reversals = [0]
        
        def dfs(node,seen):
            seen.add(node)
            for neigh,dirr in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen)
                    reversals[0] += dirr
                    
        dfs(0,seen)
        return reversals[0]

#we can also do the parent to child thing inset of keeping a set, so we dont immediatle go back
class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        '''
        parent to child paradigm
        '''
        adj_list = defaultdict(list)
        for u,v in connections:
            adj_list[u].append((v,1))
            adj_list[v].append((u,0))
        
        
        self.reversals = 0
        
        def dfs(node,parent):
            for neigh,dirr in adj_list[node]:
                if neigh != parent:
                    dfs(neigh,node)
                    self.reversals += dirr
                    
        dfs(0,-1)
        return self.reversals

class Solution:
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        '''
        we can also use bfs
        '''
        adj_list = defaultdict(list)
        for u,v in connections:
            adj_list[u].append((v,1))
            adj_list[v].append((u,0))
        
        
        q = deque([(0,-1)])
        reversals = 0
        
        while q:
            node,parent = q.popleft()
            for neigh,dirr in adj_list[node]:
                if neigh != parent:
                    reversals += dirr
                    q.append((neigh,node))
                    
        return reversals
