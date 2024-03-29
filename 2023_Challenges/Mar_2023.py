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

#################################################################
# 2316. Count Unreachable Pairs of Nodes in an Undirected Graph 
# 25MAR23
#################################################################
#TLE
class Solution:
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        '''
        for a connected component, all pairs are reach able from each other
        for the second example
        we have
            comp = size 4
            comp = size 2
            comp = size 1
            
            4*2 + 4*1 + 2*1
            8 + 4 + 2
            14
            
        if we did have the number of component sizes, we just spend 0((num compoenents)^2) time getting the product sums
        cartesian product between compoennets
        '''
        adj_list = defaultdict(list)
        for u,v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            
        
        def dfs(node,seen,curr_group):
            seen.add(node)
            curr_group.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen,curr_group)
                    
        
        seen = set()
        group_sizes = []
        for i in range(n):
            if i not in seen:
                curr_group = set()
                dfs(i,seen,curr_group)
                group_sizes.append(len(curr_group))
        
        if len(group_sizes) == 1:
            return 0
        
        ans = 0
        for i in range(len(group_sizes)):
            for j in range(i+1,len(group_sizes)):
                ans += group_sizes[i]*group_sizes[j]
        
        return ans

#this is a counting problem
class Solution:
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        '''
        for a connected component, all pairs are reach able from each other
        for the second example
        we have
            comp = size 4
            comp = size 2
            comp = size 1
            
            4*2 + 4*1 + 2*1
            8 + 4 + 2
            14
            
        if we did have the number of component sizes, we just spend 0((num compoenents)^2) time getting the product sums
        cartesian product between compoennets
        
        from the hint
        used only two hints
        for a node u, the number of nodes that are unreachable from u is the number of nodes not in the same compoenent
        this is a just counting problem
        '''
        adj_list = defaultdict(list)
        for u,v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            
        
        def dfs(node,seen,curr_group):
            seen.add(node)
            curr_group.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen,curr_group)
                    
        
        seen = set()
        ans = 0
        for i in range(n):
            if i not in seen:
                curr_group = set()
                dfs(i,seen,curr_group)
                group_size = len(curr_group)
                ans += (n - group_size)*group_size
        
        #we divided by 2 because we dobule count!
        return ans // 2
    
#bfs
class Solution:
    def countPairs(self, n: int, edges: List[List[int]]) -> int:
        '''
        bfs
        '''
        adj_list = defaultdict(list)
        for u,v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
            
        
        def bfs(node,seen,curr_group):
            seen.add(node)
            curr_group.add(node)
            q = deque([node])
            while q:
                node = q.popleft()
                for neigh in adj_list[node]:
                    if neigh not in seen:
                        q.append(neigh)
                        seen.add(neigh)
                        curr_group.add(neigh)
                    
        
        seen = set()
        ans = 0
        for i in range(n):
            if i not in seen:
                curr_group = set()
                bfs(i,seen,curr_group)
                group_size = len(curr_group)
                ans += (n - group_size)*group_size
        
        #we divided by 2 because we dobule count!
        return ans // 2
    
#for union find see C++ file

############################################
# 2360. Longest Cycle in a Graph
# 26APR23
############################################
#fuck me....
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        '''
        graph is directed, in the form of parent points
        edges[i] means i points to edges[i]
        n = len(edges) - 1
        
        return length of longest cycle
        
        if edges[i] == -1:
            there is no outgoing edge
        well first check if there is a cycle, if there isn't, return -1
        
        start from each node and find the cycle
        '''
        adj_list = defaultdict(list)
        n = len(edges)
        for i in range(len(edges)):
            if edges[i] != -1:
                adj_list[i].append(edges[i])
        
        def dfs(node,curr_path,dist):
            if node in curr_path:
                return 
            curr_path.add(node)
            for neigh in adj_list[node]:
                if neigh not in curr_path:
                    dfs(neigh,curr_path,dist+1)
            
            
        
        global_seen = set()
        for i in range(n):
            if i not in global_seen:
                curr_path = set()
                size = dfs(i,curr_path,0)
                print(curr_path)
                global_seen |= curr_path


#dp using dist matrix
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        '''
        intuition:
            if a node is part of a cycle, it cannot be part of another cycle
            for a node in a cycle, itt would impyl that the only outgoingg edge of node would also be in this cycle
            in a graph with only one outgoing edge, a node cannot be a part of more than one cycle
        
        if we traversed from any node in a cycle, we would eventuallt touch all nodes
        also there is no point in visint nodes that are not part of a cycle, iterate only on an outgoign edge
        we need to store distances in hashmap
            and if we were to visit this node again, we can get the distance
                dist = dist[curr] - dist[visited again], (where visited again is a node we have already seen)
                distances actually just store the number of edges, which is just the dist
                i.e if there is  a cycle, its dist would be:
                    cycle_dist = dist[curr] - dist[visited again] + 1
                    we do + 1 to close the cycle
                    
        in dfs traversal, we check if the neigh node using edges[node]
        if neigh is not visted, update
            dist[neigh] = dist[node] + 1
            recurse
        
        if neigh is already visited, 
            1.it is part of cycle on the current dfs we are on
            2. it was touched by a previosu dfs call
                we can verify if the dist map has the neighbord in it
                because we create a enw map for every traversal
        
        if dist contians neigh:
            it mena we visited neigh during the current dfs travertsla
            update maximum cycle longthe globallay
            ans = max(ans, dist[node] - dist[neigh] + 1)
        '''
        self.longest = -1
        
        def dfs(node,dist,visted):
            visted[node] = True
            neigh = edges[node]
            
            if neigh != -1 and not visted[neigh]:
                dist[neigh] = dist[node] + 1
                dfs(neigh,dist,visted)
            #part of the cucle
            elif (neigh != -1 and neigh in dist):
                #node would have already been populate with an even greate dist
                curr_longest_cycle = dist[node] - dist[neigh] + 1
                self.longest = max(self.longest,curr_longest_cycle)
                
        
        N = len(edges)
        visited = [False]*N
        
        for i in range(N):
            if not visited[i]:
                dist = defaultdict()
                dist[i] = 1
                dfs(i,dist,visited)
        
        return self.longest
    
#no dfs
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        '''
        we actually do not need to use dfs, becaause there is no more than one outgoing edge
        for each node, just follow its path, until e haven't seen its neighbor node
        if we have seen this node, it must have been part of a cycle
        
        just walking a linked list
        '''
        N = len(edges)
        #this stores dists node: (neigh,curr_dist), because we have at most 1 out going edge
        dist = [[0,0] for _ in range(N)] #take and out going edge from i?
        ans = -1
        
        for i in range(N):
            curr_path = 1
            curr_node = i
            
            while curr_node != -1 and dist[curr_node][1] == 0:
                dist[curr_node] = [i,curr_path]
                curr_path += 1
                curr_node = edges[curr_node]
            
            #make sure we have an outgoing edge, and that it is part of the curent cycle
            if curr_node != -1 and dist[curr_node][0] == i:
                ans = max(ans, curr_path - dist[curr_node][1])
        
        return ans
    
#another way, depth of iteration as time, which is just the number of edges, which is just the path length
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        '''
        idea,
            use time a the current distance (or the current depth of the iteration we are on)
            initally all distances for the nodes are set to 0
            for each node in the graph, check it has been visited before, i.e the time is > 0
            set start time to curr time
            while we have an out going edge, move and update the distnaces by 1
            otherwise maximize
        '''
        ans = -1
        curr_time = 1
        N = len(edges)
        
        visited_times = [0]*N
        
        for i in range(N):
            #skip nodes that have already been visited
            if visited_times[i] > 0:
                continue
            
            start_time = curr_time
            #need to hold start time to find max distance
            curr_node = i
            
            #while we have an out going edge, and it remains unvisited
            while curr_node != -1 and visited_times[curr_node] == 0:
                visited_times[curr_node] = curr_time
                curr_time += 1
                curr_node = edges[curr_node]
            
            #check if cycle has been found and if its longer than the current max cycle
            #i.e if this current time is larger than what's already there, it must have already been visited, and can be a potential answer for 
            #the largest cycle
            if curr_node != -1 and visited_times[curr_node] >= start_time:
                cycle_length = curr_time - visited_times[curr_node]
                ans = max(ans, cycle_length)
        
        
        return ans

#kahns algo starting at leaves, then just remove them
class Solution:
    def longestCycle(self, edges: List[int]) -> int:
        '''
        we can also use kahns algorithm
        since there is no more than one outgoing edge for each node, we will have nodes that have 0 in degree
        start with the nodes that have 0 indegree, visit them, then drop edges
        then visit the next nodes
        
        if we do this nodes, in a cycle will be unvisited
        for all the unvisited nodes, dfs and get the maximum componenet size
        '''
        N = len(edges)
        seen = [False]*N
        in_degree = defaultdict()
        
        for i in range(N):
            if edges[i] != -1:
                count = in_degree.get(edges[i],0)
                in_degree[edges[i]] = count + 1
        
        
        #load up those with 0 in degree
        q = deque([])
        visited = [False]*N
        for i in range(N):
            if i not in in_degree:
                q.append(i)
        
        #begin kahns
        while q:
            curr = q.popleft()
            visited[curr] = True
            
            neigh = edges[curr]
            if neigh != -1:
                in_degree[neigh] -= 1
                if in_degree[neigh] == 0:
                    q.append(neigh)
        
        
        #follow parent points for all unvisited nodes
        ans = -1
        for i in range(N):
            if not visited[i]:
                curr = i
                path_length = 1
                visited[i] = True
                #while we don't come back to the original node from where we started
                while edges[curr] != i:
                    path_length += 1
                    visited[edges[curr]] = True
                    curr = edges[curr]
                ans = max(ans,path_length)
                
        return ans
    
######################################
# 64. Minimum Path Sum (REVISTED)
# 27MAR23
######################################
#top down
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        '''
        notes to me when solving:
            when normally thinking about entire path, think about storing smallest path so far 
            so think dp
        
        
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        memo = {}
        
        def dp(i,j):
            if (i,j) == (0,0):
                return grid[0][0]
            
            if i < 0 or j < 0:
                return float('inf')
            
            if (i,j) in memo:
                return memo[(i,j)]
            
            up = grid[i][j] + dp(i-1,j)
            left = grid[i][j] + dp(i,j-1)
            ans = min(up,left)
            
            memo[(i,j)] = ans
            return ans
            
        
        return dp(rows-1,cols-1)
    
#bottom up
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        '''
        bottom up, we start from 0,0
        '''
        n = len(grid)
        m = len(grid[0])
        dp=[[0]*m for i in range(n)]
    
        for i in range(n):
            for j in range(m):
                if i==0 and j==0:
                    dp[i][j]= grid[i][j]

                else:
                    up=grid[i][j]
                    left=grid[i][j]

                    if i>0:
                        up+= dp[i-1][j]
                    else:
                        up+= float('inf')
                    if j>0:
                        left+=dp[i][j-1]
                    else:
                        left+=float('inf')

                    dp[i][j]= min(up,left)

        return dp[n-1][m-1]


#starting from (0,0)
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        '''
        i had it
        dp(i,j) = grid[i][j] + min(dp(i+1,j), dp(i,j+1))
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        memo = {}
        
        def dp(i,j):
            if i == rows or j == cols:
                return float('inf')
            if (i,j) == (rows-1,cols-1):
                return grid[i][j]
            
            if (i,j) in memo:
                return memo[(i,j)]
            
            ans = grid[i][j] + min(dp(i+1,j), dp(i,j+1))
            memo[(i,j)] = ans
            return ans
        
        return dp(0,0)
    
#bottom up another way
#for the boundary conditions, just check if we are in the last column or row
#then just limit the states we can take from
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        '''
        bottom up
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dp = [[0]*(cols) for _ in range(rows)]
        
        for i in range(rows-1, -1,-1):
            for j in range(cols-1,-1,-1):
                #if on the last row
                if (i == rows-1) and (j != cols-1):
                    dp[i][j] = grid[i][j] + dp[i][j+1]
                #last col
                elif (j == cols - 1) and (i != rows -1):
                    dp[i][j] = grid[i][j] + dp[i+1][j]
                #transition
                elif (i,j) != (rows-1,cols-1):
                    dp[i][j] = grid[i][j] + min(dp[i+1][j], dp[i][j+1])
                #increment
                else:
                    dp[i][j] = grid[i][j]
        
        return dp[0][0]
    
#keping only one row
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        '''
        bottom up 1 row
        i and i+1 are the same row
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dp = [0]*(cols)
        
        for i in range(rows-1, -1,-1):
            for j in range(cols-1,-1,-1):
                #if on the last row
                if (i == rows-1) and (j != cols-1):
                    dp[j] = grid[i][j] + dp[j+1]
                #last col
                elif (j == cols - 1) and (i != rows -1):
                    dp[j] = grid[i][j] + dp[j]
                #transition
                elif (i,j) != (rows-1,cols-1):
                    dp[j] = grid[i][j] + min(dp[j], dp[j+1])
                #increment
                else:
                    dp[j] = grid[i][j]
        
        return dp[0]
    
#constant space, in place using grid
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        '''
        bottom up 1 row
        i and i+1 are the same row
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        for i in range(rows-1, -1,-1):
            for j in range(cols-1,-1,-1):
                #if on the last row
                if (i == rows-1) and (j != cols-1):
                    grid[i][j] = grid[i][j] + grid[i][j+1]
                #last col
                elif (j == cols - 1) and (i != rows -1):
                    grid[i][j] = grid[i][j] + grid[i+1][j]
                #transition
                elif (i,j) != (rows-1,cols-1):
                    grid[i][j] = grid[i][j] + min(grid[i+1][j], grid[i][j+1])
                #increment
                else:
                    grid[i][j] = grid[i][j]
        
        return grid[0][0]

#############################################
# 2101. Detonate the Maximum Bombs
# 28MAR23
#############################################
#TLE, but works, 134/140
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        '''
        we are given a list of bombs, (x_pos, y_pos, radiuss)
        when we detonate a bomb, all bombs in its blast radius are detonated
        return max number of bombs that can be deontated if we fire one bomb
        
        try detonating each of bomb, one by one, then see how many bombs get detonated
        for each bomb detonated, the range could be extended
        input is smalle enough for N^2 solution
        i was thinking dfs
        
        for each bomb, see if we can reach it with dfs
        we'll need to dfs for each bomb
        '''
        ans = 1
        
        def dfs(idx,seen,count):
            #this gets the count
            x,y,r = bombs[idx]
            seen.add(idx)
            count[0] += 1
            for j in range(len(bombs)):
                neigh_x,neigh_y,neigh_r = bombs[j]
                dist_away = (x - neigh_x)**2 + (y - neigh_y)**2
                if j not in seen:
                    #can detonate
                    if dist_away <= r*r:
                        dfs(j,seen,count)
                        
                        
        for i in range(len(bombs)):
            seen = set()
            count = [0]
            dfs(i,seen,count)
            ans = max(ans,count[0])
        
        return ans
            
#turn it into a graph first, then use dfs
#preprocess
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        '''
        we are given a list of bombs, (x_pos, y_pos, radiuss)
        when we detonate a bomb, all bombs in its blast radius are detonated
        return max number of bombs that can be deontated if we fire one bomb
        
        try detonating each of bomb, one by one, then see how many bombs get detonated
        for each bomb detonated, the range could be extended
        input is smalle enough for N^2 solution
        i was thinking dfs
        
        for each bomb, see if we can reach it with dfs
        we'll need to dfs for each bomb
        '''
        ans = 1
        adj_list = defaultdict(list)
        N = len(bombs)
        for i in range(N):
            for j in range(N):
                if i != j:
                    x,y,r = bombs[i]
                    neigh_x,neigh_y,neigh_r = bombs[j]
                    dist_away = (x - neigh_x)**2 + (y - neigh_y)**2
                    if dist_away <= r*r:
                        adj_list[i].append(j)
        
        
        
        def dfs(node,seen):
            #this gets the count
            seen.add(node)
            for child in adj_list[node]:
                if child not in seen:
                    dfs(child,seen)
                        
                        
        for i in range(len(bombs)):
            seen = set()
            dfs(i,seen)
            ans = max(ans,len(seen))
        
        return ans
    
######################################
# 983. Minimum Cost For Tickets
# 28MAR23
######################################
#yessss, needed to use days
#day variant
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        '''
        we are traveling on days, each days[i] is  aday we travel
        train tickets cost:
            1-day = costs[0]
            7-day = costs[1]
            30-day = costs[2]
            
        passwed all for that many consective days of travle
        return minimum number of dollars you need to travel every day in days
        dp obviosuly
        
        prefix dp
        let dp(d) be the minimut cost traveling days[:i]
        if we have gotten to the end, we don't need to buy ticket
        if i'm on days[i], call it d could have gotten here from d-1, or d-7, or d-30
        
        
        
        dp(d) = {
            #only on the days we travel
            min(dp(d+1) + cost to travel 1, dp(d+7) + cost for seven days, dp(d+30) costs for traveling 30)
            
            #otherwise we just go on th next day, keeping on to thatt minimum
        }
        
        
        '''
        memo = {}
        days_travled = set(days)
        
        def dp(d):
            if d > 365:
                #doens't cost me anything
                return 0
            if d in memo:
                return memo[d]
            #if im' traveling on this day
            if d in days_travled:
                first = dp(d+1) + costs[0]
                second = dp(d+7) + costs[1]
                third = dp(d+30) + costs[2]
                ans = min(first,second,third)
                memo[d] = ans
                return ans
            #othewise just wait until we can by
            else:
                ans = dp(d+1)
                memo[d] = ans
                return ans
        
        return dp(1)

#topdown from the end
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        '''
        instead of starting from day 1, start from the last day
        only call on days i have traveled
        
        if dp(d) is the min cost for traveling up to day d
        then check d-1, d-7, d-30, and to each of these add the cost and minimiz
        '''
        memo = {}
        days_traveled = set(days)
        
        def dp(d):
            if d < 1:
                return 0
            if d in memo:
                return memo[d]
            #if this is a day i'm traveling on
            if d in days_traveled:
                first = dp(d-1) + costs[0]
                second = dp(d-7) + costs[1]
                third = dp(d-30) + costs[2]
                ans = min(first,second,third)
                memo[d] = ans
                return ans
            else:
                ans = dp(d-1)
                memo[d] = ans
                return ans
            
        
        return dp(days[-1])
    
#bottom up
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        '''
        bottom up
        '''
        days_traveled = set(days)
        dp = [0]*(365+1+30) #pad to the largest, many ways to check boundary conditions
        
        for d in range(365,0,-1):
            if d in days_traveled:
                first = dp[d+1] + costs[0]
                second = dp[d+7] + costs[1]
                third = dp[d+30] + costs[2]
                ans = min(first,second,third)
                dp[d] = ans
            else:
                ans = dp[d+1]
                dp[d] = ans
        
        return dp[1]
    
#recursive, index variant
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        '''
        instead of using the days, we can also use the index (i) into the array
        the only caveat is that we have to keep moving up a day when chevking the current number of days per pass
        '''
        memo = {}
        durations = [1,7,30]
        
        def dp(i):
            #need to allow to fall out of index
            if i < 0:
                return 0
            
            if i in memo:
                return memo[i]
            
            ans = float('inf')
            j = i
            #want to minize for all options
            #want largest index that its not more than the current days[i] + duration
            for c,d in zip(costs,durations):
                while j >= 0 and days[j] > days[i] - d: #as long as we are more than durations days away than the current days[i]
                    #falling out of index calls dp(out of index), which hits the cases
                    j -= 1
                ans = min(ans,dp(j)+c)
            
            memo[i] = ans
            return ans
        
        return dp(len(days)-1)

#######################################
# 836. Rectangle Overlap
# 28MAR23
########################################
#close one
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        '''
        if there is an overlap
            left bound of rec2 must be in between the bounds of rec1
            upper boound must be in the bounds of rec1
        '''
        a,b,c,d = rec1
        w,x,y,z = rec2
        #there are going to be eight conditions to check for
        if a < w < c:
            return True
        if w < a < y:
            return True
        if x < b < z:
            return True
        if b < x < d:
            return True
        if  w < c < y:
            return True
        if a < y < c:
            return True
        if x < d < z:
            return True
        if b < z < d:
            return True
        
        return False

#close again
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        '''
        if there is an overlap
            left bound of rec2 must be in between the bounds of rec1
            upper boound must be in the bounds of rec1
        '''
        a,b,c,d = rec1
        w,x,y,z = rec2
        #there are going to be eight conditions to check for
        if a < w < c and  b < x < d:
            return True
        if w < a < y and x < b < z:
            return True
        if  w < c < y and  x < d < z:
            return True
        if a < y < c and b < z < d:
            return True
        
        return False
    
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        '''
        check if coordinates form valid rectablge
    /**
        x1y2_____x2y2
        |         | 
        |    X1Y2_|_______X2Y2
        |     |   |         |
        x1y1__|__x2y1       |
              |             |
             X1Y1_________X2Y1
    **/
        '''
        A, B, C, D = rec1[0], rec1[1], rec1[2], rec1[3]
        E, F, G, H = rec2[0], rec2[1], rec2[2], rec2[3]
        x1 = max(A, E)
        y1 = max(B, F)
        x2 = min(C, G)
        y2 = min(D, H)
        if x1 < x2 and y1 < y2:
            return True
        return False
    
class Solution:
    def isRectangleOverlap(self, rec1: List[int], rec2: List[int]) -> bool:
        '''
        using de mograns law, ask the converse question
            are rec1 and rec2 not intersection
            checking for not intersection is easier than checking for an intersection
            the answer is just not (not intserction)
            negation
            
        first check if rectangles make a line then check left, bottom, right, top
        '''
        if (rec1[0] == rec1[2]) or (rec1[1] == rec1[3]) or \
            (rec2[0] == rec2[2]) or (rec2[1] == rec2[3]):
            return False
        
        #comparing rec1 to rec2
        left = rec1[2] <= rec2[0]
        bottom = rec1[3] <= rec2[1]
        right = rec1[0] >= rec2[2]
        up = rec1[1] >= rec2[3]
        
        
        return not (left or bottom or right or up)
    
#we can also just check area
class Solution(object):
    '''
    if there is overlap, there must be postive area
    so just check the projections of intersection along x and y

    Say the area of the intersection is width * height, where width is the intersection of the rectangles projected onto the x-axis, 
    and height is the same for the y-axis. We want both quantities to be positive.

The width is positive when min(rec1[2], rec2[2]) > max(rec1[0], rec2[0]), that is when the smaller of (the largest x-coordinates) 
is larger than the larger of (the smallest x-coordinates). The height is similar.
    '''
    def isRectangleOverlap(self, rec1, rec2):
        def intersect(p_left, p_right, q_left, q_right):
            return min(p_right, q_right) > max(p_left, q_left)
        return (intersect(rec1[0], rec1[2], rec2[0], rec2[2]) and # width > 0
                intersect(rec1[1], rec1[3], rec2[1], rec2[3]))    # height > 0
    
####################################
# 1402. Reducing Dishes
# 29MAR23
####################################
#TLE, subset enumeration, maximize, but floor to 0
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        '''
        we want the maximum like-time coefficent
        which is the largest sum of
            satisfaction[i]*times[i]
            
        we can remove any dishes from satisfaction
        return maximum like-time coeffificent
        
        one way would be to examin all possible subsets of satifcations
        then just take the dot product with their indices
        
        brute force would be to generate all subsets, and find the largest like-time coeffcients
        '''
        N = len(satisfaction)
        self.ans = float('-inf')
        #can sort before hand
        satisfaction.sort()
        
        def rec(i,path):
            if i >= N:
                temp = 0
                for i in range(len(path)):
                    temp += (i+1)*path[i]
                    self.ans = max(self.ans,temp)
                return
            rec(i+1,path+[satisfaction[i]])
            rec(i+1,path)
        
        
        rec(0,[])
        return max(self.ans,0)
    

#damn it, i dont what states to cache......
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        '''
        we want the maximum like-time coefficent
        which is the largest sum of
            satisfaction[i]*times[i]
            
        we can remove any dishes from satisfaction
        return maximum like-time coeffificent
        
        one way would be to examin all possible subsets of satifcations
        then just take the dot product with their indices
        
        brute force would be to generate all subsets, and find the largest like-time coeffcients
        
        do i need to save, paths? can i just save sums
        '''
        N = len(satisfaction)
        memo = {}
        #can sort before hand
        satisfaction.sort()
        
        def rec(i,pos,curr_sum):
            if i >= N:
                return curr_sum
            if (i,pos,curr_sum) in memo:
                return memo[(i,pos,curr_sum)]
                
            take = rec(i+1,pos+1,curr_sum+(satisfaction[i]*(pos+1)))
            no_take = rec(i+1,pos,curr_sum)
            ans = max(take,no_take)
            memo[(i,pos,curr_sum)] = ans
            return ans
        
        
        return max(rec(0,0,0),0)
    
#states are position in index and current time
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        '''
        hints say keep previous bust time coef and corresponding element sum
        if adding curr element to previous best time and its correspond best, then go ahead and add it
        otherwise keep previous
        '''
        N = len(satisfaction)
        memo = {}
        #can sort before hand
        satisfaction.sort()
        
        def dp(i,time):
            if i >= N:
                return 0
            if (i,time) in memo:
                return memo[(i,time)]
            
            #cook dish at this time and move on to the next index at i + 1
            take = satisfaction[i]*time + dp(i+1,time+1)
            no_take = dp(i+1,time)
            ans = max(take,no_take)
            memo[(i,time)] = ans
            return ans
        
        
        return dp(0,1)
    
#bottom up
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        '''
        bottom up
        '''
        N = len(satisfaction)
        dp = [[0]*(N+2) for _ in range(N+2)]
        satisfaction.sort()
        
        #starting at N
        for i in range(N-1,-1,-1):
            #but for time, we need to see all prevtimes up to N
            #so we go forward in this direction
            for time in range(1,N+1,1):
                take = satisfaction[i]*time + dp[i+1][time+1]
                no_take = dp[i+1][time]
                ans = max(take,no_take)
                dp[i][time] = ans
        
        
        return dp[0][1]
    
#bottom up space optimized, we only need to save the prev row
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        '''
        bottom up, but really we only just need two rows (along with the requisite number of columns)
        if you don't like this, then just keep two rows instead, but you'd have to update the two rows again
        think about it this way
        dp matrix
        [
        [..] <- this is i 
        [..] <- this is i+1
        ]
        '''
        N = len(satisfaction)
        prev = [0]*(N+2)
        satisfaction.sort()
        
        #starting at N
        for i in range(N-1,-1,-1):
            dp = [0]*(N+2)
            #but for time, we need to see all prevtimes up to N
            #so we go forward in this direction
            for time in range(1,N+1,1):
                take = satisfaction[i]*time + prev[time+1]
                no_take = prev[time]
                ans = max(take,no_take)
                dp[time] = ans
            prev = dp
        
        
        return prev[1]
    
#greedy, stepping stone into intuition
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        '''
        if we sorted the array, we essential could just evalute all suffix sums and take the max in O(N**2)
        '''
        N = len(satisfaction)
        ans = 0
        satisfaction.sort()
        for i in range(N):
            curr_sum = 0
            for j in range(i,N):
                curr_sum += satisfaction[j]*(j-i+1)
            
            ans = max(ans,curr_sum)
        
        return max(ans,0)

#true greedy
class Solution:
    def maxSatisfaction(self, satisfaction: List[int]) -> int:
        '''
        after sorting, we can start from the end of the array, but accumulate like time coefficinets starting with time = 1 at the end
        consider the total sum for the last element, it would be its satisfaction value
        then for the next index (i-1), the value goes up by satisfaction[i-1] + 2*satisfaction[i]
        since we are starting at the end of the array but with time 1, it doesn't releft that would should be at time N
        
        example:
            dishes = [a,b,c,d]
            like-time-coeffcients = [a+2b+3c+2d,b+2c+3d,c+2d,d,0]
            the first order differences between these terms is 
            [a+b+c+d,b+c+d,c+d,d]
            
            which is just the individual suffix sums
        
        intution:
            accumualte the suffix sums
            As an optimization, we can stop iterating early because we have sorted the array in ascending order; 
            hence the moment the suffix array sum becomes less than zero, we can break and return the current sum, 
            as adding it would only decrease the sum and the suffix array sum will always be negative after that 
            because the values would keep decreasing.
            
        
        '''
        N = len(satisfaction)
        satisfaction.sort()
        ans = 0
        
        curr_suff_sum = 0
        for i in range(N-1,-1,-1):
            curr_suff_sum += satisfaction[i]
            if curr_suff_sum < 0:
                return ans
            ans += curr_suff_sum
        
        return ans
    
###################################
# 651. 4 Keys Keyboard
# 29MAR23
####################################
#close one, transition states are hard with this one.....
class Solution:
    def maxA(self, n: int) -> int:
        '''
        this is a graph problem
        we are given n key presses, and given n key presses, return the maximum number of A's on the screen
        
        let dp(i) be the maximum number of A's i can get with i moves
        then there are two ways to get another a
            it goes up by 1, consuming a step
            or it goes up by i, consuming 3 steps
        '''
        memo = {}
        
        def dp(i,count):
            if i <= 0:
                return count
            if (i,count) in memo:
                return memo[(i,count)]
            print_one = dp(i-1,count+1)
            copy_paste = dp(i-3,count*2) 
            ans = max(print_one,copy_paste)
            memo[i] = ans
            return ans
        
        return dp(n,0)

#three states, total, and whats in buffer, the question is poorly worded...
class Solution:
    def maxA(self, n: int) -> int:
        '''
        this is a graph problem
        we are given n key presses, and given n key presses, return the maximum number of A's on the screen
        
        let dp(i) be the maximum number of A's i can get with i moves
        then there are two ways to get another a
            it goes up by 1, consuming a step
            or it goes up by i, consuming 3 steps
        '''
        memo = {}
        
        def dp(i,total,buffer):
            if i == 1:
                return total + 1
            if i == 2:
                return total + max(2,buffer*2)
            
            if (i,total,buffer) in memo:
                return memo[(i,total,buffer)]
            
            #these states represent whether we "can" go down this path
            #then we take the maximum
            pressA = 0
            #option when to press A, doesn't make sense to press A if buffer has more than 1, if it did, we would press CtrlV
            if buffer < 2:
                pressA = dp(i-1,total+1,buffer)
            #can press, if total is less than 2
            press_CtrlA_CtrlC = 0
            if total > 2:
                press_CtrlA_CtrlC = dp(i-2,total,total) #this overwrites buffer
            press_CtrlV = 0
            if buffer != 0:
                press_CtrlV = dp(i-1,total+buffer,buffer) #buffer stays no matter what
                
            ans = max(pressA,press_CtrlA_CtrlC,press_CtrlV)
            memo[(i,total,buffer)] = ans
            return ans
        
        return dp(n,0,0)
    
class Solution:
    def maxA(self, n: int) -> int:
        '''
        so we have used m presses and have current string length l
            from here, we can do a copy paste operations in three key presses: which would give us 2*l with m+3
            we can do another copy paster opertion in 4 steps: which would give 3*l with m+4 presses
            we can keep pasting: general from is k*l with m+k+1 presses, when k >= 2
            
        but there is no no need to press CtrlV more than 4 times in a row
        starting with l, we can do CtrlA, CtrlC, CtrlV 5 times, to give us 6l with 5 presses
        but if we did  Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+V, the length also becomes 6l, but the buffer now contains 2l
        which is worse than the former case, both scenarios used 7 presses
        
        so it is ineffcient to press ctrlV more than 5 times or more in a row
        because its not as effecient as doing  Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+A, Ctrl+C, Ctrl+V, Ctrl+V, which would prep our buffer with 2l
        
        dp[i+3] = 2*dp[i] #these are all Ctrl-V's after the first CtrlA, CtrlC, we can can Ctrl V up to 4 rimes
        dp[i+4] = 3*dp[i]
        dp[i+5] = 4*dp[i]
        dp[i+6] = 5*dp[i]
        
        in general dp[j] = (j-i-1)*dp[i], where i + 3 <= j < i + 6
        '''
        dp = list(range(n + 1))
        for i in range(n - 2):
            #from i, we can only check to the left not more than 3 away, or more than 5 away, but floored by the number of presses
            for j in range(i + 3, min(n, i + 6) + 1):
                dp[j] = max(dp[j], (j - i - 1) * dp[i])
        return dp[n]
    
class Solution:
    def maxA(self, n: int) -> int:
        dp = [0]*(n+1)
        
        for i in range(1,n+1):
            if i <= 5:
                #the base cases, we cannot press ctrl v more than 5 times
                dp[i] = i
            else:
                # There are following cases:
                # Type A, which is just dp[i-1]+1
                # Select, copy and paste previous A list generated by three steps before, which is dp[i-3]*2
                # Select, copy, paste, paste. So it is actually pasting dp[i-4] which is the list four steps before and we times three here
                # Select, copy, paste, paste, paste. So it is actually pasting dp[i-5] which is the list five steps before and we times four here
                # We will not have the case of pasting more than 3 times because we can select, copy and paste again to paste with longer A list.
                dp[i] = max(dp[i-1] + 1, dp[i-3]*2,dp[i-4]*3,dp[i-5]*4)
                
        
        return dp[n]
    
class Solution:
    def maxA(self, n: int) -> int:
        '''
        top down
        '''
        memo = {}
        
        def dp(i):
            if i <= 5:
                return i
            if i in memo:
                return memo[i]
            
            ans = max(dp(i-1)+1, dp(i-3)*2,dp(i-4)*3,dp(i-5)*4)
            memo[i] = ans
            return ans
        
        return dp(n)
    

####################################
# 87. Scramble String 
# 30MAR23
####################################
#OMG, this actually worked to get all scambles
class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        '''
        generate all scrambles and check if s1 == s2
        '''
        
        def scramble(string):
            N = len(string)
            if N == 1:
                return [string]
            #random split, well reall just try all splits
            scrambles = []
            for i in range(1,len(string)):
                left = scramble(string[:i])
                right = scramble(string[i:])
                for l in left:
                    for r in right:
                        scrambles.append(l+r)
                        scrambles.append(r+l)
            
            return scrambles
                
        
        possible = set(scramble(s1))
        return s2 in possible

#fuck....
#https://leetcode.com/problems/scramble-string/discuss/3357546/Python3-oror-35ms-oror-Beats-99.38-(recursion-with-memoization)
class Solution:
    def isScramble(self,s1, s2):
        m ={}
        #cache string states
        def func(s1, s2):
            if (s1, s2) in m:
                return m[(s1, s2)]
            #cannot possible be srambles of one another
            if not sorted(s1) == sorted(s2):
                return False
            #base case
            if len(s1) == 1:
                return True
            
            #try all scrambles and check
            for i in range(1, len(s1)):
                #scramble cases are:
                #1. (left part of s1 and right part of s2) and (right part of s1 and left part of 2)
                #2. (left part of s1 and left part of s2) and (right part of s1 and right part of s2)

                if func(s1[:i], s2[-i:]) and func(s1[i:], s2[:-i]) or func(s1[:i], s2[:i]) and func(s1[i:], s2[i:]):
                    m[(s1, s2)] = True
                    return True
            m[(s1, s2)] = False
            return False
        return func(s1, s2)
    

class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        '''
        intution:
            s can be divided into x and y
            we can rewrite s as x+ y, y+x, or the scrambles of x and y
            x' + y' or y' + x'
        if we are given strings s and t, how can we check of t is a scramble of s
        s = x + y and t = x' + y'
        
        dp(s1,s2) returns whether or not s1 is a scramble of s2
        partition s1 into all possible paritionts, we can get left and right parts
        '''
        memo = {}
        
        def dp(s1,s2):
            #cache string states
            if (s1,s2) in memo:
                return memo[(s1,s2)]
            #strings are equal
            if s1 == s2:
                memo[(s1,s2)] = True
                return True
            
            count1 = Counter(s1)
            count2 = Counter(s2)
            
            if count1 != count2:
                memo[(s1,s2)] = False
                return False
            
            n = len(s1)
            for i in range(1,n):
                left = s1[:i]
                right = s1[i:]
                #not swapping
                #we are checking if the scramble of hte left parts of both s1 and s2 and the right parts of s1 and s2 are scrambles of other
                case1 = dp(left, s2[:i]) and (right, s2[i:])
                #if we did swap
                #we swapped the right parts of s1 and s2, and the left parts of s1 and s2, either of these are true results in a scramble
                case2 = dp(right,s2[:n-i]) and (left, s2[n-i:])
                if case1 or case2:
                    memo[(s1,s2)] = True
                    return True
            
            memo[(s1,s2)] = False
            return False
        
        
        return dp(s1,s2)

class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        '''
        s1,s2 is a scramble when
            s1[0:k],s2[0:k] is scramble and s1[k:n] and s2[k:n] is a scramble
        or
        s1[0:k], s2[n-k:n] and s1[k:n],s2[0:n-k] is a scramble
        
        base case, is when s1[i] == s2[k] and length == 1
        
        states are (i,j,length)
        
        1. the first on will be a substring of 1 starting at index i with length == lenght, call this s
        2. the second one will be a substring of s1, starting at idnex j, with length == length, call this t
        
        dp(i,j,length) will be treu of t is a scramble of s
        
        base case is when length == 1 and char's at both s1 and s2 are the same
        
        at each state, we perform a split on s1, and try all possible splits
        '''
        memo = {}
        def dp(i,j,length):
            #base case
            if length == 1:
                ans = s1[i] == s2[j]
                memo[(i,j)] = ans
                return ans
                
            if (i,j,length) in memo:
                return memo[(i,j,length)]
            
            #try all k splits for the current length
            for k in range(1,length):
                # gr|eat and rg|tea
                if dp(i,j,k) == True and dp(i+k,j+k,length-k) == True:
                    memo[(i,j,length)] = True
                    return True
                # gr|eat and tea|gr
                if dp(i,j+length-k,k) == True and dp(i+k,j,length-k) == True:
                    memo[(i,j,length)] = True
                    return True
            
            memo[(i,j,length)] = False
            return False
        
        
        return dp(0,0,len(s1))
        
#one more way
class Solution:
    def isScramble(self, s: str, t: str) -> bool:
        @cache
        def dfs(s1, s2):
            if s1 == s2:
                return True
            if Counter(s1) != Counter(s2):
                return False
            
            N = len(s1)
            for k in range(1, N):
                # gr|eat and rg|tea
                if (dfs(s1[:k], s2[:k]) and dfs(s1[k:], s2[k:]) or
                    # gr|eat and tea|gr, this is very hard to come up with
                    dfs(s1[:k], s2[N - k:]) and dfs(s1[k:], s2[:N - k])):
                    return True
            return False       
        return dfs(s,t)

##########################################
# 1444. Number of Ways of Cutting a Pizza
# 31MAR23
###########################################
#fuck it...
class Solution:
    def ways(self, pizza: List[str], k: int) -> int:
        '''
        i have to cut the pizza into k pieces using k-1 cuts
        ok lets try using the hints
        1. after each cut, the remaining pizza always has the lower right coord (rows -1, cols - 1)
        2. dp states are : (row1,col1, c) which gives number of ways to cut pizza using c cuts
            where the current piece of pizza has  upper left cord at (row1,col1) and lower right (rows-1,cols-1)
        3. for all transitions, try all verti and horiz cuts such that the piece you give to a person has at least 1 apple
        4. base case is whne c == k - 1
        
        let dp(row,col,cuts) = {
        
            sum of all ways
        }
        '''
        mod = 10**9 + 7
        rows = len(pizza)
        cols = len(pizza[0])
        
        memo = {}
        
        def dp(row,col,cuts):
            if row > rows:
                return 0
            if col > cols:
                return 0
            if cuts == k - 1:
                return 0
            ans = 0
            #try all vertical and horizontal cuts
            for r in range(row,rows)
            

class Solution:
    def ways(self, pizza: List[str], k: int) -> int:
        '''
        need to use suffix sums, recall the dp solution from 304. Range Sum Query 2D - Immutable 
        here's the solution for reference:
class NumMatrix:
    
    #we can dp to solve this problem
    #dp(i,j) represents the sum of the elements in the rectangle bounded by (0,0) and (i-1,j-1)
    #if we are asked to find the sum of the elemnets in a sqaure marked by points ABCD
    #SUM(ABCD) = SUM(OD) - SUM(OB) - SUM(OC) + SUM(OA)
    
    #to calculate a subpriblem dp(i,j) = dp(i-1,j) + dp(i,j-1) + matrix(i,j) - dp(i,j)
    #Try drawing the rectangle. you will see that the dp[r][c] is added twice when adding dp[r+1][c] and dp[r][c+1].

    def __init__(self, matrix: List[List[int]]):
        rows = len(matrix)
        cols = len(matrix[0])
        
        if rows == 0 or cols == 0:
            return
        
        self.dp = [[0]*(cols+1) for _ in range(rows+1)]
        
        for i in range(rows):
            for j in range(cols):
                self.dp[i+1][j+1] = self.dp[i][j+1] + self.dp[i+1][j] + matrix[i][j] - self.dp[i][j]

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        OD = self.dp[row2+1][col2+1]
        OB = self.dp[row1][col2+1]
        OC = self.dp[row2+1][col1]
        OA = self.dp[row1][col1]
        return OD - OB - OC + OA

    we need this to quickly query the number of apples in a cut!
    otherwise we'd have to dp, count, and increment on each cut
    
    intution:
        we need to know cuts_left
        we need to know what part of the pizza remains after cutting, since we always give away the top and the left if it contains an apple
        let left row denote the topmost row, and col denote the leftmost col
            the remaining pizzal is then pizza[row: rows+1][cols: cols-1]
            where rows and cls denote the number of rows and columns in the original piza
            
    state:
        (row,col,cuts_left)
        dp(row,col,cuts_left) be the number of wats to cut the pizza part pizza[row..rows-1][col..cols-1] with remanin cuts
        we want dp(0,0,k-1)
        
        base case is when cuts_left = 0
        if we are at dp(row,col,0) and there is at least 1 apple in pizze[row..rows-1][cols..cols-1] then dp(row,col,0) = 0
        because with no cuts, we still get an apple, reulting in a valid way
        otherwise dp(row,col,0) = 0
        
    transition:
        when we cut hoirzontally, the next row (next_row) must be between row < next_row < rows, 
        the upper part after the cutt will be pizza[row..next_row-1][cols-1] and the bottom part will be pizza[next_row...rows-1][cols-1]
            recall we give the top oart to a person if pizza[row...next_row -1][col .. cols -1] has at leat one apple
        for the vertical cut, the next_col must be between col < next_col < cols
            the left part of the pissza will be pizza[row..rows-1][col .. next_col-1] and the right part will be pizza[row..rowx-1][next_col..cols-1]
            there must be an apple in the left part pizzza[row..rows-1][col..next_col-1]
            
        say we want to calculate dp(r,c,cuts_left) where cuts_left > 0
            we need to try all cuts and accumalte the number of ways
            * for rows, iterate on all row cuts (next_row) where: row < next_row < rows
                * if pizza[row..next_row-1][col .. cols -1] has at least on apple, we can cut it further, recurse
            * after this cut, we need to try cutting the pizza gain with cuts_left - 1 and the remaning parts of the pizza
            * which would be pizza[next_row..rows-1][col..cols-1], this is a subproblem
            * for cols, tierate on all col cuts (next_col) wher col < next_col < cols
                if pizza[row..rows-1][col..next_col-1] has an apple, we recurse further
                    * dp(r,next_col,cuts_left -1 )
        
        there is a transition from dp[remain-1][next_row][col] to dp[remain][row][col]
        same thing with the coltransistion 
        so: dp(row,col,remain)  = {
            sum for (next_row) in row < next_row < rows of dp(next_row,col,remain - 1) +
            sum for (next_col) in col < next_col < col of dp (row,next_col,remain-1)
        }
                    
        we need to quickly verfiy apples in a candidate cutted piece
        recall 2d range query
        '''
        rows = len(pizza)
        cols = len(pizza[0])
        
        apples = [[0]*(cols+1) for _ in range(rows+1)]
        #we need to start from the end, suffix sums
        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                apples[i][j] = (pizza[i][j] == 'A') + apples[i+1][j] + apples[i][j+1] - apples[i+1][j+1]
        
        memo = {}
        mod = 10**9 + 7
        
        def dp(r,c,cuts_left):
            if cuts_left == 0:
                return apples[r][c] > 0
            if (r,c,cuts_left) in memo:
                return memo[(r,c,cuts_left)]
            ans = 0
            for row_cut in range(r+1,rows):
                if apples[r][c] - apples[row_cut][c] > 0:
                    ans += dp(row_cut,c,cuts_left - 1)
            
            for col_cut in range(c+1,cols):
                if apples[r][c] - apples[r][col_cut] > 0:
                    ans += dp(r,col_cut,cuts_left - 1)
            
            memo[(r,c,cuts_left)] = ans
            return ans
        
        return dp(0,0,k-1) % mod
        
#bottom up
class Solution:
    def ways(self, pizza: List[str], k: int) -> int:
        rows = len(pizza)
        cols = len(pizza[0])
        apples = [[0] * (cols + 1) for row in range(rows + 1)]
        for row in range(rows - 1, -1, -1):
            for col in range(cols - 1, -1, -1):
                apples[row][col] = ((pizza[row][col] == 'A')
                                    + apples[row + 1][col]
                                    + apples[row][col + 1]
                                    - apples[row + 1][col + 1])
        dp = [[[0 for col in range(cols)] for row in range(rows)] for remain in range(k)]
        dp[0] = [[int(apples[row][col] > 0) for col in range(cols)]
             for row in range(rows)]
        mod = 1000000007
        for remain in range(1, k):
            for row in range(rows):
                for col in range(cols):
                    val = 0
                    for next_row in range(row + 1, rows):
                        if apples[row][col] - apples[next_row][col] > 0:
                            val += dp[remain - 1][next_row][col]
                    for next_col in range(col + 1, cols):
                        if apples[row][col] - apples[row][next_col] > 0:
                            val += dp[remain - 1][row][next_col]
                    dp[remain][row][col] = val % mod
        return dp[k - 1][0][0]