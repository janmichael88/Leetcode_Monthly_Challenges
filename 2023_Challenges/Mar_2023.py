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
    
