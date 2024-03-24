###########################################
# 2864. Maximum Odd Binary Number
# 01MAR24
###########################################
class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        '''
        odds only have 1 set bit at the 0th posiition
        count up all bits, place 1 on the 0th pos, then load the 1 bits on the MSB positions
        '''
        N = len(s)
        count_ones = 0
        for ch in s:
            count_ones += ch == '1'
        
        return '1'*(count_ones-1)+'0'*(N - count_ones)+'1'
    
#################################################
# 3062. Winner of the Linked List Game
# 01MAR24
##################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def gameResult(self, head: Optional[ListNode]) -> str:
        '''
        just advance two steps at a time and check
        curr and curr.next
        '''
        odd = 0
        even = 0
        
        curr = head
        while curr:
            if curr.next.val > curr.val:
                odd += 1
            else:
                even += 1
            curr = curr.next.next
        
        if odd > even:
            return 'Odd'
        elif even > odd:
            return 'Even'
        else:
            return 'Tie'
        
#we can just maintain point difference too
        # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def gameResult(self, head: Optional[ListNode]) -> str:
        '''
        just advance two steps at a time and check
        curr and curr.next
        '''
        score = 0
        
        curr = head
        while curr:
            if curr.next.val > curr.val:
                score += 1
            else:
                score -= 1
            curr = curr.next.next
        
        if score > 0:
            return 'Odd'
        elif score < 0:
            return 'Even'
        else:
            return 'Tie'

#####################################################
# 1474. Delete N Nodes After M Nodes of a Linked List
# 03MAR24
#####################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteNodes(self, head: ListNode, m: int, n: int) -> ListNode:
        curr = head
        prev = head
        
        while curr:
            curr_m = m
            curr_n = n
            while curr and curr_m > 0:
                prev = curr
                curr = curr.next
                curr_m -= 1
            
            while curr and curr_n > 0:
                curr = curr.next
                curr_n -= 1
            
            prev.next = curr
        
        return head
    
################################################
# 1589. Maximum Sum Obtained of Any Permutation
# 04MAR24
#################################################
#harder than just building the highest order perm T.T
class Solution:
    def maxSumRangeQuery(self, nums: List[int], requests: List[List[int]]) -> int:
        '''
        no matter how we permute nums, the entire sum will always remain the same
        say i have some requests (i,j), if we want sum(nums[i:j+1]) to be maxed out
        then we want the the largest numbers in nums[i:j+1]
        but we need to use the same permutation for all requests
        indicies with higher frequncies should be bound to larger values
        
        get counts of indicies in requests, then map this to larger values
        order should be decreasing but starting with highest values
        '''
        counts = Counter()
        n = len(nums)
        for left,right in requests:
            counts[left] += 1
            counts[right] += 1
        #put indices not in counts with 0 feq
        for i in range(n):
            if i not in counts:
                counts[i] = 0

        
        #sort by decreasing 
        counts = [(freq,idx) for idx,freq in counts.items()]
        counts.sort(key = lambda x : (-x[0],x[1]))
        
        #build highest order permutation
        highest_perm = [0]*n
        nums.sort(reverse = True)
        
        ptr = 0
        for _,idx in counts:
            highest_perm[idx] = nums[ptr]
            ptr += 1
            
        print(highest_perm)
        #build pref_sum
        pref_sum = [0]
        for num in highest_perm:
            pref_sum.append(pref_sum[-1] + num)
        
        ans = 0
        for l,r in requests:
            curr_sum = pref_sum[r+1] - pref_sum[l]
            ans += curr_sum
        
        return ans % 1_000_000_007
        
#sweep line into pref sum
class Solution:
    def maxSumRangeQuery(self, nums: List[int], requests: List[List[int]]) -> int:
        '''
        sweep line paradigm, to count the number of times we visit an index that cover this request
        recall intervals needs to be accumulated to count up the number of times
        pref_sum of the intervals array is just the number of times index i has been accessed
        then greedily assign each index the largest valuei nums
        
        '''
        N = len(nums)
        intervals = [0]*(N+1)
        for start,end in requests:
            intervals[start] += 1
            intervals[end+1] -= 1
        
        #running sum to get the number of times this index was requested
        pref_sum = [0]*N
        pref_sum[0] = intervals[0]
        for i in range(1,N):
            pref_sum[i] = intervals[i] + pref_sum[i-1]
        
        
        mod = 10**9 + 7
        #sort nums and prevf_sum
        nums.sort()
        pref_sum.sort()
        
        ans = 0
        for num,count in zip(nums,pref_sum):
            ans += (num*count) % mod
        
        return ans % mod
    
#############################################################
# 1750. Minimum Length of String After Deleting Similar Ends
# 05MAR24
##############################################################
class Solution:
    def minimumLength(self, s: str) -> int:
        '''
        two pointers
        pref and suffix cannot be intersection
        '''
        N = len(s)
        if N == 1:
            return 1
        
        left = 0
        right = N-1
        while left < right and s[left] == s[right]:
            curr_char = s[left]
            
            #remember prefix and suffix cannot be intersecting
            while left <= right and s[left] == curr_char:
                left += 1
            
            while right > left and s[right] == curr_char:
                right -= 1
        
        return right - left + 1
    
###############################################
# 1181. Before and After Puzzle
# 06MAR24
###############################################
class Solution:
    def beforeAndAfterPuzzles(self, phrases: List[str]) -> List[str]:
        '''
        split each phrase on spaces to turn into list of list
        '''
        phrases = [p.split(" ") for p in phrases]
        ans = set()
        N = len(phrases)
        for i in range(N):
            for j in range(N):
                if i != j:
                    first = phrases[i]
                    second = phrases[j]
                    if first[-1] == second[0]:
                        temp = first + second[1:]
                        ans.add(" ".join(temp))
                    if second[-1] == first[0]:
                        temp = second + first[1:]
                        ans.add(" ".join(temp))
        
        ans = list(ans)
        ans.sort()
        return ans
    
#right idea with hasmap
class Solution:
    def beforeAndAfterPuzzles(self, phrases: List[str]) -> List[str]:
        '''
        use hashmap to map first word to remaning phrase
        and end word to remnaining
        mapp word to remaining phrase
        '''
        phrases = [p.split(" ") for p in phrases]
        beginning = defaultdict(set)
        ending = defaultdict(set)
        ans = set()
        N = len(phrases)
        for p in phrases:
            #ending word
            if p[-1] in beginning:
                for e in beginning[p[-1]]:
                    temp = " ".join(p) + " " + e
                    ans.add(temp)
            if p[0] in ending:
                for s in ending[p[0]]:
                    temp = s + " " + " ".join(p)
                    ans.add(temp)
            #populdate
            beginning[p[0]].add(" ".join(p[1:]))
            ending[p[-1]].add(" ".join(p[:-1]))
        
        ans = list(ans)
        ans.sort()
        return ans
    
class Solution:
    def beforeAndAfterPuzzles(self, phrases: List[str]) -> List[str]:
        first = collections.defaultdict(set)
        last = collections.defaultdict(set)
        res = set()
        for p in phrases:
            strs = p.split(' ')
            if strs[0] in last:
                res |= {a + p[len(strs[0]):] for a in last[strs[0]]}
            if strs[-1] in first:
                res |= {p + b for b in first[strs[-1]]}
            first[strs[0]].add(p[len(strs[0]):])
            last[strs[-1]].add(p)
        return sorted(list(res))

##############################################
# 2570. Merge Two 2D Arrays by Summing Values
# 07MAR24
###############################################
class Solution:
    def mergeArrays(self, nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
        '''
        they are sorted so use two pointers
        '''
        ans = []
        i,j = 0,0
        
        while i < len(nums1) and j < len(nums2):
            if nums1[i][0] < nums2[j][0]:
                ans.append(nums1[i])
                i += 1
            elif nums2[j][0] < nums1[i][0]:
                ans.append(nums2[j])
                j += 1
            
            #if they are equal, combine
            else:
                entry = [nums1[i][0], nums1[i][1] + nums2[j][1]]
                ans.append(entry)
                i += 1
                j += 1
        

        #reached the end of one array but not the other
        while i < len(nums1):
            ans.append(nums1[i])
            i += 1
        
        while j < len(nums2):
            ans.append(nums2[j])
            j += 1
        
        return ans
        
################################################
# 3005. Count Elements With Maximum Frequency
# 08MAR24
#################################################
class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        '''
        find max frequency then count up elements who's max freq is that
        '''
        counts = Counter(nums)
        max_freq = max(counts.values())
        ans = 0
        for k,v in counts.items():
            if v == max_freq:
                ans += v
        
        return ans
    
#one pass
class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        '''
        we can do it on the fly
        just keep of current max freq
        either add to ans or update max_freq and reset
        '''
        counts = {}
        max_freq = 0
        max_freq_count = 0
        
        for num in nums:
            counts[num] = counts.get(num,0) + 1
            curr_freq = counts[num]
            
            #update
            if curr_freq > max_freq:
                max_freq = curr_freq
                max_freq_count = curr_freq
            #increment
            elif curr_freq == max_freq:
                max_freq_count += curr_freq
        
        return max_freq_count
    
#######################################
# 2540. Minimum Common Value
# 09MAR24
#######################################
#two pointer
class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        this is just merge sort
        keep moving two pointers until we find a common one
        '''
        i,j = 0,0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums2[j] < nums1[i]:
                j += 1
            else:
                return nums1[i]
        
        return -1
        
#binary search
class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can also use binary search, search for one in the other
        to make the algorithm more effecient we need to do binary search on the larger array
        '''
        if len(nums1) > len(nums2):
            return self.getCommon(nums2,nums1)
        
        for num in nums1:
            if self.binary_search(num,nums2):
                return num
        
        return -1
    
    
    def binary_search(self, num,arr):
        left = 0
        right = len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            #found it
            if arr[mid] == num:
                return True
            elif arr[mid] > num:
                right = mid - 1
            elif arr[mid] < num:
                left = mid + 1
        
        return False


#####################################
# 3063. Linked List Frequency
# 08MAR24
#####################################
#two pass
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def frequenciesOfElements(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        traverse and keep temp count map
        '''
        counts = {}
        curr = head
        while curr:
            counts[curr.val] = counts.get(curr.val,0) + 1
            curr = curr.next
        
        dummy = ListNode(-1)
        curr = dummy
        for k,v in counts.items():
            newNode = ListNode(v)
            curr.next = newNode
            curr = curr.next
        
        return dummy.next
    
#one pass
#map value of num to counts but a ListNode objects
#then pointer manip
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def frequenciesOfElements(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        we can do it in any more
        so just map an element to a ListNode with its current count
        '''
        counts = {}
        curr = head
        dummy = ListNode(-1)
        freq_head = dummy
        
        while curr:
            if curr.val in counts:
                #get the count node
                count_node = counts[curr.val]
                count_node.val += 1
            else:
                #create new node
                newNode = ListNode(1)
                freq_head.next = newNode
                counts[curr.val] = newNode
                freq_head = freq_head.next
            
            curr = curr.next
        
        return dummy.next
    
########################################################
# 1475. Final Prices With a Special Discount in a Shop
# 10MAR24
########################################################
#brute force
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        '''
        search for the minimum j
        '''
        ans = []
        N = len(prices)
        
        for i in range(N):
            discount = 0
            for j in range(i+1,N):
                if prices[j] <= prices[i]:
                    discount = prices[j]
                    break
            
            ans.append(prices[i] - discount)
        
        return ans
    
#monostack solution
class Solution:
    def finalPrices(self, prices: List[int]) -> List[int]:
        '''
        monostack
        '''
        N = len(prices)
        discounted = prices[:]
        stack = [] #store as [idx]
        
        #strictly increasing prices
        for i,p in enumerate(prices):
            #apply discounts for all indices who's price is larger than current price
            while stack and prices[stack[-1]] >= p:
                idx = stack.pop()
                discounted[idx] -= p
            #add back in
            stack.append(i)
        
        return discounted
                
############################################
# 2545. Sort the Students by Their Kth Score
# 11MAR24
############################################
class Solution:
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        '''
        custom comparator
        '''
        #store is [student id, kth score]
        pairs = []
        N = len(score)
        for i in range(N):
            pairs.append([i,score[i][k]])
        
        #sort high to low
        pairs.sort(key = lambda x : -x[1])
        
        ans = []
        for i,kth in pairs:
            ans.append(score[i][:])
        
        return ans

class Solution:
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        #one liner
        return sorted(score, key = lambda x : -x[k])

#########################
# 699. Falling Squares
# 10MAR24
#########################
class Solution:
    def fallingSquares(self, positions: List[List[int]]) -> List[int]:
        '''
        advent of code problem!
        say we given sqaure with left edge i and side length k
        then it covers i to i + k
        initialy each is at height 0
        but sidelngth is too big
        now the question is given range [i, i+k], efficiently find the largest height
        or, given state (i,i+k) what can we do to make the search more mangageable
        
        N squared works!
        '''
        height = defaultdict(lambda : 0)
        ans = []
        for left,side in positions:
            start = left
            end = left + side
            curr_height = 0
            for k in range(start,end+1):
                height[k] += side
                curr_height = max(curr_height, height[k])
            ans.append(curr_height)
        

        max_ans = [ans[0]]
        for num in ans[1:]:
            max_ans.append(max(num,max_ans[-1]))
        
        return max_ans
            
#offline prop/coor compression
class Solution:
    def fallingSquares(self, positions: List[List[int]]) -> List[int]:
        '''
        framework alludes to segment tree
        we have two operations -> update (after dropping square)
                                -> query (find largest height)
        since there are only up to 2*len(postitions) critical points, the left and rights of each sqyar
        we can use "coordinate compression" which maps these critical points to adjacent integers
        example:
            coords = set()
            for left,size in positions:
                coords.add(left)
                coords.add(left + size - 1)
        index = {x:i for i,x in enumerated(sorted(coords))}
        
        approach 1; offline propgations
            insteaf of asking the qeustions what quares effect this query, we cask what queries are affected by this square
            let aans be the max height of the interval specified by posiitions[i]
            in the end return running max of q ans
        
        for each square at positions[i], the max height will get higher by the size fo the swuare we drop
        then for any future squares in the interval [left,right], where left = pos[i][0], right = pos[i][0] + pos[i][1] we update the height
        '''
        N = len(positions)
        ans = [0]*N
        for i,(left,size) in enumerate(positions):
            right = left + size
            ans[i] += size
            #look ahead and see if we get another max height update
            for j in range(i+1,N):
                left2,size2 = positions[j]
                right2 = left2 + size2
                #intersect, which means heights get update
                if left2 < right and left < right2:
                    ans[j] = max(ans[j], ans[i])
        
        #need running max
        final_ans = [ans[0]]
        for num in ans[1:]:
            final_ans.append(max(final_ans[-1],num))
        
        return final_ans
    
#############################################################
# 1171. Remove Zero Sum Consecutive Nodes from Linked List
# 12MAR24
###############################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        keep running sum, when zero, we need to delete
        there are regions in where sum == 0, not running sum
        '''
        dummy = ListNode(0)
        dummy.next = head
        left = dummy
        while left:
            curr_sum = 0
            right = left.next
            while right:
                curr_sum += right.val
                #everythign between left and right inclusive is 0
                if curr_sum == 0:
                    #print(left.val,right.val)
                    right = right.next
                    left.next = right
                    left = right
                    if right:
                        right = right.next
                else:
                    right = right.next
            if left:
                left = left.next
        
        
        return dummy.next
    
#finally!
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        keep running sum, when zero, we need to delete
        there are regions in where sum == 0, not running sum
        '''
        dummy = ListNode(0)
        dummy.next = head
        left = dummy
        while left:
            curr_sum = 0
            right = left.next
            while right:
                curr_sum += right.val
                #everythign between left and right inclusive is 0
                if curr_sum == 0:
                    #print(left.val,right.val)
                    left.next = right.next
                right = right.next
                    
            left = left.next
        
        
        return dummy.next
    
#one pass
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        imagine that we were working with an array intead of a linked list
        [1, 2, -3, 3, 1]
        generating pref sum
        [0, 1, 3, 0, 3, 4]
               ^     ^
        these two spots have the same pref_sum
        which means the subarray between these two must be zero
        intuition -> a zero sum conescutive sequence will have a pref sum of zer0
        rather the prefum before and at the end of the sequence will be the same
        
        subarray sum equals K
        so we have pref_sum[j] == pref_sum[i], where j >= i
        then the sum between i and j must be zero, this must be the case inorder to make the two differnt pref_sums equal
        
        so when we enctouner a pref sum we have seen before, then we know that this curren subequence must be zero sum
        If you accumulate a prefix sum, any repeating values necessitate that the segment in the middle equals 0. 
        For example, in a prefix sum of [1,2,5,2], the values in between the 2nd and 4th nodes must sum to 0, or the prefix sum would not have repeated
        
        crucial insight:
        pref_sum from the front of some node A == the sum of the front of some node B, if and only if the sum from A.next to B is zero
        we use hasmap and store pref_sum values to their nodes
        '''
        dummy = ListNode(0)
        dummy.next = head
        pref_sum = 0
        mapp = {0:dummy}
        
        curr = dummy
        while curr:
            pref_sum += curr.val
            mapp[pref_sum] = curr
            curr = curr.next
        
        curr = dummy
        pref_sum = 0
        
        while curr:
            pref_sum += curr.val
            curr.next = mapp[pref_sum].next
            curr = curr.next
        
        return dummy.next
    
#########################################
# 2485. Find the Pivot Integer
# 13MAR24
##########################################
class Solution:
    def pivotInteger(self, n: int) -> int:
        '''
        we can get the pref_sum from to n, then just check if we can pivot
        '''
        pref_sum = [0]
        for i in range(1,n+1):
            pref_sum.append(pref_sum[-1] + i)
        
        for i in range(len(pref_sum)):
            if (pref_sum[i] == pref_sum[-1] - pref_sum[i] + i):
                return i
        
        return -1
    
#binary search, weeee!
class Solution:
    def pivotInteger(self, n: int) -> int:
        '''
        we can use binary search
        just use gauss trick to get sum of eleemnts for some number n
        '''
        def getSum(num):
            return (num*(num+1)) // 2
        
        left = 1
        right = n
        SUM = getSum(n)
        
        while left <= right:
            mid = left + (right - left) // 2
            #compute pivot
            pivot_sum = getSum(mid)
            if pivot_sum == SUM - pivot_sum + mid:
                return mid
            elif pivot_sum > SUM - pivot_sum + mid:
                right = mid - 1
            else:
                left = mid + 1
        
        return -1
            
#o(1) true
class Solution:
    def pivotInteger(self, n: int) -> int:
        '''
        we want
        1 + 2 + ... x = x + (x+1) + ... +  (x+n)
        which is just (x*(x+1)) // 2 = ((x+n)*(n-x+1)) // 2
        
        here we can get
        x * (x*x) /2 = (nx - x*x + x + n*n - n*x + n) / 2
        simplygin we can get
        2x*x = n*n + n
        x = sqrt((n*n + n) / 2)
        '''
        SUM = (n*(n+1)) // 2
        pivot = int(math.sqrt(SUM))
        if pivot*pivot == SUM:
            return pivot
        return -1
    
###############################################
# 238. Product of Array Except Self (REVISTED)
# 15MAR24
#############################################
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        use products going to the left and products going to the right
        '''
        N = len(nums)
        left_prods = [0]*(N)
        left_prods[0] = 1
        for i in range(1,N):
            left_prods[i] = left_prods[i-1]*nums[i-1]
        
        right_prods = [0]*N
        right_prods[N-1] = 1
        for i in range(N-2,-1,-1):
            right_prods[i] = nums[i+1]*right_prods[i+1]
        
        ans = [0]*N
        for i in range(N):
            ans[i] = left_prods[i]*right_prods[i]
        
        return ans

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        we could also padd, and keep prefix up to
        and suffix up t0
        '''
        N = len(nums)
        left_prods = [1]
        for num in nums:
            left_prods.append(left_prods[-1]*num)
        
        right_prods = [1]
        for num in nums[::-1]:
            right_prods.append(right_prods[-1]*num)
        right_prods = right_prods[::-1]
        
        ans = []
        for i in range(len(left_prods)-1):
            ans.append(left_prods[i]*right_prods[i+1])
        
        return ans
    
#optimized
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        accumulate into ans is 0(1)
        '''
        N = len(nums)
        ans = [0]*N
        ans[0] = 1
        for i in range(1,N):
            ans[i] = ans[i-1]*nums[i-1]
            
        right_prods = 1
        for i in reversed(range(N)):
            ans[i] *= right_prods
            right_prods *= nums[i]
        
        return ans


#########################################
# 930. Binary Subarrays With Sum
# 14MAR24
#########################################
#subarray sum == k
#if pref_sum[i] == pref_sum[j] + k, then in between sum is k
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        '''
        this is just subarray sum equals k
        but it cannot be an empty subarray
        '''
        count = 0
        curr_sum = 0
        count_mapp = {}
        count_mapp[0] = 1
        
        for num in nums:
            curr_sum += num
            count += count_mapp.get(curr_sum - goal , 0)
            count_mapp[curr_sum] = count_mapp.get(curr_sum,0) + 1
        
        return count
    
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        '''
        if we have seen a pref sum up to know we increment a count
        if we see pref_sum - goal, we also increment up
        idea is that pref_sum (up until now) - goal represents another subarray that when added tot he current subarray, makes goal
        '''
        count = 0
        freq_count = {}
        curr_sum = 0
        
        for num in nums:
            curr_sum += num
            if (curr_sum == goal):
                count += 1
            
            if (curr_sum - goal) in freq_count:
                count += freq_count[(curr_sum - goal)]
            
            freq_count[(curr_sum)] = freq_count.get(curr_sum,0) + 1
        
        return count

#indireclty count
class Solution:
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        '''
        the problem with regular sliding window is that adding a zero does not increment
        the current window sum, in regular sliding window problems we keep adding until some constraint is satisfied
        if we were to keep adding zero's we might miss valid arrays
            i.e the presence of zeros allows us to combine them to reach the goal
        
        subarrays exceeding the goal are not needed
        we only care about the arrays where sum <= goal
        
        so we can indirectly count the binary subarrays <= goal
        after find this count we need to isolate the subarrays that strictly meet the target goal
        say we had a function sliding window at most, that counts subarrays at most goal
        the answer is just sliding_window_at_most(nums,goal) - sliding_window_at_most(nums,goal-1)
        '''
        
        return self.count_at_most(nums,goal) - self.count_at_most(nums,goal-1) 
        
    
    def count_at_most(self, nums : List[int], goal : int) -> int:
        start = 0
        curr_sum = 0
        count = 0
        
        for end in range(len(nums)):
            curr_sum += nums[end]
            #shrink window if we are over
            while start <= end and curr_sum > goal:
                curr_sum -= nums[start]
                start += 1
            
            
            #get count for current subarray
            #if subarray has length k, and curr sum is < goal, then number of subarray is just k
            count += end - start + 1
        
        return count

########################################
# 525. Contiguous Array (REVISTED)
# 17MAR24
########################################
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        '''
        sliding window with count mapp
        counts will be increasing
        if we are in a subarray where the counts are uneuqal, we keep tring to incorportate elements
        for an array with counts 1s == count 0s
        sum(curr_sub_array) == count 0s
        
        subarray sum == k, but instead of counts, keep track of index
        instead of counting for equal zeros and ones, record difference

        idea is that we see another occurence of cumlative difference, there are the same number of 0s and 1s
        '''
        mapp = {}
        mapp[0] = -1
        
        max_length = 0
        curr = 0
        
        for i,num in enumerate(nums):
            curr += 1 if num == 1 else -1
            if curr in mapp:
                max_length = max(max_length, i - mapp[curr])
            else:
                mapp[curr] = i
        
        return max_length

###########################################
# 621. Task Scheduler (REVISTED)
# 20MAR24
###########################################
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        '''
        first show that using lowest freqs increases minimum time
        using high freq taskses decreases minimum time
        need to consider cycle length + 1 (n+1)
            pick highest, and if tie it doesn't matter
            reduce task by 1
            when picking the next counts maintain temp array to pick tasks and make sure it doesn't go beyong the cycle length
        
        the issue is that there needs to be at least n intervals between the next scheduled task
        '''
        counts = Counter(tasks)
        #we actually dont need the task chars, just their counts
        pq = [-v for _,v in counts.items()]
        heapq.heapify(pq)
        
        total_time = 0
        while pq:
            cycle_time = n + 1 #cooling interval for a cycle
            counts_used = []
            task_count = 0
            while cycle_time > 0 and pq:
                curr_freq = heapq.heappop(pq)
                curr_freq += 1
                task_count += 1
                cycle_time -= 1
                if curr_freq < 0:
                    counts_used.append(curr_freq)
            
            for c in counts_used:
                heapq.heappush(pq,c)
            
            #if there needs to be a cooling time
            if len(pq) > 0:
                total_time += n + 1
            #otherwise increment by the cooling time
            else:
                total_time += task_count
            #total_time += task_count if not pq else n + 1
        
        return total_time
    
#heap and waiting queue paradigm
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        '''
        use heap and waiting queue,
        if a task needs cool down put it in the q
        '''
        counts = Counter(tasks)
        #we actually dont need the task chars, just their counts
        pq = [-v for _,v in counts.items()]
        heapq.heapify(pq)
        q = deque([])
        
        time = 0
        while q or pq:
            time += 1
            if pq:
                curr_freq = heapq.heappop(pq)
                curr_freq += 1
                if curr_freq < 0:
                    q.append([curr_freq, time + n])
            
            if q and q[0][1] <= time:
                #put back into heap
                heapq.heappush(pq, q.popleft()[0])
        
        return time
            
#####################################
# 1669. Merge In Between Linked Lists
# 20MAR24
#####################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        '''
        advance to the ath nodes, and keep prev and curr pointers
        then advance to the bith node and keep prev and curr pointers
        then connect
        '''
        a_prev,a_curr = None,list1
        for _ in range(a):
            a_prev = a_curr
            a_curr = a_curr.next
        
        b_prev,b_curr = a_prev,a_curr
        for _ in range(b-a+1):
            b_prev = b_curr
            b_curr = b_curr.next
        
        a_prev.next = list2
        #print(a_prev.val,a_curr.val)
        #print(b_prev.val,b_curr.val)
        #need to connect end of list2 to b_curr
        c_prev,c_curr = None,list2
        while c_curr:
            c_prev = c_curr
            c_curr = c_curr.next
        
        c_prev.next = b_curr
        
        return list1
    
######################################
# 2365. Task Scheduler II
# 21MAR24
######################################
class Solution:
    def taskSchedulerII(self, tasks: List[int], space: int) -> int:
        '''
        tasks must be done in order, can only do one task in one day
        space must be in between tasks of the same type
        say i have completed task 1 on day i, if i see it again on some day j, j - i > space, its its not take a break until we get space
        so keep track of last time done
        '''
        ans = 0
        time_last = {}
        for t in tasks:
            ans += 1
            if t in time_last and ans - time_last[t] <= space:
                ans += space - (ans - time_last[t]) + 1
            #updae
            time_last[t] = ans
        
        return ans
    
########################################
# 2848. Points That Intersect With Cars
# 21MAR24
#########################################
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        '''
        just need number of points touchiung cars
        sort on starts, then used cooridnate compression
        '''
        points = [False]*(100+1)
        for s,e in nums:
            for p in range(s,e+1):
                points[p] = True
        
        return sum(points)
    
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        '''
        just need number of points touchiung cars
        sort on starts, then used cooridnate compression
        '''
        nums.sort(key = lambda x: x[0])
        points = 0
        curr_start,curr_end = nums[0]
        
        for s,e in nums[1:]:
            if s <= curr_end:
                curr_end = max(curr_end,e)
            else:
                points += (curr_end - curr_start) + 1
                curr_start,curr_end = s,e
        
        points += (curr_end - curr_start) + 1
        return points
    
#line sweep
#https://leetcode.com/discuss/study-guide/2166045/line-sweep-algorithms
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        '''
        just need number of points touchiung cars
        sort on starts, then used cooridnate compression
        +1 for all in range
        -1 just after to negate the change, then we can count them all up
        '''
        points = [0]*(100+2)
        
        for s,e in nums:
            points[s] += 1
            points[e+1] -= 1
        
        ans = 0
        pref_sum = 0
        for num in points:
            pref_sum += num
            if pref_sum > 0:
                ans += 1
        
        return ans
    
###########################################
# 1054. Distant Barcodes
# 24MAR24
##########################################
class Solution:
    def rearrangeBarcodes(self, barcodes: List[int]) -> List[int]:
        
        counts = Counter(barcodes)
        pq = [(-count,code) for code,count in counts.items()]
        #ans will always exsists
        heapq.heapify(pq)
        ans = []
        
        while pq:
            first_count,first_code = heapq.heappop(pq)
            first_count += 1
            ans.append(first_code)
            if not pq:
                return ans
            second_count,second_code = heapq.heappop(pq)
            second_count += 1
            ans.append(second_code)
            
            if first_count < 0:
                heapq.heappush(pq, (first_count,first_code))
            
            if second_count < 0:
                heapq.heappush(pq, (second_count, second_code))
        
        return ans