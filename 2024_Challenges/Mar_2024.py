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