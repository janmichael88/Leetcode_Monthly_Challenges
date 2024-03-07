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
