#####################################################################
# 3067. Count Pairs of Connectable Servers in a Weighted Tree Network
# 24APR24
######################################################################
#closeeee
class Solution:
    def countPairsOfConnectableServers(self, edges: List[List[int]], signalSpeed: int) -> List[int]:
        '''
        notice how nodes with only degree 1 will always have zero
        for a node to be even considered connectable, it must be in the middle
        hint1: take each node as root, run dfs rooted at that node i and get nodes whose distnace is divisible by signal speed
        '''
        graph = defaultdict(list)
        for u,v,w in edges:
            graph[u].append((v,w))
            graph[v].append((u,w))
        
        n = len(graph)
        num = [0]*n
        
        for i in range(n):
            self.dfs(i,i,None,graph,num,signalSpeed,0)
        print(num)
    def dfs(self, root,node, parent, graph, num, signalSpeed,curr_dist):
        if curr_dist % signalSpeed == 0:
            num[root] += 1
        for neigh,weight in graph[node]:
            if neigh != parent:
                self.dfs(root,neigh,node,graph,num,signalSpeed,curr_dist + weight)

###############################################################
# 2441. Largest Positive Integer That Exists With Its Negative
# 02MAY24
###############################################################
class Solution:
    def findMaxK(self, nums: List[int]) -> int:
        '''
        can just find the max and min and check if opposite in sign
        sort and use hash set
        '''
        nums.sort(reverse = True)
        set_nums = set(nums)
        
        i = 0
        while i < len(nums) and nums[i] > 0:
            if -nums[i] in set_nums:
                return nums[i]
            i += 1
        
        return -1
    
#################################################
# 165. Compare Version Numbers (REVISITED)
# 03MAY24
#################################################
class Solution:
    def compareVersion(self, version1: str, version2: str) -> int:
        '''
        if i conver them to arrays
        1.01 -> [1,1]
        1.001 -> [1,1] ==
        then pad the arrays  so that they equal to 
        '''
        v1 = [int(num) for num in version1.split('.')]
        v2 = [int(num) for num in version2.split('.')]
        
        if len(v1) < len(v2):
            v1 = v1 + [0]*(len(v2) - len(v1))
        
        if len(v2) < len(v1):
            v2 = v2 + [0]*(len(v1) - len(v2))

        for a,b in zip(v1,v2):
            if a != b:
                return 1 if a > b else -1
        
        return 0
    
########################################
# 817. Linked List Components
# 04MAY24
########################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        '''
        two values are connected if they appear consecutively in the list
        we need to return the connected componentes in nums
        note the values are uniuqe in nums
        kepp order or linked list as array, then mark and count streaks
        [0,1,2,3]
        used another linked list to mark connections
        now count streaks in linked list (similar to count streaks in array)
        
        '''
        nums = set(nums)
        marked = ListNode(-1)
        
        ptr_marked = marked
        ptr_head = head
        while ptr_head:
            if ptr_head.val in nums:
                new_node = ListNode(True)
            else:
                new_node = ListNode(False)
            
            ptr_marked.next = new_node
            ptr_marked = ptr_marked.next
            ptr_head = ptr_head.next
        
        
        curr = head
        marked = marked.next
        ans = 0
        is_connected = False
        
        while marked:
            if marked.val and is_connected:
                marked = marked.next
            elif marked.val and not is_connected:
                is_connected = True
                marked = marked.next
            elif not marked.val and is_connected:
                ans += 1
                is_connected = False
                marked = marked.next
            else:
                marked = marked.next
        
        if is_connected:
            ans += 1
        return ans
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def numComponents(self, head: Optional[ListNode], nums: List[int]) -> int:
        '''
        just increment count whenver we break a connection
        we check if curr val in nums and its next isn't
        '''
        nums = set(nums)
        curr = head
        ans = 0
        while curr:
            if curr.val in nums and (curr.next == None or curr.next.val not in nums):
                ans += 1
            curr = curr.next
        
        return ans
    
#############################################################
# 3125. Maximum Number That Makes Result of Bitwise AND Zero
# 05MAY24
#############################################################
class Solution:
    def maxNumber(self, n: int) -> int:
        '''
        so we have 7
        111 need to make this 0
        when performing bitwise and operations from n, the last set bit to turn 0, identieis the higehst set bit
        find the num with the higest set bit
        we need to clear all the bits anyway, might as well find the largest number that clears the left most bit position
        '''
        #find index of highest set bit
        for i in range(64,-1,-1):
            mask = 1 << i
            if n & mask:
                return 2**i - 1
            
class Solution:
    def maxNumber(self, n: int) -> int:
        '''
        keep shifting and count positiosn
        '''
        pos = 0
        while n:
            n = n >> 1
            pos += 1
        
        return 2**(pos-1) - 1
    
class Solution:
    def maxNumber(self, n: int) -> int:
        '''
        keep shifting and count positiosn
        '''
        pos = 0
        while n:
            n = n >> 1
            pos += 1
        
        return (1 << (pos - 1)) - 1
    
#########################################
# 1885. Count Pairs in Two Arrays
# 06MAY24
#########################################
#binary search
class Solution:
    def countPairs(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can rewrite: 
            nums1[i] + nums1[j] > nums2[i] + nums2[j]
        
        as:
            nums1[i] - nums2[j] > nums2[i] - nums1[j]
        
        now we are left with finding (i,j) pairs such that they satisfy this inequailtiy
        we can further rewrite as:
            (nums1[i] - nums2[j]) + (nums1[i] - nums2[j]) > 0
        
        now we have context from nums1 - nums2, and we need to find (i,j) pairs such that if we add them their differences, the ans is > 0
        sort and use binary search
        the only constraint is that i < j,
            binary search into the differences array and find the idx, where nums1[idx] - nums2[idx] > 0
        '''
        N = len(nums1)
        diffs = [nums1[i] - nums2[i] for i in range(N)]
        diffs.sort()
        
        count = 0
        for i in range(N):
            if diffs[i] > 0:
                #anything to the right would make a pair with the current i
                count += N - i - 1
            else:
                left = i+1
                right = N - 1
                while left <= right:
                    mid = left + ((right - left) // 2)
                    #look for the smallest index that satisfies the ineaulity
                    if diffs[mid] + diffs[i] > 0:
                        right = mid - 1
                    else:
                        left = mid + 1
                
                count += N - left
        
        return count
    
#two pointers
class Solution:
    def countPairs(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        two pointers
        left and right and just count pairs
        (nums1[i] - nums2[j]) + (nums1[i] - nums2[j]) > 0
        '''
        N = len(nums1)
        diffs = [nums1[i] - nums2[i] for i in range(N)]
        diffs.sort()
        
        count = 0
        left = 0
        right = N - 1
        
        while left < right:
            if diffs[left] + diffs[right] > 0:
                count += right - left
                right -= 1
            else:
                left += 1
        
        return count
            