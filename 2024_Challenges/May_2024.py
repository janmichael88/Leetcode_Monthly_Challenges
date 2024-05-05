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
    
    