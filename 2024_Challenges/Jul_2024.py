##################################################################
# 1909. Remove One Element to Make the Array Strictly Increasing
# 01JUL24
##################################################################
class Solution:
    def canBeIncreasing(self, nums: List[int]) -> bool:
        '''
        keep stack and count
        '''
        N = len(nums)
        for i in range(N):
            new_arr = nums[:i] + nums[i+1:]
            if self.isIncreasing(new_arr):
                return True
        return False
    
    def isIncreasing(self,arr):
        N = len(arr)
        for i in range(1,N):
            if arr[i] <= arr[i-1]:
                return False
        return True
    
class Solution:
    def canBeIncreasing(self, nums: List[int]) -> bool:
        '''
        if we find two consecutive eleements which are not increasing, we need to remove the bigger element
        greedily remove the bigger element
        '''
        prev = nums[0]
        used = False
        N = len(nums)
        for i in range(1,N):
            if nums[i] <= prev:
                if used:
                    return False
                used = True
                #rmoeve the elemnt from the i-1 pos, beacuse its bigger
                if (i == 1) or (nums[i] > nums[i-2]):
                    prev = nums[i]
            #remove curr eleemtn and leave prev to next num
            else:
                prev = nums[i]
            
        return True
    
#count non_increasing points
class Solution:
    def canBeIncreasing(self, nums: List[int]) -> bool:
        '''
        we need to count non_increasing points
        if 0, we dont need to remove anything
        if more than 1, we cant do
        if 1, we need to check
            removal at start and ends are allowed
            if somehwere in the middle
            [1,2,1,3,4], remove the second one
            or [1,10,2,3,4]
        '''
        indx = -1
        count = 0
        n = len(nums)
        
        # count the number of non-increasing elements
        for i in range(n-1):
            if nums[i] >= nums[i+1]:
                indx = i
                count += 1
        
        #the cases explained above
        if count==0:
            return True
        
        if count == 1:
            if indx == 0 or indx == n-2:
                return True
            if nums[indx-1] < nums[indx+1] or(indx+2 < n and nums[indx] < nums[indx+2]):
                return True
            
        return False
    
##############################################################################
# 1509. Minimum Difference Between Largest and Smallest Value in Three Moves
# 03JUL24
#############################################################################
#nice try :(
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        '''
        we need to shrink the gap in three moves
        we can either drop the maximum or raise the minimum
        does it matter?
        [5,3,2,4]
        
        make 5 to 4
        [4,4,3,2]
        make 2 to 4
        [4,4,3,4]
        make 3 to 4
        [4,4,4,4]
        
        sorted
        [5,4,3,2]
        [4,4,3,2]
        [4,4,4,2]
        [4,4,4,4]
        
        [14,10,5,1,0]
        i need to remove one of them
        if i remove 14, id get [10 - 0]
        if i remove 0, id get [14 - 1]
        remove 14
        [10,5,1,0]
        if i remove 10, id get [5 - 0]
        if i remove 0 id get [10 - 1]
        remove 10
        [5,1,0]
        if i remove 5 id get [1-0]
        if i remove 0 id get [5- 1]
        remove 5
        [1,0]
        
        what ones should we pick, try 3 largest or 3 smallest
        try them all
        '''
        N = len(nums)
        nums.sort(reverse = True)
        left = 0
        right = N - 1
        removals = 3
        while removals > 0 and left < right:
            #check if removing bigger
            if nums[left + 1] - nums[right] <= nums[left] - nums[right-1]:
                left += 1
                removals -= 1
            else:
                right -= 1
                removals -= 1
        
        return max(nums[left:right+1]) - min(nums[left:right+1])
    
#finally!
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        '''
        another brute force, try all scnearios
        '''
        N = len(nums)
        #corner case
        if N <= 4:
            return 0
        
        nums.sort()
        #we can take out first 3, last three, (first one, last two), (first two, last one)
        intervals = [[0,3],[1,2],[2,1],[3,0]]
        min_diff = float('inf')
        for left,right in intervals:
            temp = nums[left:N-right]
            min_diff = min(min_diff, max(temp) - min(temp))
        
        return min_diff
        
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        '''
        need to remove some combinatino of three smallest and three largest
        or remove 2 from largest and 1 from smallest
        or remove 2 from smallest and 1 from largest
        we just need to find min and max for all start and ends
        if we remove the three smallest, then we are looking in the nums[3:]
        removing three largest, then we are lookin in range nums[:-3]
        '''
        N = len(nums)
        #edge case
        if N <= 4:
            return 0
        nums.sort()
        min_diff = float('inf')
        #try all
        #remove three laregst
        min_diff = min(min_diff, nums[N-4] - nums[0])
        min_diff = min(min_diff, nums[N-3] - nums[1])
        min_diff = min(min_diff, nums[N-2] - nums[2])
        min_diff = min(min_diff, nums[N-1] - nums[3])
        
        return min_diff
                
#sinlge loop
class Solution:
    def minDifference(self, nums: List[int]) -> int:
        '''
        need to remove some combinatino of three smallest and three largest
        or remove 2 from largest and 1 from smallest
        or remove 2 from smallest and 1 from largest
        we just need to find min and max for all start and ends
        if we remove the three smallest, then we are looking in the nums[3:]
        removing three largest, then we are lookin in range nums[:-3]
        '''
        N = len(nums)
        #edge case
        if N <= 4:
            return 0
        nums.sort()
        min_diff = float('inf')
        #try all
        #remove three laregst
        for left in range(4):
            right = N - 4 + left
            min_diff = min(min_diff, nums[right] - nums[left])

        return min_diff

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        '''
        another brute force, try all scnearios
        proof by contraction
        assumption: that there is a better strategy removing elements not from the ends of the sorted array
        assumption2: take some index i to be deleted and 3 <= i < n-3 
            i.e i is outside range
        
        case 1: j,k < 3
            call j or k l, and nums[n-1] - nums[l], where l <= 3
            we can decrease this difference by removing l intsead of i becasme nums[l] <= nums[3] 
            only moving l would decrease the difference
        
        foolw same for cases
        case 2: j < 3 and k >= n-2
        case 3: j,k >= n-3
        both decrease minimum different by moving l instead of i
        
        initial assumption is wrong
        '''
        N = len(nums)
        #corner case
        if N <= 4:
            return 0
        
        nums.sort()
        #we can take out first 3, last three, (first one, last two), (first two, last one)
        intervals = [[0,3],[1,2],[2,1],[3,0]]
        min_diff = float('inf')
        for left,right in intervals:
            temp = nums[left:N-right]
            min_diff = min(min_diff, max(temp) - min(temp))
        
        return min_diff     
    
#################################################
# 2307. Check for Contradictions in Equations
# 03JUL24
################################################
#close one
class Solution:
    def checkContradictions(self, equations: List[List[str]], values: List[float]) -> bool:
        '''
        we can follow along a path
        say we start at node u and through some path end up at v, then (u/v) is the product of the edges
        '''
        graph = defaultdict(list)
        eqns_mapp = {}
        N = len(equations)
        
        for i in range(N):
            u,v = equations[i]
            w = values[i]
            eqns_mapp[(u,v)] = w
            eqns_mapp[(v,u)] = 1/w
            graph[u].append((v,w))
            graph[v].append((u,1/w))
        
        
        def dfs(start,curr,parent,product):
            if (start,curr) in eqns_mapp:
                #contradiction
                if eqns_mapp[(start,curr)] != product:
                    return False
                return True
            if (curr,start) in eqns_mapp:
                if eqns_mapp[(curr,start)] != product:
                    return False
                return True
            for neigh,weight in graph[curr]:
                if neigh == parent:
                    continue
                if not dfs(start,neigh,curr,product*weight):
                    return False
            
            return True
        
        for node in graph:
            if not dfs(node,node,None,1):
                return True
        
        return False
    
#dfs, 
#endition condition is that nodes computed value is within tolerance
class Solution:
    def checkContradictions(self, equations: List[List[str]], values: List[float]) -> bool:
        '''
        convert equations to graph
            a to b has weight 2
            b to a has weight 1/w
        
        dfs from any univisted node, mark value as 1
        if we reach anode that already has a value, check if the xisting value is the same as we computed so far
        if not its a contraction
        '''
        graph = defaultdict(list)
        N = len(equations)
        
        for i in range(N):
            u,v = equations[i]
            w = values[i]
            graph[u].append((v,w))
            graph[v].append((u,1/w))
        
        visited = {}
        for node in graph:
            if node not in visited:
                if self.dfs(node,visited,graph,1.0):
                    return True
        return False
        
    
    def dfs(self,node,visited,graph,path):
        if node not in visited:
            visited[node] = path
            for neigh,weight in graph[node]:
                if self.dfs(neigh,visited,graph,path/weight):
                    return True
        
        #return in tolerance
        return abs(visited[node] - path) >= 0.000001
    
#union find



##################################################
# 2181. Merge Nodes in Between Zeros
#  04JUL24
##################################################
#no dummy head
#two pass though
#need to remove last zero
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        two pointers, if we are at a node and its zero, advance it and accumlate the sum,
        then move reconnect its next  pointer
        need dummy node
        '''
        curr = head
        
        
        while curr != None:
            if curr.val != 0:
                prev = curr
                curr = curr.next
            else:
                fast = curr.next
                while fast != None and fast.val != 0:
                    curr.val += fast.val
                    fast = fast.next
                
                curr.next = fast
                prev = curr
                curr = fast
        #trim last zero
        prev = None
        curr = head
        
        while curr != None:
            if curr.val == 0:
                prev.next = None
                break
            prev = curr
            curr = curr.next
        return head
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        two pointers, if we are at a node and its zero, advance it and accumlate the sum,
        then move reconnect its next  pointer
        need dummy node
        '''

        curr = head
        
        while curr != None:
            #leave it
            if curr.val != 0:
                prev = curr
                curr = curr.next
            #if we are at a zero
            else:
                fast = curr.next
                while fast != None and fast.val != 0:
                    curr.val += fast.val
                    fast = fast.next
                
                if fast.next == None:
                    curr.next = None
                    break
                curr.next = fast
                curr = fast
        return head
                
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        omg first and ending nodes will always be zero
        '''
        fast  = head.next
        slow = head
        
        while fast:
            #if its not a zero, accumlate
            if fast.val:
                slow.val += fast.val
            elif fast.next:
                slow = slow.next
                slow.val = 0
            else:
                slow.next = None
            
            fast = fast.next
        
        return head

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        recursion, finish a block and return the sum
        '''
        
        def rec(node):
            node = node.next
            if not node:
                return node
            curr = node
            curr_sum = 0
            while curr.val != 0:
                curr_sum += curr.val
                curr = curr.next
            
            #this node's val becomes sum
            node.val = curr_sum
            #recurse on the next part
            node.next = rec(curr)
            return node
        
        return rec(head)         
            
            
#############################################################################
# 2058. Find the Minimum and Maximum Number of Nodes Between Critical Points
# 05JUL24
#############################################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        '''
        just keep track of positions of each node,
        then update min and max distance between any two critical points
        to check critical points, i need to have prev and a next
        max distance is the distance between the first and last cp
        min is just the min between the two points
        '''
        min_dist = float('inf')

        first_cp = float('inf') 
        last_cp = float('-inf')
        last_seen_cp = -1
        curr = head
        prev = None
        i = 0
        
        while curr:
            #if there is a critical point
            if prev and curr.next:
                #maximum
                if prev.val < curr.val and curr.val > curr.next.val:
                    first_cp = min(first_cp,i)
                    last_cp = max(last_cp,i)
                    if last_seen_cp == -1:
                        last_seen_cp = i
                    else:
                        min_dist = min(min_dist, i - last_seen_cp)
                        last_seen_cp = i
                        
                elif prev.val > curr.val and curr.val < curr.next.val:
                    first_cp = min(first_cp,i)
                    last_cp = max(last_cp,i)
                    if last_seen_cp == -1:
                        last_seen_cp = i
                    else:
                        min_dist = min(min_dist, i - last_seen_cp)
                        last_seen_cp = i
            prev = curr
            curr = curr.next
            i += 1
        
        #now compute
        #if onyl on cp
        if first_cp == last_cp:
            return [-1,-1]
        
        #not criptical point
        if first_cp == float('inf') or last_cp == float('-inf'):
            return [-1,-1]
        return [min_dist, last_cp - first_cp]
        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def nodesBetweenCriticalPoints(self, head: Optional[ListNode]) -> List[int]:
        '''
        clearning up code
        removing second conditional
        '''
        min_dist = float('inf')

        first_cp = float('inf') 
        last_cp = float('-inf')
        last_seen_cp = -1
        curr = head
        prev = None
        i = 0
        
        while curr:
            #if there is a critical point
            if prev and curr.next:
                #maximum
                if (prev.val < curr.val and curr.val > curr.next.val) or (prev.val > curr.val and curr.val < curr.next.val):
                    first_cp = min(first_cp,i)
                    last_cp = max(last_cp,i)
                    if last_seen_cp == -1:
                        last_seen_cp = i
                    else:
                        min_dist = min(min_dist, i - last_seen_cp)
                        last_seen_cp = i
            prev = curr
            curr = curr.next
            i += 1
        
        #now compute
        #if onyl on cp
        if first_cp == last_cp:
            return [-1,-1]
        
        #not criptical point
        if first_cp == float('inf') or last_cp == float('-inf'):
            return [-1,-1]
        return [min_dist, last_cp - first_cp]
        
#################################################
# 2582. Pass the Pillow
# 06JUL24 
################################################     
class Solution:
    def passThePillow(self, n: int, time: int) -> int:
        '''
        i cant use an array of 2*n people
        then return at index = time % len(arrayy)
        the array should be
        1,2,3,4,3,2
        its a circular array
        '''
        people = [i for i in range(1,n+1)] + [i for i in range(n-1,1,-1)]
        print(people)
        return people[time % len(people)]

#constant space
class Solution:
    def passThePillow(self, n: int, time: int) -> int:
        '''
        constant space
        
        if time < n
            return time + 1
            1,2,3,4,5
            
        if time > n, are we in the forward regime or reverse regime?
            complete forward pass takes, n-1 time
            1,2,3,4
            complete reverse pass takes another n-1 time
            4,3,2,1
        
        so every (n-1) seconds a pass is completed
        
        '''
        if time < n:
            return time + 1
        
        passes,position = divmod(time,n-1)
        #forward pass
        if passes % 2 == 0:
            return position + 1
        #reverse pass
        else:
            return n - position
        
#simulte,
class Solution:
    def passThePillow(self, n: int, time: int) -> int:
        current_pillow_position = 1
        current_time = 0
        direction = 1
        while current_time < time:
            if 0 < current_pillow_position + direction <= n:
                current_pillow_position += direction
                current_time += 1
            else:
                # Reverse the direction if the next position is out of bounds
                direction *= -1
        return current_pillow_position
    
########################################
# 1518. Water Bottles (REVISTED)
# 07JUL24
########################################
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        '''
        we can drink d bottles, and go down to d / numExhange
        try all possible drinks
        '''
        memo = {}
        def dp(numBottles,numExchange):
            #no bottles, no drinks
            if numBottles == 0:
                return 0
            if numBottles in memo:
                return memo[numBottles]
            ans = 0
            for drink in range(1,numBottles+1):
                new_bottles = numBottles - drink + (drink // numExchange)
                ans = max(ans, drink + dp(new_bottles,numExchange))
            
            memo[numBottles] = ans
            return ans
        
        return dp(numBottles,numExchange)
    

class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        '''
        need to keep track of empty and full
        '''
        full = numBottles
        empty = 0
        max_drank = 0
        
        while full > 0:
            #drink all full ones
            max_drank += full
            #exchange
            new_full,left_over_empty = divmod(full + empty, numExchange)
            empty = left_over_empty
            full = new_full
        
        return max_drank
    
##########################################################
# 1836. Remove Duplicates From an Unsorted Linked List
# 08JUL24
##########################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
        '''
        keep count map and only include nodes that are unique, two pass
        '''
        counts = Counter()
        curr = head
        while curr != None:
            counts[curr.val] += 1
            curr = curr.next
        
        dummy = ListNode(-1)
        prev = dummy
        curr = head
        
        while curr != None:
            if counts[curr.val] == 1:
                prev.next = ListNode(curr.val)
                prev = prev.next
            curr = curr.next
        
        return dummy.next
    
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
        '''
        keep count map and only include nodes that are unique, two pass
        '''
        counts = Counter()
        curr = head
        while curr != None:
            counts[curr.val] += 1
            curr = curr.next
        
        dummy = ListNode(-1)
        dummy.next = head
        prev = dummy
        curr = head
        
        while curr:
            if counts[curr.val] > 1:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        
        return dummy.next
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
        '''
        keep count map and only include nodes that are unique, two pass
        '''
        counts = Counter()
        curr = head
        while curr != None:
            counts[curr.val] += 1
            curr = curr.next
            
        return self.delete(head,counts)
    
    def delete(self, node, counts):
        if not node:
            return None
        
        removed = self.delete(node.next,counts)
        node.next = removed
        
        if counts[node.val] > 1:
            return removed
        return node
        
        
        
        
        