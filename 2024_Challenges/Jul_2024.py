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
class DSU:
    def __init__(self,):
        self.parents = {}
        self.values = {}
    
    def add(self,x):
        if x not in self.parents:
            self.parents[x] = x
            self.values[x] = 1
            
    def find(self,x):
        if self.parents[x] != x:
            self.parents[x], val = self.find(self.parents[x])
            #not really like path compression
            self.values[x] *= val
        
        return self.parents[x], self.values[x]
    
    def union(self,x,y,val):
        (x_par,x_val) = self.find(x)
        (y_par,y_val) = self.find(y)
        
        self.parents[x_par] = y_par
        self.values[x_par] *= (y_val/x_val)*val
        
class Solution:
    def checkContradictions(self, equations: List[List[str]], values: List[float]) -> bool:
        '''
        we can use union find, to detect contractions, but slighlty modified
        keep nodes value at 1, and update when there is a relationshop
        idea:
            knowing a/b, a/c, a/d, this is all just one group
            a/root, b/root, c/root, d/root
        
        say for example, if we want to know b/d, we can just do
        b/root *(root/d) = b/d
        initially all roots are just 1
        idea is that if we have already found a value, we can check in constant time using union find
        need to initialize the struct with parents to itself, and values as 1
        '''
        #initilaize
        dsu = DSU()
        for u,v in equations:
            dsu.add(u)
            dsu.add(v)
        
        for (x,y),val in zip(equations,values):
            x_par,x_val = dsu.find(x)
            y_par,y_val = dsu.find(y)
            
            #point to same root, checl
            if x_par == y_par:
                to_check = x_val/y_val
                if abs(to_check - val) > 0.00001:
                    return True
            else:
                dsu.union(x,y,val)
            
        return False


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
    
#recursive 
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
    
############################################
# 3100. Water Bottles II
# 08JUL24
############################################
#top down dp works just fine
class Solution:
    def maxBottlesDrunk(self, numBottles: int, numExchange: int) -> int:
        '''
        try dp first
        '''
        memo = {}
        def dp(full,empty,k):
            #no fulls, check if we can make more
            if full == 0:
                if empty < k:
                    #cant drink anymore
                    return 0
                return dp(full + 1, empty - k, k+1)
            if (full,empty,k) in memo:
                return memo[(full,empty,k)]
            #get more fulls from empty
            exchange = 0
            if empty >= k:
                exchange = dp(full + 1, empty - k, k+1)
            
            for drink in range(1,full+1):
                exchange = max(exchange, drink + dp(full - drink, empty + drink,k))
            
            memo[(full,empty,k)] = exchange
            return exchange
        
        return dp(numBottles,0,numExchange)
        
class Solution:
    def maxBottlesDrunk(self, numBottles: int, numExchange: int) -> int:
        '''
        keep drinking numExchange
        '''
        ans = 0
        while numBottles >= numExchange:
            ans += numExchange
            numBottles -= numExchange
            #dont forget we gain a bottle
            numBottles += 1
            numExchange += 1
            
        return ans + numBottles #dont forget to add the remanining
        
#constant space, mathzzz
'''
Let's formulate the problem setup in mathematical terms. Define the following:

$$N$$ = numBottles
$$d$$ = numExchange
$$x$$ = number of exchanges possible (earned bottles)
Hence, the total number of available bottles is:
$$N + x$$

The total number of bottles exchanged can be expressed as the sum:
$$\sum_{i=0}^{x - 1} (d + i)$$

Given that it's impossible to exchange more bottles than we possess, we derive the inequality:
$$\sum_{i=0}^{x - 1} (d + i) \leq N + x$$
$$\sum_{i=0}^{x - 1} d + \sum_{i=1}^{x - 1} i \leq N + x$$
$$xd + \frac{x(x-1)}{2} \leq N + x$$
$$x^2 + (2d-3)x - 2N \leq 0$$

We aim to find the maximal positive $$x$$ that satisfies this inequality. The left side forms a parabola with branches pointing upwards, so we seek the larger root, rounded down.

Approach
To solve the equation $$x^2 + (2d-3)x - 2N = 0$$, we use the discriminant:
$$D = (2d-3)^2 + 8N$$
$$x = \Big\lceil{\frac{-(2d-3) + \sqrt{D}}{2}}\Big\rceil$$
'''

from math import sqrt

class Solution:
    def maxBottlesDrunk(self, numBottles: int, numExchange: int) -> int:
        numBottles -= 1
        D = (2 * numExchange - 3)**2 + 8 * numBottles
        res = int((-(2 * numExchange - 3) + sqrt(D)) / 2)
        return numBottles + res + 1

#######################################
# 1701. Average Waiting Time
# 09JUL24
########################################
class Solution:
    def averageWaitingTime(self, customers: List[List[int]]) -> float:
        '''
        cheif can only prepare one meal at a time
        arrives, then cooks -> waititime time is cooked_time - arrived
        order of arrival is given in the interal input
        just comute total waiting time and get average
        the issue is that customers could arrive at the same time, and they'll be waiting for all other customers to be done
        '''
        N = len(customers)
        total_wait_time = 0
        earliest_available_time = 0
        for arrival,prep in customers:
            #chef is available
            if earliest_available_time < arrival:
                total_wait_time += prep
                earliest_available_time = arrival + prep
            else:
                total_wait_time += (earliest_available_time - arrival) + prep
                earliest_available_time += prep
        
        
        return total_wait_time / N
    
class Solution:
    def averageWaitingTime(self, customers: List[List[int]]) -> float:
        '''
        we can keep update using max
        '''
        N = len(customers)
        total_wait_time = 0
        earliest_available_time = 0
        for arrival,prep in customers:
            earliest_available_time = max(earliest_available_time, arrival) + prep
            total_wait_time += earliest_available_time - arrival
        
        return total_wait_time / N
    
#################################################
# 1752. Check if Array Is Sorted and Rotated
# 10JUL24
#################################################
class Solution:
    def check(self, nums: List[int]) -> bool:
        '''
        try all rotations and check that its sorted
        '''
        N = len(nums)
        for i in range(N+1):
            rotated = self.rotate(nums,i)
            if self.isSorted(rotated):
                return True
        return False
    
    def rotate(self,nums,k):
        return nums[k:] + nums[:k]
    
    def isSorted(self,nums):
        N = len(nums)
        for i in range(1,N):
            if nums[i] < nums[i-1]:
                return False
        return True
        
class Solution:
    def check(self, nums: List[int]) -> bool:
        '''
        there can only be one peak
        only works becasue rotation from 0 is allowed
        '''
        peaks = 0
        for i in range(len(nums)): 
            if nums[i-1] > nums[i]: peaks += 1
        return peaks <= 1
    
class Solution:
    def check(self, nums: List[int]) -> bool:
        cnt = 0
        for i in range(1, len(nums)): 
            if nums[i-1] > nums[i]: cnt += 1
        return cnt == 0 or cnt == 1 and nums[-1] <= nums[0]

############################################################
# 1190. Reverse Substrings Between Each Pair of Parentheses 
# 11JUL24
############################################################
class Solution:
    def reverseParentheses(self, s: str) -> str:
        '''
        when we hit a close, we need to revese the chars in the stack, up until thw first opening
        '''
        stack = []
        for ch in s:
            if ch == ')':
                on_stack = []
                while stack and stack[-1] != '(':
                    on_stack.append(stack.pop())
                
                #matching opening
                stack.pop()
                stack.extend(on_stack)
            else:
                stack.append(ch)
        
        return "".join(stack)
                
class Solution:
    def reverseParentheses(self, s: str) -> str:
        '''
        instead of ferrying with list and stack, we can use one q to store the lengths of our result so far
        each time we hit an opening '(' we push the current length
        when we encounter a ')' we pop the lat index from the stack
        then we can reverse from the last '(' to the current ')'
        
        '''
        open_brackets_idxs = []
        result = []
        
        for ch in s:
            #opening, mark starting of reverse
            if ch == '(':
                open_brackets_idxs.append(len(result))
            #clsoing,reverse up to heare
            elif ch == ')':
                starting = open_brackets_idxs.pop()
                result[starting:] = result[starting:][::-1]
            else:
                result.append(ch)
        
        return "".join(result)
    
#wormhole teleportation technique, LMAOOO
class Solution:
    def reverseParentheses(self, s: str) -> str:
        '''
        O(N) wormhole technique
        firt pass:
            use stack to pair up matchind parantheses idxs with each other using stack
        
        second pass:
            travse string but keep pointers into string and direction
            if ch is in '()' then move to its matching parantehse and swap direction
            otherwise add to char to the result
        '''
        N = len(s)
        open_ps_stack = []
        pairs = [0]*N
        
        for i,ch in enumerate(s):
            if ch == '(':
                open_ps_stack.append(i)
            elif ch == ')':
                j = open_ps_stack.pop()
                pairs[i] = j
                pairs[j] = i
            
        ans = []
        i = 0
        direction = 1
        while i < N:
            if s[i] in '()':
                i = pairs[i]
                direction *= -1
            else:
                ans.append(s[i])
            i += direction
        
        return "".join(ans)
    
################################################
# 1717. Maximum Score From Removing Substrings
# 12JUL24
################################################
class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        '''
        we can only remove 'ab' from s, if 'ab' isn't in there we can't gain any points
        does order matter in this removal? probably not, constraints are 10**5
        length 2
            ab, ba
        length 4
            abab, baba, abba baab
        if ab and ab (or wtv other pair combo) is sperated by a substring not a pair combo, 
        then removing the pair combon wont bring them together
        
        take the more optimal substring
        cdbcbbaaabab
        
        cdbc
        
        remove high priorty pair first, then low priority
        bbaaabab
        
        
        '''
        high_score, high_code = None,None
        low_score, low_code = None,None
        
        if x >= y:
            high_score,high_code = x, 'ab'
            low_score, low_code =  y, 'ba'
        else:
            high_score,high_code = y, 'ba'
            low_score, low_code =  x, 'ab'
            
        #do high first
        stack = []
        score = 0
        for ch in s:
            if stack and stack[-1] == high_code[0] and ch == high_code[1]:
                stack.pop()
                score += high_score
            else:
                stack.append(ch)
        
        
        #now low score, but do on removed characters from high code
        s = stack
        stack = []
        for ch in s:
            if stack and stack[-1] == low_code[0] and ch == low_code[1]:
                stack.pop()
                score += low_score
            else:
                stack.append(ch)
        
        return score
    
#consolidate 
class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        
        codes = [('a','b',x), ('b','a',y)]
        if x < y:
            codes[0],codes[1] = codes[1],codes[0]
        
        ans = 0
        for left,right,score in codes:
            stack = []
            for ch in s:
                if stack and stack[-1] == left and ch == right:
                    stack.pop()
                    ans += score
                else:
                    stack.append(ch)
            
            #swap
            s = stack
        
        return ans
    
#greedy, stack and counting
class Solution:
    def maximumGain(self, s: str, x: int, y: int) -> int:
        '''
        another way is to removal, and find the different in lengths between removals
        the difference // 2 is multiplied by the corresponding score
        '''
        score = 0
        high = 'ab' if x > y else 'ba'
        low = 'ba' if high == 'ab' else 'ab'
        
        first_pass = self.remove_target(s,high)
        score += ((len(s) - len(first_pass))//2)*max(x,y)
        
        second_pass = self.remove_target(first_pass,low)
        score += ((len(first_pass) - len(second_pass))//2)*min(x,y)
        
        return score
    
    def remove_target(self,s,target_pair):
        stack = []
        for ch in s:
            if stack and stack[-1] == target_pair[0] and ch == target_pair[1]:
                stack.pop()
            else:
                stack.append(ch)
        
        return "".join(stack)
    
#########################################
# 2751. Robot Collisions
# 13JUL24
#########################################
class Solution:
    def survivedRobotsHealths(self, positions: List[int], healths: List[int], directions: str) -> List[int]:
        '''
        rules:
            if two robots collide and have same health:
                remove robot with lower health and larger health -= 1
                surviving robot continues in same direction
            if both robots have == health:
                remove them from the line
        only include them if they survive
        make sure to return the array in the initial order they were given in positions
        simulating would take too long
        
        theres only one way it can collide, stack robot must be doing right, the other one must be going left
        '''
        #first sort them
        sortedEntries = []
        for p,h,d in zip(positions,healths,directions):
            entry = [p,h,d]
            sortedEntries.append(entry)
        
        sortedEntries.sort()
        stack = [sortedEntries[0]]
        for i in range(1,len(sortedEntries)):
            curr_robot = sortedEntries[i]
            while stack and stack[-1][2] == 'R' and curr_robot[2] == 'L':
                #equal healths
                if stack[-1][1] == curr_robot[1]:
                    stack.pop()
                    break
                #curr robot can keep going left
                elif curr_robot[1] > stack[-1][1]:
                    stack.pop()
                    curr_robot[1] -= 1
                #curr robot cant beat top of stack robot
                elif stack[-1][1] > curr_robot[1]:
                    stack[-1][1] -= 1
                    break
            else:
                stack.append(curr_robot)
        
        #rebuild ans
        mapp = {}
        for p,h,d in stack:
            mapp[p] = h
        
        ans = []
        
        for p in positions:
            if p in mapp:
                ans.append(mapp[p])
        
        return ans
            
#############################################
# 2196. Create Binary Tree From Descriptions
# 15JUL24
#############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
        '''
        values are unique, we need to build the graph and start with the root
        the build the tree, left to right
        '''
        graph = defaultdict(list)
        indegree = defaultdict(int)
        
        for parent,child,direction in descriptions:
            graph[parent].append((child,direction))
            if parent not in indegree:
                indegree[parent] = 0
            indegree[child] += 1
            
        #find root
        root = None
        for node,count in indegree.items():
            if count == 0:
                root = TreeNode(node)
                break
                
        curr = root
        q = deque([curr])
        while q:
            node = q.popleft()
            children = graph[node.val]
            #sort
            children.sort(key = lambda x: -x[1])
            for v,direction in children:
                if direction == 1:
                    left = TreeNode(v)
                    node.left = left
                    q.append(left)
                if direction == 0:
                    right = TreeNode(v)
                    node.right = right
                    q.append(right)
        return curr
        
#no need to sort
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def createBinaryTree(self, descriptions: List[List[int]]) -> Optional[TreeNode]:
        '''
        values are unique, we need to build the graph and start with the root
        the build the tree, left to right
        using dfs
        '''
        graph = defaultdict(list)
        indegree = defaultdict(int)
        
        for parent,child,direction in descriptions:
            graph[parent].append((child,direction))
            if parent not in indegree:
                indegree[parent] = 0
            indegree[child] += 1
            
        #find root
        root = None
        for node,count in indegree.items():
            if count == 0:
                root = node
                break
        
        def build(val,graph):
            node = TreeNode(val)
            for child,is_left in graph[val]:
                #we dont need to sort
                if is_left == 1:
                    node.left = build(child,graph)
                else:
                    node.right = build(child,graph)
            
            return node
        
        
        return build(root,graph)
    
###################################################################
# 2096. Step-By-Step Directions From a Binary Tree Node to Another
# 16JUL24
###################################################################
#yessss
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        '''
        find LCA for start value and end value
        need reverse directions from start to LCA, and forward directions from lcs to dest
        '''
        lca = self.lca(root,startValue,destValue)
        _,lca_to_s = self.getPath(lca,startValue)
        _,lca_to_d = self.getPath(lca,destValue)

        return "U"*len(lca_to_s) + lca_to_d
    
    def lca(self,root,p,q):
        if not root:
            return None
        if root.val == p or root.val == q:
            return root
        left = self.lca(root.left,p,q)
        right = self.lca(root.right,p,q)
        #ans is root
        if left != None and right != None:
            return root
        #get left of right acnestors
        elif left:
            return left
        return right

    def getPath(self,root,val):
        if not root:
            return [False,""]
        if root.val == val:
            return [True, ""]
        found_left,path_left = self.getPath(root.left,val)
        found_right,path_right = self.getPath(root.right,val)
        if found_left:
            return [True,"L"+path_left]
        elif found_right:
            return [True,"R"+path_right]
        return [False, ""]
        
#converting to graph
class Solution:
    def getDirections(
        self, root: TreeNode, startValue: int, destValue: int
    ) -> str:
        # Map to store parent nodes
        parent_map = {}

        # Find the start node and populate parent map
        start_node = self._find_start_node(root, startValue)
        self._populate_parent_map(root, parent_map)

        # Perform BFS to find the path
        q = deque([start_node])
        visited_nodes = set()
        # Key: next node, Value: <current node, direction>
        path_tracker = {}
        visited_nodes.add(start_node)

        while q:
            current_element = q.popleft()

            # If destination is reached, return the path
            if current_element.val == destValue:
                return self._backtrack_path(current_element, path_tracker)

            # Check and add parent node
            if current_element.val in parent_map:
                parent_node = parent_map[current_element.val]
                if parent_node not in visited_nodes:
                    q.append(parent_node)
                    path_tracker[parent_node] = (current_element, "U")
                    visited_nodes.add(parent_node)

            # Check and add left child
            if (
                current_element.left
                and current_element.left not in visited_nodes
            ):
                q.append(current_element.left)
                path_tracker[current_element.left] = (current_element, "L")
                visited_nodes.add(current_element.left)

            # Check and add right child
            if (
                current_element.right
                and current_element.right not in visited_nodes
            ):
                q.append(current_element.right)
                path_tracker[current_element.right] = (current_element, "R")
                visited_nodes.add(current_element.right)

        # This line should never be reached if the tree is valid
        return ""

    def _backtrack_path(self, node, path_tracker):
        path = []
        # Construct the path
        while node in path_tracker:
            # Add the directions in reverse order and move on to the previous node
            path.append(path_tracker[node][1])
            node = path_tracker[node][0]
        path.reverse()
        return "".join(path)

    def _populate_parent_map(self, node, parent_map):
        if not node:
            return

        # Add children to the map and recurse further
        if node.left:
            parent_map[node.left.val] = node
            self._populate_parent_map(node.left, parent_map)

        if node.right:
            parent_map[node.right.val] = node
            self._populate_parent_map(node.right, parent_map)

    def _find_start_node(self, node, start_value):
        if not node:
            return None

        if node.val == start_value:
            return node

        left_result = self._find_start_node(node.left, start_value)

        # If left subtree returns a node, it must be StartNode. Return it
        # Otherwise, return whatever is returned by right subtree.
        if left_result:
            return left_result
        return self._find_start_node(node.right, start_value)
    
#find longest common prefix in paths
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        '''
        instead of fiding the LCA, we can directly find the full pahts from root to both startValue and destValue
        then trim off the common parts, then reverse the directions from root to startValue and replace them with U
        for finding paths, we can use backtracking
        '''
        start_path = []
        dest_path = []
        
        self.getPath(root,startValue,start_path)
        self.getPath(root,destValue,dest_path)
        
        #find longest common prefix
        i = 0
        while i < len(start_path) and i < len(dest_path) and start_path[i] == dest_path[i]:
            i += 1
        
        
        return 'U'*(len(start_path) - i)+"".join(dest_path[i:])
    
    def getPath(self, node, target,path):
        if not node:
            return False
        if node.val == target:
            return True
        path.append('L')
        if self.getPath(node.left,target,path):
            return True
        path.pop()
        
        #try going right
        path.append('R')
        if self.getPath(node.right,target,path):
            return True
        path.pop()
        
        return False
    
##################################################
# 1110. Delete Nodes And Return Forest (REVISTED)
# 17JUL24
#################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        '''
        keep global forest, and if we need to delete, add its children
        as we recurse, we need to delete in place!
        '''
        forest = []
        to_delete = set(to_delete)
        
        def rec(node,to_delete):
            if not node:
                return node
            
            #first delete on the subtrees
            node.left = rec(node.left,to_delete)
            node.right = rec(node.right,to_delete)
            if node.val in to_delete:
                #add children
                if node.left:
                    forest.append(node.left)
                if node.right:
                    forest.append(node.right)
                #delete current node
                return None
            
            return node
    
        root = rec(root,to_delete)
        if root:
            forest.append(root)
        
        return forest

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        '''
        we can do bfs, check if we need to delte node, if we need to delete, set children to non
        '''
        if not root:
            return []
        
        to_delete = set(to_delete)
        forest = []
        
        q = deque([root])
        
        while q:
            curr = q.popleft()
            if curr.left:
                q.append(curr.left)
                if curr.left.val in to_delete:
                    curr.left = None
            
            if curr.right:
                q.append(curr.right)
                if curr.right.val in to_delete:
                    curr.right = None
            
            if curr.val in to_delete:
                if curr.left:
                    forest.append(curr.left)
                if curr.right:
                    forest.append(curr.right)
        
        if root.val not in to_delete:
            forest.append(root)
        
        return forest

#############################################
# 1530. Number of Good Leaf Nodes Pairs
# 18JUL24
#############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        '''
        there are 2**10 nodes, 1024, if i had the graph, i could just do pairwise dists for each
        fuck, values won't be unique, need to mapp nodes to other nodes instead
        oh fuck its shortes path
        make sure to acutally start at the leaf nodes
        '''
        graph = defaultdict(list)
        degree = defaultdict(int)
        leaf_nodes = []
        self.populate(root,None,graph,degree,leaf_nodes)
        #print([v.val for v in leaf_nodes])
        #for k,v in graph.items():
        #    print(k.val, [x.val for x in v ])
        #dfs on each leaf node
        ans = 0
        for l in leaf_nodes:
            ans += self.dfs(l,None,graph,distance)
        
        return ans // 2
    
    def populate(self,node,parent,graph,degree,leaf_nodes):
        if not node:
            return
        if not node.left and not node.right:
            leaf_nodes.append(node)
        if parent != None:
            graph[node].append(parent)
            graph[parent].append(node)
            degree[node] += degree.get(node,0) + 1
            degree[parent] += degree.get(parent,0) + 1
        
        self.populate(node.left,node,graph,degree,leaf_nodes)
        self.populate(node.right,node,graph,degree,leaf_nodes)
    
    def dfs(self,node,parent,graph,steps):
        if not node:
            return 0
        if steps < 0:
            return 0
        #cant be the first node we start at
        if parent != None and not node.left and not node.right and steps >= 0:
            return 1
        count = 0
        for neigh in graph[node]:
            if neigh != parent:
                count += self.dfs(neigh,node,graph,steps-1)
        return count
            
#bfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        '''
        we can also do bfs
        but we only do bfs in distance stpes
        '''
        graph = defaultdict(list)
        leaf_nodes = set()
        self.populate(root,None,graph,leaf_nodes)
        ans = [0]
        for l in leaf_nodes:
            self.bfs(l,graph,distance,leaf_nodes,ans)
        
        return ans[0] // 2
    
    def populate(self,node,parent,graph,leaf_nodes):
        if not node:
            return
        if not node.left and not node.right:
            leaf_nodes.add(node)
        if parent != None:
            graph[node].append(parent)
            graph[parent].append(node)
        
        self.populate(node.left,node,graph,leaf_nodes)
        self.populate(node.right,node,graph,leaf_nodes)
    
    def bfs(self,node,graph,steps,leaf_nodes,ans):
        q = deque([node])
        seen = set()
        seen.add(node)
        for _ in range(steps+1):
            N = len(q)
            for _ in range(N):
                curr_node = q.popleft()
                if curr_node in leaf_nodes and curr_node != node:
                    ans[0] += 1
                for neigh in graph[curr_node]:
                    if neigh not in seen:
                        q.append(neigh)
                        seen.add(neigh)
            
#lca intuition
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        '''
        post-order traversal
        intuition: the shortest path between any to nodes will always go through their LCA
        for every node n, we consider paths between all pairs of decendatns in the current subtree rooted at n and check if they are wthin the 
        specified distance, since n servers as the LCA for these leaves, these paths are inherntly the shortest
        at each call we need to return the number of leaf nodes and the distance that each leaf node is at -> YIKES
        
        intution 2
        each recursive call return the count of leaf nodes that are a distance d way for all possible values of d
        base case is if leaf node, return distnace 0 and 1 leaf node from current subtree
        
        rec(node) returns the number of leaf nodes that that are a distance d, for each d not more than the given distance
        knowing the number of "leaf" nodes that are some distance d, we can count paths
        
        given some node, the distance of thee shortest leaf node path thropugh node is
            number of leaf nodes rooted at 2 + rec(node.left) + rec(node.right)
            number of pairs is just the product of left and right
            we only count the pairs whose shortest path distance is < our given distance
        
        the next step is to return the counts of leaf nodes for all distances d from t he current node
        we can get this by shifting all the counts returned from left and right by 1
        exmplae, 1 leaf node that is distance 0 from some node X will translate to 1 leaf node taht is a distance 1 from another node Y
        
        '''
        def dp(node,distance):
            #input d <= 10
            if node == None:
                return [0]*12
            elif node.left == None and node.right == None:
                current = [0]*12
                current[0] = 1 #we have one leaf node that is 0 distance away from itselft
                return current
            
            left = dp(node.left,distance)
            right = dp(node.right,distance)
            
            current = [0]*12
            #combine counts from left and rights
            for i in range(10):
                current[i+1] += left[i] + right[i] #counts at each distance d
            #initialize to total number of good leaf nodes pairs from left and right
            current[-1] = left[-1] + right[-1]
            #iterate through possible leaf node distance pairs
            for d1 in range(distance+1):
                for d2 in range(distance + 1):
                    if 2 + d1 + d2 <= distance:
                        current[-1] += left[d1]*right[d2]
            
            return current
        
        return dp(root,distance)[-1]
    
#another way
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPairs(self, root: TreeNode, distance: int) -> int:
        
        res = [0]
        
        #dp returns an array of distances where each d in distances is a  distance of a leafnode from the
        #current subtree we recursed on
        #global count and increment when we have a leaf to leaf node path not more than distance
        def dp(node):
            if not node:
                return [] #no leafs nodes that are distance 1 
            #leaf node, dist 1 away
            if not node.left and not node.right:
                return [1]
            
            left_dists = dp(node.left)
            right_dists = dp(node.right)
            
            #count up pairs that are less than distance
            for l in left_dists:
                for r in right_dists:
                    res[0] += l + r <= distance
            
            new_dists = []
            for d in left_dists + right_dists:
                #we are now one more away
                d += 1
                new_dists.append(d)
            
            return new_dists
        
        dp(root,)
        return res[0]
    

############################################
# 1380. Lucky Numbers in a Matrix (REVISTED)
# 19JUL24
############################################
class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        '''
        there can only be one lucky number
        proff by contraction
        assume we have X and its lucky at (r1,c1)
        and assume we have another Y and its lucky at (r2,c2)
        now let A be (r2,c1)
        if Y is lucky,
        Y < A, (Y is minimum here)
        X > A, (X is maximum here)
        
        but this cannot be, because if Y is also lucky, then there is another number B at (r1,c2)
        Y > B (Y ix max here)
        X < B (X is min here)
        
        so Y > X, contradiction!
        
        first find the min element in reach row, then max them up
        then find the max of each col, and min them up
        if == , then we have our answer
        '''
        rows, cols = len(matrix),len(matrix[0])
        
        r_min_max = float('-inf')
        for r in matrix:
            r_min_max = max(r_min_max,min(r))
        
        c_max_min = float('inf')
        #find max along cols
        for c in zip(*matrix):
            c_max_min = min(c_max_min, max(c))
        
        if r_min_max == c_max_min:
            return [r_min_max]
        return []
    
#################################################################
# 1725. Number Of Rectangles That Can Form The Largest Square
# 19JUL24
#################################################################
class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        '''
        just try making the squares
        '''
        counts = Counter()
        maxLen = float('-inf')
        for l,w in rectangles:
            if l <= w:
                s = l
            else:
                s = w
            maxLen = max(maxLen,s)
            counts[s] += 1
            
        return counts[maxLen]
    

class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        '''
        just try making the squares
        '''
        maxLen = float('-inf')
        for l,w in rectangles:
            if l <= w:
                s = l
            else:
                s = w
            maxLen = max(maxLen,s)
        
        counts = 0
        for l,w in rectangles:
            if l <= w:
                s = l
            else:
                s = w
            counts += s == maxLen
        return counts
    
class Solution:
    def countGoodRectangles(self, rectangles: List[List[int]]) -> int:
        '''
        one pass, keep count and update
        '''
        maxLen = float('-inf')
        counts = 1
        for l,w in rectangles:
            if l <= w:
                s = l
            else:
                s = w
            
            if s > maxLen:
                maxLen = s
                counts = 1
            elif s == maxLen:
                counts += 1
        return counts
    
#######################################################
# 1605. Find Valid Matrix Given Row and Column Sums
# 20JUN24
######################################################
#greedy
class Solution:
    def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
        '''
        its guaranteed to exist
        notice the sum(rowSum) == sum(colSum)
        start with smallest rowSum or colSum
        it works, but idk why
        
        cant place negatives, so sums never go below zero
        we are free to place 0s -> put down limit and fill with 0s
        so at (r,c) what rowSum or colSum fo we use
        if we go over limit, we can build matix
        but what if we put min, put min and decrement
        '''
        rows, cols = len(rowSum),len(colSum)
        ans = [[0]*cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                entry = min(rowSum[i],colSum[j])
                ans[i][j] = entry
                rowSum[i] -= entry
                colSum[j] -= entry
        
        return ans
            
#brute force
class Solution:
    def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
        '''
        place as much as we can at each (i,j) up the the minimum limit base on rows and cols,
        '''
        rows,cols = len(rowSum),len(colSum)
        ans = [[0 for _ in range(cols)] for _ in range(rows)]
        
        currRowSums = [0]*rows
        currColSums =[0]*cols
        
        for i in range(rows):
            for j in range(cols):
                entry = min(rowSum[i] - currRowSums[i], colSum[j] - currColSums[j])
                ans[i][j] = entry
                #tally up
                currRowSums[i] += entry
                currColSums[j] += entry
        
        return ans
    
#space optimized
class Solution:
    def restoreMatrix(self, rowSum: List[int], colSum: List[int]) -> List[List[int]]:
        '''
        effecient traversal is like saddleback search
        If we dont have a rowSum or colSum to contribute to an (i,j) just go on to the next one.
        '''
        rows,cols = len(rowSum),len(colSum)
        ans = [[0 for _ in range(cols)] for _ in range(rows)]
        
        i,j = 0,0
        
        while i < rows and j < cols:
            entry = min(rowSum[i],colSum[j])
            ans[i][j] = entry
            
            rowSum[i] -= entry
            colSum[j] -= entry
            
            if rowSum[i] == 0:
                i += 1
            elif colSum[j] == 0:
                j += 1
        
        return ans
    
############################################
# 2392. Build a Matrix With Conditions
# 21JUL24
#############################################
#right idea
class Solution:
    def buildMatrix(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        '''
        need to build k by k matrix and matrix must follow rowConditions and colConditions
        must use numbers uniqely from 1 to k,
        makes sense to build it from topleft to bottom right
        i need ordering going left to right and order from top to bottom
        top sort, even if i had the ordering i still have to build the matrix
        treat conditions as edges and get top sort
        '''
        topOrderRows = self.topSort(k,rowConditions)
        topOrderCols = self.topSort(k,colConditions)

        #can't do
        if not topOrderRows or not topOrderCols:
            return []
        
        #how on earth to i figure out how to build
        ans = [[0]*k for _ in range(k)]
        
        for i in range(k):
            for j in range(k):
                if topOrderRows[i] == topOrderCols[j]:
                    ans[i][j] = topOrderRows[i]
        return ans
        
    def topSort(self,k,edges):
        #need to check for cycles too
        graph = defaultdict(list)
        indegree = {}
        for u,v in edges:
            if u not in indegree:
                indegree[u] = 0
            if v not in indegree:
                indegree[v] = 0
            
            graph[u].append(v)
            indegree[v] += 1
        
        #start with nodes that have 0 indegree
        q = deque([])
        for node,ind in indegree.items():
            if ind == 0:
                q.append(node)
        visited = set()
        ordering = []
        while q:
            curr = q.popleft()
            visited.add(curr)
            ordering.append(curr)
            for neigh in graph[curr]:
                indegree[neigh] -= 1
                if indegree[neigh] == 0:
                    q.append(neigh)
        
        return ordering
        
#finally
class Solution:
    def buildMatrix(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        '''
        need to build k by k matrix and matrix must follow rowConditions and colConditions
        must use numbers uniqely from 1 to k,
        makes sense to build it from topleft to bottom right
        i need ordering going left to right and order from top to bottom
        top sort, even if i had the ordering i still have to build the matrix
        treat conditions as edges and get top sort
        
        notes for top order
        need to make sure we visit all k rows and all k cols
        if we dont return []
        make sure there isn't a cycle too
        
        how to generate the matrix though
        need to compare the order in rows to the order in cols
        if they match, then that number belongs there
        '''
        topOrderRows = self.topSort(k,rowConditions)
        topOrderCols = self.topSort(k,colConditions)

        #can't do
        if not topOrderRows or not topOrderCols:
            return []
        
        #how on earth to i figure out how to build
        ans = [[0]*k for _ in range(k)]
        

        for i in range(k):
            for j in range(k):
                if topOrderRows[i] == topOrderCols[j]:
                    ans[i][j] = topOrderRows[i]
        return ans
        
    def topSort(self,k,edges):
        #need to check for cycles too
        graph = defaultdict(list)
        indegree = [0]*(k+1)
        for u,v in edges:            
            graph[u].append(v)
            indegree[v] += 1
        
        #start with nodes that have 0 indegree
        q = deque([])
        for i in range(1,k+1):
            if indegree[i] == 0:
                q.append(i)
        visited = set()
        ordering = []
        while q:
            curr = q.popleft()
            visited.add(curr)
            ordering.append(curr)
            for neigh in graph[curr]:
                indegree[neigh] -= 1
                if indegree[neigh] == 0:
                    q.append(neigh)
        if len(visited) != k:
            return []
        return ordering
        
#we can also use dfs and cycle detection to find the topolgical ordering
class Solution:
    def buildMatrix(self, k: int, rowConditions: List[List[int]], colConditions: List[List[int]]) -> List[List[int]]:
        '''
        need to build k by k matrix and matrix must follow rowConditions and colConditions
        must use numbers uniqely from 1 to k,
        makes sense to build it from topleft to bottom right
        i need ordering going left to right and order from top to bottom
        top sort, even if i had the ordering i still have to build the matrix
        treat conditions as edges and get top sort
        
        notes for top order
        need to make sure we visit all k rows and all k cols
        if we dont return []
        make sure there isn't a cycle too
        
        how to generate the matrix though
        need to compare the order in rows to the order in cols
        if they match, then that number belongs there
        '''
        topOrderRows = self.topSort(k,rowConditions)
        topOrderCols = self.topSort(k,colConditions)

        #can't do
        if not topOrderRows or not topOrderCols:
            return []
        
        #how on earth to i figure out how to build
        ans = [[0]*k for _ in range(k)]
        
        for i in range(k):
            for j in range(k):
                if topOrderRows[i] == topOrderCols[j]:
                    ans[i][j] = topOrderRows[i]
        return ans
        
    def topSort(self,k,edges):
        #need to check for cycles too
        graph = defaultdict(list)
        ordering = []
        visited = set()
        for u,v in edges:            
            graph[u].append(v)
        
        for i in range(1,k+1):
            if i not in visited:
                if self.has_cycle(i,graph,visited,ordering):
                    return []
        
        return ordering[::-1]
            
    
    def has_cycle(self,node,graph,visited,ordering):
        visited.add(node)
        for neigh in graph[node]:
            if neigh not in visited:
                if self.has_cycle(neigh,graph,visited,ordering):
                    return True


        ordering.append(node)
        return False
        
################################################################
# 2093. Minimum Cost to Reach City With Discounts (REVISITED)
# 22JUL24
################################################################
class Solution:
    def minimumCost(self, n: int, highways: List[List[int]], discounts: int) -> int:
        '''
        need to use djikstras ssp, but in addition to the edge weight for a toll
        we need to keep track of the number of discounts
        state is min distatance to each vertex with d discounts, when taking the smallest edge and we have a discount apply it
        apply and dont apply
        
        need 2d array for dists, node and number if discounts
        '''
        
        graph = defaultdict(list)
        for u,v,w in highways:
            graph[u].append((v,w))
            graph[v].append((u,w))
        
        #like in djisktras where state is node, here state is (node,curr_number_discounts)
        dists = [[float('inf')]*(discounts + 1) for _ in range(n)]
        dists[0][discounts] = 0
        pq = [(0,discounts,0)] #entry is (min_dist,discounts left, node)
        #can keep visited to prune if we want too
        visited = set()

        while pq:
            min_dist_so_far, curr_discounts, node = heapq.heappop(pq)
            #if we cant improve
            if dists[node][curr_discounts] < min_dist_so_far:
                continue
            visited.add((node,curr_discounts))
            for neigh,dist_to in graph[node]:
                if (neigh,curr_discounts) in visited:
                    continue
                #if we can apply a discount
                if curr_discounts > 0:
                    new_dist = min_dist_so_far + dist_to // 2
                    if dists[neigh][curr_discounts - 1] > new_dist:
                        dists[neigh][curr_discounts - 1] = new_dist
                        heapq.heappush(pq, (new_dist,curr_discounts - 1, neigh))
                #no discount
                if curr_discounts == 0:
                    new_dist = min_dist_so_far + dist_to
                    if dists[neigh][curr_discounts] > new_dist:
                        dists[neigh][curr_discounts] = new_dist
                        heapq.heappush(pq, (new_dist,curr_discounts, neigh))
        
        min_cost = min(dists[n-1])
        return min_cost if min_cost != float('inf') else -1
    
#bleaghhh
class Solution:
    def minimumCost(self, n: int, highways: List[List[int]], discounts: int) -> int:
        # Construct the graph from the given highways array
        graph = [[] for _ in range(n)]
        for highway in highways:
            u, v, toll = highway
            graph[u].append((v, toll))
            graph[v].append((u, toll))

        # Min-heap priority queue to store tuples of (cost, city, discounts used)
        pq = [(0, 0, 0)]  # Start from city 0 with cost 0 and 0 discounts used

        # 2D array to track minimum distance to each city with a given number of discounts used
        dist = [[float("inf")] * (discounts + 1) for _ in range(n)]
        dist[0][0] = 0

        visited = [[False] * (discounts + 1) for _ in range(n)]

        while pq:
            current_cost, city, discounts_used = heapq.heappop(pq)

            # Skip processing if already visited with the same number of discounts used
            if visited[city][discounts_used]:
                continue
            visited[city][discounts_used] = True

            # Explore all neighbors of the current city
            for neighbor, toll in graph[city]:

                # Case 1: Move to the neighbor without using a discount
                if current_cost + toll < dist[neighbor][discounts_used]:
                    dist[neighbor][discounts_used] = current_cost + toll
                    heapq.heappush(pq,(dist[neighbor][discounts_used],neighbor,discounts_used,),) #interesting bit with the commas here

                # Case 2: Move to the neighbor using a discount if available
                if discounts_used < discounts:
                    new_cost_with_discount = current_cost + toll // 2
                    if (new_cost_with_discount < dist[neighbor][discounts_used + 1]):
                        dist[neighbor][discounts_used + 1] = new_cost_with_discount
                        heapq.heappush(pq,(new_cost_with_discount,neighbor,discounts_used + 1,),)

        # Find the minimum cost to reach city n-1 with any number of discounts used
        min_cost = min(dist[n - 1])
        return -1 if min_cost == float("inf") else min_cost
    
######################################################
# 1636. Sort Array by Increasing Frequency (REVISTED)
# 23JUL24
######################################################
class Solution:
    def frequencySort(self, nums: List[int]) -> List[int]:
        '''
        counting sort, nums can only be in range -100 to 100
        '''
        #get the frequency array
        counts = [0]*201
        for num in nums:
            counts[num + 100] += 1
        
        #get count of counts
        count_of_counts = [0]*(max(counts) + 1)
        #if nums have same frequence, we need to bucket them
        #i.e for each count, store the numbers that have that count
        bucket_of_counts = defaultdict(list)
        
        #need array in increasing order of frequency, if ties, sort values decreasingly
        for i in range(len(counts)-1,-1,-1):
            if counts[i]:
                #print(i-k,counts[i])
                count_of_counts[counts[i]] += 1
                #this essentially puts the largest values first in the bucket for shared counts
                bucket_of_counts[counts[i]].append(i - 100)
        
        print(bucket_of_counts)
        print(count_of_counts)
        
        #build the array
        ans = []
        #by increasing frequency
        for i in range(len(count_of_counts)):
            #ordering decreasing values
            for num in bucket_of_counts[i]:
                for _ in range(i):
                    ans.append(num)
        
###########################################
# 1742. Maximum Number of Balls in a Box
# 23JUL24
###########################################
class Solution:
    def countBalls(self, lowLimit: int, highLimit: int) -> int:
        '''
        the anser is just the the number whos sum digits appear the most
        we dont want the number though,we just need the count
        '''
        counts = defaultdict(int)
        
        for num in range(lowLimit,highLimit + 1):
            sum_digits = self.sumDigits(num)
            counts[sum_digits] += 1
        
        max_count = 0
        
        for k,v in counts.items():
            if v > max_count:
                max_count = v
       
        return max_count
    
    def sumDigits(self,num):
        ans = 0
        while num > 0:
            ans += num % 10
            num = num // 10
        
        return ans

###############################################
# 2247. Maximum Cost of Trip With K Highways
# 23JUL24
###############################################
#nice try
class Solution:
    def maximumCost(self, n: int, highways: List[List[int]], k: int) -> int:
        '''
        dp on graphs
        store visited nodes in mask
        we are done when we have the ones mask ans k is zero
        ending mask is (1 << n) - 1
        
        '''
        graph = defaultdict(list)
        for u,v,toll in highways:
            graph[u].append((v,toll))
            graph[v].append((u,toll))
        
        memo = {}
        
        def dp(path,last_node,k):
            if path == (1 << n) - 1:
                if k == 0:
                    return 0
                return float('-inf')
            
            if k < 0:
                return float('-inf')
            
            if (path,last_node,k) in memo:
                return memo[(path,last_node,k)]
            
            child_ans = 0
            for i in range(n):
                if (path & (1 << i)) == 1:
                    for neigh,weight in graph[i]:
                        if neigh != last_node and (path & (1 << neigh)) == 0:
                            next_path = path | (1 << neigh)
                            child_ans = max(child_ans, weight + dp(next_path,i,k-1))
            
            memo[(path,last_node,k)] = child_ans
            return child_ans
        
        for i in range(n):
            print(dp(1 << i,-1,k))

#fuck yeah
class Solution:
    def maximumCost(self, n: int, highways: List[List[int]], k: int) -> int:
        '''
        need to keep track of the last node in path
        we dont need to touch all cities! we just cant visit a previosuly visited city
        ending mask is when we have k+1 set bits
        need fast way to count set bits -> brian kernighan
        '''
        #corner case, too many edges
        if k + 1 > n:
            return -1
        graph = defaultdict(list)
        for u,v,toll in highways:
            graph[u].append((v,toll))
            graph[v].append((u,toll))
        
        memo = {}
        
        def dp(mask,last_node_visited,k):
            #visited exaclty k+1 highways, means we have seen k nodes
            if self.countSetBits(mask) == k+1:
                return 0
            if (mask,last_node_visited) in memo:
                return memo[(mask,last_node_visited)]
            
            ans = float('-inf')
            for neigh,weight in graph[last_node_visited]:
                #need to vist neigh from this node
                if (mask & (1 << neigh)) == 0:
                    new_mask = mask | (1 << neigh)
                    ans = max(ans, weight + dp(new_mask,neigh,k))
            
            memo[(mask,last_node_visited)] = ans
            return ans
        
        #for all all n nodes
        ans = float('-inf')
        for i in range(n):
            ans = max(ans,dp(1 << i,i,k))
        
        if ans != float('-inf'):
            return ans
        return -1
            
    
    def countSetBits(self,num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        
        return count
    

#bfs solution, this is just search, explore all states, it may or may not TLE
class Solution:
    def maximumCost(self, n: int, highways: List[List[int]], k: int) -> int:
        '''
        we can do bfs, just pass states and maximize
        '''
        if k + 1 > n:
            return -1
        graph = defaultdict(list)
        for u,v,toll in highways:
            graph[u].append((v,toll))
            graph[v].append((u,toll))
        
        #for all all n nodes
        ans = float('-inf')
        for i in range(n):
            ans = max(ans,self.bfs(n,graph,i,k))
        
        if ans != float('-inf'):
            return ans
        return -1
            
    def bfs(self, n, graph, starting_city,k):
        ans = float('-inf')
        starting_mask = 1 << starting_city
        q = deque([(starting_city,starting_mask,0)])
        
        while q:
            city,mask,cost = q.popleft()
            #valid mask, we need the maximum
            if self.countSetBits(mask) == k+1:
                ans = max(ans,cost)
                continue
            
            for neigh,weight in graph[city]:
                #need to vist neigh from this node
                if (mask & (1 << neigh)) == 0:
                    new_mask = mask | (1 << neigh)
                    q.append((neigh,new_mask,cost + weight))
        
        return ans

    def countSetBits(self,num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        
        return count

#########################################
# 2191. Sort the Jumbled Numbers
# 24JUL24
#########################################
class Solution:
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        '''
        sort lexographically based on the mapping
        but order should be presevered in case of ties, so we keep the index too
        sorted non-decreasing on new numbers conversion, then increasing on its original index position
        
        wtf is this edge case
        '''
            
        new_nums = []
        for i,num in enumerate(nums):
            new_num = self.convert_number(num,mapping)
            entry = (new_num,i,num)
            new_nums.append(entry)
        
        new_nums.sort(key = lambda x : (x[0],x[1]))

        
        return [v for _,_,v in new_nums]
    
    def convert_number(self,num,mapping):
        new_num = 0
        mult = 0
        #ah the problem is that it begins with a zero
        if num == 0:
            return mapping[0]
        while num:
            digit = num % 10
            new_digit = mapping[digit]
            new_num += new_digit*(10**mult)
            mult += 1
            num = num // 10
        
        return new_num

#functinoal programming oy vay
class Solution:
    def sortJumbled(self, mapping: List[int], nums: List[int]) -> List[int]:
        #one line just for kicks
        
        def f(num,mapping):
            return int("".join(map(lambda x : str(mapping[int(x)]),str(num))))
        
        return sorted(nums, key = lambda num : f(num,mapping))

#####################################
# 962. Maximum Width Ramp
# 25JUL24
#####################################
#greedily process in order
#if we start with number k, keeps this index minimzed, becaue any number >= k can still use as the leftmost
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        '''
        for some index i, i want to know the furthest j, where i < j and nums[i] <= nums[j]
        i can mapp a number to a list of indicies
        tricky
        '''
        ans = 0
        min_idx = float('inf')
        mapp = defaultdict(list)
        for i,num in enumerate(nums):
            mapp[num].append(i)
            
        #no go in order
        for num in sorted(nums):
            #need index furthest away from the smallest index
            ans = max(ans, mapp[num][-1] - min_idx)
            #upate the smallest index to be used for the next larget element
            min_idx = min(min_idx, mapp[num][0])

        
        return ans
    
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        '''
        we can also avoid using hashmapp
        '''
        ans = 0
        min_idx = float('inf')
        mapp = [(num,i) for i,num in enumerate(nums)]
        mapp.sort()
            
        #no go in order
        for num,i in mapp:
            #need index furthest away from the smallest index
            ans = max(ans, i - min_idx)
            #upate the smallest index to be used for the next larget element
            min_idx = min(min_idx, i)

        
        return ans
    
######################################################################################
# 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance
# 26JUL24
#####################################################################################
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        '''
        i can do djikstras for each city
        then for each city count the neighbording cities that are reachale and not more then k
        '''
        graph = defaultdict(list)
        for u,v,w in edges:
            graph[u].append((v,w))
            graph[v].append((u,w))
            
        ans_city = 0
        min_cities = float('inf')
        
        for i in range(n):
            cities = self.djikstras(graph,i,n,distanceThreshold)
            if cities <= min_cities:
                min_cities = cities
                ans_city = i
        
        return ans_city
        
    
    def djikstras(self,graph,start_node,n,k):
        
        dists = [float('inf')]*n
        dists[start_node] = 0
        visited = [False]*n
        
        pq = [(0,start_node)]
        
        while pq:
            min_dist, node = heapq.heappop(pq)
            if dists[node] < min_dist:
                continue
            visited[node] = True
            for neigh,dist_to in graph[node]:
                if visited[neigh] == True:
                    continue
                new_dist = dists[node] + dist_to
                if new_dist < dists[neigh]:
                    dists[neigh] = new_dist
                    heapq.heappush(pq, (new_dist,neigh))
        
        #search through dists and check dist isnt more than k
        cities = 0
        for i in range(n):
            if i != start_node and dists[i] != float('inf'):
                if dists[i] <= k:
                    cities += 1
        return cities
                    
###########################################
# 2976. Minimum Cost to Convert String I
# 27JUL24
###########################################
class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        '''
        source and target will always be the same length
        original,chnaged,cost will aways eb the same length
        the only steps im allwoed to make are from originla to changed
        rememebr djisktrs is starting at some node u, the reverse may be not be the same

		need to precompute because there are only 26 chars.
		if we didn't we would be doiing djikstras for a really long time
        '''
        graph = defaultdict(list)
        for u,v,w in zip(original,changed,cost):
            graph[u].append((v,w))
        
        min_costs = []
        for i in range(26):
            start_letter = chr(ord('a') + i)
            dists = self.djikstras(graph,start_letter)
            min_costs.append(dists)
        
        min_cost = 0
        for start,end in zip(source,target):
            if start != end:
                start_idx = ord(start) - ord('a')
                end_idx = ord(end) - ord('a')
                temp = min_costs[start_idx][end_idx]
                if temp == float('inf'):
                    return -1
                min_cost  += temp
        
        return min_cost
        
    def djikstras(self,graph,start_letter):
        
        dists = [float('inf')]*26
        dists[ord(start_letter) - ord('a')] = 0

        
        pq = [(0,start_letter)]
        
        while pq:
            min_dist, node = heapq.heappop(pq)
            node_idx = ord(node) - ord('a')
            if dists[node_idx] < min_dist:
                continue
            for neigh,dist_to in graph[node]:
                neigh_idx = ord(neigh) - ord('a')
                new_dist = dists[node_idx] + dist_to
                if new_dist < dists[neigh_idx]:
                    dists[neigh_idx] = new_dist
                    heapq.heappush(pq, (new_dist,neigh))
        
        return dists
        
##################################################
# 2045. Second Minimum Time to Reach Destination
# 28JUL24
####################################################
#close one
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        '''
        bi-directional edge, each vertex is connected by at most one edge
        i.e there could be vertex pairs that dont have an edge
        need second minimum time to from vertex 1 to vertex n
        each node has singal that can turn green to red, and change every chagne minutes
        cannot wait on green signal, so if its green just go
        but i can wait on red, 
        need to keep track of the number of minutes that have passed, say i have at a node in k minutes
        i need to check k % change, i can than compare the quotient and remainder if quotient is even its green, if quotient is odd, its red
        and i have to wait remainder minutes, knowing this i can easily find the minimum, but what about the second minimum?
        note: each vertext can be reached directly or indirectly from every othe vertex
        thought 1; find min path, then drop edges, and then find min again, wont work, see example 2
        just two bfs twice! rather keep doing bfs until w hit n-1 twice
        the second time we hit n-1, that's our time
        
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        count_end = 0
        total_mins = 0
        q = deque([(0,1)]) #store as (minutes, and node)
        
        while count_end < 2:
            curr_mins,curr_node = q.popleft()
            
            if curr_node == n:
                #second time we've seen the end
                if count_end == 1:
                    total_mins = curr_mins
                    break
                else:
                    count_end += 1
            for neigh in graph[curr_node]:
                #if we need to wait
                if (curr_mins // change) % 2 == 1:
                    wait = change - (curr_mins % change)
                else:
                    wait = 0
                new_mins = curr_mins + wait + time
                    
                q.append((new_mins,neigh))
        
        return total_mins
        
        
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        '''
        we can use djikstras and keep count of the number of times we have visited a node
        we also keep 2 distances array, one for the first minimum and the other for the second minimum
        we also keep track of the numebr of times we have visited a node
        if can updates dists1 we do so, otherwise we try updating dists2
        if it pops out the first time, its the first minimm, if it pops out the second time, its the scond minimum
            we need to be careful we dont visit any node more than twice
        
        green-red light contraint
        its green if [0,2c]...[2c,3c], 
        ist red if [c,3c}...[3c,4c]
        
        in code
        if (timetaken/ change % 2) #redligh
            neigh_time = change*(timeTaken/change + 1) + time
        else
            #green light
            neigh_time - timetaken + time
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        dists1 = [float('inf')]*(n+1)
        dists2 = [float('inf')]*(n+1)
        counts = [0]*(n+1)
        
        pq = [(1,0)]
        dists1[1] = 0 #no second minimum yet
        while pq:
            node,time_taken = heapq.heappop(pq)
            counts[node] += 1
            
            if counts[node] == 2 and node == n:
                return time_taken
            
            #waiting times increment 
            #if red light 
            if ((time_taken // change) % 2 == 1):
                time_taken = change*(time_taken // change + 1) + time
            else:
                time_taken = time_taken + time
            
            for neigh in graph[node]:
                if counts[neigh] == 2:
                    continue
                #djikstra update
                #update dist1 if its more than the current time, and swap values with dist 2
                if (dists1[neigh] > time_taken):
                    dists2[neigh] = dists1[neigh]
                    dists1[neigh] = time_taken
                    heapq.heappush(pq, (neigh,time_taken))
                elif (dists2[neigh] > time_taken and dists1[neigh] != time_taken):
                    dists2[neigh] = time_taken
                    heapq.heappush(pq, (neigh,time_taken))
            
        return 0

#almost
#need to be careful when chosing the next node, its not always skpping the second visit
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        '''

        
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        counts = [0]*(n+1)
        q = deque([(0,1)]) #store as (minutes, and node)
        
        while q:
            curr_mins,curr_node = q.popleft()
            counts[curr_node] += 1
            if counts[curr_node] == 2 and curr_node == n:
                return curr_mins
            for neigh in graph[curr_node]:
                if counts[neigh] == 2:
                    continue
                #if we need to wait
                if (curr_mins // change) % 2 == 1:
                    wait = change - (curr_mins % change)
                else:
                    wait = 0
                new_mins = curr_mins + wait + time
                    
                q.append((new_mins,neigh))
        
        return total_mins
    
class Solution:
    def secondMinimum(self, n: int, edges: List[List[int]], time: int, change: int) -> int:
        '''
        for BFS, we need to be careful if we want the first pr se
        if the freq of the visited node is 1 we use dist1
        if freq of visited node is 2, we use dist2
        
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        dist1 = [-1]*(n+1)
        dist2 = [-1]*(n+1)
        q = deque([(1,1)]) #entries are (node,count)
        dist1[1] = 0
        
        while q:
            node,count = q.popleft()
            if count == 1:
                curr_mins = dist1[node]
            else:
                if node == n:
                    return dist2[node]
                curr_mins = dist2[node]
            
            if ((curr_mins // change) % 2) == 1:
                curr_mins = change*((curr_mins // change) + 1) + time
            else:
                curr_mins = curr_mins + time
            
            for neigh in graph[node]:
                if dist1[neigh] == -1:
                    dist1[neigh] = curr_mins
                    q.append((neigh,1))
                elif dist2[neigh] == -1 and dist1[neigh] != curr_mins:
                    dist2[neigh] = curr_mins
                    q.append((neigh,2))
                    
        
        return 0
        
###########################################
# 1395. Count Number of Teams
# 29JUL24
##########################################
class Solution:
    def numTeams(self, rating: List[int]) -> int:
        '''
        pick a center, then search left the number of increasing and decreasing,
        same for the right,
        answer is the product
        '''
        ans = 0
        N = len(rating)
        
        for i in range(1,N-1):
            left = i - 1
            inc_smaller_than = 0
            dec_greater_than = 0
            while left >= 0:
                if rating[left] < rating[i]:
                    inc_smaller_than += 1
                elif rating[left] > rating[i]:
                    dec_greater_than += 1
                
                left -= 1
            
            right = i+1
            inc_greater_than = 0
            dec_smaller_than = 0
            while right < N:
                if rating[right] > rating[i]:
                    inc_greater_than += 1
                elif rating[right] < rating[i]:
                    dec_smaller_than += 1
            
                right += 1
            
            ans += inc_smaller_than*inc_greater_than
            ans += dec_greater_than*dec_smaller_than
        
        return ans
                
#dp, top down
#similar to LIS invariant
class Solution:
    def numTeams(self, rating: List[int]) -> int:
        '''
        now try dp
        dp states are index i, and k, where i represents the index of the starting soldier
        and k being the number of spots left,
        if im at (i,1), i need to look at all numbers at indices j, where j > i and nums[j] > nums[i]
        when k == 2, we found a triplet
        solve dp(i,k) for all i
        '''
        memo_inc = {}
        memo_dec = {}
        N = len(rating)
        
        def dp_inc(i,k,memo):
            if k == 3:
                return 1
            if i >= N:
                return 0
            if (i,k) in memo:
                return memo[(i,k)]
            ways = 0
            for j in range(i+1,N):
                if rating[j] > rating[i]:
                    ways += dp_inc(j,k+1,memo)
            
            memo[(i,k)] = ways
            return ways
        
        def dp_dec(i,k,memo):
            if i >= N:
                return 0
            if k == 3:
                return 1
            if (i,k) in memo:
                return memo[(i,k)]
            ways = 0
            for j in range(i+1,N):
                if rating[j] < rating[i]:
                    ways += dp_dec(j,k+1,memo)
            
            memo[(i,k)] = ways
            return ways
        
        ans = 0
        for i in range(N):
            ans += dp_inc(i,1,memo_inc)
            ans += dp_dec(i,1,memo_dec)
        
        return ans
        
#can also so sorted list
#note insert into sorted list is actuall sqrt(N)
#and not logN        


######################################
# 1230. Toss Strange Coins (REVISTED)
# 29JUL24
######################################
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        '''
        dp states are index i into probs and count of heads
        want to solve for state (i = 0, and count = 0)
        dp(i,count) gives the probability of gettin count heads with coins up to i
        if we get TLE wih bottom up, make sure we prune the tree
        '''
        memo = {}
        N = len(prob)
        
        def dp(i,count):
            if count > target:
                return 0
            if i == N:
                if count == target:
                    return 1
                return 0
            if (i,count) in memo:
                return memo[(i,count)]
            #need prob getting heads. #but also 1 - prob of getting tails
            #we add them
            ans = prob[i]*dp(i+1,count+1) + (1 - prob[i])*(dp(i+1,count))
            memo[(i,count)] = ans
            return ans
        
        return dp(0,0)
            
################################################
# 1653. Minimum Deletions to Make String Balanced
# 30JUL24
################################################
#three pass
class Solution:
    def minimumDeletions(self, s: str) -> int:
        '''
        need to delete characters to make s balance
        s i balance is there is no pair of idicies (i,j), where i < j and s[i] == 'b' and s[j] == 'a'
        basically all the a's have to come before all the bs
        need to find the number of B's before an index 
        and the number of As after an index, but does it need include the inde
        if string is balanced
            the prefx_bs array will have all zeros to the left
            the suffx_as array will have all zeros to the right
        
        to balance at index i means to have no bs before it and no as after it
        we need to take the minimum of the two pref + suff arrays
        '''
        N = len(s)
        prefix_bs = [0]*N
        prefix_bs[0] += s[0] == 'b'
        for i in range(1,N):
            prefix_bs[i] = prefix_bs[i-1] + (s[i] == 'b')
        
        suffix_as = [0]*N
        suffix_as[-1] += s[-1] == 'a'
        for i in range(N-2,-1,-1):
            suffix_as[i] = suffix_as[i+1] + (s[i] == 'a')
        
        
        ans = N
        for i in range(N):
            temp = prefix_bs[i] + suffix_as[i]
            temp -= 1
            ans = min(ans,temp)
        
        return ans
    
#two pass
class Solution:
    def minimumDeletions(self, s: str) -> int:
        '''
        two pass,
        instead of using two passes to compute bs to the left and as to the right, we can do this in one pass
        we need to count either bs to the left first or as to the right
        then when we minimize on the second pass, we can compute the bs to th left on the fly
        '''
        N = len(s)
        suffix_as = [0]*N
        count_a = 0
        for i in range(N-1,-1,-1):
            suffix_as[i] = count_a
            count_a += s[i] == 'a'
        
        #find in deletions, but also count bs on the fly
        min_deletions = N
        count_b = 0
        for i in range(N):
            min_deletions = min(min_deletions, count_b + suffix_as[i])
            count_b += s[i] == 'b'
        
        return min_deletions
    
class Solution:
    def minimumDeletions(self, s: str) -> int:
        '''
        two pass, with two variables
        compute count as on first pass
        sexond pass when going left to right, do the same thing with counting bs on the fly
        but also decrement the as when we see it because we wanted as to the right of the current index
        '''
        N = len(s)
        count_a = 0
        for ch in s:
            count_a += ch == 'a'
            
        min_deletions = N
        count_b = 0
        for ch in s:
            count_a -= ch == 'a'
            min_deletions = min(min_deletions, count_a + count_b)
            count_b += ch == 'b'
        
        return min_deletions

#dp variant
class Solution:
    def minimumDeletions(self, s: str) -> int:
        '''
        let dp(i) be the minimum deletions to make string balanced from s[:i]
        either delete the a or delete the b
        '''
        N = len(s)
        prefix_bs = [0]*N
        count_b = 0
        for i in range(N):
            count_b += s[i] == 'b'
            prefix_bs[i] = count_b
        
        memo = {}
        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            ans = min(1 + dp(i+1), prefix_bs[i])
            memo[i] = ans
            return ans
            
        return dp(0)
            
class Solution:
    def minimumDeletions(self, s: str) -> int:
        n = len(s)
        dp = [0] * (n + 1)
        b_count = 0

        # dp[i]: The number of deletions required to
        # balance the substring s[0, i)
        for i in range(n):
            if s[i] == "b":
                dp[i + 1] = dp[i]
                b_count += 1
            else:
                # Two cases: remove 'a' or keep 'a'
                dp[i + 1] = min(dp[i] + 1, b_count)

        return dp[n]

#############################################
# 1105. Filling Bookcase Shelves (REVISTED)
# 31JUL24
############################################
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        '''
        dp states are index of book and curr shelf
        we can pass in curr height to the function, but states are only based on position and shelf with
        add to current shelf, or make new shelf andd add height
        
        notes on why we dont need to cache curr_height
        because the books are processed in order, we keep pushing books left, it actually never changes
        it just moves
        '''
        N = len(books)
        memo = {}
        
        def dp(i,curr_height,curr_shelf):
            if curr_shelf < 0:
                return float('inf')
            if i == N:
                return curr_height
            if (i,curr_shelf) in memo:
                return memo[(i,curr_shelf)]
            
            width,height = books[i]
            #keep book on same shelf
            same_shelf = dp(i+1, max(curr_height,height), curr_shelf - width)
            #new shelft
            new_shelf = curr_height + dp(i+1,height,shelfWidth - width)
            ans = min(same_shelf,new_shelf)
            memo[(i,curr_shelf)] = ans
            return ans
        
        return dp(0,0,shelfWidth)

class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        #another way
        @cache
        def place(book_pos, cur_width, max_height):
            if book_pos == len(books):
                return 0
            width, height = books[book_pos]
            ans = height + place(book_pos + 1, width, height)  # new shelf
            if book_pos and cur_width + width <= shelfWidth:   # same shelf
                height_increase = max(0, height - max_height)
                ans = min(ans, height_increase + place(book_pos + 1, cur_width + width, max_height + height_increase))

            return ans

        return place(0, 0, 0)

class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:

        N = len(books)

        memo = {}        
        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            curr_shelf = 0
            max_height = 0
            ans = float('inf')
            j = i
            while j < N and curr_shelf + books[j][0] <= shelfWidth:
                curr_shelf += books[j][0]
                max_height = max(max_height, books[j][1] )
                j += 1
                #could also choose to make a new shelft here
                new_shelf = max_height + dp(j)
                #minimize answer
                ans = min(ans, new_shelf)
            
            memo[i] = ans
            return ans
        
        return dp(0)
            