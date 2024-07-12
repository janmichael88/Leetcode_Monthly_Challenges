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