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

class Solution:
    def countPairsOfConnectableServers(self, edges: List[List[int]], signalSpeed: int) -> List[int]:
        '''
        for it to be path, there must be > 2 nodes in the path
        let dp(nodes) be the number of nodes in this subtree 
        then the number of paths for this node is
            paths = 1
            for child dp(nocde):
                paths *= child
        dfs is doing two things
            count number of nodes at distances divisible by speed from this node
            count number of pairs for each root
        '''
        adj = defaultdict(list)
        for s,d,w in edges:
            adj[s].append([d,w])
            adj[d].append([s,w])
        
        def dfs(node,prev,dist):
            #this node has count 1 if we have some dist and its path is divisible by signal spee
            count = 1 if dist > 0 and dist % signalSpeed == 0 else 0
            #count pairs in all directions
            pairs = 0
            for nei,weight in adj[node]:
                if nei != prev:
                    child_count = dfs(nei, node, dist + weight)
                    pairs += count*child_count #count pairs
                    #count for children at this node
                    count += child_count
            #just return the pairs for the node
            if node == prev:
                return pairs
            #return the count
            return count
        
        n = len(edges) + 1
        res = [0]*n
        for node in range(n):
            paths = dfs(node, node, 0)
            res[node] = paths
        return res
    
class Solution:
    def countPairsOfConnectableServers(self, edges: List[List[int]], signalSpeed: int) -> List[int]:
        '''
        count nodes divisible by distnacce (for each node)
        then count pairs
        '''
        graph, res = defaultdict(list), []
        for startNode, endNode, weight in edges:
            graph[startNode].append((endNode, weight))
            graph[endNode].append((startNode, weight))
            
        for node in range(len(edges) + 1):
            res.append(self.solve(node,graph,signalSpeed))
        return res

    #counts nodes divisible by signalSpeed from curNode
    def dfs(self, curNode: int, parent: int, weight: int,signalSpeed,graph) -> int:
        res = 1 if weight % signalSpeed == 0 else 0
        for childNode, curWeight in graph[curNode]:
            if childNode == parent: continue
            res += self.dfs(childNode, curNode, weight + curWeight,signalSpeed,graph)
        return res

    def solve(self, curNode: int,graph,signalSpeed) -> int:
        res = cur = 0
        for neighbor, weight in graph[curNode]:
            dfsResult = self.dfs(neighbor, curNode, weight,signalSpeed,graph)
            res += (dfsResult * cur)
            cur += dfsResult
        return res

        for node in range(len(edges) + 1):
            res.append(solve(node))
        return res
        
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
            
########################################
# 2487. Remove Nodes From Linked List
# 06MAY24
########################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        say if we had the array
        [5,2,13,3,8]
        keep array max to right
        [13,13,13,8,8]
        then we can rebuild
        '''
        arr = []
        curr = head
        while curr:
            arr.append(curr.val)
            curr = curr.next
        
        right_maxs = [0]*len(arr)
        right_maxs[-1] = arr[-1]
        for i in range(len(arr)-2,-1,-1):
            right_maxs[i] = max(arr[i],right_maxs[i+1])
        
        ans = ListNode(-1)
        curr = ans
        for i in range(len(arr)):
            if arr[i] >= right_maxs[i]:
                curr.next = ListNode(arr[i])
                curr = curr.next
        
        return ans.next

#reverse once, then pass and build
#in order to add to head of LL, we need to add in reverse
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        iterate in reverse order and save max
        then just make new head
        '''
        revd = self.reverse(head)
        curr_max = revd.val
        ans = ListNode(revd.val)
        revd = revd.next
        
        while revd:
            if revd.val < curr_max:
                revd = revd.next
            else:
                new_node = ListNode(revd.val)
                new_node.next = ans
                ans = new_node
                curr_max = revd.val
                revd = revd.next
        
        return ans
    
    def reverse(self, node):
        prev = None
        curr = node
        while curr:
            next_ = curr.next
            curr.next = prev
            prev = curr
            curr = next_
        
        return prev
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        recursion 
        rec(node) deltes node if there is a larger node to the right
        base cases empty node or single node, then just return it 
        otherwise recurse 
        '''
        def rec(node):
            if not node or not node.next:
                return node
            #call to next node
            next_ = rec(node.next)
            #delete if smaller, or just return the next node
            if node.val < next_.val:
                return next_
            #otherwise we can include this node, and it should come before next_
            node.next = next_
            return node
        
        return rec(head)
    
######################################################
# 2816. Double a Number Represented as a Linked List
# 07MAY24
######################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        reverse and implement carry
        '''
        revd = self.rev(head)
        carry = 0
        #hold prev in case we have carry
        prev = None
        curr = revd
        while curr:
            new_val = curr.val*2 + carry
            curr.val = new_val % 10
            carry = new_val // 10
            prev = curr
            curr = curr.next
        
        if carry:
            prev.next = ListNode(1)
        
        return self.rev(revd)
    
    def rev(self, node):
        prev = None
        curr = node
        while curr:
            next_ = curr.next
            curr.next = prev
            prev = curr
            curr = next_
        
        return prev
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        using stack
        '''
        vals = []
        curr = head
        while curr:
            vals.append(curr.val)
            curr = curr.next
        
        ans = None
        carry = 0
        while vals:
            curr_val = vals.pop()
            ans = ListNode(0,ans)
            ans.val = (curr_val*2 + carry) % 10
            carry = (curr_val*2 + carry) // 10
        
        if carry:
            ans = ListNode(1,ans)
        return ans

#recursion
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        for recrsion, we will have it return the doubled value, 
        since we need to reverse it, recursino should work perfectly since in recursion we will move through the whole LL
        and return the value that should be the node
        '''
        
        def rec(node):
            if not node:
                return 0
            #get the double value
            next_node = rec(node.next) + 2*node.val
            node.val = next_node % 10
            return next_node // 10
        
        #find final carry
        carry = rec(head)
        if carry:
            head = ListNode(carry,head)
        
        return head #remember we modified this in place during the recursion

#two pointers
# Definition for singly-linked list.
class ListNode:
     def __init__(self, val=0, next=None):
         self.val = val
         self.next = next
class Solution:
    def doubleIt(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        we can actually add left to right, not just right to left!
        but if there's carry on this node, it should go to the previous node
        and if there isn't a previous node, we make one
        there are 3 cases
        1. doubled value < 10, we can just dounle it
        2. doubled value >= 10, then prev should get the carry
        3. if first node need to be updated, we make a new node
        '''
        curr = head
        prev = None
        
        while curr:
            doubled = curr.val*2
            if doubled < 10:
                curr.val = doubled
            elif prev:
                curr.val = doubled % 10
                prev.val += doubled // 10
            else:
                #first node, make a new one
                head = ListNode(1,curr)
                curr.val = doubled % 10
            
            prev = curr
            curr = curr.next
        
        return head

######################################################################
# 3067. Count Pairs of Connectable Servers in a Weighted Tree Network
# 07MAY24
######################################################################
class Solution:
    def countPairsOfConnectableServers(self, edges: List[List[int]], signalSpeed: int) -> List[int]:
        '''
        for it to be path, there must be > 2 nodes in the path
        let dp(nodes) be the number of nodes in this subtree 
        then the number of paths for this node is
            paths = 1
            for child dp(nocde):
                paths *= child
        dfs is doing two things
            count number of nodes at distances divisible by speed from this node
            count number of pairs for each root
        '''
        adj = defaultdict(list)
        for s,d,w in edges:
            adj[s].append([d,w])
            adj[d].append([s,w])
        
        def dfs(node,prev,dist):
            #this node has count 1 if we have some dist and its path is divisible by signal spee
            count = 1 if dist > 0 and dist % signalSpeed == 0 else 0
            #count pairs in all directions
            pairs = 0
            for nei,weight in adj[node]:
                if nei != prev:
                    child_count = dfs(nei, node, dist + weight)
                    pairs += count*child_count #count pairs
                    #count for children at this node
                    count += child_count
            #just return the pairs for the node
            if node == prev:
                return pairs
            #return the count
            return count
        
        n = len(edges) + 1
        res = [0]*n
        for node in range(n):
            paths = dfs(node, node, 0)
            res[node] = paths
        return res

##########################################
# 755. Pour Water
# 08MAY24
##########################################
class Solution:
    def pourWater(self, heights: List[int], volume: int, k: int) -> List[int]:
        '''
        this is kinda like an advent of code problem
        make the chamber, as boolean array, 2d, then you just drop until we cant
            no chance of overflow because the walls on left and right are infinitely high
        start backwards from where it would drop to?
            it would drop to its left, closted to k towards the bottom
            update the next drop spot, this is the hard part of the problem
        look left, and drop at lowest point
        if we can't look right and drop at lowest point
        '''
        while volume > 0:
            #try finding left first
            left_index = -1
            for i in range(k-1,-1,-1):
                if heights[i] > heights[i+1]:
                    break
                elif heights[i] < heights[i+1]:
                    left_index = i
            
            if left_index != -1:
                heights[left_index] += 1
            else:

                #try finding right
                right_index = -1
                for i in range(k+1,len(heights)):
                    if heights[i] > heights[i-1]:
                        break
                    elif heights[i] < heights[i-1]:
                        right_index = i

                if right_index != -1:
                    heights[right_index] += 1
                else:
                    heights[k] += 1
            volume -= 1
        
        return heights
        
##################################
# 506. Relative Ranks (REVISTED)
# 08MAY24
##################################
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        '''
        linear or rather O(n + max(score)) can be done with counting sort variant, sincce all scores are unique
        '''
        max_score = max(score)
        marked_scores = []
        for _ in range(max_score+1):
            marked_scores.append([False,-1])
        
        for i,s in enumerate(score):
            marked_scores[s] = [True,i]
        
        N = len(score)
        ranks = [0]*N
        medals = ['Bronze Medal', 'Silver Medal', 'Gold Medal']
        place = 1
        for i in range(max_score,-1,-1):
            if marked_scores[i][0]:
                s,idx = marked_scores[i]
                if medals:
                    ranks[idx] = medals.pop()
                else:
                    ranks[idx] = str(place)
                place += 1
        
        return ranks

########################################
# 2473. Minimum Cost to Buy Apples
# 08MAY24
########################################
#fucking ez bruhhhhh
class Solution:
    def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
        '''
        use dp with states, pass in curr and parent to make sure we dont go back and minimize
        we only want to be one apple, all costs are positive, so it makes sense to chose the cheapest one
        we need to come back to the city we started at
        its the return path that could be different, im not convinced that some path to a city will have the same path returning
        use djikstras on each city, this time we will actually need to save the shortest path
        then find the min cost
        '''
        graph = defaultdict(list)
        for u,v,weight in roads:
            graph[u].append((v,weight))
            graph[v].append((u,weight))
        
        ans = []
        for city in range(1,n+1):
            dists = self.ssp(n,graph,city)
            local_ans = float('inf')
            for j,dist in enumerate(dists):
                if dist != float('inf'):
                    local_ans = min(local_ans, dist + dist*k + appleCost[j-1])
            ans.append(local_ans)
        
        return ans
                    
    def ssp(self,n,graph,start):
        dists = [float('inf')]*(n+1)
        dists[start] = 0
        pq = [(0,start)]
        visited = set()
        
        while pq:
            curr_dist,curr_city = heapq.heappop(pq)
            #cant minmize
            if dists[curr_city] < curr_dist:
                continue
            visited.add(curr_city)
            for neigh_city,weight in graph[curr_city]:
                if neigh_city in visited:
                    continue
                new_dist = dists[curr_city] + weight
                #can improve?
                if new_dist < dists[neigh_city]:
                    #update and add
                    dists[neigh_city] = new_dist
                    heapq.heappush(pq, (new_dist,neigh_city))
        
        return dists

#we dont need to process the dists array, but we can do it on the fly
class Solution:
    def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
        '''
        we dont need to process the dits array, after doing djikstra
        we can include the cost of buying the apple with the road cost as well as the return cost back to the starting city
        new_cost = curr_cost + (k+1)*road_cost + appleCost ; for the nieghbor
        '''
        graph = defaultdict(list)
        for u,v,weight in roads:
            graph[u-1].append((v-1,weight))
            graph[v-1].append((u-1,weight))
            
        ans = []
        for i in range(n):
            min_cost = self.ssp(n,graph,i,appleCost,k)
            ans.append(min_cost)
        
        return ans
                    
    def ssp(self,n,graph,start,apples,k):
        dists = [float('inf')]*(n)
        dists[start] = 0
        pq = [(0,start)]
        visited = set()
        min_cost = float('inf')
        
        while pq:
            curr_dist,curr_city = heapq.heappop(pq)
            #cant minmize
            if dists[curr_city] < curr_dist:
                continue
            min_cost = min(min_cost, curr_dist*(k+1) + apples[curr_city])
            visited.add(curr_city)
            for neigh_city,weight in graph[curr_city]:
                if neigh_city in visited:
                    continue
                new_dist = dists[curr_city] + weight
                #can improve?
                if new_dist < dists[neigh_city]:
                    #update and add
                    dists[neigh_city] = new_dist
                    heapq.heappush(pq, (new_dist,neigh_city))
        
        return min_cost
    
#we can do in one pass if we just start with the appleCost at each city
class Solution:
    def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
        '''
        we dont need to process the dits array, after doing djikstra
        we can include the cost of buying the apple with the road cost as well as the return cost back to the starting city
        new_cost = curr_cost + (k+1)*road_cost + appleCost ; for the nieghbor
        
        we can even do one pass by just starting with appleCost for each city
        then we can do djikstras with multi-point source
        the intuition is that for at least one city, it is chepeast to buy apples from that one city, so long as we dont travel to other cities
        also note, the return path is shortest if we just take it back, since the path to that city was already shortest
        
        so we compute the total min cost starting with cities with the minimum apple cost
        we also need to stroe info about any city visited during thessp, not just tarting
        '''
        graph = defaultdict(list)
        for u,v,weight in roads:
            graph[u-1].append((v-1,weight))
            graph[v-1].append((u-1,weight))
            
        return self.ssp(n,graph,appleCost,k)
            
                    
    def ssp(self,n,graph,apples,k):
        #we know we are at a minimum if we dont travel and just take applecost
        dists = apples[:]
        pq = [(apple_cost,city) for (city,apple_cost) in enumerate(apples)]
        heapq.heapify(pq)
        visited = set()
        
        while pq:
            curr_dist,curr_city = heapq.heappop(pq)
            #cant minmize
            if dists[curr_city] < curr_dist:
                continue
            visited.add(curr_city)
            for neigh_city,weight in graph[curr_city]:
                if neigh_city in visited:
                    continue
                new_dist = dists[curr_city] + (k + 1)*weight
                #can improve?
                if new_dist < dists[neigh_city]:
                    #update and add
                    dists[neigh_city] = new_dist
                    heapq.heappush(pq, (new_dist,neigh_city))
        
        return dists

#dp, but it tles
class Solution:
    def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
        '''
        we can also use dp to find shortest paths
        '''
        graph = defaultdict(list)
        for u,v,weight in roads:
            graph[u].append((v,weight))
            graph[v].append((u,weight))
        
        
        ans = []
        for start in range(1,n+1):
            dists = []
            visited = set()
            for end in range(1,n+1):
                dists.append(self.dp(graph,start,end,visited))
            local_ans = float('inf')
            for i,d in enumerate(dists):
                local_ans = min(local_ans, d + d*k + appleCost[i])
            ans.append(local_ans)
        
        return ans
                
    def dp(self,graph,curr,end,visited):
        if curr == end:
            return 0
        visited.add(curr)
        ans = float('inf')
        for neigh,weight in graph[curr]:
            if neigh not in visited:
                visited.add(neigh)
                ans = min(ans, weight + self.dp(graph,neigh,end,visited))
                visited.remove(neigh)
        
        return ans
        
#tles still
class Solution:
    def minCost(self, n: int, roads: List[List[int]], appleCost: List[int], k: int) -> List[int]:
        '''
        we can also us dp and backtracking
        '''
        graph = defaultdict(list)
        for u,v,weight in roads:
            graph[u-1].append((v-1,weight))
            graph[v-1].append((u-1,weight))
            
        ans = []
        visited = set()
        for i in range(n):
            visited.add(i)
            ans.append(self.dp(appleCost,i,graph,k,0,float('inf'),visited))
            visited.remove(i)
        
        return ans
            
                    
    def dp(self,apples,curr,graph,k,roadCost,minPrice,visited):
        
        totalRoadCost = (k+1)*roadCost
        #minmize final
        if minPrice < totalRoadCost:
            return minPrice
        ans = min(minPrice, apples[curr] + totalRoadCost)
        for neigh,weight in graph[curr]:
            if neigh in visited:
                continue
            
            visited.add(neigh)
            next_dist = self.dp(apples,neigh,graph,k,roadCost + weight,minPrice,visited)
            ans = min(ans,next_dist)
            visited.remove(neigh)
        
        return ans
    
################################################
# 3075. Maximize Happiness of Selected Children
# 09MAY24
################################################
class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        '''
        n children in happiness, want k children in k turns
        in each turn, when child is selected, the happiness value of all children that have not been select until decrease by 1
        happiness cannot be negtaive, and only decremented if positive
        return max sum of happiness after selecting k children
        makes sens to take the largest happiness, but after taking we need to decrement the sum
        get whole sum first, and decrement by (len of children left)
        [1,2,3]
        take 3
        [1,2] becomes [0,1]
        issue is now to make sure if happiness is zero, we dont count it as a decrement
        keep track of children that has some happiness left
        '''
        ans = 0
        children_taken = 0
        happiness.sort(reverse = True)
        left = 0
        for _ in range(k):
            #calculate new happiness
            new_happiness = max(happiness[left] - children_taken,0)
            ans += new_happiness
            left += 1
            children_taken += 1
        
        return ans
    
############################################
# 786. K-th Smallest Prime Fraction
# 10MAY24
############################################
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        '''
        its increasing, 
        we need kth
        '''
        temp = []
        N = len(arr)
        for i in range(N):
            for j in range(i+1,N):
                temp.append((arr[i]/arr[j], arr[i],arr[j]))
        
        temp.sort(key = lambda x: x[0])
        k -= 1
        return [temp[k][1],temp[k][2]]
    
#this solution is really fast
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        '''
        they are all sorted
        if i fix i as the num, than any of the other js will be the denom
        find lcm of numbers?
        
        intuition:
            say we have som fraction with pairs (arr[i],arr[j])
            since the array is sorted, the number of fractions smaller than this pair
            is when (arr[i] / arr[k]) > (arr[i],arr[j]), this will occur when k > j 
        
        its workable solution paradigm, count all fractions from some given fraction
        if this fraction works we can move up one
        we need to use binary search to find the kth smallest fraction
        
        intuition 2:
            if we have encounter k or more fractions smaller than or euqal to this amx fraction, this this max fraction is the kth smallest
            
        '''
        n = len(arr)
        left,right = 0, 1.0
        
        #binary search on a workable fraction, and check count (i,j) pairs where arr[i]/arr[j] < fraction
        while left < right:
            mid = left + ((right - left) / 2)
            max_frac = 0.0
            count_smaller = 0
            num_idx,denom_idx = 0,0 #initall start at beginnging
            j = 1
            
            #try all arr[i]
            for i in range(n-1):
                while j < n and arr[i] >= mid*arr[j]: #want largest (arr[i] / arr[j]) < mid  in order to contribute count
                    #essentially findinf upper bound of current numerator
                    j += 1
                
                count_smaller += (n - j)
                #got to the end
                if j == n:
                    break
                
                #update fraction to store index
                fraction = arr[i] / arr[j]
                if fraction > max_frac:
                    max_frac = fraction
                    num_idx = i
                    denom_idx = j
            
            #adjust search on workable fraction
            if count_smaller == k:
                return [arr[num_idx], arr[denom_idx]]
            elif count_smaller > k:
                #try a smaller frac
                right = mid
            else:
                left = mid
        
        return []
    
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        '''
        heap is easier,
        fractions are smallest when the arr[i] / arr[j] where j is just the last index
        when we pop, we need to introduce the next smallest
        do this k-1 times
        '''
        N = len(arr)
        min_heap = []
        for i in range(N):
            fraction = arr[i] / arr[-1]
            min_heap.append((-fraction,i,N-1))
            
        heapq.heapify(min_heap)
        for _ in range(k-1):
            curr = heapq.heappop(min_heap)
            #get nest smallest
            num_idx = curr[1]
            denom_idx = curr[2] - 1
            next_frac = arr[num_idx] / arr[denom_idx]
            entry = (-next_frac, num_idx,denom_idx)
            if denom_idx > num_idx:
                heapq.heappush(min_heap,entry)
        
        curr_frac, num_idx, denom_idx = heapq.heappop(min_heap)
        
        return [arr[num_idx],arr[denom_idx]]
    
class Solution:
    def kthSmallestPrimeFraction(self, arr: List[int], k: int) -> List[int]:
        '''
        heap is easier,
        fractions are smallest when the arr[i] / arr[j] where j is just the last index
        when we pop, we need to introduce the next smallest
        do this k-1 times
        '''
        heap = []
        n = len(arr)
        for i in range(n-1):
            heappush(heap,(arr[i]/arr[-1],i,n-1))
        
        for i in range(k-1):
            res,l,r = heappop(heap)
            heappush(heap,(arr[l]/arr[r-1],l,r-1))

        res,l,r = heappop(heap)

        return [arr[l],arr[r]]
    
#########################################
# 857. Minimum Cost to Hire K Workers
# 11MAY24
##########################################
class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
        '''
        need min cost to satisfy paid group
        workers pay must be direclty proportioanl to their quality
            if work's quality is double that of another worke in the group, that worker must be paid twise as much as the other worker
        if we have k workers, we need to ensure the the pay is directly proportional amoung the other k-1 woekrs
        they didn't give enough examples for balacing the proportions....
        
        say we have two workers (i,j), the conditions must be
        wage[i]/wage[j] = quality[i]/quality[j]
        rewrite as:
        wage[i] /quality[i] = wage[j]/quality[j]
        say we have two workers with i and j quality, then i should get:
            quality[i] / (quality[i] + quality[j]) of the payment
            and j should get: 
            quality[j] / (quality[i] + quality[j]) 
        
        infact for any ith person in k people, the ith person should get:
            quality[i] / (sum of qualities for all k people) * payment for all
        in the given example, worker 0 gets (2/3) of payment and worker 2 gets (1/3) of payment
        worker 0 has the higher min wage so we can do:
        (1/3) / (2/3) = x / 70 x -> 35
        we need to find the least  amount of money to form such a group
        we need to use the wage/quaility ratio for each person
        to determine the optimal worker ppol, we compute the max quality per unit multiplie by total quantiy for every 2 woekrs
        For worker 0 and 1: max quaility ratio = 7
        7*(10 + 20) = 210, going through the other pairs (0,1) ans (1,2), we get 105,150, so 105 is the smallest
        how can we build up the k people to minimize, instead of checkking all k?
        depends on workes quality and ratio of wage/quality
        intution: try taking workers with lowest wage/quality ratio -> implies sorting
        also need to keep track of qualities of workers so far
            max heap stores quality of workers
            recall total cost == sum of quality of chosen workers * max ratio
        
        if size > k, we need to remove the worker with the highest quality 
        once we have k workers in pq, we can calc the total cost for the curent set of workers by multiplying each workers euqailty 
        by their wage to quality ratio and summing products
        basically we need to maintin the current total quality, and the smalest wage to quality ratio for the k workers
        because we are limited by the smallest ratio
    
        '''
        N = len(quality)
        min_cost = float('inf')
        curr_total_quality = 0
        ratios = []
        for i in range(N):
            entry = (wage[i]/quality[i], quality[i])
            ratios.append(entry)
        
        #sort ratios
        ratios.sort(key = lambda x: x[0])
        highest_qualities = [] #max heap
        
        #go in increasing ratios
        for i in range(N):
            heapq.heappush(highest_qualities, -ratios[i][1])
            curr_total_quality += ratios[i][1]
            
            #if we have too many workers
            if len(highest_qualities) > k:
                curr_total_quality += heapq.heappop(highest_qualities)
            
            #if we have a valid k group, find min
            if len(highest_qualities) == k:
                curr_cost = curr_total_quality*ratios[i][0]
                min_cost = min(min_cost,curr_cost)
        
        return min_cost
        
#########################################
# 2373. Largest Local Values in a Matrix
# 12MAY24
#########################################
class Solution:
    def largestLocal(self, grid: List[List[int]]) -> List[List[int]]:
        '''
        need to partition in smaller 3 by 3 grids
        '''
        N = len(grid)
        ans = []
        for i in range(N-2):
            row_ans = []
            for j in range(N-2):
                max_ = self.getMax(i+1,j+1,grid)
                row_ans.append(max_)
            
            ans.append(row_ans)
        
        return ans
    
    def getMax(self,i,j,grid):
        
        max_ = 0
        for d_i in [-1,0,1]:
            for d_j in [-1,0,1]:
                max_ = max(max_, grid[i + d_i][j + d_j])
        
        return max_
    
##########################################
# 861. Score After Flipping Matrix
# 13MAY24
#########################################
#good idea though...
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        '''
        we can flip bits along rows or along cols, but score is row sum of all the bits
        if we can turn each row to all ones, we know our score is maximized
        does it matter of we flips rows before cols? what about cols before rows?
        0011
        1010
        1100
        
        go down rows, and if it results in abigger number, flip it
        then go across cols, if it results in a bigger number flip it
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        #flip along rows
        for i in range(rows):
            curr_num = 0
            flipped_num = 0
            for j in range(cols):
                curr_num = (curr_num << 1) | grid[i][j]
                flipped_num = (flipped_num << 1) | (0 if grid[i][j] == 1 else 1)
            #if its bigger, flip the bits
            if flipped_num > curr_num:
                for j in range(cols):
                    grid[i][j] = 0 if grid[i][j] == 1 else 1
        
        #flip along cols
        for col in range(cols):
            curr_num = 0
            flipped_num = 0
            for row in range(rows):
                curr_num = (curr_num << 1) | grid[row][col]
                flipped_num = (flipped_num << 1) | (0 if grid[row][col] == 1 else 1)
            if flipped_num > curr_num:
                for row in range(rows):
                    grid[row][col] = 0 if grid[row][col] == 1 else 1
        
        score = 0
        for row in grid:
            num = 0
            for r in row:
                num = (num << 1) | r
            
            score += num
        
        return score

#ezzzzz
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        '''
        maximize rows in the first position (i.e at first col)
        then maximize cols based on the number of zeros and ones
        if we have more zeros flip that entire column
        we dont need to modify row by row then col by col, if we already optimized a row in the first position
        because on the nex steps we optimize cols
        its because the score is based on only the rows
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        #promote in the first col of each row if we can
        for row in range(rows):
            if grid[row][0] == 0:
                for col in range(cols):
                    grid[row][col] ^= 1
        
        #now optimize columns
        for col in range(1,cols):
            count_zeros = 0
            for row in range(rows):
                count_zeros += grid[row][col] == 0
            
            if count_zeros > rows - count_zeros:
                for row in range(rows):
                    grid[row][col] ^= 1
        
        score = 0
        for row in grid:
            num = 0
            for r in row:
                num = (num << 1) | r
            
            score += num
        
        return score

#in place
class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        '''
        we don't need to overwrite the grid, we can just cant on the fly and accumulate the bits in our ans
        for the first col of each row, we need to promote to 1
        so we just add (1 << first position), since they are going to be one anway
        break up the numbers into its parts, for exmple:
            20 + 12 = 10 + 10 + 6 + 6, as long as the sums add up
            
        now for cols, we just need to count up the bits that we change
        if the first element in a particular row is 0, it means this rows been flipped, becasue we made it 1
        first_elem  curr_elem   curr_elem(after flipping)
        0           0           1
        0           1           0
        1           0           0
        1           1           1
        
        count_zeros after fliping is when they aren't the same
        so we count of up the ones, then we need to check if flipping to get all these ones is profitable
        we either use all the current ones or the ones we get from flipping (we take the max of these two)
        the col_score is contributed base on the position k
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        score = 0
        for row in range(rows):
            score += (1 << (cols - 1))
        
        #now optimize columns
        for col in range(1,cols):
            count_same_bits = 0
            for row in range(rows):
                count_same_bits += grid[row][col] == grid[row][0]
            
            count_same_bits = max(count_same_bits,rows - count_same_bits)
            column_score = (1 << (cols - col - 1))*count_same_bits
            score += column_score

        return score