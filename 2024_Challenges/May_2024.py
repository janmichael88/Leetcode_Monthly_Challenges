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
    
######################################
# 1219. Path with Maximum Gold
# 14MAY24
#######################################
#ezzz
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        '''
        back tracking with global maximum value
        '''
        self.ans = 0
        rows = len(grid)
        cols = len(grid[0])
           
        def rec(i,j,gold,seen):
            self.ans = max(self.ans,gold)
            seen.add((i,j))
            for di,dj in [[1,0],[-1,0],[0,1],[0,-1]]:
                neigh_i = i + di
                neigh_j = j + dj
                if 0 <= neigh_i < rows and 0 <= neigh_j < cols:
                    if (neigh_i,neigh_j) not in seen and grid[neigh_i][neigh_j] != 0:
                        rec(neigh_i,neigh_j,gold + grid[neigh_i][neigh_j],seen)
                        seen.remove((neigh_i,neigh_j))
                
            return
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != 0:
                    seen = set()
                    rec(i,j,grid[i][j],seen)
        
        return self.ans

#no global
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        '''
        we can also keep it as no global variable and inplace grid to mark as visited
        base case it when are outside the bounds, there's no gold here, so return 0
        also for directions we can do
        dirrs = [0,1,0,-1,0]
            then we can get all pairs doing [i,i+1]
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [0,1,0,-1,0]
        def rec(i,j,grid,rows,cols):
            #out of bounds
            if not (0 <= i < rows) or not (0 <= j < cols):
                return 0
            #if zero cell/visited
            if grid[i][j] == 0:
                return 0
            
            max_gold = 0
            #mark
            curr_gold = grid[i][j]
            grid[i][j] = 0
            
            for d in range(4):
                neigh_i = i + dirrs[d]
                neigh_j = j + dirrs[d+1]
                #maximize
                max_gold = max(max_gold, rec(neigh_i,neigh_j,grid,rows,cols) + curr_gold)
            
            #backtrack
            grid[i][j] = curr_gold
            return max_gold
        
        ans = 0
        for i in range(rows):
            for j in range(cols):
                ans = max(ans,rec(i,j,grid,rows,cols))
        
        return ans
    
class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:
        DIRECTIONS = [0, 1, 0, -1, 0]
        rows = len(grid)
        cols = len(grid[0])
        
        def bfs_backtrack(row: int, col: int) -> int:
            queue = deque()
            visited = set()
            max_gold = 0
            visited.add((row, col))
            queue.append((row, col, grid[row][col], visited))
            while queue:
                curr_row, curr_col, curr_gold, curr_vis = queue.popleft()
                max_gold = max(max_gold, curr_gold)

                # Search for gold in each of the 4 neighbor cells
                for direction in range(4):
                    next_row = curr_row + DIRECTIONS[direction]
                    next_col = curr_col + DIRECTIONS[direction + 1]

                    # If the next cell is in the matrix, has gold, 
                    # and has not been visited, add it to the queue
                    if 0 <= next_row < rows and 0 <= next_col < cols and \
                            grid[next_row][next_col] != 0 and \
                            (next_row, next_col) not in curr_vis:
                        curr_vis.add((next_row, next_col))
                        queue.append((next_row, next_col, 
                                      curr_gold + grid[next_row][next_col], 
                                      curr_vis.copy()))
                        curr_vis.remove((next_row, next_col))
            return max_gold

        # Find the total amount of gold in the grid
        total_gold = sum(sum(row) for row in grid)
        
        # Search for the path with the maximum gold starting from each cell
        max_gold = 0
        for row in range(rows):
            for col in range(cols):
                if grid[row][col] != 0:
                    max_gold = max(max_gold, bfs_backtrack(row, col))
                    # If we found a path with the total gold, it's the max gold
                    if max_gold == total_gold:
                        return total_gold
        return max_gold

################################################
# 2812. Find the Safest Path in a Grid (REVISTED)
# 16MAY24
################################################
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        '''
        need to find safness factor for each cell
            multipoint bfs and get distances
        then we can check if there is a path this some safeness factor
            binary search for workable solution
            
        '''
        dists = self.findDists(grid)
        left,right,res = 0,0,-1
        for i in range(len(dists)):
            for j in range(len(dists[0])):
                right = max(right,dists[i][j])
        
        while left <= right:
            mid = left + (right - left)//2
            seen = set()
            if self.dfsWithSafness(0,0,dists,seen,mid):
                res = mid 
                left = mid + 1
            else:
                right = mid - 1
        
        return res
            
    
    def dfsWithSafness(self,curr_row,curr_col,grid,seen,v):
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [0,1,0,-1,0]
        if grid[0][0] < v or grid[rows-1][cols-1] < v:
            return False
        if (curr_row,curr_col) == (rows-1,cols-1):
            return True
        seen.add((curr_row,curr_col))
        for i in range(4):
            neigh_row = curr_row + dirrs[i]
            neigh_col = curr_col + dirrs[i+1] 
            if (neigh_row,neigh_col) in seen:
                continue
            if 0 <= neigh_row < rows and 0 <= neigh_col < cols and grid[neigh_row][neigh_col] >= v:
                if self.dfsWithSafness(neigh_row,neigh_col,grid,seen,v):
                    return True
        
        return False
        
        
    def findDists(self,grid):
        rows = len(grid)
        cols = len(grid[0])
        dists = [[float('inf')]*cols for _ in range(rows)]
        dirrs = [0,1,0,-1,0]
        q = deque([])
        #find safness
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    q.append((i,j,0))
        
        while q:
            curr_row,curr_col,curr_dist = q.popleft()
            dists[curr_row][curr_col] = curr_dist
            
            for i in range(4):
                neigh_row = curr_row + dirrs[i]
                neigh_col = curr_col + dirrs[i+1]
                if 0 <= neigh_row < rows and 0 <= neigh_col < cols and dists[neigh_row][neigh_col] > curr_dist:
                    q.append((neigh_row,neigh_col,curr_dist+1))
        
        return dists
            
#using djikstras
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        '''
        we can also use djikstras, note finding longest path in undirected graph is NP-hard
            paths can be infinitely long (going back and forth between nodes)
        
        but we can use it to find longest path in DAG, which is the case for this problem
        '''
        rows = len(grid)
        cols = len(grid)
        dists = self.findDists(grid)
        dirrs = [0,1,0,-1,0]
        pq_dists = [[float('inf')]*cols for _ in range(rows)]
        seen = set()
        pq = []
        pq.append((-dists[0][0],0,0)) #maxheap to get largest safeness factors first, then minimize in pq_dists
        while pq:
            curr_safeness,curr_row,curr_col = heapq.heappop(pq)
            pq_dists[curr_row][curr_col] = min(pq_dists[curr_row][curr_col],-curr_safeness)
            seen.add((curr_row,curr_col))
            for i in range(4):
                neigh_row = curr_row + dirrs[i]
                neigh_col = curr_col + dirrs[i+1]
                if 0 <= neigh_row < rows and 0 <= neigh_col < cols and (neigh_row,neigh_col) not in seen:
                    entry = (-dists[neigh_row][neigh_col], neigh_row,neigh_col)
                    heapq.heappush(pq,entry)
        
        return pq_dists[rows-1][cols-1]
            
    def findDists(self,grid):
        rows = len(grid)
        cols = len(grid[0])
        dists = [[float('inf')]*cols for _ in range(rows)]
        dirrs = [0,1,0,-1,0]
        q = deque([])
        #find safness
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    q.append((i,j,0))
        
        while q:
            curr_row,curr_col,curr_dist = q.popleft()
            dists[curr_row][curr_col] = curr_dist
            
            for i in range(4):
                neigh_row = curr_row + dirrs[i]
                neigh_col = curr_col + dirrs[i+1]
                if 0 <= neigh_row < rows and 0 <= neigh_col < cols and dists[neigh_row][neigh_col] > curr_dist:
                    q.append((neigh_row,neigh_col,curr_dist+1))
        
        return dists


##################################################
# 2128. Remove All Ones With Row and Column Flips
# 13MAY24
##################################################
class Solution:
    def removeOnes(self, grid: List[List[int]]) -> bool:
        '''
        on the last operation, it had to be the case that there was all 1s in a col or all 1s in a row
        doing more the one operation on a row or col jsut flips it back, so we dont need to flip more than once (if we do flip)
        for each row and col
        if counts ones
        pattern of each row should be the same as the first row or its invers
        '''
        pattern = grid[0]
        pattern_inv = [1 - v for v in grid[0]]
        for r in grid[1:]:
            if r != pattern and r != pattern_inv:
                return False
        
        return True
            
class Solution:
    def removeOnes(self, grid: List[List[int]]) -> bool:
        '''
        the actual way is to flip a row, if leading column is zero
        then count up the column values
            each col should be either all zero or call ones
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        for col in range(cols):
            if grid[0][col] == 1:
                for row in range(rows):
                    grid[row][col] = 1 - grid[row][col]
                    
        #check rows all zeros
        for row in range(1,rows):
            count_ones = 0
            for col in range(cols):
                count_ones += grid[row][col]
            
            if count_ones != cols and count_ones != 0:
                return False
        
        return True
        
##################################################
# 2061. Number of Spaces Cleaning Robot Cleaned
# 17MAY24
###################################################
#ezzzz bruv
class Solution:
    def numberOfCleanRooms(self, room: List[List[int]]) -> int:
        '''
        directions will be R->D->L->U, and then it just repeats
        when do we stop?
            when the robot reaches a space that it has already cleaned and is facing the same direction as before
            so keep states (i,j,k) (i,j) is cell and k is directions
        '''
        dirrs = [[0,1],[1,0],[0,-1],[-1,0]]
        rows = len(room)
        cols = len(room[0])
        seen = set()
        cleaned_cells = set()
        
        curr_state = (0,0,0)
        while curr_state not in seen:
            seen.add(curr_state)
            curr_row,curr_col,curr_dirr = curr_state
            cleaned_cells.add((curr_row,curr_col))
            #get next move
            for i in range(4):
                next_dirr = (curr_dirr + i) % 4
                d_row,d_col = dirrs[next_dirr]
                next_row = curr_row + d_row
                next_col = curr_col + d_col
                #must not be 1 and in bounds
                if 0 <= next_row < rows and 0 <= next_col < cols and room[next_row][next_col] == 0:
                    curr_state = (next_row,next_col,next_dirr)
                    break
            
        return len(cleaned_cells)

#can also use DFS
class Solution:
    def numberOfCleanRooms(self, room: List[List[int]]) -> int:
        DIRECTIONS = (0, 1, 0, -1, 0)
        rows, cols = len(room), len(room[0])
        visited = set()
        cleaned = set()

        def clean(row, col, direction):
            #alrady seen
            if (row, col, direction) in visited:
                return len(cleaned)

            # mark
            visited.add((row, col, direction))
            cleaned.add((row, col))

            # clean next cell
            next_row = row + DIRECTIONS[direction] 
            next_col = col + DIRECTIONS[direction + 1]
            if 0 <= next_row < rows and 0 <= next_col < cols and not room[next_row][next_col]:
                return clean(next_row, next_col, direction)

            return clean(row, col, (direction + 1) % 4)

        return clean(0, 0, 0)

#bfs
class Solution:
    def numberOfCleanRooms(self, room: List[List[int]]) -> int:
        '''
        we can also use BFS and bit masks for each direction
        2,4,6,8, R,D,L,U
        then we just upate the celss states with 1 << curr_dirr
        if a cell is marked 0, we know it hasn't been visited yet
        '''
        rows = len(room)
        cols = len(room[0])
        dirrs = (0,1,0,-1,0)
        visited = [[0]*cols for _ in range(rows)] #note we could have overwtied the 2d array
        cleaned = 0
        
        q = deque([])
        q.append((0,0,0))
        
        while q:
            c_r, c_c, c_d = q.popleft()
            #un cleand room
            if visited[c_r][c_c] == False:
                cleaned += 1
            #mark this cell with thid direction
            visited[c_r][c_c] |= 1 << c_d
            for i in range(4):
                n_d = (c_d + i) % 4
                n_r = c_r + dirrs[n_d]
                n_c = c_c + dirrs[n_d+1]
                if 0 <= n_r < rows and 0 <= n_c < cols and room[n_r][n_c] == 0:
                    #check direction
                    if (visited[n_r][n_c] >> n_d) & 1:
                        return cleaned
                    else:
                        q.append((n_r,n_c,n_d))
                        break
        
        return cleaned
    
##############################################
# 1325. Delete Leaves With a Given Value
# 17MAY24
#############################################
#Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def removeLeafNodes(self, root: Optional[TreeNode], target: int) -> Optional[TreeNode]:
        '''
        resursion if leaf just return none
        '''
        def rec(node,target):
            if not node:
                return None
            if not node.left and not node.right:
                if node.val == target:
                    return None
                return node
            node.left = rec(node.left,target)
            node.right = rec(node.right,target)
            if not node.left and not node.right:
                if node.val == target:
                    return None

            return node
        
        return rec(root,target)
            
###################################################
# 979. Distribute Coins in Binary Tree (REVISITED)
# 17MAY24
###################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        '''
        number of coins == number of nodes
        represent extra coins as positive values and needed coins as negative values (for node)
        if we examine leaf nodes
        we can calculate the coints exchanged in the subtree rooted ad the a node
        curr.val = curr.val + left_counts + right_counts
        there are three cases for distributing coins from a leaf node
            1. the leaf node doesn't have any coins, so we take from parent
            2. leaf node has exactly one coin, no exchagne here
            3. leaf node has more than coint, leave one to it and move the rest
        
        from leaf node we can determing how to distribute coins -> the only neighbor is its parent!
        so we hand child nodes before parent nodes -> need post order, L,R,N
        at each node, ask, how many coints can the current node pass to its parent
        we could modify the nodes and pass them up, but its easier to pass the exchanging
            try coding modifying solution after
        
        we calculate the number of coins a parent node can pass on to its parent by subtracting one from its value
        then add the number of conts its left and right subtrees need to exahcnage, then add up for all the ndoes
        so dfs(node) passes the number of coins it needs to exhcange from some node
        the answer is just the sum of all the exchanges
        and the function returnes the exhcnage
        '''
        ans = [0]
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            ans[0] += abs(left) + abs(right)
            return (node.val - 1) + left + right
        
        dfs(root)
        return ans[0]
    
#######################################
# 789. Escape The Ghosts
# 18MAY24
#######################################
class Solution:
    def escapeGhosts(self, ghosts: List[List[int]], target: List[int]) -> bool:
        '''
        grid is infinite, cant just BFS and move them all at once -> it may never terminate
        if a single ghost blocks my path, i cant reach it
        there can be multiple ghosts in one location
        
        X 0 0 1
        1 0 0 0 
        0 0 0 0
        0 0 0 0
        
        there just needs to be a time, where i am in the desintations row or col, and no ghosts are blocking me
        for each ghosts find its L2 distance to x, if any are less then my L2, we cann't do it
        '''
        my_l2 = abs(target[0]) + abs(target[1])
        for x,y in ghosts:
            g_l2 = abs(x - target[0]) + abs(y - target[1])
            if g_l2 <= my_l2:
                return False
        return True
    
#ezzz
class Solution:
    def escapeGhosts(self, ghosts: List[List[int]], target: List[int]) -> bool:
        '''
        grid is infinite, cant just BFS and move them all at once -> it may never terminate
        if a single ghost blocks my path, i cant reach it
        there can be multiple ghosts in one location
        
        X 0 0 1
        1 0 0 0 
        0 0 0 0
        0 0 0 0
        
        there just needs to be a time, where i am in the desintations row or col, and no ghosts are blocking me
        for each ghosts find its L2 distance to x, if any are less then my L2, we cann't do it
        '''
        def l2(p,q):
            return abs(p[0] - q[0])  + abs(p[1] - q[1])
        
        for g in ghosts:
            my_dist = l2([0,0],target)
            ghost_dist = l2(g,target)
            if ghost_dist <= my_dist:
                return False
        return True
    
################################################
# 3068. Find the Maximum Sum of Node Values
# 19MAY24
###############################################
#nice try
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        '''
        we have n nodes and we are given n-1 edges
        graph will be a tree, and we have values at each node, given by nums
        maximize sum of node values
        we can perform the following operation
            choes any edge [u,v] and update nums[u] = nums[u] ^ k
        any node in the tree can be the root
        essecntially indices are connected through the edge list
        flipping the edge more than once, just returns it back to the original
        and xoring with itse;f result zero
        a ^ a = 0
        a ^ b ^ b = 0
        a ^ b = b ^ a
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        #flip or dont flip
        memo = {}
        def dp(curr,prev,k):
            if (curr,prev) in memo:
                return memo[(curr,prev)]
            ans = float('-inf')
            flip_sum = 0
            no_flip_sum = 0
            
            for neigh in graph[curr]:
                if neigh == prev:
                    continue
                #flip
                flip_sum = (nums[curr] ^ k) + (nums[neigh] ^ k)
                #dont
                no_flip_sum = (nums[curr] + nums[neigh])
                ans = max(flip_sum,no_flip_sum) + dp(neigh,curr,memo)
            
            ans = max(ans,flip_sum,no_flip_sum)
            memo[(curr,prev)] = ans
            return ans
        
        return max(dp(0,-1,k) + nums[0], dp(0,-1,k) + nums[0] ^ k)
    
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        '''
        its turn out the path doesn't matter
        so we had path form u to v 
        u -> p -> p1 -> p2 ... -> v
        because of the nature of the opertiona the sum would be
        u^k + p1 ^ k + (p1 ^ k + p2 ^ k) ... + (p2 ^ k + v ^ k)
        intutiion:
            if we apply the operations on a path from u to v, the nodes in that path remain unchainged
            only u and v get xord
            so we can choose to apply the operation, but its only valid if we applied the operation an evne number of times
            odd number of times is invalud
        all the p's would remain unchanged, only node vals at u and k would be XORd with k, the internal nodes would remain unchanged
        so eseentially there is a path that conect all thd noes, and it doesn't matter the order which we apply the operation
        and applying the operations more than once on an edge is pointless
        we need to maximize the sum of the values, wheere there is an even number of operations performed
            because operations are done on pairs
        
        base case is when we have gone through all the nodes
            if even ops, return 0, other return float('inf')
        
        states become even and parity of operations (could also do number, but that would be more states)
        parity determines if its a valid assignment (i.e a valid taking of edges to apply the operations)
        '''
        memo = {}
        
        #better way is to use number of operations
        #parity += 1, and check
        def dp(i,parity):
            if i == len(nums):
                if parity % 2 == 0:
                    return 0
                return float('-inf')
            if (i,parity) in memo:
                return memo[(i,parity)]
            no_op = nums[i] + dp(i+1,parity)
            op = (nums[i] ^ k) + dp(i+1, 1 - parity)
            ans = max(no_op,op)
            memo[(i,parity)] = ans
            return ans
        
        return dp(0,0)
    
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        '''
        bottom up
        '''
        n = len(nums)
        dp = [[0]*2 for _ in range(n+1)]
        #base case inf fill
        for i in range(n+1):
            for parity in range(2):
                if i == n and parity % 2 == 1:
                    dp[i][parity] = float('-inf')
        
        for i in range(n-1,-1,-1):
            for parity in range(2):
                no_op = nums[i] + dp[i+1][parity]
                op = (nums[i] ^ k) + dp[i+1][1 - parity]
                ans = max(no_op,op)
                dp[i][parity] = ans
        
        return dp[0][0]
    
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        '''
        if we do an operation at some node u
        nums[u] = nums[u] ^ k, we can do this for every node and find the net change
        netChage[u] = (nums[u] ^ k) - (nums[u])
        we want the apply the aoperations that have a bigger (positive net change)
        chose opertions that provide the greaest increment to node sum
        we need to do them in pairs,
            all the nodes are connected
            we never needed the edges anyway
            
        our smallest sum will just be the sum of all nums
        so we only need to find the positive changes
        '''
        N = len(nums)
        change = []
        for i in range(N):
            delta = (nums[i] ^ k) - nums[i]
            change.append(delta)
        
        change.sort(reverse = True)
        node_sum = sum(nums)
        
        for i in range(0,N,2):
            if i + 1 < N:
                pair_sum = change[i] + change[i+1]
                if pair_sum > 0:
                    node_sum += pair_sum
        
        return node_sum
    
######################################
# 1863. Sum of All Subset XOR Totals
# 20MAY24
#####################################
#recursion
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        #recursion
        N = len(nums)
        
        def rec(i,curr_sum):
            if i >= N:
                return curr_sum
            
            xor = rec(i+1, curr_sum^nums[i])
            noxor = rec(i+1,curr_sum)
            return xor + noxor
        
        return rec(0,0)
    
#bit magic
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        '''
        idea is to count up the number of times each bit it set in each subset
        example
        nums = [1,2], ans = 6 ->110
        nums = [5,1,6] ans = 26 -> 11100
        right most bits are 0
        try breaking rule
        nums = [5,20] ans = 42 -> 101010
        
        need way to determine most significant bits
        any set bit in num, is set in the out put
        '''
        ans = 0
        N = len(nums)
        for num in nums:
            for i in range(32):
                #check if set
                mask = 1 << i
                if num & mask != 0:
                    ans |= mask
        
        return ans << (N-1) #need to append N-1 zeros to the answer
    
class Solution:
    def subsetXORSum(self, nums: List[int]) -> int:
        '''
        idea is to count up the number of times each bit it set in each subset
        example
        nums = [1,2], ans = 6 ->110
        nums = [5,1,6] ans = 26 -> 11100
        right most bits are 0
        try breaking rule
        nums = [5,20] ans = 42 -> 101010
        
        need way to determine most significant bits
        any set bit in num, is set in the out put
        intution, all th binary representations fo the XOR subsets will have each set bi appear 2^(n-1) times
        for a bit position to be set in the subset XOR total, it must be set in an odd number of elements
        a given element will be included in half of the subsets
        if we have 2**N subsets, then it will be set in 2**(N-1) of them
        
        for a given bit position x, how many subset XOR totals have the xth bit set
            if not set anywhere, it will remain unset in the XOR totals
            if set somewhere it will be set 2**(N-1) times
        '''
        ans = 0
        xor_sum = 0
        N = len(nums)
        for num in nums:
            for i in range(32):
                #check if set
                mask = 1 << i
                if num & mask != 0:
                    ans |= mask
        
        #in ans mask, each set bit will apear 2**(N-1) times
        for i in range(32):
            mask = 1 << i
            if ans & mask:
                xor_sum += mask*(1 << len(nums)-1)
        
        return xor_sum
    
###################################
# 78. Subsets (REVISTED)
# 21MAY24
##################################
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        '''
        just use recursion
        '''
        
        def rec(i):
            if i >= len(nums):
                return [[]]
            first_num = nums[i]
            
            subsets_without = rec(i+1)
            subsets_with = []
            for s in subsets_without:
                temp = [first_num] + s
                subsets_with.append(temp)
            
            return subsets_without + subsets_with
        
        return rec(0)
    
###################################################
# 2044. Count Number of Maximum Bitwise-OR Subsets
# 21MAY24
##################################################
class Solution:
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        '''
        generate subsets with mask ans find xors
        '''
        entries = []
        N = len(nums)
        max_OR = 0
        for mask in range(2**N):
            OR = 0
            for i in range(N):
                if mask & (1 << i):
                    OR = OR | nums[i]
            
            entries.append((mask,OR))
            max_OR = max(max_OR,OR)
        
        ans = set()
        for mask,OR in entries:
            if OR == max_OR:
                ans.add(mask)
        
        return len(ans)
    
#knap sack with bitmasks solution
class Solution:
    def countMaxOrSubsets(self, nums: List[int]) -> int:
        '''
        we can actually find max OR, and check each subset
        sum(nums) is max and accumulating OR is also max
        '''
        entries = []
        N = len(nums)
        max_OR = 0
        for num in nums:
            max_OR |= num
        ans = 0
        
        for mask in range(2**N):
            OR = 0
            for i in range(N):
                if mask & (1 << i):
                    OR = OR | nums[i]
            if OR == max_OR:
                ans += 1
        
        return ans
        
    
##########################################
# 131. Palindrome Partitioning (REVISTED)
# 22MAY24
#########################################
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''
        just try all partitions
        '''
        ans = []
        N = len(s)
        
        def rec(i,path):
            if i >= N:
                ans.append(path[:])
                return
            for j in range(i+1,N+1):
                temp = s[i:j]
                if temp == temp[::-1]:
                    rec(j,path + [temp])
                    
        rec(0,[])
        return ans
    
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''
        we can also pass reference to substrings
        '''
        ans = []
        N = len(s)
        
        def rec(s,path):
            if not s:
                ans.append(path[:])
                return
            for i in range(1,len(s)+1):
                temp = s[:i]
                if temp == temp[::-1]:
                    rec(s[i:],path + [temp])
                    
        rec(s,[])
        return ans

#########################################
# 1745. Palindrome Partitioning IV
# 22MAY24
########################################
#almost!
#right idea, need to build it up the right way first
class Solution:
    def checkPartitioning(self, s: str) -> bool:
        '''
        pre-process checking palindromes
        then fix the start and ends, and check inside for a palindrome
        '''
        N = len(s)
        memo = {}
        for i in range(N):
            for j in range(N-1,i-1,-1):
                if (i == j):
                    memo[(i,j)] = True
                else:
                    self.dp(i,j,memo,s)
        for i in range(N+1):
            for j in range(N-2,i,-1):
                pref = memo[(0,i)]
                #print(s[:i+1], pref)
                middle = memo[(i+1,j)]
                suff = memo[(j+1,N-1)]
                if sum([pref,middle,suff]) == 3:
                    return True
        return False
                
        

    def dp(self, i,j,memo,s):
        if i >= j:
            return True
        if (i,j) in memo:
            return memo[(i,j)]
        if s[i] == s[j] and self.dp(i+1,j-1,memo,s):
            memo[(i,j)] = True
            return True
        memo[(i,j)] = False
        return False
        
class Solution:
    def checkPartitioning(self, s: str) -> bool:
        '''
        pre-process checking palindromes
        then fix the start and ends, and check inside for a palindrome
        '''
        N = len(s)
        dp = [[False]*(N) for _ in range(N)]
        
        for i in range(N-1,-1,-1):
            for j in range(N):
                if i >= j:
                    dp[i][j] = True
                elif s[i] == s[j]:
                    dp[i][j] = dp[i+1][j-1] 
                    
        for i in range(N+1):
            for j in range(N-2,i,-1):
                pref = dp[0][i]
                #print(s[:i+1], pref)
                middle = dp[i+1][j]
                suff = dp[j+1][N-1]
                if sum([pref,middle,suff]) == 3:
                    return True
        return False

########################################
# 2597. The Number of Beautiful Subsets
# 23MAY24
#######################################
class Solution:
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        '''
        genearting all subsets fits in TC, but need fast way of determining if absolute difference is allowable
        we need to do counts, because there could be repeated elements!
        '''
        nums.sort()
        N = len(nums)
        curr_subset = defaultdict(int)
        ans = [0]
        def rec(i,curr_subset):
            if i >= N:
                ans[0] += 1
                return
            #we always need to skip
            rec(i+1,curr_subset)
            prev = nums[i] - k
            if prev not in curr_subset:
                curr_subset[nums[i]] += 1
                rec(i+1,curr_subset)
                curr_subset[nums[i]] -= 1
                if curr_subset[nums[i]] == 0:
                    del curr_subset[nums[i]]
        
        rec(0,curr_subset)
        return ans[0] - 1

class Solution:
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        '''
        no global solution
        '''
        nums.sort()
        N = len(nums)
        curr_subset = defaultdict(int)
        def rec(i,curr_subset):
            if i >= N:
                return 1
            #we always need to skip
            skip = rec(i+1,curr_subset)
            prev = nums[i] - k
            no_skip = 0
            if prev not in curr_subset:
                curr_subset[nums[i]] += 1
                no_skip = rec(i+1,curr_subset)
                curr_subset[nums[i]] -= 1
                if curr_subset[nums[i]] == 0:
                    del curr_subset[nums[i]]
            return skip + no_skip
        return rec(0,curr_subset) - 1
    
#recursion tree (count up n-nary tree problem)
class Solution:
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        '''
        we can examine all subsets using mask
        keep index i and current mask, and check that adding this index to this subset is allowed
        then count up the ways, benefit of this solution is that we dont need to sort
        dont forget counting in n-ary tree recursino problems
        '''
        N = len(nums)
        
        def rec(i,mask,k):
            if i >= N:
                if mask != 0:
                    return 1
                return 0
            
            can_add = True
            #check all in curr mask
            for j in range(N):
                pos_mask = (1 << j)
                if (mask & pos_mask == 0) or abs(nums[i] - nums[j]) != k:
                    continue
                else:
                    can_add = False
                    break
            
            skip = rec(i+1,mask,k)
            no_skip = 0
            if can_add:
                #take i
                next_mask = mask | (1 << i)
                no_skip = rec(i+1, next_mask,k)
            
            return skip + no_skip
        
        return rec(0,0,k)
    
##############################################
# 1255. Maximum Score Words Formed by Letters
# 24MAY24
############################################
#not the most effecitny but it works
class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        '''
        need to find max score for any valid set of words
        find the possible subsets we can create using the letters, then find the max score for of all subsets
        there aren't that many subsets
        '''
        ans = 0
        letters = Counter(letters)
        N = len(words)
        for subset in range(2**N):
            curr_subset = []
            for i in range(N):
                pos_mask = (1 << i)
                if (subset & pos_mask):
                    curr_subset.append(i)
            
            curr_score = 0
            is_valid = True
            curr_letters = Counter()
            for i in curr_subset:
                for ch in words[i]:
                    curr_letters[ch] += 1
            
            for ch,cnt in curr_letters.items():
                if cnt > letters[ch]:
                    is_valid = False
                    break
                else:
                    idx = ord(ch) - ord('a')
                    curr_score += score[idx]*cnt
            
            if is_valid:
                ans = max(ans,curr_score)
        
        return ans
    
#anohter iterative way
class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        W = len(words)
        # Count how many times each letter occurs
        freq = [0 for i in range(26)]
        for c in letters:
            freq[ord(c) - 97] += 1

        # Calculate score of subset
        def subset_score(subset_letters, score, freq):
            total_score = 0
            for c in range(26):
                total_score += subset_letters[c] * score[c]
                # Check if we have enough of each letter to build this subset of words
                if subset_letters[c] > freq[c]:
                    return 0
            return total_score

        max_score = 0
        # Iterate over every subset of words
        subset_letters = {}
        for mask in range(1 << W):
            # Reset the subset_letters map
            subset_letters = [0 for i in range(26)]
            # Find words in this subset
            for i in range(W):
                if (mask & (1 << i)) > 0:
                    # Count the letters in this word
                    L = len(words[i])
                    for j in range(L):
                        subset_letters[ord(words[i][j]) - 97] += 1
            # Calculate score of subset
            max_score = max(max_score, subset_score(subset_letters, score, freq))
        # Return max_score as the result
        return max_score
    
class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        '''
        backtracking, pass in index, subset letters and curr score
        '''
        ans = [0]
        N = len(words)
        all_letters = [0]*26
        for ch in letters:
            all_letters[ord(ch) - ord('a')] += 1
        subset_letters = [0]*26
        
        self.backtrack(0,words,subset_letters,score,all_letters,ans,0)
        return ans[0]
    
    def backtrack(self,i,words,subset_letters,score,all_letters,ans,total_score):
        print(total_score)
        if i >= len(words):
            ans[0] = max(ans[0],total_score)
            return
        
        self.backtrack(i+1,words,subset_letters,score,all_letters,ans,total_score)
        curr_word = words[i]
        for ch in curr_word:
            subset_letters[ord(ch) - ord('a')] += 1
            total_score += score[ord(ch) - ord('a')]
            
        if self.is_valid_subset(subset_letters,all_letters):
            self.backtrack(i+1,words,subset_letters,score,all_letters,ans,total_score)
        #backtrack
        for ch in curr_word:
            subset_letters[ord(ch) - ord('a')] -= 1
            total_score -= score[ord(ch) - ord('a')]
            
    
    def is_valid_subset(self, subset_letters, all_letters):
        for i in range(26):
            if all_letters[i] < subset_letters[i]:
                return False
        
        return True

###########################################
# 140. Word Break II (REVISTED)
# 25MAY24
############################################
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        '''
        backtracking
        if we can make a word, break it and add to is path
        '''
        wordDict = set(wordDict)
        ans = set()
        
        def rec(i,s,ans,path,seen):
            if i >= len(s):
                ans.add(path[1:])
                return
            for j in range(i,len(s)+1):
                temp = s[i:j+1]
                if temp in seen:
                    rec(j+1,s,ans,path+" "+temp,seen)
        
        rec(0,s,ans,"",wordDict)
        return ans
    
#bactrakcing if we want too
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        '''
        backtracking
        if we can make a word, break it and add to is path
        '''
        wordDict = set(wordDict)
        ans = []
        
        def rec(i,s,ans,path,seen):
            if i == len(s):
                ans.append(" ".join(path))
                return
            for j in range(i,len(s)+1):
                temp = s[i:j+1]
                if temp in seen:
                    path.append(temp)
                    rec(j+1,s,ans,path,seen)
                    path.pop()
        
        rec(0,s,ans,[],wordDict)
        return ans
                
#we can cache states if we pass in string instead of string suffix
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        '''
        backtracking
        if we can make a word, break it and add to is path
        '''
        wordDict = set(wordDict)
        memo = {}
        
        def rec(i,s,memo,seen):
            if i == len(s):
                return [""]
            if i in memo:
                return memo[i]
            ans = []
            for j in range(i,len(s)+1):
                temp = s[i:j+1]
                if temp in seen:
                    for child_ans in rec(j+1,s,memo,seen):
                        entry = temp + (" " if child_ans else "") + child_ans
                        ans.append(entry)
            memo[i] = ans
            return ans
        
        return rec(0,s,memo,wordDict)
                