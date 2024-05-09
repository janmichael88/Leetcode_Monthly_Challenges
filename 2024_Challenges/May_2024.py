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