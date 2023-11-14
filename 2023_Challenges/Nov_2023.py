####################################################
#1916. Count Ways to Build Rooms in an Ant Colony
# 01NOV23
###################################################
import math #needed for comb
class Solution:
    def waysToBuildRooms(self, prevRoom: List[int]) -> int:
        '''
        hints, dp(i) is the number of ways to solve the problem for node i
        dp(i) = multiplications of the number of ways to distribute the subtrees of the childrren of i
        we want the number of ways do the root 0
        question really turns out to be how many topological orderings are there for a directed tree
        start off with this solution:
            consider a parent node with two children, x and y
            in x, there are x_edges, and in y there are y_edges
            we need to followe some order tp build x_edge and y_edges, but as lone as we follow we can mix edges
            Example: X1 -> X2 -> X3; Y1 -> Y2 -> Y3.
            It can be either: X1 -> X2 -> Y1 -> Y2 -> Y3 -> X3 or X1 -> Y1 -> Y2 -> X2 -> Y3 -> X3 or many others.
        
        total number of combinations is combinations(x_edges + y_edges,x_edges)
        but since x and y are subtrees, we can also build them in different ways
            so it becomes combinations(x_edges + y_edges,x_edges)*dp(x)*dp(y)
        '''
        mod = 10**9 + 7
        graph = defaultdict(list)
        N = len(prevRoom)
        for i in range(1,N):
            graph[prevRoom[i]].append(i)
            
        #count edges, you could use this to get edges in each subtree
        def dfs(i):
            edges = 0
            for neigh in graph[i]:
                edges += dfs(neigh)
            
            return edges + 1
        
        #print(dfs(0))
        def dp(i):
            #returns edges in subtree and ways for this node (edges,ways)
            edges = 0
            ways = 1
            for neigh in graph[i]: #accumulate products into node i for all children of the current i
                child_edges,child_ways = dp(neigh)
                edges += child_edges
                ways *= comb(edges,child_edges) % mod
                ways *= child_ways % mod
                ways %= mod
            
            return (edges+1,ways)
        
        edges,ways = dp(0)
        print(edges,ways)
        return ways % mod
    
#another way
import math
class Solution:
    def waysToBuildRooms(self, prevRoom: List[int]) -> int:
        '''
        there are some caveats to this problem
        recursion is trciky, the hard part doing fast calculation of nCr under modulo operations
        nCr requires some divisions, we have:
            (a*b) % MOD = (a % MOD)*(b % MOD)
            but (a/b) % MOD != (a % MOD) / (b % MOD), we need to use the modular multiplicative inverse
            so to find : (a / b ) % MOD = (a * (1/b)) % MOD
            we need (1/b) % MOD = pow(a, MOD-2) % MOD -> fermats little theorm
            and repeates exponeniotn is slow for large numbers
            so we need to fast powwer function
            
        def fast_pow(n,p):
            res = 1
            while p:
                if (p % 2):
                    res *= n
                    res % MOD
                p //= 2 #or p >>= 1
                n = n*n % MOD
            
            return res
            
        https://leetcode.com/problems/count-ways-to-build-rooms-in-an-ant-colony/discuss/1299599/Python-dfs-how-to-avoid-TLE-discussion
        you can avoid using combs with divions, by computing the binomial coefficients using pascal's triangle, but given the contraints, this really shouldn't be allowed
        PYTHON cheats, very good at mutiplying long numbers
        '''
        mod = 10**9 + 7
        m = 10**5 #number of nodes
        facts = [1]*(m+1)
        for i in range(1,m+1):
            facts[i] = (facts[i-1]*i) % mod
        
        facts_inv = [1]*(m) + [pow(facts[m],mod -2,mod)]
        for i in range(m-1,0,-1):
            facts_inv[i] = facts_inv[i+1]*(i+1) % mod
        
        graph = defaultdict(list)
        N = len(prevRoom)
        for i in range(1,N):
            graph[prevRoom[i]].append(i)
            
        
        def dp(i):
            ways,counts = 1,1
            for neigh in graph[i]:
                child_ways, child_counts = dp(neigh)
                ways = (ways * facts_inv[child_counts]*child_ways) % mod
                counts  += child_counts
            
            return [(ways*facts[counts-1]) % mod, counts]
        
        return dp(0)[0]

#######################################################
# 2265. Count Nodes Equal to Average of Subtree
# 02NOV23
#######################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfSubtree(self, root: Optional[TreeNode]) -> int:
        '''
        need to count the number of nodes in a tree and the sum of elements in tree
        if sum_elements / sum_nodes == node.val, increment ans by 1
        '''
        self.ans = 0
        
        def dp(node):
            if not node:
                return [0,0]
            
            count_left,sum_left = dp(node.left)
            count_right,sum_right = dp(node.right)
            sum_root = node.val + sum_left + sum_right
            count_root = 1 + count_left + count_right
            if node.val == sum_root // count_root:
                self.ans += 1
            
            return [count_root,sum_root]
        
        root_sum, root_count = dp(root)
        return self.ans
    
#######################################################
# 1441. Build an Array With Stack Operations
# 03NOV23
#######################################################
class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        '''
        rules for operations
            * we are given intergers [1,n], inclusive, its a stream
            * if stream of the integers is not empty, pick new integer from stream and push it to the top
            * if stack is not empty, pop from stack
            * if at any moment, elements in stack (from top to bottom) == target, do not read new integers from the stream and do do moe operations on the stack
            
        return stack operations needed
        hint:
            1. use push for number to be kept in array, and push pop for number to be discarded
        is numbers in target are just increasing, itst just push
        for each number in array find the difference
        if difference is 1, it just push 
        otherwise is (diff -2)*(push,pop)
        
        tricky stack problem
        '''
        #end dont matter its just the beginning
        #set beginning state
        #add zero  and n + 1 t0 the ends
        target = [0] + target + [n+1]
        ans = []
        for i in range(1,len(target)-1):
            diff = target[i] - target[i-1]
            if diff > 1:
                for _ in range(diff-1):
                    ans.append('Push')
                    ans.append('Pop')
                    #ans.extend(['Push','Pop'])
            ans.append('Push')
        
        return ans
    
class Solution:
    def buildArray(self, target: List[int], n: int) -> List[str]:
        '''
        we want to pop every number that does not appear in target and should never pop a number that does appear in target
        '''
        ans = []
        start = 0
        for num in target:
            while start < num - 1:
                ans.extend(['Push', 'Pop'])
                start += 1
            start += 1
            ans.append('Push')
        
        return ans
                
###########################################
# 1057. Campus Bikes (REVISITED)
# 03NOV23
###########################################
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        '''
        we have n workes and m bikes, n <= m, i.e we always have enough bikes to cover workers
        hints say we need to used sorting
        entrys (dist,worker_idx,bike_idx)
        sort on this, three way sort
        smallest worker,index and smallest bik index
        '''
        entries = []
        for i in range(len(workers)):
            for j in range(len(bikes)):
                dist = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1])
                entry = (dist,i,j)
                entries.append(entry)
        
        entries.sort()
        ans = [-1]*len(workers)
        needed = 0
        
        #need to keep boolean array for bikes taken
        bikes_taken = [False]*len(bikes)
        for dist,i,j in entries:
            if needed == len(workers):
                return ans
            if ans[i] == -1 and bikes_taken[j] == False:
                ans[i] = j
                needed += 1
                bikes_taken[j] = True
        
        return ans
    
#bucket sort
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        '''
        we can use bucket sort closed and known range
        the coordinate for both bike and worker are in the range [0,1000), so the max range is 1998, 
            rather entries are (0,0) to (999,999)
        
        we can use bucket sort, then just interate over the distancces
        notes:
            goal was to order pairs accoirding to distance first, worker index second, and thne bike index last
            we got distances anyway by going from worker index 0 to len(workers) 
            and then bike index from 0 to len(bikes) we are are sorted anway

        '''
        min_dist = float('inf')
        dist_to_pairs = collections.defaultdict(list)
        for i in range(len(workers)):
            for j in range(len(bikes)):
                dist = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1])
                dist_to_pairs[dist].append((i,j))
                min_dist = min(min_dist, dist)
                
        #we could loop through all distances, but we are bounded by pairs, we do pair loopin invariant rathern than distance iteration
        curr_dist = min_dist
        bikes_taken = [False]*len(bikes)
        workers_assigned = [-1]*len(workers)
        pairs = 0
        
        while pairs < len(workers):
            for w,b in dist_to_pairs[curr_dist]:
                #work and bike not assigned
                if not bikes_taken[b] and workers_assigned[w] == -1:
                    bikes_taken[b] = True
                    workers_assigned[w] = b
                    pairs += 1
            
            curr_dist += 1
    
        return workers_assigned
    
#pq
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        '''
        we can also use min heap
            for a worker, push the min distance to a bike from the work into the heap
            for the other (worker bike pairs), we same, but maintain some sorted order
            in this way dont need to keep all (worker,bike) pairs, just the pairs that are closest
        
        when going through the heap,
            if the bike hasn't been assigned to a work, assign it
            otherwise fetch from the other pairs we have saved
        '''
        min_heap = []
        #other pairs
        other_pairs = []
        for i in range(len(workers)):
            curr_pairs = []
            for j in range(len(bikes)):
                dist = abs(workers[i][0] - bikes[j][0]) + abs(workers[i][1] - bikes[j][1])
                entry = (dist,i,j)
                curr_pairs.append(entry)
            
            #sort
            curr_pairs.sort(reverse = True)
            heapq.heappush(min_heap,curr_pairs.pop())
            other_pairs.append(curr_pairs) #we we will have Array<Integer> len(workers) times
                
        #we could loop through all distances, but we are bounded by pairs, we do pair loopin invariant rathern than distance iteration
        bikes_taken = [False]*len(bikes)
        workers_assigned = [-1]*len(workers)
        
        while min_heap:
            dist,w,b = heapq.heappop(min_heap)
            if not bikes_taken[b] and workers_assigned[w] == -1:
                bikes_taken[b] = True
                workers_assigned[w] = b
            else:
                heapq.heappush(min_heap, other_pairs[w].pop())
    
        return workers_assigned

#######################################################
# 1503. Last Moment Before All Ants Fall Out of a Plank
# 04NOV23
#######################################################
class Solution:
    def getLastMoment(self, n: int, left: List[int], right: List[int]) -> int:
        '''
        given plank of length n, and two arrays left and right
        left contains indicis of ants that are going left same with right
        each at moves at 1 unit per unit time 
        when ants collide they switch directions
        return time when they fall out of the plank
            reather last moment when the last ant falls off the plank
        
        if i have ant going at i going left, and j going right
        
        importnant for transistion:
            they switch directions and both ants are at (j+i) // 2
        
        turns out ants colliding is the same as ants moving past each other
        i can see the case where two ants from opposite directions hitting being the case
        good strategy, think of a simple example and find pattern
        
        hints unfortunaely gave it away... :(
            ans is just the max distance for one ant to reach the end

        derivation
        we have ant at i going right and and at j going left and j-i > 0
        intersectino would be at (i + (j-i)/2) in (j - i)/2 unit time
        at intsection the flip and have to walk the same distance in the same time
        '''
        ans = 0
        
        #distance for an ant going left to walk off 0
        for l in left:
            ans = max(ans, l)
        
        #distance for an ant walking off n
        for r in right:
            ans = max(ans,n-r)
        
        return ans
        
##############################################
# 1535. Find the Winner of an Array Game
# 05NOV23
###############################################
class Solution:
    def getWinner(self, arr: List[int], k: int) -> int:
        '''
        looks like it simulation week
        given array of distinct integers
        if arr[0] > arr[1], arr[0] stays theres and arr[1] goest to eht end
        the array just shifts to the left
        return which integer will win the game after winning k consecutiv times
        k is too big to simulate
        
        the array rotates left
        say we have first two as (i,j), i > j, j gets sent back and shifts
        i, will continue to win of the array is decreasing and i+1 < i
         
        if k is bigger than the length of the array, just return max
        we cant simply return max if k < len(arr), 
            because there may be elements that win k consectuive times before the max is promoted to the first element in the array
        we cant simply keep playing rounds until we get to k consective wins
            k is can be too big, up to a billion
        
        
        '''
        N = len(arr)
        if k >= N:
            return max(arr)
        #simulate, need to effencitenly rotate
        #used deque
        q = deque(arr)
        curr_winner = -1
        curr_streak = 0
        
        while q:
            first = q.popleft()
            second = q.popleft()
            if first > second:
                #continue streak
                if curr_winner == first:
                    curr_streak += 1
                else:
                    curr_winner = first
                    curr_streak = 1
                #rotate
                q.append(second)
                q.appendleft(first)
                if curr_streak == k:
                    return first
            
            else:
                if curr_winner == second:
                    curr_streak += 1
                else:
                    curr_winner = second
                    curr_streak = 1
                
                #orate
                q.append(first)
                q.appendleft(second)
                if curr_streak == k:
                    return second
                
class Solution:
    def getWinner(self, arr: List[int], k: int) -> int:
        '''
        effeicient queue,
            let curr be arr[0] and the opponnets be q(arr[1:])
            also keep track of the max of the whole array
                if curr ever gets promoted to the max, the max should never lose
        '''
        max_num = max(arr)
        curr = arr[0]
        q = deque(arr[1:])
        streak = 0
        
        while q:
            opp = q.popleft()
            if curr > opp:
                q.append(opp)
                streak += 1
            else:
                q.append(curr)
                curr = opp
                streak = 1
            
            if streak == k or curr == max_num:
                return curr
            
#turns out we dont need a dequ
class Solution:
    def getWinner(self, arr: List[int], k: int) -> int:
        '''
        for a player that is not the max number, it either comes before or after
        coming before, it would lose ad get sent to the back
        coming after, it would never win
        so if an opponent loses, it will never win again
        '''
        max_num = max(arr)
        curr = arr[0]
        streak = 0
        
        for opp in arr[1:]:
            if curr > opp:
                streak += 1
            else:
                curr = opp
                streak = 1
            
            if streak == k or curr == max_num:
                return curr         
            
#############################################
# 1845. Seat Reservation Manager
# 06NOV23
#############################################
#minheap
class SeatManager:

    def __init__(self, n: int):
        '''
        hard part is trying to return the minimum seat that is available
        two heaps, seats available
        and seats taken, all seats are available
        reserve always gets the minium, but unresever could unreserve any seat
        '''
        self.seats = [i for i in range(1,n+1)]

    def reserve(self) -> int:
        ans = heapq.heappop(self.seats)
        return ans

        

    def unreserve(self, seatNumber: int) -> None:
        heapq.heappush(self.seats, seatNumber)


# Your SeatManager object will be instantiated and called as such:
# obj = SeatManager(n)
# param_1 = obj.reserve()
# obj.unreserve(seatNumber)

class SeatManager:

    def __init__(self, n: int):
        '''
        can we do it without initializaing the constructor with n seats??
        we keep track of a the smallest unservered seat
            this means that all seats >= this curr seat is remain unreserved
        the caveat with the design is that unreserve will only be called after some seat has been reserved
        i.e  elements in the min heap will alwasy be less than the marker
        if any element in this container then it contains the numun numbered seat
        otherwise an empty container means the marker is the mininum un-reserved seat
        '''
        self.min_seat = 1
        self.available_seats = []
        

    def reserve(self) -> int:
        #if there is something on this heap, its the min
        if self.available_seats:
            return heapq.heappop(self.available_seats)
        seat_number = self.min_seat
        self.min_seat += 1
        return seat_number
        

    def unreserve(self, seatNumber: int) -> None:
        heapq.heappush(self.available_seats,seatNumber)


# Your SeatManager object will be instantiated and called as such:
# obj = SeatManager(n)
# param_1 = obj.reserve()
# obj.unreserve(seatNumber)

#buultin, review Orderedset

from sortedcontainers import SortedSet
class SeatManager:

    def __init__(self, n: int):
        '''
        instead of using heap use ordered set 
        just a height balanced tree
        '''
        self.min_seat = 1
        self.available_seats = SortedSet()
        

    def reserve(self) -> int:
        #if there is something on this heap, its the min
        if self.available_seats:
            return self.available_seats.pop(0)
        seat_number = self.min_seat
        self.min_seat += 1
        return seat_number
        

    def unreserve(self, seatNumber: int) -> None:
        self.available_seats.add(seatNumber)


# Your SeatManager object will be instantiated and called as such:
# obj = SeatManager(n)
# param_1 = obj.reserve()
# obj.unreserve(seatNumber)

#############################################
# 1921. Eliminate Maximum Number of Monsters
# 07NOV23
##############################################
#yeeeeee
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        '''
        we have dist and speed arrays
            dist stores the current distance for the ith monster and speed gives speed at wchihc monster travels
        
        we can only use weapons once every minute
            i.e weapon can only kill one monster every minute
        we can only have a monster coming at every minute
        we can get their arrival times by doing dist/speed for each eleemnt,
        then sort, time in between must be greater than zero
        
        hard part is the logic for killing
        can just compare differecnes, i can kill monsters with an arrow at any distance
        '''
        N = len(dist)
        times = [dist[i]/speed[i] for i in range(N) ]
        #sort
        times.sort()
        ans = 0
        curr_time = 0
        
        for t in times:
            if t > curr_time:
                ans += 1
                curr_time += 1
            else:
                return ans
        
        return ans
    
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        '''
        take one unit time to reload, so we just compare time with the indices
        '''
        N = len(dist)
        times = [dist[i]/speed[i] for i in range(N) ]
        #sort
        times.sort()
        ans = 0
        
        for i in range(len(times)):
            if times[i] > i:
                ans += 1
            else:
                break
        
        return ans
    
#min heap
class Solution:
    def eliminateMaximum(self, dist: List[int], speed: List[int]) -> int:
        '''
        using heap, just make sure the current time isnt more than the smallest time we pull
        '''
        N = len(dist)
        times = [dist[i]/speed[i] for i in range(N) ]
        
        heapq.heapify(times)
        ans = 0
        
        while times:
            if heapq.heappop(times) > ans:
                ans += 1
            else:
                break
        
        return ans

##########################################################
# 2849. Determine if a Cell Is Reachable at a Given Time
# 08NOV23
##########################################################
class Solution:
    def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
        '''
        inputs are two big, 10^9 to do bfs
        we are allowed to move in all 8 directions
        fastest way obviosuly is to walk diagonally for some distance, then finish up with going horiz or vert
        find optimal path, and check if <= than t
        chebyshev is max distance between two points along any dimnesion
        also know as chesboard distance
        
        i dont understand the edge case though??
        omfg, AFTER EXACTLT t seconds lmaooooooo
        
        if we are allowed to go in all eight directions, the shortest is just the chebyshev distance 
        or the max(heigh,width)
        
        intution:
            imagine width > hiehgt, when going diag steps, we can height diag steps and the remaning (width - height) horiz steps
            so the distance is bounded by the maximum, same with height > widht, but just opposite
        
        the other caveat is the t seconds have to pass
        we define min_time is the chebysehv distance, and if t >= min_time, we can do it
        
        edge case, when cells are the same and t == 1
        we can revisit a cell, but going back adds another unit of time, we can't do it if t == 1, but for t > 1 we can do it
        '''
        d_cheb = max(abs(sx - fx), abs(sy - fy))
        
        #edge case
        if (sx == fx) and (sy == fy) and t == 1:
            return False
        else:
            #dist must be at least t seconds
            return d_cheb <= t
        
#########################################
# 573. Squirrel Simulation (REVISTED)
# 08NOV23
##########################################
class Solution:
    def minDistance(self, height: int, width: int, tree: List[int], squirrel: List[int], nuts: List[List[int]]) -> int:
        '''
        this kinda goes with LC. 2849, good follow up
        if squirrel started at tree, the answer is just the distances for each of the nuts times two
        but the squirrel doesn't start at the tree, in which case
            we walk to the nut that is closest, walk it to the tree, then its just 2*(sum distance of remaing nuts)
        '''
        #find closes nut to squirrel
        closest_nut = -1
        smallest_dist = float('inf')
        for i,(x,y) in enumerate(nuts):
            dist = abs(squirrel[0] - x) + abs(squirrel[1] - y)
            if dist < smallest_dist:
                smallest_dist = dist
                closest_nut = i
        
        #ans starts of with dist from sq to closest nut and from closes nut to tree
        ans = smallest_dist + (abs(tree[0] - nuts[closest_nut][0]) + abs(tree[1] - nuts[closest_nut][1]))
        for i,(x,y) in enumerate(nuts):
            if i != closest_nut:
                dist = abs(tree[0] - x) + abs(tree[1] - y)
                ans += dist*2
        
        return ans

###############################################
# 1759. Count Number of Homogenous Substrings
# 09NOV23
###############################################
#bahhh, it works, butt it aint pretty
class Solution:
    def countHomogenous(self, s: str) -> int:
        '''
        a homogenous substring is where all characters are the same
        this is a counting problem
        brute force would be to examine all substrings where there is, then count them up
        another way would be to build a homegenhoous string on the fly and put into count mapp
        say we have zzzzz
        1 + 2 + 3 + 4 + 5, its contribution is this sum
        if we have a string of the same chars of length k
        its contribution of homegenous substrings is (k*(k+1) // 2)
        which is the same as (k choose 2) + k
        count streaks
        '''
        ans = 0
        curr_letter = s[0]
        curr_streak_size = 1
        mod = 10**9 + 7
        
        for ch in s[1:]:
            if ch != curr_letter:
                #get contribution of homgenous strings
                count = (curr_streak_size*(curr_streak_size + 1) // 2) % mod
                ans += count % mod
                curr_streak_size = 1
                curr_letter = ch
            else:
                curr_streak_size += 1
        

        count = (curr_streak_size*(curr_streak_size + 1) // 2) % mod
        ans += count % mod
        return ans
        
class Solution:
    def countHomogenous(self, s: str) -> int:
        '''
        the resason why this approach doen'st work on the fly is because i have to wait for the streak to end 
        then calculate the contribution count, 
        because i have to wait for the streak to end, i need to add in the contribution count one more final time
        
        counting trick
        if there is a string of length n, then there are n substring that end with the final character
        i.e we lock the final character and can choose any of the n characters as the first
        so answer is always the length of the string
        
        if we count streaks we can just add up the streak sizes
        '''
        ans = 0
        curr_streak = 0
        mod = 10**9 + 7
        
        for i in range(len(s)):
            #edge case for startig at index zero
            if i == 0 or s[i] == s[i-1]:
                curr_streak += 1
            else:
                curr_streak = 1
            
            ans += (curr_streak) % mod
        
        return ans % mod
        
#############################################
# 1743. Restore the Array From Adjacent Pairs
# 10NOV23
##############################################
class Solution:
    def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
        '''
        we need to restore the original array given the adjacent pairs
        if the array has n elements, then adjacentPairs will have n-1 pairs
        adjacentPairs guranteed to have pairs, and return any valid array
        this is a graph problem
        dfs from the first element
        '''
        #find first eleemnt
        graph = defaultdict(list)
        counts = Counter()
        for u,v in adjacentPairs:
            counts[u] += 1
            counts[v] += 1
            graph[u].append(v)
            graph[v].append(u)
        
        #find first element, it will appear only once
        first = -1
        for k,v in counts.items():
            if v == 1:
                first = k
                break
        
        ans = []
        seen = set()
        
        def dfs(node):
            ans.append(node)
            seen.add(node)
            for neigh in graph[node]:
                if neigh not in seen:
                    dfs(neigh)
        
        dfs(first)
        return ans
    
#dont forget parent prev pardigm so we dont go back
#we also dont need a seperate counts object
class Solution:
    def restoreArray(self, adjacentPairs: List[List[int]]) -> List[int]:
        '''
        notice that we can form a double linked list
        we just need to find the roots, and the root should have only one edge
        '''
        #find first eleemnt
        graph = defaultdict(list)
        for u,v in adjacentPairs:
            graph[u].append(v)
            graph[v].append(u)
        
        #find first element, it will appear only once
        first = -1
        for num in graph:
            if len(graph[num]) == 1:
                first = num
                break
        
        ans = []
        
        def dfs(node,parent):
            ans.append(node)
            for neigh in graph[node]:
                if neigh != parent:
                    dfs(neigh,node)
        
        dfs(first,-1)
        return ans
    
###################################################
# 2642. Design Graph With Shortest Path Calculator
# 11NOV23
###################################################
#yessss
class Graph:

    def __init__(self, n: int, edges: List[List[int]]):
        '''
        this is just djikstras but implement one edge at a time
        graph is directed so we good
        we are going to need a dist array of size n by n so we cn check the shorted paths from node1 to node2
        initially all are far away, we can put edges in a min a heap
        we cant just run djikstra in the contructor
        add edge adds to our graph, but also adds to our min heap
        we run djikstras on the shortedpath call
        assume calls are all valid given graph build up to now, for add edge it is guaranteed that there is no edge betweenthe two noes before adding this one
        so do we need states as for all starting nodes?
            no because djikstrs already makes the paths optimal, so when we use node1 ad the startpoint
            
        problem is confusing, we can just do djikstras from node1
        
        '''
        self.graph = defaultdict(list)
        self.n = n
        #there's an inital state for the grpah
        for u,v,edge in edges:
            self.graph[u].append((v,edge))

    def addEdge(self, edge: List[int]) -> None:
        #add to graph and pq
        u,v,dist = edge
        self.graph[u].append((v,dist))
        

    def shortestPath(self, node1: int, node2: int) -> int:
        #do djikstras
        dist = [float('inf')]*self.n
        dist[node1] = 0
        pq = [(0,node1)]
        visited = set()
        
        while pq:
            min_dist,node = heapq.heappop(pq)
            #if its already smaller, no need to update
            if dist[node] < min_dist:
                continue
            visited.add(node)
            for neigh,dist_to in self.graph[node]:
                if neigh in visited:
                    continue
                #we can also directly return from here
                if node == node2:
                    return min_dist
                new_dist = dist[node] + dist_to
                #update?
                if new_dist < dist[neigh]:
                    dist[neigh] = new_dist
                    heapq.heappush(pq, (new_dist,neigh))
        
        #now we check for paths
        if (dist[node1] == float('inf')) or (dist[node2] == float('inf')):
            return -1
        
        return dist[node2]
        


# Your Graph object will be instantiated and called as such:
# obj = Graph(n, edges)
# obj.addEdge(edge)
# param_2 = obj.shortestPath(node1,node2)

#we can also use floyd warshall
class Graph:
    '''
    for floyd warshall, we exmaine all start and ends (i,j) and for all start and end we examin all intermediatee nodes (k)
    for all intermediate nodes k we just take the minimum
    iniutally build graph but for add edge, we try all intermediate nodes and minimuze
    brute force relaxation step
    '''

    def __init__(self, n: int, edges: List[List[int]]):
        self.n = n
        self.graph = [[float('inf')]*n for _ in range(n)]
        for u,v,edge in edges:
            self.graph[u][v] = edge
        
        #diag entries
        for i in range(n):
            self.graph[i][i] = 0
        
        #minimuze/relax
        #i is all intermediate nodes
        for i in range(n):
            #j is all source nodes
            for j in range(n):
                #k is desintation node
                for k in range(n):
                    self.graph[j][k] = min(self.graph[j][k], self.graph[j][i] + self.graph[i][k])

    def addEdge(self, edge: List[int]) -> None:
        u,v,dist = edge
        #update the minimum for all start and source nodes through this intermediate edge
        for i in range(self.n):
            for j in range(self.n):
                self.graph[i][j] = min(self.graph[i][j], self.graph[i][u] + self.graph[v][j] + dist)

    def shortestPath(self, node1: int, node2: int) -> int:
        if self.graph[node1][node2] == float('inf'):
            return -1
        return self.graph[node1][node2]
        


# Your Graph object will be instantiated and called as such:
# obj = Graph(n, edges)
# obj.addEdge(edge)
# param_2 = obj.shortestPath(node1,node2)

#####################################
# 815. Bus Routes
# 12NOV23
######################################
#fuck....
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        '''
        i can make a graph where each stop maps to every other stop
        we are just counting number of buses, if im on some route i
        then any stop in route i is reachable with 1 bus
        '''
        graph = defaultdict(set)
        for r in routes:
            for stop in r:
                neighs = set(r)
                neighs.remove(stop)
                graph[stop] = neighs
        
        q = deque([(source,1,graph[source])]) #entry is (curr stop, number buses where i can eventuallt get to)
        seen = set()
        
        if target == source:
            return 0
        while q:
            curr_stop,curr_buses,possible = q.popleft()
            if target in possible:
                return curr_buses
            if curr_stop in seen:
                continue
            seen.add(curr_stop)
            
            for neigh in graph[curr_stop]:
                if neigh not in seen:
                    next_possible = graph[neigh] | possible
                    entry = (neigh, curr_buses + 1, next_possible)
                    q.append(entry)
        
        return -1
                
#sheesh
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        '''
        thre can be multiple routes that hae this stop, 
        we cna go to any bus stop present in all these routes
        we could store all bus stops in the routes, but that would take up too mush space
        insteaf we only store the indices of the routes, i.e a bus stop maps to a route index
        '''
        if target == source:
            return 0
        adj_list = defaultdict(set)
        for i, route in enumerate(routes):
            for stop in route:
                adj_list[stop].add(i)
        q = deque([])
        visited_routes = set()
        busCount = 1
        #insert all routes that have this source bus stop
        for start_route in adj_list[source]:
            q.append((start_route))
            visited_routes.add(start_route)
        
        
        while q:
            N = len(q)
            for _ in range(N):
                curr_route = q.popleft()
                
                #get next stoprs
                for next_stop in routes[curr_route]:
                    if next_stop == target:
                        return busCount
                    
                    #get next route for this stop
                    for next_route in adj_list[next_stop]:
                        if next_route not in visited_routes:
                            visited_routes.add(next_route)
                            q.append(next_route)
            
            busCount += 1
        return -1


#we need to map stops to route indices
#then we know what stops we can eventually reach from this curr stop
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        '''
        i can make a graph where each stop maps to every other stop
        we are just counting number of buses, if im on some route i
        then any stop in route i is reachable with 1 bus
        '''
        if target == source:
            return 0
        adj_list = defaultdict(set)
        for i, route in enumerate(routes):
            for stop in route:
                adj_list[stop].add(i)
        q = deque([(source, 0)])
        visited = set()
        
        while q:
            stop,buses = q.popleft()
            if stop == target:
                return buses
            for group in adj_list[stop]:
                for nei in routes[group]:
                    if nei not in visited:
                        visited.add(nei)
                        q.append((nei, buses + 1))
                #clear the group
                #after vising this group we can clear it, becase we already would have seen any top from this group
                routes[group] = []
        return -1

#routes as nodes
class Solution:
    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        '''
        we can also treat routes as the nodes, 
        when making the graph, we need to find if there is shared stop between routes
        then the graph becomes undirected,
        we can keep visited set and use multipoint BFS
        need to sort stops in each route increaisnly
        '''
        #same source and target
        if source == target:
            return 0
        
        #sort
        N = len(routes)
        for i in range(N):
            routes[i].sort()
        
        graph = defaultdict(list)
        for i in range(N):
            for j in range(i+1,N):
                if self.hasCommonNode(routes[i],routes[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        
        #add starting routes that contain source
        q = deque([])
        for i in range(N):
            if self.stopExsists(routes[i],source):
                q.append(i)
        visited_routes = set()
        #same as before
        busCount = 1
        
        while q:
            N = len(q)
            
            for _ in range(N):
                curr_route = q.popleft()
                visited_routes.add(curr_route)
                if self.stopExsists(routes[curr_route],target):
                    return busCount
                
                #check net routes
                for neigh_route in graph[curr_route]:
                    if neigh_route not in visited_routes:
                        q.append(neigh_route)
            busCount += 1
            
        return -1
        
    
    def hasCommonNode(self,route1,route2) -> bool:
        i,j = 0,0
        while i < len(route1) and j < len(route2):
            if route1[i] == route2[j]:
                return True
            elif route1[i] < route2[j]:
                i += 1
            else:
                j += 1
        
        return False
    
    def stopExsists(self,route,stop) -> bool:
        for i in range(len(route)):
            if route[i] == stop:
                return True
        
        return False
    
#######################################################
# 2785. Sort Vowels in a String
# 13NOV23
#######################################################
class Solution:
    def sortVowels(self, s: str) -> str:
        '''
        premute s, such that the voewls are sorted in nondecreassing ASCII values
        unpack string into an array, identiy vowel indices
        sort them and put them pack into array
        
        '''
        s = list(s)
        N = len(s)
        vowels = 'AEIOUaeiou'
        vowel_indicies = [False]*N
        vowels_to_sort = []
        
        for i in range(N):
            if s[i] in vowels:
                vowel_indicies[i] = True
                vowels_to_sort.append(s[i])
        
        vowels_to_sort.sort()
        curr_vowel_ptr = 0
        
        for i in range(N):
            if vowel_indicies[i]:
                s[i] = vowels_to_sort[curr_vowel_ptr]
                curr_vowel_ptr += 1
        
        return "".join(s)

class Solution:
    def sortVowels(self, s: str) -> str:
        '''
        we can also use counting sort 
        our buckets will only be AEIOUaieou
        for using counting sorts:
            when the length is much greater than the number of distinct characters, in this case 10**5 >> number of vowels (ie cahracters)
        '''
        counts = Counter(s)
        vowels = 'AEIOUaeiou' #keep in inceasing ASCII
        i = 0 #ptr into vowels
        res = ""
        for ch in s:
            if ch in vowels: #set it smalle anway
                #need effeicent way to go through vowels in order
                while counts[vowels[i]] == 0:
                    i += 1
                
                res += vowels[i]
                counts[vowels[i]] -= 1
            
            else:
                res += ch
        
        return res