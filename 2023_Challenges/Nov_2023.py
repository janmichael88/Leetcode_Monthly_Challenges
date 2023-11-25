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
    
#####################################
# 248. Strobogrammatic Number III
# 13NOV23
#####################################
class Solution:
    def strobogrammaticInRange(self, low: str, high: str) -> int:
        '''
        mappings for strobgrammitc are
        we already have the solution for strobrgammitc number II
        which gives us all sb numbers of length n
        we dont need all the actual numbers, just the count of the numbers
        we can take the level order traversal solution from sb number II, and recreated valid sb numbers
        whenver we add back into the level, we increment a count
        
        recap on strobrogrammtic number II
        recursive idea;
            odd lenghts, cetners can only be 0,1,8
            even lenggth, n = 2, 11,69,88,96
            transistion if to go back to dp(n-1) or dp(n-2)
            and then we appened a digit left and right, checking or valid numbers of course
        dp(n) = {
            to_return = []
            for num in dp(n-2):
                for (digit1,digit2) in reversiblePairs:
                    to_return.append(digit1 + num + digit2)
            return to_return
        }
        
        for level order traversal solution, we intelligently build a strobgrammitc number

        '''
        reversible_pairs = [
            ['0', '0'], ['1', '1'], 
            ['6', '9'], ['8', '8'], ['9', '6']
        ]
        #keep track og string lengthrs for low and high, we can't make an SB number smaller than this string length or greater than the high
        low_limit = len(low)
        hi_limit = len(high)
        count = 0 
        q = deque(["","0","1","8"]) #start off with base sb nubers of length 1
        while q:
            curr_sb_num = q.popleft() #remember this is a string
            #if string length in bounds
            if len(curr_sb_num) < hi_limit or len(curr_sb_num) == hi_limit and curr_sb_num <= high:
                if len(curr_sb_num) > low_limit or len(curr_sb_num) == low_limit and curr_sb_num >= low:
                    #increment count if this is the '0' or doen'st start with
                    if curr_sb_num == '0' or not curr_sb_num.startswith("0"):
                        count += 1
                #check if we can append digits to start and end
                if hi_limit - len(curr_sb_num) >= 2:
                    for left,right in reversible_pairs:
                        next_sb_num = left + curr_sb_num + right
                        q.append(next_sb_num)
        
        return count
    
#################################################
# 1930. Unique Length-3 Palindromic Subsequences
# 14NOV23
#################################################
#70/70 !! but judge says it took too long
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        '''
        we only care about length three palindromic subsequences
        hints
            1. what is the maximum number of length 3 palindromic strings?
            2. how can we keep track of the characters that appeared to the left of a given position?
        
        say we are at index i, and we want to check if we can make a 3 length palindrome
        we need to see if there is some char x before i and after i
        try all charax a to z
        i can use hashtable, and mapp chars to indices
        we can use binary search
        say we are at i = 5 and want to look left
        need to find the insertion point of 4
        [0,1,2,3]
        '''
        mapp = defaultdict(list)
        for i,ch in enumerate(s):
            mapp[ch].append(i)
            
        possible = set()
        N = len(s)
        
        
        def find_left(arr,target):
            left = 0
            right = len(arr) - 1
            ans = 0
            while left <= right:
                mid = left + (right - left) // 2
                if arr[mid] <= target:
                    ans = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            return ans
        
        def find_right(arr,target):
            left = 0
            right = len(arr) - 1
            ans = right
            while left <= right:
                mid = left + (right - left) // 2
                if arr[mid] >= target:
                    ans = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            return ans
        

        for i in range(1,N-1):
            for j in range(26):
                ends = chr(ord('a') + j)
                candidate = ends+s[i]+ends
                if candidate in possible:
                    continue
                #look for ends to the left of i and to the right of i
                if ends not in mapp:
                    continue
                #for the left there needs to be an occurence of ends where the index <= i - 1 and >= 0
                #for the right there needs to be an index where index >= i + 1 and < N - 1
                left = find_left(mapp[ends],i-1)
                right = find_right(mapp[ends],i+1)
                if (0 <= mapp[ends][left] <= i - 1) and (i + 1 <= mapp[ends][right] < N):
                    possible.add(candidate)
        
        return len(possible)
    
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        '''
        turns out we can abuse .index methods in python here
        for each letter, find its first occurence and last occurence
        then we just checek for all unique chars between this first and last occurence
        '''
        letters = set(s)
        count = 0
        
        for ch in letters:
            left = s.index(ch)
            right = s.rindex(ch)
            
            unique = set()
            
            for k in range(left+1,right):
                unique.add(s[k])
            
            count += len(unique)
        
        return count
            
#without using builtin
class Solution:
    def countPalindromicSubsequence(self, s: str) -> int:
        '''
        no builtin .index
        '''
        letters = set(s)
        count = 0
        
        for ch in letters:
            left = -1
            right = 0
            
            for k in range(len(s)):
                if s[k] == ch: #first occurence
                    if left == -1:
                        left = k
                    #update every other last occurence
                    right = k
                
            unique = set()
            for k in range(left+1,right):
                unique.add(s[k])
            
            count += len(unique)
        
        return count

#########################################################
# 1846. Maximum Element After Decreasing and Rearranging
# 15NOV23
########################################################
#easy
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        '''
        we have two operations
        1. decrease the value of any element of arr to a smaller positive interger
        2. rearrange elements in arr to any order
        
        useing these operations we want an array that satisfies this criteria:
        1. first value of arr must be 1
        2. absolute difference between any two adjacent elements must be <= 1
            i.e abs(arr[i] - arr[i - 1]) <= 1 for each i where 1 <= i < arr.length (0-indexed
            
        the obviouse answer is to just turn it into the increasing array [1,2,3...n]
        sort the array and decrement each element to the next largest integer that satisfies the conditions
        
        [1,6]
        after sorting its goint to be increasig, so we need to decremen it
        we need i to be arr[i-1] + 1
        
        this problem is stupid lol
        '''
        arr.sort()
        N = len(arr)
        arr[0] = 1
        for i in range(1,N):
            if abs(arr[i] - arr[i-1]) > 1:
                arr[i] = arr[i-1] + 1
        

        return max(arr)
    
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        '''
        notes on official solution
        strictly increasing array is not possible if the elements in the original array do no support the counting
            becuase we are only allowed to decrease a number to a smallerr number
            sort increasinly, if the max num of arr is k, we can't raise it beyond k
        '''
        arr.sort()
        ans = 1 #start off at 1
        for num in arr[1:]:
            #is its more than run, we just raise it
            if num >= ans + 1:
                ans += 1
        
        return ans
    
#no sort
class Solution:
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        '''
        we can use counting sort;
        keep in mind that we are only allowed to decrement numbers, not increment
        recall in the best case we have the strictly increasing array [1,n], and its impossible to have elements greater than n, givne n is len(arr)
        we can just count up the number of nums, since we can't be greater than n
        if num >= n, we simply treat the num as n 
        so counts array gives number of occurences in arr when num is min(num,n)
        i.e in array [1,100,100,100], we cant use 100, since n is 4, so we treat 100 as 4
        1. if ans + count[num] <= nums, it menas there are less occrucnes of num in arr than there are sports in the range [ans+1,num]
            so we can REDUCE every instance of num to improve ans and do ans += counts[num]
        2. if ans + count[num] > num, it means there are more num than there are spots in the array
            so we set ans = num
            
        we can combine and get
        ans = min(ans + counts[num],num)
        '''
        N = len(arr)
        counts = [0]*(N+1)
        
        for num in arr:
            counts[min(num,N)] += 1
        
        ans = 1
        for num in range(2,N+1):
            ans = min(ans + counts[num],num)
        
        return ans
    
##########################################
# 1980. Find Unique Binary String
# 16NOV23
###########################################
#backtrakcnig works
#but its slow
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        N is length 16, so we can use backtracking to generate the strings
        the strings are binary, so we can just pick one that is not in nums
        '''
        N = len(nums)
        nums = set(nums)
        self.ans = ""
        
        def backtrack(N,path):
            if len(path) == N:
                temp = "".join(path)
                if temp not in nums:
                    self.ans = temp
                return
            
            for char in ['0','1']:
                path.append(char)
                backtrack(N,path)
                path.pop()
        
        backtrack(N,[])
        return self.ans
    
#stupid way is to just generate sll integers
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        we dont even need to use backtracking
        convert string binary strings to integers
        '''
        #convert binary strings to in
        def convert(string):
            ans = 0
            power = 0
            for ch in string[::-1]:
                ans += int(ch)*2**power
                power += 1
            
            return ans
                
        base_10 = []
        for num in nums:
            base_10.append(convert(num))
        
        N = len(nums)
        seen = set(base_10)
        
        for n in range(2**N):
            if n not in seen:
                #convert to string base make sure to add 0s left to pad to lenght N
                string_ans = bin(n)[2:]
                string_ans = '0'*(N-len(string_ans)) + string_ans
                return string_ans
            
#optimized recursion
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        trick to recursion is cache a call to the function and check if we want return here
        if we check n+1 different string of length n, we will surely find an answer
        early termination means we wont check more than n+1 string of length n, otherwise it becomes 2**n
        '''
        N = len(nums)
        nums = set(nums)
        
        def check(string):
            if len(string) == N:
                if string not in nums:
                    return string
                return ""
            
            #first try with zero
            zero = check(string + "0")
            if zero: #len(zero) != 0
                return zero
            
            return check(string + "1")
        
        return check("")
    
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        iterative, length nums is bounded
        since there are only n string in nums, i.e len(nums) = n, and each string is of size n, we dont need to chece n+1 different binary strings
        its because each string in nums is of length(n)
        carefully look at the n inputs, there are only going to be n strings in nums
        which means at least of the integers from 0 to n is going to be missing!
        '''
        base_10 = set()
        for num in nums:
            base_10.add(int(num,2))
            
        N = len(nums)
        for num in range(N+1):
            if num not in base_10:
                string_ans = bin(num)[2:]
                string_ans = '0'*(N-len(string_ans)) + string_ans
                return string_ans
        
        return ""
    
#random simulation
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        there are only n strings present
        and in total there are 2**n strings possible
        so the probability of randomly drawing one is:
        (2**n - n) // 2**n
        for n = 16, this is 99.9 %
        so we keep drawing lol
        '''
        base_10 = set()
        for num in nums:
            base_10.add(int(num,2))
            
        N = len(nums)
        ans = int(nums[0],2)
        
        while ans in base_10:
            ans = random.randrange(0,2**N)
        
        string_ans = bin(ans)[2:]
        string_ans = '0'*(N-len(string_ans)) + string_ans
        return string_ans
    
#cantors diagnoal 
class Solution:
    def findDifferentBinaryString(self, nums: List[str]) -> str:
        '''
        brief on cantor's diagonal arguement
        proves that there are infinite sets which cannot be put into a one to one correspondence with the infinite set of natural numbers
        example, examine enumerates of sequences s coming from elements T, we let T be chosen from 0 or 1
        from these sets we make a new set, where we take an elmenet at the ith index, neagate it and make a new set
        this new set is an enumerate, but cannot be part of the infinite set of all s
        just proves that  T is uncountable, an uncountable set cannot be mapped to an infiniset set of numbers
        '''
        ans = ""
        N = len(nums)
        for i in range(N):
            ans += '1' if nums[i][i] == '0' else '0'
        
        return ans

########################################
# 1877. Minimize Maximum Pair Sum in Array
# 17NOV23
########################################
class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        '''
        need to pair such that
            * each element of nums is in on pair
            * the maximum pair sum is minimized
        return minimuzed maximum pair
        n is even so we will always have valid pairs
        sort and two pointers and take one from each end

        proof
        assum nums is sorted ; [a1,a2,a3,a4]; a1 < a2 < a3 < a4
        assume two nums a_i and a_j
        proof by contradiction
        assume the pairing [(a1,a_i), (a_j,a_n)] is optimal than [(a1,a_n), (a_i, a_j)]
        this cont be true becase a_j + a_n is always bigger thatn a_i + a_1
        '''
        nums.sort()
        N = len(nums)
        ans = 0
        for i in range(N//2):
            ans = max(ans, nums[i] + nums[-(i+1)])
        
        return ans

##############################################
# 1838. Frequency of the Most Frequent Element
# 18NOV23
##############################################
#this doesn't quite work because its not always the best to increment all elements starting from the end
#we need to be smart in how increment elements that are <= the curr_num we are trying to raise to
#ideall we need them to be closer to num so we can raise more of them
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        '''
        in one operation we can chose an index in nums and raise it by 1
        return the maximum possible frequency of an element after peforming at most k opearations
        doest it make sense to raise every element to the one that is already the most frequent?
            in some cases yes, but the currnet most frequenct element may not be reachable from other numbers
            we dont have to do operations if we are already maximum
        
        sort non-decreasing, we can only raise a number
        [1,2,4], k = 5
        try 1, its the smallest, so for 1 we only get a frequency of 1
        try 2, we can only raise numbers <= 2 by their difference, so we raise the 1
        try, 4, we can raise 1 and 2
        so the currnet number we want to raising to is x, then for every number smaller than x we need to do:
            steps
            for num [less than x]:
                steps  += x - num, 
                until we cant get k
        
        '''
        nums.sort()
        N = len(nums)
        ans = 1
        for num in nums:
            curr_freq = 0
            start = 0
            curr_k = k
            while start < N and nums[start] <= num:
                #find diff from num
                diff = num - nums[start]
                if curr_k - diff >= 0:
                    curr_freq += 1
                    curr_k -= diff
                start += 1
            
            ans = max(ans, curr_freq)
        
        return ans
    
#TLE, this is essentially O(N^2)
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        '''
        its clear that the number we want to raise to must already exsist in the range
        say we want to raise to target-1 or target + 1, if neight are in the array we would was operations trying to make it
        bascially we are trying to imporove the frequency
        proof that number must be in nums is kinda hand wavy
        the idea is to increment the elemnets that are closes to the target
        i.e we treat each num in nums as a target to raise to, if we sort, the closest ones are to the left
        '''
        nums.sort()
        N = len(nums)
        ans = 0
        for i, num in enumerate(nums):
            curr_k = k
            curr_freq = 0
            while i >= 0:
                diff = num - nums[i]
                if curr_k - diff >= 0:
                    curr_freq += 1
                    curr_k -= diff
                    i -= 1
                else:
                    break
            
            ans = max(ans,curr_freq)
        
        return ans
    
#sliding window
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        '''
        short proof on why the target must exist in nums
            examine some number absent in nums that is not in nums
            now examine absent', the largest number in nums that is smaller than absent
            it is always the case that we save operations raising to absent' than absent, 
        we need to optimize the way we count opertoins
        say we have target as num, and its current frequency is 4
        the array will essentiall have sum = 4*target
        then number of operations would be sum in the orignal array less than 4*target
        example
        [3,6,7,12,19,22,44,150]
        and we are trying 12
        so we want:
        [12,12,12,12], which has sum 48
        original sum = 3 + 6 + 7 + 12 = 28 
            now think prefix sum!
        so we need 48 - 28 = 20 operations
        sliding window!
            shrink windwo when we operations require more than k
        
        
        '''
        nums.sort()
        N = len(nums)
        left = 0
        ans = 0
        curr_sum = 0
        for right,target in enumerate(nums):
            curr_sum += target
            while (right - left + 1)*target > k + curr_sum: # (right - left + 1)*traget - curr_sum > k
                #shrink if it requires more than k operations
                curr_sum -= nums[left]
                left += 1
            
            ans = max(ans, right - left + 1)
        
        return ans
    
#advanced sliding window
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        '''
        as with all sliding windows, we can optimize by not actually sliding
        we only care about the length of the window and not what is actually in the window
        if we find a valid window of length len, then we dont really care about windows that are smaller
            because a smaller window implies a smaller frequeny
            i.e once we have found a valid window, this is guaranteed to be maximum
        
        we just make it so that the current [left,right] is the best windwo so far
        we always try to grow the window, we never shrink
        note, left never increase by more than one in each iteration
        '''
        nums.sort()
        N = len(nums)
        left = 0
        ans = 0
        curr_sum = 0
        for right,target in enumerate(nums):
            curr_sum += target
            if (right - left + 1)*target > k + curr_sum: # (right - left + 1)*traget - curr_sum > k
                #only shrink when we have found a valid array BY ONLY expanding right
                curr_sum -= nums[left]
                left += 1
            
        
        return N - left

#binary search
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        def check(i):
            target = nums[i]
            left = 0
            right = i
            best = i
            
            while left <= right:
                mid = (left + right) // 2
                count = i - mid + 1
                final_sum = count * target
                original_sum = prefix[i] - prefix[mid] + nums[mid]
                operations_required = final_sum - original_sum

                if operations_required > k:
                    left = mid + 1
                else:
                    best = mid
                    right = mid - 1
                    
            return i - best + 1
        
        nums.sort()
        prefix = [nums[0]]
        
        for i in range(1, len(nums)):
            prefix.append(nums[i] + prefix[-1])
        
        ans = 0
        for i in range(len(nums)):
            ans = max(ans, check(i))
            
        return ans
    
##############################################################
# 1887. Reduction Operations to Make the Array Elements Equal
# 19NOV23
##############################################################
#its not just the number between left and right
#think of 1,2,3,4,5
class Solution:
    def reductionOperations(self, nums: List[int]) -> int:
        '''
        we want to make all numbers in nums equal, we are allowed one operation on these steps
            1. finad largest values in nums, get its index, if there are ties, pick the smalllest i
            2. find the next largest value in nums that is just smaller than largest, nextLargest
            3. reduce nums[i] to next largest
        
        return number of operations to make all elements equal
        sort the array and try to reduce all elements with maximum value to the next maximum value in one operation
        eventuall all numbers will get sent to the smallest value
        [5,1,3] after sorting [1,3,5]
        need to reduce to 3
        [1,3,3] one operation
        need to reduce to 1
        [1,1,1] twp operations, 3 in total
        sort the array and work backwards
         '''
        nums.sort()
        operations = 0
        N = len(nums)
        
        right = N - 1
        while right >= 0:
            first_max = nums[right]
            left = right - 1
            while left > 0 and nums[left] == first_max:
                left -= 1
            
            second_max = nums[left]
            if second_max != first_max:
                operations += right - left
            
            right = left
        
        return operations

class Solution:
    def reductionOperations(self, nums: List[int]) -> int:
        '''
        we want to make all numbers in nums equal, we are allowed one operation on these steps
            1. finad largest values in nums, get its index, if there are ties, pick the smalllest i
            2. find the next largest value in nums that is just smaller than largest, nextLargest
            3. reduce nums[i] to next largest
        
        we can just use two poiners
        imagine
        [1,2,3,4,5]
        (4,5) is (second,first)
        [1,2,3,4,4] 1 op
        (3,4)
        [1,2,3,3,3] 2 ops
        (2,3)
        [1,2,2,2,2] 3 ops
        (1,2)
        [1,1,1,1,1] 4 ops
        
        10 ops in total
         '''
        nums.sort()
        operations = 0
        N = len(nums)
        print(nums)
        right = N - 1
        while right >= 0:
            first_max = nums[right]
            left = right - 1
            while left >= 0 and nums[left] == first_max:
                left -= 1
            
            second_max = nums[left]
            if second_max != first_max:
                #need to to changer all numbers to the right, from right
                operations += N - right - 1
            
            right = left
        
        return operations
    
#another way is to just go backwards and add N - i
class Solution:
    def reductionOperations(self, nums: List[int]) -> int:
        '''
        we can sort and just go backwards
        we need to compare of nums[i-1] != nums[i]
        '''
        nums.sort()
        N = len(nums)
        ans = 0
        for i in range(N-1,0,-1):
            if nums[i-1] != nums[i]:
                ans += N - i
        
        
        return ans

###################################################
# 2391. Minimum Amount of Time to Collect Garbage
# 20NOV23
###################################################
class Solution:
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        '''
        we have N = len(garbage) houses, each having MPG
        len(travel) = N - 1, and travel[i] is time it takes to go from i to i + 1
        three garbage trucks, picks up each kind only, each starts at house 0, and msut visit in order, but do not need to visit every hous
        only on truck may be used at any given moment (i.e we can split driving and pick up tasks)
        return min time needed to pick up all garbage
        pick up takes 1 unit of time
        
        we dont need to visit all the house
        for each type of garbage, find the house the the highest index that has at least 1 unit of this type of garbage
        ill need a fast way to get traval times
            prefisum sum on travel times
        '''
        pref_travel = [0]
        for num in travel:
            pref_travel.append(num + pref_travel[-1])
        #(i,j) inclusive j
        #travel time (i,j) is pref_travel[j] - pref_travel[i]
        #from_ = 3
        #to_ = 3
        #print(pref_travel[to_] - pref_travel[from_]) #this is how to calculate from_ and to_
        
        #now this is just a pref_sum problem
        #mapp indices to chars, and follow paths
        mapp = defaultdict(list)
        for i,trash in enumerate(garbage):
            for t in trash:
                mapp[t].append(i)
        
        time = 0
        for kind,houses in mapp.items():
            start_house = 0
            for end_house in houses:
                travel_time = pref_travel[end_house] - pref_travel[start_house]
                time += travel_time
                time += 1
                start_house = end_house
        
        return time

class Solution:
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        '''
        turns out we dont need to roll the individual to and from times
        if we just count up the numbers of each type of garbage and keep track of the last position for each type of garbage
        we only need to keep track of the last position for each type of garbage
            becaseu anything smaller than the last positino would have been picked up anyway!
            stupid me!
        '''
        pref_travel = [0]
        for num in travel:
            pref_travel.append(num + pref_travel[-1])
        #(i,j) inclusive j
        #travel time (i,j) is pref_travel[j] - pref_travel[i]
        #from_ = 3
        #to_ = 3
        #print(pref_travel[to_] - pref_travel[from_]) #this is how to calculate from_ and to_
        
        #now this is just a pref_sum problem
        #mapp indices to chars, and follow paths
        lastTrashPosition = defaultdict()
        trashCounts = Counter()
        for i,trash in enumerate(garbage):
            for t in trash:
                trashCounts[t] += 1
                lastTrashPosition[t] = i
        
        time = 0
        for t in 'MPG':
            if t in lastTrashPosition:
                last_position = lastTrashPosition[t]
                garbageCount = trashCounts[t]
                time += pref_travel[last_position] + garbageCount
        
        return time
    
##############################################
# 2589. Minimum Time to Complete All Tasks
# 20NOV23
##############################################
class Solution:
    def findMinimumTime(self, tasks: List[List[int]]) -> int:
        '''
        O(N^2) works, and its alwasy better to run the task as late as possible so that other tasks can run simultanoeusly
        this is an overlappping interval problem, need to determine if we sort or start or on ends
        what if we try sorting on start
        [1,5,1], [2,3,1]
        we know we want to turn computer on at 2 or 3, becase we get both, but since this is sorted on start, we would have assumes pick time between [1,5]
        [2,3] would have never been considered
        so we at least try sorting on end, here's another example
        [15,20,3],[10,25,5],[6,30,7], after sorting on ends
        [15,20,3], need 3 units of overlap with other tasks
            this happens at [18,19,20]
        
        we cant do ealier than 18
        say we had a task like [13,16,2], because we sroted by end, we would enver see this taks after [15,20,2]
        rather because we sort an end, we can only look at overlappint interval times ENDING and EQUAL to 2
        
        now examins [10,25,5], 
        this ends at 25, we use 18,19,20, need duration of 2 more
        [25,24]
        
        now [6,30,7]
        ends at 30, need duration of (7-5)
        so 29,30
        
        if the computer is on during a time in out interval, we want to use that time to run it
        iterated and see whihc times our interval time our computer is on and subtract one from each duration for this task
        then we still have any duration left, we will turn it on at the latest second possible so that we are most likely to overlap with another task
        the next task has an end point >= the current task, its starting point is unknown
        when we choose the latest ssecond possible, we minimize our computer usage in all cases
        
        essentially mark times when our compute should be on, but only when a task is running
        after sorting of cours

        extra notes:
            for the first task with the closest end time, the best strategy is to finish it as late as possible
            this allows for the best change for the compute time to be resued by tasks that end later
            use array to mark time slows
            could also have use BIT 
        '''
        tasks.sort(key = lambda x: (x[1],x[0]))
        times = [False]*2001 #imes are fixes

        for start,end,duration in tasks:
            curr_durr, curr_end = duration,end
            for s in range(start,end+1):
                #if our copmute is altready on, we are covered
                if times[s]:
                    curr_durr -= 1
            
            #while we still need duratinos to cover, extend to other times
            #we need to cover as many time spots from the current end that hasn't been covered
            #while we still have a duration that needs to be covered
            while curr_durr > 0:
                if not times[curr_end]:
                    times[curr_end] = True
                    curr_durr -= 1
                
                curr_end -= 1
        
        #the array just stores the times the computer should be one
        return sum(times)
                
##############################################
# 1814. Count Nice Pairs in an Array
# 20NOV23
##############################################
#unfortunately i don't think i could have gotten this without the hints
class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        '''
        reverse of a number is reverse, after trimming leading zeros if any
        (i,j) is a nice pair if
            0 <= i < j < len(nums)
            nums[i] + rev(nums[j]) == nums[j] + rev(nums[i])
        
        return number of nice pairs
        we can also write:
            nums[i] - rev(nums[i]) == nums[j] - rev(nums[i])
            transform each nums[i] to (nums[i] - rev(nums[i])), then count the pairs
            then look if it exists in the hashmap

        notes;
        When you see something like this ...[i] + ...[j] == ...[j] + ...[i]
        always regroup so that the terms with the same i/j are on the same side of the equation
        e.g. ...[i] - ...[i] == [j] - ...[j]
        '''
        mod = 10**9 + 7
        
        def rev(num):
            rev_num = 0
            while num:
                rev_num *= 10
                rev_num += num % 10
                num = num // 10
            
            return rev_num
        
        nums = [num - rev(num) for num in nums]
        counts = Counter()
        ans = 0
        
        for num in nums:
            ans += counts[num] % mod
            counts[num] += 1
        
        return ans % mod
    
'''
notes on the math
for some (i,j) we want pairs such that
nums[i] + rev(nums[j]) == nums[j] + rev(nums[i])
rewrite as
nums[i] - rev(nums[i]) = nums[j] - rev(nums[j])

tranforms some num to num - rev(num)
the problem then becomes how manu pairs are equal
count along the way, and check if we have seen this num
each num we have sen before can be paired with the current one!
'''

###########################################
# 2364. Count Number of Bad Pairs
# 21NOV23
###########################################
class Solution:
    def countBadPairs(self, nums: List[int]) -> int:
        '''
        a bad pair (i,j) is
            i < j and 
            j - i != nums[j] - nums[i]
            rewrite as
            -i + nums[i] != nums[j] - j
            nums[i] - i != nums[j] - j
        '''
        counts = Counter()
        N = len(nums)
        ans = 0
        for i,num in enumerate(nums):
            temp = num - i
            #need number of pairs not == temp and before i
            #i - temp is the number of bad pairs, its good pair if its equal
            ans += i - counts[temp]
            counts[temp] += 1
    
        return ans

#########################################
# 1424. Diagonal Traverse II
# 22NOV23
#########################################
#TLE, cant just walk diags
class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        '''
        pad the array so that all have the same column size,
        then walk the diagonals
        only add to the ans if it wasn' part of the padding
        each eleemnt will be >= 1, can use 0 to pad
        
        '''
        rows = len(nums)
        max_cols = 0
        for r in nums:
            max_cols = max(max_cols,len(r))
        
        new_rows = []
        for r in nums:
            r = r +[0]*(max_cols - len(r))
            new_rows.append(r)
        
        ans = []
        #walk diag up starting with each row from the first col
        for r in range(rows):
            start_row, start_col = r, 0
            while start_row >= 0 and start_col < max_cols:
                if new_rows[start_row][start_col] != 0:
                    ans.append(new_rows[start_row][start_col])
                
                start_row -= 1
                start_col += 1
        
        #whoops forgot the last row stargint with the first col
        for c in range(1,max_cols):
            start_row, start_col = rows - 1, c
            while start_row >= 0 and start_col < max_cols:
                if new_rows[start_row][start_col] != 0:
                    ans.append(new_rows[start_row][start_col])
                
                start_row -= 1
                start_col += 1
        return ans

#need to daig trick and doule sort
class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        '''
        i need to use the diagonal trick T.T
        '''
        diags = defaultdict(list)
        for i in range(len(nums)):
            for j in range(len(nums[i])):
                #entry = (i + j, i, nums[i][j])
                d = i + j
                diags[d].append((i, nums[i][j]))
        
        #print(diags)
        ans = []
        for d in sorted(diags.keys()):
            for row,val in sorted(diags[d], key = lambda x: (-x[0],x[1])):
                ans.append(val)
        
        return ans

class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        '''
        we can start at the last row and first col, and keep doing in that direction
        also note that the row + col diag identifes keeps increasing from 0 to the last diag
        '''
        diags = defaultdict(list)
        for row in range(len(nums) - 1,-1,-1):
            for col in range(len(nums[row])):
                d = row + col
                diags[d].append(nums[row][col])
        
        ans = []
        curr_d = 0
        while curr_d in diags:
            ans.extend(diags[curr_d])
            curr_d += 1
        
        return ans
    
#BFS
class Solution:
    def findDiagonalOrder(self, nums: List[List[int]]) -> List[int]:
        '''
        we can inteligently use BFS
        we know we start from (0,0)
        we only need to consider (row + 1, col) if we are the start of the diagonal
            
        otherwise for every other square in the diag , it mist have already been visited
        if you didnt know this, we coul have used a hash set
        intution
            (0,0) is the source and each diagonal represents a distance from the source
        
        note the was add the square (row + 1,col) before (row,col+1)
        queue will only be as large as the largest diagonal
        '''
        q = deque([(0,0)])
        ans = []
        while q:
            row,col = q.popleft()
            ans.append(nums[row][col])
            
            #if we are teh start of a diagonal
            if col == 0 and row + 1 < len(nums):
                q.append((row+1,col))
            
            if col + 1 < len(nums[row]):
                q.append((row,col+1))
    
        return ans
    
#############################################
# 1630. Arithmetic Subarrays
# 23NOV23
#############################################
#sort and check, ez pz
class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        '''
        for each query, we need to determine if the nums in the given array can be rearrange to for an arithmetic sequence
        we can check if there is an arithmetic subsequence in each of the queries by sorting and checking the consectuive differences
        '''
        def check(arr):
            arr.sort()
            N = len(arr)
            for i in range(1,N-1):
                if arr[i] - arr[i-1] != arr[i+1] - arr[i]:
                    return False
            
            return True
        
        N = len(nums)
        M = len(l)
        ans = [False]*M
        
        for i,(j,k) in enumerate(zip(l,r)):
            ans[i] = check(nums[j:k+1])
        
        return ans
    
#no sort
class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        '''
        for the check we dont need to sort
        for a given array to check say we have MIN and MAX with N elements
        the consecutive difference should be
        (MIN - MAX) / (N-1)
        we can hash all the elements and just make sure they are in the array when we advance by diff
        '''
        def check(arr):
            MIN = min(arr)
            MAX = max(arr)
            
            #does not evenly divide
            #if there n elements, we have n-1 gaps
            #total diff should span array
            if (MAX - MIN) % (len(arr) - 1) != 0:
                return False
            
            diff = (MAX - MIN) // (len(arr) - 1)
            seen = set(arr)
            curr = MIN
            while curr < MAX:
                if curr not in seen:
                    return False
                curr += diff
            
            return True
        
        N = len(nums)
        M = len(l)
        ans = [False]*M
        
        for i,(j,k) in enumerate(zip(l,r)):
            ans[i] = check(nums[j:k+1])
        
        return ans
    
#MOS algorithm

    
#############################################
# 1561. Maximum Number of Coins You Can Get
# 24NOV23
#############################################
#taking second max doesnt always work
# class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        '''
        we have multiples of 3 piles, three players me, bob and alice
        i always pick the 3 piles,
        first alice takes, then me, then bob
        since im picking and alice always picsk first, i just pick the second max of three every time
        '''
        ans = 0
        #negate for max heap
        new_piles = [-p for p in piles]
        heapq.heapify(new_piles)
        while new_piles:
            alice = heapq.heappop(new_piles)
            me = heapq.heappop(new_piles)
            bob = heapq.heappop(new_piles)
            ans -= me
        
        return ans

#tricky two pointer
class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        '''
        when we pick piles, alice will always get the max, and bob will always get the minimum
        i can sort and use two pointers
        alice gets r, bob gets l, and i should get the max of r + 1 or l + 1
        '''
        ans = 0
        piles.sort()
        left = 0
        right = len(piles) - 1
        while left < right:
            #alice always takes max
            right -= 1
            #i take the bigger one
            if piles[right] > piles[left]:
                ans += piles[right]
                right -= 1
                left += 1
            else:
                ans += piles[left]
                left += 1
                right -= 1
        
        return ans

#could have used deque
class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        '''
        we can also use deque, and first two go to alice and me, and bob gets left
        '''
        piles.sort()
        q = deque(piles)
        ans = 0
        
        while q:
            q.pop()
            ans += q.pop()
            q.popleft()
        
        return ans

class Solution:
    def maxCoins(self, piles: List[int]) -> int:
        '''
        bob will always get the smallest, so we can just assign the the bottom third to him
        then for the upper two thirds it alternates between alice and me
        '''
        piles.sort()
        ans = 0
        
        for i in range(len(piles) // 3, len(piles), 2):
            ans += piles[i]
        
        return ans   

#####################################################
# 1685. Sum of Absolute Differences in a Sorted Array
# 25NOV23
#####################################################
#not quite
class Solution:
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        '''
        the array is sorting in non-dcreasing order
        comptunig the summation of the absolute difference between nums[i] and all ther other elements in the array requires to go through all the elements
        [a,b,c,d]
        choose a
        abs(a-b) + abs(a-a) + abc(a-c) + abc(a-d)
        
        really its just
        abs(a-b) + abs(a-c) + abs(a-d)
        abs difference is
            max(a,b) - min(a,b)
        
        for nums[i] its:
            (nums[i] - nums[0]) + (nums[i] - nums[1]) + ... + (nums[i] - nums[i-1]) + ... + (nums[n-1] - nums[i])
        
        [a,b,c,d]
        (a - a) + (b - a) + (c - a) + (d - a)
        (b + c + d) - a*3 
        find pref and suff sums then substract out the number 
        
        (a - b) + (b-b) + (c-b) + (d-b)
        (a + c + d) - 3*b
        ''' 
        pref_sums = [0]
        for num in nums:
            pref_sums.append(pref_sums[-1] + num)
        
        suff_sums = [0]
        for num in reversed(nums):
            suff_sums.append(suff_sums[-1] + num)
        suff_sums.reverse()
        
        ans = []
        N = len(nums)
        
        for i in range(N):
            left_sum = pref_sums[i]
            right_sum = suff_sums[i]
            #print(left_sum,right_sum)
            temp = nums[i] + left_sum + right_sum - nums[i]
            temp -= nums[i]*(N-i-1)
            ans.append(temp)
        
        return ans
    
class Solution:
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        '''
        issue is that there could be repeated eleements, in which case the abs differnce is zero
        [a,b,c,d]
        if a == b === c == d, everything clears and its all just zeros
        that last algo didn't zero out the one where nums[i] could be multiple
        we can use pref_sum array to find the left sum, not inclduing nums[i]
            left_sum = pref_sum[i+1] - nums[i]
            #note actual inclusive sum would have pref_sum[i+1]
            right_sum = pref_sum[-1] - pref_sim[i+1]
        
        the sum of the absolute difference == the sum we would have to add to the numbers to make then equal to the current nums[i]
        number of elements to the left not including nums[i] is just i
            left_count = i
            right_count = N -i - 1
        
        we  can get the totals on each sides as:
            left_total = left_count*nums[i] - left_sum
            right_total = right_sum - right_count*nums[i]
        
        ans is left_total + right_total
        '''
        n = len(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        ans = []
        for i in range(len(nums)):
            left_sum = prefix[i+1] - nums[i]
            #essentially suffix sum
            #we dont need pref_sum and suff_sum if we know pref_sum
            right_sum = prefix[-1] - prefix[i+1]
            
            left_count = i
            right_count = n - i - 1
            
            #for the left side
            left_total = left_count * nums[i] - left_sum
            #for the right side
            right_total = right_sum - right_count * nums[i]

            ans.append(left_total + right_total)
        
        return ans
    
#one pass,pref_sum on the fly
class Solution:
    def getSumAbsoluteDifferences(self, nums: List[int]) -> List[int]:
        '''
        we dont need pref_sum array, we can just find it by building up left sum and substracting from the total_sum
        '''
        n = len(nums)
        total_sum = sum(nums)
        left_sum = 0
        
        ans = []
        for i in range(len(nums)):
            #essentially suffix sum
            #we dont need pref_sum and suff_sum if we know pref_sum
            right_sum = total_sum - left_sum - nums[i]
            
            left_count = i
            right_count = n - i - 1
            
            #for the left side
            left_total = left_count * nums[i] - left_sum
            #for the right side
            right_total = right_sum - right_count * nums[i]

            ans.append(left_total + right_total)
            left_sum += nums[i]
        
        return ans
