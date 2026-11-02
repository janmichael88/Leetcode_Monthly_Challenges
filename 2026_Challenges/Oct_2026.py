#################################################
# 2333. Minimum Sum of Squared Difference
# 10OCT26
#################################################
#diff counter
class Solution:
    def minSumSquareDiff(self, nums1: list[int], nums2: list[int], k1: int, k2: int) -> int:
        '''
        we can only dp +/- 1 to each element k1 or k2 times per array
        we want to minimuzize the sum squared differenece
        that means we want to minimize the individual squared difference
        which means we want to raise the smaller number and lower the larger number (the gap)
        but to which elements do we act on?
            can do this greedily
        but we can repeatedly do this k times on an element if we want to
        store as absolulte diffs, no we need to try to reduce each diff as small as possible
        heap?, yeah but k1 and k2 are too big
        i need to efficiently reduce the largest difference
        i need to be careful about the ordering
        '''
        diffs = [abs(u - v) for u, v in zip(nums1, nums2)]
        k = k1 + k2
        total = sum(diffs)

        if sum(diffs) <= k:
            return 0

        counts = Counter(diffs)

        for diff in range(max(diffs), 0, -1):
            cnt = counts[diff]
            if cnt == 0:
                continue

            if k >= cnt:
                k -= cnt
                counts[diff - 1] += cnt
                counts[diff] = 0
            else:
                counts[diff] -= k
                counts[diff - 1] += k
                k = 0


            if k == 0:
                break

        return sum(diff * diff * cnt for diff, cnt in counts.items())

#heap solution, tryick case handling repeated k operations
class Solution:
    def minSumSquareDiff(self, nums1: list[int], nums2: list[int], k1: int, k2: int) -> int:
        '''
        heap
        '''
        heap = [ -abs(x - y) for x, y in zip(nums1, nums2)]
        s = -sum(heap)
        
        #can bring them all down to zero
        if k1 + k2 >= s: 
            return 0
        #combine
        delta = k1 + k2
        heapify(heap)
        n = len(nums1)
        
        while delta > 0:
            d = -heappop(heap)
            #check if we can reduce each diff in heap by 1
            gap = max(delta//n, 1) if heap else delta
            #lower the differ
            d -= gap
            #push back
            heappush(heap, -d)
            #use up operations
            delta -= gap

        return sum(e*e for e in heap)

#############################################################
# 2897. Apply Operations on Array to Maximize Sum of Squares
# 13OCT26
#############################################################
class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        '''
        make each number as big as possible using bit wise ops
        if i pick two indices (i,j)
        nums[i] = nums[i] & nums[j]
        nums[j] = nums[i] | nums[j]
        make each number as big as possible and take the top k
        but whats a valid transformation given the operations?
            we transfer bits from one element to another

        try counting bits

        think like this too:
        Intuition: Think about this like gravity.

Look at one bit of i and j: the pairs (0,0), (1,1) and (0,1) are unchanged, and (1,0) becomes (0,1). 
So the operation does one thing: in every column where i has a chip and j has a gap, the chip falls from i to j. 
Chips never change columns.
        '''
        mod = 10**9 + 7
        #count current bits
        bit_counts = [0]*32
        
        for num in nums:
            for b in range(32):
                if num & (1 << b):
                    bit_counts[b] += 1

        ans = 0
        for _ in range(k):
            curr = 0
            for b in range(32):
                if bit_counts[b] > 0:
                    #contribute it
                    curr = curr | (1 << b)
                    bit_counts[b] -= 1
            ans = (ans + curr*curr) % mod

        return ans % mod

##################################################
# 3727. Maximum Alternating Sum of Squares
# 13OCT26
##################################################
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        '''
        indices contribute positively and odds contribute negativel
        numbers are all sqrd so it doest matter
        once we square them, we need to alternate going up as large as possibile, then down as small as possible
        two pointers
        '''
        n = len(nums)
        arr = [num*num for num in nums]
        arr.sort()
        op = 1
        l,r = 0,n-1
        ans = 0
        while l <= r:
            if op == 1:
                ans += arr[r]
                r -= 1
            else:
                ans -= arr[l]
                l += 1
            op *= -1
        return ans

#split half the elements
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            nums[i] *= nums[i]
        nums.sort()
        return -sum(nums[: n // 2]) + sum(nums[n // 2 :])

##################################################
# 3723. Maximize Sum of Squares of Digits
# 14OCT26
#################################################
class Solution:
    def maxSumOfSquares(self, num: int, SUM: int) -> str:
        '''
        a good digit needs num digits in it
        and the sum of its digits == sum
        the score is the sum of the squares of its digits
        return the maximim on
        construct the num greedily
        fill leftmost with nines as much as possible
        '''
        ans = [0]*num
        curr_sum = 0
        for i in range(num):
            for dig in range(9,-1,-1):
                if curr_sum + dig <= SUM:
                    ans[i] = dig
                    curr_sum += dig
                    break
        

        if sum(ans) == SUM:
            return "".join([str(n) for n in ans])
        return ""

#fill left most with nines, then the rest is just the remainder
class Solution:
    def maxSumOfSquares(self, num: int, sum: int) -> str:
        if sum > 9 * num:
            return ""
        q, r = divmod(sum, 9)
        ans = "9" * q
        if r > 0:
            ans += str(r)
        ans += "0" * (num - len(ans))
        return ans

####################################################
# 2913. Subarrays Distinct Element Sum of Squares I
# 15OCT26
#####################################################
class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        '''
        for each subarray, count distinct
        '''
        ans = 0
        n = len(nums)
        for i in range(n):
            curr = set()
            for j in range(i,n):
                curr.add(nums[j])
                ans += len(curr)**2
        
        return ans

#################################################
# 3551. Minimum Swaps to Sort by Digit Sum
# 18OCT26
#################################################
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        '''
        i can sort by criteria:
            (sum(dig for dig in num), num) 
            sort increasingly
        but we need to count swaps to get in sorted array
        problem now is how to find min swaps
        classic min swaps to sort a permutation
        each permutation cycle of length k takes k-1 swaps
        '''
        arr = []
        n = len(nums)
        mapp = {}
        for i,num in enumerate(nums):
            mapp[num] = i
            digits = [int(dig) for dig in str(num)]
            arr.append((sum(digits),num))
        
        arr.sort()
        new_sorted = [num for _,num in arr]
        target_indices = [mapp[num] for num in new_sorted]

        swaps = 0
        seen = set()
        for i in range(n):
            if i in seen:
                continue
            cycle_length = 0
            j = i
            while j not in seen:
                seen.add(j)
                cycle_length += 1
                j = target_indices[j]
            if cycle_length > 1:
                swaps += cycle_length - 1

        return swaps

###################################################
# 3886. Sum of Sortable Integers
# 19OCT26
###################################################
#TLE
class Solution:
    def sortableIntegers(self, nums: list[int]) -> int:
        '''
        if n = len(nums)
        the candidate k's are ones the divide n
        for each candidate among k, partition into subarrays of length k
            there must be subarrays if length k
        so this limits our search on k
        we can only rotate a block
        we can just concat! and check if it exsists at all for each block
        '''

        def is_rotation(block,target):
            doubled = block + block
            k = len(block)
            return any(doubled[i:i+k] == target for i in range(k))

        n = len(nums)
        arr = sorted(nums)
        ans = 0
        for size in range(1,n+1):
            if n % size == 0:
                valid = True
                for i in range(0,n,size):
                    block = nums[i:i+size]
                    target = arr[i:i+size]
                    if not is_rotation(block,target):
                        valid = False
                        break
                
                if valid:
                    ans += size
        
        return ans

#couting peaks
class Solution:
    def sortableIntegers(self, nums: list[int]) -> int:
        '''
        if n = len(nums)
        the candidate k's are ones the divide n
        for each candidate among k, partition into subarrays of length k
            there must be subarrays if length k
        so this limits our search on k
        we can only rotate a block
        we can just concat! and check if it exsists at all for each block
        '''

        n = len(nums)
        arr = sorted(nums)
        ans = 0

        #check they contain same elements, including duplicates
        #block has at most on cylic decsnse
        def is_rotation(block, target):
            k = len(block)
            
            if Counter(block) != Counter(target):
                return False

            #count peaks in block
            descents = sum(block[i] > block[(i + 1) % k] for i in range(k))
            
            return descents <= 1

        for size in range(1, n + 1):
            if n % size != 0:
                continue

            valid = True

            for i in range(0, n, size):
                block = nums[i:i + size]
                target = arr[i:i + size]

                if not is_rotation(block, target):
                    valid = False
                    break

            if valid:
                ans += size

        return ans

################################################
# 3863. Minimum Operations to Sort a String
# 21OCT26
################################################
class Solution:
    def minOperations(self, s: str) -> int:
        '''
        we are only allowed to pick a substring that isn't the whole string and sort it
        we need min ops of these 
        strict edge casing i gues....

        if the string is already sorted, return 0
        if the fist char is min or the last char is max
            sort the inner part between 0 and n-1 for 1 one opertaion
        if we dont have an anchor, we need to move at least on min char to the front or one max char to the end
        Case	Condition	                                        Return
        0	    s is sorted	                                        0
        1	    First character is min, or last character is max	1
        2	    No anchor, (an inner min or max exists)	            2
        3	    Reversed extrema, (no inner min or max exists)	    3
        E	    len=2, and s is not sorted	                       -1
        '''
        n = len(s)
        mn = mx = s[0]
        isSorted = 1

        for i in range(1, n):
            if s[i] < s[i - 1]:
                isSorted = 0
            mn = min(mn, s[i])
            mx = max(mx, s[i])

        if isSorted: 
            return 0
        if n == 2: 
            return -1
        
        if s[0] == mn or s[-1] == mx:
            return 1

        for i in range(1, n - 1):
            if s[i] == mn or s[i] == mx:
                return 2

        return 3

#################################################
# 3644. Maximum K to Sort a Permutation
# 24OCT26
#################################################
class Solution:
    def sortPermutation(self, nums: List[int]) -> int:
        '''
        we can swap indices (i,j) if nums[i] & nums[j] == k
        return the maximum k value that can sort the array in non decreasing order using 'any' number of swaps
        if a number is out of place, we need to swap it, so we need that bit position
            take bitwise AND of all lement snot in there correct position
        '''

        sorted_nums = sorted(nums)
        ans = None
        for i,num in enumerate(nums):
            if num != sorted_nums[i]:
                if ans is None:
                    ans = num
                else:
                    ans = ans & num
        
        
        return ans if ans is not None else 0 

########################################################
# 2409. Count Days Spent Together
# 24OCT26
###########################################################
class Solution:
    def countDaysTogether(self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str) -> int:
        '''
        hash of common, return size
        arrival days are earlier than leave days
        '''
        limits = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        def get_dates(start,end):
            a,b = start.split("-")
            c,d = end.split("-")

            in_rome = set()

            a,b,c,d = int(a),int(b),int(c),int(d)
            while True:
                #log in day
                in_rome.add((a,b))
                if [a,b] == [c,d]:
                    break
                #check we reach end of month
                if b == limits[a-1]:
                    b = 1
                    a += 1
                else:
                    b += 1
            
            return in_rome
        alice_rome = get_dates(arriveAlice,leaveAlice)
        bob_rome = get_dates(arriveBob,leaveBob)
        both_rome = alice_rome & bob_rome
        return len(both_rome)



#convert to day of year, then its just diff
class Solution:
    def countDaysTogether(
        self, arriveAlice: str, leaveAlice: str, arriveBob: str, leaveBob: str
    ) -> int:
        datesOfMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        prefixSum = [0]
        for date in datesOfMonths:
            prefixSum.append(prefixSum[-1] + date)

        arriveAliceDay = self.calculateDayOfYear(arriveAlice, prefixSum)
        leaveAliceDay = self.calculateDayOfYear(leaveAlice, prefixSum)
        arriveBobDay = self.calculateDayOfYear(arriveBob, prefixSum)
        leaveBobDay = self.calculateDayOfYear(leaveBob, prefixSum)
        return max(
            0,
            min(leaveAliceDay, leaveBobDay)
            - max(arriveAliceDay, arriveBobDay)
            + 1,
        )

    def calculateDayOfYear(self, day: str, prefixSum: List[int]) -> int:
        month = int(day[:2])
        date = int(day[3:])
        return prefixSum[month - 1] + date

#################################################
# 3619. Count Islands With Total Value Divisible by K
# 26OCT26
#################################################
class Solution:
    def countIslands(self, grid: List[List[int]], k: int) -> int:
        '''
        looks like we need to check all islands
        find all componenets
        '''
        rows,cols = len(grid),len(grid[0])
        seen = set()
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]

        def dfs(i,j,seen,comp):
            seen.add((i,j))
            comp.append((i,j))
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    if grid[neigh_x][neigh_y] > 0 and (neigh_x,neigh_y) not in seen:
                        dfs(neigh_x,neigh_y,seen,comp)
        
        comps = []
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] > 0 and (i,j) not in seen:
                    comp = []
                    dfs(i,j,seen,comp)
                    comps.append(comp)
        ans = 0

        for c in comps:
            curr_sum = 0
            for i,j in c:
                curr_sum += grid[i][j]
            if curr_sum % k == 0:
                ans += 1
        
        return ans

#return sum
#while dfsing, make grid[i][j] = 0
class Solution:
    def countIslands(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        dirs = (-1, 0, 1, 0, -1)

        def dfs(i: int, j: int) -> int:
            total = grid[i][j]
            grid[i][j] = 0
            for d in range(4):
                x, y = i + dirs[d], j + dirs[d + 1]
                if 0 <= x < m and 0 <= y < n and grid[x][y] > 0:
                    total += dfs(x, y)
            return total

        return sum(grid[i][j] > 0 and dfs(i, j) % k == 0 for i in range(m) for j in range(n))

################################################
# 3963. Create Grid With Exactly One Path
# 26OCT26
################################################
class Solution:
    def createGrid(self, m: int, n: int) -> list[str]:
        '''
        need to make onl on path
        '''
        ans = [["#"]*n for _ in range(m)]

        for j in range(n):
            ans[0][j] = "."
        
        for i in range(m):
            ans[i][n-1] = "."
        
        return ["".join(r) for r in ans]

#############################################
# 3882. Minimum XOR Path in a Grid
# 29OCT26
#############################################
class Solution:
    def minCost(self, grid: list[list[int]]) -> int:
        '''
        dfs, can't just xor each state, 
        i could xor with another cell that would make it smaller
            i.e storing a local minimum doen't result in a global minimum
            notice how each grid[i][j] is between [0,1023]
        '''
        rows, cols = len(grid), len(grid[0])
        dp = [[set() for _ in range(cols)] for _ in range(rows)]
        dp[0][0].add(grid[0][0])

        for i in range(rows):
            for j in range(cols):
                if i == 0 and j == 0:
                    continue

                if i > 0:
                    for x in dp[i-1][j]:
                        dp[i][j].add(x ^ grid[i][j])

                if j > 0:
                    for x in dp[i][j-1]:
                        dp[i][j].add(x ^ grid[i][j])

        return min(dp[rows-1][cols-1])
        
######################################################################
# 3970. Shortest Path With At Most K Consecutive Identical Characters
# 30OCT26
######################################################################
#almost
class Solution:
    def shortestPath(self, n: int, edges: List[List[int]], labels: str, k: int) -> int:
        '''
        djikstras
        path can only have k consectuive identical labels
        '''
        graph = defaultdict(list)
        for u,v,w in edges:
            graph[u].append((v,w))

        dists = [float('inf')]*n
        visited = set()
        dists[0] = 0
        visited.add(0)
        pq = [(0,1,0)] #(curr_dist,count,node)

        while pq:
            curr_dist,count,node = heapq.heappop(pq)
            if dists[node] < curr_dist:
                continue
            visited.add(node)

            for neigh,dist_to in graph[node]:
                if neigh in visited:
                    continue
                new_dist = dists[node] + dist_to
                if labels[node] == labels[neigh]:
                    new_count = count + 1
                else:
                    new_count = 1

                if new_count > k:
                    continue
                if new_dist < dists[neigh]:
                    dists[neigh] = new_dist
                    heapq.heappush(pq, (new_dist,new_count,neigh))
        ans = dists[n-1]
        if ans == float('inf'):
            return -1
        return ans

#need states (node,count), then minimume over that
class Solution:
    def shortestPath(self, n: int, edges: List[List[int]], labels: str, k: int) -> int:
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))

        dists = [[float('inf')] * (k + 1) for _ in range(n)]
        dists[0][1] = 0

        pq = [(0, 1, 0)]

        while pq:
            curr_dist, count, node = heapq.heappop(pq)

            if curr_dist > dists[node][count]:
                continue

            if node == n - 1:
                return curr_dist

            for neigh, dist_to in graph[node]:
                new_dist = curr_dist + dist_to
                new_count = count + 1 if labels[node] == labels[neigh] else 1

                if new_count > k:
                    continue

                if new_dist < dists[neigh][new_count]:
                    dists[neigh][new_count] = new_dist
                    heapq.heappush(pq, (new_dist, new_count, neigh))

        return -1

###########################################
# 3665. Twisted Mirror Path Count
# 01NOV26
###########################################
class Solution:
    def uniquePaths(self, grid: List[List[int]]) -> int:
        MOD = 10**9 + 7
        rows, cols = len(grid), len(grid[0])
        memo = {}

        def resolve(i, j, moving_right):
            # Follow reflections until we reach an empty cell
            while 0 <= i < rows and 0 <= j < cols and grid[i][j] == 1:

                if moving_right:
                    # Entered mirror while moving right:
                    # reflect down
                    i += 1
                else:
                    # Entered mirror while moving down:
                    # reflect right
                    j += 1

                moving_right = not moving_right

            if i >= rows or j >= cols:
                return None

            return (i, j)

        def dp(i, j):
            if (i, j) == (rows - 1, cols - 1):
                return 1

            if (i, j) in memo:
                return memo[(i, j)]

            ways = 0

            # Try moving right
            nxt = resolve(i, j + 1, True)
            if nxt is not None:
                ways += dp(*nxt)

            # Try moving down
            nxt = resolve(i + 1, j, False)
            if nxt is not None:
                ways += dp(*nxt)

            memo[(i, j)] = ways % MOD
            return memo[(i, j)]

        return dp(0, 0)