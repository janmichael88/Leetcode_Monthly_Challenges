##################################################
# 2865. Beautiful Towers I
# 02NOV26
##################################################
class Solution:
    def maximumSumOfHeights(self, heights: List[int]) -> int:
        '''
        we can only remove bricks from some towers
        left and right arrays
        '''
        n = len(heights)
        ans = 0
        for i in range(n):
            curr_height = heights[i]
            curr_sum = heights[i]
            #go left to right
            for j in range(i+1,n):
                curr_height = min(curr_height,heights[j])
                curr_sum += curr_height
            
            #go right to left
            curr_height = heights[i]
            for j in range(i-1,-1,-1):
                curr_height = min(curr_height,heights[j])
                curr_sum += curr_height
            ans = max(ans,curr_sum)
        
        return ans

###################################################
# 3760. Maximum Substrings With Distinct Start
# 02NOV26  
###################################################
class Solution:
    def maximumSumOfHeights(self, heights: List[int]) -> int:
        '''
        we can only remove bricks from some towers
        left and right arrays
        '''
        n = len(heights)
        ans = 0
        for i in range(n):
            curr_height = heights[i]
            curr_sum = heights[i]
            #go left to right
            for j in range(i+1,n):
                curr_height = min(curr_height,heights[j])
                curr_sum += curr_height
            
            #go right to left
            curr_height = heights[i]
            for j in range(i-1,-1,-1):
                curr_height = min(curr_height,heights[j])
                curr_sum += curr_height
            ans = max(ans,curr_sum)
        
        return ans

##################################################
# 3895. Count Digit Appearances
# 02NOV26
##################################################
class Solution:
    def countDigitOccurrences(self, nums: list[int], digit: int) -> int:
        '''
        count them
        '''
        ans = 0
        for num in nums:
            counts = Counter(str(num))
            ans += counts[str(digit)]
        
        return ans

##############################################
# 3736. Minimum Moves to Equal Array Elements III
# 03NOV26
##############################################
class Solution:
    def minMoves(self, nums: List[int]) -> int:
        '''
        raise to median?
        raise to max?
        '''
        max_ = max(nums)
        ans = 0
        for num in nums:
            ans += max_ - num
        return ans

######################################################################
# 3505. Minimum Operations to Make Elements Within K Subarrays Equal
# 03NOV26
######################################################################
#TLE
#need to optimize median window calculation
class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        '''
        target number for a subarray should be the median
        for each index i, find the median for i to i+x
        '''
        arr = []
        n = len(nums)
        #for each index i, find the median for i to i+x
        for i in range(n-x+1):
            sub = nums[i:i+x]
            sub.sort()
            arr.append(sub[x//2])
        
        #now do dp on this array
        #if i chose this subarray, i need [num - median] for num in the current subarray, then move on to the next subarry
        #kinda like knapsack

        memo = {}
        def dp(i,count):
            if i >= len(arr):
                if count == k:
                    return 0
                return float('inf')
            if count > k:
                return float('inf')
            
            if (i,count) in memo:
                return memo[(i,count)]
            
            #if i take this subarray compute steps
            cost = 0
            median = arr[i]
            for j in range(i,i+x):
                cost += abs(nums[j] - median)
            take = cost + dp(i+x+1,count+1)
            no_take = dp(i+1,count)
            ans = min(take,no_take)
            memo[(i,count)] = ans
            return ans

        return dp(0,0)

#supposed to work, but they want bottom up

from sortedcontainers import SortedList

class MedianFinder:
    def __init__(self):
        self.lo = SortedList()
        self.hi = SortedList()
        self.sum_lo = 0
        self.sum_hi = 0

    def addNum(self, num: int) -> None:
        if not self.lo or num <= self.lo[-1]:
            self.lo.add(num)
            self.sum_lo += num
        else:
            self.hi.add(num)
            self.sum_hi += num

        self._rebalance()

    def removeNum(self, num: int) -> None:
        if self.lo and num <= self.lo[-1]:
            self.lo.remove(num)
            self.sum_lo -= num
        else:
            self.hi.remove(num)
            self.sum_hi -= num

        self._rebalance()

    def _rebalance(self):
        # lo should have either the same size as hi,
        # or one more element.

        if len(self.lo) > len(self.hi) + 1:
            val = self.lo.pop()
            self.sum_lo -= val

            self.hi.add(val)
            self.sum_hi += val

        elif len(self.lo) < len(self.hi):
            val = self.hi.pop(0)
            self.sum_hi -= val

            self.lo.add(val)
            self.sum_lo += val

    def findMedian(self):
        return self.lo[-1]

    def getCost(self):
        median = self.findMedian()
        return (median * len(self.lo) - self.sum_lo + self.sum_hi - median * len(self.hi))

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        '''
        target number for a subarray should be the median
        for each index i, find the median for i to i+x
        optimize with SortedList and deque
        '''
        arr = []
        n = len(nums)
        q = deque([])
        mf = MedianFinder()

        # Calculate cost of every sliding window
        for num in nums:
            q.append(num)
            mf.addNum(num)

            if len(q) > x:
                temp = q.popleft()
                mf.removeNum(temp)

            if len(q) == x:
                arr.append(mf.getCost())

        # DP
        memo = {}

        def dp(i, count):
            if i >= len(arr):
                if count == k:
                    return 0
                return float('inf')

            if count > k:
                return float('inf')

            if (i, count) in memo:
                return memo[(i, count)]

            # Take this subarray
            cost = arr[i]
            take = cost + dp(i+x, count+1)

            # Don't take this subarray
            no_take = dp(i+1, count)

            ans = min(take, no_take)
            memo[(i, count)] = ans
            return ans

        return dp(0, 0)

#bottom up

from sortedcontainers import SortedList

class MedianFinder:
    def __init__(self):
        self.lo = SortedList()
        self.hi = SortedList()
        self.sum_lo = 0
        self.sum_hi = 0

    def addNum(self, num: int) -> None:
        if not self.lo or num <= self.lo[-1]:
            self.lo.add(num)
            self.sum_lo += num
        else:
            self.hi.add(num)
            self.sum_hi += num

        self._rebalance()

    def removeNum(self, num: int) -> None:
        if self.lo and num <= self.lo[-1]:
            self.lo.remove(num)
            self.sum_lo -= num
        else:
            self.hi.remove(num)
            self.sum_hi -= num

        self._rebalance()

    def _rebalance(self):
        # lo should have either the same size as hi,
        # or one more element.

        if len(self.lo) > len(self.hi) + 1:
            val = self.lo.pop()
            self.sum_lo -= val

            self.hi.add(val)
            self.sum_hi += val

        elif len(self.lo) < len(self.hi):
            val = self.hi.pop(0)
            self.sum_hi -= val

            self.lo.add(val)
            self.sum_lo += val

    def findMedian(self):
        return self.lo[-1]

    def getCost(self):
        median = self.findMedian()
        return (median * len(self.lo) - self.sum_lo + self.sum_hi - median * len(self.hi))

class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        '''
        target number for a subarray should be the median
        for each index i, find the median for i to i+x
        optimize with SortedList and deque
        '''
        arr = []
        n = len(nums)
        q = deque([])
        mf = MedianFinder()

        # Calculate cost of every sliding window
        for num in nums:
            q.append(num)
            mf.addNum(num)

            if len(q) > x:
                temp = q.popleft()
                mf.removeNum(temp)

            if len(q) == x:
                arr.append(mf.getCost())

        # Bottom-up DP
        m = len(arr)
        INF = float('inf')

        dp = [[INF] * (k+1) for _ in range(m+x+1)]

        # Base case: selecting 0 windows costs 0
        for i in range(m+x+1):
            dp[i][0] = 0

        # Iterate backwards, just like your recursion
        for i in range(m-1, -1, -1):
            for count in range(1, k+1):

                # Skip current window
                no_take = dp[i+1][count]

                # Take current window
                take = arr[i] + dp[i+x][count-1]

                dp[i][count] = min(take, no_take)

        return dp[0][k]

################################################
# 3294. Convert Doubly Linked List to Array II
# 06NOV26
###############################################
"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next
"""
class Solution:
    def toArray(self, node: 'Optional[Node]') -> List[int]:
        '''
        get the left and right parts
        '''
        curr = node
        left,right = [],[]
        while curr is not None:
            right.append(curr.val)
            curr = curr.next
        
        curr = node
        while curr is not None:
            left.append(curr.val)
            curr = curr.prev
        
        #print(left[::-1],right)
        return left[::-1] + right[1:]

############################################
# 1882. Process Tasks Using Servers
# 07NOV26
###########################################
class Solution:
    def assignTasks(self, servers: list[int], tasks: list[int]) -> list[int]:
        '''
        two heaps, available and unavailable
        keep track of time
        '''
        available = [(w, i) for i, w in enumerate(servers)]
        heapq.heapify(available)

        unavailable = []
        ans = []
        time = 0

        for j, t in enumerate(tasks):
            time = max(time, j)

            # Release servers available at current time
            while unavailable and unavailable[0][0] <= time:
                end_time, weight, index = heapq.heappop(unavailable)
                heapq.heappush(available, (weight, index))

            # If no server is available, jump to next completion
            if not available:
                time = unavailable[0][0]

                #make servers available, same as above
                while unavailable and unavailable[0][0] <= time:
                    end_time, weight, index = heapq.heappop(unavailable)
                    heapq.heappush(available, (weight, index))

            # Assign task
            weight, index = heapq.heappop(available)
            ans.append(index)

            heapq.heappush(unavailable, (time + t, weight, index))

        return ans

#method for maintaining variant
class Solution:
    def assignTasks(self, servers: list[int], tasks: list[int]) -> list[int]:
        '''
        two heaps, available and unavailable
        keep track of time
        '''
        available = [(w, i) for i, w in enumerate(servers)]
        heapq.heapify(available)

        unavailable = []
        ans = []
        time = 0

        def update(unavailable,available,time):
            while unavailable and unavailable[0][0] <= time:
                end_time, weight, index = heapq.heappop(unavailable)
                heapq.heappush(available, (weight, index))

        for j, t in enumerate(tasks):
            time = max(time, j)

            # Release servers available at current time
            update(unavailable,available,time)
            # If no server is available, jump to next completion
            if not available:
                #next available max time
                time = unavailable[0][0]

                #make servers available, same as above
                update(unavailable,available,time)
                
            # Assign task
            weight, index = heapq.heappop(available)
            ans.append(index)

            heapq.heappush(unavailable, (time + t, weight, index))

        return ans

######################################################
# 1606. Find Servers That Handled Most Number of Requests
# 08NOV26
######################################################
#TLE
class Solution:
    def busiestServers(self, k: int, arrival: list[int], load: list[int]) -> list[int]:
        '''
        we have a ring of servers of size k
        arrival time if reqhquest, and load time (the time it takes to complete)
        need to find the busiest sever(s)
            a sever us is busiest if it handles the most number of request succesffyly among all servers
        return list of busiest servers
        similar to 1882. Process Tasks Using Servers
        but we need to use servers in the ring in order
        brute force would be to keep going around the ring and assign
        '''
        hits = [0]*k
        times = [0]*k
        
        for request_index, (start_time, request_time) in enumerate(zip(arrival, load)):
            end_time = start_time + request_time
            start = request_index % k

            for offset in range(k):
                i = (start + offset) % k

                if times[i] <= start_time:
                    hits[i] += 1
                    times[i] = end_time
                    break

        max_hits = max(hits)
        return [i for i in range(k) if hits[i] == max_hits]

from sortedcontainers import SortedList
class Solution:
    def busiestServers(self, k: int, arrival: list[int], load: list[int]) -> list[int]:
        '''
        we have a ring of servers of size k
        arrival time if reqhquest, and load time (the time it takes to complete)
        need to find the busiest sever(s)
            a sever us is busiest if it handles the most number of request succesffyly among all servers
        return list of busiest servers
        similar to 1882. Process Tasks Using Servers
        but we need to use servers in the ring in order
        brute force would be to keep going around the ring and assign
        
        i can use SortedContainer to speed up the search
        '''
        hits = [0]*k
        unavailable = []
        available = SortedList([i for i in range(k)])

        for request_index, (start_time, request_time) in enumerate(zip(arrival, load)):
            end_time = start_time + request_time
            start = request_index % k

            #move from unavaiblae to available
            while unavailable and unavailable[0][0] <= start_time :
                curr_time,index = heapq.heappop(unavailable)
                available.add(index)

            #seach available with wrap around
            if available:
                target = available.bisect_left(start)
                idx = available[target] if target < len(available) else available[0] #revert to the 0th one from the beginning
                available.remove(idx)
                heapq.heappush(unavailable, (end_time,idx))
                hits[idx] += 1

        max_hits = max(hits)
        return [i for i in range(k) if hits[i] == max_hits]


###########################################
# 3693. Climbing Stairs II
# 09NOV26
############################################
class Solution:
    def climbStairs(self, n: int, costs: List[int]) -> int:
        '''
        this is just dp
        from i, we can go to i+1,i+1,i+3
        '''
        memo = {}

        def dp(i):
            if i >= n:
                return 0

            if i in memo:
                return memo[i]

            ans = float('inf')

            for j in range(i + 1, min(i + 3, n)+1):
                curr_cost = costs[j - 1] + (j - i) ** 2
                ans = min(ans, curr_cost + dp(j))

            memo[i] = ans
            return ans

        return dp(0)

#bottom up
class Solution:
    def climbStairs(self, n: int, costs: List[int]) -> int:
        '''
        this is just dp
        from i, we can go to i+1,i+1,i+3
        '''

        dp = [0]*(n+1)
        for i in range(n-1,-1,-1):
            ans = float('inf')
            for j in range(i + 1, min(i + 3, n)+1):
                curr_cost = costs[j - 1] + (j - i) ** 2
                ans = min(ans, curr_cost + dp[j])
            dp[i] = ans
        
        return dp[0]

#we can do this in constant time to and w/o the for loop, its either
#the jump dists are only 1,4,9
class Solution:
    def climbStairs(self, n, costs: List[int]) -> int:
        prev_three = prev_two = prev_one = 0

        for cost in costs:
            prev_three, prev_two, prev_one = (
                prev_two,
                prev_one,
                min(prev_three + 9, prev_two + 4, prev_one + 1) + cost
            )

        return prev_one