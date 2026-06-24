###################################################
# 2144. Minimum Cost of Buying Candies With Discount
# 01JUN26
######################################################
class Solution:
    def minimumCost(self, cost: List[int]) -> int:
        '''
        sort, and take the first 2, then the third for free
        '''
        cost.sort(reverse = True)
        ans = 0
        n = len(cost)
        for i in range(n):
            if (i + 1) % 3 == 0:
                continue
            ans += cost[i]
        return ans

#######################################################
# 3633. Earliest Finish Time for Land and Water Rides I
# 02JUN26
#######################################################
class Solution:
    def earliestFinishTime(self, landStartTime: List[int], landDuration: List[int], waterStartTime: List[int], waterDuration: List[int]) -> int:
        '''
        we can get the intervals
        we need earliest time tourist can finish both rides
        example
        not really an intervals question
        we pick a ride (land or water) at its start or any later moment, and go on for its duration
        try all pairwise rides
        '''
        n = len(landStartTime)
        m = len(waterStartTime)

        ans = float('inf')
        for i in range(n):
            land_start,land_duration = landStartTime[i],landDuration[i]
            for j in range(m):
                water_start,water_duration = waterStartTime[j],waterDuration[j]
                #can either for land first or water first, could be the case where we have to wait too
                op1 = max(land_start + land_duration,water_start) + water_duration
                op2 = max(water_start + water_duration,land_start) + land_duration
                ans = min(ans,op1,op2)

        return ans
    
#######################################################
# 3635. Earliest Finish Time for Land and Water Rides II
# 02JUN26
#######################################################
class Solution:
    def earliestFinishTime(self, landStartTime: List[int], landDuration: List[int], waterStartTime: List[int], waterDuration: List[int]) -> int:
        '''
        for a fixed ride finish time, what the best second best ride? can solve doing binary search
        say we pick the land ride, whatss the best water ride
        say we pick a water ride, whats the best land ride
        let A be the time we finish a land ride:
        A = landStartTime[i] + landDuration[i]
        then land water is max(A,waterStart) + waterDuration
        if already started
            waterStart <= A
            finish = A + waterDuration
        if waster hasnt started
            waterStart > A
            finish = waterStart + waterDuration
        so for a given A we have:
            min(A + min(waterDuration where waterStart <= A),min(waterStart + waterDuration where waterStart > A))

        precomputiation
            water = sorted(zip(waterStartTime,waterDuation))
            starts = [w[0] for w in water]
            dur = [w[1] for w in water]
        
        do pref min on duration
        pref[i] = min duration among rides 0..i
        which gives us min(waterDuration where waterStart <= A) after binary search

        suff min of start + duration
        suff[i] = min(starts[j] + dur[j] for j >= i)
        which gives min(waterStart + waterDuration where waterStart > A) after binary search

        for each land ride
        A = landStart + landDuration
        idx = bisect_right(starts, A)

        and then validate based on criteria
        # waterStart <= A
        if idx > 0:
            ans = min(ans, A + pref[idx-1])

        # waterStart > A
        if idx < m:
            ans = min(ans, suff[idx]) 
        '''
        def build(starts, durations):
            rides = sorted(zip(starts, durations))
            s = [x[0] for x in rides]
            d = [x[1] for x in rides]
            n = len(rides)

            # pref[i] = min duration among rides[0..i]
            pref = [0] * n
            pref[0] = d[0]
            for i in range(1, n):
                pref[i] = min(pref[i - 1], d[i])

            # suff[i] = min(start + duration) among rides[i..]
            suff = [0] * n
            suff[-1] = s[-1] + d[-1]
            for i in range(n - 2, -1, -1):
                suff[i] = min(suff[i + 1], s[i] + d[i])

            return s, pref, suff
        
        def query(finish_time, starts, pref, suff):
            n = len(starts)
            idx = bisect_right(starts, finish_time)

            res = float("inf")

            # there is an opposite ride where start <= finish_time
            if idx > 0:
                res = min(res, finish_time + pref[idx - 1])

            # there isn't a ride that hasn not started yet; rides with start > finish_time
            if idx < n:
                res = min(res, suff[idx])

            return res
        
        water_starts, water_pref, water_suff = build(waterStartTime, waterDuration)
        land_starts, land_pref, land_suff = build(landStartTime, landDuration)
        ans = float("inf")

        # land to water
        for ls, ld in zip(landStartTime, landDuration):
            finish_land = ls + ld
            temp = query(finish_land,water_starts,water_pref,water_suff)
            ans = min(ans,temp)

        # water to land
        for ws, wd in zip(waterStartTime, waterDuration):
            finish_water = ws + wd
            temp = query(finish_water,land_starts,land_pref,land_suff)
            ans = min(ans,temp)

        return ans
    
#############################################
# 3751. Total Waviness of Numbers in Range I
# 04JUN26
##############################################
class Solution:
    def totalWaviness(self, num1: int, num2: int) -> int:
        '''
        count peaks and valleys in number
        '''
        def compute(num):
            num = str(num)
            n = len(num)
            waviness = 0
            for i in range(1,n-1):
                if num[i-1] < num[i] > num[i+1]:
                    waviness += 1
                elif num[i-1] > num[i] < num[i+1]:
                    waviness += 1
            
            return waviness
        
        ans = 0
        for num in range(num1, num2+1):
            ans += compute(num)
        
        return ans
    
######################################################
# 3753. Total Waviness of Numbers in Range II
# 05JUN06
########################################################
class Solution:
    def totalWaviness(self, num1: int, num2: int) -> int:
        '''
        digit dp
        do dp(num2) - dp(num1-1)
        how to build 
        its of the form (pos,tight,started,state) usually
        but now its Build a digit-DP state (position, tight, lastDigit, secondLastDigit)
        say im building a number and we are at position i
        its a peak if the digit at dig_at(i-1) < i > dig_at(i+1)
        '''
        @cache
        def dp(pos, prev, curr, tight, leading, num):

            s = str(num)
            n = len(s)

            if pos == n:
                return (1, 0)

            limit = int(s[pos]) if tight else 9

            total_count = 0
            total_wavy = 0

            for d in range(limit + 1):
                
                ntight = tight and d == limit
                nleading = leading and d == 0

                if nleading:
                    nprev = -1
                    ncurr = -1

                elif leading:
                    nprev = -1
                    ncurr = d

                else:
                    nprev = curr
                    ncurr = d

                child_count, child_wavy = dp(pos + 1,nprev,ncurr,ntight,nleading,num)

                total_count += child_count
                total_wavy += child_wavy
                
                #wavy state (keeping tracking of pref and curr digit)
                if prev >= 0 and curr >= 0:
                    if (prev < curr > d) or(prev > curr < d):
                        #total_wayv should go up buy count
                        total_wavy += child_count

            return total_count, total_wavy
                
        _, wav2 = dp(0, -1, -1, True, True, num2)
        #dp.cache_clear()
        #can do without cache clear
        _, wav1 = dp(0, -1, -1, True, True, num1 - 1)
        answer = wav2 - wav1
        return answer
        
#####################################################
# 1031. Maximum Sum of Two Non-Overlapping Subarrays
# 08JUN26
######################################################
class Solution:
    def maxSumTwoNoOverlap(self, nums: List[int], firstLen: int, secondLen: int) -> int:
        '''
        use pref sum to fix the over subarray
        then use it again to find the best sum that occurs before or after that sub array
        remember we are limited to the size of first len, so we don't need to check all (i,j) endpoints for the first

        '''
        n = len(nums)
        pref_sum = [0]
        for i in range(n):
            pref_sum.append(nums[i] + pref_sum[-1])
        
        ans = 0
        for i in range(n - firstLen + 1):
            sum1 = pref_sum[i + firstLen] - pref_sum[i]

            # before
            for j in range(i):
                if j + secondLen <= i:
                    sum2 = pref_sum[j + secondLen] - pref_sum[j]
                    ans = max(ans, sum1 + sum2)

            # after
            for j in range(i + firstLen, n - secondLen + 1):
                sum2 = pref_sum[j + secondLen] - pref_sum[j]
                ans = max(ans, sum1 + sum2)
        return ans
    
############################################
# 3689. Maximum Total Subarray Value I
# 09JUN26
############################################
class Solution:
    def maxTotalValue(self, nums: List[int], k: int) -> int:
        '''
        welp the hint was not helpful
        need to choose k subarrays, could be overlapping
        the value of a subarry is max(sub) - min(sub)
        total value os the sum of these values of all chosen subarrays
        we need to maximize total value
        
        exact same subarray can be chosen more than once
        its just (max(nums) - min(nums))*k
        wtf lmaooo
        '''

        return k*(max(nums) - min(nums))
    
#############################################
# 3691. Maximum Total Subarray Value II
# 10JUN26
##############################################
import math
class Solution:
    def maxTotalValue(self, nums: List[int], k: int) -> int:
        '''
        i can use range min range max queries
        can you sqrt decomp or segment tree for min and max
        you have opiones for RMQ, sprase tables, seg tree, bishop fenwick
        but ask if you need to update between points or even between queries
        sqrt decomp is still too slow
        '''
        n = len(nums)
        b = int(math.sqrt(n)) + 1 #avoid the case when = 0
        blocks = math.ceil(n / b)

        block_min = [float('inf')] * blocks
        block_max = [float('-inf')] * blocks

        for i, x in enumerate(nums):
            block = i // b
            block_min[block] = min(block_min[block], x)
            block_max[block] = max(block_max[block], x)
        
        def query_min(l, r):
            ans = float('inf')

            while l <= r:
                # entire block is inside [l, r]
                if l % b == 0 and l + b - 1 <= r:
                    ans = min(ans, block_min[l // b])
                    l += b
                else:
                    ans = min(ans, nums[l])
                    l += 1

            return ans
        
        def query_max(l, r):
            ans = float('-inf')

            while l <= r:
                # entire block is inside [l, r]
                if l % b == 0 and l + b - 1 <= r:
                    ans = max(ans, block_max[l // b])
                    l += b
                else:
                    ans = max(ans, nums[l])
                    l += 1

            return ans
        
        pq = []
        for l in range(n):
            pq.append((-(query_max(l, n - 1) - query_min(l, n - 1)), l, n - 1))
        
        heapq.heapify(pq)
        ans = 0
        while k:
            negVal, l, r = heapq.heappop(pq)
            ans -= negVal
            k -= 1
            if r > l:
                heapq.heappush(pq, (-(query_max(l, r - 1) - query_min(l, r - 1)), l, r - 1))
        return ans

class SegTree:
    def __init__(self, nums: List[int]):
        self.n = len(nums)
        self.maxv = [0] * (4 * self.n)
        self.minv = [0] * (4 * self.n)
        self.build(1, 0, self.n - 1, nums)

    def build(self, node: int, l: int, r: int, nums: List[int]):
        if l == r:
            self.maxv[node] = self.minv[node] = nums[l]
            return
        m = (l + r) // 2
        self.build(node * 2, l, m, nums)
        self.build(node * 2 + 1, m + 1, r, nums)
        self.maxv[node] = max(self.maxv[node * 2], self.maxv[node * 2 + 1])
        self.minv[node] = min(self.minv[node * 2], self.minv[node * 2 + 1])

    def queryMax(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        if ql <= l and r <= qr:
            return self.maxv[node]
        m = (l + r) // 2
        res = float('-inf')
        if ql <= m:
            res = max(res, self.queryMax(node * 2, l, m, ql, qr))
        if qr > m:
            res = max(res, self.queryMax(node * 2 + 1, m + 1, r, ql, qr))
        return res

    def queryMin(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        if ql <= l and r <= qr:
            return self.minv[node]
        m = (l + r) // 2
        res = float('inf')
        if ql <= m:
            res = min(res, self.queryMin(node * 2, l, m, ql, qr))
        if qr > m:
            res = min(res, self.queryMin(node * 2 + 1, m + 1, r, ql, qr))
        return res


class Solution:
    def maxTotalValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        seg = SegTree(nums)
        pq = []
        for l in range(n):
            curr_max = seg.queryMax(1, 0, n - 1, l, n - 1)
            curr_min = seg.queryMin(1, 0, n - 1, l, n - 1)
            value = curr_max - curr_min
            entry  = (-value,l,n-1)
            pq.append(entry)

        heapq.heapify(pq)
        ans = 0
        while k:
            negVal, l, r = heapq.heappop(pq)
            ans -= negVal
            k -= 1
            if r > l:
                curr_max = seg.queryMax(1, 0, n - 1, l, r - 1)
                curr_min = seg.queryMin(1, 0, n - 1, l, r - 1)
                value = curr_max - curr_min
                entry = (-value,l,r-1)
                heapq.heappush(pq,entry)
        
        return ans
    
##################################################
# 3612. Process String with Special Operations I
# 16JUN26
##################################################
class Solution:
    def processStr(self, s: str) -> str:
        '''
        just follow the rules,

        '''
        res = []
        for ch in s:
            if ch.islower():
                res.append(ch)
            elif ch == "*" and res:
                res.pop()
            elif ch == "#":
                res.extend(res)
            else:
                res = res[::-1]
        
        return "".join(res)
            
####################################################
# 3614. Process String with Special Operations II
# 17JUN26
######################################################
#FML.,..
class Solution:
    def processStr(self, s: str, k: int) -> str:
        '''
        Instead of asking what the kth character is after all operations,
        work backwards and ask where that character came from before the
        most recent operation.
        '''

        # Compute the final length without constructing the string.
        length = 0
        for c in s:
            if c.islower():
                length += 1
            elif c == '*' and length > 0:
                length -= 1
            elif c == '#':
                length *= 2

        if k >= length:
            return '.'

        for i in range(len(s) - 1, -1, -1):
            c = s[i]

            # Before appending c, the string length was length - 1.
            # The appended character occupies index length - 1.
            if c.islower():
                if k == length - 1:
                    return c

                # Undo the append.
                length -= 1

            # Forward operation deleted one character,
            # so undo it by restoring the length.
            elif c == '*':
                length += 1

            # Forward operation duplicated the entire string:
            # length -> 2 * length.
            # Undo by recovering the original length.
            elif c == '#':
                length //= 2

                # If k is in the second copy, map it back
                # to the corresponding position in the first copy.
                if k >= length:
                    k -= length

            # Reversal does not change length.
            # Map k back to its position before the reversal.
            elif c == '%':
                k = length - 1 - k

        return '.'
    
#############################################
# 1840. Maximum Building Height
# 20JUN26
##############################################
#scratch work attempt 1
class Solution:
    def maxBuilding(self, n: int, arr: List[List[int]]) -> int:
        '''
        yikesss
        first building must be 0, each building must be non-negative
        diff between any two buildings cannot exceed 1
        and we have height restrictions on each building
        we only want to record the tallest building
        doesnt make sense to fix each building to its resrictions
        if we have some indices (i,j) that are fixed at heights, the best strategy would be make then increasing by 1 in between
        notice there has to be at least two buildings, they are very specific
        say we have (i,hi) as ith building with restriction hi
            then the building at i-1 <= hi + 1
            then the building at i + 1 <= hi + 1
        
        generally for a the height of a building j < hi + abs(i-j)
        i got the limit part, fix each building with limit
        for each possible restriction (left, right) it woull be increasing by 1 up to that point
        adjacent restrictions are no the tightst bount
        we need to tighten the retrictions in both directions
        then find the peak between adjacent restrictions
        '''
        arr.extend([[1, 0], [n, n - 1]])
        arr.sort()  # sort by building index

        m = len(arr)

        # Left -> Right pass
        #
        # If building i-1 has maximum feasible height h,
        # then building i cannot exceed:
        #
        #     h + distance_between_buildings
        #
        # because height can increase by at most 1 per step.
        #
        # Tighten each restriction accordingly.
        for i in range(1, m):
            arr[i][1] = min(arr[i][1],arr[i - 1][1] + (arr[i][0] - arr[i - 1][0])
            )

        # Right -> Left pass
        #
        # Same idea, but now enforce constraints coming from the right.
        #
        # If building i+1 has maximum feasible height h,
        # then building i cannot exceed:
        #
        #     h + distance_between_buildings
        #
        # because height can decrease by at most 1 per step.
        for i in range(m - 2, -1, -1):
            arr[i][1] = min(arr[i][1],arr[i + 1][1] + (arr[i + 1][0] - arr[i][0]))

        ans = 0

        # After both passes, all restrictions are globally feasible.
        #
        # Consider each adjacent pair of restrictions:
        #
        #   (l, h1) ------------ (r, h2)
        #
        # We want the tallest peak that can exist between them.
        for i in range(1, m):
            l, h1 = arr[i - 1]
            r, h2 = arr[i]

            distance = r - l

            # Some of the distance is "used up" matching the
            # endpoint height difference.
            #
            # Example:
            #   h1 = 3, h2 = 7
            #
            # We already need 4 steps just to go from 3 -> 7.
            #
            # Remaining distance can be split between
            # climbing and descending around a peak.
            #But the slope rule (|h[i] - h[i-1]| <= 1) creates additional restrictions on those restrictions.
            extra = (distance - abs(h1 - h2)) // 2
            peak = max(h1, h2) + extra
            ans = max(ans, peak)

        return ans
    
############################################
# 3699. Number of ZigZag Arrays I
# 23JUN26
##############################################
#TLE
class Solution:
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        '''
        dp states (i,dir,ending)
        this is trivial dp, but we need to speed it up
        '''
        memo = {}
        mod = 10**9 + 7
        def dp(i,direction,ending):
            if i == n:
                return 1
            if (i,direction,ending) in memo:
                return memo[(i,direction,ending)]
            ways = 0
            #must go up
            if direction == 0:
                for x in range(ending+1,r+1):
                    ways += dp(i+1,1,x)
            else:
                for x in range(l,ending):
                    ways += dp(i+1,0,x)
            
            ways = ways % mod
            memo[(i,direction,ending)] = ways
            return ways
        
        ans = 0
        for num in range(l,r+1):
            ans += dp(1,0,num)
            ans += dp(1,1,num)
        
        ans = ans % mod
        return ans
    
#instead of calling it for each num between (l and r)
class Solution:
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        '''
        dp states (i,dir,ending)
        this is trivial dp, but we need to speed it up
        '''
        memo = {}
        mod = 10**9 + 7
        def dp(i,direction,ending):
            if i == n:
                return 1
            if (i,direction,ending) in memo:
                return memo[(i,direction,ending)]
            ways = 0
            if i == 0:
                for num in range(l,r+1):
                    ways += dp(1,0,num)
                    ways += dp(1,1,num)
            #must go up
            elif direction == 0:
                for x in range(ending+1,r+1):
                    ways += dp(i+1,1,x)
            elif direction == 1:
                for x in range(l,ending):
                    ways += dp(i+1,0,x)
            
            ways = ways % mod
            memo[(i,direction,ending)] = ways
            return ways
        
        ans = dp(0,-1,-1)
        return ans % mod
    
#now doing bottom up
class Solution:
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        MOD = 10**9 + 7

        dp = {}

        for val in range(l, r + 1):
            dp[(n, 0, val)] = 1
            dp[(n, 1, val)] = 1

        for i in range(n - 1, 0, -1):
            for ending in range(l, r + 1):

                ways = 0
                for x in range(ending + 1, r + 1):
                    ways += dp[(i + 1, 1, x)]

                dp[(i, 0, ending)] = ways % MOD

                ways = 0
                for x in range(l, ending):
                    ways += dp[(i + 1, 0, x)]

                dp[(i, 1, ending)] = ways % MOD

        ans = 0

        for num in range(l, r + 1):
            ans += dp[(1, 0, num)]
            ans += dp[(1, 1, num)]

        return ans % MOD
    
#bottom up with faster sum query
class Solution:
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        '''
        Let m = r - l + 1 be the number of values available.

        Original DP:

            dp(i, 0, val) =
                sum(dp(i + 1, 1, x) for x > val)

            dp(i, 1, val) =
                sum(dp(i + 1, 0, x) for x < val)

        where:
            i   = current position
            val = previous value chosen
            0   = next value must be larger
            1   = next value must be smaller

        The naive solution is O(n * m²) because every state loops
        through all possible next values.

        Observe that each transition is just a range sum on the next layer:

            dp(i, 0, val)
                = sum of all states strictly greater than val

            dp(i, 1, val)
                = sum of all states strictly smaller than val

        If we store:

            down[v] = dp(i + 1, 1, v)
            up[v]   = dp(i + 1, 0, v)

        then every transition becomes either a suffix sum or a prefix sum.

        Example:

            dp(i, 0, v)
                = sum(down[x] for x > v)

            dp(i, 1, v)
                = sum(up[x] for x < v)

        By building prefix sums of the next layer, we can answer
        both of these in O(1), reducing the overall complexity to:

            Time:  O(n * m)
            Space: O(m)
        '''

        MOD = 10**9 + 7
        m = r - l + 1

        # Base case:
        # We have already placed all n elements.
        # Regardless of the required direction, there is exactly
        # one valid way to finish (do nothing).
        up = [1] * m
        down = [1] * m

        # Build DP layers backwards.
        # After each iteration:
        #
        #   up[v]   = dp(i, 0, v)
        #   down[v] = dp(i, 1, v)
        #
        for _ in range(n - 1):

            # Prefix sums for the next layer.
            pref_up = [0] * m
            pref_down = [0] * m

            pref_up[0] = up[0]
            pref_down[0] = down[0]

            for v in range(1, m):
                pref_up[v] = (pref_up[v - 1] + up[v]) % MOD
                pref_down[v] = (pref_down[v - 1] + down[v]) % MOD

            total_up = pref_up[-1]
            total_down = pref_down[-1]

            new_up = [0] * m
            new_down = [0] * m

            for v in range(m):

                # Need a larger value next:
                #
                #   sum(down[x] for x > v)
                #
                # Remove everything <= v from the total.
                new_up[v] = (total_down - pref_down[v]) % MOD

                # Need a smaller value next:
                #
                #   sum(up[x] for x < v)
                #
                # This is exactly the prefix ending at v - 1.
                new_down[v] = pref_up[v - 1] if v > 0 else 0

            up = new_up
            down = new_down

        # First value can be any value in [l, r]
        # and the initial direction can be either up or down.
        return (sum(up) + sum(down)) % MOD
    
###################################################
# 3700. Number of ZigZag Arrays II
# 24JUN26
###################################################
class Solution:
    def zigZagArrays(self, n: int, l: int, r: int) -> int:
        """
        ------------------------------------------------------------
        PART I RECAP (DP FORMULATION)
        ------------------------------------------------------------

        We define:

            dp(i, 0, v) = number of ways starting at index i
                          if the previous value is v and the
                          next value must be larger

            dp(i, 1, v) = number of ways starting at index i
                          if the previous value is v and the
                          next value must be smaller

        Recurrence:

            dp(i, 0, v) = sum(dp(i+1, 1, x) for x > v)
            dp(i, 1, v) = sum(dp(i+1, 0, x) for x < v)

        Base case:

            dp(n, dir, v) = 1


        ------------------------------------------------------------
        STEP 1: BUILD A STATE VECTOR
        ------------------------------------------------------------

        Instead of storing two arrays:

            up[v]
            down[v]

        combine them into one vector of size 2m.

        For example:

            state =
            [
                up[0],
                up[1],
                ...
                up[m-1],

                down[0],
                down[1],
                ...
                down[m-1]
            ]

        where:
            m = r - l + 1


        ------------------------------------------------------------
        STEP 2: WRITE TRANSITIONS AS A MATRIX
        ------------------------------------------------------------

        Suppose:

            m = 3

        States:

            u0 u1 u2 d0 d1 d2

        Then:

            u0 = d1 + d2
            u1 = d2
            u2 = 0

            d0 = 0
            d1 = u0
            d2 = u0 + u1

        This can be written as:

                  u0 u1 u2 d0 d1 d2

            u0 =   0  0  0  0  1  1
            u1 =   0  0  0  0  0  1
            u2 =   0  0  0  0  0  0

            d0 =   0  0  0  0  0  0
            d1 =   1  0  0  0  0  0
            d2 =   1  1  0  0  0  0

        So:

            next_state = T @ state

        where T is a fixed 2m × 2m matrix.


        ------------------------------------------------------------
        STEP 3: WHY MATRIX EXPONENTIATION WORKS
        ------------------------------------------------------------

        Your DP is repeatedly applying the same transition.

        Part I does:

            state_n
            state_{n-1} = T * state_n
            state_{n-2} = T * state_{n-1}
            ...

        Therefore:

            state_1 = T^(n-1) * state_n

        and matrix exponentiation computes:

            T^(n-1)

        in:

            O((2m)^3 log n)

        instead of:

            O(n m)

        which is necessary when n is enormous.


        ------------------------------------------------------------
        STEP 4: WHAT IS THE BASE VECTOR?
        ------------------------------------------------------------

        In Part I your base was:

            up = [1] * m
            down = [1] * m

        because:

            dp(n, dir, value) = 1

        So:

            base =
            [1] * (2*m)

        Then compute:

            result = T^(n-1) * base

        Finally:

            answer = sum(result)

        because your original recursion started with:

            for start in values:
                ans += dp(1,0,start)
                ans += dp(1,1,start)

        which is exactly summing every component of state_1.
        """


        MOD = 10**9 + 7

        m = r - l + 1
        size = 2 * m

        def up_idx(v):
            return v

        def down_idx(v):
            return m + v

        # ---------------------------------------------------------
        # Build transition matrix T
        #
        # state_i = T * state_(i+1)
        # ---------------------------------------------------------
        T = [[0] * size for _ in range(size)]

        # up[v] = sum(down[x] for x > v)
        for v in range(m):
            for x in range(v + 1, m):
                T[up_idx(v)][down_idx(x)] = 1

        # down[v] = sum(up[x] for x < v)
        for v in range(m):
            for x in range(v):
                T[down_idx(v)][up_idx(x)] = 1

        def mat_mul(A, B):
            """
            Matrix multiplication modulo MOD.

            A: rows x mid
            B: mid x cols
            """
            rows = len(A)
            mid = len(B)
            cols = len(B[0])

            C = [[0] * cols for _ in range(rows)]

            for i in range(rows):
                for k in range(mid):

                    # Skip useless work on zero entries.
                    if A[i][k] == 0:
                        continue

                    a = A[i][k]

                    for j in range(cols):
                        if B[k][j]:
                            C[i][j] = (C[i][j] + a * B[k][j]) % MOD

            return C

        def mat_pow(M, power):
            """
            Fast exponentiation.

            Computes M^power in O(log power).
            """
            result = [
                [1 if i == j else 0 for j in range(size)]
                for i in range(size)
            ]

            while power:
                if power & 1:
                    result = mat_mul(result, M)

                M = mat_mul(M, M)
                power >>= 1

            return result

        # ---------------------------------------------------------
        # Base vector = state_n
        #
        # dp(n, dir, value) = 1
        #
        # Therefore every state at layer n is 1.
        # ---------------------------------------------------------
        base = [[1] for _ in range(size)] #base needs to be a matrix

        # state_1 = T^(n-1) * state_n
        T_pow = mat_pow(T, n - 1)
        state_1 = mat_mul(T_pow, base)

        # Original recursion started by trying every value and
        # both possible initial directions:
        #
        #   sum(dp(1,0,v) + dp(1,1,v))
        #
        # which is simply the sum of every entry in state_1.
        ans = 0
        print(state_1)
        for i in range(size):
            ans = (ans + state_1[i][0]) % MOD

        return ans