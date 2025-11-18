#############################################
# 3318. Find X-Sum of All K-Long Subarrays I
# 04NOV25
#############################################
class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        '''
        follow the instructions
        '''

        def x_sum(nums,x):
            counts = Counter(nums)
            counts = sorted(counts.items(), key = lambda x : (x[1],x[0]), reverse = True)
            ans = 0
            for num,count in counts[:x]:
                ans += num*count
            
            return ans
        
        ans = []
        n = len(nums)
        for i in range(n - k + 1):
            arr = nums[i:i+k]
            ans.append(x_sum(arr,x))
        
        return ans

#######################################################
# 1578. Minimum Time to Make Rope Colorful (REVISTED)
# 04NOV25
######################################################
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        for each group of consectuvie letters we need to keep the one with the maximum time
        and delete the rest
        so time for this group is
        time = cost(group) - max(group)
        '''
        min_time = 0
        curr_group_cost = neededTime[0]
        curr_group_size = 1
        curr_group_max = neededTime[0]
        in_group = colors[0]
        n = len(colors)
        for i in range(1,n):
            #if part of group, iincrement countiers
            if colors[i] == in_group:
                curr_group_cost += neededTime[i]
                curr_group_size += 1
                curr_group_max = max(curr_group_max, neededTime[i])
            #ending consectuive here
            else:
                if curr_group_size > 1:
                    min_time += curr_group_cost - curr_group_max
                #reset
                curr_group_cost = neededTime[i]
                curr_group_size = 1
                curr_group_max = neededTime[i]
                in_group = colors[i]
        
        if curr_group_size > 1:
            min_time += curr_group_cost - curr_group_max
        
        return min_time
    

###############################################
# 3607. Power Grid Maintenance
# 06NOV25
###############################################
class Solution:
    def processQueries(self, c: int, connections: List[List[int]], queries: List[List[int]]) -> List[int]:
        '''
        all power stations are initally online
        we can turn them off/on
        identify power stations to connected components
        '''
        graph = defaultdict(list)
        for u,v in connections:
            graph[u].append(v)
            graph[v].append(u)

        components = {}
        seen = set()
        group_id = 0

        def dfs(curr,group):
            components[curr] = group
            seen.add(curr)
            for neigh in graph[curr]:
                if neigh not in seen:
                    dfs(neigh,group)
        for i in range(1,c+1):
            if i not in seen:
                dfs(i,group_id)
                group_id += 1
        
        group_sortedlist = defaultdict(SortedList)
        for k,v in components.items():
            group_sortedlist[v].add(k)
        
        ans = []
        for type_,station in queries:
            group = components[station]
            if type_ == 1:
                if len(group_sortedlist[group]) == 0:
                    ans.append(-1)
                elif station in group_sortedlist[group]:
                    ans.append(station)
                else:
                    ans.append(group_sortedlist[group][0])
            else:
                if station in group_sortedlist[group]:
                    group_sortedlist[group].remove(station)

        return ans

############################################
# 2528. Maximize the Minimum Powered City
# 07NOV25
############################################
#brute force, cool way
class Solution:
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        '''
        if we have stations
        [1,2,4,5,0]
        the power of each city i can be computed as 
        power[i] = sum(stations[max(0,i-r):min(n-1,i+r)])
        power = []
        n = len(stations)
        for i in range(n):
            left,right = max(0,i-r),min(n-1,i+r)
            power.append(sum(stations[left:right+1]))
        
        print(power)
        recall difference array 
            range(l,r) is 
            arr[l] += x
            arr[r+1] -= x
        we can compute a sum using prefix sums
        '''
        n = len(stations)
        ans = 0
        for p in combinations_with_replacement(range(n),k):
            temp = stations[:]
            for i in p:
                temp[i] += 1
            
            power = [0]*n
            for i in range(n):
                for j in range(max(0,i-r), min(n,i+r+1)):
                    power[i] += temp[j]
            ans = max(ans,min(power))
        
        return ans

#line sweep and binary search
class Solution:
    def maxPower(self, stations: List[int], r: int, k: int) -> int:
        '''
        if we have stations
        [1,2,4,5,0]
        the power of each city i can be computed as 
        power[i] = sum(stations[max(0,i-r):min(n-1,i+r)])
        power = []
        n = len(stations)
        for i in range(n):
            left,right = max(0,i-r),min(n-1,i+r)
            power.append(sum(stations[left:right+1]))
        
        print(power)
        recall difference array 
            range(l,r) is 
            arr[l] += x
            arr[r+1] -= x
        we can compute a sum using prefix sums
        binary search, can we do (x), if so, we try x +1, x + 2
        can turn into binary search
        rather can we make every city have at least power x by adding <= k new startions
        if we at power station i, then adding a power startion at i+r also covers it
        then the actual coverage goes to i to i + 2*r
        '''
        #this is line sleep, power of a city[i] is sum(counts[:i+1])
        n = len(stations)
        counts = [0]*(n+1)
        for i in range(n):
            left,right = max(0,i-r),min(n,i+r+1)
            counts[left] += stations[i]
            counts[right] -= stations[i]

        def check(val: int) -> bool:
            diff = counts.copy()
            total = 0
            remaining = k

            for i in range(n):
                total += diff[i]
                if total < val:
                    add = val - total
                    if remaining < add:
                        return False
                    remaining -= add
                    #range sum update
                    end = min(n, i + 2 * r + 1)
                    diff[end] -= add
                    #increase power
                    total += add
            return True

        lo, hi = min(stations), sum(stations) + k
        res = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                res = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return res
    

###########################################
# 2169. Count Operations to Obtain Zero
# 09NOV25
###########################################
class Solution:
    def countOperations(self, num1: int, num2: int) -> int:
        '''
        need to do either num1 - num2 or num2 - num1
        need operations to make num1 or num2 0
        '''
        ops = 0
        while num1 > 0 and num2 > 0:
            if num1 >= num2:
                num1 -= num2
            else:
                num2 -= num1
            ops += 1
        
        return ops
    
##########################################################
# 3542. Minimum Operations to Convert All Elements to Zero
# 10NOV24
##########################################################
#dammit
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        we can pick any subarray nums[i:j], and set all the those numbers to 0
        say we have array, [3,1,2,1]
        process in order, except 0
        starting with 1, we have range with indices [1,3]
            set to zero
        next with 2, we have range with indicies [2,2]
            set to zero
        next with 3, we have range with indices [0,0]
            set to zero
        we can only set the occurences to zero, not that whole subarray :(
             [1,2,1,2,1,2]
        for 1, it spans range [0,5]
        for 2 it spans [2,5], in fat 
        pick the whole thing, and make them all zero
        becomes [0,2,0,2,0,2]
        but now the original contig sequence has changed, [1,5] no longer has 2 as the minimum
        in fact it didn't before!
        mono stack
        say we have [3,1,2,1]
        [], load 3, [3]
        [3], 1 is smaller, need to use op on 3, op +1, and load [1]
        [1], load 2, [1,2],
        if we dont and an increasing array, need to use its length
        '''
        stack = []
        counts = Counter()
        ops = 0
        for num in nums:
            if not stack:
                stack.append(num)
                counts[num] += 1
            elif stack and stack[-1] <= num:
                stack.append(num)
                counts[num] += 1
            else:
                print(stack)
                stack = [num]
        print(stack)
        return ops

#aye yai yai
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        we only have to process each new increasing step
        say we have [1,2,3,4]
        [1], ops + 1
        [1,2] ops += 1
        when we see a number num < the top of thes tack, we can go back and deal with using a new operation
        when we see a number num larger than stack[-1], it means we must use a new opertaion to remove a because all smmaler numbers would have been cleared
        '''
        stack = []
        ops = 0
        for num in nums:
            while stack and stack[-1] > num:
                stack.pop()
            if num == 0:
                continue
            if not stack or stack[-1] < num:
                ops += 1
                stack.append(num)
        
        return ops
    
##########################################################################
# 2654. Minimum Number of Operations to Make All Array Elements Equal to 1
# 12NOV25
##########################################################################
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        when the gcd of two numbers is 1, the gcd of 1 and any other number is always 1 itself
        find the number of operations required to make any element as 1 and then the remaining n-1 elements can be made 1 in just n-1 steps
        if i'm at index i, with nums[i] = 1
        i can make nums[i-1] = gcd(nums[i], nums[i-1])
        then we peform the same at i-2,i-3...., which is just n-1 times
        now how can we get the minimum number of operations to make any of the elements 1?
            for any element i, traverse through all elements i+1 to n and find gcd
            as soon ad gcd becomes 1, we can inlcude that taking the gcd in the reverse direction (from j to i) would make nums[i] == 1
        if there are any ones already present in the array, we can just use them without trying to make an extta 1
        '''
        n = len(nums)
        ones = nums.count(1)  # if there's at least one '1'
        if c != 0:
            return n - ones
        
        res = float('inf')
        for i in range(n):
            g = nums[i]
            for j in range(i + 1, n):
                g = math.gcd(g, nums[j])
                if g == 1:
                    #everything in between
                    res = min(res, j - i + (n - 1))
                    
        
        return -1 if res == float('inf') else res

#######################################################
# 3228. Maximum Number of Operations to Move Ones to the End
# 14NOV25
#######################################################
class Solution:
    def maxOperations(self, s: str) -> int:
        '''
        need to pick valid indices i and i + 1 both < len(s)
        and its of the form '10'
        move i to the right until it reaches end of string or another 1
        [1,0,0,1,1,0,1]
        two pointers?
        if its 1, increment ones count
        if it 0 then we need ones op to move it to this zero
        '''
        res = ones = 0
        for v, l in groupby(s):
            print(v,[ch for ch in l])
            if v == '1':
                ones += len(list(l))
            else:
                res += ones
        return res
    
class Solution:
    def maxOperations(self, s: str) -> int:
        '''
        need to pick valid indices i and i + 1 both < len(s)
        and its of the form '10'
        move i to the right until it reaches end of string or another 1
        [1,0,0,1,1,0,1]
        two pointers?
        if its 1, increment ones count
        if it 0 then we need ones op to move it to this zero
        '''
        res = ones = 0
        n = len(s)
        for i,ch in enumerate(s):
            if ch == '1':
                ones += 1
            elif (i + 1 < n and s[i+1] == '1') or i+1 == n:
                res += ones
        
        return res
    
################################################
# 2536. Increment Submatrices by One
# 14NOV25
################################################
class Solution:
    def rangeAddQueries(self, n: int, queries: List[List[int]]) -> List[List[int]]:
        '''
        imagine the case we have 1d array and updates across range
        we can use prefix sums
        1d prefix sum
        In 1D, if you want to add +1 to all elements in a range [l, r], you can do:
        arr[l] += 1
        arr[r+1] -= 1
        In 2D pref sum its:
        pref_sum[r][col1] += 1 → mark the start of the horizontal range in row r.
        pref_sum[r][col2 + 1] -= 1 → mark the end of the horizontal range in row r.

        '''
        pref_sum = [[0]*(n+1) for _ in range(n+1)]

        #this is similar to line sweep
        #where we do +1 at l and -1 at r
        for row1,col1,row2,col2 in queries:
            for r in range(row1,row2+1):
                pref_sum[r][col1] += 1
                pref_sum[r][col2 + 1] -= 1
        
        for i in range(n+1):
            for j in range(1,n+1):
                pref_sum[i][j] += pref_sum[i][j-1]
        
        return [r[:n] for r in pref_sum[:-1]]
    

##########################################################
# 3234. Count the Number of Substrings With Dominant Ones
# 15NOV25
#########################################################
#effecient N**2, passes in C++ and Java but not in python
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        n = len(s)

        # prefix[i] = number of 1's in s[0..i]
        prefix = [0] * n
        prefix[0] = int(s[0])

        for i in range(1, n):
            prefix[i] = prefix[i-1] + int(s[i])

        ans = 0

        for i in range(n):
            o = 0
            z = 0

            j = i
            while j < n:
                o = prefix[j] - (0 if i == 0 else prefix[i-1])
                z = (j - i + 1) - o

                if (z * z) > o:
                    # jump ahead
                    j += (z*z - o - 1)

                elif (z * z) == o:
                    ans += 1

                else:  # (z*z < o)
                    ans += 1
                    skipNum = int(o**0.5) - z
                    jumpj = j + skipNum

                    if jumpj >= n:
                        ans += (n - j - 1)
                    else:
                        ans += skipNum

                    j = jumpj

                j += 1

        return ans

class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        '''
        store prefix sum of array to count ones in a range [i:j]
        conditions:
            if count_zeros**2 > ones, we need more ones, so we need to traverse the pref_sum array (intelligently)
            by moving pointer forward, or you can say we skip ucesscary indexs
        
            if count_zeros **2 == cont ones, valid substring
            if count_zeros**2 < ones, we have excess ones, so we can skip some indices by moving j pointer
            sqrt(ones - zeros)

            if j pointer exceedds length of string, simply add all
        '''
        n = len(s)
        #pref_sum to countones
        pref_sum = [0]*n
        pref_sum[0] = 0 + (s[0] == '1')
        for i in range(1,n):
            pref_sum[i] += pref_sum[i-1] + (s[i] == '1')
        
        ans = 0
        for i in range(n):
            ones = 0
            zeros = 0
            j = i
            while j < n:
                ones = pref_sum[j] - (0 if i == 0 else pref_sum[i-1])
                zeros = (j - i + 1) - ones
                #not enough ones
                if (zeros * zeros) > ones:
                    gap = zeros**2 - ones - 1
                    j += gap
                #valid dominant substring
                elif zeros**2 == ones:
                    ans += 1
                
                #more than enough ones
                elif zeros**2 < ones:
                    ans += 1
                    skipnum = int(ones**0.5) - zeros #gap
                    jump = j + skipnum
                    if jump >= n:
                        ans += (n - j - 1) #the any substring from j to end works
                    else:
                        ans += skipnum
                    j = jump
                j += 1
            
        return ans
    
############################################
# 717. 1-bit and 2-bit Characters (REVISTED)
# 18NOV25
############################################
#dp
class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        '''
        recursively
        '''
        n = len(bits)
        memo = {}

        def dp(i):
            if i >= n:
                return False
            if i == n-1:
                return True
            if i in memo:
                return memo[i]
            one_bit = False
            two_bit = False
            if bits[i] == 0:
                one_bit = dp(i+1)
            elif i + 1 < n:
                if bits[i] == 1 and bits[i+1] == 0:
                    two_bit = dp(i+2)
                elif bits[i] == 1 and bits[i+1] == 1:
                    two_bit = dp(i+2)
            ans = one_bit or two_bit
            memo[i] = ans
            return ans
        
        return dp(0)
                