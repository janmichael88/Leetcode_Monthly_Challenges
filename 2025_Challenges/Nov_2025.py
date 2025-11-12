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
import math
class Solution:
    def minOperations(self, nums):
        n = len(nums)
        c = nums.count(1)  # if there's at least one '1'
        if c != 0:
            return n - c
        
        res = float('inf')
        for i in range(n):
            g = nums[i]
            for j in range(i + 1, n):
                g = math.gcd(g, nums[j])
                if g == 1:
                    res = min(res, j - i + (n - 1))
                    
        
        return -1 if res == float('inf') else res
