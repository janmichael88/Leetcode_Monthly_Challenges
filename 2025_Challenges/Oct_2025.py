###################################################
# 3168. Minimum Number of Chairs in a Waiting Room
# 01OCT25
###################################################
class Solution:
    def minimumChairs(self, s: str) -> int:
        '''
        minimize delta at any ont time
        '''
        ans = float('-inf')
        delta = 0
        for ch in s:
            if ch == 'E':
                delta -= 1
            else:
                delta += 1
            
            ans = max(ans,abs(delta))
        
        return ans
    
###############################################################
# 3629. Minimum Jumps to Reach End via Prime Teleportation
# 03OCT25
################################################################
#jesus fuck
class Solution:
    def minJumps(self, nums: List[int]) -> int:
        '''
        bfs
        precompute primes with seive
        only need up to max(nums)
        the problem is that the prime telportation steps from some index could be larger
        '''
        #need reverse mapp, num to index
        mapp = defaultdict(list)
        for i,num in enumerate(nums):
            mapp[num].append(i)
        
        #to store primes and jumps
        bucket = defaultdict(list)
        #compute primes
        n = len(nums)
        max_num = max(nums)
        is_prime = [True]*(max_num + 1)
        is_prime[0] = is_prime[1] = False
        p = 2
        while p <= max_num:
            if is_prime[p]:
                for i in range(p*p,max_num + 1, p):
                    if i in mapp:
                        print(p,i)
                        bucket[p].extend(mapp[i])
                    is_prime[i] = False
            p += 1
        print(bucket)
        #need to add in primes again
        for i in range(2,max_num+1):
            if is_prime[i]:
                bucket[i].extend(mapp[i])

        dists = [float('inf')]*(n)
        dists[0] = 0
        q = deque([0])
        seen = set()
        seen.add(0)

        while q:
            curr = q.popleft()
            seen.add(curr)
            #add in left and right
            for di in [1,-1]:
                ii = curr + di
                if 0 <= ii < n and ii not in seen:
                    dists[ii] = min(dists[ii], dists[curr] + 1)
                    q.append(ii)
            #add in prime telportation
            if is_prime[nums[curr]]:
                for neigh in bucket[nums[curr]]:
                    if neigh not in seen:
                        dists[neigh] = min(dists[neigh],dists[curr] + 1)
                        q.append(neigh)

        print(dists)
        return dists[n-1]

#kicker is we need smallest prime factor for each number up to  
class Solution:
    def minJumps(self, nums: List[int]) -> int:
        def compute_spf(n):
            spf = list(range(n+1))

            for i in range(2, int(n ** 0.5) + 1):
                #like seive, but if number itself, its prime (but in this case its only prime factor is itself)
                if(spf[i] == i):
                    for j in range(i**2, n+1, i):
                        #set as spf
                        if(spf[j] == j):
                            spf[j] = i
        
            return spf
        
        def getAllPrimeFactors(x, spf):
            #if we keep diving by the smallest prime factor, we get all multiples of x
            prime = set()
            while(x > 1):
                prime.add(spf[x])
                x //= spf[x]
            
            return prime
        
        smallest_prime_factor = compute_spf(max(nums))

        bucket = defaultdict(list)
        n = len(nums)

        if(n <= 1):
            return 0

        for i in range(n):
            for prime in getAllPrimeFactors(nums[i], smallest_prime_factor):
                bucket[prime].append(i)
        
        dists = [float('inf')]*(n)
        dists[0] = 0
        q = deque([0])
        seen = set()
        seen.add(0)

        while q:
            curr = q.popleft()
            seen.add(curr)
            #add in left and right
            for di in [1,-1]:
                ii = curr + di
                if 0 <= ii < n and ii not in seen:
                    dists[ii] = min(dists[ii], dists[curr] + 1)
                    q.append(ii)
            #add in prime telportation
            if nums[curr] in bucket:
                for neigh in bucket[nums[curr]]:
                    if neigh not in seen:
                        dists[neigh] = min(dists[neigh],dists[curr] + 1)
                        q.append(neigh)
                #only optimization is to clear the niegbors
                #pruning after adding
                del bucket[nums[curr]]
                

        return dists[n-1]


################################################
# 2517. Maximum Tastiness of Candy Basket
# 04OCT25
################################################
class Solution:
    def maximumTastiness(self, price: List[int], k: int) -> int:
        '''
        tastiness is the smallest absolute differ if prices of any two candies in the baskets
        binary search on answer
        '''
        price.sort()
        max_price = max(price)
        #need rightmost True
        '''
        for i in range(max_price + 1):
            ans = self.f(price,i,k)
            print(ans)
        '''
        left,right = 0, max(price)
        ans = left

        while left <= right:
            mid = left + (right - left) // 2
            if self.f(price,mid,k):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans 

    def f(self, candies,x,k):
        basket = 1
        n = len(candies)
        last = candies[0]
        for i in range(1,n):
            if candies[i] - last >= x:
                basket += 1
                last = candies[i]
        return basket >= k
    
#############################################
# 1488. Avoid Flood in The City
# 07OCT25
#############################################
#nice try :(
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        '''
        need to avoid floods in any lake
        if rains[i] > 0, then there will be rains over the rains[i], like
        i can dry an already dried lake, i can also assume, just to be safe there are at least n lakes
        '''
        n = len(rains)
        ans = [i for i in range(n)]
        flooded_lakes = set()
        for i,l in enumerate(rains):
            #rains
            if l > 0:
                #can't do
                if l in flooded_lakes:
                    return []
                ans[i] = -1
                flooded_lakes.add(l)
            elif l == 0:
                if len(flooded_lakes) == 0:
                    continue
                else:
                    ans[i] = flooded_lakes.pop()
        
        return ans
    
from sortedcontainers import SortedList
class Solution:
    def avoidFlood(self, rains: List[int]) -> List[int]:
        '''
        need to avoid floods in any lake
        if rains[i] > 0, then there will be rains over the rains[i], like
        i can dry an already dried lake, i can also assume, just to be safe there are at least n lakes
        stop immedialtey when we cant dry a lake
        something about binary search on a sorted array as we traverse rains
        if it rains
            1. if the lake is already full, we need to check if we could have dried it
            2. if the lake is not already full, we make it full
        
        we need to keep track of dry days, and use a dry day to drain a lake when it rains
        premrptively drain a lake on day i when rains[i] > 0
        i.e if it rains over a lake that is already full, it will flood unless we dried it on a previous day
        so stupid, instead of storing wet days, store dry days increasing
        '''
        n = len(rains)
        ans = [i+1 for i in range(n)] #assume we just dry the ith lake, could just be 1
        dry_days = SortedList([])
        lake_to_full = {}

        for day,lake in enumerate(rains):
            #if its dry, store dry days
            if lake == 0:
                dry_days.add(day)
            #if this lake is already full, try to dry it
            elif lake in lake_to_full:
                ans[day] = -1
                idx = dry_days.bisect(lake_to_full[lake])
                #no days to dry
                if idx == len(dry_days):
                    return []
                #on that day, dry current lake
                ans[dry_days[idx]] = lake
                dry_days.discard(dry_days[idx])
                lake_to_full[lake] = day
            else:
                ans[day] = -1
                lake_to_full[lake] = day
        
        return ans

#########################################################
# 3494. Find the Minimum Amount of Time to Brew Potions
# 09OCT25
########################################################
#gahhh
class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        '''
        need n wizards to make m potions
        time taken for the ith wizard to make the jth potion is skill[i]*mana[j]
        potions musy be brewed in order
        wizards work on potion when it arrives
        n*m is ok
        can i use dp?
        we need to use travers potions first
        '''
        n,m = len(skill),len(mana)
        earliest_time = [0]*(n+1)
        for j in range(m):
            now = earliest_time[0]
            for i in range(1,n):
                now = max(now + skill[i-1]*mana[j],earliest_time[i-1])
                earliest_time[i] = now
            earliest_time[n] = now + skill[n-1]*mana[j]
            #update in reverse order
            for i in range(n-1,-1,-1):
                earliest_time[i] = earliest_time[i+1] - skill[i]*mana[j]
            
            print(earliest_time)

#yikes, learned something though
class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        '''
        need n wizards to make m potions
        time taken for the ith wizard to make the jth potion is skill[i]*mana[j]
        potions musy be brewed in order
        wizards work on potion when it arrives
        n*m is ok
        can i use dp?
        we need to use travers potions first
        '''
        n, m = len(skill), len(mana)
        done = [0] * (n + 1)
        
        for j in range(m):
            for i in range(n):
                done[i + 1] = max(done[i + 1], done[i]) + mana[j] * skill[i]
            print(done)
            #need to syncrhonize backwards
            #on the first pass we take the max done time between wizards and add mana[j]*skill[i]
            #we then subtract this from the max time, since we are capped with the earliest wizard time completing the jth potion
            #delay syncrhonization
            for i in range(n - 1, 0, -1):
                done[i] = done[i + 1] - mana[j] * skill[i]
            print(done)
            print('-------------------')
        return done[n]
    
class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        '''
        using hints
        transition in the hint isn't quite right
        '''
        n, m = len(skill), len(mana)
        times = [0] * n
        for j in range(m):
            now = 0
            for i in range(n):
                now = max(now,times[i]) + skill[i]*mana[j]
            times[n - 1] = now
            for i in range(n - 2, -1, -1):
                times[i] = times[i + 1] - skill[i + 1] * mana[j]
        return times[n - 1]
    

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        '''
        using hints directyl
        '''
        n, m = len(skill), len(mana)
        done = [0]*n

        for j in range(m):
            x = mana[j]
            now = done[0]
            for i in range(1,n):
                now = max(now + skill[i-1]*x,done[i])
            done[n-1] = now + skill[n-1]*x

            for i in range(n-2,-1,-1):
                done[i] = done[i+1] - skill[i+1]*x
        
        return done[n-1]
    
#######################################################
# 3147. Taking Maximum Energy From the Mystic Dungeon
# 10OCT25
######################################################
#TLE
class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        '''
        dp[i] enery we gain starting at index i
        brute force will TLE
        '''
        n = len(energy)
        ans = float('-inf')
        for i in range(n):
            curr_energy = energy[i]
            while i + k < n:
                curr_energy += energy[i+k]
                i += k
            ans = max(ans,curr_energy)
        
        return ans

#finally
class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        '''
        dp[i] enery we gain starting at index i
        brute force will TLE, because k can be as big as the array
        prefix sum, but accumulate at intervals of k
        go backwards and just accumulate
        '''
        n = len(energy)
        dp = energy[:]
        for i in range(n-1,-1,-1):
            if i + k < n:
                dp[i] += dp[i+k]
        
        return max(dp)

###############################################
# 3186. Maximum Total Damage With Spell Casting
# 11OCT25
###############################################
#nice try T.T
class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        '''
        if i cast spell power[i]
        i cannot cast another spell with power[i] - 2, power[i] - 1, power[i] + 1, powe[i+2]
        but i cant cast multiple spells with same damage value
        casting a new spell with a different power means the i cant cast spell with a high power in range [+1,+2]
        if i want to cast power[i], i need to make sure i havent cast power[i] - 2, and power[i] - 1
        what if i sort and and keep counts, then iterate in order and when taking make sure i haven't taken power[i] -2 or power[i]- 1
        '''
        counts = Counter(power)
        used = set()
        ans = 0
        for k in sorted(counts):
            if k - 2 not in used and k-1 not in used:
                ans += k*counts[k]
                used.add(k)
        
        return ans
    
#gahhh, nice try again
class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        '''
        if i cast spell power[i]
        i cannot cast another spell with power[i] - 2, power[i] - 1, power[i] + 1, powe[i+2]
        but i cant cast multiple spells with same damage value
        casting a new spell with a different power means the i cant cast spell with a high power in range [+1,+2]
        if i want to cast power[i], i need to make sure i havent cast power[i] - 2, and power[i] - 1
        what if i sort and and keep counts, then iterate in order and when taking make sure i haven't taken power[i] -2 or power[i]- 1
        need to use dp, but go in order
        '''
        counts = Counter(power)
        used = set()
        ans1 = 0
        for k in reversed(sorted(counts)):
            if k + 2 not in used and k + 1 not in used:
                ans1 += k*counts[k]
                used.add(k)
        #check the other way
        used = set()
        ans2 = 0
        for k in sorted(counts):
            if k - 2 not in used and k - 1 not in used:
                ans2 += k*counts[k]
                used.add(k)

        return max(ans1,ans2)
    
class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        '''
        if i cast spell power[i]
        i cannot cast another spell with power[i] - 2, power[i] - 1, power[i] + 1, powe[i+2]
        but i cant cast multiple spells with same damage value
        casting a new spell with a different power means the i cant cast spell with a high power in range [+1,+2]
        if i want to cast power[i], i need to make sure i havent cast power[i] - 2, and power[i] - 1
        what if i sort and and keep counts, then iterate in order and when taking make sure i haven't taken power[i] -2 or power[i]- 1
        need to use dp, but go in order
        '''
        counts = Counter(power)
        arr = []
        for k in sorted(counts):
            arr.append((k,counts[k]))
        #i know i need to do dp on this array
        n = len(arr)
        memo = {}
        def dp(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            take = arr[i][0]*arr[i][1]
            no_take = dp(i+1)
            j = i + 1
            #if take, skip
            while j < n and arr[j][0] - 2 <= arr[i][0]:
                take += dp(j)
                j += 1
            ans = max(take,no_take)
            memo[i] = ans
            return ans
        
        return dp(0)
    
#finally!
class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        '''
        if i cast spell power[i]
        i cannot cast another spell with power[i] - 2, power[i] - 1, power[i] + 1, powe[i+2]
        but i cant cast multiple spells with same damage value
        casting a new spell with a different power means the i cant cast spell with a high power in range [+1,+2]
        if i want to cast power[i], i need to make sure i havent cast power[i] - 2, and power[i] - 1
        what if i sort and and keep counts, then iterate in order and when taking make sure i haven't taken power[i] -2 or power[i]- 1
        need to use dp, but go in order
        '''
        counts = Counter(power)
        arr = sorted(counts.items())  # [(power, count), ...]
        n = len(arr)
        memo = {}

        def dp(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            
            skip = dp(i + 1)
            #take it
            damage = arr[i][0] * arr[i][1]
            j = i + 1
            # find next valid allowed index
            while j < n and arr[j][0] <= arr[i][0] + 2:
                j += 1
            take = damage + dp(j)

            memo[i] = max(skip, take)
            return memo[i]

        return dp(0)
    
#######################################################
# 3539. Find Sum of Array Product of Magical Sequences
# 12OCT25
######################################################
#ughh
MOD = 10**9 + 7
from functools import lru_cache
import math
from typing import List
class Solution:
    def magicalSum(self, m: int, k: int, nums: List[int]) -> int:
        '''
        seq is magigal if it has
        size m, and its binary rep 2^(i) + ... 2^{m-1} has k set bits
        we define array product as:
        prod(seq) = (nums[seq[0]] * nums[seq[1]] * ... * nums[seq[m - 1]])
        dp, states are (i,j,bitmask)
        dp on subsets
        m could have repeated numbers in them
        for the binary rep portion, so we have indices [a,b,c,d]
        its rep will always be the same, no matter what the order
        2^a + 2 ^b + 2 ^c + 2^d
        flipping and b does nothing
        2^b + 2^a + 2^c + 2^d, so really we only care about subsets
        same thing with prod(seq)
        prod([a,b,c,d]) = nums[a]*nums[b]*nums[c]*nums[d] = prd([b,a,c,d])
        '''
        @lru_cache(None)
        def dfs(remaining, odd_needed, index, carry):
            if remaining < 0 or odd_needed < 0 or remaining + carry.bit_count() < odd_needed:
                return 0
            if remaining == 0:
                return 1 if odd_needed == carry.bit_count() else 0
            if index >= len(nums):
                return 0
            
            ans = 0
            for take in range(remaining + 1):
                ways = math.comb(remaining, take) * pow(nums[index], take, MOD) % MOD
                new_carry = carry + take
                ans += ways * dfs(remaining - take, odd_needed - (new_carry & 1), index + 1, new_carry >> 1)
                ans %= MOD
            return ans
        
        return dfs(m, k, 0, 0)


        
#brute force
class Solution:
    def magicalSum(self, m: int, k: int, nums: List[int]) -> int:
        '''
        just try doing brute force, pick and index i 
        '''
        n = len(nums)
        mod = 10**9 + 7
        ans = [0]

        def rec(i,mask_so_far,prod_so_far):
            #reached m picks
            if i == m:
                #check binary rep of picks
                if mask_so_far.bit_count() == k:
                    #add the products
                    ans[0] = (ans[0] + prod_so_far) % mod
                return

            for j in range(n):
                #use up a pick, add to make and prod
                rec(i + 1 ,mask_so_far + 2**j, (prod_so_far*nums[j]) % mod)
        
        rec(0,0,1)
        return ans[0] % mod
    
import math
mod = 10**9 + 7
class Solution:
    def magicalSum(self, m: int, k: int, nums: List[int]) -> int:
        '''
        just try doing brute force, pick and index i 
        '''
        n = len(nums)
        memo = {}

        def dp(m,k,i,flag):
            if m < 0 or k < 0 or m + flag.bit_count() < k:
                return 0
            if m == 0:
                if flag.bit_count() == k:
                    return 1
                return 0
            
            if i >= n:
                return 0
            
            key = (m,k,i,flag)
            if key in memo:
                return memo[key]
            ans = 0
            #try picking c copies of nums[i]
            for c in range(m+1):
                #mCc
                ways = math.comb(m,c)*pow(nums[i],c,mod) % mod
                new_flag = flag + c
                next_flag = new_flag >> 1 #divide by 2
                bit_contribution = new_flag & 1 #check odd
                ans += ways * dp(m - c, k - bit_contribution, i + 1, next_flag)
                ans %= mod
            
            memo[key] = ans
            return ans
        
        return dp(m,k,0,0)

#######################################################
# 3349. Adjacent Increasing Subarrays Detection I
# 14OCT25
########################################################
class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        '''
        conditions allow for checking all subarrays
        do this first
        but you can do it linearly
        if they are adjacent, then check the first k/2 and then the second k/2
        '''
        n = len(nums)
        for i in range(n-2*k+1):
            left,right = nums[i:i+k], nums[i+k:i+2*k]
            if self.check(left) and self.check(right):
                return True
        return False

    def check(self,arr):
        for i in range(1,len(arr)):
            if arr[i-1] >= arr[i]:
                return False
        return True

#check streaks
class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        '''
        we can record streak length at each index i
        '''
        streaks = [1]
        n = len(nums)
        for i in range(1,n):
            if nums[i-1] < nums[i]:
                streaks.append(streaks[-1] + 1)
            else:
                streaks.append(1)
        
        #check streaks and steaks[i+1]
        for i in range(len(streaks)-k):
            if streaks[i] >= k and streaks[i+k] >= k:
                return True
        return False

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        '''
        conditions allow for checking all subarrays
        do this first
        but you can do it linearly
        if they are adjacent, then check the first k/2 and then the second k/2
        '''
        pre, curr = 0, 1
        
        for i in range(1, len(nums)): 
            if nums[i] > nums[i - 1]: 
                curr += 1 
            
            else: 
                pre = curr 
                curr = 1 
            
            if curr >= k and pre >= k or curr >= 2*k: 
                return True
        
        return False 