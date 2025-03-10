#########################################
#2460. Apply Operations to an Array
# 02MAR25
#######################################
class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        '''
        just follow the rules and apply on nums
        then move zeros to the end
        '''
        n = len(nums)
        ans = []
        for i in range(n-1):
            if nums[i] == nums[i+1]:
                nums[i] *= 2
                nums[i+1] = 0
            else:
                continue
        
        for num in nums:
            if num:
                ans.append(num)
        
        return ans + [0]*(n-len(ans))

#inplace two pass
class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        '''
        in place two pass
        first modify the array
        '''
        n = len(nums)
        for i in range(n-1):
            if nums[i] == nums[i+1]:
                nums[i] *= 2
                nums[i+1] = 0
        
        place_idx = 0
        for num in nums:
            if num:
                nums[place_idx] = num
                place_idx += 1
        #the rest are zeros
        while place_idx < len(nums):
            nums[place_idx] = 0
            place_idx += 1
        
        return nums
    
##############################################
# 2570. Merge Two 2D Arrays by Summing Values
# 02MAR25
###############################################
class Solution:
    def mergeArrays(self, nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
        '''
        hashmap problem, how to return in sorted order after merging?
        the arrays are already in ascneding order,
        advance the smaller of the two, merge part of merge sort
        '''
        ans = []
        i,j = 0,0
        while i < len(nums1) and j < len(nums2):
            if nums1[i][0] < nums2[j][0]:
                ans.append(nums1[i])
                i += 1
            elif nums2[j][0] < nums1[i][0]:
                ans.append(nums2[j])
                j += 1
            #equal id
            else:
                entry = []
                entry.append(nums1[i][0])
                entry.append(nums1[i][1] + nums2[j][1])
                ans.append(entry)
                i += 1
                j += 1
        while i < len(nums1):
            ans.append(nums1[i])
            i += 1
        while j < len(nums2):
            ans.append(nums2[j])
            j += 1
        
        return ans
    
#############################################################
# 2161. Partition Array According to Given Pivot (REVISITED)
# 03MAR25
#############################################################
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        '''
        im sure there is an in place algo, but we can use quick sort partition scheme
        three arrays
            lesss,than,equal, everything else
        '''
        less_than = []
        equal_to = []
        remaining = []

        for num in nums:
            if num < pivot:
                less_than.append(num)
            elif num == pivot:
                equal_to.append(num)
            else:
                remaining.append(num)
        
        return less_than + equal_to + remaining
    
#two pass pointers
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        '''
        use pointers and keep track of where they should belong
        count up less and equal too
        greater elements start at less + equal
        '''
        less_than = 0
        equal_to = 0
        for num in nums:
            if num < pivot:
                less_than += 1
            elif num == pivot:
                equal_to += 1
        
        ans = [0]*len(nums)
        less_ptr = 0
        #equal starts at less than count
        equal_ptr = less_than
        greater_ptr = less_than + equal_to

        for num in nums:
            if num < pivot:
                ans[less_ptr] = num
                less_ptr += 1
            elif num > pivot:
                ans[greater_ptr] = num
                greater_ptr += 1
            else:
                ans[equal_ptr] = num
                equal_ptr += 1
        
        return ans
    
########################################
# 2747. Count Zero Request Servers
# 04MAR25
#########################################
#wtf???
class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        '''
        sort queries and logs in increasing order,
        then sliding window to find answers
        i need to maintain the original query order
        '''
        logs.sort(key = lambda x : x[1])
        queries = [(i,q) for i,q in enumerate(queries)]
        queries.sort(key = lambda x: x[1])
        ans = [0]*len(queries)
        left = 0
        right = 0
        servers = Counter()
        used = 0
        for i,q in queries:
            while right < len(logs) and logs[right][1] <= q:
                #add to window
                servers[logs[right][0]] += 1
                used += servers[logs[right][0]] == 1
                right += 1
            
            #shrink
            while left < right and logs[left][1] < q - x:
                servers[logs[left][0]] -= 1
                used -= servers[logs[left][0]] == 0
                left += 1
            
            ans[i] = n - used
        
        return ans

####################################################
# 1780. Check if Number is a Sum of Powers of Three
# 04MAR25
####################################################
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        '''
        how many powers of three are there?
        there are log_3(n) powers of three
        of these powers of three try and see if we can make a n
        must be distinc
        '''
        powers_of_three = []
        x = -1
        while 3**x <= n:
            x += 1
            powers_of_three.append(3**x)
        
        def rec(n,i):
            if i >= len(powers_of_three):
                return False
            if n < 0:
                return False
            if n == 0:
                return True
            for j in range(i,len(powers_of_three)):
                power = powers_of_three[j]
                if rec(n-power,j+1):
                    return True
            return False
        
        return rec(n,0)
    
#this TLE's though
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        '''
        how many powers of three are there?
        there are log_3(n) powers of three
        of these powers of three try and see if we can make a n
        must be distinct,
        im not satisfied that the test cases are strong enough

        also could have done backtracking
        '''
        powers_of_three = []
        x = -1
        while 3**x <= n:
            x += 1
            powers_of_three.append(3**x)
        
        used = set()
        
        def rec(n,used):
            if n < 0:
                return False
            if n == 0:
                return True
            for power in powers_of_three:
                if power not in used:
                    used.add(power)
                    if rec(n-power,used):
                        return True
                    used.remove(power)
            return False
        
        return rec(n,used)

class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        '''
        we can treat as knapsack and keep track of power and n - 3^(some_power)
        i don't think there are repated subproblems for this one though
        '''
        memo = {}
        def rec(power,n):
            if n == 0:
                return True
            if 3**power > n:
                return False
            if (power,n) in memo:
                return memo[(power,n)]
            take = rec(power+1,n-3**power)
            no_take = rec(power+1,n)
            ans = take or no_take
            memo[(power,n)] = ans
            return ans
        
        return rec(0,n)
    
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        '''
        assume we have all powers of three as a,b,c...wtv char as needed
        is n is a sum of all these then we can write
        n = a + b + c + d + e ....
        keep using the largest power
        '''
        largest_power = 0
        while n >= 3**largest_power:
            largest_power += 1
        
        while n > 0:
            #try using power
            if n >= 3**largest_power:
                n -= 3**largest_power
            #cannot use same power twise
            if n >= 3**largest_power:
                return False
            largest_power -= 1
        
        return True
    
class Solution:
    def checkPowersOfThree(self, n: int) -> bool:
        '''
        keep diving by 3 and check mod 3
        if the numbers a,b,c,d,e are all powers of three
        then a + b + c + d + e is divible by 3
        '''
        while n > 0:
            #check if power should be used only once
            print(n % 3)
            if n % 3 == 2:
                return False
            n = n // 3
        
        return True
    
###########################################
# 2579. Count Total Number of Colored Cells
# 05MAR24
###########################################
class Solution:
    def coloredCells(self, n: int) -> int:
        '''
        we keep adding 4 cells every time
        1,5,13
        each time we add, we need to add 4 more cells

        '''
        ans = 1
        curr_cells = 4
        for i in range(n-1):
            ans += curr_cells
            curr_cells += 4
        
        return ans
    
class Solution:
    def coloredCells(self, n: int) -> int:
        '''
        we keep adding 4 cells every time
        1,5,13
        each time we add, we need to add 4 more cells
        the pattern is 1 + (4*1) + (4*2) + ... + 4*(n-1)
        inside sum is sum series for n-1
        1 + 4*(n*(n-1)//2)
         1 + 2*(n*(n-1))
        '''
        return 1 + 2*(n*(n-1))
    
#########################################
# 2965. Find Missing and Repeated Values
# 06MAR25
########################################
class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        '''
        a appears twice and b is missing
        return [a,b]
        values are in teh range [1,n**2]
        and i can use hashmap as temp space
        or i can use the grid and mark

        '''
        repeated = -1
        n = len(grid)
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                abs_num = abs(num) - 1
                #find index
                ii,jj = divmod(abs_num,n)
                if grid[ii][jj] < 0:
                    repeated = abs(num)
                grid[ii][jj] *= -1
        
        #second traverse
        missing = -1
        for i in range(n):
            for j in range(n):
                num = i*n + j + 1
                if grid[i][j] > 0 and num != repeated:
                    missing = num
        
        return [repeated,missing]

#single pass using xor!
class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        '''
        a appears twice and b is missing
        return [a,b]
        values are in teh range [1,n**2]
        and i can use hashmap as temp space
        or i can use the grid and mark
        a^b^c^c
        we want
        xor1 = a^b^c^d
        xor2 = a^b^c
        missing = xor1^xor2

        '''
        repeated = -1
        xor1 = 0
        xor2 = 0
        n = len(grid)
        for i in range(n):
            for j in range(n):
                num = grid[i][j]
                abs_num = abs(num) - 1
                #find index
                ii,jj = divmod(abs_num,n)
                #if its already negative, this is the repeated one
                if grid[ii][jj] < 0:
                    repeated = abs(num)
                else:
                    #othweise add to mask of seen
                    xor2 = xor2^(abs_num + 1)
                #mark
                grid[ii][jj] *= -1
                #mask to get all of them
                xor1 = xor1^(i*n + j + 1)
        
        
        return [repeated,xor1^xor2] #undo the seen mask, and retreive the missing

#mathssss
class Solution:
    def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
        '''
        sum of numbers from 1 to n**2 is just n^2*(n^2 + 1) / 2
        sum of squares is just n^2(n^2 + 1)*(2n^2 + 1) // 6
        or just \sum_{i=1}^{n} i**2
        now if we compute the sum of the numbers in grid
        sum_numbers = perfect_sum  + x - y
        so the difference between the actual sum and perfect sum is
        sum_diff = x - y
        sqr_diff = x**2 - y**2
        x**2 - y**2 = (x+y)*(x-y)
        we eventually end up with two euqations
        x-y == sum_diff
        x+y = sqrdiff/sumdiff
        we can solve for x and y
        '''
        n = len(grid)
        actual_sum = 0
        actual_square_sum = 0
        for i in range(n):
            for j in range(n):
                actual_sum += grid[i][j]
                actual_square_sum += grid[i][j]**2
            
        expected_sum = ((n*n)*(n*n + 1))//2
        expected_square_sum = ((n*n)*(n*n + 1)*(2*n*n + 1)) // 6
        sum_diff = actual_sum - expected_sum
        sqr_diff = actual_square_sum - expected_square_sum
        repeat = (sqr_diff // sum_diff + sum_diff) // 2
        missing = (sqr_diff // sum_diff - sum_diff) // 2

        return [repeat, missing]
    
########################################
# 2523. Closest Prime Numbers in Range
# 08MAR24
########################################
class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        '''
        compute prime numbers between left and right using seive
        then check all pairs and minimize
        no need to check all pairs, primes would be increasing, just check every consecutive pair
        '''
        primes = self.seive(left,right)
        ans = [-1,-1]
        diff = float('inf')
        for i in range(1,len(primes)):
            if primes[i] - primes[i-1] < diff:
                diff = primes[i] - primes[i-1]
                ans = [primes[i-1],primes[i]]
        
        return ans
    
    def seive(self,left,right):
        n = right
        is_prime = [True]*(n+1)
        is_prime[1] = False
        p = 2
        while p*p <= n:
            if is_prime[p] == True:
                #every multiple is not
                for i in range(p*p,n+1,p):
                    is_prime[i] = False
            
            p += 1
        
        primes = []
        for i in range(left,right+1):
            if is_prime[i]:
                primes.append(i)
        
        return primes
    
class Solution:
    def closestPrimes(self, left: int, right: int) -> List[int]:
        '''
        we can also avoid using the seive
        if a number is prime, find its next prime and check
        need to take advantage of twin primes property, primes that differ by 2
        if we have the range [l,r] and r-l >= 1452, there is always at least one twin prime pair
        this is valid for the given range of the problem 1 <= l,r <= 10**6)
            this must be t he answer 
            since no pairs of primes can be smaller than a twin prime
        
        code to check max twin prime dist under range limits
        limit = 1000000
        primes = self.seive(1,limit)
        twin_primes = []
        for i in range(2,limit-1):
            if primes[i] and primes[i+2]:
                twin_primes.append(i)
        
        max_dist = 0
        temp = [-1,-1]
        for i in range(1,len(twin_primes)):
            dist = twin_primes[i] - twin_primes[i-1]
            if dist > max_dist:
                max_dist = dist
                temp = [twin_primes[i],twin_primes[i-1]]
        print(max_dist,temp)
    def seive(self,left,right):
        n = right
        is_prime = [True]*(n+1)
        is_prime[1] = False
        p = 2
        while p*p <= n:
            if is_prime[p] == True:
                #every multiple is not
                for i in range(p*p,n+1,p):
                    is_prime[i] = False
            
            p += 1
        return is_prime
        algo is just try all primes, in the range, and check for its immediate next prime
        '''
        prev_prime = -1
        closestA = -1
        closestB = -1
        min_difference = float("inf")

        # Iterate over the range of numbers and find primes
        for candidate in range(left, right + 1):
            if self.isPrime(candidate):
                if prev_prime != -1:
                    difference = candidate - prev_prime
                    if difference < min_difference:
                        min_difference = difference
                        closestA = prev_prime
                        closestB = candidate
                    #early termination
                    if difference == 1 or difference == 2:
                        return [prev_prime, candidate]
                prev_prime = candidate

        return [closestA, closestB] if closestA != -1 else [-1, -1]
    def isPrime(self, num):
        if num < 2:
            return False
        if num == 2 or num == 3:
            return True
        if num % 2 == 0:
            return False
        divisor = 3
        while divisor * divisor <= num:
            if num % divisor == 0:
                return False
            divisor += 2
        return True

############################################################
# 2379. Minimum Recolors to Get K Consecutive Black Blocks
# 08MAR25
#############################################################
class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        '''
        keep window counter of rocks
        when we have a valid size k, we need to conver the count W to B
        minimize this
        '''
        window = Counter()
        ans = float('inf')
        left = 0
        n = len(blocks)
        for right in range(n):
            color = blocks[right]
            if right - left + 1 > k:
                window[blocks[left]] -= 1
                left += 1
            
            window[color] += 1
            if right - left + 1 == k:
                ans = min(ans,window['W'])
        
        return ans

#q of size k
class Solution:
    def minimumRecolors(self, blocks: str, k: int) -> int:
        '''
        can also use deque of size k
        '''
        q = deque([])
        whites = 0
        for i in range(k):
            color = blocks[i]
            whites += color == 'W'
            q.append(color)
        
        ans = whites
        for i in range(k,len(blocks)):
            color = q.popleft()
            whites -= color == 'W'
            color = blocks[i]
            whites += color == 'W'
            q.append(color)
            ans = min(ans,whites)
        
        return ans
    
###############################
# 3208. Alternating Groups II
# 09MAR25
###############################
#finally!
class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], k: int) -> int:
        '''
        we are given a cirular array
        0 is red, 1 is blue
        keep deque of size k
        keep track of prev color and curr color

        '''
        n = len(colors)
        groups = 0
        q = deque([])
        for i in range(n+k-1):
            curr_color = colors[i%n]
            #we can add in the current color
            if q and q[-1] != curr_color:
                q.append(curr_color)
            else:
                q = deque([curr_color])
            
            if len(q) == k:
                groups += 1
                q.popleft()
        return groups

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], k: int) -> int:
        '''
        instead of using q, we can use two pointers
        we don't need to extend out the colors array, just keep moving up to n + k
        '''        
        n = len(colors)
        groups = 0
        left = 0
        right = 1
        while right < n + k - 1:
            #if same, move left to right and right + 1
            if colors[right % n] == colors[(right - 1) % n]:
                left = right
                right += 1
                continue
            right += 1
            #its the number of elements, not the bounds
            if right - left == k:
                groups += 1
                left += 1

        return groups 
