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

########################################################################
# 3306. Count of Substrings Containing Every Vowel and K Consonants II
# 10MAR25
########################################################################
class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        '''
        need to keep track of vowel counts, use as seperate hashmap
        and cononat counts, we don't care what the cosonants are, just that they are consonants
        the problem is that if we shrink the window, we lose vowels
        shrinking solves the too many consonants problem
        '''
        vowels = 'aeiou'
        window_vowels = Counter()
        consonant_counts = 0
        ans = 0
        left = 0
        n = len(word)
        for right,ch in enumerate(word):
            if ch in vowels:
                window_vowels[ch] += 1
            else:
                consonant_counts += 1
            #valid
            if len(window_vowels) == len(vowels) and consonant_counts == k:
                ans += 1
            while left < right and consonant_counts > k and len(window_vowels) == 5:
                ch = word[left]
                if ch in vowels:
                    window_vowels[ch] -= 1
                    if window_vowels[ch] == 0:
                        del window_vowels[ch]
                else:
                    consonant_counts -= 1
                left += 1
        
        return ans

class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        '''
        need to keep track of vowel counts, use as seperate hashmap
        and cononat counts, we don't care what the cosonants are, just that they are consonants
        the problem is that if we shrink the window, we lose vowels
        shrinking solves the too many consonants problem
        there is a special case, so we have the requiste number of vowels and k consoants
        expanding to another vowel is still valid
        record the next vowel for each index going fromg right to left
        '''
        vowels = 'aeiou'
        window_vowels = Counter()
        consonant_counts = 0
        ans = 0
        left = 0
        n = len(word)
        next_consonant = [0]*n
        consonant_index = n
        for i in range(n-1,-1,-1):
            next_consonant[i] = consonant_index
            if word[i] not in vowels:
                consonant_index = i

        for right,ch in enumerate(word):
            if ch in vowels:
                window_vowels[ch] += 1
            else:
                consonant_counts += 1
            #shrink if we have too many consonants
            while left < right and consonant_counts > k:
                ch = word[left]
                if ch in vowels:
                    window_vowels[ch] -= 1
                    if window_vowels[ch] == 0:
                        del window_vowels[ch]
                else:
                    consonant_counts -= 1
                left += 1
            #while we have valid criteria,then every substring between the current right and the next consoant is also valid
            #this is a shrink, but still valid criteria for substring count
            while left < right and consonant_counts == k and len(window_vowels) == 5:
                ans += next_consonant[right] - right
                ch = word[left]
                if ch in vowels:
                    window_vowels[ch] -= 1
                    if window_vowels[ch] == 0:
                        del window_vowels[ch]
                else:
                    consonant_counts -= 1
                #we can find more valid substrings by shrinking our window until we no longer have a valid ssubstring
                #this is a new substring starting at index left, as long as we mainting 5 vowels, we can shrink
                #the new shrunken substring still has next_consonant[right] - right substrings
                left += 1
        
        return ans

class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        '''
        if we instead using sliding window to find 'at least k consonants'
        then exaclty k would be
            at_least(k) - at_least(k+1)
        paradigm, sliding window reduction
            inclusion/exlcusion satsisfiability
            say we need k of something, try looking for k + 1
            then ans is just some_function(k) - some_function(k+1)
        
        intution, instead of asking for exactly k, try at least k
        then find at least k+1,
        take difference
        '''
        return self.at_least_k(word,k) - self.at_least_k(word, k + 1)
    
    def at_least_k(self,word,k):
        vowels = 'aeiou'
        window_vowels = Counter()
        consonant_counts = 0
        ans = 0
        left = 0
        n = len(word)
        for right,ch in enumerate(word):
            if ch in vowels:
                window_vowels[ch] += 1
            else:
                consonant_counts += 1
            #need substrings with at least k cosonants
            #if we have a valid window ending at right, then any subtrings ending in len(word) - word would also have at least k
            #so we shrink and count, like in previous set up
            while len(window_vowels) == 5 and consonant_counts >= k:
                ans += len(word) - right
                ch = word[left]
                if ch in vowels:
                    window_vowels[ch] -= 1
                    if window_vowels[ch] == 0:
                        del window_vowels[ch]
                else:
                    consonant_counts -= 1
                left += 1
            
        return ans

#############################################################
# 1358. Number of Substrings Containing All Three Characters
# 11MAR25
#############################################################
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        '''
        assume we have a valid window with pointers [left,right]
        expanding right would just mean we have more valid substrings
        really we have len(s) - right substrings
        '''
        n = len(s)
        window = Counter()
        left = 0
        ans = 0
        for right in range(n):
            window[s[right]] += 1
            while left < right and len(window) == 3:
                ans += n - right
                window[s[left]] -= 1
                if window[s[left]] == 0:
                    del window[s[left]]
                left += 1
        
        return ans
    
##############################
# 749. Contain Virus
# 12MAR25
##############################
#dammit
#close one
class Solution:
    def containVirus(self, isInfected: List[List[int]]) -> int:
        '''
        each wall is one unit lone
        can be install between any two 4 direciontal adjacent cells
        first identify the regions
        hint looks ghastly LMAOOO
            not horribly hard, just a bitch 
        steps:
        1. find all viral regions (ans the possible next cells to infect)
        2. disinfect the largest reason (i.e find the area)
        3. then spread
        4. keep going until we can no longer spread

        need utility to find peremited given connected componenets
        for each (i,j) in the comp, check if there is an exposing side that is not inefected nor a boundary
        seal off the regions with the largest perimeters first

        hardest part is calculating the permiter
        '''
        rows, cols = len(isInfected),len(isInfected[0])
        seen = set()
        comps = self.find_comps(isInfected,seen)
        walls = 0
        while comps:
            largest_size,largest_comp = heapq.heappop(comps)
            #compute walls needed
            walls_used = self.get_perimeter(largest_comp,isInfected,rows,cols)
            walls += walls_used
            #infect the remaining
            next_day = []
            for _,comp in comps:
                next_comp = self.infect(comp,isInfected,rows,cols)
                perim = self.get_perimeter(next_comp,isInfected,rows,cols)
                entry = [-perim,next_comp]
                heapq.heappush(next_day,entry)
            comps = next_day
        
        return walls

    def infect(self,comp,grid,rows,cols):
        dirrs = [[1,0],[-1,0],[0,-1],[0,1]]
        next_comp = set()
        for i,j in comp:
            for di, dj in dirrs:
                ii = i + di
                jj = j + dj    
                if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] == 0 and (ii,jj) not in comp:
                    next_comp.add((ii,jj))
                    grid[ii][jj] = 1
        return next_comp | comp
    def get_perimeter(self,comp,grid,rows,cols):
        #sides = set()
        dirrs = [[1,0],[-1,0],[0,-1],[0,1]]
        perim = 0
        for i,j in comp:
            for di, dj in dirrs:
                ii = i + di
                jj = j + dj
                if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] == 0 and (ii,jj) not in comp:
                    perim += 1
        return perim

    
    def dfs(self,i,j,grid,seen,curr_comp):
        rows,cols = len(grid),len(grid[0])
        dirrs = dirrs = [[1,0],[-1,0],[0,-1],[0,1]]
        curr_comp.add((i,j))
        seen.add((i,j))
        for di, dj in dirrs:
            ii = i + di
            jj = j + dj
            if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] == 1 and (ii,jj) not in seen:
                self.dfs(ii,jj,grid,seen,curr_comp)
    
    def find_comps(self,grid,seen):
        rows,cols = len(grid),len(grid[0])
        comps = []
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1 and (i,j) not in seen:
                    curr_comp = set()
                    self.dfs(i,j,grid,seen,curr_comp)
                    perim = self.get_perimeter(curr_comp,grid,rows,cols)
                    entry = [-perim,curr_comp]
                    heapq.heappush(comps,entry)
        return comps
        

class Solution(object):
    def containVirus(self, grid):
        R, C = len(grid), len(grid[0])
        def neighbors(r, c):
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < R and 0 <= nc < C:
                    yield nr, nc

        def dfs(r, c):
            if (r, c) not in seen:
                seen.add((r, c))
                regions[-1].add((r, c))
                for nr, nc in neighbors(r, c):
                    if grid[nr][nc] == 1:
                        dfs(nr, nc)
                    elif grid[nr][nc] == 0:
                        frontiers[-1].add((nr, nc))
                        perimeters[-1] += 1

        ans = 0
        while True:
            #Find all regions, with associated frontiers and perimeters.
            seen = set()
            regions = []
            frontiers = []
            perimeters = []
            for r, row in enumerate(grid):
                for c, val in enumerate(row):
                    if val == 1 and (r, c) not in seen:
                        regions.append(set())
                        frontiers.append(set())
                        perimeters.append(0)
                        dfs(r, c)

            #If there are no regions left, break.
            if not regions: break

            #Add the perimeter of the region which will infect the most squares.
            triage_index = frontiers.index(max(frontiers, key = len))
            ans += perimeters[triage_index]

            #Triage the most infectious region, and spread the rest of the regions.
            for i, reg in enumerate(regions):
                if i == triage_index:
                    for r, c in reg:
                        grid[r][c] = -1
                else:
                    for r, c in reg:
                        for nr, nc in neighbors(r, c):
                            if grid[nr][nc] == 0:
                                grid[nr][nc] = 1

        return ans

################################################################
# 2529. Maximum Count of Positive Integer and Negative Integer
# 12MAR25
##################################################################
class Solution:
    def maximumCount(self, nums: List[int]) -> int:
        '''
        counts
        '''
        negatives = 0
        positives = 0
        for num in nums:
            negatives += num < 0
            positives += num > 0
        
        return max(negatives,positives)
    
#binary search
#jesus....
class Solution:
    def maximumCount(self, nums: List[int]) -> int:
        '''
        binary search to find the index of the largest negative
        is this index is k
        find smallest positive
        two binary searches
        '''
        largest_negative = self.largest(nums)
        smallest_positive = self.smallest(nums)
        #using indices find counts of negatives and positives
        #print(largest_negative,smallest_positive)
        count_negatives = largest_negative + 1 if largest_negative != -1 else 0
        count_positives = len(nums) - smallest_positive if smallest_positive != -1 else 0
        return max(count_negatives,count_positives)
    
    def largest(self,nums):
        largest_negative = -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= 0:
                right = mid - 1
            else:
                largest_negative = mid
                left = mid + 1
        return largest_negative
    
    def smallest(self,nums):
        smallest_positive = -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= 0:
                left = mid + 1
            else:
                right = mid - 1
                smallest_positive = mid
        return smallest_positive

##############################################
# 3356. Zero Array Transformation II
# 13MAR25
##############################################
#TLE with linear seach
class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        '''
        need to make nusm array the 0 array, 
        we need to find the minimum value of k (the value to decrement by) on each index (given in the queries array)
        that will make it the 0 array
        i'm thinking binary search for the correct k
        so we have queries [[0,2,1],[0,2,1],[1,1,3]]
        we can make possible decreemnt counts
        [0,0,0]
        [1,1,1]
        [2,2,2]
        [2,5,2]
        we can us 2 on nums to reduce to zero, and its minimum
        opps, we don't use all the queries, only the first k queries
        binary search on k
        '''
        N = len(queries)
        for i in range(N+1):
            ints = self.get_intervals(queries,nums,i)
            if self.can_do(nums,ints):
                return i
        
        return -1
    
    def can_do(self,nums,ints):
        for i in range(len(nums)):
            if nums[i] > ints[i]:
                return False
        return True
    
    def get_intervals(self,queries,nums,k):
        possible_decrements = [0]*(len(nums)+1)
        for l,r,v in queries[:k]:
            possible_decrements[l] += v
            possible_decrements[r+1] -= v
        #roll up
        for i in range(1,len(possible_decrements)):
            possible_decrements[i] += possible_decrements[i-1]
        
        return possible_decrements[:-1]
        
#now binary search
#jesus, stupid fucking problem....
class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        '''
        need to make nusm array the 0 array, 
        we need to find the minimum value of k (the value to decrement by) on each index (given in the queries array)
        that will make it the 0 array
        i'm thinking binary search for the correct k
        so we have queries [[0,2,1],[0,2,1],[1,1,3]]
        we can make possible decreemnt counts
        [0,0,0]
        [1,1,1]
        [2,2,2]
        [2,5,2]
        we can us 2 on nums to reduce to zero, and its minimum
        opps, we don't use all the queries, only the first k queries
        binary search on k
        '''
        left = 0
        right = len(queries)
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            ints = self.get_intervals(queries,nums,mid)
            if self.can_do(nums,ints):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
    
    def can_do(self,nums,ints):
        for i in range(len(nums)):
            if nums[i] > ints[i]:
                return False
        return True
    
    def get_intervals(self,queries,nums,k):
        possible_decrements = [0]*(len(nums)+1)
        for l,r,v in queries[:k]:
            possible_decrements[l] += v
            possible_decrements[r+1] -= v
        #roll up
        for i in range(1,len(possible_decrements)):
            possible_decrements[i] += possible_decrements[i-1]
        
        return possible_decrements[:-1]
        
class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        '''
        we can use sleep one, and check each value in nums against the queries
        instead of processing all the k queries up front, 
        maintain an active set of queries and update nums only when necessary
        the idea is that if we are at index i in nums it can be:
            * i < left, save query since we haven't applied it yet
            * left <= i <= right, apply it on nums
            * right < i, ignore this query
        '''
        n = len(nums)
        #we are keeping the possible decrements at this index
        #that are available to use
        available_decrements = 0
        k = 0 #used queries
        possible_decrements = [0]*(n+1)

        for i in range(n):
            #keep using queries to bring down nums[i]
            while available_decrements + possible_decrements[i] < nums[i]:
                k += 1
                if k > len(queries):
                    return -1
                l,r,val = queries[k-1]
                if r >= i:
                    possible_decrements[max(l,i)] += val
                    possible_decrements[r + 1] -= val
            
            #available decrements to be used up the the current index i
            available_decrements += possible_decrements[i]

        return k
    
################################################
# 2226. Maximum Candies Allocated to K Children
# 14MAR25
###############################################
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        '''
        another binary search on answer one
        say we have piles [5,8,6]
        if we make it so that way each pile has 1 candy, we can do
        try to see if we can make k piles with num_per_pile candies each
        then binary serach on answer
        '''
        sum_ = sum(candies)
        left = 1
        right = max(candies)
        ans = 0
        while left <= right:
            mid = left + (right - left) // 2
            if self.can_make(candies,mid,k):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
            
        return ans
    
    def can_make(self,candies,num_per_pile,k):
        piles = 0
        for c in candies:
            piles += c // num_per_pile
        
        return piles >= k

########################################
# 2560. House Robber IV
# 16MAR25
########################################
class Solution:
    def minCapability(self, nums: List[int], k: int) -> int:
        '''
        cannot steal from adjacent houses
        capability is the maximum amount of money steals
        needs to rob at least k houses
        need minimmum capability

        capbility will change depending on the number of houses we rob, and what houses we rob
        minimuze/maximize at least k is binary search paradigm
        '''
        left = min(nums)
        right = max(nums)
        ans = left
        while left <= right:
            mid = left + (right - left) // 2
            if self.check(nums,mid,k):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
    
    #if i can rob at least k houses with this value, i can certainly do it with a greater guess value
    def check(self,nums,guess,k):
        houses = 0
        robbed = False
        for n in nums:
            if n <= guess and not robbed:
                houses += 1
                robbed = True
            elif robbed:
                robbed = False
        
        return houses >= k
    
#############################################
# 2594. Minimum Time to Repair Cars (REVISTED)
# 16MAR24
############################################
class Solution:
    def repairCars(self, ranks: List[int], cars: int) -> int:
        '''
        a mechanic can repiars n cairs in r*n^2 mins
        for each mechanic, calcualte how many cars they can repair in n minutes
        all mechanics can work simultaneously

        for exmaple
        rank = 2
        2 cars can be done in 2*2*2 mins
        time = r*n**2
        for this amounf of time find cars fixed by mech
        time / r = n**2
        (time /r)**.5 = n
        can't do partial cars
        n = int(time / r)**.5
        if we can do call cars in this number of minutes, we can certainly fix it in any minutes >= the current min
        binary search 
        '''
        left = 1
        right = sum(ranks)*cars*cars
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            if self.func(ranks,cars,mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans
        #for m in range(1,20):
        #    print(m,self.func(ranks,cars,m))
    
    #func to check all cars repaired given minutes
    def func(self, ranks, cars,mins):
        cars_repaired = 0
        for r in ranks:
            cars_repaired += int((mins / r)**.5)
        
        return cars_repaired >= cars
            
############################################
# 2206. Divide Array Into Equal Pairs
# 17MAR25
#############################################
#two pass
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        '''
        count and check all freqs are even
        '''
        mapp = Counter(nums)
        for num,count in mapp.items():
            if count % 2:
                return False
        return True

#boolean array, just tracking parity of numbers
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        '''
        parity array
        '''
        unpaired = [0]*(max(nums) + 1)
        for num in nums:
            unpaired[num] = 1 - unpaired[num]
        
        return sum(unpaired) == 0

#one pass???
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        '''
        sinlge pass hashset
        if its not in there, add it in, otherwise remove
        set shouold be empty
        '''
        unpaired = set()
        for num in nums:
            if num not in unpaired:
                unpaired.add(num)
            else:
                unpaired.remove(num)
        
        return not unpaired
    
#########################################
# 1172. Dinner Plate Stacks
# 17MAR25
##########################################
#dammit
from sortedcontainers import SortedList
class DinnerPlates:
    '''
    keep track of the left most available stack for push
    keep track of right most nonempty stack for pop
    keep way to pop at index of stack
    problem is that was have infinite stacks 
    we can try doing popAtStack(index) where index is beyond the last
    index is bound to a maximum of 10**5
    can i do sorted list with two entries
    first if size of stack and next entry is size of stack
    '''

    def __init__(self, capacity: int):
        self.max_size = 10**5
        self.capacity = capacity
        self.stacks = SortedList([])
        for _ in range(self.max_size + 1):
            self.stacks.append([0,[]])
    def push(self, val: int) -> None:
        return

    def pop(self) -> int:
        return

    def popAtStack(self, index: int) -> int:
        if len(self.stacks[index][0sl ]) == 0:
            return -1
        else:
            return self.stacks[index].pop()


# Your DinnerPlates object will be instantiated and called as such:
# obj = DinnerPlates(capacity)
# obj.push(val)
# param_2 = obj.pop()
# param_3 = obj.popAtStack(index)


############################################
# 2401. Longest Nice Subarray (REVISTED)
# 18MAR25
############################################
#sliding window on mask, don't forget bitwise!
class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        '''
        say we have subarray [a,b,c] and it is nice
        this means
        (a & b) + (b & c) + (a % c) = 0
        if we have indices (i,j) we treat indices (j,i) as equivalent
        keep current mask of bit positions
        then we just check if mask & current number to add is zero or not

        '''
        ans = 1
        left = 0
        set_bits = nums[left]
        for right in range(1,len(nums)):
            #if addinf nums[right] doesn;t make it nice, we need to  shrink
            while left < right and set_bits & nums[right] != 0:
                #remove bits
                set_bits = set_bits ^ nums[left]
                left += 1
            
            #otherwise we are free to include and extend
            set_bits = set_bits | nums[right]
            ans = max(ans, right - left  + 1)
        
        return ans
    
#brute force works
#nice property contrains the size of the subarray
class Solution:
    def longestNiceSubarray(self, nums: List[int]) -> int:
        '''
        we can brute force by checking in the each subarray
        the nice constraint of the subarray is limited by the largest numbers
        so its n*lgN
        '''
        ans = 1
        n = len(nums)
        for i in range(n):
            size = 1
            mask = nums[i]
            for j in range(i+1,n):
                if mask & nums[j]:
                    break
                #include
                mask = mask | nums[j]
                size += 1
                ans = max(ans,size)
        
        return ans
    
########################################################################
# 3191. Minimum Operations to Make Binary Array Elements Equal to One I
# 19MAR25
########################################################################
#flip all length three subarrays
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        we can only flip 3 consecutive elements
        we can do this any number of times, possibly  zero
        return min number of operations to get the ones array
        we have to flip if we're at a zero
        try flipping when ever we have a zero and count flips
        if all ones return flips
        '''
        flips = 0
        n = len(nums)
        for i in range(n-3+1):
            if nums[i] == 0:
                flips += 1
                for j in range(i,i+3):
                    nums[j] = 1 - nums[j]
        
        if sum(nums) == n:
            return flips
        
        return -1
    
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        no inner loops        
        '''
        flips = 0
        n = len(nums)
        for i in range(n-3+1):
            if nums[i] == 0:
                flips += 1
                nums[i] = 1 - nums[i]
                nums[i+1] = 1 - nums[i+1]
                nums[i+2] = 1 - nums[i+2]
        
        if sum(nums) == n:
            return flips
        
        return -1
            
#single pass q, this was cool!
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        sliding window one pass
        if an index has been flipped an odd number of times, its value is opposite
        if an index has been flipped an even number of times, it remains the same
        if the last two positions are zero, we can't flip them
        if we flip an index i, it affect the next i+1 and i+2 indices
        instead of modifying the array we keep track of indices and check how it affects later indices
        '''
        flipped_indices = deque([])
        flips = 0
        n = len(nums)
        for i in range(n):
            #clear stale indices
            while flipped_indices and i - flipped_indices[0] > 2:
                flipped_indices.popleft()
            
            #if itz zero
            if (nums[i] + len(flipped_indices)) % 2 == 0:
                #check we can flip the whole triple
                if i + 2 >= n:
                    return -1
                flips += 1
                flipped_indices.append(i)
        
        return flips
    
########################################
# 3108. Minimum Cost Walk in Weighted Graph
# 20MAR25
########################################
from functools import reduce
class DSU:
    def __init__(self,n):
        self.n = n
        self.parent = [i for i in range(n)]
        self.size = [1]*n
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)

        if x_par == y_par:
            return
        elif self.size[x_par] >= self.size[y_par]:
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
            self.parent[y_par] = x_par
        else:
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
            self.parent[x_par] = self.parent[y_par]


class Solution:
    def minimumCost(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        '''
        properties of and
        X & X = X, going back along a previous edge does nothing to undo it
        but taking it again might clear another future edge
        union find on the nodes
        if u and v aren't connected return -1, otherwise can use all edges from the connected components
        
        '''
        dsu = DSU(n)
        #i dont actually need the graph, just a nodes edge adjacent weights
        node_edges = defaultdict(list)
        for u,v,w in edges:
            node_edges[u].append(w)
            node_edges[v].append(w)
            dsu.union(u,v)
        
        groups = defaultdict(list)
        for i in range(n):
            groups[dsu.find(i)].append(i)
        
        group_ands = {}
        for g,nodes in groups.items():
            edges = []
            for n in nodes:
                for edge in node_edges[n]:
                    edges.append(edge)
            if len(edges) > 1:
                edges = reduce(lambda x , y : x & y, edges)
            else:
                edges = - 1
            group_ands[g] = edges
        
        ans = []
        for u,v in query:
            if dsu.find(u) == dsu.find(v):
                ans.append(group_ands[dsu.find(u)])
            else:
                ans.append(-1)
        return ans
    
#optomize union find, group them fiurst, then AND them with -1
from functools import reduce
class DSU:
    def __init__(self,n):
        self.n = n
        self.parent = [i for i in range(n)]
        self.size = [1]*n
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)

        if x_par == y_par:
            return
        elif self.size[x_par] >= self.size[y_par]:
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
            self.parent[y_par] = x_par
        else:
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
            self.parent[x_par] = self.parent[y_par]


class Solution:
    def minimumCost(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        '''
        properties of and
        X & X = X, going back along a previous edge does nothing to undo it
        but taking it again might clear another future edge
        union find on the nodes
        if u and v aren't connected return -1, otherwise can use all edges from the connected components
        identity = -1 & a = 8
        '''
        dsu = DSU(n)
        for u,v,w in edges:
            dsu.union(u,v)
        
        comp_cost = [-1]*n
        for u,v,w in edges:
            root = dsu.find(u)
            #and each edge weight o its group
            comp_cost[root] &= w
        
        ans = []
        for u,v in query:
            if dsu.find(u) == dsu.find(v):
                ans.append(comp_cost[dsu.find(u)])
            else:
                ans.append(-1)
        return ans
            

#can replace dsu wihth either dfs/bfs
class Solution:
    def minimumCost(self, n, edges, queries):
        # Create the adjacency list of the graph
        adj_list = [[] for _ in range(n)]
        for edge in edges:
            adj_list[edge[0]].append((edge[1], edge[2]))
            adj_list[edge[1]].append((edge[0], edge[2]))

        visited = [False] * n

        # Array to store the component ID of each node
        components = [0] * n
        component_cost = []

        component_id = 0

        # Perform DFS for each unvisited node to identify components and calculate their costs
        for node in range(n):
            if not visited[node]:
                # Get the component cost and mark all nodes in the component
                component_cost.append(
                    self._get_component_cost(
                        node, adj_list, visited, components, component_id
                    )
                )
                component_id += 1

        result = []
        for query in queries:
            start, end = query

            if components[start] == components[end]:
                # If they are in the same component, return the precomputed cost for the component
                result.append(component_cost[components[start]])
            else:
                # If they are in different components, return -1
                result.append(-1)

        return result

    # Helper function to calculate the cost of a component using BFS
    def _get_component_cost(
        self, node, adj_list, visited, components, component_id
    ):

        # Initialize the cost to the number that has only 1s in its binary representation
        current_cost = -1

        # Mark the node as part of the current component
        components[node] = component_id
        visited[node] = True

        # Explore all neighbors of the current node
        for neighbor, weight in adj_list[node]:
            # Update the component cost by performing a bitwise AND of the edge weights
            current_cost &= weight
            if not visited[neighbor]:
                # Recursively calculate the cost of the rest of the component
                # and accumulate it into currentCost
                current_cost &= self._get_component_cost(
                    neighbor, adj_list, visited, components, component_id
                )

        return current_cost
            
#################################################
# 1839. Longest Substring Of All Vowels in Order
# 20MAR25
#################################################
class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:
        '''
        store indices of a,e,i,o,u
        '''
        window = {'a': deque([]),
                  'e': deque([]),
                  'i': deque([]),
                  'o': deque([]),
                  'u': deque([]),}
        ans = 0
        left = 0
        for right,ch in enumerate(word):
            #add to window
            window[ch].append(right)
            #while in violation shrink
            while left < right and self.in_violation(window):
                ch = word[left]
                window[ch].popleft()
                left += 1
            if self.is_valid(window):
                ans = max(ans, right - left + 1)
            
        return ans
        
    def in_violation(self,window):
        if (any([len(q) > 0 for _,q in window.items() ])):
            if window['a'] and window['e'] and window['a'][-1] > window['e'][-1]:
                return True
            if window['e'] and window['i'] and window['e'][-1] > window['i'][-1]:
                return True
            if window['i'] and window['o'] and window['i'][-1] > window['o'][-1]:
                return True
            if window['o'] and window['u'] and window['o'][-1] > window['u'][-1]:
                return True
        return False
    
    def is_valid(self,window):
        return all([len(q) > 0 for _,q in window.items() ])
    
#just need to check if in order, since a,e,i,o,u are already in order
class Solution:
    def longestBeautifulSubstring(self, word: str) -> int:
        '''
        just check in order
        '''
        uniques = set()
        left = 0
        ans = 0
        for right,ch in enumerate(word):
            #not increasing
            if right > 0 and ch < word[right-1]:
                uniques = set()
                left = right
            uniques.add(ch)
            if len(uniques)== 5:
                ans = max(ans,right - left + 1)
        
        return ans

    
#########################################################
# 2115. Find All Possible Recipes from Given Supplies
# 21MAR25
########################################################
#hashamp alone won't work, idk why though
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        '''
        this is just a hashmap problem
        for each recipe, check if we have it ingredients as a supplies
        oh we have an infinite supply of supplies but only fixex number of recipes
        evidently you cant just hasmap all of them :(
        can treat as graph problem
        '''
        ans = []
        supplies = set(supplies)
        made_rec = []
        for rec,ing in zip(recipes,ingredients):
            if all([i in supplies for i in ing]):
                supplies.add(rec)
                made_rec.append(rec)
        

        for rec,ing in zip(recipes,ingredients):
            if all([i in supplies for i in ing]):
                ans.append(rec)
        
        made_rec = set(made_rec)
        ans = set(ans)
        for rec in recipes:
            if rec in made_rec and rec not in ans:
                ans.append(rec)
        return list(ans)
    
#just keep making
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        '''
        basically check if we can keep making more recipes
        '''
        supplies = set(supplies)
        mapp = defaultdict(set)
        for rec,ing in zip(recipes,ingredients):
            for i in ing:
                mapp[rec].add(i)
        
        can_make = []
        already_made = set()
        while True:
            made_new = False
            for rec,ings in mapp.items():
                if rec in already_made:
                    continue
                if all([i in supplies for i in ings]):
                    can_make.append(rec)
                    supplies.add(rec)
                    already_made.add(rec)
                    made_new = True
            
            if not made_new:
                break
        
        return can_make
    
#can use queue, kinda like BFS for attemping to make new recipes
class Solution:
    def findAllRecipes(
        self,
        recipes: list[str],
        ingredients: list[list[str]],
        supplies: list[str],
    ) -> list[str]:
        # Track available ingredients and recipes
        available = set(supplies)

        # Queue to process recipe indices
        recipe_queue = deque(range(len(recipes)))
        created_recipes = []
        last_size = -1  # Tracks last known available count

        # Continue while we keep finding new recipes
        while len(available) > last_size:
            last_size = len(available)
            queue_size = len(recipe_queue)

            # Process all recipes in current queue
            while queue_size > 0:
                queue_size -= 1
                recipe_idx = recipe_queue.popleft()
                if all(
                    ingredient in available
                    for ingredient in ingredients[recipe_idx]
                ):
                    # Recipe can be created - add to available items
                    available.add(recipes[recipe_idx])
                    created_recipes.append(recipes[recipe_idx])
                else:
                    recipe_queue.append(recipe_idx)

        return created_recipes
    
#dfs
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        '''
        using dfs like cycle detection
        '''
        can_make = dict.fromkeys(supplies, True)
        mapp = defaultdict(set)
        for rec,ing in zip(recipes,ingredients):
            for i in ing:
                mapp[rec].add(i)

        def _check_recipe(recipe: str, visited: set) -> bool:
            if can_make.get(recipe, False):
                return True

            # Not a valid recipe or cycle detected
            if recipe not in mapp or recipe in visited:
                return False

            visited.add(recipe)

            possible = []
            for ingredient in mapp[recipe]:
                #for each of the dependencies, there shouldn't be a cycle
                possible.append(_check_recipe(ingredient, visited))
            can_make[recipe] = all(possible)

            return can_make[recipe]
        
        ans = []
        for recipe in recipes:
            if _check_recipe(recipe, set()):
                ans.append(recipe)
        
        return ans
            
#kahns top sort
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        '''
        to do top sort, we can't just do the usual thing and mapp recipes to ingredients
        we only draw an edge if we DO NOT have an indredient
            i.e when ingredient is not in supplies
        this ensures that the in-degree of a recipe reflects onl the number of unavialble
        when we do the sort, this ingredient could become available
        intuition
        1. if a recipe has indegree 0, all of it depdnecies (i.e ingredients) have been fulfilled, so we can make it
        2. when we complete a new recipe, we satisfuy its dependencies on others
        3. when indegree is zero, we are free to make it

        '''
        available_supplies = set(supplies)
        mapp = defaultdict(set)
        indegree = Counter()
        for rec,ings in zip(recipes,ingredients):
            for i in ings:
                mapp[rec].add(i)
                indegree[rec] = 0
        #make graph of what we dont have
        graph = defaultdict(list)
        for rec,ings in mapp.items():
            for i in ings:
                if i not in available_supplies:
                    #meaning we have dependencies on this guy
                    graph[i].append(rec)
                    indegree[rec] += 1
        
        q = deque([])
        ans = []
        #starting with rec that don't have any dependencies
        for rec,ind in indegree.items():
            if ind == 0:
                q.append(rec)
        
        while q:
            curr_rec = q.popleft()
            ans.append(curr_rec)
            for neigh in graph[curr_rec]:
                indegree[neigh] -= 1
                if indegree[neigh] == 0:
                    q.append(neigh)
        
        return ans

###############################################
# 2685. Count the Number of Complete Components
# 22MAR25
###############################################
class Solution:
    def countCompleteComponents(self, n: int, edges: List[List[int]]) -> int:
        '''
        one we get the components, count them up
        a connected comp is complete if and only iff the number of edges == m*(m-1)/2
        where me is the number of nodes in the componenets,
        kinda different, we normally just think about nodes in a graph/comp, we don't usually think about the edges!
        meh works, but i dont like it
        '''
        graph = defaultdict(list)
        hash_edges = set()
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            hash_edges.add((u,v))
            hash_edges.add((v,u))
        
        comps = []
        seen = set()
        for i in range(n):
            if i not in seen:
                #dfs
                curr_comp = []
                self.dfs(i,graph,seen,curr_comp)
                comps.append(curr_comp)
        #validate each comp
        count = 0
        for comp in comps:
            nodes = len(comp)
            edges_have = 0
            for i in range(nodes):
                for j in range(i+1,nodes):
                    u,v = (comp[i],comp[j])
                    if (u,v) in hash_edges or (v,u) in hash_edges:
                        edges_have += 1
            if edges_have == (nodes*(nodes - 1)) // 2:
                count += 1
        
        return count
    
    def dfs(self,node,graph,seen,curr_comp):
        seen.add(node)
        curr_comp.append(node)
        for neigh in graph[node]:
            if neigh not in seen:
                self.dfs(neigh,graph,seen,curr_comp)

##########################################################
# 1976. Number of Ways to Arrive at Destination (REVISTED)
# 23MAR25
###########################################################
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        '''
        dijkstra's when we have a new min cost, we carry over ways from 
        if we have equal path, we need to add up ways
        number of ways with dijkstra paradism
        in langs like C++ and Java, int max won't work
        there are 200 nodes, and each edge can be as high as 10**9
        the shortest path could be 199*10^9 which exceedns INT MAX
        should use LONG MAX
        python doesn't matter
        '''
        #first make graph
        graph = defaultdict(list)
        for u,v,w in roads:
            graph[u].append((v,w))
            graph[v].append((u,w))
        
        dists = [float('inf')]*n
        ways = [0]*n
        mod = 10**9 + 7
        ways[0] = 1
        dists[0] = 0

        min_heap = [(0,0)]
        seen = set()

        while min_heap:
            min_dist,curr_node = heapq.heappop(min_heap)
            #can't minimize
            if dists[curr_node] < min_dist:
                continue
            seen.add(curr_node)
            for neigh,weight in graph[curr_node]:
                if neigh in seen:
                    continue
                neigh_dist = dists[curr_node] + weight
                if neigh_dist < dists[neigh]:
                    dists[neigh] = neigh_dist
                    ways[neigh] = ways[curr_node]
                    heapq.heappush(min_heap,(neigh_dist,neigh))
                elif neigh_dist == dists[neigh]:
                    ways[neigh] += (ways[curr_node] % mod) % mod
        return ways[n-1] % mod
    
##############################################
# 3169. Count Days Without Meetings
# 24MAR25
##############################################
class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        '''
        need to merge meetings first!
        sort and count the days in between intervals where there is no overlap
        meetings bounded by days
        '''
        meetings.sort()
        merged = [meetings[0]]
        for start,end in meetings[1:]:
            #if in between
            if merged[-1][0] <= start <=  merged[-1][1]:
                merged[-1][1] = max(end,merged[-1][1])
            else:
                merged.append([start,end])

        ans = 0
        for i in range(1,len(merged)):
            start_one,end_one = merged[i-1]
            start_two,end_two = merged[i]
            if start_two > end_one:
                ans += start_two - end_one - 1
        print(merged)
        #include ending days
        ans += (days - merged[-1][1])
        ans += (merged[0][0] - 1)
        return ans

class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        '''
        another way is to use line sleep, but since days can be big, 10**9
        we need to use hashmpa instead
        we can use a difference map,
            line sweep with an array, we need to roll up
        '''
        difference_map = defaultdict(int)
        smallest_day = float('inf')
        largest_day = 0
        for start,end in meetings:
            smallest_day = min(smallest_day,start)
            largest_day = max(largest_day,end)
            #appyl to difference map
            difference_map[start] += 1
            difference_map[end + 1] -= 1
        
        #any day before the smallest day is free
        free_days = smallest_day - 1
        free_days += (days - largest_day)

        pref_sum = 0
        prev_day = smallest_day
        #need to check the differnce beteen day and prev_day
        #in the line sweep array, zeros mean there was a gap
        for day in sorted(difference_map):
            if pref_sum == 0:
                free_days += day - prev_day
            pref_sum += difference_map[day]
            prev_day = day
        
        return free_days
    
class Solution:
    def countDays(self, days: int, meetings: List[List[int]]) -> int:
        '''
        sinlge pass
        '''
        free_days = 0
        latest_end = 0

        # Sort meetings based on starting times
        meetings.sort()

        for start, end in meetings:
            # Add current range of days without a meeting
            if start > latest_end + 1:
                free_days += start - latest_end - 1

            # Update latest meeting end
            latest_end = max(latest_end, end)

        # Add all days after the last day of meetings
        free_days += days - latest_end

        return free_days
    
#################################################
# 3394. Check if Grid can be Cut into Sections
# 26MAR25
###############################################
class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        '''
        coords of rectagnles are (start_x,start_y,end_x,end_y)
        rectangles do not overlap
        need to check if we can make two horiz cuts or two vert cuts 
        such that:
            each section has at least one rectagnle
            every rectable belongs to only one section
        
        for the one dimensional problem, we can use line sleep, and check gaps, there should be at least two gaps
        '''
        xs = []
        ys = []
        for start_x,start_y,end_x,end_y in rectangles:
            x = [start_x,end_x]
            xs.append(x)
            y = [start_y,end_y]
            ys.append(y)
        #sort them on start
        xs.sort()
        ys.sort()

        return self.check_gaps(xs) >= 2 or self.check_gaps(ys) >= 2
    def check_gaps(self,arr):
        gaps = 0
        prev_end = arr[0][1]
        for start,end in arr[1:]:
            if start >= prev_end:
                gaps += 1
            prev_end = max(prev_end,end)
        
        return gaps

####################################################
# 2033. Minimum Operations to Make a Uni-Value Grid
# 27MAR25
####################################################
class Solution:
    def minOperations(self, grid: List[List[int]], x: int) -> int:
        '''
        we can only add/subtract x, but we can do it any number of times on any element
        m*n is  <= 10**5
        we cant brind them to same number if they have different their remainders are different when divided by x
        need to bring to the middle number, median

        if all numbers must have the same remainder when divided by x
        say a target is v, we need (v - a) % x == 0 and (v - b) % x == 0
        (v - a) % x = (v - b) % x
        this is just congruene form
        a conguent to b % x
        a % x = b % x
        and v congruent to b % x
        so v % x = b % x!

        now why is it the median?
        say we have [a,b,c] and v is some number
        we cant to miinize (v - a)^2 + (v - b)^2 + (v - c)^2
        derive wrt to v
        (a**2 - 2av + v^2) +  (b**2 - 2bv + v^2) +  (c**2 - 2cv + v^2)
        (-2a + 2v) + (-2b + 2v) + (-2c + 2v) = 0
        solve v
        6v = -2(a+b+c)
        v = -(a+b+c) // 2, this is just the aveage
        '''
        arr = []
        for row in grid:
            for num in row:
                arr.append(num)
        
        if not self.can_do(arr,x):
            return -1
        arr.sort()
        target = arr[len(arr)//2]
        ops = 0

        for num in arr:
            ops += abs(target - num) // x
        
        return ops
    
    def can_do(self,arr,x):
        remainder = arr[0] % x
        for num in arr[1:]:
            if num % x != remainder:
                return False
        return True

###############################################
# 2780. Minimum Index of a Valid Split
# 27MAR25
###############################################
#ezzzz
class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        '''
        a value of nums x, is dominant if more than half of the elemnts of arr have value of x
        first find the dominant element
        i can keep track of the number of dominant elements at each index, and return the minimum i
        '''
        #find dom element
        n = len(nums)
        counts = Counter(nums)
        dominant = -1
        count_dominant = -1
        for k,v in counts.items():
            if v > n//2:
                dominant = k
                count_dominant = v
            
        curr_count_dom = 0
        for i in range(n-1):
            num = nums[i]
            curr_count_dom += num == dominant
            left_size = i + 1
            right_size = n - i - 1
            if curr_count_dom > left_size//2 and count_dominant - curr_count_dom > right_size//2:
                return i
        
        return -1
    
class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        '''
        we actuall don't need to keep track of the dominant element,
        just counts for the left and the right parts
        '''
        left_counts = Counter()
        right_counts = Counter()
        n = len(nums)

        for num in nums:
            #inially left is empty, and right has all
            right_counts[num] += 1
        
        for i in range(n-1):
            num = nums[i]
            left_counts[num] += 1
            right_counts[num] -= 1
            left_size = i + 1
            right_size = n - i - 1
            if left_counts[num] > left_size//2 and right_counts[num] > right_size//2:
                return i
        return -1

class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        '''
        we actuall don't need to keep track of the dominant element,
        just counts for the left and the right parts

        another optimization would be to use Boyer-moore to find the majority element
        the idea is that if the elements appears len(nums)/2 times, then it must appear after cancelling out the other elements!

        '''
        #find dom
        n = len(nums)
        dominant,count_dominant = self.boyer_moore(nums)

        curr_count_dom = 0
        for i in range(n-1):
            num = nums[i]
            curr_count_dom += num == dominant
            left_size = i + 1
            right_size = n - i - 1
            if curr_count_dom > left_size//2 and count_dominant - curr_count_dom > right_size//2:
                return i
        
        return -1
    
    def boyer_moore(self,arr):
        dom = arr[0]
        count = 0
        dom_count = 0
        for num in arr:
            if num == dom:
                count += 1
            else:
                count -= 1
            if count == 0:
                dom = num
                dom_count = 1
        return [dom,dom_count]
    
###########################################
# 2737. Find the Closest Marked Node
# 27MAR25
###########################################
class Solution:
    def minimumDistance(self, n: int, edges: List[List[int]], s: int, marked: List[int]) -> int:
        '''
        djikstras from s
        then check all distances from s to marked and take the minimum
        '''
        #make graph
        graph = defaultdict(list)
        for u,v,w in edges:
            graph[u].append((v,w))


        dists = [float('inf')]*n
        visited = [False]*n
        dists[s] = 0

        min_heap = [(0,s)]
        while min_heap:
            min_dist,curr_node = heapq.heappop(min_heap)
            if visited[curr_node]:
                continue
            visited[curr_node] = True
            for neigh,weight in graph[curr_node]:
                neigh_dist = min_dist + weight
                if dists[neigh] < neigh_dist or visited[neigh]:
                    continue
                else:
                    dists[neigh] = neigh_dist
                    heapq.heappush(min_heap, (neigh_dist,neigh))

        ans = float('inf')
        for m in marked:
            if dists[m] != float('inf'):
                ans = min(ans,dists[m])
        print(dists)
        return ans if ans != float('inf') else -1

#we can just return if we find a marked node!
class Solution:
    def minimumDistance(self, n: int, edges: List[List[int]], s: int, marked: List[int]) -> int:
        '''
        djikstras from s
        then check all distances from s to marked and take the minimum
        '''
        #make graph
        graph = defaultdict(list)
        for u,v,w in edges:
            graph[u].append((v,w))
        
        marked = set(marked)
        dists = [float('inf')]*n
        visited = [False]*n
        dists[s] = 0

        min_heap = [(0,s)]
        while min_heap:
            min_dist,curr_node = heapq.heappop(min_heap)
            if curr_node in marked:
                return dists[curr_node]
            if visited[curr_node]:
                continue
            visited[curr_node] = True
            for neigh,weight in graph[curr_node]:
                neigh_dist = min_dist + weight
                if dists[neigh] < neigh_dist or visited[neigh]:
                    continue
                else:
                    dists[neigh] = neigh_dist
                    heapq.heappush(min_heap, (neigh_dist,neigh))
        
        return -1
    
#bellman ford
class Solution:
    def minimumDistance(self, n: int, edges: List[List[int]], s: int, marked: List[int]) -> int:
        '''
        bellman ford,
        basically relax all edges if we can n - 1 times
        intuition
            if there is a shortest path, then it can contain at most n-1 edges
        '''
        dists = [float('inf')]*n
        dists[s] = 0
        for _ in range(n-1):
            for u,v,w in edges:
                if dists[u] != float('inf') and dists[u] + w < dists[v]:
                    dists[v] = dists[u] + w
        
        min_dist = min([dists[node] for node in marked],default=float('inf'))
        return -1 if min_dist == float("inf") else min_dist

###################################################
# 2503. Maximum Number of Points From Grid Queries
# 28MAR25
###################################################
#almost
class Solution:
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        '''
        notice that for a query, its just the number of cells i can visit
        for each query we can revist the same cell multiple times
        brute force would be to bfs/dfs for each query and count the number of cells i can visit
            i.e all (i,j) in visited whre grid[i][j] < current query that i'm on
            this would obviously take too long, m*n*k
        
        sort queries, and bfs from upper left
        as we bfs, keep track of number of cells that are < the current query
        in the queue,keep track of the (i,j) as well at the current query
        '''
        n = len(queries)
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        queries_sorted = [(q,i) for (i,q) in enumerate(queries)]
        queries_sorted.sort(key = lambda x: x[0])
        ans = [0]*n
        seen = set()
        q = deque([(0,0,0)]) #entry is (i,j,kth query)
        while q:
            i,j,k = q.popleft()
            #if we added here we can conttinue to explore neighbors
            if grid[i][j] < queries_sorted[k][0]:
                ans[k] += 1
                seen.add((i,j)) #mark cell as seen
                for di,dj in dirrs:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < rows and 0 <= jj < cols and (ii,jj) not in seen:
                        if grid[ii][jj] <= queries_sorted[k][0]:
                            q.append((ii,jj,k))
            #if wecan't add, move on to the next index
            else:
                if k + 1 < n:
                    q.append((i,j,k+1))
        
        return ans

#yesss
from queue import PriorityQueue
import heapq
class Solution:
    def maxPoints(self, grid: List[List[int]], queries: List[int]) -> List[int]:
        '''
        notice that for a query, its just the number of cells i can visit
        for each query we can revist the same cell multiple times
        brute force would be to bfs/dfs for each query and count the number of cells i can visit
            i.e all (i,j) in visited whre grid[i][j] < current query that i'm on
            this would obviously take too long, m*n*k
        
        sort queries, and bfs from upper left
        as we bfs, keep track of number of cells that are < the current query
        in the queue,keep track of the (i,j) as well at the current query

        we need to use heap, to keep track of the minimum value in our expansion
        as long as the min value is smaller than the current query, we can keep expanding

        new thing, there is a 
        from queue import PriorityQueue

        '''
        n = len(queries)
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        queries_sorted = [(q,i) for (i,q) in enumerate(queries)]
        queries_sorted.sort(key = lambda x: x[0])
        ans = [0]*n
        visited = [[False]*cols for _ in range(rows)]
        visited[0][0] = True
        min_heap = [(grid[0][0],0,0)]
        #min_heap = PriorityQueue()
        #min_heap.put((grid[0][0],0,0))
        curr_points = 0

        for query_bound,idx in queries_sorted:
            #only expand when we query bound is higher
            while min_heap and min_heap[0][0] < query_bound:
                curr_val, i,j = heapq.heappop(min_heap)
                #curr_val, i,j = min_heap.get()
                curr_points += 1
                #neigh expansion
                for di,dj in dirrs:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < rows and 0 <= jj < cols and not visited[ii][jj]:
                        heapq.heappush(min_heap, (grid[ii][jj],ii,jj))
                        #min_heap.put((grid[ii][jj],i,j))
                        #mark
                        visited[ii][jj] = True
            #place ans
            ans[idx] = curr_points
        
        return ans
                
#############################################
# 2818. Apply Operations to Maximize Score
# 30MAR25
##############################################
#still TLEs...
class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        '''
        we can only use a subarray from i:j only one time
        but subarrays can overlap, i.e we can do [1:4] and [2:5]
        prime score is number of prime factors
        for each number in the array, i can calculate its prime score in logarithmic time, once we have the prime scores
        we need to keep track of the prime score of each number
        the issue is that there are n*n subarrays, but for each subarray we want the num with largest prime factor
            tiebreaker would be the leftmost index
        problem is that we can't do in O(k) time either, because thats lower bounded by n*n
        prime factors could have been any hueristic, just a wrench lol
        intuitions:
            for each index i, we need to know the number or ranges where s[i] is the maximum
            if we could repeatedly us a subarray, then we would just pick the max prime score num every time
            a number has the largest pimre score until another on shows up, with a greater prime score to the left or the right

        we then need to use fast exponentiation since the numbers could be large
        '''
        #prime score
        n = len(nums)
        s = [0]*n
        for i,num in enumerate(nums):
            s[i] = self.prime_factors(num)

        #for each index in nums, determine how many subarrays nums[i], or s[i] is the maximum, monstock for left and right
        #left[i]: neareast index to the left where prime score is >= s[i]
        #right[i]: nearest index to the right where prime score > s[i]
        left = [-1]*n
        right = [n]*n
        #monostack for decreasing prime score, it contains the indices,and for each index at the top, we can check is primescore
        stack = []
        for i in range(n):
            #need s[i] to be largest, i.e stack is no longer decreasing
            while stack and s[stack[-1]] < s[i]:
                idx = stack.pop()
                right[idx] = i
            #if stack is not empty, set prev largest prime score elemnt to current index
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        
        #count number of subarrays where nums[i] has largest s[i]
        count_subarrays = [0]*n
        for i in range(n):
            count_subarrays[i] = (i - left[i])*(right[i] - i)
        
        #sort elemments in desceding order, we want to use the largest nums to get the largest score, along with index
        sorted_elements = [(num,i) for i,num in enumerate(nums)]
        sorted_elements.sort(key = lambda x: -x[0])
        score = 1
        mod = 10**9 + 7
        for num,index in sorted_elements:
            operations = min(k,count_subarrays[index])
            #get this num's prime score
            curr_ps = s[index]
            print(num,operations)
            #multiply, need to use fast exponentaion here
            #recursive seems to TLE here :(
            temp = self._power(num,operations,mod)
            score *= temp
            score %= mod
            k -= operations
            if k == 0:
                return score
        return score

    def prime_factors(self,num):
        factors = set()
        curr_prime = 2
        #keep dividing it
        
        while curr_prime <= num:
            if num % curr_prime == 0:
                factors.add(curr_prime)
                num = num // curr_prime
            else:
                curr_prime += 1
        
        return len(factors)

    def fast_power(self,x,power,mod):
        if power == 0:
            return 1
        half_power = self.fast_power(x,power//2,mod) % mod
        if power % 2 == 0:
            return (half_power*half_power) % mod
        return (x*half_power*half_power) % mod
    

    def _power(self,base, exponent,mod):
        res = 1

        # Calculate the exponentiation using binary exponentiation
        while exponent > 0:
            # If the exponent is odd, multiply the result by the base
            if exponent % 2 == 1:
                res = (res * base) % mod

            # Square the base and halve the exponent
            base = (base * base) % mod
            exponent //= 2
        
        return res

#just brute foce all factors
class Solution:
    def maximumScore(self, nums: List[int], k: int) -> int:
        '''
        we can only use a subarray from i:j only one time
        but subarrays can overlap, i.e we can do [1:4] and [2:5]
        prime score is number of prime factors
        for each number in the array, i can calculate its prime score in logarithmic time, once we have the prime scores
        we need to keep track of the prime score of each number
        the issue is that there are n*n subarrays, but for each subarray we want the num with largest prime factor
            tiebreaker would be the leftmost index
        problem is that we can't do in O(k) time either, because thats lower bounded by n*n
        prime factors could have been any hueristic, just a wrench lol
        intuitions:
            for each index i, we need to know the number or ranges where s[i] is the maximum
            if we could repeatedly us a subarray, then we would just pick the max prime score num every time
            a number has the largest pimre score until another on shows up, with a greater prime score to the left or the right

        we then need to use fast exponentiation since the numbers could be large
        '''
        #prime score
        n = len(nums)
        s = [0]*n
        for i,num in enumerate(nums):
            s[i] = self.prime_factors(num)

        #for each index in nums, determine how many subarrays nums[i], or s[i] is the maximum, monstock for left and right
        #left[i]: neareast index to the left where prime score is >= s[i]
        #right[i]: nearest index to the right where prime score > s[i]
        left = [-1]*n
        right = [n]*n
        #monostack for decreasing prime score, it contains the indices,and for each index at the top, we can check is primescore
        stack = []
        for i in range(n):
            #need s[i] to be largest, i.e stack is no longer decreasing
            while stack and s[stack[-1]] < s[i]:
                idx = stack.pop()
                right[idx] = i
            #if stack is not empty, set prev largest prime score elemnt to current index
            if stack:
                left[i] = stack[-1]
            stack.append(i)
        
        #count number of subarrays where nums[i] has largest s[i]
        count_subarrays = [0]*n
        for i in range(n):
            count_subarrays[i] = (i - left[i])*(right[i] - i)
        
        #sort elemments in desceding order, we want to use the largest nums to get the largest score, along with index
        sorted_elements = [(num,i) for i,num in enumerate(nums)]
        sorted_elements.sort(key = lambda x: -x[0])
        score = 1
        mod = 10**9 + 7
        for num,index in sorted_elements:
            operations = min(k,count_subarrays[index])
            #get this num's prime score
            curr_ps = s[index]
            print(num,operations)
            #multiply, need to use fast exponentaion here
            #recursive seems to TLE here :(
            temp = self.fast_power(num,operations,mod)
            score *= temp
            score %= mod
            k -= operations
            if k == 0:
                return score
        return score

    def prime_factors(self,num):
        #this part needs to be speed up
        factors = 0
        # Check for prime factors from 2 to sqrt(n)
        for factor in range(2, int(math.sqrt(num)) + 1):
            if num % factor == 0:
                factors += 1
                # Remove all occurrences of the prime factor from num
                while num % factor == 0:
                    num //= factor
        if num >= 2:
            factors += 1

        return factors

    def fast_power(self,x,power,mod):
        if power == 0:
            return 1
        half_power = self.fast_power(x,power//2,mod) % mod
        if power % 2 == 0:
            return (half_power*half_power) % mod
        return (x*half_power*half_power) % mod
    

    def _power(self,base, exponent,mod):
        res = 1

        # Calculate the exponentiation using binary exponentiation
        while exponent > 0:
            # If the exponent is odd, multiply the result by the base
            if exponent % 2 == 1:
                res = (res * base) % mod

            # Square the base and halve the exponent
            base = (base * base) % mod
            exponent //= 2
        
        return res
