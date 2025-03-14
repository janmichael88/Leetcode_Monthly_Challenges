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
        sides = set()
        for i,j in comp:
            for di, dj in dirrs:
                ii = i + di
                jj = j + dj
                if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] == 0 and (ii,jj) not in comp:
                    sides.add((ii,jj,i,j))
        return len(sides) 

    
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