##################################
# 2678. Number of Senior Citizens
# 01AUG24
###################################
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        '''
        first ten chars are phone numbers
        next char is geneder
        next to chars are age
        10+1 
        ''' 
        age_idx = 10 + 1
        ans = 0
        for d in details:
            if int(d[age_idx:age_idx + 2]) > 60:
                ans += 1
        
        return ans
    
#doing char by char insteaf of converting the whole thing
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        '''
        first ten chars are phone numbers
        next char is geneder
        next to chars are age
        10+1 
        ''' 
        age_idx = 10 + 1
        ans = 0
        for d in details:
            tens = ord(d[age_idx]) - ord('0')
            ones = ord(d[age_idx + 1]) - ord('0')
            
            age = tens*10 + ones
            if age > 60:
                ans += 1
        
        return ans
    
###################################################
# 2134. Minimum Swaps to Group All 1's Together II
# 02AUG24
##################################################
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        '''
        there could be multiple arrangments for which a ciruclar array is divided accordigly, we want the minimum
        if there are k's, then each subarray of length k should all 1
        we want the the subarray wit the smallest number of zeros
        we dont want to keep computing the number of zeros, so we can use sliding window
        '''
        ones = sum([num == 1 for num in nums])
        if ones == 0:
            return 0
        nums_doubled = nums + nums
        ans = float('inf')
        left = 0
        count_zeros = 0
        
        for right in range(len(nums_doubled)):
            count_zeros += nums_doubled[right] == 0
            if right - left + 1 == ones:
                ans = min(ans,count_zeros)
                count_zeros -= nums_doubled[left] == 0
                left += 1
            
        return ans
    
#we dont need to concat, jsut use mod N
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        '''
        no need to concat, just use mod N
        '''
        ones = sum([num == 1 for num in nums])
        if ones == 0:
            return 0
        ans = float('inf')
        left = 0
        count_zeros = 0
        N = len(nums)
        
        for right in range(2*N):
            count_zeros += nums[right % N] == 0
            if right - left + 1 == ones:
                ans = min(ans,count_zeros)
                count_zeros -= nums[left % N] == 0
                left += 1

        return ans
    
###########################################
# 1062. Longest Repeating Substring
# 02JUL24
############################################
#cheeze
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        generate all possible substrings
        '''
        seen = set()
        N = len(s)
        ans = 0
        for i in range(N):
            for j in range(i+1,N+1):
                substring = s[i:j]
                if substring in seen:
                    ans = max(ans,len(substring))
                else:
                    seen.add(substring)
        
        return ans
    
#using rolling hash
#doesnt quite work collisiosn???
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        need to use rolling hash instead of getting subsstring
        hash_value = (hash_value + (c - 'a' + 1) * p_pow) % m;
        p_pow = (p_pow * p) % m;
        '''
        print(len(s))
        seen = set()
        N = len(s)
        ans = 0
        mod = 10**9 + 7
        base = 31
        for i in range(N):
            curr_hash = 0
            p_pow = 1
            for j in range(i,N):
                dig = ord(s[j]) - ord('a')
                curr_hash = (curr_hash + (dig + 1)*p_pow) % mod
                p_pow = (p_pow*base) % mod
                if curr_hash in seen:
                    ans = max(ans,j-i+1)
                else:
                    seen.add(curr_hash)
        print(len(seen))
        return ans


class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        
        def check_length(length):
            seen = set()
            base = 26
            mod = 2**31 - 1
            current_hash = 0
            base_power = 1
            for i in range(length):
                current_hash = (current_hash * base + ord(s[i])) % mod
                base_power = (base_power * base) % mod
            
            seen.add(current_hash)
            
            for start in range(1, len(s) - length + 1):
                current_hash = (current_hash * base - ord(s[start - 1]) * base_power + ord(s[start + length - 1])) % mod
                if current_hash in seen:
                    return True
                seen.add(current_hash)
            
            return False
        
        left, right = 1, len(s)
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check_length(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans

##########################################
# 1508. Range Sum of Sorted Subarray Sums
# 04AUG24
##########################################
class Solution:
    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
        '''
        say we have three numbers. a,b,c
        a + b + c + (a + b) + (a + b + c) + (b + c)
        sums are sorted increasingly
        a + b + c + (a + b) + (b+c) + (a+b+c)
        use pref sums, then find all (i,j) sums to get subarrays
        
        '''
        pref_sum = [0]
        for num in nums:
            pref_sum.append(pref_sum[-1] + num)
        
        N = len(nums)
        all_sums = []
        for i in range(N):
            for j in range(i,N):
                sub_sum = pref_sum[j+1] - pref_sum[i]
                all_sums.append(sub_sum)
        
        all_sums.sort()
        ans = 0
        mod = 10**9 + 7
        for i in range(left,right+1):
            ans += all_sums[i-1]
            ans %= mod
        
        return ans
    
#dont need to use pref sum
class Solution:
    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
        '''
        say we have three numbers. a,b,c
        a + b + c + (a + b) + (a + b + c) + (b + c)
        sums are sorted increasingly
        a + b + c + (a + b) + (b+c) + (a+b+c)
        use pref sums, then find all (i,j) sums to get subarrays
        
        '''
        N = len(nums)
        all_sums = []
        for i in range(N):
            sub_sum = 0
            for j in range(i,N):
                sub_sum += nums[j]
                all_sums.append(sub_sum)
        
        all_sums.sort()
        ans = 0
        mod = 10**9 + 7
        for i in range(left,right+1):
            ans += all_sums[i-1]
            ans %= mod
        
        return ans
    
#min heap, in inceasing sum order
class Solution:
    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
        '''
        we can use priority queue to go in order of increasing sums
        insert all the length 1 subarray sums into a min heap (increaisng order) but pair with its index,
        if index is greater >= left, this sum must be included in the answer, then we can add in the next sum to the current subaray sum
        '''
        min_heap = []
        N = len(nums)
        for i in range(N):
            min_heap.append((nums[i],i))
        
        
        heapq.heapify(min_heap)
        mod = 10**9 + 7
        ans = 0
        for i in range(1,right+1):
            top = heapq.heappop(min_heap)
            if i >= left:
                ans += (top[0] % mod) % mod
            
            #get the next sum
            if top[1] < n - 1:
                next_sum = top[0] + nums[top[1] + 1]
                next_idx = top[1] + 1
                heapq.heappush(min_heap, (next_sum, next_idx))
        
        return ans % mod
    
#binary search
class Solution:
    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
        # we have a monotonic search space
        #   - for a certain sum := target, there are either =k subarrays whose sum is leq target
        #   - we want to find the minimum target s.t. left-1 subarrs sum to it and the same for right
        #   - after finding this minimum target, we need the total sum of subarrs up to this target
        #   - the difference of these 2 total sums is the intermediate sum
        # now we need to calculate the number of subarrays whose sum is leq target in linear time
        #   - we can use a sliding window to find the (left, right) pairs that sum to leq target
        #   - once we have a left, right pair leq target, we know all subarrs within this pair also sum leq target
        #   - this is the subarrays sliding window pattern: at each valid window,
        #       - count += length because you can take subarrays 1st elem to this, 2nd etc
        #       - window_contribition += nums[i] * length bc nums[i] will contribute to count subarrays
        #       - window_sum += nums[i]
        #   - at each invalid window:
        #       - window_contribition -= window_sum because this whole window is invalid
        #       - window_sum -= nums[i]
        #   - total sum is sum of all window contributions
        
        def count_subarrays(target):
            left = right = 0
            total_sum = 0
            window_sum = 0
            window_contribution = 0
            count = 0
            while right < n:
                if window_sum <= target:  # if statement not needed but following template
                    window_sum += nums[right]
                    window_contribution += nums[right] * (right - left + 1)
                    right += 1
                while window_sum > target:
                    window_contribution -= window_sum
                    window_sum -= nums[left]
                    left += 1
                count += right - left
                total_sum += window_contribution

            return count, total_sum

        def first_k_sum(k,lo,hi):
            left = lo
            right = hi

            # minimize target sum s.t. count of subarrays leq target sum is >=k
            while left < right:
                mid = (left + right) // 2
                if count_subarrays(mid)[0] >= k:
                    right = mid
                else:
                    left = mid + 1

            count, total_sum = count_subarrays(left)
            # subarrs(left) >= k but it may be greater if many subarrays have sum=k
            # we need to shave off the extra subarrays and return only the first k
            extra_subarray_count = count - k
            return total_sum - left*extra_subarray_count

        lo = min(nums)
        hi = sum(nums)
        return (first_k_sum(right,lo,hi) - first_k_sum(left-1,lo,hi)) % (10**9 + 7)

###########################################
# 2053. Kth Distinct String in an Array
# 05AUG24
###########################################
class Solution:
    def kthDistinct(self, arr: List[str], k: int) -> str:
        '''
        count them up, then going left to right, check if count(s) > 1
        return the kth one
        '''
        counts = Counter(arr)
        for s in arr:
            if counts[s] == 1:
                k -= 1
            if k == 0:
                return s
        
        return ""
    
#################################################
# 3016. Minimum Number of Pushes to Type Word II
# 06AUG24
################################################
#nice try, come back to this one
class Solution:
    def minimumPushes(self, word: str) -> int:
        '''
        if a char is un mapped to a number, mapp it
        if its already been mapped, then access how many times that char has already been mapped 
        ill need two mapps, one mapping char to num, and one mapping num to the its chars
        but what number do i mapp it too? just the first unmapped number
        just check if char has already been mapped or not, but remember there are only 8 numbers we can mapp too (2 through 9)
        keep count of unmapped numbers
        '''
        positions = [0]*26
        unmapped_numbers = 0
        
        ans = 0
        for ch in word:
            pos = ord(ch) - ord('a')
            #already been mapped
            if positions[pos] != 0:
                ans += positions[pos]
            #not mapped yet
            else:
                #havent mapped all 8 numbers yet, so number is available
                if unmapped_numbers < 8:
                    positions[pos] += 1
                    unmapped_numbers += 1
                else:
                    new_counts = unmapped_numbers // 8
                    positions[pos] = new_counts + 1

                    ans += positions[pos]

        return ans
                
class Solution:
    def minimumPushes(self, word: str) -> int:
        '''
        sort, count, and go in groups of 8
        '''
        counts = Counter(word)
        ans = 0
        unmapped_numbers = 0
        for k,v in sorted(counts.items(), key = lambda x: -x[1]):
            ans += ((unmapped_numbers // 8) + 1)*v
            unmapped_numbers += 1
        
        return ans
    
###############################################
# 1812. Determine Color of a Chessboard Square
# 07AUG24
###############################################
class Solution:
    def squareIsWhite(self, coordinates: str) -> bool:
        '''
        even numbers going left to right start with white then alternate
        odd numbers are the opposite, the order zigzags, i can go up fromm 1 to 8, and then alternate
        convert cell to cooridnate
            if row is odd, we are going left to right
            if row is even we are going right to left
        '''
        row = ord(coordinates[1]) - ord('0')
        col = ord(coordinates[0]) - ord('a') + 1
        
        if row % 2 == 1:
            return 1- (col) % 2
        return (col % 2)

############################################
# 273. Integer to English Words (REVISITED)
# 07AUG24
###########################################
class Solution:
    def numberToWords(self, num: int) -> str:
        '''
        need english numbers < 10, i.e Ten,Nine,Eight...
        need enligh numbers < 20, like Nineteen,Eighteen, etc
        then below hundred in multiples of 10, like Twenty,Thirty,Forty, etc
        for numbers bigger, we process recursively by divided by 1000,100000,1000000, and a billion
        
        base case, for numbers < 10, return enligh of belowTen
        between 10 and 19, return bewlow Twenty
        for numbers beteween 20 and 99, return belowHundred
        then reusviely call on num / 100, num / 1000, num / millions, num / billions
        
        ie.
        Base Case: For numbers less than 10, the function directly maps to a word using belowTen. 
        For numbers between 10 and 19, belowTwenty handles these unique cases. 
        Ror numbers between 20 and 99, it combines words from belowHundred for tens and recursively processes the remainder for units.
        
        recursive case:
        Numbers from 100 to 999:
        Combine the recursive result for the hundreds place with "Hundred", and the recursive result for the remaining part.
        Numbers from 1000 to 999,999:
        Combine the recursive result for thousands with "Thousand", and the recursive result for the remaining part.
        Numbers from 1,000,000 to 999,999,999:
        Combine the recursive result for millions with "Million", and the recursive result for the remaining part.
        Numbers 1,000,000,000 and above:
        Combine the recursive result for billions with "Billion", and the recursive result for the remaining part.
        '''
        below_ten = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        below_twenty = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        below_hundred = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        
        def rec(num):
            if num < 10:
                return below_ten[num]
            if num < 20:
                return below_twenty[num - 10]
            if num < 100:
                return below_hundred[num // 10] + (" " + rec(num % 10) if num % 10 != 0 else "")
            if num < 1000:
                return rec(num // 100) + " Hundred" + (" " + rec(num % 100) if num % 100 != 0 else "")
            if num < 1000000:
                return rec(num // 1000) + " Thousand" + (" " + rec(num % 1000) if num % 1000 != 0 else "")
            if num < 1000000000:
                return rec(num // 1000000) + " Million" + (" " + rec(num % 1000000) if num % 1000000 != 0 else "")
            return rec(num // 1000000000) + " Billion" + (" " + rec(num % 1000000000) if num % 1000000000 != 0 else "")
        
        if num == 0:
            return "Zero"
        
        return rec(num)
    
#another way
class Solution:
    def numberToWords(self, num: int) -> str:
        '''
        using walrus operator
        '''
        digit_name = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        teens_name = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        tens_name = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        
        def rec(num):
            out = ""
            if billions := num // 1_000_000_000:
                out += rec(billions) + " Billion "
            if millions := (num // 1_000_000) % 1000:
                out += rec(millions) + " Million "
            if thousands := (num // 1000) % 1000:
                out += rec(thousands) + " Thousand "
            if hundreds := (num // 100) % 10:
                out += digit_name[hundreds] + " Hundred "
            if (tens := (num // 10) % 10) > 1:
                out += tens_name[tens] + " "
            if tens == 1:
                out += teens_name[num % 10] + " "
            elif num % 10 or not out:
                out += digit_name[num % 10] + " "

            return out[:-1]

        return rec(num)
    
#no walrus
class Solution:
    def numberToWords(self, num: int) -> str:
        '''
        using walrus operator
        for billions just check if we can divite by billions
        for millions and thousands, check that we can divide by part and grab $ 1000
        '''
        digit_name = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        teens_name = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
        tens_name = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
        
        def rec(num):
            out = ""
            #check billions first
            billions = num // 1_000_000_000
            if billions > 0:
                out += rec(billions) + " Billion "
            #check milions
            millions = (num // 1_000_000) % 1000
            if millions:
                out += rec(millions) + " Million "
            #check thousands
            thousands = (num // 1000) % 1000
            if thousands:
                out += rec(thousands) + " Thousand "
            #check hundreds, but can all be preceded by a digit name
            hundreds = (num // 100) % 10
            if hundreds:
                out += digit_name[hundreds] + " Hundred "
            #if in 20 and 99
            tens = (num // 10) % 10
            if tens  > 1:
                out += tens_name[tens] + " "
            #if in between 10 and 19
            if tens == 1:
                out += teens_name[num % 10] + " "
            #just the single number
            elif num % 10 or not out:
                out += digit_name[num % 10] + " "

            return out[:-1]

        return rec(num)

########################################
# 885. Spiral Matrix III (REVISTED)
# 08AUG24
#######################################
class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        '''
        just keep walking in a spiral until we get all values
        step increments are 1,1,2,2,3,3,4,4,5,5
        its every two chage in directions step count goes up by 1
        '''
        ans = []
        steps = 1
        while len(ans) < rows*cols:
            #walkig right
            for _ in range(steps):
                if 0 <= rStart < rows and 0 <= cStart < cols:
                    ans.append([rStart,cStart])
                cStart += 1
            
            #walking down
            for _ in range(steps):
                if 0 <= rStart < rows and 0 <= cStart < cols:
                    ans.append([rStart,cStart])
                rStart += 1
            
            steps += 1
            
            #walking left
            for _ in range(steps):
                if 0 <= rStart < rows and 0 <= cStart < cols:
                    ans.append([rStart,cStart])
                cStart -= 1
                
            #walking up
            for _ in range(steps):
                if 0 <= rStart < rows and 0 <= cStart < cols:
                    ans.append([rStart,cStart])
                rStart -= 1
            
            steps += 1
        
        return ans
                    
####################################
# 2326. Spiral Matrix IV
# 08AUG24
####################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
        '''
        walk the matrix in spiral fasion,
        
        '''
        mat = [[-1]*n for _ in range(m)]
        curr = head
        start_row = 0
        end_row = m-1
        start_col = 0
        end_col = n - 1
        
        while start_row <= end_row and  start_col <= end_col:
            for c in range(start_col,end_col+1):
                if curr:
                    mat[start_row][c] = curr.val
                    curr = curr.next
            
            start_row += 1
            
            for r in range(start_row,end_row+1):
                if curr:
                    mat[r][end_col] = curr.val
                    curr = curr.next
            
            end_col -= 1
            
            if (start_row <= end_row):
                for c in range(end_col, start_col-1, -1):
                    if curr:
                        mat[end_row][c] = curr.val
                        curr = curr.next
                
                end_row -= 1
            
            if (start_col <= end_col):
                for r in range(end_row, start_row-1, -1):
                    if curr:
                        mat[r][start_col] = curr.val
                        curr = curr.next
                start_col += 1
                
        
        return mat
    
##############################
# 840. Magic Squares In Grid
# 08AUG24
##############################
class Solution:
    def numMagicSquaresInside(self, grid: List[List[int]]) -> int:
        '''
        a magic square is a 3 x 3 grid filled with distinct numbers from 1 to 9, 
        and all rows, cols, diag, and anti diags have the same sum
        for 3 by 3 anti-diags are (0,2), (1,1), (2,0)
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        if rows < 3 or cols < 3:
            return 0
        
        #try lookin at all squares, by fixing upper left corner
        ans = 0
        for i in range(rows-2):
            for j in range(cols-2):
                square = self.check_square(grid,i,j)
                if not square:
                    continue
                if self.check_magic(square):
                    ans += 1
        
        return ans
                
    
    def check_square(self,grid,i,j):
        seen = set()
        square = []
        for ii in range(3):
            row = []
            for jj in range(3):
                if grid[i+ii][j+jj] > 9 or grid[i+ii][j+jj] in seen or grid[i+ii][j+jj] == 0:
                    return []
                seen.add(grid[i+ii][j+jj])
                row.append(grid[i+ii][j+jj])

            square.append(row)
        
        return square
    
    def check_magic(self,square):
        row_sums = [0,0,0]
        col_sums = [0,0,0]
        diag_sum = 0
        anti_diag_sum = 0
        
        for i in range(3):
            for j in range(3):
                row_sums[i] += square[i][j]
                col_sums[j] += square[i][j]
                if i == j:
                    diag_sum += square[i][j]
                if i + j == 2:
                    anti_diag_sum += square[i][j]
        check_sums = row_sums + col_sums + [diag_sum] + [anti_diag_sum]
        return len(set(check_sums)) == 1
                
                
###########################################
# 959. Regions Cut By Slashes
# 12JUL24
###########################################
#need to blow up the grid
'''
need to expand each (i,j) cell to a 3 x 3
for back slack, we walk diagonllay down to the right in steps (1,1)
for forward salsh, this is we walk diagonallly down to the left
'''
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        '''
        flood fill the empty spots, you can use numbers for
        maintain largest number when doing flood fill
        problem is that we need to be able to squeeze into tight spaces, this is from advent of code!
        i can modify the grid, so that way the lines are longer, if we have a line at (i,j) and its /, make another line at (i+1,j-1)
        if tis \\, then it should go to (i+1,j+1)
        '''
        new_grid = self.make_new_grid(grid)
        n = len(new_grid)
        regions = [[-1]*n for _ in range(n)]
        curr_region = 0
        
        for i in range(n):
            for j in range(n):
                if new_grid[i][j] == ' ' and regions[i][j] == -1:
                    self.dfs(new_grid,i,j,n,curr_region,regions)
                    curr_region += 1

        return curr_region
        
    
    def dfs(self,grid,i,j,n,curr_region,regions):
        regions[i][j] = curr_region
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        for di,dj in dirrs:
            ii = i + di
            jj = j + dj
            #bounds
            if 0 <= ii < n and 0 <= jj < n and grid[ii][jj] == ' ' and regions[ii][jj] == -1:
                self.dfs(grid,ii,jj,n,curr_region,regions)
        
    def make_new_grid(self,grid):
        n = len(grid)
        expanded_grid = [[" "] * (n * 3) for _ in range(n * 3)]

        # Populate the expanded grid based on the original grid
        for i in range(n):
            for j in range(n):
                base_row = i * 3
                base_col = j * 3
                # Check the character in the original grid
                if grid[i][j] == "\\":
                    # Mark diagonal for backslash
                    expanded_grid[base_row][base_col] = "\\"
                    expanded_grid[base_row + 1][base_col + 1] = "\\"
                    expanded_grid[base_row + 2][base_col + 2] = "\\"
                elif grid[i][j] == "/":
                    # Mark diagonal for forward slash
                    expanded_grid[base_row][base_col + 2] = "/"
                    expanded_grid[base_row + 1][base_col + 1] = "/"
                    expanded_grid[base_row + 2][base_col] = "/"
        
        return expanded_grid
    
#union find on triangles
class Solution:
    def regionsBySlashes(self, grid: List[str]) -> int:
        '''
        we can do union find on triangles
        imagine each slash divide the cell into 4 quadrants, going clockwise, and (i,j) cells is divided into quads 0,1,2,3
        forward slack joins regions 0,1 and joins regions 3 and 2
        back slach joins regions 0 and 3, and regions 1 and 2, so for ids we have (rows*cols)*4
        for joining
            the top triagnle of a cell will alwauys connect to the bottom trianlge above it
            a left triangle will connect right
            a slash divide the cell digonally allowing us to combine the two adjacent triangles on each side of the digonals
            
        union rules
        1. if there is a cell above the current cell, union bottom triganle with top
        2. if there is a cell to the left, unino the right to the left
            if not "/"
                union top triangle with right triagnle
            if not '\'
                union top triagnle with left triangle and bottomm tirangle with right
                
        if empty space, connect all 4 triangles
        '''
        grid_size = len(grid)
        total_triangles = grid_size * grid_size * 4
        parent_array = [-1] * total_triangles

        # Initially, each small triangle is a separate region
        region_count = total_triangles

        for row in range(grid_size):
            for col in range(grid_size):
                # Connect with the cell above
                if row > 0:
                    region_count -= self._union_triangles(parent_array,self._get_triangle_index(grid_size, row - 1, col, 2),self._get_triangle_index(grid_size, row, col, 0))
                # Connect with the cell to the left
                if col > 0:
                    region_count -= self._union_triangles(parent_array,self._get_triangle_index(grid_size, row, col - 1, 1),self._get_triangle_index(grid_size, row, col, 3))
                #if /, (0,3) and (2,1)
                if grid[row][col] == "/":
                    region_count -= self._union_triangles(parent_array,self._get_triangle_index(grid_size, row, col, 0),self._get_triangle_index(grid_size, row, col, 3))
                    region_count -= self._union_triangles(parent_array,self._get_triangle_index(grid_size, row, col, 2),self._get_triangle_index(grid_size, row, col, 1))

                # If \\, union (0,1) and (2,3)
                if grid[row][col] == "\\":
                    region_count -= self._union_triangles(parent_array,self._get_triangle_index(grid_size, row, col, 0),self._get_triangle_index(grid_size, row, col, 1))
                    region_count -= self._union_triangles(parent_array,self._get_triangle_index(grid_size, row, col, 3),self._get_triangle_index(grid_size, row, col, 2))
                else:
                    #if space, union all 0 to 3
                    if grid[row][col] == " ":
                        for i in range(1,4):
                            region_count -= self._union_triangles(parent_array,self._get_triangle_index(grid_size, row, col, 0),self._get_triangle_index(grid_size, row, col, i))

        return region_count

    def _get_triangle_index(self, grid_size, row, col, triangle_num):
        return (grid_size * row + col) * 4 + triangle_num

    def _union_triangles(self, parent_array, x, y):
        parent_x = self._find_parent(parent_array, x)
        parent_y = self._find_parent(parent_array, y)
        if parent_x != parent_y:
            parent_array[parent_x] = parent_y
            return 1  # Regions were merged, so count decreases by 1
        return 0  # Regions were already connected

    def _find_parent(self, parent_array, x):
        if parent_array[x] == -1:
            return x
        parent_array[x] = self._find_parent(parent_array, parent_array[x])
        return parent_array[x]
    
#######################################################
# 1568. Minimum Number of Days to Disconnect Island
# 12AUG24
#######################################################
#close one, so fucking annoying
class Solution:
    def minDays(self, grid: List[List[int]]) -> int:
        '''
        if there is only one island, we need to disconnect it from all the water
        but if all islands are disconnect, we'are done
        if there exactly one island, othwerwise its disconnected, we need to make it disconnected
        we wont need more than two days to disconnect the grid? wtf
        
        if disconnected, meaning there isn't exaclty one island return 0
        return 1 if chaning a single land to water disconnectes the island
        othewise return 2
        '''
        rows = len(grid)
        cols = len(grid[0])
        islands = self.count_islands(grid,rows,cols)
        if islands != 1:
            return 0
        #try swapping 
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    islands = self.count_islands(grid,rows,cols)
                    if islands != 1:
                        return 1
                    grid[i][j] = 0
        
        return 2
    
    #function for exactly one island
    def dfs(self,i,j,grid,seen,rows,cols):
        seen.add((i,j))
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        for di,dj in dirrs:
            ii = i + di
            jj = j + dj
            if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] == 1 and (ii,jj) not in seen:
                self.dfs(ii,jj,grid,seen,rows,cols)
    
    def count_islands(self,grid,rows,cols):
        seen = set()
        islands = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1 and (i,j) not in seen:
                    self.dfs(i,j,grid,seen,rows,cols)
                    islands += len(seen) > 0
        return islands
    
#actual solution
class Solution:
    def minDays(self, grid: List[List[int]]) -> int:
        '''
        if there is only one island, we need to disconnect it from all the water
        but if all islands are disconnect, we'are done
        if there exactly one island, othwerwise its disconnected, we need to make it disconnected
        we wont need more than two days to disconnect the grid? wtf
        
        if disconnected, meaning there isn't exaclty one island return 0
        return 1 if chaning a single land to water disconnectes the island
        othewise return 2
        '''
        rows, cols = len(grid), len(grid[0])

        def _count_islands():
            visited = set()
            count = 0
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] == 1 and (i, j) not in visited:
                        _explore_island(i, j, visited)
                        count += 1
            return count

        def _explore_island(i, j, visited):
            if (i < 0 or i >= rows or j < 0 or j >= cols or grid[i][j] == 0 or (i, j) in visited):
                return
            visited.add((i, j))
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                _explore_island(i + di, j + dj, visited)

        # Check if already disconnected
        if _count_islands() != 1:
            return 0

        # Check if can be disconnected in 1 day
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    grid[i][j] = 0
                    if _count_islands() != 1:
                        return 1
                    grid[i][j] = 1

        # If can't be disconnected in 0 or 1 day, return 2
        return 2
    
#####################################
# 40. Combination Sum II (REVISITED)
# 13AUG24
#####################################
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        i need to sort them first
        similart to combindation sum I
        imagine examples like
        [1,1,1,1,1,1]
        2
        
        we need combinations to be unique, so sorting makes sure we dont pick the same partial combination
        if i group all the same numbers together, i dont want to repeat freqeunces of that number k in a combiation
        '''
        candidates.sort()
        results = []
        
        #print(candidates)
        def recurse(i,target,path):
            if target <= 0:
                if target == 0 and path:
                    results.append(path[:])
                
                return 
            
            for j in range(i,len(candidates)):
                if j > i and candidates[j] == candidates[j-1]:
                    #print('skip',i,j)
                    continue
                if candidates[j] > target:
                    break
                recurse(j+1, target - candidates[j], path + [candidates[j]])
        
        recurse(0,target,[])
        return results
    
#bottom up variant
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        '''
        filter out elements <= target
        '''
        candidates = [i for i in candidates if i <= target]

        candidates.sort()

        dp = [set() for _ in range(target+1)]
        dp[0].add(())

        for i, c in enumerate(candidates):
            
            #need to start backwrds from target
            for j in range(target - c, -1, -1):
                for tup in dp[j]:
                    dp[j+c].add(tuple(list(tup) + [c]))
        
        print(dp)
        return list(dp[target])

##############################################
# 351. Android Unlock Patterns (REVISITED)
# 13AUG24
##############################################
class Solution:
    
    def numberOfPatterns(self, m: int, n: int) -> int:
        '''
        grid is
        1 2 3
        4 5 6
        7 8 9
        there only an obsacle between some digit (u,v) if we move through another dot, it must have been previously pressed
        we can brute force all possible ways to make unlock pattern with key presses between m and n
        
        then we push push states using recursion
        '''
        obstacles = defaultdict()
        #for a number on the pad, look for numbers that need to be pressed at least on1 before pressing the next one
        dirrs = [[2,0,1,0],[-2,0,-1,0],[0,2,0,1],[0,-2,0,-1],[-2,-2,-1,-1],[2,2,1,1],[-2,2,-1,1],[2,-2,1,-1]] #(x,y,x,y)
        for i in range(3):
            for j in range(3):
                for di,dj,dk,dl in dirrs:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii <= 2 and 0 <= jj <= 2:
                        u = i*3 + j
                        v = ii*3 + jj
                        skip = (i + dk)*3 + (j + dl)
                        obstacles[(u+1,v+1)] = skip+1
        ans = [0]
        for i in range(1,10):
            seen = set()
            self.count_ways(i,obstacles,ans,1,seen,m,n)
        
        return ans[0]
        
    
    def count_ways(self,num,graph,ans,count,seen,m,n):
        if m <= count <= n:
            ans[0] += 1
        if count == n:
            return
        seen.add(num)
        for neigh in range(1,10):
            if neigh not in seen:
                if (num,neigh) in graph and graph[(num,neigh)] not in seen:
                    continue
                self.count_ways(neigh,graph,ans,count+1,seen,m,n)
                
        seen.remove(num)
        
##########################################
# 719. Find K-th Smallest Pair Distance
# 14AUG24
###########################################
#almost had it!
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        '''
        need the kth smallest distance among all pairs, where (0 <= i < j < len(nums))
        binary search on answer, so we need a linear time funcino that checks how many pairs (i,j) 
        are <= x
        if pair count > k, x was too big, other wise x is too small
        if we sort sums, then the largest distance would be with pair (i,n-1)
        say we are at some pair (i,j) with dist == Y, 
        if Y <= X, then the number of pairs <= X is (j - i), so move left
        if Y > X, we cant count this pair and the distance is too big, but moving the left pointer up 1 and right pointer down 1
        would both decrease the distance
        [1,2,3,4,5]
        '''
        nums.sort()
        #neew count_pairs_leq == k, but we want the smallest k, so the left most one
        left = nums[1] - nums[0]
        right = nums[-1] - nums[0]
        
        while left < right:
            mid = left + (right - left) // 2
            count = self.count_pairs_leq(nums,mid)
            if count < k:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    #count pairs <= x
    def count_pairs_leq(self, nums,x):
        left = 0
        right = len(nums) - 1
        pairs = 0
        while left < right:
            dist = nums[right] - nums[left]
            if dist <= x:
                pairs += (right - left)
                left += 1
            else:
                right -= 1
        
        return pairs
    
#yesssss
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        '''
        need the kth smallest distance among all pairs, where (0 <= i < j < len(nums))
        binary search on answer, so we need a linear time funcino that checks how many pairs (i,j) 
        are <= x
        if pair count > k, x was too big, other wise x is too small
        if we sort sums, then the largest distance would be with pair (i,n-1)
        say we are at some pair (i,j) with dist == Y, 
        if Y <= X, then the number of pairs <= X is (j - i), so move left
        if Y > X, we cant count this pair and the distance is too big, but moving the left pointer up 1 and right pointer down 1
        would both decrease the distance
        [1,2,3,4,5]
        
        we need to use sliding window, anchor the right, and move left when too big
        '''
        nums.sort()
        #neew count_pairs_leq == k, but we want the smallest k, so the left most one
        left = 0
        right = nums[-1] - nums[0]
        ans = -1 #save ans and look for a better onw
        
        while left < right:
            mid = left + (right - left) // 2
            count = self.count_pairs_leq(nums,mid)
            if count == k:
                ans = mid
            if count < k:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    #count pairs <= x
    def count_pairs_leq(self, nums,x):
        left = 0
        pairs = 0
        for right in range(len(nums)):
            while nums[right] - nums[left] > x:
                left += 1
            #other wise count up all pairs anchored with right
            pairs += (right - left)
        
        return pairs

#brute force just for fun
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        '''
        for brute force, we can just do bucket sort,
        put pair distance in buckets,
        then find the kth smallest, after decremtning the counts in the buckets
        '''
        max_distance = max(nums)
        distance_counts = [0]*(max_distance + 1)
        N = len(nums)
        for i in range(N):
            for j in range(i+1,N):
                dist = abs(nums[i] - nums[j])
                distance_counts[dist] += 1
        
        for d in range(max_distance + 1):
            k -= distance_counts[d]
            if k <= 0:
                return d
        
        return -1