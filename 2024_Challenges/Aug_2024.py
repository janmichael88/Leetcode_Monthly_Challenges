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

#brute force
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        brute force with set,
        intution, start with the longest length substring, and keep looking for a repat
        if it repeats we are done, starting from the largest one, and going down means we will always find the largest one possible
        '''
        seen = set()
        max_length = len(s) - 1
        
        while max_length > 0:
            seen.clear()
            for start in range(len(s) - max_length + 1):
                substring = s[start:start+max_length]
                if substring in seen:
                    return max_length
                seen.add(substring)
            
            max_length -= 1
        
        return 0
    
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        suffix array with sorting, when all sufixes are sorted, common prefixes will be adjacent to each other
        then we can just scan the sorted suffixes
        then its just finding the longest common prefix
        verify this!
        if two suffxies share a common prefix, this prefix must appear more than one time!
        so the problem then becomes, given the suffixes, find the longest common prefix
        '''
        N = len(s)
        suffixes = []
        for i in range(N):
            suffixes.append(s[i:])
        
        suffixes.sort()
        #lcp
        def commonPref(str1,str2):
            M = len(str1)
            N = len(str2)
            for i in range(min(M,N)):
                #it was match up until i, so prefix up to i, non-inclusive
                if str1[i] != str2[i]:
                    return str1[:i]

            return str1[:min(M,N)]
        
        ans = 0
        for i in range(len(suffixes)):
            for j in range(i+1,len(suffixes)):
                common = commonPref(suffixes[i],suffixes[j])
                ans = max(ans,len(common))
        
        return ans

#N*N*lnN
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        suffix array with sorting, when all sufixes are sorted, common prefixes will be adjacent to each other
        then we can just scan the sorted suffixes
        then its just finding the longest common prefix
        verify this!
        if two suffxies share a common prefix, this prefix must appear more than one time!
        so the problem then becomes, given the suffixes, find the longest common prefix
        '''
        N = len(s)
        suffixes = []
        for i in range(N):
            suffixes.append(s[i:])
        
        suffixes.sort()
        max_length = 0
        # Compare adjacent suffixes to find the longest common prefix
        for i in range(1, len(suffixes)):
            j = 0
            # Compare characters one by one until
            # they differ or end of one suffix is reached
            while (j < min(len(suffixes[i]), len(suffixes[i - 1])) and suffixes[i][j] == suffixes[i - 1][j]):
                j += 1
            # Update max_length with the length of the common prefix
            max_length = max(max_length, j)
        return max_length
    
#binary search
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        we can also do binary search
        our search space is the length of all possible subtrings
        our function is to check a repeating substring of some length k
        '''
        left = 1
        right = len(s) - 1
        ans = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            if self.repeating(s,mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans
    
    def repeating(self,s,k):
        seen = set()
        for i in range(len(s) - k + 1):
            substring = s[i:i+k]
            if substring in seen:
                return True
            seen.add(substring)
        
        return False
    
#dp



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
    
##############################################################
# 1566. Detect Pattern of Length M Repeated K or More Times
# 15AUG24
##############################################################
class Solution:
    def containsPattern(self, arr: List[int], m: int, k: int) -> bool:
        '''
        pattern needs to repeat itselft
        '''
        N = len(arr)
        for i in range(N-m+1):
            pattern = arr[i:i+m]
            if pattern*k == arr[i:i+m*k]:
                return True
        
        return False

######################################################
# 624. Maximum Distance in Arrays (REVISITED)
# 16AUG24
######################################################
#dammit
class Solution:
    def maxDistance(self, arrays: List[List[int]]) -> int:
        '''
        we need to pick two integers from two different arrays
        i can put everything in array, sort, then use two pointers, and keep walking until we get two differnt integers from
        two different arrays
        '''
        nums = []
        for i,arr in enumerate(arrays):
            for num in arr:
                nums.append((num,i))
        
        nums.sort()
        left = 0
        right = len(nums) - 1
        while left < right and nums[left][1] == nums[right][1]:
            #go in direction of maximum gap
            if left + 1 < right and right - 1 > left:
                if abs(nums[left+1][0] - nums[right][0]) > abs(nums[right-1][0] - nums[left][0]):
                    left += 1
                else:
                    right -= 1
            elif left + 1 < right:
                left += 1
            else:
                right -= 1
        
        return abs(nums[left][0] - nums[right][0])
    
class Solution:
    def maxDistance(self, arrays: List[List[int]]) -> int:
        '''
        they are sorted, so we just need to find the smallest min and the largest max, but both need to come 
        from different arrays
        '''
        smallest_min = float('inf')
        largest_max = float('-inf')
        ans = 0
        
        for arr in arrays:
            #update before picking the min and max!
            #ie we didn't, the two integers might have been from the same array
            ans = max(ans, largest_max - arr[0], arr[-1] - smallest_min)
            smallest_min = min(smallest_min,arr[0])
            largest_max = max(largest_max,arr[-1])
            print(ans,smallest_min,largest_max)
        
        return ans
    
##########################################
# 1937. Maximum Number of Points with Cost
# 17AUG24
##########################################
#unoptimized DP, TLE
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        '''
        i can only pick a cell in each row, so we can only go down
        howevery we loost points if we pick a cell too far from the cell we are currently at
        if we are at (i,j) gain points[i][j]
        and we move to (i+1,j+1), we lose abs(j - (j+1))
        if we pick (i+1, j+2) we lose (abs(j - (j+2)))
        so we want to pick the largest in the row below, but not too far from the current (i,j)
        dp would take N*N time
        '''
        rows = len(points)
        cols = len(points[0])
        
        memo = {}
        
        def dp(i,j):
            #base case
            if i >= rows:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            
            ans = 0
            for k in range(cols):
                #get point at next
                curr_points = points[i][k] - abs(j-k) + dp(i+1,k)
                ans = max(ans,curr_points)
            
            memo[(i,j)] = ans
            return ans
        
        ans = 0
        for j in range(cols):
            #add in first point value for first row, the call dp
            ans = max(ans,points[0][j] + dp(1,j))
        
        return ans
                
#need linear time, input is too big for (two states)
#need lefts array and rights array
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        '''
        the problem with the dp approach is that we tried all cells below row i
        applied penalty, and took the max, recall there aren't many ways to optimize a top down
        recursive dp solution, and most of the time these optimizations come from
        tabulation
        idea is to keep left max and right max
        precompute these, and take the maximum of the two choices
        if we compute left to right and right to left, the penality only goes up by 1
        '''
        rows = len(points)
        cols = len(points[0])
        
        dp = [[0]*cols for _ in range(rows)]
        #fill first row
        for c in range(cols):
            dp[0][c] = points[0][c]
        
        for r in range(1,rows):
            left_max = [0]*cols
            right_max = [0]*cols
            
            #find max going left and apply penality
            left_max[0] = dp[r-1][0]
            for c in range(1,cols):
                left_max[c] = max(left_max[c-1] - 1, dp[r-1][c])
            
            right_max[-1] = dp[r-1][-1]
            for c in range(cols-2,-1,-1):
                right_max[c] = max(right_max[c+1] - 1, dp[r-1][c])
            
            #fill in current row ans
            for col in range(cols):
                dp[r][col] = max(left_max[col],right_max[col]) + points[r][col]
        
        return max(dp[-1])

#####################################
# 264. Ugly Number II (REVISTED)
# 18AUG24
####################################
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        '''
        try merging three sorted lists
        if we have the kth ugly, the next ugly will be min(L1*2,L2*2,L3*5)
        generate the sequence
        note 1, can be exrpessed as an ugly, because we can have 2**0 * 3**0 * 5**0
        we need to keep pointers to the last three ugly numbers
        '''
        uglies = [1]
        i2,i3,i5 = 0,0,0
        while len(uglies) < n:
            ugly2 = uglies[i2]*2
            ugly3 = uglies[i3]*3
            ugly5 = uglies[i5]*5
            
            #need to move all pointers!
            next_ugly = min(ugly2,ugly3,ugly5)
            #promote pointers
            if next_ugly == ugly2:
                i2 += 1
            if next_ugly == ugly3:
                i3 += 1
            if next_ugly == ugly5:
                i5 += 1
            uglies.append(next_ugly)
        
        print(uglies)
        return uglies[n-1]
            
#bottom up dp
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        '''
        idea is to keep pointers for the last three ugly numbers
        '''
        uglies = [0]*(n+1)
        uglies[1] = 1
        first = 1
        second = 1
        third = 1
        
        for i in range(2,n+1):
            next_ugly = min(uglies[first]*2,uglies[second]*3,uglies[third]*5)
            uglies[i] = next_ugly
            #move pointers
            if next_ugly == uglies[first]*2:
                first += 1
            if next_ugly == uglies[second]*3:
                second += 1
            if next_ugly == uglies[third]*5:
                third += 1
        
        return uglies[n]
    
#using sorted or minheap
from sortedcontainers import SortedList
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        '''
        brute force, 
        just keep track of all possible uglies and always pick the minimum
        used sortedList to speed up
        '''
        uglies = SortedList([])
        seen = set()
        uglies.add(1)
        seen.add(1)
        
        for _ in range(n-1):
            curr_ugly = uglies.pop(0)
            seen.remove(curr_ugly)
            for mult in [2,3,5]:
                next_ugly = curr_ugly*mult
                if next_ugly not in seen:
                    uglies.add(next_ugly)
                    seen.add(next_ugly)
        
        return uglies[0]
    
###########################################
# 1014. Best Sightseeing Pair
# 18AUG24
###########################################
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        '''
        we want to maximize
        values[i] + values[j] + i - j, for i < j
        rewrite as (values[i] + i) + (values[j] - j)
        for some value i, we need the max of values[j] - j to the left
        '''
        n = len(values)
        max_js = [0]*n
        for j in range(n-1,-1,-1):
            if j == n - 1:
                max_js[j] = values[j] - j
            else:
                max_js[j] = max(max_js[j+1],values[j] - j)
        
        #find max i
        max_is = [0]*n
        for i in range(n):
            if i == 0:
                max_is[i] = values[i] + i
            else:
                max_is[i] = max(max_is[i-1], values[i] + i)
        
        ans = 0
        for i in range(n-1):
            ans = max(ans, max_is[i] + max_js[i+1])
            
        return ans

#we can reduce to two pass
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        '''
        we can reduce to two pass, 
        on the second pass find the maximum of values[i] + i and compare with the largest values[j] - j
        '''
        n = len(values)
        max_js = [0]*n
        for j in range(n-1,-1,-1):
            if j == n - 1:
                max_js[j] = values[j] - j
            else:
                max_js[j] = max(max_js[j+1],values[j] - j)
        
        #find max i
        ans = 0
        max_is = 0
        for i in range(n-1):
            max_is = max(values[i] + i,max_is)
            ans = max(ans, max_is + max_js[i+1])
            
        return ans

#one pass
#maximize max_is, and when we discover a new j, we save it
class Solution:
    def maxScoreSightseeingPair(self, values: List[int]) -> int:
        '''
        we actually don't need to maximize js if we go from left to right, 
        as long as we maximize i, we can discover a new j, that it the right of the current i
        so we only maximize max_is, and maximize ans for the current i
        '''
        n = len(values)
        ans = 0
        max_is = values[0] + 0  # values[i] + i, initialized at i = 0

        for i in range(1, n):
            curr_js = values[i] - i
            ans = max(ans, max_is + curr_js)
            max_is = max(max_is, values[i] + i)

        return ans

#######################################
# 650. 2 Keys Keyboard (REVISTED)
# 19AUG24
#######################################
#dp
class Solution:
    def minSteps(self, n: int) -> int:
        '''
        dp states
            (chars_on_screen, chars_in_buffer)
        
        when we have n chars_on_screen we are done
        copy operation = put chars on screen into buffer and add 1
        past operation, put chars in buffer on screen
        min of copy and paste
        copy needs to be followed by a paste
        make sure to only do actions when we can
        '''
        memo = {}
        
        def dp(screen_chars,buffer_chars):
            if screen_chars == n:
                return 0
            if screen_chars > n or buffer_chars > n:
                return float('inf')
            if (screen_chars,buffer_chars) in memo:
                return memo[(screen_chars,buffer_chars)]
            
            #it doubles, and we have screen_chars in buffer
            copy_paste = 2 + dp(screen_chars*2,screen_chars)
            #paste operation, we just move screen to bufer
            paste = 0
            if buffer_chars > 0:
                paste = 1 + dp(screen_chars + buffer_chars,buffer_chars)
            if paste:
                ans = min(copy_paste,paste)
            else:
                ans = copy_paste
            memo[(screen_chars,buffer_chars)] = ans
            return ans
            
        return dp(1,0)
    
#bottom up
class Solution:
    def minSteps(self, n: int) -> int:
        '''
        bottom up
        '''
        dp = [[0]*(n+1) for _ in range(n+1)]
        
        for screen_chars in range(n-1,-1,-1):
            #order doenst matter for buffer chars
            for buffer_chars in range(screen_chars-1,-1,-1):
                #copy paste
                if screen_chars*2 > n:
                    copy_paste = float('inf')
                else:
                    copy_paste = 2 + dp[screen_chars*2][screen_chars]
                
                paste = 0
                if buffer_chars > 0:
                    if screen_chars + buffer_chars > n:
                        paste = 0
                    else:
                        paste = 1 + dp[screen_chars + buffer_chars][buffer_chars]
                
                if paste:
                    ans = min(paste,copy_paste)
                else:
                    ans = copy_paste
                
                dp[screen_chars][buffer_chars] = ans
        
        return dp[1][0]

#another dp solution

class Solution:
    def minSteps(self, n: int) -> int:
        '''
        for this, the assumption is that we start with 1 char in screen on one alredy in buffer
        '''
        if n==1:
            return 0
        
        memo = {}
        def helper(el, cp):
            if el>n or cp>n:
                return float('inf')            
            if el==n:
                return 0
            if (el,cp) in memo:
                return memo[(el,cp)]
            only_paste=1+helper(el+cp,cp)
            copy_paste=2+helper(el*2,el)
            ans = min(only_paste,copy_paste)
            memo[(el,cp)] = ans
            return ans
            
            
        return 1+helper(1,1)

#top down
class Solution:
    def minSteps(self, n: int) -> int:
        '''
        we can do 1 state dp
        let dp(i) be the mininum number of moves to get i A's
        now say have 6As
        we could have gotten to here by
        pasting 1A 5 times
        pasting 2A 4 times
        pasting 3A 1 times
        it is more advantages to just paste whatver is in buffer
        more so, if we have i As, the previous As on the screen must have been factor of i
        one possible way to make i As is to the use the copy all operation for j A's when j is a factor of i
        we can the paste the j A's (i-j) // j times
        '''
        memo = {}
        
        def dp(i):
            if i == 1:
                return 0
            if i in memo:
                return memo[i]
            ans = float('inf')
            for j in range(1,i):
                if i % j == 0:
                    ans = min(ans,dp(j) + i // j)
            
            memo[i] = ans
            return ans
        
        
        return dp(n)
    
#notice that for some count of i A's we must have come from a factor of i


##################################################################
# 2507. Smallest Value After Replacing With Sum of Prime Factors
# 19AUG24
###################################################################
class Solution:
    def smallestValue(self, n: int) -> int:
        '''
        simulate, good review on prime factorization
        start from 2 and keep dividing as long as its a factor
        
        '''
        ans = float('inf')
        
        while True:
            sum_primes = self.sumprimefactors(n)
            ans = min(ans,sum_primes)
            if sum_primes == n:
                break
            n = sum_primes
        return ans

        return 0
    def sumprimefactors(self,n):
        ans = 0
        d = 2
        while n > 1:
            while n % d == 0:
                ans += d
                n = n // d
            
            d += 1
        
        return ans
    
##############################################
# 3189. Minimum Moves to Get a Peaceful Board
# 20AUG24
##############################################
class Solution:
    def minMoves(self, rooks: List[List[int]]) -> int:
        '''
        meed min moves to have board state where there is exactly one rook
        '''
        #sort rooks on rows, then assign
        n = len(rooks)
        rooks.sort()
        moves = 0
        
        for i in range(n):
            moves += abs(rooks[i][0] - i)
        
        #now sort on columns
        rooks.sort(key = lambda x: x[1])
        
        for i in range(n):
            moves += abs(rooks[i][1] - i)
        
        return moves
        
#counting sort
class Solution:
    def minMoves(self, rooks: List[List[int]]) -> int:
        '''
        instead of sorting by by rook position, we count up the rooks at each position
        we can use counting sort
        '''
        n = len(rooks)
        rows = Counter([r for r,c in rooks])
        cols = Counter([c for r,c in rooks])
        
        moves = 0
        
        for i,r in enumerate(self.countingSort(rows,n)):
            moves += abs(r - i)
        
        for i,c in enumerate(self.countingSort(cols,n)):
            moves += abs(c - i)
        
        return moves
    
    def countingSort(self,counts,n):
        order = []
        for i in range(n):
            if i in counts:
                for _ in range(counts[i]):
                    order.append(i)
        
        return order
    
class Solution:
    def minMoves(self, rooks: List[List[int]]) -> int:
        '''
        we calculate the number of rooks at each row and col seperately
        recall there can only be one rook in reach row and col
        the excess needs to be moved
        so the cost of moving the excess rooks is just count_rooks at this (row or col) - 1
        we add this up for both row and cols
        maintain difference and accumulate difference
        '''
        n = len(rooks)
        moves = 0
        rows = [0]*n
        cols = [0]*n
        
        for r,c in rooks:
            rows[r] += 1
            cols[c] += 1
            
        moves
        excess_rooks_rows = 0
        excess_rooks_cols = 0
        for i in range(n):
            excess_rooks_rows += rows[i] - 1
            excess_rooks_cols += cols[i] - 1
            moves += abs(excess_rooks_rows) + abs(excess_rooks_cols)
        
        return moves
    
#######################################
# 1140. Stone Game II (REVISITED)
# 20AUG24
#######################################
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        '''
        alice starts first, M starts as 1
        each player can take all the stones in the range(1,2M + 1)
        then set M to be max(M,X)
        states are (i,m)
        return scores for both alice and bob, but swap after each call
        i can also keep track of alice turn or bob turn
        if its alices turn, we need to maximize, if its bob's turn we need to minimize
        player_turn (0 for alice, 1 for bob)
        '''
        n = len(piles)
        memo = {}
        
        def dp(i,m,player_turn):
            if i >= n:
                return 0
            if (i,m,player_turn) in memo:
                return memo[(i,m,player_turn)]
            #if its alice turn we are maximizing
            if player_turn == 0:
                max_score = float('-inf')
                stone_sum = 0
                for j in range(1,min(2*m,n-i)+1):
                    stone_sum += piles[i+j-1]
                    next_m = max(m,j)
                    max_score = max(max_score, stone_sum + dp(i+j,next_m,1))
                memo[(i,m,player_turn)] = max_score
                return max_score
            #on bob's turn we minimize, and we dont take any stones
            elif player_turn == 1:
                min_score = float('inf')
                for j in range(1,min(2*m,n-i)+1):
                    next_m = max(m,j)
                    min_score = min(min_score, dp(i+j,next_m,0))
                
                memo[(i,m,player_turn)] = min_score
                return min_score
            
        return dp(0,1,0)
                
#check this solution too
#https://leetcode.com/problems/stone-game-ii/discuss/793881/python-DP-Thought-process-explained       
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        suffix_sum = self._suffix_sum(piles)
        '''
        idea is to maximize the difference in piles
        '''
        @lru_cache(None)
        def dfs(pile: int, M: int, turn: bool) -> Tuple[int, int]:
            # turn: true - alex, false - lee
            sum_alex, sum_lee = suffix_sum[pile], suffix_sum[pile]

            for next_pile in range(pile + 1, min(pile + 2 * M + 1, len(piles) + 1)):
                sum_alex_next, sum_lee_next = dfs(
                    next_pile, max(M, next_pile - pile), not turn
                )
                range_sum = suffix_sum[pile] - suffix_sum[next_pile]

                if turn:
                    if sum_lee_next < sum_lee:
                        sum_alex = sum_alex_next + range_sum
                        sum_lee = sum_lee_next
                else:
                    if sum_alex_next < sum_alex:
                        sum_alex = sum_alex_next
                        sum_lee = sum_lee_next + range_sum

            return sum_alex, sum_lee

        return dfs(0, 1, True)[0]

#######################
# 1686. Stone Game VI
# 20AUG24
########################
class Solution:
    def stoneGameVI(self, aliceValues: List[int], bobValues: List[int]) -> int:
        '''
        both alice and bob value stones differently
        and they can take any stone in the array, not just ends or beginning
        have alice choose the biggest stone, but also the biggest stone for bob
        '''
        new_stones = [[a + b, a,b] for (a,b) in zip(aliceValues,bobValues)]
        new_stones.sort(key = lambda x: -x[0])
        alice_score = 0
        bob_score = 0
        n = len(new_stones)
        for i in range(n):
            if i % 2 == 0:
                alice_score += new_stones[i][1]
            else:
                bob_score += new_stones[i][2]
        
        if alice_score > bob_score:
            return 1
        elif alice_score < bob_score:
            return -1
        return 0

#############################################
# 1920. Build Array from Permutation
# 22AUG24
#############################################
#fuck yeah
class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:
        '''
        i can save space using bitwise operators
        store the nums[i] and nums[nums[i]] as integer left part nums[i] and right part is nums[nums[i]]
        store origina number in first ten bits, 
        nums only go up to 1000
        2**10 covers
        increment each value by one first and shift to the left 20, then we have space for its pair
        
        '''
        mask = (1 << 11) - 1
        n = len(nums)
        for i in range(n):
            curr_num = nums[i] + 1
            curr_num = curr_num << 20
            nums[i] = curr_num
        
        #retravesre again and get pair
        for i in range(n):
            curr_num = nums[i]
            #get actual num
            actual_num = (curr_num >> 20) - 1
            #find its pair
            pair = nums[actual_num]
            actual_pair = (pair >> 20 ) - 1
            #put acutal pair in the first 10 spots of curr_num, but + 1
            curr_num |= (actual_pair + 1)
            nums[i] = curr_num
        
        for i in range(n):
            curr_num = nums[i]
            pair = (curr_num & mask) - 1
            nums[i] = pair
        
        return nums
    
#need to right number as (a,b) -> a = b*q + r
class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:
        '''
        intution is that we need to encode both the new value and the old value into the same value
        if we left a = q*b + r, we put the old value into r and the new value into b
        and since there are only number froms 0 to n-1, we use q as n
        we can retreive the old value by taking % q and new value as // q
        rather new num will be a multiple of q
        and old num is % q
        '''
        q = len(nums)
        
        for i,num in enumerate(nums):
            r = nums[i]
            b = nums[nums[i]] % q
            a = b*q + r
            nums[i] = a
        
        for i,num in enumerate(nums):
            #print(num % q,num//q)
            nums[i] = num // q
        
        return nums
    
###################################
# 664. Strange Printer (REVISITED)
# 22AUG24
###################################
class Solution:
    def strangePrinter(self, s: str) -> int:
        '''
        the problem is that we can replace existing chars
        in one press we can print as many chars as we want to
        we can choose to place the next sequence at the end (in any length) or replace them at any position
        and at any length
        idea is to pring s in a few presses as possible, since we are allowed to 'delete' chars by overwriting
        lets examine way to split a string
        'cabad'
        let dp(i,j) be the minimum number of presses to get s[i:j+1]
        so we wan dp(0,len(s))
        then we can check for all k from i to j and minimize
            if s[i] == s[k], its part of the current press
        '''
        memo = {}
        n = len(s) 
        
        def dp(i,j):
            if i > j:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = 1 + dp(i+1,j) #advance and press
            for k in range(i+1,j+1):
                if s[i] == s[k]:
                    left = dp(i,k-1)
                    right = dp(k+1,j)
                    combine = left + right
                    ans = min(ans,combine)
            
            memo[(i,j)] = ans
            return ans
        
        return dp(0,n-1)
                    
        
#bottom up
class Solution:
    def strangePrinter(self, s: str) -> int:
        '''
        bottom up
        '''
        n = len(s)
    
        # Initialize the DP table
        dp = [[0] * n for _ in range(n)]

        # Fill the DP table
        for length in range(1, n + 1):  # Length of the substring
            for i in range(n - length + 1):
                j = i + length - 1

                # Base case for substrings of length 1
                dp[i][j] = 1 if length == 1 else 1 + dp[i + 1][j]

                # Check for all possible split points
                for k in range(i + 1, j + 1):
                    if s[i] == s[k]:
                        left = dp[i][k - 1]
                        right = dp[k + 1][j] if k + 1 <= j else 0
                        dp[i][j] = min(dp[i][j], left + right)

        # The answer for the whole string is stored in dp[0][n-1]
        return dp[0][n-1]

####################################################
# 592. Fraction Addition and Subtraction (REVISITED)
# 23AUG24
####################################################
#using regex
import re
class Solution:
    def fractionAddition(self, expression: str) -> str:
        '''
        get fractions
        then for each fraction split into numerators and denoms
        lcm, of denoms, regardless of sign
        then reduce, using gcd
        '''
        pattern = r'[+-]?\d+/\d+'
        fractions = re.findall(pattern,expression)
        nums = []
        denoms = []
        lcm = 1
        for f in fractions:
            num,denom = f.split("/")
            nums.append(int(num))
            denoms.append(int(denom))
            lcm = (lcm*abs(int(denom))) / self.gcd(lcm,abs(int(denom)))
        
        ans_num = 0
        for n,d in zip(nums,denoms):
            ans_num += (lcm/d)*n
        
        sign = -1 if ans_num < 0 else 1
        #print(ans_num,lcm)
        #find gcd
        GCD = self.gcd(ans_num,lcm)
        final_num = int(abs(ans_num/GCD))
        final_denom = int(abs(lcm/GCD))
        
        ans = str(final_num)+'/'+str(final_denom)
        
        if sign == -1:
            ans = '-'+ans
        return ans

#process and summation on fly
import re
class Solution:
    def fractionAddition(self, expression: str) -> str:
        '''
        we dont need to grab all the nums and denoms in a seperate pass
        we can reduce and add
        when addint tow fraction, use product of two denoms as common denom
        '''
        pattern = r'[+-]?\d+/\d+'
        fractions = re.findall(pattern,expression)
        nums = []
        denoms = []
        num = 0
        denom = 1
        for f in fractions:
            f = f.split("/")
            curr_num,curr_denom = int(f[0]),int(f[1])
            #update num
            common_denom = denom*curr_denom
            num = (common_denom//denom)*num + (common_denom//curr_denom)*curr_num
            denom = common_denom
        
        gcd = abs(self.gcd(num,denom))
        #reduce
        num = num // gcd
        denom = denom // gcd
        return f"{num}/{denom}"
            

    
    def gcd(self,a,b):
        if a == 0:
            return b
        return self.gcd(b % a,a)
        
######################################
# 1837. Sum of Digits in Base K
# 23AUG24
######################################
class Solution:
    def sumBase(self, n: int, k: int) -> int:
        
        ans = 0
        while n:
            ans += n % k
            n = n // k
        
        return ans
    
#################
# 853. Car Fleet
# 23AUG24
##################
#nice try :(
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        '''
        they only count as a fleet if car catches up to another car at target mile
        ideas, sorting, left/right max_array or min_array
        if i car is at the end it has no car to catch up to
        if we are at some car i its speed will be limited by i+1
        sort and monostack, i dont think i need a stack, just keep track of cars in a fleet
        and distance fleet is formed
        and see if the current one can catch up
        if it can catch up before target its part of the fleet
        all position values are unique!
        if i have two cars that can catch up, at what distance will the first car catch up?
            i dont care about time, just the distance
        say we have two cars with (speed,pos) at (2,1) and (1,5) -> remmber the second car hasn't started moving yet
        lcm of (pos_left + speed_left) and (pos_right,speed_right)
        lcm(a,b) = a*b / gcd(a,b)
        6 9 12 15
        8 10
        '''
        cars = [(p,s) for p,s in zip(position,speed)]
        cars.sort()

        fleets = []
        count = 0
        for p,s in cars:
            #store as (pos,speed)
            #and can catch up

            if fleets and fleets[-1][1] > s:
                pos_left,speed_left = fleets[-1]
                dist_meet = ((pos_left + speed_left)*(p+s)) // self.gcd(pos_left + speed_left,p + s)
                entry = (dist_meet,min(s,fleets[-1][1]))
                #fleets.pop()
                fleets.append(entry)
            else:
                fleets.append((p,s))
        print(fleets)
        for p,s in fleets:
            if p <= target:
                count += 1
        return count
    
    def gcd(self,a,b):
        if a == 0:
            return b
        return self.gcd(b % a,a)
                
                
class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        '''
        we can actually compute the time for cars
        if a car can catch up (i.e get to it at faster car it becomes a fleet)
        need to go in reverse and see if a previous car can catch up
        could also do w/o stack and just record last time
        '''
        cars = [(p,s) for p,s in zip(position,speed)]
        cars.sort(reverse = True)
        stack = []
        
        for pos,velocity in cars:
            #get dist from target
            dist = target - pos
            #find time
            time = dist / velocity
            if not stack:
                stack.append(time)
            elif time > stack[-1]:
                stack.append(time)
        
        return len(stack)
    
####################################
# 564. Find the Closest Palindrome
# 24AUG24
####################################
class Solution:
    def nearestPalindromic(self, n: str) -> str:
        '''
        need to intellgiently build the closes palindrome,
        if there is a tie, return the closest one
        examine numbers like 1234,9999,1000
        for 1234, it would be 1221
        for 9999, it would be 10001
        for 1000, it would be [1001 or 999] but we want the smaller one
        we can consider both left and right, and take the smaller
        what avut 56789? -> 56765
        5678 -> 5665
        200 -> 202
        299 -> 292
        in the general case is just the left half reversed to the right side
        12932 -> 12921
        99800 -> 99899 or 99799
        12120 -> 12121
        10001 -> 9999
        use this code to compare and see patterns
        
        '''
        #ez way to to just check up and down
        down = int(n) - 1
        while str(down) != str(down)[::-1]:
            down -= 1
        
        up = int(n) + 1
        while str(up) != str(up)[::-1]:
            up += 1
        
        print(down,up)
        return ""
        
#almost, aye yai yai
class Solution:
    def nearestPalindromic(self, n: str) -> str:
        '''
        need to intellgiently build the closes palindrome,
        if there is a tie, return the closest one
        examine numbers like 1234,9999,1000
        for 1234, it would be 1221
        for 9999, it would be 10001
        for 1000, it would be [1001 or 999] but we want the smaller one
        we can consider both left and right, and take the smaller
        what avut 56789? -> 56765
        5678 -> 5665
        200 -> 202
        299 -> 292
        in the general case is just the left half reversed to the right side
        12932 -> 12921
        99800 -> 99899 or 99799
        12120 -> 12121
        10001 -> 9999
        use this code to compare and see patterns
        can probably get smaller palindrome and larger palindrome, compare dist from n and return smaller
        get the left most half, then add 1 it or subtract 1 from it or mirror it
        but hwen also check 9999 and 10001
        and is just the closes of these possibilites
        '''
        k = len(n)
        #check 
        if k % 2 == 0:
            left = n[:k // 2]
        else:
            left = n[:(k//2) + 1]
        
        #gather possibilties
        candidates = []
        if len(left) % 2 == 1:
            #dont add
            no_add = int(left)
            no_add = str(no_add) + str(no_add)[::-1][1:]
            #first add1
            add1 = int(left) + 1
            add1 = str(add1) + str(add1)[::-1][1:]
            #sub 1
            sub1 = int(left) - 1
            sub1 = str(sub1) + str(sub1)[::-1][1:]
            candidates.extend([no_add,add1,sub1])

        elif len(left) % 2 == 0:
            #dont add
            no_add = int(left)
            no_add = str(no_add) + str(no_add)[::-1]
            #first add1
            add1 = int(left) + 1
            add1 = str(add1) + str(add1)[::-1]
            #sub 1
            sub1 = int(left) - 1
            sub1 = str(sub1) + str(sub1)[::-1]
            candidates.extend([no_add,add1,sub1])
        
        #add in ones of the form 10000, and 99999
        nines = str(10**(k-1) - 1)
        tens = str(10**k + 1)
        candidates.extend([nines,tens])
        print(candidates)
        ans = -1
        min_dist = float('inf')
        for c in candidates:
            if c == n:
                continue
            curr_dist = abs(int(n) - int(c))
            if curr_dist < min_dist:
                min_dist = curr_dist
                ans = int(c)
            elif curr_dist == min_dist:
                ans = min(ans, int(c))
        
        return str(ans)
                
        
class Solution:
    def nearestPalindromic(self, n: str) -> str:
        len_n = len(n)
        i = len_n // 2 - 1 if len_n % 2 == 0 else len_n // 2
        first_half = int(n[: i + 1])
        """
        Generate possible palindromic candidates:
        1. Create a palindrome by mirroring the first half.
        2. Create a palindrome by mirroring the first half incremented by 1.
        3. Create a palindrome by mirroring the first half decremented by 1.
        4. Handle edge cases by considering palindromes of the form 999... 
           and 100...001 (smallest and largest n-digit palindromes).
        """
        possibilities = []
        possibilities.append(
            self.half_to_palindrome(first_half, len_n % 2 == 0)
        )
        possibilities.append(
            self.half_to_palindrome(first_half + 1, len_n % 2 == 0)
        )
        possibilities.append(
            self.half_to_palindrome(first_half - 1, len_n % 2 == 0)
        )
        possibilities.append(10 ** (len_n - 1) - 1)
        possibilities.append(10**len_n + 1)

        diff = float("inf")
        res = 0
        nl = int(n)
        for cand in possibilities:
            if cand == nl:
                continue
            if abs(cand - nl) < diff:
                diff = abs(cand - nl)
                res = cand
            elif abs(cand - nl) == diff:
                res = min(res, cand)
        return str(res)

    def half_to_palindrome(self, left: int, even: bool) -> int:
        res = left
        if not even:
            left = left // 10
        while left > 0:
            res = res * 10 + left % 10
            left //= 10
        return res

##########################################
# 919. Complete Binary Tree Inserter
# 25AUG24
##########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class CBTInserter:

    def __init__(self, root: Optional[TreeNode]):
        '''
        if we were doing bfs, we just build level by level
        when we insert, we pop from the queue, and if empty, prepare the next level
        the issue is that we alwasy need to keep reference to the root
        we intialize with an existing root
        and we need to find the next available spot to insert
        what if we make is to that way the parent is at the front of the q
        ok so input will ALWAYS be complete if its complete, i can bfs level by level
        need to maintain two deques, one to process the tree, and the other to make connections
        '''
        self.q = deque([])
        self.root = root
        self.q.append(root)
        
        #i had this part!
        while self.q:
            curr = self.q[0]
            if not curr.left or not curr.right:
                if curr.left:
                    self.q.append(curr.left)
                break
            self.q.popleft()
            if curr.left:
                self.q.append(curr.left)
            if curr.right:
                self.q.append(curr.right)
         
    def insert(self, val: int) -> int:
        curr = self.q[0]
        newNode = TreeNode(val)
        self.q.append(newNode)
        if curr.left == None:
            curr.left = newNode
        else:
            self.q.popleft()
            curr.right = newNode
        
        return curr.val
            
    def get_root(self) -> Optional[TreeNode]:
        return self.root

# Your CBTInserter object will be instantiated and called as such:
# obj = CBTInserter(root)
# param_1 = obj.insert(val)
# param_2 = obj.get_root()

#another way
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class CBTInserter:
    '''
    idea is to maintain queue, where the first element in the queue is the node we wish to add to
    if parent node is complete (i.e has both children) we pop it out
    '''

    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        self.q = deque([root])
        while True:
            curr = self.q[0]
            if curr:
                if curr.left:
                    self.q.append(curr.left)
                    if curr.right:
                        self.q.append(curr.right)
                        #this node cannot by the one we wish to add too
                        self.q.popleft()
                    else:
                        break
                else:
                    #if there isn't a left, it means this node (curr) is the one we want to add to
                    break

    def insert(self, val: int) -> int:
        curr = self.q[0]
        new_node = TreeNode(val)
        if not curr.left:
            curr.left = new_node
            self.q.append(new_node)
        else:
            curr.right = new_node
            self.q.append(new_node)
            self.q.popleft()
        
        return curr.val

    def get_root(self) -> Optional[TreeNode]:
        return self.root


# Your CBTInserter object will be instantiated and called as such:
# obj = CBTInserter(root)
# param_1 = obj.insert(val)
# param_2 = obj.get_root()

################################################
# 590. N-ary Tree Postorder Traversal (REVISTED)
# 26AUG24
################################################
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        '''
        need to in flag variablt
        first encounter of a node is that we need to do it children first
        second encounter is when we process and get its value
        '''
        ans = []
        
        if not root:
            return []
        
        stack = [(root,False)]
        
        while stack:
            curr,visited = stack.pop()
            #if we have seen it, add to ans
            if visited:
                ans.append(curr.val)
            else:
                stack.append((curr,True))
                for child in reversed(curr.children):
                    stack.append((child,False))
        
        return ans
    
########################################
# 998. Maximum Binary Tree II
# 26APR24
#########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoMaxTree(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        '''
        similar to max binary tree i
        if i had the inital a array, i could just append b to it and rebuild the array
        '''
        a = self.dfs(root)
        #add b
        a.append(val)
        #build using max binary tree algo
        return self.build(a,0,len(a))
    
    #get a array
    def dfs(self,root):
        if not root:
            return []

        return  self.dfs(root.left) + [root.val] + self.dfs(root.right)
    
    def get_max(self,a,left,right):
        curr = left
        for i in range(left,right):
            if a[i] > a[curr]:
                curr = i
        
        return curr
    
    def build(self,a,left,right):
        if left >= right:
            return None
        max_idx = self.get_max(a,left,right)
        node = TreeNode(a[max_idx])
        node.left = self.build(a,left,max_idx)
        node.right = self.build(a,max_idx+1,right)
        return node

#################################################
# 1514. Path with Maximum Probability (REVISTED)
# 27AUG24
#################################################
#bfs, only add if it makes it bigger/smaller
import math
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        '''
        largest path in undirected graph is NP hard
        instead of multiplying to log probabilites, then do minimum after negating all costs
        a*b*c*d, can be rewriten as log(a) + log(b) * log(c) * log(d)
        can also do bfs and update when we ahve a new max
        retaking an edge repeatedly will only make the probability smaller
        '''
        graph = defaultdict(list)
        for i in range(len(edges)):
            u = edges[i][0]
            v = edges[i][1]
            weight = math.log(succProb[i],2)
            graph[u].append((v,weight))
            graph[v].append((u,weight))
        
        max_probs = [float('-inf')]*n
        max_probs[start] = 0
        q = deque([start])
        
        while q:
            curr = q.popleft()
            for neigh,neigh_prob in graph[curr]:
                next_prob = max_probs[curr] + neigh_prob
                if next_prob > max_probs[neigh]:
                    max_probs[neigh] = next_prob
                    q.append(neigh)
        
        if max_probs[end] == float('-inf'):
            return 0
        return 2**max_probs[end]
    

import math
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        '''
        for djikstrs version we need the least negative number
        since they are all negative anyway, we use max_heap
        now i wish there was a way to enforce the heap api in python to strictly max instead of negating
        becasue we need to least negative number
        '''
        graph = defaultdict(list)
        for i in range(len(edges)):
            u = edges[i][0]
            v = edges[i][1]
            weight = math.log(succProb[i],2)
            graph[u].append((v,weight))
            graph[v].append((u,weight))
        
        
        max_probs = [float('-inf')]*n #this is accumulated log probs
        max_probs[start] = 0.0
        max_heap = [(0.0, start)]
        seen = set()
        
        while max_heap:
            curr_log_prob,node = heapq.heappop(max_heap)
            curr_log_prob *= -1
            if max_probs[node] > curr_log_prob:
                continue
            seen.add(node)
            for neigh,neigh_prob in graph[node]:
                if neigh in seen:
                    continue
                next_prob = max_probs[node] + neigh_prob
                if next_prob > max_probs[neigh]:
                    max_probs[neigh] = next_prob
                    heapq.heappush(max_heap, (-next_prob,neigh))
        
        if max_probs[end] == float('-inf'):
            return 0
        return 2**max_probs[end]

################################################
# 1976. Number of Ways to Arrive at Destination
# 27AUG24
################################################
#TLE
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        '''
        use any ssp algo to generate the dists array
        find the min dist and use only edges
        then us dp!
        '''
        graph = defaultdict(list)
        for u,v,w in roads:
            graph[u].append((v,w))
            graph[v].append((u,w))
        
        dists = [float('inf')]*n
        dists[0] = 0
        
        min_heap = [(0,0)]
        seen = set()
        
        while min_heap:
            min_dist_so_far,curr = heapq.heappop(min_heap)
            if dists[curr] < min_dist_so_far:
                continue
            seen.add(curr)
            for neigh,dist_to in graph[curr]:
                if neigh in seen:
                    continue
                next_dist = dists[curr] + dist_to
                if dists[neigh] > next_dist:
                    dists[neigh] = next_dist
                    heapq.heappush(min_heap, (next_dist,neigh))
        
        min_time = dists[n-1]
        @cache
        def dp(curr,total_time):
            if total_time > min_time:
                return 0
            if curr == n -1:
                if total_time == min_time:
                    return 1
                return 0
            ways = 0
            for neigh,dist in graph[curr]:
                ways += dp(neigh,total_time + dist)

            return ways
        
        return dp(0,0)
                    
class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        '''
        need to count on the fly while we do djisktras
        if we find a path from 0 to some node v, with time k and it == the min_time so far
        we carry the number of ways from 0 to b
        '''
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
            min_dist_so_far,curr = heapq.heappop(min_heap)
            if dists[curr] < min_dist_so_far:
                continue
            seen.add(curr)
            for neigh,dist_to in graph[curr]:
                if neigh in seen:
                    continue
                next_dist = dists[curr] + dist_to
                if dists[neigh] > next_dist:
                    dists[neigh] = next_dist
                    #carry over the number of ways for this min_dist
                    ways[neigh] = ways[curr]
                    heapq.heappush(min_heap, (next_dist,neigh))
                #we have found an additional way to reach neigh with the current min_time
                #so we add it
                elif next_dist == dists[neigh]:
                    ways[neigh] += ways[curr]
                    ways[neigh] %= mod
                    
        return ways[n-1] % mod
        
#############################
# 1905. Count Sub Islands
# 28AUG24
#############################
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        '''
        do union find in grid 1 to map (i,j) to its island
        then for each island in grid2, check that all the cell in this island belong to an island in grid1
        i dont have to do union find, i can just dfs on grid1 and paint the cells accordinlgy
        '''
        #color islands
        seen = set()
        color = 2
        rows = len(grid1)
        cols = len(grid2[0])
        for i in range(rows):
            for j in range(cols):
                if grid1[i][j] == 1 and (i,j) not in seen:
                    self.paint(grid1,i,j,rows,cols,seen,color)
                    color += 1
        
        #now find islands in grid2 and check
        count = 0
        seen = set()
        for i in range(rows):
            for j in range(cols):
                if grid2[i][j] == 1 and (i,j) not in seen:
                    if self.same_color(grid1,grid2,i,j,rows,cols,seen,grid1[i][j]):
                        count += 1
        return count
    
    def paint(self,grid,i,j,rows,cols,seen,color):
        #color it
        grid[i][j] = color
        seen.add((i,j))
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        for d_i,d_j in dirrs:
            ii,jj = i + d_i, j + d_j
            if (0 <= ii < rows) and (0 <= jj < cols):
                if (ii,jj) not in seen and grid[ii][jj] == 1:
                    self.paint(grid,ii,jj,rows,cols,seen,color)
    
    def same_color(self,g1,g2,i,j,rows,cols,seen,color):
        #g2 is the one we are in, compare color to g1
        if g1[i][j] != color:
            return False
        if g1[i][j] == 0:
            return False
        seen.add((i,j))
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        for d_i,d_j in dirrs:
            ii,jj = i + d_i, j + d_j
            if (0 <= ii < rows) and (0 <= jj < cols):
                if (ii,jj) not in seen and g2[ii][jj] == 1:
                    if not self.same_color(g1,g2,ii,jj,rows,cols,seen,color):
                        return False
        return True
    
#aye yai yai, ugly but is passes....not too happy about it
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        '''
        do union find in grid 1 to map (i,j) to its island
        then for each island in grid2, check that all the cell in this island belong to an island in grid1
        i dont have to do union find, i can just dfs on grid1 and paint the cells accordinlgy
        
        i need to make sure i get the whole island in the second pass
        i can just use dfs to get the island and check they are all the same color!
        '''
        #color islands
        seen = set()
        color = 2
        rows = len(grid1)
        cols = len(grid2[0])
        for i in range(rows):
            for j in range(cols):
                if grid1[i][j] == 1 and (i,j) not in seen:
                    self.paint(grid1,i,j,rows,cols,seen,color)
                    color += 1

        #now find islands in grid2 and check
        count = 0
        seen = set()
        curr_island = []
        for i in range(rows):
            for j in range(cols):
                if grid2[i][j] == 1 and (i,j) not in seen:
                    self.capture(grid2,i,j,rows,cols,seen,curr_island)
                    #check them all that they are the same color
                    ii,jj = curr_island[0]
                    color = grid1[ii][jj]
                    valid = True
                    for ii,jj in curr_island:
                        if grid1[ii][jj] == 0:
                            valid = False
                            break
                    for ii,jj in curr_island[1:]:
                        if grid1[ii][jj] != color or grid1[ii][jj] == 0:
                            valid = False
                            break
                    if valid:
                        print(curr_island)
                        count += 1
                    curr_island = []
        return count
    
    def paint(self,grid,i,j,rows,cols,seen,color):
        #color it
        grid[i][j] = color
        seen.add((i,j))
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        for d_i,d_j in dirrs:
            ii,jj = i + d_i, j + d_j
            if (0 <= ii < rows) and (0 <= jj < cols):
                if (ii,jj) not in seen and grid[ii][jj] == 1:
                    self.paint(grid,ii,jj,rows,cols,seen,color)
    
    def capture(self,g2,i,j,rows,cols,seen,curr_island):
        seen.add((i,j))
        curr_island.append((i,j))
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        for d_i,d_j in dirrs:
            ii,jj = i + d_i, j + d_j
            if (0 <= ii < rows) and (0 <= jj < cols):
                if (ii,jj) not in seen and g2[ii][jj] == 1:
                    self.capture(g2,ii,jj,rows,cols,seen,curr_island)

########################################
# 320. Generalized Abbreviation (REVISTED)
# 28AUG24
#########################################
#finally!
class Solution:
    def generateAbbreviations(self, word: str) -> List[str]:
        '''
        keep index i,path, and count of the lenght we are currently abbreviating
        '''
        paths = []
        N = len(word)
        
        def dfs(i,path,count):
            #gotten to the end, append ans and count
            if i == N:
                if count:
                    path.append(str(count))
                paths.append("".join(path))
                return
            #abbeviate
            if count:
                dfs(i+1,path + [str(count)] + [word[i]],0)
            else:
                dfs(i+1,path + [word[i]],0)
            
            dfs(i+1,path,count+1)
            
        dfs(0,[],0)
        return paths

#iterative
class Solution:
    def generateAbbreviations(self, word: str) -> List[str]:
        '''
        we can do bottom up and treat each abbreviation as a bit mask
        1 at this bit position means it was abbreviate, 0 means it was not abbreviated
        let n be len(word), there are 2**n possible masks, and so 2**N abbreivations
        including the null abbreviation
        '''
        n = len(word)
        abbreviations = []
        
        for mask in range(1 << n):
            curr_abbreviation = ""
            count = 0
            
            for i in range(n):
                #if 1 is set, abbreviate
                if mask & (1 << i):
                    count += 1
                else:
                    #abbreivate so far up to this i
                    if count > 0:
                        curr_abbreviation += str(count)
                        count = 0
                    curr_abbreviation += word[i]
                
            #if we still have count
            if count > 0:
                curr_abbreviation += str(count)

            abbreviations.append(curr_abbreviation)
        
        return abbreviations