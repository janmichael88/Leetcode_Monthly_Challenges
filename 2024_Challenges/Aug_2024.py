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