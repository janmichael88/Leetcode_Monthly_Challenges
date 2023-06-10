######################################
# 1230. Toss Strange Coins
# 02JUN23
#####################################
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        '''
        if i just had 1 coin and flipped it, this would just have been a bernoulli trial
        but we are given an array of coins
        keep track of count heads, count tails, position, and curr probabilty
            
        
        '''
        n = len(prob)
        memo = {}
        def dp(i,count):
            if count > target:
                return 0
            if i == n:
                if count == target:
                    return 1
                else:
                    return 0
            
            if (i,count) in memo:
                return memo[(i,count)]
            
            #getting a head
            get_head = prob[i]*dp(i+1,count+1)
            #getting tails
            get_tails = (1-prob[i])*dp(i+1,count) #doesn't go up because its a tail
            ans = get_head + get_tails
            memo[(i,count)] = ans
            return ans
        
        
        return dp(0,0)
    

#bottom up
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        '''
        if i just had 1 coin and flipped it, this would just have been a bernoulli trial
        but we are given an array of coins
        keep track of count heads, count tails, position, and curr probabilty
            
        
        '''
        n = len(prob)
        dp = [[0]*(target+1) for _ in range(n+1)]
        
        #base case fill
        for i in range(n,-1,-1):
            for count in range(target,-1,-1):
                if count > target:
                    dp[i][count] = 0
                if i == n:
                    if count == target:
                        dp[i][count] = 1
                    else:
                        dp[i][count] = 0
                        
        
        #one away from base case
        for i in range(n-1,-1,-1):
            for count in range(target-1,-1,-1):
                #getting a head
                get_head = prob[i]*dp[i+1][count+1]
                #getting tails
                get_tails = (1-prob[i])*dp[i+1][count] #doesn't go up because its a tail
                ans = get_head + get_tails
                dp[i][count] = ans
        
        print(dp)
        return dp[0][0]

class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        n = len(prob)
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] * (1 - prob[i - 1])
            for j in range(1, target + 1):
                if j > i:
                    break
                dp[i][j] = dp[i - 1][j - 1] * prob[i - 1] + dp[i - 1][j] * (1 - prob[i - 1])

        return dp[n][target]

#######################################################
# 1502. Can Make Arithmetic Progression From Sequence
# 06JUN23
######################################################
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        '''
        there should only be one difference
        try sorting increasnly, then check,
        then decresilngly then check
        '''
        N = len(arr)
        if N == 2:
            return True
        
        arr_inc = sorted(arr)
        for i in range(1,N-1):
            x = arr_inc[i-1]
            y = arr_inc[i]
            z = arr_inc[i+1]
            
            if y - x != z - y:
                return False
            
        return True
    
#no sorting
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        '''
        we don't  need to sort
        assume the array is arithmetic, with common differecne diif
        let the first term by min+valie
            then the differeence between each eleement arr[i] and min_value must be a multiple of diff
        example
        [1,4,7,10]
        10 - 1 / 3 = 3
        4 -  1 % 3 == 3
        
        we just check that (arr[i] - min_value) % diff is a multiple
            i.e modulo == 0
        
        careful with duplicate elements in the array, like [1,2,3,2,5]
        this will fail the first algo since each arr[i] - min is divisible by diff
        use set, and check size of set == len(arr)
        
        '''
        min_value,max_value = min(arr),max(arr)
        N = len(arr)
        
        #all the same, return true
        if max_value - min_value == 0:
            return True
        #range should be also divisible by steps
        if (max_value - min_value) % (N-1):
            return False
        
        diff = (max_value - min_value) // (N-1)
        seen = set()
        for num in arr:
            if (num - min_value) % diff:
                return False
            seen.add(num)
        
        return len(seen) == N
    
#constant space
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        '''
        in order to use O(1) space we need to modifiy the input
        say an array is arithmetic, with range_diff == diff,tecnically diff / (n-1) should be the step size
        we can difine its position in the array as
            j = (arr[i] - min_value) / diff
        
        if j == i, then this number is in the right place, othrwise swap
        
        '''
        min_value,max_value = min(arr),max(arr)
        N = len(arr)
        
        #all the same, return true
        if max_value - min_value == 0:
            return True
        #range should be also divisible by steps
        if (max_value - min_value) % (N-1):
            return False
        
        diff = (max_value - min_value) // (N-1)
        
        i = 0
        while i < N:
            #arr[i] is correctly place
            if arr[i] == min_value + i*diff:
                i += 1
            
            #doesn't belong to this arithemtic sequence
            elif (arr[i] - min_value) % diff:
                return False
            #otherwise swap
            else:
                j = (arr[i] - min_value) // diff
                #dupliate?
                if arr[i] == arr[j]:
                    return False
                arr[i],arr[j] = arr[j], arr[i]
        
        return True
    
##################################
# 999. Available Captures for Rook
# 06JUN23
##################################
#sheesh, works through
class Solution:
    def numRookCaptures(self, board: List[List[str]]) -> int:
        '''
        the highest attacking position for a rook is 4
        a rook is considered attacking a pawn if th erook can capture on this turn
        the number of available captures for the white rook is the number of pawns that the rook is attacking
        
        just check all the way up, all the way down, left and right
        '''
        n = 8
        rook_i,rook_j = -1,-1
        for i in range(n):
            for j in range(n):
                if board[i][j] == 'R':
                    rook_i,rook_j = i,j
                    break
        
        
        pawns = 0
        #check up
        curr_i,curr_j = rook_i,rook_j
        curr_i -= 1
        while curr_i > 0 and board[curr_i][curr_j] == '.':
            curr_i -= 1
        pawns += board[curr_i][curr_j] == 'p'
        
        #check down
        curr_i,curr_j = rook_i,rook_j
        curr_i += 1
        while curr_i < n-1 and board[curr_i][curr_j] == '.':
            curr_i += 1
        pawns += board[curr_i][curr_j] == 'p'
        
        #check left
        curr_i,curr_j = rook_i,rook_j
        curr_j -= 1
        while curr_j > 0  and board[curr_i][curr_j] == '.':
            curr_j -= 1
        pawns += board[curr_i][curr_j] == 'p'
        
        curr_i,curr_j = rook_i,rook_j
        curr_j += 1
        while curr_j < n-1  and board[curr_i][curr_j] == '.':
            curr_j += 1
        pawns += board[curr_i][curr_j] == 'p'
        

#consolidate while loops
class Solution:
    def numRookCaptures(self, board):
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'R':
                    x0, y0 = i, j
                    break
        res = 0
        #store directions and paired steps
        for i, j in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
            x, y = x0 + i, y0 + j
            while 0 <= x < 8 and 0 <= y < 8:
                if board[x][y] == 'p': 
                    res += 1
                if board[x][y] != '.': 
                    break
                x, y = x + i, y + j
        return res

################################################
# 1318. Minimum Flips to Make a OR b Equal to c
# 07JUN23
################################################
#ugly but it works
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        '''
        find min bit flips in a and/or b such that a or b == c
        hint says to check bits 1 by 1 and see if they need to be flipped
        
        '''
        bits_a = bin(a)[2:]
        bits_b = bin(b)[2:]
        bits_c = bin(c)[2:]
        
        #need pad so that the bits are equallength
        largest_size = max(len(bits_a),len(bits_b),len(bits_c))
        bits_a = '0'*(largest_size - len(bits_a))+bits_a
        bits_b = '0'*(largest_size - len(bits_b))+bits_b
        bits_c = '0'*(largest_size - len(bits_c))+bits_c
        
        ans = 0
        
        print(bits_a,bits_b,bits_c)
        for i in range(largest_size):
            x = int(bits_a[i])
            y = int(bits_b[i])
            z = int(bits_c[i])
            
            if (x | y == z):
                continue
            else:
                if (x == 1) and (y == 1) and (z == 0):
                    ans += 2
                else:
                    ans += 1
        
        return ans
    
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        '''
        i don't need to convert string, just keep shifting and use the & operator to get the bit
        
        '''
        
        ans = 0
        
        while a or b or c:
            #& operatore to get the bit value
            x = a & 1
            y = b & 1
            z = c & 1
            
            if (x | y == z):
                a = a >> 1
                b = b >> 1
                c = c >> 1
            else:
                if (x == 1) and (y == 1) and (z == 0):
                    ans += 2
                else:
                    ans += 1
            
                #shift entirely
                a = a >> 1
                b = b >> 1
                c = c >> 1
        
        return ans
    
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        '''
        we can get the least significant bit of any integer P using P & 1
        case1:
            c & 1 = 1
            we need at least one bit in either a & 1 or b & 1, if either is, we can go on to the next bit, otherwise we need to flip
            
        case2:
            c & 1 = 0:
            both (a & 1) == 0 and (b & 1) == 0
            if either (a & 1) or (b & 1) == q, we need on flip to make it zero
            number of flips == (a & 1) + (b & 1)
        '''
        flips = 0
        
        while a or b or c:
            #current c bit is 1 
            if c & 1:
                #need a 1 from either a or b
                flips += 0 if ((a & 1) or (b & 1)) else 1
            else:
                #its a zero, only way it could be zero if they are bother zero
                flips += (a & 1) + (b & 1)
            
            a >>= 1
            b >>= 1
            c >>= 1
        
        return flips
    
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        '''
        recall the XOR operation
            returns 1 if the bits are different, else 0
            if we do (a | b) ^ c, then every bit that is difference will have a value of 1
        
        we are trying to see if a | b == c
        if we do (a | b) ^ c, and we know that XOR shows which bits are differenet, then we can just sum the set bits
        
        
        however, there is one exception, when (c & 1) = 0 and both (a & 1) and (b & 1) are both 1
        we need an extra flip here,
        we can use the & operator to find the needed extra flips
        (a & b) & ((a | b) ^ c)
        
        he final step is to count the number of digits 1 in the binary representation of the two numbers (a | b) ^ c and (a & b) & ((a | b) ^ c).


            
        '''
        first = (a | b) ^ c
        second = (a & b) & first
        return bin(first).count("1") + bin(second).count("1")
    
#################################################
# 1351. Count Negative Numbers in a Sorted Matrix
# 08JUN23
#################################################
#binary search with error checking
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        '''
        could also binary search to find negative numbers along rows or cols
        find upper bound
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        count = 0
        for row in grid:
            left = 0
            right = len(grid)
            
            while left < right:
                mid = left + (right - left) // 2
                if row[mid] < 0: #there are more negatives to the right
                    right = mid - 1
                else:
                    left = mid + 1
            
            if left < cols and row[left] > 0:
                left += 1
            count += (cols - left)
        
        return count
    

class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        count = 0
        n = len(grid[0])
        # Iterate on all rows of the matrix one by one.
        for row in grid:
            # Using binary search find the index
            # which has the first negative element.
            left, right = 0, n - 1
            while left <= right:
                mid = (right + left) // 2
                if row[mid] < 0:
                    right = mid - 1
                else:
                    left = mid + 1
            # 'left' points to the first negative element,
            # which means 'n - left' is the number of all negative elements.
            count += (n - left)
        return count
    
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        '''
        both rows and cols are sorted in non-increasing order
        if we are on row[i] with first negtaive index k, then we know that for any row [i+l]
        the first index cannot be greater than k
        
        traverse right to left, and going down each row, find the first negative index
        '''
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        
        curr_negative_idx = cols - 1
        #this will point to the last positive element
        #so currnegtaive is going to be + 1
        
        for row in grid:
            while curr_negative_idx >= 0 and row[curr_negative_idx] < 0:
                curr_negative_idx -= 1
            
            #increase count
            count += (cols - (curr_negative_idx + 1))
        
        return count

#################################################################
# 1150. Check If a Number Is Majority Element in a Sorted Array
# 09JUN23
#################################################################
#fucking edge contitions
class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        '''
        need to use binary search to find the frequencies since the array is sorted
        binary search twice to find the first and last occurences of target
        '''
        N = len(nums)
        #if there is a majority eleemnt, it should be found at the middle index
        if nums[N // 2] != target:
            return False
        #find lower bound
        left = 0
        right = N
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        
        lower_bound = left
        #find upper bound
        left = 0
        right = N
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid
            else:
                left = mid + 1
        
        upper_bound = left
        if upper_bound == N:
            upper_bound -= 1
        #recall these are the insertion points
        print(lower_bound,upper_bound)
        if nums[lower_bound] == nums[upper_bound] == target:
            return (upper_bound - lower_bound + 1) > N // 2
        else:
            return False
        
class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        '''
        recall lower bound returns the index of the first element >= to the target
        if there is no instance, it retuns the length of the list
        
        more succintly, lower bound is the FIRST element >= than the target
        upper bound is the first element that is greater than the given element
        '''
        N = len(nums)
        #if there is a majority eleemnt, it should be found at the middle index
        if nums[N // 2] != target:
            return False
        #find lower bound
        left = 0
        right = N-1
        lower_bound = N
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
                lower_bound = mid
            else:
                left = mid + 1
        
        #find upper bound
        upper_bound = N
        left = 0
        right = N - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
                upper_bound = mid
            else:
                left = mid + 1
        
        upper_bound = left
        print(lower_bound,upper_bound)
        return (upper_bound - lower_bound + 1) > N // 2

#much easier to use builtins
class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        '''
        uisng the builtins
        '''
        N = len(nums)
        if nums[N//2] != target:
            return False
        
        lo = bisect.bisect_left(nums,target)
        hi = bisect.bisect_right(nums,target)
        return hi - lo > N // 2

class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        '''
        custome, lo == target
        hi == target + 1
        '''

        def search(a, x):
            lo, hi = 0, len(a)
            while lo < hi:
                mid = (lo + hi) // 2
                if a[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
            
        N = len(nums)
        if nums[N // 2] != target:
            return False
        lo = search(nums, target)
        hi = search(nums, target + 1)
        return hi - lo > N // 2

class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:
        '''
        we only need to search for the first index of target, or the first element that is <= target
        then we just check if nums[this index + len(nums) /2] is also the target
        this would mean we have a majority for this element
        '''
        N = len(nums)
        left = 0
        right = N - 1
        lower_bound = N
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                right = mid - 1
                lower_bound = mid
            else:
                left = mid + 1
        
        if (lower_bound + N // 2) < N:
            return nums[lower_bound + (N//2)] == target
        
###############################################
# 744. Find Smallest Letter Greater Than Target
# 09JUN23
###############################################
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        '''
        binary search to find the upper bound
        '''
        left = 0
        right = len(letters)
        while left < right:
            mid = left + (right - left) // 2
            if letters[mid] > target:
                right = mid
            else:
                left = mid + 1
        
        if left == len(letters):
            return letters[0]
        return letters[left]
    
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        '''
        upper bound with wrap around
        '''
        upper = bisect.bisect_right(letters,target)
        return letters[upper % len(letters)]