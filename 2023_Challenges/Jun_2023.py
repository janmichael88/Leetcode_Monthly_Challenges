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

#########################################################
# 1802. Maximum Value at a Given Index in a Bounded Array
# 10JUN23
##########################################################
class Solution:
    def maxValue(self, n: int, index: int, maxSum: int) -> int:
        '''
        if we were to constuct an array, it should have:
            len(arry) == n
            nums[i] > 0 for i in range(0,n+1), must be all positive, and cannot be zero
            abs(arra[i]-array[i+1]) <= 1 for i in range(0,n), difference between consecutive elements cannot be more than 1
            sum(array) does not exceed maxSum
            array[index] is maximized
            
        return array[index]
        
        notes:
            nums[index] cannot be maxSum
            nums[index] must be the largest weight
            lets all this largest_weight
            if nums[index] == largest_weight
            then sum of remaining == max_sum - largest_weight
            rather:
            left = sum(nums[:index -1])
            right = sum(nums[index+1:])
            and left + right == max_sum - largest_weight
            we can always make a valid array equaling left and right

        guranteed to be valid array becayse we have 1 <= n <= maxSum <= 10**9
        if we wanted we could check for invalid array
            if n > maxSum, canot do it
        
        and if index > n; cannot do it
        
        try all targets and see if we can make it?
        objective is to maximize nums[index] while ensuring sum array does not exceed maxSum, but we cannot abritiatly make left and right sums 
        anything we want
        
        easiet approach would be after setting nums[index] = target
        let numbers to its left decrease by one until 1, and left numbers to the right decreased by 1 until 1
        calculating sum of array parts?
        take left for example, this would be an increasing arithmetic sequence with a sectino of consectuvie 1s if nums[index] < number of elements to the left
        we need to determing the arithmetic sequence cased on the size of index and value
        one we have determing the length of the seqeucne, we can sum the seuqnce using
        (nums[start] + nums[end]) *n/2
        1 + 2 + ... n = n*(n+1)/2
        if start === 1 and ending is n
        (1 + k) + (2 + k) + ... + (n+k) = (n+k)*(n+k+1)/2
        
        if value <= index, it means in addition to the arithmetic sequecne from value to 1, we have a region of consecutive 1's of lenfth index - value + 1
            sum of arithmetic sequecne will be [1,2,3...value-1,value] == (value)*(value+1)/2
            sume of consetive 1s sequence will be index - value + 1 
        
        if value > index, there are is consecutive 1s
            The sum of arithmetic sequence [value - index, ..., value - 1, value], which is (value + value - index) * (index + 1) / 2.
            really its just ((value-index) + (value)) *(index+1) / 2
            
        for the right:
            we compare with n-index
            if value <= n  index:
                we have the arithemtic sequcne and string of consective 1s of length n-index - value in addition to the arithemtic sequence from value to 1
                sum of arithemtic sequence if (1 + value)*value/2
                sume of consectives 1s will be n-index - value
                
            if vvalue > n-index, we jsut have the sequence
                [value, value - 1, ..., value - n + 1 + index], which is (value + value - n + 1 + index) * (n - index) / 2.
                
            ALSO! dont forget that we have added the actual value at index twice, so we need to subtract the final sum by value (the one that we try)
            
        
        we can just use binary search to find the workable solution, and keep trying it
        first we need to deinfe a function getSum(index,value) to calculate the minsum of the array, given that we fix nums[index] = value
            have it return the sum of the array, 
            and check if we <= maxSum or > maxSum
        '''
        left, right = 1, maxSum
        while left < right:
            mid = (left + right + 1) // 2
            if self.getSum(index, mid, n) <= maxSum:
                left = mid
            else:
                right = mid - 1
        
        return left
    def getSum(self, index: int, value: int, n: int) -> int:
        count = 0

        # On index's left:
        # If value > index, there are index + 1 numbers in the arithmetic sequence:
        # [value - index, ..., value - 1, value].
        # Otherwise, there are value numbers in the arithmetic sequence:
        # [1, 2, ..., value - 1, value], plus a sequence of length (index - value + 1) of 1s. 
        if value > index:
            count += (value + value - index) * (index + 1) // 2
        else:
            count += (value + 1) * value // 2 + index - value + 1

        # On index's right:
        # If value >= n - index, there are n - index numbers in the arithmetic sequence:
        # [value, value - 1, ..., value - n + 1 + index].
        # Otherwise, there are value numbers in the arithmetic sequence:
        # [value, value - 1, ..., 1], plus a sequence of length (n - index - value) of 1s. 
        if value >= n - index:
            count += (value + value - n + 1 + index) * (n - index) // 2
        else:
            count += (value + 1) * value // 2 + n - index - value

        return count - value
    
class Solution:
    def maxValue(self, n: int, index: int, maxSum: int) -> int:
        '''
        another eay
        '''
        #lambda function get get arithmetics sequence sum
        calc_sum = lambda x,y: (x-y)*y//2
        left = 1
        right = maxSum + 1
        
        while left < right:
            mid = left + (right - left) // 2
            #get sizes of subarray
            left_size = index
            right_size = n - index - 1
            #get counsective ones on each side
            #don't forget this max trick
            ones_left = max(left_size - mid + 1,0)
            ones_right = max(right_size - mid +1,0)
            #get sums
            left_sum = calc_sum(2*mid-1, left_size - ones_left)
            right_sum = calc_sum(2*mid -1, right_size - ones_right)
            if ones_left + left_sum + mid + right_sum + ones_right > maxSum:
                right = mid
            else:
                left = mid + 1
        
        return left - 1
    
######################################
# 1146. Snapshot Array
# 11JUN23
######################################
class SnapshotArray:
    def __init__(self, length: int):
        '''
        hint1: use list of lists adding both element and snap id to each index
        '''
        self.container = [[[0,0]] for _ in range(length)] #at the index we have (snap_id,val)
        self.curr_snap = 0
        

    def set(self, index: int, val: int) -> None:
        #check most recent for updating
        if self.container[index][-1][0] == self.curr_snap:
            self.container[index][-1][1] = val
            return
        #otherise append
        self.container[index].append([self.curr_snap,val])

    def snap(self) -> int:
        self.curr_snap += 1
        return self.curr_snap - 1

        
    def get(self, index: int, snap_id: int) -> int:
        #binary search to find the snap id
        curr_list = self.container[index]
        left = 0
        right = len(curr_list)
        #we are searching for the upper bound, so return the one snap shot just before it
        while left < right:
            mid = left + (right - left) // 2
            if curr_list[mid][0] > snap_id:
                right = mid
            else:
                left = mid+1
        return curr_list[left-1][1]


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)

class SnapshotArray:
    def __init__(self, length: int):
        '''
        hint1: use list of lists adding both element and snap id to each index
        '''
        self.container = [[[0,0]] for _ in range(length)] #at the index we have (snap_id,val)
        self.curr_snap = 0
        

    def set(self, index: int, val: int) -> None:
        #check most recent for updating
        if self.container[index][-1][0] == self.curr_snap:
            self.container[index][-1][1] = val
            return
        #otherise append
        self.container[index].append([self.curr_snap,val])

    def snap(self) -> int:
        self.curr_snap += 1
        return self.curr_snap - 1

        
    def get(self, index: int, snap_id: int) -> int:
        #binary search to find the snap id
        curr_list = self.container[index]
        #can also use biect right
        snap_index = bisect.bisect_right(curr_list,[snap_id, float('inf')]) #need to specify list of lists
        return curr_list[snap_index-1][1]


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)

#########################################
# 1002. Find Common Characters
# 12JUN23
#########################################
#close
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        '''
        make hashset of letters for words[0]
        for then for words[1:], use &
        including duplicates huh
        need to cu
        
        count map, and store as tuple (char,count)
        '''
        common = Counter(words[0])
        common = set([(k,v) for k,v in common.items()])
        
        for word in words[1:]:
            word_count = Counter(word)
            temp = set([(k,v) for k,v in word_count.items()])
            common = common & temp
        
        ans = []
        while common:
            char,count = common.pop()
            for _ in range(count):
                ans.append(char)
        
        return ans
    
#oh god, should be medium
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        '''
        keep count and check
        '''
        common = Counter(words[0])
        
        for word in words[1:]:
            word_count = Counter(word)
            #need to generate new common
            new_common = Counter()
            first = set(common.keys())
            second = set(word_count.keys())
            #get current intersection
            curr_intersection = list(first & second)
            for ch in curr_intersection:
                new_common[ch] = min(common[ch],word_count[ch])
            #reassign
            common = new_common
        
        ans = []
        
        for ch,count in common.items():
            for _ in range(count):
                ans.append(ch)
        
        return ans
    
class Solution:
    def commonChars(self, A: List[str]) -> List[str]):
        """
        :type A: List[str]
        :rtype: List[str]
        """
        # counter generator:
        def CreateCounter(string):
            counter = {}
            for c in string:
                if c not in counter:
                    counter[c] = 0
                counter[c] += 1
            return counter
        
        if A is None or len(A) == 0: return A
        counter = CreateCounter(A[0])
     
        for word in A[1:]:
            currCounter = CreateCounter(word)
            copy = list(counter.items())
            for key, value in copy:
                if key in currCounter:
                    counter[key] = min(value, currCounter[key])
                else:
                    del counter[key]
        ans = []
        for c, count in counter.items():            
            for _ in range(count):
                ans.append(c)
        return ans
            
class Solution:
    def commonChars(self, words: List[str]) -> List[str]:
        common = Counter(words[0])
        
        for word in words[1:]:
            new_common = Counter()
            for ch in word:
                if common[ch] > 0:
                    new_common[ch] += 1
                    common[ch] -= 1
            
            common = new_common
        
        ans = []
        for c, count in common.items():            
            for _ in range(count):
                ans.append(c)
        return ans
                
######################################
# 2352. Equal Row and Column Pairs
# 13JUN23
######################################
#brute force passes??
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        '''
        i can un pack all the rows as rows and cols as rows
        then we need to check that all (i row) and (j pairs)
        would be O(N*N + N*N)
        carch
        '''
        N = len(grid)
        rows = defaultdict(list)
        cols = defaultdict(list)
        
        for i in range(N):
            for j in range(N):
                val = grid[i][j]
                rows[i].append(val)
                cols[j].append(val)
        
        
        ans = 0
        for i in range(N):
            for j in range(N):
                curr_row = rows[i]
                curr_col = cols[j]
                if curr_row == curr_col:
                    ans += 1
        
        return ans
    
#lock at row r and col c, then move and check
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        N = len(grid)
        ans = 0
        
        for i in range(N):
            for j in range(N):
                same = True
                
                for k in range(N):
                    if grid[i][k] != grid[k][j]:
                        same = False
                        break
                
                ans += int(same)
        
        return ans
    
#row and col signatures
class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        '''
        store rows as keys in count maps
        then just get cols, and check if this col signature is in the hahsmap
        '''
        ans = 0
        row_counts = Counter()
        N = len(grid)
        
        for row in grid:
            row_counts[tuple(row)] += 1
            
        for c in range(N):
            curr_col = []
            for r in range(N):
                curr_col.append(grid[r][c])
            
            #col can be repeated as many times tuple(col) in row_counts
            curr_col = tuple(curr_col)
            ans += row_counts[curr_col]
        
        return ans

#Trie
class Node:
    def __init__(self,):
        self.count = 0
        self.children = {}
        

class Trie:
    def __init__(self,):
        self.root = Node()
    
    def insert(self,array):
        curr = self.root
        for num in array:
            if num not in curr.children:
                curr.children[num] = Node()
            curr = curr.children[num]
        
        curr.count += 1
        
    def search(self,array):
        curr = self.root
        for num in array:
            if num not in curr.children:
                return 0
            else:
                curr = curr.children[num]
        
        return curr.count
        
        

class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        '''
        we can use a Trie, where each node represents an eleemnts in a row /col
        the leaves of the nodes
            or when we get to the end, store counts
        '''
        N = len(grid)
        t = Trie()
        ans = 0
        
        for row in grid:
            t.insert(row)
        
        for c in range(N):
            curr_col = []
            for r in range(N):
                curr_col.append(grid[r][c])
            
            ans += t.search(curr_col)
        
        return ans
        
##########################################
# 1161. Maximum Level Sum of a Binary Tree
# 15JUN23
###########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        '''
        dfs and store levels
        '''
        level_sums = []
        def dfs(node,curr_level):
            if not node:
                return
            if curr_level >= len(level_sums):
                level_sums.append([])
            #do sum on the fly
            if len(level_sums[curr_level]) == 1:
                level_sums[curr_level][0] += node.val
            else:
                level_sums[curr_level].append(node.val)
            dfs(node.left,curr_level+1)
            dfs(node.right,curr_level+1)
        
        dfs(root,0)
        print(level_sums)
        max_sum = float('-inf')
        ans = 1
        for i in range(len(level_sums)):
            curr_sum = level_sums[i]
            if curr_sum[0] > max_sum:
                ans = i+1
                max_sum = curr_sum[0]
        
        return ans
    
#can we reduce second part on the fly?
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        '''
        bfs and keep level sums along with a max sum
        '''
        max_sum = float('-inf')
        min_level = 1
        curr_level = 1
        
        q = deque([root])
        
        while q:
            curr_sum = 0
            N = len(q)
            for _ in range(N):
                curr_node = q.popleft()
                curr_sum += curr_node.val
                if curr_node.left:
                    q.append(curr_node.left)
                if curr_node.right:
                    q.append(curr_node.right)
            
            if curr_sum > max_sum:
                max_sum = curr_sum
                min_level = curr_level
            
            
            curr_level += 1
        
        return min_level

########################################
# 163. Missing Ranges (REVISTED)
# 15JUN23
#######################################
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        '''
        number is considered missing if in the inclusive range [lower,uppper]
        and not in nums
        return shortest shorted list of ranges that covers all missing numbers
        [0,1,3,50,75], lower = 0, upper = 99
        append bounds
        [0,1,3,50,75,99]
        [
        [2,2],
        [4,49],
        [51,75],
        [76,99]
        ]
        '''
        if lower < nums[0]:
            nums = [lower] + nums
        
        if upper > nums[-1]:
            nums = nums + [upper]
        
        intervals = []
        #now just find the missing gapes
        curr_interval = []
        for i in range(1,len(nums)):
            if nums[i] - nums[i-1] != 1:
                lower_bound = nums[i-1] + 1
                upper_bound = nums[i] - 1
                curr_interval = [lower_bound,upper_bound]
                intervals.append(curr_interval)
            
            curr_interval = []
        
        #adjust beginning and ends
        intervals[0][0] = min(lower,nums[0])
        intervals[-1][1] = max(upper,nums[-1])
        return intervals
    
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        '''
        number is considered missing if in the inclusive range [lower,uppper]
        and not in nums
        return shortest shorted list of ranges that covers all missing numbers
        [0,1,3,50,75], lower = 0, upper = 99
        append bounds
        [0,1,3,50,75,99]
        [
        [2,2],
        [4,49],
        [51,75],
        [76,99]
        ]
        
        #edge cases
            if we dont start with lower as the first element in the array, we need to include [lower,nums[0]-1]
            if we don't end with upper as the last element of the array, we need to cinlude it at the end [nums[-1]+1,upper]
        '''
        n = len(nums)
        
        if n == 0:
            return [[lower,upper]]
        intervals = []
        
        if lower < nums[0]:
            intervals.append([lower,nums[0]-1])

        #now just find the missing gapes
        curr_interval = []
        for i in range(1,len(nums)):
            if nums[i] - nums[i-1] != 1:
                lower_bound = nums[i-1] + 1
                upper_bound = nums[i] - 1
                curr_interval = [lower_bound,upper_bound]
                intervals.append(curr_interval)
            
            curr_interval = []
        
        if upper > nums[-1]:
            intervals.append([nums[-1]+1,upper])
        return intervals
    
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        '''
        cool use of pairwise
        if i do [lower-1] + nums + [upper+1], just one before lower and one greater than upper
        [1,2,3,4,5,6], [1,6]
        [0,1,2,3,4,5,6,7]
        [1,2]
        a = 1
        b = 2
        [2,2]
        '''
        nums = [lower-1] + nums + [upper+1]
        intervals = []
        for a,b in pairwise(nums):
            if a + 1 <= b - 1:
                intervals.append([a+1,b-1])
        
        return intervals
    
#one line
class Solution:
    def findMissingRanges(self, nums, lower, upper) -> List[List[int]]:
        return ([a+1,b-1] for a,b in pairwise([lower-1]+nums+[upper+1]) if a+1 <= b-1)
    
########################################################
# 1569. Number of Ways to Reorder Array to Get Same BST
# 16JUN23
#########################################################
#close one
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        '''
        divide and conquer
        first number will be the root, consdier the numbers smaller and larger than the root seperately
        when merging the results together, how many ways can you order x elements in x + y spots
        counting problem
        
        inserting into a BST, find point then rebalance, we just want perms of nums such that when we insert in order of the perm
        we get the same BST tree
        starting with [2,1,3]
        2 will always be the root, so i can't start with anything larger then 2
        so fix [2], then we have the remaining part of the array [1,3]
        in this pushing 1 before 3 or 3 before 1 makes the same BST, so we wan do [2,1,3], [2,3,1]
        
        '''
        mod = 10**9+7
        def rec(arr):
            if len(arr) == 0:
                return 0
            if len(arr) == 1:
                return 1
            root = arr[0]
            smaller = []
            larger = []
            for i in range(1,len(arr)):
                if nums[i] < root:
                    smaller.append(arr[i])
                else:
                    larger.append(arr[i])
            left = rec(smaller)
            right = rec(larger)
            return (left*right + 1) % mod
        
        return rec(nums) % mod
    
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        '''
        fist eleemtn in nums will alwasy be the root of the bst tree
        let dfs(nums) be the number of perms of nums that result in the same BST
        we just keep calling nums[1:] recursively
        example
        [3,4,5,1,2]
            [3]
        [1,2] [4,5]
        
        we are free to swap to [2,1] and to [5,4]
        the realtive positions to nums[0] could be anything
        so really we can rearrange [1,2] and [4,5] any number of ways
        so we need to call dfs([1,2]) and dfs([4,5]) and do something to them to count the number of ways with
        nums[0] as the root
        
        dfs(nums)=dfs(left_nodes)⋅dfs(right_nodes) so far we have
        however, it is important to note that the actual numbers of valid pemrs may exceed the calculate number from above
        this is because there are some perms that do not alter the relative order in left and in right
        this given the same BST
        
        example
        [3,4,5,1,2]
            [3]
        [1,2] [4,5] for these two we can permute them 2 ways
        2*2 = 4, but we are missing 1, so we don't get all the ways here
        this implies that we need to adjust the formulate by multiplying it with a coeffient (P) that represents the number
        of permuttaions that preserve the relative order of nodes in left and right
        
        dfs(nums)=P⋅dfs(left_nodes)⋅dfs(right_nodes)
        
        it is possible to abritraily select two cells to hold the nodes in left and there are 6 permutions that generat the same
        left and right
        exmplae [3,_,_,_,_] where we want to place [4,5] in any two spots but we want to preserve the ordering of [4,5]
        can we [3,4,5,_,_], [3,4,_,5,_], [3,4,_,_,5], [3,_,4,5,_], [3,_,4,_,5], [3,_,_,4,5]
        so really its num_ways(left)*num_ways(right)*(something includes preserving the ordering)
        
        in generatl, for an array len(left) == m nodes in the left subtree
        the number of valid permutatinos == the numger of ways of selecting k cells from m-1 cells (first cell includes the root)
        (m-1) C left = (m-1,left) = (m-1)! / (left!(m-1-left))
        we can use python's builtin or use pascals triangle to find the binom coefficents
        generte a lenght m by m table
        then index into table[n][k] = n choose k 
        
        so we now have the transition equations
        dfs(nums)=P⋅dfs(left_nodes)⋅dfs(right_nodes) = (n C k)*dfs(left_nodes)*dfs(right_nodes)
        base case
            if the array is not more than 3 elements long, it only has one permutation to contruct the same BST (for that root)
          
         exmaple
         [5, 1, 8, 3, 7, 9, 4, 2, 6], we need to keep the relative order in [1, 3, 4, 2] and [8, 7, 9, 6] unchanged
         first coeffcient is (4 choose 8)
         on left [1,3,4,2]
         nothing can be on th left side, so return 1
         presrver order on right [3,4,2], coefficent now is 0 choose 3
         
         for the return, dont include the starting array itself
        '''
        mod = 10**9 + 7
        def dfs(nums):
            m = len(nums)
            if m < 3:
                return 1
            left_nodes = []
            right_nodes = []
            for num in nums:
                if num < nums[0]:
                    left_nodes.append(num)
                elif num > nums[0]:
                    right_nodes.append(num)
            
            left_count = dfs(left_nodes)
            right_count = dfs(right_nodes)
            binom_nCk = comb(m-1,len(left_nodes))
            ans = left_count*right_count*binom_nCk
            return ans % mod
            
#using pacals to find the binomial coefs
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        '''
        try building out pascals trianlge to compute the binomCoefficient
        
        '''
        #generate pascal's triangle
        m = len(nums)
        mod = 10**9 + 7
        pascals = [[0]*m for _ in range(m)]
        
        for i in range(m):
            pascals[i][0] = 1
            pascals[i][i] = 1
            
        for i in range(2,m):
            for j in range(1,i):
                pascals[i][j] = (pascals[i-1][j-1] + pascals[i-1][j]) % mod

        
        def dfs(nums):
            m = len(nums)
            if m < 3:
                return 1
            left_nodes = []
            right_nodes = []
            for num in nums:
                if num < nums[0]:
                    left_nodes.append(num)
                elif num > nums[0]:
                    right_nodes.append(num)
            
            left_count = dfs(left_nodes)
            right_count = dfs(right_nodes)
            binom_nCk = pascals[m-1][len(left_nodes)]
            ans = left_count*right_count*binom_nCk
            return ans % mod
        
        return (dfs(nums) - 1) % mod
    
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.right = None
        self.left = None
        self.size = 1
        
    
    def insert(self,node,val):
        if not node:
            self.root = TreeNode(val)
            return self.root
        else:
            node.size += 1
            if val < node.val:
                node.left = self.insert(node.left,val)
            else:
                node.right = self.insert(node.right,val)
            
            return self.root
        
class Solution:
    def numOfWays(self, nums: List[int]) -> int:
        '''
        we can actually try building the tree first starting with nums
        in the BST tree api, include the number of nodes in each each subtree
        if we are at a node (call it curr_node)
        and we have orderings on left and orderings on right
        we can do left*right orderings butt also to,es C, where C is the number of ways in which left and right can be interleaved
        
        
        Example 2 is good for thinking through this. At the root, both the left and right trees have one possible ordering. 
        For the whole tree, we just need to figure out the number of ways in which both arrays, of size 2, .can be interleaved, which is comb(4, 2) = 6. 
        The final solution is 1*1*6 - 1 = 5 (subtracting 1 since the problem asks for reorderings).

why we use comb(ls + rs, ls) and not comb(ls + rs, rs): The binomial coefficient (n k) is the same as (n n-k).
Which means comb(ls + rs, ls) == comb(ls + rs, rs).
        '''
        def insert(node,val):
            if not node:
                return TreeNode(val)
            else:
                node.size += 1
                if val < node.val:
                    node.left = insert(node.left,val)
                else:
                    node.right = insert(node.right,val)

                return node

        bst = None
        mod = 10**9 + 7
        for num in nums:
            bst = insert(bst,num)
        
        
        def dfs(node):
            if not node:
                return 1
            left_ways = dfs(node.left)
            right_ways = dfs(node.right)
            left_nodes = node.left.size if node.left else 0
            right_nodes = node.right.size if node.right else 0
            binom_coef = comb(left_nodes + right_nodes, left_nodes) #could else use right_nodes as (k)
            return left_ways*right_ways*binom_coef % mod
        
        return dfs(bst) - 1

##########################################
# 1187. Make Array Strictly Increasing
# 17JUN23
##########################################
#bleacghhh
class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        '''
        in on move i can pick two indices (i,j)
        and do the assigment arr1[i] = arr2[j], if course i and j must be in bounds
        greddily going left to right and searching for an element in arr2 where the is a violation in arr1 does not always work
        [1,5,3,6,7] and [4,3,1]
        going left to right we hit violation at 5, but looking for number between 1 and 3 in arr2, we don't
        but we could have swapped 5 with 3 if we had swapped the 3 with 4
        
        hint says state is
            index in arr1 and the index of the previous element in arr2 after sotring it and removing duplicate
        the arrays must be strictly increasing, so we can't reuse elements anyway
        it makes sense to sort arr2 and remove duplicates
        [1,5,3,6,7]
        [1,2,3,4]
        
        let dp(i,prev) be the min operations to make arr1[i:] increasing when arr1[i-1] > prev
        if arr[i] > prev, we can just go on to the next tep
            dp(i+1,arr[i]) #set prev to a new arr[i]
        
        if its not we need to find a replacement in arr2
            the replacement in nums 2 needs be just greater then prev
            so we can find the upper bound arr2 (i.e) the smallest element in arr2 just greater than prev
            if such an element is in arr2 we have a replacement event
            1 + dp(i+1,arr2[idex of element to be found])
            but here we just take the minimum of out two options
            
        cases1:
            if arr[i] < arr[i-1], we must replace with the smallest number in arr2 just greater than arr[i-1]
            binar seach
        
        case2.a,.2b
            if arr[i] > arr[i-1]
                leave it and advance
            try replacing it with an even emsaller value. why?
            by doing so itt may make it easier to ensure that subsequent numbers are greater than arr1[i]
        
        in both cases we also look for the upper bound from arr[i-1]
        
        importantn step
            if we canont replace arr[i] with a value from arr2, need to return a larger number float('inf')
            to indicate that it cannot be replace
        '''
        arr2 = sorted(list(set(arr2)))
        memo = {}
        
        def dp(i,prev):
            if i == len(arr1):
                return 0
            if (i,prev) in memo:
                return memo[(i,prev)]
            
            curr_min_cost = float('inf')
            #if the array is sorted at this arr[i] and prev
            if arr1[i] > prev:
                curr_min_cost = dp(i+1,arr1[i])
            
            #try finding largest
            upper_bound = bisect.bisect_right(arr2,prev)
            #if we can replace it
            if upper_bound < len(arr2):
                curr_min_cost = min(curr_min_cost, 1 + dp(i+1,arr2[upper_bound]))
                
            memo[(i,prev)] = curr_min_cost
            return curr_min_cost
        ans = dp(0,float('-inf'))
        
        if ans != float('inf'):
            return ans
        return -1
    
#starting to get the hang of upper bound
class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        '''
        coding out binary search 
        '''
        arr2 = sorted(list(set(arr2)))
        memo = {}
        
        def bin_search(arr,search):
            left = 0
            right = len(arr)
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] <= search:
                    left = mid + 1
                else:
                    right = mid
            
            return left
        
        def dp(i,prev):
            if i == len(arr1):
                return 0
            if (i,prev) in memo:
                return memo[(i,prev)]
            
            curr_min_cost = float('inf')
            #if the array is sorted at this arr[i] and prev
            if arr1[i] > prev:
                curr_min_cost = dp(i+1,arr1[i])
            
            #try finding largest
            upper_bound = bin_search(arr2,prev)
            #if we can replace it
            if upper_bound < len(arr2):
                curr_min_cost = min(curr_min_cost, 1 + dp(i+1,arr2[upper_bound]))
                
            memo[(i,prev)] = curr_min_cost
            return curr_min_cost
        ans = dp(0,float('-inf'))
        
        if ans != float('inf'):
            return ans
        return -1
    
#another cool way with boundar conditins
#https://leetcode.com/problems/make-array-strictly-increasing/discuss/3648097/Python-Elegant-and-Short-or-Top-Down-DP-or-Binary-Search
class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        arr2.sort()
        memo = {}
        
        def dp(i,prev_max):
            if i == len(arr1):
                return 0
            if (i,prev_max) in memo:
                return memo[(i,prev_max)]
            
            j = bisect.bisect_right(arr2,prev_max)
            take = dp(i+1,arr2[j])  + 1 if j < len(arr2) else float('inf')
            no_take = dp(i+1,arr1[i]) if arr1[i] > prev_max else float('inf')
            ans = min(take,no_take)
            memo[(i,prev_max)] = ans
            return ans
        
        return dp(0,float('-inf'))
    
#bottom up is trickier, and needs slight modification
class Solution:
    def makeArrayIncreasing(self, arr1: List[int], arr2: List[int]) -> int:
        '''
        for bottom up we need to use a mapp of the form
            (i:{prev:count}) where prev i the prev value and cound is th minimum operations to reach this state
            we can't do the thing where we loop through all i and through all prev from prev to max_prev
            why? we don't know how many prevs can exists, other than the fact the we need to seach for prev in arr2
        
        loop through all indices i, and for all states in dp
            each state in dp is {prev:count}
            
        again:
            if arr1[i] <= prev, wemust replace arr1[i] with the smallest value in arr2 just greart than prev
            this new state becomes {arr2[index for bin seach] : count + 1}
            otherwise we can't update, so we just return float('inf')
        
        if arr[i] > prev:
            leave it unchanged {arr[i]:count} -> new_dp
            try replacing it with a smaller value, i.e the smallest value > prev: {arr2[index for bin search]: count + 1}
            then we take the ,minimum
        
        then we just reassign dp's
        
        
            
        '''
        #boundary conditions, we have min count 0 for the start of the array
        dp = {-1:0}
        arr2.sort()
        N = len(arr2)
        
        for i in range(len(arr1)):
            next_dp = collections.defaultdict(lambda: float('inf')) #set default value to float('inf') it not in hahsmap
            for prev in dp:
                if arr1[i] > prev:
                    next_dp[arr1[i]] = min(dp[prev],next_dp[arr1[i]])
                #try binary search
                idx = bisect.bisect_right(arr2,prev)
                if idx < len(arr2):
                    next_dp[arr2[idx]] = min(dp[prev] + 1, next_dp[arr2[idx]])
            
            dp = next_dp
            
        return min(dp.values()) if dp else -1
    
###################################################
# 2328. Number of Increasing Paths in a Grid
# 18JUN23
###################################################
#YESSSS
class Solution:
    def countPaths(self, grid: List[List[int]]) -> int:
        '''
        if i let dp(i,j) be the number of increasing paths starting from cell (i,j)
        dp(i,j) = {
                for (neigh_i,neigh_j) in neighbors of (i,j):
                    if grid[i][j] < grid[neigh_i][neigh_j]:
                        dp(i,j) += dp(neigh_i,neigh_j)
        }
        
        do dp(i,j) for all (i,j) in rows, then sum for each
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        mod = 10**9 + 7
        
        memo = {}
        
        def dp(i,j):
            #we don't need a termination case go out of bounds, because we will check before recursing below
            if (i,j) in memo:
                return memo[(i,j)]
            ans = 1
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #increasing
                    if grid[neigh_x][neigh_y] > grid[i][j]:
                        ans += dp(neigh_x,neigh_y)
                        ans %= mod
            
            #one more time
            ans %= mod
            memo[(i,j)] = ans
            return ans
        
        
        paths = 0
        for i in range(rows):
            for j in range(cols):
                paths += dp(i,j)
                paths %= mod
        
        return paths % mod
            
class Solution:
    def countPaths(self, grid: List[List[int]]) -> int:
        '''
        to do bottom up, we need to sort the cells by their value
        and we cna keep a temp dp arry
        where dp[i][j] represents the number of paths ending at dp[i][j]
        then we check all in bound neighs (neigh_i,neigh_j)
        and if grid[neigh_i][neigh_j] > grid[i][j]
            we cane exxtend all paths to neigh_i,neigh_j, so we weincremant by dp[i][j]
        '''
        rows = len(grid)
        cols = len(grid[0])
        mod = 10**9 + 7
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        
        #counts
        dp = [[1]*cols for _ in range(rows)]
        cells = []
        for i in range(rows):
            for j in range(cols):
                cells.append([grid[i][j],i,j])
        
        #sort increasingly by value
        cells.sort(key = lambda x: x[0])
        for val,i,j, in cells:
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #increasing
                    if grid[neigh_x][neigh_y] > grid[i][j]:
                        dp[neigh_x][neigh_y] += dp[i][j]
                        dp[neigh_x][neigh_y] %= mod
                        
        ans = 0
        for i in range(rows):
            for j in range(cols):
                ans += dp[i][j]
                ans %= mod
        
        return ans % mod
    
#more pythonic
class Solution:
    def countPaths(self, grid: List[List[int]]) -> int:
        '''
        to do bottom up, we need to sort the cells by their value
        and we cna keep a temp dp arry
        where dp[i][j] represents the number of paths ending at dp[i][j]
        then we check all in bound neighs (neigh_i,neigh_j)
        and if grid[neigh_i][neigh_j] > grid[i][j]
            we cane exxtend all paths to neigh_i,neigh_j, so we weincremant by dp[i][j]
        '''
        rows = len(grid)
        cols = len(grid[0])
        mod = 10**9 + 7
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        
        #counts
        dp = [[1]*cols for _ in range(rows)]
        cells = [[i,j] for i in range(rows) for j in range(cols)]
        cells.sort(key = lambda x: grid[x[0]][x[1]])

        for i,j in cells:
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #increasing
                    if grid[neigh_x][neigh_y] > grid[i][j]:
                        dp[neigh_x][neigh_y] += dp[i][j]
                        dp[neigh_x][neigh_y] %= mod
                        
        ans = 0
        for i in range(rows):
            for j in range(cols):
                ans += dp[i][j]
                ans %= mod
        
        return ans % mod