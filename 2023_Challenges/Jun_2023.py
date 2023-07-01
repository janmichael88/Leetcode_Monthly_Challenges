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
    
class Solution:
    def countPaths(self, grid: List[List[int]]) -> int:
        MOD = 10 ** 9 + 7
        R,C = len(grid), len(grid[0])
        dp = defaultdict(int)
        for val,r,c in sorted(((grid[r][c],r,c) for r,c in product(range(R), range(C))), reverse=True):
            for dr, dc in (0,1),(1,0),(0,-1),(-1,0):
                nr,nc = r+dr,c+dc
                if R > nr >= 0 <= nc < C and grid[r][c] < grid[nr][nc]:
                    dp[r,c] += 1 + dp[nr,nc]
                    dp[r,c] %= MOD
        return (R * C + sum(dp.values())) % MOD
    
###############################################################
# 158. Read N Characters Given read4 II - Call Multiple Times
# 19JUN23
#################################################################
# The read4 API is already defined for you.
# def read4(buf4: List[str]) -> int:

class Solution:
    def __init__(self,):
        self.internal_buffer = ['']*4
        self.buffer_ptr = 0
        self.characters_in_buffer = 0
    '''
    https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/discuss/989807/Clean-Python-solution
    * the idea is to read from the file into an internal buffer,
    * store number of characters in the internal buffer
    * keep track of the current interal buffer position
    
    for each
        use local varibale to keep track of the number of copied charcters (from file into internal buffer)
        whiel its lees than n, copy characters from interal buffer into the our buffer
        move both pointers
    
    if the current buffer pointer to char == the size of the buffer
        we need to read another chunk from the file using read4
        ipdate and reset the poitners
    '''
    def read(self, buf: List[str], n: int) -> int:
        #recall we arte trying to read in n characters
        i = 0
        while i < n:
            #if we need to read in a new set of 4
            if self.buffer_ptr == self.characters_in_buffer:
                self.buffer_ptr = 0
                self.characters_in_buffer = read4(self.internal_buffer)
                if self.characters_in_buffer == 0:
                    #nothing else to read
                    break
            #write into file buffer
            buf[i] = self.internal_buffer[self.buffer_ptr]
            self.buffer_ptr += 1
            i += 1
        
        return i
    
#an additional way
# The read4 API is already defined for you.
# def read4(buf4: List[str]) -> int:

class Solution:
    
    def __init__(self,):
        self.buf4 = ["" for _ in range(4)]
        #pointer in buffer
        self.ptr = 0
        self.n = 0 #spaces used in buffer
        
    def read(self, buf: List[str], n: int) -> int:
        ans = 0
        
        for i in range(n):
            if self.ptr >= self.n:
                #read next check
                r = read4(self.buf4)
                #reset
                self.ptr = 0
                self.n = r
            
            if self.ptr < self.n:
                buf[i] = self.buf4[self.ptr]
                self.ptr += 1
                ans += 1
            else:
                break
        
        return ans
    
#Google follow up, speed up read
#essentially the same as the first one
#maybe brute force would have been to call read4, create temp, and copy over
#here we copy on the fly

#https://leetcode.com/problems/read-n-characters-given-read4-ii-call-multiple-times/discuss/188293/Google-follow-up-question.-Speed-up-the-copy.
class Solution:
    def __init__(self):
        self.buf4 = [''] * 4
        self.i4 = 0
        self.n4 = 0
        
    def read(self, buf: List[str], n: int) -> int:
        idx = 0
        
        while idx < n:
            if self.i4 < self.n4:
                buf[idx] = self.buf4[self.i4]
                idx += 1
                self.i4 += 1
            else:
                self.n4 = read4(self.buf4)
                self.i4 = 0

                if not self.n4:
                    return idx

        return idx
    
##################################
# 2090. K Radius Subarray Averages
# 20JUN23
##################################
#jesus
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        '''
        i can build prefix sum, then i should be able to find the subarray sums in constant time
        then divide by k for each i
        just remmber for prefix sum array with [0] at the beginning
        given points (i,j)
        sum us pref_sum[j+1] - pref_sum[i]
        '''
        N = len(nums)
        pref_sum = [0]
        for num in nums:
            pref_sum.append(num + pref_sum[-1])
        
        
        ans = [-1]*N
        
        for i in range(N):
            #get the bounds
            left_bounds = i - k
            right_bounds = i + k
            if left_bounds >= 0 and right_bounds < N:
                #print(temp)
                #print(nums[left_bounds],nums[right_bounds])
                #find sum
                curr_sum = pref_sum[right_bounds+1] - pref_sum[left_bounds]
                num_elements = right_bounds - left_bounds + 1
                #print(temp)
                #print(curr_sum,sum(temp),num_elements)
                ans[i] = curr_sum // num_elements

        
        return ans

class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        '''
        for any K radius subarray in nums, there is going to be 2*k + 1 eleemnts in that array
        if there are not, this avearge will be - 1
        
        notes on pref_sum array
            given n elements, we get pref_sum array of size n + 1
            given (left,right), we can find sum as : pref_sum[right+1] - pref_sum[left]
        
        special cases to spee up
        if k == 0, we only consinder average of subarrays of size 1, the average of a sigle eleemnt is just the elemnet itself
            return nums
        
        if 2*k + 1 > n, which means we have to find more than n numbers of the left and right for in i in range [0,n-1]
        just return [-1]*n
        otherwise we can check all i for i in range [k to n-k]
        
        '''
        N = len(nums)
        if k == 0:
            return nums
        
        ans = [-1]*N
        if 2*k + 1 > N:
            return ans
        
        pref_sum = [0]*(N+1)
        for i in range(N):
            pref_sum[i+1] = pref_sum[i] + nums[i]
            
        #centers from k to n - k
        for i in range(k,N-k):
            left = i - k
            right = i + k
            entry = (pref_sum[right+1] - pref_sum[left]) // (2*k + 1)
            ans[i] = entry
        
        return ans
            
#sliding window for O(1) space
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        '''
        we can using a sliding window
        we know there must be 2*k + 1 elements in a subarray
        say we are at some index x and we knows is K radius subarray, call it S_x
        when we go to the next x+1, we just add the next element in right and remove left most window
        S_{x+1} = S_{x} + nums[k+x+1] - nums[k-x+1]
        '''
        N = len(nums)
        if k == 0:
            return nums
        
        ans = [-1]*N
        if 2*k + 1 > N:
            return ans
        
        #get the first window sum
        window_sum = sum(nums[:2*k+1])
        ans[k] = window_sum // (2*k+1)
        
        #iterate on the next ending indicies, i.e right
        #from a right bounds, we can find its left start
        for right in range(2*k+1,N):
            window_sum += nums[right] - nums[right - (2*k+1)]
            ans[right-k] = window_sum//(2*k+1)
        
        return ans

#another way
class Solution:
    def getAverages(self, nums: List[int], k: int) -> List[int]:
        N = len(nums)
        ans = [-1]*N
        
        running_sum = 0
        left = 0
        for right,num in enumerate(nums):
            running_sum += num
            if right >= 2*k:
                ans[left+k] = running_sum // (2*k+1)
                running_sum -= nums[left]
                left += 1
        
        return ans

############################################
# 2448. Minimum Cost to Make Array Equal
# 21JUN23
#############################################
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        '''
        in one operation we can increase or decrease any element in nums by 1
        the cost of doing this at nums[i] is costs[i]
        return min cost such that all array nums become equal
        
        its optimal to try changing all the elements to an element existing in nums already
        
        tryingg all posbile nums to raise to would be
            O(N^2)
        say we have elements: [a,b,c,d,e]
        we know we need to raise each element to any one of them,
        if we try a, then the number of steps for the remaining would be abs(b-a), abs(c-a)....abs(e-a)
        and cost would be abs(b-a)*cost[1] + abs(c-a)*cost[2] + ... _ abs(e-a)*cost[-1]
        
        what if we were to sort (nums[i],cost[i]) for i in range(n), on increasing cost
        if we know the current cost to changing nums[i] to x, lets call it total cost
        if we were to chagne x to dx, then the cost would increas by dx*(total_cost)
        
        intutions
            nums = [1,3,5,2]
            cost = [2,3,1,14]
            cost to change to 3 would be
            abs(3-2)*2 + abs(3-3)*3 + abs(5-3)*2 + abs(3-1)*14 = 20
            no try raising to 5
            abs(5-1)*2 + abs(5-3)*3 + abs(5-5)*0 + abs(5-2)*14 = 56
            F(x) = \sum_{i=0}^{N} abs(nums[i]-x)*cost[i]
            now try increasin x by something x' = x + dx
           F(x+dx) = \sum_{i=0}^{N} abs(nums[i]-(x+dx))*cost[i] 
           F(x+dx) = F(x) + \sum_{i=0}^{n} (dx*cost[i])
           depending on the direction of dx, could be positive or could be negative
           if we had sorted the array, and we know the current cost for changing to nums[i], then we can get the change from nums[i-1]        
        two different costs
            1. the cost of increasing numbers smaller than nums[i] is pref_sum up to i
            2. the cost of decreasing largers numbers than nums[i] would be suffix_sum at i to the end of the array
        def compute_cost(nums,cost,x):
            N = len(nums)
            total_cost = 0
            for i in range(N):
                total_cost += abs(nums[i] - x)*cost[i]
            
            return total_cost
        
        print(compute_cost(nums,cost,5))
        print(compute_cost(nums,cost,6))
        print(sum(cost))
        
        intution:
            the increament of the cost is proporital to the pref_sum (or suff_sum) of cost at i
         
        why do we only need to pick from an existing element in nums?
            F(x) is montonic when x is in the range nums[i] and nums[i+1]
            F(x) is motnoic when the array is sorted
        why does F(x) being montonic imply that the base must be an element in nums?
        
        algo:
            1. pair num with cost and sort increaisnly on cost
            2. build pref_sum array for the sorted costs
            3. start with nums[0], as the bast, and calculate the cost of makgin evey element euqal to nums[0]
            4. then iteratinf from nums[1] to nums[len(nums)-1]
                to get the delta x change in costs
                minimize each delta x
                there will be a delta x for (0,1), (1,2),(2,3)....all the way to (N-2,N-1)
        
        intution 2:
            for the function F(x), start with the smallast x in nums
            to raise x to x', which would be the next largest element in nums
            find delta x = x' - x
            The current totalCost made by nums[i], compared with the previous cost made by nums[i - 1], 
            is increased by gap times the prefix sum of costs prefixCost[i - 1] and decreased by gap 
            times the suffix sum of costs prefixCost[n - 1] - prefixCost[i - 1].

        Since F(x) is monotonic for a range nums[i] <= x <= nums[j], where i, j are just random indices, it must mean that F(x) is a minimum at either nums[i] or nums[j]. 
        Think about it, if a function monotonic increasing, you start at the top and end at the bottom. 
        If a function is monotonic decreasing, you start at the bottom and end at the top. In both scenarios, 
        the bottom is at one of the endpoints.

        So we are trying to find the x that makes F(x) at minimum, and since the function is monotonic on the range [i,j] where nums[i] <= x <= nums[j], 
        the minimum must be at one of the endpoints i or j. Anything in between would not make F(x) the minimum.
            
        '''
        def compute_cost(nums,cost,x):
            N = len(nums)
            total_cost = 0
            for i in range(N):
                total_cost += abs(nums[i] - x)*cost[i]
            
            return total_cost
        
        #print(compute_cost(nums,cost,5))
        #print(compute_cost(nums,cost,6))
        #print(sum(cost))
        
        N = len(nums)
        pairs = [(num,c) for num,c in zip(nums,cost)] #(num[i],cost[i])
        #sort increasinly on nums
        pairs.sort(key = lambda x: x[0])
        
        pref_sum_cost = [0]*(N+1)
        for i in range(N):
            pref_sum_cost[i+1] = pref_sum_cost[i] + pairs[i][1]
        
        #then we try every interger and change ot ti nums[0]
        starting_cost = 0 
        for i in range(1,N):
            cost_at_i = pairs[i][1]
            diff = pairs[i][0] - pairs[0][0]
            starting_cost += cost_at_i*diff
        
        ans = starting_cost
        
        #thenw e try nums[1] and nums[2] and so on
        #the cost difference is made by the cnage of two parts 1. pref_sum of costs 2. suff_sum of costs
        #print(pref_sum_cost)
        for i in range(1,N):
            delta_x = pairs[i][0] - pairs[i-1][0]
            pref_sum = pref_sum_cost[i]
            suff_sum = pref_sum_cost[-1] - pref_sum_cost[i]
            starting_cost += pref_sum*delta_x
            starting_cost  -= suff_sum*delta_x
            ans = min(ans,starting_cost)
        
        return ans
    
#binary search
class Solution:
    def minCost(self, nums: List[int], cost: List[int]) -> int:
        '''
        binary search approach
        intution:
            a linear combination (with) non negative coeffients of convex functinos is convex
            a function is convex if a line segment between any two points lies above the graph
            rather, the second derivative must be non negative in its entire domain
            i.e must be positive semi definite
            
        if we define f_i{x} as the cost function for one elements nums[i]
        then f_i(x) is convex
        
        if nums consists of multiple elements, F(X) = f1(x) + f2(x) + fx(x) + ... + fn(x)
        which means F(x) is also convent, and the minmum must be in range [min(nums),max(nums)]
        we can use binary search to find the min by comparing F(x) and F(x+1)
        
        case1:
            if F(x) < F(x+1), we can i.e its increasing, we discard the right half
            if F(x) > F(x+1), its decreasing, and we discard the left half
        '''
        # Get the cost of making every element equals base.
        def get_cost(base):
            return sum(abs(base - num) * c for num, c in zip(nums, cost))
        
        # Initialize the left and the right boundary of the binary search.
        left, right = min(nums), max(nums)
        answer = get_cost(nums[0])
        
        # As shown in the previous picture, if F(mid) > F(mid + 1), then the minimum
        # is to the right of mid, otherwise, the minimum is to the left of mid.
        while left < right:
            mid = (left + right) // 2
            cost_1 = get_cost(mid)
            cost_2 = get_cost(mid + 1)
            answer = min(cost_1, cost_2)
            
            if cost_1 > cost_2:
                left = mid + 1
            else:
                right = mid
        
        return answer
            
############################################################
# 714. Best Time to Buy and Sell Stock with Transaction Fee
# 22JUN23
############################################################
#YAYYYYYYY
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        '''
        can only hold 1 stock at a time
        states:
            position
            holding and not holding stock
        
        transition
            if im holding stock at i, i can sell on this day which means i go up in price, but i lose profit due to fee
            if im holding stock at i, i can just do nothing and move on
            
            if im not holding stock at i, i can continue not holding
            if im not holding stock at i, buy stock at i and go down in price
        '''
        memo = {}
        N = len(prices)
        
        def dp(i,holding):
            if i == N:
                return 0
            if (i,holding) in memo:
                return memo[(i,holding)]
            
            ans = float('-inf')
            #holding stock then buy
            if holding:
                option1 = prices[i] - fee + dp(i+1,False) #no longer holding
                option2 = dp(i+1,True) #do nothing and still else
                ans = max(ans,option1,option2)
            
            else:
                #not holding 
                option3 = dp(i+1,False)
                option4 = -prices[i] + dp(i+1,True)
                ans = max(ans,option3,option4)
            
            memo[(i,holding)] = ans
            return ans
        
        
        return dp(0,False)
    
#bottom up
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        '''
        bottom up,
        dp arrays of size N+1, two of them
        '''
        N = len(prices)
        dp = [[0]*(2) for _ in range(N+1)]
        
        for i in range(N-1,-1,-1):
            ans = float('-inf')
            for holding in [0,1]:
                if holding:
                    option1 = prices[i] - fee + dp[i+1][0]
                    option2 = dp[i+1][1]
                    ans = max(ans,option1,option2)
                else:
                    option3 = dp[i+1][0]
                    option4 = -prices[i] + dp[i+1][1]
                    ans = max(option3,option4)
                
                dp[i][holding] = ans
        
        return dp[0][0]

#########################
# 1214. Two Sum BSTs
# 22JUN23
########################
#meh it works
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def twoSumBSTs(self, root1: Optional[TreeNode], root2: Optional[TreeNode], target: int) -> bool:
        '''
        dump one tree into a hashset than traverse the other tree and check for its complement
        '''
        hash_set1 = set()
        self.ans = False
        def dfs(node,curr_hash_set):
            if not node:
                return
            curr_hash_set.add(node.val)
            dfs(node.left,curr_hash_set)
            dfs(node.right,curr_hash_set)
        
        
        def dfs_2(node,curr_hash_set):
            if not node:
                return
            #find complement
            comp = target - node.val
            if comp in curr_hash_set:
                self.ans = True
            dfs_2(node.left,curr_hash_set)
            dfs_2(node.right,curr_hash_set)
            
        dfs(root1,hash_set1)
        dfs_2(root2,hash_set1)
        return self.ans

##############################################
# 1027. Longest Arithmetic Subsequence
# 23JUN23
#############################################
#dammit this sucks
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        '''
        states:
            need position in array, call it i
            at that position keep track of the arithmetic difference, call it prev_diff
            need a way to check prev_diff
        
        transition:
            we either take at this i to extend if there is a common differnce
            or we don't take and reset to 1
        
        if im at i, i need to check for the existence of nums[i] - nums[i-1] for a start index j
        beteween (0 and i-1)
        
        start off at index 1
        
        '''
        memo = {}
        N = len(nums)
        
        def dp(i,prev):
            if i == N:
                return 0
            if (i,prev) in memo:
                return memo[(i,prev)]
            #if we can extend
            extend = 1 + dp(i+1,nums[i]-nums[i-1])
            skip = dp(i+1,prev)
            ans = 0
            #check the other ones
            for j in range(i):
                child = 1 + dp(j,nums[i]-nums[j])
                ans = max(ans,child,extend,skip)
            
            memo[(i,prev)] = ans
            return ans
        
        
        return dp(1,nums[1]-nums[0])
        
#python TLE's but hava does not
#O(N^3)
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        '''
        states:
            need position in array, call it i
            at that position keep track of the arithmetic difference, call it prev_diff
            need a way to check prev_diff
        
        transition:
            we either take at this i to extend if there is a common differnce
            or we don't take and reset to 1
        
        if im at i, i need to check for the existence of nums[i] - nums[i-1] for a start index j
        beteween (0 and i-1)
        
        this would be longest starting i
        
        start off at index 1
        
        '''
        memo = {}
        N = len(nums)
        
        def dp(i,prev):
            if i == N:
                return 0
            if (i,prev) in memo:
                return memo[(i,prev)]
            ans = 0
            #check the other ones
            for j in range(i+1,N):
                #find the next idff
                next_diff = nums[j] - nums[i]
                if next_diff == prev:
                    ans = max(ans,1+dp(j,prev))
            
            memo[(i,prev)] = ans
            return ans
        
        ans = 0
        #smallest can be length 2
        for i in range(N):
            for j in range(i+1,N):
                start_diff = nums[j] - nums[i]
                temp = 1 + dp(i,start_diff)
                ans = max(ans,temp)
        
        return ans

#top down O(N^2), it only gets AC with java though
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        '''
        states:
            need position in array, call it i
            at that position keep track of the arithmetic difference, call it prev_diff
            need a way to check prev_diff
        
        transition:
            we either take at this i to extend if there is a common differnce
            or we don't take and reset to 1
        
        if im at i, i need to check for the existence of nums[i] - nums[i-1] for a start index j
        beteween (0 and i-1)
        
        this would be longest starting i
        
        start off at index 1
        
        '''
        memo = {}
        N = len(nums)
        
        def dp(i,prev):
            if i == N:
                return 0
            if (i,prev) in memo:
                return memo[(i,prev)]
            ans = 1
            
            if prev == float('-inf'):
                for j in range(i+1,N):
                    extend = 1 + dp(j,nums[j]-nums[i])
                    no_extend = dp(j,prev)
                    ans = max(ans,max(extend,no_extend))
            else:
                for j in range(i+1,N):
                    if nums[j] - nums[i] == prev:
                        extend = 1 + dp(j,prev)
                        ans = max(ans,extend)
            
            memo[(i,prev)] = ans
            return ans
        
        
        return dp(0,float('-inf'))

#need to do bottom up with dp as hash table
class Solution:
    def longestArithSeqLength(self, nums: List[int]) -> int:
        '''
        bottom up, dp(i,prev)
            represents the longest arightmetic subsequence ending at i, with common difference prev
        '''
        dp = {}
        N = len(nums)
        
        for right in range(N):
            for left in range(0,right):
                diff = nums[right] - nums[left]
                if (left,diff) in dp:
                    dp[(right,diff)] = dp[(left,diff)] + 1
                else:
                    dp[(right,diff)] = 2

                    
        
        return max(dp.values())
    
##########################################
# 956. Tallest Billboard
# 25JUN23
##########################################
#yasss
class Solution:
    def tallestBillboard(self, rods: List[int]) -> int:
        '''
        keep track to two different sums
        for each each we can either add to sum1 or sum2 or skip
        if we got to the end, check the sums
        '''
        memo = {}
        N = len(rods)
        
        def dp(i,sum1,sum2):
            if i == N:
                if sum1 == sum2:
                    return 0
                else:
                    return float('-inf')
            
            if (i,sum1,sum2) in memo:
                return memo[(i,sum1,sum2)]
            
            op1 = rods[i] + dp(i+1,sum1+rods[i],sum2)
            op2 = rods[i] + dp(i+1,sum1,sum2+rods[i])
            op3 = dp(i+1,sum1,sum2)
            ans = max(op1,op2,op3)
            memo[(i,sum1,sum2)] = ans
            return ans
        
        
        return dp(0,0,0) // 2

#state reudction to 2
#if we know one sum, we inriscally know the other
#the sum actually represents the sum of both stacks, do we need to divide by two
class Solution:
    def tallestBillboard(self, rods: List[int]) -> int:
        '''
        i want to make the tallest billboard, i need at least two rods to make a biull board
        all rods have positive length
        brute force:
            ans = 0
            for subset_sum in all subset_sums:
                if subset_sum === sum(rodes) - subset_sum:
                    ans = max(ans,subset_sum)
            you don't need to use all the rods
            
        maintain states 
            i, positoin, and difference between two susbet sums
            if difference is zero, we have an an answer

        state reduction is tricky, FYI
            
        '''
        memo = {}
        N = len(rods)
        
        def dp(i,diff):
            if i == N:
                if diff == 0:
                    return 0
                else:
                    return float('-inf')
            
            if (i,diff) in memo:
                return memo[(i,diff)]
            
            op1 = rods[i] + dp(i+1,diff + rods[i])
            op2 = rods[i] + dp(i+1, diff - rods[i])
            skip = dp(i+1,diff)
            ans = max(op1,op2,skip)
            memo[(i,diff)] = ans
            return ans
        
        return dp(0,0) // 2
    
#brute force, but with reduction in input by constant
class Solution:
    def tallestBillboard(self, rods: List[int]) -> int:
        '''
        brute force involves finding subets for
            using a subset on the left stand
            using a subset on the right stand
            or not using a stand at all
            
            which would be O(3^n)
            given the inputs 3^20 is too much, but if we would get down to
            3^10, it would be work, for stuff with exponential time, try to find a way to reduce by a constant at least; constant in the input
            we need to divide rods into halves
        
        intuition is to split the rods in two halves
        then we start from states (0,0) and we essentially BFS from them
            for r1 we can
                use in left (r1,0)
                use in right (0,r1)
                not use at all (0,0)
            
            for r2, we need to add to the exisiting stgates, brancing factor of three
                states:
                    {(0, 0), (r1, 0), (0, r1), (r2, 0), (0, r2), (r2 + r1, 0), (r1, r2), (r2, r1), (0, r2 + r1)}
                this will grow, but the point was the we limited the exponent to 10, which is acceptable
        
        given the states, how can we find the anser
            say we have some left combinations with height 5, and some right combinations with height 2
            left is higher by 3, so we need to find a combidnation giving 3 extra
        
        so lets store combindations in the first half, where the keys are:
            diff = left - right
        the value should be either the left or right rode height
            an answer for a combindation between two halvews owuld be either the left or the right
           
         we stotre first_half[left-right] = left
         second_half must be the same: second_half[left-right] = left
         
         after builing the hashmapes for the left and right halves we traverse over the first half
         and for each combindation represetned as first_half[diff] = left
         we check whether the second half contains the combiantion with diff to componestate
         if it does we take irst_half[diff] + second_half[-diff] as a valid billboard height

         if we know the height difference between the left and right
         and the -heigh difference exsists in the other side, combining thme both wild give the same heights for both left and right
        '''
        # Helper function to collect every combination `(left, right)`
        def helper(half_rods):
            states = set()
            states.add((0, 0))
            for r in half_rods:
                new_states = set()
                for left, right in states:
                    new_states.add((left + r, right))
                    new_states.add((left, right + r))
                states |= new_states
                
            dp = {}
            for left, right in states:
                dp[left - right] = max(dp.get(left - right, 0), left)
            return dp

        n = len(rods)
        first_half = helper(rods[:n // 2])
        second_half = helper(rods[n // 2:])

        answer = 0
        for diff in first_half:
            if -diff in second_half:
                answer = max(answer, first_half[diff] + second_half[-diff])
        return answer

########################################
# 1575. Count All Possible Routes
# 25JUN23
########################################
class Solution:
    def countRoutes(self, locations: List[int], start: int, finish: int, fuel: int) -> int:
        '''
        we are given locations of cities where locations[i] is hte positions
        we are given start city and finish city
        count all paths from start to finisght
        i can visit any city more than once
        
        define dp(i,fuel) as:
            the number count of all possilble paths
            if i is the finish city, there is a path, so return 1
            then we just need to add all the paths from there
            if fuel < 0:
                return 0
        '''
        memo = {}
        N = len(locations)
        mod = 10**9 + 7
        
        def dp(i,fuel):
            if fuel < 0:
                return 0
            if (i,fuel) in memo:
                return memo[(i,fuel)]
            
            #dont just return 1 on any leaf node, theres at least 1 if we hit it
            ways = 1 if i == finish else 0
            for j in range(N):
                if j != i:
                    used_fuel = abs(locations[i] - locations[j])
                    ways += dp(j, fuel - used_fuel)
                    ways %= mod
            ways %= mod
            memo[(i,fuel)] = ways
            return ways
        
        
        return dp(start,fuel)
    
#############################################
# 2462. Total Cost to Hire K Workers
# 26JUN23
#############################################
#fuck....
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        '''
        we are given array costs, where costs[i] is cost of hiring the ith work
        conditions:
            run k session and hire exactly one worker from each sessions
            in each session:
                choose the worker with the lowest cost from either the first canddiate workes ro the last canddiat workers
                break the two by the smallest indext
                
        maintain two heaps one for the left and right parts
        '''
        left_half = []
        right_half = []
        remaining = []
        ptr_rem = 0
        N = len(costs)
        
        for i in range(candidates):
            #combine entries into string????
            heapq.heappush(left_half, str(costs[i])+"_"+str(i))
            heapq.heappush(right_half, str(costs[N-i-1])+"_"+str(N-i-1))
            
        #anything remaining
        for i in range(candidates,N-candidates):
            remaining.append(str(costs[i])+"_"+str(i))
        
        
        ans = 0
        while k < 0:
            #three cases
            #no left
            if len(left_half) == 0:
                entry = heapq.heappop(right_half)
                cost,index = entry.split("_")
                ans += int(cost)
                if ptr_rem < len(remaining):
                    heapq.heappush(right_half,remaining[ptr_rem])
                    ptr_rem += 1
            #no right 
            elif len(right_half) == 0:
                entry = heapq.heappop(left_half)
                cost,index = entry.split("_")
                ans += int(cost)
                if ptr_rem < len(remaining):
                    heapq.heappush(left_half,remaining[ptr_rem])
                    ptr_rem += 1
            #find min from havles
            else:
                cost_left,index_left = left_half[0]
                cost_right,index_right = right_half[0]
                if cost_
                
#two heaps with deque of remaining
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        '''
        be careful for not taking into the right side, what we have already taken into the left side
        we don't need to keep track of the indinces if we split between left and right parts
        if there is a tie, we know we have to pick from the left side because the indices on left are before right
        so we just check the tops of the two heaps <=
        '''
        left_half = []
        right_half = []
        remaining = deque([])
        ptr_rem = 0
        N = len(costs)
        
        for i in range(candidates):
            #combine entries into string????
            left_half.append(costs[i])
        for i in range(max(candidates, len(costs) - candidates),N):
            right_half.append(costs[N-i-1])
            
        heapq.heapify(left_half)
        heapq.heapify(right_half)
        
        #anything remaining
        for i in range(candidates,N-candidates):
            remaining.append(costs[i])
        
        ans = 0
        while k > 0:
            #epty right half, take from left, or left_half[0] <= right_half
            if len(right_half) == 0 or len(left_half) > 0 and left_half[0] <= right_half[0]:
                ans += heapq.heappop(left_half)
                #rebalane the heaps
                if len(remaining) > 0:
                    heapq.heappush(left_half, remaining.popleft())
                    
            else:
                #right side
                ans += heapq.heappop(right_half)
                if len(remaining) > 0:
                    heapq.heappush(right_half,remaining.pop())
            
            k -= 1
        
        return ans

#two pointers in remaning
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        # head_workers stores the first k workers.
        # tail_workers stores at most last k workers without any workers from the first k workers. 
        head_workers = costs[:candidates]
        tail_workers = costs[max(candidates, len(costs) - candidates):]
        heapify(head_workers)
        heapify(tail_workers)
        
        answer = 0
        next_head, next_tail = candidates, len(costs) - 1 - candidates 

        for _ in range(k): 
            if not tail_workers or head_workers and head_workers[0] <= tail_workers[0]: 
                answer += heappop(head_workers)

                # Only refill the queue if there are workers outside the two queues.
                if next_head <= next_tail: 
                    heappush(head_workers, costs[next_head])
                    next_head += 1
            else: 
                answer += heappop(tail_workers)

                # Only refill the queue if there are workers outside the two queues.
                if next_head <= next_tail:  
                    heappush(tail_workers, costs[next_tail])
                    next_tail -= 1
                    
        return answer

#one heap
class Solution:
    def totalCost(self, costs: List[int], k: int, candidates: int) -> int:
        '''
        the issue with putting all the costs into one queue paired with their indices is that we can only examine candidates at a time
        we can push on to a min heap (costs,0 if first m candidates or 1 if last m candidates)
        
        when we rebalance the min heap again, we need to check we take from the remaining at the left side or right side by checking the indices
        '''
        min_heap = []
        N = len(costs)
        for i in range(candidates):
            min_heap.append((costs[i],0))
        
        for i in range(max(candidates, len(costs) - candidates),N):
            min_heap.append((costs[i],1))
        
        heapq.heapify(min_heap)
        
        ans = 0
        #get points top next left or next right
        next_left = candidates
        next_right = N - candidates - 1
        
        while k > 0:
            curr_cost, curr_side = heapq.heappop(min_heap)
            ans += curr_cost
            #rebalance
            if next_left <= next_right:
                if curr_side == 0:
                    heapq.heappush(min_heap, (costs[next_left],0))
                    next_left += 1
                
                else:
                    heapq.heappush(min_heap, (costs[next_right],1))
                    next_right -= 1
            
            k -= 1
        
        return ans

#############################################
# 373. Find K Pairs with Smallest Sums
# 28JUN23
##############################################
class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        '''
        bfs with min heap, only add to the heap if we hav
        there is no easy way to pick up the smallest sums
        so we keep track of states (index in nums1, index in nums2)
        we either pick at this state and move on to the next one
        
        intution:
            the arrays are sorted increasingly, so the smallest sum would be (0,0)
                from here the next smallest sum would could, we only need to look at these two pairs, because any other would be greater
                    (0,1), (1,0)
                if we move to (1,0), the next smallest pair would be (0,1) still
                    that state is still in contetion, or (1,1), (0,2)
                but we would want the smaller sum of these two pairs, BFS!
            at each step we chose the minimum sum pair from the remaining leftover combinations of pairs
        
        we can use heap to keep track of the smallest sum
        '''
        ans = []
        seen = set()
        
        min_heap = [(nums1[0]+nums2[0],0,0)]
        seen.add((0,0))
        
        while k > 0 and min_heap:
            sum_,i,j = heapq.heappop(min_heap)
            entry = [nums1[i],nums2[j]]
            ans.append(entry)
            
            #check neighbors
            if i+1 < len(nums1) and (i+1,j) not in seen:
                heapq.heappush(min_heap,(nums1[i+1] + nums2[j], i+1,j))
                seen.add((i+1,j))
            
            if j+1 < len(nums2) and (i,j+1) not in seen:
                heapq.heappush(min_heap,(nums1[i] + nums2[j+1], i,j+1))
                seen.add((i,j+1))
            
            k -= 1
        
        return ans
    
#########################################
# 1514. Path with Maximum Probability
# 28JUN23
#########################################
#using acutal probs
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        '''
        this is just djikstras shortest path, but using max prob as the weight, insteaf of the sum
        multiplying probabilites will take precision errors
        take logs and sum them up
        '''
        graph = defaultdict(list)
        for i in range(len(edges)):
            u = edges[i][0]
            v = edges[i][1]
            #weight = math.log(succProb[i],2)
            weight = succProb[i]
            graph[u].append((v,weight))
            graph[v].append((u,weight))
        
        
        visited = set()
        dist = [0.0]*n
        dist[start] = 1.0
        
        max_heap = [(-1.0,start)]
        
        while max_heap:
            curr_prob,node = heapq.heappop(max_heap)
            curr_prob *= -1
            
            #only need to maximuize,already maximum
            if dist[node] > curr_prob:
                continue
            visited.add(node)
            for neigh,prob_to in graph[node]:
                if neigh in visited:
                    continue
                
                new_prob = curr_prob*prob_to
                if new_prob > dist[neigh]:
                    dist[neigh] = new_prob
                    heapq.heappush(max_heap, (-new_prob,neigh))
        
        
        return dist[end]
    
#using log probs
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        '''
        this is just djikstras shortest path, but using max prob as the weight, insteaf of the sum
        multiplying probabilites will take precision errors
        take logs and sum them up
        '''
        graph = defaultdict(list)
        for i in range(len(edges)):
            u = edges[i][0]
            v = edges[i][1]
            weight = math.log(succProb[i],2)
            graph[u].append((v,weight))
            graph[v].append((u,weight))
        
        
        visited = set()
        dist = [float('-inf')]*n
        dist[start] = 0.0
        
        max_heap = [(0.0,start)]
        
        while max_heap:
            curr_prob,node = heapq.heappop(max_heap)
            curr_prob *= -1
            
            #only need to maximuize,already maximum
            if dist[node] > curr_prob:
                continue
            visited.add(node)
            for neigh,prob_to in graph[node]:
                if neigh in visited:
                    continue
                
                new_prob = dist[node] + prob_to
                if new_prob > dist[neigh]:
                    dist[neigh] = new_prob
                    heapq.heappush(max_heap, (-new_prob,neigh))
        
        
        if dist[end] != 0.0:
            return 2**(dist[end])
        else:
            return 0.0
        
#bellman ford
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        '''
        we can also use bellman ford
        intution
            a path in a graph without any cycles has at most n - 1 edges (if there are n nodes)
            so we relax the edges n-1 times for all edge in edges
            for each node in the graph, it tries to improve the shortest path from some source
        
        in the first round, we update the maximum probabily of reachining some nod u if we are allowed only 1 edge
        second round, we are allowed two edges...third round we are allowed three....and so on
        
        algo:
            init max probabilty array, to 0's and init the start to 1
            relax all edges
                for each edge (u,v) if we find a high probability of reaching u through this edge we update
            if we are unable to update break
        '''
        max_probs = [float('-inf')]*n
        max_probs[start] = 0
        
        for _ in range(n-1):
            can_update = False
            for i in range(len(edges)):
                u,v = edges[i]
                log_prob = math.log(succProb[i],2)
                #if we can reach u with a heigh prob
                prob_v_through_u = max_probs[u] + log_prob
                if prob_v_through_u > max_probs[v]:
                    max_probs[v] = prob_v_through_u
                    can_update = True
                prob_u_through_v = max_probs[v] + log_prob
                if prob_u_through_v > max_probs[u]:
                    max_probs[u] = prob_u_through_v
                    can_update = True
            
            if not can_update:
                break
        
        if max_probs[end] != float('-inf'):
            return 2**max_probs[end]
        return 0.0
    
#shortest path faster 
class Solution:
    def maxProbability(self, n: int, edges: List[List[int]], succProb: List[float], start: int, end: int) -> float:
        '''
        shortest path faster algo, similar to dijkstra's but we use q for node to edge weights
        we also only add back to the queue when we can relax an edge
            if the probability of traveling from the stating node to a neighbor node through a specific edge is greater than the current maximum
            probability for that neighbor, we update the maximum prob of this neighbor and add its neighbor
        '''
        graph = defaultdict(list)
        for i in range(len(edges)):
            u = edges[i][0]
            v = edges[i][1]
            weight = math.log(succProb[i],2)
            graph[u].append((v,weight))
            graph[v].append((u,weight))
        
        
        visited = set()
        dist = [float('-inf')]*n
        dist[start] = 0.0
        
        q = deque([(0.0,start)])
        
        while q:
            curr_prob,node = q.popleft()
            curr_prob *= -1
            
            #only need to maximuize,already maximum
            if dist[node] > curr_prob:
                continue
            visited.add(node)
            for neigh,prob_to in graph[node]:
                if neigh in visited:
                    continue
                
                new_prob = dist[node] + prob_to
                if new_prob > dist[neigh]:
                    dist[neigh] = new_prob
                    q.append((-new_prob,neigh))
        
        
        if dist[end] != 0.0:
            return 2**(dist[end])
        else:
            return 0.0

###########################################
# 864. Shortest Path to Get All Keys
# 29JUN23
###########################################
#fml
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        '''
        we have empty cells, walls, keys and locks
        we have a starting point
        if we walk over a key, we can choose to pick it up or not pick it up
            however, we can only walk over a lock if we have the key
        
        there is always a bijection to the (key,lock) pair
        return lowest number of moves to get all the keys, not to obtain all the locks
        bfs, but push step count and states 
        states:
            need to keep track of (key,lock) pairs as well as count
            if we have picked up all the keys, we are good
            if we go over a lock without having the key, we are in an invalid state, so skip it and don't add its neihbords
            but we'll need a way to keep track of already visited states
            store states as strings "key_count_lock_count" as well as if the lock has been opend up
            push the whole grid?, then check the state of this grid on this path
            there is only going to be 6 key,lock pairs
            state would be:
                (i,j,keys_count_locks_count)
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        start_row,start_col = -1,-1
        key_lock_state = {}
        keys = 0
        visited = set()
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '@':
                    start_row,start_col = i,j
                elif grid[i][j].islower():
                    key_lock_state[grid[i][j]] = 0
                    keys += 1
                elif grid[i][j].isupper():
                    key_lock_state[grid[i][j]] = 0
                    
        def get_state(mapp):
            string_state = [k+"_"+str(v) for k,v in mapp.items()]
            string_state = "_".join(string_state)
            return string_state
        
        def get_state_reverse(string):
            temp = string.split("_")
            mapp = {}
            for i in range(0,len(temp),2):
                key = temp[i]
                lock = temp[i+1]
                mapp[key] = int(lock)
            
            return mapp
        
        q = deque([(0,start_row,start_col,get_state(key_lock_state))])
        while q:
            #unpack states
            curr_moves,curr_row,curr_col,curr_state = q.popleft()
            #check if we have all keys
            print(curr_state)
            curr_state = get_state_reverse(curr_state)
            if sum(curr_state.values()) == keys:
                return curr_moves
            visited.add((curr_row,curr_col, get_state(curr_state)))
            for dx,dy in dirrs:
                neigh_x = dx + curr_row
                neigh_y = dy + curr_col
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #empty cell
                    if grid[neigh_x][neigh_y] == '.' and (neigh_x,neigh_y,get_state(curr_state)) not in visited:
                        q.append((curr_moves+1,neigh_x,neigh_y,get_state(curr_state)))
                    
                    #is a key
                    elif grid[neigh_x][neigh_y].islower() and (neigh_x,neigh_y,get_state(curr_state)) not in visited:
                        next_state = curr_state
                        next_state[grid[neigh_x][neigh_y]] += 1
                        q.append((curr_moves+1,neigh_x,neigh_y,get_state(next_state)))
                    #if its lock that we can open
                    elif grid[neigh_x][neigh_y].isupper():
                        #check if we have that key
                        if curr_state[grid[neigh_x][neigh_y].lower()] == 1:
                            next_state = curr_state
                            next_state[grid[neigh_x][neigh_y].lower()] = 0
                            q.append((curr_moves+1,neigh_x,neigh_y,get_state(next_state)))
                            
                        
        return -1
    

#close oneeee!
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        '''
        we have empty cells, walls, keys and locks
        we have a starting point
        if we walk over a key, we can choose to pick it up or not pick it up
            however, we can only walk over a lock if we have the key
        
        there is always a bijection to the (key,lock) pair
        return lowest number of moves to get all the keys, not to obtain all the locks
        bfs, but push step count and states 
        states:
            need to keep track of (key,lock) pairs as well as count
            if we have picked up all the keys, we are good
            if we go over a lock without having the key, we are in an invalid state, so skip it and don't add its neihbords
            but we'll need a way to keep track of already visited states
            store states as strings "key_count_lock_count" as well as if the lock has been opend up
            push the whole grid?, then check the state of this grid on this path
            there is only going to be 6 key,lock pairs
            state would be:
                (i,j,keys_count_locks_count)
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        start_row,start_col = -1,-1
        key_lock_state = {}
        keys = 0
        visited = set()
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '@':
                    start_row,start_col = i,j
                elif grid[i][j].islower():
                    key_lock_state[grid[i][j]] = 0
                    keys += 1
                elif grid[i][j].isupper():
                    key_lock_state[grid[i][j]] = 0
                    
        def get_state(mapp):
            string_state = [k+"_"+str(v) for k,v in mapp.items()]
            string_state = "_".join(string_state)
            return string_state
        
        def get_state_reverse(string):
            temp = string.split("_")
            mapp = {}
            for i in range(0,len(temp),2):
                key = temp[i]
                lock = temp[i+1]
                mapp[key] = int(lock)
            
            return mapp
        
        q = deque([(0,0,start_row,start_col,get_state(key_lock_state))])
        while q:
            #unpack states
            curr_moves,curr_keys,curr_row,curr_col,curr_state = q.popleft()
            #check if we have all keys
            curr_state = get_state_reverse(curr_state)
            if curr_keys == keys:
                return curr_moves
            visited.add((curr_row,curr_col, get_state(curr_state)))
            for dx,dy in dirrs:
                neigh_x = dx + curr_row
                neigh_y = dy + curr_col
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #empty cell
                    if grid[neigh_x][neigh_y] == '.' and (neigh_x,neigh_y,get_state(curr_state)) not in visited:
                        q.append((curr_moves+1,curr_keys,neigh_x,neigh_y,get_state(curr_state)))
                    
                    #is a key
                    elif grid[neigh_x][neigh_y].islower() and (neigh_x,neigh_y,get_state(curr_state)) not in visited:
                        next_state = curr_state
                        next_state[grid[neigh_x][neigh_y]] += 1
                        q.append((curr_moves+1,curr_keys+1,neigh_x,neigh_y,get_state(next_state)))
                    #if its lock that we can open
                    elif grid[neigh_x][neigh_y].isupper():
                        #check if we have that key
                        if curr_state[grid[neigh_x][neigh_y].lower()] == 1:
                            next_state = curr_state
                            next_state[grid[neigh_x][neigh_y].lower()] = 0
                            q.append((curr_moves+1,curr_keys,neigh_x,neigh_y,get_state(next_state)))
        
        return -1
    
class Solution:
    def shortestPathAllKeys(self, grid: List[str]) -> int:
        '''
        almost had it! need to use bit masks to represent key holding states to allow is to visit previosuly visited cells
        states:
            (row,col,dist,key-holding state)
        
        need to use bit mask to represent state of collecterd keys
        we only have 6 keys, we need 12 spots in total (for key lock pairs)
        when entering a cell we just check if we have that key and also check if we have seen this state
        '''
        rows = len(grid)
        cols = len(grid[0])
        #if we have all the keys
        end_key_mask = 0
        visited = set()
        q = deque([])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        for i in range(rows):
            for j in range(cols):
                #start cell add to q and seen
                if grid[i][j] == '@':
                    q.append((i,j,0,0)) #entry is (row,col,steps and state)
                    visited.add((i,j,0))
                elif grid[i][j]  in 'abcdef':
                    end_key_mask = end_key_mask | 1 << (ord(grid[i][j]) - ord('a'))
        
        while q:
            curr_row,curr_col,curr_state,curr_steps = q.popleft()
            if curr_state == end_key_mask:
                return curr_steps
            #make sure we can move from here
            if grid[curr_row][curr_row] != '#':
                for dx,dy in dirrs:
                    neigh_x = dx + curr_row
                    neigh_y = dy + curr_col
                    #bounds
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        if (curr_row,curr_col,curr_state) not in visited:
                            if grid[neigh_x][neigh_y] in 'abcdef':
                                new_state = curr_state | (ord(grid[neigh_x][neigh_y]) - ord('a'))
                                q.append((neigh_x,neigh_y,new_state,curr_steps+1))
                            #is a lock, make sure we have it 
                            elif grid[neigh_x][neigh_col] in '@.' or (grid[neigh_x][neigh_col] in 'ABCDEF' and k & (1 << ord(grid[neigh_x][neigh_col]) - ord('A'))):
                                q.append((neigh_x,neigh_y,new_state,curr_steps+1))
                                
                            visited.add((neigh_row,neigh_col,curr_state))
            
        return -1
    

########################################
# 250. Count Univalue Subtrees
# 29JUN23
########################################
#close one, can top down, but need two return arguments
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        '''
        let dp(node) be the number of univalue subtrees for this node
            if i know the number of univalue subtress on left and right
            i can add them together, only if node.val == node.left.val and node.right.val

        the algorithm doesn't hold for cases like
        [1,1,1,5,5,null,5], where we need to break the count at the 
         1
         /\
         1 1 position, this algo is carrying it up
        '''
        def dp(node):
            if not node:
                return 0
            left = dp(node.left)
            right = dp(node.right)
            #no subtrees
            if not node.left and not node.right:
                return 1 + left + right
            #two subtress
            elif node.left and node.right:
                if node.val == node.left.val == node.right.val:
                    return 1 + left + right
                else:
                    return left + right
            #only left side
            elif node.left and not node.right:
                if node.val == node.left.val:
                    return 1 + left
                else:
                    return left + right
            elif not node.left and node.right:
                if node.val == node.right.val:
                    return 1 + right
                else:
                    return left + right
        
        return dp(root)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        '''
        intution, given a node in our tree, a node is uni-value if:
            the children of this node are also uni-value
            the children have the same value as node
            leaf nodes are trivially uni-value
        
        dfs(node) returns whether or not a node in univalue
        then we need to check univalue for left and univalue for right
        return false is it is not
        for dfs function
            if node is null, we can return true
            
        keep global count variable
        '''
        self.count = 0
        
        def dfs(node):
            if not node:
                return True
            left = dfs(node.left)
            right = dfs(node.right)
            
            if left and right:
                if node.left and node.left.val != node.val:
                    return False
                if node.right and node.right.val != node.val:
                    return False
                
                self.count += 1
                return True
            
            return False

        dfs(root)
        return self.count
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        '''
        to avoid the use of a global variable, we can modify the function
        dp(node) returns two values:
            whether this node is univalue
            and the number of univalue subtrees in this node
            
            then we can just unpack
            base case:
                return True,0
            
            is_left_uni,count_left = dp(node.left)
            is_right_unit,count_right = dp(node.right)
            
            
        '''
        def dp(node):
            if not node:
                return [True,0]
            
            left_uni, left_count = dp(node.left)
            right_uni,right_count = dp(node.right)
            #maintain uni invaraint
            if left_uni and right_uni:
                if node.left and node.left.val != node.val:
                    return [False, left_count + right_count]
                if node.right and node.right.val != node.val:
                    return [False, left_count + right_count]
                
                return [True, left_count + right_count + 1]
            
            return [False, left_count + right_count]
        
        return dp(root)[1]
    
#################################################
# 1970. Last Day Where You Can Still Cross
# 30JUN23
#################################################
class Solution:
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        '''
        we are given binary matrix, 0 = land, 1 = water
        we are given a list of cells list<list> 1 indexed, where cells[i] changed to water
        find the last day where we can walk to the bottom from the top
            can start anywhere on top, and end anywhere on bottom
        
        brute force woul be to bfs for each grid where we modify the grid on that day
        if we cant reach the bottom, we are done
        bfs with binary search!
            issue is getting the grid at the right time
        
        i can create the grid in row,col time and in each cell, indicate the time when it turns to water
        then when doing bfs, check if we can reach this cell by using the time
        '''
        grid = [[0]*col for _ in range(row)]
        dirrs = [(1,0),(-1,0),(0,-1),(0,1)]
        for i,cell in enumerate(cells):
            x,y = cell
            grid[x-1][y-1] = i+1
        
        #make the trip at time t
        def can_make(grid,time):
            seen = set()
            q = deque([])
            for c in range(col):
                if grid[0][c] > time:
                    q.append((0,c))
                    seen.add((0,c))
            #early checks
            if len(q) == 0:
                return False
            
            while q:
                curr_row,curr_col = q.popleft()
                #made it to the bottom
                if curr_row == row - 1:
                    return True
                for dx,dy in dirrs:
                    neigh_x = curr_row + dx
                    neigh_y = curr_col + dy
                    #bounds
                    if 0 <= neigh_x < row and 0 <= neigh_y < col:
                        #has not changed yet at this time
                        if grid[neigh_x][neigh_y] > time and (neigh_x,neigh_y) not in seen:
                            q.append((neigh_x,neigh_y))
                            seen.add((neigh_x,neigh_y))
            return False
        
        #binary search now
        left = 1
        right = len(cells)
        
        while left < right:
            mid = left + (right - left) // 2
            if can_make(grid,mid):
                left = mid + 1
            else:
                right = mid
        
        return left - 1
        
#dfs
class Solution:
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        '''
        we can also use dfs insteaf of bfs
        note, another way would have been to build the 1's grid up the the ith day then do dfs/bfs
        
        '''
        grid = [[0]*col for _ in range(row)]
        for i,cell in enumerate(cells):
            x,y = cell
            grid[x-1][y-1] = i+1
            
        #binary search now
        left = 1
        right = len(cells)
        
        while left < right:
            mid = left + (right - left) // 2
            if self.can_make(row,col,cells,grid,mid):
                left = mid + 1
            else:
                right = mid
        
        return left - 1
        
    
    def can_make(self, row:int, col:int, cells: List[List[int]], grid: List[List[int]], time:int) -> bool:
        visited = set()
        dirrs = [(1,0),(-1,0),(0,-1),(0,1)]
        
        #make the trip at time t
        def dfs(i,j):
            if i == row - 1:
                return True
            visited.add((i,j))
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < row and 0 <= neigh_y < col:
                    #has not changed yet at this time
                    if grid[neigh_x][neigh_y] > time and (neigh_x,neigh_y) not in visited and dfs(neigh_x,neigh_y):
                        return True
            return False
        
        for i in range(col):
            if grid[0][i] > time and dfs(0,i):
                return True
        return False
    

#union find using dfs
class DSU:
    def __init__(self,n):
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
        
        #other wise not same
        if self.size[x_par] >= self.size[y_par]:
            self.parent[y_par] = x_par
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
        else:
            self.parent[x_par] = y_par
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0

class Solution:
    def latestDayToCross(self, row: int, col: int, cells: List[List[int]]) -> int:
        '''
        union find solution is a little trickier
        we need to reverse the days in cells, which is the same as replacing water celss with land cels
        if we go in reverse and we have a path, this is the first day a path becomes available
        we replaces cells[at this date] with a land cell and for each land cell we uniond the edges
        
        top row my have many disconnected components. do we need to check them one by one?
        no, we just need another group for both top row and bottom row to represent a connection
        '''
        
        dsu = DSU(row*col + 2)
        #initally all are land
        grid = [[1]*col for _ in range(row)]
        dirrs = [(1,0),(-1,0),(0,-1),(0,1)]
        
        #go in reverse days
        for i in range(len(cells)-1,-1,-1):
            curr_row,curr_col = cells[i][0] - 1, cells[i][1] - 1
            #turn to lans
            grid[curr_row][curr_col] = 0
            #find number in UF
            idx1 = curr_row*col + curr_col + 1
            for dx,dy in dirrs:
                neigh_x = curr_row + dx
                neigh_y = curr_col + dy
                idx2 = neigh_x*col + neigh_y + 1
                #bounds
                if 0 <= neigh_x < row and 0 <= neigh_y < col:
                    #is also land
                    if grid[neigh_x][neigh_y] == 0:
                        #union them
                        dsu.union(idx1,idx2)
            
            #check if we havepaths
            #if twop row, add the to top row parent in UF
            if curr_row == 0:
                dsu.union(0,idx1)
            #add to bottom row parent in UF
            if curr_row == row - 1:
                dsu.union(row*col + 1, idx1)
            #if connection
            if dsu.find(0) == dsu.find(row*col + 1):
                return i
