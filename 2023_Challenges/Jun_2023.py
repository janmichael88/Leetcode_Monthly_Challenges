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