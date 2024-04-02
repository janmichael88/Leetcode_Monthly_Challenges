#######################################
# 1518. Water Bottles
# 01APR24
#######################################
#careful with how you solve subprorblems!
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        '''
        dp recursion and reduce
        '''
        memo = {}  # Dictionary to store computed values

        def dp(n):
            if n == 0:
                return 0
            if n in memo:
                return memo[n]

            ans = 0
            # Try drinking from 1 to n + 1
            for d in range(1, n + 1):
                # Drink and no exchange
                op1 = d + dp(n - d)
                # Drink and exchange
                new_bottles = d // numExchange
                op2 = d + dp(new_bottles + (n - d))
                #need to max op1 and op2 first then globally!
                ans = max(ans, max(op1, op2))

            memo[n] = ans
            return ans

        return dp(numBottles)
    
#recursion, reduction not dp
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        '''
        let n be numBottles and k be numexchange
        when n < k, we drink all the bottles and cannot exchange
        otherwise we drink k at a time and exhcnage those k empty bottles for +1 drink
        '''
        def rec(n,k):
            if n < k:
                return n
            return k + rec(n - k + 1,k)
        
        return rec(numBottles,numExchange)
    
class Solution:
    def numWaterBottles(self, numBottles: int, numExchange: int) -> int:
        '''
        iterative
        
        '''
        drink = 0
        while numExchange <= numBottles:
            drink += numExchange
            numBottles -= numExchange
            numBottles += 1
        
        return drink + numBottles

###############################################################
# 1779. Find Nearest Point That Has the Same X or Y Coordinate
# 02APR24
###############################################################
class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        '''
        just check all points only if x or y line up
        '''
        min_dist = float('inf')
        min_index = float('inf')
        
        for i_i,point in enumerate(points):
            i,j = point
            if i == x or j == y:
                curr_dist = abs(i-x) + abs(j-y)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    min_index = i_i
        
        if min_index == float('inf'):
            return -1
        return min_index
    
################################
# 885. Spiral Matrix III
# 02APR24
################################
#close one
class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        '''
        walk the spiral and keep going until we touch all visit all cells
        spiral is
        (start) , (right,one), (down,one),(left,two), (up,two),(right,three)
        so its R,D,L,U repeating, increase dist by 1 every time
        '''
        dirrs = [
            [1,0],[0,1],[-1,0],[0,-1]
        ]
        
        cells = rows*cols
        visited = []
        dirr_ptr = 0
        curr_step = -1
        step_size = 1
        
        #start
        visited.append([rStart,cStart])
        cells -= 1
        
        while cells > 0:
            if curr_step % 2 == 0:
                step_size += 1
            d_y,d_x = dirrs[dirr_ptr]
            #move to next cell using dirr_ptr and step_size
            rStart = rStart + d_x*step_size
            cStart = cStart + d_y*step_size
            print(rStart,cStart)
            #if in bounds, add it
            if 0 <= rStart < rows and 0 <= cStart < cols:
                visited.append([rStart,cStart])
                cells -= 1
            
            dirr_ptr = (dirr_ptr + 1) % 4
            curr_step += 1
        
        return visited

#right idea, just break it into 4 walks
#incremnt step size by 2 each time
class Solution(object):
    
    def spiralMatrixIII(self, R, C, r0, c0):
        ret = [(r0, c0)] 
        is_valid = lambda row, col: row >= 0 and row < R and col >= 0 and col < C 
        
        steps = 1 
        r, c = r0, c0 
        while len(ret) < R * C: 
            # Go east 1
            for step in range(steps):
                r, c = r, c + 1 
                if is_valid(r, c): ret.append((r, c))
                    
            # Go down 1 
            for step in range(steps):
                r, c = r + 1, c 
                if is_valid(r, c): ret.append((r, c))
                    
            steps += 1
                    
            # Go west 2 
            for step in range(steps):
                r, c = r, c - 1
                if is_valid(r, c): ret.append((r, c))           
            
            # Go north 2 
            for step in range(steps):
                r, c = r - 1, c 
                if is_valid(r, c): ret.append((r, c))           
                    
            steps += 1
            
        return ret 
