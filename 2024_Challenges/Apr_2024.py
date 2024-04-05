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
class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        '''
        increase dist by 1 every two turns
        '''
        dirrs = [[1,0],[0,1],[-1,0],[0,-1]]
        
        cells = rows*cols
        visited = [(rStart,cStart)]
        dirr_ptr = 0
        step_size = 1

        while len(visited) < rows*cols:
            #right down
            for _ in range(2):
                for i in range(step_size):
                    d_y,d_x = dirrs[dirr_ptr]
                    #move to next cell using dirr_ptr and step_size
                    rStart = rStart + d_x
                    cStart = cStart + d_y
                    #if in bounds, add it
                    if 0 <= rStart < rows and 0 <= cStart < cols:
                        visited.append([rStart,cStart])
                dirr_ptr = (dirr_ptr + 1) % 4
            step_size += 1
            #left up
            for _ in range(2):
                for i in range(step_size):
                    d_y,d_x = dirrs[dirr_ptr]
                    #move to next cell using dirr_ptr and step_size
                    rStart = rStart + d_x
                    cStart = cStart + d_y
                    #if in bounds, add it
                    if 0 <= rStart < rows and 0 <= cStart < cols:
                        visited.append([rStart,cStart])
                dirr_ptr = (dirr_ptr + 1) % 4
            step_size += 1
        return visited

#right idea, just break it into 4 walks
#incremnt step size by 2 each time
class Solution:
    def spiralMatrixIII(self, rows: int, cols: int, rStart: int, cStart: int) -> List[List[int]]:
        '''
        increase dist by 1 every two turns
        '''
        r,c = rStart,cStart
        visited = [(r,c)]
        
        is_valid = lambda r,c : 0 <= r < rows and 0 <= c < cols
        steps = 1    
        
        while len(visited) < rows*cols:
            #go right
            for i in range(steps):
                r,c = r,c+1
                if is_valid(r,c):
                    visited.append((r,c))
            #go down
            for i in range(steps):
                r,c = r+1,c
                if is_valid(r,c):
                    visited.append((r,c))
            
            steps += 1
            #go left
            for i in range(steps):
                r,c = r,c-1
                if is_valid(r,c):
                    visited.append((r,c))
            
            #go up
            for i in range(steps):
                r,c = r-1,c
                if is_valid(r,c):
                    visited.append((r,c))
            steps += 1
        return visited

#############################################
# 842. Split Array into Fibonacci Sequence
# 03APR24
#############################################
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        '''
        try all splits
        review on all possible concenations
        N = len(num)
        
        def rec(i,num,path):
            print(path)
            if i == N:
                return
            
            for j in range(i,N):
                temp = num[i:j+1]
                rec(j+1,num,path + [temp])
                
        
        rec(0,num,[])
        '''
        N = len(num)
        ans = []
        
        def rec(i,num,path):
            #print(path)
            if i == N:
                if len(path) > 2:
                    ans.append(path[:])
                return
            
            for j in range(i,N):
                temp = num[i:j+1]
                if len(temp) > 1 and temp[0] == '0':
                    #print(temp)
                    continue
                if int(temp) > 2**31:
                    continue
                if len(path) < 2:
                    rec(j+1,num,path + [int(temp)])
                if len(path) >= 2:
                    if path[-1] + path[-2] == int(temp):
                        rec(j+1,num,path + [int(temp)])
                
        
        rec(0,num,[])
        if not ans:
            return []
        return ans[0]
    
#true backtracking
class Solution:
    def splitIntoFibonacci(self, num: str) -> List[int]:
        '''
        can treat this as backtracking, but need to return something
        so we just return if this path is valid
        '''
        N = len(num)
        ans = []
        
        def rec(i,num,path):
            if len(path) > 2 and path[-3] + path[-2] != path[-1]:
                return False
            if i == N:
                if len(path) > 2 and ans == None:
                    ans.append(path[:])
                    return True
                return False

            for j in range(i,N):
                temp = num[i:j+1]
                if len(temp) > 1 and temp[0] == '0':
                    #print(temp)
                    continue
                if int(temp) > 2**31:
                    continue
                path.append(temp)
                if rec(j+1,num,path):
                    return True
                path.pop()
            
            return False
        
        rec(0,num,[])
        if not ans:
            return []
        return ans[0]

##################################################
# 1614. Maximum Nesting Depth of the Parentheses
# 04APR24
###################################################
class Solution:
    def maxDepth(self, s: str) -> int:
        '''
        balance and take the max
        '''
        ans = float('-inf')
        curr_bal = 0
        
        for ch in s:
            if ch == '(':
                curr_bal += 1
            elif ch == ')':
                curr_bal -= 1
            
            ans = max(ans,curr_bal)
        
        return ans
    
#############################################
# 1544. Make The String Great (REVISITED)
# 05ARP24
#############################################
class Solution:
    def makeGood(self, s: str) -> str:
        '''
        doesn't matter what two chars we remove
        '''
        def isGreat(s):
            N = len(s)
            for i in range(0,N-1):
                #diff must be abs(32)
                left,right = s[i],s[i+1]
                if abs(ord(left) - ord(right)) == 32:
                    return False
            
            return True
        
        while not isGreat(s):
            N = len(s)
            for i in range(0,N-1):
                #diff must be abs(32)
                left,right = s[i],s[i+1]
                if abs(ord(left) - ord(right)) == 32:
                    break
            
            s = s[:i] + s[i+2:]
        
        return s

#recursion
class Solution:
    def makeGood(self, s: str) -> str:
        #recursion
        
        def rec(s):
            if not s:
                return ""
            N = len(s)
            for i in range(0,N-1):
                left,right = s[i],s[i+1]
                if abs(ord(left) - ord(right)) == 32:
                    return rec(s[:i] + s[i+2:])
            return s
        
        return rec(s)
    
#stack, make sure to clear both
class Solution:
    def makeGood(self, s: str) -> str:
        stack = []
        for ch in s:
            if stack and abs(ord(stack[-1]) - ord(ch)) == 32:
                stack.pop()
            else:
                stack.append(ch)
        
        return "".join(stack)
    
###################################
# 769. Max Chunks To Make Sorted
# 05APR24
###################################
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        '''
        keep pref max array
        then compare pref max with the sorted array
        if pref_max[i] == num at sorted position, use up a chunk
        idea is to find some splitting line so that numbers being left of this line are smaller than numbers
        to the right of the line
            no we can ask how many lines exist
        imagine the sorted array [0,1,2,3,4,5], we can do 6 chunks
        if swap the ends : [5,1,2,3,4,0], we have to use the whole chunk
        intutition:
        The key to understand this algorithms lies in the fact that when max[index] == index, 
        all the numbers before index must be smaller than max[index] (also index), 
        so they make up of a continuous unordered sequence, i.e {0,1,..., index}

        This is because numbers in array only vary in range [0, 1, ..., arr.length - 1], 
        so the most numbers you can find that are smaller than a certain number, say arr[k], 
        would be arr[k] - 1, i.e [0, 1, ..., arr[k] - 1]. So when arr[k] is the max number 
        in [arr[0], arr[1], ..., arr[k]], all the k - 1 numbers before it can only lies in [0, 1, ..., arr[k] - 1], 
        so they made up of a continuous sequence. 

        '''
        N = len(arr)
        pref_max = [0]*N
        pref_max[0] = arr[0]
        for i in range(1,N):
            pref_max[i] = max(arr[i],pref_max[i-1])
        
        chunks = 0
        for i in range(N):
            if pref_max[i] == i:
                chunks += 1
        
        return chunks