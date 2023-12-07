############################################
# 634. Find the Derangement of An Array
# 02NOV23
############################################
#top down fails??
class Solution:
    def findDerangement(self, n: int) -> int:
        '''
        no element should appear in its original posistion
        array is sorted in ascending order from numbers [1 to n]
        input is too big for brute force
        n = 1 = [1]
            0 ways
        n = 2 = [1,2]
            1 way
        n = 3 = [1,2,3]
            2 ways, 
            
        if i knew the number of dearrangements for some n, and i want to add in a new number n+1,
        i cant place i cann't place it at it index, but i can put at at all other spots (n-1) times somthing ....
        '''
        memo = {}
        mod = 10**9 + 7
        
        def dp(n):
            if n == 1:
                return 0
            if n <= 0:
                return 1
            if n in memo:
                return memo[n]
            ans = (n-1)*(dp(n-1) + dp(n-2))
            ans %= mod
            memo[n] = ans
            return ans
        
        return dp(n)
    
class Solution:
    def findDerangement(self, n: int) -> int:
        '''
        no element should appear in its original posistion
        array is sorted in ascending order from numbers [1 to n]
        input is too big for brute force
        n = 1 = [1]
            0 ways
        n = 2 = [1,2]
            1 way
        n = 3 = [1,2,3]
            2 ways, 
            
        if i knew the number of dearrangements for some n, and i want to add in a new number n+1,
        i cant place i cann't place it at it index, but i can put at at all other spots (n-1) times somthing ....
        dp(n) = (n-1)*dp(n-1) + dp(n-2)*(n-1)
            = (n-1)*(dp(n-1) + dp(n-2))
            
        we either place i with the number we are swapping:
            (n-1)*dp(n-2)
        or we don't place it
            (n-1)*dp(n-1)
        '''
        mod = 10**9 + 7
        if n == 1:
            return 0
        if n == 0:
            return 1
        
        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 0
        
        for i in range(2,n+1):
            a = (i-1)*dp[i-1] % mod
            b = (i-1)*dp[i-2] % mod
            dp[i] = (a % mod) + (b % mod) % mod
        
        return dp[n] % mod
    
#######################################
# 635. Design Log Storage System
# 03NOV23
#######################################
#finally!   
#just do string comparison
#need to use hashamp as container
class LogSystem:

    def __init__(self):
        self.times = {}
        self.g = {"Year": 4, 
                  "Month": 7, 
                  "Day": 10, 
                  "Hour": 13, 
                  "Minute": 16, 
                  "Second": 19}
        
    def put(self, id, timestamp):
        self.times[id] = timestamp

    def retrieve(self, s, e, gra):
        ind = self.g[gra]
        s, e = s[:ind], e[:ind]
        return [i for i, time in self.times.items() if s <= time[:ind] <= e]

# Your LogSystem object will be instantiated and called as such:
# obj = LogSystem()
# obj.put(id,timestamp)
# param_2 = obj.retrieve(start,end,granularity)

###################################################
# 2264. Largest 3-Same-Digit Number in String
# 04DEC23
###################################################
class Solution:
    def largestGoodInteger(self, num: str) -> str:
        '''
        just count streaks, or use regex >.<
        '''
        ans = ""
        curr_num = num[0]
        
        for ch in num[1:]:
            #extend
            if ch == curr_num[-1]:
                curr_num += ch
                #if size three
                if len(curr_num) == 3:
                    ans = max(ans,curr_num)
                
                #bigger than three
                if len(curr_num) > 3:
                    curr_num = curr_num[1:]
            
            else:
                curr_num = ch
        
        return ans
    
#check in steps of 3
class Solution:
    def largestGoodInteger(self, s):
        n = len(s)
        num = 0
        result = ""
        
        for i in range(n - 2):
            if s[i] == s[i + 1] and s[i + 1] == s[i + 2]:
                a = s[i:i + 3]
                if int(a) >= num:
                    num = int(a)
                    result = a
        
        return result
    
######################################
# 641. Design Circular Deque
# 04DEC23
######################################
#using builtin
class MyCircularDeque:

    def __init__(self, k: int):
        #cheese it and just use deque?
        self.q = deque([])
        self.k = k

    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        self.q.appendleft(value)
        return True

    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        self.q.append(value)
        return True

    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        self.q.popleft()
        return True

    def deleteLast(self) -> bool:
        #check empty 
        if self.isEmpty():
            return False
        self.q.pop()
        return True
        

    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.q[0]

    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.q[-1]

    def isEmpty(self) -> bool:
        return len(self.q) == 0
        

    def isFull(self) -> bool:
        return len(self.q) == self.k
        


# Your MyCircularDeque object will be instantiated and called as such:
# obj = MyCircularDeque(k)
# param_1 = obj.insertFront(value)
# param_2 = obj.insertLast(value)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()

class MyCircularDeque:

    def __init__(self, k: int):
        '''
        true solution using builtin array only
        concept of ring buffer
            two pointers head and tail
            empty when head == tail
            full when tail == head + 1
        '''
        self.q = [-1]*(k+1)
        self.head = 0
        self.tail = 0
        self.size = k + 1

    def insertFront(self, value: int) -> bool:
        if self.isFull():
            return False
        self.head = (self.head - 1) % self.size
        self.q[self.head] = value
        return True
        

    def insertLast(self, value: int) -> bool:
        if self.isFull():
            return False
        self.q[self.tail] = value
        self.tail = (self.tail + 1) % self.size
        return True


    def deleteFront(self) -> bool:
        if self.isEmpty():
            return False
        self.head = (self.head + 1) % self.size
        return True

    def deleteLast(self) -> bool:
        if self.isEmpty():
            return False
        self.tail = (self.tail - 1) % self.size
        return True

    def getFront(self) -> int:
        if self.isEmpty():
            return -1
        return self.q[(self.head) % self.size] 

    def getRear(self) -> int:
        if self.isEmpty():
            return -1
        return self.q[(self.tail - 1) % self.size]

    def isEmpty(self) -> bool:
        return self.tail == self.head

    def isFull(self) -> bool:
        return (self.tail + 1) % self.size == self.head
    
#######################################
# 1688. Count of Matches in Tournament
# 05DEC23
#######################################
class Solution:
    def numberOfMatches(self, n: int) -> int:
        '''
        just apply the operations
        '''
        matches = 0
        
        while n > 1:
            #eveb
            if n % 2 == 0:
                matches += n // 2
                n = n // 2
            else:
                matches += (n - 1) // 2
                n = (n - 1) // 2 + 1
        
        return matches
    
class Solution:
    def numberOfMatches(self, n: int) -> int:
        '''
        there are n teams, and there can only be 1 winner
        so there must be n-1 matches
        '''
        return n- 1


#######################################
# 666. Path Sum IV
# 04DEC23
#######################################
#nice try
class Node:
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.left = None
        self.right = None

class Solution:
    def pathSum(self, nums: List[int]) -> int:
        '''
        if depth is smaller than 5, we can represent tree with an array, where len(array) is number of nodes
        from left to right
            first digit is depth (1 <= d <= 4)
            second digit is position at this depth (from left to right), (1 <= p <= 8)
            third digit is the value of the node (0 <= v <= 9)
        
        return some of all paths from root to leaves
        if i had three, the problem is solved
        build tree using dfs/bfs
        '''
        def unpack(num):
            if num == 0:
                return []
            
            return unpack(num // 10) + [num % 10]
        
        def dfs(i,nums):
            if i == len(nums):
                return None
            depth,pos,val = unpack(nums[i])
            node = Node(val)
            if i + 1 < len(nums):
                if unpack(nums[i+1])[1] == pos:
                    node.left = dfs(i+1,nums)
                else:
                    node.right = dfs(i+1,nums)
            if not node.right and  i + 2 < len(nums):
                if unpack(nums[i+2])[1] == pos + 1:
                    node.right = dfs(i+2,nums)
            
            return node
        
        root = dfs(0,nums)
        ans = [0]
        
        def dfs2(node,path):
            if not node.left and not node.right:
                ans[0] += path + node.data
            
            if node.left:
                dfs2(node.left,path + node.data)
            
            if node.right:
                dfs2(node.right,path + node.data)
        
        dfs2(root,0)
        return ans[0]
    
#hashmap, store nodes (depth,pos)
class Node:
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.left = None
        self.right = None

class Solution:
    def pathSum(self, nums: List[int]) -> int:
        '''
        if depth is smaller than 5, we can represent tree with an array, where len(array) is number of nodes
        from left to right
            first digit is depth (1 <= d <= 4)
            second digit is position at this depth (from left to right), (1 <= p <= 8)
            third digit is the value of the node (0 <= v <= 9)
        
        return some of all paths from root to leaves
        if i had three, the problem is solved
        build tree using dfs/bfs
        if im at index 0 in the array
            its left is going to be 2**0 + 1
            and its right is going to be 2**0 + 2
            rather for some depth d
            left index = 2**(d-1) + 1
            right index = 2**(d-1) + 2
            
            order should be (using first two)
            11
            11 12
            21 22 23 24
            31 32 33 34 35 36 37 38
            41 42 43 44 45 46 47 48
        
        critical obersation
        for some node at p, its left will be 2*p, and its right will be 2*p + 1
        left_child = 2*p, so parent would be at left_child // 2
        right_child = 2*p + 1, so parent would be at (right_child -1) / 2
        use hashmap, and key will be (depth,position) mapping to node,
            
        
        '''
        def unpack(num):
            if num == 0:
                return []
            
            return unpack(num // 10) + [num % 10]
        
        root = Node(nums[0] % 10)
        nodes = {}
        nodes[(1,1)] = root
        
        for num in nums[1:]:
            depth,pos,val = unpack(num)
            node = Node(val)
            parent_pos = (pos + 1) // 2
            parent_node = nodes[(depth - 1, parent_pos)]
            
            #attach left
            if pos % 2 == 1:
                parent_node.left = node
            else:
                parent_node.right = node
            
            #cache
            nodes[(depth,pos)] = node
        
        ans = [0]
        
        def dfs(node,path):
            if not node:
                return
            path += node.data
            if not node.left and not node.right:
                ans[0] += path
            dfs(node.left,path)
            dfs(node.right,path)
        
        dfs(root,0)
        return ans[0]
        
class Node:
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.left = None
        self.right = None

class Solution:
    def pathSum(self, nums: List[int]) -> int:
        '''
        we can also generate the tree without a hashamp
        a node should be left or right is:
            pos  - 1 < 2**(depth-2)
        when depth = 4, we have pos - 1 < 2**(2) =
            pos - 1 < 4
            pos < 5, so its left when pos <= 4 and when depth is 4
        
        '''
        def unpack(num):
            if num == 0:
                return []
            
            return unpack(num // 10) + [num % 10]
        
        root = Node(nums[0] % 10)
        nodes = {}
        
        for num in nums[1:]:
            depth,pos,val = unpack(num)
            curr = root
            pos -= 1
            #zero index, first used for root, need to start at second
            #this is the search where to place the node
            for d in range(depth -2, -1, -1):
                #look for half way point, either left or right
                #really need to review counting with binary trees
                if pos < 2**d:
                    if curr.left == None:
                        curr.left = Node(val)
                    curr = curr.left
                else:
                    if curr.right == None:
                        curr.right = Node(val)
                    curr = curr.right
                #prepare for next pos at the next depth
                pos %= 2**d
        
        ans = [0]
        
        def dfs(node,path):
            if not node:
                return
            path += node.data
            if not node.left and not node.right:
                ans[0] += path
            dfs(node.left,path)
            dfs(node.right,path)
        
        dfs(root,0)
        return ans[0]
        
#bottom up approach
import collections
class Solution:
    def pathSum(self, nums: List[int]) -> int:
        total, counter = 0, collections.Counter()
        for num in reversed(nums):
            d, l, v = map(int, str(num))
            total += v * counter[(d, l)] or v
            counter[(d - 1, (l + 1) // 2)] += counter[(d, l)] or 1
        return total

#########################################
# 1716. Calculate Money in Leetcode Bank
# 06DEC23
#########################################
class Solution:
    def totalMoney(self, n: int) -> int:
        '''
        keep array of 7 days, then just follow the rules
        '''
        days = [0]*7
        bank = 0
        
        for day in range(n):
            curr_day = day % 7
            if curr_day == 0:
                days[curr_day] += 1
            else:
                days[curr_day] = days[curr_day-1] + 1
            
            bank += days[curr_day]
        
        return bank
    
class Solution:
    def totalMoney(self, n: int) -> int:
        '''
        without using days array
        '''
        bank = 0
        curr_day = 1
        
        while n > 0:
            for day in range(min(n,7)):
                bank += curr_day + day
            
            n -= 7
            curr_day += 1
        
        return bank
    
class Solution:
    def totalMoney(self, n: int) -> int:
        '''
        time for math
        for the first week we have
        1+2+3+4+5+6+7 wk 1
        2+3+4+5+6+7+8 wk 2
        3+4+5+6+7+8+9 wk 3
        
        each sum is offset by 7
        for if we have 3 weeks
        28 + (28 + 7) + (28 + 7*2) 
        which is just
        28*(num_weeks) + 7*(num_weeks - 1)
        sum of arithmetic seq:
            (num elements in seq)*(first num)*(second num) / 2
            
        after k weeks, we wouldhave k + 1 dollars on monday
        so add every day + 1
        
        now what if we have days left over
        then its going to be some additinoa contribution after the last full week
        '''
        k = n // 7
        F = 28
        L = 28 + (k - 1) * 7
        arithmetic_sum = k * (F + L) // 2
        
        monday = 1 + k
        final_week = 0
        for day in range(n % 7):
            final_week += monday + day
        
        return arithmetic_sum + final_week
    
#########################################
# 1903. Largest Odd Number in String
# 07DEC23
##########################################
class Solution:
    def largestOddNumber(self, num: str) -> str:
        '''
        if number ends in odd digit, its odd
        '''
        last_odd = -1
        N = len(num)
        for i in range(N-1,-1,-1):
            digit = num[i]
            if int(digit) % 2 == 1:
                last_odd = i
                break
        
        return num[:last_odd+1]





##########################################
# 670. Maximum Swap
# 06DEC23
##########################################
class Solution:
    def maximumSwap(self, num: int) -> int:
        '''
        convert digits to array
        does not make sense to swap that results in a larger number moving from left to right
        and it does not make sense to move a smaller number from right to left
        inputs are technically small enough tto try all
        '''
        digits = []
        temp_num = num
        while temp_num:
            digits.append(str(temp_num % 10))
            temp_num = temp_num // 10
        
        digits = digits[::-1]
        ans = num
        N = len(digits)
        if N == 1:
            return ans
        for i in range(N):
            for j in range(i+1,N):
                digits[i],digits[j] = digits[j],digits[i]
                ans = max(ans, int("".join(digits)))
                digits[i],digits[j] = digits[j],digits[i] 
        
        return ans
                
    