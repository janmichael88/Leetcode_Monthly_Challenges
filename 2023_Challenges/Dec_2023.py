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