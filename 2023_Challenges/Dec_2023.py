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
                
#right maxes array, keep right max and index (non inclusive at some index i)
class Solution:
    def maximumSwap(self, num: int) -> int:
        '''
        keep track of the right most exclusive max of each nuber
        i.e keep array, called max_to_right
        where each element is length 2 list that stores the index and value
        first entry is the number, second entry is the index
        '''
        num = list(str(num))
        N = len(num)
        right_maxes = [[0,0] for _ in range(N)]
        right_maxes[N-1] = [-1,-1]
        
        for i in range(N-2,-1,-1):
            number = int(num[i+1])
            #update
            if number > right_maxes[i+1][0]:
                right_maxes[i] = [number,i+1]
            #carry left
            else:
                right_maxes[i] = right_maxes[i+1]
        
        
        for i in range(N):
            number = int(num[i])
            #if we can find a larger number swap it
            if number < right_maxes[i][0]:
                #swap
                num[i], num[right_maxes[i][1]] = num[right_maxes[i][1]],num[i]
                return int("".join(num))
            
        
        return int("".join(num))
    
################################################
# 1973. Count Nodes Equal to Sum of Descendants
# 08DEC23
################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def equalToDescendants(self, root: Optional[TreeNode]) -> int:
        '''
        carry sums and chekc if == to node.val
        '''
        count = [0]
        
        def dp(node):
            if not node:
                return 0
            left = dp(node.left)
            right = dp(node.right)
            if left + right == node.val:
                count[0] += 1
            
            return left + right + node.val
        
        _ = dp(root)
        return count[0]
    
#######################################
# 1380. Lucky Numbers in a Matrix
# 08DEC23
#######################################
class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        rows,cols = len(matrix),len(matrix[0])
        matrix_t = list(map(list,zip(*matrix)))
        
        row_mins = []
        col_maxs = []
        
        for r in matrix:
            row_mins.append(min(r))
            
        for c in matrix_t:
            col_maxs.append(max(c))
        
        ans = []
        
        for i in range(rows):
            for j in range(cols):
                num = matrix[i][j]
                if (num == row_mins[i] and num == col_maxs[j]):
                    ans.append(num)
        
        return ans

#########################################
# 1360. Number of Days Between Two Dates
# 09DEV23
##########################################
class Solution:
    def daysBetweenDates(self, date1: str, date2: str) -> int:
        def is_leap_year(year) -> bool:
            return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            
        def get_days(date: str) -> int:
            days_in_month: dict[int, int] = {
            1: 31,
            2: 28,  # This value might be 29 for leap years
            3: 31,
            4: 30,
            5: 31,
            6: 30,
            7: 31,
            8: 31,
            9: 30,
            10: 31,
            11: 30,
            12: 31,
        }
            year, month, days = [int(number) for number in date.split("-")]

            if is_leap_year(year): days_in_month[2] = 29
                
            total: int = 0
            
            for year in range(1971, year):
                if is_leap_year(year): total += 366
                else: total += 365
                    
            for month in range(1, month):
                total += days_in_month[month]

            total += days

            return total
        return abs(get_days(date1) - get_days(date2))

###################################################################
# 1287. Element Appearing More Than 25% In Sorted Array (REVISTED)
# 10DEC23
##################################################################
class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        '''
        the candidante must be at (n/4), 2*(n//4), 3*(n//4)
        for each candidate find the first ooccrunce using binary search, then check block size
        '''
        N = len(arr)
        cands = []
        block_size = N //4
        for num in [1,2,3]:
            cand = arr[block_size*num]
            cands.append(cand)
        cands = [arr[N // 4], arr[N // 2], arr[-N // 4]]

        
        for cand in cands:
            #find its first position
            first_pos = bisect.bisect_left(arr,cand)
            print(first_pos,cand)
            if arr[first_pos + block_size] == cand:
                return cand
        
        return -1
    
########################################################
# 1385. Find the Distance Value Between Two Arrays
# 10DEC23
########################################################
class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        '''
        distance value is tnumber of elements arr[i] such that there is not any element arr2[j] where abs(arr1[i] - arr2[j]) <= d
        brute force is allowed?
        answer is going to be from 0 to len(arr1)
        use count to check
        '''
        ans = 0
        for i,n1 in enumerate(arr1):
            count = 0
            for n2 in arr2:
                val = abs(n1 - n2)
                #we dont want to count pairs
                if val <= d:
                    count += 1
            if count == 0:
                ans += 1
            
        
        return ans

class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        '''
        hint says to sort arr2 and use binary search to get the closes element for each arr1[i]
        basically we want to find the number of elements in arr1 which are greater than d for all element sin arr2
        arr1 = [4,5,8]
        after sorting arr2
        arr2 = [1,8,9,10]
        d = 2
        need to check if any element in arr2 is withing d, if its within d, return falsse, i.e this currentt num in arr1 cannot be valid

        '''   
        arr2.sort()
        
        def isValid(arr,num,d):
            left = 0
            right = len(arr) - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                
                #greater
                if arr[mid] > num:
                    diff = arr[mid] - num
                    if diff <= d:
                        return False
                    right = mid - 1
                else:
                    diff = num - arr[mid]
                    if diff <= d:
                        return False
                    left = mid + 1
            
            return True
        
        ans = 0
        for num in arr1:
            if isValid(arr2,num,d):
                ans += 1
        
        return ans
            
###################################################
# 1464. Maximum Product of Two Elements in an Array
# 11DEC23
##################################################
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        '''
        given input constraints brute force should be alllowed
        '''
        ans = float('-inf')
        N = len(nums)
        for i in range(N):
            for j in range(i+1,N):
                ans = max(ans, (nums[i]-1)*(nums[j]-1))
        
        return ans
    
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        '''
        just and grab the last sort
        say we have (a-1)*(b-1), expands
        a*b - a - b + 1
        '''
        nums.sort()
        a = nums[-1]
        b = nums[-2]
        
        return a*b - a - b + 1
    
#no sort
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        '''
        we dont need to sort, just keep track of first largest and second largest
        '''
        a = 0
        b = 0 #given inputs, 0 is the smallest
        
        for num in nums:
            if num > a:
                b = a
                a = num
            else:
                b = max(b,num)
        
        return a*b - a - b + 1

#####################################
# 1436. Destination City
# 14DEC23
####################################
class Solution:
    def destCity(self, paths: List[List[str]]) -> str:
        '''
        its just a linked list, follow pointers
        destination city is the city that has indegree of 1 and out degree of zero
        or just follow path
        '''
        graph = defaultdict()
        for u,v in paths:
            graph[u] = v
        
        curr = paths[0][0]
        
        while curr in graph:
            curr = graph[curr]
        
        return curr
            
#############################################
# 911. Online Election
# 16DEC23
#############################################
#yes!! ugly but its works
class TopVotedCandidate:

    def __init__(self, persons: List[int], times: List[int]):
        '''
        need to keeep track of vote counts at a certain time
        we have everything i the constructor
        need to query the leader at a certain time
        hash everything {t : {count maps}}
        then during the query find the left bound of the time and pick that time
        i can use an array, then just get the previous and carry over, may need to reconvert array back to hashmap later
        this is going to take too long
        make the counts array first, then apply consecutive differs
        '''
        self.votetimes = []
        #pair together
        pairs = [(t,p) for t,p in zip(times,persons)]
        pairs.sort()
        for t,p in pairs:
            curr_vote = Counter()
            curr_vote[p] += 1
            self.votetimes.append(curr_vote)
        
        #apply differences
        for i in range(1,len(self.votetimes)):
            prev_vote = self.votetimes[i-1]
            for k,v in prev_vote.items():
                self.votetimes[i][k] += v
        
        #we only need the maxes
        for i in range(len(self.votetimes)):
            self.votetimes[i] = self.votetimes[i].most_common(1)[0][0]
        
        self.time_markers = sorted(times)


    def q(self, t: int) -> int:
        #find the index into votetimes
        idx = bisect.bisect_left(self.time_markers,t)
        if idx == len(self.time_markers):
            idx -= 1
        if self.time_markers[idx] <= t:
            return self.votetimes[idx]
        return self.votetimes[idx-1]


# Your TopVotedCandidate object will be instantiated and called as such:
# obj = TopVotedCandidate(persons, times)
# param_1 = obj.q(t)
    
#binary search
class TopVotedCandidate:
    '''
    there is actually a differene between bisect_left, bisect_right, and bisect!
    but its better to code out binary search
    intuition:
        precompute top voted person for each time in times in orderedList
        binarys earch topVOted
        times will alays be increasing
    '''

    def __init__(self, persons: List[int], times: List[int]):
        self.topVotes = [0]*len(times) #each time will store person with max votes
        countVotes = Counter()
        current_leader = persons[0]
        
        for i, (p,t) in enumerate(zip(persons,times)):
            countVotes[p] += 1
            if countVotes[p] >= countVotes[current_leader]:
                #maintin most recentin case of times
                current_leader = p
            self.topVotes[i] = (t,current_leader) #topVotes is array for each time in times, but we also need tost oe the time

    def q(self, t: int) -> int:
        left, right = 0, len(self.topVotes) - 1

        while left <= right:
            mid = left + (right - left) // 2
            time,top = self.topVotes[mid]
            if time == t:
                return top
            if time < t:
                left = mid + 1
            else:
                right = mid - 1
        
        time,top = self.topVotes[right]
        return top
        


# Your TopVotedCandidate object will be instantiated and called as such:
# obj = TopVotedCandidate(persons, times)
# param_1 = obj.q(t)
    
############################################
# 2353. Design a Food Rating System
# 17NOV23
############################################
#TLE, need to speed up highest rated
    class FoodRatings:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        '''
        make mapping food to ratinng
        and another one to many cuisine to food
        '''
        self.foodToRating = {f:r for (f,r) in zip(foods,ratings)}
        self.cuisinesToFood = defaultdict(list)
        for f,c in zip(foods,cuisines):
            self.cuisinesToFood[c].append(f)

    def changeRating(self, food: str, newRating: int) -> None:
        self.foodToRating[food] = newRating

    def highestRated(self, cuisine: str) -> str:
        foods = self.cuisinesToFood[cuisine]
        ans_food = foods[0]
        ans_rating = self.foodToRating[ans_food]
        for f in foods:
            #new highest raiting
            if self.foodToRating[f] > ans_rating:
                ans_food = f
                ans_rating = self.foodToRating[f]
            #if equal take the lexogrphical smaller name
            elif self.foodToRating[f] == ans_rating:
                ans_food = min(ans_food,f)
        
        return ans_food
    


# Your FoodRatings object will be instantiated and called as such:
# obj = FoodRatings(foods, cuisines, ratings)
# obj.changeRating(food,newRating)
# param_2 = obj.highestRated(cuisine)
    
#garbage head, with validation check
#make class for food
class Food:
    def __init__(self, rating, name):
        self.rating = rating
        self.name = name
        
    def __lt__(self, other):
        #concept is called overload, need to overload the less tahn operatir for cmoparsions
        if self.rating == other.rating:
            #compare names
            return self.name < other.name
        
        return self.rating > other.rating
        
class FoodRatings:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        '''
        make mapping food to ratinng
        and another one to many cuisine to food
        when i update the rating i also need to updat the highest
        need to maintin max rating for each cuisine
        keep heap of all raitings with foods and cuisines until we need it
        '''
        self.food_to_rating = {}
        self.food_to_cuisine = {}
        self.cuisine_to_food = defaultdict(list) #cusing will make to heap of Foods
        for f,c,r in zip(foods,cuisines,ratings):
            self.food_to_rating[f] = r
            self.food_to_cuisine[f] = c
            heapq.heappush(self.cuisine_to_food[c], Food(r,f))
        


    def changeRating(self, food: str, newRating: int) -> None:
        #update
        self.food_to_rating[food] = newRating
        cuisine_name = self.food_to_cuisine[food]
        #insert new rating into the cuisines heap
        heapq.heappush(self.cuisine_to_food[cuisine_name], Food(newRating,food))

    def highestRated(self, cuisine: str) -> str:
        #we are combining foods and ratings with all its previous ratings
        #we need to compare it back to wahts in the actually food_to_rating mapp!
        highest_rated = self.cuisine_to_food[cuisine][0]
        #make sure it matches
        while self.food_to_rating[highest_rated.name] != highest_rated.rating:
            heapq.heappop(self.cuisine_to_food[cuisine])
            highest_rated = self.cuisine_to_food[cuisine][0]
        
        return highest_rated.name

# Your FoodRatings object will be instantiated and called as such:
# obj = FoodRatings(foods, cuisines, ratings)
# obj.changeRating(food,newRating)
# param_2 = obj.highestRated(cuisine)
    
#ordered set solution
#using sorted sorted
from sortedcontainers import SortedSet
        
class FoodRatings:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        '''
        using sortedSet, cuisine to food map maps to sortedSet
        '''
        self.food_to_rating = {}
        self.food_to_cuisine = {}
        self.cuisine_to_food = defaultdict(SortedSet) #cusing will make to heap of Foods
        for f,c,r in zip(foods,cuisines,ratings):
            self.food_to_rating[f] = r
            self.food_to_cuisine[f] = c
            self.cuisine_to_food[c].add((-r,f))
        


    def changeRating(self, food: str, newRating: int) -> None:
        #update
        cuisine_name = self.food_to_cuisine[food]
        #delete old elements
        old_food = (-self.food_to_rating[food],food)
        self.cuisine_to_food[cuisine_name].remove(old_food)
        #update food to rating
        self.food_to_rating[food] = newRating
        self.cuisine_to_food[cuisine_name].add((-newRating,food))


    def highestRated(self, cuisine: str) -> str:
        #we are combining foods and ratings with all its previous ratings
        #we need to compare it back to wahts in the actually food_to_rating mapp!
        highest_rated = self.cuisine_to_food[cuisine][0]
        
        return highest_rated[1]

# Your FoodRatings object will be instantiated and called as such:
# obj = FoodRatings(foods, cuisines, ratings)
# obj.changeRating(food,newRating)
# param_2 = obj.highestRated(cuisine)
    
######################################################################
# 1637. Widest Vertical Area Between Two Points Containing No Points
# 21NOV23
#####################################################################
class Solution:
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        '''
        need widest area, i,e widest gap between two points such that there are no points in between
        sort based on col position and find largest consecutive difference
        '''
        points.sort(key = lambda x: x[0])
        ans = 0
        N = len(points)
        for i in range(N-1):
            left = points[i]
            right = points[i+1]
            ans = max(ans, right[0] - left[0])
        
        return ans

################################################
# 1422. Maximum Score After Splitting a String
# 22NOV23
################################################
class Solution:
    def maxScore(self, s: str) -> int:
        '''
        prefix sum with ones, then to find zero just subtract from the size of the array
        '''
        pref_sum = [0]
        N = len(s)
        for i in range(N):
            pref_sum.append(pref_sum[-1] + (s[i] == '1'))
        
        
        ans = 0
        #make sure we look at substrings only
        for i in range(N-1):
            left_zeros = (i+1) - pref_sum[i+1]
            right_ones = pref_sum[-1] - pref_sum[i+1]
            #print(left_zeros,right_ones)
            ans = max(ans,left_zeros + right_ones)
        
        return ans

class Solution:
    def maxScore(self, s: str) -> int:
        '''
        two pass, counts ones, then retraerse and decrement count of ones and incrment count of zeros
        '''
        ones = 0
        for ch in s:
            ones += ch == '1'
        
        ans = 0
        zeros = 0
        for ch in s:
            if ch == '1':
                ones -= 1
            else:
                zeros += 1
            
            ans = max(ans, zeros + ones)
        
        return ans

#########################
# 1496. Path Crossing
# 23DEC23
##########################
#other ways to represent two numbers in one number
# Y = aX + b
# a = Y // X
# b = Y % X
# a = horizontal direction, b = vertical direction
def isPathCrossing3(self, path: str) -> bool:  # Beats 93%
    cur = 0
    seen = {cur}
    X = 10**4 + 1  # max possible b + 1
    for move in path:
        match move:
            case 'N':
                cur += 1
            case 'S':
                cur += -1
            case 'E':
                cur += X
            case 'W':
                cur += -X

        if cur in seen:
            return True
        else:
            seen.add(cur)

    return False

# Store the 2 numbers in 1 number using bit manipulation
# max possible direction = len of path = 10**4
# this means all possible values can fit in the final 14 bits: 2^14=16384
# int = 32 bits (usually); we can simply use the last 28 bits to store the number we want
# we don't even have to worry about comparing bits with |, simply use addition
def isPathCrossing4(self, path: str) -> bool:  # beats 93%
    cur = 0
    seen = {cur}
    for move in path:
        match move:
            case 'N':
                cur += 1
            case 'S':
                cur -= 1
            case 'E':
                cur += 1 << 14
            case 'W':
                cur -= 1 << 14

        if cur in seen:
            return True
        else:
            seen.add(cur)

    return False

###########################################################
# 1758. Minimum Changes To Make Alternating Binary String
# 24NOV23
###########################################################
class Solution:
    def minOperations(self, s: str) -> int:
        '''
        final string will be '01010...' or '101010'
        compare string to each final representatino and compare
        '''
        def test(s):
            N = len(s)
            first = "0"
            second = "1"
            for i in range(N-1):
                first += str(int(first[-1]) ^ 1)
                second += str(int(second[-1]) ^ 1)
            
            left = 0
            for a,b in zip(s,first):
                if a != b:
                    left += 1
            
            right = 0
            for a,b in zip(s,second):
                if a != b:
                    right += 1
            
            return min(left,right)
        
        return test(s)
        
class Solution:
    def minOperations(self, s: str) -> int:
        '''
        if we start with a zero, then there should be zeros only on even indices
        if we start with a one, then there should only be ones even indices
        keep count of changese done if we had started with zero or started with one
        then compare with the actual value in the string
        '''
        start0 = 0
        start1 = 0
        N = len(s)
        
        for i in range(N):
            #even 
            if i % 2 == 0:
                if s[i] == '0':
                    start1 += 1
                else:
                    start0 += 1
            #odd
            else:
                if s[i] == '0':
                    start0 += 1
                else:
                    start1 += 1
        
        return min(start0,start1)
    
class Solution:
    def minOperations(self, s: str) -> int:
        '''
        we only need to check one of the starting strings
        say we are comparing s with the alternating binary string starting with 0 and have x differences
        if we had start with the alternatinve string with 0, then there are len(s) - x differences
        '''
        start0 = 0
        N = len(s)
        for i in range(N):
            if i % 2 == 0:
                if s[i] != '0':
                    start0 += 1
            else:
                if s[i] != '1':
                    start0 += 1
        
        return min(start0, N - start0)
    

###############################################################
# 159. Longest Substring with At Most Two Distinct Characters (REVISITED)
# 26DEC23
#################################################################
#finally 
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        '''
        sliding window
        cannot have more than two distint characters
        '''
        counts = Counter()
        N = len(s)
        left = right = 0
        ans = 0
        
        while right < N:
            curr_char = s[right]
            counts[curr_char] += 1
            ans = max(ans, (right - left) )
            right += 1
            while left < N and len(counts) >= 3:
                curr_char = s[left]
                counts[curr_char] -= 1
                if counts[curr_char] == 0:
                    del counts[curr_char]
                left += 1
            

        #last one
        ans = max(ans, (right - left) )
        return ans
                
###########################################
# 1578. Minimum Time to Make Rope Colorful
# 27DEC23
###########################################
#dp uses up too much memory
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        dp, what are the states? position and last color seen
        if there isn't a same color, we can advance and spend no time
        otherwise we cut, change color
        take minimum of the two
        '''
        memo = {}
        N = len(colors)
        
        def dp(last_color,idx):
            if idx >= N:
                return 0
            if (last_color,idx) in memo:
                return memo[(last_color,idx)]
            #same color
            if colors[idx] == last_color:
                ans = neededTime[idx] + dp(last_color,idx+1)
                memo[(last_color,idx)] = ans
                return ans
            else:
                op1 = dp(colors[idx],idx+1)
                op2 = neededTime[idx] + dp(last_color,idx+1)
                ans = min(op1,op2)
                memo[(last_color,idx)] = ans
                return ans
            
        
        return dp("",0)
    
##############################################
# 1531. String Compression II (REVISTED)
# 28DEC23
##############################################
#fuckkk
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        '''
        states,
        need last letter in block
        position in string
        curr_size
        deletions left
        '''
        
        memo = {}
        N = len(s)
        
        def dp(letter,i,size,k):
            #no deletions
            if k < 0:
                return float('inf')
            if i >= N:
                if len(str(size)) > 1:
                    return 1 + len(str(size))
                else:
                    return 1
            if (letter,i,size,k) in memo:
                return memo[(letter,i,size,k)]
            #fist, we have to take it
            if letter == "":
                ans = dp(s[i], i+1, size+1,k)
                memo[(letter,i,size,k)] = ans
                return ans
            #matching
            if s[i] == letter:
                #extend size
                extend = dp(s[i],i+1,size+1,k)
                #delte
                delete = dp(s[i],i+1,size,k-1)
                ans = min(extend,delete)
                memo[(letter,i,size,k)] = ans
                return ans
                
            
            #no matching if
            if s[i] != letter:
                #extend
                if len(str(size)) > 1:
                    extend = (1 + len(str(size))) + dp(s[i], i+1,1,k)
                    delete =  (1 + len(str(size))) + dp(letter,i+1,size,k-1)
                    ans = min(extend,delete)
                    memo[(letter,i,size,k)] = ans
                    return ans
                else:
                    extend = 1 + dp(s[i], i+1,1,k)
                    delete = 1 + dp(letter,i+1,size,k-1)
                    ans = min(extend,delete)
                    memo[(letter,i,size,k)] = ans
                    return ans
        
        
        return dp("",0,0,k)

#######################################################
# 1335. Minimum Difficulty of a Job Schedule (REVISTED)
# 29DEC23
#######################################################
#three state dp
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        question really is partition the array into d parts, such that the sum(min(part) for parts) is minimum
        so we need to maintin index, number partitions, and the current max in this partition
        '''
        memo = {}
        N = len(jobDifficulty)
        
        def dp(i,curr_max,d):
            if d < 0:
                return float('inf')
            if i == N:
                if d == 0:
                    return curr_max
                else:
                    return float('inf')
            
            if (i,curr_max,d) in memo:
                return memo[(i,curr_max,d)]
            
            take = dp(i+1, max(curr_max,jobDifficulty[i]),d)
            no_take = curr_max + dp(i+1,jobDifficulty[i],d-1)
            ans = min(take,no_take)
            memo[(i,curr_max,d)] = ans
            return ans
        
        ans = dp(0,0,d)
        if ans == float('inf'):
            return -1
        return ans
            
#two state but trea like rod cutting
    class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        question really is partition the array into d parts, such that the sum(min(part) for parts) is minimum
        so we need to maintin index, number partitions, and the current max in this partition
        we can reduce from 3 states to 2 states
        states are (index,days remaining)
        we just need to find the max on the remaining days for any given day
        dp(i,d) = {
            min for all j from 0 to <= i-1 dp(j,d-1)
            but for each day j we use one less day, so of these j days we need to find the max to the remaining and add it
            since we go from j to i and j is < i, we can maintain a max a find iteraviely
        }
        '''
        N = len(jobDifficulty)
        if d > N:
            return -1
        memo = {}
        
        max_left = jobDifficulty[:]
        for i in range(N-2,-1,-1):
            max_left[i] = max(max_left[i], max_left[i+1])
        
        
        def dp(i,d):
            if d == 1:
                return max_left[i]
            
            if (i,d) in memo:
                return memo[(i,d)]
            
            ans = float('inf')
            daily_max = 0
            for j in range(i,N - d + 1):
                daily_max = max(daily_max,jobDifficulty[j])
                ans = min(ans, daily_max + dp(j+1,d-1))
            
            memo[(i,d)] = ans
            return ans
        
        
        return dp(0,d)


#bottom up, its kind hard to translate, 
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        question really is partition the array into d parts, such that the sum(min(part) for parts) is minimum
        so we need to maintin index, number partitions, and the current max in this partition
        we can reduce from 3 states to 2 states
        states are (index,days remaining)
        we just need to find the max on the remaining days for any given day
        dp(i,d) = {
            min for all j from 0 to <= i-1 dp(j,d-1)
            but for each day j we use one less day, so of these j days we need to find the max to the remaining and add it
            since we go from j to i and j is < i, we can maintain a max a find iteraviely
        }
        '''
        N = len(jobDifficulty)
        if d > N:
            return -1
        memo = {}
        
        max_left = jobDifficulty[:]
        for i in range(N-2,-1,-1):
            max_left[i] = max(max_left[i], max_left[i+1])
            
        dp = [[float('inf')]*(d+1) for _ in range(N+1)]
              
        
        #base case fill, wheneever d = 1
        for i in range(N):
            dp[i][1] = max_left[i]
        
        for days_remaining in range(1,d+1):
            for i in range(N - days_remaining + 1):
                ans = float('inf')
                daily_max = 0
                for j in range(i,N - days_remaining + 1):
                    daily_max = max(daily_max,jobDifficulty[j])
                    ans = min(ans, daily_max + dp[j+1][days_remaining-1])
                dp[i][days_remaining] = ans
                
        print(dp)
        
########################################
# 813. Largest Sum of Averages
# 31DEC23
########################################
#not knapsack, more like rod cutting
class Solution:
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        '''
        maximize score of the subarrays, where we can have no more then k non-empty subarrays
        states
            (index,current size of partition, and k left) thats too many states
        need to reduce to two states
        dp(i,k) gives ans using nums[i:] and k deletions
        so we want dp(0,k)
        
        we can use pref_sum array, no need to add starting becaue empty subarrays are not allowed
        i need to keep start,end and partitions left
        '''
        N = len(nums)
        pref_sum = [0]
        for num in nums:
            pref_sum.append(pref_sum[-1] + num) #for som indices [i:j] inclusive pref_sum[j+1] - pref_sum[i]
            
        memo = {}
        
        def dp(left,right,k):
            #just the single elemenet
            if left == right:
                return pref_sum[right+1] - pref_sum[left]
            #got to the end
            if right == N-1:
                return (pref_sum[right+1] - pref_sum[left]) / (right - left)
            #no more parittions need then we need the averge
            if k == 0:
                return (pref_sum[right+1] - pref_sum[left]) / (right - left)
            
            if (left,right,k) in memo:
                return memo[(left,right,k)]
            
            #we either partitino or we dont partition
            curr_avg = (pref_sum[right+1] - pref_sum[left]) / (right - left)
            partition = curr_avg + dp(right+1,right+1,k-1)
            no_partition = dp(left,right+1,k)
            ans = max(partition,no_partition)
            memo[(left,right,k)] = ans
            return ans
        
        return dp(0,N-1,k)

#rodcutting
class Solution:
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        '''
        maximize score of the subarrays, where we can have no more then k non-empty subarrays
        states
            (index,current size of partition, and k left) thats too many states
        need to reduce to two states
        dp(i,k) gives ans using nums[i:] and k deletions
        so we want dp(0,k)
        
        we can use pref_sum array, no need to add starting becaue empty subarrays are not allowed
        i need to keep start,end and partitions left
        
        use pref_sum and k, then look all pref-sums before
        '''
        N = len(nums)
        memo = {}
        
        def dp(i,k):
            if i >= N:
                return 0
            if k <= 0:
                return float('-inf')
            if (i,k) in memo:
                return memo[(i,k)]
            curr_sum = 0
            ans = float('-inf')
            for j in range(i,N):
                curr_sum += nums[j]
                ans = max(ans, curr_sum / (j - i + 1) + dp(j+1,k-1) )
            
            memo[(i,k)] = ans
            return ans
        
        return dp(0,k)
    
#bottom up
class Solution:
    def largestSumOfAverages(self, nums: List[int], k: int) -> float:
        '''
        bottom up
        '''
        N = len(nums)
        dp = [[float('-inf')] * (k + 1) for _ in range(N + 1)]
        
        for i in range(N+1):
            for curr_k in range(k+1):
                if i == N:
                    dp[i][curr_k] = 0
        
        for curr_k in range(1,k+1):
            for i in range(N-1,-1,-1):
                curr_sum = 0
                ans = float('-inf')
                for j in range(i,N):
                    curr_sum += nums[j]
                    ans = max(ans, curr_sum / (j-i+1) + dp[j+1][curr_k-1])
                
                dp[i][curr_k] = ans
        
        return dp[0][k]