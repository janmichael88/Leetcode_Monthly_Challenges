#############################
# 844. Backspace String Compare
# 01MAY22
#############################
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        '''
        just use stacks for both
        '''
        def getString(string):
            stack = []
            
            for ch in string:
                if ch == '#':
                    if stack:
                        stack.pop()
                    else:
                        continue
                else:
                    stack.append(ch)
            return "".join(stack)
        
        return getString(s) == getString(t)


##############################
# 905. Sort Array by Parity
# 02MAY22
###############################
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        '''
        allocate new are with even and off pointers at the ends
        then just advane while we traverse
        '''
        N = len(nums)
        ans = [0]*N
        evens = 0
        odds = N - 1
        
        for num in nums:
            #is even
            if num % 2 == 0:
                ans[evens] = num
                evens += 1
            else:
                ans[odds] = num
                odds -= 1
        
        return ans


##############################################
# 581. Shortest Unsorted Continuous Subarray
# 03MAY22
##############################################
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:
        '''
        i need to find the shortest unosrted continuous subarray
        or rather one continuous subarray that if i only sort this subarray, the whole thing becomes sorted ascendingly
        return the length of this subarray
        
        so the array is nearly sorted, in the worst case we would have to sort the whole thing
        i can just sort the list and check if the numbers are all the same
        
        if they are all the same, return 0, otherwise find the subarray
        
        '''
        N = len(nums)
        sorted_nums = sorted(nums)
        matched = []
        for a,b in zip(sorted_nums,nums):
            matched.append(a == b)
        
        #find the boundaries
        left = None
        right = None
        
        #left boundary
        for i,foo in enumerate(matched):
            if foo == False:
                left = i
                break
        
        #right boundary
        for i in range(N):
            if matched[N-i-1] == False:
                right = N-i-1
                break
        
        if not left and not right:
            return 0
        else:
            return right - left + 1

###############################
# 03MAR22
# 484. Find Permutation
###############################
class Solution:
    def findPermutation(self, s: str) -> List[int]:
        '''
        we are given a strin s of the chars I and D
        I means increasing
        D means decreasing
        
        we want to create an array using nums [1,len(s)+1] that are increasing consecutively or decreasing consecutively
        
        intuioin:
            when we see an I, we know are increaisng, so put the current number into the array
            when we see a D, we need to make a decreasing sequence, instead we push the numbers on to the stack
            only when we hit an I again, we start felling in the results with the stuff we pop from the stack
        
        '''
        res = [0]*(len(s) + 1)
        stack = []
        ptr = 0 #to place into res
        
        for i in range(1,len(s)+1):
            if s[i-1] == 'I':
                stack.append(i)
                while stack:
                    res[ptr] = stack.pop()
                    ptr += 1
            else:
                stack.append(i)
                
        
        #push final element
        stack.append(len(s) + 1)
        while stack:
            res[ptr] = stack.pop()
            ptr += 1
        
        return res

#two pointer
class Solution:
    def findPermutation(self, s: str) -> List[int]:
        '''
        we could also pre allocate an ans array using the numbers [1,n]
        then re-traverse s and reverse along the decreasing sequences
        '''
        res = [i for i in range(1,len(s)+2)]
        
        i = 1
        while i <= len(s):
            j = i
            while i <= len(s) and s[i-1] == 'D':
                i += 1
            res[j-1:i] = res[j-1:i][::-1]
            i += 1
        
        return res

##################################
# 1679. Max Number of K-Sum Pairs
# 03MAY22
##################################
class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        '''
        i can get the freq counts from the array
        for examples
        1:1
        2:1
        3:1
        4:1
        
        check for 1 amd k-1
        '''
        
        operations = 0
        N = len(nums)
        counts = Counter(nums)
        
        for num in nums:
            curr = num
            complement = k - curr
            #if we have numbers to used up
            if counts[curr] > 0 and counts[complement] > 0:
                #specical case, same eleemnt and less than 2,
                if curr == complement and counts[curr] < 2:
                    continue
                #use uo
                counts[curr] -= 1
                counts[complement] -= 1
                operations += 1
        
        return operations

###########################
# 371. Sum of Two Integers
# 08MAY22
############################
class Solution:
    def getSum(self, a: int, b: int) -> int:
        '''
        i could check them bit by bit using bit shift operators
        then rebuild
        
        recall we have XOR, AND, NOT
        
        notes on use cases
            1. a > b or a < b
            2. a < 0 or b < 0
            3. a > 0 or b > 0
            
            thats 2*2*2 = 8 use cases
            
        reduce to 2 simple cases:
            sume of two positive integers , x + y where x > y
            difference of two positive integers x - y, where x > y
            
        notes on XOR:
            if we take the XOR of two numbers, we get the natural sume without carry
         x^y = answer without carry
         
         carry = (x & y) << 1, gives the carry at each bit position
         
         we reduce the the problem to: find the sum of the answer with carry and without carry
         
         if we have two numbers x and y
         
         first start x = x^y, that's without carry
         then find carry y = (x & y) << 1
         
         then apply carry by xoring agian -> recall this woudl be without carry
         x = x^y
         y = (x&y) << 1
         
         we keep doing this while carry != 0
         
         
         now recall, XOR is just the difference, without taking borrow into account
         x^y = x-y without borrow
         then we can just find borrow -> ((~x) & y) << 1
         '''
                
        x, y = abs(a), abs(b)
        # ensure that abs(a) >= abs(b)
        if x < y:
            return self.getSum(b, a)
        
        # abs(a) >= abs(b) --> 
        # a determines the sign
        sign = 1 if a > 0 else -1
        
        if a * b >= 0:
            # sum of two positive integers x + y
            # where x > y
            while y:
                answer = x ^ y
                carry = (x & y) << 1
                x, y = answer, carry
        else:
            # difference of two integers x - y
            # where x > y
            while y:
                answer = x ^ y
                borrow = ((~x) & y) << 1
                x, y = answer, borrow
        
        return x * sign

#############################
# 353. Design Snake Game
# 09MAY22
#############################
#close one....im too tired
#right idea though
class SnakeGame:
    '''
    when the snake passes over food, it gains 1 value in length
    the tail will always follow the head of the snake
    wherever the last part of the snake was, just after the snake eats it, its lenght increases there
    we need to keep track of all the positions of the tail
    say for example we have an array of x,y positions [pos_1,pos_2....pos_N]
    upon a new input, we move the positions to the right once, and add the new location
    [pos_1,pos_2....pos_N,pos_N+1]
    '''

    def __init__(self, width: int, height: int, food: List[List[int]]):
        #we can make this a q, then just push and pop when we move or eat food
        self.body = deque([])
        self.body.append((0,0))
        
        #fast lookup for snake body
        self.body_set = set()
        self.body_set.add((0,0))
        
        #fast look up for soot
        self.food_set = set()
        for x,y in food:
            self.food_set.add((x,y))
            
        #direction variables
        self.dirrs = {'R': (1,0),
                      'L': (-1,0),
                      'U': (0,1),
                      'D': (0,-1)
                     }
        
        self.width = width
        self.height = height
        self.curr_pos = (0,0)
        

    def move(self, direction: str) -> int:
        #get next moves
        dx,dy = self.dirrs[direction]
        
        #find next position
        new_x,new_y = self.curr_pos[0] + dx, self.curr_pos[1] + dy
        
        #bounds check or next move causes it to eat itself
        if   (not (0 <= new_x < self.width)) or  (not(0 <= new_y < self.height)) or (new_x,new_y) in self.body_set:
            return -1
        
        #check if next move is food
        if (new_x,new_y) in self.food_set:
            self.body.append((new_x,new_y))
            self.food_set.remove((new_x,new_y))
            self.body_set.add((new_x,new_y))
            self.curr = (new_x,new_y)
        #else, not food, shift
        else:
            removed = self.body.popleft()
            self.body_set.remove(removed)
            self.body.append((new_x,new_y))
            self.body_set.add((new_x,new_y))
            self.curr = (new_x,new_y)
        
        return len(self.body) 
            


# Your SnakeGame object will be instantiated and called as such:
# obj = SnakeGame(width, height, food)

class SnakeGame:

    def __init__(self, width: int, height: int, food: List[List[int]]):
        """
        Initialize your data structure here.
        @param width - screen width
        @param height - screen height 
        @param food - A list of food positions
        E.g food = [[1,1], [1,0]] means the first food is positioned at [1,1], the second is at [1,0].
        """
        self.snake = collections.deque([(0,0)])    # snake head is at the front
        self.snake_set = {(0,0) : 1}
        self.width = width
        self.height = height
        self.food = food
        self.food_index = 0
        self.movement = {'U': [-1, 0], 'L': [0, -1], 'R': [0, 1], 'D': [1, 0]}
        

    def move(self, direction: str) -> int:
        """
        Moves the snake.
        @param direction - 'U' = Up, 'L' = Left, 'R' = Right, 'D' = Down 
        @return The game's score after the move. Return -1 if game over. 
        Game over when snake crosses the screen boundary or bites its body.
        """
        
        newHead = (self.snake[0][0] + self.movement[direction][0],
                   self.snake[0][1] + self.movement[direction][1])
        
        # Boundary conditions.
        crosses_boundary1 = newHead[0] < 0 or newHead[0] >= self.height
        crosses_boundary2 = newHead[1] < 0 or newHead[1] >= self.width
        
        # Checking if the snake bites itself.
        bites_itself = newHead in self.snake_set and newHead != self.snake[-1]
        
        # If any of the terminal conditions are satisfied, then we exit with rcode -1.
        if crosses_boundary1 or crosses_boundary2 or bites_itself:
            return -1

        # Note the food list could be empty at this point.
        next_food_item = self.food[self.food_index] if self.food_index < len(self.food) else None
        
        # If there's an available food item and it is on the cell occupied by the snake after the move, eat it
        if self.food_index < len(self.food) and \
            next_food_item[0] == newHead[0] and \
                next_food_item[1] == newHead[1]:  # eat food
            self.food_index += 1
        else:    # not eating food: delete tail                 
            tail = self.snake.pop()  
            del self.snake_set[tail]
            
        # A new head always gets added
        self.snake.appendleft(newHead)
        
        # Also add the head to the set
        self.snake_set[newHead] = 1

        return len(self.snake) - 1
        


# Your SnakeGame object will be instantiated and called as such:
# obj = SnakeGame(width, height, food)
# param_1 = obj.move(direction)
# param_1 = obj.move(direction)

##################################
# 216. Combination Sum III
# 10MAY22
##################################
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        '''
        we want to find all combindations of length k, the sum to n
        such that only numbers 1 to 9 are used, and each number is used only once
        '''
        combinations = set()
        taken = [False]*9
        def backtrack(start,criteria,path):
            #reduced to 0
            if criteria == 0:
                if len(path) == k:
                    combinations.add(frozenset(path[:]))
                    return
                elif len(path) > k:
                    return
            
            for candidate in range(start,9):
                if not taken[candidate-1]:
                    path.append(candidate)
                    taken[candidate-1] = True
                    backtrack(start+1,criteria - candidate,path)
                    path.pop()
                    taken[candidate -1] = False
        
        backtrack(1,n,[])
        return combinations
            

class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        '''
        we want to find all combindations of length k, the sum to n
        such that only numbers 1 to 9 are used, and each number is used only once
        '''
        combinations = []
        def backtrack(start,criteria,path):
            #reduced to 0
            if criteria == 0 and len(path) == k:
                combinations.append(path[:])
                return
            elif criteria < 0 or len(path) == k:
                return
            
            for candidate in range(start,10):
                path.append(candidate)
                #it is NOT start + 1, rather candidate + 1
                #watch how we backtrack for the other combinations problem series
                backtrack(candidate+1,criteria - candidate,path)
                path.pop()
        
        backtrack(1,n,[])
        return combinations

#don't forget about other solutions
from itertools import combinations
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        '''
        using combinations
        
        '''
        res = []
        for c in combinations(range(1,10),k):
            if sum(c) == n:
                res.append(c)
        
        return res
        
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        '''
        another recursive solution, but no global variable
        
        There's nothing "null" or "invalid" about those cases. If both k=0 and n=0, then there is exactly one combination, namely the empty combination. It has zero numbers and they add up to zero. So the answer is simply [[]]. In all other cases where k=0 or n=0 or even k<0 or n<0, there is no combination, so the answer is [].
        '''
        def combs(k,n,cap):
            #base case, when k == 0 or n === 0
            if not k and not n:
                return [[]]
            elif not k and n != 0:
                return []
            res = []
            #add last digit to previous computed asnwers from combs
            for last in range(1,cap):
                #for each answer in combs, add a new one, reduced by 1
                for comb in combs(k-1,n-last,last):
                    res.append(comb + [last])
            return res
        
        return combs(k,n,10)

#iterative
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        combs = [[]]
        for _ in range(k):
            next_combs = []
            for comb in combs:
                for first in range(1,comb[0] if comb else 10):
                    next_combs.append([first] + comb)
            
            combs = next_combs
        
        return [comb for comb in combs if sum(comb) == n]

##################################
# 1641. Count Sorted Vowel Strings
# 11MAY22
##################################
class Solution:
    def countVowelStrings(self, n: int) -> int:
        '''
        we know backtracking would take too long, why?
        for each positino in the string, we could take any vowel, rather our choices diminish since we must chose 
        a vowel that comes lexogrpahically after the vowel we are currently on
        the time complexity is actually really hard, and not facorial, but rather polynomial
        turns our we can use combinations formula and sum them up
        (5 choose i-1) + (4 choose i-1) + ..... + (1 choose i - 1)
        which is upper bounded by O(n^5)
        
        now for the algorithm, 
        we can encdoe the vowwles a,e,i,o,u as the numbers 1,2,3,4,5
        
        we define dp(n,vowel) as the number of lexogrpahic strings with size n and using all the current vowel
        
        recurrence:
            dp(n,vowel) = sum of dp(n-1,v) where v is all vowels after the current vowel we are on
            dp(n,i) = sum(dp(n-1,j) for j in range(i+1,5+1))
            base case is when n == 0, and we have made a possible string
        '''
        memo = {}
        def dp(n,start_vowel):
            if n == 0:
                return 1
            if (n,start_vowel) in memo:
                return memo[(n,start_vowel)]
            res = 0
            for curr_vowel in range(start_vowel,5+1):
                res += dp(n-1,curr_vowel)
            memo[(n,start_vowel)] = res
            return res
        
        return dp(n,1)

class Solution:
    def countVowelStrings(self, n: int) -> int:
        '''
        we can also turn this into dp
        '''
        dp = [[0]*(5+1) for _ in range(n+1)]
        
        #base cases when n == 0, we have at least 1
        for j in range(1,5+1):
            dp[0][j] = 1
        
        for i in range(1,n+1):
            for start_vowel in range(1,5+1):
                res = 0
                for curr_vowel in range(start_vowel,5+1):
                    res += dp[i-1][curr_vowel]
                dp[i][start_vowel] = res
        
        return dp[n][1]

#another way of breaking\class Solution:
    def countVowelStrings(self, n: int) -> int:
        '''
        another way of defining the recurrence is looking back at previous subrpoblems for both previous calcualted n
        and previously calcualted vowels
        
        we first need to breack the subproblem into dp(n,vowles)
        where dp(n,vowels) is number of strings of size n using vowels 1 to vowels
        
        for example
        dp(2,3) = 6
        aa, ae, ai, ee, ei, ii
        
        if we are at dp(n,v) = dp(n-1,v) + dp(n,v-1)
        why? 
        if we already new the answer to dp(n,v) we could examine dp(n-1,v) and we could also examine dp(n,v-1)
        
        the base case, if we only have 1 letter, there are 5 answers
        '''
        memo = {}
        
        def dp(n,v):
            if n == 1:
                return v
            if v == 1:
                return 1
            if (n,v) in memo:
                return memo[(n,v)]
            ans = dp(n-1,v) + dp(n,v-1)
            memo[(n,v)] = ans
            return ans
        
        return dp(n,5)

#bottom dp
class Solution:
    def countVowelStrings(self, n: int) -> int:
        '''
        translating into dp
        '''
        
        dp = [[0]*(5+1) for _ in range(n+2)]
        
        #base cases
        for vowel in range(1,5+1):
            dp[1][vowel] = 5
        for i in range(n+1):
            dp[i][1] = 1
        
        for i in range(1,n+2):
            for j in range(1,5+1):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[n+1][5]

################################
# 365. Water and Jug Problem
# 12MAY22
#################################
#BFS with state transitions
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        '''
        we can use bfs to to go from a state to another state
        we want to get to a point where jug1 + jug2 == target
        initally we are at the 0,0 state
        
        from a state, lets call it (a,b) we can:
        1. empty the first jug (0,b)
        2. empty the second jub (a,0)
        3. fill the first jug (x,b)
        4. fill the second jug  (a,y)
        5. pour the second jug into the first (min(x,b+a), 0 of b < x - a else b - (x-a))
        6. pur the first jug into the second (0 if a + b < y else a - (y-b),min(b+a,y))
        
        '''
        #first check if we can even do this
        if x + y < z:
            return False
        
        #initial state is the empty jug state
        q = deque([(0,0)])
        visited = set((0,0))
        
        while q:
            first,second = q.popleft()
            #we can make it
            if first + second == z:
                return True
            
            #generate next moves
            next_states = [(0,second),(first,0),(x,second),(first,y),
                          (min(x,first+second), 0 if second < x - first else second - (x - first)),
                           (0 if first + second < y else first - (y - second),min(first+second,y))
                          ]
            
            for state in next_states:
                if state in visited:
                    continue
                else:
                    q.append(state)
                    visited.add(state)
        
        return False

class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        '''
        just another way defining the transitions a little easier
        '''
        if x + y < z:
            return False
        
        visited = set()
        q = deque([(0,0)])
        
        while q:
            i,j = q.popleft()
            visited.add((i,j))
            
            if i + j == z:
                return True
            
            moves = set([
                (x, j), (i, y), (0, j), (i, 0),
                (min(i + j, x), (i + j) - min(i + j, x)),
                ((i + j) - min(i + j, y), min(i + j, y)),
            ])
            
            for move in moves:
                if move not in visited:
                    q.append((move))
        
        return False
        