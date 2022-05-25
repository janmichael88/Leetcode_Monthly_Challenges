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
        
class Solution:
    def canMeasureWater(self, x: int, y: int, z: int) -> bool:
        '''
        the problem really just aks if z is a multiple of the GCD(x,y)
        this comes froms bezouts lemma
        
        let a and b be non zero intiegers and elt d be their greatest common divisor,
        then there exist integers x and y such that ax + by = d
        
        in addition, the GCD d is the smallest positive integers that can be written as ax + by
        every integer of the form ax + by is a multiple of the GCD d
        
        If a or b is negative this means we are emptying a jug of x or y gallons respectively.

        Similarly if a or b is positive this means we are filling a jug of x or y gallons respectively.

        x = 4, y = 6, z = 8.

        GCD(4, 6) = 2

        8 is multiple of 2

        so this input is valid and we have:

        -1 * 4 + 6 * 2 = 8

        In this case, there is a solution obtained by filling the 6 gallon jug twice and emptying the 4 gallon jug once. (Solution. Fill the 6 gallon jug and empty 4 gallons to the 4 gallon jug. Empty the 4 gallon jug. Now empty the remaining two gallons from the 6 gallon jug to the 4 gallon jug. Next refill the 6 gallon jug. This gives 8 gallons in the end)
        '''
        if x + y < z:
            return False
        if x == z or y == z or x + y == z:
            return True
        
        while y != 0:
            print(x,y)
            #keep trying to see if y can go into x
            temp = y
            y = x % y
            x = temp
        
        return z % x == 0

#snippets for finding the GCD
def gcd(a,b):
	if b == 0:
		return a
	if a == 0:
		return b
	else:
		return gcd(b, a % b)



def computeGCD(x, y):
  
    if x > y:
        small = y
    else:
        small = x
    for i in range(1, small+1):
        if((x % i == 0) and (y % i == 0)):
            gcd = i
              
    return gcd

 def computeGCD(x, y):
  
   while(y):
       x, y = y, x % y
  
   return x

############################
# 117. Populating Next Right Pointers in Each Node II (REVISITED)
# 14MAY22
###########################
#covering the second approahc where space is opmtizimed
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        '''
        if we look, the connected nodes at each level form a linked list, 
        after makgin a connectiong, we go down a level
        inution: we only move on the level N+1 when we are done
        
        algo:
            1. start at root node, and establich connections
            2. we want to use contstante space, so we can keep a prev,curr pointer and move through the level 
            3. to start at a level we just need the current leftmost nodes
            
            template
            leftmost = curr
            while leftmost is not None:
                curr = leftmost
                prev = None
                while curr is None:
                    process(left)
                    process(right)
                    set leftmost
                    curr = curr.next
        
        process function:
            1. if prev is assigned a non null value, startwth null, and make left to prev
            2. no left, child, then set it right
            3. no children, maintain current prev and advance
            4. otherwise, we need to update
        '''
        if not root:
            return root
        
        #set leftmost for each level
        leftmost = root
        
        #need to keep finding leftmodt
        while leftmost:
            #prev tracks latest node on next levle, curr is our pointer on this level
            prev,curr = None,leftmost
            
            #reset we can ALWAYS find a leftmost on the next leftl
            leftmost = None
            
            while curr:
                prev,leftmost = self.process(curr.left,prev,leftmost)
                prev,leftmost = self.process(curr.right,prev,leftmost)
                curr = curr.next
        
        return root
    
    def process(self, child,prev,leftmost):
        if child:
            # If the "prev" pointer is alread set i.e. if we
            # already found atleast one node on the next level,
            # setup its next pointer
            if prev:
                prev.next = child
            # Else it means this child node is the first node
            # we have encountered on the next level, so, we
            # set the leftmost pointer
            else:
                leftmost = child
            prev = child
        
        return prev, leftmost

#############################
# 743. Network Delay Time
# 14MAY22
############################
#close one....
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        '''
        this is a graph problem
        we are given a source node, and weighted edge list
        return the time it takes for all n nodes to recevie the singnal
        if it is impossible, retyrn -1
        
        we can check that all the nodes are connected, if they are not connected, return-1
        the edges are directed
        
        '''
        #generate edge list, u: liist([v,weight])
        adj_list = defaultdict(list)
        
        for u,v,weight in times:
            adj_list[u].append([v,weight])
        
        #dfs function to check if we can at least touch
        def dfs(node,seen):
            seen.add(node)
            for (neigh,weight) in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen)
        
        seen = set()
        dfs(k,seen)
        if len(seen) != n:
            return -1
        
        #now that graph problem, we know from here the nodes are all connected
        #bfs, but for each node, update the total time with the max of its neighbors
        ans = 0
        seen = set()
        q = deque([k])
        
        while q:
            curr = q.popleft()
            seen.add(curr)
            
            largest_time = 0
            for neigh,time in adj_list[curr]:
                if neigh not in seen:
                    largest_time = max(largest_time,time)
                    q.append(neigh)
            
            ans += largest_time
    
        return ans

#this is stupid, it's really just single shortest path, starting from node k
#with the added criteria that we touch all n nodes
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        '''
        well the problem boils down to finding the time required to recieve the signal for allnodes
        turns out it is the maximum time. why? because we need to find the time at which all nodes receie the signal
        so the time stamp which the last node receives the signal is the answer!
        
        lets explore dfs and bfs first
        
        created the edge list in the usual manner,
        then while doing dfs, we try to visit all the nodes, and try relaxing each edge 
        we visited and update the edges if when going across that edge, we can reduce the amount of time
        if so, we update
        we keep going until we can't 
        return the max answer
        
        we can save time by sorting the edge list by weight
        '''
        adj_list = defaultdict(list)
        for u,v,weight in times:
            adj_list[u].append((weight,v))
            
        #sort edges by weight
        for v in adj_list:
            adj_list[v] = sorted(adj_list[v])
            
        visited = {}
        
        def dfs(node,time):
            if node in visited and time >= visited[node]:
                return
            
            visited[node] = time
            
            if node not in adj_list:
                return
            
            for t,neigh in adj_list[node]:
                dfs(neigh, time + t)
        
        dfs(k,0)
        return max(visited.values()) if len(visited) == n else -1

#bfs
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        '''
        well the problem boils down to finding the time required to recieve the signal for allnodes
        turns out it is the maximum time. why? because we need to find the time at which all nodes receie the signal
        so the time stamp which the last node receives the signal is the answer!
        
        lets explore dfs and bfs first
        
        created the edge list in the usual manner,
        then while doing dfs, we try to visit all the nodes, and try relaxing each edge 
        we visited and update the edges if when going across that edge, we can reduce the amount of time
        if so, we update
        we keep going until we can't 
        return the max answer
        
        we can save time by sorting the edge list by weight
        '''
        adj_list = defaultdict(list)
        for u,v,weight in times:
            adj_list[u].append((weight,v))
            
        #sort edges by weight
        for v in adj_list:
            adj_list[v] = sorted(adj_list[v])
            
        visited = {}
        
        q = deque([(k,0)])
        
        while q:
            node,time = q.popleft()

            if node in visited and time >= visited[node]:
                continue
            
            visited[node] = time
            
            if node not in adj_list:
                continue
            
            for t,neigh in adj_list[node]:
                q.append((neigh, time + t))
        
        return max(visited.values()) if len(visited) == n else -1
            

#insteadf of sorting all the edges initially, we can use a heap
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        '''
        we can use Dijkstra's algo, and use a heap to keep the smallest edges at each time
        then just continue to relax the edges
        '''
        adj_list = defaultdict(list)
        for u,v,weight in times:
            adj_list[u].append((weight,v))
            
            
        visited = {}
        
        #keep in mind, python is a min heap
        heap = [(0,k)]
        
        while heap:
            time,node = heapq.heappop(heap) 

            if node in visited and time >= visited[node]:
                continue
            
            visited[node] = time
            
            if node not in adj_list:
                continue
            
            for t,neigh in adj_list[node]:
                heapq.heappush(heap, ((time + t),neigh))
        
        return max(visited.values()) if len(visited) == n else -1

#######################################
# 1091. Shortest Path in Binary Matrix (Revisited)
# 16MAY22
#######################################
#TLE! this is a win in my book
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        '''
        this is the single source shortes path problem, its kinda like dijsktras but in this case all the edges are on
        we do try dfs, and try to relax an edge when we see a zero
        we can create a hash map, and into this hashmap we store (cell) : path length
        which represents the length to get to that cell, while we dfs, we see if we can get to that path by relaxing it
        one we are done, we check is the bottom left is in the mapp and return its answer
        '''
        
        if grid[0][0] == 1:
            return -1
        rows = len(grid)
        cols = len(grid[0])
        
        
        dirrs = []
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                if (x,y) != (0,0):
                    dirrs.append((x,y))
            
        
        seen = {} #<cell:path>
        def dfs(row,col,path):
            #if already seen
            if (row,col) in seen and path >= seen[(row,col)]:
                return
            #put
            seen[(row,col)] = path+1
            #recurse
            for dx,dy in dirrs:
                neigh_row = row + dx
                neigh_col = col + dy
                if (0 <= neigh_row < rows) and (0 <= neigh_col < cols):
                    if grid[neigh_row][neigh_col] == 0:
                        dfs(row+dx,col+dy,path+1)
                
        dfs(0,0,0)
        return seen[(rows-1,cols-1)] if (rows-1,cols-1) in seen else -1

class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        '''
        since the edges to each node are at a distnace of 1
        just use bfs and update distance in each cell accordingly
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dirrs = []
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                if (x,y) != (0,0):
                    dirrs.append((x,y))
        
        if grid[0][0] != 0 or grid[-1][-1] != 0:
            return -1
        
        def get_neighbors(row,col):
            for dx,dy in dirrs:
                neigh_x = row+dx
                neigh_y = col+dy
                if not (0 <= neigh_x < rows) or not (0 <= neigh_y < cols):
                    continue
                if grid[neigh_x][neigh_y] != 0: #remember we update in place
                    continue
                yield (neigh_x,neigh_y)
                
        
        q = deque([(0,0)])
        
        while q:
            row,col = q.popleft()
            #get curr_dist
            distance = grid[row][col]
            if (row,col) == (rows -1,cols -1):
                return distance + 1
            
            for neigh_x,neigh_y in get_neighbors(row,col):
                grid[neigh_x][neigh_y] = distance + 1
                q.append((neigh_x,neigh_y))
        print(grid)
        return -1

class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        '''
        since the edges to each node are at a distnace of 1
        just use bfs and update distance in each cell accordingly
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dirrs = []
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                if (x,y) != (0,0):
                    dirrs.append((x,y))
        
        if grid[0][0] != 0 or grid[-1][-1] != 0:
            return -1
        
        def get_neighbors(row,col):
            for dx,dy in dirrs:
                neigh_x = row+dx
                neigh_y = col+dy
                if not (0 <= neigh_x < rows) or not (0 <= neigh_y < cols):
                    continue
                if grid[neigh_x][neigh_y] != 0: #remember we update in place
                    continue
                yield (neigh_x,neigh_y)
                
        
        q = deque([(0,0,0)])
        seen = set()
        
        while q:
            row,col,distance = q.popleft()
            #get curr_dist
            if (row,col) == (rows -1,cols -1):
                return distance + 1
            
            for neigh_x,neigh_y in get_neighbors(row,col):
                if (neigh_x,neigh_y) in seen:
                    continue
                q.append((neigh_x,neigh_y,distance+1))
                seen.add((neigh_x,neigh_y))
        print(grid)
        return -1
            
#using A*
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        '''
        A* if very similar to BFS, but during path discovery we want to priotize more promising paths
        we need to define heuristic, a potential function, that measures how much promise the option has
        then we can priotize the options
        
        not only do we priortize by distance traveled so far, but also the most optmistic estiamte how how many more steps it would take to get to the bottom right
        
        mathematicaly, the estiamte for the remainder of the path is:
            the maximmum out of the number of rows and columns remaining
            and assume that the first path we discoverd into a cell is the best one
            instead keep track of all the options and then choose the best one when we get to it
            
            in each cell, include distance traveled so far and distant traveled so far + max(remianing)
            if we did revisit a cell, keep all possible value, then take the min
            
        one thing to notice too:
            the A* estimates from a parent cell are never more than its children
            A "child" cell could never have a lower estimate than a "parent" cell. If it did, this would mean that the "parent" cell's estimate was not the lowest possible
            No cell can have an estimate lower than that of the top-left cell
            
        proof of correctness is still really hard to get
            The key idea is that the estimates for any given path from the top-left to bottom-right cell are non-decreasing; 
            proof by contradiction
            
        we push on the min heap the A* heuristic and its corresponding cell
            
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dirrs = []
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                if (x,y) != (0,0):
                    dirrs.append((x,y))
        
        if grid[0][0] != 0 or grid[-1][-1] != 0:
            return -1
        
        def get_neighbors(row,col):
            for dx,dy in dirrs:
                neigh_x = row+dx
                neigh_y = col+dy
                if not (0 <= neigh_x < rows) or not (0 <= neigh_y < cols):
                    continue
                if grid[neigh_x][neigh_y] != 0: #remember we update in place
                    continue
                yield (neigh_x,neigh_y)
        
        #helper functino for A* heuristic
        def best_case_estimate(row,col):
            return max(rows - row, cols - col)
        
        # Entries on the priority queue are of the form
        # (total distance estimate, distance so far, (cell row, cell col))
        pq = [(1 + best_case_estimate(0, 0), 1, (0, 0))]
        visited = set()
        
        while pq:
            estimate,distance,cell = heapq.heappop(pq)
            row,col = cell[0],cell[1]
            if cell in visited:
                continue
            if cell == (rows - 1, cols -1):
                return distance
            visited.add(cell)
            for neigh_x,neigh_y in get_neighbors(row,col):
                if (neigh_x,neigh_y) in visited:
                    continue
                #get new A* estimates
                estimate = best_case_estimate(neigh_x,neigh_y) + distance + 1
                entry = (estimate,distance + 1, (neigh_x,neigh_y))
                heapq.heappush(pq,entry)
                
        
        return -1

##############################
# 694. Number of Distinct Islands (REVISITED)
# 16MAY22
##############################
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        '''
        we can use dfs
        add cells to global island path, which we do for each dfs call on each cell
        after dfsing' we need to check if the island is unique
        
        to assist in checking of island unqiqueness, subtract origin cell from all explored cells during dfs
        once we have an island, check uniqueness:   
            first ensure lenghts are the same
            then check cells equivalence
        '''
        
        rows = len(grid)
        cols = len(grid[0])
        seen = set()
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        def dfs(row,col):
            #bounds check
            if not (0 <= row < rows) or not (0 <= col < cols):
                return
            #not a 1 or already seen
            if (row,col) in seen or not grid[row][col]:
                return
            #otherwise it msut be a 1, create
            #we shoudl talk about this, we could pass all this shit in the function,
            #but create them globally within the loop (global at loop scope)
            curr_island.append((row - row_origin,col - col_origin))
            seen.add((row,col))
            
            for dx,dy in dirrs:
                dfs(row+dx,col+dy)
                
        def isUnique():
            for other_island in unique_islands:
                if len(other_island) != len(curr_island):
                    continue #no point in comparing
                #recall we offsetted from the origin
                for cell_1,cell_2 in zip(curr_island,other_island):
                    if cell_1 != cell_2:
                        break
                else:
                    return False
            return True
        
        unique_islands = []
        for row in range(rows):
            for col in range(cols):
                curr_island = []
                row_origin = row
                col_origin = col
                dfs(row,col)
                if not curr_island or not isUnique():
                    continue
                unique_islands.append(curr_island)
                
        return len(unique_islands)

#using frozen sets
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        '''
        we can also hash by local cooridnates
        instated of using array to keep an island, we can use a frozen set
        '''
        
        rows = len(grid)
        cols = len(grid[0])
        seen = set()
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        def dfs(row,col):
            #bounds check
            if not (0 <= row < rows) or not (0 <= col < cols):
                return
            #not a 1 or already seen
            if (row,col) in seen or not grid[row][col]:
                return
            #otherwise it msut be a 1, create
            #we shoudl talk about this, we could pass all this shit in the function,
            #but create them globally within the loop (global at loop scope)
            curr_island.add((row - row_origin,col - col_origin))
            seen.add((row,col))
            
            for dx,dy in dirrs:
                dfs(row+dx,col+dy)
                
        
        unique_islands = set()
        for row in range(rows):
            for col in range(cols):
                curr_island = set()
                row_origin = row
                col_origin = col
                dfs(row,col)
                if curr_island:
                    unique_islands.add(frozenset(curr_island))
                
        return len(unique_islands)

#hash by path signature
class Solution:
    def numDistinctIslands(self, grid: List[List[int]]) -> int:
        '''
        another way of uniquely hashing would be to use the directions we go in
        U,D,L,R
        which is up,down,left,right
        
        don't forget to mark when we bacttracked by adding a speical character
        '''
        
        rows = len(grid)
        cols = len(grid[0])
        seen = set()
        dirrs = [(1,0,'U'),(-1,0,'D'),(0,1,'R'),(0,-1,'L')]
        def dfs(row,col,direction):
            #bounds check
            if not (0 <= row < rows) or not (0 <= col < cols):
                return
            #not a 1 or already seen
            if (row,col) in seen or not grid[row][col]:
                return
            #otherwise it msut be a 1, create
            #we shoudl talk about this, we could pass all this shit in the function,
            #but create them globally within the loop (global at loop scope)
            curr_island.append(direction)
            seen.add((row,col))
            
            for dx,dy,d in dirrs:
                dfs(row+dx,col+dy,d)
            #backtrack
            curr_island.append('*')
                
        
        unique_islands = set()
        for row in range(rows):
            for col in range(cols):
                curr_island = []
                row_origin = row
                col_origin = col
                dfs(row,col,'*')
                if curr_island:
                    print("".join(curr_island))
                    unique_islands.add("".join(curr_island))
                
        return len(unique_islands)

#################################
# 351. Android Unlock Patterns
# 17MAY22
#################################
class Solution:
    def numberOfPatterns(self, m: int, n: int) -> int:
        '''
        for a valid unlock pattern
            all the dots in the sequence are distance
            if line segment connecting two dots passes through the CENTER of any another dot
                then the other dot MUST have previously appear in the seqience
                EXAMPLE:
                    connecting dots 2 through 9 w/ot 5 or 6 appearing beforehandisvalid because linke from 2 to 9
                    does not pass through the center of either 5 or 6
                    CENTER is they key
        
        we want the number of valid unlock patterns with at least m keys
        an at most n keys
        
        there is also a recurrence to i think
        for v value of n, we can sum up values (n=1),(n-2),(n-3)....(n-4) up to n-max(n-1,0))
        
        we could use dfs to generate all possible paths, since n can never exceed 9
        
        inution:
            lets call the current number we are on num, and its next number, nextNum
            to reach nextNum from num we have to pass obstacles
            and we can only pass this obstalce if we have previously passed it before
                recall in usual dfs we check all neighbors!
                instead we keep track of all nums visted so far, and only cross this obstacle if we have seen ti before
                
        '''
        #keep map of edges and along an edge it maps to an obstacle
        #<edge : cross through number>
        obstacles = { (1,3): 2, (1,7): 4, (1,9): 5, (2,8): 5, 
                      (3,7): 5, (3,1): 2, (3,9): 6, (4,6): 5, 
                      (6,4): 5, (7,1): 4, (7,3): 5, (7,9): 8, 
                      (8,2): 5, (9,7): 8, (9,3): 6, (9,1): 5
                    }
        
        self.num_patterns = 0
        
        def dfs(num,count,m,n):
            #consider only patterns with count in range [m,n]
            if m <= count <= n:
                self.num_patterns += 1
            #we can have no more than n
            if count == n:
                return
            
            #add to visited, we recall we dfs from each number between 1 and 0
            visited.add(num)
            for next_num in range(1,10):
                #if we haven't seen this yet
                if next_num not in visited:
                    #if edge has obstacle, and if we have yet to meet this obstacle, we cannot consider the path
                    if (num,next_num) in obstacles and obstacles[(num,next_num)] not in visited:
                        continue
                    dfs(next_num,count+1,m,n)
            #backtrack
            visited.remove(num)
            
        for i in range(1,10):
            visited = set()
            dfs(i,1,m,n)
        return self.num_patterns

###############################
# 355. Design Twitter
# 17MAY22
###############################
#close one..
class Twitter:

    def __init__(self):
        '''
        we can use two hashmaps
        one mapping userId to tweets
        one mapping userId to followers
        keep global time counter
        '''
        self.tweets = defaultdict(list)
        self.following = defaultdict(set)
        self.order = 0
        

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((self.order,tweetId))
        self.order += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        #pull all tweets from user and followers
        user_tweets = self.tweets[userId]
        follower_tweets = []
        for follower in self.following[userId]:
            follower_tweets += self.tweets[follower]
        all_tweets = user_tweets + follower_tweets
        all_tweets.sort(reverse = True)
        return [tweet for time,tweet in all_tweets[-10:]]

    def follow(self, followerId: int, followeeId: int) -> None:
        self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.following[followerId].discard(followeeId)


# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)

class Twitter:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.tweets = collections.defaultdict(list)
        self.following = collections.defaultdict(set)
        self.order = 0
    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        self.tweets[userId] += (self.order, tweetId), 
        self.order -= 1

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        tw = sorted(tw for i in self.following[userId] | {userId} for tw in self.tweets[i])[:10]
        return [news for i, news in tw]
    

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.following[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        self.following[followerId].discard(followeeId)     
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)

class Twitter:

    # Each user has a separate min heap
    # if size of heap is lesser than 10 keep pushing tweets and when it's full, poppush
    # use a defaultdict to associate user id's to their heaps
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.following = defaultdict(set)
        self.user_tweets = defaultdict(deque)
        self.post = 0

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        """
        self.post += 1
        tweets = self.user_tweets[userId]
        tweets.append(((self.post), tweetId))
        if len(tweets) > 10:
            tweets.popleft()
        

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        """
        h = []
        u = self.user_tweets[userId]
		h.extend(u)
        heapify(h)
        for user in self.following[userId]:
            tweets = self.user_tweets[user]
            for x in range(len(tweets) - 1, -1, -1):
                if len(h) < 10:
                    heappush(h, tweets[x])
                else:
                    if h[0][0] < tweets[x][0]:
                        heappushpop(h, tweets[x])
                    else:
                        break
        return [heappop(h)[1] for x in range(len(h))][::-1]

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        """
        if followerId != followeeId:
                self.following[followerId].discard(followeeId)

#############################################
# 329. Longest Increasing Path in a Matrix (REVISITED)
# 19MAY22
#############################################
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        '''
        we can use dfs to keep track of the length of the longest path starting from (i,j)
        dp(i,j) = give max length of path ending at i,j
        dp(i,j) = {
            onlf if the the neigh is increasing in value
            for all neighs from i,j:
                ans = max(current (i,j), dp(i,j) for all i,j neighbors)
        }
        
        then we invoke dp for all i,j spots in the matrix to get the final answer
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        memo = {}
        
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dp(i,j):
            if (i,j) in memo:
                return memo[(i,j)]
            ans = 0
            for dx,dy in dirs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds check and we can step
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and matrix[neigh_x][neigh_y] > matrix[i][j]:
                    #maximize from previous computations
                    if (neigh_x,neigh_y) in memo:
                        ans = max(ans,memo[(neigh_x,neigh_y)])
                    #maximize using recursive case
                    else:
                        ans = max(ans,dp(neigh_x,neigh_y))
            #move to next cell increase path length by 1
            memo[(i,j)] = ans + 1
            return ans + 1
        
        res = 0
        for i in range(rows):
            for j in range(cols):
                res = max(res,dp(i,j))
        
        return res
            
#peeling an onion
#find topolical ordering frist
#then process
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        '''
        this problem is partifularly painful because we are not given the topological sorted order
        rather, we do not have a base case
        we already have defined the recurrence relation as
        f(i,j)=max{f(x,y)∣(x,y) is a neighbor of(i,j) and matrix[x][y]>matrix[i][j]}+1
        
        we can use the peeling onion to try to define the base cases,
        then establish the ordering
        then from the ordering we can apply the recurrence
        
        the idea is that in a DAG, we will have leaf elements where we can directly comptue an answer
        we put these leaves in a list, and tehn remove from DAG
        
        after the removal, there will be new leaves
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        memo = {}
        
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        #we need to find the (i,j) cells that do not depend on previous computations, i.e their out degree is zero
        outDegree = [[0]*(cols) for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                for dx,dy in dirs:
                    neigh_x = i + dx
                    neigh_y = j + dy
                    #bounds, check and icnreasing
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols and matrix[neigh_x][neigh_y] > matrix[i][j]:
                        outDegree[i][j] += 1
        
        #bfs, q up 0 outdegree cells
        q = deque([])
        for i in range(rows):
            for j in range(cols):
                if outDegree[i][j] == 0:
                    q.append((i,j))
                    
        res = 0
        while q:
            #process first layer
            res += 1
            N = len(q)
            for _ in range(N):
                i,j = q.popleft()
                for dx,dy in dirs:
                    neigh_x = i + dx
                    neigh_y = j + dy
                    #bounds, check and icnreasing
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        if matrix[neigh_x][neigh_y] > matrix[i][j]:
                            outDegree[neigh_x][neigh_y] -= 1
                            if outDegree[neigh_x][neigh_y] == 0:
                                q.append((neigh_x,neigh_y))
                    else:
                        continue
        
        return res
                    
###################################
# 10. Regular Expression Matching
# 19MAY22
###################################
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        '''
        if we didn't have any Kleene stars and just dots, then we would just pass through both string and pattern
        continuing only on dots and verifying chars match
        
        when a star is present, we need to check different suffixes and see if they match the rest of the pattern
        
        w/o kleen star we can have:
        
        def match(s,p):
            if not p:
                return not s
            first_match = bool(s) and p[0] in {s[0],'.'}
            return first_match and match(s[1:],p[1:])
            
        inution on match,
            basicially it's true if we have a first match and we can return true for the rest of the string and the pattern
            
        if a * is present in the pattern, it will be in the second position pattern[1]
        then we can ignore this part of the pattern or delete a amtchine char in the text
        if we have match on the remaining strings after any of these operatinos, the initial inputs matched
        
        '''
        def rec(s,p):
            if not p:
                return not s
            #find first match
            first_match = bool(s) and p[0] in {s[0],'.'}
            #star case, 
            #note pattern[0] can be a char or dot
            '''
            example
            say we have s = 'aaadddd' and p = 'aaad*'
            we matched up to aad, then the *implies match any number of times d
            '''
            # the kleene * matches any number of times the char predicing char
            if len(p) >= 2 and p[1] == '*':
                return rec(s,p[2:]) or first_match and rec(s[1:],p)
            else:
                #regular case with no Kleeneww star
                return first_match and rec(s[1:],p[1:])
        
        return rec(s,p)

#############################
# 647. Palindromic Substrings (Revisited)
# 23MAY22
#############################
class Solution:
    def countSubstrings(self, s: str) -> int:
        '''
        if we let dp(i,j) return whether s[i:j] is a palindrome or not
        dp(i,j) is True when dp(i+1,j-1) is true
        then we just check all i,j substrings and increment a counter
        
        
        '''
        memo = {}
        #this is an O(N) operation
        #but by the time wew compute another i,j we already have previosuly computed its values
        def isPal(i,j):
            if i > j:
                return True
            if s[i] != s[j]:
                return False
            if (i,j) in memo:
                return memo[(i,j)]
            ans = isPal(i+1,j-1)
            memo[(i,j)] = ans
            return ans
        
        ans = 0
        N = len(s)
        for i in range(N):
            for j in range(i,N):
                ans += isPal(i,j)
                
        return ans
        
#fucck this shit
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]

        ans = 0
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j]:
                    if i+1 >= j:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1]
                        
                if dp[i][j]:
                    ans += 1
        return ans

####################################
# 474. Ones and Zeroes (Revisited)
# 23MAY22
####################################
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        '''
        this is the classical 0/1 knap sack problem
        let dp(i,m,n) be the maximum size of the subset if we include subsets from strs[i:] having m 0's and n 1's
        
        now for the current call at i,m,n we can either choose to include this subet or no include
        if we do include we need to decrement the count, otherwise we don't and keep going on
        
        dp(i,m,n) = {
            counts_at_strs[i]
            if counts_at_strs[i] has more than the current m and n
                dp(i+1,m,n)
            else:
                max(1+dp(i+1,m-count zeros,n-count ones),dp(i+1,m,n))
        }
        '''
        memo = {}
        def calc(strs,idx,zeros,ones):
            #base case, end of array no options left
            if idx == len(strs):
                return 0
            if zeros == 0 and ones == 0:
                return 0
            if (idx,zeros,ones) in memo and memo[(idx,zeros,ones)] != 0:
                return memo[(idx,zeros,ones)]
            #current one strs
            counts = Counter(strs[idx])
            #keep recursing if we have enought
            taken = -1
            if zeros - counts["0"] >= 0  and  ones - counts["1"] >= 0:
                taken = calc(strs,idx+1,zeros - counts['0'],ones - counts['1']) + 1
            #take vs not take
            not_taken = calc(strs,idx+1,zeros,ones)
            ans = max(taken,not_taken)
            memo[(idx,zeros,ones)] =  ans
            return ans
        
        return calc(strs,0,m,n)

class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        '''
        lets try turning this into bottom up DP, using 3d array
        '''
        N = len(strs)
        
        dp = [[[0]*(n+1) for _ in range(m+1)] for _ in range(N+1)]
        
        #bottom case for recurrcne started at ending of trs
        for i in range(N,-1,-1):
            for zeros in range(m+1):
                for ones in range(n+1):
                    #bottom cases
                    if i == N:
                        dp[i][zeros][ones] = 0
                    else:
                        count = Counter(strs[i])
                        taken = -1
                        if zeros - count['0'] >= 0 and ones - count['1'] >= 0:
                            taken = dp[i+1][zeros - count['0']][ones- count['1']] + 1
                        not_taken = dp[i+1][zeros][ones]
                        ans = max(taken,not_taken)
                        dp[i][zeros][ones] = ans
        
        return dp[0][m][n]

#################################
# 32. Longest Valid Parentheses (REVSITED)
# 24MAY22
####################################
#top down still fails some cases
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        '''
        if we define dp(i) as the number of valid parantheses using s[:i]
        then we call dp(i) for i in range(len(s))
            base case
                i < 0:
                return 0
            if s[i] == ')' and s[i-1] == '(', we have ()
                dp[i] = dp[i-2] + 2
            for cases like ))
                we need to check if s[i - dp(i-1)-1] == '('
                    then dp(i) = dp(i-1) + s[i-dp(i-1) -2] + 2
        '''
        memo = {}
        
        def dp(i):
            if i <= 0:
                return 0
            if i in memo:
                return memo[i]
            #ending
            ans = 0
            if s[i] == ')':
                #case 1, ()
                if s[i-1] == '(':
                    if i >= 2:
                        ans = dp(i-2) + 2
                    else:
                        ans = 2
                #case 2, ))
                elif (i - dp(i-1) > 0) and (s[i-dp(i-1)-1] == '('):
                    if (i - dp(i-1)) >= 2:
                        ans = dp(i-1) + dp(i-dp(i-1)-2) + 2
                    else:
                        ans = 2
                
            memo[i] = ans
            return ans

            
        ans = 0
        N = len(s)
        for i in range(N):
            ans = max(ans,dp(i))
        
        return ans