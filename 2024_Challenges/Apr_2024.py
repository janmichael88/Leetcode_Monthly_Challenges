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
    
#########################################
# 768. Max Chunks To Make Sorted II
# 05APR24
##########################################
#well, i can't belive that shieet worked
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        '''
        same problem, just that the numbers can be repeating
        [2,1,3,4,4], now sort
        [1,2,3,4,4]
        each k which some perm of arr[:k] == sorted(ar)[:k] is where we should cut
        so cut [2,1], [3], [4],[4]
        how can we effeciently check then?
        
        say we are int index i in arr
        if arr[:i+1] contains elements in sorted(arr[:i+1]) we split here
        holy shit worked! lmaooooo
        '''
        N = len(arr)
        sorted_arr = sorted(arr)
        chunks = 0
        
        for i in range(N):
            pref = arr[:i+1]
            if sorted(pref) == sorted_arr[:i+1]:
                chunks += 1
                
        
        return chunks
    
#################################################
# 678. Valid Parenthesis String (REVISTED)
# 07APR24
#################################################
#easssy
class Solution:
    def checkValidString(self, s: str) -> bool:
        '''
        keep track of balane use recursion
        can i cache states
        '''
        N = len(s)
        def dp(i,balance):
            if balance >= 1:
                return False
            if i == N:
                if balance == 0:
                    return True
                return False
            if s[i] == '(':
                return dp(i+1,balance-1)
            if s[i] == ')':
                return dp(i+1,balance+1)
            if s[i] == '*':
                return dp(i+1,balance) or dp(i+1,balance-1) or dp(i+1,balance+1)
            return False
        
        return dp(0,0)
    
#two stacks
class Solution:
    def checkValidString(self, s: str) -> bool:
        '''
        two stacks and greedy
        one stores indices of opened, the other stores indicies of asterisks
        '''
        opened = []
        stars = []
        
        for i,ch in enumerate(s):
            if ch == '(':
                opened.append(i)
            elif ch == '*':
                stars.append(i)
            #open
            else:
                #try closing
                if opened:
                    opened.pop()
                #resort to using asterisk
                elif stars:
                    stars.pop()
                #can't do
                else:
                    return False #nothing to pair this closing one
        
        
        #now check remaining opeened, we can close this opened if there is a star after it
        while opened and stars:
            if opened[-1] > stars[-1]:
                return False
            opened.pop()
            stars.pop()

        return not opened
    
#############################################################
# 1963. Minimum Number of Swaps to Make the String Balanced
# 08APR24
#############################################################
class Solution:
    def minSwaps(self, s: str) -> int:
        '''
        string can always be made balanced
        what string can we turn it into so that we swap a minimum number of times?
        swap and update at the same time
        check for largest disbalance in absolute value???

        idea is to just clear out the balanced parts
        put the unbalanced halves in the stack
        then we are left wil all the un balance halfs only
        ans is just size of stack + 1 // 2
        	    Pattern                          Result                      stack in the end         stack size
		 "]["                             1                                "["                  1
		"]][["                            1                                "[["                 2
		"]]][[["                          2                                "[[["                3
		"]]]][[[["                        2                                "[[[["               4
		"]]]]][[[[["                      3                                "[[[[["              5
		 
		 If you notice, result = (stack.size()+1)/2;
        '''
        stack = []
        for ch in s:
            if ch == '[':
                stack.append(ch)
            else:
                if stack:
                    stack.pop()
        
        return (len(stack) + 1) // 2


#################################################
# 1700. Number of Students Unable to Eat Lunch
# 08APR24
################################################
#ugly as hell but it works
class Solution:
    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        '''
        theres'a cycle
        if i were to simulate id need a terminating conditiion, can be empty sanwiches stack, because there are somes students 
        who won't be able to eat
        we're eventuall going to get to a state where all the students will be of one kind and that they can't eat the first sandwich
        '''
        N = len(students)
        eaten = 0
        students = deque(students)
        counts = Counter(students)
        i = 0
        
        while self.check(counts, sandwiches[i]):
            if students[0] == sandwiches[i]:
                eaten += 1
                i += 1
                if i == len(sandwiches):
                    i -= 1
                counts[students[0]] -= 1
                if counts[students[0]] == 0:
                    del counts[students[0]]
                students.popleft()
            else:
                students.append(students.popleft())
        
        return N - eaten
        
    def check(self, counts, sandwich):
        #checking to keep simultating
        if len(counts) == 2:
            return True
        elif len(counts) == 1:
            if counts[0] > 0 and 0 == sandwich:
                return True
            elif counts[1] > 0 and 1 == sandwich:
                return True
        
        return False
        
class Solution:
    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        '''
        another queue way, but how do we know when none of the students want to take top
        keep track of lastServed, and if lastServed goes all the way, we know none of them want the top
        otherwise take top and update lastServed to 0\
        
        the answer is just those students waiting in the queue
        '''
        N = len(students)
        student_q = deque([])
        sandwich_stack = []
        
        for i in range(N):
            student_q.append(students[i])
            sandwich_stack.append(sandwiches[N-i-1])
        
        #keep track of last studnet served
        last_served = 0
        while len(student_q) > 0 and last_served < len(student_q):
            #match with first student and top sandwich
            if student_q[0] == sandwich_stack[-1]:
                sandwich_stack.pop()
                student_q.popleft()
                last_served = 0
            else:
                student_q.append(student_q.popleft())
                last_served += 1
        
        return len(student_q)

#counting is tricky
class Solution:
    def countStudents(self, students: List[int], sandwiches: List[int]) -> int:
        '''
        just count and decrement
        when we want to server a sandwich but are out of students to give it too, we are done
        '''
        circle_count = 0
        square_count = 0
        
        for s in students:
            if s == 0:
                circle_count += 1
            else:
                square_count += 1
        
        for s in sandwiches:
            if s == 1 and square_count == 0:
                return circle_count
            if s == 0 and circle_count == 0:
                return square_count
            if s == 0:
                circle_count -= 1
            if s == 1:
                square_count -= 1
        
        
        return 0

#########################################
# 2073. Time Needed to Buy Tickets
# 09APR24
#########################################
class Solution:
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        '''
        use q but pair with indices
        '''
        time = 0
        q = deque([])
        for i,count in enumerate(tickets):
            q.append([i,count])
            
        kth_still_in = True
        while kth_still_in:
            curr,count = q.popleft()
            count -= 1
            if count > 0:
                q.append([curr,count])
            elif count == 0 and curr == k:
                kth_still_in = False
            time += 1
        
        return time
    
class Solution:
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        '''
        dont need to pair
        just use indices
        '''
        time = 0
        q = deque([])
        for i,count in enumerate(tickets):
            q.append(i)
            
        while q:
            time += 1
            curr = q.popleft()
            tickets[curr] -= 1
            if curr == k and tickets[curr] == 0:
                return time
            if tickets[curr] > 0:
                q.append(curr)
        
        return time
    
#no q
class Solution:
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        n = len(tickets)
        time = 0

        # If person k only needs one ticket, return the time to buy it
        if tickets[k] == 1:
            return k + 1

        # Continue buying tickets until person k buys all their tickets
        while tickets[k] > 0:
            for i in range(n):
                # Buy a ticket at index 'i' if available
                if tickets[i] != 0:
                    tickets[i] -= 1
                    time += 1

                # If person k bought all their tickets, return the time
                if tickets[k] == 0:
                    return time;

        return time
    
#one pass is tricky 
#O(N)
class Solution:
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        '''
        one pass
        intutition:
            determine if ith person is <= k or if i > k
                then this person can get up to at at most tickets[k]
                so min(tickets[k], tickets[i])
            
            determine if i > k, ith person will have less chances to get tickets, if tickets[k] < tickets[i]
                min(tickets[k] - 1, tickets[i])
                if they have less they can get all of them
                if they have more, then they can only get one less than tickets[k]
            
        '''
        ans = 0
        for i,num in enumerate(tickets):
            if i <= k:
                ans += min(tickets[i],tickets[k])
            else:
                ans += min(tickets[k] -1, tickets[i])
        
        return ans
    
########################################
# 950. Reveal Cards In Increasing Order
# 10APR24
#########################################
#christ im getting old..
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        '''
        notice how the ordering alternates, increasing on one index,
        decreasing on the other
        sort, weave and merge, nope its not that simple
        order the deck and reverse the operations!
        smallest card will alwasy be first
        need this order
        [2, 3, 5, 7, 11, 13, 17]
        
        [2, _, 3, _, 5, _, 7, ] what about thre order of [11,13,17]?
        if i have [11,13,17] left
        [11,17,13] nope
        alterntates
        we reveal smallest, but then next smallest gets sent to the end
        i can always place the the first n/2 numbers at even positision
        [17]
        [17,13]
        [11,17,13]
        [7,13,11,17]
        [5,17,7,13,11]
        [3,11,5,17,7,13]
        [2,13,3,11,5,17,7]
        '''
        N = len(deck)
        if N == 1:
            return deck
        deck.sort()
        q = deque([])
        while deck:
            if not q:
                q.appendleft(deck.pop())
            
            last_elem = q.pop()
            q.appendleft(last_elem)
            q.appendleft(deck.pop())
        
        return q
            
#we dont need to check for edge case if N == 1
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        '''
        no edge case
        '''
        deck.sort()
        q = deque([deck.pop()])
        while deck:
            
            last_elem = q.pop()
            q.appendleft(last_elem)
            q.appendleft(deck.pop())
        
        return q
    
#there' a rotate method for deque
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        '''
        no edge case
        '''
        deck.sort()
        q = deque([deck.pop()])
        while deck:
            q.rotate()
            q.appendleft(deck.pop())
        
        return list(q)
    
