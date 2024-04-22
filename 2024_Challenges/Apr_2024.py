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
    
####################################################
# 2844. Minimum Operations to Make a Special Number
# 11APR24
####################################################
class Solution:
    def minimumOperations(self, num: str) -> int:
        '''
        a number is diviible by 25 if the last digits are are divisibly by 25
        '''
        N = len(num)
        ans = N if '0' not in num else N-1
        for i in range(N):
            for j in range(i+1,N):
                last_two = int(num[i])*10 + int(num[j])
                if last_two in [0,25,50,75]:
                    ans = min(ans,N - i - 2)
        
        return ans

class Solution:
    def minimumOperations(self, num: str) -> int:
        '''
        we can also do prefix dp
        index i, and mod 25
        '''
        memo = {}
        N = len(num)
        
        def dp(i,mod):
            if i >= N:
                if mod == 0:
                    return 0
                return float('inf')
            
            if (i,mod) in memo:
                return memo[(i,mod)]
            
            next_mod = (mod*10 + int(num[i])) % 25
            take = dp(i+1,next_mod)
            remove = 1 + dp(i+1,mod)
            ans = min(take,remove)
            memo[(i,mod)] = ans
            return ans
        
        
        return dp(0,0)

###################################################################
# 255. Verify Preorder Sequence in Binary Search Tree (REVISITED)
# 13APR24
###################################################################
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        '''
        stack problem monostack
        pre order is node, left, right, so explore left subtree first
        maintain min and update it, if at any point in time we are smaller than the min, 
        it means this should have been explore first in order to have a valid bst, return false
        otherwise we are still good, and can contiue going right
        recall that regualr dfs we need to backtrack to the originl caller (the parent)
            in stack oringinal called would have been the node at the top of stack
        '''
        stack = []
        curr_min = float('inf')
        for num in preorder:
            #need to maintain largest min, not more than the next num
            while stack and stack[-1] < num:
                curr_min = stack.pop()
            #invalid bst, invalid preorder!
            if num <= curr_min:
                return False
            stack.append(num)
        
        return True

#constant space
class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        '''
        we can reduce to constant space if we simiulate stack with array
        append -> i++, set at [i]
        pop -> i--
        '''
        i = 0
        curr_min = float('-inf')
        for num in preorder:
            #need to maintain largest min, not more than the next num
            while i > 0 and preorder[i-1] < num:
                curr_min = preorder[i-1]
                i -= 1
            #invalid bst, invalid preorder!
            if num <= curr_min:
                return False
            preorder[i] = num
            i += 1
        
        return True


########################################
# 407. Trapping Rain Water II
# 12APR24
########################################
#nice try.....
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        '''
        apply the original algorithm across each row???
        try along both rows and cols and take minimum
        '''
        rows = len(heightMap)
        cols = len(heightMap[0])
        row_volume = 0
        
        #try along rows
        for r in heightMap:
            row_volume += self.trap(r)
        
        col_volume = 0
        heightMap_T = [list(row) for row in zip(*heightMap)]
        
        for col in heightMap_T:
            col_volume += self.trap(col)
        
        return min(row_volume,col_volume)
    def trap(self, height: List[int]) -> int:
        '''
        left max and right max for each height
        we also need to store the indics to find the distances
        '''
        ans = 0
        current = 0
        st = []
        
        while current < len(height):
            while st and height[current] > height[st[-1]]:
                top = st.pop()
                if not st:
                    break
                distance = current - st[-1] - 1
                #how much water is trapped in this gap?
                bounded_height = min(height[current], height[st[-1]]) - height[top]
                ans += distance * bounded_height
            
            st.append(current)
            current += 1
        
        return ans

class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        '''
        scan the grid for the lowest (i,j) cell
        if its surrounded by all 4 insides it can hold water, in fact i can fill this with water with
            volumne = (second smallest height - first smallest height)
            fill smallest heights first
            group cells by height, then order the heights increasinly
        '''
        heights = defaultdict(list)
        rows = len(heightMap)
        cols = len(heightMap[0])
        
        for i in range(rows):
            for j in range(cols):
                h = heightMap[i][j]
                heights[h].append((i,j))
        
        #sort on increasiny hieghts
        height_vals = [k for k,_ in heights.items()]
        height_vals.sort()
        N = len(height_vals)
        ans = 0
        for i in range(1,N):
            prev = height_vals[i-1]
            curr = height_vals[i]
            increase = curr - prev
            for i,j in heights[prev]:
                #cannnot be on the edge
                if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
                    continue
                #i need to move these cells over to the next level
                ans += increase
                heights[curr].append((i,j))
        
        return ans
    
#true solution is using heap

########################################
# 623. Add One Row to Tree (REVISTED)
# 17APR24
########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: TreeNode, v: int, d: int, left: bool = True) -> TreeNode:
        
        if d == 1:
            new_node = TreeNode(v)
            if left:
                new_node.left = root
            else:
                new_node.right = root
            return new_node
            #return TreeNode(v, root if left else None, root if not left else None)
        elif root:
            root.left = self.addOneRow(root.left, v, d - 1, True)
            root.right = self.addOneRow(root.right, v, d - 1, False)
        return root
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        #edge case
        if depth == 1:
            new_node = TreeNode(val)
            new_node.left = root
            return new_node
        
        def rec(node,curr_depth,insertion_depth,val):
            if not node:
                return
            if curr_depth == insertion_depth - 1:#  just before
                temp = node.left
                node.left = TreeNode(val)
                node.left.left = temp
                temp = node.right
                node.right =  TreeNode(val)
                node.right.right = temp
            else:
                rec(node.left,curr_depth+1,insertion_depth,val)
                rec(node.right,curr_depth+1,insertion_depth,val)
        rec(root,1,depth,val)
        return root

############################################
# 988. Smallest String Starting From Leaf 
# 17APR24
############################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        '''
        dont need self variable since its string, 
        get smallest, numnodes in tree is 8500
        python cheese is great!
        '''
        ans = ['z'*8501]
        
        def dfs(node,path,ans):
            path += chr(ord('a') + node.val)
            #check leaf
            if not node.left and not node.right:
                ans[0] = min(ans[0],path[::-1])
                return
            if node.left:
                dfs(node.left,path,ans)
            if node.right:
                dfs(node.right,path,ans)
        
        dfs(root,"",ans)
        return ans[0]

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def smallestFromLeaf(self, root: Optional[TreeNode]) -> str:
        '''
        no global
        '''        
        def dfs(node,path):
            if not node:
                return 'z'*8501
            path += chr(ord('a') + node.val)
            #check leaf
            if not node.left and not node.right:
                return path[::-1]
            left = dfs(node.left,path)
            right = dfs(node.right,path)
            return min(left,right)
        
        return dfs(root,"")
    
################################################
# 1560. Most Visited Sector in a Circular Track
# 19MAR24
###############################################
class Solution:
    def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
        '''
        n is the number of sectors, i can simulate and just do mod n
        sectors are numbered 1 to n
        '''
        mapp = {}
        for i in range(1,n):
            mapp[i] = i+1
        mapp[n] = 1
        counts = [0]*(n+1)
        is_first = True
        for i in range(len(rounds)-1):
            if is_first:
                start = rounds[i]
                end = rounds[i+1]
                is_first = False
            else:
                start = rounds[i]
                start = mapp[start]
                end = rounds[i+1]
                
            while start != end:
                counts[start] += 1
                start = mapp[start]
            counts[start] += 1
        
        #now find the max
        max_ = max(counts)
        ans = []
        for i in range(1,n+1):
            if counts[i] == max_:
                ans.append(i)
        
        return ans

class Solution:
    def mostVisited(self, n: int, rounds: List[int]) -> List[int]:
        '''
        Explanation:
        if n = 4, rounds = [1,3,1,2]
        than [1,2,3,4,1,2]
        output is [1,2]

        if n = 3 rounds = [3,1,2,3,1]
        than [3,1,2,3,1]
        output is [1,3]

        if n = 4 rounds = [1,4,2,3]
        than [1,2,3,4,1,2,3]
        output is [1,2,3]

        which means all steps moved in the middle will happen again and again and again, so they are useless.
        The only important things are start point and end point.
        '''
        first = rounds[0]
        last = rounds[-1]
        
        if last >= first:
            return [i for i in range(first,last+1)]
        return [i for i in range(1,n+1) if i <= last or i >= first]
    
##############################################
# 407. Trapping Rain Water II
# 19MAR24
###############################################
class Solution(object):
    def trapRainWater(self, heightMap):   
        m, n = len(heightMap), len(heightMap[0])
        heap = []
        visited = [[0]*n for _ in range(m)]

        # Push all the block on the border into heap
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0 or i == m-1 or j == n-1:
                    heapq.heappush(heap, (heightMap[i][j], i, j))
                    visited[i][j] = 1
        
        result = 0
        while heap:
            #get smalest from heap
            height, i, j = heapq.heappop(heap)    
            for x, y in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                #neighbores and in bounds check
                if 0 <= x < m and 0 <= y < n and not visited[x][y]:
                    #we can store water with differeunt, up to though
                    result += max(0, height-heightMap[x][y])
                    heapq.heappush(heap, (max(heightMap[x][y], height), x, y))
                    #mark
                    visited[x][y] = 1
        return result
    
class Solution(object):
    def trapRainWater(self, heightMap):
        '''
        heap but starting with border cells
        '''
        rows,cols = len(heightMap),len(heightMap[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        seen = [[False]*cols for _ in range(cols)]
        
        heap = []
        for i in range(rows):
            for j in range(cols):
                if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
                    entry = (heightMap[i][j], i,j)
                    heapq.heappush(heap,entry)
        
        water = 0
        while heap:
            curr_height, curr_row,curr_col = heapq.heappop(heap)
            seen[curr_row][curr_col] = True
            for dx,dy in dirrs:
                neigh_row = curr_row + dx
                neigh_col = curr_col + dy
                if 0 <= neigh_row < rows and 0 <= neigh_col < cols and not seen[neigh_row][neigh_col]:
                    if heightMap[neigh_row][neigh_col] <= curr_height:
                        water += curr_height - heightMap[neigh_row][neigh_col] 
                    entry = (heightMap[neigh_row][neigh_col], neigh_row,neigh_col)
                    heapq.heappush(heap,entry)
        
        return water
            
########################################
# 1992. Find All Groups of Farmland
# 20APR24
########################################
class Solution:
    def findFarmland(self, land: List[List[int]]) -> List[List[int]]:
        '''
        0 is forested land, 1 is farmland
        there are groups of farmland, not two groups are conected
        and groups of farmland are not four directinoally adajacent
        for each group of ones, return [topleft coords, bottomright coords]
        groups of farmland will always be in rectangular shape
        dfs on each group and find the coners
            upper left will be min of allxs and ys
            bottom right will be max of allxs and ys
        '''
        rows = len(land)
        cols = len(land[0])
        dirrs = [(1,0), (-1,0), (0,1), (0,-1)]
        seen = [[False]*cols for _ in range(rows)]
        
        ans = []
        for i in range(rows):
            for j in range(cols):
                if land[i][j] == 1 and not seen[i][j]:
                    upper_left = [i,j]
                    bottom_right = [i,j]
                    self.dfs(land,[i,j],upper_left,bottom_right,seen,dirrs,rows,cols)
                    entry = upper_left + bottom_right
                    ans.append(entry)
        return ans
        
    
    def dfs(self, land, curr_point, upper_left,bottom_right,seen,dirrs,rows,cols):
        r,c = curr_point
        #mark
        seen[r][c] = True
        #minimze
        upper_left[0] = min(upper_left[0],r)
        upper_left[1] = min(upper_left[1],c)
        #maximize
        bottom_right[0] = max(bottom_right[0],r)
        bottom_right[1] = max(bottom_right[1],c)
        for dx,dy in dirrs:
            neigh_row = r + dx
            neigh_col = c + dy
            if 0 <= neigh_row < rows and 0 <= neigh_col < cols:
                if not seen[neigh_row][neigh_col] and land[neigh_row][neigh_col] == 1:
                    self.dfs(land, [neigh_row,neigh_col], upper_left, bottom_right,seen,dirrs,rows,cols)
        

##########################################
# 752. Open the Lock (REVISTED)
# 22APR24
############################################
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        '''
        need neighbor function to generate net moves
        then use bfs dist dist array
        '''
        deadends = set(deadends)
        #edge cases
        if target in deadends: 
            return -1
        if '0000' in deadends: 
            return -1
        dists =  {}
        dists['0000'] = 0
        q = deque(['0000'])
        
        while q:
            curr = q.popleft()
            for neigh in self.get_neighbors(curr):
                if neigh not in deadends:
                    next_dist = 1 + dists[curr]
                    if next_dist < dists.get(neigh,float('inf')):
                        dists[neigh] = next_dist
                        q.append(neigh)
        
        ans = dists.get(target,float('inf'))
        if ans != float('inf'):
            return ans
        return -1
    
    def get_neighbors(self,code):
        N = len(code)
        for i in range(N):
            left = code[:i]
            right = code[i+1:]
            #can only turn one at a time!            
            for j in [1,-1]:
                temp = str((int(code[i]) + j) % 10)
                neigh = left + temp + right
                if neigh != code:
                    yield neigh
                
################################################
# 2368. Reachable Nodes With Restrictions
# 22APR24
#################################################
class Solution:
    def reachableNodes(self, n: int, edges: List[List[int]], restricted: List[int]) -> int:
        '''
        bfs with dists array
        if node is unreachable starting from 0 it doen'st count
        '''
        
        #make graph
        adj_list = defaultdict(list)
        for u,v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
    
        restricted = set(restricted)
        dists = [float('inf')]*n
        
        #start from ther 0th node
        dists[0] = 0
        q = deque([0])
        
        while q:
            curr = q.popleft()
            for neigh in adj_list[curr]:
                if neigh not in restricted:
                    next_dist = 1 + dists[curr]
                    if dists[neigh] > next_dist:
                        dists[neigh] = next_dist
                        q.append(neigh)
                        
        ans = 0
        for d in dists:
            if d != float('inf'):
                ans += 1
        
        return ans
        
        