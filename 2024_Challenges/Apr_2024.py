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
        
#without keeping dists array
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
        seen = [False]*n
        ans = 0
        q = deque([0])
        
        while q:
            curr = q.popleft()
            seen[curr] = True
            ans += 1
            for neigh in adj_list[curr]:
                if neigh in restricted:
                    continue
                if not seen[neigh]:
                    q.append(neigh)
                        
        return ans
    
#dfs
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
        seen = [False]*n
        ans = [0]
        
        def dfs(curr,seen,ans):
            seen[curr] = True
            ans[0] += 1
            for neigh in adj_list[curr]:
                if neigh in restricted:
                    continue
                if not seen[neigh]:
                    dfs(neigh,seen,ans)
        dfs(0,seen,ans)              
        return ans[0]

################################################
# 310. Minimum Height Trees (REVISITED)
# 23APR24
################################################
#first brute force
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        first of all do brute force and dfs on each node and finds its depth
        
        '''
        indegree = [0]*n
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            indegree[u] += 1
            indegree[v] += 1
        
        depths = []
        for i in range(n):
            depths.append(self.dfs(i,None,graph))
        
        min_depth = min(depths)
        ans = []
        for i in range(n):
            if depths[i] == min_depth:
                ans.append(i)
        
        return ans
        
    
    def dfs(self, curr,parent,graph):
        ans = 0
        for neigh in graph[curr]:
            if neigh != parent:
                ans = max(ans,1 + self.dfs(neigh,curr,graph))
        
        return ans
        
#top sort, but special way
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        if a graph has n nodes, then it can have at most n-1 mht trees?
        im thkning top sort, the nodes that are left must be the the roots of mht
        start off with nodes that have smallest indegree
        the thing is how do i know when to stop?
        
        essentially we are looking for the centroids of a graph
        for a tree-like graph, number of centroids <= 2
        even nodes, there could be two centroids
        odd nodes, there would be only one
        if we have more than two centroids, the graph must contain a cycle, which is not the case
        if there were three centroids, really it would justt be one
        problem boils down to finding the centroids of a tree
        trim from the leaves until we have two remaining nodes
        https://leetcode.com/problems/minimum-height-trees/discuss/827284/c%2B%2B99-TC-with-explanation-using-bfs-top-sort-%3A)
        MHT must be the midpoints of the longest leaf to leaf path in tree (could have solved it this way by finding the longest path)
        to find the longest path, first find the fathest leaf from any node, then find the fathest leafe from the found node above
        then these two nodes we found are the end points of the longest path
        '''
        if n == 0:
            return []
        
        if n == 1:
            return [0]
        
        indegree = [0]*n
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            indegree[u] += 1
            indegree[v] += 1
        
        q = deque([])
        for i in range(n):
            if indegree[i] == 1:
                q.append(i)
        
        ans = []
        while q:
            ans = []
            N = len(q)
            for _ in range(N):
                curr = q.popleft()
                ans.append(curr)
                for neigh in graph[curr]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 1:
                        q.append(neigh)
        return ans
    
#two dfs, find diameter
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        another way would be to find the furthest node from any node, once we find that node, find the futherst node from that
        and recreate path, the middle/s of the path are centroids
        '''
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        #first dfs, find furthest node from 0
        dist,parent = [-1]*n, [None]*n
        dist[0] = 0
        self.dfs(0,graph,dist,parent)
        furthest_from_zero = dist.index(max(dist))

        #second dfs, furthest from furtherst
        dist,parent = [-1]*n, [None]*n
        dist[furthest_from_zero] = 0
        self.dfs(furthest_from_zero,graph,dist,parent)
        
        furthest_from_furthest = dist.index(max(dist))
        path = []
        while furthest_from_furthest != None:
            path.append(furthest_from_furthest)
            furthest_from_furthest = parent[furthest_from_furthest]
            
        #get middle nodes
        N = len(path)
        temp = [path[N//2],path[(N-1)//2]] #careful with the middle, need lower, but N-1/(2) and now N//2 - 1
        return set(temp)

    def dfs(self,curr,graph,dist,parent):
        for neigh in graph[curr]:
            if dist[neigh] == -1:
                dist[neigh] = dist[curr] + 1
                parent[neigh] = curr
                self.dfs(neigh,graph,dist,parent)

#########################################
# 1656. Design an Ordered Stream
# 24APR24
##########################################
#stupid ass problem
class OrderedStream:

    def __init__(self, n: int):
        '''
        maintin pointer
        '''
        self.arr = [None]*(n+1)
        self.ptr = 1

    def insert(self, idKey: int, value: str) -> List[str]:
        self.arr[idKey] = value
        if idKey > self.ptr:
            return []
        if idKey == self.ptr:
            to_return = []
            while self.ptr < len(self.arr) and self.arr[self.ptr] != None:
                to_return.append(self.arr[self.ptr])
                self.ptr += 1
            
            return to_return


# Your OrderedStream object will be instantiated and called as such:
# obj = OrderedStream(n)
# param_1 = obj.insert(idKey,value)
        
####################################################
# 2370. Longest Ideal Subsequence 
# 25APR24
####################################################
#close one
class Solution:
    def longestIdealString(self, s: str, k: int) -> int:
        '''
        dp, extend or don't extend and save max, thankfully we only need to return the length and not the actual string
        dp(i) gives longest ideal string ending at i
        need ending index and last char at that index as the states
        rather ending at i, and last char before ending at i
        '''
        N = len(s)
        memo = {}
        
        def dp(i,s,k,memo):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            extend = 1
            if i < N - 1:
                curr_char,next_char = ord(s[i]),ord(s[i+1])
                if abs(curr_char - next_char) <= k:
                    extend = 1 + dp(i+1,s,k,memo)
            
            no_extend = dp(i+1,s,k,memo)
            ans = max(extend,no_extend)
            memo[i] = ans
            return ans
        
        ans = 0
        for i in range(N):
            ans = max(ans, dp(i,s,k,memo))
        
        return ans
            
#TLE
class Solution:
    def longestIdealString(self, s: str, k: int) -> int:
        '''
        dp, extend or don't extend and save max, thankfully we only need to return the length and not the actual string
        dp(i) gives longest ideal string ending at i
        need ending index and last char at that index as the states
        rather ending at i, and last char before ending at i
        '''
        N = len(s)
        memo = {}
        
        def dp(i,last_char,s,k,memo): #use '#' as dummy char
            if i >= N:
                return 0
            if (i,last_char) in memo:
                return memo[(i,last_char)]
            extend = 0
            curr_char_num,last_char_num = ord(s[i]), ord(last_char)
            if abs(curr_char_num - last_char_num) <= k or last_char == '#': #no last char, this case we mut extend
                extend = 1 + dp(i+1,s[i],s,k,memo)
            no_extend = dp(i+1,last_char,s,k,memo)
            ans = max(extend,no_extend)
            memo[(i,last_char)] = ans
            return ans
        
        
        return dp(0,'#',s,k,memo)

#bottom up
#this TLE's too
class Solution:
    def longestIdealString(self, s: str, k: int) -> int:
        '''
        dp, extend or don't extend and save max, thankfully we only need to return the length and not the actual string
        dp(i) gives longest ideal string ending at i
        need ending index and last char at that index as the states
        rather ending at i, and last char before ending at i
        '''
        N = len(s)
        dp = [[0]*(123) for _ in range(N+1)]
        
        for i in range(N-1,-1,-1):
            for last_char in range(123):
                extend = 0
                curr_char_num,last_char_num = ord(s[i]) - ord('a'), last_char
                if i == len(dp) - 1 or abs(curr_char_num - last_char_num) <= k or chr(last_char) == '#': #no last char, this case we mut extend
                    extend = 1 + dp[i+1][curr_char_num]
                no_extend = dp[i+1][last_char]
                dp[i][last_char] = max(extend,no_extend)
        
        return dp[0][ord('#')]

class Solution:
    def longestIdealString(self, s: str, k: int) -> int:
        N = len(s)

        # Initialize all dp values to -1 to indicate non-visited states
        dp = [[-1] * 26 for _ in range(N)]

        def dfs(i: int, c: int, dp: list, s: str, k: int) -> int:
            # Memoized value
            if dp[i][c] != -1:
                return dp[i][c]

            # State is not visited yet
            dp[i][c] = 0
            match = c == (ord(s[i]) - ord('a'))
            if match:
                dp[i][c] = 1

            # Non base case handling
            if i > 0:
                dp[i][c] = dfs(i - 1, c, dp, s, k)
                if match:
                    for p in range(26):
                        if abs(c - p) <= k:
                            dp[i][c] = max(dp[i][c], dfs(i - 1, p, dp, s, k) + 1)
            return dp[i][c]

        # Find the maximum dp[N-1][c] and return the result
        res = 0
        for c in range(26):
            res = max(res, dfs(N - 1, c, dp, s, k))
        return res

class Solution:
    def longestIdealString(self, s: str, k: int) -> int:
        '''
        bottom up
        '''
        N = len(s)
        dp = [0]*26
        
        for i in range(N):
            curr_char = ord(s[i]) - ord('a')
            local_best = 0
            for prev in range(26):
                if abs(prev - curr_char) <= k:
                    local_best = max(local_best, dp[prev])
            
            dp[curr_char] = max(dp[curr_char], local_best + 1)
        
        return max(dp)
    
#note instead trying all 26, we can bound the search space
class Solution:
   def longestIdealString(self, s: str, k: int) -> int:
       substr_map = collections.defaultdict(lambda : 0)
       ans = 0
       for c in s:
           max_length = 0
           for i in range(max(ord(c)-k, ord('a')), min(ord(c)+k+1, ord('z')+1)):
               max_length = max(substr_map[chr(i)] + 1,max_length)
           substr_map[c] = max_length
           ans = max(max_length, ans)
       return ans
   
######################################
# 1289. Minimum Falling Path Sum II
# 26APR24
#######################################
   #TLE, but its right
class Solution:
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        '''
        if we are at cell (i,j), we cannot go to (i+1,j) because it shares the same column
        we need to choose any of the rest
        need to optimize to pass
        '''
        rows = len(grid)
        cols = len(grid[0])
        memo = {}
        
        def dp(i,j,grid,memo,rows,cols):
            #if we reached the last row
            if i == rows - 1:
                return grid[i][j]
            #out of bounds
            if i < 0 or i > rows or j < 0 or j > cols:
                return float('inf')
            if (i,j) in memo:
                return memo[(i,j)]
            ans = float('inf')
            for k in range(cols):
                if k != j:
                    ans = min(ans, grid[i][j] + dp(i+1,k,grid,memo,rows,cols))
            memo[(i,j)] = ans
            return ans
        
        ans = float('inf')
        for j in range(cols):
            ans = min(ans, dp(0,j,grid,memo,rows,cols))
        
        return ans

#need to use cache
class Solution:
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        '''
        if we are at cell (i,j), we cannot go to (i+1,j) because it shares the same column
        we need to choose any of the rest
        need to optimize to pass
        '''
        rows = len(grid)
        cols = len(grid[0])

        @cache
        def dp(i,j):
            #if we reached the last row
            if i == rows - 1:
                return grid[i][j]
            #out of bounds
            if i < 0 or i > rows or j < 0 or j > cols:
                return float('inf')
            ans = float('inf')
            for k in range(cols):
                if k != j:
                    ans = min(ans, grid[i][j] + dp(i+1,k))
            return ans
        
        ans = float('inf')
        for j in range(cols):
            ans = min(ans, dp(0,j))
        
        return ans
    
#bottom up
class Solution:
    def minFallingPathSum(self, grid: List[List[int]]) -> int:
        '''
        bottom up
        
        '''
        N = len(grid)
        dp = [[float('inf')]*N for _ in range(N)]
        
        #fill in last roow
        for col in range(N):
            dp[N-1][col] = grid[N-1][col]
            
        for row in range(N-2,-1,-1):
            for col in range(N):
                ans = float('inf')
                for k in range(N):
                    if k != col:
                        ans = min(ans, dp[row+1][k] + grid[row][col])
                
                dp[row][col] = ans
        

        return min(dp[0])

#############################################
# 514. Freedom Trail
# 28APR24
##############################################
#ex dp with weighted graph
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        '''
        we can treat this like a graph problem and find the shortest path
        note: for each matcher char placed at  top this counts as 1 step
        so its min dist + len(key)
        find min dist
        make adj list by char, since we can rotate CW, or CCW, we need minimum
        step depends on what char we are at, and we want the min dist to the next char
        '''
        N = len(ring)
        graph = [[float('inf')]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                forward_dist = abs(i-j)
                rev_dist = N - forward_dist
                graph[i][j] = min(graph[i][j],forward_dist,rev_dist)
        

        #now just do dp
        memo = {}
        
        #states are position we are at now in ring and position where we are at in key
        def dp(ring_idx,key_idx):
            if (ring_idx,key_idx) in memo:
                return memo[(ring_idx,key_idx)]
            if key_idx == len(key):
                return 0
            ans = float('inf')
            for next_index,dist in enumerate(graph[ring_idx]):
                if ring[next_index] == key[key_idx]:
                    next_dist = dist + 1 + dp(next_index,key_idx+1)
                    ans = min(ans, next_dist)
            
            memo[(ring_idx,key_idx)] = ans
            return ans
        
        return dp(0,0)
    
#bottom up
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        '''
        bottom up
        '''
        N = len(ring)
        graph = [[float('inf')]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                forward_dist = abs(i-j)
                rev_dist = N - forward_dist
                graph[i][j] = min(graph[i][j],forward_dist,rev_dist)
        

        #now just do dp
        dp = [[float('inf')]*(len(key) + 1) for _ in range(len(ring)+1)]
        
        #base case fill
        for ring_idx in range(len(ring)+1):
            for key_idx in range(len(key)+1):
                if key_idx == len(key):
                    dp[ring_idx][key_idx] = 0

        
        for key_idx in range(len(key)-1,-1, -1):
            for ring_idx in range(len(ring)):
                ans = float('inf')
                for next_index,dist in enumerate(graph[ring_idx]):
                    if ring[next_index] == key[key_idx]:
                        next_dist = dist + 1 + dp[next_index][key_idx+1]
                        ans = min(ans, next_dist)
            
                #minimize at each steap
                dp[ring_idx][key_idx] = ans
        

        return dp[0][0]
        
#djikstras
class Solution:
    def findRotateSteps(self, ring: str, key: str) -> int:
        '''
        we can use djikstras to find the shortest path
        shortest path is number of turns needed
        but we need to do for each press
        so its len(shortest_path) + len(key)
        '''
        N = len(ring)
        graph = [[float('inf')]*N for _ in range(N)]
        
        for i in range(N):
            for j in range(N):
                forward_dist = abs(i-j)
                rev_dist = N - forward_dist
                graph[i][j] = min(graph[i][j],forward_dist,rev_dist)
        
        #first node is (0,0), (ring_idx,key_idx)
        dists = defaultdict(lambda: float('inf'))
        dists[(0,0)] = 0
        
        pq = [(0,0,0)] #entry is (min_dist,ring_idx,key_idx)
        visited = set()
        min_path = 0
        
        while pq:
            min_dist, ring_idx,key_idx = heapq.heappop(pq)
            if dists[(ring_idx,key_idx)] < min_dist:
                continue
            if (ring_idx,key_idx) in visited:
                continue
            #if we have finished the work
            if key_idx == len(key):
                min_path = min_dist
                break
            visited.add((ring_idx,key_idx))
            for next_index,dist in enumerate(graph[ring_idx]):
                next_dist = min_dist + dist + 1
                #optimize
                if next_dist < dists[(next_index,key_idx+1)]:
                    dists[(next_index,key_idx+1)] = next_dist
                    entry = (next_dist, next_index,key_idx+1)
                    heapq.heappush(pq,entry)
        return min_path + len(key)

###########################################
# 834. Sum of Distances in Tree (REVISTED)
# 28APR24
###########################################
#https://leetcode.com/problems/sum-of-distances-in-tree/discuss/1443467/Python-2-dfs-solution-explained
class Solution:
    def sumOfDistancesInTree(self, N: int, edges: List[List[int]]) -> List[int]:
        '''
        1. dfs starting from node 0 and store two things
            a. the number of nodes in that subtree, must be at least 1
            b. how far it is from the root, i.e 0
        2. dfs2(node,parent,w), let w be the answer for the current node
            how can we find neigh as if we already now ans for node?
                it must be w + N - 2*counts[neigh] because we moved from node to neigh 
                for weights[neigh] the number of nodes distances increased by 1 and for N - weights[neigh] (i.e the rest) it decreased by 1
                
        The answer for a node is the same as the answer for it's parent (w) except it is one unit distance closer 
        to all the members of its subtree (- weights[node]) and it is one unit farther away from every node 
        in the graph that is not in its subtree (+ N - weights[node]).

        All together the answer for a node is answer_for_parent - weights[node] + N - weights[node] which is the same as w + N - 2*weights[neib].
        
        intuition:
        The insight that the "sum of the depths" for any node we dfs from is the answer for that node is beautiful.
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        def dfs1(node,parent,depth,counts,depths):
            count = 1
            for neigh in graph[node]:
                if neigh != parent:
                    count += dfs1(neigh,node,depth+1,counts,depths)
            counts[node] = count
            depths[node] = depth
            return count
        
        def dfs2(node,parent,w,counts,ans):
            ans[node] = w
            for neigh in graph[node]:
                if neigh != parent:
                    dfs2(neigh,node, w + N - 2*counts[neigh],counts,ans)
        
        counts = [0]*N
        depths = [0]*N
        ans = [0]*N
        dfs1(0,None,0,counts,depths)
        dfs2(0,None,sum(depths),counts,ans) #inital ans for the first root would just be some of the depths
        return ans

########################################
# 305. Number of Islands II (REVISITED)
# 28APR24
########################################
#pretty much got it!
#be happy on this one!
class DSU:
    def __init__(self, rows,cols):
        self.parent = {}
        self.size = {}
        for i in range(rows):
            for j in range(cols):
                self.parent[(i,j)] = (i,j)
                self.size[(i,j)] = 1
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
            
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        if x_par == y_par:
            return False
        
        if self.size[x_par] > self.size[y_par]:
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
            self.parent[y_par] = x_par
        else:
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
            self.parent[x_par] = y_par
        
        return True #means we did do a union operation
    
        
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        '''
        union find on ith posiition
        when making the current (row,col) a 1, check its neighbors (up,down,left,right)
        all these should be
        '''
        dsu = DSU(m,n)
        grid = [[0]*n  for _ in range(m)]
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        islands = 0
        ans = [0]*len(positions)
        for i,(r,c) in enumerate(positions):
            grid[r][c] = 1
            connected = 0
            for dx,dy in dirrs:
                neigh_row = r + dx
                neigh_col = c + dy
                if 0 <= neigh_row < m and 0 <= neigh_col < n and grid[neigh_row][neigh_col] == 1:
                    if dsu.find((r,c)) == dsu.find((neigh_row,neigh_col)):
                        connected += 1
                        dsu.union((r,c),(neigh_row,neigh_col))
            islands += 1 - connected
            ans[i] = islands
        return ans
                
class UnionFind:
    def __init__(self, n):
        self.root = list(range(n))
        self.rank = [0] * n

    def find(self, node):
        if self.root[node] == node:
            return node
        else:
            self.root[node] = self.find(self.root[node])
            return self.root[node]

    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)

        if self.rank[root1] > self.rank[root2]:
            self.root[root2] = self.root[root1]
        elif self.rank[root1] < self.rank[root2]:
            self.root[root1] = self.root[root2]
        else:
            self.root[root2] = self.root[root1]
            self.rank[root1] += 1

    def is_connected(self, node1, node2):
        return self.find(node1) == self.find(node2)

class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        dsu = UnionFind(m * n)
        grid = [[0] * n for _ in range(m)]
        islands = 0
        ans = [0] * len(positions)

        for i, (new_r, new_c) in enumerate(positions):
            # skip repeated islands by doing nothing if it's already an island
            if grid[new_r][new_c]:
                ans[i] = islands
                continue

            connected = 0
            grid[new_r][new_c] = 1
            for old_r, old_c in ((new_r + 1, new_c), (new_r - 1, new_c), (new_r, new_c + 1), (new_r, new_c - 1)):
                if old_r in range(m) and old_c in range(n) and grid[old_r][old_c]:
                    new_land = n * new_r + new_c
                    old_land = n * old_r + old_c
                    if not dsu.is_connected(old_land, new_land):
                        connected += 1
                        dsu.union(old_land, new_land)

            islands += 1 - connected
            ans[i] = islands

        return ans
    
###################################################################
# 2997. Minimum Number of Operations to Make Array XOR Equal to K
# 29APR24
####################################################################
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        '''
        in one operation, chose any element of the array and flip a bit in its binary representation
        
        return minimum number of operations to make the bitwise XOR of all elements == k
        we can flip leading bits that are zero
        101 -> 110
        0101 -> 1101
        
        geeks for geeks problem
        to have it XOR to k, we need at least 1 in all the positions of k
        
        hints
        1. get bitwise XOR of all elements in nums and compare to k
        2. for each different bit, we need to flip exaclty one one bit of an eleemnt in nums
        
        the real insight is that XOR is mod2 addition,
        if there's an odd number of set bits, its XOR is 1
        if there's an even number of set bit, its XOR is 0
        
        we need the XOR of all nums to be equal to k
        if there is a mismatch as this position, we can flip 1 to make it equal at this position in k
        this works because flipping any bit, at any position in any num in nums, will flip this position in k
        '''
        xor_orig = nums[0]
        for num in nums[1:]:
            xor_orig = xor_orig ^ num
        
        ans = 0
        curr_pos = 0
        
        while xor_orig or k: #need to go through at least one of them
            mask = 1 << curr_pos
            if (k & mask) != (xor_orig & mask):
                ans += 1
            
            k = k >> 1
            xor_orig = xor_orig >> 1
        
        return ans

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        '''
        we can also just count the differences
        xor all the numbers and xor again with k,
        where there's a one, means there's a mismatch, so we need to flip a bit in any of the nums
        '''
        xor_orig = 0 #can also just start with 0
        for num in nums:
            xor_orig = xor_orig ^ num
            
        joined = xor_orig ^ k
        ans = 0
        while joined:
            ans += joined & 1
            joined = joined >> 1
        
        return ans               

#############################################
# 1915. Number of Wonderful Substrings
# 30APR24
#############################################
class Solution:
    def wonderfulSubstrings(self, word: str) -> int:
        '''
        a string is wonderful if there is a most one letter that appears an odd number of times
        note, we only have the first 10 letters
            also note you can't have zero chars appearing an odd number of times
            also, count each occurence of a substring supeaaretly? what if it asked only unique ones?
        string will be non empty
        use bit masks to identitfy string as wondeful
        if a string is wonderful, then there will only be a single 1 or or no ones in the mask
        
        'aba', (0,0,0,0,0,0,0,0,0,0)
        a :    (1,0,0,0,0,0,0,0,0,0)
        ab :   (1,1,0,0,0,0,0,0,0,0)
        aba:   (0,1,0,0,0,0,0,0,0,0) 
        note how (a and aba have the same mask)
        use hashmap to store counts of current masks and do the counting trick
        
        we can find parity of a count of letter doing mod 2, or XOR
        two types of wonerful strings 
            no letters appearing an odd number of times, and having one letterr appearing an odd number of times
        
        need to count number of substrings with mask of 0
        inutition:
            for any substring, we can represent it as the prefix between two prefixes os s
            for example, substring[2,5] is diff between substring[0,5] and substring[0,1], this mask should == 0 if
            substring[0,5] == substring[0,1], essentially we are subtracting prefixes
            XOR == subtration under mod 2
            we can combine prefix bit masks using XOR together to get the mask
        mask_of_whole_word = mask_word[0] ^ mask_word[1] .... ^ mas_word[end]
        if we xor a previous prefix with the current prefix, we are going to get a mask that corresponds to a wonderul string anway
        
        to count substrings with all even letters ending at some index r
            take the prefix ending at r with partiy mask m
            the differente of the two prefixes with the same bitmask == 0
        next we just need to coun up the caes where exactly one letter appears an odd number of times
            so we flip each bit in all 10 positions -> i.e its counterpart
            example, curr mask is 111, we can have 011, 101,110
        '''
        ans = 0
        counts =  Counter()
        #dont forget the zero mask
        counts[0] = 1
        curr_mask = 0
        for ch in word:
            char_idx = ord(ch) - ord('a')
            #flip it in mask
            curr_mask = curr_mask ^ (1 << char_idx) #mask will always correspond to a string that is wonderful
            ans += counts[curr_mask]
            counts[curr_mask] += 1
            
            for j in range(10):
                complement_mask = curr_mask ^ (1 << j)
                ans += counts[complement_mask]
        return ans

###############################################
# 2505. Bitwise OR of All Subsequence Sums
# 30APR24
################################################
class Solution:
    def subsequenceSumOr(self, nums: List[int]) -> int:
        '''
        all bits in array elements will be set
        and all bits in pref sum array will be set
        '''
        pref_sum = [nums[0]]
        for num in nums[1:]:
            pref_sum.append(pref_sum[-1] + num)
        
        ans = 0
        for num in nums:
            ans = ans | num
        for num in pref_sum:
            ans = ans | num
        
        return ans

class Solution:
    def subsequenceSumOr(self, nums: List[int]) -> int:
        '''
        recally bitwise or is just addition with carryover
        set each bit individually for each num in nums and check for carry
        becasue we want subsequences, we need to check if we cary over a bit using OR
        100000000000000
        18446744073709551616
        we need enough bit posistions to cover all subsequence sums, which can be no bigger than (10**9)*(10**9)
        2 | (2 + 1)
        think about actually adding the bits in binary format, the next number gets set!
        '''
        max_ = 64
        bit_set_counts = [0] * max_
        result = 0
        
        #this part would be treating each number as a single sum
        for num in nums:
            for i in range(31):
                mask = 1 << i
                if (num & mask):
                    bit_set_counts[i] += 1
                    
        for pos in range(max_ - 1):
            #if we have an occurence of this bit
            if bit_set_counts[pos] > 0:
                result = result | (1 << pos)
            #carry under mod two addition
            bit_set_counts[pos+1] += bit_set_counts[pos] // 2
        
        return result
        
class Solution:
    def subsequenceSumOr(self, nums: List[int]) -> int:
        '''
        we can just do bitwise or with the current num and its running sum
        for example [2,1,3]
        we have 2, we or with 1, and its sum, 3
        2,1, we wor with 3 and its sum 5
        this gives us the bitwise or of all subsequence sums
        
        so we have three numbers (a,b,c)
        what we want is a | b | c | (a + b) | (b + c) | (a + b + c) = a | b | c | (a + b) | (a + b + c)
        we dont need the other subsequences
        '''
        res = 0
        pref_sum = 0
        for num in nums:
            pref_sum += num
            res = res | num | pref_sum
        
        return res

