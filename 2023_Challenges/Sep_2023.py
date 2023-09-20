#########################################
# 2707. Extra Characters in a String
# 02SEP23
#########################################
#YESSSS
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        '''
        break s into substrings in the mostt optimal way such that the number of leftover characters as minimum
        try dp
        if i'm at i, i can technially try all substrings from i to n+1
        try all j, from i+1 to n
        if i has gotten all the way to the end, it means there are i leftover characters
        
        what if a maximize the used characters, then its just N - max_used
        '''
        memo = {}
        dictionary = set(dictionary)
        N = len(s)
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            ans = dp(i+1)
            for j in range(i+1,N+1):
                if s[i:j] in dictionary:
                    ans = max(ans,(j-i) + dp(j))
            
            #ans = max(ans,dp(i+1))
            memo[i] = ans
            return ans
        
        
        return N - dp(0)

#bottom up
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        '''
        '''
        dictionary = set(dictionary)
        N = len(s)
        dp = [0]*(N+1)
        dp[N] = 0
        
        
        for i in range(N-1,-1,-1):
            ans = dp[i+1]
            for j in range(i+1,N+1):
                if s[i:j] in dictionary:
                    ans = max(ans,(j-i) + dp[j])
            
            dp[i] = ans

        
        return N - dp[0]

#for directly solving
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        '''
        for direclty solving, if we cant match a character we add 1
        otherwise take the minimum of the nexdt
        '''
        
        memo = {}
        dictionary = set(dictionary)
        N = len(s)
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            ans = dp(i+1) + 1
            for j in range(i+1,N+1):
                if s[i:j] in dictionary:
                    ans = min(ans,dp(j))
            

            memo[i] = ans
            return ans
        
        
        return dp(0)
    
#insteaf of using hahset build up a trie
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        root = self.buildTrie(dictionary)
        
        @cache
        def dp(start):
            if start == n:
                return 0
            # To count this character as a left over character 
            # move to index 'start + 1'
            ans = dp(start + 1) + 1
            node = root
            for end in range(start, n):
                if s[end] not in node.children:
                    break
                node = node.children[s[end]]
                if node.is_word:
                    ans = min(ans, dp(end + 1))
            return ans
        
        return dp(0)
    
    def buildTrie(self, dictionary):
        root = TrieNode()
        for word in dictionary:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
        return root

#############################################
# 725. Split Linked List in Parts (REVISITED)
# 06SEP23
#############################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        '''
        get size of N
        if N % k == 0, then each part will have N//k nodes
        otherwise the first N % k parts wil have ceil(N/k) nodes
        the remaning N - (N % k) should have k nodes
        '''
        ans = []
        N = 0
        curr = head
        while curr:
            N += 1
            curr = curr.next
        parts_plus_one = N % k
        parts_no_plus = k - (N % k)
        
        curr = head
        #the first parts_plus_one parts will have ceil(3/5) nodes
        need = math.ceil(N/k)
        for _ in range(parts_plus_one):
            count = 0
            runner = curr
            prev = None
            while count < need:
                prev = runner
                runner = runner.next
                count += 1
            prev.next = None
            ans.append(curr)
            curr = runner
        
        #the next parts
        need = N // k
        for _ in range(parts_no_plus):
            count = 0
            runner = curr
            prev = None
            while runner and count < need:
                prev = runner
                runner = runner.next
                count += 1
            if prev:
                prev.next = None
            else:
                prev = None
            ans.append(curr)
            curr = runner
        
        return ans[:k]
    
#########################################
# 2674. Split a Circular Linked List
# 06SEP23
#########################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def splitCircularLinkedList(self, list: Optional[ListNode]) -> List[Optional[ListNode]]:
        '''
        get length, 
        get upper half
        split at upper half
        '''
        N = 1
        curr = list
        runner = curr.next
        while runner != curr:
            runner = runner.next
            N += 1
        
        ans = []
        first_half = math.ceil(N / 2)
        prev = None
        runner = curr
        for _ in range(first_half):
            prev = runner
            runner = runner.next
            
        #print(curr.val,prev.val,runner.val)
        prev.next = curr
        #break the first part
        #return [curr,runner]
        prev = None
        next_half = runner
        for _ in range(N - first_half):
            prev = runner
            runner = runner.next
            
        prev.next = next_half
        return [curr,next_half]

############################################
# 92. Reverse Linked List II (REVISTED)
# 07SEP23
############################################
#bahhh close enough this time
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        '''
        define function to reverse, and return head of new linked list as well as the tail of the original
        '''
        
        def reverse(node,size):
            prev = None
            curr = node
            tail = curr
            while size > 0:
                next_ = curr.next
                curr.next = prev
                prev = curr
                curr = next_
                size -= 1
            
            return prev,curr,tail
        
        
        curr = head
        prev = None
        for _ in range(left-1):
            prev = curr
            curr = curr.next
        
        a,b,c = reverse(curr,right - left + 1)
        if prev:
            prev.next = a
        
        c.next = b
        return head

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        '''
        good review on reverseing linkedlist recursively
        '''
        self.succ = None
        if left == 1:
            return self.reverseN(head,right)
        head.next = self.reverseBetween(head.next,left-1,right-1)
        return head
        
    def reverse(self, node):
        if not node.next:
            return node
        last = self.reverse(node.next)
        node.next.next = node
        node.next = None
        return last

    #reverse first N elements revursively
    def reverseN(self, node,n):
        #one left
        if n == 1:
            self.succ = node.next
            return node
        last = self.reverseN(node.next,n-1)
        node.next.next = node
        node.next = self.succ
        return last
        
#without keeping succ
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        '''
        we dont need to store successort because head.next.next will always point ot is
        '''
        if head == None or head.next == None:
            return head
        if left == right:
            return head
        
        if left > 1:
            head.next = self.reverseBetween(head.next,left -1, right - 1)
            return head
        
        newHead = self.reverseBetween(head.next,1,right -1)
        tail = head.next.next
        head.next.next = head
        head.next = tail
        return newHead

############################################
# 1019. Next Greater Node In Linked List
# 07SEP23
#############################################
#almost!
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        '''
        if we had the array we could just work backwards and update the max
        its next greater, not greater to the right!
        need to use motonic stack in decreasing order
        '''
        curr = head
        ans = []
        stack = []
        
        while curr:
            #empty stack we append
            if not stack:
                stack.append(curr.val)
            #maintain decreassing
            elif stack and stack[-1] > curr.val:
                stack.append(curr.val)
            else:
                temp = []
                while stack and stack[-1] < curr.val:
                    num = stack.pop()
                    if num < curr.val:
                        temp.append(curr.val)
                temp.append(0)
                ans.extend(temp[::-1])
                stack.append(curr.val)
            
            curr = curr.next
        
        return ans[::-1]
            

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        '''
        push values into an array 
        then keep montonic decreaing stack of indices
        when the value we are trying to push is bigger than the top of the stack,
        then every index in the stack has next greater value == to the one we are trying to push
        dont forget this paradigm!
        motonic stack at least
        '''
        nums = []
        curr = head
        while curr:
            nums.append(curr.val)
            curr = curr.next
            
        N = len(nums)
        ans = [0]*N
        
        stack = []
        
        #push indicies
        for i,num in enumerate(nums):
            while stack and nums[stack[-1]] < num:
                #get the index
                index = stack.pop()
                #this values next greater is num
                ans[index] = num
            
            stack.append(i)
        
        return ans

#one pass
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        '''
        no need to push answers to an integer array, we can just update the stack and answer on the flow
        we need to keep track the index and number when using stack, 
        we do this by keeping track of the count
        '''
        ans = []
        stack = []
        count = 0
        
        curr = head
        while curr:
            ans.append(0) #same as init'ing the 0s array
            while stack and curr.val > stack[-1][1]:
                curr_idx, curr_val = stack.pop()
                ans[curr_idx] = curr.val
            
            stack.append([count,curr.val])
            count += 1
            curr = curr.next
        
        return ans 


############################################
# 1199. Minimum Time to Build Blocks
# 04SEP23
############################################
#closeeee one, i dont quite understand the splitting in parallel bullshit
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        '''
        blocks[i] is time needed to build block
        we can either build this block with worker
        we can also split worker, where worker goes up by 1
        after building block a worker goes home
        split time is cost ti split, if workers split at the same time
        
        return min time to build all blocks
        need to keep track of the ith block that we are on
        if i'm at blocks[i] i use one workers and blocks[i] time
        or i can split into two workers, and go up split time
        if i split, i get three workers, but i need to make sure i keep one to split again
        sort increasingly and use dp
        dp(i,j) is min time to to build blocks[:i] using j workers
        choice1 = blocks[i] + dp(i+1,j-1)
        try all splitting up to the maximum
        i.e, 1 becomes 2, 2 becomes 4, 4 becomes 8
        each doubling event costs split ttime
        for each double event, i can move up j-i, to cover those blocks
        and go up by that sum from the current i to j
        base case
            i got to end, return 0
            if j >= N, return sum blocks
        '''
        memo = {}
        blocks.sort(reverse=True)
        N = len(blocks)
        
        def dp(i,j):
            if i == N:
                return 0
            if j == 0:
                return float('inf')
            if j >= N:
                return max(blocks[i:]) if blocks[i:] else 0
            if (i,j) in memo:
                return memo[(i,j)]
            choice1 = max(blocks[i], dp(i+1,j-1))
            curr_workers = j
            curr_split = 1
            while curr_workers <= N:
                #try doubling
                new_workers = curr_workers*2
                #splitting happend in parrelel, when we split, they work in parallel!
                new_time = split*curr_split + dp(i+new_workers-1,j+new_workers) + max(blocks[i:i +(new_workers)-1])
                ans = min(choice1,new_time)
                curr_workers = new_workers
                curr_split += 1
            
            memo[(i,j)] = ans
            return ans
        
        
        return dp(0,1)
    
#notes to why this failed, its not simple push dp
#if a worker decides to build at this i, i cant just add this time to the next sub problem
#because it would mean i exclude this i from being build where i have between 1 <= k <= N workers for some k
#i can try taking the max in choice1, but it fails when i try all splits
#because im only checking if split from this i
#so its like i, then split, so cover i+1,i+2, or i+1,i+2,i+3,i+4, or rather (i,[i:i*2]) and we only move i one step at a time
#but some nodes can work or split, the above doesnt capture that

        
#still nothing, the explanaitno is wierd i think
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:

        memo = {}
        blocks.sort(reverse=True)
        N = len(blocks)
        

        def dp(i,j):
            if i == N:
                return 0
            if j == 0:
                return float('inf')
            if j >= N:
                return sum(blocks[i:])
            if (i,j) in memo:
                return memo[(i,j)]
            choice1 = blocks[i] + dp(i+1,j-1)
            new_workers = j*2
            #splitting happend in parrelel
            new_time = split + dp(i+new_workers-1,j+new_workers)
            ans = min(choice1,new_time)

            memo[(i,j)] = ans
            return ans
        
        
        return dp(0,1)
    

#top down
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        '''
        we need at least n workers to build all blocks, and we initially have on worker
        we can imagine the splitting as a tree, and we can split in any number of ways to produce an assortment of trees
        the depth would be the number of splits taken to get that worker
        so for any leaf node, the splitting time would be depth*split, and only leaf nodes can actually build a block
        so for any leaf node, the total time taken by the worker (designated as that leaf node) would be blocks[i] + depth[i]*split
        so the total time would be max(blocks[i] + depth[i]*split) for all in in [0,N-1], FOR a given tree
        which means we have to enumrate all trees, find the max(blocks[i] + depth[i]*split) for that tree and take the minimum
        i.e split until we get N leaf nodes
        intuition:
            the block which takes the max time ot build should be assigned to the leaf node with the smllest depth
            the leaf with the highed depth should be assigend to the block with min time
            this leads to idea of sorting
        
        the best tree structure for the input depends on three and the hints suggest we have to look for all possbile optiosns
        each worker can either work -> continue as leaf node
                            split -> become an itnernal node
                            
        assume we had dp(b,w) which give the minimum for building the first b blocks with w workers
        first sort decending
        after buildingn block b, we try b+1
        so we need to solve dp(0,1)
        base cases
            b == N, build all blocks, so no time
            w = 0, no workers, return larget value
            w >= N - b, we have enough workers, just reutrn blocks[b]
        number blocks remaing would be len(blocks) - b
        if we have N workers, meaning we can build them all, the answer is just the max of blocks, but since we sorted, its just blocks[b]
        
        transsition part:
            if use this bth work to build blocks b, we need to go on to dp(b+1,w-1)
            since we can do both we take the maximum time
            option1 = max(blocks[b], dp(b+1,w-1))
            
            we can split here
            option2 = split + dp(b, min(2*w, N-b)), 
            
            then we just take min of option2 or option1
        '''
        n = len(blocks)

        # Sort the blocks in descending order
        blocks.sort(reverse=True)   

        # dp[i][j] represents the minimum time taken to 
        # build blocks[i~n-1] block using j workers
        dp = [[-1] * (n + 1) for _ in range(n)]

        def solve(b, w):
            # Base cases
            if b == n:
                return 0
            if w == 0:
                return float('inf')
            if w >= n - b:
                return blocks[b]

            # If the sub-problem is already solved, return the result
            if dp[b][w] != -1:
                return dp[b][w]

            # Two Choices
            work_here = max(blocks[b], solve(b + 1, w - 1))
            split_here = split + solve(b, min(2 * w, n - b))

            # Store the result in the dp array
            dp[b][w] = min(work_here, split_here)
            return dp[b][w]

        # For block from index 0, with 1 worker
        return solve(0, 1)
    
#note using dictionary ges MLE
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        n = len(blocks)

        # Sort the blocks in descending order
        blocks.sort(reverse=True)   
        
        memo = {}
        def dp(b, w):
            # Base cases
            if b == n:
                return 0
            if w == 0:
                return float('inf')
            if w >= n - b:
                return blocks[b]

            # If the sub-problem is already solved, return the result
            if (b,w) in memo:
                return memo[(b,w)]

            # Two Choices
            work_here = max(blocks[b], dp(b + 1, w - 1))
            split_here = split + dp(b, min(2 * w, n - b))

            # Store the result in the dp array
            ans = min(work_here, split_here)
            memo[(b,w)] = ans
            return ans

        # For block from index 0, with 1 worker
        return dp(0, 1)
    
#bottom up
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        '''
        bottom up
        '''
        n = len(blocks)
        blocks.sort(reverse = True)
        
        dp = [[0]*(n+1) for _ in range(n+1)]
        
        for b in range(n+1):
            for w in range(n+1):
                if b == n:
                    dp[b][w] = 0
                if w == 0:
                    dp[b][w] = float('inf')
                    
        #starting from n - 1, one away from base case
        #for w start at n
        for b in range(n - 1, -1, -1):
            #we stop just before 1, if we wante dp(0,0) go one more step
            for w in range(n, 0, -1):
                if w >= n - b:
                    #need to handle this boundary condition while traversing
                    dp[b][w] = blocks[b]
                    continue
                workHere = max(blocks[b], dp[b + 1][w - 1])
                split_here = split + dp[b][min(2 * w, n - b)]
                
                # Store the result in the dp array
                dp[b][w] = min(workHere, split_here)
        

        return dp[0][1]
    
#optimal merge pattern
class Solution:
    def minBuildTime(self, blocks: List[int], split: int) -> int:
        '''
        concept is known as the optimal merge patern
        we aleady know that leaf nodes at the deppest depth should be assigned blocks with mnimal time
        and leaf nodes nearest the root should be assigned blocks that take maximum time
        say we have blocks [10,50,25] with split 7
        we need to split
        we find the the next greatest two blocks [10,25], then its just 7 + max(10,25)
        imagine accimpulate the time in the each root node, split left and right
        the we just take the next two minimums
             note we cant just sort, keep taking two at time and incrementing
        need to use heap to keep track of the minimums and take the next max
        '''
        heapq.heapify(blocks)
        
        while len(blocks) > 1:
            first_min = heapq.heappop(blocks)
            second_min = heapq.heappop(blocks)
            new_node = split + max(first_min,second_min)
            heapq.heappush(blocks,new_node)
        
        return blocks[0]
        
######################################
# 118. Pascal's Triangle (REVISTED)
# 08SEP23
######################################
#top down
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        '''
        dp(i,j) = dp(i-1,j) + dp(i-1,j+1)
        '''
        memo = {}
        
        def dp(i,j):
            if i < 0:
                return 0
            if (i,j) == (0,0):
                return 1
            if i == j:
                return 1
            if j == 0:
                return 1
            if (i,j) in memo:
                return memo[(i,j)]
            ans = dp(i-1,j-1) + dp(i-1,j)
            memo[(i,j)] = ans
            return ans
        
        
        ans = []
        for i in range(numRows):
            level = []
            for j in range(i+1):
                level.append(dp(i,j))
            
            ans.append(level)
        
        return ans
    
#bottom up
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        '''
        bottom up
        '''
        
        ans = [[1]]
        
        for i in range(numRows-1):
            prev_row = ans[-1]
            next_row = [1]
            for j in range(len(prev_row) - 1):
                next_row.append(prev_row[j] + prev_row[j+1])
            next_row.append(1)
            ans.append(next_row)
        
        return ans
    
#every time i solve this i alwasy solve it a different way!

########################################
# 1252. Cells with Odd Values in a Matrix
# 09SEP23
########################################
class Solution:
    def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
        '''
        can i just simulate each indicies operation?
        optimize?
        if i have row r that im adding 1 to, then all columns go up by 1
        '''
        mat = [[0]*n for _ in range(m)]
        for r,c in indices:
            #do the rth row
            for col in range(n):
                mat[r][col] += 1
            
            for row in range(m):
                mat[row][c] += 1
        
        ans = 0
        for i in range(m):
            for j in range(n):
                ans += (mat[i][j] % 2) != 0
        
        return ans
    
#speed up (O(M*N + l))
class Solution:
    def oddCells(self, m: int, n: int, indices: List[List[int]]) -> int:
        '''
        only count row index and col index that appear odd tims
        then onle cells (i,j) that are odd will appear odd number of times
        use XOR to flipp, basically just getting parity from this
        imagine slapping 1s along some row i, these have parity odd
        if we do that same i, the are no longer 1
        now what about some column j, well it depends on if the parity of i was odd or evven!
        if some j intersected with an i, where i was aready odd, then it switches back to even!
        
        so we only want (i,j) that are odd
        '''
        rows,cols = [0]*m,[0]*n
        
        for r,c in indices:
            rows[r] ^= True
            cols[c] ^= True
        
        ans = 0
        for i in range(m):
            for j in range(n):
                ans += rows[i] ^ cols[j]
        
        return ans
    
#############################################################
# 1359. Count All Valid Pickup and Delivery Options (REVISITED)
# 10SEP23
#############################################################
#other way around
class Solution:
    def countOrders(self, n: int) -> int:
        '''
        keep track of picked and delieveted
        '''
        mod = 10**9 + 7
        memo = {}
        def dp(picked, delivered):
            if picked == n and delivered == n:
                return 1
            if picked > n or delivered > n or picked > delivered:
                return 0
            if (picked,delivered) in memo:
                return memo[(picked,delivered)]
            
            ans = 0
            #get num ways for picking
            picking = (n-picked)*dp(picked+1,delivered)
            ans += picking
            ans %= mod
            
            #beause when picking i use up a spot
            delivering = (delivered - picked + 1)*dp(picked,delivered+1)
            ans += delivering
            ans %= mod
            memo[(picked,delivered)] = ans
            return ans
        
        
        a = (dp(0,0))
        return a
    
class Solution:
    def countOrders(self, n: int) -> int:
        '''
        there are going to be 2n spots
        things to note, a pickup cannot happen at the end and a dropoff cannot happend at the beginning
        try n = 2, so there are 4 spots
        [_,_,_,_]
        i can place P1, anywhere except the end
        so for the first pickup, i can place it at 3 positions
        if i cant to drop it off, i need to place it any of the available positions that come after where i had place P1
        so i place P1 at some index i, then i can place D1 at (4-i - 1) positions
        which means there are in 3*(4 - i + 1) for a a pair
        need to keep track of the number of picked
        
        for picks ups, we can place N picks in any 2*N - 1 spots (everythign except the last)
        for delivers we can only place them after we have picked up the ith 1
        '''
        memo = {}
        
        def dp(i):
            if i > n:
                return 1
            if i in memo:
                return memo[i]
            #number of wasy to place the ith pick is i*dp(i+1) to get the total number of ways
            # to place a deliver, we have 2*i - 1 spots
            # so its just i*(2*i - 1)
            ans = i*(2*i - 1)*dp(i+1)
            ans %= 10**9 + 7
            memo[i] = ans
            return ans
        
        return dp(1)
    
#############################################################
# 1282. Group the People Given the Group Size They Belong To
# 11SEP23
#############################################################
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        '''
        there can be multiple answers and we can return any of them
        map group size to ids, the greedinly make a grop
        '''
        group_size_to_id = defaultdict(list)
        for i,size in enumerate(groupSizes):
            group_size_to_id[size].append(i)
        
        
        ans = []
        #split each entry
        for size,ids in group_size_to_id.items():
            curr_size = len(ids)
            #split ids into the correct size
            for i in range(0,curr_size,size):
                ans.append(ids[i:i+size])
        
        return ans
    
#can also do one pass!
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        groups = defaultdict(list)
        res = []
        for pid, group_size in enumerate(groupSizes):
            groups[group_size].append(pid)
            if len(groups[group_size]) == group_size:
                res.append(groups.pop(group_size))
        return res
    
###########################################
# 358. Rearrange String k Distance Apart
# 12SEP23
###########################################
#close one! 56/64
class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        '''
        note there can multiple answers
        must be a greedy solution
        say i have count k for a letter
        each has to be at least k away from each other
        get counts of chars, and greddily try to place them
        so we have: aabbcc
        a : 2
        b : 2
        c : 2
        
        does it make sense to greedily put them k distance apart?
        a b c a b c
        '''
        if k == 0:
            return s
        counts = Counter(s)
        #sort them decreasinly
        sorted_counts = sorted([(v,k) for k,v in counts.items()],reverse = True)
        N = len(s)
        ans = [""]*N
        
        start = 0
        for count,ch in sorted_counts:
            while start < N and count > 0:
                ans[start] = ch
                count -= 1
                start += k
            #impossible
            if count > 0:
                return ""

            #find next empty spot
            for i in range(N):
                if ans[i] == "":
                    start = i
                    break
        
        return "".join(ans)
            
class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        '''
        idea is to use priority queue to retreive the most frequent elements
        and to also is a deque to control when we push back into the queue
        
        intuition:
            if we place a character at index x, then then next time we can place that character must be at x + K
            which means that all characters in between x and x + k are unique
            so we need to start with characters that have the highest frequency
            use max heap and decrement char counts, push into heap to alwasy find the max
            
        we place at the current index, but we can't simply put the the char we just pushed back into the heap, because we might violate the K rule
        for this when we add to our answer, we also push into a deque
        when deque becomes size k, we know the char at the front of the q can be put back in
            not it cannot simply the the size of deque, it must be the case that the index from where the char at the front is more than k away
        '''
        freq = [0] * 26

        # Store the frequency for each character.
        for char in s:
            freq[ord(char) - ord('a')] += 1

        free = []

        # Insert the characters with their frequencies in the max heap.
        for i in range(26):
            if freq[i]:
                heapq.heappush(free, (-freq[i], i))

        ans = ""

        # This queue stores the characters that cannot be used now.
        busy = []

        while len(ans) != len(s):
            index = len(ans)

            # Insert the character that could be used now into the free heap.
            if busy and (index - busy[0][0]) >= k:
                _, char = busy.pop(0)
                heapq.heappush(free, (-freq[char], char))

            # If the free heap is empty, it implies no character can be used at this index.
            if not free:
                return ""

            _, currChar = heapq.heappop(free)
            ans += chr(currChar + ord('a'))

            # Insert the used character into busy queue with the current index.
            freq[currChar] -= 1
            if freq[currChar] > 0:
                busy.append((index, currChar))

        return ''.join(ans)
    
#we can also do the variant where we check the size of the deque
class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        #another way, using len(q) variant to checck whether we can add back
        if k == 0:
            return ""
        
        max_heap = [(-freq,key) for key,freq in Counter(s).items()]
        heapq.heapify(max_heap)
        waiting = deque([])
        
        ans = ""
        
        while max_heap:
            curr_freq, curr_char = heapq.heappop(max_heap)
            ans += curr_char
            waiting.append((curr_freq+1,curr_char))
            
            if len(waiting) < k:
                continue
            
            #add back into heap from q
            curr_freq, curr_char = waiting.popleft()
            if curr_freq < 0:
                heapq.heappush(max_heap, (curr_freq,curr_char))
        
        #before returning check if we have anything in the waitilist
        #i.e any unused characters
        while waiting:
            curr_freq,curr_char = waiting.popleft()
            if curr_freq < 0:
                return ""
        
        return ans
    
##############################################
# 135. Candy (REVISTED)
# 13SEP23
##############################################
#nice try
#but we need to keep checking if distribution is valid
class Solution:
    def candy(self, ratings: List[int]) -> int:
        '''
        initally all children can have one candy
        [1,1,1]
        ratigins
        [1,0,2]
        
        start with first child
        1 < 0 for raitings, up 1 by 1
        [2,1,1]
        [2,1,2]
        
        '''
        N = len(ratings)
        candies = [1]*N
        
        for i in range(N):
            #first one
            if i == 0 and (i+1) < N:
                #bigger raiting and not enouhg candies
                if ratings[i] > ratings[i+1] and candies[i] <= candies[i+1]:
                    candies[i] = candies[i+1] + 1
            #ending 
            elif i == N-1 and (i-1) >= 0:
                if ratings[i] > ratings[i-1] and candies[i] <= candies[i-1]:
                    candies[i] = candies[i-1] + 1
            else:
                if ratings[i] > ratings[i-1] and ratings[i] > ratings[i+1] and candies[i-1] >= candies[i] and candies[i+1] >= candies[i]:
                    candies[i] = max(candies[i-1],candies[i+1]) + 1
    
        return sum(candies)
                
#top down
class Solution:
    def candy(self, ratings: List[int]) -> int:
        '''
        dp
        if ratings[i] > ratings[i+1]
        then candies[i] should be > candies[i+1]
        which meand candies[i] is also going to greater than any other candies to the right
        same could be said checking to the left
        so for each 
        '''
        memoleft = {}
        memoright = {}
        N = len(ratings)
        
        def dpLeft(i):
            if i == 0:
                return 1
            if i in memoleft:
                return memoleft[i]
            ans = 1
            if ratings[i] > ratings[i-1]:
                ans = dpLeft(i-1) + 1
            memoleft[i] = ans
            return ans
        
        
        def dpRight(i):
            if i == N-1:
                return 1
            if i in memoright:
                return memoright[i]
            ans = 1
            if ratings[i] > ratings[i+1]:
                ans = dpRight(i+1) + 1
            memoright[i] = ans
            return ans
        
        
        ans = 0
        for i in range(N):
            ans += max(dpLeft(i),dpRight(i))
        
        return ans
    
#bottom up
class Solution:
    def candy(self, ratings: List[int]) -> int:

        N = len(ratings)
        rightwards = [1]*N
        leftwards = [1]*N
        
        #left to right
        for i in range(1,N):
            if ratings[i] > ratings[i-1]:
                rightwards[i] = rightwards[i-1] + 1
        
        #right to left
        for i in range(N-2,-1,-1):
            if ratings[i] > ratings[i+1]:
                leftwards[i] = leftwards[i+1] + 1
                
        res = 0
        for i in range(N):
            res += max(leftwards[i],rightwards[i])
        
        return res
    
###########################################
# 2182. Construct String With Repeat Limit
# 14SEP23
###########################################
class Solution:
    def repeatLimitedString(self, s: str, repeatLimit: int) -> str:
        '''
        given string s and repeatLimit, 
        use chars of s such that no letter appears more than repeatLimit times in a row
        we do not need to use all chars from s, we cannot use more than the counts in s
        need to keep track of the largest lexographical character
        but you can't simply discard a char if we hit the repeat limits
        so if we hit the repeatlimit while in pq, we need to cache tht char somehwere, then put it back in the heap
        we dont need to wait, we can just alternate between the two largest chars
        '''
        counts = [0]*26
        for ch in s:
            counts[ord(ch) - ord('a')] += 1
        
        largest_chars = [-(i+1) for i in range(26) if counts[i] != 0] #one based indexing to allow for ordering
        heapq.heapify(largest_chars)
        
        ans = ""
        waiting = -1
        
        while largest_chars:
            first_largest = -heapq.heappop(largest_chars) - 1
            
            #use up first largest as much as we can
            repeat = 0
            while repeat < repeatLimit and counts[first_largest] > 0:
                ans += chr(ord('a') + first_largest)
                repeat += 1
                counts[first_largest] -= 1
            
            #no more of firt largest
            if counts[first_largest] == 0:
                continue
            #otherwise we can break this wit the second largest
            else:
                if not largest_chars:
                    return ans
                second_largest = -heapq.heappop(largest_chars) - 1
                ans += chr(ord('a') + second_largest)
                counts[second_largest] -= 1
                
                #add back in
                if counts[second_largest] > 0:
                    heapq.heappush(largest_chars, -(second_largest + 1))
                    
                #dont forget first largest
                heapq.heappush(largest_chars, -(first_largest +1))
        
        
        return ans
            
#########################################
# 332. Reconstruct Itinerary
# 14SEP23
#########################################
#we can't just greedily take the smallest lexographical neighbor
#nice try though!
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        '''
        this is not just top sort
        note, we also want the itinerary that has smallest lexographicla order
        tickets are directed
        if the there were no cycles, we just return the traversal, the issue is that there could be cycles 
        we need to visit every EDGE, exactly once
        concept is called Eulerian Path
        
        remove edges while we traverse
        when making the adjlist sort them, and when taking that edge just pop from the front
        cant just greedily take the next lexographical on
        '''
        graph = defaultdict(list)
        for u,v in tickets:
            graph[u].append(v)
        for node,neighs in graph.items():
            graph[node] = sorted(neighs)
        
        ans = []
        
        def dfs(node):
            ans.append(node)
            
            #has edges
            if graph[node]:
                dfs(graph[node].pop(0))
        
        dfs('JFK')
        return ans
        
#need to use backtracking after sorting
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        '''
        remove edges while we traverse
        when making the adjlist sort them, and when taking that edge just pop from the front
        cant just greedily take the next lexographical on
        we need to backtrack, 
        the ending conditions is when we have we touched all the edges
        
        for each node, we keep track of a visited set if we can touch all edges from this node
        if there are N edges, we need to take them all, which means there will be N+1 nodes in the path
        start by adding JFK to the curr path, then recursively try all neighbors until we get N+1 nodes in the path
        we return True once we get N+1 nodes, and for each call, check if we can get a valid path
        otherwise return False
        '''
        graph = defaultdict(list)
        for u,v in tickets:
            graph[u].append(v)
        
        visited_edges = {}
        for node,neighs in graph.items():
            graph[node] = sorted(neighs)
            visited_edges[node] = [False]*len(neighs)
        
        self.ans = []
        N = len(tickets)
        
        def dfs(node,path):
            if len(path) == N+1:
                self.ans = path[:]
                return True
            
            for i,neigh in enumerate(graph[node]):
                if not visited_edges[node][i]:
                    #mark as this being visited fomr this node to its neigh, we mark neigh as visit
                    visited_edges[node][i] = True
                    if dfs(neigh, path + [neigh]):
                        return True
                    visited_edges[node][i] = False
                
            
            
            return False
        
        dfs('JFK',['JFK'])
        return self.ans
                
#Hierholzer algo
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        '''
        i almost had it with the first attempt,
        concept is something called Eulerian Cycle, or Eulerian path
        in Eulerian path, we touch all edges only once and in Eulerian Cycel touch all edges but also come back to the vertext we started at
        Hierholzer algorithm
            idea is stepwise constructions of the Eulerian cycle by connecting disjunctive circles
            start with a random node and then follows an abritary unvisited edge to a neighbor, and repeat until one return to the starting node
            this gives the first cyle
            if the first cycle covers all the nodes, it must Eulerian
            othwise choose another node among the cycle's nodes with unvisited edges and construct another circle
            this is called a subtour
            
        
        for eulerian path, rather than stopping at the starting point we stop at the vertext where we do not have any unvisited edges
        algo:
        1. start from any vertex, we keep following the unused edges until we get stuck at certain vertex where there are no more unvisite edges
        2. backtrack to the nearest neighbord vertex in the current path that has unused edges and repeat
        
        the first vertex we got stuck at would be the endpoint
        if we follow stuck points backwards we get the Eulerian path
        
        since we know an Eulerian path exsits, we dont need to backtrack
        i.e we keep dfsing until we get stuck, when we get stuck we just add the node started the first traversal from
        notes
            there is an eulerian path so we change the question as given a list of flights, find an order to use each flight once and only once
            before visiting the last airport (call it V) we can say that we have already used all the rest of the flights
            rather, before adding the last airport in the path, we make sure visit all the other in that path
        '''
        graph = defaultdict(list)
        for u,v in tickets:
            graph[u].append(v)
        
        for node,neighs in graph.items():
            graph[node] = sorted(neighs,reverse = True) #do this decreasinly
        
        self.ans = []
        
        def dfs(node):
            neighs = graph[node]
            while neighs:
                dfs(neighs.pop())
            
            self.ans.append(node)
            
        dfs('JFK')
        return self.ans[::-1]
    
##############################################
# 2097. Valid Arrangement of Pairs
# 14SEP23
###############################################
class Solution:
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        '''
        turn into graph problem and return eulerian path
        a pair i can be connected another pair j if the end  for this i == start for this j
        consider pairs as edges, and each number as a node
        then run HeirHolzer
        start on any node with outdegree - indegree = 1, if all zero, start anywhere in cycle
        '''
        graph = defaultdict(list)
        degree = Counter()
        for u,v in pairs:
            graph[u].append(v)
            degree[u] += 1
            degree[v] -= 1
            
        for start in graph:
            if degree[start] == 1:
                break
        
        self.ans = []
        
        def dfs(node):
            neighs = graph[node]
            while neighs:
                neigh = neighs.pop()
                dfs(neigh)
                self.ans.append([node,neigh])
            
        dfs(start)
        return self.ans[::-1]

#not adding pairs on the fly, but adding desitination
#then recreate the answer
class Solution:
    def validArrangement(self, pairs: List[List[int]]) -> List[List[int]]:
        '''
        turn into graph problem and return eulerian path
        a pair i can be connected another pair j if the end  for this i == start for this j
        consider pairs as edges, and each number as a node
        then run HeirHolzer
        start on any node with outdegree - indegree = 1, if all zero, start anywhere in cycle
        '''
        graph = defaultdict(list)
        degree = Counter()
        for u,v in pairs:
            graph[u].append(v)
            degree[u] += 1
            degree[v] -= 1
            
        for start in graph:
            if degree[start] == 1:
                break

        self.ans = []
        
        def dfs(node):
            neighs = graph[node]
            while neighs:
                neigh = neighs.pop()
                dfs(neigh)
            self.ans.append(node)
            
        
        dfs(start)
        self.ans = self.ans[::-1]
        return [[self.ans[i], self.ans[i+1]] for i in range(len(self.ans)-1)]
    
##################################################
# 1584. Min Cost to Connect All Points (REVISTED)
# 15SEP23
#################################################
#kruskal
class DSU:
    def __init__(self,n):
        self.size = [1]*n
        self.parent = [i for i in range(n)]
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        if self.size[x_par] >= self.size[y_par]:
            self.parent[y_par] = x_par
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
        else:
            self.parent[x_par] = y_par
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
        

class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        '''
        minimum spanning tree problem, kruskal or primms
        make directed edge_list with weights, sort on weights
        '''
        edge_weights = []
        N = len(points)
        for i in range(N):
            for j in range(i+1,N):
                x1,y1 = points[i]
                x2,y2 = points[j]
                weight = abs(x1-x2) + abs(y1 - y2)
                entry = (weight, i,j)
                edge_weights.append(entry)
        
        edge_weights.sort(key = lambda x: x[0])
        
        dsu = DSU(N)
        min_cost = 0
        for weight,u,v in edge_weights:
            #check if we can join
            if dsu.find(u) != dsu.find(v):
                min_cost += weight
                dsu.union(u,v)
        
        return min_cost
    
#primms
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        '''
        we can also use primmms 
        start with any abritray node and find all its neighbors, find its weight and push into a heap
        we greediblly add edges to mst by taking the smaller ones first
        keep track of nodes in mst and edges used
        invariant is until we use up the edges
        if there are n nodes in an mst, then ther will be n-1 edges
        '''
        N = len(points)
        min_heap = [(0,0)] #(weight,node)
        in_mst = [False]*N
        
        edges_used = 0
        min_cost = 0
        
        while edges_used < N:
            curr_weight,curr_node = heapq.heappop(min_heap)
            
            if in_mst[curr_node]:
                continue
            
            #mark
            in_mst[curr_node] = True
            min_cost += curr_weight
            edges_used += 1
            
            #check neighs
            for neigh in range(N):
                if in_mst[neigh] == False:
                    x1,y1 = points[curr_node]
                    x2,y2 = points[neigh]
                    weight = abs(x1-x2) + abs(y1 - y2)
                    heapq.heappush(min_heap,(weight,neigh))
        
        return min_cost

#################################################
# 1631. Path With Minimum Effort (REVISITED)
# 16SEP23
#################################################
#binary search dfs
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        '''
        binary search to find the right k value
        to see if we can reach the end, use dfs
        smallest value can be 0 and largest value will be the the maximum
        if i can get to the path using this k, it means i can use any k greater than it
        '''
        rows = len(heights)
        cols = len(heights[0])
        dirrs = [(0,1),(0,-1),(1,0),(-1,0)]
        
        
        def canReach(i,j,seen,k):
            if (i,j) == (rows-1,cols-1):
                return True
            seen.add((i,j))
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #un seen and within k
                    if (neigh_x,neigh_y) not in seen and abs(heights[i][j] - heights[neigh_x][neigh_y]) <= k:
                        if canReach(neigh_x,neigh_y,seen,k) == True:
                            return True
            
            return False
        
        
        left = 0
        right = max([max(row) for row in heights])
        #seen = set()
        #temp = canReach(0,0,seen, 2)
        #print(temp)
        ans = -1
        while left < right:
            mid = left + (right - left) // 2
            seen = set()
            #which means we can use this k
            if canReach(0,0,seen,mid):
                ans = mid
                right = mid
            #we need to try a bigger k
            else:
                left = mid + 1
        
        return ans
            
#djikstras
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        '''
        we can turn this into a single source shortest path problem (SSP) with djikstras
        we can take the abs difference between two cells as the edge wiehgt
        recall with djikastras they are all float('inf') or float('-inf') away
        
        intsteaf or looking for the shortest path, we define the cells as the minimum effor required to reach that cell from ALL possible paths
        update all from the current cell
        use min heap to store edges weights that are closest
        
        i.e we are looking for the path from the current cell to the adjcent cell that takes less efforst than other paths that have reached the adjacent cell
        i.e going from (i,j) to (neigh_i, neigh_j) requires less effort (on a different path)
        we dont care about the path, we only care about the min effor
        
        the whole point is that we ARRVIVED at some cell from another cell with a smaller effort
        '''
        rows = len(heights)
        cols = len(heights[0])
        dirrs = [(0,1),(0,-1),(1,0),(-1,0)]
        
        distances = [[float('inf')]*cols for _ in range(rows)]
        #starting dista
        distances[0][0] = 0
        seen = set()
        
        min_heap = [(0,0,0)] #(weight, i,j)
        
        while min_heap:
            diff, i,j = heapq.heappop(min_heap)
            seen.add((i,j))
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #un seen and within k
                    if (neigh_x,neigh_y) not in seen:
                        curr_diff = abs(heights[i][j] - heights[neigh_x][neigh_y])
                        max_diff = max(curr_diff, distances[i][j])
                        #arrive at a smaller effort
                        if distances[neigh_x][neigh_y] > max_diff:
                            distances[neigh_x][neigh_y] = max_diff
                            #can do so, so push to heap
                            heapq.heappush(min_heap, (max_diff, neigh_x,neigh_y))

        return distances[rows-1][cols-1]
    
#union find
class DSU:
    def __init__(self,size):
        self.parent = [i for i in range(size)]
        self.rank = [1]*(size)
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        if self.rank[x_par] >= self.rank[y_par]:
            self.parent[y_par] = x_par
            self.rank[x_par] += self.rank[y_par]
            self.rank[y_par] = 0
        else:
            self.parent[x_par] = y_par
            self.rank[y_par] += self.rank[x_par]
            self.rank[x_par] = 0
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        '''
        we can also do union find, recall each cell (i,j) is connected to any one of is neighbords (neigh_i,neigh_j) throough an edge
        abs(cell(i,j) - cell(neigh_i,neigh_j))
        so we just need to check if (0,0) and (rows-1,cols-1) are connected
        we join components greedily start with the smallest edges first
        need to flatten 2d matrixn to 1d matrix: (currentRow * col + currentCol)
        
        '''
        rows = len(heights)
        cols = len(heights[0])
        dirrs = [(0,1),(0,-1),(1,0),(-1,0)]
        
        if rows == 1 and cols == 1:
            return 0
        
        edge_list = []
        for i in range(rows):
            for j in range(cols):
                for dx,dy in dirrs:
                    neigh_x = i + dx
                    neigh_y = j + dy
                    #bounds
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        curr_diff = abs(heights[i][j] - heights[neigh_x][neigh_y])
                        #push as (diff, to, from)
                        entry = (curr_diff, i*cols + j, neigh_x*cols + neigh_y) #will have repeated edges, but its ok becase we sort anyway
                        edge_list.append(entry)
        edge_list.sort(key = lambda x: x[0])
        dsu = DSU(rows*cols)
        for diff,u,v in edge_list:
            dsu.union(u,v)
            if dsu.find(0) == dsu.find(rows*cols-1):
                return diff
        
        return -1
    
####################################################
# 847. Shortest Path Visiting All Nodes (REVSITED)
# 17SEP23
####################################################
#TLE
#could also have used frozen states here too
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        '''
        this is variant traveling salesman problem
        find shortest path touching all nodes
            may reuse nodes and edges
        BFS, and push states to next levels
            each state needs to track current cells visited and path length
            when all cells have been visited return the path length
        '''
        q = deque([])
        N = len(graph)
        for i in range(N):
            entry = (0,i,set([i]))
            q.append(entry)
        
        while q:
            dist,node,seen = q.popleft()
            if len(seen) == N:
                return dist
            
            for neigh in graph[node]:
                copy_seen = copy.deepcopy(seen)
                copy_seen.add(neigh)
                #print(node,copy_seen)
                q.append((dist+1,neigh,copy_seen))
                
        return -1
    
#use bit mask instead of seen set
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        '''
        this is variant traveling salesman problem
        find shortest path touching all nodes
            may reuse nodes and edges
        BFS, and push states to next levels
            each state needs to track current cells visited and path length
            when all cells have been visited return the path length
        '''
        q = deque([])
        N = len(graph)
        for i in range(N):
            mask = 0
            #set this bit
            mask |= (1 << i)
            entry = (0,i,mask)
            q.append(entry)
            
        
        
        while q:
            dist,node,seen = q.popleft()
            if seen == 2**N - 1:
                return dist
            
            for neigh in graph[node]:
                next_mask = seen | (1 << neigh)
                #print(node,copy_seen)
                q.append((dist+1,neigh,next_mask))
                
        return -1
    
#careful not to revisit mask states
#for some reason deque gets TLE on this problems, even with caching,you need to use array lists insteaf
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        '''
        this is variant traveling salesman problem
        find shortest path touching all nodes
            may reuse nodes and edges
        BFS, and push states to next levels
            each state needs to track current cells visited and path length
            when all cells have been visited return the path length
        '''
        q = deque([])
        N = len(graph)
        for i in range(N):
            mask = 0
            #set this bit
            mask |= (1 << i)
            entry = (0,i,mask)
            q.append(entry)
        
        #state are combindaito of node and mask
        seen_states = set()
        while q:
            dist,node,seen = q.popleft()
            seen_states.add((node,seen))
            if seen == 2**N - 1:
                return dist
            
            for neigh in graph[node]:
                next_mask = seen | (1 << neigh)
                if next_mask == 2**N - 1:
                    return dist + 1
                if (neigh,next_mask) not in seen_states:
                    #print(node,copy_seen)
                    q.append((dist+1,neigh,next_mask))
                
        return -1
    
#top down dp review
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        '''
        so we have some group of k nodes, call it state, and we know this for state we are at a minimum path length
        we then want to add some node i
        there are two options:
            1. we have already visted node
            2. we have not visited this node
            so then are awnser would be Min((1 + option1), (1 + option2))
            
            if we have already visited, we dont need to update state, but simply add1
            if we haven't, identfy new mask and recurse
        '''
        memo = {}
        N = len(graph)
        
        def dp(node,mask):
            if mask == 2**N - 1:
                return 0
            if (node,mask) in memo:
                return memo[(node,mask)]
            
            ans = float('inf')
            memo[(node,mask)] = ans
            for neigh in graph[node]:
                #not seen
                if mask & (1 << neigh) == 0:
                    visited = 1 + dp(neigh,mask)
                    not_visited = 1 + dp(neigh, mask | (1 << neigh))
                    ans = min(ans,visited,not_visited)
            
            memo[(node,mask)] = ans
            return ans
        
        ans = float('inf')
        for i in range(N):
            ans = min(ans, dp(i, 1 << i))

        return ans
    
###############################################
# 2812. Find the Safest Path in a Grid
# 17SEP23
###############################################
#jesus too many fucking edge cases
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        '''
        cells at (i,j) contain 1 if theif
        0 if empty
        safeness factor of a paht is defined as the min minhat distance from any cell in the path to any theif in the grid
        return maximum safeness factor for all paths
        define path as (0,0) to (N-1,N-1)
        
        its manhattan distance! not step distance
        
        i can do the binary search workable solution thingy again
        for each (i,j) cell find the minimum safeness factor
        then try reaching start to end with some safeness factor k, if k works, then anything less than equal to k should work
        
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0), (-1,0), (0,1), (0,-1)]
        
        if (grid[0][0],grid[rows-1][cols-1]) == (1,1):
            return 0
        
        safeness = [[-1]*cols for _ in range(rows)]
        #find min dist for each (i,j) using bfs
        #multipoint BFS
        seen = set()
        q = deque([])
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    safeness[i][j] = 0
                    q.append((0,i,j))
                    
        while q:
            dist,i,j = q.popleft()
            for dx,dy in dirrs:
                neigh_x = dx + i
                neigh_y = dy + j
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and safeness[neigh_x][neigh_y] == -1:
                    safeness[neigh_x][neigh_y] = dist + 1
                    q.append((dist+1,neigh_x,neigh_y))
        
        #dfs function to see if we can reach (rows-1,cols-1) from (0,0) using this k safeness factor
        def dfs(i,j,k,seen):
            if (i,j) == (rows-1,cols-1):
                return True
            seen.add((i,j))
            for dx,dy in dirrs:
                neigh_x = dx + i
                neigh_y = dy + j
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and (neigh_x,neigh_y) not in seen:
                    if safeness[neigh_x][neigh_y] >= k:
                        if dfs(neigh_x,neigh_y,k,seen) == True:
                            return True
            
            return False
    

        left = 0
        right = max([max(row) for row in safeness])
        ans = -1
        while left < right:
            seen = set()
            mid = left + (right - left) // 2
            if dfs(0,0,mid,seen) and safeness[0][0] >= mid:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
                
        return ans if ans != -1 else 0

class Solution:
    
    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def manhattanDist(self, grid: List[List[int]], n: int) -> List[List[int]]:
        queue = collections.deque()
        visited = [[False for _ in range(n)] for _ in range(n)]

        distances = [[0 for _ in range(n)] for _ in range(n)]

        for r in range(n):
            for c in range(n):
                if grid[r][c]:
                    queue.append((r, c))
                    visited[r][c] = True
        
        dist = 0
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                distances[r][c] = dist

                for dr, dc in self.DIRS:
                    nr, nc = r + dr, c + dc

                    if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                        queue.append((nr, nc))
                        visited[nr][nc] = True

            dist += 1
        
        return distances

    def reachable(self, dist_grid: List[List[int]], min_dist: int, n: int) -> bool:
        visited = [[False for _ in range(n)] for _ in range(n)]

        def dfs(r: int, c: int) -> bool:
            if r == n - 1 and c == n - 1:
                return True
            
            visited[r][c] = True
            for dr, dc in self.DIRS:
                nr, nc = r + dr, c + dc

                if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                    if dist_grid[nr][nc] >= min_dist:
                        if dfs(nr, nc):
                            return True
            
            return False
        
        return dfs(0, 0) if dist_grid[0][0] >= min_dist else False
    

    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        
        n = len(grid)

        l, r = 0, 400
        l -= 1

        dist_grid = self.manhattanDist(grid, n)
        while l < r:
            mid = (l + r + 1) >> 1

            if self.reachable(dist_grid, mid, n):
                l = mid
            else:
                r = mid - 1
        
        return l if l != -1 else 0
    
class Solution:
    def maximumSafenessFactor(self, grid: List[List[int]]) -> int:
        '''
        we can also use djikstras to find the path with maximum safeness factor
        dist[i][j] is stores max safness factor getting to (i,j) on any path
        
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(1,0), (-1,0), (0,1), (0,-1)]
        
        if (grid[0][0],grid[rows-1][cols-1]) == (1,1):
            return 0
        
        safeness = [[-1]*cols for _ in range(rows)]
        #find min dist for each (i,j) using bfs
        #multipoint BFS
        seen = set()
        q = deque([])
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    safeness[i][j] = 0
                    q.append((0,i,j))
                    
        while q:
            dist,i,j = q.popleft()
            for dx,dy in dirrs:
                neigh_x = dx + i
                neigh_y = dy + j
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and safeness[neigh_x][neigh_y] == -1:
                    safeness[neigh_x][neigh_y] = dist + 1
                    q.append((dist+1,neigh_x,neigh_y))
        
        
        distances = [[0]*cols for _ in range(rows)]
        distances[0][0] = safeness[0][0]
        max_heap = [(-distances[0][0],0,0)]
        
        while max_heap:
            dist,i,j = heapq.heappop(max_heap)
            dist *= -1
            if (i,j) == (rows-1,cols-1):
                return dist
            for dx,dy in dirrs:
                neigh_x = dx + i
                neigh_y = dy + j
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    max_safeness = max(dist, distances[neigh_x][neigh_y])
                    if max_safeness > distances[neigh_x][neigh_y]:
                        distances[neigh_x][neigh_y] = max_safeness
                        heapq.heappush(max_heap,(-max_safeness,neigh_x,neigh_y))
        
        return distances[rows-1][cols-1]
    

################################################
# 2152. Minimum Number of Lines to Cover Points
# 15SEP23
#################################################
class Solution:
    def minimumLines(self, points: List[List[int]]) -> int:
        '''
        need to touch all points using the minimum number of lines
        inputs are small, we will only have 10 points
        if there are n points, than at most we can n lines, one line for each point 
        but even better if a line connects at leat two points, we only need n/2 lines + 1 is odd
        
        so i had a group of points call it S with indeix (i,j,k,l)
        and for this group we know the min lines to be a minimum
        then we want to introduct another point m
        if m is is on any intsersecting in this group, we do not need to add a new line
        otherwise, we do need to add a new line
        bass case is when we get all 1s in the mask
        
        need to preomcpute pairwise slopes for all (i,j)
        then i can say for this ith point at this slope it connects j
        '''
        memo = {}
        N = len(points)
        target = (1 << N) - 1
        
        if N == 1:
            return 1
        
        #precompute pairwise point slope
        graph = defaultdict(lambda: defaultdict(list))
        for i in range(N):
            for j in range(i+1,N):
                p1 = points[i]
                p2 = points[j]
                slope = self.getSlope(p1,p2)
                #note python dictionary can handle floating points
                graph[i][slope].append(j)
                graph[j][slope].append(i)
        
        
        return self.dp(0,memo,target,graph,N,points)
                
        
    def getSlope(self,p1,p2):
        x1,y1 = p1
        x2,y2 = p2

        if x1 == x2:
            return float('inf')
        return (y1 - y2) / (x1 - x2)

    def dp(self,mask,memo,target,graph,N,points):
        #we have taken all points
        if mask == target:
            return 0
        if mask in memo:
            return memo[mask]

        ans = float('inf')
        for i in range(N):
            #not taken
            if (mask & (1 << i)) == 0:
                p1 = points[i]
                #find another point
                for j in range(N):
                    p2 = points[j]
                    if j == i:
                        continue
                    #include both in new mask
                    next_mask = mask | (1 << i)
                    next_mask = next_mask | (1 << j)
                    next_slope = self.getSlope(p1,p2)
                    #check points from i that match this slope
                    for k in graph[i][next_slope]:
                        #these points don't add 1 to the count since they are on the same line
                        next_mask = next_mask | (1 << k)
                        ans = min(ans, 1 + self.dp(next_mask,memo,target,graph,N,points))
                        #note, can also minimize here on the fly, or at the end of looking through neighbors
        
        memo[mask] = ans
        return ans

import math
class Solution:
    def minimumLines(self, points: List[List[int]]) -> int:
        '''
        if we didn't want to store floats in the hash map, we need to normalaize by the GCD
        '''
        memo = {}
        N = len(points)
        target = (1 << N) - 1
        
        if N == 1:
            return 1
        
        #precompute pairwise point slope
        graph = defaultdict(lambda: defaultdict(list))
        for i in range(N):
            for j in range(i+1,N):
                p1 = points[i]
                p2 = points[j]
                slope = self.getSlope(p1,p2)
                #note python dictionary can handle floating points
                graph[i][slope].append(j)
                graph[j][slope].append(i)
        
        
        return self.dp(0,memo,target,graph,N,points)
                
        
    def getSlope(self,p1,p2):
        x1,y1 = p1
        x2,y2 = p2

        if x1 == x2:
            return float('inf')
        num = (y1 - y2) 
        denom = (x1 - x2)
        GCD = math.gcd(num,denom)
        entry = (num // GCD, denom // GCD)
        return entry

    def dp(self,mask,memo,target,graph,N,points):
        #we have taken all points
        if mask == target:
            return 0
        if mask in memo:
            return memo[mask]

        ans = float('inf')
        for i in range(N):
            #not taken
            if (mask & (1 << i)) == 0:
                p1 = points[i]
                #find another point
                for j in range(N):
                    p2 = points[j]
                    if j == i:
                        continue
                    #include both in new mask
                    next_mask = mask | (1 << i)
                    next_mask = next_mask | (1 << j)
                    next_slope = self.getSlope(p1,p2)
                    #check points from i that match this slope
                    for k in graph[i][next_slope]:
                        #these points don't add 1 to the count since they are on the same line
                        next_mask = next_mask | (1 << k)
                        ans = min(ans, 1 + self.dp(next_mask,memo,target,graph,N,points))
                        #note, can also minimize here on the fly, or at the end of looking through neighbors
        
        memo[mask] = ans
        return ans
                    
############################################
# 1136. Parallel Courses (REVISTED)
# 19SEP23
############################################
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        '''
        this is just longest path given a dag, or number of edges in longest path + 1
        if there is a cycle we can't do it
        another issue is there the components could be disconnected
        
        for cycle detection remember three states: visited, visiting, unvisited
        '''
        graph = defaultdict(list)
        for u,v in relations:
            graph[u].append(v)
        
        seen = {}
        for i in range(1,n+1):
            if self.hasCycle(i,seen,graph):
                return -1
        
        #no cycle, use dp to find longest path
        memo = {}
        ans = 0
        for i in range(1,n+1):
            if i not in memo:
                ans = max(ans, self.dp(i,memo,graph))

        return ans

        

    def hasCycle(self,node,seen,graph):
        if node in seen:
            return seen[node]
        else:
            seen[node] = -1
        for neigh in graph[node]:
            if self.hasCycle(neigh,seen,graph):
                return True
        
        
        seen[node] = False
        return False
        
        '''
        for i in range(1,n+1):
            seen = set()
            if hasCycle(i,seen):
                return -1
        '''
        #no cycle, use dp to find longest path
        memo = {}
        
    def dp(self,node,memo,graph):
        if len(graph[node]) == 0:
            return 1
        if node in memo:
            return memo[node]
        ans = 0
        for neigh in graph[node]:
            ans = max(ans,1 + self.dp(neigh,memo,graph))
        memo[node] = ans
        return ans

#kahns
class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        '''
        topsort using kahns
        clases with zero in degree can be taken first 
        visit neighbors, but only add back into queue when when indegree is zero
        instead of passing step count into queue, layer out and increment
        
        '''
        graph = defaultdict(list)
        indegree = Counter()
        for u,v in relations:
            graph[u].append(v)
            indegree[v] += 1
        
        #quee up zero indedree
        q = []
        for i in range(1,n+1):
            if indegree[i] == 0:
                q.append(i)
        
        q = deque(q)
        
        classes = 0
        semesters = 0
        while q:
            N = len(q)
            for _ in range(N):
                curr_class = q.popleft()
                classes += 1
                for neigh in graph[curr_class]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 0:
                        q.append(neigh)
            
            semesters += 1
        
        if classes != n:
            return -1
        return semesters
    
################################################
# 1494. Parallel Courses II
# 19SEP23
################################################
class Solution:
    def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
        '''
        from parallel coures the min number of semesters is just the longest path, given the constraints
        we can use dp with bitmasks, bitmask will store classes taken
        backtrack using indegree
        offset classes by 1
        need to try all possible ways to take k classes at each step, not  just any k
        then just take or dont take up to k times
        hard search problem
        '''
        graph = defaultdict(list)
        indegree = [0]*n
        
        for u,v in relations:
            graph[u-1].append(v-1)
            indegree[v-1] += 1
        
        final_mask = (1 << n) - 1
        start_mask = 0
        
        memo = {}
        
        def dp(mask,memo,indegree):
            if mask == final_mask:
                return 0
            if (mask,tuple(indegree)) in memo:
                return memo[(mask,tuple(indegree))]
            
            #get classes we can possible take
            possible_classes = []
            for i in range(n):
                if (mask & (1 << i)) == 0 and indegree[i] == 0:
                    possible_classes.append(i)
                    
            ans = float('inf')
            #generate k combinatinos of classes we can take
            for next_classes in combinations(possible_classes, min(k,len(possible_classes))):
                next_mask = mask
                next_indegree = indegree[:] #copy it
                #minimize againat all possible classes
                #need to minimze for all possible ways!
                for next_class in next_classes:
                    next_mask |= (1 << next_class)
                    for neigh in graph[next_class]:
                        next_indegree[neigh] -= 1
                    
                ans = min(ans, 1 + dp(next_mask,memo,next_indegree))
                #note i could indent here but that would TLE, because i call it that many more times
                #why does the answer have to here
            
            memo[(mask,tuple(indegree))] = ans
            return ans
        
        return dp(0,memo,indegree)
    
#we can also do bfs
class Solution:
    def minNumberOfSemesters(self, n: int, relations: List[List[int]], k: int) -> int:
        '''
        from parallel coures the min number of semesters is just the longest path, given the constraints
        we can use dp with bitmasks, bitmask will store classes taken
        backtrack using indegree
        offset classes by 1
        need to try all possible ways to take k classes at each step, not  just any k
        then just take or dont take up to k times
        hard search problem
        '''
        graph = defaultdict(list)
        indegree = [0]*n
        
        for u,v in relations:
            graph[u-1].append(v-1)
            indegree[v-1] += 1
        
        final_mask = (1 << n) - 1
        start_mask = 0
        
        q = deque([(0,start_mask,indegree)])
        seen = set()
        seen.add((start_mask,tuple(indegree)))
        
        while q:            
            curr_semesters, mask, indegree = q.popleft()
            
            if mask == final_mask:
                return curr_semesters
            
            #get classes we can possible take
            possible_classes = []
            for i in range(n):
                if (mask & (1 << i)) == 0 and indegree[i] == 0:
                    possible_classes.append(i)

            #generate k combinatinos of classes we can take
            for next_classes in combinations(possible_classes, min(k,len(possible_classes))):
                next_mask = mask
                next_indegree = indegree[:] #copy it
                #minimize againat all possible classes
                #need to minimze for all possible ways!
                for next_class in next_classes:
                    next_mask |= (1 << next_class)
                    for neigh in graph[next_class]:
                        next_indegree[neigh] -= 1
                if (curr_semesters + 1, next_mask, tuple(next_indegree)) not in seen:
                    entry = (curr_semesters + 1, next_mask, next_indegree)
                    seen.add((curr_semesters + 1, next_mask, tuple(next_indegree)))
                    q.append(entry)
