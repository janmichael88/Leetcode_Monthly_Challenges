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