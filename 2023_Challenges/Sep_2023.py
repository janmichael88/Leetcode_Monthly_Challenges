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

#recursively is tricky
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

    #dervere first N elements revursively
    def reverseN(self, node,n):
        #one left
        if n == 1:
            self.succ = node.next
            return node
        last = self.reverseN(node.next,n-1)
        node.next.next = node
        node.next = self.succ
        return last
        

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
        
        