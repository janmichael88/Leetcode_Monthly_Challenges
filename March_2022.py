############################
# 338. Counting Bits
# 01MAR2022
############################
class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        i can count the ones in each num by shifting
        and using num & 1
        '''
        def count_ones(num):
            ones = 0
            while num:
                ones += num & 1
                num = num >> 1
            
            return ones
        
        
        ans = []
        for i in range(n+1):
            ans.append(count_ones(i))
        
        return ans

class Solution:
    def countBits(self, n: int) -> List[int]:
        
        def pop_count(x: int) -> int:
            count = 0
            while x != 0:
                x &= x - 1 # zeroing out the least significant nonzero bit
                count += 1
            return count
            
        ans = [0] * (n + 1)
        for x in range(n + 1):
            ans[x] = pop_count(x)
    
        return ans       

class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can use dp to find the count of currnet set bits in number i from number i-1
        we first need the transisiton function
        say we have x = 605 = 1001011101
        it differs by a number we have previosuly calculated
        x = 93 = 1011101
        0001011101 -> 605
        1001011101 -> 93
        count(x+b) = count(x) + 1
        
        if we have the numbers
        0 -> 0
        1 -> 1
        2 -> 10
        3 -> 11
        4 -> 100
        5 -> 101
        6 -> 110
        7 -> 111
        
        we can find the count for 2 and 3 just by adding 1 from 0 and 1
        we can find the count for 4 through 7, by adding one to the counts from the counts in 0 to 3
        
        we can write the transitino functon as
        P(x+b) = P(x) + 1. for b in range(0,2**b)
        
        rather dp(x) = dp(x-b) + 1 for b in all the bits up to x
        '''
        dp = [0]*(n+1)
        x = 0
        b = 1
        
        #for all bit positions <= n
        while b <= n:
            #check intervals [b,2b) or [b,n] from [0,b)
            #check all allowable bit positions for this b, and increment by 1
            #i.e move through the positions < b, and use the transition to generate the counts
            while x < b and x + b <= n:
                print(x,b)
                dp[x+b] = dp[x] + 1
                x += 1
            
            #reset x
            #why? we need to start back to the first bit position
            x = 0
            b <<= 1
        
        return dp

class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can also use the least significant bit, right most bit
        if we look at the relation between a number x and x//2
        x = 605 = 1001011101
        x = 302 = 100101110
        
        we just popped off the last bit 
        P(x) = P(x/2) + (x mod 2)
        '''
        ans = [0] * (n + 1)
        for x in range(1, n + 1):
            # x // 2 is x >> 1 and x % 2 is x & 1
            #not x mod 2 and x & 1 are the same
            ans[x] = ans[x >> 1] + (x % 2) 
        return ans 
        
class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can just get the last set bit using Brian Kernighan's trick
        P(x) = P(x & (x-1)) + 1
        '''
        ans = [0]*(n+1)
        for x in range(1,n+1):
            ans[x] = ans[x & (x-1)] + 1
        
        return ans

#recursive
class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can just get the last set bit using Brian Kernighan's trick
        P(x) = P(x & (x-1)) + 1
        '''
        memo = {}
        
        def dp(n):
            if n == 0:
                return 0
            if n == 1:
                return 1
            #otherwise fetch
            if n in memo:
                return memo[n]
            #res = dp(n >> 1) + (x % 2)
            res = dp(n & (n-1)) + 1
            memo[n] = res
            return res
        
        ans = [0]*(n+1)
        for i in range(1,n+1):
            ans[i] = dp(i)
        
        return ans

###############################
# 01MAR22
# 1490. Clone N-ary Tree
###############################
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        '''
        i can use dfs to traverse the tree but keep cloned copy outside the rescursive function
        then move
        '''
        def dfs(node):
            if not node:
                return
            cloned_node = Node(val = node.val)
            for child in node.children:
                cloned_node.children.append(dfs(child))
            
            return cloned_node
        
        return dfs(root)

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        '''
        iterative dfs
        mainitin root and cloned in stack
        
        note for bfs just replace stack with deque
        '''
        if not root:
            return root
        
        cloned = Node(val = root.val)
        stack = [(root,cloned)]
        
        while stack:
            curr,copy = stack.pop()
            if not curr:
                continue
            for child in curr.children:
                copy_child = Node(child.val)
                copy.children.append(copy_child)
                stack.append((child,copy_child))
            
        return cloned

###############################
# 02MAR22
# 392. Is Subsequence
###############################
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        '''
        we just want to check if s is a substring of t
        just move pointers in s and t and return if we moved all the way to s
        '''
        #s is too big
        if len(s) > len(t):
            return False
        
        ptr_s = 0
        ptr_t = 0
        
        while ptr_t < len(t) and ptr_s < len(s):
            #match
            if s[ptr_s] == t[ptr_t]:
                ptr_s += 1
            ptr_t += 1
        
        return  ptr_s == len(s)

#recursive solution
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        '''
        we can use recursion 
        dp(i,j) answers the question if s[:i] is a subsequence of s[:j]
        if s[i] == t[j] then we are left so solve the subproblem on s[i+1:] and s[j+1]
        if s[i] != t[j], then we are forced to advance the pointer j into target and see if we can 
        find a subsequence s[:i] in t[:j+1]
        
        base cases:
            if we have advanced i, then it must be a subsequence
            if we have advance j, but still letters are unmatched
        '''
        memo = {}
        
        def dp(i,j):
            #base caes
            if i == len(s):
                return True
            if j == len(t):
                return False
            if (i,j) in memo:
                return memo[(i,j)]
            #recursive case
            if s[i] == t[j]:
                res = dp(i+1,j+1)
                memo[(i,j)] = res
                return res
            res = dp(i,j+1)
            memo[(i,j)] = res
            return res
        
        dp(0,0)
        print(memo)

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        '''
        follow up question can be solved greedily using hashmap
        if we repeatedly have incoming chars s1,s2,s3....n
        we have to repeatedly search the target string each time a new char s comes in
        one way is to precompute a hash for each char in t mapping to its index
        
        now with the hashmap precomputed, we match greedily
        when searching indices, we can use binary search to avoid scaning linearly
        we want the index that is just greater than the current index
        if the index we find is >= len(s) it can't be dont because there are no more indices to choose from
        
        We use the pointer to check if an index is suitable or not. For instance, for the character a whose corresponding indices are [0, 3], we need to pick an index out of all the appearances as a match. Suppose at certain moment, the pointer is located at the index 1. Then, the suitable greedy match would be the index of 3, which is the first index that is larger than the current position of the target pointer.
        '''
        mapp = defaultdict(list)
        for i,char in enumerate(t):
            mapp[char].append(i)
            
        curr_idx = -1
        for char in s:
            if char not in mapp:
                return False
            idxs = mapp[char]
            candidate = bisect.bisect_right(idxs,curr_idx)
            #if i got to the end of the idxs list, there isn't an index just one greate
            if candidate != len(idxs):
                curr_idx = idxs[candidate]
            else:
                return False
            
        return True