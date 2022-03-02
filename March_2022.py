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