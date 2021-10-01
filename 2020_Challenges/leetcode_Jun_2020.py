######################################################
#INVERTING BINARY TREE
#######################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        #for practice i want you to code all traversals
        def pre_order_traverse(root):
            if root:
                print root.val
                pre_order_traverse(root.left)
                pre_order_traverse(root.right)
        def in_order_traverse(root):
            if root:
                in_order_traverse(root.left)
                print root.val
                in_order_traverse(root.right)
        def post_order_traversal(root):
            if root:
                post_order_traversal(root.left)
                post_order_traversal(root.right)
                print root.val
        def invert_traverse(root):
            if root:
                invert_traverse(root.right)
                print root.val
                invert_traverse(root.left)

# Definition for a binary tree node.
# recursive call
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return root
        
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        
        #swap
        root.right = left
        root.left = right
        
        return root

##iterative stack solution
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        #DFS
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop(-1)
            if node:
                node.left, node.right = node.right, node.left
                stack.append(node.left)
                stack.append(node.right)
        return stack

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

class Solution(object):
    def twoCitySchedCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        #get the difference between the two cities [0] from  [1]
        #sort on the difference increasing
        #split the sorted lists in two, in the first half take [0] in the other half take [1]
        
        costs.sort(key = lambda x: x[0] - x[1])
        
        N = len(costs)
        half = N // 2
        
        min_sum = sum([a for a,b in costs[:half]]) + sum([b for a,b in costs[half:]])
        
        return min_sum

class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        a_pointer = 0
        b_pointer = len(s) - 1
        
        while a_pointer <= b_pointer:
            #swap
            temp = s[b_pointer]
            s[b_pointer] = s[a_pointer]
            s[a_pointer] = temp
            
            #inc and dec
            a_pointer += 1
            b_pointer -= 1

class Solution(object):
    
    def __init__(self, w):
        """
        :type w: List[int]
        """
        #create cum sum of weights for each w in weight
        total_weight = 0
        self.cumsum_weights = []
        for i in w:
            #update total
            total_weight += i
            self.cumsum_weights.append(total_weight)
        

    def pickIndex(self):
        """
        :rtype: int
        """
        #pick random number between 1 and the largest weight
        ran = random.randint(1,self.cumsum_weights[-1])
        lo = 0
        hi = len(self.cumsum_weights) - 1
        while lo <= hi:
            mid = (lo+hi) // 2
            if self.cumsum_weights[mid] < ran:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo


class Solution(object):
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        position of the shortest person wiil always have n-1 people in front
        
        [7,0] , [4,4] , [7,1] , [5,0] , [6,1] , [5,2]
        #sort by height, and then by q
        [4,4], [5,0], [5,2], [6,1], [7,0], [7,1]
        first smallet person can be assigned that index in the list
        None , None, None, None, [4,4], None
        dump the next smallest person at their kth index, so long as there isn't another repeated person
        
        [5,0] , None, None, None, [4,4], None
        [5,0] , None, [5,2],None, [4,4], None
        [5,0] , None, [5,2],[6,1],[4,4], None
        [5,0] , [7,0],[5,2],[6,1],[4,4], [7,1]
        '''
        N = len(people)
        results = [None]*N
        people.sort()
        
        for h,k in people:
            #set counters, i is the index in the output
            #j counts the number of nones
            i,j = 0,-1
            while (i < N):
                #if an element does not exist in the output or if height at the ith output is the height if the current person
                if not results[i] or results[i][0] == h:
                    j += 1
                #check if the number of nones if equal to q
                if j == k:
                    break
                i += 1
            results[i] = [h,k]
        return results
                
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        '''
        5 [1,2,5]
        ones: 
        twos: 
        fives: 
        5 minus 1 > 0 ones+1
        4 minus 1 > 0 ones + 1
        3 minus 1 > 0 ones + 1
        ...
        1 minus 1 == 0 stop
        start with the amount and subtract the lowest denomination until it this zero or less than zero, count the number of times this fuction is called
        go on to the next deomination,
        recurse
        
        '''
        def decrement_amount(amount, denom, counter):
            if amount - denom > 0:
                amount = amount - denom
                counter[0] += 1
                decrement_amount(amount, denom, counter)
                return counter[0]
        a = decrement_amount(amount, coins[0], [0])
        print a


class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        '''
        use as dp array
        columns are amount number
        rows are the denominations available
             0 1 2 3 4 5
        []   1 0 0 0 0 0
        [1]  1 1 1 1 1 1
        [2]  1 1 2 2 3 3
        [5]  1 1 2 2 2 4
        first row populatate 1, 0, 0, 0 , amount+1
        first col populations 1, 1, 1, 1....len(denoms) + 1
        
        
        '''
        rows = len(coins) + 1
        cols = amount + 1
        
        dp = [[0 for i in range(cols)] for j in range(rows)]
        #populate dp[1,1]
        dp[0][0] = 1 
        #populate first row
        for c in range(1,cols):
            dp[0][c] = 0
        #populations first col
        for r in range(1,rows):
            dp[r][0] = 1
        for r in range(1,rows):
            for c in range(1,cols):
                #take only from row above
                if c - coins[r-1] >= 0:
                    dp[r][c] = dp[r-1][c] + dp[r][c-coins[r-1]]
                else:
                    dp[r][c] = dp[r-1][c]
        
            
        return dp[rows-1][cols-1]

n = 1024

def get_binary(n,res):
    res = ''
    if n % 2 == 0:

class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        if int is a power of two, there should only be 1 in in its binary represenation
        '''
    
        if (n % 2 == 0):
            while (n % 2 == 0):
                n = n / 2
            if n == 1:
                return True
            else:
                return False
        else:
            return False

class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        keep incrementing by one and see if that number is 2**n
        '''
        check, i = 1,1
        while check <= n:
            if check == n:
                return True
            check = 2**i
            i += 1
        return False

class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n <= 0:
            return False
        return n & (n-1) == 0
        
class Solution(object):
    def isSubsequence(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        #pop elements off a stack
        stack = list(s)
        
        for char in t:
            if stack and char == stack[0]:
                stack.pop(0)
                
        return stack == []

class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        #in the case that the target is beyond the last element in nums
        nums.append(float('inf'))
        lo = 0
        hi = len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if target <= nums[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        #this is just a sorting algorithm
        #do insertion sort first
        for current in range(0,len(nums)):
            #set pointer 
            i = current
            while (i > 0) and (nums[i] < nums[i-1]):
                #swap
                temp = nums[i]
                nums[i] = nums[i-1]
                nums[i-1] = temp
                i -= 1
        return nums

class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        #three pointers, red, white, blue
        #[2,0,2,1,1,0]
        #update each pointer after an occurence
        red, white, blue = 0,0, len(nums)-1
        while (white <= blue):
            if nums[white] == 0:
                #swap with  a red
                nums[red], nums[white] = nums[white], nums[red]
                white += 1
                red += 1
            #if white is 1 good, leave it there but update the pointer
            elif nums[white] == 1:
                white += 1
            else:
                nums[white],nums[blue] = nums[blue], nums[white]
                blue -= 1


class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.items = {}

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if self.items.get(val):
            return False
        self.items[val] = True
        return True

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if self.items.get(val):
            self.items.pop(val)
            return True
        return False

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return random.choice(list(self.items.keys()))


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

class Solution(object):
    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        [1,2,4,8]
        [1], [1,2], [1,2,4],[1,2,4,8]
        '''
        nums.sort()
        N = len(nums)
        if N < 2:
            return nums
        results = [[num] for num in nums]
        
        for i in range(N):
            for j in range(i):
                if nums[i] % nums[j] == 0 and len(results[i]) < len(results[j]) + 1:
                    results[i] = results[j] + [nums[i]]
                    
        return max(results,key = lambda x: len(x))

'''
Dijkstra review

Let distance of start vertex from start vertex = 0
Let distance of all other vertices from start be inf

while len(vertices visted) > 0
do:
    visit the unvisited vertex with the smallest known distance
    for the current vertext, examines its unvisited neighbors
    for the current vertext, calculate the distance of each neighbor from the start vetex
        if the calculated distances of a vertex is less than the known distance, 
            update shortest distance
            updates previous vertext for each updated distances
            add the current vertext to the list of visited vertices
'''

def Dijkstra(graph, source):
    '''
    graph is a list of lists where the index represents the node number
    at the nodes if gives the distance to anoterh node
    [[1,100],[2,500]]
    means from the zeroth node to the first node is a distane of 100
    from the zeroth nnode to the second node is a distance of 500

    '''
    n = len(graph)

    Q = [i for i in range(n)]
    dist = [float('inf') for _ in range(n)]
    prev = [None for ) in range(n)]
    #create adjacency ist
    adj = [[] for _ in range(n)]
    #populate adj list
    for u,v,w in graph:
        adj[u].append((v,w))
    
    dist[source] = 0

    while len(Q) > 0:
        #find vertext in Q that has the least distance to u
        for vertex in Q:
            if graph[vertex][1] == dist[source]:
                u = vertex
        Q.popleft()

        for v in adj[u]:
            alt = dist[u] + graph[v][1]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    return(dist,prev)

import collections
class Solution(object):
    def findCheapestPrice(self, n, flights, src, dst, K):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type K: int
        :rtype: int
        """
        #init adjaceny list
        adj = [[] for i in range(n)]
        output = [float('inf') for i in range(n)]
        output[src] =  0
        
        for u,v,w in flights:
            adj[u].append((v,w))
        
        graph = collections.deque()
        #append src node with a -1 to track if we stopped by it
        graph.append((src,-1,0))
        
        while graph:
            u,stops,cost = graph.popleft()
            #stop updating graph when K has been met
            if stops >= K:
                continue
            for v,w in adj[u]:
                #output update, if the output at our source plus the weight is less than than the output going to v, update it
                if cost + w < output[v]:
                    output[v] = cost + w
                    graph.append((v,stops+1,cost+w))
                
        if output[dst] == float('inf'):
            return -1
        else:
            return output[dst]

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        def dfs(node):
            if not node:
                return 
            if node.val == val:
                return node
            elif val < node.val:
                #traverse left
                return dfs(node.left)
            else:
                #traverse right
                return dfs(node.right)
        return dfs(root)

import re
class Solution(object):
    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        #first check if IP address is IPv4
        #if not check  if IPv6
        #if not return "Neither"
        ip4, ip6 = True, True
        ip = IP.split(".")
        
        #check thats its a set for 4
        if len(ip) == 4:
            for n in ip:
                if not n or re.search('\D',n) or int(n) > 255 or (n != "0" and len(n) - len(n.lstrip("0")) !=0):
                    ip4 = False
                    break
        else:
            ip4 = False
        
        #function to check that a number is hex
        def is_hex(x):
            try:
                int(x,16)
                return True
            except ValueError:
                return False
            
        ip = IP.split(":")
        if len(ip) == 8:
            for n in ip:
                if not n or re.search('\W',n) or len(n) > 4 or not is_hex(n):
                    ip6 = False
                    break
        else:
            ip6=False
            
        if ip4:
            return "IPv4"
        elif ip6:
            return "IPv6"
        else:
            return "Neither"

#alternative methods
#two functions
def isIP4(s):
    try:
        return str(int(s)) == s and 0 <= int(s) <= 255
    except:
        return False

def isIP6(s):
    if len(s) > 4:
        return False
    if re.match("\W",s):
        return False
    try:
        int(s,16)
    except:
        return False

ip4 = IP.split(".")
ip6 = IP.split(":")

if len(ip4) == 4 and all(isIP4(n) for n in ip4):
    return "IPv4"
if len(ip6) == 8 and all(isIP6(n) for n in ip6):
    return "IPv4"
return "Neither"


class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        #check borders recursively, if O on a border and touching  other O's mark as dont flip
        #this part if DFS
        #otherwise flip everything  else
        #than traverse the  board  flipping each O to an X passing on those ones that our DFS disocvered
        if not board:
            return 
        height = len(board)
        width = len(board[0])
        
        
        def dfs_marker(r,c):
            #must be within the board
            #invoke within solve after
            if 0<=r<height and 0<=c<width and board[r][c] == 'O':
                #recrusively call
                board[r][c] = 'C'
                dfs_marker(r-1,c)
                dfs_marker(r+1,c)
                dfs_marker(r,c-1)
                dfs_marker(r,c+1)
        
        #invoke only on first/last row/column
        #rows
        for r in [0, height -1]:
            for c in range(width):
                dfs_marker(r,c)
                
        #cols
        for r in range(height):
            for c in [0, width-1]:
                dfs_marker(r,c)
                
        #final traverse
        #if C flip back into  an O
        #if O flip back inoto an X
        
        for r in range(height):
            for c in range(width):
                if board[r][c] == 'C':
                    board[r][c]  = 'O'
                elif board[r][c]  == 'O':
                    board[r][c] = 'X'
        
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        citations.sort()
        h_index = 0
        papers = 0
        #create pointers
        i = 0
        while (i <= len(citations)-1):
            for cit in citations:
                if cit >= h_index:
                    papers += 1
            if papers >= citations[i]:
                #update
                h_index = citations[i]
            papers = 0    
            i +=  1
        return h_index


class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        '''
        [0,1,3,5,6] citations
        [5,4,3,2,1] h index is just len(citations) - index
        stops when h index < citations[index]
        binary search
        return h index
        '''
        if not citations:
            return 0
        N = len(citations)
        l = 0
        r = N - 1
        
        while (l < r):
            mid =  (l+r) // 2
            if citations[mid] >= N - mid:
                r = mid
            else:
                l = mid + 1
        
        if citations[l] == 0:
            return 0
                
        return N  - l

'''
knuth-morris-pratt algo
store prefix table, where its the longest pattern that is both a prefix and a suffix
m is the pattern length P
i if the longest prefix that has been found in the pattern
(prefix meaning that is also a suffix at P[i])
j i the current index of the pattern for which are calcualting pi
pi is the prefix table
'''

def kmp_prefix_table(p):
    '''
    p is apttern
    pi i the prefix table
    '''
    m = len(p)
    pi = [0]*m
    i = 0

    for j in range(1,m):
        while (i > 0) and (p[i+1] != p[j]):
            i = pi[i]
        if p[i+1] == p[j]:
            i += 1
        pi[j] = i

    print(pi)

kmp_prefix_table(p='ATAGGGGG')

class Solution(object):
    def longestDupSubstring(self, S):
        """
        :type S: str
        :rtype: str
        """
        '''
        largest matching substring in string would be length - 1
        brute force: check if each substring  is in the actual string
        two ways, from the front and back
        which is just O(2*len(s)*len(s-1))
        in pseudo code
        occurence = 0
        for i in range(0,len(S)):
            pattern = S[i:len(S)]
            #scan array for this patternn
            for j in range(0,len(S)):
                window = S[j:j+len(pattern)]
                if pattern == window and len(pattern) > occruence:
                    occurence = pattern
        this won't work
        binary search method, if we had a helper function
        use helper function to see if there is a string of length n that matches, check if there  is more, conversely, if less check to see if there is even less
        '''
        p = 2**63 -2
        def rabin_karp(mid):
            #finds a repesting  subsequence
            #load in a sequence
            cur_hash = 0
            for i in range(mid):
                #update cur_hash
                cur_hash = (cur_hash*26 + nums[i]) % p
            hashes = {cur_hash}
            #-1 for pos  means we did not find a repeating  subseqence
            pos = -1
            max_pow = pow(26,mid,p)
            for i in range(mid, len(S)):
                cur_hash = (26*cur_hash - nums[i- mid]*max_pow + nums[i]) % p
                if cur_hash in hashes:
                    pos = i + 1 - mid
                hashes.add(cur_hash)
            return pos
        
        #binary search
        l,r = 0,len(S)-1
        start,end = 0,0
        #creates nums hash
        nums = [ord(c) for  c in S]
        
        while (l<=r):
            mid = (l+r) // 2
            pos = rabin_karp(mid)
            if pos == -1:
                #checj shorted subsequence
                hi = mid -1
            else:
                start,end = pos, pos+mid
                l = mid + 1
                
        return S[start:start+l-1]


#wirting the permute function
#the base case is when the input is just one character
def permute(s):
    out = []

    if len(s) == 1:
        out = [s]
    else:
        #loop through each charcter
        for i,let in enumerate(s):
            #permuate through al other posibilites
            for perm in permute(s[:i]+s[i+1:]):
                out += [let+perm]
    return out

print(permute('ABC'))


#permutations sequence
'''
n = 6,k = 314
start with [1,2,3,4,5,6]
can have 6 * 5!, or 120 different permutations  that have 1 - 6 as the first digit
314 < 2*120 and > 3*120
first digit must be 3
3 [1,2,4,5,6]

past to so 2*5! =  314 - 2*120
74
what permutation starting with digit 3 is 74 after
thre are 5 digits left, if we place the next digit we can have it 4! ways
74 - 3*4! = 2, we take the third digit in the list it must 5
35 [1,2,4,6]

there are 2 indices left
plae the nth digit, there 3! ways, or 6 ways, i cant push anymore since there
only 2 left it must be 1
351 [2,4,6]
2-0*3! = 2

3512
'''
import math
class Solution(object):
    def getPermutation(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        #creating a list of numbers
        numList = [i for i in range(1,n+1)]
        #initiliaze string
        results = ""
        #caulcate inital factorial value
        fact = math.factorial(n)
        while (n>0):
            #get partitions for the permutations list
            fact /= n
            #get the ceiling from the partition list at this iteration
            div = math.ceil(k/fact)
            #pop off the right digit from numList using div as the index
            results += str(numList.pop(int(div-1)))
            #update k
            k %= fact
            #decrement n until we finish
            n -= 1
        return results
        

            class Solution(object):
    def calculateMinimumHP(self, dungeon):
        """
        :type dungeon: List[List[int]]
        :rtype: int
        """
        '''
        start at the P cell and dp back
        '''
        height, width = len(dungeon), len(dungeon[0])
        m,n = height - 1, width - 1
        
        dp = dungeon
        #get minimum health at princess squard
        dp[m][n] = max(1, 1 - dp[m][n]) #want the minmum health as a positive number
        
        #go up from P at right most oclumn
        for r in range(m-1,-1,-1):
            #difference of row below and current cell
            dp[r][n] = max(1, dp[r+1][n] - dp[r][n])
        
        #for columns
        for c in range(n-1,-1,-1):
            dp[m][c] = max(1, dp[m][c+1] - dp[m][c])
            
        #traverse the rest of the array
        for r in range(m-1,-1,-1):
            for c in range(n-1,-1,-1):
                dp[r][c] = max(1,min(dp[r+1][c],dp[r][c+1]) - dp[r][c])
                
        return dp[0][0]
        
import collections
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        lazy method using counter
        '''
        count = collections.Counter(nums)
        
        for k,v in count.items():
            if v == 1:
                return k

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        #traverse the tree in any order
        #every time the function is executed increament a counter
        def traverse_count(node):
            if not node:
                return 0
            return traverse_count(node.left) + traverse_count(node.right) + 1
        
        return traverse_count(root)

class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        notes
        this is just a twist on the catalan numbers
        which is just:
        C_{n} = \frac{1}{n+1} combinations(2n,n)
        or
        \frac{(2n)!}{(n+1)!n!}
        or
        product_{k=2}^{n} \frac{n+k}{k}
        first we  define G(n): as the unique BST for a seq lenght of n
        G(0) = 1 and G(1) = 1
        For each element in n, we place it and recurse left and right
        We can define F(i,n)  as the unique number of BST we can create when we place i at the root given n and i is in n
        so G(n) = \sum_{n}^{F(i,n)}
        We realize that:
        F(i,n) = G(i-1)*G(n-1)
        we can solve using a dynamic programming array
        the base case is G(0) = G(1) = 1
        then G(3) = F(1,3)  + F(2,3) + F(3,3)
                  = G(0)*G(2) + G(1)*G(1) + G(2)*G(0)
                  = G(0)*G(2) + G(1)*G(1) + (G(1)*G(1))*G(0)
        '''
        #create dp array to store up to n+1 elements
        dp = [0]*(n+1)
        #solve the base case
        dp[0], dp[1] = 1,1
        #fill up the dp array
        #i is each n
        #j is the root given n
        for i in range(2,n+1):
            for j in range(1,i+1):
                dp[i] = dp[i] + (dp[i-j]*dp[j-1])
                
        return dp[n]
        

class Solution(object):
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        memo = {}
        def helper(start,end):
            #beyond bounds of 1 and n
            if (start < 1) or (end > n) or (start >= end):
                return 1
            count =  0
            if (start,end) in memo:
                return memo[(start,end)]
            else:
                for i in range(start,end+1):
                    count += helper(start,i-1)*helper(i+1,end)
                    memo[(start,end)] = count
                
            return count
        return helper(1,n)

import collections
class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        count = collections.Counter(nums)
        
        for k,v in count.items():
            if v  ==  2:
                return k
            elif v > 2:
                return k

class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #ffloayds algorithm, cycle detection
        #there a 1 to n unique integers of list len(n)
        #there must be a duplicate
        #so a cycle exists
        #floyds algo
        turtle = hare = nums[0]
        #advance turtle nums[turtle] times
        turtle = nums[turtle]
        #advance hare
        hare = nums[nums[hare]]
        
        while (turtle != hare):
            #repeat
            turtle = nums[turtle]
            hare = nums[nums[hare]]
            
        #move turtle back to beginnning
        turtle = nums[0]
        while turtle != hare:
            turtle = nums[turtle]
            hare = nums[hare]
            
        return turtle

'''
review
[1,3,4,2,2]
start:
    turle = 1
    hare = 1

    turtle = 3
    hare = 2
while t != h:
    turtle = 2
    hare = 2

#move back to beginning:
    turtle = 1

while t != h:
    turle = 3
    hare = 4

    tutrle = 2
    hare = 2

'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        #dfs to the end of each leaf
        #a leaf has no left or right node
        #edge case, there is an empty binary tree
        if not root:
            return 0
        
        output = []
        
        def pre_order_traverse(node, sofar):
            #if this node is a leaf, add node value to list so far
            if not node.left and not node.right:
                output.append("".join(sofar+[str(node.val)]))
                
            if node.left:
                pre_order_traverse(node.left,sofar+[str(node.val)])
            if node.right:
                pre_order_traverse(node.right,sofar+[str(node.val)])
            
                
            
        pre_order_traverse(root,[])
        
        return sum([int(x) for x in output])
                
            
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        similart to the min number of coins problem
        warm up, first generate list of perfect squars up n
        '''
        i,k,squares = 1,1,[]
        while (k<=n):
            squares.append(k)
            i += 1
            k = i**2
        
        #create dp array
        #this is just 1 d
        '''
        n = 12
        0 1 2 3 4 5 6 7 8 9 10 11 12
        1 1 2 3 4 5 6 7 8 9 10 11 12
        4 0 0 0 1 2 3 4 2 3 4  5  3
        9 0 0 0 0 0 0 0 0 1 2  3  4
        
        #for each square update the minmum number of times it goes into n
        '''
        dp = [float('inf') for _ in range(0,n+1)]
        
        for s in squares:
            for i in range(1,n+1):
                #if square can go in to  number:
                if s <= i:
                    #first check if sqaure can evenly go into n
                    if i % s == 0:
                        candidate = i / s
                        #update dp
                        dp[i] = min(candidate,dp[i])
                    else:
                        dp[i] = min(1+dp[i-s], dp[i])
        
        return dp[n]

class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        #scan the list of lists to get the pair that has jfk
        #add jfk to output  list
        #get the next airport
        #find the nextairport
        #keep updating the next airport
        output = []
        pointer = "JFK"
        while (len(tickets) > 1):
            for trip in tickets:
                if trip[0] == pointer:
                    output.append(trip[0])
                    pointer = trip[1]
                    tickets.remove(trip)
        output =  output + tickets[0]
        return output

#import collections
class Solution(object):
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        #create  instance vars in calss
        
        #first step, build the adj list
        self.adj = collections.defaultdict(list)
        tickets.sort(key = lambda x:x[1])
        
        for u,v in tickets:
            self.adj[u].append(v)
        
        self.result = []
        self.dfs('JFK')
        
        return self.result[::-1]
        
    #def dfs is method 
    def dfs(self,s):
        while (s in self.adj) and (len(self.adj[s]) > 0):
            v = self.adj[s][0]
            self.adj[s].pop(0)
            self.dfs(v)
        self.result.append(s)
        
        
'''
[
[0,1]
[0,1]
[1,0]
]
'''

class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        #solve bottom up
        #let m be the number of rows
        #n be the number of columns
        
        if m == 1 and n == 1:
            return 1
        
        dp = [[0 for _ in range(n)] for _ in range(m)]
        
        #populate last row and last column
        #row first
        for c in range(n-1):
            dp[m-1][c] = 1
        #then columns
        for r in range(m-1):
            dp[r][n-1] = 1
        #go backwards in our dp
        #start at second to last row  and second to last column
        for r in range(m-2, -1, -1):
            for c in range(n-2,-1,-1):
                dp[r][c] =  dp[r+1][c] + dp[r][c+1]
            
        return dp[0][0]


class Trie(object):
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        #create root, which is just the dictionary
        self.root = dict()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        #get the structure
        p = self.root
        for char in word:
            #if char isn't in the dict
            if char not in p:
                #add char to dict with another dict
                p[char] = dict()
            #get new root
            p = p[char]
        p["#"] = True
    
    def find(self,prefix):
        p = self.root
        for char in prefix:
            if char not in p:
                return None
            p = p[char]
        return p
        
                
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.find(word)
        return node is not None and "#" in node
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        return self.find(prefix) is not None

#create Trie class
class Trie(object):
    #initialize the tree
    def __init__(self):
        #instance variable
        self.root = dict()
    def insert(self,word):
        p = self.root
        for char in word:
            if char not in p:
                p[char] = dict()
            #get the root, and update p
            p = p[char]
        #once you get the end of the word
        p['#'] = True
    #finds prefix in trei
    def find(self,prefix):
        #get the struct
        p = self.root
        for char in prefix:
            if char not in p:
                return None
            p = p[char] #this updates p
        return p
    def search(self,word):
        node = self.find(word)
        return node is not None and "#" in node
    def startsWith(self, prefix):
        return self.find(prefix) is not None
    

class Solution(object):
    def findWords(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        t = Trie()
        for w in words:
            t.insert(w)
        N = len(board)
        M = len(board[0])
        output = []
        
        def helper_dfs(row,col,path):
            #base case or visited
            #marking visited cells with a 0
            if (row < 0) or (row >=N) or (col < 0) or (col >=M) or (board[row][col] == 0):
                #this just ends the dfs, meaning we are done
                return
            #base case again
            #if the visited string does not start with stop
            temp = path+[board[row][col]]
            string  = "".join(temp)
            if not t.startsWith(string):
                return #just ends
            #if starts with is tru
            elif t.search(string) and string not in output:
                output.append(string)
            #start dfs checkign all four directions
            placeholder = board[row][col]
            board[row][col] = 0
            helper_dfs(row-1,col,temp)
            helper_dfs(row+1,col,temp)
            helper_dfs(row,col-1,temp)
            helper_dfs(row,col+1,temp)
            board[row][col] = placeholder
            
        #invoke dfs
        for n in range(N):
            for m in range(M):
                helper_dfs(n,m,[])
        return output
            







