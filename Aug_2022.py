#!/usr/local/bin/python3

###############################################
# 744. Find Smallest Letter Greater Than Target
# 02AUG22
###############################################
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        '''
        just binary search and return the right boung
        '''
        index = bisect.bisect_right(letters,target)
        if index == len(letters):
            return letters[0]
        else:
            return letters[index]

#can also use modulo
class Solution(object):
    def nextGreatestLetter(self, letters, target):
        index = bisect.bisect(letters, target)
        return letters[index % len(letters)]


class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        '''
        just binary search and return the right boung
        '''
        
        # if the number is out of bound
        if target >= letters[-1] or target < letters[0]:
            return letters[0]
        
        
        left = 0
        right = len(letters) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            #we need to find the one just greater than the target
            if letters[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        
        return letters[left % len(letters)]


###################################
# 760. Find Anagram Mappings
# 02AUG22
####################################
class Solution:
    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        '''
        for nums2, mapp nums to index, then retreive idnex for nums1
        '''
        nums2_mapp = {}
        for i,num in enumerate(nums2):
            nums2_mapp[num] = i
        
        ans = []
        for num in nums1:
            ans.append(nums2_mapp[num])
        
        return ans

############################
# 378. Kth Smallest Element in a Sorted Matrix (REVISTED)
# 02JUL22
#############################
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        '''
        really we can reframe the porblem as finding the Kth smallest element among N sorted lists
        think about finding the kth smallest in two sorted lists
            advance two pointers increasinly, then return the min of the two pointers
            
        IMPORTANT:
            say our matrix has 100 rows, and we want to find the kth smallest
            if the matrix is sorted increasingly left to right, then up to down
            then we would only need to look in the first 5 rows of the matrix
            O(min(rows,k)); overall time complexity would be
            O(min(rows,k)) + K*min(rows,K)
        
        algo:
            maintain min heap, where entry is (value,row,column)
            we extrat smallest then keep advancing along column
                note: we only advance along a row, if we can stay in the current row
        '''
        N = len(matrix)
        
        #minheap
        heap = []
        for r in range(min(N,k)):
            heap.append((matrix[r][0],r,0))
            
        #heapify
        heapq.heapify(heap)
        
        #keep extracing min and advancing along a row
        while k:
            value,row,col = heapq.heappop(heap)
            #if we can advance along row
            if col < N - 1:
                heapq.heappush(heap, (matrix[row][col+1],row,col+1))
                
            #use up k
            k -= 1
        
        return value

class Solution:
    
    def countLessEqual(self, matrix, mid, smaller, larger):
        
        count, n = 0, len(matrix)
        row, col = n - 1, 0
        
        while row >= 0 and col < n:
            if matrix[row][col] > mid:
               
                # As matrix[row][col] is bigger than the mid, let's keep track of the
                # smallest number greater than the mid
                larger = min(larger, matrix[row][col])
                row -= 1
                
            else:
                
                # As matrix[row][col] is less than or equal to the mid, let's keep track of the
                # biggest number less than or equal to the mid
                
                smaller = max(smaller, matrix[row][col])
                count += row + 1
                col += 1

        return count, smaller, larger
    
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        
        n = len(matrix)
        start, end = matrix[0][0], matrix[n - 1][n - 1]
        while start < end:
            mid = start + (end - start) / 2
            smaller, larger = (matrix[0][0], matrix[n - 1][n - 1])

            count, smaller, larger = self.countLessEqual(matrix, mid, smaller, larger)

            if count == k:
                return smaller
            if count < k:
                start = larger  # search higher
            else:
                end = smaller  # search lower

        return start

############################
# 427. Construct Quad Tree
# 03AUG22
############################
"""
# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""

class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        '''
        a quad tree is a tree struture where each inteal node has four children
        each node has two attributes
            val: True if the node representas a grid of 1's or False if node represents a grid of 0s
            isLeaf: True if the node is a leaf node or False if the nodes had four children
            
        algo:
            if the current grid has the same values (all 1's or all 0's) set isLeaf to True and val to to be the value of the grid and set children to None
            if the current grid has different values, set isLeaf to false and set val to any value and divide the grid into four
            recurse on the four
            
        Start with the full grid and keep on diving it four parts.
        Once we have a grid of size 1 then start with merging.
        Compare if all the leaf nodes,
        if yes then merge all into a single node.
        else, return all four nodes separately.
        
        each call will define the grid as the top left corner
        '''
        
        def build(row,col,grid_length):
            #base case, single element
            if grid_length == 1:
                val = grid[row][col]
                node = Node(val,True,None,None,None,None)
                return node
            
            #otherwise recurse, dividing them
            #represent each grid defined by coordinate in top left
            topLeft = build(row,col,grid_length //2)
            topRight = build(row,col + grid_length // 2, grid_length//2)
            bottomLeft = build(row + grid_length // 2, col, grid_length//2)
            bottomRight = build(row + grid_length //2, col + grid_length //2, grid_length//2)
            
            #check all leaves
            if topLeft.isLeaf == topRight.isLeaf == bottomLeft.isLeaf == bottomRight.isLeaf == True:
                #check all valuess are the same
                if topLeft.val == topRight.val == bottomLeft.val == bottomRight.val:
                    #build
                    node = Node(topLeft.val,True,None,None,None,None)
                    return node
            
            #make false node
            node = Node(-1,False,topLeft,topRight,bottomLeft,bottomRight)
            return node
        
        return build(0,0,len(grid))

#####################
# 729. My Calendar I
# 03AUG22
#####################
class MyCalendar:

    def __init__(self):
        '''
        we can use binary search to search for the position in the array who's start is just smaller then the start we want to add
        then we must check that this end is works
        '''
        self.intervals = []
    def book(self, start: int, end: int) -> bool:
        i = self.binarySearch(start)
        #i is where we want to insert
        #check the one previous to it and make sure there is no overlap
        if i > 0 and self.intervals[i-1][1] > start:
            return False
        #check starts
        if i < len(self.intervals) and end > self.intervals[i][0]:
            return False
        self.intervals.insert(i,[start,end])
        return True
        
    def binarySearch(self,target):
        left = 0
        right = len(self.intervals)
        while left < right:
            mid = left + (right - left) //2
            if self.intervals[mid][0] >= target:
                right = mid
            else:
                left = mid + 1
        return left


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)

'''
using sorted container
keep track all 2n points
then binary search twice
1. make sure that q1 == q2, which is the point we need to insert start and end
2. also check for q1 % 2 == 0
    i.e, it lies at an even index, because we alaways need to insert two pointts
    i the index were odd, it means we over lap on an end
'''
from sortedcontainers import SortedList

class MyCalendar:
    def __init__(self):
        self.arr = SortedList()
        
    def book(self, start, end):
        #we need to find the index that is just greater than start, and just smaller than end
        q1 = SortedList.bisect_right(self.arr, start)
        q2 = SortedList.bisect_left(self.arr, end)
        if q1 == q2 and q1 % 2 == 0:
            self.arr.add(start)
            self.arr.add(end)
            return True
        return False

# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)


#using balanced binary tree
'''
we can use a self balacing binary tree, i don't think you'd be expected
and build the insertion method recursively
'''

class Node:
    def __init__(self,start,end):
        self.start = start
        self.end = end
        self.left = None
        self.right = None
    
    def insert(self,start,end):
        #if the end we want to insert is bigger then our start
        if self.start >= end:
            #put to its left
            if self.left is None:
                self.left = Node(start,end)
                return True
            else:
                return self.left.insert(start,end)
        elif self.end <= start:
            if self.right is None:
                self.right = Node(start,end)
                return True
            else:
                return self.right.insert(start,end)
        else:
            return False

class MyCalendar:

    def __init__(self):
        self.root = None
        

    def book(self, start: int, end: int) -> bool:
        if not self.root:
            self.root = Node(start,end)
            return True
        else:
            return self.root.insert(start,end)
        


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)

###########################
# 858. Mirror Reflection (REVISITED)
# 04AUG22
###########################
class Solution:
    def mirrorReflection(self, p: int, q: int) -> int:
        '''
        we need to determine the number receptor the ray hits first, either 0,1,or 2    
    
        if we were to stack rooms on top of each other multiple times, the laser beam would just keep boucing
        specifically if we fire from the southwest corner, it will travel up a distance of q to the right wall
        then up another distance of q to the left well
        then again to the right wall up a distance of q
        draw the case for when p = 3 and q = 2
        we can translate this to m*p = n*q
        
        where m is the number of room extensions + 1
        and q is the number of laser beam reflections+ 1
        
        cases
            1. if the number of light reflections is odd (which mean n is even) the possible corner is on the left hand side, so the possible corner is 2. otherwise the corner is on the right hand side
            2. if the corner is on the right hand side, 
                if the number of room extensions is even (implying m is odd) it means the corner is 1
                else 0
                
        we can conclude
        
        m is even & n is odd => return 0.
        m is odd & n is odd => return 1.
        m is odd & n is even => return 2.
        
        Note: The case m is even & n is even is impossible. Because in the equation m * q = n * p, if m and n are even, we can divide both m and n by 2. Then, m or n must be odd.

https://leetcode.com/problems/mirror-reflection/discuss/2377070/Pseudocode-Explain-Why-Odd-and-Even-Matter
https://leetcode.com/problems/mirror-reflection/discuss/2377070/Pseudocode-Explain-Why-Odd-and-Even-Matter
        
        m = the number of rooms after the extension.
n = the number of laser ray lights traveling inside these rooms before it hits corner 0, 1 or 2.

In the example of p=3 and q=2, we can derive m = 2 and n = 3, i.e. there are two floors for the rooms and there are 3 light turning back and forth to finally reach a corner.

Now, to decide the condition of m and n when deriving the corner, here are the tricks:
1.1 odd m indicates the light ends up upwards
1.2 even m indicates the light ends up downwards
2.1 even n indicates round-trips and it should end up hitting the left wall
2.2 odd n indicate round-trips+1 trip to the right wall

So we can conclude:
(m % 2 == 0 && n % 2 == 1) return 0; //downwards and right wall
(m % 2 == 1 && n % 2 == 1) return 1; // upwards and right wall
(m % 2 == 1 && n % 2 == 0) return 2; // upwards and left wall
        '''
        #solve m*p = n*q
        m = q
        n = p
        
        while m % 2 == 0 and n % 2 == 0:
            #reduce away from even
            m //= 2
            n //= 2
        
        #check the cases
        if m % 2 == 0 and n % 2 == 1:
            return 0
        if m % 2 == 1 and n % 2 == 1:
            return 1
        
        if m % 2 == 1 and n % 2 == 0:
            return 2
        
        return -1

#using lcm
#https://leetcode.com/problems/mirror-reflection/discuss/2376355/Python3-oror-4-lines-geometry-w-explanation-oror-TM%3A-9281
class Solution:
    def mirrorReflection(self, p: int, q: int) -> int:

        L = lcm(p,q)

        if (L//q)%2 == 0:
            return 2

        return (L//p)%2

#########################
# 458. Poor Pigs (REVISITED)
# 07AUG22
#########################
class Solution:
    def poorPigs(self, buckets: int, minutesToDie: int, minutesToTest: int) -> int:
        '''
        there is exactly one poisonus bucket, if the bucket is poisonous, the big will die in minutesToDie
        we are left with only minutesToTest, we can use and many pigs as we need
        return the minimum number of pigs need to figure which is poisonous
        
        how many states does a pig have?
        * if there is no time to test, i.e minToTest / minToDie = 0, only one state, alive
        * if minToTest / minToDie = 1, then the pig has time to die, so it can be alive or dead
        * if minToTest / minToDie = 2, then there are three states available, alive/dead after first, dead after second
        
        number of available states is minToTest / minToDie + 1
        if one pig has two available states, then x pids can test 2^x buckets, since one pig can test two buckets (find out if one of the two is poisonus)
        
        how many buckets could test x pigs with s available states?
            we can text s^x buckets
            
        we degenerate the problem to:
            find x, such that states^x >= buckets
            where x is the number of pigs
            states = minToTest /minToDie + `
            
        x = log(buckets) / log(states)
        '''
        states = minutesToTest // minutesToDie + 1
        return math.ceil(math.log(buckets) / math.log(states))

##############################
# 366. Find Leaves of Binary Tree (REVISITED)
# 08AUG22
##############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        '''
        we can use topsort on this problem
        '''
        #track indegree and  build graph, we treat the leaf nodes as prereqs meaning we start with them first
        indegree = defaultdict(int)
        graph = defaultdict(list)
        
        stack = [root]
        while stack:
            curr = stack.pop()
            child = 0
            if curr.left:
                stack.append(curr.left)
                graph[curr.left].append(curr)
                child += 1
            if curr.right:
                stack.append(curr.right)
                graph[curr.right].append(curr)
                child += 1
            indegree[curr] = child
        
        #start with leave nodes
        q = deque([])
        for node,deg in indegree.items():
            if deg == 0:
                q.append(node)
        
        #topsort
        ans = []
        while q:
            ans.append([])
            N = len(q)
            for i in range(N):
                curr = q.popleft()
                ans[-1].append(curr.val)
                for neigh in graph[curr]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 0:
                        q.append(neigh)
        
        return ans

###############################
# 433. Minimum Genetic Mutation
# 08AUG22
###############################
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        '''
        we are given a start string and an end string
        we want to get from start to end, using the minimum number of muttations,
        we are only allowed to use mutations in the gene bank
        we define a mutation a s sinlge char being changed in the string
        
        i can treat this as a graph problem, where each node is a mutation (must be a mutation in the gene bank)
        this then become shortest path from start to end, then i can solve this using bfs
        '''
        #if end is not in the bank, we can immmeidatly return a -1
        #we may also need to add the start gene to the bank
        bank = set(bank)
        if end not in bank:
            return -1
        
        #need generationg function for a gene
        def getAllowedMutations(gene):
            for i in range(8):
                ch = gene[i]
                for mut in ['A','C','G','T']:
                    if ch != mut:
                        new = gene[:i]+mut+gene[i+1:]
                        if new in bank:
                            yield new
        
        #bfs
        q = deque([(start,0)])
        
        while q:
            curr,path = q.popleft()
            if curr == end:
                return path
            for neigh in getAllowedMutations(curr):
                q.append((neigh,path+1))
                #we can prune the space by removing a mutation from the bank#
                #bank.remove(neigh)
        
        return -1

###########################
# 439. Ternary Expression Parser
# 09AUG22
###########################
class Solution:
    def parseTernary(self, expression: str) -> str:
        '''
        https://leetcode.com/problems/ternary-expression-parser/discuss/122935/Python-solution-with-detailed-explanation
        we need to process from right to left
        use stack and push the digits on to the stack, 
        we can ignored ':' and when we hit '?' we need to test whehter the char preceeding it is a T or F
        depending on that, pop values from the stack and eval the exrpession
        then push the evaluated expression back on to the stack
        key insight is wwe need to find the first '?' from the right, evaluate it and push it back on the stack
        return what the expression that is left on the stack at the end
        '''
        #starting from the right
        i = len(expression) - 1
        
        stack = []
        
        while i >= 0:
            #push on stack the most recent digit
            ch = expression[i]
            if ch.isdigit():
                stack.append(ch)
                i -= 1
            #push T or F
            elif ch in ['T','F']:
                stack.append(ch)
                i -= 1
            #pass on colon
            elif ch == ':':
                i -= 1
            #we need to eval
            elif ch == '?':
                #move one more
                i -= 1
                #get the expression for T and F
                #the string will be guaranteed to be a valid expression, no need to check for edge caes
                true = stack.pop()
                false = stack.pop()
                
                if expression[i] == 'T':
                    stack.append(true)
                else:
                    stack.append(false)
                
                i -= 1
            
        
        return stack[-1]


class Solution:
    def parseTernary(self, expression: str) -> str:
        '''
        another way
        '''
        stack = []
        
        for ch in reversed(expression):
            #if there is something on the stack and is ?
            if stack and stack[-1] == "?":
                stack.pop()
                first = stack.pop()
                stack.pop()
                second = stack.pop()
                #eval
                if ch == "T":
                    stack.append(first)
                else:
                    stack.append(second)
            else:
                stack.append(ch)
        
        return stack[0]

class Solution:
    def parseTernary(self, expression: str) -> str:
        '''
        ternary expressions exhibit optimal substructure
        rather:
            expr ? expr if true:expr is false
        
        so we first need to grab the expression to the left of the question mark (which is always single T or F)
        and the recursively falle t o the vlaues for expre if trye and expre if false
        
        base case:
            when the next char is a value (T,F, or digit) and not another expression
            this is determined by checking whaat is next to it
            if there is a ':' it is a value, else it is another expression
            each recursive call return the evaluated value and the next position to be processed by the parent call
        '''
        self.N = len(expression)
        
        def rec(i):
            #base case, bext char is a value, or we reach the end of the expression
            #return the the value
            if (i + 1 >= self.N) or (i + 1 < self.N and expression[i+1] == ':'):
                return expression[i], i + 2
            exp = expression[i]
            left, i = rec(i+2)
            right,i = rec(i)
            if exp == 'T':
                return left,i
            else:
                return right,i
        
        right,i = rec(0)
        return right

###################################
# 748. Shortest Completing Word
# 09AUG22
###################################
class Solution:
    def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
        '''
        clean up license plate to make counter object of on chars only
        '''
        counts = Counter()
        for ch in licensePlate:
            ch = ch.lower()
            if ch.isalpha():
                counts[ch] += 1
        
        #sort words based on length and starting index
        words = [(word,i) for i,word in enumerate(words)]
        words.sort(key = lambda x: (len(x[0]),x[1]))
        
        for word,i in words:
            temp = copy.deepcopy(counts)
            for ch in word:
                if ch in temp and temp[ch] > 0:
                    temp[ch] -= 1
                if temp[ch] == 0:
                    del temp[ch]
            
            if len(temp) == 0:
                return word

#cool way using and operator and lambda filter
#note, sorting by length would have also sorted by earliest index anyway
def shortestCompletingWord(self, licensePlate: str, words: List[str]) -> str:
    pc = Counter(filter(lambda x : x.isalpha(), licensePlate.lower()))
    return min([w for w in words if Counter(w) & pc == pc], key=len) 

##########################
# 10AUG22
# 443. String Compression
##########################
class Solution:
    def compress(self, chars: List[str]) -> int:
        '''
        we can keep pointers i and j
        we start by sending j out and we update the chars array in place using i
        chars[i] = chars[j], then advnace j
        we keep sending out j if it matches the current char at i
        and increment the count
        if the count is greater than 1, we need to update it the following spaces with each string(int)
        '''
        left,right = 0,0
        N = len(chars)
        
        while right < N:
            #set first ocrrurence
            chars[left] = chars[right]
            count = 1
            
            #try to incrementthe count
            while right + 1 < N and chars[right] == chars[right+1]:
                right += 1
                count += 1
            
            #if we have a count larger than 1
            if count > 1:
                for c in str(count):
                    chars[left+1] = c
                    left += 1
            
            left += 1
            right += 1
        
        return left

#########################################################
# 762. Prime Number of Set Bits in Binary Representation
# 12JUN22
#########################################################
class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        '''
        make helper function to count the number of bits
        if the largest number is 10**6,
        then there can only be at most log2(10**6) + 1 bit positions
        there are only 20 bit positions,
        how many primes are between 2 and 20
        
        largest = 10**6
        num_bit_positions = 0
        while largest:
            num_bit_positions += 1
            largest >>= 1
        
        '''
        possible_primes = set([2,3,5,7,11,13,17,19])
        
        def count_bits(num):
            count = 0
            while num:
                count += num & 1
                num >>= 1
            return count
        
        ans = 0
        for num in range(left,right+1):
            count = count_bits(num)
            if count in possible_primes:
                ans += 1
        
        return ans
            
##########################################################
# 762. Prime Number of Set Bits in Binary Representation
# 12AUG22
##########################################################
class Solution:
    def countPrimeSetBits(self, left: int, right: int) -> int:
        '''
        make helper function to count the number of bits
        if the largest number is 10**6,
        then there can only be at most log2(10**6) + 1 bit positions
        there are only 20 bit positions,
        how many primes are between 2 and 20
        
        largest = 10**6
        num_bit_positions = 0
        while largest:
            num_bit_positions += 1
            largest >>= 1
        
        '''
        possible_primes = set([2,3,5,7,11,13,17,19])
        
        def count_bits(num):
            count = 0
            while num:
                count += num & 1
                num >>= 1
            return count
        
        ans = 0
        for num in range(left,right+1):
            count = count_bits(num)
            if count in possible_primes:
                ans += 1
        
        return ans
            
#########################################
# 783. Minimum Distance Between BST Nodes
# 12AUG22
##########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        '''
        inorder traversal checking difference each time between prev and curr 
        then update when new minimum has been reached
        '''
        self.prev = float('-inf')
        self.min_dist = float('inf')
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            #update
            self.min_dist = min(self.min_dist,node.val - self.prev)
            self.prev = node.val
            inorder(node.right)
        
        inorder(root)
        return self.min_dist

###########################################################
# 30. Substring with Concatenation of All Words (Revisited)
# 13AUG22
############################################################
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        '''
        first solutino involved keeping hashmap of counts of words, and trying using the words in the substring up
        and check if we could have done this from this index
        the only problem is that we have to recompute the hashmaps of words every time to get the counts
        we also repeated alot because may have travesed an index already part of the substring 
        
        turns out we can also use a sliding window
        instead of calling the check functions, we try to find all valid substrings in one pass using sliding iwndow
        
        left and right pointers, with right moving in increments of wordlength
        if we find a word not in words, move up left past this word and start again, also if we exceed the length of the substring size
        
        we also need to make sure of multiplicity of each words
        example,s = 'foofoobar'
        words = ['foo','bar'], we need to only foo once
        
        this yields another criterios for moving the left bound
            we move left until we find the exceees word and remove it
        '''
        n = len(s)
        k = len(words)
        word_length = len(words[0])
        substring_size = word_length * k
        word_count = collections.Counter(words)
        
        def sliding_window(left):
            words_found = collections.defaultdict(int)
            words_used = 0
            excess_word = False
            
            # Do the same iteration pattern as the previous approach - iterate
            # word_length at a time, and at each iteration we focus on one word
            for right in range(left, n, word_length):
                if right + word_length > n:
                    break

                sub = s[right : right + word_length]
                if sub not in word_count:
                    # Mismatched word - reset the window
                    words_found = collections.defaultdict(int)
                    words_used = 0
                    excess_word = False
                    left = right + word_length # Retry at the next index
                else:
                    # If we reached max window size or have an excess word
                    while right - left == substring_size or excess_word:
                        # Move the left bound over continously
                        leftmost_word = s[left : left + word_length]
                        left += word_length
                        words_found[leftmost_word] -= 1

                        if words_found[leftmost_word] == word_count[leftmost_word]:
                            # This word was the excess word
                            excess_word = False
                        else:
                            # Otherwise we actually needed it
                            words_used -= 1
                    
                    # Keep track of how many times this word occurs in the window
                    words_found[sub] += 1
                    if words_found[sub] <= word_count[sub]:
                        words_used += 1
                    else:
                        # Found too many instances already
                        excess_word = True
                    
                    if words_used == k and not excess_word:
                        # Found a valid substring
                        answer.append(left)
        
        answer = []
        for i in range(word_length):
            sliding_window(i)

        return answer

##############################
# 126. Word Ladder II (REVISITED)
# 14AUG22
##############################
#https://leetcode.com/problems/word-ladder-ii/discuss/2423485/Python-Three-steps-approach
#https://leetcode.com/problems/word-ladder-ii/discuss/2367587/Python-BFS-%2B-DFS-With-Explanation-Why-Optimization-Is-Needed-to-Not-TLE
#the last previous approaches get TLE because we keep going down paths that are dead ends
class Solution:

    WILDCARD = "."
    
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        """
        Given a wordlist, we perform BFS traversal to generate a word tree where
        every node points to its parent node.
        
        Then we perform a DFS traversal on this tree starting at the endWord.
        """
        if endWord not in wordList:
            # end word is unreachable
            return []
        
        # first generate a word tree from the wordlist
        word_tree = self.getWordTree(beginWord, endWord, wordList)
        
        # then generate a word ladder from the word tree
        return self.getLadders(beginWord, endWord, word_tree)
    
    
    def getWordTree(self,
                    beginWord: str,
                    endWord: str,
                    wordList: List[str]) -> Dict[str, List[str]]:
        """
        BFS traversal from begin word until end word is encountered.
        
        This functions constructs a tree in reverse, starting at the endWord.
        """
        # Build an adjacency list using patterns as keys
        # For example: ".it" -> ("hit"), "h.t" -> ("hit"), "hi." -> ("hit")
        adjacency_list = defaultdict(list)
        for word in wordList:
            for i in range(len(word)):
                pattern = word[:i] + Solution.WILDCARD + word[i+1:]
                adjacency_list[pattern].append(word)
        
        # Holds the tree of words in reverse order
        # The key is an encountered word.
        # The value is a list of preceding words.
        # For example, we got to beginWord from no other nodes.
        # {a: [b,c]} means we got to "a" from "b" and "c"
        visited_tree = {beginWord: []}
        
        # start off the traversal without finding the word
        found = False
        
        q = deque([beginWord])
        while q and not found:
            n = len(q)
            
            # keep track of words visited at this level of BFS
            visited_this_level = {}

            for i in range(n):
                word = q.popleft()
                
                for i in range(len(word)):
                    # for each pattern of the current word
                    pattern = word[:i] + Solution.WILDCARD + word[i+1:]

                    for next_word in adjacency_list[pattern]:
                        if next_word == endWord:
                            # we don't return immediately because other
                            # sequences might reach the endWord in the same
                            # BFS level, this is important here
                            found = True
                        if next_word not in visited_tree:
                            if next_word not in visited_this_level:
                                visited_this_level[next_word] = [word]
                                # queue up next word iff we haven't visited it yet
                                # or already are planning to visit it
                                q.append(next_word)
                            else:
                                visited_this_level[next_word].append(word)
            
            # add all seen words at this level to the global visited tree
            visited_tree.update(visited_this_level)
            
        return visited_tree
    
    
    def getLadders(self,
                   beginWord: str,
                   endWord: str,
                   wordTree: Dict[str, List[str]]) -> List[List[str]]:
        """
        DFS traversal from endWord to beginWord in a given tree.
        """
        def dfs(node: str) -> List[List[str]]:
            if node == beginWord:
                return [[beginWord]]
            if node not in wordTree:
                return []

            res = []
            parents = wordTree[node]
            for parent in parents:
                res += dfs(parent)
            for r in res:
                r.append(node)
            return res

        return dfs(endWord)

#another way
class Solution:
    def findLadders(
        self, beginWord: str, endWord: str, wordList: list[str]
    ) -> list[list[str]]:

        # 1. Create adjacency list
        def adjacencyList():

            # Initialize the adjacency list
            adj = defaultdict(list)

            # Iterate through all words
            for word in wordList:

                # Iterate through all characters in a word
                for i, _ in enumerate(word):

                    # Create the pattern
                    pattern = word[:i] + "*" + word[i + 1 :]

                    # Add a word into the adjacency list based on its pattern
                    adj[pattern].append(word)

            return adj

        # 2. Create reversed adjacency list
        def bfs(adj):

            # Initialize the reversed adjacency list
            reversedAdj = defaultdict(list)

            # Initialize the queue
            queue = deque([beginWord])

            # Initialize a set to keep track of used words at previous level
            visited = set([beginWord])

            while queue:

                # Initialize a set to keep track of used words at the current level
                visitedCurrentLevel = set()

                # Get the number of words at this level
                n = len(queue)

                # Iterate through all words
                for _ in range(n):

                    # Pop a word from the front of the queue
                    word = queue.popleft()

                    # Generate pattern based on the current word
                    for i, _ in enumerate(word):

                        pattern = word[:i] + "*" + word[i + 1 :]

                        # Itereate through all next words
                        for nextWord in adj[pattern]:

                            # If the next word hasn't been used in previous levels
                            if nextWord not in visited:

                                # Add such word to the reversed adjacency list
                                reversedAdj[nextWord].append(word)

                                # If the next word hasn't been used in the current level
                                if nextWord not in visitedCurrentLevel:

                                    # Add such word to the queue
                                    queue.append(nextWord)

                                    # Mark such word as visited
                                    visitedCurrentLevel.add(nextWord)

                # Once we done with a level, add all words visited at this level to the visited set
                visited.update(visitedCurrentLevel)

                # If we visited the endWord, end the search
                if endWord in visited:
                    break

            return reversedAdj

        # 3. Construct paths based on the reversed adjacency list using DFS
        def dfs(reversedAdj, res, path):

            # If the first word in a path is beginWord, we have succesfully constructed a path
            if path[0] == beginWord:

                # Add such path to the result
                res.append(list(path))

                return

            # Else, get the first word in a path
            word = path[0]

            # Find next words using the reversed adjacency list
            for nextWord in reversedAdj[word]:

                # Add such next word to the path
                path.appendleft(nextWord)

                # Recursively go to the next word
                dfs(reversedAdj, res, path)

                # Remove such next word from the path
                path.popleft()

            # Return the result
            return res

        # Do all three steps
        adj = adjacencyList()
        reversedAdj = bfs(adj)
        res = dfs(reversedAdj, [], deque([endWord]))

        return res

################################
# 68. Text Justification
# 15AUG22
################################
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        '''
        we want to format words such that each line has exactly maxWidth characters and is fully (left or right justified)
        we want to pack words in greedy approach, that is pack as many words as you can in each line
            * pad extra spaces when necessary so that each line has exactly maxWidth characters
            
        notes
            * extra psaces between words should be distributed as evenly as possible
            * if the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than on right
            * for the last line of text, it should be left justified and no extra space is inserted between words
            
        intuition:
            assigngin extra spaces on the left just really means round robbing
            
            for i in range(maxWidth - num_of_letters):
                cur[i%(len(cur)-1 or 1)] += ' '
        
        once you determine that there only k words that can fit on a given line, you know what the total length of those words is num_of_letters
        then the rest are spaces, and there are (maxWidth - num_of_letters) of spaces
        the 'or 1' part is for dealing iwth the edge case len(curr) == 1
        
        
        '''
        result = [] #stores answer
        curr_line = [] #the current line
        num_letters = 0 #calculate letters to be put on this line
        
        for word in words:
            #the total number of chars in curr_line _ total number of hcars in word + total number of words
            #i.e if we cant fit the next word, we need to justify
            if num_letters + len(word) + len(curr_line) > maxWidth:
                #we need to adjust
                #size will be used for round robbing
                #use max 1, because at least one word would be there
                size = max(1,len(curr_line)-1)
                
                for i in range(maxWidth - num_letters):
                    #add space to this word at this line
                    index = i % size
                    curr_line[index] += ' '
                
                #add curr line to ans and reset
                result.append("".join(curr_line))
                curr_line = []
                num_letters = 0
            
            #otherwise add word to current line
            curr_line.append(word)
            num_letters += len(word)
        
        #last line, we need to left justifty
        curr_line = " ".join(curr_line).ljust(maxWidth)
        result.append(curr_line)
        return result

########################
# 796. Rotate String
# 16AUG22
########################
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        if i simulate shifting, for all possible shifts
        i can use q to simulate in O(1) time
        '''
        s = list(s)
        q = deque(s)
        N = len(s)
        count = 0
        
        while count < N:
            temp = "".join(q)
            print(temp)

            if temp == goal:
                return True

            q.append(q.popleft())
            count += 1
        
        return False

class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        if we were to concate s with itselft
        s+s, then we just need to ensure goal in in this substring
        '''
        return len(s) == len(goal) and goal in s+s

#rolling hash
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        our goal is to check whether goal is a substring of s + s
        we can use a rolling hash to solve this questions
        idea behind hashing:
            hash(a substinrg of s + s) is uniformly distributed between [0,1,2...mod-1]
            and of hash(substring of s) == hash(goal), it is very liekly to be a match
        
        recall pow(x,y,z) == x**y mod z
        
        first step, get hash of both s and goal then check
        otherwise we need to go through s again update hash and and check if it matches goal
        
        another way would have been to contacte s and then use rolling hash with that

        '''
        mod = 10**9 + 7
        prime = 113 #use large prime
        pinv = pow(prime,mod-2,mod)
        
        hash_goal = 0
        power = 1
        for ch in goal:
            code = ord(ch) - ord('a')
            hash_goal = (hash_goal + power*code) % mod
            power = power*prime % mod
        
        hash_s = 0
        power = 1
        for ch in s:
            code = ord(ch) - ord('a')
            hash_s += (power*code) % mod
            power *= prime % mod
        
        if hash_s == hash_goal and s == goal:
            return True
        
        for i,ch in enumerate(s):
            code = ord(ch) - ord('a')
            hash_s += power*code
            hash_s -= code
            hash_s *= pinv
            hash_s %= mod
            if hash_s == hash_goal and s[i+1:] + s[:i+1] == goal:
                return True
        
        return False

#kmp
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        using kmp
        remember we have degenerated the problem of finding goal in s+s
        https://web.stanford.edu/class/cs97si/10-string-algorithms.pdf
        https://www.youtube.com/watch?v=JoF0Z7nVSrA&ab_channel=NeetCode
        lps, longest prefix also a suffic, that is not the length of the current prefix
        '''
        if len(s) != len(goal):
            return False
        if len(s) == 0:
            return True
        
        #make strings 1 index
        N = len(s)
        s = ' '+s+s
        goal = ' '+goal

        #caclualte the pi table, i.e longest prefix also a suffic
        pi = [0]*(N+1)
        left = -1
        pi[0] = -1
        
        for right in range(1,N+1):
            while left >= 0 and goal[left+1] != goal[right]:
                left = pi[left] #recurse
                #pi[i] answers the question, what it the longest prefix in goal[i] that is also a suffix
            left += 1
            pi[right] = left
            
        #matching part
        j = 0
        for i in range(1,(2*N)+1):
            #try to match, and when we can't we need to advance our pointer
            while j >= 0 and goal[j+1] != s[i]:
                j = pi[j]
            j += 1
            if j == N:
                return True
        return False
        

#############################
# 800. Similar RGB Color 
# 16AUG22
#############################
class Solution:
    def similarRGB(self, color: str) -> str:
        '''
        this is hex
        can be [0-9,a-f]
        which is base 16
        
        instead of minimzing the squred differnces, we can minimuze the absolute value, 
        this still gives us the same orerder
        
        first find the length 2 pairs [00,11,22....ff]
        then break apart color in 3 pairs of two
        caluclate the absolutdifference, and get the minimum
        '''
        
        #function to get closest for a len 2 code
        def closest(code):
            possible = ['00', '11', '22', '33', '44', '55', '66', '77', '88', '99', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff']
            sims = []
            
            #get all possible scores
            for cand in possible:
                score = abs(int(code,16) - int(cand,16))
                sims.append((score,cand))
                
            #find closest one, through sorting
            sims.sort(key = lambda x: x[0])
            return sims[0][1]
        
        ans = "#"
        for i in range(1,len(color),2):
            ans+= closest(color[i:i+2])
        
        return ans

##########################################
# 1570. Dot Product of Two Sparse Vectors
# 17AUG22
##########################################
#lazy implementation works
class SparseVector:
    def __init__(self, nums: List[int]):
        self.nums = nums
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        N = len(self.nums)
        v1 = self.nums
        v2 = vec.nums
        ans = 0
        for a,b in zip(v1,v2):
            ans += a*b
        return ans
            
# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)

#still slow???
class SparseVector:
    def __init__(self, nums: List[int]):
        self.sparse = {}
        self.size = len(nums)
        for i,num in enumerate(nums):
            if num:
                self.sparse[i] = num
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        N = self.size
        v1 = self.sparse
        v2 = vec.sparse
        ans = 0
        #just take one, it doesn't matter anyway if there is a zero
        for i, num in v1.items():
            if i in v2:
                ans += num*v2[i]
        
        return ans
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)

class SparseVector:
    def __init__(self, nums: List[int]):
        '''
        we can repsresent the array is tuple pairs (index,num)
        then we can use two pointers to pass them pait
        also note the time complexity of the hashing function as well
        and priotrize looping over the sparse vector who's length of non zero values is shorter
        
        '''
        self.pairs = []
        for i,num in enumerate(nums):
            if num != 0:
                self.pairs.append((i,num))
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        ans = 0
        i = 0
        j = 0
        #i into first, j into second
        while i < len(self.pairs) and j < len(vec.pairs):
            if self.pairs[i][0] == vec.pairs[j][0]:
                ans += self.pairs[i][1]*vec.pairs[j][1]
                i += 1
                j += 1
            #always advance the lagging pointer to catch up
            elif self.pairs[i][0] < vec.pairs[j][0]:
                i += 1
            else:
                j += 1
        
        return ans
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)

################################
# 539. Minimum Time Difference
# 17AUG22
################################
#close one
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        '''
        convert the string to minuts past 0, then sort
        first take the min difference between adjacent intervals
        but don't forget to do the ends wit the beginning
        edge case 00:00 must be set to 1440
        '''
        times = []
        for t in timePoints:
            if t == "00:00":
                times.append(1440)
            else:
                total_mins = 0
                hr,mins = t.split(":")
                total_mins += int(hr)*60 + int(mins)
                times.append(total_mins)
        
        times.sort()
        return min(times[1] - times[0],times[-1] - times[-2])

#what the fuck?!?!?
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        '''
        convert the string to minuts past 0, then sort
        first take the min difference between adjacent intervals
        but don't forget to do the ends wit the beginning
        edge case 00:00 must be set to 1440
        '''
        times = []
        for t in timePoints:
            if t == '00:00':
                times.append(24*60)
            else:
                total_mins = 0
                hr,mins = t.split(":")
                total_mins += int(hr)*60 + int(mins)
                times.append(total_mins)
        
        times.sort()
        ans = float('inf')
        for i in range(1,len(times)):
            ans = min(ans, times[i] - times[i-1])
        
        #one more final one
        #minutes from mindnight, plus the first time
        ans = min(ans,60*24 - times[-1] + times[0])
        return ans

################################################
# 659. Split Array into Consecutive Subsequences
# 18AUG22
################################################
#close one, 160/187
#donest quite pass all the cases
class SubSeq:
    def __init__(self,start,end,size):
        #class to store elements in subsequence
        self.start = start
        self.end = end
        self.size = size
    
    #comparator
    def __lt__(self,nxt):
        #increasing end and increasing size
        if self.end < nxt.end and self.size < nxt.size:
            return self.end
        elif self.end < nxt.end and self.size > nxt.size:
            return nxt.end
        else:
            return self.end
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        '''
        we are given a non-decreasing order array
        we need to determine if it is possible to split nums into one or more subsequences such that:
            1. each subsequence is a consectuvie increasing sequence (one more than the other)
            2. all subsequences have length of 3 or more
            
        using repeated numbers could not possibly make a subsequence
        i need to use all the nums, but the order can be shifted around (must maintain relative ordering)
        
        intuition:
            if we have x subsequences, then we simply need to add nums[i] to the right xth subsequence
            or make a new onw
            
            example, in the case there are no repeated numbers
            [1,2,3,5,6,8,9,10]
            [1,2,3] [5,6] [8,9,10]
            
        we don't need to store all the subsequences, just [first num,last num] in the subsequence onto a heap, and the size of the subsequnce can be deduced
        [last - first + 1]
        if we are at nums[i], we need to decide where to put it, greedily add it to the subsequence who's last elements i exactly one less (max heap by last element)
        [0,1,1,2,2,3]
        we need to do [0,1,2], [1,2,3]
            if the last element is the same for two or more subsequences, sort subsenqce on increasing order of their size
            
        heap invariant
            Increasing order of their last element.
            When the subsequences' last elements are equal, then the two subsequences will be placed in increasing order of length.
            
        for each element nums[i] we compare nums[i] with the last element heap[0][1] of the existing subsequence
        1. nums[i] > last + 1, nums[i] cannot be part of this subsequence, so we rmeot this subseqeunce from the heap and compare to the next one
        2. nums[i] == last (or heap is empty), nums[i] cannot be part here either
            since we stored subsequences by increasing order of last element, there will be no other subseqneces in the heap to append to, so we must create a new subseqnece here
        3. nums[i] == last + 1, add this to that subequence
        

        '''
        heap = []
        for num in nums:
            #condition 1, remove non qualifying subsequences
            while len(heap) != 0 and heap[0].end + 1 < num:
                curr_subseq = heapq.heappop(heap)
                #check for size requiremnt
                if curr_subseq.size  < 3:
                    print(heap)
                    return False
            
            #condition 2, new subsequence
            if len(heap) == 0 or heap[0].end == num:
                sub_seq = SubSeq(num,num,1)
                heapq.heappush(heap,sub_seq)
            else:
                #condition three, add to an existing subseqnece
                curr_subseq = heapq.heappop(heap)
                sub_seq = SubSeq(curr_subseq.start,num,curr_subseq.size+1)
                heapq.heappush(heap,sub_seq)
        
        while heap:
            curr_subseq = heapq.heappop(heap)
            if curr_subseq.size < 3:
                return False
        
        return True

class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        '''
        the easier to way is to implement using hash map
        
        '''
        #map needed val to heap of len(lists)
        #we want to estend the shortest list of that needed val first
        ending_num_to_length = defaultdict(list)
        
        for num in nums:
            #look for last possible ending num
            if num - 1 in ending_num_to_length and len(ending_num_to_length[num-1]) != 0:
                #add to shorest one
                prev_length = heapq.heappop(ending_num_to_length[num-1])
                heapq.heappush(ending_num_to_length[num],prev_length+1)
            else:
                #make new entry
                heapq.heappush(ending_num_to_length[num],1)
                
        #check all
        for lengths in ending_num_to_length.values():
            if lengths and any([l < 3 for l in lengths]):
                return False
        return True

#another way
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        subsequences = []
        
        for num in nums:
            while subsequences and subsequences[0][0] + 1 < num:
                sub = heapq.heappop(subsequences)
                if sub[1] < 3:
                    return False
            
            if not subsequences or subsequences[0][0] == num:
                heapq.heappush(subsequences, [num, 1]) # end, len
            else:
                # Pop and push to maintain order
                sub = heapq.heappop(subsequences)
                sub[0] += 1
                sub[1] += 1
                heapq.heappush(subsequences, sub)
                
        while subsequences:
            sub = heapq.heappop(subsequences)
            if sub[1] < 3:
                return False
            
        return True

#greedy using maps
class Solution:
    def isPossible(self, nums: List[int]) -> bool:
        '''
        now lets try saying that for each element nums[i], we will try to see if is it possible to form a valid subsequence with the reamining elements
        i.e if we can start of build a subsequence using nums[i]
        while creating a new subsequence with nums[i] as the starting element, we will check if a vlid subseuence is possible or not with nums[i] as the starting element
        for nums[i] to be valid, we need at least nums[i] + 1 and nums[i] + 2 in the array (hashmap!!)
        we can count and make sure they are avilable
        
        what if we ant to add nums[i] to one of the existing subequences?
            when we create a new subseuqnece using nums[i] we ensur that a valid subsequence is possible
            so we dont need a heap to record the length of sort the subsequences
            for adding nums[i] to an esiting subseuence we only need to know if such a subequecne exsists == nums[i] - 1 (seen hash set)
            
        we just need to store the last element in the subseuence in a set when a subsequence is modified
        this way if we want to add nums[i] to an exisiting subsequence, we can just check if nums[i] - 1 exists in the set
        why is a set enough? because we can get the answer in o(1) time
        if there is more than one subsequence with the same last element, 
        we can use another count mapp to store the frequency
        
        algo:
        1. init two mapps, one to store the inital counts and the other that store the counts of for subsequences ending with num
        2. iterate over the nums array and update the frequency map
            * if the frequency of the current element is 0, that means the num is already consider to be a part of a valid subsequence, continue
            * check if we can add this current num to an existing subsequence, so we check for nums - 1 in mapp
                if there exists an entry we can add numt to one of them
                if no such subseuecne exists, we need to create a new one
                    before doing so, we need to check if nums + 1 and nums + 2 exists
                    if the don't its not possbile and reutnr false
                    otherwise make the changes
        '''
        counts = Counter(nums)
        count_subseqs_ending = Counter()
        
        for num in nums:
            #num already in use
            if counts[num] == 0:
                continue
            
            #if valid subseq exsits with last elemnt num - 1, we ensure this a valid to being with because of the third rule
            if count_subseqs_ending[num-1] > 0:
                #get the current count
                curr_count = count_subseqs_ending[num-1]
                # consumer it
                count_subseqs_ending[num-1] = curr_count - 1
                #new count ending
                count_subseqs_ending[num] += 1
            #if we want to start a new subsequence, chec for num+1 and num+2 and update
            elif (counts[num+1] > 0) and (counts[num+2] > 0):
                count_subseqs_ending[num+2] += 1
                #consume
                counts[num+1] -= 1
                counts[num+2] -= 1
            else:
                return False
            
            counts[num] -= 1
        
        return True

#################################
# 871. Minimum Number of Refueling Stops (REVISITED)
# 21AUG22
#################################
#this is just 0/1 knapsack
class Solution:
    def minRefuelStops(self, target: int, startFuel: int, stations: List[List[int]]) -> int:
        '''
        this is a 0/1 knapsack problem
        we let dp(i,j) represent the max fuel we can get when we are at station i and have used j fuel stops
        
        base cases
        when (i,j) == (0,0), there is no fuel or stops to be used, return -
        #if j isn't between 1 and i, return a really larger negative number
        '''
        #add start and end conditions to statinos array
        stations = [[0, startFuel]] + stations + [[target, 0]]
        
        # dp(i, j) = maximum amount of fuel we can get when we are at station i and we have used j fuel stops (i not included)
        
        # j should be between [1, i]
        memo = {}
        def dp(i, j):
            if i == 0 and j == 0:
                return 0
            if not (1 <= j <= i):
                return float("-inf")
            
            if (i,j) in memo:
                return memo[(i,j)]
            #take, we need to used up fuel from i-1 to i
            ans1 = dp(i-1, j) - (stations[i][0]-stations[i-1][0])
            # don'\ take, we gain fuel from i-1 to i
            ans2 = dp(i-1, j-1) - (stations[i][0]-stations[i-1][0]) + stations[i-1][1]
            
            #check of edge condtions when taking previous subproblems
            ans = max(ans1, ans2)
            if ans < 0:
                ans = float('-inf')
                memo[(i,j)] = ans
                return ans
            else:
                memo[(i,j)] = ans
                return ans
        
        for j in range(len(stations)):
            if dp(len(stations)-1, j) >= 0:
                return j-1  # we return j-1 since first stop (we added the starting point) is mandatory and hence should be removed from final solution 
            
        return -1

################################
# 936. Stamping The Sequence (Revisited)
# 21AUG22
#################################
#https://leetcode.com/problems/stamping-the-sequence/discuss/189576/C%2B%2B-simple-greedy
class Solution:
    def movesToStamp(self, stamp: str, target: str) -> List[int]:
        '''
        recall we want to convert s to target using at most 10*len(target) turns
        it easier if we were to start from the target
        say we have target: 'aabccbc'
        and stamp 'abc'
        
        first we try to find subtring abs, and repalce with wildecard
        after there are no more replacements, we wil try '*bc' and 'ab*' turn by turn
        
        idea leads to greedy solution that produces the minimum number of stamps
        '''
        res = []
        cnt = 0
        #helper function to turn go backwards from current target to wildcarding with temp
        def f(cnt, target): 
            for i in range(len(stamp)): 
                for j in range(i + 1): 
                    stp = '*' * j + stamp[j: len(stamp) - (i - j)] + (i - j) * '*'
                    #using walrus operator to help with loop initialization
                    #could also put k here
                    #k = target.find(strp)
                    while (k := target.find(stp)) > -1: 
                        res.append(k)
                        target = target[:k] + '*' * len(stamp) + target[k + len(stamp):]
                        #incremant count for stamp use, each letter replacement counts as a turn
                        #we are counting the number of times we replaced it with a '*'
                        cnt += len(stamp) - i
                        print(k, stp, target,cnt)
                        #change it again
                        #k = target.find(stp)
            return cnt, target
        
        while True: 
            #simulate to try and replace
            cnt1, target1 = f(cnt, target)
            #if we couldn't replace it
            if cnt1 == cnt: 
                break
            #reassign
            cnt = cnt1
            target = target1
        #print(target)
        return res[::-1] if cnt == len(target) else [] #chck if all characters are converted

###############################
# 562. Longest Line of Consecutive One in Matrix
# 22AUG22
###############################
class Solution:
    def longestLine(self, mat: List[List[int]]) -> int:
        '''
        i really only need to check row,col,diags going down diagnoally
        just check all rows, cols, diags, and anti diags for streaks
        for checking diags and anti diags, use hashamp
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        diags = defaultdict(list)
        anti_diags = defaultdict(list)
        
        
        for i in range(rows):
            for j in range(cols):
                diags[i-j].append(mat[i][j])
                anti_diags[i+j].append(mat[i][j])
        
        longest_line = 0
        
        #check rows
        for row in mat:
            streak = 0
            for num in row:
                if num == 1:
                    streak += 1
                    longest_line = max(longest_line,streak)
                else:
                    streak = 0
        
        #check cols
        for c in range(cols):
            streak = 0
            for r in range(rows):
                if mat[r][c] == 1:
                    streak += 1
                    longest_line = max(longest_line,streak)
                else:
                    streak = 0
        
        #check diags
        for diag in diags.values():
            streak = 0
            for num in diag:
                if num == 1:
                    streak += 1
                    longest_line = max(longest_line,streak)
                else:
                    streak = 0
        
        #check anti-diags
        for diag in anti_diags.values():
            streak = 0
            for num in diag:
                if num == 1:
                    streak += 1
                    longest_line = max(longest_line,streak)
                else:
                    streak = 0
        return longest_line

#using dp
class Solution:
    def longestLine(self, mat: List[List[int]]) -> int:
        '''
        we can use dp
        make memo for rows, cols, diags, and anti diags
        then update each
        each dp represetns the longest streak going in that direction
        then add 1 to each as we traverse if there is a one
        '''
        rows = defaultdict(int)
        cols = defaultdict(int)
        dds = defaultdict(int)
        ads = defaultdict(int)
        
        longest_line = 0
        M = len(mat)
        N = len(mat[0])
        
        for i in range(M):
            for j in range(N):
                if mat[i][j] == 0:
                    #reset
                    rows[i] = cols[j] = ads[j+i] = dds[j-i] = 0
                else:
                    #update
                    rows[i] += 1
                    cols[j] += 1
                    ads[j + i] += 1
                    dds[j - i] += 1
                    longest_line = max(longest_line, rows[i], cols[j], ads[j+i], dds[j-i])
        
        return longest_line

############################
# 342. Power of Four (Revisited)
# 22AUG22
############################
class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        '''
        recall to find power of two,
        n > 0 and n & (n-1) == 0
        
        if we look at the binary rep for powers of 4, the bits are set at even positions
        if we do bitwise & with (101010....)_2, a power of 4 wouuld give zero
        so we just check is power of 2 and bitwise and with 0xaaaaaaaa 
        '''
        return num > 0 and num & (num-1) == 0 and num & (0xaaaaaaaa) == 0

########################
# 234. Palindrome Linked List (REVISITED)
# 23AUG22
########################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        '''
        if we want to print nodes in reverse recursively
        #funtion to print recursively
        def rec_print_rev(node):
            if node:
                rec_print_rev(node.next)
                print(node.val)
                
        rec_print_rev(head)
        
        notice in the algorithm we keep advancine until we  hit the last node
        where it finally prints
        
        algorithm
            * when given the head node (or another node) referred to a currnode, we first check the rest of the LL
            * if it discovers further down that the LL is not a palindomr, return false
            * otherwise move front pointer by 1 and return true
        '''
        self.front = head
        
        #the function return true whwether or not if the LL is a palindrom from this node
        def rec_check(node):
            if node:
                #first check progress so far
                if not rec_check(node.next):
                    return False
                #check
                if self.front.val != node.val:
                    return False
                #advance
                self.front = self.front.next
            
            return True
        
        return rec_check(head)
            
###################################################
# 549. Binary Tree Longest Consecutive Sequence II (REVISITED)
# 23AUG22
###################################################
#close one
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        '''
        return the length of the longest consecutive path in the tree
        the path and be in the child-parent-child order
        
        i can build the graph from a Tree in O(N) time, this would be an undirected graph
        degenerate the question to:
            in an undirected graph,  find the length of the longest consecutive path
            
        '''
        graph = defaultdict(list)
        
        def dfs(node,parent=None):
            if not node:
                return
            if parent:
                graph[parent].append(node)
                graph[node].append(parent)
            
            dfs(node.left,node)
            dfs(node.right,node)
        
        dfs(root)
        global_seen = set([root])
        q = deque([(root,1)])
        longest_path = 1
        
        while q:
            #to keep track at this level
            local_seen = set()
            #current size of frontier
            N = len(q)
            for _ in range(N):
                curr_node,path_size = q.popleft()
                #maximize
                longest_path = max(longest_path,path_size)
                #neighbor search
                for neigh in graph[curr_node]:
                    #seen updates
                    if neigh not in global_seen:
                        if neigh not in local_seen:
                            #must be increasing
                            if neigh.val == curr_node.val + 1:
                                q.append((neigh,path_size+1))
                                local_seen.add(neigh)
            #global update
            global_seen.update(local_seen)
        
        return longest_path
                                
#using dp
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        '''
        imagine we had a function longest(node)
            which returns an array [inc,dec], where inc is the length of the longest increainsg path below node
            and dec is the length of the longest decreasing path below node
            trivial this is [1,1] for a leaf node (or for any node that matter)
            
        then we can obtain longest path for left child
            longest(node.left)
                if the child's value is one less than the current node, it forms a decreasing sequences with the current node
                so the dec value for the current node is sortes as the left childs dec value + 1 (because we can extend this decreasing sequence)
                but if the left child's value is 1 greater,than this forms in increasing sequence with the current node
                so we update the inc value + 1
            we do the same for the right
                BUT: for obtaining the inc and ec value for the current node, we need to consider the maximum value out of the two values obtained from left and right for both the inc and dec
                
            further, after we've obtained the final update values of inc and dec, we update the lenght of the longst consective path
        
        notes top down, global update solution
        '''
        self.ans = 0
        
        def dp(node):
            if not node:
                return [0,0]
            #alwasy have at least one node in the sequence for a node
            inc = 1
            dec = 1
            
            #left side
            if node.left:
                left = dp(node.left)
                #decreasing from above
                if node.val == node.left.val + 1:
                    dec = left[1] + 1
                #inceasing
                elif node.val == node.left.val - 1:
                    inc = left[0] + 1
            
            #right side
            if node.right:
                right = dp(node.right)
                #decreasing from above
                if node.val == node.right.val + 1:
                    dec = max(dec,right[1] + 1)
                elif node.val == node.right.val - 1:
                    inc = max(inc,right[0]+1)
            
            #global update
            self.ans = max(self.ans, inc+dec-1)
            return [inc,dec]

        
        dp(root)
        return self.ans

########################
# 796. Rotate String
# 16AUG22
########################
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        if i simulate shifting, for all possible shifts
        i can use q to simulate in O(1) time
        '''
        s = list(s)
        q = deque(s)
        N = len(s)
        count = 0
        
        while count < N:
            temp = "".join(q)
            print(temp)

            if temp == goal:
                return True

            q.append(q.popleft())
            count += 1
        
        return False

class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        if we were to concate s with itselft
        s+s, then we just need to ensure goal in in this substring
        '''
        return len(s) == len(goal) and goal in s+s

#rolling hash
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        our goal is to check whether goal is a substring of s + s
        we can use a rolling hash to solve this questions
        idea behind hashing:
            hash(a substinrg of s + s) is uniformly distributed between [0,1,2...mod-1]
            and of hash(substring of s) == hash(goal), it is very liekly to be a match
        
        recall pow(x,y,z) == x**y mod z
        
        first step, get hash of both s and goal then check
        otherwise we need to go through s again update hash and and check if it matches goal
        
        another way would have been to contacte s and then use rolling hash with that

        '''
        mod = 10**9 + 7
        prime = 113 #use large prime
        pinv = pow(prime,mod-2,mod)
        
        hash_goal = 0
        power = 1
        for ch in goal:
            code = ord(ch) - ord('a')
            hash_goal = (hash_goal + power*code) % mod
            power = power*prime % mod
        
        hash_s = 0
        power = 1
        for ch in s:
            code = ord(ch) - ord('a')
            hash_s += (power*code) % mod
            power *= prime % mod
        
        if hash_s == hash_goal and s == goal:
            return True
        
        for i,ch in enumerate(s):
            code = ord(ch) - ord('a')
            hash_s += power*code
            hash_s -= code
            hash_s *= pinv
            hash_s %= mod
            if hash_s == hash_goal and s[i+1:] + s[:i+1] == goal:
                return True
        
        return False

#kmp
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        '''
        using kmp
        remember we have degenerated the problem of finding goal in s+s
        https://web.stanford.edu/class/cs97si/10-string-algorithms.pdf
        https://www.youtube.com/watch?v=JoF0Z7nVSrA&ab_channel=NeetCode
        lps, longest prefix also a suffic, that is not the length of the current prefix
        '''
        if len(s) != len(goal):
            return False
        if len(s) == 0:
            return True
        
        #make strings 1 index
        N = len(s)
        s = ' '+s+s
        goal = ' '+goal

        #caclualte the pi table, i.e longest prefix also a suffic
        pi = [0]*(N+1)
        left = -1
        pi[0] = -1
        
        for right in range(1,N+1):
            while left >= 0 and goal[left+1] != goal[right]:
                left = pi[left] #recurse
                #pi[i] answers the question, what it the longest prefix in goal[i] that is also a suffix
            left += 1
            pi[right] = left
            
        #matching part
        j = 0
        for i in range(1,(2*N)+1):
            #try to match, and when we can't we need to advance our pointer
            while j >= 0 and goal[j+1] != s[i]:
                j = pi[j]
            j += 1
            if j == N:
                return True
        return False
        

#############################
# 800. Similar RGB Color 
# 16AUG22
#############################
class Solution:
    def similarRGB(self, color: str) -> str:
        '''
        this is hex
        can be [0-9,a-f]
        which is base 16
        
        instead of minimzing the squred differnces, we can minimuze the absolute value, 
        this still gives us the same orerder
        
        first find the length 2 pairs [00,11,22....ff]
        then break apart color in 3 pairs of two
        caluclate the absolutdifference, and get the minimum
        '''
        
        #function to get closest for a len 2 code
        def closest(code):
            possible = ['00', '11', '22', '33', '44', '55', '66', '77', '88', '99', 'aa', 'bb', 'cc', 'dd', 'ee', 'ff']
            sims = []
            
            #get all possible scores
            for cand in possible:
                score = abs(int(code,16) - int(cand,16))
                sims.append((score,cand))
                
            #find closest one, through sorting
            sims.sort(key = lambda x: x[0])
            return sims[0][1]
        
        ans = "#"
        for i in range(1,len(color),2):
            ans+= closest(color[i:i+2])
        
        return ans

##########################################
# 1570. Dot Product of Two Sparse Vectors
# 17AUG22
##########################################
#lazy implementation works
class SparseVector:
    def __init__(self, nums: List[int]):
        self.nums = nums
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        N = len(self.nums)
        v1 = self.nums
        v2 = vec.nums
        ans = 0
        for a,b in zip(v1,v2):
            ans += a*b
        return ans
            
# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)

#still slow???
class SparseVector:
    def __init__(self, nums: List[int]):
        self.sparse = {}
        self.size = len(nums)
        for i,num in enumerate(nums):
            if num:
                self.sparse[i] = num
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        N = self.size
        v1 = self.sparse
        v2 = vec.sparse
        ans = 0
        #just take one, it doesn't matter anyway if there is a zero
        for i, num in v1.items():
            if i in v2:
                ans += num*v2[i]
        
        return ans
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)

class SparseVector:
    def __init__(self, nums: List[int]):
        '''
        we can repsresent the array is tuple pairs (index,num)
        then we can use two pointers to pass them pait
        also note the time complexity of the hashing function as well
        and priotrize looping over the sparse vector who's length of non zero values is shorter
        
        '''
        self.pairs = []
        for i,num in enumerate(nums):
            if num != 0:
                self.pairs.append((i,num))
        

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        ans = 0
        i = 0
        j = 0
        #i into first, j into second
        while i < len(self.pairs) and j < len(vec.pairs):
            if self.pairs[i][0] == vec.pairs[j][0]:
                ans += self.pairs[i][1]*vec.pairs[j][1]
                i += 1
                j += 1
            #always advance the lagging pointer to catch up
            elif self.pairs[i][0] < vec.pairs[j][0]:
                i += 1
            else:
                j += 1
        
        return ans
        

# Your SparseVector object will be instantiated and called as such:
# v1 = SparseVector(nums1)
# v2 = SparseVector(nums2)
# ans = v1.dotProduct(v2)