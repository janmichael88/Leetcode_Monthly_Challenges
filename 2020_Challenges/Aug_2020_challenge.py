#######################################
#DETECT CAPITAL 08/01/2020
#######################################
class Solution(object):
    def detectCapitalUse(self, word):
        """
        :type word: str
        :rtype: bool
        """
        '''
        correct uppercase is when all are upper or when the only the leftmost char is upper
        '''
        count_uppers = 0
        word_chars = list(word)
        if len(word_chars) == 1:
            return True
        for i in range(0,len(word_chars)):
            if word_chars[i].isupper():
                count_uppers += 1
        if (count_uppers == len(word_chars)) or (word_chars[0].isupper() and count_uppers == 1):
            return True
        elif count_uppers == 0:
            return True
        else:
            return False
    ##or a one liner
    def detectCapitalUse(self,word):
    	return word.isupper() or word.islower() or (word[:1].isupper() and word[1:].islower())


###############################
#Cousins in Binary Tree, this is from May, but need to add it somewhere
###############################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        '''
        nodes are cousins if they are at the same depth but of different parents
        we can tell if they are of the same parent if node.left(parent) for x is 
        this is just a modified version of BFS - access level by level
        fix node to quue, pop, add children, while q do work
        use counter to demarcate generations
        ask two questions if node is a sibling
        and if node is a child of someone on generation
        '''
        #create queue
        q = []
        q.append(root)
        genpop = 1
        while q:
            #start of new generation when pop is 0
            if genpop == 0:
                genpop = len(q)
            node = q.pop()
            #get rid of pop counter
            genpop -= 1
            #init target to - 1
            target = -1
            if node.left != None:
                if node.left.val == x:
                    target = y
                elif node.left.val == y:
                    target = x
                else:
                    q.append(node.left)
            if node.right != None:
                #do sibling check
                if node.right.val == target:
                    return False
                if node.right.val == x:
                    target = y
                elif node.right.val == y:
                    target = x
                else:
                    q.append(node.right)
                    
            #now if we havent found our target, or when target is still -1, we are seaching for cuzn
            if target != -1:
                #now seach for in generatour
                while genpop > 0:
                    node = q.pop()
                    genpop -= 1
                    if (node.left != None) and (node.left.val == target):
                        return True
                    if (node.right != None) and (node.right.val == target):
                        return True
                return False
        return False


       

###################################################
#Design Hash Set 08/02/2020
##################################################
#brute force
class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.array = [None for _ in range(1000000)]

    def add(self, key: int) -> None:
        if not self.contains(key):
            self.array[key] = True

    def remove(self, key: int) -> None:
        if self.contains(key):
            self.array[key] = None

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        return self.array[key]
        


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)

class MyHashSet:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        #using buckets
        self.bucketsize = 15000
        self.buckets = [[] for _ in range(self.bucketsize)]
    
    def hash_function(self,key):
        return key % self.bucketsize
        

    def add(self, key: int) -> None:
        i = self.hash_function(key)
        if not key in self.buckets[i]:
            self.buckets[i].append(key)

    def remove(self, key: int) -> None:
        i = self.hash_function(key)
        if key in self.buckets[i]:
            self.buckets[i].remove(key)

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        i = self.hash_function(key)
        if key in self.buckets[i]:
            return True
        return False
        


# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)

        


#################################
#Valid Palindrome 08/03/2020
################################

class Solution(object):
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        #just check if the reverse is the same as the foward
        #first make into lower and remove speical charcters
        s = s.lower()
        s = "".join(char for char in s if char.isalnum())
        beg,end = 0,len(s)-1
        while beg < end:
            if s[beg] != s[end]:
                return False
            beg += 1
            end -= 1
        
        return True

#####################################
#  Maximum Sum Circular Subarray, from Mays Challegene
#########################################
class Solution(object):
    def maxSubarraySumCircular(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        #notes circular aray mean A[i] = A[i + len(C)]
        #A is the array
        #there must exist a subarray C that has the max sum
        #they also say there is no overlap
        '''
        exaine the subarray
        [2,2,2,-1,10,10]
        if circular we can imagine it being 
        [2,2,2,-1,10,10][2,2,2,-1,10,10]
        so not its from 10on the left to the third 2 on the right
        cumsum from end of the array, store each cum sum from len(array) to 0
        then find the the largest sums on the left and right sides, then add them
        '''
        #case 1, kadanes
        N = len(A)
        cur_sum, max_sum = float('-inf'), float('-inf')
        for n in A:
            cur_sum = max(cur_sum,0) + n
            max_sum = max(max_sum,cur_sum)
        
        #case 2
        left_sum = [0]*N
        left_max = [0]*N
        right_sum = [0]*N
        
        #find cumsum on the left,remeber we need two passes
        cum_sum_left = 0
        #go backwards, first pass
        for i in range(N-1,-1,-1):
            cum_sum_left += A[i]
            left_sum[i] = cum_sum_left
        #second pass
        cur_max_left = float('-inf')
        for i in range(N-1,-1,-1):
            cur_max_left = max(cur_max_left, left_sum[i])
            left_max[i] = cur_max_left
            
        #RIGHTSIDE
        cur_sum_right = 0
        for i in range(N):
            cur_sum_right += A[i]
            right_sum[i] = cur_sum_right
        
        cur_max_2,max_sum_2 = float('-inf'),float('-inf')
        for i in range(N-1):
            cur_max_2 = right_sum[i] + left_max[i+1]
            max_sum_2 = max(cur_max_2,max_sum_2)
        return max(max_sum,max_sum_2)


#################################
#isPowerofFour 08/04/2020
##################################
class Solution(object):
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        '''
        if a number is a power of 4, i can write the name as 4^n
        the naive way would be to just keep checking all powers of four
        and return true if it hits, if we've passed it break the loop and return false
        '''
        #warm up with naive
        i = 0
        while num >= 4**i:
            if 4**i == num:
                return True
            i += 1
        return False

    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        #just take the log
        return (num > 0) and (math.log(num) / math.log(4)).is_integer()

import math
class Solution(object):
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        '''
        now try do it without any loops or recursion
        couldn't you just check that there is only a single bit in its binary rep?
        bits??
        2 is '0010'
        4 is '0100'
        16 is '10000'
        i can rewrite 4^n as (2^2)^n which can be written as 2^(2n)
        1,4,16,64
              1
            100
          10000
        1000000
        they ones at an odd position
        check if there is 1 zero in binary
        do num & num-1
        
        '''
        if num <=0:
            return False
        #now check if there is only one bit on the binrep
        if num & num - 1 !=0:
            return False
        #now fine position at 1
        b =bin(num)[::-1]
        p = b.index("1")
        return p % 2 == 0 #since power at four is at even indices with zero indexing


########################################
#Contiguous Subarray from may
#########################################
class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #brute force is to examine each subarray, count ones and zeros, update a maxlen
        maxlen = 0
        for i in range(0,len(nums)):
            zeros,ones = 0,0
            for j in range(i, len(nums)):
                if nums[j] == 0:
                    zeros += 1
                else:
                    ones += 1
                if zeros == ones:
                    maxlen = max(maxlen,j-i+1)
                    
        return maxlen


class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        create hashmap storing unique values of counts
        so it will be like <count>:<index>
        
        '''
        counts = dict()
        maxlen,count = 0,0
        counts[0] = -1
        for i in range(0,len(nums)):
            if nums[i] == 0:
                count -= 1
            else:
                count += 1
            
            if count in set(counts.keys()):
                maxlen = max(maxlen, i - counts[count])
            else:
                counts[count] = i
        return maxlen

############################################################
#Add and Search Word - Data structure design 08/05/2020
###########################################################
#recursive solution from Tim
class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = dict()

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        p = self.root
        #if i can't find it
        if not self.search(word):
            for i in range(0,len(word)):
                if word[i] not in p:
                    p[word[i]] = dict()
                p = p[word[i]]
            
            p['#'] = dict()
            
    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        p = self.root
        
        #use a recursive function to handle the wildcard symbol
        def rec(start,p):
            #base condition
            #think about what conditions to recurse
            #what conditions to end the recursion
            if start >= len(word):
                if "#" in p:
                    return True
                else:
                    return False
            #so in the case that char is a period, we need to keep searching each dictionary
            #recurse only on period and when there is a match
            if word[start] == '.':
                for k,v in p.items():
                    if rec(start+1,v):
                        return True
                return False
            elif word[start] in p:
                if rec(start+1,p[word[start]]):
                    return True
            else:
                return False
        return rec(0,p)
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)


##############################################
#Find All Duplicates in an Array 08/06/2020
##############################################
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #warm up,count and then return numbers with two occurrences
        counts = dict()
        results = []
        for num in nums:
            if num not in counts:
                counts[num] = 1
            else:
                counts[num] += 1
        
        for k,v in counts.items():
            if v == 2:
                results.append(k)
                
        return results

class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #now try doing it in O(n) time and O(1) space
        #the idea is to swap each number to its place and go down the place
        #when you try to swap a number to its place and the number is already there, it means you've done it before and can count that as a dupliate
        #stop when this si found (while loop)
        #do this for the whole nums (for loop)
        #finally return the array at the numers with whtr wrong numbers and return the places
        #https://leetcode.com/problems/find-all-duplicates-in-an-array/discuss/775738/Python-2-solutions-with-O(n)-timeO(1)-space-explained
        
        for i in range(0,len(nums)):
            while i != nums[i] - 1 and nums[i] != nums[nums[i]-1]:
                #swap 
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        return [nums[it] for it in range(len(nums)) if it != nums[it]-1]

###############################################
#  Vertical Order Traversal of a Binary Tree
###############################################

#works on some of the cases 11/30

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def verticalTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        '''
        traverse tree and add in xy points
        add each point with its correspoding xy. so like..[node.val,[x,y]]
        this will be a list of lists
        then reorder the list
        '''
        node_coords = []
        results = []
        def dfs(root,left,right):
            if root:
                position = root.val
                node_coords.append([position,[left,right]])
                dfs(root.left,left-1,right-1)
                dfs(root.right,left+1,right-1)
        dfs(root,0,0)
        #now sort
        node_coords = sorted(node_coords, key = lambda x:(x[1][0],x[1][1]))
        #now pass once to find the uniqe xcoords
        un_xs = set()
        for i in range(0,len(node_coords)):
            un_xs.add(node_coords[i][1][0])
        un_xs = sorted(un_xs, reverse=False)
        #pass one moretime creating the groups
        for x in un_xs:
            group = []
            for i in range(0,len(node_coords)):
                if node_coords[i][1][0] == x:
                    group.append(node_coords[i][0])
            group = sorted(group,reverse=False)        
            results.append(group)
            
        return results

from collections import defaultdict
from heapq import heappush, heappop

class Solution:
    def verticalTraversal(self, root: TreeNode) -> List[List[int]]:
        maf = defaultdict(list)
        
        def traverse(node, x, y):
            if not node: 
                return
            
            maf[x].append((y, node.val))
            traverse(node.left, x - 1, y + 1)
            traverse(node.right, x + 1, y + 1)
        
        traverse(root, 0, 0)
        heap, ans = [], []
        
        for x, lst in maf.items():
            heappush(heap, (x, sorted(lst)))
        
        while heap:
            ans.append([v for _, v in heappop(heap)[1]])
        
        return ans

#####################################
#Path Sum III 08/08/2020
#####################################

#nice first try

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        '''
        im thinking a recurive solution would solve this
        bottom up?
        start from the simplest subtree, and check sum
        is sum == sum increment path by 1
        if it doesn't go up into to the tree
        dfs
        well you would just dfs on the node and see if the paths sum to to them
        '''
        paths = 0
        
        def dfs_2(node,path_sum):
            if node:
                path_sum += node.val
                dfs_2(node.left,path_sum)
                dfs_2(node.right,path_sum)
            
            
        
        def dfs(node):
            if not node:
                return
            elif not node.left and not node.right:
                if node.val == sum:
                    paths += 1
            else:
                left = node.left
                right = node.right


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        '''
        im thinking a recurive solution would solve this
        bottom up?
        start from the simplest subtree, and check sum
        is sum == sum increment path by 1
        if it doesn't go up into to the tree
        dfs
        well you would just dfs on the node and see if the paths sum to to them
        '''
        self.total = 0
        def helper(node,cur):
            if not node:
                return
            helper(node.left,cur+node.val)
            helper(node.right,cur+node.val)
            if cur + node.val == sum:
                self.total += 1
        
        def dfs(node):
            if not node:
                return
            helper(node,0)
            dfs(node.left)
            dfs(node.right)
            
        dfs(root)
        return self.total
from collections import defaultdict
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        '''
        im thinking a recurive solution would solve this
        bottom up?
        start from the simplest subtree, and check sum
        is sum == sum increment path by 1
        if it doesn't go up into to the tree
        dfs
        well you would just dfs on the node and see if the paths sum to to them
        '''
        self.total = 0
        self.lookup = defaultdict(int)
        self.lookup[sum] = 1

        def dfs(node,root_sum):
            if not node:
                return
            root_sum += node.val
            #now update total
            self.total += self.lookup[root_sum]
            #update dict
            self.lookup[root_sum+sum] += 1
            dfs(node.left,root_sum)
            dfs(node.right,root_sum)
            #bracktrack
            self.lookup[root_sum+sum] -= 1
            
        dfs(root,0)
        return self.total
       
######################################
#Rotting Oranges 08/09/2020
#####################################

#well dfs was not the way to go on this...
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        '''
        similar to capturing islands no, more like flood fill
        take a look at that again
        dfs approach
        but how do you find the minimum number of minutes
        greedy approach would be just to spread in all four directions
        so flood fill on the first pass updating the array
        second pass, scan the array seeing all rotten, if all rotten return the number of times dfs was called
        if not all rotten, return - 1
        '''
        #edge case
        if len(grid) == 1 and len(grid[0]==1):
            if grid[0] == 2:
                return 1
            else:
                return -1
        
        #count 
        ones = 0
        twos = 0 
        
        rows, cols = len(grid), len(grid[0])
        mins_passed = 0
        def dfs(r,c):
            #bound condition check 
            if (r<0) or (r>=rows) or (c<0) or (c>= cols):
                return
            #now check all four direcitons
            elif grid[r][c] == 2:
                #recurse


class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        '''
        graph traversal problem, bfs
        know state of board, nums fresh and rotten
        put rotten oranges in q at = 0
        while q is not empty:
            pop
            look at neighbors and put into q
            t += 1 only when we have finished infecting neighbords
            and pop len of curent q
        return -1 if we werent able to infect all the oranges on the board
        keep decrementing count of fresh
        '''
        rotten = []
        r,c,fresh,t = len(grid), len(grid[0]),0,0
        for i in range(r):
            for j in range(c):
                if grid[i][j] == 2:
                    rotten.append([i,j])
                elif grid[i][j] == 1:
                    fresh += 1
        #do bfs
        while len(rotten) > 0:
            num = len(rotten) #keep track of q lengh every time
            for i in range(num):
                x,y = rotten[0] #taking the first rotten oragne in q
                rotten.pop(0)
                if (x>0) and (grid[x-1][y]) == 1: #to the left
                    grid[x-1][y] = 2
                    fresh -= 1
                    rotten.append([x-1,y])
                if (y>0) and (grid[x][y-1]) == 1: #below
                    grid[x][y-1] = 2
                    fresh -= 1
                    rotten.append([x,y-1])
                if (x<r-1) and grid[x+1][y] == 1: #to the right
                    grid[x+1][y] = 2
                    fresh -= 1
                    rotten.append([x+1,y])
                if (y<c-1) and grid[x][y+1] == 1:
                    grid[x][y+1] = 2
                    fresh -= 1
                    rotten.append([x,y+1])
            if len(rotten) > 0:
                t+=1
        
        return t if (fresh == 0) else -1


#unoptimized, not using q
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        #count fresh oranges
        fresh = 0 
        M = len(grid)
        N = len(grid[0])
        
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 1:
                    fresh += 1
        
        minutes = 0
        prev = fresh #keep to see if we can't decrease the number of fresh oranges
        while True:
            cp = copy.deepcopy(grid)
            #scan grid
            for i in range(M):
                for j in range(N):
                    #if rotten, update adjacebt
                    #make changes to copy
                    if grid[i][j] == 2:
                        #check
                        for r,c in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                            #check if in bounds
                            if 0<=r<M and 0<=c<N and cp[r][c] == 1:
                                #update temp
                                cp[r][c] = 2
                                fresh -= 1
                                
                                
            #if we haven't been able to make changes, before reassigning cp back to gridc
            if prev ==fresh: 
                break
            minutes += 1
            #updates to grid
            grid = cp
            #updates to fresh
            prev = fresh
            
        if fresh > 0:
            return -1 #still rotten left
        else:
            return minutes

from collections import deque
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        #count fresh oranges
        fresh = 0 
        q = deque()
        M = len(grid)
        N = len(grid[0])
        
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 2:
                    q.append((i,j,0))
                if grid[i][j] == 1:
                    fresh += 1
        
        minutes = 0
        while q:
            i,j,minutes = q.popleft()
            #check
            if grid[i][j] == 2:
                for r,c in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
                    #check if in bounds
                    if 0<=r<M and 0<=c<N and grid[r][c] == 1:
                        #update temp
                        grid[r][c] = 2
                        fresh -= 1
                        q.append((r,c,minutes+1))

        if fresh > 0:
            return -1
        else:
            return minutes

#######################################
#Excel Sheet Column Number 08/10/20
#######################################
class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        1,2,3....26
        A,B,C,...Z
        AB is just 26 + 2
        ZY is just 26*26 + 25
        i could make a mapper
        A is 1, Z is 26
        ZA is 26*26 + 1 ZZ is 26*26+26
        so it just base 26
        i can write A as
        1*26^0 is 1
        B is
        2*26^ which is 2
        AA is 1*26^1 + 1*26^0 = 27
        ord(char) offset by 64
        '''
        number = 0
        for i in range(0,len(s)):
            number += (ord(s[i])-64)*26**(len(s)-1-i)
        return number


###############################
# hIndex 08/11/2020
##############################
#bad solution but it works

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        '''
        even without sorting, the naive way would be to check each index from 0 to N
        and see if we have at least that many number of papers
        then break, we would have to scan the whole array every time
        if i sort
        3,0,6,1,5
        0,1,3,6,5
        the hindex will always be a value in the list of papers
        what if i just counted the number of papers?
        just try this first
        '''
        if not citations:
            return 0
        elif len(citations) == 1:
            if citations[0] == 0:
                return 0
            else:
                return 1
        papers = len(citations)
        counts = dict()
        for i in range(papers+1):
            papsAtLeasti = 0
            for c in citations:
                if c >= i:
                    papsAtLeasti += 1
            counts[i] = papsAtLeasti
        #now traverse the dictionary
        for k,v in counts.items():
            if v < k:
                return k - 1
            elif v == k:
                return k

#another way
N = len(citations)
citations.sort()
for i,v in enumerate(citations):
	if N - i <= v:
		return N-i
	return 0

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        '''
        [3,0,6,1,5]
        [0,1,3,5,6]
        [5,4,3,2,1]
        #this is just nlogn + n, but we can do it in N time without sorting
        what if we create an array of zeros holding the counts at each index
        [0,0,0,0,0,0]
        [1,1,0,1,0,2]
        [5,4,3,3,2,2]
         0 1 2 3 4 5
        '''
        N = len(citations)
        tmp = [0 for _ in range(N+1)]
        for i,v in enumerate(citations):
            if v > N:
                #increment last value of h-index
                tmp[N] += 1
            else:
                tmp[v] += 1
        #go backwards and get sum
        total = 0
        for i in range(N,-1,-1):
            total += tmp[i]
            if total >= i:
                return i
** after sorting
if not citations: return 0
citations.sort()
        # [0,0,0,0,0]
        #  5 4 3 2 1  N-index
        
        # not return index, return h-index
        N = len(citations)
        l = 0
        r = N-1
        
        while (l<r):
            mid = (l+r)//2
            if citations[mid] >= N-mid:
                r = mid
            else:
                l = mid+1
        
        if citations[l] == 0: return 0
        return N-l

############################
#Pascals triangle II 08/12/20
############################
class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        '''
        pacals triangle just start
         1              1
        1 1             2
       1 2 1            3
      1 3 3 1           4
     1 4 6 4 1,         5  which are juse the binomial coefficients (a+b)^n
    1 5 10 10 5 1       6
    dp solution buliding up
        
        
        '''
        if rowIndex == 0:
            return [1]
        if rowIndex == 1:
            return [1,1]
        dp = [[] for i in range(rowIndex+1)]
        dp[0] = [1]
        dp[1] = [1,1]
        
        if rowIndex < 2:
            return dp[rowIndex]
        for i in range(2,rowIndex+1):
            #sliding window on the previous row
            temp = []
            temp.append(1)
            for j in range(0,len(dp[i-1])-1):
                temp.append(dp[i-1][j]+dp[i-1][j+1])
            temp.append(1)
            dp[i] = temp
        return dp[rowIndex]

class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        '''
        pacals triangle just start
         1              1
        1 1             2
       1 2 1            3
      1 3 3 1           4
     1 4 6 4 1,         5  which are juse the binomial coefficients (a+b)^n
    1 5 10 10 5 1       6
    dp solution buliding up
        
        
        '''
        output = [1]
        for i in range(1,rowIndex+1):
            output.append(1)
            for j in range(len(output)-2,0,-1):
                output[j] += output[j-1]
        return output

output = []
for i in range(0,rowIndex+1):
    output.append(math.factorial(rowIndex) /(math.factorial(i)*math.factorial(rowIndex-i)))
return output



#####################################
#Iterator for Combindation 08/13/2020
#####################################
class CombinationIterator(object):

    def __init__(self, characters, combinationLength):
        """
        :type characters: str
        :type combinationLength: int
        """
        #recursive call to create combindations
        #recurse only on non containing elements
        #iterative solution won't work because we have combindationLength arg
        def rec_comb(characters, combinationLength,prev = []):
            #base case
            if len(prev) == combinationLength:
                return ["".join(prev)]
            combs = []
            for i,v in enumerate(characters):
                prev_extended = copy.copy(prev)
                prev_extended.append(v)
                combs += rec_comb(characters[i+1:],combinationLength,prev_extended)
            return combs
        self.combs = rec_comb(characters, combinationLength,[])
            
            
    def next(self):
        """
        :rtype: str
        """
        if self.combs:
            return self.combs.pop(0)

    def hasNext(self):
        """
        :rtype: bool
        """
        #return true if stack
        if self.combs:
            return True

#another way iteratively
from collections import deque
class CombinationIterator(object):

    def __init__(self, characters, combinationLength):
        """
        :type characters: str
        :type combinationLength: int
        """
        self.combs = []
        
        q = deque()
        q.append(("",0))
        N = len(characters)
        
        while q:
            temp,cur = q.popleft()
            if len(temp) == combinationLength:
                self.combs.append(temp)
                continue
            for i in range(cur,N):
                q.append((temp+characters[i],i+1))
            
    def next(self):
        """
        :rtype: str
        """
        if self.combs:
            return self.combs.pop(0)

    def hasNext(self):
        """
        :rtype: bool
        """
        #return true if stack
        if self.combs:
            return True

class CombinationIterator(object):

    def __init__(self, characters, combinationLength):
        """
        :type characters: str
        :type combinationLength: int
        """
        self.q = []
        def rec_comb(start,length,txt):
            if length == 0:
                self.q.append(txt)
                return
            for i in range(start, len(characters)-length+1):
                rec_comb(i+1,length-1, txt + characters[i])
        rec_comb(0,combinationLength,"")

    def next(self):
        """
        :rtype: str
        """
        if self.q:
            return self.q.pop(0)
        

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.q:
            return True


################################
#Longest Palindrom 08/14/2020
#################################

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        generating all permutations of length n is definitely not the way to go
        given how large the string can be, an n squared solution might be the way to go
        my initial thought would be to start at the largest length, and see if any of these permuations is palindrome, if not, decrement down and try with length - 1
        what if i count the characters?
        then greedily build up the largest palindrome
        then i would just use up all the even counts first, then just add one, because it doesnt matter what the center point is if odd
        '''
        counts = dict()
        for char in s:
            if char in counts:
                counts[char] += 1
            else:
                counts[char] = 1
        num_evens,count_evens = 0,0
        num_odds,count_odds = 0,0
        for k,v in counts.items():
            if v % 2 == 0:
                num_evens += 1
                count_evens += v
            else:
                num_odds += 1
                count_odds += v
                
        if num_odds == 0:
            return count_evens
        if 
        return count_evens + (count_odds // num_odds)
        '''
        print num_evens,count_evens,num_odds,count_odds
        '''


class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        generating all permutations of length n is definitely not the way to go
        given how large the string can be, an n squared solution might be the way to go
        my initial thought would be to start at the largest length, and see if any of these permuations is palindrome, if not, decrement down and try with length - 1
        what if i count the characters?
        then greedily build up the largest palindrome
        then i would just use up all the even counts first, then just add one, because it doesnt matter what the center point is if odd
        '''
        counts = dict()
        for char in s:
            if char in counts:
                counts[char] += 1
            else:
                counts[char] = 1
        ans = 0
        for k,v in counts.items():
            #this part is important for captring, if we have a letter v times, it can be captures with another v // 2 * 2 times
            #example if we have 'aaaaa' it can be paired with 'aaaa'
            ans += v / 2*2
            if ans % 2 == 0 and v % 2 == 1:
                ans += 1
        return ans

    def longestPalindrome(self, s):
    	#another way but more explicltiy
    	counts = dict()
        for char in s:
            if char in counts:
                counts[char] += 1
            else:
                counts[char] = 1
      	ans = 0
      	odd_found = False
        for k,v in counts.items():
        	if odd_found:
        		if v > 1:
        			if v % 2 == 0:
        				ans += v
        			else:
        				ans += v-1
        	else:
        		if v % 2 == 0:
        			ans += v
        		else:
        			ans+=v
        			odd_found = True

        return ans


######################################
#Non-overlapping Intervals 08/15/2020
######################################
#almost got it, 10/18 solutions 
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        creating capture interval variable
        update this as you traverse the array
        if the and of the next elements elements are in capture, increment output
        '''
        if len(intervals) == 0:
            return 0
        
        intervals = sorted(intervals, key = lambda x:x[1])
 
        beg = intervals[0][0]
        end = intervals[0][1]
        output = 0
        for i in range(1,len(intervals)):
            if intervals[i][0] < beg:
                beg = intervals[i][0]
            if intervals[i][1] > end:
                end = intervals[i][1]
            else:
                output += 1
        return output

class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/non-overlapping-intervals/discuss/792805/Python-sorting-O(nlogn)-solution-w-explanation-beats-94
        reframe the problem, instead of asking to remove the minimum number of elements
        maximize the number of overlaping intervals
        sort on the starting interval
        now examine the interval
        [[1,2],[1,3],[2,3],[3,4]]
        compare [1,2] and [1,3] there is overlap so update interval and keep the smaller interval ending 
        now [1,2] and [2,3] no overlap
        now [2,3] and [3,4] no overlap
        we can take three of the intervals,since we took three it means we dropped 1
        '''
        if not intervals:
            return 0
        #sort
        intervals.sort()
        end = intervals[0][0]
        num_overlap = 0
        for interval in intervals:
            if end > interval[0]:
                end = min(end,interval[1])
            else:
                num_overlap += 1
                end = interval[1]
        return len(intervals) - num_overlap

class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        [[1,2],[1,3],[2,3],[3,4]]
        compare start and previous end, updating each and incrementing output as we go along
        '''
        if not intervals:
            return 0
        intervals.sort(key = lambda x: x[1])
        max_end, output = float('-inf'),0
        for start,end in intervals:
            if start >= max_end:
                max_end = end
            else:
                output += 1
        return output


#######################################
#Best Time to Buy and Sell Stock III
######################################

#using dp array
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        first think about 1 transactino problem
        where is the min and where is the max
        https://www.youtube.com/watch?v=Pw6lrYANjz4
        well think about 1 transction,use the example array
        5  11  3  50  60  90
        maxsofar = -5
        profit = 0

        i = 1:
        	maxsofar = 6
        	profit = 
        '''
        if not prices:
            return 0
        #array will hold the max profit obtained for a certain number of transactions
        #keep updating when a new max is create
        dp = [0 for _ in range(len(prices))]
        for t in range(1,2+1):
            pos = -prices[0]
            profit = 0
            for i in range(1,len(prices)):
                pos = max(pos,dp[i]-prices[i]) #if buying or not buying
                profit = max(profit,pos+prices[i]) #if selling or not selling
                #update 
                dp[i] = profit
        
        return dp[-1]

#using state representations
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        using state representations as one pass
        firstbuy, secondbuy, firstprofit, secondprofit
        you want the firstbuy to be the minimum of it currentvalent and price[i]
        you want firstprofit to be the max of its currenvt value and price[i]-firstbuy
        you want seconduy to be the mini of current value and price[i] - firstprofit
        you want secondprofit to tbe the max of its currentvalue and price[i] - secondbuy
        5  11  3  50  60  90
        (firstbuy,secondbuy,firstprofit,secondprofit)
        for 5
        (5,0,0,5)
        
        '''
        firstBuy,secondBuy, firstProfit, secondProfit = float('inf'),float('inf'),0,0
        for p in prices:
            firstBuy = min(firstBuy,p)
            firstProfit = max(firstProfit, p - firstBuy)
            secondBuy = min(secondBuy, p - firstProfit)
            secondProfit = max(secondProfit, p - secondBuy)
        return secondProfit
 
#using O(nk) time and O(nk) space, from algo expert
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        first think about 1 transactino problem
        where is the min and where is the max
        https://www.youtube.com/watch?v=Pw6lrYANjz4
        example dp array
        X   5  11  3  50  60  90
        0   0   0  0   0   0   0
        1   0   6  6  47  57  87      
        2   0   6  6  53  63  93
        #with one transaction it becomes dp[i+1] = max()
        #two choices, sell at the ith day, meanin we had bought before i
        #or we don't sell, and the profit is just the i-1 day
        #carry over max profit from the i-1 day or at the current day
        '''
        if not prices:
            return 0
        #array will hold the max profit obtained for a certain number of transactions
        #keep updating when a new max is create, storing max for the previous to avoid uncesssary calcs
        profits = [[0 for _ in prices] for _ in range(2+1) ]
        for t in range(1,2+1):
            #store max profit from 0 to t
            maxsofar = float('-inf')
            for d in range(1,len(prices)):
                #update massofr far
                maxsofar = max(maxsofar,profits[t-1][d-1]-prices[d-1])
                #update dp
                profits[t][d] = max(profits[t][d-1], maxsofar + prices[d])
                
        return profits[-1][-1]


#########################################
#Distribute Candies to People
#########################################
class Solution(object):
    def distributeCandies(self, candies, num_people):
        """
        :type candies: int
        :type num_people: int
        :rtype: List[int]
        """
        '''
        so we have 7 candies and 2 people
        first case is keep giving the nth person n candies, if one left, the last person get's the remining
        only add when candies remaining is 1
        '''
        passes = 0
        people = [0 for _ in range(num_people)]
        pointer = 0
        while candies > 0:
            if candies == 1:
                people[pointer] += pointer
                candies -= pointer
            elif pointer == len(people)-2:
                pointer = 0
                passes += 1
            
            elif passes > 0:
                people[pointer] += num_people + pointer + 1
                pointer += 1
                candies -= pointer
            else:
                people[pointer] += pointer
                candies -= pointer
                pointer += 1
            
        return people

class Solution(object):
    def distributeCandies(self, candies, num_people):
        """
        :type candies: int
        :type num_people: int
        :rtype: List[int]
        """
        '''
        so we have 7 candies and 2 people
        first case is keep giving the nth person n candies, if one left, the last person get's the remining
        only add when candies remaining is 1
        '''
        results = [0]*num_people
        c = 0
        while candies > 0:
            #candies may not be enough on last round, therefore get the minmum of candies and p
            results[c % num_people] += min(candies,c+1)
            #add one more for the next person
            c += 1
            #update candies
            candies -= c
            
        return results

class Solution(object):
    def distributeCandies(self, candies, num_people):
        """
        :type candies: int
        :type num_people: int
        :rtype: List[int]
        """
        '''
        straight forward brute force approach
        '''
        output = [0]*num_people
        #init candy to allocate
        candy = 0
        while candies > 0:
            for i in range(num_people):
                candy += 1
                output[i] += min(candy,candies)
                candies -= candy
                if candies < 0:
                    break
        return output

#very good binary search solution
class Solution:
    def distributeCandies(self, candies, num_people):
       '''
        Idea: Round number k (starting from 1)
              -> give away
              (k-1)*n+1 + (k-1)*n+2 + ... + (k-1)*n + n = 
              (k-1)*n^2 + n*(n+1)/2 candies
              
        Assume we have completed K full rounds, then K is the largest integer >= 0 with
        
        K*n*(n+1)/2 + K * (K-1)/2 * n^2 <= candies 
        
        Find K by binary search and then simulate the last round.
        
        The person at index i gets
    
        0*n+i+1 + ... + (K-1)*n+i+1 = K*(i+1) + n*K*(K-1)/2 
        
        candies from rounds 1 to K, plus everything they get on their
        last round.
        
        Important: Allow for the fact that we may not complete a single round.

        REVIEW
		'''
		
        lo, hi = 0, candies
        K = 0
        while lo <= hi:
            k = (lo + hi)//2
            if k*(num_people*(num_people+1))//2 + (k*(k-1))//2 * num_people**2 <= candies:
                K = k
                lo = k + 1
            else:
                hi = k - 1
        result = [(i+1)*K+num_people*(K*(K-1))//2 for i in range(num_people)]
        candies -= sum(result)
        for i in range(num_people):
            add = min(candies, K * num_people + i + 1)
            result[i] += add
            candies -= add
            if candies == 0:
                break
        return result  


from math import floor
class Solution(object):
    def distributeCandies(self, candies, num_people):
        """
        :type candies: int
        :type num_people: int
        :rtype: List[int]
        """
        '''
        1  2  3  4  5
        6  7  8  9  10
        the second row is just
        (1+5) (2+5) (3+5) (4+5) (5+5)
        the sum for the L'th round is just
        \frac{k(k+1)}{2} + (L-1)k^2
        we need this for L rounds, so it just becomes
        (\frac{k(k+1)}{2} + (L-1)k^2)L <=candies
        solve for L using binary search
        simulate the last round, and if there are remaining they go to the last person
        '''
        lo,hi = 0, candies
        L = 0
        while lo <= hi:
            l = lo + (hi-lo) // 2
            first_term = (num_people*(num_people + 1)) // 2
            second_term = (l-1)*num_people**2
            expression = l*(first_term+second_term)
            if expression <= candies:
                L = l
                lo = l+1
            else:
                hi = l -1
        #add candies using last round
        result = [(i+1)*L+num_people*(L*(L-1))//2 for i in range(num_people)]
        candies -= sum(result)
        
        #add in remaining
        for i in range(num_people):
            add = min(candies, L*num_people+i+1)
            result[i] += add
            candies -= add
            if candies == 0:
                break
        return result


######################################################
#Numbers with Same Consecutive Differences 08/18/2020
######################################################
#TLE, the brute force approach
class Solution(object):
    def numsSameConsecDiff(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: List[int]
        """
        '''
        the naive way would be to first generate the numbers of length N
        then sliding window across the number checking abs difference is K, is so add
        this results in TLE
        but there might be a pattern
        what if we generate the smallest one first, then keep adding to it?
        well hold on, what are two numbers who's difference euqlas N?
        then we can just make them
        7, the only two numbers are 1 and 8, 2 and 9
        9, 0 or 9
        8 9 1
        9 - 9 = 0
        9 - 8 = 1
        9 - 7 = 2
        this is just a permutation creator with a constraint, similar to the knap sacking problem
        this is just dfs
        '''
        if N == 1:
            results = []
            for i in range(0,10):
                results.append(i)
            return results
        results = []
        for i in range(10**(N-1), 10**(N)):
            number = list(str(i))
            matches = len(number) - 1 #there needs to be this many matches
            j = 0 #starting pointer
            while j < len(number) - 1:
                if abs(int(number[j]) - abs(int(number[j+1]))) == K:
                    matches -= 1
                j += 1
            if matches == 0:
                results.append(i)
        
            
        return results


class Solution(object):
    def numsSameConsecDiff(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: List[int]
        """
        '''
        9, 0 or 9
        8 9 1
        9 - 9 = 0
        9 - 8 = 1
        9 - 7 = 2
        this is just a permutation creator with a constraint, similar to the knap sacking problem
        this is just dfs, similar to permutation generator, but adding creating a number two integers at a time
        for the first digit, the space is [1,9], any digit after thant i can use 0 to 0
        build up the number accoridnlty so land as difference between the previous digit and the next digit to add is equal to K
        '''
        results = []
        def create_number(N,K, str_num):
            #base condition, when to add the number
            if len(str_num) == N:
                results.append(int(str_num))
                return
            
            #first digit space
            if len(str_num) == 0:
                for i in range(1,10):
                    create_number(N,K,str_num+str(i)) #i will create this with an empty string first
            
            else:
                for i in range(0,10):
                    if abs(ord(str(i)) - ord(str(str_num[-1]))) == K:
                        #go ahead and create
                        create_number(N,K,str_num+str(i))
                        
        #in the case N is 1
        if N == 1:
            results.append(0)
        create_number(N,K,"")
        return results

#official solutions from leetcode
#DFS
class Solution(object):
    def numsSameConsecDiff(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: List[int]
        """
        '''
        https://leetcode.com/problems/numbers-with-same-consecutive-differences/solution/
        this is just a tree traversal problem, where we generate the number digit by digit
        starting with the highest digit
        the values at the leaf nodes along each path is a number
        the function would be dfs(N,num), where num is the number made so far, and N is the numbe of digits to be added, you keep recursing so long as N was not zero
        and adding it to a results
        we need to keep adding a tail digit until N == 0, 
        the tail digit could be tail digit +- K
        '''
        if N == 1:
            return [i for i in range(10)]
        results = []
        def dfs(N,number):
            if N == 0:
                results.append(number)
            last_digit = num % 10
            next_digits = set([last_digit + K, last_digit - K])
            
            #now keep adding to the last gigit
            for digit in next_digits:
                if 0 <= digit < 10:
                    new_num = num*10 + digit
                    dfs(N-1,num)
        #invoke for n from 1 to 9
        #each invokation is recursion tree startin with n
        for num in range(1,10):
            dfs(N-1,num)
        
        return results

#BFS solution
from collections import deque
class Solution(object):
    def numsSameConsecDiff(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: List[int]
        """
        #using BFS and using a Q
        #check if len number == N then append to output
        #if its not, get the next digit, woudl be -K + K and add go the Q
        q = deque()
        output = []
        #add inital 
        for i in range(10):
            q.append([i])
            
        while q:
            l = q.popleft()
            if len(l) == N:
                l = [str(i) for i in l]
                num = int("".join(l))
                if len(str(num)) == N:
                    output.append(num)
            else: #keep adding digits
                next_digit = l[-1] - K
                if 0<= next_digit < 10:
                    q.append(l+[next_digit])
                next_digit = l[-1] + K
                if 0 <= next_digit < 10:
                    q.append(l+[next_digit])
        
        return list(set(output))

#another way not using a que
class Solution(object):
    def numsSameConsecDiff(self, N, K):
        """
        :type N: int
        :type K: int
        :rtype: List[int]
        """
        temp = range(10)
        for i in range(N-1):
            temp = {x*10+y for x in temp for y in [x%10+K,x%10-K] if x and 0<=y<10}
            
        return list(temp)

##########################
#Goat Latin 08/19/2020
##########################
class Solution(object):
    def toGoatLatin(self, S):
        """
        :type S: str
        :rtype: str
        """
        S = S.split(" ")
        vowels = {'a','e','i','o','u'}
        for i in range(0,len(S)):
            if S[i][0].lower() in vowels:
                S[i] = S[i]+str('ma')+"a"*(i+1)
            else:
                S[i] = S[i][1:]+S[i][0]+'ma'+"a"*(i+1)
        return " ".join(S)

##########################
#Reorder List 08/20/20
##########################
#nice try but no dice
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        '''
        warm up not doing in place first
        zero index stays the same
        1 becomes n
        2 becomes 1
        3 becomes n-1
        4 becomes 2
        5 becomes n -2
        6 becomes 3
        so its 0,n,1,n-1,2,n-3,4,n-4
        '''
        dummy = point =  ListNode(0)
        #reasign head to dummy.next
        values = []
        cur = head

        while cur.next != None:
            values.append(cur.val)
            cur = cur.next
        #since there might be one left
        values.append(cur.val)
        #now traverse values in the right way adding it to dummy.next
        for v in range(10):
            point.next = ListNode(v)
            point = point.next
        head = dummy.next   


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        '''
        go through the linked list add each node to an array
        then reorder with two pointers left and right
        https://leetcode.com/problems/reorder-list/discuss/801971/Python-O(n)-by-two-pointers-w-Visualization
        O(n) time and O(n) space
        '''
        array = []
        cur = head
        length = 0
        while cur:
            array.append(cur)
            cur = cur.next
            length += 1
        
        #now re-order with two pointers
        left,right = 0, length -1
        #reassign head
        last = head
        
        #traverse from both ends
        while left < right:
            #left pointer next
            array[left].next = array[right]
            left += 1
            
            #when left catches up to right
            if left == right:
                last = array[right]
                break
            #right pointer next
            array[right].next = array[left]
            right -= 1
            
            #update last
            last = array[left]
            
        if last:
            last.next = None

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        '''
        second approach involves findin the middle of the linkedlist
        reversing the second half, then mergeing the two lists
        if we have 
        1 2 3 4 5 6 7
        we can split at the middle
        1 2 3 4   and 5 6 7
        we rerevese the second list
        1 2 3 4   and 7 6 5
        then combine
        1 7 2 6 3 5 4
        to find the middle we use the turtle and the hare pointer method
        we revese the second
        and then combine
        '''
        if not head:
            return None
        #locate the middle, first half is before mid, second is after mid
        hare, turtle = head,head
        
        while hare and hare.next:
            turtle = turtle.next
            hard = hare.next.next
        mid = turtle
        
        #reverse the second half,starting from mid
        prev,cur = None,mid
        
        while cur:
            #update cur
            cur.next = prev
            prev = cur
            cur = cur.next
        
        head_of_second_rev = prev
        
        #update the link between the first hald and the reversed second half
        first,second = head, head_of_second_rev
        
        while second.next:
            next_hop = first.next
            first.next = second
            first = next_hop
            
            next_hop = second.next
            second.next = first
            second = next_hop
            
from collections import deque
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        '''
        https://www.youtube.com/watch?v=XIJMdQUzs-I
        the idea is to use a stack
        and rebuild the linkedlist alternating left and right sides
         2 3 
        1  4 2 3
        '''
        q = deque()
        dummy = ListNode(0)
        dummy.next = head
        cur = dummy.next
        while cur:
            q.append(cur)
            cur = cur.next
            
        #rest
        cur = dummy
        #need boolean to pop
        even = False
        while q:
            node = q.pop() if even else q.popleft()
            node.next = None
            cur.next = node
            cur = cur.next
            even ^= True #keep flipping
        return head

from collections import deque
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        '''
        https://www.youtube.com/watch?v=XIJMdQUzs-I
        split the list at the middle
        1  4 2 3
        '''
        if not head:
            return []
        #find the middle node
        #slow and fast pointers, advance slow one, fast by two, until fast cant go anymore
        slow,fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        #reverse the second half
        #three pointers, prev,cur, and a temp that stores the current node to 
        prev,cur = None,slow.next #not as slow but slow's next
        while cur:
            tmp = cur.next
            cur.next = prev
            prev = cur
            cur = tmp
        slow.next = None
        #should loke like 1 2 3 4 7 6 5
        #                 *       *
        
        #merge first and second half
        head1,head2 = head, prev
        while head2:
            tmp = head1.next
            head1.next = head2
            head1 = head2
            head2 = tmp
        return head

################################
#Sort Array by Parity 08/21/2020
################################

class Solution(object):
    def sortArrayByParity(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        '''
        allocate two lists, evens odds
        traverse once dumping into each list
        return concat
        '''
        evens = []
        odds = []
        for num in A:
            if num % 2 == 0:
                evens.append(num)
            else:
                odds.append(num)
        return evens + odds
         

class Solution(object):
    def sortArrayByParity(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        '''
        i could do it in place, using two pointers
        if conditinos if number even on the left, move left point
        if number is odd in right, move the right pointer
        i swap when i is odd and j is even
        '''
        left,right = 0, len(A) - 1
        while left < right:
            if A[left] % 2 > A[right] % 2:
                #swap
                A[left], A[right] = A[right], A[left]
            if A[left] % 2 == 0:
                left += 1
            if A[right] % 2 == 1:
                right -= 1
        return A

######################
#Find Permutation
######################     
class Solution(object):
    def findPermutation(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        '''
        I increasing from I given the set (1,2)
        [1,2]
        DI decreasming then increasing
        cant start with 1
        so [2,1]
        but now we are at I which isincrearing it must be 3,
        [2,1,3]
        DID (1,2,3,4)
        [2,1,4,3]
          D I D
        they are the middle of the two numbers+
        we are going to create a number, can do this recursively or iteratively
        let's do this iteratively because it can be too high, greedy approach?
        from the solution
        so we start with DDIIIID, numbers [1,2,3,4,5,6,7,8]
        to satisfy the first DD lexgrpahically we must ust at least 1 2 3, if they were anywhere else this would not be a minimum
        the III itself minimizes the number
        keep the larger numbers towars the end of the string, unless DDDDDD
        start with the min number that can be formed given n, which is just range(n)
        to satify the pattern we need to reverse only those subsectino of the min array which have a D in the pattern at their corresponding positions
        we use a stack, and stat cnonisdieirng he numbers o from 1 to n
        for ever curr number, wehn we see D in the pattern we just pushe that number on to the stack
        when we find thenext I, we can pop off these numbers from thes tack leading to the formaitno of a revertsed descenidng subpattern of those numbers correspoding to the D's
        '''
        results = [0]*(len(s)+2)
        stack = []
        j = 0 #give reference to the elements in S
        for i in range(1,len(s)+1):
            if s[i-1] == 'I':
                stack.append(i)
                while stack:
                    j += 1
                    results[j] = stack.pop()
            else:
                stack.append(i)
        stack.append(len(s)+1)
        while stack:
            j += 1 
            results[j] = stack.pop()
        return results[1:]

#############################################
#Random Point in Non-overlapping Rectangles
#############################################
import random
class Solution(object):

    def __init__(self, rects):
        """
        :type rects: List[List[int]]
        """
        '''
        count the number of total points in the space
        w[i] is weight vector with number of points for the ith rectangle
        pick a rectangel from this weight vector
        then uniformly pick an x and y from this rectangle
        '''
        self.rects = rects
        #create variables to get the total number of points
        self.total = 0
        self.weights = []
        for x1,y1,x2,y2 in self.rects:
            #increment the total
            self.total += (x2-x1+1)*(y2-y1+1)
            #dnump into weights, no need for normalzigin
            self.weights.append(self.total)
        
        

    def pick(self):
        """
        :rtype: List[int]
        """
        '''
        pick a random number from (0,total)
        use binary search to find the find the ith rectanlge that it less than the target
        this index the rectanlge we want to draw from!
        '''
        target = random.randint(0,self.total+1)
        lo,hi = 0,len(self.rects)-1
        while lo != hi:
            mid = lo + (hi - lo) // 2
            if target >= self.weights[mid]:
                lo = mid + 1
            else:
                hi = mid
        #at this point lo correpsonds to the index for the rectanlge we wnat
        #now we can just draw point from the x_min x_max and y_min and y_max from this rectangle
        x1,y1,x2,y2 = self.rects[lo]
        p_x = random.randint(x1,x2)
        p_y = random.randint(y1,y2)
        return [p_x,p_y]


# Your Solution object will be instantiated and called as such:
# obj = Solution(rects)
# param_1 = obj.pick()


###########################
#Stream of Characters 08/23/20
###########################
from collections import deque
class StreamChecker(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        '''
        a basic Trie would not work in this case, becase we don't know how many of the last chars to match, for example, the last three, do we match only on 'j' or 'jk' or 'jkl'
        one way to get around this is to create a Trie backwards, that way we can start at the end,
        time complexity if num words times len of each word
        '''
        self.trie = dict()
        self.stream = deque([])
        
        for word in set(words):
            node = self.trie
            for char in word[::-1]:
                if char not in node:
                    node[char] = dict()
                node = node[char]
            node['#'] = word
        

    def query(self, letter):
        """
        :type letter: str
        :rtype: bool
        """
        '''
        we start from the end of the stream and check each char going down the trie
        so append each letter to deque and check if we can make it all the way donw the trie
        we append left to the que and see if we can make it down our reversed Trie
        '''
        self.stream.appendleft(letter)
        
        #set the node
        node = self.trie
        for ch in self.stream:
            if "#" in node:
                return True
            if not ch in node:
                return False
            node = node[ch]
        return "#" in node


# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)

#from Time, unoptimized
from collections import deque
class StreamChecker(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.trie = {}
        self.stream = deque()
        for word in words:
            cur = self.trie
            for char in word:
                if char not in cur:
                    cur[char] = {}
                cur = cur[char]
            cur['#'] = True
        

    def query(self, letter):
        """
        :type letter: str
        :rtype: bool
        """
        #for this we want to check if we can make it down the path for any word
        #examine 'abcab' what a are we at? we need to point at both inside of our stream
        temp = deque()
        self.stream.append(self.trie)
        for node in self.stream:
            #each node will be a nested dictionary
            if letter in node:
                temp.append(node[letter])
        self.stream = temp
        for node in self.stream:
            if '#' in node:
                return True
        return False


#from Time, optimized
from collections import deque
class StreamChecker(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.trie = {}
        self.stream = deque()
        for word in words:
            cur = self.trie
            for char in reversed(word):
                if char not in cur:
                    cur[char] = {}
                cur = cur[char]
            cur['#'] = True
            
    def query(self, letter):
        """
        :type letter: str
        :rtype: bool
        """
        #for this we want to check if we can make it down the path for any word
        #examine 'abcab' what a are we at? we need to point at both inside of our stream
        self.stream.appendleft(letter)
        #start
        cur = self.trie
        for l in self.stream:
            #each node will be a nested dictionary
            if l in cur:
                cur = cur[l]
            else:
                break #i dont need to keep goig because im not going down a path
            if "#" in cur:
                return True
        return False

###############################
#Sum of Left Leaves 08/24/2020
###############################
#works on 73/102
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        its a left leave if i can get to it and if there are not left or right nodes
        '''
        left_leaves = []
        def traverse(root):
            if not root or not root.left:
                return
            left = root.left
            if not left.left and not left.right:
                left_leaves.append(left.val)
            traverse(root.left)
            traverse(root.right)
        traverse(root)
        return sum(left_leaves)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        self.total = 0
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        its a left leave if i can get to it and if there are not left or right nodes
        if nodes have no left or right leaes, and if we came left
        '''
        left_leaves = []
        def traverse(root,left):
            if not root:
                return
            
            if not root.left and not root.right and left:
                self.total += root.val
            
            traverse(root.left,True)
            traverse(root.right,False)
            
            
                
        traverse(root,False)
        return self.total

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        from Kevin Naughton
        https://www.youtube.com/watch?v=_gnyuO2uquA
        '''
        def traverse(root):
            if not root:
                return 0
            elif (root.left and not root.left.left and not root.left.right):
                return root.left.val + traverse(root.right)
            else:
                return traverse(root.left) + traverse(root.right)
        return traverse(root) 


#####################################
#Minimum Costs for Tickets 08/25/2020
#####################################
class Solution(object):
    def mincostTickets(self, days, costs):
        """
        :type days: List[int]
        :type costs: List[int]
        :rtype: int
        """
        '''
        this is just a dp array problem
        i can solve this in pass over the 2d array
        
        lets review
        X  1  4  6  7  8  20
        2  2  4  8  10 12 14  this row is using only the two dollars one, the next row can be 7
        7  7  7  7  7  9  11 
        15 15 15 15 15 15 17
        i would just return the minimum at the last col,
        but how do i find the logic for each i,j entry
        lets try another example
        X  1  3  6  7  8  14  31
        5  5  10 15 20 25 30  35 when you advance to the next row look at the previous rows day
        14 14 14 14 14 
        30 14 14 14 14 14 14  19 its still 19
        min(costs[0]*len(days),cost[1],cost[2])
        min(dp[i][j])
        use a 1 day array for all days up to the last day
        set(days) for easier lookup
        we update our dp for the cheapest
        min(dp[i-1]+costs[0],dp[max(0,i-7)]+costs[1],dp[max(0,-30)]+costs[2])
        we use the max of 0 or the day to avoid negative indexing
        https://leetcode.com/problems/minimum-cost-for-tickets/discuss/811437/Python-DP-95
        '''
        number_days = days[-1]
        dp = [0]*(number_days+1)
        days = set(days) #for easier lookup
        
        #starting at 1 and not zero
        for i in range(1,number_days+1):
            if i in days:
                dp[i] = min(dp[i-1]+costs[0],
                           dp[max(0,i-7)]+costs[1],
                           dp[max(0,i-30)]+costs[2])
            #if we aren't at a day we carry over the previous
            else:
                dp[i] = dp[i-1]
        return dp[-1]

from collections import deque
class Solution(object):
    def mincostTickets(self, days, costs):
        """
        :type days: List[int]
        :type costs: List[int]
        :rtype: int
        """
        '''
        using a q for both 7 and 30, on each day simulate what the chepeast is
        one we get past the 7th and 30th day, we need to look back and see if buying this pass made us pay more or made us pay less
        example
          1  4  6  7  8  20
        1 2  4  6  8  10 12
        7 7  7  7  
        30
        but we keep building up 7 and 30 days queues
        q7  
        '''
        #for each q, keep track of day and cost
        sevenday = deque()
        thirtyday = deque()
        cost = 0
        for d in days:
            while sevenday and sevenday[0][0] <= d-7:
                sevenday.popleft()
            while thirtyday and thirtyday[0][0] <= d-30:
                thirtyday.popleft()
            sevenday.append((d,cost + costs[1]))
            thirtyday.append((d,cost + costs[2]))
            cost = min(cost+costs[0],sevenday[0][1],thirtyday[0][1])
        return cost
        

from collections import deque
class Solution(object):
    def mincostTickets(self, days, costs):
        """
        :type days: List[int]
        :type costs: List[int]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/minimum-cost-for-tickets/discuss/810791/Python-Universal-true-O(days)-solution-explained
        using queues so top down first
        keep track of number of passes and the prices for each pass
        use Q for each pass, becasue we need to look back 7 or 30days,
        we'll pop it off each time after the day gets outdated
        '''
        k = 3
        P = [1,7,30]
        cost = 0
        Q = [deque() for _ in range(k)]
        
        for d in days:
            for i in range(k):
                while Q[i]  and Q[i][0][0] + P[i] <= d: #remove from Q because was cant look back that far
                    Q[i].popleft()
                Q[i].append([d+cost +costs[i]])
            #now take the minimum for that day
            cost = min([Q[i][0][1] for i in range(k)])
        return cost

 #another way
 #https://leetcode.com/discuss/explore/august-leetcoding-challenge/810806/august-leetcode-challenge-minimum-cost-for-tickets-dynamic-programming-with-explanation
 '''
 dp[i] is the overall costs until the ithday
 so we have three options
 1 pass dp[i] = dp[i-1] + costs[0]
 7 pass dp[i] = dp[i-7] + costs[1]
 30 pass dp[i] = dp[i-30] + costs[2]
 to void negative indexing, we mast have passed the 1st, 7th, or 30th day
 max(0,i-1)
 '''

def mincostTickets(self, days: List[int], costs: List[int]) -> int:
	dp=[0 for i in range(days[-1]+1)]
	dy = set(days)
	for i in range(days[-1]+1):
		if i not in dy: dp[i]=dp[i-1]
		else: dp[i]=min(dp[max(0,i-7)]+costs[1],dp[max(0,i-1)]+costs[0],dp[max(0,i-30)]+costs[2])
	return dp[-1]


#######################
#The Maze 08/25/2020
#######################
#DFS implementaiton first  
class Solution(object):
    def hasPath(self, maze, start, destination):
        """
        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: bool
        """
        '''
        dfs implementation first
        keep track of visited nodes
        check direcitions in left, right, down, and up
        '''
        visited = set()
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
        
        def dfs(pos):
            '''
            we are going to recurse on each position finding new stops
            '''
            toSearch = []
            for d in directions:
                currentX = pos[0]
                currentY = pos[1]
                while True:
                    possibleX = currentX + d[0]
                    possibleY = currentY + d[1]
                    #now check that we are in bounds
                    if (0<= possibleX < len(maze)) and(0<= possibleY < len(maze[0])) and (maze[possibleX][possibleY] !=1):
                        currentX = possibleX
                        currentY = possibleY
                        continue
                    else:
                        break
                newStop = (currentX,currentY)
                if newStop == destination:
                    return True
                toSearch.append(newStop)
            #mark visited
            visited.add(pos)
            
            #recurse for each node in toSerach
            for node in toSearch:
                if node not in visited:
                    if dfs(node):
                        return True
            return False
        
        return dfs(start)

###BFS approach
from collections import deque
class Solution(object):
    def hasPath(self, maze, start, destination):
        """
        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: bool
        """
        '''
        BFS approach, https://leetcode.com/problems/the-maze/discuss/811130/Python-Explanation-BFS
        '''
        #an edge case where there could be a wall at teh start and desitination
        if maze[start[0]][start[1]] == 1 or maze[destination[0]][destination[1]] == 1:
            return False
        
        #visted and q
        visited = set()
        q = deque([(start[0],start[1])]) #each element in q is of type (x,y), same thing with set
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        
        while q:
            #create next elements to search
            next_elements = []
            #process the coordiantes if they have not been visited
            for x,y in q:
                if (x,y) not in visited:
                    visited.add((x,y))
                    
                    #now do proces this xy
                    for i,j in directions:
                        next_x, next_y = x + i,y+j
                        
                        #while we are in bounds and not at a wall keep moving
                        while (0 <= next_x < len(maze)) and (0<= next_y < len(maze[0])) and maze[next_x][next_y] == 0:
                            #advance in the current direction
                            next_x += i
                            next_y += j
                        #once we have reached a while check if the previous direcrtion is the desitnation coordinate
                        if (destination[0] == next_x - i) and (destination[1] == next_y - j):
                            return True
                        
                        #id not add the previous coordiante as a starting point for BFS
                        next_elements.append((next_x - i,next_y - j))
            q = next_elements
            
        #if i finish this it means there wasn't a path so return False
        return False

#########################
#Fizz Buzz 08/26/20
#########################
class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        output = []
        for i in range(1,n+1):
            if i % 3 == 0 and i % 5 == 0:
                output.append("FizzBuzz")
            elif i % 3 == 0:
                output.append("Fizz")
            elif i % 5 == 0:
                output.append("Buzz")
            else:
                output.append(str(i))
        return output


#############################
#Find Right Interval 08/27/20
#############################
#no dice
class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[int]
        """
        '''
        check if there exists an interval j whose start point is bigger than or equal to the end point of the interval i
        traverse the array
        for each interval in the array, compare to every other interval except itself
        see iff the start points in the other intervals are bigger than the current end
        dump into a list
        return
        n squared
        '''
        if len(intervals) == 1:
            return [-1]
        
        results = []
        for i in range(0,len(intervals)):
            current_end = intervals[i][1]
            for j in range(0,len(intervals)):
                marker = -1
                if j == i:
                    continue
                if intervals[j][0] >= current_end:
                    marker = j
                    break
            results.append(marker)
        return results

#correct implementation for brute force but we get a TLE
class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[int]
        """
        '''
        brute force solution
        compare every interval with every other interval, looking for the interval whose startpoint (by a minmum difference) thatn the chose interval's end point
        keeep track of the interval with the min start point satisfying the given criteria w index
        '''
        if len(intervals) == 1:
            return [-1]
        results = [0]*(len(intervals))
        for i in range(0,len(intervals)):
            minimum =  float('inf')
            min_index = -1 # if there isn't a min
            for j in range(0, len(intervals)):
                if (intervals[j][0] >= intervals[i][1] and intervals[j][0] < minimum):
                    #update the minmum
                    minimum = intervals[j][0]
                    #record the index
                    min_index = j
            #update the results
            results[i] = min_index
        return results
            results.append(marker)
        return results

#half brutforce, hash map retainin indices after sorting
class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[int]
        """
        '''
        still brute force, but this time we order with a hash map
        hash map is <interval:index> entry to keep
        we then sort the intervals on the starting value
        when going in the inner scane, we just look at the points after and point back to the hashmap to grab the intervals
        so now our j is i + 1 < j < n
        the first element found in this scan is the correct one!
        if nothing found after the inner scan, dump -1 into results
        we are reducing teh second loop  by n-1 times
        O(n) * O(n-1) / 2
        '''
        results = [0]*(len(intervals))
        mapp = dict()
        for i in range(0,len(intervals)):
            mapp[str(intervals[i])] = i
            
        intervals = sorted(intervals,key = lambda x: x[0])
        
        #scan
        for i in range(0,len(intervals)):
            minimum = float('inf')
            min_index = -1
            #now i cans earch after i +1
            for j in range(i+1,len(intervals)):
                if (intervals[j][0] >= intervals[i][1]) and (intervals[j][0] < minimum):
                    #update the min
                    minmum = intervals[j][0]
                    #update the min_index
                    min_index = mapp[str(intervals[j])]
            #update the resulkts
            results[mapp[str(intervals[i])]] = min_index
        return results


#hashmap and binary search
class Solution(object):
    def recursive_binary_search(self,intervals,target,lo,hi):
        #base case
        if lo >= hi:
            if (intervals[lo][0] >= target):
                return intervals[lo]
            return None
        
        mid = lo + (hi - lo) // 2
        if intervals[lo][0] < target:
            return self.recursive_binary_search(intervals,target,mid+1,hi)
        else:
            return self.recursive_binary_search(intervals, target,lo,mid)
            
    def findRightInterval(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[int]
        """
        '''
        HASH map creation and then using binary search
        hash map is <interval:index> entry to keep
        we then sort the intervals on the starting value
        when going in the inner scane, we can perform binary search to find the start point that is the minmum from the current i'd endpoint
        we then use the intervals point and look it back in the hashmap
        '''
        results = [0]*(len(intervals))
        mapp = dict()
        for i in range(0,len(intervals)):
            mapp[str(intervals[i])] = i
            
        intervals = sorted(intervals,key = lambda x: x[0])
        
        for i in range(0,len(intervals)):
            interval = self.recursive_binary_search(intervals, intervals[i][1],0,len(intervals)-1)
            results[mapp[str(intervals[i])]] = -1 if interval == None else mapp[str(interval)]
        return results


import heapq
class Solution(object):
    def findRightInterval(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[int]
        """
        '''
        [[1,4], [2,3], [3,4], [10,11]]
        sort starts and ends and keep index or origina
        start = 1,2,3,10
        ends = 3,4,4,11
        then just check if an end is equal to or greater than the start
        but we need to pair the starts to the index
        '''
        starts = []
        ends = []
        output = []
        
        #use heapq for starts and ends
        #remember to store the indices
        for i,(start,end) in enumerate(intervals):
            #these are min heaps so i don't need to use the max trick
            heapq.heappush(starts,(start,i))
            heapq.heappush(ends,(end,i))
        #add negative -1 to starts, in case we can't find anything
        heapq.heappush(starts,(float('inf'),-1))
        start = float('-inf')
        
        while ends:
            end,i_end = heapq.heappop(ends)
            while starts and start < end:
                start,i_start = heapq.heappop(starts)
                if start >= end:
                    break
            output.append((i_end,i_start))
        
        return [b for a,b in sorted(output)]


##########################################
# Implement Rand10() Using Rand7() 08/28/20
##########################################
# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

class Solution(object):
    def rand10(self):
        """
        :rtype: int
        """
        '''
        the idea is called rejection sampling
        when genearting a number in a range, if generated number is in range, output, otherwise reject it and sample aganin
        we call rand7 twice, which gives as an index between 1 to 49, we can created 4 groups of 10, 1:10,11:20,21:30,31:40,
        the last 40:49 does not capture a number that could be mapped to 1:10
        if this happens, we call rand7 twice again until
        we conver the numbers rand7 rand7 back to a number
        '''
        max_idx = 40
        while max_idx >= 40:
            max_idx = rand7()*rand7()
        return max_idx % 10 + 1

# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

class Solution(object):
    def rand10(self):
        """
        :rtype: int
        """
        '''
        the idea is called rejection sampling
        when genearting a number in a range, if generated number is in range, output, otherwise reject it and sample aganin
        we call rand7 twice, which gives as an index between 1 to 49, we can created 4 groups of 10, 1:10,11:20,21:30,31:40,
        the last 40:49 does not capture a number that could be mapped to 1:10
        if this happens, we call rand7 twice again until
        we conver the numbers rand7 rand7 back to a number
        '''
        curr = 40
        while curr >= 40:
            curr = (rand7() - 1) * 7 + rand7() - 1
        return curr % 10 + 1


##########################
#Pancake Sorting 08/29/2020
##########################
class Solution(object):
    def pancakeSort(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        '''
        [3,2,4,1]
        flip at k = 4
        [1,4,2,3]
        flip at k = 2
        [4,1,2,3]
        flip at k = 4
        [3,2,1,4]
        flip at k = 3
        [1,2,3,4]
        output it [4,2,3,4]
        first check if it is sorted, if sorted, return empty list
        len(out) < 10*len(out) k is the number of elements
        
        the trick is to put the largest pancake in its place then we can look at the previous numbers not the the last elemtne we orderd
        [3,2,4,6,5,1] k = 3
        [6,4,2,3,5,1] k = 5
        [1,5,3,2,4,6] k = 1
        [5,1,3,2,4,6] k = 4
        [4,2,3,1,5,6] k = 3
        [1,3,2,4,5,6] k = 1
        [3,1,2,4,5,6] k = 2
        [2,1,3,4,5,6] k = 1
        [1,2,3,4,5,6] DONE
        
        '''
        results = []
        n = len(A)
        for i in range(n,0,-1): #giving us unique numbers in range, A is a permutation of numbers in range (N)
            #find the index of the current number
            i_index = A.index(i)
            #now check if the current number is at its place
            if i_index == i -1: #leave it
                continue
            if i_index != 0:
                #not at the beginning, move the current i to the front
                results.append(i_index+1)
                #rearange A 
                flipped = A[:i_index+1][::-1]
                A[:i_index+1] = flipped
            #bring the current i from the front to the back by flipping,appending the current position
            results.append(i)
            #rearrange A
            A[:i] = A[:i][::-1]
        return results

class Solution(object):
    def pancakeSort(self, A):
        """
        :type A: List[int]
        :rtype: List[int]
        """
        '''
        from tim, similar to leetcode solution
        define flip function that pancake flips from the curren position
        i dentifty the max, assuming max is in the right position
        if its not reassign the max index
        '''
        def flip(index):
            start = 0
            while start < index:
                A[start],A[index] = A[index],A[start]
                start += 1
                index -= 1
        N = len(A)
        results = []
        #start backwards
        for i in range(N-1,-1,-1):
            max_index = i
            for j in range(i,-1,-1):
                if A[j] > A[max_index]:
                    max_index = j
            #now check if we need to flip it
            if max_index != i:
                #send it to the fron
                flip(max_index)
                #send it to is position
                flip(i)
                #now add at most two flips
                results.append(max_index+1)
                results.append(i+1)
        return results

#############################################
#  Largest Component Size by Common Factor 08/30/20
#############################################
def primes(n):
    #find all primes that are fators of n
    #return a set, 1 is irrelvant
    #mathematial formula for finding a prime: sqrt(candidate) + 1
    #this is done recursively
    for i in range(2,int(n**(1/2)) + 1):
        if n % i == 0: #if divisible by i
            #resurn set including i, with the union of the recursive call with n // i ??
            return set([i]).union(primes(n//i))
    return set([n])

print(primes(10))


A = (10 / 100) + 200*.01
B = (10/100) + 20*.01

print(A / B)
