class Solution(object):
    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        i'm building coins up 1 at a time
        for every step that is created make sure it is more than than the previous
        otherwise break at i and return i
        start   steps
        8 - 1       1
        7 - 2       2
        5 - 3       3
        2 - 4
        
        '''
        stairs,coins,i = 0,1,1
        while coins <= n:
            stairs += 1
            i += 1
            coins += i
        return stairs

'''
one thing to notice is that the total number of coins needed for a 
staircase of size k is just
k(k+1) / 2

we can k/(k+1) /2 < n
'''

import math
class Solution(object):
    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        we can write an equation
        k(k+1) / 2 <= n
        k^2 + k + 2n <= 0
        solve for k
        k < -1 +- sqrt(1 -8n) / 2
        '''
        return int(math.floor((-1 +  math.sqrt(1 + 8*n)) / 2))

class Solution(object):
    def arrangeCoins(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        binary search using the number of coins formula
        k(k+1) // 2
        '''
        l,r = 0,n
        while l <= r:
            mid = (l+r) //2
            #get the coins at mid
            coins = mid*(mid+1) // 2
            if coins == n:
                return mid
            elif coins > n:
                r = mid -1
            else:
                l = mid + 1
                
        return r
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return
        output = []
        def traverse(root):
            if root:
                output.append(root.val)
                traverse(root.left)
                traverse(root.right)
        traverse(root)
        #reorder output
        start = output.pop(0)
        new_output = []
        for i in range(0,len(output) -1,2):
            level = [output[i],output[i+1]]
            new_output.append(level)
        new_output = list(reversed(new_output)) 
        new_output.append([start])
        return new_output

class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        #do bfs, but keep track of what level we are on
        #store level and node pointer
        q = collections.deque([(root,0)])
        output = []
        temp = []
        prev_level = 0
        
        while q:
            node,level = q.popleft()
            if node:
                if level != prev_level:
                    output.append(temp)
                    temp = []
                    prev_level = level
                temp += [node.val]
                q.append((node.left,level+1))
                q.append((node.right,level+1))
        if temp:
            output.append(temp)
            
        return output[::-1]

import collections
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        #do bfs, but keep track of what level we are on
        #store level and node pointer
        q = collections.deque([(root,0)])
        output = []
        temp = []
        prev_level = 0
        
        while q:
            node,level = q.popleft()
            if node:
                if level != prev_level:
                    output.append(temp)
                    temp = []
                    prev_level = level
                temp += [node.val]
                q.append((node.left,level+1))
                q.append((node.right,level+1))
        if temp:
            output.append(temp)
            
        return output[::-1]
                

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        #dfs approach, using the index of our output to indicate the level 
        output = []
        
        def dfs(root, level):
            if not root:
                return
            if len(output) < level+1:
                output.append([]) #for a new level
            dfs(root.left, level+1)
            dfs(root.right, level+1)
            output[level].append(root.val)
            
        dfs(root,0)
        return output[::-1]

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        #bfs from knowledge center
        #create a result and first add the root to a Que
        #while there is a Q, initalize a nodes list to hold the nodes at that level
        #then for each thing in the quee:
            #pop left, and add the node vale to nodes, which is local
            #if there is a node on the left, add it to the que
            #if there is a node on the right, ad it tot the ques
        result = []
        if root == None:
            return result
        Q = []
        Q.append(root)
        while (len(Q) > 0):
            nodes = []
            for i in range(len(Q)):
                node =  Q.pop(0)
                nodes.append(node.val)
                if node.left != None:
                    Q.append(node.left)
                if node.right != None:
                    Q.append(node.right)
            result.insert(0,nodes)
        return result

class Solution(object):
    def prisonAfterNDays(self, cells, N):
        """
        :type cells: List[int]
        :type N: int
        :rtype: List[int]
        """
        #empty edge case
        if cells == []:
            return []
        new_cells = [0]*8
        for i in range(N):
            for j in range(1,7):
                #get left
                left = cells[j-1]
                right = cells[j+1]
                if left == right:
                    new_cells[j] = 1
                else:
                    new_cells[j] = 0
                new_cells[0] = 0
                new_cells[7] = 0
            cells = new_cells
            
                    
        return new_cells[::-1]

class Solution:
    def next_step(self, cells):
        res = [0] * 8
        for i in range(1,7):
            res[i] = int(cells[i-1] == cells[i+1])
        return res
    
    def prisonAfterNDays(self, cells, N):
        """
        :type cells: List[int]
        :type N: int
        :rtype: List[int]
        """
        '''
        #there can only be 2^8 possibilites, in fact there are only 2^6 possibilties after day 0
        #so we will be repeating
        #we need to find the loop!
        #to fine a loop we keep out statts in hash table, so we can find the loop_len
        #we do iteration by iteration and if we found a loop, we need (N-1)% loop_len more steps into our hash table
        '''
        found_states = {}
        
        for i in range(N):
            #conver to string to use as a key
            cell_str = str(cells)
            #discovered a state?
            if cell_str in found_states:
                #to discover the length of the loop
                loop_len = i - found_states[cell+str]
                return self.prisonAfterNDays(cells, (N-i) % loop_len)
            #if there isn't a loop
            else:
                #add new state
                found_states[cell_str] =  i
                cells = self.next_step(cells)
        return cells

def maxDivide(a,b):
    while a % b == 0:
        a = a / b
    return a
def isUgly(num):
    num = maxDivide(num,2)
    num = maxDivide(num,3)
    num = maxDivide(num,5)
    if num == 1:
        return 1
    else:
        return 0

class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        i = 1
        count = 1
        while n > count:
            i += 1
            if isUgly(i):
                count += 1
        return i

#this gets a time limit exceed

class Solution(object):
    def prisonAfterNDays(self, cells, N):
        """
        :type cells: List[int]
        :type N: int
        :rtype: List[int]
        """
        memo = {}
        while N: 
            if str(cells) in memo:
                N %= memo[str(cells)] - N
            memo[str(cells)] = N
            
            if N > 0:
                temp = [0]*8
                for i in range(1,7):
                    if cells[i-1] == cells[i+1]:
                        temp[i] = 1
                    else:
                        temp[i] = 0
                cells = temp
                N -= 1
        return cells

class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        #just use bitwise operatiors
        binary = bin(x ^ y)
        setBits = [ones for ones in binary[2:] if ones=='1'] 
        return len(setBits)

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        string = ""
        for i in digits:
            string += str(i)
        string = str(int(string) + 1)
        results = []
        for i in string:
            results.append(int(i))
        return results

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        #another way, increment the digit by 1 if less than 0
        #keep doing this for all indices in digits until the last one
        #staring index
        index = len(digits) - 1
        while index >= 0:
            if digits[index] == 9:
                digits[index] = 0
            else:
                digits[index] += 1
                return digits
            index -= 1
        return [1] + digits
                

class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        #notes, width and height do not exceed 100, so i can use a double for loop 10^4
        #the corners can only add up to a max of two sides
        rows = len(grid)
        cols = len(grid[0])
        if (rows == 0) or (cols == 0):
            return 0
        p = 0
        
        #travers the whole array adding 4 for each one
        #after adding check if there is is neighbor to the left and above, and for each neighbord subtratct 2
        for i in range(0,rows):
            for j in range(0,cols):
                if grid[i][j] == 1:
                    p += 4
                    if (i>0) and (grid[i-1][j] == 1):
                        p -= 2
                    if (j>0) and (grid[i][j-1] == 1):
                        p -= 2
                        
        return p
                #now check for in bounds and for left neighbow

##DFS recursive
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        #dfs seach. check to see if it is an island, add one if surrounded by water
        #also mark each node as being visited
        rows = len(grid)
        cols = len(grid[0])
        
        def dfs(r,c):
            #checking out of bounds, and for water, add 1 to the perimeter
            if (r < 0) or (r >= rows) or (c < 0) or (c >= cols) or grid[r][c] == 0:
                return 1
            if grid[r][c] == 1:
                #mark as visited
                grid[r][r] = 2
                return dfs(r-1,c) + dfs(r+1,c) + dfs(r,c-1) + dfs(r,c+1)
            return 0
        
        #invoke dfs
        p = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    p += dfs(i,j)
        return p

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        results = []
        #call numbers i,j,k
        for i in range(0,len(nums)):
            first_num = nums[i]
            two_sum = 0 - first_num
            #now find numbers for two sum using hash
            mapp = {}
            #first pass
            for j in range(i+1,len(nums)):
                mapp[nums[j]] = j
            #second pass searching for complement
            for j in range(i+1,len(nums)):
                complement = two_sum - nums[j]
                if (complement in mapp.keys()):
                    results.append([first_num,nums[j],nums[mapp[complement]]])
        #now remove duplicates
        #this gets confusing, use the two pointers approach
        return results

##two pointers approach
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        #choose an element number i
        #find pair of eleemnts beg and end such that i < beg < end
        #and numbers[beg] + nums[end] = target = -nums[i], which is just the complement
        #approach, stargin frmo beg,end = i+1, n-1, and move beg to tehr giht and n to th left
        #comparing nums[beg] + nums[end] to our target
        #if it euqal to target we add it to our resulkts and move two pointers
        #because we can have equal numbbers in num, we need to check taht we reurn only unique, as we apply set
        #time complexity if (nlogn + n^2)
        nums.sort()
        n, result = len(nums), []

        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]: continue

            target = -nums[i]
            beg, end = i + 1, n - 1

            while beg < end:
                if nums[beg] + nums[end] < target:
                    beg += 1
                elif nums[beg] + nums[end] > target:
                    end -= 1
                else:
                    result.append((nums[i], nums[beg], nums[end]))
                    beg += 1
                    end -= 1

        return set(result)


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def widthOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        ##bfs see knowledge center video
        if root == None:
            return 0
        #niit results to 1, we always have at least 1 as the root, so size 1
        results = 1
        #init que
        Q = [[root,0]]
        while len(Q) > 0:
            count = len(Q)
            #get indices for the node, binary heap
            start = Q[0][1]
            end = Q[-1][1]
            results = max(results, end-start+1)
            #then for eeach nod in the Q
            for i in range(count):
                #i dont get what is goin on here
                p = Q[0] #this pulls the node,idx pair
                idx = p[1] - start #this gets the index of the next pair in the Q
                Q.pop(0)
                if p[0].left != None:
                    Q.append([p[0].left,2*idx +1])
                if p[0].right != None:
                    Q.append([p[0].right, 2*idx +2]) #1 and 2 for zero indexing
                    
        return results

#another way of doing it
import collections
class Solution(object):
    def widthOfBinaryTree(self, root):
        q = collections.deque([root,1])
        #starting width 
        width = 0
        while q:
            #starting indices
            _,left = q[0]
            _,right = q[-1]
            width = max(width,right-left+1)
            #new q for next level
            next_level = collections.deque()
            while q:
                node,index = q.popleft()
                if node.left:
                    next_level.append([node.left,index*2])
                if node.right:
                    next_level.append([node.right, index*2 + 1])
                #reset q
                q = next_level
        return(width)

###########################################
#Flatten a Multilevel Doubly Linked List
##########################################

'''
1 -- 2 -- 3 -- 4 -- 5 -- 6
          7 -- 8 -- 9 -- 10
              11 -- 12

              1
             /
            2
           /
          3
         /\
        4  7  
traverse next until child:
    if not node.child:
        print node.val
    if node.child:
        node = node.child
        traverse
'''
class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        results = []
        def dfs(node):
            if node.next:
                results.append(node.val)
                dfs(node.next)
            if node.child:
                node = node.child
                dfs(node)
            if not node.next:
                results.append(node.val)
                

        dfs(head)
        return results

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution(object):
    def flatten(self, head):
        """
        :type head: Node
        :rtype: Node
        if node has a child, make its node.next be node.child
        and make the child's node.previous by the node.child
        and reurse when there is a node with a child
        use stack that contains both next and child nodes
        this is done easier with a stack
        
        1. put head if our list to a staock, pop it and add the two elements in this order, child and next. We want to visit the child first
        2. each time we pop the last elemtn from stack
        """
        if not head:
            return head
        starting = Node(0)
        #set the pointers
        curr = starting
        stack = [head]
        
        while stack:
            temp = stack.pop()
            if temp.next:
                stack.append(temp.next)
            if temp.child:
                stack.append(temp.child)
            #swap the pointers
            curr.next = temp
            temp.prev = curr
            temp.child = None #so that no child pointers have any values
            curr = temp
        starting.next.prev = None
        return starting.next

###################
#SUBSETS
###################

'''
this is going to help me under stand rercursion
so we want to to permuate a string a recively
the set would be:
[] = [a + permuate(using all elements that are not a) + b + permuate(all ele not be)...]

def permute(s):
    if len(s) == 1:
        return [s]
    perm_list = []

    for a in s:
        remaining = [x for x in s if x != a]
        z = permuate(remaining)
        for t in z:
            perm_list.append([a] + t)
    return 

def permute(s):
    out = []
    if len(s) == 1:
        out = [s]
    for i,let in enumerate(s):
        for perm in permute(s[:i]+s[i+1:]): #each perm is call to permuate, so it adds to output, doing that for each letter
            out += [let +perm]
    return out
'''
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        #use recursion
        def comb_rec(nums):
            if len(nums) == 0:
                return [[]]
            cs = []
            for c in comb_rec(nums[1:]):
                cs += [c,c+[nums[0]]]
            return cs
        return comb_rec(nums)
test = Solution()
print(test.subsets([1,2,3]))
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        #do it recursively
        # 1 leave [2,3]
        # 2 leave [3]
        # 3 leave []
        #ade [] [3], [2], [2,3], [1], [3,1], [2,1], [2,3,1]
        
        output = []
        def rec_subset(start,temp, output):
            output += [temp]
            for i in range(start,len(nums)):
                rec_subset(i+1,temp+[nums[i]],output)
        rec_subset(0,[],output)
        return output
test = Solution()
print(test.subsets([1,2,3]))

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        #the iterative solution
        if len(nums) == 0:
            return [[]]
        results = [[]]
        for x in nums:
            length = len(results)
            for i in range(length):
                to_add = results[i] + [x] #adding two lists
                results.append(to_add)
        return results
###################
#REVERSE BITS########
####################

class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        #naive way would be convet the binary rep to a string
        #revese it
        #recase the revesed string back into a binary
        string_n = '{:032b}'.format(n)
        string_n = ''.join(list(reversed(string_n)))
        num = int(string_n,2)
        num = bin(num)
        return num

class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        #naive way would be convet the binary rep to a string
        #revese it
        #recase the revesed string back into a binary
        rep = list('{:032b}'.format(n))
        rep.reverse()
        return int("".join(rep),2)

    def reverseBits(self, n): #another wise using the operators
        output, power = 0,31
        while n:
            output += (n&1) << power
            n = n >> 1
            power -= 1
        return output


########################
#SAME TREE
#######################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        '''
        #travese each tree in wtv traversal
        #for each traverse, append to a list
        #compare lists at the and return if they match
        '''
        
        def traverse(node,order=[]):
            if node:
                order.append(node.val)
                traverse(node.left,order)
                traverse(node.right,order)
            return order
        p_order = traverse(p)
        q_order = traverse(q)
        
        for i in range(0,min(len(p_order),len(q_order))):
            if p_order[i] != q_order[i]:
                return False
                break
            else:
                return True

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        def traverse_compare(t1,t2):
            if t1.val != t2.val:
                return False
            traverse_compare(t1.left,t2.left)
            traverse_compare(t1.right,t2.right)
            return True

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        def dfs(p,q):
            #this is the base case
            if not p and not q:
                return True
            #this is the case we are testing for and it can happen other ways
            elif (p and not q) or (q and not p) or (p.val != q.val):
                return False
            #compare both treesm making sure its left and right subtrees are used in the function
            return dfs(p.left,q.left) and dfs(p.right, q.right)
        return dfs(p,q)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        '''
        use the function is same Tree
        '''
        #if there isn't anything they are the same tree
        if not p and not q:
            return True
        #the case where one it not and the other is
        elif not p or not q:
            return False
        #the case we have values and they are different
        elif p.val != q.val:
            return False
        else:
            return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)

###################################
#ANGLE BETWEWEN TWO HANDS OF A CLOCK
###################################

class Solution(object):
    def angleClock(self, hour, minutes):
        """
        :type hour: int
        :type minutes: int
        :rtype: float
        """
        '''
        notes, the largest angle that can be formed is 360
        there are 60 ticks, each tick can form at most 180/60
        the trick with the minute had is that the hour hand moves (360 / 12) every hour or 360/720 in a minute
        see how many minutes are past the hour hand
        get the one angle and take the minmum of that angle or 360 - angle
        the hour hand can only move 30 degrees in an hour
        so 12:30 is
        180 - (30/60)*30 = 165
        3:30 is
        90 - (30/60)*30 = 75
        
        '''
        angle_minutes = (float(minutes) / 60)*360
        angle_hour = (float(hour)*(360/12) % 360) + (float(minutes) / float(60))*30
        return min(abs(angle_minutes - angle_hour), 360 - abs(angle_minutes - angle_hour))

#######################
#REVERSE STRING
######################

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        #split on white spaces
        #then reverse
        return " ".join(reversed(s.split()))

#############################
#POW(x,n)
#############################

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        output = x
        temp_n = n
        counter = n - 1
        while(counter > 0):
            output *= x
            counter -= 1
        if temp_n < 0:
            return 1/output
        else:
            return output

'''
the trick is to divide it into subproblesm
we realize that 2^8 is just 2^4 times 2^4
in the base case we return just 1, which is 2^0
'''

def power(x,y):
    if y == 0:
        return 1
    elif (int(y/2) % 2 == 0):
        #recruse mutpiplying teh two function calls
        return power(x, int(y/2))*power(x, int(y/2))
    else: # the case where you may have one more times to mutiple
        return x*power(x,y/2)*power(x,y/2)


'''
but we can optimize by storying only y/2 onice
'''

def power(x,y):
    if y == 0:
        return 1
    temp = y/2
    if (y%2 == 0):
        return temp*temp
    else:
        return x*temp*temp

'''
now include negatives
'''

def power(x, y): 
  
    if(y == 0): return 1
    temp = power(x, int(y / 2))  
      
    if (y % 2 == 0): 
        return temp * temp 
    else: 
        if(y > 0): return x * temp * temp 
        else: return (temp * temp) / x 

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        '''
        binary search
        '''
        if n == 0:
            return 1
        #get the first calc
        temp = self.myPow(x, int(n/2))
        if n % 2 == 0:
            return temp*temp
        else:
            if n > 0:
                return x*temp*temp
            else:
                return (temp*temp) / x
        

        class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        def power(x,n):
            if n == 0:
                return 1
            #get the first calc
            temp = power(x, int(n/2))
            if n % 2 == 0:
                return temp*temp
            else:
                if n > 0:
                    return x*temp*temp
                else:
                    return (temp*temp) / x
        return power(x,n)

#binary exponentiation
class Solution:
    def myPow(self, x, n):
        if n == 0: 
            return 1
        if n == 1:
            return x
        if n < 0:
            return self.myPow(1/x,-n)
        result = self.myPow(x, n//2)
        result *= result
        if n %2 == 1:
            result *= x
        return result


########################
#TOP K ELEMENTS
#######################
import collections
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        c = collections.Counter(nums)
        c = sorted(c, key = lambda x: c[x], reverse = True)
        
        return c[:k]

#using heaps


##################################
#COURSE SCHEDULE I
#################################
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        #nodes can be unvisted, visited, and complete
        #cycle detection looking for back edge( edge going to visted edge on dfs call)
        #create adjaceny list
        adj = dict()
        for i in range(0,numCourses):
            adj[i] = []
        for p in prerequisites:
            adj[p[0]].append(p[1])
        
        #created visited, index position is course number, mark all unknown, 
        #then with each dfs call change to V or C
        #0 is unknown, 1 is visited, 2 is complete
        visited = [0]*numCourses
            
        #recursive call
        def dfs_helper(v): 
            #when to recruse, call only on adjacent neighbors of a node
            ##this just visites each node and marks it as visited
            if visited[v] == 1: #when to fired dfs
                return False
            visited[v] = 1 #mark it if not
            for ad in adj[v]:
                if not dfs_helper(ad):
                    return False
            visited[v] = 2 #mark completed 
            
            return True
        #invoke for each course in numCourse
        for i in range(0,numCourses):
            if visited[i] == 0 and not dfs_helper(i):
                return False
        return True
                
            
            

##################################
#COURSE SCHEDULE II
#################################

class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        #build adjaceny list
        #start with a node, dfs on the node until i reach a node that can't be explored
        #add to the stack
        #backtrack
        #if i cant go anywhere else from this backtrack node, add to the stack and backtracka gain
        #recurse until all nodes have been visited
        #pop the whole stack
        #if stack is empty, return an empty list
        adj = {}
        for i in range(numCourses):
            adj[i] = []
        for p in prerequisites: #b --> a in the pairs watch for this
            adj[p[1]].append(p[0])
        s = []
        visited = [0]*numCourses
        
        def dfs_cycle_detection(u):
            visited[u] = 1
            for v in adj[u]:
                if visited[v] == 1:
                    return True
                if (visited[v] == 0) and (dfs_cycle_detection(v)):
                    return True
            visited[u] = 2
            s.append(u)
            return False
        
        
        #to invoke
        for i in range(numCourses):
            if (visited[i] == 0) & dfs_cycle_detection(i):
                return []
        s.reverse()
        if len(s) == 2:
            return s
        else:
            return s[(len(s) // 2):]

class Solution(object):
    def dfs(self,u):
        self.visited[u] = 1
        for v in self.adj[u]:
            if self.visited[v] == 1:
                return True
            if self.visited[v] == 0 and self.dfs(v):
                return True
        self.visited[u] = 2
        self.s.append(u) 
        return False
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        #build adjaceny list
        #start with a node, dfs on the node until i reach a node that can't be explored
        #add to the stack
        #backtrack
        #if i cant go anywhere else from this backtrack node, add to the stack and backtracka gain
        #recurse until all nodes have been visited
        #pop the whole stack
        #if stack is empty, return an empty list
        self.adj = [[] for i in range(numCourses)]
        for p in prerequisites:
            self.adj[p[1]].append(p[0])
        
        self.s = []
        self.visited = [0]*numCourses
        for i in range(numCourses):
            if self.visited[i] == 0 and self.dfs(i):
                return []
        self.s.reverse()
        return self.s

class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        #dfs on each node, keep track of being visited
        #but need to keep track of a cycle, basically a node pointing to a node in a visited set
        #terminate dfs if cycle is detected
        adj = [[] for _ in range(numCourses)]
        for c,p in prerequisites:
            adj[p].append(c)
        
        stack = [] #keep track
        visited = set()
        tracker = set()
        self.cycle = False #to detect a cucle
        
        def dfs(node,visited,tracker,stack):
            visited.add(node)
            tracker.add(node)
            for nx in adj[node]:
                if nx in tracker:
                    self.cycle = True
                if nx not in visited: #fire dfs
                    dfs(nx,visited,tracker,stack)
            tracker.remove(node)
            stack.append(node)
            
        for node in range(numCourses):
            if node not in visited:
                dfs(node,visited,tracker,stack)
        if self.cycle:
            return []
        return stack[::-1]
            

########################
#Add Binary
#######################

class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        #the naive way would be to recast each as an int
        #add them
        #then recast as binary
        #warm up with this first
        return str(bin((int(a,2)+int(b,2)))[2:])

class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        #start at the end of the string, using two pointers
        #while loop so that both pointers are >= 0, or conidtion, travese all for both
        #use carry variable
        result = []
        a_pointer = len(a) - 1
        b_pointer = len(b) - 1
        carry = 0
        
        #traverse both lists
        #but we may have uneven lengths, so one will have to keep going
        while (a_pointer >= 0) or (b_pointer >= 0):
            sum_two = carry
            #making sure it is in bounds
            if a_pointer >= 0:
                sum_two += int(a[a_pointer])
            if b_pointer >= 0:
                sum_two += int(b[b_pointer])
            result.append(str(sum_two % 2))
            carry = int(sum_two / 2)
            a_pointer -= 1
            b_pointer -= 1
        if carry != 0:
            result.append(str(carry))
        result.reverse()
        return ''.join(result)

#######################################
#Remove Linked List
######################################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        #dummy pointer
        #store previous node, and if pre.val == val, then pre becomes next
        #keep previous node
        dummy = ListNode(0)
        dummy.next = head
        
        curr,prev = head,dummy
        
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return dummy.next

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        #while loop to take care if the head val is equal to the valu
        while (head and head.val == val):
            head = head.next
        
        current_node = head #with head updated
        while (current_node and current_node.next):
            if current_node.next.val == val:
                current_node.next = current_node.next.next
            else:
                #move the current node to next
                current_node = current_node.next
        return head

################################
#WORD SEARCH
###############################
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        #level order DFS in a 2d matrix, DFS around the found i,j, if we haven't found the next char, return false, else return true
        N = len(board)
        M = len(board[0])
        
        def dfs(board,row,col,count,word):
            #if we have found the word
            if count == len(word):
                return True
            #check boundaries
            if (row < 0) or (row >= N) or (col < 0) or (col >= M) or (board[row][col] != word[count]):
                return False
            #so now we are in bounds, and we need to recurse
            #mark a cell us used, just use empty space
            temp = board[row][col] #add value back
            board[i][j] = ' '
            #recurse in all possible directions
            found = dfs(board,row+1,col,count+1,word) or dfs(board,row-1,col,count+1,word) or dfs(board,row,col+1,count+1,word) or dfs(board,row,col-1,count+1,word)
            #replace the board
            board[row][col] = temp
            return found
            
        
        #traverse board
        for i in range(0,N):
            for j in range(0,M):
                if board[i][j] == word[0] and dfs(board,i,j,0,word):
                    return True #after dfs fires and returns true
        
        return False
        
#works on 80 or 89 cases

class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        #level order DFS in a 2d matrix, DFS around the found i,j, if we haven't found the next char, return false, else return true
        N = len(board)
        M = len(board[0])
        P = len(word)
        
        def dfs(row,col,pos):
            if pos >= P:
                return True #meaning we've found the word
            elif 0 <= row < N and 0 <= col < M and board[row][col] == word[pos]:
                temp = board[row][col]
                board[row][col] = None
                #when to do dfs
                if dfs(row-1,col,pos+1) or dfs(row+1,col,pos+1) or dfs(row,col+1,pos+1) or dfs(row,col-1,pos+1):
                    return True
                board[row][col] = temp
            return False
        
        for i in range(0,N):
            for j in range(0,M):
                if dfs(i,j,0):
                    return True
        return False


###########################################
#Binary Tree Zigzag Level Order Traversal
##########################################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        #need to keeptrack of the level we are one
        #so the direction travled on a level is the reverse of the previous level
        #we are going level by level, so it has to be BFS
        #level order BFS
        #as warm up implement level order BFS first
        results = []
        
        def BFS_level_order(root):
            if root is None:
                return
            q = []
            #initliaze the q
            q.append(root)
            while (len(q) > 0):
                results.append(q[0].val)
                #get the next node
                node = q.pop(0)
                #add the left child node
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
            return results
        print BFS_level_order(root)
        
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        results = []
        def BFS_by_level(root):
            if root is None:
                return []
            if root == []:
                return []
            q = []
            q.append(root)
            while q:
                count = len(q)
                level = []
                while count > 0:
                    node = q.pop(0)
                    level.append(node.val)
                    if node.left:
                        q.append(node.left)
                    if node.right:
                        q.append(node.right)
                    count -= 1
                results.append(level)
                level = []
                
            for i in range(1,len(results),2):
                results[i] = list(reversed(results[i]))
            return results
        return BFS_by_level(root)
            def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        #doing dfs instead
        #pass in the node, a level and an output
        #if odd level, prepend
        #if even append
        output = []
        def dfs(node,level,output):
            if not node:
                return
            if len(output) >= level: #meaning we are at a new level
                output += [[]]
            dfs(node.left, level+1,output)
            dfs(node.right, level+1,output)
            if level % 2 == 0:
                output[level].append(node.val)
            else:
                output[level].insert(0,node.val)
                
        #invoke
        dfs(root,0,output)
        return output

#############################
#Singler Number III
############################

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #just count, then search the entire count
        #two pass
        count = dict()
        for num in nums:
            if num not in count.keys():
                count[num] = 1
            else:
                count[num] += 1
        #search counts for 1 occurence
        results = []
        for k,v in count.items():
            if v == 1:
                results.append(k)
        return results

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #lookup implementation
        lookup = set()
        for num in nums:
            if num not in lookup:
                lookup.add(num)
            else:
                lookup.remove(num)
        return list(lookup)

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        #bit manupilations XOR
        #start xoring with 0 for the first number, update the xor to xor with xor result
        #so now we have num1 xor num2 = updated xor
        #but how do we get num1 and num2
        #we need the frist bit in this xor that is 1
        #xor^(xor -1) and xor = num1, this is tricky
        #pass again through the array

        #recall the properties A^0 = A and A^A
        #if there are duplicates, we will end up getting xor of the two numbers which are single in our array
        #i.e if a and b are the two singel numbers tha a^b = xor, the number
        #and operator is to check of the nth bit is set
        #or operator can be used to set a bit of certain number
        xor = 0
        for n in nums:
            xor ^= n
        
        firstbit = xor & (xor -1) ^ xor
        num1 = 0
        for n in nums:
            if n & firstbit:
                num1 ^= n
        return [num1,num1^xor]
 
 ########################################
 #All paths from source target
 ##########################################

 class Solution(object):
    def allPathsSourceTarget(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[List[int]]
        """
        #DFS approach
        #mark each node as visited
        #results list initlizaed
        #keep adding paths so long as they reach the end
        #base condition is at the last node
        end = len(graph) - 1
        output = []
        def dfs(node,path):
            if node == end:
                output.append(path)
            for adj in graph[node]:
                #recurse
                dfs(adj,path +[adj]) #assumption is that DAG has no cycles, no need for marking anything
        dfs(0,[0])
        return output

import collections
 #graph approach, connecting intervals by an edge
        #adjacency list
        #connected nodes means the intervals can be combined
        #store visited nodes in a set - constant time adn contaniment
        #essentially brute force calculate - compare every interval to every other interval
        #recall notes on the connected nodes via two edges

########################################
#153 Find Minimum in Rotated Sorted Array II
#########################################
#this is just a follow up to the actual coding challenge

class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #i can keep dividing by two
        #for each division, used two pointers going to the middle
        #if at anypoint the right pointers value is greater than the left we can return the left pointers value
        #is not join the two again
        #binary search with recursion?
        #rotation means there is an inflection point a jump basically
        #soo all elements to the left should be greater than the min
        #all elements to the right should be less
        #steps, find the mid element in the arrays
        #if mid elemnt > first elemnt of array, its to the right
        #else it is to the left
        #we stop the search when we have our infelciton point
        #or when nums[mid] > nums[mid] > nums[mid + 1]:return mid + 1 # the even case
        #or when numsp[mid-1] > nums[mid] return mid

        #notes from Nick, look on the unsorted side, because of inflection point

        
        if len(nums) == 1:
            return nums[0]
        #pointers
        l = 0
        r = len(nums) - 1
        #chekcing the rotation condition
        if nums[r] > nums[l]:
            return nums[0] #no rotation
        
        #binary search!
        while l <= r:
            mid = l + (r - l) // 2
            #if the mid is greater than its next elemnt, than mid_1 is the smalles
            if nums[mid] > nums[mid+1]:
                return nums[mid+1]
            #if mid elemnt is less than previous, than you know its at mid!
            if nums[mid-1] > nums[mid]:
                return nums[mid]
            #advancing pointers
            if nums[mid] > nums[0]:
                l = mid + 1
            else:
                r = mid - 1


    #another variaiont
    def findMin(self, nums):
        if len(nums) == 1:
            return nums[0]

        l,r = 0, len(nums) -1
        while l <= r:
            mid = l + (r-l) // 2
            if (mid > 0) and(nums[mid] < nums[mid-1]):
                return nums[mid]
            elif (nums[l] <= nums[mid]) and (nums[mid] >nums[r]):
                l = mid + 1
            else:
                r = mid - 1
        return nums[l]


        class Solution(object):
def findMin(self, nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    left, right = 0, len(nums) - 1
    #condition since this is a sorted array
    #the second conditional only works on this problem
    while left < right and nums[left] >= nums[right]:
        mid = left + (right - left) / 2
        #adjusts the boundes to the unsorted side
        if nums[mid] < nums[left]:
            right = mid
        else:
            left = mid + 1
    
    return nums[left]


class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l,r = 0,len(nums)-1
        while l < r:
            mid = l + (r-l) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] < nums[r]:
                r = mid
            else:
                return nums[r]
        return nums[l]
        
########################################
#Find Minimum in sorted array II
########################################
#same as the first one
#but we cannnot get an extra infomration at the last conidiional
#but then we have to scan the whole thing
#nlogn
class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l,r = 0,len(nums)-1
        while l < r:
            mid = l + (r-l) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            elif nums[mid] < nums[r]:
                r = mid
            else:
                r -= 1 #since there can be duplicates we have to reduce the seach space by 1
        return nums[l]


#######################
#ADD DIGITS
######################
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        #the naive way would be to keep adding until the length of the digit is only one
        while(len(str(num)) > 1):
            num = sum([int(a) for a in list(str(num))])
        return num
    #another way
    def addDigits(self, num):
        while num > 9:
            temp = 0
            while temp:
                temp += num%10
                num //=10
            num = temp
        return num

#review of the following article
#https://en.wikipedia.org/wiki/Digital_root
'''
F_b(n) = sum_{i=0}^{k-1} d_{i}
where k = \floort(\log_b n)
and d_i = \frac{n mod b^{i+1} - n mod b^i}{b^i}
'''

class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        #you can just use the floor function
        if num == 0:
            return 0
        return num - (10-1)*((num-1)//(10-1))

#can use disivion  by 9
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        digital_root = 0
        while num > 0:
            digital_root += num % 10
            num = num // 10
            
            if num == 0 and digital_root > 9:
                num = digital_root
                digital_root = 0
            
        return digital_root


##############################################################
#  Construct Binary Tree from Inorder and Postorder Traversal
###############################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        #last node in post order traversal is the root first?
        #so why do we have an in roder?
        #well the in order gives us the left and right subtrees
        #recursive method just set the root of a subtree
        #pass in the right from the inorder list
        #pss in theelft from the in order list
        mapp = {}
        for i,v in enumerate(inorder):
            mapp[v] = i
        
        def build_rec(low,high):
            #base condition?
            if low > high:
                return
            
            root = TreeNode(postorder.pop())
            #now pass in left of that root from the inorder 
            mid = mapp[root.val]
            root.right = build_rec(mid+1,high)
            root.left = build_rec(low,mid-1)
            return root
        
        #invoke
        return build_rec(0,len(inorder) - 1)


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        #the last node in a post order traversal is the roote node
        #look up this last value on the in order traversal to find the left and right subtrees
        #recruse for each of these left and right subtrees
        #we need to keep track of the the indices for starting in order, ending in order, and endng post
        #we decrement each ending post after a stack call
        def build_rec(i1,i2,p1,p2):
            if i1 >= i2 or p1 >=p2:
                return None
            root = TreeNode(postorder[p2-1])
            it = inorder.index(postorder[p2-1])
            diff = it - i1
            root.left = build_rec(i1, i1+diff,p1,p1+diff)
            root.right = build_rec(i1+diff + 1, i2, p1+diff, p2 -1) #instead of passing in the lists, pass the indeices, but these need to be update
            return root
        
        n = len(inorder)
        if n == 0:
            return None
        return build_rec(0,n,0,n)


#####################
#TASK SCEHDULER
#######################

import collections
import heapq
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        #thing we want:
        #counter of tasks, use up the tasks with the highest amount first
        #and then do more combinations of tasks so that we are less than n
        #this is kinda of greedy
        #use a heap
        '''
        AAABBC
        ABC--AB---A = 11
        (3,A)
        (2,B)
        (1,C)
        
        push to temp list decrementing the counts
        temp
        [(2,A),(1,B),(0,C)]
        
        counter is +1+1+1 = 3
        but cooldown is 2
        conditional i <=n
        
        add back to the heap
        '''
        output = 0
        heap = []
        counter = collections.Counter(tasks)
        #turn into a heap
        #reminder heap is like a stack, but the smalles element of the heap is popped off
        #pyhon is a maxhheap
        for k,v in counter.items():
            heapq.heappush(heap,(-v,k))
            
        while heap:
            temp = [] #does not need to be a heap
            i = 0 #counter for the number of tasks so far
            #to find the cool down
            while i <= n:
                output += 1
                if heap:
                    nums,key = heapq.heappop(heap) #this gets the task with the max amount
                    nums += 1 #using up this task
                    #now we need to add it back to the heap
                    if nums < 0: #meaning we still can used it up
                        temp.append((nums,key))
                #break out of the counting loop
                if not heap and not temp:
                    break
                i += 1
            #now reappend temp to the heap
            for k,v in temp:
                heapq.heappush(heap,(k,v))
  
        return output

#my attempt after learning
import heapq
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        #create counter object
        cycles = 0
        counter = {}
        for task in tasks:
            if task not in counter.keys():
                counter[task] = 1
            else:
                counter[task] += 1
        #making a max heap
        heap = []
        for k,v in counter.items():
            heapq.heappush(heap,(-v,k))
        
        #do work from the heap
        while heap:
            temp = []
            for i in range(0,n+1):
                if heap: #to take tasks
                    temp.append(heapq.heappop(heap)) #adding tasks 
            #now we need to decrement each and put back into the heap or in this case add
            for nums,key in temp:
                nums += 1
                if nums < 0:
                    heapq.heappush(heap,(nums,key))
            
            if not heap:
                cycles += len(temp)
            else:
                cycles += n  + 1
            
        return cycles

#the mathy ways of seeing it
#basically you return the maxof the len(tasks), (n+1)*(max_num-1) + max_num_count
#max_num is the maximum number of the same taskes that appears in the tasks list
#max_num_count the number of tasks with max_num
#n is the cool down period
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        count = list(collections.Counter(tasks).values())
        max_num = max(count)
        max_num_count = count.count(max_num)
        return max(len(tasks), (n+1)*(max_num-1)+ max_num_count)
        
        
#####################################################
#Best Time to Buy and Sell Stock with Cooldown
#####################################################
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        this is a DP problem
        #two states, need max profit while holding a stock, and need max profit when we are not holding anythin
        '''
        #base condition
        if not prices:
            return 0
        #two DP arrays, while holding and not holding, this is the max profit
        hold = [0 for _ in range(len(prices)+1)]
        nothold = [0 for _ in range (len(prices)+1)]
        #states on what we can do each day
        
        #on the first day of buying stock
        hold[1] = -prices[0]
        nothold[1] = 0
        #these are just two initliaze the first days in each of the two states
        #traverse prices updating hold and nothold
        for i in range(1, len(prices)):
            #update dp arrays
            hold[i+1] = max(hold[i],nothold[i-1]-prices[i])
            nothold[i+1] = max(nothold[i], hold[i]+prices[i])
            
        return nothold[-1]

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        #calculate the differences array, which is what i thought initially
        #now take as many subarrays with adjacent elemnts that have the biggest sum, but there is not a gap size of 1
        #[1,2,3,1,4]
        #[1,1,-2,3]
        n = len(prices)
        if n <= 1:
            return 0 #edge case
        
        #get the differences arrays
        diff = [prices[i+1] - prices[i] for i in range(n-1)]
        
        #create two arrays
        #dp is the max gain for the first i elements where we use the ith element
        #dp_max is the max gain for the first i where we can and cant use the ith element
        dp, dp_max = [0]*(n+1), [0]*(n+1)
        
        #want to include the last element and one after
        for i in range(n-1):
            #we take the ith element, which is just increaing it by diff[i] plus the option of skipping 2 elements or not skippingand taking the previous            
            dp[i] = diff[i] + max(dp_max[i-3], dp[i-1])
            dp_max[i] = max(dp_max[i-1], dp[i])
        
        return dp_max[-3]

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        https://twchen.gitbook.io/leetcode/dynamic-programming/best-time-to-buy-and-sell-stock-with-cooldown
        let stock[i+1] by the max profit as day i holding the stock
        let money[i+1] be the max profite at day w/o stock
        to have stock at day i we must have had stock at day i -1, so the profit can be stock[i]; or but at day[i,], which means we sold at i-1, and the profit it money[i-1]-prices[i]
        so stock[i+1] = max(stock[i], money[i-1] - prices[i])
        
        to not have stock at day i:
        we didnt have stock at day i-1 and dont buy either at day i, the the profit is just money[i-1]; or
        had stock at i-1 and cooled down, so the profit is stock[i-1]+prices[i]
        and money is money[i+1] = max(stock[i] + prices[i], money[i])
        '''
        if len(prices) <2:
            return 0
        stock = [0]*(len(prices)+1)
        money = [0]*(len(prices)+1)
        stock[1] = -prices[0]
        money[1] = 0
        for i in range(1,len(prices)):
            stock[i+1] = max(stock[i], money[i-1]-prices[i])
            money[i+1] = max(stock[i] + prices[i],money[i])
        return money[-1]
        

############################
#WORD BREAK II
############################
#nice attemp, but no dice
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        #traverse the wordDict
        #for each word in the dict see if i can find it in s, if found remove the leters
        #dump into a temp
        #remove each of the characetrs from s
        #i have to run this for every word
        #the first char in s has to match up to be used
        #i could sort the wordict first, so that the first char of words show up, if there isn't a matching first char, i cant split it up
        
        if len(wordDict) == 0 or s is None:
            return []
        
        output = []
        sorted(wordDict, key = lambda x:x[0] ==s[0])[::-1]
        
        #traverse each word, but only need to do the words starting with the first char in s
        for word in wordDict:
            #check
            if word[0] != s[0]:
                break
            s_chars = list(s)
            for word2 in wordDict:
                found_words = ""
                length = len(word2)
                if ''.join(s_chars[0:length-1]) == word2:
                    found_words += word2 + ' '
                    #remove from s_chars
                    s_chars = s[length-1:]
                #now to add back to ouput
                if len(s_chars) == 0:
                    output.append(found_words)
            wordDict.pop(0)
        return output

#DFS solution
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        '''
        and of course this another DP problem but we could also use recursion
        recursion review
        start with a single char and see if it is dictionary
        if its, go to the next one
        if s = 'abc'
        dict = {a,b,bc}
        rec(a) a in dict
        rec(bc) bc in dict
        rec(c) its not
        back to rec(bc)
        add [a,bc]
        rec(ab) fail
        rec(abc) fail
        
        first create a set of all the ltetters in the words
        then all leters in the string
        and return the empty list of the set difference is non empty
        then perform DFS on the reminang cases
        dfs takes in a list which we update, and if the index is equal to the end of string, we make the list of words found into a string add it to the final list to be returned, otherwise whenver a rod is found we add it to sofar and ind is incremented for the DFS to continue searchinf the string from that index
        '''
        word_letters = set(''.join(wordDict))
        wordDict = set(wordDict)
        stringLets = set(s)
        
        if stringLets - word_letters:
            return []
        
        word_list_list = []
        n = len(s)
        
        def DFS(so_far = [], ind = 0):
            if ind == n:
                word_list_list.append(' '.join(so_far))
                return
            for i in range(ind, n+1):
                if s[ind:i] in wordDict:
                    DFS(so_far+[s[ind:i]],i)
        DFS()
        return word_list_list

#another way
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """
        dp = {}
        def word_break(s):
            if s in dp:
                return dp[s]
            results = []
            for w in wordDict:
                if s[:len(w)] == w: #matching case
                    if len(w) == len(s): #when to stop recursing and adding the word
                        results.append(w)
                    else:
                        tmp = word_break(s[len(w):]) #if i am not at the ending case
                        for t in tmp:
                            results.append(w+" "+t) #adding back all the macthes
            dp[s] = results
            return results
        
        return word_break(s)
                    
        
        
        
















