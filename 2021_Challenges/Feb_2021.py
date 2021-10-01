###################
# Number of 1 Bits
###################
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        naive way would be to convert into to binary and just count up ones
        '''
        return Counter(bin(n)[2:])["1"]


class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        naive way would be to convert into to binary and just count up ones
        '''
        def rec_binary(n):
            if n == 0:
                return "0"
            else:
                return rec_binary(n//2) + str(n%2)
        
        return Counter(rec_binary(n))["1"]

class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        loop and flip, we keep checking each position with &
        we can maintain mask with bit shift left
        max is 32, so we run onl 32 times
        '''
        ones = 0
        mask = 1
        for i in range(32):
            #if its a 1
            if n & mask:
                ones += 1
            mask <<= 1
        
        return ones


class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        another way would be to talke n & n-1
        this flips the least siginigicnat 1 bit and also reduces n
        one n is zero, we know there are no numbers to flips
        ... 1 1 0 1 0 0 ... n
        ... 1 1 0 0 1 1 ... n - 1
        ... 1 1 0 0 0 0 ... n & n-1
        '''
        ones = 0
        while n:
            ones += 1
            n = n & n - 1
        return ones

#############################
#Squirrel Simulation
#############################
#so close 77/122!
#i think there is an error for some of the test casses
class Solution(object):
    def minDistance(self, height, width, tree, squirrel, nuts):
        """
        :type height: int
        :type width: int
        :type tree: List[int]
        :type squirrel: List[int]
        :type nuts: List[List[int]]
        :rtype: int
        """
        '''
        for each of the nuts, i cant get its l1 disatnce fromt eh squirrel
        i couild also get pairwaise distances for each of the nuts, same with squirrel to tree and tree to nuts
        i would simulate the squirrel moving
        start to first closest nut: +distance
        closest nut to tree: + distance
        tree to second closest nut: + distance
        second closet nut to tree +distance
        repeat until i get all the nuts, return distance
        heap
        '''
        distance  = 0
        #find closest nut to squrrel
        heap = []
        for nut in nuts:
            #distance from squirrel
            dist_sq = abs(squirrel[0] - nut[0]) + abs(squirrel[1] - nut[1])
            heappush(heap,(dist_sq,nut))
        #take first nust
        dist, nut = heappop(heap)
        distance += dist
        #now back to tree
        dist = abs(tree[0]-nut[0]) + abs(tree[1]-nut[1])
        distance += dist
        
        #now get the distance traveling for the remaining nuts
        while heap:
            _,nut = heappop(heap)
            #the squirrel has to travel twice the distance for each nut
            dist = abs(tree[0]-nut[0]) + abs(tree[1]-nut[1])
            distance += 2*dist
    
        return distance

class Solution(object):
    def minDistance(self, height, width, tree, squirrel, nuts):
        """
        :type height: int
        :type width: int
        :type tree: List[int]
        :type squirrel: List[int]
        :type nuts: List[List[int]]
        :rtype: int
        """
        '''
        it doesn't matter what order we go in reall, but to minimize the disance, want to start with the nut that is closer to the squirrel and nearest the tree
        for the first visited nut, given by d, the difference between the distance betweent the tree and current nut and the distance between the current nut andt he squirrel
        while traversing over the nuts array and adding to the distance, we find out the saving,d, which can be obtain if the squirrel goes to the current nut first.
        out of all the nust, we find out which maximise the saving and then decut the savin fromt eh sim total distance
        '''
        distance = 0
        offset = float('-inf')
        for nut in nuts:
            dist = abs(nut[0] - tree[0]) + abs(nut[1] - tree[1])
            distance += 2*dist
            nut_squirrel = abs(squirrel[0]-nut[0]) + abs(squirrel[1]-nut[1])
            #the offset if the maximum of the current offset or the distance btween
            #nut_tree and nut_squirrel
            offset = max(offset, dist - nut_squirrel)
        return distance - offset

#########################
#Trim Binary Search Tree
########################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def trimBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: TreeNode
        """
        '''
        if a nodes val is out of the limit, delete that node make the next  connection
        if a nodes val  is greater than high, we can prune left
        if a nodes val is lower than low, we can prunce right
        at each call give reference to the left and right subtrees recursively
        '''
        def dfs(node):
            if not node:
                return
            #refercence to subtrees 
            node.left = dfs(node.left)
            node.right = dfs(node.right)
            if node.val < low:
                #then anything to the left must be pruned
                return node.right
            elif node.val  > high:
                #everything to the right must be prine
                return node.left
            return node
        return dfs(root)
            
class Solution(object):
    def trimBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: TreeNode
        """
        '''
        if a nodes val is out of the limit, delete that node make the next  connection
        if a nodes val  is greater than high, we can prune left
        if a nodes val is lower than low, we can prunce right
        at each call give reference to the left and right subtrees recursively
        '''
        def dfs(node):
            if not node:
                return
            if node.val < low:
                return dfs(node.right)
            elif node.val > high:
                return dfs(node.left)
            else:
                node.left = dfs(node.left)
                node.right = dfs(node.right)
                return node
        
        return dfs(root)

####################
#Linked List Cycle
###################### Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        '''
        i can just traverse the linked list, dumping each node into a set
        if the node is already in the set return false
        otherwise we've gotten to the end
    
        '''
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        
        return False
        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        slow,fast = head,head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
                #found the point
                
        #if we've gotten to the tail witoutht finding a cycle
        if not fast.next or not fast.next.next:
            return False

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        slow = head
        fast = head
        #this works because if there isn't a cycle, fast has to envetuall get to the end
        #time complexit
        #N is nodes
        #M is the cycle
        #think about the case when fast next is none
        #and fast.next.next is also
        #we nneed to make sure the invariant holds
        while fast and fast.next:
            #in this case advnce first
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

###################################
# Longest Harmonious Subsequence
##################################
#brute force
class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        brute force
        we can generate all subsequences using the binary mark trick
        examples num =[5,9,6], we have 2^3 or 4 binary reps
        0 [0,0,0], []
        4 [1,0,0] = [5]
        7 [1,1,1], [5,9,6]
        '''
        N = len(nums)
        length = 0
        #numbers up to N-1 as binary
        for i in range(1<<N):
            current_length = 0 #keeping track of current length
            minn = float('inf') #updating these on the fly
            maxx = float('-inf')
            for j in range(N):
                #get the index which corresponds to number after convertin to binary
                if i & (1 << j) != 0:
                    current_length += 1
                    minn = min(minn,nums[j])
                    maxx = max(maxx,nums[j])
            #now check this
            if maxx - minn == 1:
                length = max(length, current_length)
        return length
        
#N squared
class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        n squared
        we examin each element
        for the elemnt i, we scan again increase the count when it meets the harmonicty criteria
        if we have found a s subequence meeting the criteria, we update the length
        
        '''
        length = 0
        N = len(nums)
        for i in range(N):
            curr_length = 0
            harmonic = False
            for j in range(N):
                #we would take this elment if nums[i] == nums[j] to add to the length
                #but only update when we have found a harmonic seuqnece
                if nums[i] == nums[j]:
                    curr_length += 1
                elif nums[j] + 1 == nums[i]:
                    curr_length += 1
                    harmonic = True
            if harmonic:
                length = max(length, curr_length)
        
        return length

#using hash
class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        using hash
        we want the max and min elements to be within one of each other
        for each element in nums, check to see how many of its n-1 and n+1 are in there
        what if we counted up nums
        '''
        counts = defaultdict(int)
        uniques = set(nums)
        for num in nums:
            counts[(num,num+1)] += 1
            counts[(num-1,num)] += 1
        
        max_length = 0
        unqs = set(nums)
        for k,v in counts.items():
            if k[0] in unqs and k[1] in unqs:
                max_length = max(max_length,v)
        return max_length

class Solution(object):
    def findLHS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        counts = Counter(nums)
        
        output = 0
        for k,v in counts.items():
            if k + 1 in counts:
                output = max(counts[k]+counts[k+1],output)
        return output
            

##################
#Simplify Path
##################
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        '''
        paths always start with /
        '.' refers to directory
        '..' referts up two levels and so on
        // are reduced to /
        im thinin stack
        we keep pushing on to the stack when we encounter . / we have some operations to do
        '''
        stack = []
        for char in path:
            if char == '/':
                while stack and stack[-1] == '/':
                    stack.pop()
                stack.append(char)
            elif char == '.':
                #clear the whole thing
                while stack:
                    stack.pop()
            else:
                stack.append(char)
        stack = "".join(stack)
        return stack

class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        '''
        paths always start with /
        '.' refers to directory
        '..' referts up two levels and so on
        // are reduced to /
        im thinin stack
        we keep pushing on to the stack when we encounter . / we have some operations to do
        we split in /
        when we see .. that just means we go back up a level and no longer in that director
        '''
        path = path.split('/')
        stack = []
        for p in path:
            if p == "..":
                #keeping popping from out stack, clear the stack
                if stack:
                    stack.pop()
            elif p == '.' or not p:
                continue
            else:
                stack.append(p)
        return "/"+"/".join(stack)

#another way
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        path = [foo for foo in path.split('/') if foo != '']
        stack = []
        for p in path:
            if p == '..':
                if stack:
                    stack.pop()
            elif p == '.':
                continue
            else:
                stack.append(p)
        return '/'+'/'.join(stack)

#############################
#Binary Tree Right Side View
############################
#won't work in certain edge cases
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        '''
        we can only go right
        we can never see anything on the left
        it could be the case where we go left once, but nothing on the right
        '''
        rights = []
        def go_right(node):
            if not node:
                return
            rights.append(node.val)
            go_right(node.right)
        go_right(root)
        return rights

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        '''
        if did BFS, level by level,
        well you would only want the ones all the way to the right! 
        '''
        if not root:
            return []
        q = deque([root])
        
        rights  = []
        while q:
            #we need to make sure we get the one all the way on the right
            N = len(q)
            for i in range(N):
                current = q.popleft()
                if i == N-1:
                    rights.append(current.val)
                if current.left:
                    q.append(current.left)
                if current.right:
                    q.append(current.right)
        return rights

class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        '''
        well i did it with one q keeping track of size, but lets try two qs
        one q for current level, another q for second level
        the idea is to pop from current level and move to next level
        Initiate the list of the right side view rightside.

    Initiate two queues: one for the current level, and one for the next. Add root into nextLevel queue.

    While nextLevel queue is not empty:

    Initiate the current level: currLevel = nextLevel, and empty the next level nextLevel.

    While current level queue is not empty:

    Pop out a node from the current level queue.

    Add first left and then right child node into nextLevel queue.

    Now currLevel is empty, and the node we have in hands is the last one, and makes a part of the right side view. Add it into rightside.

    Return rightside.
        '''
        if not root:
            return
        rights = []
        next_level = deque([root])
        
        while next_level:
            #set current to next
            current_level = next_level
            #make new next
            next_level = deque()
            while current_level:
                curr_node = current_level.popleft()
                #usual BFS
                if curr_node.left:
                    next_level.append(curr_node.left)
                if curr_node.right:
                    next_level.append(curr_node.right)
            rights.append(curr_node.val)
            
        return rights

#DFS
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        '''
        recursively...
        we traverse the tree level by level starting each time forom the right most child
        '''
        if not root: 
            return []
        
        rights = []
        
        def dfs(node,level):
        	#becasue we have reached this level from the right the first time
        	#we reach a newpth, and since we are starting from the right we need to add it
            if level == len(rights):
                rights.append(node.val)
            #we always start going right
            if node.right:
                dfs(node.right,level+1)
            if node.left:
                dfs(node.left, level+1)
        
        dfs(root,0)
        return rights

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        '''
        anothey dfs by level and each level key into a hash the value of the node
        we are doing pre order first
        '''
        mapp = defaultdict(list)
        def dfs(node,level):
            if not node:
                return
            mapp[level].append(node.val)
            dfs(node.left,level+1)
            dfs(node.right,level+1)
            
        dfs(root,0)
        return [v[-1] for k,v in sorted(mapp.items())]

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        '''
        anothey dfs by level and each level key into a hash the value of the node
        we are doing pre order first
        '''
        mapp = defaultdict(int)
        def dfs(node,level):
            if not node:
                return
            if level not in mapp:
                mapp[level] = node.val
            dfs(node.right,level+1)
            dfs(node.left,level+1)
            
        dfs(root,0)
        return [v for k,v in mapp.items()]

##################################
#Shortest Distance to a Character
##################################
#close one 18 of 76
class Solution(object):
    def shortestToChar(self, s, c):
        """
        :type s: str
        :type c: str
        :rtype: List[int]
        """
        '''
        this is just a distance array for each char in s to the nearest c
        we would have to keep going left and right from each s, if we can
        '''
        distances = []
        N = len(s)
        for i in range(N):
            if s[i] == c:
                distances.append(0)
            elif i == 0:
                for j in range(i+1,N):
                    if s[j] == c:
                        break
                distances.append(abs(i-j))
            elif i == N-1:
                for j in range(N-2,-1,-1):
                    if s[j] == c:
                        break
                distances.append(abs(i-j))
            else:
                l,r = i,i
                while l >= 0 and r <= N-1:
                    if s[l] == c or s[r] == c:
                        break
                    else:
                        l -= 1
                        r += 1
                distances.append(min(abs(i-l),abs(i-r)))
        return distances

#two min arrays
class Solution(object):
    def shortestToChar(self, s, c):
        """
        :type s: str
        :type c: str
        :rtype: List[int]
        """
        '''
        we pass the array twice
        on the first pass we check how far the current char is to c (going left to right) and dump each distance into a min array
        when we pass the array we keep track of the position c lies in and update anytime we see it
        on the second pass (right to left)
        we put into the array at i the minimum of the two distances
        '''
        prev = float('-inf')
        distances = []
        for i,char in enumerate(s):
            if char == c:
                prev = i
            distances.append(i - prev) #we want this positive since its i - prev, thats way prev is -in for this pass
        #print distances
        #reset prev to this time poistive inf
        prev = float('inf')
        for i in range(len(s)-1,-1,-1):
            if s[i] == c:
                prev = i
            #prev - i will be positive inf, allow min to take the min distance
            #remember this trick
            distances[i] = min(distances[i],prev-i)
        return distances

class Solution(object):
    def shortestToChar(self, s, c):
        """
        :type s: str
        :type c: str
        :rtype: List[int]
        """
        '''
        ok lets just write this out the long way
        '''
        N = len(s)
        lefts,rights,dists = [N]*N,[N]*N,[N]*N
        
        #left to right
        temp = float('inf')
        for i in range(N):
            if s[i] == c:
                temp = 0
            lefts[i] = temp
            temp += 1
        #right to left
        temp = float('inf')
        for i in range(N-1,-1,-1):
            if s[i] == c:
                temp = 0
            rights[i] = temp
            temp += 1
        for i in range(N):
            dists[i] = min(lefts[i],rights[i])
        return dists

class Solution(object):
    def shortestToChar(self, s, c):
        """
        :type s: str
        :type c: str
        :rtype: List[int]
        """
        '''
        ok lets just write this out the long way
        '''
        N = len(s)
        dists = [N]*N
        
        #left to right
        temp = float('inf')
        for i in range(N):
            if s[i] == c:
                temp = 0
            dists[i] = temp
            temp += 1
        #right to left
        temp = float('inf')
        for i in range(N-1,-1,-1):
            if s[i] == c:
                temp = 0
            dists[i] = min(dists[i],temp)
            temp += 1
        return dists

class Solution(object):
    def shortestToChar(self, s, c):
        """
        :type s: str
        :type c: str
        :rtype: List[int]
        """
        '''
        ok lets just write this out the long way
        '''
        N = len(s)
        dists = [N]*N
        
        #left to right
        temp = float('-inf')
        for i in range(N):
            if s[i] == c:
                temp = i
            dists[i] = i - temp
        #right to left
        temp = float('inf')
        for i in range(N-1,-1,-1):
            if s[i] == c:
                temp = i
            dists[i] = min(dists[i],temp-i)
        return dists

#################
#Peeking Iterator
#################
#not bad, but kinda cheeky lol

# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.itr = iterator
        #just dump the contents into another container
        self.items = []
        while self.itr.hasNext():
            self.items.append(self.itr.next())
        
        self.N = len(self.items)
        self.ptr = 0

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.items[self.ptr]

    def next(self):
        """
        :rtype: int
        """
        output = self.items[self.ptr]
        self.ptr += 1
        return output

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.ptr < self.N
      
# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].

#An aside on iterators
'''
convert an an array using constant space
'''
class LinkedListIterator:
	def __init__(self,head):
		self.node = head
	def hasNext(self):
		return self.node is not None
	def next(self):
		result = self.node.value
		self.node = sefl.node.next
		return result

#we need to give reference to the next variable everytime 
#lese it would get destroyed

class RangeIterator:
	def __init__(self,minn,maxx):
		self.maxx = maxx
		self.current = minn
	def hasNext(self):
		return self.current < self.maxx
	def next(self):
		self.current += 1
		return self.current - 1


foo = RangeIterator(minn=0,maxx = 10)
print(foo.next())
while foo.hasNext():
	print(foo.next())

class SquaresIterator:
    def __init__(self):
        self._n = 0

    def hasNext(self):
        # It continues forever, 
        # so definitely has a next!
        return True

    def next(self):
        result = self._n
        self._current += 1
        return result ** 2

#Saving Peeked Value
class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        #init a peeked value
        self.peeked_val = None
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        #we check if we have stored a valute to peek at
        if not self.peeked_val:
            if not self.iter.hasNext():
                #if there is no value of there is a next error out
                raise StopIteration()
            #otheriwse we can get teh next
            self.peeked_val = self.iter.next()
        #otherwise return the already peekdvale
        return self.peeked_val
        

    def next(self):
        """
        :rtype: int
        """
        #we need to check if we already have a value stored in peekf
        if self.peeked_val:
            output = self.peeked_val
            #reset 
            self.peeked_val = None
            return output
        #if there isn't a value we need to get it
        #but make sure we can
        if not self.iter.hasNext():
            raise StopIteration()
        
        return self.iter.next()

    def hasNext(self):
        """
        :rtype: bool
        """
        #if there is nothing to get next or peeked is null
        return self.peeked_val is not None or self.iter.hasNext()

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        '''
        instead of only storing the next value after we'eve picked it
        we can just assign next to global var in the constructor
        '''
        self.iter = iterator
        self.next = iterator.next()
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.next
        

    def next(self):
        """
        :rtype: int
        """
        if not self.next:
            raise StopIteration()
        output = self.next
        #store the next value again
        if self.iter.hasNext():
            self.next = selt.iter.next()
        return output
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.next is not None

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.itr = iterator
        self.peeked = False
        self.val = None
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        #if we have peeked we must have v al
        #otherwise peek and assing the val
        if self.peeked == True:
            return self.val
        else:
            self.peeked = True
            self.val = self.itr.next()
            return self.val
        

    def next(self):
        """
        :rtype: int
        """
        if self.peeked == True:
            self.peeked = False
            return self.val
        else:
            return self.itr.next()

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.peeked or self.itr.hasNext():
            return True
        else:
            return False

#writing to a buffer first

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        if self.iter.hasNext():
            self.buffer = self.iter.next()
        else:
            self.buffer = None
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.buffer
        

    def next(self):
        """
        :rtype: int
        """
        temp = self.buffer
        if self.iter.hasNext():
            self.buffer = self.iter.next()
        else:
            self.buffer = None
        return temp
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.buffer is not None

##############################
#Convert BST to Greater Tree
###############################
#214/215!
# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        '''
        a greater tree is the value from the orig BST, plus sum all elements greater than that node
        for each node, we would dfs all and get the sum for all elements on the right
        then that value would become the node
        then just dfs all the way down the tree
        at each node keep track of the left and right sums
        thats too tricky, how but for each node we visit, we find its greater sum
        '''
        def greater_sum(candidate,node):
            self.summ = 0
            def helper(candidate,node):
                if not node:
                    return 
                if node.val > candidate:
                    self.summ += node.val
                helper(candidate,node.left)
                helper(candidate,node.right)
            helper(candidate,node)
            return self.summ + candidate
        
        vals = []
        def dfs(node):
            if not node:
                return
            vals.append(greater_sum(node.val,root))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        
        vals = deque(vals)
        def dfs_2(node):
            if not node:
                return
            node.val = vals.popleft()
            dfs_2(node.left)
            dfs_2(node.right)
            
        dfs_2(root)
        return root

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        '''
        a node's greater sum is just the entire sum of the right subtree
        more specifically, a parent node is the just the sum of the right subtree elements plus its element
        a nodes left child is just its sum plues everything on the right
        node.val = sum(node.right) + node.val
        node.left.val = sum(node.left.right) + node.left.val + node.val
        node.right.val = node.val + sum(node.right)
                def right_sum(node,right_sum):
            if not node:
                return
            if node.left and node.right:
                node.val = node.val + right_sum(node.right)
                node.left.val = node.left.val + right_sum
        
        
        we maintain some minor global state so each recurive call can access and modify the current total sum
        this way we ensure that the current node exists, recurse on the right, and visit the left subtree
        if we know that recursing on root.right properuly updates the right subtree and that recursing left update the left subtree, then we are guaranteeds to update all nodes with larger values before the current node and all nodes with smaller values after
        '''
        self.total = 0

class Solution(object):
    def convertBST(self, root):
        
        def dfs(node):
            if not node:
                return
            dfs(node.right)
            self.total += node.val
            node.val = self.total
            dfs(node.left)
            return root
        
        return dfs(root)



        def dfs(node,accum):
            if not node:
                return accum
            accum = dfs(node.right,accum)
            accum += node.val
            node.val = accum
            accum = dfs(node.left,accum)
            return accum
        
        dfs(root,0)
        return root
        
###########################
#Number of Distinct Islands
###########################
#this is the solution for counting islands
#but not for distinct islands
#NEED DISTINCT
class Solution(object):
    def numDistinctIslands(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        '''
        brute force dfs
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        self.islands = 0
        
        seen = set()
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(row,col):
            if row < 0 or col < 0 or row >= rows or col >= cols:
                return
            if (row,col) in seen:
                return
            seen.add((row,col))
            for dx,dy in dirrs:
                new_row = row + dx
                new_col = col + dy
                if new_row < 0 or new_col < 0 or new_row >= rows or new_col >= cols:
                    return
                    if (new_row,new_col) not in seen and grid[new_row][new_col] == 1:
                        seen.add((new_row,new_col))
                        dfs(new_row,new_col)
            self.islands += 1
            
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    dfs(i,j)

class Solution(object):
    def numDistinctIslands(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        '''
        brute force dfs, we already reviewed the problem how to count islands
        now we need to identify if islands are unique
        inutition:
            we've already used DFS to make a list of islands, where each island is repsrsented as a list of coordinates
            if two islandsa are the same, their cooridantes can be translated
            example: [(2, 1), (3, 1), (1, 2), (2, 2)], it would become [(1, 0), (2, 0), (0, 1), (1,1)] when anchored at the top-left corner.
            you just offset by subtracting the current row and current col for each of the cooridnates
            we are anchording at the top left corner
        algo:
            1. use DFS to make a list of islands, where each island is list of coordinates (top left 
            anchored)
            2. init uniqe count of islands to zero
            3. after genrating a candidate island, compare to every island in the uniqe set, if it matches, do not add it
            4. just return the length of the list of island coordinate
        '''
        rows = len(grid)
        cols = len(grid[0])
        #this time is a boolean array
        seen = [[False]*cols]*rows
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        #dfs call
        def dfs(row,col):
            if row < 0 or col < 0 or row >= rows or col >= cols:
                return
            if seen[row][col] == True or grid[row][col] == 0:
                return
            seen[row][col] = True
            #current island defind outside
            current_island.append((row - row_origin, col - col_origin))
            for dx,dy in dirrs:
                dfs(row+dx,col+dy)
        
        #unique checker function, for each created current island, compare to unique island
        def is_unique():
            for other_island in unique_islands:
                if len(other_island) != len(current_island):
                    continue
                for cell_1, cell_2 in zip(current_island, other_island):
                    if cell_1 != cell_2:
                        break
                else:
                    return False
            return True
            
        unique_islands = []
        #rpeated call dfs
        for i in range(rows):
            for j in range(cols):
                current_island = []
                row_origin = i
                col_origin = j
                dfs(i,j)
                if not current_island or not is_unique():
                    continue
                unique_islands.append(current_island)
        return len(unique_islands)
                    
        return self.islands

#################################
#Copy List with Random Pointer
##################################
#cheeky way
"""
# Definition for a Node.
class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        return copy.deepcopy(head)
        
#almost....
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        '''
        i can first make copy of nodes with next
        then i can go back to head and move random
        what if had two hash maps, 
        one would be node:next and the other node.random, i can make these by first passing the through the head
        all i really care about are the valeues, 
        then pass through the head again makign the structure
        '''
        mapp_nexts = {}
        mapp_randoms = {}
        temp = head
        while temp:
            mapp_nexts[temp] = temp.next
            mapp_randoms[temp] = temp.random
            temp = temp.next
        
        dummy = Node(x=0)
        temp = dummy
        

        #reset temp to build dummy and rebuild on to dumm
        while head:
            temp.val = head.val
            #now get the nodes
            node_next = mapp_nexts[head]
            node_random = mapp_randoms[head]
            
            #set vals
            temp.next = Node(x=node_next.val if node_next else None)
            temp.random = Node(x = node_random.val if node_random else None)
            temp = temp.next
            head = head.next
        return dummy

#recursively
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        '''
        there are a couple of approaches here, so lets go through all of them
        we can use recursion and think of this like a graph
        of course the cycles make it harder, but we can use a visited set
        all we do here is just traverse and clone
        cloning means for every unseen node, we make a enw one
        traverse happens in a dfs manner
        algo:
            1. start traversing from the head
            2. if we already have a cloned copy of the currnet node in teh visited dict, use the cloned as referece
            3. if we haven't made a clone, we make a new one and add it
            4. we then reurse twice, one using random pointer and one for next
                we are mkaing the recursive calls for the children of the curent node
        '''
        #hash holding old nodes as keys and new nodes as values
        cloned_visited = {}
        def dfs(node):
            if node == None:
                return None
            #if we have seen this node, just return it
            if node in cloned_visited:
                return cloned_visited[node]
            #otherwise we have work to do
            #make new one and save into our hash
            cloned_node = Node(node.val, None,None)
            #store it
            cloned_visited[node] = cloned_node
            
            #recurse
            #copy remainined linked list starting once form teh next pointer and then randome pointer
            cloned_node.next = dfs(node.next)
            cloned_node.random = dfs(node.random)
            return cloned_node
        return dfs(head)

#iterative O(N) time and O(N) space
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        '''
        pass head twice connected next node first
        then connect random
        '''
        #generate new list update hash
        #when we update our hash, we make a new node first and then mape it back to the old
        dummy = Node(-1)
        new = dummy
        temp = head
        old_new = {}
        
        #first pass, create lists with next but hash along the way
        while temp:
            #make the new connection
            new.next = Node(temp.val)
            #mapp old node to new
            old_new[temp] = new.next
            temp = temp.next
            new = new.next
        
        #now connect randoms
        temp = head
        #why in this case it is dummy.next????
        #we made it so that the head is located at dummy.next
        new = dummy.next
        while temp:
            if temp.random:
                new.random = old_new[temp.random]
            temp = temp.next
            new = new.next
        return dummy.next

#O(1) space  iterattive
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        '''
        instead of using a hash and O(N) space
        Old List: A --> B --> C --> D
InterWeaved List: A --> A' --> B --> B' --> C --> C' --> D --> D'
where primes indicate the copies
the interweaving is done using the next pointers and we can make use of the interweaved structure 
this becomes our hash
1. insert new nodes into list
2. update random pointers to to pointt o enw list
3. remove old nodes
        '''
        #interweave
        dummy = Node(-1)
        dummy.next = head
        cur = head
        
        while cur:
            #store for new node
            copy = Node(cur.val)
            copy.next = cur.next
            cur.next = copy
            cur = copy.next

        #update random pointers, we no longer have hash, but our values are in the linked list now
        cur = head
        while cur:
            #only the old ones would have a random pointer
            if cur.random:
                #we go to the new one, and set is random as the old's next
                cur.next.random = cur.random.next
            #skip over old ones advance twice
            cur = cur.next.next
            
        #remove old nodes
        cur = dummy
        old = head
        while old:
            cur.next = old.next #build dummy by getting the old node
            cur = old
            old = cur.next
        return dummy.next

##############
#Valid Anagram
##############
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        '''
        just compare hash of counts
        '''
        return Counter(s) == Counter(t)

###############################################
#  Number of Steps to Reduce a Number to Zero
#############################################
class Solution(object):
    def numberOfSteps (self, num):
        """
        :type num: int
        :rtype: int
        """
        '''
        from the hint, it just say simulate
        rules:
         1. if num % 2 == 0, divide by two
         2. else subtract 1
        '''
        
        steps = 0
        
        while num > 0:
            if num % 2 == 0:
                num //= 2
            else:
                num -= 1
            print num
            steps += 1
            
        return steps

class Solution(object):
    def numberOfSteps (self, num):
        """
        :type num: int
        :rtype: int
        """
        '''
        a follow up, notice that when when we divide the number by two we just shift the bits right by 1
        and when we subtract 1, we just pop it of
        so the number of steps to reduct it to zero would be just the length of the binary number plus the number of ones
        i can get the length of the number in binary with floor(log2(num))
        how to get ones?
        
        '''
       	if num == 0:
            return 0
        import math 
        def get_ones(num):
            ones = 0
            while num > 0:
                ones += num & 1
                num = num >> 1
            return ones
        return int(math.floor(math.log(num,2))) + get_ones(num) 

#####################################
#Shortest Path in Binary Matrix
##################################
#closeeee 55/84
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        '''
        hint says BFS
        each cell is a node with an edge of 1
        BFS from start to end and return the path length
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dirrs = [(i,j) for i in (-1,0,1) for j in (-1,0,1) if (i,i)]
        
        end = (rows-1,cols-1)
        
        visited = set()
        
        q = deque([(0,0,1)])
        
        if grid[0][0] != 0:
            return -1
        
        while q:
            x,y,distance = q.popleft()
            if (x,y) == end:
                return distance
            #mark as visited
            visited.add((x,y))
            
            for dx,dy in dirrs:
                new_x = x + dx
                new_y = y + dy
                #check bounds
                if new_x >= 0 and new_x < rows and new_y >= 0 and new_y < cols:
                    #check if zero and not visited
                    if grid[new_x][new_y] == 0 and not (new_x,new_y) in visited:
                        q.append((new_x,new_y,distance + 1))
        
        return -1

#####so i gues i need to learn A*
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        
        dirrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        #helper function to yeild neirhgbors of a cells
        def get_neighbors(row,col):
            for dx,dy in dirrs:
                new_x = row + dx
                new_y = col + dy
                if 0 <= new_x < rows and 0 <= new_y < cols and grid[new_x][new_y] == 0 and (new_x,new_y) not in visited:
                    yield (new_x,new_y)
        
        if grid[0][0] != 0 or grid[rows-1][cols-1] != 0:
            return -1
        
        q = deque([(0,0,1)])
        visited = set()
        visited.add((0,0))
        #grid[0][0] = 1
        while q:
            x,y,distance = q.popleft()
            if (x,y) == (rows-1,cols-1):
                return distance
            for neigh in get_neighbors(x,y):
                visited.add((x,y))
                #grid[new_x][new+y] = 1
                q.append((*neigh,distance+1))
        return -1


#####################
# Is Graph Bipartite?
######################
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        '''
        bipartite means every edge in the graph connects a node in set A and a node in set B
        this is silly:
        we can mark each node with a different color, -1 to one set, 1 to the other
        then we can use dfs to visit the nodes and check if we haven't marked it and haven't seen it it must be other other color
        if we get to the case where we have seen it and it hasn't been marked it must not be bipartite?
        i still don't get how though...if it's connected, it NEEDS to be a different color, then we can return False
        no two adjacent vertices can be the same color! - KEY
        we attempt to 2 color the graph by traversing the graph and marking the neighbords of node to a different color than the one node we are at marked of a different color
        if we succseflly 2 color the graph, we can ren turn True, but if we come across two adjacent verticses of the same color, we must return False
        DFS first
        mark with 1 and -1
        
        '''
        
        colors = {}
        
        def dfs(node):
            #examine nodes neighrbos
            for neighbor in graph[node]:
                #check we have seen it 
                if neighbor in colors:
                    #color check
                    if colors[neighbor] == colors[node]:
                        return False
                #mark with different color
                else:
                    colors[neighbor] = colors[node]*(-1)
                    #then just recurse
                    if dfs(neighbor) == False:
                        return False
            #we finish exploring and color this node's path/neighbor it passes!
            return True
        
        #invoke for each nieghtbor
        for i in range(len(graph)):
            if i not in colors:
                colors[i] = 1
                if dfs(i) == False:
                    return False
        return True
        
#for some reason, on this case iterative stack DFS == BFS with q
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        '''
        bipartite means every edge in the graph connects a node in set A and a node in set B
        this is silly:
        we can mark each node with a different color, -1 to one set, 1 to the other
        then we can use dfs to visit the nodes and check if we haven't marked it and haven't seen it it must be other other color
        if we get to the case where we have seen it and it hasn't been marked it must not be bipartite?
        i still don't get how though...if it's connected, it NEEDS to be a different color, then we can return False
        no two adjacent vertices can be the same color! - KEY
        we attempt to 2 color the graph by traversing the graph and marking the neighbords of node to a different color than the one node we are at marked of a different color
        if we succseflly 2 color the graph, we can ren turn True, but if we come across two adjacent verticses of the same color, we must return False
        DFS first
        mark with 1 and -1
        
        DFS with stack
        '''
        colors = {}
        N = len(graph)
        
        for node in range(N):
            #if we haven't seen it we need to dfs
            if node not in colors:
                stack = [node]
                colors[node] = 1
                while stack:
                    #dfs
                    current_node = stack.pop()
                    for neigh in graph[current_node]:
                        #not marked or seen, push back on to stack and mark it
                        if neigh not in colors:
                            stack.append(neigh)
                            colors[neigh] = colors[current_node]*(-1)
                        elif colors[neigh] == colors[current_node]:
                            return False
        return True

#we need to dfs on each node beause it would be the case there may be unconnect parts
#what if there were part of the graph that were unconnected?
#well if we just DFS on the connected part, and that part was bipartite, we wan't say that?
#a graph with more than 1 disconnected parts cannot be bipartite
class Solution(object):
    def isBipartite(self, graph):
        colors = {}
        N = len(graph)
        
        for node in range(N):
            if node not in colors:
                colors[node] = 1
                stack = [node]
                while stack:
                    current = stack.pop()
                    for neigh in graph[current]:
                        if neigh in colors:
                            if colors[neigh] == colors[current]:
                                return False
                        else:
                            colors[neigh] = colors[current]*(-1)
                            stack.append(neigh)
        return True


##################################
# The K Weakest Rows in a Matrix
##################################
class Solution(object):
    def kWeakestRows(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[int]
        """
        '''
        for each row, count up the ones and push the value into a hash row_idx:counts 1
        then sort and grab the ks
        '''
        counts = {}
        rows = len(mat)
        cols = len(mat[0])
        
        for i in range(rows):
            count = 0
            for j in range(cols):
                if mat[i][j] == 1:
                    count += 1
            
            counts[i] = count
        
        output = []
        
        for kv in sorted(counts.items(),key=lambda kv: kv[1], reverse=False):
            output.append(kv[0])
        
        return output[:k]

#more pythonic
class Solution(object):
    def kWeakestRows(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[int]
        """
        strengths = [(sum(row),i) for i,row in enumerate(mat)]
        strengths.sort()
        
        return [idx for foo,idx in strengths[:k]]

'''
notes on time complexity
counting O(cols)
sorting O(rowslong(rows))
O(row+cols) + O(cols) + O(rowslog(rows)) 
'''

#using binary search
class Solution(object):
    def kWeakestRows(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[int]
        """
        '''
        we can count using binary serach?
        use binary search to find the first zero in each row
        then use index calculations to get the number of ones, i.e just return the index
        instead of returning a mid value here, we keep binary seraching until we get a zero and the left of the zero is 1 and right a zero
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        def binary_search_count(row):
            lo = 0
            high = len(row) - 1 
            while lo < high:
                mid = lo + (high - lo) // 2
                if row[mid] == 1:
                    lo = mid + 1
                else:
                    high = mid
            #in this case we retorn our left pointer, but it would;nt matter in thise cse
            return lo
        
        row_strengths =  [(binary_search_count(row),i) for i,row in enumerate(mat)]
        
        #dfault sorts by first value
        row_strengths.sort()
        
        return [row for strength, row in row_strengths[:k]]

#pq with binary search
class Solution(object):
    def kWeakestRows(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[int]
        """
        '''
        we can count using binary serach?
        use binary search to find the first zero in each row
        then use index calculations to get the number of ones, i.e just return the index
        instead of returning a mid value here, we keep binary seraching until we get a zero and the left of the zero is 1 and right a zero
        #addition we can also use a heapq of fixed size k
        then just pop off the heap whenever k gets to big
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        def binary_search_count(row):
            #notice here for this binary search implementation, we go up N
            #not N -1, N being the length of the row
            low = 0
            high = len(row)
            while low < high:
                mid = low + (high - low) // 2
                if row[mid] == 1:
                    low = mid + 1
                else:
                    high = mid
            return low
        
        pq = []
        #note when heapq pushed a tuple, it takes into consideration the ordering for all elements in the tuple
        for i,row in enumerate(mat):
            strength = binary_search_count(row)
            entry = (-strength,-i)
            #contrain k and also there is no need to push a row strength if it isn't as strong a the weakest row in pq
            if len(pq) < k or entry > pq[0]:
                heappush(pq,entry)
            if len(pq) > k:
                heappop(pq)
                
        #pull the indices and renegate
        idxs = []
        while pq:
            idxs.append(-heappop(pq)[1])
        return idxs[::-1]

###################
# Kill Process
###################
class Solution(object):
    def killProcess(self, pid, ppid, kill):
        """
        :type pid: List[int]
        :type ppid: List[int]
        :type kill: int
        :rtype: List[int]
        """
        '''
        the index who's ppid  0 is the root of the tree
        i read pid as for each p in pid, comes from ppid at index(p)
        do i need to recursiely make the tree????????
        find the ppid == 0
        create adj_list first
        '''
        N = len(pid)
        adj_list = defaultdict(list) #parent:child
        for i in range(N):
            adj_list[ppid[i]].append(pid[i])
        
        #now this just becomes a graph, explore using...DFS!
        #once i get to the kill value, add it to a container and keep recursing
        #since we have the hash, we can just dfs starting at the kill node!
        #fuckkkk!!
        
        def dfs(node):
            #invoke at kill node
            killed = []
            for child in adj_list[node]:
                killed += dfs(child)
            #add in the node once we are done recursing
            killed.append(node)
            return killed
        
        return dfs(kill)

#another way doing it globally instead
class Solution(object):
    def killProcess(self, pid, ppid, kill):
        """
        :type pid: List[int]
        :type ppid: List[int]
        :type kill: int
        :rtype: List[int]
        """
        '''
        the index who's ppid  0 is the root of the tree
        i read pid as for each p in pid, comes from ppid at index(p)
        do i need to recursiely make the tree????????
        find the ppid == 0
        create adj_list first
        '''
        N = len(pid)
        adj_list = defaultdict(list) #parent:child
        for i in range(N):
            adj_list[ppid[i]].append(pid[i])
        
        #now this just becomes a graph, explore using...DFS!
        #once i get to the kill value, add it to a container and keep recursing
        #since we have the hash, we can just dfs starting at the kill node!
        #fuckkkk!!
        
        killed = []
        
        def dfs(node):
            for child in adj_list[node]:
                dfs(child)
            killed.append(node)
        
        dfs(kill)
        return killed

#brute force recursive
class Solution(object):
    def killProcess(self, pid, ppid, kill):
        """
        :type pid: List[int]
        :type ppid: List[int]
        :type kill: int
        :rtype: List[int]
        """
        '''
        lets go over some of these approaches one by one
        BRUTE FORCE RECURSIVE
        inution; 
            find the kill node and kill all its children
            traverse the ppid and find out all the children processed to be killed
            we recurse for every child node
        
        '''
        self.killed = []
        if kill == 0:
            return self.killed
        self.killed.append(kill)
        for i in range(len(ppid)):
            if ppid[i] == 0:
                self.killed += self.killProcess(pid,ppid,pid[i])
        return self.killed

#iteratrive DFS stack
class Solution(object):
    def killProcess(self, pid, ppid, kill):
        """
        :type pid: List[int]
        :type ppid: List[int]
        :type kill: int
        :rtype: List[int]
        """
        '''
        DFS but using stack
        '''
        N = len(ppid)
        adj_list = defaultdict(list)
        for i in range(N):
            adj_list[ppid[i]].append(pid[i])
            
        killed = []
        
        #since i start at killed anyway, youre gonna have to add them the killed list
        #what if i didn't start at the kill node?
        #start from the root, then dfs until i get the kill node
        #stop, then it just becomes this problem!
        #note, you could also solve this using a deque
        stack = [kill]
        
        while stack:
            current = stack.pop()
            killed.append(current)
            for child in adj_list[current]:
                stack.append(child)
        return killed

#########################
# Letter Case Permutation
########################
#recursive with global results
class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        '''
        for every char in string S, if its alpha, we can mutate it by lowering or uppering
        recursion
        not you cannot mutate a string
        '''
        N = len(S)
        results = []
        
        def mutate(idx,substring):
            if len(substring) == N:
                results.append(substring)
            else:
                #get the char
                char = S[idx]
                if char.isalpha():
                    if char.islower():
                        mutate(idx+1, substring+char.upper())
                    else:
                        mutate(idx+1, substring+char.lower())
                #once we are done chekcing the cases we need to backtrack and add the last char
                #FUCK YEAH!!!!!
                mutate(idx+1,substring+char)
        mutate(0,'')
        return results

#recusrive non global
class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        '''
        for every char in string S, if its alpha, we can mutate it by lowering or uppering
        recursion
        not you cannot mutate a string
        #now trying doing it where you don't have a global results
        there are 2^N function calls in the eexecution tree
        each call fors though the length N 2^n times N

        '''
        N = len(S)
        
        def mutate(idx,substring):
            results = []
            if idx == N:
                return [substring]
            else:
                #get the char
                char = S[idx]
                if char.isalpha():
                    if char.islower():
                        results += mutate(idx+1, substring+char.upper())
                    else:
                        results += mutate(idx+1, substring+char.lower())
                #once we are done chekcing the cases we need to backtrack and add the last char
                #FUCK YEAH!!!!!
                results += mutate(idx+1,substring+char)
                return results
        return mutate(0,'')

#iterative cascading
class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        '''
        another doing this iteratively
        this is similar to the subsets iterative problem
        CASCADING
        example, say we are give S ='abc'
        first maintain
        and ['']
        ['a','A']
        ['ab','aB'] + ['Ab','AB']
        ['ab','aB','Ab','AB']
        ['abc','aBc','Abc','ABc'] + ['abC', 'aBC','AbC', 'ABC']
        ['abc','aBc','Abc','ABc', 'abC', 'aBC','AbC', 'ABC']
        '''
        results = [[]]
        for char in S:
            N = len(results)
            if char.isalpha():
                for i in range(N):
                    results.append(results[i][:])
                    results[i].append(char.lower())
                    #make the next set of lists, double index by N+i
                    results[N+i].append(char.upper())
            else:
                for i in range(N):
                    results[i].append(char)
                    
        return map("".join,results)

#Cartesian Product using *map and itertools.product
class Solution(object):
    def letterCasePermutation(self, S):
        f = lambda x: (x.lower(), x.upper()) if x.isalpha() else x
        return map("".join, itertools.product(*map(f, S)))

#iterative
class Solution(object):
    def letterCasePermutation(self, S):
        """
        :type S: str
        :rtype: List[str]
        """
        '''
        another iterative way
        '''
        
        results = [""]
        for char in S:
            temp = []
            if char.isalpha():
                for foo in results:
                    temp.append(foo+char.lower())
                    temp.append(foo+char.upper())
            else:
                for foo in results:
                    temp.append(foo+char)
            
            results = temp
        
        return results

##############################
#Container with most water
############################
#brute force TLE
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        '''
        do the brute force way first,
        examine every possible pair and just do a maxupdate
        '''
        max_area = float('-inf')
        N = len(height)
        
        for i in range(N):
            for j in range(i+1,N):
                width = j - i
                highest = min(height[i],height[j])
                max_area = max(max_area, width*highest)
        
        return max_area
        
#two pointers
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        '''
        the brute force suggested starting with the maximum width containter
        we can go to a shorted width container if there is a vertical line loner thant he current ocnatiners shorter line
        we can use two pointers 
        at anytime we always advance the pointer of the shorter height and recalculate
        '''
        max_area = float('-inf')
        left = 0
        right = len(height) - 1
        while left < right:
            width = right - left
            max_area = max(max_area,width*min(height[left],height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return max_area

################################
#
################################
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        well first determine if the sequence is arithmetic
        if it is
        determine how many sequences in A are arithmetics, derive formula for this
        well it looks like there could be sequnees that are arithmetic in an entire sequence that is not arithmetic
        
        '''
        N = len(A)
        if N < 3:
            return 0
        i = 0
        while i + 2 < N:
            if A[i+1] - A[i] != A[i+2] - A[i+1]:
                return 0
            i += 1
        #we got here the sequnce must be arithmetic
        #how many windows if size 3 to N are in A? including the whole sequence
        #for a window if size N, the number of windows = the lenght of the array - windoow size + 1
        window_sizes = list(range(3,N))
        output = 0
        for size in window_sizes:
            output += N - size + 1
        return output + 1

#welp....i tried
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        welp, lets go through all the solutions once again
        the naive soluiont would be to consider every pair of start and end points
        then we pass over that slight and check for arithmetic sequence
        '''
        count = 0
        N = len(A)
        for start in range(N-2):
            for end in range(start+3,N+1):
                sub = A[start:end]
                #now examine each slice
                sub_length = len(sub)
                diff = sub[1] - sub[0]
                for i in range(1,sub_length-1):
                    if sub[i+1] - sub[i] != diff:
                        break
                if i+1 == sub_length-1:
                    count += 1
        return count

#doesn't work on all cases
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        welp, lets go through all the solutions once again
        the naive soluiont would be to consider every pair of start and end points
        then we pass over that slight and check for arithmetic sequence
        '''
        count = 0
        #allowing up the last window size 3
        for start in range(len(A)-2):
            diff = A[start+1] - A[start]
            #branch out for all possible ends points
            for end in range(start+2,len(A)):
                for i in range(start+1,end+1):
                    if A[i] - A[i-1] != diff:
                        break
                if i == end:
                    count += 1
        
        return count

#recursive
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        better brute force
        in the last approach we considered every possible range and then pass over the range to check each consecutive differece is the same
        We can optimize just a little bit by noticing
        in stead of checking every coneuctive difference just check that the new ends conseuctive difference is the same
        #recursive approach, continunig on this path, if a sequence on the range (i,j) is arithmetics then another element at j+1 must have the same difference, i.e A[j+1] - A[j] is the same
        if so, all the ranges from (i,j+1) must also be arithmetic
        furthrmore if the sequence on the range (i,j) isn't arithmetic, adding another element won't do use any good
        assumwe we have a sum variable used to store the total number of arithmetic slices in teh array
        we can define a recusrive function slices(A,i), whihc returns the number of slices in ther ange (k,i), but not part of any range (k,j)
        k refers to the minimum inde such that the range (k,i) is valid arithmetic
        if we know the number of slices on the right (0,i-1) to be the set x
        if this range is indeed arithemtics, all consec elemnets have the same difference
        adding a new element, say a_i, to extend range to (0,i)
        and this new addition increases the a variable ap, which is the number or arithemtics slices 
        the new additional slices will be (0,i), (1,i),(2,i),(i-2,i)
        which is a total of x+1 additioanl arithemtic slices
        thus, in every call, the ith elemnt has the same common differences with the last element as the previous common differences

        '''
        self.sum = 0
        def slices(A,i):
            #returns the number of aritmetic slices of array A up to i
            if i < 2:
                return 0
            additional = 0
            if A[i] - A[i-1] == A[i-1] - A[i-2]:
                additional = 1 + slices(A,i-1)
                self.sum += additional
            else:
                slices(A,i-1)
            return additional
        
        slices(A,len(A)-1)
        return self.sum
                
#dp O(N) time and O(N) space
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        this is easier to see as DP problem
        the minimum requiremtn for an arith seq must be at least length 3
        we can use a sliding window of three, and when we do find it, we push 1 into our dp array
        when we examin the next window, if it is also an arithmetic sequence, we add one but also acarry thre previous element in our dp array
        update sum on the fly
        
        can also use constante space by just keeping track of previous entry and current entry
        '''
        N = len(A)
        dp = [0]*N
        count = 0
        for i in range(2,N):
            if A[i] - A[i-1] == A[i-1] - A[i-2]:
                dp[i] = 1+dp[i-1]
                count += dp[i]
        return count

        N = len(A)
        current = 0
        count = 0
        for i in range(2,N):
            if A[i] - A[i-1] == A[i-1] - A[i-2]:
                current = 1 + current
                count += current
            else:
                current = 0
        return count

class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        123468
         11122
          1334
        we can start by getting the differenes array
        then whenever we have the same conseuctive differences we have found a match
        '''
        first_diffs = []
        N = len(A)
        for i in range(1,N):
            first_diffs.append(A[i]-A[i-1])
        
        output = 0
        offset = 1
        #starting at the second 1
        for i in range(1,len(first_diffs)):
            if first_diffs[i] == first_diffs[i-1]:
                output += offset
                offset += 1
            else:
                offset = 1
        return output

#########################################
#Minimum Remove to Make Valid Parentheses
#########################################
class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        a string is balances if the count('(') == cont(')')
        we can pass the string and check if it is balances by increamting 1 if open
        and decrenting 1 if closed
        if at anytime the balance is negtaive, than the string up that point is unbalnaced
        we can use this to our advantage
        if we encounter a close at when the balance is zero, we know not to include that in
        but this doesn't work for all cases
        if we have a balance of zero after reaching the end it's notvalid
        meaning we didn't have enough closings at the end to make the string balanced
        we need to identify which '(' each of our ')' is actually pairing with
        we need to know the indices of the problematic '('
        we use a stack and each time we should add its index to tehs tack
        each time we see an ')' we should removen an index from the stack beause the ')' will match with wtv '(' was at the top of the stack
        algo:
            1. keep track of problematic openings by pushing their indices onto a stack
            2. always add opening paran first
            3. if there is somthing on the stack and we encoutner a closing, pop it off
            4. if there ins't a stack and we get to a closing pass it
            5. rebuildg the string skipping over the indices in the stack
        '''
        def is_balanced(string):
            balance = 0
            for char in s:
                if char == '(':
                    balance += 1
                if char == ')':
                    balance -= 1
                if balance < 0:
                    return False
            return balance == 0
        
        unmatched_closings = set()
        stack_matched_pairs = [] #whats left on this stack are the problematic openings
        for i,char in enumerate(s):
            if char not in '()':
                continue
            if char == '(':
                stack_matched_pairs.append(i)
            if char == ')':
                #if there is nothing to pair it with, its probematic
                if not stack_matched_pairs:
                    unmatched_closings.add(i)
                #there must be a pair
                else:
                    stack_matched_pairs.pop()
        #gather problemtic indices;
        bad_indices = unmatched_closings.union(set(stack_matched_pairs))
        
        result = []
        for i,c in enumerate(s):
            if i not in bad_indices:
                result.append(c)
                    
        return ''.join(result)

class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        another way would be to just treat it like the valid paranthese problem
        just match close with open on to the stack and if you can't then we need to get rid of it
        '''
        stack = []
        s = list(s)
        N = len(s)
        
        stack = []
        
        for i in range(N):
            if s[i] == '(':
                stack.append(i)
            if s[i] == ')':
                if stack:
                    stack.pop()
                else:
                    s[i] = ''
        while stack:
            s[stack.pop()] = ''
        return ''.join(s)

##################
#Roman To Integer
###################

class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        there are a couple of approaches here, lets first walk through a left to right pass
        recall that each symbol adds its own value except for when a samlle values symbold is before a large valued symbol
        in this case we need to subtract the alrge from the small and add the difference
        the simplest algorithm is to use a two to scan ethrough string and examine one and two places ahead
        i fucking had it!
        two mainin the invariant just check that we at least two symbols to check every time we left ot right
        '''
        letters = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
        numbers = [1,5,10,50,100,500,1000]
        mapp =  {k:v for k,v in zip(letters,numbers)}
        
        #get the lenght of the string
        N = len(s)
        value = 0
        left = 0
        while left < N:
            if left + 1 < N and mapp[s[left]] < mapp[s[left+1]]:
                value += mapp[s[left+1]] - mapp[s[left]]
                left += 2
            else:
                value += mapp[s[left]]
                left += 1
        return value
        
#using extra numberals for multipels 4 and 9
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        we can im prove by increasing the the state representation for each char
        if we though of the dictionary to include represeantions of:
        4,9,40,90,400,900
        as IV, IX,XL,XC,CD,CM
        then we can look up each pair and them togetehr
        we then hash everythin and check that its int he hash
        remember to fucking maintain the invariant!!!!
        '''
        mapp = {}
        letters = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
        numbers = [1,5,10,50,100,500,1000]
        numbers += [4,9,40,90,400,900]
        letters += ['IV', 'IX','XL','XC','CD','CM']
        mapp = {k:v for k,v in zip(letters,numbers)}
        
        result = 0
        i = 0
        while i < len(s):
            single = s[i:i+2]
            double = s[i]
            if i - 1  < len(s) and double in mapp:
                result += mapp[double]
                i += 2
            else:
                result += mapp[single]
                i += 1
        
        return result

#right to left
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        right to left,
        recall from the left to right approach XC was just the sum of
        mapp[C] - mapp[X]
        which isjust the same as
        sum += mapp[C]
        sum -= mapp[X]
        then we can process 1 by 1 one from the right,
        we still need to examine its nieghbor though, unless we noticne the following
        inution:
            1. withotu looking at the enxt symbol, we don't know wheter we should increment or decrmenet
            2. but the right most symbol will alwyas be added regardless! (think about this)
        what we can is initalize the sum frist to be the value of the right most, 
        then we can work backwards, starting from the second to last char
        '''
        letters = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
        numbers = [1,5,10,50,100,500,1000]
        mapp =  {k:v for k,v in zip(letters,numbers)}
        
        value = mapp[s[-1]]
        for i in reversed(range(len(s)-1)):
            #if the previous neighbor is smaller we need to subtract
            if mapp[s[i+1]] > mapp[s[i]]:
                value -= mapp[s[i]]
            else:
                value += mapp[s[i]]
        return value

        #an additional way not storing the final value numeral as the iniital value
        letters = ['I', 'V', 'X', 'L', 'C', 'D', 'M']
        numbers = [1,5,10,50,100,500,1000]
        mapp =  {k:v for k,v in zip(letters,numbers)}
        
        N = len(s)
        i = N-1
        output = 0
        while i >= 0:
            if i < N-1 and mapp[s[i]] < mapp[s[i+1]]:
                #in the case IX
                output -= mapp[s[i]]
            else: 
                output += mapp[s[i]]
            i -= 1
        
        return output

####################
#Broken Calculator
###################
#well BFS was a good try
class Solution(object):
    def brokenCalc(self, X, Y):
        """
        :type X: int
        :type Y: int
        :rtype: int
        """
        '''
        intiallay we are starting with the value X
        we can only double
        or decrement one
        return number of min ops to get Y
        can we always get to Y?
        well im thnking BFS, just a graph
        '''
        
        q = deque([(X,0)])
        
        while q:
            number,ops = q.popleft()
            if number == Y:
                return ops
            q.append((number*2,ops+1))
            q.append((number-1,ops+1))

#fail again
class Solution(object):
    def brokenCalc(self, X, Y):
        """
        :type X: int
        :type Y: int
        :rtype: int
        """
        '''
        intiallay we are starting with the value X
        we can only double
        or decrement one
        return number of min ops to get Y
        can we always get to Y?
        well im thnking BFS, just a graph
        if at anytime Y is less than X, we must decrease
        otherwise we must double it
        nope that was stupid
        '''

        ops = 0
        
        while X != Y:
            if X > Y:
                X -= 1
            else:
                X *= 2
            ops += 1
        return ops

class Solution(object):
    def brokenCalc(self, X, Y):
        """
        :type X: int
        :type Y: int
        :rtype: int
        """
        '''
        work backawards
        instead of multiplying by 2 or subtracting from 1 (from X both cases)
        we divied by 2 when Y is even otherewsie subtract
        we need to greedily divide by two, why?
        well by multiplying we saw that did not have the optimal answer, so we do they opposite
        motivation:
            if y is even, if we were to perform two additions and one division (three steps), we could instead peform one division and on addition
            if say y is odd, then if we perform 3 additionas and one division, we could isntead perform 1 addition, 1 division, and 1 additoin for less operatrions
        '''

        ops = 0
        
        while Y > X:
            ops += 1
            if Y % 2 == 1:
                Y += 1
            else:
                Y /= 2
        return ops + (X-Y)


class Solution(object):
    def brokenCalc(self, X, Y):
        """
        :type X: int
        :type Y: int
        :rtype: int
        """
        '''
        you know going forwards won't work
        lets think about cases
        if Y < X, then we can only go in one steps
        so its just X-Y steps
        if even, just divide by two
        if odd, add one then divide by two
        greedily divide by two if you can
        if yo can't and you're still greater than Y, add 1 then divide
        
        '''
        steps = 0
        while X != Y:
            if Y < X:
                steps += X-Y
                X = Y
            else:
                if Y % 2 == 0:
                    Y /= 2
                else:
                    Y += 1
                steps += 1
        return steps

class Solution(object):
    def brokenCalc(self, X, Y):
        """
        :type X: int
        :type Y: int
        :rtype: int
        """
        '''
        recursive
        '''
        def steps(X,Y):
            num_steps = 0
            if X >= Y:
                num_steps = X-Y
            elif Y % 2 == 0:
                num_steps +=  steps(X,Y/2) + 1
            else:
                num_steps +=  steps(X,Y+1) + 1
            return num_steps
        return steps(X,Y)

##################################################
# Longest Word in Dictionary through Deleting 
##################################################
#substring decomp
class Solution(object):
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        '''
        the naive way would be to check if i can make each word recurively
        '''
        N = len(s)
        d = set(d)
        self.output = ""
        
        def substring_decomp(s,path):
            if len(s) == 0:
                print path[:]
                return
                
            #shrink s and take, shrink s and don't take
            substring_decomp(s[:-1],path+s[-1])
            substring_decomp(s[:-1],path)
        
        substring_decomp(s,"")

class Solution(object):
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        '''
        well this was dumb, we did not need to anthing recursive
        we can just check if each word in d is a subsequence of s
        '''
        #sort the dictionary by decreasing length and lexogrphic order
        d.sort(key = lambda x: (-len(x),x))
        
        #helper function to determine
        def is_subsequence(sub,whole):
            len_sub = len(sub)
            len_whole = len(whole)
            i,j = 0,0
            while i < len_sub and j < len_whole:
                if sub[i] == whole[j]:
                    i += 1
                j += 1
            return i == len_sub
        
        #apply helper
        for word in d:
            if is_subsequence(word,s):
                return word
        return ""

class Solution(object):
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        '''
        we also don't need to sort and just check everything in d
        sorting requires extra space
        in this case, it woudl required log(d) space
        
        '''
        #helper function to determine
        def is_subsequence(sub,whole):
            len_sub = len(sub)
            len_whole = len(whole)
            i,j = 0,0
            while i < len_sub and j < len_whole:
                if sub[i] == whole[j]:
                    i += 1
                j += 1
            return i == len_sub
        
        #apply helper
        output = ""
        for word in d:
            if is_subsequence(word,s):
                if len(word) > len(output):
                    output = word
                elif len(word) == len(output) and output[0] > word[0]:
                    output = word
        return output

#########################
#Find The Celebrity
########################
#close one
# The knows API is already defined for you.
# @param a, person a
# @param b, person b
# @return a boolean, whether a knows b
# def knows(a, b):

class Solution(object):
    def findCelebrity(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        i could make the adjaceny list, which means i would have to call knows for every pair
        that does not seem unreasonable given than n is at most 100
        '''
        if n == 2:
            return -1
        adj_list = defaultdict(list)
        for i in range(n):
            for j in range(n):
                adj_list[i].append(knows(i,j))
        temp = [[None]*n]*n
        for k,v in adj_list.items():
            temp[k] = v
        
        #now sum across columns 
        col_sums = []
        for c in range(n):
            col_sums.append(sum([temp[r][c] for r in range(n)]))
            
        for i,val in enumerate(col_sums):
            if val == n:
                return i
        return -1
        
class Solution(object):
    def findCelebrity(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        instead of cheaking each pairwise comparison, we can view it as a grpah
        which is what i said at first
        if there is a celbrity, it had n-1 indirections and 0 out directions
        we don't have the edges to begin with, but do we need to call the API so many times to find the celebrity
        if A know's somone, we know that A cannot be the celebrity and can eliniate A
        if we already know that someone knows another person then that person cannot be the celebrity
        if A knows B, a cannot be the celeb
        if not A knows B, B cannot be the celeb either
        therefore, each call to knows, can rule out if one person is a celbrity
        algo:
            ranomdize the first candidate to zero, and call known with zero and 1
            if true, it cannot be zero, so candiate becomes on
            and we we just keep updating the candiatne in n-1 time ruling out celebrities
            #watch the anumation
            at the end, we have our final canddiate, that should be the celebrity
            we check agian using our celbrity function
        '''
        #to determin if a person is a celebrity in O(N) time
        def is_celebrity(person):
            for other in range(n):
                if person == other:
                    continue
                if knows(person,other) or not knows(other,person):
                    return False
            return True
        
        candidate = 0 #start at the frist one
        for i in range(1,n):
            if knows(candidate,i):
                candidate = i
        #now check
        if is_celebrity(candidate):
            return candidate
        return -1

class Solution(object):
    def findCelebrity(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        instead of cheaking each pairwise comparison, we can view it as a grpah
        which is what i said at first
        if there is a celbrity, it had n-1 indirections and 0 out directions
        we don't have the edges to begin with, but do we need to call the API so many times to find the celebrity
        if A know's somone, we know that A cannot be the celebrity and can eliniate A
        if we already know that someone knows another person then that person cannot be the celebrity
        if A knows B, a cannot be the celeb
        if not A knows B, B cannot be the celeb either
        therefore, each call to knows, can rule out if one person is a celbrity
        algo:
            ranomdize the first candidate to zero, and call known with zero and 1
            if true, it cannot be zero, so candiate becomes on
            and we we just keep updating the candiatne in n-1 time ruling out celebrities
            #watch the anumation
            at the end, we have our final canddiate, that should be the celebrity
            we check agian using our celbrity function
        '''
        #to determin if a person is a celebrity in O(N) time
        def is_celebrity(person):
            for other in range(n):
                if person == other:
                    continue
                if knows(person,other) or not knows(other,person):
                    return False
            return True
        
        candidate = 0 #start at the frist one
        for i in range(1,n):
            if knows(candidate,i):
                candidate = i
        #now check
        if is_celebrity(candidate):
            return candidate
        return -1

########################
#Search a 2D Matrix II
########################
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        cheeky way is just pass the whole thing
        but there must be a log(cols)*log(rows) solution
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows):
            for j in range(cols):
                if target == matrix[i][j]:
                    return True
        return False

#binary searching each row for all rows
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        cheeky way is just pass the whole thing
        but there must be a log(cols)*log(rows) solution
        i could binary search each row individually
        binary search row by row
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        for row in range(rows):
            #binary search the row
            lo, hi = 0,cols-1
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if matrix[row][mid] == target:
                    return True
                elif matrix[row][mid] > target:
                    hi = mid
                else:
                    lo = mid + 1
            if matrix[row][lo] == target:
                return True
        return False

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        there are a couple of approaches here, lets go over them 1 by 1
        we can binary search on the row and column in succsession
        if we haven't found the target, we block off the row and column
        essentially we go down the diagonals from upper left to lower right
        and binary search at the current row and column
        XXXX
        XYYY
        XYZZ
        XYZA
        going down diagonally and blocking off
        '''
        if not matrix:
            return False
        #binary search functino
        def binary_serach(matrix,target,start,vert):
            low = start
            high = len(matrix[0]) - 1 if vert else len(matrix)-1
            
            while high >= low: #we are shrinkgin low in each pass anyways, so limit high
                mid = low + (high - low) // 2
                #going across column
                if vert:
                    if matrix[start][mid] < target:
                        low = mid+1
                    elif matrix[start][mid] > target:
                        hi = mid-1
                    else:
                        return True
                else: #now on row
                    if matrix[mid][start] < target:
                        low = mid+1
                    elif matrix[mid][start] > target:
                        hi = mid-1
                    return True
            return False
        
        for i in range(min(len(matrix),len(matrix[0]))):
            vert_found = binary_serach(matrix,target,i,True)
            horz_found = binary_search(matrix,target,i,False)
            if vert_found or horz_found:
                return True
            
        return False

#recursive
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        we can also use recursion to solve this
        intuition:
            at any time, we can partition the two d matrix into four sorted sub matrices
            two of which might contain the target, and the other two cannot
        algo:
            base case, if the array has no size, it cannot contain the target
            if the target is smaller than the arrays smallest element (top left) or larger than the arrays largest elment (bottom right) it doesn't contain the target
            recursive case, target it not met and the array has positive area
            thereforse we seek along the matrix's middle column for an index row such taht matrix[row-1][mid] < target < matrix[row][mid]
            we are pruning the search space, and recurse on the the matrix with the aforementioned bounds
        '''
        if not matrix:
            return False
        
        def rec_search(left,up, right,down):
            #mark the bounds of the slice of the amtrix
            #out of bounds check
            if left > right or up > down:
                return False
            #first base condition, lower than upper left and higher that lower right
            elif target < matrix[up][left] or target > matrix[down][right]:
                return False
            #otherwise recurse
            else:
                #the col we want
                mid = left + (right - left) // 2
                #we need to move through the rows
                row = up
                while row <= down and matrix[row][mid] <= target:
                    if matrix[row][mid] == target:
                        return True
                    row += 1
            possibility_1 = rec_search(left,row,mid-1,down)
            possibility_2 = rec_search(mid+1,up,right, row-1)
            return possibility_1 or possibility_2
        
        return rec_search(0,0,len(matrix[0])-1,len(matrix)-1)

#another way
'''
The idea behind this code is to recursively split the matrix into 4 sub-matrices.
For each sub-matrix, the upper left corner (ri, ci) is the min value and the lower right corner (rj, cj) is the max value.
If target is in that range, target may be in that sub-matrix, in which case we split it in 4 again. Otherwise, we can safely rule out the submatrix.
'''
class Solution:
    def searchMatrix(self, matrix, target):
        
        if not matrix or not matrix[0]:
            return False
        
        def explore(ri, ci, rj, cj):
            if ri > rj or ci > cj :
                return False
            
            if ri == rj and ci == cj:
                return matrix[ri][ci] == target
            
            _min, _max = matrix[ri][ci], matrix[rj][cj]
            if _min <= target <= _max:
                rm, cm = (ri + rj) // 2, (ci + cj) // 2
                return explore(ri, ci, rm, cm) or explore(ri, cm+1, rm, cj) or explore(rm+1, ci, rj, cm) or explore(rm+1, cm+1, rj, cj)
            return False
            
        rows, columns = len(matrix), len(matrix[0])
        return explore(0, 0, rows-1, columns-1)


#starting from bottom left
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        we can reduce the search space every time
        since the rows and columns are sorted increasing
        if we start all the way at end, then we are alwasy left with two options
        and adjacent lement will alwasy be greater or smaller
        '''
        #edge cases empty matrix 
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        #start bottom left
        i,j = rows-1, 0
        
        while 0 <= i < rows and 0 <= j < cols:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                i -= 1
            elif matrix[i][j] < target:
                j += 1
        return False

#starting from upper right
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        we can reduce the search space every time
        since the rows and columns are sorted increasing
        if we start all the way at end, then we are alwasy left with two options
        and adjacent lement will alwasy be greater or smaller
        '''
        #edge cases empty matrix 
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        #start bottom left
        i,j = 0, cols-1
        
        while 0 <= i < rows and 0 <= j < cols:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            elif matrix[i][j] < target:
                i += 1
        return False

##########################
# Score of Parentheses
##########################
#i couldn't generalize this one ...
#fail
class Solution(object):
    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        '''
        (()(()))
        
        (
        2
        '''
        score = 0
        S = list(S)

        #eliminate () and put those case 1
        temp = [] #this is our stack
        for char in S:
            if char == ')':
                if temp and temp[-1] == '(':
                    temp.pop()
                    temp.append(1)
                else:
                    temp.append(char)

            else:
                temp.append(char)
        print temp
        #evalute all (1)
        temp_2 = []
        for char in temp:
            if char == ')':
                if temp_2:
                    if temp_2[-1].isnumeric():
                        temp_2.pop()
                        temp_2.append(2)
                    else:
                        temp_2.append(char)
                else:
                    temp_2.append(char)
            else:
                temp_2.append(char)
        print temp_2


        #evaluate all additions

        #final ()

class Solution(object):
    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        '''
        recall in a balanced parantheses, number of open will be be equal or greater number closed
        as we go forward
        we can use stack
        and every time we see an open just add a zero
        only then we encouner a close we pop off stack
        if that value it zero we know we are supposed to lcose so the value beocmes 1
        otherwise that val get multiplie by two
        '''
        stack = []
        output, curr_val = 0,0
        for char in S:
            if char == '(':
                stack.append(0)
            elif char == ')':
                mult = stack.pop()
                if mult == 0:
                    curr_val = 1
                else:
                    curr_val = mult*2
                #if we are done closing out all paranthese, that meant we don't have stack
                #just increment the output
                if not stack:
                    output += curr_val
                else:
                    #there is a value dangling and we need to increment that
                    stack[-1] += curr_val
        return output

class Solution(object):
    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        '''
        the recursive approach
        we can split string S into A + B, and A and B are balanced
        and they are bothe the smallest possible non empty prefix of S
        
        algo:
            we call a balanced string primitive if it cannot be partitonied into two empty non-empy balance strings
            by keeping track of balance (the number of open parantheses min close) we can partition S into primitive strings p1 + p2...pn
            and the score S can be written as a recurrences
            score(S) = score(p1) + score(p2)
            
        For each primitive substring (S[i], S[i+1], ..., S[k]), if the string is length 2, then the score of this string is 1. Otherwise, it's twice the score of the substring (S[i+1], S[i+2], ..., S[k-1]).
        '''
        def F(i,j):
            #score of the balanaced stgring S[i:j]
            ans = bal = 0
            #split string into primitives
            for k in range(i,j):
                if S[k] == '(':
                    bal += 1
                else:
                    bal -= 1
                
                if bal == 0:
                    #if the indices differe by 1, i.e not primitive
                    if k - i == 1:
                        ans += 1
                    else:
                        #reurce
                        ans += 2*F(i+1,k)
                    i = k + 1
            return ans
        
        return F(0,len(S))
#come back to this problem
class Solution:
    def scoreOfParentheses(self, s: str) -> int:
        score=0
        bal=0;j=0
        for i in range(len(s)):
            if s[i] == '(':
                bal += 1
            else:
                bal -= 1
            if bal==0:
                if len(s[j:i+1])==2:
                    score += 1;
                else:
                    score+=2 * self.scoreOfParentheses(s[j+1:i])
                j=i+1
        return score

class Solution(object):
    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        '''
        every position in the string has a depth associated to it
        our gail is to mainitain the score at the curent depth
        when encoutering an opening, we add a zero, which is the the score for the curent depth
        when close we add twice the previous depths score, unlues () which is jsut one
        '''
        stack = [0]
        
        for char in S: 
            print stack
            if char == '(':
                stack.append(0)
            else:
                value = stack.pop()
                stack[-1] += max(2*value,1)
        print stack
class Solution(object):
    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        '''
        count cores
        the final sum will be a sum of powers of 2
        as every core (substring (), with score 1) will have its core multiplied by 2 for each exterior sef of parantehsesis
        algo:
            keept rack of the balance of the string
            for every close that immediately follows an open, 1 << balance is the number of exterior set of parantheses that contains the core
        '''
        ans = bal = 0
        for i,char in enumerate(S):
            if char == '(':
                bal += 1
            else:
                bal -= 1
                if S[i-1] == '(':
                    ans += 1 << bal
        return ans
        
class Solution(object):
    def scoreOfParentheses(self, S):
        """
        :type S: str
        :rtype: int
        """
        '''
        count cores
        the final sum will be a sum of powers of 2
        as every core (substring (), with score 1) will have its core multiplied by 2 for each exterior sef of parantehsesis
        algo:
            keept rack of the balance of the string
            for every close that immediately follows an open, 1 << balance is the number of exterior set of parantheses that contains the core
        '''
        ans = bal = 0
        for i,char in enumerate(S):
            print bal
            if char == '(':
                bal += 1
            else:
                bal -= 1
                if S[i-1] == '(':
                    ans += 2**bal
        return ans
        

##########################################
# Shortest Unsorted Continuous Subarray
##########################################
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        well i can first check of the nums array is already sorted
        brute force would be to check all subsequences
        '''
        N = len(nums)
        if N == 1:
            return 0
        
        #if we look, if there is sequence, its smallest value is greater than the left-1 idx
        #and its largest value is less than right + 1 idx
        
        #first find the start of the array by going left and check increasing
        left = 0
        while left < N:
            if  left + 1 < N and nums[left+1] >= nums[left]:
                left += 1
            else:
                break
        if left == N-1:
            return 0
        #now go from the right
        right = N - 1
        while right >= 0:
            if right - 1 >= 0 and nums[right-1] <= nums[right]:
                right -=1
            else:
                break
        if right == 0:
            return N
        print left,right
        return right - left + 1
            
#brute force
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        BRUTE FORCE
        examine every subarray nums[i:j] and get their min and max
        then if nums[0:i-1] and nums[j:n-1] are sorted, then this is a viable candidate subarray
        also all elements from nums[0:i-1] must be < min(nums[i:j])
        and all elements from nums[j:n-1] > max(nums[i:j])
        furthere, we also need to check if nums[0:i-1 ] and nums[j:n-1] are sorted correctly
        if all conditions match, we can minimuze the length
        '''
        result = len(nums)
        for i in range(len(nums)):
            for j in range(i,len(nums)+1):
                mini = float('inf')
                maxi = float('-inf')
                prev = float('-inf')
                #k is just the index in this subarray
                for k in range(i,j):
                    mini = min(mini,nums[k])
                    maxi = max(maxi,nums[k])
                #now compare agains the maxi and mini
                #if left side exceeds minie
                #right less than maxi
                if (i > 0 and nums[i-1] > mini) or (j < len(nums) and nums[j] < maxi):
                    continue
                #check nums[0:i] is sorted
                k = 0
                while k < i and prev <= nums[k]:
                    prev = nums[k]
                    k += 1
                if k != i:
                    continue
                #check nums[j:n-1] is sorted
                k = j
                while k < len(nums) and prev <= nums[k]:
                    prev = nums[k]
                    k += 1
                if k == len(nums):
                    result = min(result, j-i)
                    
        return result

#Better Brute Force
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        better brute force
        we can take some properites from selection sort
        we traverse over the give nums array and for every eelemnt chosen we tru to determine its correct positions in the sorted array
        for this we compare nums[i] with every nums[j] such that i <j < n
        if nums[j] happens to be smaller than nums[i] it means nums[i] and nums[j] are not in their right spot
        so we need to swap the two elements to bring them to their correct spots
        instead of swapping we just note the position of nums[i] and nums[j] given by i and j
        these two elments now mark the boundary
        this is actually a very brilliant idea
        '''
        #if we need to swap the whole array
        largest_left = len(nums) #minimu this
        smallest_right = 0 #maximuse thise
        for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if nums[j] < nums[i]:
                    largest_left = min(largest_left,i)
                    smallest_right = max(smallest_right,j)
        if smallest_right - largest_left < 0:
            return 0
        else:
            return smallest_right - largest_left + 1

#using sorting
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        using sorting
        we can sort a copy of the given nums array
        then we just compare elements
        '''
        sorted_nums = copy.deepcopy(nums)
        sorted_nums.sort()
        #again minimuze this range
        largest_left = len(nums)
        smallest_right = 0
        for i in range(len(nums)):
            if sorted_nums[i] != nums[i]:
                largest_left = min(largest_left,i)
                smallest_right = max(smallest_right,i)
                
        if smallest_right - largest_left < 0:
            return 0
        else:
            return smallest_right - largest_left + 1

#O(N) times
class Solution(object):
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        using stack, stop here, im fucking done
        the concept is similar to selection sort
        the goal can degenerate to finding the positions ot the min and max of the unosrted subarray
        traverse the array, and so long as we are increasing keep pushing on to the stack
        once we encounter a falling slop we know nums[j] is out of place
        on a decreasing sequence we pop from the stack and keep popping until we reach the stage where the element (corresponding to the index) is lesser than nums[j]
        now we mark this as the boundary
        do the same for the right side
        #KEY:
        #in order to determine the correct position of nums[j], we keep popping from the top of the stack
        #until we reach the elements that is smaller than nums[j], call it a k
        once the popping stops at k, we know that nums[j] should be at k + 1
        '''
        stack  = []
        largest_left = len(nums)
        smallest_right =  0
        for i in range(len(nums)):
            while stack and nums[stack[-1]] > nums[i]:
                #keep minimiging the largest left
                largest_left = min(largest_left,stack.pop())
            stack.append(i)

        #clear the stack
        stack = []
        for i in range(len(nums)-1,-1,-1):
            while stack and nums[stack[-1]] < nums[i]:
                smallest_right = max(smallest_right, stack.pop())
            stack.append(i)
            
        if smallest_right - largest_left < 0:
            return 0
        else:
            return smallest_right - largest_left + 1

##########################
# Validate Stack Sequences
##########################
#FUCK YEAH! GOT IT! 
#THATS HOW YOU DO IT!
class Solution(object):
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        '''
        first start with an empty stack
        and try pushing each element, but also examins if i can pop the element
        if ive simulated and there is nothing left to pop its valid
        otherwise it is not
        '''
        N = len(pushed)
        pushed_ptr = 0
        popped_ptr = 0
        stack = []
        while pushed_ptr < N or popped_ptr < N:
            if stack and stack[-1] == popped[popped_ptr]:
                stack.pop()
                popped_ptr += 1
            else:
                if pushed_ptr >= N:
                    return False
                else:
                    stack.append(pushed[pushed_ptr])
                    pushed_ptr += 1
        if not stack:
            return True
        else:
            return False

class Solution(object):
    def validateStackSequences(self, pushed, popped):
        """
        :type pushed: List[int]
        :type popped: List[int]
        :rtype: bool
        """
        '''
        well for every element in pushed, we push onto and empty stack
        and if there is a stack and the top is the current value at popped well we pop
        '''
        stack = []
        popped_ptr = 0
        for num in pushed:
            stack.append(num)
            while stack and popped_ptr < len(popped) and stack[-1] == popped[popped_ptr]:
                stack.pop()
                popped_ptr += 1
        return len(stack) == 0 
        
