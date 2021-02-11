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

        