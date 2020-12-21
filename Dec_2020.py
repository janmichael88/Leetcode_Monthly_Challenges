# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        i bfs all the way down until i hit the final leaf
        '''
        if not root:
            return 0
        q = deque([(root,1)])
        while q:
            current, level = q.popleft()
            if current.left:
                q.append((current.left,level+1))
            if current.right:
                q.append((current.right,level+1))
                
        return level

#Recusrive
#note for the min depth problem simply change max to min
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        i can descend the tree going left and right
        everytime i invoke dfs, add 1 and return the max
        '''
        if not root:
            return 0
        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            right = dfs(root.right)
            return max(left,right) + 1
        
        return dfs(root)


#########################
#Shortest Word Distance
#########################
class Solution(object):
    def shortestDistance(self, words, word1, word2):
        """
        :type words: List[str]
        :type word1: str
        :type word2: str
        :rtype: int
        """
        '''
        use a hashmap storing the most recent index positions for each word
        i need to update this hashmap every time i get the distance
        keep taking the min and then go through the whole array
        '''
        mapp = {}
        distance = float('inf')
        for i in range(len(words)):
            if words[i] == word1:
                mapp[word1] = i
            if words[i] == word2:
                mapp[word2] = i
            if word1 in mapp and word2 in mapp:
                distance = min(distance, abs(mapp[word1] - mapp[word2]))
        
        return distance

#########################
#Linked List Random Node
#########################
#Naive Soltion
class Solution(object):

    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        '''
        the naive way would be to just dump all elements in a list and use a random funfction
        '''
        self.nums = []
        while head:
            self.nums.append(head.val)
            head = head.next
        

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        rand = int(random.random()*len(self.nums))
        return self.nums[rand]


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):

    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        self.LL = head
        #get the number of nodes
        self.N = 0
        temp = self.LL
        while temp:
            self.N += 1
            temp = temp.next
        
        

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        '''
        now find a way to get a randome number between 1 and N
        traverse up to that N, and return it
        '''
        rand = int(random.random()*self.N)
        i = 0
        temp = self.LL
        result = None
        while i < rand:
            result = temp.val
            temp = temp.next
            i += 1
        return result

#using reservoir sampling
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):

    def __init__(self, head):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        :type head: ListNode
        """
        '''
        the idea it to use Reservoir sampling
        say we want to take k elements w/ equal probability 1/N
        but we cannot keep track of N because the list is infinte (streaming)
        we keep as an array R of size k, sampling from S of unknown size
        we load up k elemnts from S in order
        then we continue from k+1, each time generating a random number from 1 up to k + 1, and of that random number is less than k (i.e in our R) we update our R
        at the end we should obtain k samples from S where len(S) is unknown
        
        '''
        self.head = head
        

    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        n = 1
        ans = 0
        curr = self.head
        
        while curr:
            #generate random number and see if less than 1/
            #if less than 1/n, we need to update our res
            if random.random() < 1/n:
                ans = curr.val
            curr = curr.next
            n += 1
        return ans

##############################
#Increasing Order Search Tree
##############################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def increasingBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        '''
        i can an inorder traversal getting the values
        then remake the tree alwasy going right? with left child null
        '''
        vals = []
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            vals.append(node.val)
            dfs(node.right)
        dfs(root)
        #now make a new tree
        newroot = TreeNode(val = vals[0])
        temp = newroot
        for n in vals[1:]:
            temp.left = None
            temp.right = TreeNode(val=n)
            temp = temp.right
        return newroot

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def increasingBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        '''
        recursive solution
        we can assing an empty tree structure outside our call
        recursively move through our tree as we make changes to our tree
        in the call we examine a node
        set the node to left
        in our tree structure we will have apointer, and it is in this dfs call we asign left and right pointers, but also move our pointer
        at that node, assign cur pointer right to node
        descend right
        making sure we call on both left and right
        '''
        newTree = TreeNode(None)
        self.temp = newTree
        def dfs(node):
            if not node:
                return
            dfs(node.left)
            node.left = None
            self.temp.right = node
            self.temp = node
            dfs(node.right)
        dfs(root)
        return newTree.right


#######################
#THe kth Factor of N
########################
class Solution(object):
    def kthFactor(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        '''
        just keep generating factors up to k and if it works dump into a list
        then return the k-1 in the list
        '''
        factors = []
        cand = 1
        while cand <= n:
            if n % cand == 0:
                factors.append(cand)
            cand += 1
        
        if len(factors) < k:
            return -1
        else:
            return factors[k-1]

class Solution(object):
    def kthFactor(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        '''
        iterate from x to n // 2
        if x is a divisor of N k-= 1
        return x once k gets to 0
        other wise return n (no divisor)
        '''
        for i in range(1, n//2 + 1):
            if n % i == 0:
                k -= 1
                if k == 0:
                    return i
                
        return n if k == 1 else -1
        
#another way
class Solution(object):
    def kthFactor(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        '''
        using a heap 
        O (sqrt(N)*log(k))
        initalize a heap of size k, (well populate this with divsors)
        and iterate from 1 to sqrt(N)
        if cand is a divors of N then N/cand is also a divisor
        return the head of the heap
        '''
        heap = []
        #heapify(heap)
        for x in range(1,int(n**0.5)+1):
            #if i can divide n
            if n % x == 0:
                heappush(heap,-x)
                if len(heap) > k:
                    heappop(heap)
                    #edge case
                    if x != n // x:
                        heappush(heap,-n//x)
                        if len(heap) > k:
                            heappop(heap)
        return -heappop(heap) if k == len(heap) else -1

class Solution(object):
    def kthFactor(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        '''
        O(sqrt(N))
        init set of divisors
        now iteratr from 1 up to sqrt(N)
        if x is a divros N, decreas k by one
        now we need to find the kth divisor
        i can get its complement by
        N / listdivors[len(listdivors) - k]
        but we need to check in the case of perfect square at which the complement will be the kth divisor
        the list of divisors will contain a duplicate, so we incrment k by 1
        '''
        divisors = []
        sqrtn = int(n**0.5)
        for x in range(1, int(n**0.5) + 1):
            if n % x == 0:
                k -= 1
                divisors.append(x)
                if k == 0:
                    return x
        #now if we have not found the kth divisor check for perfect square
        if sqrtn*sqrtn == n:
            k += 1
        
        n_div = len(divisors)
        if k <= n_div: #meaning we want its pair
            return n // divisors[n_div-k]
        else:
            return -1


###################
#Can Place Flowers
###################
#well its half
class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        '''
        this is similar to the seat problem
        can i simulate placing the flowers
        when placing flowers  check for adjacent positions
        if i want to place a flower i need to check that i-1 and i+1 are zeros
        and once n hits 0, return True
        otherwise i got to the end without placing flowsers so its false
        '''
        N = len(flowerbed)
        i = 1
        curr = 0
        while i < N:
            if n == 0:
                return True
            #check ends
            elif i == 0:
                if flowerbed[i] == 0 and flowerbed[i+1] == 0:
                    flowerbed[i] = 1
                    n -= 1
            elif i == N-1:
                if flowerbed[i] == 0 and flowerbed[i-1] == 0:
                    flowerbed[i] = 1
                    n -= 1
            elif flowerbed[i-1] == 0 and flowerbed[i+1] == 0:
                flowerbed[i] = 1
                n -= 1
            i += 1
        
        print N

class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        '''
        count up the extra available slots that meet the constraints
        we can only add in a flower if it zero and adjacent positions are also zero
        for the first and last eleemnts we need not check the previos and next adjacent
        '''
        N = len(flowerbed)
        i = 0
        count = 0
        while i < N:
            if (flowerbed[i] == 0) and (i == 0 or flowerbed[i-1] == 0) and (i == N-1 or flowerbed[i+1] == 0):
                flowerbed[i] = 1
                count += 1
            i += 1
        return count >= n

class Solution(object):
    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        '''
        early termination
        '''
        spots = 0
        for i in range(len(flowerbed)):
            if flowerbed[i] == 0 and (i==0 or flowerbed[i-1] == 0) and (i == len(flowerbed)-1 or flowerbed[i+1] == 0):
                flowerbed[i] = 1
                spots += 1
            if spots >= n:
                return True
        return False
        

#################################################
# Populating Next Right Pointers in Each Node II
#################################################
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        '''
        similar to the first one, but we may not be given a binary tree
        bfs might be better here
        '''
        if not root:
            return root
        q = deque([root])
        while q:
            size = len(q)
            for i in range(size):
                current = q.popleft()
                #making sure to alwasy connect to the left most first
                if i < size - 1:
                    current.next = q[0]
                if current.left:
                    q.append(current.left)
                if current.right:
                    q.append(current.right)
        return root

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        '''
        reduce space by eliminating queue structure
        treat each level as a lnked list
        we only move on the level N+1 hwen we are done esatblishing the pointers for N
        record a left most node for each level, this acts as our head
        note that the leftmost node always starts a new level
        curr is initally set to left most most aand we just treat this is a linked list traversal
        https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/discuss/37826/Concise-python-solution-9-lines-space-O(1)
        keep a variable for the current level
        and create a dummy variable holding head of linked list for level
        we also set a pointer to the dumm head and set pointer.next to store the first node of the cilde level for us
        keep another variable to connect each node on the same level without adavnacing ahaead
        '''
        #pointer to root
        old_root = root
        #dummy for linked iked
        dummy = Node(0)
        #current pointer
        current = dummy
        while root:
            while root:
                if root.left:
                    current.next = root.left
                    current = current.next
                if root.right:
                    current.next = root.right
                    current = current.next
                #move the root
                root = root.next
            #move the level, remember we have populated dummy from the inner loop
            #giving us access to the left most node from the current level
            #just descend here
            root = dummy.next
            current = dummy # a new level
            current.next = None
        
        return old_root

###################
#Spiral Matrix II
###################
#slow but it works!
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        '''
        similar to the first problem
        generate the elemnts first
        and add them to an output
        i need to output the i,j elements in spiral order
        and then each i,j will be just put into its positions
        '''
        #generate matrix for place holders
        matrix = [[0]*n for _ in range(n)]
        #numbers
        numbers = list(range(1,n*n+1))
        
        #pointer to numbers to adavnce
        pointer = 0

        start_row,start_col = 0,0
        end_row,end_col = len(matrix),len(matrix[0])
        
        while start_row < end_row or start_col < end_col:
            #right
            if start_row < end_row:
                for i in range(start_col,end_col):
                    matrix[start_row][i] = numbers[pointer]
                    pointer += 1
                start_row += 1 #don't do this row
            #down
            if start_col < end_col:
                for i in range(start_row,end_row):
                    matrix[i][end_col-1] = numbers[pointer]
                    pointer += 1
                end_col -= 1
            
            #left, going backwards now
            if start_row < end_row:
                for i in range(end_col-1,start_col-1,-1):
                    matrix[end_row-1][i] = numbers[pointer]
                    pointer += 1
                end_row -= 1
            
            #up
            if start_col < end_col:
                for i in range(end_row-1,start_row-1,-1):
                    matrix[i][start_col] = numbers[pointer]
                    pointer += 1
                start_col += 1
        return matrix

#another way to walk the spiral
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        '''
        apporach 1
        traverse layer by layer in spiral form
        for a given n we have floor((n+1)/2) layers
        we can traverse in all four directions and increment a counter populating the i,j fields
        '''
        matrix = [[0]*n for _ in range(n)]
        number = 1
        #walk over layers
        for layer in range((n+1)//2):
            #now go over directions
            #direction 1, left to right
            for col in range(layer,n-layer):
                matrix[layer][col] = number
                number += 1
            #direction 2, top to bottom
            for row in range(layer+1,n-layer):
                matrix[row][n-layer-1] = number
                number += 1
            #direction 3, right to left
            for col in range(layer+1,n-layer):
                matrix[n-layer-1][n-col-1] = number
                number += 1
            #direction 4, bottom ot top
            for row in range(layer+1,n-layer-1):
                matrix[n-row-1][layer] = number
                number += 1
        return matrix

#optimized traversal
class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        '''
        we can make improvments to the time complexity
        recallwe have 4 directions to walk in
        with directions 1, row is constant, but we advance col
        in direction 2, row is also constant, but we decrement col
        we can define a directions array [[0,1][0,-1],etc]
        and define a pointer to the array using modular arithmetic, in this case dir_pointer + 1 mod 4
        we change direction when we find the enxt row or col in a partircular direction has a non zero value, duh!
        '''
        matrix =[[0]*n for _ in range(n)]
        num = 1
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        dirr = 0
        row,col = 0,0
        while num <= n*n:
            matrix[row][col] = num
            num += 1
            #advance row and col t0 check
            r = (row + directions[dirr][0]) % n
            c = (col + directions[dirr][1]) % n
            #change direction when entry to update is nonzero
            if matrix[r][c] != 0:
                dirr = (dirr+1) % 4
            row += directions[dirr][0]
            col += directions[dirr][1]
            
        return matrix
            
#####################################################
# Pairs of Songs With Total Durations Divisible by 60
#####################################################
#TLE Obvie
class Solution(object):
    def numPairsDivisibleBy60(self, time):
        """
        :type time: List[int]
        :rtype: int
        """
        '''
        brute force would be to check all pairs and that their sum is divisble by 60
        then incrment a count do this first
        '''
        count = 0
        for i in range(0,len(time)):
            for j in range(i,len(time)-1):
                if (time[i] + time[j+1]) % 60 == 0:
                    count += 1
        return count

class Solution(object):
    def numPairsDivisibleBy60(self, time):
        """
        :type time: List[int]
        :rtype: int
        """
        '''
        we can write (a+b) % k = ((a % k) + (b % k)) % k
        in our case we have
        ((a % 60) + (b % 60)) % 60 = 0
        which means a % 60, b % 60, a % 60 + b % 60 = 60
        we pass through the array and for each element a, we want to know the number of elements b such that b % 60 = 0 if a %60 =0
        and
        b % 60 = 60 - a %60 if a % 60 does not equal 0
        there are 60 ways two songs could have length % 60 == 0
        a % 60 and b % 60 == 0
        a % 60 = 1 and b % 60 = 59
        ...
        a % 60 = 59 and b % 60 = 1
        we create a hash map holding the counts of the % 60 for each song (which is in range [0,59]) incremeting their count
        each time we hit an element we also check for the complements which is just 60 - elemtn % 60 indexing back into the array, and we incrment the count
        if elemnt % 0 = 0, and reaminder[0] tot eh result else add remainder[60 - t % 60]
        and also update the reaminders after upating count
        '''
        remainders = collections.defaultdict(int)
        count = 0
        for t in time:
            #if t is divisible by 60
            if t % 60 == 0:
                count += remainders[0]
            #now check the number of complements 60 - t % 60
            else: #similar to dp
                count += remainders[60 - t % 60]
            #update reaminder
            remainders[t % 60] += 1
        return count

####################
#Missing Ranges
####################
class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: List[str]
        """
        '''
        we are looking for nums in the range lower to upper that do not exsist in nums
        if we have the array
        [1,2,4,10,12] and limits (-1,-15)
        the result will be
        [-1->0,3,5->9,11,13->15]
        not that if two adjacent numbers in the nums array differ by one we don't include them
        if they differ by two, well just just take nums[i]
        if greater than 2, we take nums[i+1]- and nums[i+2] - 1
        we also need to affix the lower and upper limits to out nums array
        '''
        #add lower and upper bounds for edge cases
        nums = [lower-1] + nums + [upper+1]
        ranges = []
        #start from index 1 bease we need to compare differcne
        for i in range(0,len(nums)-1):
            #if nums[i+1] - nums[i] == 2, take nums[i]-1
            if nums[i+1] - nums[i] == 2:
                ranges.append(str(nums[i]+1))
            elif nums[i+1] - nums[i] > 2:
                ranges.append(str(nums[i]+1)+"->"+str(nums[i+1]-1))
        
        return ranges

##############################
#Binary Search Tree Iterator
#############################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator(object):

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        '''
        easy case first, we traverse the root in order adding an array
        then we manipluate a pointer to go to next and has next
        '''
        self.nodes = []
        self.pointer = -1
        def dfs(root):
            if not root:
                return
            dfs(root.left)
            self.nodes.append(root.val)
            dfs(root.right)
        dfs(root)

    def next(self):
        """
        :rtype: int
        """
        '''
        i need to move a pointer iteratively
        '''
        self.pointer += 1
        return self.nodes[self.pointer]


    def hasNext(self):
        """
        :rtype: bool
        """
        '''
        and then check 'inorder' which is just left, nope,right
        '''
        return self.pointer  < len(self.nodes) - 1

class BSTIterator(object):
    '''
    we can simulate recursion using a stack, this would allow for control during regular recursino
    init an empty stack which wil be used to similar.
    in the usual recursive call we would invoke once then go left, node, right
    the inital state will keep descending left until it hits a leaf
    for a given node root, the next smallest element will always be the left most element in tree
    the fist time next function is made the smalles element of the BST has to be returns
    the next smalest node wold be sitting at the top of the stack
    the best case when visiting a node would be if there were no childre
    if a node has a right childe, we dont need for the left because we've already done that going in order
    '''

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.root = root
        self.stack = []
        

    def next(self):
        """
        :rtype: int
        """
        #instead of creaitng a function to call invoke here
        #we want to keep going left until we can't, but as we go left
        while self.root:
            self.stack.append(self.root)
            self.root = self.root.left
        top = self.stack.pop()
        #return this value after traversung
        result = top.val
        #now the we have gotten the next, move right for the next call
        self.root = top.right
        return result
        

    def hasNext(self):
        """
        :rtype: bool
        """
        #if there is nothing else to pop we should be done
        #
        return len(self.stack) > 0 or self.root != None


#with creating another method
class BSTIterator(object):
    '''
    we can simulate recursion using a stack, this would allow for control during regular recursino
    init an empty stack which wil be used to similar.
    in the usual recursive call we would invoke once then go left, node, right
    the inital state will keep descending left until it hits a leaf
    for a given node root, the next smallest element will always be the left most element in tree
    the fist time next function is made the smalles element of the BST has to be returns
    the next smalest node wold be sitting at the top of the stack
    the best case when visiting a node would be if there were no childre
    if a node has a right childe, we dont need for the left because we've already done that going in order
    '''

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        #create instance var, since i'm not creatin a method function for inorder traverse
        self.root = root
        #stack init
        self.stack = []
        #get initla state
        while self.root:
            self.stack.append(self.root)
            self.root = self.root.left
        

    def next(self):
        """
        :rtype: int
        """
        #the next node to return would always be at the top of the stack
        #but we need to maintain the inariant if the node has a right childe
        top = self.stack.pop()
        if top.right:
            while top.right:
                self.stack.append(top.right)
                top.right = top.right.left
        return top.val

    def hasNext(self):
        """
        :rtype: bool
        """
        #if there is nothing else to pop we should be done
        #
        return len(self.stack) > 0


#joust a review for iterative inroder traversale of BST
stack = []
while True:
    if root:
        stack.append(root)
        root = root.left
    else:
        if not stack:
            break
        root = stack.pop()
        print(root.val)
        root = root.right

#########################
#Valid Mountain Array
##########################
#close but maybe thinking too hard?
class Solution(object):
    def validMountainArray(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        '''
        traverse the array finding the peak,
        then use two pointers from the peak to make sure boths sides are strictly increasing
        both pointers should reach the end with out a pleatua
        '''
        N = len(arr)
        for i in range(1,N-1):
            #peak
            if arr[i-1] < arr[i] < arr[i+1]:
                l,r = i,i
                while l > 0 and r < N-1:
                    #keep checking
                    if arr[l-1] < arr[l]:
                        l -= 1
                    if arr[r+1] > arr[r]:
                        r += 1
                #check ends
                if l == 0 and r == N-1:
                    return True
                else:
                    return False
                return True
        
        return False

class Solution(object):
    def validMountainArray(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        '''
        we can walk along the array strictly increasing
        if we get the point where we are no longer increasing, check that the peack is not at the end
        then walk down and return whether or not we got to the end
        '''
        N = len(arr)
        i = 0
        while i+1 < N and arr[i+1] > arr[i]:
            i += 1
        #until we can't
        if i == 0 or i == N-1:
            return False
        #walk down
        while i+1 < N and arr[i+1] < arr[i]:
            i += 1
        return i == N-1


###########################################
#Remove Duplicates from Sorted Array II
###########################################
#since popping from an array is O(N), the worst case for this would be N^2
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        elements should only appear at most twice
        go through the array and check that i + 1 is i if not delete it
        keep track of a counter and last seen number
        while loop but update N upon deletion
        '''
        count = 1
        i = 1
        while i < len(nums):
            #when adjacenet elements are the same incrment counter
            if nums[i] == nums[i-1]:
                count += 1
                #same but greater than 2, pop it off
                if count > 2:
                    del nums[i]
                    i -= 1
            #reset counter, not that we've popped them all off
            else:
                count = 1
            #we always advane our i
            i += 1

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        we can just overwrite unwanted duplicates
        we define two pointers i and k
        i goes across the array one at atime and j keeps track of the next location in the array that needs to be overwrited
        keep a count var that tracks the count of a particular element in the array, min count is always 1
        start a 1 
        if we find the the current element is the same as the previous we increment count
        if count exceeds 2, then we have a dplicate, move our i and not j
        if count less than 2 we cna move thelement from i to j
        if nums[i] != nums[i-1] then we have a new element, reset count to 1 and move j
        then j would have just advance all the way up to the last unwanted duplicate
        at which point we just return j
        
        in summary
        i pointer passes through the array
        when we get to a duplicate, we keep passsing the dupliate until a new elmenet
        once we get to the new element, we assing nums[i] to nums[j]
        then move our j!
        '''
        j = 1
        count = 1
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                count += 1
            else:
                count = 1
            #no duplicates, we just copy over, this won't hit until i finds a new occurence
            #at which point we can update nums[j] to nums[i]
            if count <= 2:
                nums[j] = nums[i]
                j += 1
                
        return j
            
        return len(nums)
            

#he first problem class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        another way
        keep a lenght pointer and only update (i.e increament 1 when we pass all unwatned duplicates)
        '''
        length = 0
        for i in range(len(nums)):
            if i == 0 or nums[i-1] != nums[i]:
                nums[length] = nums[i]
                length += 1
        return length
        

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        another way would be to just use a while loop with a single pointer
        and just keep popping the next i + 2 element if it is the same as i
        '''
        i = 0
        while i + 2 < len(nums):
            if nums[i] == nums[i+2]:
                nums.pop(i+2)
            else:
                i += 1
        return len(nums)
        
###############################################
#  Smallest Subtree with all the Deepest Nodes
##############################################
#so closeeee :(
#note thie could be rewritten to get the max depth recursively
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def subtreeWithAllDeepest(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        '''
        the deepest node will by the parent node of the deepest leaf
        level order BFS, but give reference to the parent node
        no dfs might be better
        we examine from a node, and if its child nodes are leaves possible candidate

        '''
        levels = []
        def dfs(parent,node, level):
            if not node:
                return 
            if not node.left and not node.right:
                levels.append((parent,level))
            parent = node
            dfs(parent, node.left, level+1)
            dfs(parent, node.right, level+1)
        dfs(root,root,0)
        #now just return the nodes with the deepest level
        max_level = 0
        for p,l in levels:
            max_level = max(max_level,l)
        ans = None
        for p,l in levels:
            if l == max_level:
                ans = p
                break
        return ans

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def subtreeWithAllDeepest(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        '''
        similar to my idea before, use hash the node with its level
        then dfs one more time on the hashmap
        tje first phase it ot idneity the nodes of tree that are depest
        sing annoatoin:
            if the node in question has max deppth, thats it
            botht he left and right child oa node have deepst desendant, answer is the parent
            otherwise if some child has an even deeper descendant, than the answer is that child
            otherise no answer
        algo:
            first dfs to a hash annotating the node
            second dfs, dfs again on the tree referencing the node in ht hassh
        '''
        depths = {}
        def dfs(node,level):
            if not node:
                return
            depths[node] = level
            dfs(node.left,level+1)
            dfs(node.right,level+1)
            
        dfs(root,0)
        max_depth = max(depths.itervalues())
        
        #now find the node with the maxdepth
        def dfs_answer(node):
            if not node:
                return
            if depths[node] == max_depth:
                return node
            #recurse for each subtree
            left = dfs_answer(node.left)
            right = dfs_answer(node.right)
            if left and right:
                return node
            else:
                return left or right
        return dfs_answer(root)

########################
#Burst Balloons
########################
class Solution(object):
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        at first glance we can think of this recursively, 
        we examine every possibilty of order for popping ballons, 
        since we pop one balloon every time we get N*(N-1)*(N-2)...1
        which is just N! time
        we can cache the set of existing balloons, since each balloon can be burst or not we we are incurtring extra time creatin a set of balloons each time
        which is still a worse case 2^N
        1. Divide a and conqure
        for every balloon we pop we can treat it is solving a problem for the left and right sides
        nums[:i] and nums[i+1:]
        and to find the optimal solution we check every optimal solution after bursting each balloon
        since we will find the optimal solution for every range in nums, we we burst every ballong in every range to find the optimal solutions we have N^2 times N N^3
        however if we try to divide our problem in the order where we burst ballons first, we run into adjaceny issues
        we can start in reverse order of how they were popped
        each time we add a ballon we can divide and conquer on its left and right sides
        https://leetcode.com/problems/burst-balloons/discuss/519207/python-solutions-with-visualized-explanations
        given the array
        [3,1,5,8]
        we can add 1 ballons at the head and tail (for edge cases if we pop the beginning and dn)
        [1,3,1,5,8,1]
        so here N= 6
        we define cache of matrix N*N, where dp[i][j] is ht emax contains gained by bursting ballongs from i to j
        no the question degenerates, finding dp[1][4]
        assume we popping the last ballong k, in our range i to j
        we define pointers:
            left = nums[i-1]
            right = nums[i+1]
            dp[i][j] = dp[i][k-1] + left *nums[k] + right + dp[k+1,j]
        we then recurse for each popped baloon in range of i to j and reference the dp array
        '''
        #append ones to head and tail
        nums = [1] + nums + [1]
        N = len(nums)
        #dp[i][j] max coins we get by bursting i to j
        dp = [[0]*N for _ in range(N)]
        
        def get_max(i,j):
            #memory
            if dp[i][j]:
                return dp[i][j]
            #gone past the last index
            elif i > j: 
                return dp[i][j]
            #get the amx coins
            max_coins = 0
            #loop through j+1 - i positions
            for k in range(i,j+1):
                left = nums[i-1]
                right = nums[j+1]
                coins_k_max = get_max(i,k-1) + left*nums[k]*right + get_max(k+1,j)
                max_coins = max(max_coins,coins_k_max)
            #put into memeory
            dp[i][j] = max_coins
            return dp[i][j]
        #invoke for all possible i to j
        get_max(1,N-2)
        #we want to get the max coins possible from i to j, this is just the frist row and N-2
        return dp[1][N-2]

#########################
#Palindrmo Partitioning
#########################
#well this was dumb...
class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        '''
        brute force, generate all substrings for all ranges and check is valid palindrome
        for len(3)
        3 + 2 + 1 substrings
        (16(16+1) / 2)
        O(272*3)
        '''
        def check(string):
            
        N = len(s)
        #all posible ranges
        for size in range(0,N):
            subs = []
            for i in range(0,len(s)-size):
                subs.append(s[i:i+size+1])
            #now check each sub reads as a palindrome
            for s in subs:

class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        '''
        first step, generate all possible substrings
        subtring decompisition recursively
        after generating all decompositoins, determing if each substring can be a palindrome
        if all possible subtrings are such, add them to results
        '''
        #first generate function for determin palindrome
        def palindrome(string):
            N = len(string)
            start,end = 0,N-1
            while start <=end:
                if string[start] != string[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        #second generate all possible substrings
        self.subs = []
        def rec_build(start_idx,path):
            #base case is when start_inx gets to the length of s
            if start_idx >= len(s):
                self.subs.append(path)
            
            #now recurse for on each length but move up start index
            for size in range(len(s)-start_idx):
                #check for palindrome
                if palindrome(s[start_idx:start_idx+size+1]):
                    #id palindrome, recurse on the otherside but not before adding the current substring
                    rec_build(start_idx+size+1,path+[s[start_idx:start_idx+size+1]])
        
        rec_build(0,[])
        return self.subs

class Solution(object):
    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        '''
        top down with memoziaton
        
        '''
        #special case
        if len(s) == 0:
            return []
        
        #function to check if string is palindrom
        def palindrome(string):
            start,end = 0,len(string)-1
            while start <= end:
                if string[start] != string[end]:
                    return False
                start += 1
                end -= 1
            return True
        
        #recursive function with memo, we hash the string to all possible substrings
        memo = {}
        def dfs(string):
            if string in memo:
                return memo[string]
            
            results = [] #for each string add its substring
            for i in range(len(string)):
                if palindrome(string[:i+1]):
                    #once we get to the end of the string we need to terminate
                    if i == len(string) - 1:
                        results.append([string[:i+1]])
                    #otherwise recurse on the other side
                    else:
                        subs = dfs(string[i+1:])
                        for s in subs:
                            results.append([string[:i+1]] + s)
            #dump into memo
            memo[string] = results
            return results
        
        return dfs(s)

###############################
#Squares of a Sorted Array
###############################
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        well just square each number then sort increasingly
        '''
        return sorted([num**2 for num in nums])

class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        we can use a two pointer approach and fill the array backwards
        taking the sqaure of the bigger absolution value of the L or R pointers
        '''
        N = len(nums)
        results = [0]*N
        l,r = 0, N-1
        for i in range(N-1,-1,-1):
            #compare left and right of numes
            if abs(nums[l]) > abs(nums[r]):
                square = nums[l]
                l += 1
            else:
                #take right
                square = nums[r]
                r -= 1
            results[i] = square*square
        return results
            
#####################
#Plus One Linked List
#####################
#hacky way, but i got it
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def plusOne(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        traverse the entire linked list, concat each node.val, conver to int, add 1, recreate the number using a linked list
        '''
        number = ""
        while head:
            number += str(head.val)
            head = head.next
        #add 1
        number = str(int(number)+1)
        #create new linked list
        dummy = ListNode()
        temp = dummy
        for char in number[:-1]:
            temp.val = char
            temp.next = ListNode()
            temp = temp.next
        temp.val = number[-1]
        return dummy

class Solution(object):
    def plusOne(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        if we have a case such as 1->2->3
        we just go all the way to right most digit this is not a 9 and a one to it
        1->2->4
        but if we have a 9 we need to carry over
        1->2->9
        1->3->0
        we can keep a sentinnel node holding a 0 value for edge cases if they are all nines and we gain a diigit
        if the sentinel node isnt zero, return sentinel, otherise return sentinenl.next
        algo:
            1. init setninel node as ListNode(0)
            2. find right most digit not euqal ot nine
            3. increase that node by 1
            4. set all following nices to zero
            5. return seninel node if 1
        '''
        #sentinely head
        dummy = ListNode(0) #to control for edge cases
        dummy.next = head
        not_nine = dummy
        
        #find right most not nine digit
        while head:
            if head.val != 9:
                not_nine = head
            head = head.next
        
        #add 1 to the not nine digit, should be pointing to digit that is not 9
        not_nine.val += 1
        #move to next, since it should be 0
        not_nine = not_nine.next
        
        #flip 9's to zeros, if there weren't any intermediate 9's this would neve execute
        #and the fist non_none should have been the only node upped 1
        while not_nine:
            not_nine.val = 0
            not_nine = not_nine.next
        
        #return dummy if a value, else return next
        if dummy.val:
            return dummy
        else:
            return dummy.next
        
#############################
#Validate Binary Search Tree
#############################
#so close, 57/77. i think it needs to hold for all cases, ie we need ranges for max and min
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        recall that a BST has the propery that from a node, its left is less than the current node
        and its right is greater than the node
        we can recurse on each node checking left and right
        if not, return false and stop
        '''
        def dfs(node):
            if not node:
                return
            if not node.left and not node.right:
                return True
            if node.left and node.right:
                return node.left.val < node.val and node.right.val > node.val
            if node.left and not node.right:
                return node.left.val < node.val
            if node.right and not node.left:
                return node.right.val > node.val
            dfs(node.left)
            dfs(node.right)
            
        return dfs(root)

#recursive with ranges
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        recursively checking from each node will not work, since a BST has the propery tat all nodes in the left substree are 
        we need to carry over the lower and upper limits everytime we descend left or rigth
        '''
        def dfs(node, low=float('-inf'), high=float('inf')):
            if not node:
                return True
            #the node we are examining must be in range, if not false
            if node.val <= low or node.val >= high:
                return False
            #recurse left and right but change rnages
            left  = dfs(node.left, low=low,high=node.val)
            right = dfs(node.right,low=node.val, high=high)
            return left and right
        return dfs(root)

#iterative with ranges
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        iterative solution with ranges
        '''
        if not root:
            return True
        stack = [(root,float('-inf'),float('inf'))]
        
        while stack:
            current, low, high = stack.pop()
            #if a node is null we need to keep checking
            if not current:
                continue
            if current.val <= low or current.val >=high:
                return False
            
            stack.append((current.left,low,current.val ))
            stack.append((current.right,current.val,high))
        return True

#recursive in order
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        recursive inorder traversal, we process each node and check that it is greater than the lower limit, if this is true, update limit to node, and process the next inordr node
        '''
        self.low = float('-inf')
        
        def dfs(node):
            if not node:
                return True
            #we check left, and if we can't return false
            if not dfs(node.left):
                return False
            if node.val <= self.low:
                return False
            self.low = node.val
            return dfs(node.right)
        
        return dfs(root)
        

#iterative in order, keep in back pocket like recursive pre,in,post
 stack = []
        lower = float('-inf')
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            print(root.val)
            root = root.right

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        iterative inorder traverse
        two while loops, first one need soemting in stack or a root
        while wehave a root we keep going left
        '''
        stack = []
        lower = float('-inf')
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= lower:
                return False
            lower = root.val
            root = root.right
        return True
        
########################
# 4Sum II
###################
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        '''
        well i can just brute force across all lengths
        '''
        count = 0
        N = len(A)
        
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        if A[i] + B[j] + C[k] + D[l] == 0:
                            count += 1
        return count

class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        '''
        brute force would be enumerating all combindatins
        a better approach would be to use three next loops and find sum a + b + c and then seach for its complement d == -(a+b+c) in the fourth array
        *note we need to track the frequency of each element in the fourth arrays (because reated complements increase the count) - hash to store counts
        we also notic that a + b == -(c+d)
        for we count sums of elements a + b using hash
        then enumerate elemnets from this and fourht searching for their complements
        '''
        count = 0
        mapp = {}
        #store counts of first two sumes
        for a in A:
            for b in B:
                if (a+b) in mapp.keys():
                    mapp[a+b] += 1
                else:
                    mapp[a+b] = 1
        #now find complements -(c+d in map)
        for c in C:
            for d in D:
                if -(c+d) in mapp.keys():
                    count += mapp[-(c+d)]
        return count
        
#nnote the use of .get in this sitatuion
        cnt = 0
        m = {}
        for a in A:
            for b in B:
                m[a + b] = m.get(a + b, 0) + 1
        for c in C:
            for d in D:
                cnt += m.get(-(c + d), 0)
        return cnt

#the general solution done recursivelu
class Solution(object):
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        '''
        we can generalize this for k/sum 
        where time would be O(num arrays / 2)
        using recursions
        we can finds for the first i/2 lists
        then recurse again finding the next i/2 lists, but this time counting the complements in the hash
        '''
        mapp = {}
        def addtohash(lists,i,summ):
            #pass in list of lists, stop after i//2
            if i == len(lists) // 2:
                mapp[summ] = mapp.get(summ,0) + 1
            else:
                for num in lists[i]:
                    #back tracking like, on to next array caruing over sum
                    addtohash(lists,i+1,summ + num)
        def getcomps(lists,i,comp):
            if i == len(lists):
                return mapp.get(comp,0)
            count = 0
            for num in lists[i]:
                count += getcomps(lists,i+1, comp - num)
            
            return count
        
        addtohash([A,B,C,D],0,0)
        return getcomps([A,B,C,D],len([A,B,C,D]) // 2,0)


##################
#Increasing Triplet Subsequence
###################
#55/61 nened to take care of special cases where adjacent elemtns are not incerasing
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        sliding window of size 3 would work but only for consevtuive indexes
        try it and see of we can get all of them
        '''
        for i in range(len(nums)-2):
            window = nums[i:i+3]
            if window[0] < window[1] < window[2]:
                return True
        
        return False

#well i tried...
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        i can use three pointers
        each start at the first element
        when i approach a case where first < second keep them there and advance second and third
        keep moving second and third until i find a triplet
        '''
        N = len(nums)
        first, second, third = 0,1,2
        while first < N and second < N and third < N:
            #always check triples
            if nums[first] < nums[second] < nums[third]:
                return True
            if first < N and second < N:
                while nums[second] <= nums[first]:
                    first += 1
                    second += 1
                third = second + 1
            if second < N and third < N:
                while nums[third] <= nums[second]:
                    third += 1
        
        #final check
        if nums[first] < nums[second] < nums[third]:
            return True
        else:
            return False
        

class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        we can do this in one pass
        we just need to find numbers in increasing order
        we assign two numbers first and second both to inf
        traverse the array and if n <= first update first to n
        else if n <= second update second to n
        else return True
        if nums is in descending order we would always update first, and the loop would enver terminrate return False
        in ascending order first and second get updated with the next number being grearter than both so return True
        we're always looking for the next smallest first and second numbers
        there is a special case we need to think about
        [1,2,0,3]
        first = 1
        second = 2
        first = 0
        here we would return true
        becasue there exists another number before second number which is bigger than the last update first num but smalled than second num
        '''
        first, second = float('inf'),float('inf')
        for n in nums:
            if n <= first:
                first = n
            elif n <= second:
                second = n
            else:
                return True
        False

#trying to stored the indices, but i think we need to use recursion
first, second = float('inf'),float('inf')
first_idx,second_idx,third_idx = 0,0,0
triplets = []
for i,n in enumerate(nums):
    if first_idx < second_idx < third_idx:
        triplets.append([first_idx,second_idx,third_idx])
    if n <= first:
        first = n
        first_idx = i
    elif n <= second:
        second = n
        second_idx = i
    else:
        third_idx = i
print triplets

#generatinte combinations
#i cant figure out how to get triplets, but for now just review geerationg combnindatio
#and permutations
class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        using recursion to generate indices of valid triplets
        '''
        triplets = []
        
        def rec_build(nums, build):
            if len(build) == 3:
                triplets.append(build)
            for i in range(len(nums)):
                rec_build(nums[i+1:],build+[nums[i]])
        rec_build(nums,[])        
        print triplets

#permuations of ength 3
#keep this in back pocket, both cmobindatins and permuations
#try to figure out how to get certain length permutations
#figure out rcursion
#perm([a,b,c,d]) = perm([a] + perm[b,c,d])...
def permutation(s):
   if len(s) == 1:
     return [s]

   perm_list = [] # resulting list
   for i,a in enumerate(s):
     remaining_elements = [x for j,x in enumerate(s) if i != j]
     z = permutation(remaining_elements) # permutations of sublist

     for t in z:
        perm_list.append([a] + t)

   return perm_list
   print(permutation(['cat','dog','bunny']))

##################
#Cherry Pickup II
###################
class Solution(object):
    def cherryPickup(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        '''
        well this was a hard problem
        we can only move in three drections
        reaching bottom row means they are done, cannot move left and right staying at teh same row, always moving down
        we need to move both robots at the same time!
        keeping track both states of two robots is too much, we need to reduce the states
        we can deinfe the DP state as (row1,col1,row2,col2)
        if we move them synchronsly, then we get row1=row2
        so we get our dp states as (row,col1,col2)
        we can define dp(row,col1,col2) as the max cherries we can pick if robot1 stars at row,col2 and robot 2 starts at row,col2
        bases cases are that both roboots start at bottom line
        we need to add the maximum cherries robots can pick in the future
        we have 9 possible movments from a sinlge dp state 3*3 (look at link)
        https://leetcode.com/problems/cherry-pickup-ii/solution/
        (left down,left down), (left down, down), (left down, right down).. etc
        the max cherries would be max(dp(all 9 states))
        algo:
            define dp function taking in 3 arguments our of our state (row,col1,col2)
            dp function returns max cherries if robot1 starts at row,col1 and robot 2 starts at row, col2
            collect cherry at (row,col1) and (row, col2) do not doubkle count at col1 == col2
            if we do not reach the last row, add max cherries that can be picked in the future
        '''
        rows = len(grid)
        cols = len(grid[0])
        cache = {}
        
        def dfs(row,col1,col2):
            #retrieving from memory
            if (row,col1,col2) in cache:
                return cache[(row,col1,col2)]
            #boundary check
            if col1 < 0 or col1 >= cols or col2 < 0 or col2 >= cols:
                return float('-inf') #becase we are taking max
            #current cell
            result = 0
            result += grid[row][col1]
            #don't double count if they get to the ssame col
            if col1 != col2:
                result += grid[row][col2]
            #recurse as long as we are not in the final row
            if row != rows -1:
                #dfs from all 9 states
                #stor possible cherry picks ups from all 9 directions
                temp = []
                for c1 in [col1,col1+1,col1-1]:
                    for c2 in [col2,col2+1,col2-1]:
                        temp.append(dfs(row+1,c1,c2))
                #take the max from all of these
                result += max(temp)
            #put back into memeory
            cache[(row,col1,col2)] = result
            return result
        
        return dfs(0,0,cols-1)

#bottom up solutioon
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [[[0]*n for _ in range(n)] for __ in range(m)]

        for row in reversed(range(m)):
            for col1 in range(n):
                for col2 in range(n):
                    result = 0
                    # current cell
                    result += grid[row][col1]
                    if col1 != col2:
                        result += grid[row][col2]
                    # transition
                    if row != m-1:
                        result += max(dp[row+1][new_col1][new_col2]
                                      for new_col1 in [col1, col1+1, col1-1]
                                      for new_col2 in [col2, col2+1, col2-1]
                                      if 0 <= new_col1 < n and 0 <= new_col2 < n)
                    dp[row][col1][col2] = result
        return dp[0][0][n-1]

##########################
#Decoded String at Index
#########################
#TLE, that weird edge case 'aaa345325454'
class Solution(object):
    def decodeAtIndex(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        decoded = ""
        N = len(S)
        for i in range(N):
            if ord('2') <= ord(S[i]) <= ord('9'):
                #get the digit
                d = int(S[i]) - 1
                to_add = decoded*d
                decoded += to_add
            else:
                decoded += S[i]
        return decoded[K-1]

#this one is tricky!
class Solution(object):
    def decodeAtIndex(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        '''
        imageine if we have a string that repeats it selft
        appleappleapple
        the Kth index char will always be K %len(string) becaust it repeats itselft
        first find the length of the decoded string
        hi2bob3
        ((2*2)+3)*7
        hihibobhihibobhihibob
        we then work backars keeping the track of the size
        if we ecntouner a digit, we need to divide the size by that (rmember we multipled in the begiing)
        and if not  just reduce the size
        '''
        size = 0
        for i,ch in enumerate(S):
            if ch.isdigit():
                size *= int(ch)
            else:
                size += 1
        K -= 1 #zero indexing
        #now go backwards
        for j in range(i,-1,-1):
            if S[j].isdigit():
                size //= int(S[j])
                K %= size
            elif size == K + 1:
                return S[j]
            else: 
                size -= 1