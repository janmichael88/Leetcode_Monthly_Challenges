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
        #whenever the decoded string is just a string repeated d times we return K % size
        #whenever the decoded string would equal some word repeated d times, we can reduce K tp K % size
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

###################
#Smallest Range II
###################
#close one
class Solution(object):
    def smallestRangeII(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        '''
        might be helpful to look at the solution fo Smallest Range I
        if we have A[i] < A[j], we dont need to consider when A[i] goes down
        while A[j] goes up 
        this is because the interval (A[i] + K, A[j]-K) is a subset of the interval (A[i]-K, A[j]+K)
        this means it is never worse to choose (up,down) instead (down,up)
        we ca prove this claim that one interval is a subset of another by showing that both A[i] + K and A[j] - K are between A[i] - K and A[j] + k
        for a sorted array A, say A[i] is the largest that i goes up
        then A[0] + K, A[i] + K, A[i+1] - K, A[len(A)-1] - K are the only candidates for calculathe the answer
        
        '''
        A.sort()
        #sort to get the initial answer
        curr_min, curr_max = A[0], A[-1]
        curr_result = curr_max - curr_min #minimuze this
        for i in range(len(A)-1):
            curr_min = min(curr_min,A[i]-K,A[i]+K)
            curr_max = max(curr_max, A[i+1]-K, A[i+1]+K)
            curr_result = min(curr_result, curr_max-curr_min)
        return curr_result

class Solution(object):
    def smallestRangeII(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        '''
        might be helpful to look at the solution fo Smallest Range I
        if we have A[i] < A[j], we dont need to consider when A[i] goes down
        while A[j] goes up 
        this is because the interval (A[i] + K, A[j]-K) is a subset of the interval (A[i]-K, A[j]+K)
        this means it is never worse to choose (up,down) instead (down,up)
        we ca prove this claim that one interval is a subset of another by showing that both A[i] + K and A[j] - K are between A[i] - K and A[j] + k
        for a sorted array A, say A[i] is the largest that i goes up
        then A[0] + K, A[i] + K, A[i+1] - K, A[len(A)-1] - K are the only candidates for calculathe the answer
        #why is it -2k?
        suppose that array A i sorted and we have i < j and A[i] < A[j]
        to minimuze the difference between A[i] and A[j] we need to increment A[i] by K and decrement A[j] by k so min becomes A[i] - K and max becomes A[j] + K
        taking the difference we have A[j] - K - (A[i] + K) = A[j] - A[i] - 2K
        https://leetcode.com/problems/smallest-range-ii/discuss/980294/Python-O(n-log-n)-solution-explained 
        '''
        #sort
        A.sort()
        N = len(A)
        result = float('inf') #minimize this
        #pass the array and for each element examine its candidates
        #it can either be the min, the max - 2*k, the ith element, or the i+1 element -2*k
        #find the min and max for each candidate in the range and update the min result
        for i in range(N-1):
            candidates = [A[0], A[-1]-2*K, A[i], A[i+1]-2*K]
            result = min(result, max(candidates)-min(candidates))
        
        return min(result, A[-1]-A[0])

#another way of thinking about it
def smallestRangeII(self, A, K):
        A.sort()
        res = A[-1] - A[0]
        for i in range(len(A) - 1):
            big = max(A[-1], A[i] + 2 * K)
            small = min(A[i + 1], A[0] + 2 * K)
            res = min(res, big - small)
        return res

#better explanation
#https://leetcode.com/problems/smallest-range-ii/discuss/173495/Actual-explanation-for-people-who-don't-understand-(I-hope)
#given a sorted Array we pass through it
#we assume that A[i] is the max, for this to be true we must also assert that A[-1] - K is less than A[i]
#we take the max
#to find the min, we assume that A[i+1] is smaller than A[i] so we subrtarct K
#in doing so, we assert that A[0] is no longer the min so we add K
#then we just update the result  
class Solution:
    def smallestRangeII(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        A.sort()
        first, last = A[0], A[-1]
        res = last - first
        for i in range(len(A) - 1):
            maxi = max(A[i] + K, last - K)
            mini = min(first + K, A[i + 1] - K)
            res = min(res, maxi - mini)
        return res


#######################
#Balanced Binary Tree
#######################
#so close! 203/228
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        a balanced binary tree is a tree in which every substree does not differ by a height more than 1
        the problem then becomes get the heights for the left and right subtrees
        '''
        if not root:
            return True
        #function to get height, i could always implement bfs
        def dfs_height(node):
            if not node:
                return 0
            left = dfs_height(node.left)
            right = dfs_height(node.right)
            return max(left,right) + 1
        
        #now apply heigh functino for each node
        def dfs_main(node):
            if not node:
                return -1
            if abs(dfs_height(node.left) - dfs_height(node.right)) >1:
                return False
            else:
                return True
            dfs_main(node.left)
            dfs_main(node.right)

            
        return dfs_main(root)
            

#wooohoo
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        a balanced binary tree is a tree in which every substree does not differ by a height more than 1
        the problem then becomes get the heights for the left and right subtrees
        #nlogn because we call dfs_main on each node, and height is called lonN times
        '''
        if not root:
            return True
        #function to get height, i could always implement bfs
        def dfs_height(node):
            if not node:
                return 0 #can be -1 too and would still work
            left = dfs_height(node.left)
            right = dfs_height(node.right)
            return max(left,right) + 1
        
        #now apply heigh functino for each node
        def dfs_main(node):
            if not node:
                return True
            #now check heights are within 1
            if abs(dfs_height(node.left) - dfs_height(node.right)) <2:
                #now we must check that subtrees are also balanced, this is the kicker
                #notice how in this call we nest within an if block
                #not just the usual dfs after the elif blocks
                if dfs_main(node.left) and dfs_main(node.right):
                    return True
                else:
                    return False
        return dfs_main(root)
#with cachine
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        #cacahing height results into memory
        a balanced binary tree is a tree in which every substree does not differ by a height more than 1
        the problem then becomes get the heights for the left and right subtrees
        
        '''
        if not root:
            return True
        self.cache= {}
        #function to get height, i could always implement bfs
        def dfs_height(node):
            if not node:
                return 0
            if node in self.cache:
                return self.cache[node]
            left = dfs_height(node.left)
            right = dfs_height(node.right)
            result = max(left,right) + 1
            #put into cache
            self.cache[node] = result
            return result
        
        #now apply heigh functino for each node
        def dfs_main(node):
            if not node:
                return True
            #now check heights are within 1
            if abs(dfs_height(node.left) - dfs_height(node.right)) <2:
                #now we must check that subtrees are also balanced, this is the kicker
                #notice how in this call we nest within an if block
                #not just the usual dfs after the elif blocks
                if dfs_main(node.left) and dfs_main(node.right):
                    return True
                else:
                    return False
            
        return dfs_main(root)
            
#optimize O(N)
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        #in the first approach, we performed redundanct caclculations in calling the ehgiht functions
        #instead of top down we go bottom up
        #we can remove redundcany by first recursino on the children of curr node
        #and then use their computer height to determing whether node current node is balanced
        #Check if the child subtrees are balanced. If they are, use their heights to determine if the current subtree is balanced as well as to calculate the current subtree's height.
        def helper(node):
            #base case, empty tree with node is balance
            if not node:
                return True,0
            #check subtrees to see if they are balanced
            left,left_height = helper(node.left)
            if not left:
                return False,0
            right,right_height = helper(node.right)
            if not right:
                return False, 0
            #if subtrees are balanced, check if the current tree is balance
            if abs(left_height - right_height) < 2:
                return [True, 1+max(left_height,right_height)]
            else:
                return [False,1+max(left_height,right_height)]
                
        return helper(root)[0]

###########################################
#Find Nearesrest Right Node in Binary Tree
###########################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def findNearestRightNode(self, root, u):
        """
        :type root: TreeNode
        :type u: TreeNode
        :rtype: TreeNode
        """
        '''
        using two Queues
        one for the current level, and one for the next level
        the idea is to pop the nodes one by one from the current level and push their children on the the next q level
        algo:
            while nextlevel q is not empty:
                initiatine curr = next
                and empty next
            while curr level is not empty
                pop node from q
                if node is u return next node if thehre is one,else null
                add children
        '''
        if not root:
            return []
        next_level = deque([root])
        while next_level:
            #for not set current to next and make a new next
            current_level = next_level
            #new next
            next_level = deque()
            #while we have a current
            while current_level:
                node = current_level.popleft()
                if node == u:
                    if current_level:
                        #we need to get the next node if there is something in the q
                        return current_level.popleft()
                    else:
                        return None
                #else ad
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)

class Solution(object):
    def findNearestRightNode(self, root, u):
        """
        :type root: TreeNode
        :type u: TreeNode
        :rtype: TreeNode
        """
        '''
        BFS but keep track of the length of the q
        move through the q
        and keep track of i
        the node is the right most if i == len(q) - 1
        '''
        if not root:
            return None
        q = deque([root])
        while q:
            size = len(q)
            for i in range(size):
                node = q.popleft()
                if node == u:
                    if i == size - 1:
                        return None
                    else:
                        return q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)

#########################
#Next Geater Element III
#########################
#sooooo close
#the approach is right, but needs tweaking
class Solution(object):
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        the next greater digit is just going to one more the smalest 10s place
        all i need is for that place to be increament by at least 1
        if i can't try the 10's place
        '''
        str_n = list(str(n))
        N = len(str_n)
        digit = 0
        while digit < N:
            #scan the sring the find the next greatest digit
            for i in range(N):
                if int(str_n[i]) > int(str_n[digit]):
                    #found my i
                    break
            if i != digit:
                #swap
                str_n[i], str_n[digit] = str_n[digit], str_n[i]
                #join and get the number
                number = int("".join(str_n)) 
                return number if number > n else -1
            
            else:
                digit +=1
        
        return -1

class Solution(object):
    def nextGreaterElement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        https://leetcode.com/problems/next-greater-element-iii/discuss/983076/Python-O(m)-solution-explained
        lets start with an example :
        say we are given n = 234157641
        can we get a larger digit starting with 2, yes
        how about 23, yes
        234, yes
        2341,yes,
        23415...no
        the last part 7641 is decresing, so we cannot make a lrger digit
        algo:
            start from the end and look for an increasing pattern in this cases 7641
            if it happens that al the numbers in n have in increase patter, we cant do it (54321) cant be done
            no we need to find the first digit in out ending which is less or euql to digits[i-1]
            in our example we have 57641, we can replace 5 with 6! 67541
            finally weed need to revesre the last figits
            61457, we reverse because we want the smallest sequence after we have swppaed our found digit (remember the sequence is increasing)
            
        '''
        digits = list(str(n))
        #start from end
        i = len(digits) - 1
        #find the idx starting increasing sequence
        while i - 1 > 0 and digits[i] <= digits[i-1]:
            i -= 1
        
        #if we've gone through the sequence can't be done
        if i == 0:
            return -1
        #mark start of decreasing
        j = i
        #go until we get a larger digit than the one a j
        while j + 1 < len(digits) and digits[j+1] > digits[i-1]:
            j += 1
        #now swap
        digits[i-1],digits[j] = digits[j],digits[i-1]
        #flip
        digits[i:] = digits[i:][::-1]
        result = int(''.join(digits))
        #edge case if results is too big
        return result if result < 1<<31 else -1

######################
#Swap Nodes in Pairs
######################
#close on3 43/55
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None:
            return head

        i = 0
        dummy = head
        temp = dummy #return dummy
        while temp.next:
            #swap every other
            if temp.val:
                if i % 2 == 0:
                    current_val = temp.val
                    temp.val = temp.next.val
                    temp.next.val = current_val
            temp = temp.next
            i += 1
        
        return dummy

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        good illustration
        https://leetcode.com/explore/featured/card/december-leetcoding-challenge/572/week-4-december-22nd-december-28th/3579/
        allocate a dummy and set dummy.next to head
        moveing point cur set to dumm
        while moving pointer has a next and a next next
            first pointer co curr.next
            second pointer to curr.next.next
            swap cur next with sec
            and first next to sec next
            move back
            advance cur
        '''
        #edge case
        if not head or not head.next:
            return head
        dummy = ListNode(0)
        dummy.next = head #return dummy.next
        curr = dummy
        while curr.next and curr.next.next:
            #reference first ands econd
            first = curr.next
            second = curr.next.next
            #swap
            curr.next = second
            first.next = second.next
            second.next = first
            #advance two
            curr = curr.next.next
        return dummy.next

    class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        recursion
        reach the end of the linked lists in steps of two 
        in every function call we take out tow nodes which would be swappaed
        and the remaining node inthe next rec call
        we can do this recursively because a sublist is still part of the linked list
        algo:
            start the recursion with head of node
            every call is responsible for swapping a pair of nodes
            next rec is made by calling the functioniwth the head of next pair o dnoes
            once we get the pointer to the remaining swapped list from the recursion call
            we can swap the first ands econd node
            the nodes in the current recursive call and then return the pointer to the secondNode since it will be the new head after swapping.
            we are really swapping the last two first and the we back track
        '''
        if not head or not head.next:
            return head
        #give reference to the nodes
        first = head
        second = head.next
        
        #swap and recurse
        # Swapping
        first.next  = self.swapPairs(second.next)
        second.next = first

        # Now the head is the second node
        return second

####################
#Diagnoal Traverse
###################
class Solution(object):
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        '''
        (0,0), (0,1), (1,0), (2,0) (1,1),(0,2),(1,2),(2,1),(2,2)
        ether going diag up or diag down
        start diag up
        imagine the problem where we just went down a diagnoal
        from the starting point it would just
        [row+1,col-1]
        we ssimply need to reverse the odd numbered diagnoalds before we add the elemnts ot the final results array
        algo:
            init result array
            we would have an outer loop that will go over each of the diagonals one by one
            inner looping going along a diagonola (keep iterating until one of indices goes out of bounds)
            
        '''
        if not matrix or not matrix[0]:
            return []
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        results = []
        temp = [] #store current diag elemetns
        
        #heads of each temp array will be all elements in first row and last column
        #so we loop for each of them
        for head in range(rows+cols+1):
            temp = []
            #establish starts of diagonsgls
            if head < cols:
                r = 0
            else:
                r = head - cols + 1
            
            if head < cols:
                c = head
            else:
                c = cols - 1
            
            while r < rows and c >= 0:
                temp.append(matrix[r][c])
                r+= 1
                c -= 1
            
            if head % 2 == 0: #reverse
                results += temp[::-1]
            else:
                results += temp
        
        return results
        
class Solution(object):
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        '''
        a propety about a matrix is that elements lying on the same diagnoal shar the same sum
        i can traverse the matrix and dump the summed indices into a hash mapped to alist
        we then go through the hash in order of diagnolas and for every even one reverse it
        '''
        if not matrix or not matrix[0]:
            return []
        mapp = {}
        results = []
        rows = len(matrix)
        cols = len(matrix[0])
        for r in range(rows):
            for c in range(cols):
                if r + c not in mapp:
                    mapp[r+c]= [matrix[r][c]]
                else:
                    mapp[r+c].append(matrix[r][c])
        #now build the results
        for k,v in mapp.items():
            if k % 2 == 0:
                results += mapp[k][::-1]
            else:
                results += mapp[k]
        return results
                
        
##################
#Decode Ways
##################
#well good review on subsequence patitionsing
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        this is a recursion, which means we can do top down with a memo
        similar to subsequence decomposition
        partitioning to get sub sequences
        '''
        self.ways = 0
        def rec_build(start_idx,build):
            if start_idx >= len(s):
                #check that each is between 1 and 26
                for num in build:
                    if int(num) <= 0 or int(num) > 26:
                        return
                self.ways += 1
            for size in range(len(s)-start_idx):
                rec_build(start_idx+size+1,build+[s[start_idx:start_idx+size+1]])
        rec_build(0,[])
        return self.ways

class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        at any one time we need to consinder taking one digit or two digits
        from the recursion tree, we only do a function call when we have a valid mapping from either usone one digit or two
        we can take care of overlapping subproblems using a cache
        if there was just a single didgit decode, then there is only one choice to make at teach step
        at any given time for a string, we enter a recursion only after successfuly decoding two digits to a single char
        if a given path leads to the end of the string, this means we could have succefully decoded the string
        algo:
            enter recursion with starting index 0 (move through using this pointer)
            terminate the case, we check for the end of the string, and return 1 for valid way
            every time we enter a recursion for a subg string we terminate of the digit is a zero (it won't add to the number of ways)
            we can add the result to a memo for the substrin we are on and return the answer
            If the result is already in memo we return the result. Otherwise the number of ways for the given string is determined by making a recursive call to the function with index + 1 for next substring string and index + 2 after checking for valid 2-digit decode. The result is also stored in memo with key as current index, for saving for future overlapping subproblems.
        '''
        memo = {}
        def rec_build(start_idx,s):
            #base case, moved pointer to end, valid way
            if start_idx == len(s):
                return 1
            if start_idx == len(s) - 1:
                return 1
            #at any point a char is zero, we cant do anything
            if s[start_idx] == '0':
                return 0
            
            #memo retrieval
            if start_idx in memo:
                return memo[start_idx]
            
            #keep carrying the answer so long a the digits are less thenn 26
            if int(s[start_idx:start_idx+2]) <= 26:
                result = rec_build(start_idx+1,s) + rec_build(start_idx +2,s)
            else:
                result = 0
            memo[start_idx] = result
            return result
        if not s:
            return 0
        if s == 0:
            return 0
        
        return rec_build(0,s)

#not as inutitive
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        the dp solution is not as intutive
        we allocate a 1d dp array, and read the cell as the number of ways decoding string s from 0 to i-1
        we must initialize the first two positions (for the cases where we look back two)
        it follows that the general rule us ust dp[i] = dp[i-1] + dp[i-2]
        algo:
            if the string is empty or null return 0
            init dp[0] = 1; to allow us to at least look back 2
            if the first cha is zero, then we can't decode it so that becomes 0, otherwie 1
            iteratore the dpp start at 2 (index into sting using i -1)
            now we chech if a vlaid single digit decode is possible, this just means looking at index[s-1] is non zero
            f the valid single digit decoding is possible then we add dp[i-1] to dp[i]. Since all the ways up to (i-1)-th character now lead up to i-th character too.
            We check if valid two digit decode is possible. This means the substring s[i-2]s[i-1] is between 10 to 26. If the valid two digit decoding is possible then we add dp[i-2] to dp[i].
        '''
        if not s:
            return 0
        dp = [0]*(len(s)+1)
        
        #init the first two spots
        dp[0]  = 1
        if s[0] == '0':
            dp[1] = 0
        else:
            dp[1] = 1
        
        #go into the second cell of the dp (i-1)
        for i in range(2,len(dp)):
            if s[i-1] != '0':
                dp[i] += dp[i-1]
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]
        return dp[-1]

################
#Jump Game IV
################
#omg my first hard problem pretty much solved in 15 mins!
#21 of 28 sooooo close, job job though!
class Solution(object):
    def minJumps(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        treat as a graph problem
        i can build a graph of n nodes, and from nth node list indices to where i can go from
        then start bfs fomr node 0 and keep a distance, answer is the distance when i get to the n-1 node
        '''
        N = len(arr)
        adj = defaultdict(list)
        for i in range(N):
            for j in range(0,N):
                if arr[j] == arr[i] and j !=i:
                    adj[i].append(j)
            if i == 0:
                adj[i].append(i+1)
            elif i == N-1:
                adj[i].append(i-1)
            else:
                adj[i].append(i+1)
                adj[i].append(i-1)
        #man that sucked, now bfs from node 0
        visited = set()
        q = deque([(0,0)])
        while q:
            current, distance = q.popleft()
            if current == N-1:
                return distance
            if current not in visited:
                for neigh in adj[current]:
                    q.append((neigh,distance+1))
            visited.add(current)


class Solution(object):
    def minJumps(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        bfs problem with a twist
        create deafult dict the usal way 
        consider three types of neighbors from n -> n+1 and n-2 and n+p where arr[n] == arr[n+p] and p != 0
        keep a visited set
        but also keep a visitedgroups set
        imagine we have arr = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
         Then first time we see 1, we visit all other 1. Second time we see 1, we do not need to check its neibors of type 3, we already know that we visited them. Without this optimization time complexity can be potentially O(n^2).
         Consider we have [a,b,b,b,b,b,b,b,b,b,b,b,c]

If we are at the first b (of index 1) we can either get to a or to any other b including the last b in a single step.

If we are at the last b then we can either get to c or to any other b including the first b in a single step.

Additionally we notice that it is always sub-optimal to move to any b that is between the first b and the last b.

So instead we can create a new array that looks like [a,b,b,c] and run BFS on that, meaning for any array containing n consecutive elements we only keep the first and last occurrences of this element.
        we do bfs for each of the nodes neighbors check type and 2 and if we did not visit of this type we can check for type 3
        #note, making the adj list with type 3 neighbors caused the TLE
        '''
        #make adj in the following way: num:index
        N = len(arr)
        adj = defaultdict(list)
        for i,num in enumerate(arr):
            adj[num].append(i)
        
        q = deque([(0,0)])
        visited_nodes = set()
        visited_groups = set() #this is  set of visited values
        
        while q:
            current, distance = q.popleft()
            if current == N-1:
                return distance
            
            #now check neigbors + and - currnet node
            for neigh in [current+1,current - 1]:
                #remember neigh must be in bounds
                if 0 <= neigh < N and neigh not in visited_nodes:
                    visited_nodes.add(neigh)
                    q.append((neigh, distance + 1))
            
            #now we must check for type 3 neighbors matching value
            if arr[current] not in visited_groups: #if we havent seen this value
                for neigh in adj[arr[current]]:
                    #find the value
                    if neigh not in visited_nodes:
                        visited_nodes.add(neigh)
                        q.append((neigh, distance+ 1))
                visited_groups.add(arr[current])

################
#Reach a Number
###############
#i cant figure out how to start the recursion
class Solution(object):
    def reachNumber(self, target):
        """
        :type target: int
        :rtype: int
        """
        '''
        starting at zero
        we can only go in n steps at the nth step either left or right
        brute force would be to examine all left right step combindations until i get the target
        think recursino tree, i can either take -n or +n steps
        '''
        def rec_step(current,steps,target):
            if current + steps == target or current - steps == target:
                return steps
            else:
                print current
                rec_step(current+steps,steps+1, target)
                rec_step(current-steps,steps+1,target)
        
        return rec_step(0,1,target)


class Solution(object):
    def reachNumber(self, target):
        """
        :type target: int
        :rtype: int
        """
        '''
        starting at zero
        we can only go in n steps at the nth step either left or right
        brute force would be to examine all left right step combindations until i get the target
        think recursino tree, i can either take -n or +n steps
        the targets will always by the nth sum of
        sum_{i=1}^i=N x_{n}
        if we pass the same we can get that passed sum only of it is divisble by 2
        https://leetcode.com/problems/reach-a-number/discuss/990901/%22Python%22-easy-explanation-blackboard
        but before that we must show that the min steps for target is the same as abs(target)
        examine the target 5
        which is just 0 + 1 + 2 + 3
        i could have easily just gotten to -5 buy reverse
        0 -1 -2 -3
        you can always add a 1 if you are behind -4+5-6+7-8+9 = 3
        '''
        step, summ = 0,0
        #keep going until our sum either goes past our target or is the target
        target = abs(target)
        while summ < target:
            summ += step
            step += 1
        #get the different between where we are at now and target, the goal is to get rid fo the different to reach target
        #for the ith move, it we switch the right move to the left, the change in sum whill b 2*i
        while (summ - target) % 2 != 0:
            summ += step
            step += 1
        
        return step - 1

#better explanation
#https://leetcode.com/problems/reach-a-number/discuss/188999/Throw-Out-All-Fucking-Explanations-This-is-nice-explanation-(c%2B%2B)-I-think-.......
class Solution(object):
    def reachNumber(self, target):
        """
        :type target: int
        :rtype: int
        """
        target = abs(target)
        step = 0
        summ = 0
        while True:
            step += 1
            summ += step
            if summ == target:
                return step
            elif summ > target and (summ - target) % 2 == 0:
                return step


#############################################
# Pseudo-Palindromic Paths in a Binary Tree
############################################
#woooooooo
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pseudoPalindromicPaths (self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        well the naive way would be to enumerate all possible paths, and check if each path is a permutation of itself (just count it)
        if it is it is psueo palindromic
        well hint 1 one helps
        keep track if freqeunce counts of chars
        '''
        self.temp = []
        self.ways = 0
        
        def paths(node,path):
            if not node:
                return
            if not node.left and not node.right:
                self.temp.append(path+[node.val])
            paths(node.left,path+[node.val]) #rember for paths you need to pass it along this took a long time to get
            paths(node.right,path+[node.val])
 
        paths(root,[])
        #now get freq counts for each path and observer that at most digit has odd frequencey
        for path in self.temp:
            count = Counter(path)
            odds = 0
            for k,v in count.items():
                if v % 2 != 0:
                    odds += 1
            if odds == 1 or odds == 0:
                self.ways += 1
                
        return self.ways

class Solution(object):
    def pseudoPalindromicPaths (self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        i've done the recursive all paths traversal,
        now lets try the iterative but we compute freq counts on the fly using bitsh shifts
        path will be defind a binary type with slots 1 to 9
        if at most one bit is set, it must be a power of two
        XOR of zero and a bit results in that bit
        XOR of two equal bits (even if they are zeros) results in z ero
        hence, one could see that a bit in a path only if it appears an odd number of times
        #we can compute the occurneces of each digit in the correspdong bit by:
        path = path ^ (1 << node.val)
        now to ensure that at most one digit it must be a power of two
        this could be done by turning off the right most bit
        path & path - 1
        so in general if its a leaf we can check at most one digit having an odd frequence
        if path & (path -1) == 0:
        count += 1
        '''
        count = 0
        stack = [(root,0)]
        while stack:
            node,path = stack.pop()
            if node:
                #compute occruecne of each digit
                path = path ^ (1 << node.val)
                #now we check if leaf
                if not node.left and not node.right:
                    #check path contains at most one digit having odd freq
                    #or jsut check path must be poer of two
                    if path & (path -1) == 0:
                        count +=1
                else:
                    stack.append((node.left,path))
                    stack.append((node.right,path))
                    
        return count
        
#iterative all root to leaf paths
        paths = []
        stack = [(root,[])]
        while stack:
            node,path = stack.pop()
            if node:
                if not node.left and not node.right:
                    paths.append(path+[node.val])
                else:
                    stack.append((node.left,path+[node.val]))
                    stack.append((node.right,path+[node.val]))
        print paths
###########################################################
# Longest Substring with At Most K Distinct Characters
############################################################
#TLE 139/141
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        '''
        well the brute force would be to examin all possible lengths from range(2,len(s))
        and check that the lenght of the set made is less than = k
        '''
        #special cases
        n = len(s)
        if n * k == 0:
            return 0
        max_length = float('-inf')
        for size in range(1,len(s)+1):
            for i in range(0,len(s)-size+1):
                substring =  s[i:i+size]
                if len(set(substring)) <= k:
                    max_length = max(max_length,len(substring))
        
        return max_length

class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        '''
        similar to longest substring with 2
        sliding windwo with hash
        two poitners
        hahse is char:index
        we update our hash whenver we exceed k+1 items and move the leftmost character
        we update in our hash the indices
        '''
        #edge cases
        N = len(s)
        if N*k == 0:
            return 0
        
        l,r = 0,0
        mapp = defaultdict()
        #in mapp most recent entry is right most char
        max_len = 1
        while r < N:
            mapp[s[r]] = r
            r += 1
            
            if len(mapp) >= k + 1:
                #delete the last recently seen
                last_seen_idx = min(mapp.values())
                del mapp[s[last_seen_idx]]
                #move l past it
                l = last_seen_idx + 1
            
            #always update
            max_len = max(max_len, r-l)
        
        return max_len

#using ordered dict
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        '''
        using a hasmap does not give us access to the first or last hasehd key value pair
        however there is a structure called an orderdict (recall finding the min in a hash of size k is O(K))
        using orderded dict:
            return 0 if the string is empty of k is equal to zero
            set boht pointer to beinggin l,r = 0,0 and maxlength = 0
            while right is less than N
            if s[r] is already in ordered dict, delete it to ensure the first key in hash is left most
            add the current s[r] and move right
            if ordered duct contains k+1 remove letmost O(1) instead O(N)
        '''
        N = len(s)
        if k == 0 or N == 0:
            return 0
        l,r = 0,0
        mapp = OrderedDict()
        max_len = 1
        while r < N:
            char = s[r]
            #if char is already in mapp, delted to ensure that right most element is the last added in
            #and that left most if first added in
            if char in mapp:
                del mapp[char]
            mapp[char] = r
            r += 1
            
            #when we go over k
            if len(mapp) == k + 1:
                #delete left most, this is 0(1)
                _,del_idx = mapp.popitem(last= False)
                #move left
                l = del_idx + 1
            #alwyas update
            max_len = max(max_len, r-l)
        
        return max_len

###############
#Game of Life
##############
#cheeky but works
class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        '''
        i would need to make a copy of the boar firs, then it becomese asier
        you just check each i,j on the conditions, and mutate the board
        but watch for boundaires
        i could make a helper function to act on the board
        make a results board and mutate that N squared must be ok because the dimensions are not more than 25
        
        '''
        results = copy.deepcopy(board) #mutate this but apply rules to board
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
        rows = len(board)
        cols = len(board[0])
        
        #helper
        def helper(cell):
            i,j = cell
            #if cell is 1
            if board[i][j] == 1:
                ones = 0
                zeros = 0
                for dirr in directions:
                    new_i,new_j = i + dirr[0], j + dirr[1]
                    #bouandary check
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        if board[new_i][new_j] == 1:
                            ones += 1
                        else:
                            zeros += 1
                #rules
                if ones < 2:
                    return 0
                elif ones == 2 or ones == 3:
                    return 1
                else:
                    return 0
            #now if cell is 0
            if board[i][j] == 0:
                ones = 0
                zeros = 0
                for dirr in directions:
                    new_i,new_j = i + dirr[0], j + dirr[1]
                    #bouandary check
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        if board[new_i][new_j] == 1:
                            ones += 1
                        else:
                            zeros += 1
                if ones == 3:
                    return 1
                else:
                    return 0
        for i in range(rows):
            for j in range(cols):
                res = helper((i,j))
                results[i][j] = res
        
        #one more passt to reassign
        for i in range(rows):
            for j in range(cols):
                board[i][j] = results[i][j]


class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        '''
        O(1) space, in place, and elif block consolidation
        realy just two if states for rule 1, rule 2, rule 3
        if element == 0 and live neighbors < 2 and live_neighbors > 3, its a zero else 1
        we can solve the problem in places by using a dummy cell value to signify previous state of the cell along with the new changed value
        example, if the cell was originally 1 but become zero after the rile, we can mark it -1; negative means dead, but was oringinally 1
        also if the value of the cell was 0, but became 1, then we make it a two, + indicats the change
        algo:
            1. iterate acoress the board one by one
            2. update rules now:
                rule 1: any live cell with < 2 neighors dies, the cell becomes - 1
                rule 2: any live cell with == 2 or == 3 live, live on so no change
                rule 3: anye live cell iwth >3 dies, to its - 1
                rule 4: andy dead cell with == 3 lives becomes 2
            3. apply new rules
            4. one more pass to convert -1 and 2 to live(1) and dead
        '''
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
        rows = len(board)
        cols = len(board[0])
        for i in range(rows):
            for j in range(cols):
                #count live
                live = 0
                for dirr in directions:
                    new_i,new_j = i + dirr[0], j + dirr[1]
                    #bouandary check
                    if 0 <= new_i < rows and 0 <= new_j < cols:
                        if abs(board[new_i][new_j]) == 1: #remember it used to be alive
                            live += 1
                #new rule update
                #live cell and <2 >3 neighbors
                if board[i][j] == 1 and (live <2 or live >3):
                    board[i][j] = -1 #alive now dead
                if board[i][j] == 0 and live == 3:
                    board[i][j] = 2 #dead now alive
        
        #second pass to decode board
        for i in range(rows):
            for j in range(cols):
                if board[i][j] > 0:
                    board[i][j] = 1
                else:
                    board[i][j] = 0

class Solution(object):
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        '''
        on the infinite board follow up
        we would have sparse matrix, so it might be better to store the locations if 1s
        #revisits this
        https://leetcode.com/problems/game-of-life/discuss/73217/Infinite-board-solution/201780
        '''
        #store live
        lives = set()
        for i,row in enumerate(board):
            for j,live in enumerate(row):
                if live:
                    lives.add((i,j))
        #init counter
        #just counting up all the neighbord cooridantes of live cell cooridantes
        counts = []
        for i,j in lives:
            for I in range(i-1,i+1):
                for J in range(j-1,j+1):
                    if I != i or J != J:
                        counts.append((I,J))
        counts = Counter(counts)
        
        #now store locations if counts == 2 or counts == 2 and also in lives
        results = set()
        for ij in counts:
            if counts[ij] == 3 or counts[ij] == 2 and ij in lives:
                results.add(ij)
        
        for i, row in enumerate(board):
            for j in range(len(row)):
                row[j] = int((i, j) in lives)

##################################
#Largest Rectangle in Histogram   
###################################
#O(N^3)
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        '''
        go through all approaches
        brute force
        we can can consider all possible sequence of rectangles and just take the max area
        '''
        max_area = 0
        N = len(heights)
        for i in range(N):
            for j in range(N):
                #first find the minimu
                min_height = float('inf')
                for k in range(i,j+1):
                    min_height = min(min_height,heights[k])
                    #print min_height
                #update afer finding min
                max_area = max(max_area,min_height*(j-i+1))
        return max_area

#O(N^2)
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        '''
        better brute force
        we can do one slight modification in the previous approach to optimize it some height
        insgtead of taking every possible pair of itnervals when findint the bar of min height lying between them every time, 
        we can fint eh bar of the minheight for current pair by using the minimum height of the previous bar
        min_height = min(min_height, heights(j))
        so keep track if the minimum found, and update i
        '''
        N = len(heights)
        max_area = 0
        for i in range(N):
            min_height = float('inf')
            for j in range(i,N):
                min_height = min(min_height,heights[j])
                #udate max
                max_area = max(max_area, min_height*(j-i+1))
        return max_area

#first two are standard approaches   
#O(NlogN)

class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        '''
        we can perform a divide and conquer strategy 
        we note three things about the rectangle with max area:
            1. it will be the maximum of the widest possible rectangle with height euqal to the height of the shortest bar
            2. the largest rectangle will be confined to the left of the shortest bar (sub problem)
            3. the largest rectanble will be confined the right of the shortest bar (sub problem again)
        and it is the max of all three of these conditions
        '''
        memo = {}
        def calc_area(heights,start,end):
            if (start,end) in memo:
                return memo[start_end]
            if start > end:
                return 0
            min_idx = start
            for i in range(start,end+1):
                if heights[min_idx] > heights[i]:
                    min_idx = i
            
            shortest_bar = heights[min_idx]*(end-start+1)
            largest_left = calc_area(heights,start,min_idx-1)
            largest_right = calc_area(heights, min_idx+1,end)
            result = max(shortest_bar, largest_left,largest_right)
            memo[start,end] = result
            return result
        
        return calc_area(heights,0,len(heights)-1)

#O(N) using a stack 
#watch video
class Solution(object):
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        #using stack
        stack = [-1]
        N = len(heights)
        max_area = 0
        for i in range(N):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                current_height = heights[stack.pop()]
                current_width = i - stack[-1] - 1
                max_area = max(max_area,current_height*current_width)
            stack.append(i)
            
        #remaining elements
        while stack[-1] != -1:
            current_height = heights[stack.pop()]
            current_width = N - stack[-1] - 1
            max_area = max(max_area, current_height*current_width)
            
        return max_area       
