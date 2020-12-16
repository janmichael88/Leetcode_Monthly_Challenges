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
