#####################################################
# Convert Binary Number in a Linked List to Integer
###################################################

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def getDecimalValue(self, head):
        """
        :type head: ListNode
        :rtype: int
        """
        '''
        O(1) space, traverse once counting the number of nodes
        traverse the linked list and conver the number
        '''
        result = 0
        N = 0
        temp = head
        while head:
            N += 1
            head = head.next
            
        N -= 1
        while temp:
            result += (temp.val)*2**N
            temp = temp.next
            N -= 1
        return result
         
#################################
#  Maximum Depth of Binary Tree
##################################
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
        this is just level order bfs
        bfs all the way down and return the last depth level
        '''
        if not root:
            return 0
        
        q = deque([(root,1)])
        while q:
            current,level = q.popleft()
            if current.left:
                q.append((current.left, level+1))
            if current.right:
                q.append((current.right, level+1))
        return level


#######################
# Insertion Sort List
#######################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        i can't go back in a linked list
        i need to allocate another one...dummy head
        always give reference to the start of the dummy
        dummy->1->2->3->4
        
        4->2->1->3
        
        in a singly linked list, each node has only one pointer that points to hte enxt nodt
        we need to use two pointes prev_node and next_node, which are always referenced in the dummy
        
        
        '''
        dummy = ListNode()
        
        current = head
        
        while current:
            prev_node, next_node = dummy,dummy.next
            while next_node:
                if next_node.val > current.val:
                    break
                prev_node, next_node = next_node, next_node.next
            #give reference to the next node after current
            current_next = current.next
            #re link between inputs and dummy
            current.next = next_node
            prev_node.next = current
            
            current = current_next
        return dummy.next


###########################
# Meeting Rooms
##########################
class Solution(object):
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: bool
        """
        '''
        if there is an intersection between the end of any meeting and the start of another
        then that person cannot attend all of them and return false
        sort on start, and check end
        '''
        if not intervals:
            return True
        
        #sort
        intervals.sort()
        
        for i in range(1,len(intervals)):
            if intervals[i][0] < intervals[i-1][1]:
                return False
        return True

#############################
# Search Insert Position
############################
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        '''
        this is just binary search
        in the special case where i can't find a match, let the loop finish and return the left pointer
        '''
        N = len(nums)
        lo,hi = 0, N-1
        
        while lo <= hi:
            mid = lo + (hi-lo)//2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                hi = mid -1
            else:
                lo = mid + 1
        return lo


##########################
# Consecutive Characters
############################
class Solution(object):
    def maxPower(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        this is a baby dp problem
        allocate extra power array
        '''
        power_array = [0]*len(s)
        power_array[0] = 1
        
        for i in range(1,len(s)):
            if s[i-1] == s[i]:
                power_array[i] = power_array[i-1] + 1
            else:
                power_array[i] = 1
        
        return max(power_array)


###########################
# Lenght of Last Word
############################
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        go backwards in the array trimming white spaces
        onces we have reached the last word count of the length
        '''
        i = len(s) - 1
        while i >= 0 and s[i] == ' ':
            i -= 1
            
        size = 0
        while i >= 0 and s[i] != ' ':
            size += 1
            i -= 1
        
        return size


class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        go backwards in the array trimming white spaces
        onces we have reached the last word count of the length
        '''
        i = len(s)
        size = 0
        while i > 0:
            i -= 1
            if s[i] != ' ':
                size += 1
            elif size > 0:
                return size
        return size


###################################
# Maximum Depth of N-nary Tree
##################################
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: Node
        :rtype: int
        """
        '''
        same thing as max depth of binary tree, use q and keep decseding
        '''
        if not root:
            return 0
        q = deque([(root,1)])
        while q:
            current, level = q.popleft()
            if current:
                for node in current.children:
                    q.append((node,level+1))
        return level  

#########################
# Minimum Height Trees 11/4/2020
########################
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        '''
        https://leetcode.com/problems/minimum-height-trees/discuss/76055/Share-some-thoughts
        enumerate all possible trees using each node as the root, that would take a long time...
        topological sort? similar to the course schedule problem
        make the adjacency list
        intuition:
            the distances between thwo nodes is the number of edghes that connects the two nodes
            and there exists one and only one path between two nodes from the constraints
            we take the height as the max distance between the root and its leaves (all of them the same)
            we rephrase th eproblem as finidng out the nopdes that are overall close to all other nodes
        
        if we view the graphs an area of a circle, and the leaves lie along the circumference, then what we are looking for are actuall the centroids of the circle, with min radii
        NOTE: for a tree like graph, the number of centroids is no more than 2
            if number of nodes is even, there can be no more than two
            if odd nodes, there can be one and only one
            proof by contradiction for the odd number case...i.e there can be cycle making centroids be n, but the tree is unidirected
        
        algo:
            the problem now becomes looking for centroids, which can be no more than two
            trim out the leaves layer by layer, leaving the final centroids
        
        implementation:
            get the adjacenct list
            q up holding the leaves
            do work while there is at least 2 nodes in the q
            at each iter, pop the current leaves nodes from the q. while removing the nodes, also removed the edges that are linked to those nodes
            (as a consequence some of the non-leaf nodes could becomes leaves, but these are trimmed out in the next ieration)
            
        '''
        #base cases
        if n <= 2:
            return [i for i in range(n)]
        
        #build adj list
        adj = collections.defaultdict(list)
        for start,end in edges:
            adj[start].append(end)
            adj[end].append(start)
        
        #we need the first layers of leaves to q up
        leaves = []
        for i in range(n):
            if len(adj[i]) == 1:
                leaves.append(i)
                
        #trim until we get to the middle, 
        nodes_left = n
        while nodes_left > 2:
            nodes_left -= len(leaves)
            #new leaves finder
            new_leaves = []
            while leaves:
                leaf = leaves.pop()
                #check its neighbors and q up
                for nei in adj[leaf]:
                    #remove its neighbords
                    adj[nei].remove(leaf)
                    #add back in to q
                    if len(adj[nei]) == 1:
                        new_leaves.append(nei)
            leaves = new_leaves
            
        
        return leaves

#brute force
class Solution(object):
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        '''
        enumerate all possible trees using each node as the root, that would take a long time...
        brute force algo:
            generate adj list once
            for each node do bfs finding the max height
            dump the neights into a list
            fin the min height
            return those nodes correpsoding to the min height
        '''
        #base cases
        if n <= 2:
            return [i for i in range(n)]
        
        #build adj list
        adj = collections.defaultdict(list)
        for start,end in edges:
            adj[start].append(end)
            adj[end].append(start)
        
        
        heights = [0]*n
        
        #bfs finding the max height for each tree with the rooted node
        for i in range(n):
            #q up and keep track of already visited nodes
            visited = set()
            q = deque([(node,1) for node in adj[i]]) #tuple is list with list[0] being nodes list[1] level
            visited.add(i)
            while q:
                node,level = q.popleft()
                visited.add(node)
                for neigh in adj[node]:
                    if neigh not in visited:
                        q.append((neigh,level+1))

            heights[i] = level
        
        #find the min
        mini = min(heights)
        
        #return indices where it equals min
        results = []
        for i in range(n):
            if heights[i] == mini:
                results.append(i)
        
        return results


###################################
#Remove duplicates from Sorted List
###################################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return None
        cur = head
        while cur and cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head
        

###############################
# Minimum Cost to Move Chips to the Same Position
##############################
# class Solution(object):
    def minCostToMoveChips(self, position):
        """
        :type position: List[int]
        :rtype: int
        """
        '''
        moving a chip 2 costs 0, moving a chip 1 costs 1
        to minimize cost you'd want stacks of coins nearest each other
        i can bring all even coins to the 0 index for free
        i can bring all the odd couns to the 1 index for free
        move the smaller stack to the largest one
        '''
        evens = 0
        odds = 0
        for p in position:
            if p % 2 == 0:
                evens += 1
            else:
                odds += 1
        return min(evens,odds)

##########################
# 69. Sqrt(x)
###########################
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        '''
        this is just newtons method of successive approximations
        recall:
        x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}
        
        
        '''
        if x == 0:
            return 0
        guess = x
        #as we get close to our guess
        for i in range(20):
            guess = (guess + x/guess)*0.5
        
        return int(guess)

#using the log function
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        '''
        recall sqrt(x) = e^.5 log x
        
        '''
        if x < 2:
        	return x

        left = int(e**(0.5)*log(x))
        right = left + 1
        return left if right*right > x else right #because we truncated

class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        #newtons method
        '''
        this is just newtons method of successive approximations
        recall:
        x_{n+1} = x_{n} - \frac{f(x_{n})}{f'(x_{n})}
        
        
        
        if x == 0:
            return 0
        guess = x
        #as we get close to our guess
        for i in range(20):
            guess = (guess + x/guess)*0.5
        
        return int(guess)
        '''
        #linear generation
        '''
        generate squares until we can't anymore
        
        if x < 2:
            return x
        
        guess = 1
        while guess*guess <= x:
            guess += 1
        return guess - 1
        '''
        #binary search
        '''
        i can improve the linear scan using binary search
        since 0<sqrt(x) < x/2
        '''
        if x < 2:
            return x
        
        lo, hi = 2, x // 2
        
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if mid*mid > x:
                hi = mid - 1
            elif mid*mid < x:
                lo = mid + 1
            else:
                return mid
        return hi

################################################
# Find the Smallest Divisor Given a Threshold
################################################
#Time Limit Exceed
class Solution(object):
    def smallestDivisor(self, nums, threshold):
        """
        :type nums: List[int]
        :type threshold: int
        :rtype: int
        """
        '''
        brute force try all possible divsors and stop when sum exceeds threhold
        return the min
        '''
        def helper(a,b):
            return int((a + (b-1)) / b)
        
        smallest = float('inf')
        candidate = max(nums)
        summ = sum([helper(foo,candidate) for foo in nums])
        
        while candidate >= 1:
            summ = sum([helper(foo,candidate) for foo in nums])
            if summ <= threshold:
                smallest = min(smallest, candidate)
            candidate -= 1
        
        return smallest
    

class Solution(object):
    def smallestDivisor(self, nums, threshold):
        """
        :type nums: List[int]
        :type threshold: int
        :rtype: int
        """
        '''
        binary search, start with 1 and max of nums
        '''
        def helper(a,b):
            return int((a + (b-1)) / b)
        
        lo = 1
        hi = max(nums)
        
        
        while lo <= hi:
            mid = lo + ((hi-lo) // 2)
            #get the sum using our mid
            summ = sum([helper(foo,mid) for foo in nums])
            
            if summ > threshold:
                #use a bigger divor
                lo = mid + 1
            else:
                hi = mid - 1
                
        #we want the smallest divosor so we take the left
        #note in this case we don't have a target to search, we would keep binary searching until we found it
        #this is important, at the end of the loop, lo > hi
        #and so the sum(hi) > threshold
        #and sum(lo) <= threshold, but one less than hi
        return hi + 1


########################
#Add Two Numbers II
########################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        just dump the numbers in two lists, converts string to in, add, recrate the list
        7243
         564
        7807
        '''
        num1 = ""
        num2 = ""
        
        h1,h2 = l1,l2
        while h1:
            num1 += str(h1.val)
            h1 = h1.next
        
        while h2:
            num2 += str(h2.val)
            h2 = h2.next
        
        new_num = int(num1) + int(num2)
        new_num = str(new_num)
        #recreate
        dummy = ListNode(val = int(new_num[0]))
        current = dummy
        for i in range(1,len(new_num)):
            newNode = ListNode(val = int(new_num[i]))
            current.next = newNode
            current = current.next
        return dummy

#another way 
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        this combines a lot of problems,
        the naive way, which i already did would be to dump the vals in a list
        add regularly, and recreate the one
        another way:
            reverse both lists
            implement addtion with carry for each list
            dump into a now linkedlist
        '''
        def reverse(node):
            prev = None
            cur = node
            while cur:
                cur_next = cur.next
                cur.next = prev
                prev = cur
                cur = cur_next
            return prev
        
        l1 = reverse(l1)
        l2 = reverse(l2)
        
        #implement addion with carry, we need to make sure for each next call 
        #in l1 and l2 we add theem to the front
        results = None
        carry = 0
        
        while l1 or l2:
            if l1:
                x1 = l1.val
            else:
                x1 = 0
            if l2:
                x2 = l2.val
            else:
                x2 = 0
            
            #add and carry
            val = (x1 + x2 + carry) % 10
            carry  = (x1 + x2 + carry) // 10
            
            #add to the front
            newNode = ListNode(val)
            newNode.next  = results
            results = newNode
            
            #move pointers
            if l1:
                l1 = l1.next
            else:
                l1 = None
            if l2:
                l2 = l2.next
            else:
                l2 = None
                
        #final carry
        if carry:
            newNode = ListNode(carry)
            newNode.next = results
            results = newNode
        
        return newNode


########################
# Add Strings
########################
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        '''
        this is a review to on how to carry with digits
        init  res structure
        start from carry 0
        set pointers to the end of boths trings
        travese both at the same time and stop when strings are used up
            set x1 to be euqla  to a digit froom string num1 at  p1
            same thing with x2
            copmute the value  with carry, don't forget  to mod it
            update the carrry
            reverse the string and return  it
        '''
        results = []
        carry  = 0
        p1 = len(num1) -1
        p2 = len(num2) -1
        
        #work while eithr of the pointers  exists
        while p1 >= 0 or p2 >=0:
            #either p1 or p2 could go below zero at any poit
            if p1 >= 0:
                x1 = ord(num1[p1]) - ord('0')
            else:
                x1 = 0
            if p2 >= 0:
                x2 = ord(num2[p2]) - ord('0')
            else:
                x2 = 0
            #compute value
            val = (x1 + x2 + carry) % 10
            #upcate carry
            carry  = (x1 + x2 + carry) // 10
            results.append(val)
            p1 -= 1
            p2 -= 1
        
        #if there is one last carry
        if carry:
            results.append(carry)
            
        return ''.join(str(x) for x in results[::-1])


######################## 
#Reverse Linked List
#######################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        since i cannt look back in a single linked list
        i need to store the previous element before hand
        also another pointer is needed to store the next node before changing the reference
        '''
        prev = None
        cur = head
        while cur:
            cur_next = cur.next
            cur.next = prev
            prev = cur
            cur = cur_next

#recursive solution
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        recursive solution
        recurse all the way down to single node, this will fire and reverse
        the reversed list is the input to the previous caller
        
        '''
        def dfs(node):
            if not node or not node.next:
                return node
            temp = dfs(node.next)
            node.next.next = node
            node.next = None
            return temp
        return dfs(head)
        

####################
#Binary Tree Tilt
###################
#so close...
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        make the tree first and then sum up all the nodes, any of the three traversals will work
        '''
        def dfs(node):
            if not node:
                return
            if not node.left and not node.right:
                return 0
            if node.left:
                sum_left = 0
                sum_left += node.val
                dfs(node.left)
                dfs(node.right)
            if node.right:
                sum_right = 0
                sum_right += node.val
                dfs(node.left)
                dfs(node.right)
            
            node.left = ListNode(val=sum_left)
            node.right = ListNode(val=sum_right)
            node = abs(node.left.val - node.right.val)
            
        #invoke in place
        dfs(root)


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        to get the tilt at each node, we need the sum of its left and right subtrees
        tilt(node) = sum(node.left) + sum(node.right)
        and 
        sum(node) = node.val + sum(node.left) + sum(node.right)
        '''
        self.answer = 0
        
        def dfs(node):
            if not node:
                return 0
            
            left_sum = dfs(node.left)
            right_sum = dfs(node.right)
            tilt = abs(left_sum-right_sum)
            self.answer += tilt
            
            return node.val +left_sum+right_sum
        
        dfs(root)
        return self.answer

#####################
# Symmetric Tree
#####################
#so clsoe 119/165

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        a subtree is symmetric if at a node
        the node.left == node.right and node.right == node.left
        '''
        if not root:
            return True
        def dfs(node):
            if not node:
                return
            if node.left == node.right and node.right == node.left:
                return True
            if dfs(node.left) and dfs(node.right):
                return True
            return False

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        a subtree is symmetric if at a node
        the node.left == node.right and node.right == node.left
        use two poiners and compare subtrees
        '''
        if not root:
            return True
        def dfs(node1,node2):
            if not node1 and not node2:
                return True
            if not node1 or not node2:
                return False
            return (node1.val == node2.val) and dfs(node1.left,node2.right) and dfs(node1.right,node2.left)
        
        return dfs(root,root)
        

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        '''
        iteratively?
        use q, add in both roots and check if left == right, right == left
        '''
        q = deque([])
        q.append(root)
        q.append(root)
        while q:
            node1 = q.popleft()
            node2 = q.popleft()
            if not node1 and not node2:
                continue
            if not node1 or not node2:
                return False
            if node1.val != node2.val:
                return False
            #add the children
            q.append(node1.left)
            q.append(node2.right)
            q.append(node1.right)
            q.append(node2.left)
        
        return True


#################################################
#   Maximum Difference Between Node and Ancestor
#################################################
# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxAncestorDiff(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        well the hint gave it away...LOL
        for each node find the min and max of its decnedants
        the min would be all the way to the left, and the max would be all the way to the right
        update a self variable outside the recursive call
        '''
        self.max = float('-inf')
        
        def find_min_max(node):
            if not node:
                return
            L = node.left
            R = node.right
            while L:
                if L:
                    L = L.left
            mini = L.val
            while R.right:
                R = R.right
            maxi = R.val
            return abs(mini-maxi)
        
        def dfs(node):
            if not node:
                return
            self.max = max(self.max,find_min_max(node))
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxAncestorDiff(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        brute force recursion
        we can compare every desendant of a node, and just update a max
        lets define our helper function, which taks in a node and two inters which are the max and min
        the function will update outside its scope
        we are really just going down all paths ane keeping track of a max and min along a path
        at the end we update the largest difference
        '''
        if not root:
            return 0
        
        self.result = float('-inf')
        def dfs(node, mini,maxi):
            if not node:
                return
            #update
            self.result = max(self.result,abs(node.val - mini), abs(node.val- maxi))
            mini = min(node.val,mini)
            maxi = max(node.val,maxi)
            dfs(node.left,mini,maxi)
            dfs(node.right,mini,maxi)
            
        dfs(root,root.val,root.val)
        return self.result

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxAncestorDiff(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        brute force recursion
        we can compare every desendant of a node, and just update a max
        lets define our helper function, which taks in a node and two inters which are the max and min
        the function will update outside its scope
        '''
        if not root:
            return 0
        
        def dfs(node, mini,maxi):
            if not node:
                return maxi - mini
            #update
            mini = min(node.val,mini)
            maxi = max(node.val,maxi)
            left = dfs(node.left,mini,maxi)
            right = dfs(node.right,mini,maxi)
            return max(left,right)
            
        return dfs(root,root.val,root.val)
            
            
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def maxAncestorDiff(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        iterative dfs
        https://www.youtube.com/watch?v=f37BCBHGFGA&ab_channel=RenZhang
        '''
        result = 0
        q = [(root,root.val,root.val)]
        while q:
            node, cur_min,cur_max = q.pop()
            cur_min = min(cur_min,node.val)
            cur_max = max(cur_max,node.val)
            for child in [node.left,node.right]:
                if not child:
                    continue
                q.append((child,cur_min,cur_max))
            #at leaf
            if not node.left or node.right:
                result = max(result,cur_max-cur_min)
                
        return result

################################
# 172. Factorial Trailing Zeroes
################################
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        the naive way is to just compute and count up the leading zeros
        '''
        product = 1
        while n >= 1:
            product *= n
            n -= 1
        
        zeros = 0
        while product % 10 == 0:
            zeros += 1
            product //= 10
        return zeros
        

class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        anytime i multiple by 10, we increase our zero count by 1
        so the quesiton becomes, how many times do we multiply by 10?
        well we can count up the pairs of 2s and 5s 
        and take the min of either 2 or 5
        in pseudo code:
        twos = 0
        for i in range(1,n):
            if i % 2 == 0:
                twos += 1
        fives = 0
        for i in range(1,n):
            if i % 5 == 0:
                fives += 1
        return min(twos,fives)
        but what is we have factors that contain multiples 5s and twos, well we just keep dividing until we can't anymore
        twos = 0
        for i in range(1,n):
            while i % 2 == 0:
                twos += 1
                i //= 2
        fives = 0
        for i in range(1,n):
            while i % 5 == 0:
                fives += 1
                i //= 2
        return min(twos,fives)
        i can remove first pass for twos because it will always be greater than 5
        and we can finally pass in incrementes of 5
        fives = 0
        for i in range(5,n+1,5):
            while i % 5 == 0:
                fives += 1
                i //= 5
        return fives
        fives = 0
        for i in range(5,n+1,5):
            #starting power
            power = 5
            while i % power == 0:
                fives += 1
                power *= 5
        return fives
        
        how about logarithmic time
        just keep diving n by 5 and incremding that count of zeros by how many times 5 went into the current n
        since its integer division, n will eventually get to zero
        at that point we just get the number after increamting
        
        '''
        zeros = 0
        while n > 0:
            n //=5
            zeros += n
        return zeros


######################
# Flippig an Image
######################
class Solution(object):
    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        well just reverse the rows and flips te bits
        '''
        rows = len(A)
        cols = len(A[0])
        #reverse
        for i in range(0,rows):
            A[i] = A[i][::-1]
        #invert
        for i in range(0,rows):
            for j in range(0,cols):
                if A[i][j] == 1:
                    A[i][j] = 0
                else:
                    A[i][j] = 1
        return A


#####################
# Two Sum Less than K
#####################
class Solution(object):
    def twoSumLessThanK(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        '''
        [34,23,1,24,75,33,54,8]
        after sotring
        [1, 8, 23, 24, 33, 34, 54, 75]
        1
        '''
        result = -1
        A.sort()
        
        lo, hi = 0,len(A)-1
        while lo < hi:
            if (A[lo] + A[hi]) < K:
                result = max(result, A[lo]+A[hi])
                lo += 1
            else:
                hi -= 1
        
        return result


class Solution(object):
    def twoSumLessThanK(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        '''
        [34,23,1,24,75,33,54,8]
        after sotring
        [1, 8, 23, 24, 33, 34, 54, 75]
        1
        '''
        result = -1
        A.sort()
        
        for i in range(0,len(A)):
            lo = i + 1
            hi = len(A) - 1
            while lo <= hi:
                mid = lo + (hi - lo) //2
                if A[i] + A[mid] >= K:
                    hi = mid -1
                else:
                    lo = mid + 1
                    result = max(result, A[i]+A[mid])
        return result
                

#######################
# Valid Square
#######################
#105 of 244
class Solution(object):
    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """
        '''
        its a sqaure of a pair of points are equal at x and y
        and the distance between all points are the same
        finding matching x, get dis apart
        check of dist apart is same dist for matching y
        '''
        matching_x = []
        matching_y = []
        points = deque([p1, p2, p3, p4])
        
        count = 0
        while count < 4:
            count += 1
            cand = points.popleft()
            for p in points:
                if cand[0] == p[0]:
                    matching_x.append(cand)
                    matching_x.append(p)
                if cand[1] == p[1]:
                    matching_y.append(cand)
                    matching_y.append(p)
        if not matching_x or not matching_y:
            return False


        return matching_x[0][1]-matching_x[1][1] == matching_y[0][0]-matching_y[1][0]

#after sorting the points
class Solution(object):
    def validSquare(self, p1, p2, p3, p4):
        """
        :type p1: List[int]
        :type p2: List[int]
        :type p3: List[int]
        :type p4: List[int]
        :rtype: bool
        """
        '''
        we can sort the points by the x axis,
        once we do this, we know the points are in clockwise order from the bottom left of the square
        p1  p3
        
        p0  p2
        we can then check:
        p1p2, p2p4, p4p3,p3p1 for equality
        and p2p3 == p1p4
        '''
        def dist(a,b):
            return ((b[1]-a[1])**2 + (b[0]-a[0])**2)
        
        points = [p1,p2,p3,p4]
        points.sort()
        
        return dist(points[0],points[1]) != 0 and dist(points[0],points[1]) == dist(points[1],points[3]) and dist(points[1],points[3]) == dist(points[3],points[2]) and dist(points[3],points[2]) == dist(points[2],points[0]) and dist(points[0],points[3]) == dist(points[1],points[2])


########################
# Permutations II
#######################
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        '''
        standard backtracking approach
        recurse, and add builds to container outside of function call
        they want unique ones so use hash set, remember to make into a tuple
        '''
        results = set()
        N = len(nums)
        def rec_build(nums,build):
            #base case
            if len(build) == N:
                if tuple(build) not in results:
                    results.add(tuple(build))
                else:
                    return
            for i in range(len(nums)):
                rec_build(nums[:i]+nums[i+1:],build+[nums[i]])
                
        rec_build(nums,[])
        return results

class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        '''
        https://leetcode.com/problems/permutations-ii/solution/
        the same thing goes with the previous permulation problem
        recall backtracking
            general algo for finding all (or some) solutions to some problems with constraints
            constaint satisfations problems
            it incremntally builds candidates and abandons candidates when the path does not meet the constraint
        lets start with [1,1,2]
        only two choises [1,2]
        each has two choise[1,2] or [1,1]
        a key insight to avoid generating any redundant permutation is that at each step rather than viewing each number as a candidate, we consider each unique number as the true candidate
        original problem [1,1,2] three potential candidates
        but really only [1,2]
        '''
        results = []
        
        def rec_build(build,counts):
            #when your counts is empty we've made a valid permutation
            if not counts:
                results.append(build)
                return
            #recurse
            
            for num in counts.keys():
                #use up on occurence of the candidate
                if counts[num] == 1:
                    del counts[num]
                else:
                    counts[num] -= 1
                rec_build(build + [num],counts)
                #backtrack
                counts[num] += 1
        
        rec_build([],Counter(nums))
        return results

#########################################
#167. Two Sum II - Input array is sorted
#########################################
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        lo = 0
        hi = len(numbers) -  1
        while lo < hi:
            if numbers[lo] + numbers[hi] > target:
                hi -= 1
            elif numbers[lo] + numbers[hi] < target:
                lo += 1
            elif numbers[lo] + numbers[hi] == target:
                return lo+1,hi+1

##############################################
# Populating Next Right Pointers in Each Node
##############################################
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
        recursive call, first input is the node we are at, second input is the one we are connecting to
        if you notice every nodes right pointer is pointer to the left of a parent node!
        https://leetcode.com/problems/populating-next-right-pointers-in-each-node/discuss/690375/Python-2-Solutions-with-Explanation-and-comments
        '''
        def dfs(node):
            if not node:
                return
            #if the node can descen right and has a next
            if node.right and node.next:
                node.right.next = node.next.left
            #only a left
            if node.left:
                node.left.next = node.right
            dfs(node.right)
            dfs(node.left)
        dfs(root)
        return root

#level order bfs
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
        instead of doing this recursively, use level order BFS
        add in the current node, then check to see if we can assign its next pointer to the first element in the que
        
        '''
        if not root:
            return root
        q = deque([root])
        while q:
            size = len(q)
            for i in range(size):
                current = q.popleft()
                #check that we are only making the current's next immedialty next to the head of the q
                #so not at the end
                if i < size -1:
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
        using previously established pointers
        there are two kinds of connecitons we need to make
        between nodes of teh same parent
        parent.left.next = parent.right
        and from two different partents, which is not as trivial
        we only descend to N+1 after we have finisehd making connections at N
        since we have access to all hte nodes at a particular level via the next pointers, we can use these next pointers to establish the connections for the next levle or the level containing children
        '''
        #edge case
        if not root:
            return root
        
        #give refrence to the head
        leftmost = root
        
        #stay in a level until we are down
        while leftmost.left:
            #start the 'linked list'
            head = leftmost
            while head:
                #first connection
                head.left.next = head.right
                #second connection
                if head.next:
                    head.right.next = head.next.left
                #advance into the connection
                head = head.next
            #reassign start of level pointers and descend into the next level
            leftmost = leftmost.left
        return root

#####################
#Poor Pigs
#####################
class Solution(object):
    def poorPigs(self, buckets, minutesToDie, minutesToTest):
        """
        :type buckets: int
        :type minutesToDie: int
        :type minutesToTest: int
        :rtype: int
        """
        '''
        for only one pig
        say for example we take the ratio:
        minTest/minDie  = 0
        this means we have no time to test at constat minDie
        and so the states are only alive for the pig
        now take minTest/minDie = 1
        for this pigs its alive or dead
        now take minTst / minDie = 2
        now the pig can have three states :
        alive, dead after first, dead after second
        so the number of available seats for the pig is
        states = minTest / minDie + 1
        now one pig could test at most two buckets (alive if not p, dead if p)
        thats two states
        and so two pigs have 4 states
        2^x, where x is the number of buckets
        so then the problem gets reduces to find x such that states^x >= bukets
        x = log_states(buckets)
        or x >= log(buckets) / log(states)
        '''
        states = minutesToTest // minutesToDie + 1
        return int(math.ceil(math.log(buckets) / math.log(states)))


class Solution(object):
    def poorPigs(self, buckets, minutesToDie, minutesToTest):
        """
        :type buckets: int
        :type minutesToDie: int
        :type minutesToTest: int
        :rtype: int
        """
        '''
        https://www.youtube.com/watch?v=_JcO3fqoG2M&ab_channel=AnishMalla
        we notice that if we have minutesTest/ minDie = 60/15 = 4 +1 ***for the +1 case
        it takes one pig (across a row)
        two pigs across row and col
        3 pigs
        25 < 3 pigs < 125
        we keep adding pigs such that ((minTest / minDIe) + 1)**pigs < buckets
        '''
        pigs = 0
        while (int(minutesToTest/minutesToDie)+1)**pigs < buckets:
            pigs += 1
        return pigs


###############
#Range Sum BST
##############
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rangeSumBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: int
        """
        '''
        dfs along the tree to touch each node
        at node check lo hi constraints and add to sum outside of recursive call
        return the sum
        '''
        self.sum = 0
        
        def dfs(node):
            if not node:
                return
            if low <= node.val <= high:
                self.sum+= node.val
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        return self.sum
                

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rangeSumBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: int
        """
        '''
        faster appraoch
        dfs along the tree to touch each node
        at node check lo hi constraints and add to sum outside of recursive call
        return the sum
        in this call you would never be smaller than the low or greater than the high
        think in terms of exploring possible values in the left and right subtrees
        '''
        self.sum = 0

        def dfs(node):
            if not node:
                return
            if low <= node.val <= high:
                self.sum+= node.val
            if node.val > low:
                dfs(node.left)
            if node.val < high:
                dfs(node.right)

        dfs(root)
        return self.sum

#iterative bfs
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rangeSumBST(self, root, low, high):
        """
        :type root: TreeNode
        :type low: int
        :type high: int
        :rtype: int
        """
        '''
        interative approach uisng stack
        '''
        result = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                if low <= node.val <= high:
                    result += node.val
                if node.val > low:
                    stack.append(node.left)
                if node.val < high:
                    stack.append(node.right)
        return result

############################
#Longest Mountain in Array
############################
#fuck this problem...
class Solution(object):
    def longestMountain(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        what if i took the first difference array
        a mountain is defined as increasing up to a point and then decreasing
        there should be only one flip, the second flip is the start of the next mountain
        keep in mind that we only want ones that have length 3
        it should only change from positive to negative once
        '''
        N = len(A)
        dp = [0]*N
        for i in range(1,N-1):
            if A[i] > A[i+1]:
                dp[i] = -1
            elif A[i] < A[i+1]:
                dp[i] = 1
            else:
                dp[i] = 0
        #now i can go across my dp array and increment my counter at the first occurnece of 1
        #allow for a change of -1
        #terminate the counter
        mountains = []
        count = 0
        increasing = True
        for i in range()

#well this problem wasn't as bad as i thought..
class Solution(object):
    def longestMountain(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        we can define a peak as an element at i that is greater than i-1 and i+1
        find peak and use two pointers
        move left is decreasing and right if decreasing, they should both be decreasing from each side
        of the peak
        '''
        results = 0
        N = len(A)
        
        for i in range(1,N-1):
            #search for peak
            if A[i-1] < A[i] > A[i+1]:
                #split points
                l,r = i,i
                #check left side
                while l > 0 and A[l] > A[l-1]:
                    l -= 1
                #move right
                while r +1 < N and A[r+1] < A[r]: #watch for the bounds condition on the right pointer
                    r += 1
                #update
                results = max(results, r-l+1)
        
        return results

###################
#MMirror Reflection
###################
class Solution(object):
    def mirrorReflection(self, p, q):
        """
        :type p: int
        :type q: int
        :rtype: int
        """
        '''
        even q will hit 2
        but q needs to be odd
        but in the case where q is larger, draw out extra room
        as long a q is odd, it will hit 0 or 1 at least draw diagram
        we have p*m == q*n
        if n is even it must reach receptor 2
        if n is odd, either 0 or 1, based on the number of q's
        we simulate gcd
        
        '''
        #look up gcd trick
        m = n = 1
        while p*m != q*n:
            #m will always be greater then n, since p is greater than q
            #increment n
            n += 1
            #adjust m
            m = q*n // p #note we will eventually get an answer
        
        if n % 2 == 0:
            return 2
        elif m % 2 != 0:
            return 1
        else:
            return 0


######################
# Merge Intervals
######################
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        sort in the starting intervals
        and then just keep updating the last added interval
        '''
        intervals.sort(key=lambda x: x[0])
        output = [intervals[0]]
        for start,end in intervals[1:]:
            if start > output[-1][1]: #insert
                output.append([start,end])
            elif end > output[-1][1]: #update
                output[-1][1] = end
        return output

###########################
# Subtreee of Another Tree
###########################
#close on this one, but still shaky on the implementation
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        '''
        descend tree s in any order traversal until we find the node that matches the root of t
        keep traversing at that point in the same traversal as s
        return true once done
        if not return false
        '''
        def dfs(root1,root2):
            if not root1:
                return
            if root1.val == root2.val:
                #when i found the matching node recurse
                dfs(root1.left,root2.left)
                dfs(root1.right,root2.right)
                return True
            dfs(root1.left,root2) and dfs(root1.right,root2)
            return True
        
        dfs(s,t)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        def dfs(tree,subtree):
            if tree == None and subtree == None:
                return True
            if tree == None or subtree == None:
                #not the same
                return False
            #recurse
            if tree.val == subtree.val:
                return dfs(tree.left,subtree.left) and dfs(tree.right, subtree.right)
        
        #edgecases
        if s == None:
            return False
        elif dfs(s,t):
            return True
        else:
            return dfs(s.left,t) or dfs(s.right,t)


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        '''
        this is kind of a wierd dfs question
        we recurse on two nodes
        but we also have to recurse on the node in the main tree
        '''
        def dfs(node):
            #inner function to recurse again
            def same(n1,n2):
                if not n1 and not n2:
                    return True
                if not n1 or not n2 or n1.val!= n2.val:
                    return False
                return same(n1.left,n2.left) and same(n1.right, n2.right)
            if not node:
                return False
            if node.val == t.val and same(node,t):
                return True
            return dfs(node.left) or dfs(node.right)
        
        return dfs(s)


#####################
#Decode String
#####################
#pretty good
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        i need to detect the inner brackets first before evaluating
        it could be nested. but how?
        i can use two pointers
        first on detects opening bracket the other detects closing bracket
        '''
        decoded = ""
        N = len(s)
        i,j = 0,0
        while i < N and j < N:
            #i detects opening, j is closing
            while s[i] != '[':
                i += 1
            j = i
            while s[j] != ']':
                j += 1
            #take the coeffcient
            string = int(s[i-1])*s[i+1:j]
            decoded += string
            i = j
            i += 1
            j += 1
        return decoded

#almost, but remember the coef can b greater than 9
class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        well i knew we had to use a stack
        the idea is to keep passing over the string pushing each char o the stack until it is not a closing bracket
        one we encounter closing bracket we need to decode
        there are two cases we need to worry about
            1. current chart is not a closing bracket, push on to stack
            2. char is a clsoing bracket
        start decpdomg the last traversed string by popping the string and number from the top of the stack
        pop from stack until the next char is not an opening bracke and append
        pop opening bracket from stack
        pop from stack until next char is a digit and build the number using k
        
        '''
        stack = []
        for i in range(0,len(s)):
            #when closing bracket decode the stack up to previous opening
            if s[i] == ']':
                decoded_string = ''
                #keep popping as until opening
                while stack[-1] != '[':
                    decoded_string += stack.pop()
                #last opening [
                stack.pop()
                coef = stack.pop()
                decoded_string *= int(coef)
                #push back in reverse order
                for j in range(len(decoded_string)-1,-1,-1):
                    stack.append(decoded_string[j])
            else:
                stack.append(s[i])
        output = ''
        while stack:
            output += stack.pop()
        return output[::-1]

class Solution(object):
    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        well i knew we had to use a stack
        the idea is to keep passing over the string pushing each char o the stack until it is not a closing bracket
        one we encounter closing bracket we need to decode
        there are two cases we need to worry about
            1. current chart is not a closing bracket, push on to stack
            2. char is a clsoing bracket
        start decpdomg the last traversed string by popping the string and number from the top of the stack
        pop from stack until the next char is not an opening bracke and append
        pop opening bracket from stack
        pop from stack until next char is a digit and build the number using k
        
        '''
        stack = []
        for i in range(0,len(s)):
            #when closing bracket decode the stack up to previous opening
            if s[i] == ']':
                decoded_string = ''
                #keep popping as until opening
                while stack[-1] != '[':
                    decoded_string += stack.pop()
                #last opening [
                stack.pop()
                #in the case the coef is greater than 9
                base = 1
                k = 0
                while stack and ord('0') <= ord(stack[-1]) <= ord('9'):
                    #increment k
                    k = k + int(stack.pop())*base
                    base *= 10
                decoded_string *= int(k)
                #push back in reverse order
                for j in range(len(decoded_string)-1,-1,-1):
                    stack.append(decoded_string[j])
            else:
                stack.append(s[i])
        output = ''
        while stack:
            output += stack.pop()
        return output[::-1]

####################
# Third Maximum Number
###################
#close
class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        sliding window of three and keep each three orders
        sort the sliding window
        '''
        if len(nums) < 3:
            return max(nums)
        N = len(nums)
        candidates = nums[:3]
        candidates.sort(reverse=True)

        for i in range(3,N-3):
            candidates[0] = max(candidates[0],nums[i])
            candidates[1] = max(candidates[1],nums[i+1])
            candidates[2] = max(candidates[2],nums[i+2])
        
        return candidates[-1]
        

class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        remove the maxs and just return max after drpppp
        '''
        nums = set(nums)
        first_max = max(nums)
        
        if len(nums) < 3:
            return first_max
        
        
        nums.remove(first_max)
        second_max = max(nums)
        nums.remove(second_max)
        return max(nums)

class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        keep track of max seen set, when past three remove the min
        then just return the min of the max set
        '''
        maxes = set()
        for num in nums:
            maxes.add(num)
            if len(maxes) > 3:
                maxes.remove(min(maxes))
                
        if len(maxes) == 3:
            return min(maxes)
        
        return max(maxes)
        

##########################3#######
#Search In Rotated Sorted Array II
###################################
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        '''
        from the first problem:
        you need to find the pivot point you will find that nums[i] > nums[i+1]
        this wouldn't be the case if the array was unpivoted and sorted
        
        '''
        if not nums: 
            return False
        if len(nums)==1: 
            return nums[0]==target 
        if nums[0]==target or nums[-1]==target: 
            return True 
        if nums[0]==nums[-1]: 
            return target in nums 
        
        #double binary search
        #first binary serach
        l,r = 0, len(nums) - 1
        
        while l < r:
            mid = l + (r-l) // 2
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid - 1 #not minus 1
                
        start = l
        l = 0
        r = len(nums)-1
        
        #now check where to search
        if (target >= nums[start] and target <= nums[r]):
            l = start
        else:
            r = start
        #second binary search
        while l <= r:
            mid = l + (r-l) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] < target:
                l = mid+1
            else:
                r = mid - 1
        return False


 #FROM KC
 class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        '''
        https://www.youtube.com/watch?v=WkihvY2rJjc&t=119s&ab_channel=KnowledgeCenter
        '''
        l,r = 0, len(nums)-1
        while l <= r:
            mid = l + (r-l)//2
            #check if target
            if nums[mid] == target:
                return True
            #when l == mid, and r == mid we cant tell, keep decrementing l and r
            if (nums[l]==nums[mid]) and (nums[r]==nums[mid]):
                l += 1
                r -= 1
            #two special cases
            elif nums[l] <= nums[mid]: #actually increasing
                if (nums[l] <= target) and (nums[mid] > target):
                    #reject other half
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if (nums[mid] < target) and (nums[r] >= target):
                    l = mid + 1
                else:
                    r = mid - 1
        return False

######################################
#Numbers At Most N givine Digit Set
#######################################
#recurison doesn't really work here 
class Solution(object):
    def atMostNGivenDigitSet(self, digits, n):
        """
        :type digits: List[str]
        :type n: int
        :rtype: int
        """
        '''
        brute force
        recursive function to build up a digit
        check if casting to int is less than n
        if so, append to results outside of call
        '''
        results = []
        def rec_build(build, digits,n):
            if build == "":
                return
            if int(build) < n:
                results.append(build)
            for i in range(0,len(digits)):
                rec_build(build+digits[i], digits[:i]+digits[i+1:],n)
            print build
        rec_build("",digits,n)


class Solution(object):
    def atMostNGivenDigitSet(self, digits, n):
        """
        :type digits: List[str]
        :type n: int
        :rtype: int
        """
        '''
        this was a hard problem, but its better seeing how its done than skipping it
        first call a pos int X valid if X <= N
        only ocnsists of digits form D
        if N has K digits, we can only write a vliad number if little k digits such that k < K
        then there are (len(digits))^k possible numbers we could write 
        note the recusrive solution is harder 
        DP!
        for samplex N = 2345, K = 4
        and D is the set 1-9
            if the first digit we write is less than the first digit of N then we could write any numbers after for a totla of (len(D))^K-1 valid numbers from the prefix
            in our example we could take 1111 to 1999
            
            if the first digit we write is the same, then we need the next figit equal or lower
            if we start with 2, the nex digit needs to be <= 3
            we can't write a larger digit because if we started with 3, then the number is too big
        
        algo:
        let dp[i] be the number of ways to write a valid number N 
        for example if N = 2345, then dp[0] would be the number of vlad numbers at most 2345, dp[1] would be the ones up to 345, dp[2] 45, dp[3] 5
        
        WLOG, dp[i] = (number of d in D with d < S[i])*((D.length(**K-i-1))) if S[i] in D
        uhhhhh!??!?!
        '''
        #convert digits into a string
        S = str(n)
        K = len(S)
        dp = [0]*K + [1]
        
        #allocate dp array, recall dp[i] is the totla number of valid integers of N was N[i:]
        for i in range(K-1,-1,-1):
            for d in digits:
                if int(d) < int(S[i]):
                    dp[i] += len(digits)**(K-i-1)
                elif int(d) == int(S[i]):
                    dp[i] += dp[i+1]
        
        return dp[0] + sum(len(digits)**i for i in range(1,K))
        

#########################
#Unique Morse Code Words
#########################
class Solution(object):
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        convert each word in words to its morse
        set the morse
        get the length
        '''
        morse = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        #a-z indexed from 0 to 25
        #off set with ord(a)
        decoded = []
        for w in words:
            temp = ""
            for char in w:
                symbol = morse[ord(char) - ord('a')]
                temp += symbol
            decoded.append(temp)
        return len(set(decoded))

#########################################################
#  Longest Substring with At Most Two Distinct Characters
#########################################################
class Solution(object):
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        brute force
        n squared would be to check all possible substrings
        update max length under contstraint len(set(substring)) == 2
        TLE obvie
        '''
        if len(s) < 3:
            return len(s)
        max_length = float('-inf')
        for i in range(0,len(s)):
            for j in range(i,len(s)):
                substring = s[i:j+1]
                if len(set(substring)) <= 2:
                    max_length = max(max_length,len(substring))
        return max_length

#LC solution
class Solution(object):
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        two pointer apporach, left and right
        keep advancing right to examine each char
        for the left, only advance once we have more than two unq letters in our hash
        move it up the +1 from the least recently seen char
        difference of left and right +1 with max update
        '''
        n = len(s)
        if n < 3:
            return n
        
        l,r = 0,0
        mapp = defaultdict()
        max_length = 2
        
        while r < n:
            #add char to mapp 
            mapp[s[r]] = r
            #move up
            r += 1
            
            #now check constraint
            if len(mapp) == 3:
                #find the last rececntly seen
                last_seen_idx = min(mapp.values())
                #remove from maap
                del mapp[s[last_seen_idx]]
                #move up left
                l = last_seen_idx + 1
            #update
            max_length = max(max_length, r-l)
            
        return max_length


#############################
# House Robber III 23Nov2020
#############################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        go through this carefully
        since the problem asks us to find out the maximum amount the their can 
        get starting form this noe
        dfs(node):
            if not node:
                retun -
            else:
                two choices, rob this node or not?
                if not rob:
                    helper(node.left) + helper(node.right)
                if rob:
                    return node.val
                    #but now we cannot take from the left or right
                    #so we need to touch the nodes siblings to see if we could get more
                    
        the best practice is touch only its children, WE FUCKING RECURSE AGAIN FROM THE CHILDREN
        the ideal situation is to make node.left and node. right handle grandchildren
        how? we can let them know wheter we robbed this node or not by passing this information as input
            two choices? rob or not
            rob = node.val + helper(node.left, parent_robbed = True) + helper(node.right parent_robbed = True)
            not_rob = helper(node.left, parent_robbed=False) + helper(node.right, parent_robbed=False)
            return max(rob, not_rob)
        another problem is that helper is called too many times, in fact FOUR TIMES at one invocation
        also note then when we call helper(node.left, True) and helper(node.right, False), helper(node.left.left, False) is called....I DONT GET WHY?!?!
        ohhhh now i do
        so we can combine them
        we return the results of helper(node.left,True) and helper(node.left, False) as one single return call in an array
        left = helper(node.left)
        right = helper(node.right)
        some calculation...
        return [max_if_rob, max_if_not_rob]
        
        #algo,
        use a helper function which received a node as input and returns a two element array, where the first element representes the max amount of money they theif can rob if starting form this node without robbing this node,
        the second eleemtn represents the max amount of money the theif can rob if starte fromt his node and robbing this node
    
        '''
        def dfs(node):
            #first element money theif gets starting from this node and robbing
            #second element, starting form this and not robbing
            if not node:
                return [0,0]
            left = dfs(node.left)
            right = dfs(node.right)
            #now we decide we we rob this node or nor
            #if we rob, we cannot rob its children
            rob = node.val + left[1] + right[1]
            #else we choose to take from children or not
            not_rob = max(left) + max(right)
            return [rob, not_rob]
        return max(dfs(root))


#https://www.youtube.com/watch?v=mSzz_bZUVCQ&ab_channel=AnishMalla
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        stolen = {}
        not_stolen = {}
        def dfs(node,parent_stolen):
            #parent_stolen keeps track of whether or not we can steal or notsteal its children
            if not node:
                return 0
            if parent_stolen:
                #if stealing
                #only optein is to not steal the current node we are at
                if node in stolen:
                    return stolen[node]
                result =  dfs(node.left,False) + dfs(node.right,False) 
                stolen[node] = result
                return result
            else:
                #if not stealing
                #given a choise b/tsteam and not stealing
                #condition for stealing at current node
                if node in not_stolen:
                    return not_stolen[node]
                steal = node.val + dfs(node.left,True) + dfs(node.right, True)
                #not stealing
                not_steal = dfs(node.left, False) + dfs(node.right, False)
                result = max(steal,not_steal)
                not_stolen[node] = result
                return result
        return dfs(root,False)

######################
# Basic calculator II
########################
#welll it workds
class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        because of nested functions we need to use a stack
        but we need to remembed order of opreations
        *  and / get precedence over + and -
        do nothing in space
        there are no () tahnk go
        dump into stack taking care of * and / first going right ot left
        then just pass over the stack left to right doing normal evals with + and -
        first pass to trim white spaces
        '''
        #edge cases
        ops = set(["+", "-", "*", "/"])
        count_ops = 0
        for i in range(len(s)):
            if s[i] in ops:
                count_ops += 1
        if count_ops == 0:
            return int(s)
        N = len(s)
        #take care of white spaces first
        i = 0
        trimmed_s = ""
        while i < N:
            if s[i] == ' ':
                pass
                i += 1
            else:
                trimmed_s += s[i]
                i += 1
        #now lets make sure we get the whole int and individual operations, dump into list ['122','+']
        #i can use the two pointer method to mark the start and beg of my number
        temp  = []
        l,r = 0,0
        while r < len(trimmed_s):
            if trimmed_s[l] in ops:
                temp.append(trimmed_s[l])
                l += 1
                r += 1
            while r  < len(trimmed_s) and ord('0') <= ord(trimmed_s[r]) <= ord('9'):
                r += 1
            temp.append(trimmed_s[l:r])
            l = r

        # now do * / and make sure to get the ints
        N = len(temp)
        stack = []
        i = 0
        while i < N:
            if temp[i] == '*':
                stack.append(str(int(stack.pop())*int(temp[i+1])))
                i += 2
            elif temp[i] == "/":
                stack.append(str(int(stack.pop())/int(temp[i+1])))
                i += 2
            else:
                stack.append(temp[i])
                i += 1

        #now take care of addition and substracton by passing the stack
        result = int(stack[0])
        N = len(stack)
        i = 1
        while i < N:
            if stack[i] == "+":
                result += int(stack[i+1])
                i += 2
            elif stack[i] == "-":
                result -= int(stack[i+1])
                i += 2
        return result


class Solution(object):
    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        note we can treat subtraction as adding a negative
        take care of multiple ans divide first
        scane the input string fomr left ot righ and eal expressions based:
        store three variabils 
        currnet var
        current op
        and currnet chat
        '''
        if not s:
            return 0
        stack = []
        curr_num = 0
        curr_op = "+" 
        
        all_ops = {"+","-","*","/"}
        nums = set(str(x) for x in range(10))
        
        for i in range(len(s)):
            char = s[i]
            
            if char in nums:
                curr_num = curr_num*10 + int(char) #cool trick
            if char in all_ops or i == len(s)-1:
                if curr_op == "+":
                    stack.append(curr_num)
                elif curr_op == "-":
                    stack.append(-curr_num)
                elif curr_op == "*":
                    #carry out
                    stack[-1] *= curr_num
                elif curr_op == "/":
                    #note at the end of the loop we have yet to execute * or /
                    #we still need to carry that ou
                    stack[-1] =  round(stack[-1] / curr_num)
                
                curr_num = 0
                curr_op = char
        
        return int(sum(stack))

###################################
# Smallest Integer Divisible by K
###################################
class Solution(object):
    def smallestRepunitDivByK(self, K):
        """
        :type K: int
        :rtype: int
        """
        '''
        keep generating 1 11 11 1111 111
        stop when remainer by K is zero
        '''
        if K % 2 == 0 or K % 5 == 0:
            return -1
        candidate = 1
        while candidate % K != 0:
            candidate = candidate*10 + 1
            
        if candidate % K == 0:
            return len(str(candidate))
        else:
            return -1
            

class Solution(object):
    def smallestRepunitDivByK(self, K):
        """
        :type K: int
        :rtype: int
        """
        '''
        do to interger overflow we need to use the remainder
        the while loop will only terminate if a solution exists
        so we need to keep track of seen remainders
        if we see it again, we have encounter a cycle
        '''
        remainder = 1
        length_N = 1
        seen = set()
        
        while remainder  % K != 0: #cant go into K
            N = remainder*10 + 1
            #new remainder
            remainder = N%K
            length_N += 1
            
            if remainder in seen:
                return -1
            else:
                seen.add(remainder)
                
        return length_N

class Solution(object):
    def smallestRepunitDivByK(self, K):
        """
        :type K: int
        :rtype: int
        """
        '''
        do to interger overflow we need to use the remainder
        the while loop will only terminate if a solution exists
        so we need to keep track of seen remainders
        if we see it again, we have encounter a cycle
        
        using th pigeonhole principle
        recall that the numbe rof possible values or eaminder is in range 0,K-1
        in fact it is K (try a few numbers to see that this is true)
        as a result the while loop should only continue K times
        otherwise the reamidners reperat
        if N exists, the while loop must return length_N in the first K loops
        '''
        reaminder = 0
        for L in range(1,K+1):
            reaminder = (reaminder*10 + 1) % K
            if reaminder == 0:
                return L
            
        return -1

###########################################################
# Longest Substring with At Least K Repeating Characters
#############################################################
class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        '''
        brute force would be to generate all subtrings counter it and check each char constaint
        if satified update the length
        TLE ovbiee
        
        '''
        max_length = 0
        N = len(s)
        for i in range(N):
            for j in range(i,N):
                count = Counter(s[i:j+1])
                #now we need to check that all elements count is >= k
                #just check the min
                if min(count.values()) >= k:
                    max_length = max(max_length, len(s[i:j+1]))
        return max_length
                
#recursively
class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        #usind divide and conquer
        #we can write a function
        #longest(start,end) = max(longest(start,mid),longest(mid,end))
        #we find the split position mid which is the first invalid char
        #which would be the char whose count is < k
        def rec_split(s,k):
            if len(s) == 0 or k > len(s):
                #when to terminae the callt
                return 0
            
            count = Counter(s)
            sub1 = ""
            sub2 = ""
            
            for i,char in enumerate(s):
                if count[char] < k:
                    #invalid char
                    sub1 = rec_split(s[:i],k)
                    sub2 = rec_split(s[i+1:],k)
                    break
            else:
                return len(s)
            
            return max(sub1,sub2)
        
        return rec_split(s,k)

class Solution(object):
    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        '''
        i can begin by counting up the char occurences
        we divide the string into regions that only contain valid char counts (i.e counts >= k)
        since inclding those counts would yeild an invalid string
        this is just anothe recursive solution
        '''
        def rec_build(s,k):
            N = len(s)
            if N == 0 or N < k:
                return 0
            if k <= 1:
                return N
            
            counts = Counter(s)
            l = 0
            while l < N and counts[s[l]] >= k:
                l += 1
            if l >= N-1: #got to the end
                return l
            
            s1 = rec_build(s[0:l],k)
            #if there are still parts of the string with char counts < k
            while l < N and counts[s[l]] < k:
                l += 1
            if l < N:
                s2 = rec_build(s[l:],k)
            else:
                #if we have gotten to the end, there is nothing
                s2 = 0
            
            return max(s1,s2)
        
        return rec_build(s,k)

############################
#Partition Equal subset sum
############################
#close but it wont work for all cases
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        similar to building all subsets but build one subset, and keep track of the complement subset
        then at each iteration check if sums of subset and comp subset match
        
        '''
        N = len(nums)
        chosen_idxs = []
        not_chosen = []
        def build_subset(start,temp):
            chosen_idxs.append(temp)
            not_chosen.append([foo for foo in list(range(N)) if foo not in temp])
            for i in range(start,len(nums)):
                build_subset(i+1, temp+[i])
        build_subset(0,[])
        
        #now only start at the first one
        N = len(chosen_idxs)
        for i in range(1,N):
            subset = []
            comp = []
            for idx in chosen_idxs[i]:
                subset.append(nums[idx])
            for idx in not_chosen[i]:
                comp.append(nums[idx])
            if sum(subset) == sum(comp):
                return True
        
        return False

#official LC< TLE Again
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        note that the sum of the whole array must be even in order for us to split it up
        we find a subset in na ray where the sum must be equal to the subset
        ehaustively build them up
        assume there is an array nums of size n
        and we heave to find if there exists a subset with total sum = sumsetsum
        two cases at element x:
            1. x is part of the the subset, at which subsetsum = subsetsum - x
            2. x i not in the subset, so we take teh previous sum without x
        def dfs function that finds if a subset can equal that sum
        dfs(subsetsum,n) = dfs(subsetsum - nums[n],n-1) or dfs(subsetsum,n-1)
        review knapsacking
        this gets TLE
        '''
        def dfs(nums,n,subsetsum):
            if subsetsum == 0:
                return True
            if n == 0 or subsetsum < 0:
                return False
            return dfs(nums,n-1,subsetsum - nums[n-1]) or dfs(nums,n-1,subsetsum)
        
        totalsum = sum(nums)
        if totalsum  % 2 != 0:
            return False
        subsetsum = totalsum / 2
        N = len(nums)
        
        #now find the subset sum
        return dfs(nums,N-1,subsetsum)


class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        of course we can add memoization to the dfs calls
        but how?
        we can create a 2d boolean memo, that has dimensions len(nums) + 1 by subsetsum + 1
        we then look at the memo to check our result
        '''
        def dfs(nums,n,subsetsum,memo):
            if subsetsum == 0:
                return True
            if n == 0 or subsetsum < 0:
                return False
            if memo[n][subsetsum] is not None:
                return memo[n][subsetsum]
            
            result = dfs(nums,n-1,subsetsum - nums[n-1],memo) or dfs(nums,n-1,subsetsum,memo)
            memo[n][subsetsum] = result
            return result
        totalsum = sum(nums)
        if totalsum % 2 != 0:
            return False
        subsetsum = totalsum / 2
        N = len(nums)
        memo = [[None]*(subsetsum + 1) for _ in range(N+1)]
        
        return dfs(nums, N-1, subsetsum, memo)

class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        another IDDFS approach
        this is a variant of the knapsacking problem
        from the recursino tree we go left when we take element x as part of the subset
        but decrement the target
        we go right when we do not include element x as part of the subset
        add set of targets, why?
        if we have descned a path that includes a target, it means we haven't fired true for our dfs
        so we end the call there
        
        '''
        self.cache = set()
        def dfs(target, nums):
            if target < 0:
                return False
            if target == 0:
                return True
            if target in self.cache:
                return False
            self.cache.add(target)
            
            for i,n in enumerate(nums):
                if dfs(target - n, nums[i+1:]) or dfs(target, nums[i+1:]):
                    return True
            
            return False
        
        subsetsum = sum(nums) 
        if subsetsum % 2 != 0:
            return False
        
        return dfs(subsetsum/2,nums)
            
##########################
#Sliding Maximum Window
###########################
#TLE, im not sure why this is a hard one
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        '''
        i dont get why this is hard?
        just keep moving the sliding window until you can't
        add the max to a list, and return the list
        '''
        N = len(nums)
        maxes = []
        for i in range(N-k+1):
            maxes.append(max(nums[i:k+i]))
            
        return maxes
        

class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        '''
        LC article sucks
        https://leetcode.com/problems/sliding-window-maximum/discuss/237611/Simplest-O(n)-Python-Solution-with-Explanation
        1. maintain deque of indexes which store the largest nums we've seen so far - the len of deq will never be greater tjna k
        2. we need to keep sanity of the deque
            the deque should never point to elements smaller than the current one (pop em off)
            deque should enver point to elements outside the bounds of our window
        3. keep furnishing from the front of the deque - it happens that the max will always be from the front
        
        '''
        d = deque()
        results = []
        for i,v in enumerate(nums):
            #top of the stack must always be greater than the current eleemtn
            while d and nums[d[-1]] < v:
                d.pop()
            #q up the index
            d.append(i)
            #if the index it outside k
            if d[0] == i - k:
                d.popleft()
            #we are always appending after i >= k
            if i >= k - 1:
                results.append(nums[d[0]])
        return results
        

###################
#Jump Game III
###################
class Solution(object):
    def canReach(self, arr, start):
        """
        :type arr: List[int]
        :type start: int
        :rtype: bool
        """
        '''
        from the hint, i can think of this using BFS, whic means graph
        each index is connected to another index by end edge, the edge is i +arr[i] or i - arr[i]
        if a path exists to the zero val, i should be able to traverse the graph
        algo
            add node to q, and mark if we have visited
            than checkits paths
        '''
        N = len(arr)
        visited = set() #mark with index
        q = [start]
        
        while q:
            node = q.pop(0)
            if arr[node] == 0:
                return True
            if node in visited:
                continue
            
            for i in [node + arr[node],node - arr[node]]:
                if 0 <= i < N:
                    q.append(i)
                    
            visited.add(node)
            
        return False

#############################
#Maximum Average Subarray II
############################
class Solution(object):
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        '''
        the naive way would be to find all sub arrays, average them, and update a max
        number of posisble windows is len(nums) - k + 1
        TLE obvie
        '''
        N = len(nums)
        averages = [float(sum(nums))/float(N)]
        for window_size in range(k,N):
            for i in range(N-window_size+1):
                averages.append(float(sum(nums[i:i+window_size])) / window_size)
                
        return max(averages)

class Solution(object):
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        '''
        https://leetcode.com/problems/maximum-average-subarray-ii/solution/
        we know the average must lie beteween min an max, but we need subarray with at least k eleetns
        we can use binary search to find an array with at least k elements that is greater than the mid value we got using binary seach
        thus we can see that if after subtracting the mid numbers from the elements in our candidate array, with moe than k - 1 elments, withinnums, if the sum of elements of this reduced subarray is greater than 0, we can achieve an average value greater than mid!
        the trick is that for array elements a,b,c
        we can write (a + b + c) / 3 >= mid
        (a-mid) + (b - mid) + (c- mind) >= 0
        if sum candidata elements after mid subtracting is of length >=k and >=0
        we can achieve thid mid
        otherwise if the sum is less then 0, we cant use this mid
        so we need a smaller one, which is just binary search
        in order to determing is such a subaray exits ina linear manner
        we keep on adding nums[i] - mid to the sum tilt he ith element until we get at least k elements
        when sum becomes >= 0 we can directly determine that we can increase the avg beyond mid
        otherwise we continue making additions to sum for elements beoung the kth element
        if we know the cum sum up to indices i and j, say sum_i and sum_j, respectively we can determine the sum of the subarray between these indices as sumj - sumi
        instead of checking all possible vluaes of sum- we can just conisder the min cum cum up to the index j-k
        this is because if the required condition can't be satisfied with the min sumi it can enver be stafied with  alrge vale
        To fulfil this, we make use of a prevprev variable which again stores the cumulative sums but, its current index(for cumulative sum) lies behind the current index for sumsum at an offset of kk units. 
        '''
        def check(nums,mid,k):
            #check if we have valid subarray
            prev = 0
            min_sum = 0
            total = sum(nums[i] - mid for i in range(k))
            if total >= 0:
                return True
            #now check
            for i in range(k,len(nums)):
                total += nums[i] - mid
                prev += nums[i-k] - mid
                min_sum = min(min_sum,prev)
                if total > min_sum:
                    return True
            return False
        
        #binary search away
        max_val = max(nums)
        min_val = min(nums)
        error = float('inf')
        prev_mid = max_val
        while  error > 0.00001:
            mid = float((max_val + min_val)) / float(2)
            if check(nums,mid,k):
                #can have a potential higher average
                min_val = mid
            else:
                max_val = mid
            #update error
            error = abs(prev_mid-mid)
            prev_mid = mid
        return min_val

######################
#The Skyline Problem
######################
class Solution(object):
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        https://www.youtube.com/watch?v=GSBLe8cKu0s&t=724s&ab_channel=TusharRoy-CodingMadeSimple
        similar to merge intervals
        iterate over the points, move point up to edge of building
        at start and at end of buildings
        move from left to right enctouner start of building
        push height of building into heap
        if max changes, there is an overlap
        if at end, remove building from heap
        sort points on x, edge cases are if xs are the same
        furnish out y from max of heap
        edge cases
        
        when starts of two or more buildings is same, look for builidng with higher height
        when ends of two or more buildings are the same, look at lower height first
        when end is start of next building, only start, endstart,end
        use heap q
        '''
        #create building points array
        #three element 0 for start 1 for end
        building_points = []
        for start,end,height in buildings:
            building_points.append([start,height,0])
            building_points.append([end,height,1])
        #sort, recall the edge cases for sorting
        #sort on start, if starts are same, give precendence if building demarcation is start of builiding
        #negate height, to take the higher buildin first
        #
        building_points.sort(key = lambda x:(x[0],x[2],-x[1] if x[2] == 0 else [1]))
        results = []
        q_heights = [0] #init with 0
        heapify(q_heights)
        for b in building_points:
            #unpack
            index,height,event_type = b
            if event_type == 0: #start
                if height > -q_heights[0]:
                    results.append([index,height])
                heappush(q_heights,-height)
            else:
                #end
                heappop(q_heights)
                if height > -q_heights[0]:
                    results.append([index,-q_heights[0]])
        
        return results

class Solution(object):
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        https://leetcode.com/problems/the-skyline-problem/discuss/954585/Python-O(n-log-n)-solution-using-heap-explained
        Notes:
        1. Keep a so called active set, to what buildings we currently have in given point
        2. We need to quickly determine the highest builiding os far, so use a heap
        and perform lazy deletions
        
        Algo:
        1. For each building, put two tuples of info, left corner first marked by -1, then concat with right corner, 1
        2. Now we need to sort, which is where the edge cases come in. sort by x. if euqal sor tem by height multiplied by -1, we want taller buildings first
        3. load a dummy element into out heap and set
        4. go through oints, if we meeet the beginning of some building, add to active set, if end, remove from active set
        5. if we meet the left corner of a building and it is also height than we alread met so far, that is h > heap[0][0]. then we add this element to ans. we also add (-h.ind) to ouy heap (min heap)
        6. If we meet right corner of building and it is less than highest current building, we do nothing with our heap: (actually, what we need to do is to say, that current building is not active any more and we already did this). 
        7. Finally, if h == -heap[0][0], it means, that we need to pop some element from our heap. How many? Here we have our lazy updates: we look if building is active or not and if it is not, we remove and continue, until we have highest building which is active
        8. Finally, if new point we want to add has different height from what we have as the last element of our ans, we add new point to ans: [x, -heap[0][0]]: note, that we add point with height after we removed all inactive elements from heap
    
        '''
        #create new points, left marked with -1, right with 1
        #give reference to ith building with index
        points = []
        for i,(l,r,h) in enumerate(buildings):
            points.append((l,h,-1,i))
            points.append((r,h,1,i))
        #sort by x, then taller building in front
        points.sort(key = lambda x: (x[0],x[1]*x[2]))
        #init, heap is our heights with (height, index)
        heap_heights  = [(0,-1)]
        visited_buildings = set([-1])
        results = []
        
        for x,h,lr,ind in points:
            #start of building
            if lr == -1:
                visited_buildings.add(ind)
            else:
                visited_buildings.remove(ind)
            #print visited_buildings
            
            #start
            if lr == -1:
                if h > heap_heights[0][0]:
                    #include in answer
                    results.append([x,h])
                #max update
                heappush(heap_heights,(-h,ind))
            #at the end instead
            else:
                if h == heap_heights[0][0]: #the same
                    #we keep removing from our heap
                    while heap and heap[0][1] not in visited_buildings:
                        heappop(heap)
                #for the last building
                if -heap_heights[0][0] != results[-1][1]:
                    results.append([x,-heap_heights[0][0]])
        return results
        
