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