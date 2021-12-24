##########################
# 198. House Robber
# 01DEC21
##########################
#top down recursion with memozation
class Solution:
    def rob(self, nums: List[int]) -> int:
        '''
        this is just a 0/1 knapsack
        we can either chose to rob this house  and take its value
        or skip this house
        dp(i) represents the amount of profit i have atfter robbing house i
        need to hold if we robbed or not
        rather dp(i) gives the max profit i can get if start robbing at house i
        
        take = dp(i+2) + nums[i]
        no_take = dp(i+1) then advance 
        dp(i) = max(take,notake)
        '''
        memo = {}
        N = len(nums)
        def dp(i):
            if i >= N:
                return 0 #no profit
            if i in memo:
                return memo[i]
            
            rob = nums[i] + dp(i+2)
            no_rob = dp(i+1)
            ans = max(rob,no_rob)
            memo[i] = ans
            return ans
        
        return dp(0)

#bottom up dp
class Solution:
    def rob(self, nums: List[int]) -> int:
        '''
        to get dp(0)
        we need dp(1)
        to get dp(1) we need dp(2)
        so start from the end of the array
        '''
        if not nums:
            return 0
        
        N = len(nums)
        dp = [0]*(N+2)
        #edge case for zero beyond 2
        
        for i in range(N-1,-1,-1):
            rob = nums[i] + dp[i+2]
            no_rob = dp[i+1]
            dp[i] = max(rob,no_rob)
            
        return dp[0]
        
#bottom up constant space
class Solution:
    def rob(self, nums: List[int]) -> int:
        '''
        to get dp(0)
        we need dp(1)
        to get dp(1) we need dp(2)
        so start from the end of the array
        '''
        # Special handling for empty case.
        if not nums:
            return 0
        
        N = len(nums)
        
        rob_next_plus_one = 0
        rob_next = nums[N - 1]
        
        # DP table calculations.
        for i in range(N - 2, -1, -1):
            
            # Same as recursive solution.
            current = max(rob_next, rob_next_plus_one + nums[i])
            
            # Update the variables
            rob_next_plus_one = rob_next
            rob_next = current
            
        return rob_next

###################################
# 328. Odd Even Linked List
# 02DEC21
###################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can keep two pointers
        one odd and one even
        if i odds on one side and evens on one side
        i would want to make odd.next = even.next
        then just move the odd and even pointers
        
        at the end, connect odds tail to evens head
        
        to maintain the invariant, we make sure even and its next exists
        '''
        if not head:
            return None
        odd = head
        even = head.next
        evenHead = even
        
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        #tails
        odd.next = evenHead
        return head


##################################################
# 708. Insert into a Sorted Circular Linked List
# 01DEC21
##################################################
#close one, but you pretty much had it on an interview
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        '''
        i can find the break in the circular list
        before finding the break, pull all values into a list
        find place where insert val can be inserted, then put it there
        rebuild
        '''
        if not head:
            dummy = Node(val = insertVal)
            dummy.next = dummy
            return dummy
        seen = set()
        values = []
        
        curr = head
        while curr not in seen:
            values.append(curr.val)
            seen.add(curr)
            curr = curr.next
            
        #now scan values to find where to match, find the left bound
        left_bound = len(values)
        for i in range(len(values)-1):
            if values[i] <= insertVal <= values[i+1]:
                left_bound = i+1
                break
                
        lower_half = values[:left_bound]
        lower_half.append(insertVal)
        values = lower_half + values[left_bound:]
        
        #now build a new one and return it
        dummy = Node()
        temp = dummy
        for num in values:
            temp.next = Node(val = num)
            temp = temp.next
        print(values)
        print(dummy.next.val)
        print(temp.val)
        #reconnect
        temp.next = dummy.next
        return dummy.next

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        '''
        we can use two pointers to traverse the linked list
        we need to keep two pointers to keep track of curr and prev
        we can move through the list and find a suitable place for insertVal to go in between prev and curr
        
        algo:
            loop with two pointers
            termination condition occurs when prev gets back to head
            during loop, check if current place bounded by two pointers is the right place to insert
            if not go forward
        
        casework:
            1. the value of the new nodes sits between the min and max values of the current list
                or rather, prev.val <= insertVal <= curr.val
            2. value of new node goes beyon the min and max values of the curr list
            either less than the minimal value or greater than the maximal values
            in eitehr case the new node should be added right after the tail node
                first we should locate the position of the tail node, by finding the descending order between adjacent 
                check if new value goes beyong vlaues of tail and head nodes, which are pointed by prev and curr
                
                2.1 insertval >= prev.val
                2.2 insertval <= curr.val
                
                once we located the tail and head ndoes, we extend the original list by inserting the value in between the tail and head ndoes 
                or in between prev and curr pointers
                
            3. there is one case that does not fall into any of hte above, list contains uniform values
            just add the node anywhere
            
            4. empty LL, return the inservalt pointing to itself
        '''
        if head == None:
            newNode = Node(insertVal,None)
            newNode.next = newNode
            return newNode
        
        prev,curr = head, head.next
        toInsert = False
        
        while True:
            
            #case 1
            if prev.val <= insertVal <= curr.val:
                toInsert = True
            #case 2, locate tail element, prev points to the tail, or the largest element!
            elif prev.val > curr.val:
                if insertVal >= prev.val or insertVal <= curr.val:
                    toInsert = True
            
            if toInsert:
                prev.next = Node(insertVal,curr)
                #we made an insert!
                return head
            prev = curr
            curr = curr.next
            
            #break out of loop if prev gets to head
            if prev == head:
                break
        
        #we broke out of the loop, must be in case 3, just add at end
        prev.next = Node(insertVal,curr)

        return head

#with looping invariant
# https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/discuss/1294608/Python-solution-with-comments
class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        node = Node(insertVal)
        
        #empty list
        if not head:
            node.next = node
            return node
        
        #set min and max paointers, if list contains all the same values, stop advancing the pointers when 
        #we reach the head again
        max_node = head
        while max_node.val <= max_node.next.val and max_node.next != head:
            max_node = max_node.next
        
        #set min
        min_node = max_node.next
        
        #case 2, min < insertVal < max
        if min_node.val < insertVal < max_node.val:
            curr = min_node
            while curr.next.val < insertVal:
                curr = curr.next
            node.next = curr.next
            curr.next = node
            
        #case 3, at the ends
        else:
            node.next = min_node
            max_node.next = node
        
############################
# 152. Maximum Product Subarray
# 03Dec21
############################
#dp, linear space
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        '''
        i cant simply just treat this like Kadane's, where we take a max or zero
        because i negative later on in the array could re negate the negative and drive it up
        
        if all the numbers are positive, then the answer is just the whole array
        what is if used an accumlation of products
        [2,3,-2,4]
        [2,6,-12,-48]
        i could rescan, and find that largest is at 6
        then i want the part of the array where its increasing the most
        
        notes:
            zeros in the array would immeditale reset the streak
            negatives could re negate them
        
        i can store two dp arrays
        dp max will store the max up to this points
        dp min will store the min up this point
        '''
        N = len(nums)
        dp_max = [0]*N
        dp_min = [0]*N
        
        dp_max[0] = dp_min[0] = nums[0]
        
        #first pass, find min and max at each pointin the array
        for i in range(1,N):
            dp_max[i] = max(nums[i]*dp_max[i-1],nums[i],dp_min[i-1]*nums[i])
            dp_min[i] = min(nums[i]*dp_min[i-1],nums[i],dp_max[i-1]*nums[i])
        
        #second pass, find the max that could be obtained at each point using the two dp
        maxes = [0]*N
        #starting off take max of the beginning
        
        for i in range(N):
            maxes[i] = max(dp_max[i],dp_min[i])
            
        return max(maxes)

#we can eliminate the dp arrays and save values along the way instead
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        N = len(nums)
        
        dp_max = dp_min = nums[0]
        ans = dp_max
        
        #first pass, find min and max at each pointin the array
        for i in range(1,N):
            curr_max = max(nums[i]*dp_max,nums[i],dp_min*nums[i])
            curr_min = min(nums[i]*dp_min,nums[i],dp_max*nums[i])
            dp_max = curr_max
            dp_min = curr_min
            
            ans = max(ans,dp_max)
        
            
        return ans

###############################
# 1032. Stream of Characters
# 04DEC21
###############################
class StreamChecker:
    '''
    we need to check if stream of letters form a suffux
    we can build a Trie in reverse
    then mainin set of poiners during the query
    
    we also start from the end of the stream and ehck character by characeter going down the Trie
    when querying, we can use q deque, and just check into tree if we have formed a valid suffix
    '''

    def __init__(self, words: List[str]):
        self.trie = {}
        self.stream = deque([])
        
        for word in set(words):
            node = self.trie
            for ch in reversed(word):
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            
            #mark completion of word
            node['#'] = word
        

    def query(self, letter: str) -> bool:
        #load into stream
        self.stream.appendleft(letter)
        #now check
        node = self.trie
        
        for ch in self.stream:
            #valid suffic
            if '#' in node:
                return True
            if ch not in node:
                return False
            #move current node if stream matched
            node = node[ch]
        return '#' in node


# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)

#brute force
class StreamChecker:
    '''
    lets try brute force
    save all input chars and check each word for suffux
    '''

    def __init__(self, words: List[str]):
        self.words = set(words)
        self.stream = []
        self.suffixes = set()
        

    def query(self, letter: str) -> bool:
        self.stream.append(letter)
        #get all suffixes from this stream
        temp = "".join(self.stream)
        #add suffixes
        for i in range(len(temp)+1):
            self.suffixes.add(temp[-i:])
        #for each suffix check ends withw word
        for word in self.words:
            for suff in self.suffixes:
                if word.endswith(suff):
                    return True
        return False

# Your StreamChecker object will be instantiated and called as such:
# obj = StreamChecker(words)
# param_1 = obj.query(letter)


###########################
# 05DEC21
# 337. House Robber III
###########################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        '''
        same as the first house robber problem,
        but if i have robbed from this parent, i cannot rob any of its children
        return the max that i can rob
        that max that i robbed at a node is just the max(left,right)
        brute force recursino would be to just pass in wheter or not we robbed from the parent
        if we have robbed from parent, the just return from the children
        if we have'nt robbed, we have two choices so take the max
        TLE though
        
        '''
        def dfs(node,parent_robbed = False):
            if not node:
                return 0
            #if i have robbed this parent, just retun the calls from the childre
            if parent_robbed:
                return dfs(node.left,False) + dfs(node.right,False)
            if not parent_robbed:
                #two choices 
                rob = node.val + dfs(node.left,True) + dfs(node.right,True)
                not_rob = dfs(node.left,False) + dfs(node.right,False)
                return max(rob,not_rob)
        
        return dfs(root)
        
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        '''
        the problem with brute force recusion is that we have to many dfs calls
        we need to reduce the number of times we call our recursive function
        we have a ncessary call to both sides of three, so lets call them first
        we can pass the results up, first element if money we got from stating this node and robbing
        second element, starting and not robbing
        '''
        def dfs(node):
            if not node:
                return [0,0]
            left = dfs(node.left)
            right = dfs(node.right)
            rob = node.val + left[1] + right[1]
            not_rob = max(left) + max(right)
            return [rob,not_rob]
        
        return max(dfs(root))

#using memo
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        '''
        we can improve this using memoeization but we need to stored the robbed and not robbed states
        for each node, then we can just retreive
        
        '''
        robbed = {}
        not_robbed = {}
        
        def dfs(node,parent_robbed):
            if not node:
                return 0
            if parent_robbed:
                #first check
                if node in robbed:
                    return robbed[node]
                res = dfs(node.left,False) + dfs(node.right,False)
                robbed[node] = res
                return res
            #not robbed
            if not parent_robbed:
                if node in not_robbed:
                    return not_robbed[node]
                rob_here = node.val + dfs(node.left, True) + dfs(node.right,True)
                no_rob_here = dfs(node.left,False) + dfs(node.right,False)
                res = max(rob_here, no_rob_here)
                not_robbed[node] = res
                return res
        
        return dfs(root,False)

########################################################
# 1217. Minimum Cost to Move Chips to The Same Position
# 06DEC21
########################################################
# move to zeroth and first columns
class Solution:
    def minCostToMoveChips(self, position: List[int]) -> int:
        '''
        we are given n chips, and position[i] marks the position of the ith chip
        if we move a chip two spaces, it costs zero
        if we move a chip 1 space, it costs 1
        return min cost of moving chips to the same position
        greedy:
            first move chips that are two away to some spot, total costs 0
            then just move the ones that are one away
            * first move keeps parity of element as it is
            * second move changes parity of element
            * since the first move is free, if all number have same parity, the answer would be zero
            * find minimum cost to make all numbers same parity
            * if they are the same parity, it costs zero to move them
        
        try moving all chips to the zero position
        then move all chips to the one positions
        then just move the the smallest pile over
            
        '''
        zero_position = 0
        one_position = 0
        
        for pos in position:
            #if positino if even, move it to zero
            if pos % 2 == 0:
                zero_position += 1
            #goes to the one position
            else:
                one_position += 1
        
        #move the smaller pile over
        return min(zero_position, one_position)

############################
# 06DEC21
# 434. Number of Segments in a String
############################
class Solution:
    def countSegments(self, s: str) -> int:
        '''
        split count does not work
        need to advance pointers and a white space, then capture segments when not
        '''
        N = len(s)
        ans = 0
        i = 0
        while i < N:
            #keep advnacing it is a white space
            while i < N and s[i] == ' ':
                i += 1
            #start of segment
            if i < N:
                ans += 1
            #advance on chars
            while i < N and s[i] != ' ':
                i += 1
        
        return ans

##############################
# 1290. Convert Binary Number in a Linked List to Integer
# 07DEC21
##############################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        '''
        first pass find number of nodes
        then use number of nodes as exponent
        '''
        N = 0
        temp = head
        while temp:
            N += 1
            temp = temp.next
        
        ans = 0
        temp = head
        while temp:
            ans += temp.val << N-1
            temp = temp.next
            N -= 1
        return ans

class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        '''
        we also don't need to find the number of nodes
        treat this like building new int
        multiply by base and ad next val
        '''
        ans = head.val
        while head.next:
            #move over one position
            ans <<= 1
            #set first bit position
            ans |= head.next.val
            head = head.next
        return ans

##################################
# 349. Intersection of Two Arrays
# 07DEC21
#################################
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        '''
        hash both and check of one is in the other
        '''
        nums1 = set(nums1)
        nums2 = set(nums2)
        
        ans = []
        for num in nums1:
            if num in nums2:
                ans.append(num)
        
        return ans

##############################
# 563. Binary Tree Tilt
# 07DEC21
##############################
class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        '''
        we want to return the sum of all a nodes tilt
        we define tilt as the  abs diff between the the sum of left and the sum of right
        if a node has no left child, the sum of left subtree node is zero
        same for the right
        for a node, we want to pass in left subtree sums and right subtree sums
        '''
        def sumTree(node):
            if not node:
                return 0
            left = sumTree(node.left)
            right = sumTree(node.right)
            return left + right + node.val
        
        
        self.tilts = 0
        def sumTilt(node):
            if not node:
                return 0
            left_sum = sumTree(node.left)
            right_sum = sumTree(node.right)
            tilt = abs(left_sum -right_sum)
            self.tilts += tilt
            sumTilt(node.left)
            sumTilt(node.right)
        
        sumTilt(root)
        return self.tilts

class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        '''
        we want to return the sum of all a nodes tilt
        we define tilt as the  abs diff between the the sum of left and the sum of right
        if a node has no left child, the sum of left subtree node is zero
        same for the right
        for a node, we want to pass in left subtree sums and right subtree sums
        '''
        self.tilts = 0
        def sumTree(node):
            if not node:
                return 0
            left = sumTree(node.left)
            right = sumTree(node.right)
            tilt = abs(left - right)
            self.tilts += tilt
            return left + right + node.val
        
        sumTree(root)
        return self.tilts
            
class Solution:
    def findTilt(self, root: Optional[TreeNode]) -> int:
        '''
        we want to return the sum of all a nodes tilt
        we define tilt as the  abs diff between the the sum of left and the sum of right
        if a node has no left child, the sum of left subtree node is zero
        same for the right
        for a node, we want to pass in left subtree sums and right subtree sums
        '''
        def sumTree(node,tiltsSoFar):
            if not node:
                return [0,0]
            left = sumTree(node.left,tiltsSoFar)
            right = sumTree(node.right,tiltsSoFar)
            tilt = abs(left[0] - right[0]) + left[1] + right[1]
            tiltsSoFar += tilt
            return [left[0] + right[0] + node.val,tiltsSoFar]
        
        return sumTree(root,0)[1]

#############################
# 364. Nested List Weight Sum II
# 08DEC21
############################
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution:
    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
        '''
        i first need to find the depths of the integers
        the find the max
        then find its weight
        then take dot product
        i can use dfs to flatten the nested list
        append to array (integer,depth)
        '''
        self.int_depth_array = []
        def dfs(nestedList,depth):
            for nested in nestedList:
                if nested.isInteger():
                    self.int_depth_array.append([nested.getInteger(),depth+1])
                else:
                    dfs(nested.getList(),depth+1)
                    
        dfs(nestedList,0)
        #now find max depth
        max_depth = 0
        for num,depth in self.int_depth_array:
            max_depth = max(max_depth,depth)
        ans = 0
        for num,depth in self.int_depth_array:
            ans += (max_depth - depth + 1)*num
        return ans
        
#another dfs
class Solution:
    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
        '''
        another way, just find max depth and the just re traverse
        
        '''
        #find max
        def find_max(nestedList):
            max_depth = 1
            for nested in nestedList:
                if not nested.isInteger():
                    max_depth = max(max_depth, 1+ find_max(nested.getList()))
            return max_depth
        
        def dfs(nestedList,depth,max_depth):
            weights = 0
            for nested in nestedList:
                if nested.isInteger():
                    weights += nested.getInteger()*(max_depth - depth + 1)
                else:
                    weights += dfs(nested.getList(),depth+1,max_depth)
            return weights
        
        max_depth = find_max(nestedList)
        #remember we start at depth of 1
        return dfs(nestedList, 1,max_depth)

#single pass dfs, using class
class WeightedSumTriplet(object):
    #note i could have used an integer array just fine
    def __init__(self):
        self.maxDepth = 0
        self.sumOfElements = 0
        self.sumOfProducts = 0
    
    def dfs(self,nestedList,depth):
        for nl in nestedList:
            #if its an int, record and increment
            if nl.isInteger():
                self.sumOfElements += nl.getInteger()
                self.sumOfProducts += nl.getInteger()*depth
                self.maxDepth = max(self.maxDepth,depth)
            #otherwise recurse
            else:
                #return new triplet
                newTriplet = WeightedSumTriplet()
                newTriplet.dfs(nl.getList(),depth+1)
                self.sumOfElements += newTriplet.sumOfElements
                self.sumOfProducts += newTriplet.sumOfProducts
                self.maxDepth = max(self.maxDepth, newTriplet.maxDepth)
                
   
class Solution:
    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
        '''
        if there were  a way to do this one time, rather, find the sum of weights
        where we only need to use the maxdepth at the end
        turns out the final derivation is:
        (maxDepth + 1)*sumOfElements - sumOfProducts
        which means we only need to keep track of sumof Elements and sum of products
        sumofproducts being the the element in the list times its depth!
        '''
        Triplet = WeightedSumTriplet()
        Triplet.dfs(nestedList,1)
        return (Triplet.maxDepth + 1)*(Triplet.sumOfElements) - (Triplet.sumOfProducts)

class Solution:
    def depthSumInverse(self, nestedList: List[NestedInteger]) -> int:
        '''
        we also could have use bfs, 
        keep track of sum of elements, sum of products, and maxdepth
        again we just use (maxDepth + 1)*sumOfElements - sumOfProducts
        '''
        q = deque([])
        for nl in nestedList:
            q.append(nl)
            
        curr_depth = 1
        max_depth = 0
        sumOfElements = 0
        sumOfProducts = 0
        
        while q:
            size = len(q)
            max_depth = max(max_depth,curr_depth)
            
            #pop each thin
            for i in range(size):
                curr = q.popleft()
                #if its an interger update
                if curr.isInteger():
                    sumOfElements += curr.getInteger()
                    sumOfProducts += curr.getInteger()*curr_depth
                #otherwise q up
                else:
                    for neigh in curr.getList():
                        q.append(neigh)            
            #done with this level,go down
            curr_depth += 1
        
        return (max_depth+1)*sumOfElements - sumOfProducts

#############################
# 1306. Jump Game III
# 09DEC21
#############################
#bfs FTW
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        '''
        bfs from start and check if i can hit an element zero
        '''
        
        N = len(arr)
        visited = set()
        #start off q
        q = deque([start])
        
        while q:
            curr = q.popleft()
            visited.add(curr)
            #hit target
            if arr[curr] == 0:
                return True
            for neigh in [curr+arr[curr],curr-arr[curr]]:
                #in bounds
                if 0 <= neigh < N:
                    if neigh not in visited:
                        q.append(neigh)
        
        return False

#sorta?
class Solution:
    def canReach(self, arr: List[int], start: int) -> bool:
        '''
        we also could use dfs
        but this time we can save on space by making the node negative
        '''
        N = len(arr)
        def dfs(idx):
            #check
            if arr[idx] == 0:
                return True
            arr[idx] = -arr[idx]
            #check for neighbors
            for neigh in [idx+arr[idx],idx-arr[idx]]:
                if 0 <= neigh < N:
                    if arr[neigh] >= 0:
                        if dfs(neigh):
                            return True
                        else:
                            return False
            return False 
        
        return dfs(0)

class Solution:
    def canReach(self, A, cur):
        if cur < 0 or cur >= len(A) or A[cur] < 0: return False
        A[cur] *= -1
        return A[cur] == 0 or self.canReach(A, cur + A[cur]) or self.canReach(A, cur - A[cur])

#################################
# 790. Domino and Tromino Tiling
# 10NOV21
#################################
#well this was fail...
class Solution:
    def numTilings(self, n: int) -> int:
        '''
        we have two shapes, domino and tromino, we can rotate, but does it really matter
        if we are only trying to figure out the number of ways
        we want to tmake a 2*n board
        n is from 1 to 1000 inclusive
        recusrive?
        for n = 1,
        theres only one way
        for n  == 2
        there are two ways
        for n = 3, there are 5 ways
        if i could reduce 3 to 1 or 2, 
        using dom only
        f(3) = f(1) + f(2)
        
        if i used trominoes
        f(5) = 2
        
        so f(5) = f(1) + f(2) + f(5) using tromnios
        
        '''
        return 

class Solution:
    def numTilings(self, n: int) -> int:
        '''
        notice that for some values of n, we can derive a board placement usint a previous n-k
        this might give a fully covered board, but it could also give a partially covered board
        to generate all board positions, we need to look back at previous fully board positions and partially board positions
        
        we can define f(k) number of way to fully cover board of width k
        we can define p(k) number of wayt so partially cover board with width k
        
        we can determine the number of ways to fully or partially tile a boad with width k, by looking at the number of ways to arrive at f(k) or p(k) by placing additional dominos or trominos
        
        from f(k-1) we can add 1 vertical domino for each tiliing with width of k-1
        from f(k-2) we can add 2 horiontal domnios for each tiling
            note we don't need to add 2 vertical dominoes since f(k-1) will cover that case
            
        from p(k-1) we can add an L shaped tromino for each tiling in a partially covred board with a width of k-1
            we will multiply by p(k-1) by 2, because for any partially covered tiling, there will be a horizontally symmetrical tiling of it, could place tromino upside down or rightside up
            
        summing ways to reach f(k) gives us
            f(k) = f(k-1) * f(k-2) + 2*p(k-1)
            
        how about p(k)
        think about ways to get to p(4) or partially covered 4
        
        this is FUCKING IMPORTANT:
            Take a pen and start drawing scenarios that contribute to p(4)p(4) (this is a good technique to aid critical thinking during an interview). Start by drawing p(4)p(4), remember p(4)p(4) is a board of width 4 with the first 3 columns fully covered and the last column half covered. Now, try removing a domino or a tromino to find which scenarios contribute to p(4)p(4). Notice that p(k)p(k) can come from the below scenarios:

        
        adding a tromino to a fully covered board of width k-2 (i.f f(k-2))
        adding horizontal domino to a partially covered board widwith k-1 p(k-1)
        p(k) = p(k-1) + f(k-2)
        
        algo:
            1. first dervie base cases for f(1), f(2), p(2):
                f(1) = 1
                f(2) = 2
                p(2) = 1
            2. define the following:
                f(k): The number of ways to fully cover a board of width k
                p(k): The number of ways to partially cover a board of width k
            3. recurse
            4. don't forget to cache
        '''
        mod = 10**9 + 7
        f_memo = {}
        p_memo = {}
        
        def p(n):
            if n == 2:
                return 1
            if n in p_memo:
                return p_memo[n]
            res =  (p(n-1) + f(n-2)) % mod
            p_memo[n] = res
            return res
        
        def f(n):
            if n <= 2:
                return n
            if n in f_memo:
                return f_memo[n]
            res = (f(n-1) + f(n-2) + 2*p(n-1)) % mod
            f_memo[n] = res
            return res
        
        return f(n)
        
#bottom up dp
class Solution:
    def numTilings(self, n: int) -> int:
        '''
        we can just translte the memozied version to bottom up dp
        '''
        if n <= 2:
            return n
        mod = 10**9 + 7
        N = n
        
        f_dp = [0]*(N+1)
        p_dp = [0]*(N+1)
        
        #fill in base cases
        p_dp[2] = 1
        f_dp[1] = 1
        f_dp[2] = 2
        
        for i in range(3,N+1):
            f_dp[i] = (f_dp[i-1] + f_dp[i-2] +2*p_dp[i-1]) % mod
            p_dp[i] = (p_dp[i-1] + f_dp[i-2]) % mod
        
        return f_dp[n]

class Solution:
    def numTilings(self, n: int) -> int:
        '''
        we can save on space by only keeping track of the actual previous valyes
        we only ever need f(n-1), f(n-2) and p(n-1)
        
        f_curr represents f(k-1)
        p_curr represents p(k-1)
        start off with f_curr being 2
        and p_curr being 1
        
        state transitions are:
            fCurrent = fCurrent + fPrevious + 2 * pCurrent
            pCurrent = pCurrent + fPrevious
            fPrevious = fCurrent (use the value of fCurrent before its update in the first step)
        
        '''
        if n <= 2:
            return n
        
        mod = 10**9 + 7
        
        f_curr = 2
        f_prev = 1
        p_curr = 1
        
        for k in range(3,n+1):
            temp = f_curr
            #udates f and pr
            f_curr = (f_curr + f_prev + 2*p_curr) % mod
            p_curr = (p_curr + f_prev) % mod
            f_prev = temp
        return f_curr

#note there are few matrix exponential equations, just revist this
class Solution:
    def numTilings(self, n: int) -> int:
        '''
        we can use the matrix exponential after dervinf the transition functions
        recall our functions:
        
        f(k)=f(k−1)+f(k−2)+2∗p(k−1)
        p(k) = p(k-1) + f(k-2)p(k)=p(k−1)+f(k−2)
        
        we can right this as matrix multiplacation
        
        [f(k),f(k-1),p(k)] = [[1,1,2],[1,0,0],[0,1,1]] dot [f(k-1),f(k-2),p(k-1)]
        where the just exponential the matrix k-2 times
        '''
        self.mod = 10**9 + 7
        self.mat = [[1,1,2],[1,0,0],[0,1,1]]
        self.size = 3
        
        def matrix_product(m1, m2):  
            """Return product of 2 square matrices."""
            # Result matrix `ans` will also be a square matrix with same dimensions.
            ans = [[0] * self.size for _ in range(self.size)]  
            for row in range(self.size):
                for col in range(self.size):
                    cur_sum = 0
                    for k in range(self.size):
                        cur_sum += (m1[row][k] * m2[k][col]) % self.mod
                    ans[row][col] = cur_sum
            return ans
        
        def matrix_expo(n):  
            """Perform matrix multiplication n times."""
            cur = self.mat
            for _ in range(1, n):
                cur = matrix_product(cur, self.mat)
            # The answer will be cur[0][0] * f(2) + cur[0][1] * f(1) + cur[0][2] * p(2)
            return (cur[0][0] * 2 + cur[0][1] * 1 + cur[0][2] * 1) % self.mod
        
        # Handle base cases
        if n <= 2:
            return n  
        
        return matrix_expo(n - 2)

#fast matrix exponentiation
class Solution:
    def numTilings(self, n: int) -> int:
        MOD = 1_000_000_007
        SQ_MATRIX = [[1, 1, 2], [1, 0, 0], [0, 1, 1]]  # Initialize square matrix
        SIZE = 3  # Width/Length of square matrix

        def matrix_product(m1, m2):  
            """Return product of 2 square matrices."""
            nonlocal MOD, SIZE
            # Result matrix `ans` will also be a square matrix with same dimension
            ans = [[0] * SIZE for _ in range(SIZE)]  
            for row in range(SIZE):
                for col in range(SIZE):
                    cur_sum = 0
                    for k in range(SIZE):
                        cur_sum = (cur_sum + m1[row][k] * m2[k][col]) % MOD
                    ans[row][col] = cur_sum
            return ans

        @cache  
        def matrix_expo(n):
            nonlocal SQ_MATRIX
            if n == 1:  # base case
                return SQ_MATRIX
            elif n % 2:  # If `n` is odd
                return matrix_product(matrix_expo(n - 1), SQ_MATRIX)
            else:  # If `n` is even
                return matrix_product(matrix_expo(n // 2), matrix_expo(n // 2))

        if n <= 2:
            return n

        # The answer will be cur[0][0] * f(2) + cur[0][1] * f(1) + cur[0][2] * p(2)
        ans = matrix_expo(n - 2)[0]
        return (ans[0] * 2 + ans[1] * 1 + ans[2] * 1) % MOD

#################################
# 878. Nth Magical Number
# 11DEC21
##################################
#TLE
class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        '''
        brute force would be to start with 1, and check if divisible by a or b
        if its mark, then return if nth is the nth
        '''
        curr = 1
        rank = 0
        
        while curr <= 10**9:
            if (curr % a == 0) or (curr % b == 0):
                rank += 1
                #print(curr,rank)
                if rank == n:
                    return curr
            curr += 1

#another brute force
class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        '''
        brute force would be to simply iterate from min(A,B) until we find B magical numbers
        
        '''
        mod = 10**9 + 7
        start = min(a,b)
        while n > 0:
            if (start % a == 0) or (start % b == 0):
                n -= 1
            start += 1
        return (start - 1) % mod


class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        '''
        brute force would be to start with 1, and check if divisible by a or b
        if its mark, then return if nth is the nth
        
        first realize the the pattern of magical numbers repeats itself
        example, if A = 6, and B = 10, the fist magial numbers are
        [6,10,12,18,20,24,30]
        with the same patten repeating for +30 times some multiple
        in general for a pattern there will be lcm(A,B) // A + lcm(A,B) // B - 1 numbers in a group
        algo:
            lets try to count the nth number mathematically
            if L is the LCM of A and B, and if X <= L is is magical, the X+L is also magical
            same thing for B
            
            if there M magical numbers, then we can represent:
            M = (L/A) + (L/B) - 1 magical numbers less than or equal to L
            instead of counting 1 at a time, we can count by M at a time
            
            Now supporse:
            N = M*q + r, with r < M, then the first L*q numbers countain M*q magical numbers,
            and within the next numbers (L*q + 1, L*q + 2.....)
            
            we can brute foce, for the next magical numbers less (L*q)
            if the r'th such magical number is Y, then the final anwer will be L*q + Y
            
        
        '''
        def gcd(x,y):
            small = x if x > y else y
            gcd = 1
            for i in range(1,small+1):
                if (x % i == 0) and (y % i == 0):
                    gcd = i
            return gcd
        
        mod = 10**9 + 7
        #find GCD
        gcd = gcd(a,b)
        #using GCD find LCM
        lcm = (a // gcd)*b
        #using inclusion/exlucion find the number of magical numbers (the length of the pattern)
        m = (lcm//A) + (lcm//b) - 1
        #find number of timmes m goes into n
        q,r = divmod(n,m)
        
        #if it goes evenly, then its just the end
        if r == 0:
            return q*L % mod
        
        #otherwise we need to find
        heads = [a,b]
        for _ in range(r-1):
            if heads[0] <= heads[1]:
                heads[0] += a
            else:
                heads[1] += b
        
        return (q*lcm + min(heads)) % mod

#better way
#https://leetcode.com/problems/nth-magical-number/discuss/1622665/Python-2-solutions%3A-find-patternbinary-search
class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        '''
        brute force would be to simply iterate from min(A,B) until we find B magical numbers
        
        '''
        def gcd(x,y):
            small = x if x > y else y
            gcd = 1
            for i in range(1,small+1):
                if (x % i == 0) and (y % i == 0):
                    gcd = i
            return gcd
        #find lcm of a and b
        mod = 10**9 + 7
        lcm = a*b // gcd(a,b)
        #generate candidates
        candidates = []
        #start with multiples of A
        for i in range(1,lcm //a):
            candidates.append(i*a)
        for i in range(1,lcm//b+1):
            candidates.append(i*b)
        
        #sorte
        candidates = sorted(candidates)
        #find the length of the pattern
        m = len(candidates)
        #when returning we want to check how multiples of m we haegone
        return (candidates[(n-1) % m] + lcm*((n-1)//m)) % mod 

class Solution:
    def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
        '''
        notes on the binary search solution,
            lcm(a,b) = a*b // gcd(a,b)
            the initial right pointer is N*min(a,b)
            we want to find the number of magical numbers we have before our x
            we define a functino: f(x), which is th number of magical numbers <= x
            by inclusion/exclusion, or just rule of double couting
            f(x) = x//a + x//b - x//lcm
            why the last part? we need to make sure we don't double count
            then we just check if we have numbers gretar than n
        '''
        def gcd(x,y):
            small = x if x > y else y
            gcd = 1
            for i in range(1,small+1):
                if (x % i == 0) and (y % i == 0):
                    gcd = i
            return gcd
        
        mod = 10**9 + 7
        lcm = a*b // gcd(a,b)
        
        left = 0
        right = n*min(a,b)
        
        while left < right:
            mid = left + (right - left) // 2
            #if i don't have enough numbers, extend the range
            if mid//a + mid//b - mid//lcm < n:
                left = mid + 1
            else:
                right = mid
        
        return left % mod

#also, recall the recursive definition for defining the gcd
def gcd(x,y):
    if x == 0:
        return y
    return gcd(y % x,x)

#################################
# 416. Partition Equal Subset Sum
# 12Dec21
#################################
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        '''
        if i find the sum of the array, the two paritions must be equal
        i can try building a subset until the subset becomes SUM // 2
        if it any point the subset grows beyond that i need to abandon
        it would help to sort the array first
        '''
        N = len(nums)
        SUM = sum(nums)
        
        #cannot be partitioned evenly
        if SUM % 2 == 1:
            return False
        
        #sort
        nums = sorted(nums)
        
        memo = {}
        
        def rec(i,curr_sum):
            #got to end
            if i == N:
                if curr_sum == SUM // 2:
                    return True
                else:
                    return False
            
            if curr_sum > SUM // 2:
                return False
            
            #got here
            if curr_sum == SUM // 2:
                return True
            
            if (i,curr_sum) in memo:
                return memo[(i,curr_sum)]
            take = rec(i+1,curr_sum+nums[i])
            no_take = rec(i+1,curr_sum)
            res = take or no_take
            memo[(i,curr_sum)] = res
            return res
        
        return rec(0,0)

#another recursion
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        '''
        if i find the sum of the array, the two paritions must be equal
        i can try building a subset until the subset becomes SUM // 2
        if it any point the subset grows beyond that i need to abandon
        it would help to sort the array first
        '''
        N = len(nums)
        SUM = sum(nums)
        
        #cannot be partitioned evenly
        if SUM % 2 == 1:
            return False
        
        #find the target sum
        subset_sum = SUM // 2
        memo = {}
        
        def rec(i,subset_sum):
            if subset_sum == 0:
                return True
            if i == 0 or subset_sum < 0:
                return False
            if (i,subset_sum) in memo:
                return memo[(i,subset_sum)]
            take = rec(i-1,subset_sum - nums[i-1])
            no_take = rec(i-1,subset_sum)
            res = take or no_take
            memo[(i,subset_sum)] = res
            return res
        
        return rec(N-1,subset_sum)
            

#bottom up dp
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        '''
        translating recusion to dp
        dp[i][j] returns true of all nums including nums up to index x and == j
        flase if i can't
        
        '''
        N = len(nums)
        SUM = sum(nums)
        
        #cannot be partitioned evenly
        if SUM % 2 == 1:
            return False
        
        #find the target sum
        subset_sum = SUM // 2
        
        dp = [[False]*(subset_sum+1) for _ in range(N+1)]
        
        for i in range(1,N+1):
            curr = nums[i-1]
            for j in range(subset_sum+1):
                #base case
                if j == 0:
                    dp[i][j] = True
                elif j < curr:
                    #carry over
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j - curr]
        
        return dp[N][subset_sum]

############################
# 1446. Consecutive Characters
# 12DEC21
############################
class Solution:
    def maxPower(self, s: str) -> int:
        '''
        just count the streaks
        and keep the longest one
        '''
        max_streak = 1
        curr_streak = 1
        N = len(s)
        for i in range(1,N):
            if s[i-1] == s[i]:
                curr_streak += 1
                max_streak = max(max_streak,curr_streak)
            else:
                curr_streak = 1
        
        return max_streak

############################
# 13Dec21
# 263. Ugly Number
############################
class Solution:
    def isUgly(self, n: int) -> bool:
        '''
        an ugly numebr is a positive integer whose prime factors are limited to 2,3, and 5
        well if a number if less than 0, its not ugly
        1 is an edge cass as well
        carefull for overflow
        keep reuding to 1 until we can't
        one we've gone as far as we can check if it is 1
        '''
        if n <= 0:
            return False
        if n == 1:
            return True
        for x in [2, 3, 5]:
            while n % x == 0:
                n = n / x
        return n == 1
        
################################
# 938. Range Sum of BST
# 14DEC21
################################
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        '''
        traverse all nodes in tree and increment sum
        '''
        self.sum = 0
        
        def dfs(node,low,high):
            if not node:
                return
            if low <= node.val <= high:
                self.sum += node.val
            dfs(node.left,low,high)
            dfs(node.right,low,high)
        
        dfs(root,low,high)
        return self.sum

class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        '''
        without the use of global variable
        
        '''
        def dfs(root,low,high):
            if not root:
                return 0
            total = 0
            if low <= root.val <= high:
                total += root.val
            if root.val > low:
                total += dfs(root.left,low,high)
            if root.val < high:
                total += dfs(root.right,low,high)
            return total
        
        return dfs(root,low,high)

class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        '''
        iterative stack
        '''
        ans = 0
        stack = [root]
        
        while stack:
            curr = stack.pop()
            if not curr:
                continue
            if low <= curr.val <= high:
                ans += curr.val
            if curr.left:
                stack.append(curr.left)
            if curr.right:
                stack.append(curr.right)
        
        return ans

#########################
# 293. Flip Game
# 14DEC21
#########################
class Solution:
    def generatePossibleNextMoves(self, s: str) -> List[str]:
        ans = []
        for i in range(len(s)-1):
            if s[i:i+2] == "++": ans.append(s[:i]+"--"+s[i+2:])
        return ans 
    
#############################
# 147. Insertion Sort List
# 15DEC21
#############################
#pull apart and sort
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        for insertion sort, in an array, we traverse each element and try to place the element in the correst position
        two parts, a sorted part, and an unsorted part
        however we need to scan the sorted part every time for an insertion
        first approach, pull all nums into an an array and code insertion sorts
        '''
        nums = []
        curr = head
        while curr:
            nums.append(curr.val)
            curr = curr.next
        
        #now do insertion sort
        for i in range(len(nums)):
            temp = nums[i]
            j = i
            while j > 0 and nums[j-1] > temp:
                nums[j] = nums[j-1]
                j -= 1
            nums[j] = temp
        
        dummy = ListNode()
        curr = dummy
        for num in nums:
            curr.next = ListNode(val=num)
            curr = curr.next
        return dummy.next

#swapping values
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        to do insertion sort in a linked list
        we need to use a pair of pointers, prev and next
        '''
        cur = head
        while cur:
            temp = head #alway start from beginning
            while temp != cur:
                if temp.val > cur.val:
                    #swap
                    temp.val,cur.val = cur.val, temp.val
                #otheriwse adanve temp
                temp = temp.next
            #always advance
            cur = cur.next
        
        return head

#swapping nodes
#https://leetcode.com/problems/insertion-sort-list/discuss/1629811/C%2B%2BPythonJava-2-Simple-Solution-w-Explanation-or-Swap-values-%2B-Pointer-Manipulation-Approaches
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        by using the swap values idea, we head to iterate from head to curr, every time
        usually insertion sort works by finding the current elements place to be in the sorted arrayr
        recall in the last, we always had to shit adjacent elements up or down
        
        algo: 
        we can find the correct position of cur by iterating in sorted portions of list until we find a node which has value less than cur
        then remove curr from its original positison and insert
        1. update point of cur to j, which is the position before which cur needs to be inserted
        2. update enxt point nodes of j to cur
            this is because cur is now inserted before j and this prevs next ndoe should point at cur
        3. update enxt pointer of previous node to enxt of oc
        4. the current node is now placed at its proper position and all pointers have been updated
        '''
        #starting off pointers
        dummy = ListNode(val= -1, next = head)
        cur_prev = head
        cur = head.next
        
        while cur:
            #state arouod cur
            j_prev = dummy
            j = dummy.next
            cur_next = cur.next
            if cur.val > cur_prev.val:
                cur_prev = cur
            else:
                while j.val < cur.val:
                    j_prev = j
                    j = j.next
                cur.next = j
                j_prev.next = cur
                cur_prev.next = cur_next
            cur = cur_next
        return dummy.next

#official solution
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        official solution
        move prev pointer to the corret positionon
        than reassign pointers
        
        '''
        #starting off pointers
        dummy = ListNode()
        curr = head

        while curr:
            # At each iteration, we insert an element into the resulting list.
            prev = dummy

            # find the position to insert the current node
            while prev.next and prev.next.val < curr.val:
                prev = prev.next

            next = curr.next
            # insert the current node to the new list
            curr.next = prev.next
            prev.next = curr

            # moving on to the next iteration
            curr = next

        return dummy.next

##################################
# 624. Maximum Distance in Arrays
# 15DEC21
##################################
#TLE brute force
class Solution:
    def maxDistance(self, arrays: List[List[int]]) -> int:
        '''
        find mins  and  maxs
        crap, note that the min and max cannot come from the same array
        note, that there will be at most 10**5 integers in the array
        what if i mapp each num to its index array
        
        brute force would be to check all i,j where i != j
        '''
        ans = 0
        N = len(arrays)
        for i in range(N):
            for j in range(N):
                if i != j:
                    ans = max(ans,abs(arrays[i][0]-arrays[j][-1]))
        
        return ans

#better brute force
class Solution:
    def maxDistance(self, arrays: List[List[int]]) -> int:
        '''
        find mins  and  maxs
        crap, note that the min and max cannot come from the same array
        note, that there will be at most 10**5 integers in the array
        what if i mapp each num to its index array
        
        brute force would be to check all i,j where i != j
        
        we can do better, since there only i*j over two pairs
        '''
        ans = 0
        N = len(arrays)
        for i in range(N):
            for j in range(i+1,N):
                #swapping i and j
                ans = max(ans,abs(arrays[i][0]-arrays[j][-1]))
                ans = max(ans,abs(arrays[j][0]-arrays[i][-1]))
        
        return ans

class Solution:
    def maxDistance(self, arrays: List[List[int]]) -> int:
        '''
        we don't need to compare every i,j array if i != j
        really what we are asking is for the largest gap between any two arrays in arrrays
        we can keep track of a min so far and a max so far
        update the largest abs diff, but also update the min so far and max so far
        we initalize with the starting array at index 0 and check every array after that
        
        
        digress:
            we don't need to another res update after the last 1?
            because if we did, the curr_min and curr_max would belong to the same array
        '''
        res = 0
        curr_min = arrays[0][0]
        curr_max = arrays[0][-1]
        
        for i in range(1,len(arrays)):
            #first find the curr diff, abs(min_val - max of curr array)
            diff1 = abs(arrays[i][-1] - curr_min)
            #second diff, abs(max_val - min array)
            diff2 = abs(arrays[i][0] - curr_max)
            #update answer
            res = max(res,max(diff1,diff2))
            #update currmin and currmax
            curr_min = min(curr_min, arrays[i][0])
            curr_max = max(curr_max, arrays[i][-1])
    
        return res
        
############################
# 310. Minimum Height Trees
# 16DEC21
############################
#TLE, max highet of N-ary tree for each node
#find MHT for each rooted node
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        notes:
            any connected graph without a cycle is a tree
            we are given an undirected graph
            with n nodes, and an edge list
        we can used any node we want as the root,
        return the min height of all trees
        
        brute force would be to dfs each tree using 0 to n-1 as the root
        get the max depth for each traversal and return the min
        '''
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        def dfs(node,seen,depth):
            #note, depth will start off at 0, but keep track of min
            #keep track of global max depth
            self.max_depth = max(self.max_depth,depth)
            if node not in seen:
                seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen,depth+1)

        '''            
        self.max_depth = 0
        dfs(0,set(),0)
        print(self.max_depth)
        '''
        temp = []
        for i in range(n):
            self.max_depth = 0
            dfs(i,set(),0)
            temp.append([i,self.max_depth])
            
        #find min height
        min_height = min([h for n,h in temp])
        #find nodes having min height
        ans = []
        for n,h in temp:
            if h == min_height:
                ans.append(n)
        
        return ans

#BFS layer by layer
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        note that the brute force gives TLE,
        very similat to top sort on course scheduel i and ii
        lets define the distance between two numbers as the number of edges that connects the two nodes
        because this is a tree and there are no cycles, there must only be one path between any two nodes
        lets define the height of a tre as the maximum distance bettween the root and all it leaf nodes
        reduction:
            the problem is finding out the nodes that are overall close to all other nodes
        
        this way we are actuall looking for centrods
        IMPORTANT assertion: for a tree like graph, the number of centroids is no more than 2
        if nodes form a chain there are two cases:
            if the number of nodes is even, there are only two centroids
            if number of nodes is odd, there is only one
        
        proof by contradiction:
            for three nodes, if there were three centroids, this would HAVE to form a cycle
            there cannot be more than two centroids
        
        algo:
            reudce the problem:
                look for all centroid nodes in a tree like grpah, which is bounded by two
            idea: trim out leaf nodes layer by layer until we reach the core of the graph
            really its top sort from centroids
            in this case, we trim out leaf nodes which are farther away from the centroids
            at each step, the nodes we trim out are closer to the centroids than the nodes in the previous step
        BFS strategy:
            * make adj lists
            * create queue which would be used to hold all leaf nodes
            * at beginning, put all current leage nodes into q
            * loop until there is only two nodes left in graph
            * at each iteration
                * remove current leaf nodes from q
                * while removing ndoes, we also drop edges that link the  nodes
                * as a consequence, some of the non lead nodes would become leaf ndoes
                * and these ndoes would be trimmed out in next iterations
            * terminate when there are no more than ndoes left in the graph
        '''
        if not edges:
            return [0]
        seen = [False]*n
        adj_list = defaultdict(set)
        
        for u,v in edges:
            adj_list[u].add(v)
            adj_list[v].add(u)
        
        leaves = []
        new_leaves = []
        in_degree = []
        
        #find leaves in degree for all nods
        for i in range(n):
            if len(adj_list[i]) == 1:
                leaves.append(i)
            in_degree.append(len(adj_list[i]))
            
        while n > 2:
            for leaf in leaves:
                for neigh in adj_list[leaf]:
                    #reduce indegre
                    in_degree[neigh] -= 1
                    #get ready for next layr
                    if in_degree[neigh] == 1:
                        new_leaves.append(neigh)
            
            #remove leave
            n -= len(leaves)
            leaves = new_leaves[:]
            new_leaves = []
        
        return leaves

class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        another way to dfs twice to find the longest path in given graph
        there can only be a max of 2 MHTS, and the node which uyeild the min height tree  are the mid points of a longes path in the given graph
        for an MHT, we want the minimum distance to all leaves from a root
        so its obvious that what we are actually tring to find  is a node that balances the amx distance from itselt to its extreme nodes
        choosing the middle nodes of the longest path will ensure the tree has its two longes branches
            if 1 root, L//2
            if 2 root, L//2 - 1
        
        we can abritrailty dfs from a node to find its first leaf, call it node1
        then dfs from node1 as far as we can, node2
        node1 to node2 is guranteed to be the longest path
        for this path, the middle nodes minimuze the depth of three
        '''
        graph = defaultdict(set)
        seen = [False]*n
        
        for u,v in edges:
            graph[u].add(v)
            graph[v].add(u)
        
        def dfs(i):
            #this returns the path
            if seen[i]:
                return []
            seen[i] = True
            longest_path = []
            for neigh in graph[i]:
                longest_path.append(dfs(neigh))
            #find the longest one
            longest_path = max(longest_path,key = len, default=[])
            #don't forget to include the currnet node
            longest_path += [i]
            #backtrak
            seen[i] = False
            return longest_path
        
        #first dfs to find the first leaf
        node1 = dfs(0)[0]
        #second one to recrod the ptath
        path = dfs(node1)
        return set([path[len(path)//2], path[(len(path)-1)//2]])

#######################
# 221. Maximal Square
# 17DEC21
#######################
#welp this almsot works
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        '''
        i can probably use dynamic programming for this problem
        a single cell is trivially a square
        so for the first row and col, is there's a 1, its area is 1
        now if im at (i,j), look at i-1, and j-1, is they were ones, then take the its max in
        put 1
        dp(i,j) = if 1, check dp(i-1,j) and dp(i,j-1) and dp(i-1,j-1) is one
        dp(i,j) answers the question, the largest square that can be made with lower right corner ending at (i,j)
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        dp = [[0]*cols for _ in range(rows)]
        
        #first base cases for first row and cols
        for i in range(rows):
            if matrix[i][0] == '1':
                dp[i][0] = 1
        #cols
        for j in range(cols):
            if matrix[0][j] == '1':
                dp[0][j] = 1
        
        ans = 0
        for i in range(1,rows):
            for j in range(1,cols):
                #if this is a one, see if i can make q sqaure
                if matrix[i][j] == '1':
                    dp[i][j] = min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1]) + 1
                #can't make it, well its at elast 1 from here
                else:
                    dp[i][j] = 1
                
                ans = max(ans,dp[i][j])
        
        return ans*ans

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        '''
        i can probably use dynamic programming for this problem
        a single cell is trivially a square
        so for the first row and col, is there's a 1, its area is 1
        now if im at (i,j), look at i-1, and j-1, is they were ones, then take the its max in
        put 1
        dp(i,j) = if 1, check dp(i-1,j) and dp(i,j-1) and dp(i-1,j-1) is one
        dp(i,j) answers the question, the largest square that can be made with lower right corner ending at (i,j)
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        dp = [[0]*(cols+1) for _ in range(rows+1)]
        
        
        ans = 0
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                #if this is a one, see if i can make q sqaure
                if matrix[i-1][j-1] == '1':
                    dp[i][j] = min(min(dp[i-1][j],dp[i][j-1]),dp[i-1][j-1]) + 1
                
                    ans = max(ans,dp[i][j])
        
        return ans*ans

#lets go over a few of the solutions 
#https://leetcode.com/problems/maximal-square/discuss/1632376/C%2B%2BPython-6-Simple-Solution-w-Explanation-or-Optimizations-from-Brute-Force-to-DP
#brute force
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        '''
        brute force, consider all squares with all possible side lenghts
        for each cell and for each side length, check all 1s, then just return sum
        or rather return the product of sideLen
        '''
        def isSquare(row,col,size):
            #coll way of using all
            return all(all(matrix[i][j] == '1' for j in range(col,col+size)) for i in range(row,row+size))
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        for size in range(min(rows,cols),0,-1):
            for row in range(rows-size+1):
                for col in range(cols-size+1):
                    if isSquare(row,col,size):
                        return size*size
        
        return 0

#better brute force, keep track of consecutive ones
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        '''
        we can optimize the brute force
        instead of checking each possible side length for a square starting at a given cell, we can optimize the process by starting from row of that cell and expaning the side length for that till it is possible
        continue this for below rows as well until consec ones > curr row number
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        ans = 0
        
        def get_max_square_len(row,col):
            all_ones_row = min(rows - row,cols - col)
            sq_len = 0
            i = j = 0
            while i < all_ones_row:
                j = 0
                while j < all_ones_row and matrix[i+row][j+col] != '0':
                    j += 1
                all_ones_row = j
                sq_len = min(all_ones_row,i := i + 1)
            
            return sq_len
        
        for row in range(rows):
            for col in range(cols):
                ans = max(ans, get_max_square_len(row,col))
        
        return ans*ans

#optimized, consecutives ones arrays
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        '''
        note in the brute force, we were iteratively calculating the maxinum consective ones multiple times
        we can precompute, and for each row, record the number of max consective ones
        The following solution uses ones matrix where ones[i][j] denotes number of consecutive ones to the right of the (i, j) cell.
        
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        ans = 0
        
        ones = [[0]*(cols+1) for _ in range(rows)]
        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                ones[i][j] = 1 + ones[i][j+1] if matrix[i][j] == '1' else 0
        
        def get_max_square_len(row, col):
            all_ones_row_len, sq_len, i, j = min(rows-row, cols-col), 0, 0, 0
            while i < all_ones_row_len:                
                all_ones_row_len = min(all_ones_row_len, ones[i+row][col])
                sq_len = min(all_ones_row_len, i := i + 1)
            return sq_len
        
        for row in range(rows):
            for col in range(cols):
                ans = max(ans, get_max_square_len(row, col))
        return ans * ans

#dp, using up m*n space
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        '''
        dp solution
        if we represent dp(i,j) as the largest square that can be formed with bottom right corner at (i,j)
        if matrix[i,j] = 1
        dp(i,j) = min(dp(i-1,j),dp(i-1,j-1),dp(i,j-1)) + 1
        we add one because we can make a new square
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        dp = [[0]*(cols+1) for _ in range(rows+1)]
        
        ans = 0
        for i in range(1, rows+1):
            for j in range(1,cols+1):
                if matrix[i-1][j-1] == '1':
                    dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) + 1
                    ans = max(ans,dp[i][j])
        
        return ans*ans

#saving on space, we only care about the curr row and previous row
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        '''
        dp solution
        if we represent dp(i,j) as the largest square that can be formed with bottom right corner at (i,j)
        if matrix[i,j] = 1
        dp(i,j) = min(dp(i-1,j),dp(i-1,j-1),dp(i,j-1)) + 1
        we add one because we can make a new square
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        curr = prev =  [0]*(cols+1)
        
        ans = 0
        for i in range(1, rows+1):
            for j in range(1,cols+1):
                if matrix[i-1][j-1] == '1':
                    curr[j] = min(prev[j],curr[j-1],prev[j-1]) + 1
                    ans = max(ans,curr[j])
            prev = curr
            curr = [0]*(cols+1)
        return ans*ans

#########################################
# 902. Numbers At Most N Given Digit Set
# 18DEC21
#########################################
#fuck...
#note backtracking has too big of input sizes to work
class Solution:
    def atMostNGivenDigitSet(self, digits: List[str], n: int) -> int:
        '''
        this is a back tracking problem
        the only issue is that i can keep taking as many digits, i can use digits[i] as many times as i want
        '''
        ans = []
        
        def backtrack(i,path):
            #if the path is greater than n, terminated
            if int("".join(path)) > n:
                return
            #anytime we have valid answer add it
            if int("".join(path)) <= n:
                ans.append(''.join(path))
                return
            
            for j in range(i,len(digits)):
                backtrack(j,path+[digits[j]])
        
        backtrack(0,[])
        return ans

class Solution:
    def atMostNGivenDigitSet(self, D, N):
        '''
        first, lets call a positive integer X, valid if X <= N and X only consits of digits from D
        now, say N has K digits
        if we write a vlid number with k digits (k < K), then there are len(D)**k possible numbers we could write
        since all are less than N
        now say we are to write a valid K digit number left to right, 
        N = 2345
        K = 4
        D = ['1','2'....'9']
        consider what happens when we write the first digit
        if the firt digit is less than the first digit of N, then we can continue to wrte any numbers after
        for a total of len(D)**(k-1)
        if we starrt with 1, we could get 1111 to 1999 from this prefix
        if the first digit we write is the same, then we requirer that the next digit we write is equal to or lower than the nexr digit N, in our example, N = 2345, then if we start with 2, the next digit we write must be 3 or less
        we can't write a larger digit because if we start wtih 3, then evne number 3000 > N
        
        algo:
            dp[i] be the number of ways to write a valid number if N becamse N[i]
            i.e N = 2345
            dp[0] be numbers at most 2345
            dp[1] be numbers at most 345
            dp[2] be numberts at most 45
            dp[3] be numbers at most 5
            
            rather dp[i] = (number of d in D with d < S[i])*(len(D)**(K-i-1)) + dp[i+1] if S[i] in D
        '''
        #get the number of digits in N
        S = str(N)
        K = len(S)
        dp = [0]*K + [1]
        #dp[i] = total number of valid integers is N was N[i:]
        for i in range(K-1,-1,-1):
            #now check for each digit in D
            for d in D:
                if d < S[i]:
                    #i can make len(D)**(k-1) digits for this i
                    #recal digits are in order
                    dp[i] += len(D)**(K-i-1)
                #if i can't carry it over
                elif d == S[i]:
                    #if this digit is equal, the next i+1 will not work because the digits are orderd
                    dp[i] += dp[i+1]
        '''
        trying to make numbes for each size len(D) - 1
        add this to the first entry in dp
        for i in range(1,K):
            print(i,len(D)**i)
        '''
        #making digits for N[i:] for i in range(len(N)) - 1
        digits = 0
        for i in range(1,K):
            digits += len(D)**i
        return dp[0] + digits

#another way, but i get the first approach
class Solution:
    def atMostNGivenDigitSet(self, D: List[str], N: int) -> int:
        '''
        we can use binary search onf this problem,
        recall that int eh first approach, a positive interge X is valid if X <= N, and only contains digits in D
        we let B = len(digits)
        there is a bijection between valid integers 
        examples:
            D = ['1','3','5','7']
            we could write numbers
            '1', '3', '5', '7', '11', '13', '15', '17', '31', ... 
            as a bijective base (B) numbers 
            1', '2', '3', '4', '11', '12', '13', '14', '21', ....
        our approach then becomes finding the largest valid integer and convert it into a bijectiv base -b
        from which it easy to find its rank (position in the seqeuence)
        becaue of the bijection, the rank of this element must by the number of valid integers
        examples:
        N = 64, using same D digits
        we have 1,33,....55,57
        which can be written as bijective base 4 numbers 1,2..33,34
        algo:
            1. convert N into the largest possible valid integer X
            2. convert X into bijective base b
            3. conver ther result into a decimal answer
            
            example D = ['2','4','6','8']
            
            if the firdt digit of N is in D, we write that digit and continue, example N = 25123
            if the first digits of N > min(D) then we write the largest possible number from D, less than the digits
            Example: n = 5123, the write 4888
            if first digit of N is < min(D) then we must subtract 1
            For example, if N = 123, we will write 88. If N = 4123, we will write 2888. And if N = 22123, we will write 8888. This is because "subtracting 1" from '', '4', '22' yields '', '2', '8' (can't go below 0).
            
        '''
        B = len(D) # bijective-base B
        S = str(N)
        K = len(S)
        A = []  #  The largest valid number in bijective-base-B.

        for c in S:
            if c in D:
                A.append(D.index(c) + 1)
            else:
                i = bisect.bisect(D, c)
                A.append(i)
                # i = 1 + (largest index j with c >= D[j], or -1 if impossible)
                if i == 0:
                    # subtract 1
                    for j in range(len(A) - 1, 0, -1):
                        if A[j]: break
                        A[j] += B
                        A[j-1] -= 1

                A.extend([B] * (K - len(A)))
                break

        ans = 0
        for x in A:
            ans = ans * B + x
        return ans

##########################
# 19DEC21
# 394. Decode String
##########################
class Solution:
    def decodeString(self, s: str) -> str:
        '''
        i can use a stack and work inside to tou
        we keep pushing onto a stack until we hit a closing bracket, 
        once we hit the close we can push back
        but we need to make sure we reverse when pushing back
        '''
        stack = []
        for char in s:
            #decode if closing
            if char == "]":
                #prepare
                decoded_string = ""
                while stack[-1] != '[':
                    decoded_string += stack.pop()
                #clear the opening
                stack.pop()
                #recall its a string, so we could get a string int lager than 9
                base = 1
                k = 0
                while stack and '0' <= stack[-1] <= '9':
                    k = k + (ord(stack.pop()) - ord('0'))*base
                    base *= 10
                
                string_len = len(decoded_string)
                #we need to push this bask k times, but in revers
                while k != 0:
                    for char2 in reversed(decoded_string):
                        stack.append(char2)
                    
                    k -= 1
            #otherwise append to stack
            else:
                stack.append(char)
        
        return "".join(stack)

class Solution:
    def decodeString(self, s: str) -> str:
        '''
        we can also solve this recursively
        start by buillding k ands tring and recusrively decode at each nested level
        then return the current decoding and recurse
        
        algo:
            build result while enxt char is lettter and buildg number while next char is digit
            ignore the next '[' 
            decode the current pattern k[decoded] and append to result
            return curr result
            
            base case:
                traverseed through all of s or the next char is ] , which prompts us to evaluate
        '''
        index = [0]
        def rec(s):
            result = ""
            while index[0] < len(s) and s[index[0]] != ']':
                if not ('0' <= s[index[0]] <= '9'):
                    result += s[index[0]]
                    index[0] += 1
                else:
                    k = 0
                    while index[0] < len(s) and '0' <= s[index[0]] <= '9':
                        k = k*10 + ord(s[index[0]]) - ord('0')
                        index[0] += 1
                    #ignore opening
                    index[0] += 1
                    #decode
                    decoded_string = rec(s)
                    #ignore enxt
                    index[0] += 1
                    #buuild k[decoded]
                    while k > 0:
                        result += decoded_string
                        k -= 1
            
            return result
        
        return rec(s)

####################################
# 1200. Minimum Absolute Difference
# 19DEC21
####################################
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        '''
        we want to find all pairs of elements with min abs diff of any two elements
        rather return list of all pairs [a,b]
        such that 
            a,b in arr
            a < b
            b - a == min abs different of any two elements in array
        
        sort the array
        the find the min difference
        the find elements that have that difference
        '''
        arr = sorted(arr)
        N = len(arr)
        absoluteMinDiff = float('inf')
        for i in range(1,N):
            absoluteMinDiff = min(absoluteMinDiff, abs(arr[i]-arr[i-1]))
        
        ans = []
        for i in range(1,N):
            if abs(arr[i] - arr[i-1]) == absoluteMinDiff:
                ans.append([arr[i-1],arr[i]])
        
        return ans

#counting sort
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        '''
        we can use counting sort, create items an array to store [min(arr),max(arr)]
        zero inclusive
        since the numbers couold be negative, we use an aux array of size 2*10^6
        so we need to build a bijection between the element value and the index in the aux array
        we can map the value to index where this element should be palced by adding a term.
        shift = 10^6, or rather the largest element in arr
        to sort the array, we can find the corresponding index of this intger index = value + shift
        and increase the vlaue at index in the aux array lie by 1
        note: the shift == -min(arr)
        to mapp num -> num + shigt
        this gives us the cout array,
        not the array is strictly increasing, so the items array, which was iniitally zero will contains ones
        we can traverse  the line again, and where line[index] != 0 this signifies the value index - shift
        question: how to collect and compare pairs of adjacent eleemnts
        we scan the lines array and check if a 1 is present at this index
        if a 1 is presnet, then the number lines[index] - shift is in the original array
        keep not of curr and prev 1, update if min diff gets larger
        algo;
            1. find the min and max of the array
            2. init aux array of size min - max + 1
            3. pass over arr, and increment num + shift in aux by 1
            4.  travere aux array and for each check for 1 and 0
        '''
        smallest = min(arr)
        largest = max(arr)
        shift = -smallest
        line = [0]*(largest - smallest + 1)
        #first pass, mapp nums to line
        for num in arr:
            line[num+shift] = 1
            
        min_diff_pair = largest - smallest
        prev = 0
        res = []
        
        for curr in range(1,largest + shift + 1):
            if line[curr] == 0:
                continue
            ##otherwise its a 1
            #first check if curr diff is the  min diff
            if curr - prev == min_diff_pair:
                res.append([prev - shift,curr-shift])
            #otherwise update min diff
            elif curr - prev < min_diff_pair:
                #reupdate the new res, because we found a smaller one
                res = [[prev - shift,curr-shift]]
                min_diff_pair = curr - prev
            #update prev
            prev = curr
        
        return res
        print(line)

####################
# 38. Count and Say
# 20DEC21
####################
class Solution:
    def countAndSay(self, n: int) -> str:
        '''
        base case is '1'
        for n = 4
        n = 1   '1' 
        n = 2   '11'
        n = 3   '21'
        n = 4   '1211'
        n = 5   '111221'
        n = 6   '312211'
        n = 7   '13112221'
        i'm given the recurrence already, so then lets start from the base case
        and build out way to n
        to generate the nth tearm just count and say n-1'term
        '''
        
        def countSayHelper(s):
            N = len(s)
            left = 0
            right  = 1
            res = ""
            while left < N:
                while right < N and s[left] == s[right]:
                    right += 1
                count = right - left
                res += str(count)
                res += str(s[left])
                left = right
                right += 1
            
            return res
        
        if n == 1:
            return '1'
        
        curr = '1'
        for i in range(n-1):
            ans = countSayHelper(curr)
            curr = ans
        return curr

########################
# 231. Power of Two
# 21DEC21
#######################
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        '''
        i can just divide by two and check i can reduce to 1
        also careful of negatives
        '''
        if n <= 0:
            return False
        while n % 2 == 0:
            n //= 2
        
        return n == 1

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        '''
        if its a power of two, there should only 1 in its bitset
        '''
        if n <= 0:
            return False
        ones = 0
        while n:
            ones += n & 1
            n >>= 1
        
        return ones == 1

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        '''
        recall the tricks x & (-x)
        twos complement, -x is the same as ~x + 1
        recall, if we have a number x, we get its complement (flipping all bits) 
        using ~x
        -x = ~x + 1
        x & (-x) clears all bits and sets right most bit of x to 1
        a number is a power of two if x & (-x) == x
        '''
        if n <= 0:
            return False
        
        return n & (-n) == n

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        '''
        brian kernighan, x & (x-1)
        subtracting 1 means decremting right most bit by 1
        and set all lower zero bits to 1
        for a power of two, it just has one bit
        so by clearing the right most bit and checking == 0
        n must be power of two
        '''
        if n <= 0:
            return False
        
        return n & (n-1) == 0

#################################
# 345. Reverse Vowels of a String
# 21DEC21
#################################
class Solution:
    def reverseVowels(self, s: str) -> str:
        '''
        swap only if both left and right pointers are vowels
        '''
        N = len(s)
        s = list(s)
        left = 0
        right = N - 1
        vowels = 'aeiouAEIOU'
        
        while left < right:
            #check for vowels
            if s[left] in vowels:
                while s[right] not in vowels:
                    right -= 1
                #swap
                s[left],s[right] = s[right],s[left]
                left += 1
                right -= 1
            elif s[right] in vowels:
                while s[left] not in vowels:
                    left += 1
                #swap
                s[left],s[right] = s[right],s[left]
                left += 1
                right -= 1
            else:
                left += 1
                right -= 1
        
        return "".join(s)

########################
# 143. Reorder List
# 22DEC21
########################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        '''
        cheeky way is to pull values from list and modify.val for each node
        then do it every other starting from the end
        then i can two pointers for each and reset the nodes
        '''
        vals = []
        temp = head
        while temp:
            vals.append(temp.val)
            temp = temp.next
            
        left = 0
        right = len(vals) - 1
        ptr = 0
        temp = head
        while temp:
            #take from beginning
            if ptr % 2 == 0:
                temp.val = vals[left]
                left += 1
            elif ptr % 2 == 1:
                temp.val = vals[right]
                right -= 1
            temp = temp.next
            ptr += 1
            

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        '''
        really if we just found the head of the middle node
        reversed the middle linked list
        then merged the two lists we would get the answer
        we can find the middle of the linked list using two pointers
        then reverse the seoncd part of the list
        then merge
        '''
        if not head:
            return 
        
        #find middle
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        #reverse second par of list, #slow is not pointing to head of middle
        prev = None
        curr = slow
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        
        #merged the two
        first = head
        second = prev
        while second.next:
            first.next, first = second, first.next
            second.next,second = first, second.next
            

##############################################
# 1153. String Transforms Into Another String
# 22DEC21
##############################################
class Solution:
    def canConvert(self, str1: str, str2: str) -> bool:
        '''
        the solution is kinda long winded so lets break this down in to parts
        1. One to One Mapping:
            if each char in str1 can map to str2, we can transfrom this with zero or more conversion
        2. One to Many mapping
            if chars in str1 are not different and one of the smae chars in str1 is mapped to another in str2
        3. Linked List
            if the mappings form a linked list, must be careful in the order we transform
            by there is no cycle in this mapping, so still possible
        4. Cylic Linked List
            we need to break the cycle mapping in str1, which any char not in str1 and str2
        5. Multiple Linked Lists
            we can break the cycle, so long as we have a unique char
        6. Cylic Linked List with 26 letters
            cannot do, beause we need ane extra char to brea the cycle
        7. Linked List with 26 Letters and One Loop
            say we have str1, containing all 26 unique lower case chars
            says str2, only contains 25, we can still convert
            why? two char from string 1 mapp to the sam char
            we adopt greey stretgy and conver yh to z
            the idea is to make these to chars (y and z) converted both to z
        inution:
            if str1 has 26 unique chars, it is still possible to convert str1 to str2 
            as long as str2 has less than 26 chars
            SO, if str1 has 26 uique chard and str2 does not, there will always be a way to transform str1 into str2
        
        
        '''
        if str1 == str2:
            return True
        
        conversion_mappings = dict()
        unique_characters_in_str2 = set()
        
        # Make sure that no character in str1 is mapped to multiple characters in str2.
        for letter1, letter2 in zip(str1, str2):
            if letter1 not in conversion_mappings:
                conversion_mappings[letter1] = letter2
                unique_characters_in_str2.add(letter2)
            elif conversion_mappings[letter1] != letter2:
                # letter1 maps to 2 different characters, so str1 cannot transform into str2.
                return False
        
        
        if len(unique_characters_in_str2) < 26:
            # No character in str1 maps to 2 or more different characters in str2 and there
            # is at least one temporary character that can be used to break any loops.
            return True
        
        # The conversion mapping forms one or more cycles and there are no temporary 
        # characters that we can use to break the loops, so str1 cannot transform into str2.
        return False

###########################
# 23May21
# 23DEC21
############################
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        '''
        we can used dfs to find a topologiclat sorted order
        we are given edges of the form [a,b], where b is a prereq to a
        so the edge is of the form a -> b, directed and unweighted
        the graph is cyclic, if there were a cycle the wouln't be a way to take all coures
        problem says it might not be possible, so it must be cyclic
        apporach 1, dfs
        
        consider all paths starting from A, once we've gone as far as we can go, we know we can at least start from this course
        intution:
            let S be a stack of courses
            dfs(node):
                for neigh in adjlist at node
                    dfs(neigh)
                S.append(node)
        algo:
            1. init stack S, that contains the order
            2. make adjlist, b needs to be taken before a
            3. for each node in the graph, dfs, in case that node was not visited during a previous nodes' traversal
            4. suppose we are executing the dfs for a node N, 
                we recursively tarverse all the neighbrods of N, which have not been procesed
            5. once processsing all neighs, and N to stack
            6. return order of stack from top to bottom
        
        to differentiate whether we have visited a node on this path while dfsin'g or dfsing a fist time we can color the nodes
        1, means not visited
        2, means visited while dfsing
        3, means visited and added to stack
        
        '''
        adj_list = defaultdict(list)
        
        #build adj list
        for a,b in prerequisites:
            adj_list[b].append(a)
        
        top_sort = []
        self.is_possible = True
        color = {node:1 for node in range(numCourses)}
        
        def dfs(node):
            #if there is a cycle, stop recursing
            if not self.is_possible:
                return
            #mark as 2
            color[node] = 2
            #traverse on neighs
            if node in adj_list:
                for neigh in adj_list[node]:
                    #if its white dfs
                    if color[neigh] == 1:
                        dfs(neigh)
                    elif color[neigh] == 2:
                        self.is_possible = False
            #mark 3 and add
            color[node] = 3
            top_sort.append(node)
        
        for v in range(numCourses):
            if color[v] == 1:
                dfs(v)
        
        return top_sort[::-1] if self.is_possible else []

#top sort using in degree
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        '''
        keeping track of in degree
        intuition:
            first node in top ordering will be a node or nodes with 0 in degree
            process zero in degree ndoes frist, along with out going edges
            one we process a node, reduce its in degree
            
        algo:
            1. init q to keept rack of all the dnoes in the graph with 0 in degree
            2. iterate over all the edges inthe input and create adj list
            3. add all the nodes with 0 in degree to q
                pop node from q
                bfs, reuding neigh's indegree by 1
                add to q if zero
                add node to top sorted order
        '''
                # Prepare the graph
        adj_list = defaultdict(list)
        indegree = {}
        for dest, src in prerequisites:
            adj_list[src].append(dest)

            # Record each node's in-degree
            indegree[dest] = indegree.get(dest, 0) + 1

        # Queue for maintainig list of nodes that have 0 in-degree
        zero_indegree_queue = deque([k for k in range(numCourses) if k not in indegree])

        topological_sorted_order = []

        # Until there are nodes in the Q
        while zero_indegree_queue:

            # Pop one node with 0 in-degree
            vertex = zero_indegree_queue.popleft()
            topological_sorted_order.append(vertex)

            # Reduce in-degree for all the neighbors
            if vertex in adj_list:
                for neighbor in adj_list[vertex]:
                    indegree[neighbor] -= 1

                    # Add neighbor to Q if in-degree becomes 0
                    if indegree[neighbor] == 0:
                        zero_indegree_queue.append(neighbor)

        return topological_sorted_order if len(topological_sorted_order) == numCourses else []










