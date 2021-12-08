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













