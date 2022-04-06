#############################
# 680. Valid Palindrome II
# 02APR22
##############################
class Solution:
    def validPalindrome(self, s: str) -> bool:
        '''
        we want to see if string s can form a palindrom after deleting at most one char
        the brute force way would b to delete every possible single char then check if a palindrom can be made
        which would be Nsquared
        
        we can define a helper function that checks for palindrome using i and j
        if the chars at i and j don't match then we just move up i to i + 1 and i to j - 1 and check that substring
        '''
        def is_pal(i,j):
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
            return True
        
        
        i = 0
        j = len(s) - 1
        
        while i < j:
            if s[i] != s[j]:
                return is_pal(i+1,j) or is_pal(i,j-1)
            i += 1
            j -= 1
        
        return True
            
#just another way
class Solution(object):
    def validPalindrome(self, s):
        left = 0
        right = len(s) - 1
        
        while left < right:
            if s[left] != s[right]:
                one = s[left:right]
                two = s[left+1:right+1]
                return one == one[::-1] or two == two[::-1]
            left += 1
            right -= 1
            
        return True

################################
# 1087. Brace Expansion
# 02MAR22
################################
#close one!, almost getting backtracking!
class Solution:
    def expand(self, s: str) -> List[str]:
        '''
        generate using dfs, then sort
        first, i need to find out how long the actuayl created sring will be
        then when i dfs, find the start end ending for each brace
        if we got through the whole string return
        '''
        N = len(s)
        
        self.ans = []
        def rec(i,path):
            print(path)
            if i == N:
                self.ans.append("".join(path))
            #opening brace, we need to back track
            if s[i] == "{":
                #find closing brace ending
                right = i
                while right < N and s[right] != "}":
                    right += 1
                #i and right should be { and }, generae candidates between i and right
                cands = s[i+1:right-1].split(',')
                for c in cands:
                    rec(right+1,path.append(c))

        rec(0,[])
        print(self.ans)
        

#yess!
class Solution:
    def expand(self, s: str) -> List[str]:
        '''
        generate using dfs, then sort
        first, i need to find out how long the actuayl created sring will be
        then when i dfs, find the start end ending for each brace
        if we got through the whole string return
        note how we don't need to bactrack, its not an all possible paths type of problem
        '''
        N = len(s)
        
        self.ans = []
        def rec(i,path):
            if i == N:
                self.ans.append("".join(path))
                return
            #opening brace, we need to back track
            if s[i] == "{":
                #find closing brace ending
                right = i + 1
                while s[right] != "}":
                    right += 1
                #i and right should be { and }, generae candidates between i and right
                cands = s[i+1:right].split(',')
                for c in cands:
                    rec(right+1,path+ [c])
            else:
                rec(i+1,path+[s[i]])

        rec(0,[])
        return sorted(self.ans)
        
#############################
# 599. Minimum Index Sum of Two Lists
# 02APR22
#############################
class Solution:
    def findRestaurant(self, list1: List[str], list2: List[str]) -> List[str]:
        '''
        in each list we want to find out their common interest with least list index sum
        if there are multiplie least sum answers, output all of them with no order requirement
        
        i can make hashmap for both lists mapp rest. to index
        then check if any rest in list1 mapp is in list2 mapp
        put into anoth mapp of common rest to sums
        '''
        mapp1 = {rest:i for i,rest in enumerate(list1)}
        mapp2 = {rest:i for i,rest in enumerate(list2)}

        
        mapp3 = defaultdict(list)
        min_index = float('inf')
        for rest,index in mapp1.items():
            if rest in mapp2:
                #get index sum
                index_sum = index + mapp2[rest]
                min_index = min(min_index,index_sum)
                mapp3[index_sum].append(rest)
        
        return mapp3[min_index]

###############################
# 31. Next Permutation
# 03MAR22
###############################
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        the doh way would be to generate all permutations in order
        then find the current one and get the next one if we can, other wise just return the smallest one in order
        obvie this will time limit, but lets try this one first
        
        '''
        self.ans = []
        N = len(nums)
        def backtrack(perm,i,picked):
            if len(perm) == N:
                self.ans.append(perm[:])
                return
            else:
                for j in range(N):
                    if not picked[j]:
                        picked[j] = True
                        perm.append(nums[j])
                        backtrack(perm,i+1,picked)
                        perm.pop()
                        picked[j] = False
        
        picked = [False]*N
        backtrack([],0,picked)
        

        
        for i in range(len(self.ans)):
            if self.ans[i][:] == nums[:] and i + 1 < len(self.ans):
                print(self.ans[i+1])
                nums[:] = self.ans[i+1][:]
                break
        
        nums[:] = self.ans[0][:]

#two pointer, find stricly icnreasing, swap, then reverse
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        the doh way would be to generate all permutations in order
        then find the current one and get the next one if we can, other wise just return the smallest one in order
        obvie this will time limit, but lets try this one first
        
        the first observartion is that if nums are in decreasng order, there is no next permutation
        [9,5,3,3,1]
        
        neext we need to find thef irst pair of two numbers a[i] and s[i-1] such that a[i] > a[i-1], i.e just strictly greater
        from here, we know that no rearragnements to the the right of a[i-1] can create a larger permutation sice that subarray consitsts of numbers in descending order
        so we need to rearrange the numbers tot her ight if a[i-1] including itself
        
        how do we rearrange?
            we want to create the permutation just larger than the current one
            so we replace number a[i-1] with number that is just larger then it among the number lay to its right, call it a[j]
            we then swap a[i-1] with a[j]
        but we still aren't done
            we need the smalelst permualtion taht can beformed by using numbes only to the right of a[i-1]
            therefore we need to replace the numebrs ina secneindg order
        but recall that while scanning the numebrs from the right, we had already checked for stricly increasging
        we just need tor reverse them
        '''
        #starting from the right, find the first decreasing element
        i = len(nums) - 1
        while i - 1 >= 0 and nums[i-1] >= nums[i]:
            i -= 1
            
        #i-1 is our current swap point
        if i - 1 >= 0:
            j = len(nums) - 1
            while nums[j] <= nums[i-1]:
                j -= 1
            #swap
            nums[i-1],nums[j] = nums[j], nums[i-1]
            
        #reverse
        nums[i:] = nums[i:][::-1]

########################################
# 1721. Swapping Nodes in a Linked List
# 04APR22
########################################
#close one
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        '''
        we want to swap nodes from the kth beginning to the kth end
        indexing starts at 1
        
        three passes
        first find N
        then advnace K
        then addnace N-k
        get values and swap
        '''
        N = 0
        curr = head
        while curr:
            N += 1
            curr = curr.next
            
        if N == 1:
            return head
            
        dummy = ListNode(-1)
        dummy.next = head
        curr = dummy
        
        curr_step = 0
        
        while curr_step <= N-k:
            if curr_step == k:
                kth_node = curr
            if curr_step == N-k:
                kth_end_node = curr.next
            curr_step += 1
            curr = curr.next
        
        #swap
        kth_node.val,kth_end_node.val = kth_end_node.val,kth_node.val
        return dummy.next


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        '''
        three pass approache
        '''
        N = 0
        curr = head
        while curr:
            N += 1
            curr = curr.next
            
        #find front
        front = head
        for _ in range(1,k):
            front = front.next
        
        #find end
        end = head
        for _ in range(1,N-k+1):
            end = end.next
        
        front.val,end.val = end.val,front.val
        return head

#two pass
class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        '''
        we can reduce this to two passes, 
        on the first pass while getting the length of the LL, if we get to legnth == k, mark front node
        then on the second pass mark the end node swap
        '''
        #find size of LL and mark front
        N = 0
        curr = head
        front = None

        while curr:
            N += 1
            if N == k:
                front = curr
            curr = curr.next
            
        #andvance N-k
        end = head
        for _ in range(1,N-k+1):
            end = end.next
            
        front.val,end.val = end.val, front.val
        return head
        
#one pass
class Solution:
    def swapNodes(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        '''
        we cam do this in a singel pass, and trick has come up in similar problems, Remove N'th Node from End of List and
        Find the middle of linked list
        
        if endNode lags behind a current node, then when current node reaches the end of the linked list, i.e the nth node, 
        the endNode would be at n-kth node
        
        algo:
            * start iterating from the head of the Linked List until the end using curr pointer
            * keep track of the number of nodes traverses so far using length and increment 1
            * when N == k, we know that the curr nodes is at the kth node
                * set front to point to the kth node nad end to head
                * now end is k behing curr
            * if end is not null, the we can advance both and and curr
            * swap
        '''
        N = 0
        front = None
        end = None
        curr = head
        
        while curr:
            N += 1
            if end:
                end = end.next
            if N == k:
                front = curr
                end = head
            curr = curr.next
        
        front.val,end.val = end.val, front.val
        return head


###############################
# 606. Construct String from Binary Tree
# 03APR22
###############################
#close one
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        '''
        preoder is node left right
        emtpy node, return and empty string
        otherwise if there is a node, get the node and put inside () only if there are children
        '''
        self.ans = ""
        
        def dfs(node):
            if not node:
                self.ans += "()"
                return
            
            #at the node
            self.ans += str(node.val)
            #opening
            if node.left:
                self.ans += '('
                dfs(node.left)
                self.ans += ')'
            if node.right:
                self.ans += '('
                dfs(node.right)
                self.ans += ')' 

        
        dfs(root)
        return self.ans

#top down
#when doing left, wrap
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:

        self.ans = ""
        
        def dfs(node):
            if not node:
                return
            self.ans += str(node.val)
            if not node.left and not node.right:
                return
            #when going left, alwasy wrap left node with parathenssis
            self.ans += '('
            dfs(node.left)
            self.ans += ')'
            
            if node.right:
                #wrap again
                self.ans += '('
                dfs(node.right)
                self.ans += ')'
                
        dfs(root)
        return self.ans

#another way
class Solution:
    def tree2str(self, t: TreeNode) -> str:
        sb = [] # init string builder
        
        # helper function to create result
        def helper(node: TreeNode) -> None:
            if not node:
                return
            
            sb.append('(')                      # add 1st wrapping parenthesis
            sb.append(str(node.val))
            
            helper(node.left)                   # process left recursively
            
            if not node.left and node.right:    # add parenthesis if left is empty but right exists
                sb.append('()')
                
            helper(node.right)                  # process right recursively
            
            sb.append(')')                      # add 2nd wrapping parenthesis
        
        helper(t)

        # trim 1st and last parenthesis build result string
        return ''.join(sb[1:-1]) 


#bottom up
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        '''
        bottom up, with reaturn
        '''
        
        def dfs(node):
            if not node:
                return ""
            if not node.left and not node.right:
                return str(node.val)
            #when going left, alwasy wrap left node with parathenssis
            if not node.right:
                return str(node.val) + '('+dfs(node.left)+')'
            else:
                return str(node.val) + '('+dfs(node.left)+')'+'('+ dfs(node.right) +')'
                
        return dfs(root)

#usings stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        '''
        iterative version, 
        we need to use a visited set, and we also push the elements right, left insteadf of left, right 
        
        '''
        stack = [root]
        seen = set()
        ans = ""
        
        while stack:
            curr = stack[-1]
            if curr in seen:
                stack.pop()
                ans += ')'
            else:
                seen.add(curr)
                ans += '('+str(curr.val)
                if not curr.left and curr.right:
                    ans += '()'
                
                if curr.right:
                    stack.append(curr.right)
                
                if curr.left:
                    stack.append(curr.left)
                    
        
        return ans[1:-1]

#another way
def tree2str(self, root: Optional[TreeNode]) -> str:
    """ O(N)TS"""
    left = f'({self.tree2str(root.left)})' if root.left else ('()' if root.right else '')
    right = f'({self.tree2str(root.right)})' if root.right else ''
    return f'{root.val}' + left + right

#using .format
def tree2str(self, t):
    if not t: return ''
    left = '({})'.format(self.tree2str(t.left)) if (t.left or t.right) else ''
    right = '({})'.format(self.tree2str(t.right)) if t.right else ''
    return '{}{}{}'.format(t.val, left, right)

############################
# 11. Container With Most Water
# 05APR22
#############################
class Solution:
    def maxArea(self, height: List[int]) -> int:
        '''
        we can use two pointers here
        maximize the current area for the two bars
        advance the smaller of the two
        '''
        left = 0
        right = len(height) -1
        ans = 0
        
        while left < right:
            curr_height = min(height[left],height[right])
            ans = max(ans, curr_height*(right - left))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        
        return ans
        
#another way using 2d representation
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        MAX = 0 
        x = len(height) - 1
        y = 0
        while x != y:
            if height[x] > height[y]:
                area = height[y] * (x - y)
                y += 1
            else:
                area = height[x] * (x - y)
                x -= 1
            MAX = max(MAX, area)
        return MAX