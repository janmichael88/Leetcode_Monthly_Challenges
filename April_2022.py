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
# 02APR22
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
# 03APR22
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

##############################
# 923. 3Sum With Multiplicity
# 06APR22
###############################
#doesn't count all multiplicites
class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        '''
        we can use standard two pointer trick
        sort, pick all starting,
        then two pointer from starting when summ == target
        '''
        mod = 10**9 + 7
        N = len(arr)
        count = 0
        
        #sort the array
        arr.sort()
        for i in range(N):
            j = i+1
            k = N-1
            while j < k:
                #matches target
                if arr[i] + arr[j] + arr[k] == target:
                    count += 1
                    j += 1
                    k -= 1
                #if less
                elif arr[i] + arr[j] + arr[k] < target:
                    j += 1
                else:
                    k -= 1
                
                count = count % mod
        
        return count
            
#count combinations
        mod = 10**9 + 7
        count = 0 
        N = len(arr)
        arr.sort()
        
        for i in range(N):
            #apply two sum with points j and k
            twoSum = target - arr[i]
            j = i + 1
            k = N -1
            while j < k:
                if arr[j] + arr[k] < twoSum:
                    j += 1
                elif arr[j] + arr[k] > twoSum:
                    k -= 1
                #at some point these two will equal twoSum, and case 1 are differnt
                elif arr[j] != arr[k]:
                    #cound the number of times weve moved left and riight
                    left = 1
                    right = 1
                    while j + 1 < k and arr[j] == arr[j+1]:
                        left += 1
                        j += 1
                    while k - 1 > j and arr[k] == arr[k-1]:
                        right += 1
                        k -= 1
                    
                    #how many pairs have we contirbuted
                    count += left*right #the multiplicities
                    count %= mod
                    #from two sum
                    j += 1
                    k -= 1
                else:
                    #htey are equal and all the same, case 2
                    M = (k-j+1)
                    count += M*(M-1) / 2
                    count %= mod
                    break
        return count

#########################
# 1046. Last Stone Weight
# 07APR22
##########################
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        '''
        we can just use a a max heap
        pop two,smash and add back in the reult of the smach
        keep doing while len(heap) > 1
        '''
        
        stones = [-stone for stone in stones]
        heapq.heapify(stones)
        
        while len(stones) > 1:
            first = - heapq.heappop(stones)
            second = - heapq.heappop(stones)
            if first == second:
                continue
            else: 
                heapq.heappush(stones, -(first-second))
                
        return -heappop(stones) if stones else 0

#bucket sort
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        '''
        we can use bucket sort so sort the weights of thes stones
        then we need to start from the max weight
        if the largest weight can be smashed together, reduce its count
        '''
        max_weight = max(stones)
        buckets = [0]*(max_weight + 1)
        
        #bucket sort
        for weight in stones:
            buckets[0] += 1
            
        #scane through buckets
        biggest_weight = 0 #we don't have a weight yet
        curr_weight = max_weight
        
        while curr_weight > 0:
            #if there are no rocks to smash as this weight keep going
            if buckets[curr_weight] == 0:
                curr_weight -= 1
            #othwerwise its not zero, but we need an update to the biggest weight
            elif biggest_weight == 0:
                #try to smash
                buckets[curr_weight] %= 2
                #if we are left with one, we have a new biggest weight
                if buckets[curr_weight] == 1:
                    biggest_weight = curr_weight
                curr_weight -= 1
            else:
                #its one, so use it up
                buckets[curr_weight] -= 1
                if biggest_weight - curr_weight <= curr_weight:
                    #add in a new stone
                    buckets[biggest_weight - curr_weight] += 1
                    biggest_weight = 0
                else:
                    #this becomes the new biggest stone
                    biggest_weight -= curr_weight
                    
        
        return biggest_weight


#######################################
# 703. Kth Largest Element in a Stream
# 08APR22
#######################################
from sortedcontainers import SortedList
class KthLargest():

    def __init__(self, k: int, nums: List[int]):
        '''
        we will always have to return the kth largest
        i can insert every time and sort, but the would be slow since add is called 10**4 times
        i can use a sorted container
        '''
        self.sl = SortedList(nums)
        self.k = k
        

    def add(self, val: int) -> int:
        self.sl.add(val)
        return self.sl[-self.k]


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)

#min heap
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        '''
        i can use a heap to keep only k elements
        if i ever for more than k, i need to clear them
        
        if we only ever have k elements, and of these k we want the kth largest
        really we just want the minimum of k elements
        '''
        #frist heappofy nums
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        #take only k elements
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap,val)
        #more than k
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]
        


# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)

#########################################
# 604. Design Compressed String Iterator
# 08APR22
##########################################
#close one
class StringIterator:

    def __init__(self, compressedString: str):
        '''
        i need a way to get the letter and its number
        the numbers could be very big, it could Z999999....999999
        start from the end of the array
        get the digit, then get the number and put into [char,num times]
        then ptr manip to read next and has next
        '''
        self.char_num = []
        N = len(compressedString)
        
        i = N - 1
        
        while i > 0:
            j = i
            #start getting the count
            count = 0
            while compressedString[j].isdigit():
                count *= 10
                count += int(compressedString[j])
                j -= 1
            #we now have its count and j is on the letter
            self.char_num.append([compressedString[j],count])
            #move over one more
            i = j - 1
            
        #just reverse char num so we don't get fucked up
        self.char_num = self.char_num[::-1]
        self.N = len(self.char_num)
        self.ptr = 0

    def next(self) -> str:
        if self.ptr < self.N and self.char_num[self.ptr][1] > 0:
            #take away the count for the curret char
            self.char_num[self.ptr][1] -= 1
            ans = self.char_num[self.ptr][0]
        else:
            #can't sue this char, go to the next one
            self.ptr += 1
            ans = self.char_num[self.ptr][0]
            self.char_num[self.ptr][1] -= 1
            
        return ans

    def hasNext(self) -> bool:
        return self.ptr < self.N
        


# Your StringIterator object will be instantiated and called as such:
# obj = StringIterator(compressedString)
# param_1 = obj.next()
# param_2 = obj.hasNext()

#better way
class StringIterator:

    def __init__(self, compressedString: str):
        '''
        i need a way to get the letter and its number
        the numbers could be very big, it could Z999999....999999
        start from the end of the array
        get the digit, then get the number and put into [char,num times]
        then ptr manip to read next and has next
        '''
        self.char_num = []
        N = len(compressedString)
        i = 0
        
        while i < N:
            char = compressedString[i]
            i += 1
            #start getting the count
            count = 0
            while i < N and compressedString[i].isdigit():
                count = count*10 + int(compressedString[i])
                i += 1
            #we now have its count and j is on the letter
            self.char_num.append([char,count])

            
        #just reverse char num so we don't get fucked up
        self.N = len(self.char_num)
        self.ptr = 0

    def next(self) -> str:
        print(self.char_num)
        if self.ptr < self.N and self.char_num[self.ptr][1] > 0:
            #take away the count for the curret char
            self.char_num[self.ptr][1] -= 1
            ans = self.char_num[self.ptr][0]
        else:
            #can't sue this char, go to the next one
            self.ptr += 1
            ans = self.char_num[self.ptr][0]
            self.char_num[self.ptr][1] -= 1
            
        return ans

    def hasNext(self) -> bool:
        return self.ptr < self.N

#using regex
import re
class StringIterator:

    def __init__(self, compressedString: str):
        '''
        we can use a regex expression to find the right patterns it would be 'letter' followed by 'num times of letter'
        we can use the pattern '\D\d+'
         means a non-digit character, followed by 1 or more digit characters. (The + denotes a kleene plus, a wildcard character meaning "one or more of the preceding match.")
        '''
        self.tokens = []
        pattern = '\D\d+'
        for token in re.findall(pattern,compressedString):
            self.tokens.append([token[0],int(token[1:])])
        #reverse
        self.tokens = self.tokens[::-1]
        

    def next(self) -> str:
        if not self.tokens:
            return ' '
        char,count = self.tokens.pop()
        if count > 1:
            self.tokens.append([char,count-1])
        return char

    def hasNext(self) -> bool:
        return len(self.tokens) > 0

#absorbing cost in the the next call
class StringIterator:

    def __init__(self, compressedString: str):
        '''
        instead of just building out all uncompressed string in the constructor, we can build it in the next call
        the contstuctor is O(1)
        next is amortized O(1)
        '''
        self.res = compressedString
        self.ptr = 0
        self.count = 0
        self.ch = ' '

    def next(self) -> str:
        #if there is no next
        if not self.hasNext:
            return ' '
        if self.count == 0:
            #get the char
            self.ch = self.res[self.ptr]
            self.ptr += 1
            while self.ptr < len(self.res) and self.res[self.ptr].isdigit():
                self.count = self.count*10 + int(self.res[self.ptr])
                self.ptr += 1
        
        #use up
        self.count -= 1
        return self.ch
        
    def hasNext(self) -> bool:
        return self.ptr != len(self.res) or self.count != 0

####################################
# 09APR22
# 347. Top K Frequent Elements
####################################
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        '''
        i cant count then sort
        '''
        counts = Counter(nums)
        ans = []
        for num,count in counts.most_common():
            ans.append(num)
            k -= 1
            if k == 0:
                break
        
        return ans

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        #special case when k == nums:
        if k == len(nums):
            return nums
        
        #lets count
        counts = Counter(nums)
        
        #init heap
        heap = []
        
        for num,count in counts.items():
            heapq.heappush(heap,(-num,count))
            if len(heap) > k:
                heapq.heappop(heap)
        
        
        ans = []
        
        while heap:
            temp = heapq.heappop(heap)
            ans.append(-temp[0])
        
        return ans

import heapq
from collections import Counter
class Solution:
    def topKFrequent(self, nums, k):
        res = []
        dic = Counter(nums)
        max_heap = [(-val, key) for key, val in dic.items()]
        heapq.heapify(max_heap)
        for i in range(k):
            res.append(heapq.heappop(max_heap)[1])
        return res   

#using n largest
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. build hash map : character and how often it appears
        # O(N) time
        count = Counter(nums)   
        # 2-3. build heap of top k frequent elements and
        # convert it into an output array
        # O(N log k) time
        return heapq.nlargest(k, count.keys(), key=count.get) 

#quick select
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        '''
        we can also use quick select here, typically involves choosing the k'th something
        it was created by Tony Hoare, and is called Hoare's algorithm
        
        average O(N), but in the worst case can be O(N**2), but probability is negligible
        
        intution:
        One chooses a pivot and defines its position in a sorted array in a linear time using so-called partition algorithm.
        count of the counts of each number, then for each numbe place it in in sorted array using the count as the pivot
        recruse on the left or right parts until we have met N-k condition
        
        it would be a quicksort if we recursed on the whole array
        but we can discard the k-1 part of the k+1 part
        
        algo:
            count the freq count of the array
            work with unique array, and use partition scheemt to place the pivot into its perfect positiosn in the sorted array
            move less frequent elements to th left and more frequent elements to the right
            compare pivot index and N-k
                if pivot_index == N-k, the pivot is the N-kth most frequent element, so we can everything to the right, inclusive
                otherrise recuse on the left side
                
        partition shcheme: we use a variant of Lomuto
            * move pivot at the end of the array using swap
            * set pointer at the beginning of the array : stor_index = left
            * itratore over teh array and move all less frequent elements to theft left (i.e swap(store_index,i))
            * move store_inde one steap to the right after each swap
            * move the pivot to its final place
        '''
        count = Counter(nums)
        unique = list(count.keys())
        
        def partition(left, right, pivot_index) -> int:
            pivot_frequency = count[unique[pivot_index]]
            # 1. move pivot to end
            unique[pivot_index], unique[right] = unique[right], unique[pivot_index]  
            
            # 2. move all less frequent elements to the left
            store_index = left
            for i in range(left, right):
                if count[unique[i]] < pivot_frequency:
                    unique[store_index], unique[i] = unique[i], unique[store_index]
                    store_index += 1

            # 3. move pivot to its final place
            unique[right], unique[store_index] = unique[store_index], unique[right]  
            
            return store_index
        
        def quickselect(left, right, k_smallest) -> None:
            """
            Sort a list within left..right till kth less frequent element
            takes its place. 
            """
            # base case: the list contains only one element
            if left == right: 
                return
            
            # select a random pivot_index
            pivot_index = random.randint(left, right)     
                            
            # find the pivot position in a sorted list   
            pivot_index = partition(left, right, pivot_index)
            
            # if the pivot is in its final sorted position
            if k_smallest == pivot_index:
                 return 
            # go left
            elif k_smallest < pivot_index:
                quickselect(left, pivot_index - 1, k_smallest)
            # go right
            else:
                quickselect(pivot_index + 1, right, k_smallest)
        
        n = len(unique) 
        # kth top frequent element is (n - k)th less frequent.
        # Do a partial sort: from less frequent to the most frequent, till
        # (n - k)th less frequent element takes its place (n - k) in a sorted array. 
        # All element on the left are less frequent.
        # All the elements on the right are more frequent.  
        quickselect(0, n - 1, n - k)
        # Return top k frequent elements
        return unique[n - k:]

#using bucket sort
class Solution:
    def topKFrequent(self, nums, k):
        bucket = [[] for _ in range(len(nums) + 1)]
        Count = Counter(nums).items()  
        for num, freq in Count: bucket[freq].append(num) 
        flat_list = list(chain(*bucket))
        return flat_list[::-1][:k]

##############################
# 10APR22
# 682. Baseball Game
##############################
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        '''
        we can just use stack to get the operations, kinda like post fix notication
        just follow the rules
        '''
        stack = []
        for op in ops:
            if op == '+':
                stack.append(stack[-1] + stack[-2])
            elif op == 'C':
                stack.pop()
            elif op == 'D':
                stack.append(2 * stack[-1])
            else:
                stack.append(int(op))

        return sum(stack)

###############################################
# 10APR22
# 1874. Minimize Product Sum of Two Arrays
##############################################
class Solution:
    def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        i want to return the minimum dot product for nums1 and nums2, where i can have any permutation of nums1
        brute force would be to examine all permutations of nums1, and take the minimum dot product with nums2
        elements are always positive
        
        given numbes a and b a*b - (a-1)*(b-1)
        a**2 + b**2 - 2ab - ab
        a**2 + b**2 - ab > 0 for all a and b that are non negative
        implying (a-1)*(b-1) > a*b
        
        a**2 - ab
        a(a-b)
        
        rather say we have number x, we can get x**2
        x(x-1) vs x**2
        x**2 - x 
        
        to get the smallest product sum of two arrays we want to multiply two numbers that are far apart
        
        the tidbit for saying that we can only permute nums1, really means we can match any element in nums1 to nums2
        we pair the smallest elemtnet in nums1 to the largetst element in nums2
        sort nums1, reverse sort nums2, find the dot product
        '''
        nums1.sort()
        nums2.sort(reverse = True)
        ans = 0
        for x,y in zip(nums1,nums2):
            ans += x*y
        
        return ans

#using PQ
class Solution:
    def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can use a max heap
        we sort num1, but push elements of nums2 on to a max heap
        '''
        nums1.sort()
        
        nums2_heap = [-num for num in nums2]
        heapq.heapify(nums2_heap)
        
        ans = 0
        
        for i in range(len(nums2)):
            ans += nums1[i]*(-heapq.heappop(nums2_heap))
            
        return ans

#counting sort
class Solution:
    def minProductSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Initialize counter1 and counter2.
        counter1, counter2 = [0] * 101, [0] * 101

        # Record the number of occurrence of elements in nums1 and nums2.
        for num in nums1:
            counter1[num] += 1
        for num in nums2:
            counter2[num] += 1
        
        # Initialize two pointers p1 = 1, p2 = 100.
        # Stands for counter1[1] and counter2[100], respectively.
        p1, p2 = 1, 100
        ans = 0
        
        # While the two pointers are in the valid range.
        while p1 <= 100 and p2 > 0:

            # If counter1[p1] == 0, meaning there is no element equals p1 in counter1,
            # thus we shall increment p1 by 1.
            while p1 <= 100 and counter1[p1] == 0:
                p1 += 1

            # If counter2[p2] == 0, meaning there is no element equals p2 in counter2,
            # thus we shall decrement p2 by 1.
            while p2 > 0 and counter2[p2] == 0:
                p2 -= 1

            # If any of the pointer goes beyond the border, we have finished the 
            # iteration, break the loop.
            if p1 == 101 or p2 == 0:
                break

            # Otherwise, we can make at most min(counter1[p1], counter2[p2]) 
            # pairs {p1, p2} from nums1 and nums2, let's call it occ. 
            # Each pair has product of p1 * p2, thus the cumulative sum is 
            # incresed by occ * p1 * p2. Update counter1[p1] and counter2[p2].
            occ = min(counter1[p1], counter2[p2])
            ans += occ * p1 * p2
            counter1[p1] -= occ
            counter2[p2] -= occ
        
        # Once we finish the loop, return ans as the product sum.
        return ans

########################
# 11APR22
# 1260. Shift 2D Grid
########################       
#using space at each iteration
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        '''
        it shifts like an S shape, with wrap around from bottom right to top left
        brute force would be
        50*50*100, which isn't too bad
        
        basically each elements gets shift over one to the right, if it can
        if we are in the last col, we move element to the next row first col
        then bottom right gets sent to top left
        '''
        num_rows, num_cols = len(grid), len(grid[0])

        for _ in range(k):
            # Create a new grid to copy into.
            new_grid = [[0] * num_cols for _ in range(num_rows)]

            # Case 1: Move everything not in the last column.
            for row in range(num_rows):
                for col in range(num_cols - 1):
                    new_grid[row][col + 1] = grid[row][col]

            # Case 2: Move everything in last column, but not last row.
            for row in range(num_rows - 1):
                 new_grid[row + 1][0] = grid[row][num_cols - 1]

            # Case 3: Move the bottom right.
            new_grid[0][0] = grid[num_rows - 1][num_cols - 1]

            grid = new_grid

        return grid

class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        '''
        we can swap the value in place
        in the last approach, we allocated a new 2d array
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        for _ in range(k):
            prev = grid[-1][-1]
            
            for row in range(rows):
                for col in range(cols):
                    #save what is curretnly there
                    temp = grid[row][col]
                    grid[row][col] = prev
                    prev = temp
        
        return grid

#using modular arithmetic
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        '''
        we can calculate what the final i,j positions will be after k steps using modular arithmetic
        the value of the column will change k times
        in fact new_col = (col + k) % num_cols
        we cant think of modulo as subtracting the number k from col + k until we have less than k left 
        similar to gcd
        
        finding the row is a little trickier
        notice that the new row value does not change as often as the new col value
        
        since the value of the row goes up when we move from the last col to the first col, we need to determine how many times the value moves from the last to first col
        
        we look at the quotionet this time
        
        new_col = (j + k) % num_cols
        
        number_of_increments = (j + k) / num_cols
        new_row = (i + number_of_increments) % num_rows
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        new_grid = [[0]*rows for _ in range(cols)]
        
        for row in range(rows):
            for col in range(cols):
                #find new row and new col
                new_col = (col + k) % cols
                num_times_around = (col + k) // cols
                new_row = (row + num_times_around) % rows
                new_grid[new_row][new_col] = grid[row][col]
        
        return new_grid

#another way
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        '''
        another way is think of the grid as 1 1d array, then we just shift k times at most rows*cols
        convert its position in the array back to it next index
        '''
        rows = len(grid)
        cols = len(grid[0])
        N = rows*cols
        
        new_grid = [[0]*cols for _ in range(rows)]
        
        for i in range(N):
            #shift i
            new_i = (i + k) % N
            #fin back
            old_row,old_col = divmod(i,cols)
            #find this column
            new_row,new_col = divmod(new_i,cols)
            new_grid[new_row][new_col] = grid[old_row][old_col]
        
        return new_grid

#in place
class Solution:
    def shiftGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        '''
        another way is think of the grid as 1 1d array, then we just shift k times at most rows*cols
        convert its position in the array back to it next index
        '''
        rows = len(grid)
        cols = len(grid[0])
        N = rows*cols
        
        count = 0
        i = 0
        
        while count < N:
            #shift i
            new_i = (i + k) % N
            #find old position back
            old_row,old_col = divmod(i,cols)
            curr  = grid[old_row][old_col]
            
            while True:
                #find this new row and column
                new_row,new_col = divmod(new_i,cols)
                #swap
                grid[new_row][new_col],curr = curr, grid[new_row][new_col]
                #increase count
                count += 1
                if i == new_i:
                    break
                #update new_i
                new_i = (new_i + k) % N
            i += 1
        
        return grid

################################
# 289. Game of Life
# 12APR22
################################
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        we can make a copy of the board and count of the live cell nieghbors for each cell in the grid
        we use the copy as the oriignal state, and update new states in the original board
        
        we first define a helper function that returns the number of neighboring live counts from the current cell
        
        '''
        rows = len(board)
        cols = len(board[0])
        
        #make copy of board
        copy_board = [[board[row][col] for col in range(cols)] for row in range(rows)]
        
        #generate dx,dy 
        steps = [0,1,-1]
        
        dirrs = []
        for i in range(len(steps)):
            for j in range(len(steps)):
                if (i,j) != (0,0):
                    dirrs.append([steps[i],steps[j]])
        
        def count_live(i,j):
            count_live = 0
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                
                #in bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    if copy_board[neigh_x][neigh_y] == 1:
                        count_live += 1
            return count_live
        
        for row in range(rows):
            for col in range(cols):
                count = count_live(row,col)
                
                #we can combine rule 1 and 3
                if copy_board[row][col] == 1 and (count < 2 or count > 3):
                    board[row][col] = 0
                #rule 2
                elif copy_board[row][col] == 1 and (count == 2 or count == 2):
                    board[row][col] = 1
                #rule 4
                elif copy_board[row][col] == 0 and count == 3:
                    board[row][col] = 1

class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        to do in place we need to store the 'change in cell state' onto the oriingal boar
        then pass the board again to apply the changes
        
        if cell was originally 1 and become 0, we can change this calue to -1
        the negative now signigies a dead cell, but the magnitude signifies the cell was alive
        
        if cell was originally 0 but become 1 after applying the rule, then we can change the value to 2
        the positive sign means cell is now alive, but 2 means the cell was oringally dead
        
        the update rules are as follows:
            rule 1: any live cell with < 2 live neigh dies, so we change this to -1
            rule 2: any live cell with 2 or 3, lives on, so no change
            rule 3: any live cell with > 3 live neighbores dies, so change value to -1, we don't need to differentiate from rule 3
            rule 4: any dead cell with == 3 live neighbords, becomes 1, so mark as 2
            
            or change values are coded as -1 and 2
        
        apply rules:
            if value is greater than 0, it becomes 1
            if values i < 0, it becomes 0
        '''
        rows = len(board)
        cols = len(board[0])
        
        #generate dx,dy 
        steps = [0,1,-1]
        
        dirrs = []
        for i in range(len(steps)):
            for j in range(len(steps)):
                if (i,j) != (0,0):
                    dirrs.append([steps[i],steps[j]])
        
        def count_live(i,j):
            count_live = 0
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                
                #in bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #need original live count
                    if abs(board[neigh_x][neigh_y]) == 1:
                        count_live += 1
            return count_live
        
        for row in range(rows):
            for col in range(cols):
                count = count_live(row,col)
                
                #we can combine rule 1 and 3
                if board[row][col] == 1 and (count < 2 or count > 3):
                    board[row][col] = -1
                #rule 2
                elif board[row][col] == 1 and (count == 2 or count == 3):
                    board[row][col] = 1
                #rule 4
                elif board[row][col] == 0 and count == 3:
                    board[row][col] = 2
        
        #apply changes
        for row in range(rows):
            for col in range(cols):
                if board[row][col] > 0:
                    board[row][col] = 1
                else:
                    board[row][col] = 0

###############################
# 617. Merge Two Binary Trees
#  14APR22
################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we want to over lap a tree with another tree
        if nodes in both trees over lap, the new node becomes the sum
        otherwise place with the exisiting node if there is a node in either of them
        
        i can dfs down a tree and mark each node with it index and value
        dfs the second tree and to it into the dictionary  of index leaf and it value
        then recreate 
        
        using dfs up until the last index
        '''
        
        if not root1 and not root2:
            return None
        index_value = defaultdict(int)
        
        def dfs(node,index):
            if not node:
                return
            index_value[index] += node.val
            if node.left:
                dfs(node.left,index*2)
            if node.right:
                dfs(node.right,index*2 + 1)
        
        dfs(root1,1)
        dfs(root2,1)
        
        #now i need to build the fucking tree
        ans = TreeNode(-1) #to return once done
        #move pointer
        curr = ans
        
        q = deque([(curr,1)])
        
        while q:
            node,index = q.popleft()
            #if i have this mapped
            if index in index_value:
                node.val = index_value[index]
            #if i can go left
            if index*2 in index_value:
                #make a new node, connect left and add it to q
                new_left = TreeNode(-1)
                node.left = new_left #this value will be added on the enxt pop
                q.append([new_left,index*2])
            #if we can go right
            if index*2 + 1 in index_value:
                #make a new right
                new_right = TreeNode(-1)
                node.right = new_right
                q.append([new_right,index*2 + 1])
        
        return ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we can use recursion, 
        if root1 is empty its obvious we must return root2
        same goes for the other way
        
        otherwise sum up their values, we use the left argument as the main
        then recurse
        '''
        def dfs(r1,r2):
            if not r1:
                return r2
            if not r2:
                return r1
            #combine
            r1.val += r2.val
            r1.left = dfs(r1.left,r2.left)
            r1.right = dfs(r1.right,r2.right)
            return r1
        
        return dfs(root1,root2)

class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we can also use stack to save on space
        push two nodes on to stack for every call
        if nodes exsist increment the sum of the frist 1 by the second one
        '''
        if not root1:
            return root2
        if not root2:
            return root1
        
        stack = [(root1,root2)]
        
        while stack:
            node1,node2 = stack.pop()
            #if noth are empty
            if not node1 or not node2:
                continue
            #increment
            node1.val += node2.val
            #if there is no left
            if not node1.left:
                node1.left = node2.left
            else:
                stack.append((node1.left,node2.left))
                
            #if there is a right
            if not node1.right:
                node1.right = node2.right
            else:
                stack.append((node1.right,node2.right))
        
        return root1

#################################
# 669. Trim a Binary Search Tree
# 15APR22
#################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        '''
        we can do this recursively
        we just trim if node is too passed it the low or high
        '''
        def dfs(node):
            if not node:
                return
            if node.val > high:
                return dfs(node.left)
            elif node.val < low:
                return dfs(node.right)
            #if we can't trim we need to reconnect
            else:
                node.left = dfs(node.left)
                node.right = dfs(node.right)
                return node
        
        return dfs(root)

#iterative with stack
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        '''
        stack implementation is tricker, but we can still do it
        we keep a global tree answer to which we can return the answer
        
        then we break and make connections depending on whehter we discard left or right subtrees, IFF there are subtrees to be discarded
        '''
        zero = TreeNode(val=-1, right=root)
        st = [(root, zero)]
        while st:
            node, parent = st.pop()
            if not node: 
                continue
            
            if node.val < low:
                if node.right: # left subtree goes away
                    if node.right.val > parent.val:
                        parent.right = node.right
                    else:
                        parent.left = node.right
                        
                    st.append((node.right, parent))
                else:
                    if node.val > parent.val:
                        parent.right = None
                    else:
                        parent.left = None
                    
                
                    
            if node.val > high:
                if node.left: # right subtree goes away
                    if node.left.val > parent.val:
                        parent.right = node.left
                    else:
                        parent.left = node.left
                        
                    st.append((node.left, parent))
                else:
                    if node.val > parent.val:
                        parent.right = None
                    else:
                        parent.left = None
                        
                
            
            st.append((node.left, node))
            st.append((node.right, node))
            
        return zero.right

###################################
# 538. Convert BST to Greater Tree
# 16APR22
###################################
#recursive
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we can solve this recusrively prioritzing the right nodes first! 
        we get their sums
        the idea is to visit the nodes in descending order! keeping the sum of all values that we have already visited
        this is reverse in order , so right, root, left
        '''
        self.sum = 0
        
        def dfs(node):
            if not node:
                return
            dfs(node.right)
            self.sum += node.val
            node.val = self.sum
            dfs(node.left)
            return node
            
        return dfs(root)
        
#iterative with stack
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        we can do this iteratively with the usual preorder way
        but we start right first instead of left
        '''
        SUM = 0
        curr = root
        
        stack = []
        
        while curr != None or len(stack) > 0:
            #go right as far as we can
            while curr != None:
                stack.append(curr)
                curr = curr.right
            
            #no lets process
            curr = stack.pop()
            #accumulate the sums
            SUM += curr.val
            #change value
            curr.val = SUM
            
            #finall move left
            curr = curr.left
        
        
        return root


#######################################
# 1586. Binary Search Tree Iterator II
# 16APR22
#######################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        '''
        the dumb way to to just unpack the tree in the constructor
        then just control flow using global pointer back to array of TreeNodes
        remember this is order, as practive alwasy do in order iteratively
        '''
        self.nodes = []
        curr = root
        stack = []
        
        while curr != None or len(stack) > 0:
            while curr != None:
                stack.append(curr)
                curr = curr.left
            
            curr = stack.pop()
            self.nodes.append(curr.val)
            curr = curr.right
        
        self.N = len(self.nodes)
        self.ptr = -1

    def hasNext(self) -> bool:
        return self.ptr  < self.N - 1

    def next(self) -> int:
        self.ptr += 1
        ans = self.nodes[self.ptr]
        return ans

    def hasPrev(self) -> bool:
        return self.ptr > 0

    def prev(self) -> int:
        self.ptr -= 1
        ans = self.nodes[self.ptr]
        return ans


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.hasNext()
# param_2 = obj.next()
# param_3 = obj.hasPrev()
# param_4 = obj.prev()

#iterative
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator:
    '''
    the problem is that we parse the whole root at the Constructor level, we want average case to be amotrized O(1)
    in the worst case, we have an un parsed left subtree that we have to traverse, so next would be worset case O(N)
    
    in addition to saving the value of the nodes in an array
    we save all parsed nodes in a list and then re-use them if we need to return next from the laready parsed area of the tree
    
    algo:
        Constructor
            initialize last processed node as root, last = root
            initialize a list to store already processed nodes arr
            initialize service data structure to  be used during in order
            init pointer to be -1, serves as indicator to alraeadt parsed area or not, we are only in the parsed area if pointer + 1 < len(arr)
        
        hasNext:
            check if pointer + 1 < len(arr)
        
        next:
            increase pointer by 1
            if we're not in the precompute part of tree parse the bare minimumu
                go left until you can
                    push last node back on to stack
                    go left
                pop last node out of stack
                append this node value to arr
                go right
            otherwise, return fro the arr
        
        hasPrev
            compare the pointer to zero, 
        prev:
            decrease the pointer by one and return arr[pointer]
            
            
    '''

    def __init__(self, root: Optional[TreeNode]):
        self.last = root #to get previous node after parsing
        self.stack = [] #hold traveresd nodes
        self.arr = [] #holds processed values
        self.ptr = -1

    def hasNext(self) -> bool:
        return self.stack or self.last or self.ptr < len(self.arr) - 1
        

    def next(self) -> int:
        #advance
        self.ptr += 1
        
        #if we are outside what is left
        if self.ptr == len(self.arr):
            #process all predecessors of the last node, i.e go as far left as you can
            while self.last:
                self.stack.append(self.last)
                self.last = self.last.left
            #finished decesding left, get the last processed node before we invoked next
            curr = self.stack.pop()
            self.last = curr.right
            
            #add this value to arr as processed
            self.arr.append(curr.val)
        
        return self.arr[self.ptr]
            
    def hasPrev(self) -> bool:
        #check we can return something from the parsed array
        return self.ptr > 0

    def prev(self) -> int:
        #go back
        self.ptr -= 1
        return self.arr[self.ptr]


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.hasNext()
# param_2 = obj.next()
# param_3 = obj.hasPrev()
# param_4 = obj.prev()

##################################
# 17APR22
# 897. Increasing Order Search Tree
###################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        '''
        we can just unpack using in order tarversal then rebuild
        '''
        self.dummy = TreeNode(-1)
        self.curr = self.dummy
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            new_node = TreeNode(node.val)
            self.curr.right = new_node
            self.curr = self.curr.right
            inorder(node.right)
        
        inorder(root)
        return self.dummy.right

#we can use the yeild function in python too
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        def inorder(node):
            if not node:
                return
            yield from inorder(node.left)
            yield node.val
            yield from inorder(node.right)
            
        ans = TreeNode(-1)
        curr = ans
        for v in inorder(root):
            curr.right = TreeNode(v)
            curr = curr.right
        
        return ans.right
        
##############################
# 17APR22
# 628. Maximum Product of Three Numbers
##############################
#close one, but cant combine this way
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        '''
        for an array of all nums > 0, the answer is trivial, just the top 3
        if i have negative numbers, i would only want to include the negative if it made a larger product
        
        i can pull the largest three, and the smallest three and find the combindations of size 3 that gives the largest
        this is constant
        '''
        N = len(nums)
        
        #size 3
        if N == 3:
            product = 1
            for num in nums:
                product *= num
            return product
        
        
        #find smalltest three and largest three
        nums.sort()
        smallest = nums[:3]
        largest = nums[-3:]
        
        candidates = smallest + largest
        
        self.prod = 0
        def rec(i,path):
            if len(path) == 3:
                #print(path)
                prod = 1
                for num in path:
                    prod *= num
                self.prod = max(self.prod, prod)
                print(prod,path)
                return
            for j in range(i,len(candidates)):
                path.append(candidates[j])
                rec(j+1,path)
                path.pop()
        rec(0,[])
        return self.prod

#sorting
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        '''
        sort and just check nums[0]*nums[1]*nums[n-1] or nums[n-3]*nums[n-2]*nums[n-1]
        '''
        N = len(nums)
        nums.sort()
        return max(nums[0] * nums[1] * nums[N - 1], nums[N - 1] * nums[N - 2] * nums[N - 3])

####################################
# 230. Kth Smallest Element in a BST
# 18APR22
####################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        '''
        return the kth smallest
        the array is 1 indexded
        would do an in order traversal and just retun once we hit k using a counter
        '''

        
        stack = []
        step = 1
        
        curr = root
        
        
        while curr != None or len(stack) > 0:
            while curr != None:
                stack.append(curr)
                curr = curr.left
            

            curr = stack.pop()
            if step == k:
                return curr.val
            step += 1
            curr = curr.right

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        '''
        recursive
        '''
        def inorder(node):
            if not node:
                return []
            left = inorder(node.left)
            right = inorder(node.right)
            return left + [node.val] + right
        
        return inorder(root)[k-1]

#################################
# 99. Recover Binary Search Tree
# 19APR22
#################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        only two nodes were swapped by mistake, so there is already a mistake -> no special cases for not having a mistake
        in order traversal should give nodes in order, but two elements will be out of place
        i can get the inorder traversal, mark the problematic nodes
        the pass the tree again updating the values
        '''
        in_order_vals = []
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            in_order_vals.append(node.val)
            inorder(node.right)
        
        
        def preorder(node,count):
            #donesn't matter what order travesal, since we want to touch all nodes
            if not node:
                return
            if node.val == first or node.val == second:
                if node.val == first:
                    node.val = second
                else:
                    node.val = first
                count -= 1
                if count == 0:
                    return
            
            preorder(node.left,count)
            preorder(node.right,count)
            
        inorder(root)
        #find problematic nodes
        first = None
        second = None
        for i in range(len(in_order_vals)-1):
            if in_order_vals[i] > in_order_vals[i+1]:
                first = in_order_vals[i+1]
                if not second:
                    second = in_order_vals[i]
                else:
                    break
        
        preorder(root,2)
                
#iterative one pass
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        we can find identify the swapped nodes by keeping track if hthe pred in the in-order traversal (i.e the predecessor of the current node)
        we compare with the curent value
        if the current node value is smaller than it spred, this is a problematic node
        since there are only two swapped nodes, we could break after finding the two
        '''
        stack = []
        first = None
        second = None
        pred = None
        
        while root != None or len(stack) > 0:
            while root:
                stack.append(root)
                root = root.left
            
            #find current
            root = stack.pop()
            #check, this should be smaller
            if pred and root.val < pred.val:
                #first problematic node
                first = root
                if second == None:
                    second = pred
                else:
                    break
            #continue traversal
            pred = root
            root = root.right
        
        #swap
        first.val, second.val = second.val, first.val

#recursive one pass
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.first = None
        self.second = None
        self.pred = None
        
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            if self.pred and node.val < self.pred.val:
                self.first = node
                if self.second == None:
                    self.second = self.pred
                else:
                    return
            
            self.pred = node
            inorder(node.right)
        
        inorder(root)
        #swap
        self.first.val,self.second.val = self.second.val,self.first.val

#inorder traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        this is good time to go over Morris Traversal
        the idea behind Morris Traversal is to linke the current node and its predessor (i.e pred.right = root)
        
        so one starts from the nodes, computes its pred and verifies if the link is present:
            if there is no link, set it and go left
            if there is a link, break and go right
            
        note inorder pred, go left once and far right
        '''
                # predecessor is a Morris predecessor. 
        # In the 'loop' cases it could be equal to the node itself predecessor == root.
        # pred is a 'true' predecessor, 
        # the previous node in the inorder traversal.
        x = y = predecessor = pred = None
        
        while root:
            # If there is a left child
            # then compute the predecessor.
            # If there is no link predecessor.right = root --> set it.
            # If there is a link predecessor.right = root --> break it.
            if root.left:       
                # Predecessor node is one step left 
                # and then right till you can.
                predecessor = root.left
                while predecessor.right and predecessor.right != root:
                    predecessor = predecessor.right
                # set link predecessor.right = root
                # and go to explore left subtree
                if predecessor.right is None:
                    predecessor.right = root
                    root = root.left
                # break link predecessor.right = root
                # link is broken : time to change subtree and go right
                else:
                    # check for the swapped nodes
                    if pred and root.val < pred.val:
                        y = root
                        if x is None:
                            x = pred 
                    pred = root
                    
                    predecessor.right = None
                    root = root.right
            # If there is no left child
            # then just go right.
            else:
                # check for the swapped nodes
                if pred and root.val < pred.val:
                    y = root
                    if x is None:
                        x = pred 
                pred = root
                
                root = root.right
        
        x.val, y.val = y.val, x.val