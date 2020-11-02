################################
#Number of Recent Calls 10/01/20
################################
#TLE exceed
class RecentCounter(object):

    def __init__(self):
        #store pings
        self.all_pings = []
        

    def ping(self, t):
        """
        :type t: int
        :rtype: int
        """
        #first append
        self.all_pings.append(t)
        #get the range
        start,end = t-3000,t
        #return call
        in_range = 0
        #traverse pings
        for ping in self.all_pings:
            if start <= ping <=end:
                in_range += 1
        return in_range

from collections import deque
class RecentCounter(object):

    def __init__(self):
        #store pings.using q, add right, popleft
        self.all_pings = deque()
        self.count = 0
        

    def ping(self, t):
        """
        :type t: int
        :rtype: int
        """
        #use q, and keep popping off pings that aren't in the arange
        self.all_pings.append(t)
        self.count += 1
        while self.all_pings and self.all_pings[0] < t - 3000:
            self.all_pings.popleft()
            self.count -= 1
        return self.count
        

############################
#Combindation Sum 10/02/2020
############################
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        '''
        good review problem, lets try the recursive solution first
        we can build up a combindation of numbers if we keep building but decremeting the targe as go along, we then check if the combindations meet the taret
        add to results outside
        '''
        results = []
        
        def recurse(comb,idx,target):
            if target <=0:
                if target == 0 and comb:
                    results.append(comb)
                return
            
            for i in range(idx,len(candidates)):
                num = candidates[i]
                recurse(comb+[num],i,target-num)
        
        recurse([],0,target)
        return results

class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        '''
        official backtracking solution,lets make sure i get it this stime
        recall backtracking, building up a solution, but then abandoning that solution when it does not meet the constraint
        build up a solution recursively, but traverse the canddiates in order!
        we can then treat this is a DFS backtracking solution
        we deinfe backtrack(remain,comb,start) which populates results outside the fal
            comb is the current build
            remain because we cant look back
            and start (this moves too)
        the base case
        if reamin == 0 we're done
        if less than 0, we've exceeded the target
        '''
        
        results = []
        def backtrack(remaining, comb,start):
            #comb is init as an empty list
            if remaining == 0: #found a set
                results.append(list(comb))
                return
            elif remaining < 0:
                return #abandon this
            
            #recurse
            for i in range(start,len(candidates)):
                comb.append(candidates[i])
                #give the current number a chance to be seen again
                backtrack(remaining - candidates[i],comb,i)
                #backtrack
                comb.pop()
                
        #invoke
        backtrack(target,[],0)
        return results

class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        '''
        dp solution
          build up a list of lists in a 2d array
        X         0     1       2       3       4               5   
        [1]       []   [1]    [1,1]   [1,1,1]  [1,1,1,1]        [1,1,1,1,1]
        [2]       []   [0]     [2]    [2,1,1]  [[2,1,1],[2,2]]  [[2,2,1],[1]]
        [3]       [] 
        '''
        dp = [[] for _ in range(target+1)]
        
        for c in candidates:
            for i in range(1,target+1):
                if i < c:
                    continue
                elif i == c:
                    dp[i].append([c])
                #now we can more the candidate when c is greater than i
                else:
                    for foo in dp[i-c]:
                        dp[i].append(foo + [c])
        return dp[target]
        

####################################
#Maximun Distance in Arrays 10/02/20
####################################
#120 of 124, TLE EXCEED! Niiiceee! now lets speed it up
class Solution(object):
    def maxDistance(self, arrays):
        """
        :type arrays: List[List[int]]
        :rtype: int
        """
        '''
        traverse the arrays pulling the min and max
        this gives as len(arrays)*2 numbers
        return abs(min(mins) - max(maxs))
        [
        [1,4],
        [0,5],
        [2,8],
        ]
        
        mins = [1,0,2]
        maxs = [4,5,8]
        
        if my indices are 0,1,2 pairs are then
        0,1
        0,2
        1,2
        
        '''
        mins, maxs = [], []
        for arr in arrays:
            mins.append(min(arr))
            maxs.append(max(arr))
        
        #now be careful to pick a unique pair, what a wrench, since only have two elements now
        #mins and maxes can only be taken from two different arrays, if i take min[i] i cannot take max[i] same goes with min[j] max[j]
        max_distance = float('-inf')
        for i in range(0,len(arrays)):
            for j in range(i,len(arrays)):
                if i == j:
                    continue
                #min from i max from j
                first = abs(maxs[j] - mins[i])
                #second, min from j max from i
                second = abs(maxs[i] - mins[j])
                max_distance = max(max_distance,first,second)
        return max_distance
                
        
        #print mins, maxs

class Solution(object):
    def maxDistance(self, arrays):
        """
        :type arrays: List[List[int]]
        :rtype: int
        """
        '''
        traverse the arrays pulling the min and max
        this gives as len(arrays)*2 numbers
        return abs(min(mins) - max(maxs))
        [
        [1,4],
        [0,5],
        [2,8],
        ]
        
        mins = [1,0,2]
        maxs = [4,5,8]
        
        if my indices are 0,1,2 pairs are then
        0,1
        0,2
        1,2
        
        '''

        mins, maxs = [], []
        for arr in arrays:
            mins.append(min(arr))
            maxs.append(max(arr))
        
        #now be careful to pick a unique pair, what a wrench, since only have two elements now
        #well we already have the mins and maxs!
        #i could sort them in a tuple keeping track on the indices
        #then traverse one more time making sure the indices do not match taking the min and max diff
        
        #add in indices
        for i in range(0,len(arrays)):
            mins[i] = (i,mins[i])
            maxs[i] = (i,maxs[i])
            
        #sort both mins and maxs on the second element, mins acending, maxs decending
        mins = sorted(mins, key = lambda x:x[1])
        maxs = sorted(maxs, key = lambda x:x[1], reverse=True)
        
        #traverse both lists only taking the diff the indices are different
        max_diff = float('-inf')
        for i in range(0,len(arrays)):
            if maxs[i][0] != mins[i][0]:
                max_diff = max(max_diff, abs(maxs[i][1] - mins[i][1]))
        #if its still -inf i need a value to return
        if max_diff == float('-inf'):
            return mins,maxs
        return max_diff
                
        
        #print mins, maxs

#linear time
class Solution(object):
    def maxDistance(self, arrays):
        """
        :type arrays: List[List[int]]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/maximum-distance-in-arrays/solution/
        the arrays are already sorted, so we only need to consider extremes
        also, points being considere should not belong to the same array
        for an array a or b we can fin the max of a[n-1] - b[0] and b[m-1] - a[0]
        where n am are the lenghts a,b
        but we dont need to compare all possible pairs
        instead keep on travering over teh arrays and keep tracking of the max distance so found so far
        
        to do this, keep tracking of the element with the min_val and one with the max_val found so far
        these new extreme values can be treated as if the represent the extremes of a cumitlative array
        
        for every new array find the distance a[n-1] - min_val and max_val - a[0]
        note that the max distace found until now needs not be always contributed by the end points of the distance being max_val and min_val
        
        but such points could help in maximizing the distance in the future.
        so we need to keep track of these maxs and mins along with the max found so far
        '''
        #max_distance init
        max_distance = 0
        #get min val from first array
        min_val = arrays[0][0]
        #max from second
        max_val = arrays[0][-1]
        #start with the second
        for i in range(1,len(arrays)):
            #but first update the current max_distance
            max_distance = max(max_distance, abs(arrays[i][-1]-min_val),abs(max_val - arrays[i][0]))
            
            #get the minimum
            min_val = min(min_val,arrays[i][0])
            #get the max
            max_val = max(max_val, arrays[i][-1])
        
        return max_distance


####################################
#K-diff Pairs in an array 10/03/2020
####################################
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        naive way
        sort and enumerate all possible pairs, but only go when i != j and when j isn't the same as j-1
        '''
        sorted_nums = sorted(nums)
        result = 0
        
        for i in range(0,len(sorted_nums)):
            if i > 0 and sorted_nums[i] == sorted_nums[i-1]:
                continue
            for j in range(i + 1,len(sorted_nums)):
                if j > j + 1 and sorted_nums[j] == sorted_nums[j-1]:
                    continue
                if abs(sorted_nums[j] - sorted_nums[i]) == k:
                    result += 1
        return result
        

class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        we can use a two pointer approach for this problem
        first we init l pointer to the frist and r t the second
        take diff every time
        if diff is less than k, move the right, if pointers are the same move right alos
        if it greater than k, we move the elft
        if it is exactly k, we found a pair
        the idea is to keep extending or contracting the range
        there is one more case, we need to make sure we do not capture any duplicates
        whenver we have a pair who's difference is k we keep incrementing the left pointer as as long the left pointer's previous does not match its current
        '''
        N = len(nums)
        nums = sorted(nums)
        left, right = 0,1
        result = 0
        while left < N and right < N:
            if left == right or nums[right] - nums[left] < k:
                #extend the range 
                right += 1
            elif nums[right] - nums[left] > k:
                #contract the range
                left += 1
            else:
                left += 1
                result += 1
                #making sure we are not duplicatin pairs
                while left < N and nums[left] == nums[left-1]:
                    left += 1
                    
        return result

from collections import Counter
class Solution(object):
    def findPairs(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        get the frequency of each num
        traverse the keys in the hash
        and if key plus k in hash, increment the results
        keep in mind the case [1,1,1,1,] k = 0
        1:4, we need to capture this as one only!
        '''
        
        result = 0
        nums = Counter(nums)
        
        for num in nums:
            if k >0 and num + k in nums:
                result += 1
            elif k == 0 and nums[num] > 1:
                result += 1
        return result


####################################
#Remove Covered Intervals 10/04/2020
####################################
class Solution(object):
    def removeCoveredIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        to check coverage, an current intervals start must be >= candidate start and <= candidate end
        ------
           ------
        ------
             ---
        what if i sort on the starting points?
        [[1,4],[3,6],[2,8]]
        [[1,4],[2,8],[3,6]]
        
        '''
        N = len(intervals)
        intervals = sorted(intervals, key = lambda x: x[0])
        min_start = intervals[0][0]
        max_end = intervals[0][1]
        
        for start,end in intervals[1:]:
            min_start = max(min_start,start)
            max_end = max(max_end,end)
            #removing covered intervals
            if start >= min_start and end <= max_end:
                intervals.remove([start,end])
        return len(intervals

class Solution(object):
    def removeCoveredIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        to check coverage, an current intervals start must be >= candidate start and <= candidate end
        ------
           ------
        ------
             ---
        what if i sort on the starting points? sort on start first, the on end
        [[1,4],[3,6],[2,8]]
        [[1,4],[2,8],[3,6]]
        
        '''
        N = len(intervals)
        intervals.sort(key = lambda x: (x[0], -x[1]))
        
        prev_end = 0

        results = []
        for start,end in intervals:
            if end > prev_end:
                results.append([start,end])
                prev_end = end
        return len(results)       

class Solution(object):
    def removeCoveredIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        from tim, just keep adding intervals that aren't covered and return the length of the results
        '''
        if not intervals:
            return 0
        intervals.sort()
        output = []
        output.append(intervals[0])
        
        for start,end in intervals[1:]:
            #not covered
            if start > output[-1][1]:
                output.append([start,end])
            else:
                #we orderd on start, now if the current end is smaller than the prev_end
                #or start is the same, update the end
                if end <= output[-1][1] or start == output[-1][0]:
                    output[-1][1] = max(output[-1][1],end)
                else:
                    #just keep adding
                    output.append([start,end])
        return len(output)

###################################
#Complement of Base 10 Integer
###################################
class Solution(object):
    def bitwiseComplement(self, N):
        """
        :type N: int
        :rtype: int
        """
        '''
        generate binary rep
        flip the bits
        return the complement
        '''
        if N == 0:
            return 1
        def rec_binary(num, rep):
            if num == 0:
                return
            rep.append(str(divmod(num,2)[1]))
            rec_binary(divmod(num,2)[0],rep)
            return rep[::-1]
        
        binary_rep =  rec_binary(N,[])
        
        #flip each and return the sum 
        number = 0
        for i in range(0,len(binary_rep)):
            number += (1-int(binary_rep[i]))*2**(len(binary_rep)-1-i)
        return number


import math
class Solution(object):
    def bitwiseComplement(self, N):
        """
        :type N: int
        :rtype: int
        """
        '''
        we want the this:
        \sum_{i=0}^{N} (1-x_{i})*2^{N-1}
        which is
        \sum_{i=0}^{N} 2^{N-1} - \sum_{i=0}^{N} x_{i}*2^{N-1}
        which is just the largest number created from the original number - number
        
        just take ceiling(log(N)) - N
        '''
        if N == 0:
            return 1
        length = math.ceil(math.log(N,2))
        max_base_2 = 0
        for i in range(int(length)):
            max_base_2 += 2**(int(length)-1-i)
        return max_base_2 - N


############################################
#Insert into a Binary Search Tree 10/06/2020
############################################
#6 of 35! so close
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        '''
        in a BST i can only go left or right, we can return any BST, so lets' just add it as a leaf
        
        '''
        def insert_into(node,target):
            #base case
            if not node.left and not node.right:
                if target < node.val:
                    node.left = TreeNode(val=target)
                else:
                    node.right = TreeNode(val=target)
            elif target < node.val:
                insert_into(node.left, target)
            elif target > node.val:
                insert_into(node.right,target)
        insert_into(root,val)
        return root

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        '''
        almost had it! keep this in mind though with the official LC solution
        recall BST insertion is logN, and searching is also logN
        if val > node.val, go right
        if val < node.val, go left
        '''
        def insert_into(node, target):
            if not node:
                return TreeNode(target)
            if target > node.val:
                node.right = insert_into(node.right, target)
            else:
                node.left = insert_into(node.left,target)
            return node
        
        return insert_into(root,val)


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        #edge case
        if not root:
            return TreeNode(val)
        
        def insert(node,val):
            if val > node.val:
                #base case 
                if not node.right:
                    node.right = TreeNode(val)
                else:
                    insert(node.right,val)
            else:
                if not node.left:
                    node.left = TreeNode(val)
                else:
                    insert(node.left,val)
            return node
        
        return insert(root,val)
        

#########################
#Rotate List 10/07/2020
#########################
#TLE good idea but slow
from collections import deque
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        '''
        for each value of k, you take the last element and move it to the front
        repeat this k times, since this is a linked list i can't access the prev
        it might be better to dump into a list, then pop append
        '''
        if not head:
            return head
        values = []
        node = head
        while node:
            values.append(node.val)
            node = node.next
        values = deque(values)
        
        #pop append
        while k > 0:
            move = values.pop()
            values.appendleft(move)
            k -= 1
        #reconstruct
        node = head
        while values:
            temp = values.popleft()
            node.val = temp
            node = node.next
        return head


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        '''
        since the nodes are linked, we can connect head to tail
        break the ring after the new tail  and just in front of the new head
        travese the linked list keeping track of the number of nodes, when i get to the end, reconnect it back to the tail
        the linked list is fixed at length n
        we break the linked list at n - k%n - 1 (its just a circle)
        '''
        #edge cases
        if not head:
            return None
        if not head.next:
            return head
        
        #make the linked list into a cycle
        old_tail = head
        n = 1
        while old_tail.next:
            old_tail = old_tail.next
            n += 1
        #close it!
        old_tail.next = head
        #find the new tail at (n-k%n-1)
        #add the new head
        new_tail = head
        for i in range(0,n-k%n-1):
            new_tail = new_tail.next
        #break it
        new_head = new_tail.next
        #add non
        new_tail.next = None
        
        return new_head


class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        '''
        https://www.youtube.com/watch?v=wU-Sr9RUISQ&ab_channel=TimothyHChang
        we don't need to keep moving for each k, just found the point we need to be after doing k
        step 1: count the nodes and make circule
        step 3: prev node point to null new node be head
        '''
        if not head or not head.next:
            return head
        
        cur = head
        N = 1
        while cur.next: #so long as there is something to be pointed to
            N += 1
            cur = cur.next
        #weve gone all the way down, now reconnect
        cur.next = head 
        
        #we need to go the point in the list after moving k times, this is just k%N
        M = N - k%N
        i = 0 
        cur = head
        while i < M:
            #give reference to the previous pointer
            prev = cur
            cur = cur.next
            i += 1
        prev.next = None
        #get the newhead
        head = cur
        return head

#########################
#Binary Search 10/08/2020
#########################
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return -1
        N = len(nums)
        lo = 0
        hi = N -1
        while lo <= hi:
            mid = lo + (hi-lo) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                hi = mid - 1
            else:
                lo = mid + 1
        return -1

###############################
#Serialize and Deserialize BST 10/09/2020
################################
#wohoo
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        '''
        i can dump the string values as an in order list, and then when i derialize, reconstruct the tree
        i need in order and pre order order
        then build
        '''
        self.string_tree_inorder = ""    
        self.string_tree_preorder = ''
        def dfs_inorder(node):
            if node:
                dfs_inorder(node.left)
                self.string_tree_inorder += str(node.val)+'.'
                dfs_inorder(node.right)
        def dfs_preorder(node):
            if node:
                self.string_tree_preorder += str(node.val)+'.'
                dfs_preorder(node.left)
                dfs_preorder(node.right)
        if not root:
            self.string_tree_inorder += "[]"
        else:
            dfs_inorder(root)
            dfs_preorder(root)
        return self.string_tree_inorder+"+"+self.string_tree_preorder
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        self.parsed = data.split('+')
        self.inorder,self.preorder = self.parsed[0].split('.')[:-1],self.parsed[1].split('.')[:-1]
        if '[]' in self.inorder:
            return []
        else:
            self.inorder = [int(foo) for foo in self.inorder]
            self.preorder = [int(foo) for foo in self.preorder]
        
        #now rebuild from inorder and preorder
        def helper(preorder,inorder):
            if len(preorder) == 0:
                return None
            #recall preorder always has root at the first element
            root = TreeNode(preorder[0])
            #give reference to the Index
            rootIndex = inorder.index(preorder[0])
            
            #in order gives the nodes that belong to the left and right sides of the tree
            #get both
            leftInorder = inorder[:rootIndex]
            rightInorder = inorder[rootIndex + 1:]
            
            #in order does not give child nodes correctly, so get them from preorder
            leftPreorder = preorder[1:rootIndex + 1]
            rightPreorder = preorder[rootIndex + 1:]
            
            root.left = helper(leftPreorder,leftInorder)
            root.right = helper(rightPreorder, rightInorder)
            return root
        return helper(self.preorder, self.inorder)
        
        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# tree = ser.serialize(root)
# ans = deser.deserialize(tree)
# return ans

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        #preorder list first
        self.list = []
        def dfs(node):
            if not node:
                return
            self.list.append(node.val)
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ",".join(map(str,self.list))
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if not data:
            return None
        lst = [int(d) for d in data.split(",")]
        #now recreate tree using preoder list
        #keeing track of upper and lower bounds from the parent node
        def build_tree(lst,lower,upper):
            if not lst:
                return None
            if not lower <=lst[0] <= upper:
                return None
            #get first
            cand = lst.pop(0)
            root = TreeNode(cand)
            #from the parent node, the upper will be its val for the left side
            #from the parent node, the lower will be its val for the right side
            root.left = build_tree(lst,lower,root.val)
            root.right = build_tree(lst,root.val, upper)
            return root


########################################
#Find Minimum Arrows to Burst Balloons
#######################################

class Solution(object):
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        '''
        this is in interval problem, min arrows means use dp
        we want to maximize the number or merges
        the naive way is to check the number of ballons i can pop at each x point
        sort on start point
        [[10,16],[2,8],[1,6],[7,12]]
        [[1,6],[2,8],[7,12],[10,16]]
        [[1,8]]
        find the interval that contains the starts and ends!
        which means i don't have to check all x points
        we could sort on the end point? why?
            to have a start copord smalle than a current end point, it means we could have popped them toegether
            to have a start coord larger than the end, we can't pop them together so we increase the nummber of arrows here
        keep track of the current enf o the ballong, ignore all ballons which end before it
        one current baloon is ended (the next balloon starts after current end) we increase the number of arrows
        '''
        if not points:
            return 0
        
        #sort on end
        points.sort(key = lambda x: x[1])
        
        current_end = points[0][1]
        arrows = 1 #starting with one arrow
        for start,end in points:
            if current_end < start:
                arrows += 1
                #update new end
                current_end = end
        return arrows

class Solution(object):
    def findMinArrowShots(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        '''
        https://www.youtube.com/watch?v=_WIFehFkkig&ab_channel=TimothyHChang
        sort in the starting point
        '''
        if not points:
            return 0
        points.sort()
        
        arrows = 1
        #keep tracking of prev_start and end
        prev_start = points[0][0]
        prev_end = points[0][1]
        
        for start,end in points[1:]:
            #if i can't pop the next balloon
            if start > prev_end:
                arrows += 1
                #update both
                prev_start,prev_end = start,end
            else:
                #in this its less than
                prev_end = min(prev_end, end)
        return arrows
        

#####################################
#Remove Duplicate Letters 10/11/2020
####################################
from collections import Counter
class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        get the counts of each char, 1 pass
        traverse along the string and remoive each string not hvaing a count of only 1
        '''
        count = Counter(s)
        results = []
        for char in s:
            if count[char] == 1:
                results.append(char)
            count[char] -= 1
        return "".join(results)

from collections import Counter
class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        https://leetcode.com/problems/remove-duplicate-letters/solution/
        lexographical means alphabetical order, but in relation to the unique chars in the strin
        strings are compared from the first char to the last one
        which string is greater depends on the comparison between the 'first unequal corresponding char'
        as a result any string beginning with a willes be laa than any string beginning with b duh!
        OPTIMAL SOLUTION will have the smallest chars as early as possible
        1. leftmost letter in our solution will be the smallest letter such taht the suffix from the letter contains every other
        this is because we know that the solution must have one copy of every letter,
        and we know that the solution will have the smallestg lexogroh. smallest left most char!
        if there are multiple letters, then we picj the leftmost one because this gives us more options
        2. as we iterate over the string, if a char at i > i + 1 and another i in set exists, deleting i leads to the optimal solution
        chars that come later in the string i dont matter in tis cal because i is in a more significant spont.
        even if char i + 1 isn't the ebst yet, we can always replace it for a smaller char down the line
        ALGO:
        chose the smalelst char such that its suffix, contains at least one copy of every character in the string
        
        '''
        #once we fix the smallest left most char, the following suffic (after the fixed positions) must contain all unique chars
        #find pos - the index of the elftmost letter in our solution
        #we create a counter and the end iteration once the suffix does not have each uniuqe char
        #pos will be the index of the smalelst char we encoutner before the iteration ends
        c = Counter(s)
        pos = 0
        for i in range(len(s)):
            #finding smallest start
            if s[i] < s[pos]:
                pos = i
            #using up char
            c[s[i]] -= 1
            if c[s[i]] == 0:
                break
        #our answer i the left most letter plus the recusrive call on the reminader of the string
        #remove further occurences of s[pos] to ensure no duplicates
        return s[pos] + self.removeDuplicateLetters(s[pos:].replace(s[pos],"")) if s else ""


class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        extension from the first solution
        keep track in stack
        at each iteration we add the current char to the solution if it hasn't already been used,
        we tro to remove as many chars a possible
        delte when char is great than current chars
        the char can be removed because it occurs later one
        '''
        results = []
        seen = set()
        #finding last char occurence position! i like this
        last_occur = {c:i for i,c in enumerate(s)}
        
        for i,c in enumerate(s):
            #only try to add if we haven't seen it
            if c not in seen:
                #conditions, it existis, is greater than than c , so we can make it smaller, not the alte occurence
                while results and c < results[-1] and i < last_occur[results[-1]]:
                    #remove from the top of the stack
                    seen.discard(results.pop())
                seen.add(c)
                results.append(c)
        return "".join(results)


#########################
#Buddy Strings 10/12/2020
##########################
class Solution(object):
    def buddyStrings(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        '''
        if the char counts are the same, its possible
        well if they are different at the points,they must be the same anywhere else
        find where they are different, then swap and check
        '''
        if not A or not B:
            return False
        
        if len(A) != len(B):
            return False
        
        #if strings are equal, check that there is a double to swap, like ab, ab
        if A == B:
            return len(A) - len(set(A)) >= 1
        
        #find where they are different
        diff = []
        N = len(A)
        for i in range(0,N):
            if A[i] != B[i]:
                diff.append(i)
                if len(diff) > 2:
                    return False
                
        #not exaclty two
        if len(diff) != 2:
            return False
        
        #check if it can be swapped
        if A[diff[0]] == B[diff[1]] and A[diff[1]] == B[diff[0]]:
            return True
        
        return False

class Solution(object):
    def buddyStrings(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: bool
        """
        #get and compare countes
        cA, cB = Counter(A),Counter(B)
        if cA != cB:
            return False
        
        diff = sum([1 for i in range(len(A)) if A[i] != B[i]])
        
        if diff == 2:
            return True
        elif diff == 0:
            return any([cnt > 1 for char,cnt in cB.items()]) #if they are they same, check if there are any doubles
        else:
            return False
        

##########################################
#Two Sum III - Data Structure Design
########################################
#12 of 17
class TwoSum(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.nums = []
        

    def add(self, number):
        """
        Add the number to an internal data structure..
        :type number: int
        :rtype: None
        """
        self.nums.append(number)
        

    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        #the usual two sum here, first pass finding difference from val
        #second pass finind the pair
        complements = {num:value - num for num in self.nums}
        #second pass
        for num in self.nums:
            if complements[num] in self.nums:
                return True
        return False
        


# Your TwoSum object will be instantiated and called as such:
# obj = TwoSum()
# obj.add(number)
# param_2 = obj.find(value)


class TwoSum(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        #keep track of num counts, in the case num is equal to complement
        self.num_counts = {}
        

    def add(self, number):
        """
        Add the number to an internal data structure..
        :type number: int
        :rtype: None
        """
        if number in self.num_counts:
            self.num_counts[number] += 1
        else:
            self.num_counts[number] = 1
        

    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        #the usual two sum here, first pass finding difference from val
        #second pass finind the pair
        #In a particular case, where the number and its complement are equal, we then need to check if there exists at least two copies of the number in the table.
        for num in self.num_counts.keys():
            #get the complement 
            complement = value - num
            if num != complement:
                #check if complement in hash
                if complement in self.num_counts:
                    return True
            elif self.num_counts[num] > 1:
                return True
        return False
        


# Your TwoSum object will be instantiated and called as such:
# obj = TwoSum()
# obj.add(number)
# param_2 = obj.find(value)


#####################
#Sort List 10/13/2020
#####################
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        the first way, pull the node values sorting and recreating the list
        '''
        cur = head
        temp  = []
        
        while cur:
            temp.append(cur.val)
            cur = cur.next
        
        temp.sort()
        
        dummy = ListNode(0)
        cur = dummy
        
        for v in temp:
            cur.next = ListNode(v)
            cur = cur.next
            
        return dummy.next

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        bubble sort n squared but constant space
        '''
        N = 0
        cur = head
        while cur:
            N += 1
            cur = cur.next
            
        i = 0
        while i < N:
            j = 0
            cur = head
            while j < N - i and cur.next:
                if cur.val > cur.next.val:
                    cur.val, cur.next.val = cur.next.val, cur.val
                cur = cur.next
                j += 1
            i += 1
        
        return head
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        i could dump all the node vals into another list and then sort, then recreate the linked list
        but lets walk through the merge sort in place solution from leet
        just go over top down merge sort, don't worry about O(1) space
        divide phase: split into nodes of length 1 recursively
        conquer phase: solve each sorting subproblem
        combine phase: meger the linked lists!
        RECURSIVE SPLIT:    
            keep splitting until only one node
        MERGE:
            with the nodes now split treat this as a merge linked list problem
            dummy head, two pointers
            keep advacning pointer as you make a connection to the next greater node val
        '''
        #define the mid function first
        def getMid(head):
            #two pointers, fast is two steps ahead slow is not
            slow,fast = head,head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            #assign the mid
            mid = slow.next
            slow.next = None #marks the end of the linked list
            return mid
        
        #merge - conqure step, the sub problem has two linked lists
        def merge(head1,head2):
            dummy = tail = ListNode(None)
            while head1 and head2:
                if head1.val < head2.val:
                    tail = head1
                    tail.next = head1
                    head1 = head1.next #moving it up
                else:
                    tail = head2
                    tail.next = head2
                    head2 = head2.next
            #final node to end it with node
            tail.next = head1 or head2
            return dummy.next
        
        #inner function for recusrive call
        def recursive_sort(head):
            if not head or not head.next:
                return head
            mid = getMid(head)
            left = recursive_sort(head)
            right = recursive_sort(mid)
            return merge(left,right)
        
        return recursive_sort(head)


class Solution(object):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        i could dump all the node vals into another list and then sort, then recreate the linked list
        but lets walk through the merge sort in place solution from leet
        just go over top down merge sort, don't worry about O(1) space
        divide phase: split into nodes of length 1 recursively
        conquer phase: solve each sorting subproblem
        combine phase: meger the linked lists!
        RECURSIVE SPLIT:    
            keep splitting until only one node
        MERGE:
            with the nodes now split treat this as a merge linked list problem
            dummy head, two pointers
            keep advacning pointer as you make a connection to the next greater node val
        '''
        def sortList(self, head):
            if not head or not head.next: return head
            mid = self.getMid(head)
            left = self.sortList(head)
            right = self.sortList(mid)
            return self.merge(left, right)

        def getMid(self, head):
            slow, fast = head, head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            mid = slow.next
            slow.next = None
            return mid

        def merge(self, head1, head2):
            dummy = tail = ListNode(None)
            while head1 and head2:
                if head1.val < head2.val:
                    tail.next, tail, head1 = head1, head1, head1.next
                else:
                    tail.next, tail, head2 = head2, head2, head2.next

            tail.next = head1 or head2
            return dummy.next


##########################
#House Robber II 10/14/2020
##########################
###TLE Exceed 52/74
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        [1,2,3,1]
        if i rob the first house, i cannot rob the last house
        from the hint, just call the recurive function twice
        1 or n -1
        2 or n
        3 or
        '''
        if not nums:
            return 0
        
        elif len(nums) < 3:
            return max(nums)
        
        N = len(nums)
        #there are two cases, to think about, if i am at the beginning (0 or -1)
        #or at the end (0 or -1)
        
        def rec_max(k,nums):
            if k == 0:
                return nums[0]
            elif k == 1:
                return max(nums[1],nums[0])
            return max(rec_max(k-2,nums)+nums[k],rec_max(k-1,nums))
        
        return max(rec_max(N-2,nums[:-1]),rec_max(N-2,nums[1:]))


class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        [1,2,3,1]
        if i rob the first house, i cannot rob the last house
        from the hint, just call the recurive function twice
        1 or n -1
        2 or n
        3 or
        '''
        if not nums:
            return 0
        
        elif len(nums) < 3:
            return max(nums)
        
        def calculate(houses):
            N = len(houses)
            money = [0]*N
            
            money[0] = houses[0]
            money[1] = houses[1]
            
            for i in range(2,N):
                money[i] = max(money[i-2] + houses[i],money[i-1])
            return money[-1]
        
        return max(calculate(nums[:-1]),calculate(nums[1:]))
            

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        just solve the original house robber problem twice from 1 to the second last
        and 2 to the last
        recall the original house robber problem
        1,2,3
        max(1+3, or 2)
        i can define a current max and the total max
        0,0
        define a prevmax and a total max
        f(k) = max(f(k – 2) + Ak, f(k – 1))
        '''
        def rob_one(nums):
            prevMax = 0
            currMax = 0
            for num in nums:
                #give reference to old currMax before updating
                #think of like thieves
                #t1 goes in and takes the max, but will leave a note to t2
                #t2 gets the note after t1 takes the max
                localMax = currMax
                currMax = max(currMax,prevMax+num)
                prevMax = localMax
            return currMax
        
        if len(nums) == 0 or not nums:
            return 0
        
        elif len(nums) == 1:
            return nums[0]
        
        return max(rob_one(nums[:-1]),rob_one(nums[1:]))


##############################
#Rotate Array 10/15/2020
##############################
from collections import deque
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        '''
        just turn this into a deque,
        then pop and appendlift k times
        '''
        nums_d = deque(nums)
        i = 0
        while i < k:
            temp = nums_d.pop()
            nums_d.appendleft(temp)
            i += 1
        for i in range(0,len(nums)):
            nums[i] = nums_d[i]

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        '''
        not totally O(1) space, but closer
        i can treat this is a cycle
        
        '''
        if len(nums) == 1:
            return nums[0]
        
        N = len(nums)
        #move to point
        M = (k % N) + 1
        i = 0
        new_nums = []
        while i < N:
            new_nums.append(nums[(M+i) % N])
            i += 1
        for i in range(N):
            nums[i] = new_nums[i]

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        '''
        i can just move it over by k, and mod it with lenght, but using an extra array
        '''
        N = len(nums)
        temp = [0]*N
        for i in range(N):
            temp[(i+k) % N] = nums[i]
            
        nums[:] = temp


class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        '''
        the trick is to reverse all the elements
        reverse the first k
        reverse k + 1 to the end IDK why?!?!?
        '''
        def reverse(nums, start,end):
            while start < end:
                nums[start],nums[end] = nums[end],nums[start]
                start += 1
                end -= 1
                
        N = len(nums)
        #find pivit point
        M = k % N
        #reverse all
        reverse(nums,0,N-1)
        #reverse up to k
        reverse(nums,0,M-1)
        #after k
        reverse(nums,M,N-1)


####from Tim 
x = k % len(nums)

temp = nums[-x:]
del nums[-x:]
nums[:0] = temp
        
##############################
#Search a 2D matrix 10/16/2020
##############################
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        just concat the lists and do binary search
        '''
        if not matrix:
            return False
      
        #concat
        new_matrix = []
        for m in matrix:
            new_matrix += m
        
        #binary seach
        lo = 0
        hi = len(new_matrix) -1
        
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if new_matrix[mid] == target:
                return True
            elif new_matrix[mid] > target:
                hi = mid - 1
            else:
                lo = mid + 1
        return False

#in place
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        '''
        instead of concating the matrix
        realize the the idx position can be be turned into an i,j position
        row = idx // 2
        col = idx % n
        keep the matrix but convert the idx to row and col
        '''
        if not matrix:
            return False
        
        cols = len(matrix[0])
        rows = len(matrix)
        lo = 0
        hi = cols*rows - 1
        
        while lo <= hi:
            mid_idx = lo +(hi - lo) // 2
            mid_element = matrix[mid_idx // cols][mid_idx % cols]
            #check
            if mid_element == target:
                return True
            else:
                if mid_element > target:
                    hi = mid_idx - 1
                else:
                    lo = mid_idx + 1
        return False


####################################
# Repeated DNA Sequences 10/17/2020
####################################
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        '''
        string matching, Rabin Karp of KMP, i dont remember this at all....
        the naive way would be to first extract all unique pattens of length 10 put into a hash
        as a traverse increase the count by 1 if it pattern exists
        finllay travese hash and output pattens that have count larger than 0
        
        '''
        if not s:
            return []
        pattern_counts = collections.defaultdict(int)
        for i in range(0,len(s)-10+1):
            cand = s[i:i+10]
            if cand in pattern_counts:
                pattern_counts[cand] += 1
            else:
                pattern_counts[cand] = 1
        #traverse counts and add results that have more than 1 count
        results =  []
        for k,v in pattern_counts.items():
            if v > 1:
                results.append(k)
        return results

class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        '''
        single pass hash
        '''
        N = len(s)
        seen =  set()
        output = set()
        for i in range(0,N-10+1):
            temp = s[i:i+10]
            if temp in seen:
                output.add(temp[:])
            seen.add(temp)
        return output

##Rabin Karp
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        '''
        Rabin Karp using rolling hash
        the idea is to slice over the string and comptue the hast of the sequence in the sliding window
        first assign each unique char a number 0 to len(uniques) - 1
        'A' -> 0, 'C' -> 1, 'G' -> 2, 'T' -> 3
        to get the hash of this use base 4
        AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT -> 00000111110000011111100000222333
        the first sequence is
        0000011111
        h_0 = \sum_{i=0}^{L-1} c_i 4^{L-1-i}
        to compute the next hash,just add onto it c_{L+1}
        or just h_1 = (h_0 \ctimes 4 c_0 4^L) + c_{L+1}
        
        '''
        L = 10
        n = len(s)
        #nothing to return
        if n < L:
            return []
        #rolling hash base 4
        a = 4
        aL = pow(a,L)
        #char to integer mapping
        to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        #convert
        nums = [to_int.get(s[i]) for i in range(n)]
        
        #starting hash 
        h = 0
        seen,output = set(), set()
        for start in range(n-L+1):
            #hash for the first sequence
            if start != 0:
                #convert the first and last char hashes
                h = h*a - nums[start-1]*aL + nums[start+L-1]
            else:
                #get the hash if the initial 
                for i in range(L):
                    h = h*a +nums[i]
                    
            #updates
            if h in seen:
                output.add(s[start:start+L])
            seen.add(h)
        return output
        
############################
#Meeting Rooms II 10/17/2020
############################
#well i tried....
import heapq
class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        sort on the start values
        i need to keep track of when a meeting ends
        after sorting on start
        -----------------
         ---
               -----
        in this case three rooms are needed
         [[0, 30],[5, 10],[15, 20]]
        start [[0,30]] rooms 1
        [5,10] rooms 2
        [15,20] 15>10 use up this room
        rooms 2
        
        '''
        intervals.sort()
        rooms = 0
        min_heap = [intervals[0][1]]
        heapq.heapify(min_heap)
        while intervals[1:]:
            #room allocation
            rooms = max(rooms, len(min_heap))
            current = intervals.pop()
            while heapq.heappop(min_heap)[1]

#you were close! but you had the right idea!
import heapq
class Solution(object):
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/meeting-rooms-ii/solution/
        notes:
        we need to be able to efficinealty find out if a rooms is available or not for the current meeting and assing a new room only if non of the assigned rooms are currently free
        PRIORITY QUEUE!
        sort on start
        but how do we find out efficinetly if a room is available or not?
        at any point in time we have multiple rooms that can be occupied and we dont really care which room is free as long as it is free
        the naive way to do this would be to iterate over all rooms every time, but we can do better
        use min heap to keep tracking of EARLIEST ending time
        anytime we need to check if a room is available or not, just check the top most element in the heap
        if the room we extraced from the top of the heap isn't free, instead of searching, just allocate a new room
        STEPS: 
            1. sort on start
            2. init a new min_heap and add the first meeting's end time
            3. for every meeting room, check if the min elemtnof the heap is free or not
            4. if it is free, pop the top, and add the current ending time into the min heap
            5. if not allocate a new room
            6. after processsing the rooms, get the lenght of the heap
        '''
        #edge case, not meetings return 0
        if not intervals:
            return 0
        
        #heap init
        free_rooms = []
        #sort!
        intervals.sort()
        #add the frist meeting
        heapq.heappush(free_rooms, intervals[0][1])
        #traverse now for all meetings after the first
        for meeting in intervals[1:]:
            #if ending time is less than this meeting's start, we are done with this room!
            #and we do not need to allocate, so pop it off
            if free_rooms[0] <= meeting[0]:
                heapq.heappop(free_rooms)
                
            #if the ending time is greater, well we need to add in a new room
            heapq.heappush(free_rooms,meeting[1])
            
        return len(free_rooms)


###################################################
# Best Time to Buy and Sell Stock IV 10/18/2020
###################################################
#TLE Exceed
class Solution(object):
    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        '''
        this is the same thing as the best time time to buy and sell stock III,
        the only difference is that i can complete up to k transactions
        with the one transaction problem we would create a dp profits array, storing the max
        then dp[i] = max(dp[i-1], profit_so_far + prices[i])
        recall in the 1 transaction problem, we kept track of curr_min and max_profit
        we then went across the prices and update
        max_profit = max(max_profit, prices[i]-curr_min)
        curr_min = min(curr_min, prices[i])
        X   5  11  3  50  60  90
        0   0   0  0   0   0   0
        1   0   6  6  47  57  87      
        2   0   6  6  53  63  93
        set up dp array k by len prices
        dp will hold the max profit found so far for a certain number of transactions
        keep track of a maxsofar for each day, this is just 
        at each day, consinder the additional amount of money you could have made by adding the max from the previous day!
        this solution does
        profit[t][d] = max(profit[t][d-1], prices[d]+max(-prices[x] + profit[t-1][x]) for x in range(d))
        but we can call the last part just a maxsofar and keep updating
        maxsofar = max(maxsofar,profits[t-1][d-1] - prices[d-1])
        '''
        if not prices:
            return 0
        #in the case k is greater than n! wish means we can trasact for every otehr day
        if 2*k > len(prices):
            res = 0
            for i, j in zip(prices[1:], prices[:-1]):
                res += max(0, i - j)
            return res
        profits = [[0]*len(prices) for _ in range(k+1)]
        for t in range(1,k+1):
            #store maxsofar from holding stock on previous days since we are allowed t +1 transactions
            maxsofar = float('-inf')
            for d in range(1,len(prices)):
                maxsofar = max(maxsofar, profits[t-1][d-1] - prices[d-1])
                profits[t][d] = max(profits[t][d-1], maxsofar + prices[d])
        return profits[-1][-1]
        

class Solution(object):
    def maxProfit(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        '''
        getting rid of the extra dp array
        '''
        #first edge case
        if not prices:
            return 0
        
        #the case where 2k > n
        #in this case we can freely trasact an many as we'd like
        #max of the sum of the first differences arary
        if 2*k > len(prices):
            results = 0
            for i,j in zip(prices[1:],prices[:-1]):
                results += max(0,i-j)
                return results
        
        #the general dp problem
        dp = [0*len(prices)]
        
        for t in range(1,k+1):
            #we update pos for each ith day
            #profit is just max(dp[-1], prices[i]+dp[i])
            pos = -prices[0]
            profit = 0
            for i in range(1,len(prices)):
                pos = max(pos, dp[i-1]-prices[i]) #buying
                profit = max(profit, pos+prices[i]) #selling
                #update
                dp[i] = profit
        return dp[-1]


###################################################
# Minimum Domino Rotations For Equal Row 10/19/2020
 ###################################################
#fuck me! this is tough
class Solution(object):
    def minDominoRotations(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        '''
        greedy approach?
        get counts for A, for the largest count in A, check if the complement is in B
        if it is, we might be able to do it?
        i can count up the number of dots in both A and B
        for me to swap, there needs to be at least len(A) occurnces of a dot
        
        '''
        #no edge cases
        N = len(A)
        countA = Counter(A)
        countB = Counter(B)
        
        to_flip = None
        #traverse CountA
        for k,v in countA.items():
            if countB[k] == N - countA[k] or countA[k] == N -countB[k]:
                #find flip
                to_flip = k
        print to_flip
        #find flipping indices in both A and B


class Solution(object):
    def minDominoRotations(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        '''
        Greedy method, just start with the first domino
        there are three cases to check, A is the top half, B is the bottom half
        1. Pick the first domino and flip so that all elements match A[0]
        2. Pick the second domino and flip so thatl all elements mattch B[1]
        3. No amount of flips and result an the in A or B being the same
        
        why the first? well ALL elements in the row have to match! if we can't do it with the first we can't do it all
        
        pick the first domino and check if we can make either A or B equal to all A[0] or B[0]
        return the minimum number of rotations
        otherwise return -1
        
        '''
        N = len(A)
        def simulate(num):
            #return min number of swaps if it were possible
            #else return -1
            
            #init
            swaps_a = swaps_b = 0
            for i in range(N):
                #we can't swap anything, so terminate
                if A[i] != num and B[i] != num:
                    return -1
                #if A doesn't match but B does
                elif A[i] != num:
                    swaps_a += 1
                #B doesn;t but A does
                elif B[i] != num:
                    swaps_b += 1
                    
            return min(swaps_a,swaps_b)
        
        #first try with A
        swaps = simulate(A[0])
        
        if swaps != -1 or A[0] == B[0]:
            return swaps
        else:
            return simulate(B[0])


  class Solution(object):
    def minDominoRotations(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        '''
        [2,1,2,4,2,2] 
        [5,2,6,2,3,2]
        
        if this were possible then either A[0] or B[0] would have to show up in tehr est of the array
        '''
        if not A:
            return -1
        N = len(A)
        
        #first check if possible
        v1,v2 = A[0],B[0]
        
        top = all([v1 in (A[i],B[i]) for i in range(1,N)])
        bottom = all([v2 in (A[i],B[i]) for i in range(1,N)])
        
        #if this were both true then the minimum number of rotations would be the same
        
        if not top and not bottom:
            return -1 #we cant accomplish this
        
        #we can now go across both arrays and count where we need to flip (i.e where the element is not v1 or v2 and we can take he mid)
        output = None
        if top:
            #go through both arrays and count up where its not v1
            #we need to rotate here, but we want the minimum from A or B
            output = min(sum([1 for i in range(N) if A[i] != v1]),sum([1 for i in range(N) if B[i] !=v1]))
        elif bottom:
            output = min(sum([1 for i in range(N) if A[i] != v2]),sum([1 for i in range(N) if B[i] !=v2]))
            
        return output
        


#######################
#Clone Graph 10/20/2020
#######################
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        #edge cases
        if node == None:
            return None
        elif node.neighbors == []:
            return Node(val=1, neighbors = [])
        
        #great! now recreate the graph, i could go through the entire graph once
        #recreating the adjaceny list as a dict
        #from the dict return a new node
        print node.neighbors[0]


"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        '''
        this is graph traversal, so i can use dfs
        this is just dfs
        i can traverse the graph recurisvely and for each node to cloned variable, use hash
        
        algo:
            allocate a hash map, call this cloned
            seen set will be the hash map
            the key will be the node, and the value will be a list of nodes
            for each node:
                check is we have seen this nodes neighbors:
                    the RECURSE for each neighbor
            after recursing put back into the hashmap
                
        '''
        #edge case 
        if not node:
            return node
        
        #init cloned, RETURN THIS SHIT!, but give reference to the node
        cloned_map = {}
        
        def dfs(node):
            #for each node visited add to cloned map
            cloned_map[node] = Node(node.val) #initilize for each first time visited
            for n in node.neighbors:
                if n not in cloned_map:
                    #FUCKING RECURSE
                    dfs(n)
                #update cloned
                #recall key is node and val is a list, adding all neighbors from the node we are at
                cloned_map[node].neighbors += [cloned_map[n]]
        
        #inovke
        dfs(node)
        return cloned_map[node]


"""
# Definition for a Node.
class Node(object):
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution(object):
    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        '''
        this is just DFS, but with a wrench
        [[2,4],[1,3],[2,4],[1,3]]
        start at 1, visited(1), go to 2
        visited(1,2), go to 3
        visted(1,2) go to 4
        before we get the node 4, mark nodes 1, and 3 in hash
        thoughts....
        if we dfs on each node we'd get to the end befor ever initalizsing nodes 1,3
        in our dfs call give reference to each node, which means we can't see the neighbors 
        before firing dfs, we need to create a copy of the node into our hash
        why? in the absence of ordering, we might get caught in the recursion because we could encouter the node again down the line, example 1,3 is at node 4, but i haven't given reference to nodes 1,3 (its later in the call stack)
        
        algo:
            dfs on a node, then build up its neighbors later
            we only for each neihbor of a node, we don't need anymore than that!
            visited set has key a node.val and val as being a copy of the node
            the return case for a dfs call is just a node, in fact the neighboring nodes/node
        '''
        #return copy.deepcopy(node)

        #edge case
        if not node:
            return None
        
        visited = {} #key is node.val and value is copy of node
        def helper_dfs(node,visited):
            #create new node, to prevent cycle, give reference to each node visited
            new = Node(node.val)
            #mark node as visited, give reference to new node COPYYYYY
            visited[node.val] = new
            #we need to find neighbors for this node,allocate as method
            new.neighbors = []
            
            for n in node.neighbors:
                if n.val not in visited:
                    #add in our neighrbos and FUCKING RECURSE!!!
                    new.neighbors.append(helper_dfs(n,visited)) #add the nodes along the path
                else:
                    new.neighbors.append(visited[n.val]) #if i have seen it, just add the one node
            #now add the neighbors for this node back in        
            return new
        
        #invoke
        return helper_dfs(node, visited)


##############################
#Asteroid Collision 10/21/2020
##############################
class Solution(object):
    def asteroidCollision(self, asteroids):
        """
        :type asteroids: List[int]
        :rtype: List[int]
        """
        '''
        simulate using stack? maybe a deque
        add in the first asterioid to the stack
        [5,10,-5]
        
        [5]
        [5,10]
        check if opposite
        [5,10,-5], hit take the bigger one
        [5,10]
        could i use heap?
        build up one asteroid at time
        a row is stable if all the negative values are before all the positive values, 
        if a negative value comes after a postive one, they are going to hit
        it could be the case the i have hits before the right most asteroid, but does that matter
        [5,10,-5,10,-6]
        start from the right and keep colliding, if i cant collide anymore go on to the next asteroid
        check if we need to add in an asteroid
        '''
        #edge case
        if not asteroids:
            return []
        
        results =  []
        results.append(asteroids[0])
        
        #starting with the second one
        for ast in asteroids[1:]:
            #check if we need pop off, or explistion
            while results and results[-1] > 0 and ast < 0 and abs(ast) > results[-1]:
                results.pop()
                
            #check if we need to append, exploed,or both
            #if empty add it
            if not results:
                results.append(ast)
            #iff the last asteroid in our stack is positive but same size they both go away
            elif results[-1] > 0 and ast < 0 and abs(ast) == results[-1]:
                results.pop()
            #iff last asteroid is negative, and the asteroid is going the other way
            elif results[-1] < 0 or ast > 0:
                results.append(ast)
        return results


#######################################
# Minimum Depth of Binary Tree 10/22/20
#########################################
#so close
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        keep doing my dfs call until i get down to a leaf
        once i get down to a leaf (making sure to add the nodes in)
        dump length(path) into results and return the min
        '''
        lengths = []
        def dfs(node,path):
            if not node:
                return
            elif not node.left and not node.right:
                lengths.append(path)
                return
            path.append(node.val)
            dfs(node.left,path)
            dfs(node.right,path)
            
        dfs(root,[])
        return lengths

#this is super slow
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        keep doing my dfs call until i get down to a leaf
        once i get down to a leaf (making sure to add the nodes in)
        dump length(path) into results and return the min
        '''
        if not root:
            return 0
        lengths = []
        def dfs(node,path):
            if not node:
                return
            if not node.left and not node.right:
                lengths.append(path+[node.val])
            else:
                dfs(node.left,path+[node.val])
                dfs(node.right,path+[node.val])

        dfs(root,[])
        return min([len(foo) for foo in lengths])

#so close again
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        dfs was too slow
        level order BFS
        keep checking if the child nodes do not have any left or right nodes
        if they don't i've found the shortes path and can stop traversing
        '''
        if not root:
            return []
        
        #starting with depth at least 1
        depth = 0
        q = [root]
        while q:
            #visit the node
            current = q.pop()
            #check for leaf
            if not current.left and not current.right:
                break
            q.append(current.left)
            q.append(current.right)
            depth += 1
        
        return depth
            

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        dfs was too slow
        level order BFS
        keep checking if the child nodes do not have any left or right nodes
        if they don't i've found the shortes path and can stop traversing
        '''
        if not root:
            return 0
        

        #use deque with and element will be (node,level)
        q = deque([(root,1)])
        while q:
            #visit the node
            current,level = q.popleft()
            #check for leaf
            if not current.left and not current.right:
                #terminate
                return level
            #if they have children
            if current.left:
                q.append((current.left,level+1))
            if current.right:
                q.append((current.right,level+1))
        return -1


 self.min = float('inf')
        def dfs(node,level):
            if not node:
                return
            if not node.left and not node.right:
                #update
                self.min = min(self.min, level)
            dfs(node.left, level+1)
            dfs(node.right,level+1)
            
        #invoke
        dfs(root,1)
        return self.min
            


#########################
#Maximun sum subarray 53
##########################
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        i can just do it greedily
        at every point in the array i keep checking if the sum will go up from a previous sum
        since its a max negatives will bring it down, so just take the max(dp[i-1],0) + nums[i]
        [-2,1,-3,4,-1,2,1,-5,4]
        [-2,1,-2,4,3,5,6,1,5]
        curr_max = 3
        absolute_max = 3
        in the dp[i] = max(dp[i-1],0) + nums[i]
        
        
        '''
        '''
        #allocating a dp array
        #edge cases
        if len(nums) < 2:
            return max(nums)
        
        #inits
        N = len(nums)
        dp = [0]*N
        
        dp[0] = nums[0]
        
        #init aboslute max holder
        abs_max = float('-inf')
        for i in range(1,N):
            dp[i] = max(dp[i-1],0) + nums[i]
            abs_max = max(abs_max,dp[i])
            
        return abs_max
        '''
        #realize im only every looking at the previous element in dp an updating
        #just collaspse into a variable
        if len(nums) < 2:
            return max(nums)
        
        N = len(nums)
        prev_max = float('-inf')
        abs_max = float('-inf')
        
        for i in range(N):
            prev_max = max(prev_max,0) + nums[i]
            abs_max = max(abs_max,prev_max)
        return abs_max

    
##############################
# 132 Pattern 10/23/2020
###############################
#O (n^3)
class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        brute force first n cubed
        
        '''
        for i in range(0,len(nums)-2):
            for j in range(i+1,len(nums)-1):
                for k in range(j+1,len(nums)):
                    if nums[i] < nums[k] < nums[j]:
                        return True

# O(N^2)
class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        n squared, start with 1 pointer call this anchor, which is just the first
        note that for a partricular nums[j] (the second element) if  we dont consider nums[k] our job is just to find nums[i] such that i<j and nums[j] < nus[k]
        after finding a pairt, we now need to finde nums[k] such that nums[k] >
        first find pair of nums[i] and nums[j], then find nums[k], which is in the range of nums[i] and nums[j], such that i < j < k
        we only check after jth index to find k
        to find k, we keep track of the min element found so far exluding nums[j]
        why? the min element alwasy serves as the nums[i] for the current j
        
        '''
        #find the smallest i as we search for the jth element
        smallest_i = float('inf')
        for j in range(0, len(nums)):
            smallest_i = min(smallest_i, nums[j])
            #k is after j but in between nums[i] and nums[j]
            for k in range(j+1,len(nums)):
                if (nums[k] < nums[j]) and (smallest_i < nums[k]):
                    return True
                
        return False

#better n squared checking intervals!
class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        this works?!?!?!
        searching intervals
        in the last n squared approach we fixed nums[j] found the smalelst i at the same time
        and created a range between (nums[i],nums[j]) for which nums[k] could fall in to
        in this solution we kept updating our nums[i],nums[j] range, and traverse again to find our k
        instead note the ranges to find k our in local rising slope in the array
        to find these points we can traverse the nums arrays and keep track of the minimum point found after the peak at nums[s]
        now when we encounter a falling slope at index i we knot that nums[i-1] was the point point locally!
        then we check in this range!
        instead of tracering of all nums to kinf k, we seach in the range(nums[s],nums[i-1])
        while traversing the nums arrays to check the rising and faling slops, we keep adding the endpoint piars to a new intercals arrays
        '''
        #create intervals
        intervals = []
        #points, i is the end of the range and s is start
        i,s = 1,0
        while i < len(nums):
            #check inreasing?
            if nums[i] < nums[i-1]:
                #check that s lagging behind i
                if s < i -1:
                    intervals.append([nums[s],nums[i-1]])
                #new interval 
                s = i
            #traverese the intervals
            for foo in intervals:
                if nums[i] > foo[0] and nums[i] < foo[1]:
                    return True
            #move our i
            i += 1
        
        return False


class Solution(object):
    def find132pattern(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        O(N) stack
        recall in the last n squared solutions we found the first nums[i] without having to search all combinations by first finding a range (nums[i],nums[j]). 
        we can do better by preprocessing to find nums[k]
        first we allocate a min array 
        we interprest this array as having the best min[i] for a value of j
        we go from the back of nums, to find the nums[k], since k has to be after i,j
        we keep track of the nums[k] values which can potentially satisfy the 132 criteria for the nums[j]
        we know that one of the conditions to be satisfied by such a nums[k] is that is must be greater than nums[i]! but we cound the best nums[i] in our first min pass!
        after creating the min array, we start traversing the nums[j] array in a backward manner
        lets say we are currently at the jth eleemnt and lets also assume that the stack is sorted now
        first check that nums[j] > min[j] is not keep going down doing nothing with the stack
        if not, pop elemtns formt ehs top of the stack until we find taht stack[top] > nums[j] > min[j]
        '''
        if len(nums) < 3:
            return False
        
        stack = []
        #allocate mins array
        mins = [0]*len(nums)
        mins[0] = nums[0]
        
        #find the smallest i for each j!
        for i in range(1,len(nums)):
            mins[i] = min(mins[i-1], nums[i])
        
        #go backward, since k > j > i 
        #and nums[j] > nums[k] > nums[i]
        #lo, hi, mid
        for j in range(len(nums)-1,-1,-1):
            if nums[j] > mins[j]:
                #keep poping our j, which is the mid
                while stack and stack[-1] <= mins[j]:
                    stack.pop()
                if stack and stack[-1] < nums[j]:
                    return True
            stack.append(nums[j])
            
        return False

############################
#Bag Of Tokens 10/24/2020
############################
#almost
class Solution(object):
    def bagOfTokensScore(self, tokens, P):
        """
        :type tokens: List[int]
        :type P: int
        :rtype: int
        """
        '''
        initial Power P, and initial score 0, and bag of tokens
        max scroes by:
            1. if current power is >= tokens[i], i can play the token face up, power -= tokens[i] and score += 1
            2. if score >=, play token[i] face down power += tokens[i], score -= 1
        max score, think greedy
        order the tokens?
        i dont think i need to DP here or recurse
        since im maxing my score, you want to do option 1 more
        score can go negative, but power can't, what if i max power?
         tokens = [100,200,300,400], P = 200
         
        score = 0, P = 200
        100 face up
        score 1, P = 100
        400 face down
        score 0, P = 500
        200 face up
        score 1, P = 300
        300 face up
        score 2, P = 0
        to play all the tokens i would need at least 1000, 800 away
        sort from largest to smallest, keep powering up until i have enough to play the remainder of the tokens!
        '''
        tokens.sort(reverse=True)
        
        
        sum_tokens = sum(tokens)
        score = 0
        for t in tokens:
            #play token 
            sum_tokens -= t
            if P == 0:
                break
            elif P >= sum_tokens:
                score += 1
                P -= t
            else:
                P += t
                score -= 1
        return score

class Solution(object):
    def bagOfTokensScore(self, tokens, P):
        """
        :type tokens: List[int]
        :type P: int
        :rtype: int
        """
        '''
        intuition: if we play a token face up, we might as well play the token with the smallest value
        if we play a token face down, play the one with the largest value
        
        play tokens face up until we can't anymore, then play face down and continue
        
        edge case
        play tokens face up until exhaustion (score goes to negative) then play token face down
        terminate when we cant play either face up or face down
        '''
        #sort increaslingly 
        tokens.sort()
        
        #two pointers, up, which is the min, and down which is the max
        N = len(tokens)
        score = 0
        up = 0
        down = N - 1
        #take token from up or down
        while up <= down and (score or P >= tokens[up]):
            #i can play the min up token
            if P >= tokens[up]:
                score += 1
                P -= tokens[up]
                up += 1
            #playing down
            elif up != down:
                score -= 1
                P += tokens[down]
                down -= 1
            else:
                break
        return score

##another way using deque
class Solution(object):
    def bagOfTokensScore(self, tokens, P):
        """
        :type tokens: List[int]
        :type P: int
        :rtype: int
        """
        q  = collections.deque(sorted(tokens))
        score = 0 
        max_score = 0
        
        while q:
            if P >= q[0]:
                t = q.popleft()
                P -= t
                score += 1
                max_score = max(max_score,score)
            
            elif P < q[0] and score > 0:
                t = q.pop()
                P += t
                score -= 1
            else:
                break
        return max_score


##########################
#Stone Gamve IV 10/25/2020
##########################
class Solution(object):
    def winnerSquareGame(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        square numbers alice can win in one move
        hint says it is a dp problem
        bob is in a losing state if alice is able to get all the stones
        final losing move?
        if it is alice's turn and the number of stones is a perfect square
        which means the last move for Bob is that there are not a square number of stones
        https://leetcode.com/problems/stone-game-iv/solution/
        
        given n strones, alice has a must win strategy or Bob does, since the game cannot end in a draw
        either bob or alice must win
        
        let the current player refer to the player no removing the stones and stat i refers to when there are i stones remaining
        if we can go to a known state where Bob must lose, then in the next state alice must wint
        all alice has to do is remove the corresponding number of stones to go to taht state
        therefore we need to find out which statte Bob must lose
        
        note that after going to the next state, bob becomes the player removing the stones, 
        which iis the positions of alice in the current state
        therefore to fin out whether Bob will lose in the next state, we just need to check if our recursive function give False
        
        let dfs(remain) represent whether the currentplayer must win with the remain stones
        to find out the result sof dfs(n) we need to iterate k from 0 to whether there exists dfs(remain - k*k) == Fakse
        '''
        cache = {}
        cache[0] = False
        
        def dfs(cache,remaining):
            #remaining stones state, win or not
            if remaining in cache:
                return cache[remaining]
            
            sqr_root = int(remaining**0.5)
            #check states from 1 to sqrt root
            for i in range(1,sqr_root+1):
                #is there is any change to mkae the opponent lost the game in the next round, then that player will win
                if not dfs(cache,remaining -i*i):
                    cache[remaining] = True
                    return True
            
            #not in the looop
            cache[remaining] = False
            return False
        
        return dfs(cache,n)

class Solution(object):
    def winnerSquareGame(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        bottom up DP solution for finding the state of the game
        if you ever get to state where there is only 0 stones, youuve lost
        so if you can get to this state from anothe state you're winning
        if we can somehow get to a false (losing state for BOB) from nth state, then we are
        in a winning position!
        0   1   2   3   4   5
        F   T   F   T   T   F
        '''
        dp = [False for _ in range(n+1)]
        
        for i in range(1,n+1):
            stones = 1
            while stones**2 <= i:
                if not dp[i - stones**2]: #if we go back from our state and see that its false we can win!
                    dp[i] = True
                    break
                stones += 1
        return dp[-1]
        


###########################
#Champagne Tower 10/26/2020
###########################
class Solution(object):
    def champagneTower(self, poured, query_row, query_glass):
        """
        :type poured: int
        :type query_row: int
        :type query_glass: int
        :rtype: float
        """
        '''
        there are a 100 rows
        odd series 
        glasses in a row all the way towards the left or the right only get filled up half as much as the other ones
        the ones in the middle will fill up before the ones on the end
        keep finding the states after your pour a glass
        keep track of the current row, once overflow, need to see the row below
        look at the rates once full
        [1]
        [0.5,0,5]
        [0.25,0.5,0.25]
        [0.125,0.25,0.25,0.125]
        this is pascals triangle
        define the rates at the ith row as:
        
        bleaghh, in stead of keeping track how much champagne should end up in a glass
        keep track of the total amount of champagne poured from the top
        for example if we pour 10 at the top, flow through should be 4.5  of the sides (10-1) / 2
        
        algo:
            in general if a glass has flow through: glass X after being full is (X-1)/2
            a glass at r,c will have flow through to glass at r+1,c and r+1,c+1
        

        
        '''
        #create the glasses array
        glasses = [[0]*102 for _ in range(102)]
        #pout all volume into top glass, and keep decremennting
        glasses[0][0] = float(poured)
        for r in range(0,query_row+1):
            for c in range(0,r+1):
                flowed_through = float((glasses[r][c] - 1) / 2)
                #print flowed_through
                if flowed_through > 0:
                    glasses[r+1][c] += flowed_through
                    glasses[r+1][c+1] += flowed_through
                
        return min(1,glasses[query_row][query_glass])
        '''
        glasses_per_row = []
        i = 1
        while len(glasses_per_row) <= 100:
            glasses_per_row.append([0]*i)
            i += 1
        
        print glasses_per_row
        '''

class Solution(object):
    def champagneTower(self, poured, query_row, query_glass):
        """
        :type poured: int
        :type query_row: int
        :type query_glass: int
        :rtype: float
        """
        '''
        1
        .5 .5
        .25 .5 .25
        instead of keeping track of all 102 rows, just keep track of the current row and the one below it
        then simulate for the query row
        '''
        glasses = [poured]
        for _ in range(query_row):
            #next row
            below = [0]*(len(glasses)+1)
            for i in range(0,len(glasses)):
                flowed_through = float((glasses[i] - 1)/2)
                #add if there is excess
                if flowed_through > 0:
                    below[i] += flowed_through
                    below[i+1] += flowed_through
                #get the new row
            glasses = below
        
        return min(1,glasses[query_glass])


#######################
#Linked List Cycle II 10/27/2020
######################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        there is a cycle if a node.next points to a node that was already passed
        i need to return the node
        if there weren't a cycle, id eventually get to the end and reach a null
        two pointers, slow and fast, fast should eventually meet slow if there is a cycle
        if there isn't a cycle slow will eventuall get to a non in the node.next, which is None
        '''
        #using a lookup set
        cur = head
        lookup = set()
        while cur:
            if cur in lookup:
                return cur
            lookup.add(cur)
            cur = cur.next
        return None
        
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        there is a cycle if a node.next points to a node that was already passed
        i need to return the node
        if there weren't a cycle, id eventually get to the end and reach a null
        this is tortoise and the hare!
        two pointers, slow and fast, fast should eventually meet slow if there is a cycle
        if there isn't a cycle slow will eventuall get to a non in the node.next, which is None
        when they do meet, move one of the pointers back to the head, and keep advnacing until it gets to the other one and return it
        '''
        if not head:
            return
        slow,fast = head,head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                break
                #found the point
                
        #if we've gotten to the tail witoutht finding a cycle
        if not fast.next or not fast.next.next:
            return
        
        #other wise move up one of the points
        newslow = head
        while slow.next:
            if newslow == slow:
                return slow
            newslow = newslow.next
            slow = slow.next
        return
            
        
######################
#Linked List Cycle I
#####################
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


############################################
# Search in a Sorted Array of Unknown Size 10/27/2020
############################################
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation LOL!
# """
#class ArrayReader(object):
#    def get(self, index):
#        """
#        :type index: int
#        :rtype int
#        """

class Solution(object):
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        if reader.get(0) == target:
            return 0
        
        #search conditions
        l,r = 0,1
        while reader.get(r) < target:
            l = r
            r <<= 1 #same as multiplying by 2
            
        #binary search
        while l <= r:
            mid = l + ((r-l) >> 1)
            num = reader.get(mid)
            
            if num == target:
                return mid
            elif num > target:
                r = mid -1
            else:
                l = mid + 1
        return -1


#############################
# Summary Ranges
#############################
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        '''
        increment by one from the min to max, 
        if there is jump close the interval, and move the pointer to the next
        use a deque!
        
        '''
        if not nums:
            return []
        
        q = collections.deque(nums)
        
        start = q[0]
        results = []
        temp = []
        while q:
            if q[0] == start:
                temp.append(q.popleft())
            elif q[0] != start:
                results.append(temp)
                temp = []
            start += 1

                
        #pass the results one last time and create the strings
        
                
        print q


class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        '''
        compare the current number with the next one, if the current numerber is no the previouse number+1, we have jumped
        also check if the range has more than one number, if not then just add the single
        
        
        '''
        if not nums:
            return []
        
        results = []
        curmax = nums[0]
        curmin = nums[0]
        
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1] + 1:
                curmax = nums[i]
            #we need to add
            else:
                if curmax != curmin:
                    results.append(str(curmin)+"->"+str(curmax))
                else:
                    results.append(str(curmin))
                #reset
                curmin = nums[i]
                curmax = nums[i]
        
        #add the last ones
        if curmax != curmin:
            results.append(str(curmin)+"->"+str(curmax))
        else:
            results.append(str(curmin)) 
        
        return results


#######################################
# Maximize Distance to Closest Person
#######################################
#so close!
class Solution(object):
    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        '''
        [1,0,0,0,1,0,1]
        when i encounter a zero use two pointers going left and right
        go left as far as can until i hit a one
        go right as far as i can
        '''
        dist_array = [0]*len(seats)
        
        for i in range(0,len(seats)):
            if seats[i] == 0:
                left,right = i,i
                while seats[left] != 0 or seats[right] != 0:
                    left -= 1
                    right += 1
                dist_array[i] = min(left,right)
        print dist_array
        


class Solution(object):
    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        '''
        left left[i] be the distance from seat to the closest person on the left
        and right[i] similar if to the right
        thus the answer for each i will min(left[i],right[i])
        to construct left[i] notice it is either left[i-1] + 1 if the sat is empty or 0 if it full
        '''
        N = len(seats)
        left,right = [N]*N,[N]*N
        
        #dp extra array
        for i in range(N):
            if seats[i] == 1:
                left[i] = 0
            elif i > 0:
                left[i] = left[i-1] + 1
                
        #now for the right
        for i in range(N-1,-1,-1):
            if seats[i] == 1:
                right[i] = 0
            elif i < N-1:
                right[i] = right[i+1] + 1
                
        #now i want the max for the min at each seat in the array
        output = float('-inf')
        
        for i in range(N):
            output = max(min(left[i],right[i]),output)
        
        return output


class Solution(object):
    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        '''
        another way
        two dp arrays first pass left, then pass from right
        but when going from right, take the min from the left
        hold max dis in a variable
        dis = 0
        [1,0,0,0,1,0,1]
        [0,1,2,3,0,1,0] left to right
        [0,1,2,1,0,1,0] right to left
        
        but if 
        [0,0,0,0,1,0,1]
        [1,2,3,4,0,1,0]
        [1,2,2,1,0,1,0]
        
        keep boolean variable to keep track of when to update
        
        '''
        N = len(seats)
        dp = [0]*N
        dis = -1 #start with negtative distance marking when we have seen a seated seat
        
        for i in range(N):
            if not seats[i] and dis != -1:
                dis += 1
                dp[i] = dis
            else:
                dis = 0
        
        #from the right
        dis = -1
        for i in range(N-1,-1,-1):
            if not seats[i] and dis != -1:
                dis += 1
                #in the case we havent touched our dp
                if dp[i] == 0:
                    dp[i] = dis
                else:
                    dp[i] = min(dp[i],dis)
            else:
                dis = 0
        
        return max(dp)


class Solution(object):
    def maxDistToClosest(self, seats):
        """
        :type seats: List[int]
        :rtype: int
        """
        '''
        O(N) greedy solution
        '''
        N = len(seats)
        start = -1
        maxgap = 0
        
        for i in range(N):
            if seats[i] == 1:
                if start == -1:
                    maxgap = i
                else:
                    maxgap = max(maxgap,(i-start)//2)
                start = i
                
        maxgap = max(maxgap,N-1-start)
        
        return maxgap
        
###########################################
# Number of Longest Increasing Subsequence
###########################################
#all possible subsequences derivation
class Solution(object):
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        recursive
        use backtracking
        invoke for each lenght possibility
        rec function will use
        '''
        subseqs = []
        def backtrack(arr,index,build):
            if index == len(arr):
                if len(build) != 0:
                    subseqs.append(build)
            else:
                backtrack(arr,index+1,build)
                backtrack(arr,index+1,build+[arr[index]])
        
        for i in range(0,len(nums)):
            backtrack(nums[i:],0,[])
        print subseqs

#nice try, good review on geneating subsets
class Solution(object):
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        recursive
        use backtracking
        invoke for each lenght possibility
        rec function will use
        '''
        if len(set(nums)) == 1:
            return len(nums)
        subseqs = []
        def backtrack(arr,index,build):
            if index == len(arr):
                if len(build) != 0:
                    if len(build) > 1:
                        #check increasing
                        temp = []
                        for a,b in zip(build[:-1],build[1:]):
                            temp.append(b-a)
                        if -1 in temp:
                            return
                        else:
                            subseqs.append(build)
            else:
                #subsequence not including the current idx element
                backtrack(arr,index+1,build)
                #subsequence inlcluding the current element at the index
                backtrack(arr,index+1,build+[arr[index]])
                
        
        for i in range(0,len(nums)):
            backtrack(nums[i:],0,[])
        counts = {}
        for s in subseqs:
            if len(s) in counts:
                counts[len(s)] += 1
            else:
                counts[len(s)] = 1
        
        maxlength = float('-inf')
        for k,v in counts.items():
            maxlength = max(maxlength,k)
        return counts[maxlength]

class Solution(object):
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        [1,3,5,4,7]
        [1,2,3,3,4] LS
        [1,1,1,1,1] count
        
        to find the frequency of the longest increasing sequence i need:
            how long the longest increasing subsequence is
            the occurence of each length
        
        store dp[i] the length of longest increasing subsequence at nums[i] (up to)
        cnt[i] the freq of the longest increasing subsequence
        if dp[i] < dp[j] + 1, then we have found a longer sequence and dp[i] needs to be updated
        and cnt[i] = count[j]
        if dp[i] == dp[j] + 1, eamnding dp[j] + 1 is one way to reaching the longest increase sequence to i, incremt bount by cnt[j]
        sum up cnt of all longest 
        '''
        
        if not nums:
            return 0
        
        N = len(nums)
        lengths = [1]*N
        counts = [1]*N
        m = 0 #for storing length of max increasing subsequence
        
        for i in range(N):
            for j in range(i):
                if nums[j] < nums[i]: #increasing 
                    #check, length at i
                    if lengths[i] < lengths[j] + 1:
                        lengths[i] = lengths[j] + 1
                        counts[i] = counts[j]
                    elif lengths[i] == lengths[j] + 1:
                        counts[i] += counts[j]
            
            #update max length
            m = max(m,lengths[i])
                        
        return sum(c for l,c in zip(lengths,counts) if l == m)
        

##############################
#Recover Binary Search Tree
##############################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        '''
        recall that an inroder traversal of a BST results in ascneding array
        since at least two of the nodes are swapped, we can find which two nodes are swapped using the folloinwg:
        traverse the array in order
        if the i + 1 element is greater than i, allocate it
        if second occurent break
        
        construct the inorder traveral
        find the two swapped nodes
        traverse the tree again inorder chaing x to y and and the y to x
        '''
        def inorder(node):
            return inorder(node.left) + [node.val] + inorder(node.right) if node else []
        
        def find_two_swapped(nums):
            N = len(nums)
            x = y = -1
            for i in range(N-1):
                if nums[i+1] < nums[i]:
                    y = nums[i+1]
                    if x == -1:
                        x = nums[i]
                    else:
                        break
            return x,y
        
        def recover(node,count):
            if node:
                if node.val == x or node.val == y:
                    node.val = y if node.val == x else x
                    count -= 1
                    if count == 0:
                        return
                recover(node.left,count)
                recover(node.right,count)
        
        nums = inorder(root)
        x,y = find_two_swapped(nums)
        recover(root,2)

####
#another way

class Solution(object):
	def recoverTree(self,root):
		self.temp = []

		def dfs(node):
			if not node:
				return

			dfs(node.left)
			self.temp.append(node.val)
			dfs(node.right)

		dfs(root)

		srt = sorted(n.val for n in self.temp)

		for i in range(len(srt)):
			self.temp[i].val = srt[i]


