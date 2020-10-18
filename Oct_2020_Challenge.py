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
        