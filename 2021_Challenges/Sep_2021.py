####################
# Array Nesting
####################
#yes! TLE!
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        '''
        brute force would be to try building a set iteratively
        and do this for each index i
        '''
        def build_set(i):
            seen = set()
            seen.add(nums[i])
            #get new i
            i = nums[i]
            while nums[i] not in seen:
                seen.add(nums[i])
                i = nums[i]
            return len(set)
        
        longest = 0
        for i in range(len(nums)):
            longest = max(longest,build_set(i))
        
        return longest

#recursive function
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        '''
        we could also do this recursively
        if we haven't seen this number yet, add it to a set, and go to that index on nums of that number
        rec(i) = 1 + rec(nums[i]) if i not in seen
        then invoke for each i
        '''
        seen = set()
        memo = {}
        def dp(start):
            if start in memo:
                return memo[start]
            if start not in seen:
                seen.add(start)
                res = 1 + dp(nums[start])
                memo[start] = res
                return res
            else:
                memo[start] = 0
                return 0
        
        longest = 0
        for i in range(len(nums)):
            longest = max(longest,dp(i))
        return longest
        
#phew
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        '''
        brute force would be to try building a set iteratively
        and do this for each index i
        this gives TLE, 
        notice that if the array was sorted the longest chain is 1
        we can optimize the N^2 solution keeping visited, and abandon this starting as an option
        '''
        
        seen = set()
        longest = 0
        for i in range(len(nums)):
            if i not in seen:
                #we can start to include it
                count = 1
                #take the first starting
                start = nums[i]
                while start != i:
                    start = nums[start]
                    count += 1
                    seen.add(start)
                    
                longest = max(longest,count)
        
        return longest

#O(1) space
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        '''
        brute force would be to try building a set iteratively
        and do this for each index i
        this gives TLE, 
        notice that if the array was sorted the longest chain is 1
        we can optimize the N^2 solution keeping visited, and abandon this starting as an option
        we can optimize by using nums array directly
        '''
        

        longest = 0
        for num in nums:
            if nums != -1:
                count = 0
                while nums[num] != -1:
                    #increment the cycle size
                    count += 1
                    #get the current next number
                    temp = nums[num]
                    #mark
                    nums[num] = -1
                    #advance
                    num = temp
                longest = max(longest,count)
        
        return longest

###########################
# Maximum Average Subtree
###########################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        '''
        i can dfs twice to find the sums of the substrees and number of nodes
        then just return the max average
        '''
        sums = []
        num_nodes = []
        
        def sum_subtree(node):
            if not node:
                return 0
            left = sum_subtree(node.left)
            right = sum_subtree(node.right)
            curr_sum = left + right + node.val
            sums.append(curr_sum)
            return curr_sum
        
        def nodes(node):
            if not node:
                return 0
            left = nodes(node.left)
            right = nodes(node.right)
            curr_num = left + right + 1
            num_nodes.append(curr_num)
            return curr_num
        
        sum_subtree(root)
        nodes(root)
        max_avg = 0
        for a,b in zip(sums,num_nodes):
            max_avg = max(max_avg, a/b)
        
        return max_avg

#intersting enough this gets TLE
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        '''
        i can dfs twice to find the sums of the substrees and number of nodes
        then just return the max average
        '''
        self.max_avg = 0
        
        def dfs(node):
            if not node:
                return 0,0
            left_sum = dfs(node.left)[0]
            right_sum = dfs(node.right)[0]
            curr_sum = left_sum + right_sum + node.val
            left_nodes = dfs(node.left)[1]
            right_nodes = dfs(node.right)[1]
            curr_nodes = left_nodes + right_nodes + 1
            self.max_avg = max(self.max_avg,curr_sum/curr_nodes)
            return [curr_sum,curr_nodes]
        
        dfs(root)
        return self.max_avg

#this works though
class Solution:
    def maximumAverageSubtree(self, root: Optional[TreeNode]) -> float:
        if not root:
            return 0
        
        def dfs(node=root):
            
            # if node is None, return 0 as sum and 0 as no. of nodes
            if node is None:
                return (0, 0)
            
            # go to left subtree and get total sum on that subtree, and the total nodes
            sum_left, node_cnt_left = dfs(node.left)
            
            # go to right subtree and get total sum on that subtree, and the total nodes
            sum_right, node_cnt_right = dfs(node.right)
            
            # calculate the total sum
            _total_sum = node.val + sum_left + sum_right
            # calculate the total no. of nodes
            _total_nodes = 1 + node_cnt_left + node_cnt_right
            # compute the avg
            _avg = _total_sum / _total_nodes
            
            # update max_avg if it is smaller than computed avg
            if _avg > self.max_avg:
                self.max_avg = _avg
            
            # return the total sum and total nodes to previous stack call.
            return (_total_sum, _total_nodes)
        
        # set a variable to track max_avg
        self.max_avg = float("-inf")
        dfs()
        return self.max_avg

##################################
#   Unique Binary Search Trees II
##################################
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        '''
        the first version of this problem  was to generate th numbe of unique trees
        turns out that is just the catalan number
        we can solve for the nth catalan number recursively:
            we pick each i as a root, then we have the sequence (1..i) on left and (i+1,n) on right
            if we know the number of ways for left and right, then the number of ways to make a unique BST if left*right
            then we just find this for all possible roots betwwen (1...n)
            sum of all left and rights for all i in n
        now how do we make a tree:
            pick all is as the right and make trees on left and right sides
            we can use two pointers start and end
        '''
        def dp(start,end):
            if start > end:
                return [None]
            trees = []
            for i in range(start,end+1):
                left = dp(start,i-1)
                right = dp(i+1,end)
                #connect
                for l in left:
                    for r in right:
                        curr = TreeNode(i)
                        curr.left = l
                        curr.right = r
                        trees.append(curr)
            return trees
        
        return dp(1,n)

#######################
#  Erect the Fence
#######################
class Solution:
    def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
        '''
        this is really just the convex hull problem, and the first algorithm is know is Jarvis
        the idea is we want to wrap all points in the set in a counterclockwise manner
        for a point p, we need to find a point q that is counter clockewise relative to p, 
        we can take the crosss product of p and q or p and r
        and add to the hull the one with the smaller cross producrt
        the cross product pq and qr should be in the same direction
        algo:
            scan over all the points r and find out which q is the most counter clockwise relative to p
            if there are two points with the same realtive orientation to the current point p
            the points i and j are collinear to p, we need to consider the point i, which lies between p and k
            we keep going to make the hull
        '''
        def orientation(p,q,r):
            diag = (r[0]-q[0])*(q[1]- p[1])
            off_diag = (r[1]-q[1])*(q[0]-p[0])
            return diag - off_diag
        
        def inBetween(p,i,q):
            a = (i[0] >= p[0] and i[0] <= q[0]) or (i[0] <= p[0] and i[0] >= q[0])
            b = (i[1] >= p[1] and i[1] <= q[1]) or (i[1] <= p[1] and i[1] >= q[1])
            return a and b
        
        hull = set()
        if len(trees) < 4:
            for t in trees:
                hull.add(tuple(t))
            return hull
        
        #find left most point
        left_most = 0
        for i in range(len(trees)):
            if trees[i][0] < trees[left_most][0]:
                left_most = i
        p = left_most
        #start off with p
        while True:
            q = (p+1) % len(trees)
            #keep finding the next point to the right of vector pq and move it
            for i in range(len(trees)):
                if orientation(trees[p],trees[i],trees[q]) < 0:
                    q = i
            #now check for colinearity and add all colinear points to hull
            for i in range(len(trees)):
                if i != p and i != q and orientation(trees[p],trees[i],trees[q]) == 0 and inBetween(trees[p],trees[i],trees[q]):
                    hull.add(tuple(trees[i]))
            #none, so add them in
            hull.add(tuple(trees[q]))
            p = q
            if p == left_most:
                break
        return hull

#graham scan
from functools import cmp_to_key
class Solution:
    def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
        '''
        we can reduce the time complextiy of jarvis march using graham scan
        algo:
            start with an inital point, in this case the one with the lowest y coordinate
            in case of a tie, choose one with lower x coordinate
            we then sort the points base on their polar angles, wrt to a vertical line drawn through the initial point
            we can use the same orientation function in the last one
            points with a lower polar angles, relative to the vertical line of the starting point are examined first
            if points have some polar angle, we break ties with the point closes to the veritcal line starting point
            SPECIAL CASE: colinear points on the last edge of the hull
                after sorting array, traverse sorted array from end and reverse the order of the points which are colinear and twoards the end of the sorted array
            starting with inital points, push them onto a stack
            if the current point being considered appears after taking a left turn or straight path, we push the point onto the stack
            we can orientation function, if orientation > 0, this is a right turn, otherwise note
            we pop off the last point from teh stack if it happens to be takin a right turn from the previous line dirction
            we keep poppinf off until we can make a right turn to the current point
            points on the stack make up the hull
        '''
        def orientation(p,q,r):
            diag = (r[0]-q[0])*(q[1]- p[1])
            off_diag = (r[1]-q[1])*(q[0]-p[0])
            return diag - off_diag
        
        def distance(p,q):
            x_diff = p[0] - q[0]
            y_diff = p[1] - q[1]
            return x_diff**2 + y_diff**2
        
        def bottom_left(points):
            bottom = points[0]
            for p in points:
                if p[1] < bottom[1]:
                    bottom = p
            return bottom
        
        if len(trees) <= 1:
            return trees
        
        bm = bottom_left(trees)
        #sort
        def comparator(p,q):
            diff = orientation(bm,p,q) - orientation(bm,q,p)
            if diff == 0:
                return distance(bm,p) - distance(bm,q)
            else:
                return 1 if diff > 0 else -1
        trees.sort(key= cmp_to_key(comparator))
        
        #start in reverse
        i = len(trees) - 1
        
        while i >= 0 and orientation(bm,trees[len(trees)-1],trees[i]) == 0:
            i -= 1
        l = i + 1
        h = len(trees) - 1
        while l < h:
            trees[l],trees[h] = trees[h],trees[l]
            l += 1
            h -= 1
        stack = []
        stack.append(trees[0])
        stack.append(trees[1])
        for i in range(2,len(trees)):
            top = stack.pop()
            while orientation(stack[-1],top,trees[i]) > 0:
                top = stack.pop()
            stack.append(top)
            stack.append(trees[i])
        return stack

#############################
# Sum of Distances in Tree
#############################
#wooohooo TLE
class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        brute force would be to examine all nodes
        for each node i in n we would want get all paths from i to all n-ith nodes
        i can dfs from each node, and for each dfs call we need to change the destination node
        
        '''
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
            

        def bfs(node,end):
            seen = set()
            seen.add(node)
            q = deque([(node,0)])
            while q:
                curr_node,size = q.popleft()
                if curr_node == end:
                    return size
                for neigh in adj_list[curr_node]:
                    if neigh not in seen:
                        seen.add(neigh)
                        q.append([neigh,size+1])
        
        answer = [0]*n
        for start in range(n):
            count = 0
            for end in range(n):
                if start != end:
                    count += bfs(start,end)
                answer[start] = count
        
        return answer
            
class Solution:
    def sumOfDistancesInTree(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        if we split the tree on an edge containing xy
        we w would have two trees with subroot x and subroot y
        the value at subroot x would contain the sum of distances of all paths going to x, same for the subtree rooted at y
        then the answer would be be the sum rooted at x + sum rooted at y + the current answer for which we split on
        X: x@X
        Y: y@Y
        #parent
        ans[parent] = parent + X + Y
        algo:
            root the tree, and for each node consider the subtree S_{node} and all it descendants
            count[node] = the number of nodes in S_{node} 
            stsum[node] = sum of distances from node to the nodes in S_{node}
            we can get count and stum using post order taversal, L,R,N
            where on exiting some node, the count[child] and stsum[node] += stsum[child] + count[child]
            this wll gives us the answer for the root: ans[root] = stsum[root]
            NOW:
                if we have a node parent and its child, then these are neighboring nodes
                so ans[child] = ans[parent] - count[childe] + (N - conut[child])
                WHY?
                because there are count[child] nodes that are 1 easier to get from child than parent
                and N-count[child] nodes that are 1 harde to get from child than parent
            usuinf second preorder traversal we can update our answewr in linear time for all of our nodes
        '''
        adj_list = defaultdict(list)
        for a,b in edges:
            adj_list[a].append(b)
            adj_list[b].append(a)
        
        count = [1]*n #num nodes in subtree
        ans = [0]*n #sum distance to this node
        
        #first pass, get num nodes and sum distances to each node for each node in the subtree
        def dfs(node,parent):
            for child in adj_list[node]:
                #if we did not come back to the parent
                if child != parent:
                    dfs(child,node)
                    count[node] += count[child]
                    ans[node] += ans[child] + count[child]
        
        #using dfs again to get ans in total, which is just dfs on a dfs! (dp on dp)
        def dfs2(node,parent):
            for child in adj_list[node]:
                if child != parent:
                    ans[child] = ans[node] - count[child] + n - count[child]
                    dfs2(child, node)
        dfs(0,None)
        dfs2(0,None)
        print(count)
        print(ans)
        return ans

#########################
# Orderly Queue
#########################
#close one
class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
        '''
        lexographic just means alphabetical, we want the most smallest possible lexographic string
        what if i just greedily take the largest char in s[:k] and keep moving until the end
        and stop if a switch to end results in a larger lexographic string
        '''
        s = list(s)
        seen = set()
        seen.add("".join(s))
        while True:
            #find min char 
            max_char = max(s[:k])
            #find its index
            idx = s.index(max_char)
            del s[idx]
            s.append(max_char)
            print("".join(s))
            if "".join(s) in seen:
                break
            seen.add("".join(s))
        seen.add("".join(s))
        print(seen)
        return min(seen)
        
class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
        '''
        for k of size 1, we notice that this is just a rotation
        for k > 1, this allways every possible permutaiton of s to be allowed, 
        
        '''
        if k == 1:
            return min([s[i:]+s[:i] for i in range(len(s))])
        else:
            return "".join(sorted(s))
        

#could also just keep rotating all k chars
class Solution:
    def orderlyQueue(self, s: str, k: int) -> str:
        if k > 1:
            return ''.join(sorted(s))
        seen = set()
        while(s not in seen):
            seen.add(s)
            s = s[k:] + s[:k]

        return min(seen)

########################
# Slowest Key
########################
#using hashmap
class Solution:
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        '''
        get the differences for each key presssed
        record max in hasmap
        find the max 
        if tie return most lexographical key
        '''
        releaseTimes = [0] + releaseTimes
        mapp = {}
        max_duration = 0
        for i,key in enumerate(keysPressed):
            duration = releaseTimes[i+1] - releaseTimes[i]
            if key in mapp:
                mapp[key] = max(mapp[key],duration)
            else:
                mapp[key] = duration
            #set max duration
            max_duration = max(max_duration,duration)
        
        max_char = chr(96)
        for k,v in mapp.items():
            if v == max_duration:
                max_char = max(max_char,k)
        return max_char

#no hashmap
class Solution:
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        '''
        without using hasmap
        '''
        releaseTimes = [0] + releaseTimes
        max_duration = 0
        max_char = chr(96)
        for i,key in enumerate(keysPressed):
            duration = releaseTimes[i+1] - releaseTimes[i]
            #update maxduration
            if duration > max_duration:
                max_duration = duration
                max_char = key
            elif duration == max_duration and key > max_char:
                max_char = key
        
        return max_char

#just another way
class Solution:
    def slowestKey(self, releaseTimes: List[int], keysPressed: str) -> str:
        ansKey = keysPressed[0]
        ansDuration = releaseTimes[0]
        for i in range(1, len(keysPressed)):
            key = keysPressed[i]
            duration = releaseTimes[i] - releaseTimes[i-1]
            if duration > ansDuration or duration == ansDuration and key > ansKey:
                ansKey = key
                ansDuration = duration
        return ansKey
        
#######################
#   Reverse Linked List
#######################
#iteratiely
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        move through list using three pointers
        make next prev and advance preve
        '''
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev

#recursively
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        '''
        recursive approach is tricky
        assume we have list n1 -> n2 -> nk -> nk+1 ->...nM -> NONE
        assume from of list from n_{k+1} to m has been reverse and we are at nk
        we want nk+1 nodes to point to k
        so we want nk.next.next = nk
        '''
        def reverse(node):
            #base case
            if not node or not node.next:
                return node
            p = reverse(node.next)
            node.next.next = node
            node.next = None
            return p
        
        return reverse(head)

#####################
#   Shifting Letters
#####################
#brute force TLE
class Solution:
    def shiftingLetters(self, s: str, shifts: List[int]) -> str:
        '''
        brute force would be to apply each shift to each char 
        taking mods up to z
        '''
        mapp = {chr(i+97):i for i in range(26)}
        mapp_2 = {i:chr(i+97) for i in range(26)}
        #code back to int
        s = [mapp[char] for char in s]
        N = len(shifts)
        for i in range(N):
            for j in range(i+1):
                s[j] += shifts[i]
                s[j] %= 26


        return "".join([mapp_2[char] for char in s])

#woohoo
class Solution:
    def shiftingLetters(self, s: str, shifts: List[int]) -> str:
        '''
        we can find the final shifts for each letter
        example
        [3,5,9]
        abc
        for each leter it would be
        a got shifted 3+5+9
        b got shifted 5 + 9
        c got shifted 9
        which is just pref sum in reverse
        '''
        N = len(shifts)
        mapp = {chr(i+97):i for i in range(26)}
        mapp_2 = {i:chr(i+97) for i in range(26)}
        shifts = shifts[::-1]
        
        final_shifts = [shifts[0] % 26]
        for i in range(1,N):
            final_shifts.append(final_shifts[-1] + shifts[i] % 26)
            
        #reverse
        final_shifts = final_shifts[::-1]
        
        #code back to int
        s = [mapp[char] for char in s]
        
        #add shifts
        for i in range(N):
            s[i] += final_shifts[i]
            s[i] %= 26
        
        return "".join([mapp_2[char] for char in s])

#written better
class Solution:
    def shiftingLetters(self, s: str, shifts: List[int]) -> str:
        '''
        we notice that the first char gets shifted by the sum of all the shifts
        the next char gets shifts sum shifts - shifts[i]
        '''
        ans = ""
        sum_shifts = sum(shifts) % 26
        for i,char in enumerate(s):
            #get num
            idx = ord(char) - ord('a')
            #shift
            shift_char = ord('a') + ((idx + sum_shifts) % 26)
            #add to ans
            ans += chr(shift_char)
            #take away from sum_shifts
            sum_shifts -= shifts[i]
            sum_shifts %= 26
        
        return ans

#######################
# Largest Plus Sign
#######################
#TLE
class Solution:
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        '''
        first make the grid and seed the mins
        '''
        #create grid
        grid = [[1]*n for _ in range(n)]
        #seed zeros
        for x,y in mines:
            grid[x][y] = 0
        
        #solve the dp problem from r,c
        def dp(r,c):
            left,right,up,down = 0,0,0,0
            #go right
            j = c
            while j < n and grid[r][j] == 1:
                j += 1
            right = j - c
            #go left
            j = c
            while  j >= 0 and grid[r][j] == 1:
                j -= 1
            left = c - j
            #go down
            i = r
            while i < n and grid[i][c] == 1:
                i += 1
            down = i - r
            #go up
            i = r
            while i >= 0 and grid[i][c] == 1:
                i -= 1
            up = r - i
            return min(left,right,up,down)
        
        
        ans = 0
        for i in range(n):
            for j in range(n):
                ans = max(ans,(dp(i,j)))
        return ans

#dp, longest consecutive ones
class Solution:
    def orderOfLargestPlusSign(self, n: int, mines: List[List[int]]) -> int:
        '''
        instead of placing zeros for mins, just keep hashset of mines 
        call k, the order of the candidate pluse sign
        if we knew the longest possible arm length in each direciton up,down,left,right
        from a center, we would just take the min
        we can find each of these using dynamic programming
        if we are at (r,c) then dp[r][c] is zero of 0, else 1 plus the count of the coorindate in the same direction
        if we had 0 1 1 1 0 1 1 0
        the counts becomes
        0 1 2 3 0 1 2 0
        and the integers are either 1 more than their successor
        for each cell (r,c) we want dp[r][c] to end up being the mini for all 4 possible counts
        max conescutive ones from a cell, but be in all directions
        '''
        banned = set((tuple(mine) for mine in mines))
        dp = [[0]*n for _ in range(n)]
        
        largest_k = 0
        
        #first start with rows
        for r in range(n):
            k = 0
            #going right
            for c in range(n):
                k = 0 if (r,c) in banned else k + 1
                dp[r][c] = k
            #going left
            k = 0
            for c in range(n-1,-1,-1):
                k = 0 if (r,c) in banned else k + 1
                #overwrite if we are close going left
                if k < dp[r][c]:
                    dp[r][c] = k
                    
        #now start with cols, doing up down
        for c in range(n):
            k = 0
            #down
            for r in range(n):
                k = 0 if (r,c) in banned else k + 1
                #update smallest
                if k < dp[r][c]:
                    dp[r][c] = k
            #up
            k = 0
            for r in range(n-1,-1,-1):
                k = 0 if (r,c) in banned else k + 1
                #update smallest
                if k < dp[r][c]:
                    dp[r][c] = k
                #this was the last direction, we want the max
                if dp[r][c] > largest_k:
                    largest_k = dp[r][c]
                    
        return largest_k


#####################
# Best Meeting Point
#####################
#TLE
class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        '''
         we could bfs from each point, and for each point find the distance to all other point
         then just return the min
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(0,1),(0,-1),(1,0),(-1,0)]
        
        def bfs(x,y):
            seen = set()
            seen.add((x,y))
            q = deque([(x,y,0)])
            total_dist = 0
            while q:
                curr_x,curr_y,d = q.popleft()
                if grid[curr_x][curr_y] == 1:
                    total_dist += d
                
                for dx,dy in dirrs:
                    neigh_x = curr_x + dx
                    neigh_y = curr_y + dy
                    
                    #bounds
                    if (0 <= neigh_x < rows) and (0 <= neigh_y < cols) and (neigh_x,neigh_y) not in seen:
                        q.append((neigh_x,neigh_y,d+1))
                        seen.add((neigh_x,neigh_y))
                        
            return total_dist
        

        min_dist = float('inf')
        for i in range(rows):
            for j in range(cols):
                min_dist = min(min_dist,bfs(i,j))
        
        return min_dist

#not using bfs, still TLE
class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        '''
        we can just put all the cells that have 1 into a list
        then find the total distance from each cell to all those 1 points
        and solve for the min
        '''
        rows = len(grid)
        cols = len(grid[0])
        ones = []
        
        def total_dist(points,x,y):
            dist = 0
            for p1,p2 in points:
                dist += abs(x-p1) + abs(y-p2)
            return dist
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    ones.append((i,j))
                    
        ans = float('inf')
        for i in range(rows):
            for j in range(cols):
                dist = total_dist(ones,i,j)
                ans = min(ans,dist)
        return ans

#collect row and col indices and add up individual distances to their middles
class Solution:
    def minTotalDistance(self, grid: List[List[int]]) -> int:
        '''
        turn outs the best meeting point (in the 1 d case is the median)
        Case #1: 1-0-0-0-1
        Case #2: 0-1-0-1-0
        pick the median, to minimze the distance
        1-0-0-0-0-0-0-1-1
        0 1 2 3 4 5 6 7 8
        indexes of ones
        [0,7,8]
        if we pick 7, the distance ins minimized
        [7-0] + [8-7] = 8
        notice the mean is 5, which gives
        [5-0] + [7-5] + [8-5] = 10
        takeaway: as long as there is an equal number of points to the left and right of the meeting point
            the distance is minimized
        
        algo:
            collect row and col indices individually and sort them
            tind their middle, and calculat the distances using their middles
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        row_idxs = []
        col_idxs = []
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_idxs.append(i)
                    col_idxs.append(j)
        
        row_idxs.sort()
        col_idxs.sort()
        
        #get middle elemtns
        median_row = row_idxs[len(row_idxs)//2]
        median_col = col_idxs[len(col_idxs)//2]
        
        ans = 0
        for x,y in zip(row_idxs,col_idxs):
            ans += abs(x-median_row) + abs(y - median_col)
        
        return ans

######################################
#  Arithmetic Slices II - Subsequence
######################################
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        generate all subseqs and check
        '''
        N = len(nums)
        self.ans = 0
        
        def rec(idx,path):
            #if gotten to end of nums
            if idx == N:
                #check if current subsequenc is at least 3
                if len(path) < 3:
                    return
                #check sequence
                for i in range(1,len(path)):
                    if path[i] - path[i-1] != path[1] - path[0]:
                        return
                
                self.ans += 1
                return
            
            rec(idx+1,path)
            path.append(nums[idx])
            rec(idx+1,path)
            path.pop()
            
        rec(0,[])
        return self.ans

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        https://leetcode.com/problems/arithmetic-slices-ii-subsequence/discuss/92852/11-line-Python-O(n2)-solution
        we use a dp array of dictionarys 
        where dp[i][j] is the numbeber of 2-length subsequences ending at i with diff of j
        the kye is to store all 2 LENGTH arithmetic slices (which helps to build up the solution)
        from its sub problems while only adding 3 length arithmetic sequences to the total
        we then iterate over all pairs in the array, each (A[j],A[i]) is a two length slices with diff == A[i] - A[j] that we haven't seenn yet,
        and so we incrment dp[i][A[i]-A[j]] by 1 (but leave totale as it, becaues its not length 3)
        if there are any slices with A[i] - A[j] that finish at indcex j
        (if A[i] - A[j]) in dp[j]
        we extend them to index i and add to the total
        since any slice that terminated at index j would have at least length 3 terminating at i
        '''
        total = 0
        dp = [defaultdict(int) for i in nums]
        for i in range(len(nums)):
            for j in range(i):
                diff = nums[i] - nums[j]
                dp[i][diff] += dp[j][diff] + 1
                total += dp[j][diff]
        return total

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        first let us come up with a POSSIBLE state tranistion for finding arithmetic sequences
        we need at least two paramters: the first or last element, and the common difference
        f[i][d] denotes the number of arithmetic subsequences ending with A[i] and having common difference d
        then we can say
        for all j < i f[i][A[i]-A[j]] += f[j][A[i]-A[j]]
        BUT WHAT IF ALL f[i][d] are set to zero, how can we from new arithmetic subseqeunces
        WEAK ARITHMETIC SUBSEQUENCES:
            arithmetic subsequences of lenth 2
            any i,j pair, where i != j, can always from a valid subsequence
            if we can append a new element to this subsequence then the created new one must be arithmetic
        then!
        f[i][d] denotes the number of weak arithmetic subsequences that ends with A[i] and its common difference is d.
        and now:
        for all j < i, f[i][A[i] - A[j]] += (f[j][A[i] - A[j]] + 1).
        now we can just sum up all of f[i][d] to get weak, but we need to be careful to remove all weak arithmetic subsequences
            * all weak subsequences are just all (i,j) pairs == n(n-1) / 2
            * when we can append a new element to this subsequence, we are increasing the anser by 1
        '''
        N = len(nums)
        ans = 0
        counts = [defaultdict(int) for _ in range(N)]
        for i in range(N):
            for j in range(i):
                diff = nums[i] - nums[j]
                counts[i][diff] += counts[j][diff] + 1
                
        ans = 0
        for d in counts:
            ans += sum(d.values())
        return ans - (N*(N-1)//2)

######################
# Basic Calculator
#####################
class Solution:
    def calculate(self, s: str) -> int:
        '''
        if we used a stack, and pushed elements on starting with opening until we got to the close
        when we pop and evaluate, we actually get the wrong answer
        we need to reverse all elements on stack
        but simplye reverseing the elements in the stack will not work either, the - does not commute
        example A+B+C, we have  (A+B) + C = A + (B+C)
        note: (A-B) - C !+ (C-B) - A
        algo:
            1. iterate the expression string in reverse order, be careful when reading digits and non digits
            2. take care when reading partial string, like 123, which is 100 + 20 + 3
                take string and multiply by power of 10 and keep adding to current if we can
            3. once we come to a char that is not a digit, we push operand on to the stack
            4. when we encounter an (, the expression has ended, remember we reversed the string
            NOTE: we evalue sub  expressions in reverse first, for the final expression, we need to
            evalute left to right
            5. push the other non digits on to the stafk
            6. do this until  we get the final result, 
                its possible we that we dont have anymore chars left to process, but the stack is still non-empty
                this would happen if the main expression is not enclosed by parantheis
                so we are done evaluating the expression
            7. check if stack is non empty, if its is, we treat the elements in at as one final expression
            8. we cn also cover the original expression with a set  of parantehsis to avooid this extra call
        '''
        #helper  function to eval stack, to eval the sub expression of or the final stack of expresions
        def eval_exp(stack):
            res = stack.pop() if stack else 0
            #keep goinf until we hit closing
            while stack and stack[-1] != ")":
                sign = stack.pop()
                if sign == "+":
                    res += stack.pop()
                else:
                    res -= stack.pop()
            return res
        
        stack = []
        n, op = 0,0
        
        #goin backward
        for i in range(len(s)-1,-1,-1):
            char = s[i]
            #if digit, generat digit in revese
            if char.isdigit():
                op = (10**n*int(char)) + op
                n += 1
            #not a digit
            elif char == " ":
                if n:
                    #save the op and put back on to stack until we gt some non-digit
                    stack.append(op)
                    n, op = 0,0
                if ch  == "(":
                    res = eval_exp(stack)
                    stack.pop()
                    #append the eval reult to the stack,
                    #this could be a sub expression within paranthe
                    stack.append(res)
                #for other non digits just push onto the stack
                else:
                    stack.append(char)
        #push the las op to the stack
        if n:
            stack.append(op)
        #eval the enire last stack elements
        return eval_exp(stack)

#recursion, stack is similar
class Solution:
    def calculate(self, s: str) -> int:
        '''
        https://leetcode.com/problems/basic-calculator/discuss/142162/Beats-100-Stack-Recursion-(Java-Scala-Python)
        using recursion
        comparing stack vs recursion
        common ground:
            * when we hit a + or - update sign accordingling
        difference:
            using stack:
                when we meet '(', push curr result on to stack, and push current sign
                when we meet ')' result = result*stack.pop() + stack.pop()
            using recursion:
                we flip the operations
        '''
        self.i = 0
        def recurse(s):
            currSign, currRes = 1,0
            
            while self.i < len(s):
                if s[self.i] == ' ':
                    self.i += 1
                    continue
                elif s[self.i] == '+':
                    currSign = 1
                elif s[self.i] == '-':
                    currSign = -1
                elif s[self.i] == '(':
                    self.i += 1
                    currRes += currSign*recurse(s)
                    currSign = 1
                elif s[self.i] == ')':
                    return currRes
                #digit
                else:
                    currNum = int(s[self.i])
                    while self.i + 1 < len(s) and s[self.i+1].isdigit():
                        currNum = currNum*10 + int(s[self.i+1])
                        self.i += 1
                    currRes += currNum*currSign
                    currSign = 1
                self.i += 1
            
            return currRes
        
        return recurse(s)

class Solution:
    def calculate(self, s: str) -> int:
        '''
        just and additional way
        https://leetcode.com/problems/basic-calculator/discuss/62424/Python-concise-solution-with-stack.
        use single stack and eval sub expression immediatly after calculating
        then put pack on teh stack
        anything remaing on the stack needs to be evaluated
        '''
        output, cur, sign, stack = 0,0,1,[]
        
        for c in s:
            #if digit, do digit trick
            if c.isdigit():
                cur = cur*10+int(c)
            elif c in "+-":
                output += (cur*sign)
                cur = 0
                if c == '-':
                    sign = -1
                else:
                    sign = 1
            elif c == '(':
                stack.append(output)
                stack.append(sign)
                output = 0
                sign = 1
            elif c == ')':
                output += (cur*sign)
                output *= stack.pop()
                output += stack.pop()
                cur = 0
        
        return output + (cur*sign)

########################################
# Reachable Nodes In Subdivided Graph
#######################################
class Solution:
    def reachableNodes(self, edges: List[List[int]], maxMoves: int, n: int) -> int:
        '''
        we are given an undireted graph, along with a number for each node
        the number means we are making this edge a an undireced edge of number nodes
        return all nodes reachable from zero after subsidivding
        we can treat the original graph, with edge having a wiehgt
        when we travel along an edge (either direction) we can keep track of how much we use it
        at the end we want to know every node we reach in the graph
        plus the sum of the edges we used up!
        '''
        graph = defaultdict(dict) #nested dict
        for u,v,w in edges:
            graph[u][v] = w
            graph[v][u] = w
        
        pq = [(0,0)] #moves left,node
        dist = {0:0}
        used = {}
        ans = 0
        
        while pq:
            d,node = heapq.heappop(pq)
            #see if we've visted a node, remember we are first counting up nodes, then count up remaining edge lengths
            if d > dist[node]: #use up later
                continue
            #otherwise countthis node
            ans += 1
            
            #now check neighs
            for neigh,weight in graph[node].items():
                #M-d is how much further we can walk from thid node
                #wiehgt is how many new nodes there are on this edge
                #v is max used up from this edge
                v = min(weight, M-d)
                used[(node,neigh)] = v
                
                #d2 is total distance to reach neigh from node
                d2 = d + weight + 1
                if d2 < dist.get(neigh, M + 1):
                    heapq.heappush(pq,(d2,neigh))
                    dist[neigh] = d2
        #at the end, each edge (u,v,w) can be used with a max of we new nodes
        #a made of used[u,v] nodes from one side
        #and the other sie
        for u,v,w in edges:
            ans += min(w, used.get((u, v), 0) + used.get((v, u), 0))
        return ans

#########################
#   Maximum Number of Balloons
#########################
#simulate until we cannot
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        '''
        get the freq counts of text
        to make the word balloon, i need 2 l's, 2 o's, 1 b,a,and n
        it could be possible that i have n completions of the word balloon, and then one last partial
        '''
        counts_balloon = Counter('balloon')
        counts_text = Counter(text)
        
        def check_counts(counts_balloon,counts):
            for k,v in counts_balloon.items():
                if counts[k] < v:
                    return False
            return True
        
        ans = 0
        while check_counts(counts_balloon,counts_text):
            ans += 1
            counts_text -= counts_balloon
        
        return ans

class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        '''
        when we count up freqs of char in tezt, we realize we are limited in contructing balloon
        by the element that appears the smallest number of times
        l and o are use up twice in text
        set them back to 1, and take the min
        i,e to make one instace of balloon, we need 2 instances in text
        so if we have x occurences of l in text, we can really only make x//2 (assuming all other chars fit criteria)
        '''
        counts = Counter(text)
        return min(counts['b'],counts['a'],counts['o']//2,counts['l']//2,counts['n'])

#general solution
class Solution:
    def maxNumberOfBalloons(self, text: str) -> int:
        '''
        general case:
        say we wanted to make a pattern from a string
        to find the maximum patterns we can make from text we need the min char
        the min char would be the min count freq for all chars in text / count in pattern
        '''
        c_text = Counter(text)
        c_pattern = Counter('balloon')
        ans = float('inf')
        for char,counts in c_pattern.items():
            if counts > 0:
                ans = min(ans,c_text[char]//counts)
        
        return ans

##########################
# Reverse Only Letters
##########################
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        '''
        two pointer reverse trick but only reverse if english letters
        '''
        s = list(s)
        N = len(s)
        left, right = 0, N-1
        
        while left < right:
            #both can be swapped
            if s[left].isalpha() and s[right].isalpha():
                s[left],s[right] = s[right],s[left]
                left += 1
                right -= 1
            #left is alphas but not right
            elif s[left].isalpha() and not s[right].isalpha():
                right -= 1
            #right is alpha
            elif not s[left].isalpha() and s[right].isalpha():
                left += 1
            #both are not alpha
            else:
                left += 1
                right -= 1
        
#using second array and stack
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        '''
        we can put all chars if they are alpha into a list
        then when we retraverse s, if a char is alpha take from the end of the list
        otherwise add in the char back
        '''
        letters = [char for char in s if char.isalpha()]
        res = ""
        for c in s:
            if c.isalpha():
                res += letters.pop()
            else:
                res += c
        
        return res
        return "".join(s)

#############################
# Longest Turbulent Subarray
#############################
class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        '''
        a turbulent subarray obeys the properties:
        For i <= k < j:
        arr[k] > arr[k + 1] when k is odd, and
        arr[k] < arr[k + 1] when k is even.
        Or, for i <= k < j:
        arr[k] > arr[k + 1] when k is even, and
        arr[k] < arr[k + 1] when k is odd.
        
        find the largest subarray with the most adjacent peaks
        [9  4  2  10  7  8  8  1  9]
          5  2  -8  3  -1  0  7  -8
        0  1  1   0  1   0  1  1   0
         we want the longest subarray with alternating differences in sign
         we can have three numbers, -1,0,1, then we are just looking for the longest sequence of -1 and 1
        '''
        def compare(left,right):
            if left > right:
                return 1
            elif left < right:
                return -1
            else:
                return 0
        
        N = len(arr)
        res = 1
        start = 0
        for i in range(1,N):
            sign = compare(arr[i-1],arr[i])
            #if new block
            if sign == 0:
                start = i
            #if we are at second from last, and it stopped alternating, get size so far
            #otherwise keep extending
            elif i == N-1 or sign*compare(arr[i],arr[i+1]) != -1:
                res = max(res, i - start + 1)
                start = i
        
        return res

class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        '''
        a turbulent subarray obeys the properties:
        For i <= k < j:
        arr[k] > arr[k + 1] when k is odd, and
        arr[k] < arr[k + 1] when k is even.
        Or, for i <= k < j:
        arr[k] > arr[k + 1] when k is even, and
        arr[k] < arr[k + 1] when k is odd.
        
        find the largest subarray with the most adjacent peaks
        [9  4  2  10  7  8  8  1  9]
          5  2  -8  3  -1  0  7  -8
        0  1  1   0  1   0  1  1   0
         we want the longest subarray with alternating differences in sign
         we can have three numbers, -1,0,1, then we are just looking for the longest sequence of -1 and 1
        '''
        #we can use a while loop and push out a right pointer so long as we are alternating
        N = len(arr)
        #edge case
        if N == 1:
            return 1
        output = 0
        l,r = 0,0
        while r < N-1:
            while r < N-1 and (arr[r-1]<arr[r]>arr[r+1] or arr[r-1]>arr[r]<arr[r+1]):
                r += 1
            #watch for eleemnts of same
            while l < r and arr[l] == arr[l+1]:
                l += 1
            #sliding window update
            output = max(output, r-l + 1)
            l = r
            r += 1
        
        return output

#########################
# Spiral Matrix
########################
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        '''
        we repeat in steps:
            1. go left to right
            2. go up to down
            3. go right to left
            4  go down to up
        and at each point we shrink start and end rows and cols
        #remember to shrink the bounds too when going in reverse
        #before re traversing any row or col in reverse, we need to make sure we did not
        traverse already in a forward pass!
        '''
        if not matrix:
            return []
        start_row,start_col = 0,0
        end_row,end_col = len(matrix), len(matrix[0])
        
        ans = []
        
        while end_row > start_row or end_col > start_col:
            #check we are on new row
            if start_row < end_row:
                #first go left to right, on  this row
                for col in range(start_col,end_col):
                    ans.append(matrix[start_row][col])
                #use up a row
                start_row += 1

            #check again
            if start_col < end_col:
                #go up to down
                for row in range(start_row,end_row):
                    ans.append(matrix[row][end_col-1])
                #use up col
                end_col -= 1

            #check again
            if start_row < end_row:
                #go right to left
                for col in range(end_col-1,start_col-1,-1):
                    ans.append(matrix[end_row-1][col])
                #use up row
                end_row -= 1
            
            #check again
            if start_col < end_col:
                #go down to up
                for row in range(end_row-1,start_row-1,-1):
                    ans.append(matrix[row][start_col])
                #use up col
                start_col += 1

        return ans

#just a coulple more ways
#https://leetcode.com/problems/spiral-matrix/discuss/394774/python-3-solution-for-spiral-matrix-one-of-the-most-easiest-you-will-never-forget
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        if len(matrix) == 0:
            return res
        row_begin = 0
        col_begin = 0
        row_end = len(matrix)-1 
        col_end = len(matrix[0])-1
        while (row_begin <= row_end and col_begin <= col_end):
            for i in range(col_begin,col_end+1):
                res.append(matrix[row_begin][i])
            row_begin += 1
            for i in range(row_begin,row_end+1):
                res.append(matrix[i][col_end])
            col_end -= 1
            if (row_begin <= row_end):
                for i in range(col_end,col_begin-1,-1):
                    res.append(matrix[row_end][i])
                row_end -= 1
            if (col_begin <= col_end):
                for i in range(row_end,row_begin-1,-1):
                    res.append(matrix[i][col_begin])
                col_begin += 1
        return res
#########################################
# Minimize Max Distance to Gas Station
########################################
class Solution:
    def minmaxGasDist(self, stations: List[int], k: int) -> float:
        '''
        first start off with dp
        we let dp[n][k] be the answer for adding k more gas stations in the first n intervals, somehere
        say we are at interval i, stations[i+1] - stations[i]
        and we want to find dp[n+1][k]
        we can put x gas stations in the n+1'th interval for a best distance of deltans[n+1] / x+1
        then the rest of the intervals can be solve with an answer of dp[n][k-x]
        the answer if the min of these overall x
        i....i+1
        i...(x / (i+1 - i))....i + 1
        i ..(x / (i+2 - i))....i + 1
        '''
        N = len(stations)
        deltas = [stations[i+1] - stations[i] for i in range(N-1)]
        #dp[n][k] answers the questions, if i'm at the nth interval, what is the maximum distance for placing no more than k gas stations
        dp = [[0.0]*(k+1) for _ in range(N-1)]
        #base condtions for dp[0]
        for i in range(k+1):
            #print(deltas[0] / float(i+1))
            dp[0][i] = deltas[0] / float(i+1)
        
        #solve the recurrence
        for p in range(1,N-1):
            for k in range(k+1):
                ans = float('inf')
                for x in range(k+1):
                    best_so_far = deltas[p] / float(x+1)
                    prev = dp[p-1][k-x]
                    #print(best_so_far,prev)
                    ans = min(ans,max(best_so_far,prev))
                dp[p][k] = ans
        return dp[-1][k]

class Solution:
    def minmaxGasDist(self, stations: List[int], k: int) -> float:
        '''
        brute force,
        lets take a look at the deltas
        we can repeatedly add a gas staion to the current largest interval, so that when we add K of them in total
        this greedy approach is correct because if we left it alone, then our answer never goes down from that point
        algo:
            to find the largest interval, we keep track of how many parts count[i] the ith original interval has become
            for example, if we added 2 gas stations to it in total, there would be 3 parts
            the new largest interval of this section of roac would be deltas[i] / count[i]
        '''
        N = len(stations)
        deltas = []
        for i in range(N-1):
            deltas.append(float(stations[i+1] - stations[i]))
        
        #initally we only place one gas station at each interval
        count = [1]*(N-1)
        
        for k in range(k):
            #find interval with largest part
            best = 0
            for i in range(N-1):
                if deltas[i] / count[i] > deltas[best] / count[best]:
                    best = i
            
            count[best] += 1
        
        ans = 0
        for i in range(N-1):
            ans = max(ans, deltas[i] / count[i])
        return ans

#still TLE with heap
class Solution:
    def minmaxGasDist(self, stations: List[int], k: int) -> float:
        '''
        max heap TLE, still
        recall in approach 2, we were trying to find max interval, linearly
        we can just use a heap
        algo:
            repeated;y add a gas station to the next largest interval K times
            we can use a heap to know which interval is largest
            negate to max heap
        '''
        N = len(stations)
        pq = [] #(-part_length,original_length,num_parts)
        for i in range(N-1):
            x,y = stations[i],stations[i+1]
            key = (x-y,y-x,1)
            heappush(pq,(key))
        
        for _ in range(k):
            negative_length, orig_length,parts = heappop(pq)
            parts += 1
            key = (-(orig_length / float(parts)),orig_length,parts)
            heappush(pq,key)
        
        return -heappop(pq)[0]

#passing TLE with HEAP!
class Solution:
    def minmaxGasDist(self, stations: List[int], k: int) -> float:
        '''
        #https://leetcode.com/problems/minimize-max-distance-to-gas-station/discuss/113632/simple-10-line-python-on-logn-priority-queue-solution
        we know the maximum distance can only stations[-1] - stations[0] / k
        '''
        N = len(stations)
        #get bound for max length
        d = (stations[N-1] - stations[0]) / float(k)
        print(d)
        
        heap = []
        for i in range(N-1):
            n = max(1, int((stations[i+1]-stations[i]) / d))
            k -= (n-1)
            key = (float(stations[i]-stations[i+1]) / n, stations[i], stations[i+1], n)
            heap.append(key)
        heapify(heap)
        
        for i in range(k):
            (d,a,b,n) = heap[0]
            key =  ((a-b)/(n+1.0), a, b, n+1)
            heapreplace(heap,key)
        
        return -heap[0][0]

################################
# Intersection of Two Arrays II
################################
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        '''
        intersection just really means counts of common elements
        we can count up the number of elements in one array
        then pass the other array only adding to a result of there is a count available
        can't just use set intserctions becasue we want the multiplicty to be right
        
        '''
        res = []
        counts = Counter(nums1)
        
        for num in nums2:
            if counts[num] > 0:
                res.append(num)
                counts[num] -= 1
        return res

#sort, and two pointers
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        '''
        i can sort the lists and use two pointers
        and advance the smaller element or append to new res
        [4,5,9]
        [4,4,8,9,9]
        '''
        res = []
        nums1.sort()
        nums2.sort()
        i,j = 0,0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] < nums2[j]:
                i += 1
            elif nums1[i] > nums2[j]:
                j += 1
            else:
                res.append(nums1[i])
                i += 1
                j += 1
        
        return res

'''
Follow up questions
What if the given array is already sorted? How would you optimize your algorithm?

We can use either Approach 2 or Approach 3, dropping the sort of course. It will give us linear time and constant memory complexity.
What if nums1's size is small compared to nums2's size? Which algorithm is better?

Approach 1 is a good choice here as we use a hash map for the smaller array.
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

If nums1 fits into the memory, we can use Approach 1 to collect counts for nums1 into a hash map. Then, we can sequentially load and process nums2.

If neither of the arrays fit into the memory, we can apply some partial processing strategies:

Split the numeric range into subranges that fits into the memory. Modify Approach 1 to collect counts only within a given subrange, and call the method multiple times (for each subrange).

Use an external sort for both arrays. Modify Approach 2 to load and process arrays sequentially.
'''

##########################
# Expression Add Operators
#########################
#toooo hard
#i can 
    def addOperators(self, num: str, target: int) -> List[str]:
        '''
        lets try to use brute force recursion to generate all possible paths
        keep track of the path an evaluate
        of the path evals to target, it's a result
        we actually don't need to eval, but just apply the op to the next number
        '''
        self.paths = []
        N = len(num)
        def rec(idx,path):
            #print(path)
            #got to the end, now check
            if idx == N:
                #print("".join(path)[:-1])
                path = "".join(path)
                if path[-1] not in "+-*":
                    if eval(path) == target:
                        self.paths.append(path)
                return
            
            rec(idx+1,path+[str(int(num[idx:idx+1]))])
            rec(idx+1,path+[num[idx]+"+"])
            rec(idx+1,path+[num[idx]+"-"])
            rec(idx+1,path+[num[idx]+"*"])
                
        rec(0,[])
        return(self.paths)

class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        '''
        the issue is really we can make many diferent digits
        123 can be 1 2 3, 12 3, 1 23
        we can consinder an additional operation to +-*, we call this NO OP
        example we can go from 12 to 123 by:
        12*10 + 3
        initial algo:
            1. look at all possibilities to find all valid expression (small size of input hint to recursion and back tracking)
            2. our rec function will have an index, which represents the current digit we're looking at in num
            3. at every step, we have 4 different rc calls
                a. NO OP extends digit
                b. the other three +-*
            4.keep building expression until we exhaust nums
        there are a few problems with this algo though
        how do we eval this string expression effecitently to see if it is the target
        not eval function in python gives TLE
        WE CAN JUST EVALUATE THE EXPRESSIONS ON THE FLY.
        insteaf of just keeping tack of what expression the string us, we also keep track of the previous computed value after appluying the op
        now the problem would have been trivial had the ops only been + and -, but *
        the reason being is that the * has larger precedence than + -
        the * op would require the ACTUAL previous operand in the expression rather than the current value of the expression
        HOW DO WE HANDLE THIS?
        we simnply need to keep track of the last operand in our expression and how it modified the expression's value overall so that when we consider the * operator, we can reverse the previous operand and consider it for multiplations
        '''
        N = len(num)
        res = []
        
        def rec(idx,prev,curr,val,string):
            #idx is poistion in nums, prev is last evaluated expresion, curr is current exrpession
            if idx == N:
                #if we get target and no operand is left unprocessed
                if val == target and curr == 0:
                    res.append("".join(string[1:]))
                return
            #entedin curr by one digit
            curr = curr*10 + int(num[idx])
            str_op = str(curr)
            
            #to avoid cases like 1+05 and 1*05
            if curr > 0:
                #no OP recursion
                rec(idx+1,prev,curr,val,string)
            
            #addition
            string.append("+")
            string.append(str_op)
            rec(idx+1,curr,0,val+curr,string)
            string.pop()
            string.pop()
            
            #can substract ot multiply only if there are previous operands
            if string:
                
                #subtraction
                string.append('-')
                string.append(str_op)
                rec(idx+1,-curr,0,val-curr,string)
                string.pop()
                string.pop()
                
                #multiplacation
                string.append('*')
                string.append(str_op)
                rec(idx+1,curr*prev,0,val-prev+(curr*prev),string)
                string.pop()
                string.pop()
        
        rec(0,0,0,0,[])
        return res

class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        '''
        using eval function
        '''
        ans = []
        def rec(i, curr, result, ApplyOpt, size):
            if i == len(num):
                result = result + str(curr)
                if eval(result) == target:
                    ans.append(result)
                return

            if size == 0 or curr > 0:
                rec(i + 1, curr * 10 + int(num[i]), result, True, 1)  # Concatenate num[i]
            if ApplyOpt:
                rec(i, 0, result + str(curr) + '+', False, 0)  # add PLUS
                rec(i, 0, result + str(curr) + '-', False, 0) # add MINUS
                rec(i, 0, result + str(curr) + '*', False, 0) # add PRODUCT

        rec(0, 0, "", False, 0)
        return ans

class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        '''
        https://leetcode.com/problems/expression-add-operators/discuss/572099/C%2B%2BJavaPython-Backtracking-and-Evaluate-on-the-fly-Clean-and-Concise
        instead of keeping the whole string path and evaluating to check target
        keep track of current result and previous state
        + operator: newRes = resSoFar + num
        - operator: newRes = resSoFar - num
        * operator: we need to keep the prevvNum so that to calc newRes we need to minus prevNum wthen plus with prevNume*num
            newRes = resSFar - prevNum + prevNum*num
        good example eplaining:
            path = "1 + 2"
            cur = 3
            prev = 2 (got newly added from the previous backtrack call)

            in the for loop
                now = 4
                to the operand "*" -> we'll get the correct result 9 if we do
                curr - prev + prev * now
                3 - 2 + 2 * 4 = 9
        '''
        paths = []
        N = len(num)
        def rec(i,path,resSoFar,prev):
            if i == N:
                if resSoFar == target:
                    paths.append(path)
                return
            
            for j in range(i,N):
                #we do not need leading zeros
                if j > i and num[i] == "0":
                    break
                curr = int(num[i:j+1])
                if i == 0:
                    #place num with no op
                    rec(j+1,path+str(curr),resSoFar + curr,curr)
                else:
                    #carry out ops
                    rec(j+1,path+"+"+str(curr),resSoFar+curr,curr)
                    rec(j+1,path+"-"+str(curr),resSoFar-curr,-curr)
                    rec(j+1,path+"*"+str(curr),resSoFar-prev+prev*curr,prev*curr)
                    
        rec(0,"",0,0)
        return paths

#############################
# Distinct Subsequences
#############################
#brute force TLE
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        '''
        we want to return the NUMBER of distinct subsequences in s 
        that can make up t
        brute force would be to generate all subsequences, which would be O(2^len(s))
        and check if subsequence is t
        do brute force
        '''
        N = len(s)
        self.ans = 0
        def rec(i,path):
            if len(path) == len(t):
                if "".join(path) == t:
                    self.ans += 1
                    return
            for j in range(i,N):
                path.append(s[j])
                rec(j+1,path)
                path.pop()
        
        rec(0,[])
        return self.ans

#recursion
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        '''
        first start by asking whether or not t is a subsequence of s
        we can use the two pointer trick and check if we've advanced the t pointer all the way
        lets examine two scenarios where we could possibly encouter and also look at what we have
        1. first scneario is that chars do not match, in this case we need to progress both pointers
            i is the pointer to s
            j is the pointer to t
            (i,j) -> (i+1,j), think recursion transition
        2. second scenario, we hace two chars that match up, we can simply move (i+1,j+1)
            but we need to find all possible subsequence matches,
            it coul be possible to find same char at i at another one down the line
            r a b b b i t
            r a b b i t
                X Y Z
            from X, we can consider positions at Y and Z
            so we have one char match (i,j) -> (i+1,j+1)
            reject the char in s and move it up (i,j) -> (i+1,j)
        given points i and j, out recursive function would return the number of distinct sequences in substring
        s[i...M] == substring t[j...N] where M and N are s and t lengths
        two cases for our rec function:
            1. chars match and we hace two choices, ignore current s pointer and adavance, or advance both
            rec(i,j) = rec(i+1,j) + rec(i+1,j+1)
            2. second scenario they dont match, so move up our s pointer
            rec(i,j) = rec(i+1,j)
            AGAIN, think of the is subseq function and how this works
        base cases:
            i == M or j == N, we return a 1 if we have moved our j all the way through
            0 otherwise since we couldn't make a subsequence
        then we can cache along the way
        pseudo recursion:
            rec(i,j):
                if (i== M or j == n){
                return j == N ? 1 : 0
                }
                if (s[i]==t[j]){
                return rec(i+1,j) + rec(i+1,j+1)
                }
                else{
                return rec(i+1,j)
                }
        '''
        memo = {}
        M = len(s)
        N = len(t)
        #i points to s, j points to t
        
        def rec(i,j):
            #base case, ending, but also if length to be exploraed in s is < t
            #we can prune to see if there is a subsequence
            if i == M or j == N or M - i < N - j:
                return 1 if j == len(t) else 0
            
            if (i,j) in memo:
                return memo[(i,j)]
            
            #note we always move the pointer
            ans = rec(i+1,j)
            
            #match, we move bnoth and add
            if s[i] == t[j]:
                ans += rec(i+1,j+1)
            
            memo[(i,j)] = ans
            return ans
        
        return rec(0,0)

#another recursive way
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        memo = {}
        M = len(s)
        N = len(t)
        def rec(i,j):
            if j == N:
                return 1
            if i == M:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = rec(i+1,j)
            if s[i] == t[j]:
                ans += rec(i+1,j+1)
            memo[(i,j)] = ans
            return ans
        
        return rec(0,0)

class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        '''
        translating from recurison to dp
        notice that in these problems, recursion top down, meaning we fill up stack until we hit base case
        then function calls in stack get popped off
        we actually start getting returns at the end,
        so in DP we can start backwards!
        init dp array, add 1 to both dims for empty string base caes
        algo:
            1. init 2d dp array M x N
            2. recall rec(i,j) means the number of distint subsequences in s[i..M] == t[j..N]
            3. loop from M-1 to 0 nested N-1 to 0
            4. inint base condtions
                dp[M][j], means we've moved all the way through s, so theres no possibility, 0
                dp[i][N], weve moved all the way through t, so we must have at least 1
            5. state transitions:
                dp[i][j] = dp[i+1][j] FIRSRT
              then if match s[i] == t[j]:
                dp[i][j] += dp[i+1][j+1]
        '''
        M = len(s)
        N = len(t)
        
        dp = [[0]*(N+1) for _ in range(M+1)]
        
        #base conditions
        for i in range(M+1):
            dp[i][N] = 1
            
        for i in range(M-1,-1,-1):
            for j in range(N-1,-1,-1):
                
                dp[i][j] = dp[i+1][j]
                
                if s[i] == t[j]:
                    dp[i][j] += dp[i+1][j+1]
        
        return dp[0][0]

#space optimized
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        '''
        using optimal space just keep on dp row and reset
        notice that in the 2d dp, we always started with a 1 in the end column
        keep this 1 as a prev state
        note we also could have done two diferent arrays, update the firsst
        and just recopy
        '''
        M = len(s)
        N = len(t)
        
        dp = [0]*(N)
        
        for i in range(M-1,-1,-1):
            prev = 1
            for j in range(N-1,-1,-1):
                curr = dp[j]
                if s[i] == t[j]:
                    dp[j] += prev
                prev = curr
        
        return dp[0]

#two arrays
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        '''
        using optimal space just keep on dp row and reset
        notice that in the 2d dp, we always started with a 1 in the end column
        keep this 1 as a prev state
        note we also could have done two diferent arrays, update the firsst
        and just recopy
        '''
        M = len(s)
        N = len(t)
        
        first_dp = [0]*(N+1)
        second_dp = [0]*(N+1)
        first_dp[-1] = 1
        second_dp[-1] = 1
        
        for i in range(M-1,-1,-1):
            for j in range(N-1,-1,-1):
                second_dp[j] = first_dp[j]
                if s[i] == t[j]:
                    second_dp[j] += first_dp[j+1]
            first_dp = second_dp
            second_dp = [0]*(N+1)
            second_dp[-1] = 1
        print(first_dp)
        return first_dp[0]

###########################
# Find Winner on a Tic Tac Toe Game
##########################
#right idea
class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        '''
        build the board using moves
        check counts of X and O,
        return if any rows,cols,diags have 3
        '''
        board = [["*"]*3 for _ in range(3)]
        for move,pos in enumerate(moves):
            row,col = pos
            if move % 2 == 0: #X's turn
                board[row][col] = 'X'
            else:
                board[row][col] = 'O'
                
        
        print(board[0])
        print(board[1])
        print(board[2])
        
        #now check rows
        for row in range(3):
            counts = Counter(board[row])
            if counts['X'] == 3:
                return 'A'
            if counts['O'] == 3:
                return 'B'
        
        #now check cols
        for row in range(3):
            curr_col = []
            for col in range(3):
                curr_col.append(board[row][col])
            counts = Counter(curr_col)
            if counts['X'] == 3:
                return 'A'
            if counts['O'] == 3:
                return 'B'
        
        #now check diags
        counts = Counter()
        for row in range(3):
            for col in range(3):
                counts[board[row][col]] += 1
        if counts['X'] == 3:
            return 'A'
        if count['O'] == 3:
            return 'B'
        
        #check anti diag
        counts = Counter()
        for row in range(3):
            for col in range(3):
                if row+col == 2:
                    counts[board[row][col]] += 1
        if counts['X'] == 3:
            return 'A'
        if count['O'] == 3:
            return 'B'
        
        
        #now winner but board is filled
        if len(moves) == 9:
            return "Draw"
        
        return "Pending"

#better solution, but still mn
class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        '''
        build the board and check,
        instead of using X's and O's, use 1 for player A and -1 for player B
        why? if sum up A's we should 3, if we sum up B's we should get -1
        recall diags have row == col
        anti diags have row + col = n - 1
        '''
        N = 3
        board = [[0]*N for _ in range(N)]
        
        #define check functions to call after every move
        
        def checkRow(row,player):
            for col in range(N):
                if board[row][col] != player:
                    return False
            return True
        
        def checkCol(col,player):
            for row in range(N):
                if board[row][col] != player:
                    return False
            return True
        
        def checkDiag(player):
            for row in range(N):
                if board[row][row] != player:
                    return False
            return True
        
        def checkAntiDiag(player):
            for row in range(N):
                if board[row][N-1-row] != player:
                    return False
            return True
        
        player = 1
        for move in moves:
            row,col = move
            board[row][col] = player
            #check
            if checkRow(row,player) or checkCol(col,player) or (row == col and checkDiag(player)) or (row + col == N-1 and checkAntiDiag(player)):
                return 'A' if player == 1 else 'B'
            
            player *= -1
        
        return 'Draw' if len(moves) == N*N else 'Pending'

class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        '''
        store arrays for row, col, each keep counts along each row and col
        also need one for diag and anti daig
        this prevents us from checking the whole board again
        '''
        N = 3
        rows = [0]*N
        cols = [0]*N
        diags = 0
        anti_diags = 0
        player = 1
        for row,col in moves:
            #always update a row and col count
            rows[row] += player
            cols[col] += player
            
            #diag
            if row == col:
                diags += player
            #anti-daig
            if row + col == N-1:
                anti_diags += player
                
            #check
            if any(abs(line) == N for line in (rows[row], cols[col], diags, anti_diags)):
                return 'A' if player == 1 else 'B'
            player *= -1
        
        return 'Draw' if len(moves) == N*N else 'Pending'

#####################
# Max Consecutive Ones
#####################
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        '''
        you can just do a linear scan and update when there is a 1
        if there isn't take maxsofar, update, and reset
        '''
        ans = 0
        ans_so_far = 0
        for num in nums:
            if num == 1:
                ans_so_far += 1
                ans = max(ans,ans_so_far)
            else:
                ans_so_far = 0
                ans = max(ans, ans_so_far)
        
        return ans

class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        ans = 0
        curr = 0
        
        for num in nums:
            if num == 1:
                curr += 1
            else:
                ans = max(ans,curr)
                curr = 0
        
        return max(ans,curr)

#just a tidbit, here's a fancy oneliner, where we just pull the lengths
#after joining the array and splitting on zero
def findMaxConsecutiveOnes(self, nums):
  return max(map(len, ''.join(map(str, nums)).split('0')))


################################################################
# Maximum Length of a Concatenated String with Unique Characters
################################################################
#brute force path generation and check
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        what if i examine all possible concatenations
        then just check if this has all unique, i.e len(set(curr concat)) == len(curr concat)
        '''
        N = len(arr)
        self.ans = 0
        
        def rec(idx,path):
            if path:
                if len(set("".join(path))) == len("".join(path)):
                    self.ans = max(self.ans,len("".join(path)))
            
            for j in range(idx,N):
                path.append(arr[j])
                rec(j+1,path)
                path.pop()
                
        rec(0,[])
        return self.ans

#iterative w/output limit exceeded
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        iterative, without bit masking, just build it up
        algo:
            inint rest with empty string
            pass through each word in arr
            go through res
            update max along the way
        '''
        res = [""]
        ans = 0
        for w in arr:
            for i in range(len(res)):
                #form new word and set length check, this is the part that can be optimized to O(1)
                new_res = res[i] + w
                if len(new_res) != len(set(new_res)):
                    continue
                
                #add valid option to result
                res.append(new_res)
                ans = max(ans,len(new_res))
        
        return ans

#iteraive with bit masking
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        # Results initialized as a Set to prevent duplicates
        results = set([0])
        best = 0
                
        # Check each string in arr and find the best length
        for word in arr:
            best = max(best, self.addWord(word, results))
        return best
    
    def addWord(self, word: str, results: List[str]) -> int:
        # Initialize an int used as a character set
        char_bitset = 0
        best = 0
        for char in word:
            # Define character mask for currrent char
            mask = 1 << ord(char) - 97

            # Bitwise AND check using character mask
            # to see if char already found and if so, exit
            if char_bitset & mask > 0:
                return 0

            # Mark char as seen in charBitSet
            char_bitset += mask

        # If the initial bitset is already a known result,
        # then any possible new results will have already been found
        if char_bitset in results:
            return 0

        # Iterate through previous results only
        for res in list(results):
            # If the two bitsets overlap, skip to the next result
            if res & char_bitset:
                continue

            # Build the new entry with bit manipulation
            new_res_len = (res >> 26) + len(word)
            new_char_bitset = char_bitset + res & ((1 << 26) - 1)

            # Merge the two into one, add it to results,
            # and keep track of the longest so far
            results.add((new_res_len << 26) + new_char_bitset)
            best = max(best, new_res_len)
        return best

#recursion with bit masking check
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        downside of iterative solution is that we need to keep a large amount
        we can use backtracking
        recursive function returns the best length seen so far
        resMap to store character counts, any count more than 1 means there are duplicates and is invalid
        first encode each arr to its bitset and length combo
        then use backtracking to find the max
        '''
        #convert each arr to bitset with length, store bitsets in set
        arr_set = set()
        def get_bitset(word):
            '''Returns bitset of word if we can make it'''
            char_bitset = 0
            for c in word:
                mask = 1 << ord(c) - 97
                if char_bitset & mask:
                    return
                char_bitset += mask
            
            return char_bitset + (len(word) << 26)
        
        def dfs(arrayBitSet,pos,res):
            old_chars = res & ((1 << 26)-1)
            old_len = res >> 26
            best = old_len
            
            #pass through remainin results
            for i in range(pos,len(arrayBitSet)):
                new_chars = arrayBitSet[i] & ((1 << 26)-1)
                new_len = arrayBitSet[i] >> 26
                
                #if two bitsets overlap, skip
                if new_chars & old_chars:
                    continue
                
                #combine res
                new_res = new_chars + old_chars + (new_len + old_len << 26)
                best = max(best, dfs(arrayBitSet, i+1,new_res))
            
            return best
        
        for word in arr:
            converted = get_bitset(word)
            if converted:
                arr_set.add(converted)
        
        print(arr_set)
        #convert set back to array for iteration
        temp = list(arr_set)
        return dfs(temp, 0,0)

########################
# Break a Palindrome
#######################
#close one
#failed to 'abb'

class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        '''
        first there is an edge case, length 1, will always be palindrome
        i can  try chaning the first non a char to a
        if the string only has a, change the last char to b
        there can multiple changes, but we want the most lexogrpical one
        
        '''
        #edge case
        N = len(palindrome)
        if N == 1:
            return ""
        #pull apart chars
        listPal = list(palindrome)
        
        #nov we can try placing letters a-z at the most significan spot in the palindrom
        start = 0
        madeChange = False
        for i in range(26):
            #get curr char
            currChar = chr(i+ord('a'))
            for j in range(start,N):
                if listPal[j] != currChar:
                    listPal[j] = currChar
                    madeChange = True
                    break
            start += 1
            if madeChange:
                break

        return "".join(listPal)
            
#brute force TLE
class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        '''
        lets take a step back, if we wanted to make a paldromic string non palindromic
        we only need to make a change in the first half of the string
        recall odd even  casses, odd length means we can have a center point
        even cases we only first half is reverse of second half
        brute force would be try chaing each char in the string with each of the 25 -th chars
        then return the most lexographic one
        '''
        def isPal(string):
            N = len(string)
            l,r = 0,N-1
            while l < r:
                if string[l] != string[r]:
                    return False
                l += 1
                r -= 1
            return True
        N = len(palindrome)
        if N == 1:
            return ""
        #pull apart chars
        listPal = list(palindrome)
        possible = []
        
        #nov we can try placing letters a-z at the most significan spot in the palindrom
        for i in range(26):
            #get curr char
            currChar = chr(i+ord('a'))
            for j in range(N):
                temp = listPal[:]
                temp[j] = currChar

                if not  isPal("".join(temp)):
                    possible.append("".join(temp))

        return min(possible)

#using the trick
class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        '''
        using the hint, make the first non a char a a then return
        make last one b

        lets take a step back, if we wanted to make a paldromic string non palindromic
        we only need to make a change in the first half of the string
        recall odd even  casses, odd length means we can have a center point
        even cases we only first half is reverse of second half
        brute force would be try chaing each char in the string with each of the 25 -th chars
        then return the most lexographic one
        greedily change every char of the string to the smallest different char among all the resulitng strings
        when there is a in the string: return it immediealty
        if we can't do this (meaning they are all a) just make it a b
        '''
        N = len(palindrome)
        listPal = list(palindrome)
        
        if N == 1:
            return ""
        
        #its a palindrome so only need to change the first half
        for i in range(N//2):
            if listPal[i] != 'a':
                listPal[i] = 'a'
                return "".join(listPal)
        
        listPal[N-1] = 'b'
        return "".join(listPal)
        
###########################
# N-th Tribonacci Number
###########################
#top down recursion
class Solution:
    def tribonacci(self, n: int) -> int:
        '''
        just convert usiing the recurrence,
        first top down with memo
        T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.
        '''
        memo = {}
        
        def trib(n):
            if n == 0:
                return 0
            elif (n == 1) or (n == 2):
                return 1
            elif n in memo:
                return memo[n]
            ans = trib(n-3) + trib(n-2) + trib(n-1)
            memo[n] = ans
            return ans
        
        return trib(n)

#bottom up dp
class Solution:
    def tribonacci(self, n: int) -> int:
        if n == 0:
            return 0
        if (n==1) or (n==2):
            return 1
        dp = [0,1,1]
        for i in range(3,n+1):
            dp.append(sum(dp[-3:]))
        return(dp[-1])

#keeping three states and update
class Solution:
    def tribonacci(self, n: int) -> int:
        '''
        we don't need to keep track the array,
        just store three values and update in linear time
        '''
        if n == 0:
            return 0
        t0 = 0
        t1 = 1
        t2 = 1
        for _ in range(n-2):
            tn =  t0 + t1 + t2
            t0 = t1
            t1 = t2
            t2 = tn
        return t2

#another way would be to precompute in the cnosturctor, then query n
class Tri:
    def __init__(self):
        n = 38
        self.nums = nums = [0] * n
        nums[1] = nums[2] = 1
        for i in range(3, n):
            nums[i] = nums[i - 1] + nums[i - 2] + nums[i - 3]
                    
class Solution:
    t = Tri()
    def tribonacci(self, n: int) -> int:
        return self.t.nums[n]

#using matrix exponential
import numpy as np
class Solution:
    def tribonacci(self, n: int) -> int:
        '''
        we can use matrix exponential along wit fast power recurrence to find the answer
        first we are given t0 = 0, t1 = 1,t2 = 1
        tn = t(n-3) + t(n-2) + t(n-1)
        we have the system:
            t1 = t0*0 + t1*1 + t2*0 
            t2 = t0*0 + t1*0 + t2*1
            t3 = t0*1 + t1*1 + t2*1
        which is just a matrix transition with initial state
        [t1,t2,t3] = [[0, 1, 0], [0, 0, 1], [1, 1, 1]] * [0,1,1]
        there is a recusrive fast pow multiple to get matrix exponenital in log(size(matrix)) time
        
        '''
        def fastPow(matrix,pow):
            if pow == 1:
                return matrix
            half = fastPow(matrix,pow//2)
            res = np.dot(half,half)
            if pow & 1:  # If pow is an odd number
                res = np.dot(res, matrix)  # res *= matrix
            return res
        
        if n == 0: 
            return 0
        matrix = np.matrix([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
        M0 = np.matrix([[0], [1], [1]])
        Mn = np.dot(fastPow(matrix, n), M0)  # Mn = matrix^n * M0
        return Mn.item((0, 0))

####################################################
#  Shortest Path in a Grid with Obstacles Elimination
#####################################################
#bfs keeping track of state
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        '''
        if there were no limits on how many obstables we coul remove, then the answer if just the manhattan distance
        if the number of times we can remove an obstable is greater than the manhattan distance, the answer is just the manhattan distance
        in some cases, we need to take a detour, and explore all posible pathas
        in nomralk bfs, explore frontier while keeping number of steps so far
        need to keep track of remainin quotq
        algo:   
            1. same dfs using deque
            2. for each entry in the deque, we keep track of the state (combination of coords and remaning quota)
            3. at each iteration of the loop pop, the elemnt contains the distance from the starting point, as well as curr state
            4. in same iteration, evalute next moves, only after seeing we have enough remeaing
            5. hash state, instead of typical coords
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        #if we have enough, just return manhattan
        if k >= rows + cols - 2:
            return rows + cols - 2
        
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        state = (0,0,k)
        q = deque([(0,state)])
        seen = set(state)
        
        while q:
            steps, (row,col,k) = q.popleft()
            
            #can reach end
            if (row,col) == (rows-1,cols-1):
                return steps
            
            for dx,dy, in dirs:
                new_x = row + dx
                new_y = col + dy
                
                #in bounds
                if (0 <= new_x < rows) and (0 <= new_y < cols):
                    #is obstacle use up quota
                    new_k = k - grid[new_x][new_y]
                    new_state = (new_x,new_y,new_k)
                    #add only to seen if it qualifies
                    if new_k >= 0 and new_state not in seen:
                        seen.add(new_state)
                        q.append((steps+1,new_state))
        
        return -1

##########################
# Transform to Chessboard
#########################
class Solution:
    def movesToChessboard(self, board: List[List[int]]) -> int:
        '''
        bits of intuition:
            1. if a row can be transormed into a chess board if there are tow unique rows, (101,010), which are inverses of each other
            2. count of 1's must be count of zeros in each row, or differ by more than 1
            if rule 1 passes, we don't need to check rule 2
        if both passed, it is possible to make board into chess board
        now how do we check the min counts to transform
        WE ONLY NEED TO COUNT FIRST ROW AND COLUMN, because the by rules 1 and 2, it should already be valid
        
        1. take first row, we don't know if first cell should be 1 or 0, if we assume 0, we know expcted counts 1 and count 0
        2. count the difference against actual values (expected - count), numswap should be diffCnt // 2
            if count is odd number, then it means the first cell cannot be 0, choose 1 as first cell\
        3. if bvoth choosing 0 and 1 makes even diffCount, then we chose the on with the smallest number of swaps
        '''
        N = len(board)
        
        #count up rows uniquely
        rows = []
        for i in range(N):
            rows.append("".join([str(cell) for cell in board[i]]))
        
        uniqRowsCount = Counter(rows)
        uniqRows = list(uniqRowsCount.keys())
        
        #more than two unique
        if len(uniqRows) != 2:
            return -1
        
        #if there are only two unique, their counts cannot differ by more than 1
        if abs(uniqRowsCount[uniqRows[0]] - uniqRowsCount[uniqRows[1]]) > 1:
            return -1
        
        #for each of the two unique rows, their counts of 1's and zeros should not be more than 1
        if abs(uniqRows[0].count('0') - uniqRows[1].count('1')) > 1:
            return -1
        
        #swapping a COLUMN, does not change the relative cell positions, same thing with ROWS
        #there fore, we only need to verify that the positions alteranted
        #CHECK THE TWO UNIQ ROWS FOR alternating
        for i in range(len(uniqRows[0])):
            if uniqRows[0][i] == uniqRows[1][i]:
                return -1
        
        #starting with first col, how many bits does it differ from the expected row cells if we start at 0
        row_diff = 0
        for j in range(N):
            row_diff += board[0][j] != j % 2
        
        if N % 2 == 0: #even size
            #we know there must be the same number of 0's and 1's, so we take the min of row_diff or n - row_diff
            #this gives us the min diff for rows
            rows_diff = min(row_diff, N - row_diff)
            
        elif row_diff % 2 != 0:
            row_diff = N - row_diff
            
        #to find the expected row cell positions with k differences, we swap with half the row_diff
        col_swaps = row_diff // 2
        
        #nor do the same checking the first col
        col_diff = 0
        for i in range(N):
            col_diff += board[i][0] != i % 2
        
        if N % 2 == 0:
            col_diff = min(col_diff,N-col_diff)
        elif col_diff % 2 != 0: 
            #first cell has to be one
            col_diff = N - col_diff
            
        row_swaps = col_diff // 2 
        return col_swaps + row_swaps

############################
# Unique Email Addresses
############################
class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        '''
        if there is a + sign, everything between + and @ gets discarded
        periods are use to concatnate
        then just add to a hash set
        there is no + in the domain name
        
        first get the domain name
        '''
        valid = set()
        for em in emails:
            #fist split on @
            first_split = em.split("@")
            local,domain = first_split[0],first_split[1]
            new_local = ""
            for ch in local:
                if ch == "+":
                    break
                elif ch == ".":
                    continue
                else:
                    new_local += ch
            new = new_local+"@"+domain
            valid.add(new)
        
        return len(valid)


########################
# Shortest Distance from All Buildings
########################
#almost had it, we can't simply walk around a building
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        '''
        this is similar to the best meeting point problem
        the kicker is that we have an obstable of , that has two
        i can bfs from each cell, until i hit all the buildings, then find this total distance
        then do this for all cells and return the min
        '''
            dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        ones = 0
        zeros = []
        rows = len(grid)
        cols = len(grid[0])
        
        if not grid or not grid[0]:
            return -1
        

        #grab zero cells
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0:
                    zeros.append((i,j))
                elif grid[i][j] == 1:
                    ones += 1
                    
        #no empty land
        if len(zeros) == 0:
            return -1
        
        #bfs call frome each cell
        def bfs(cell,seen):
            x,y = cell
            seen.add((cell))
            
            total_dist = 0
            houses = 0
            q = deque([(x,y,0)])
            while q:
                curr_x,curr_y,curr = q.popleft()
                #generate neighbors
                for dx,dy in dirrs:
                    neigh_x = curr_x + dx
                    neigh_y = curr_y + dy
                    #bounds check
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        #obstacle check
                        if grid[neigh_x][neigh_y] == 2:
                            continue
                        #if zero and not seen
                        if (neigh_x,neigh_y) not in seen and grid[neigh_x][neigh_y] == 0:
                            seen.add((neigh_x,neigh_y))
                            q.append((neigh_x,neigh_y,curr+1))
                        #if its a one and not seen
                        elif (neigh_x,neigh_y) not in seen and grid[neigh_x][neigh_y] == 1:
                            total_dist += curr + 1
                            houses += 1
                            seen.add((neigh_x,neigh_y))
            
            return total_dist if houses == ones else -1
        
        ans = float('inf')
        for zero in zeros:
            ans = min(ans,bfs(zero,set()))
        
        return ans
                       
#official solution 1: empty land to house
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        '''
        if we did not have obstacles, we could try placing a house on each empty cell
        and for each placment find the total distance to all house, then return the minimum
        but just using manhattan distance causes problems because of obstacles
        BFS from empty land to all houses
        keep track of num houses we've seenm and if we can't get to all the houses, it can't be done
        algo:
            1. for each empty cell, grid[i][j] == 0, start BFS
            2. when we hit a house increase houses reahced by 1, and increase total disum by curr distance
            3. if we hit all of the houses, return ans
            4. update total distances to min
        '''
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        ones = 0
        zeros = []
        rows = len(grid)
        cols = len(grid[0])
        totalHouses = 0
        minDistance = float('inf')
        
        def bfs(row,col):
            seen = set()
            seen.add((row,col))
            distanceSum = 0
            housesReached = 0
            q = deque([(row,col,0)])
            
            while q and housesReached != totalHouses:
                curr_row,curr_col,steps = q.popleft()
                #if cell is house, then add to distance
                if grid[curr_row][curr_col] == 1:
                    distanceSum += steps
                    housesReached += 1
                    continue 
                
                #neigh generation
                for dx,dy in dirrs:
                    neigh_x = curr_row + dx
                    neigh_y = curr_col + dy
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        if (neigh_x,neigh_y) not in seen and grid[neigh_x][neigh_y] != 2:
                            seen.add((neigh_x,neigh_y))
                            q.append((neigh_x,neigh_y,steps + 1))
            
            #now if we were unable to reach all houses, set all visited cells to 2, since we could not reach
            if housesReached != totalHouses:
                for row,col in seen:
                    if grid[row][col] == 0:
                        grid[row][col] = 2
                return float('inf')
            
            #otherwise return answer
            return distanceSum

        #count up houses
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    totalHouses += 1
        #find min
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0:
                    minDistance = min(minDistance,bfs(i,j))
        #if now change
        if minDistance == float('inf'):
            print("here")
            return -1
        else:
            return minDistance

#official solution 2: land to empty house, modify gird in place
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        '''
        in first apporach we start from empty land to house, here we start from house to empty land
        if we can reach a house from empty land, then we can reach empty land to house
        if there are fewere houses, this apporach is better, on average is is not an improvement
        in the last bfs, we wanted total distances from emptyy spot to all houses
        BUT, starting from house to empty land, we generate partial distances
        hence we need extra space to store partial sum dsitances
        algo:
            1. for each house, start bfs
                for each empty cell we reach, increase the cell's sum of distancea by the steps taken to reach that cell from this house
                for each empty cell, also icnremaent cells house counter by 1
            2. after traversing all houses, get the min distance from all empty cells which have the right number of houses
            3. if is is possible for all house to reach a specific empty land cell, return min found, otherwise return -1
        '''
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        rows = len(grid)
        cols = len(grid[0])
        totalHouses = 0
        minDistance = float('inf')
        
        distances = [[[0]*2 for _ in range(cols)] for _ in range(rows)]
        
        def bfs(row,col):
            seen = set()
            seen.add((row,col))
            q = deque([(row,col,0)])
            
            while q:
                curr_row,curr_col,steps = q.popleft()
                #if cell is house, then add to distance
                if grid[curr_row][curr_col] == 0:
                    #distance entry
                    distances[row][col][0] += steps
                    distances[row][col][1] += 1
                    
                #neigh generation
                for dx,dy in dirrs:
                    neigh_x = curr_row + dx
                    neigh_y = curr_col + dy
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        if (neigh_x,neigh_y) not in seen and grid[neigh_x][neigh_y] == 0:
                            seen.add((neigh_x,neigh_y))
                            q.append((neigh_x,neigh_y,steps + 1))

        #count up houses
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    totalHouses += 1
                    bfs(i,j)
        
        #check all empty lands with count equal total houses and get min
        for i in range(rows):
            for j in range(cols):
                if distances[i][j][1] == totalHouses:
                    minDistance = min(minDistance,distances[i][j][0])
        #if now change
        if minDistance == float('inf'):
            return -1
        else:
            return minDistance

#the only one that passes
class Solution(object):
    def shortestDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """        
        def bfs(y0, x0):
            seen = [[0]*n for _ in range(m)]
            q = collections.deque([(y0, x0, 0)])
            while q:
                y, x, d = q.popleft()
                for yp, xp in {(y-1,x), (y+1, x), (y, x+1), (y, x-1)}:
                    if 0 <= yp < m and 0 <= xp < n and grid[yp][xp] == 0 and seen[yp][xp] == 0:
                        seen[yp][xp] = 1
                        if not rec[yp][xp]:
                            rec[yp][xp] = [d+1]
                        else:
                            rec[yp][xp].append(d+1)
                        q.append((yp, xp, d+1))
                        
        m = len(grid)
        n = len(grid[0])
        rec = [[None]*n for _ in range(m)]  
        building_count = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    building_count += 1
                    bfs(i, j)
        minimum = float('inf')
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    if rec[i][j] and len(rec[i][j]) == building_count:
                        minimum = min(minimum, sum(rec[i][j]))
        return minimum if minimum != float('inf') else -1

####################
#  Sort Array By Parity II
#################### 
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        '''
        break up the array into arrays of odd and even
        then just alternate taking from each list
        '''
        odds = deque([])
        evens = deque([])
        N = len(nums)
        for num in nums:
            if num % 2:
                odds.append(num)
            else:
                evens.append(num)
        
        ans = []
        for i in range(N):
            if i % 2:
                ans.append(odds.popleft())
            else:
                ans.append(evens.popleft())
        
        return ans

#onepass
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        '''
        allocate an array of size len(nums) for answers
        then have pointer to add and even indices
        place accordinly and advance accordingly
        '''
        N = len(nums)
        ans = [None]*N
        
        even = 0
        odd = 1
        for num in nums:
            #odd
            if num % 2:
                ans[odd] = num
                odd += 2
            else:
                ans[even] = num
                even += 2
        
        return ans

#constant space
class Solution:
    def sortArrayByParityII(self, nums: List[int]) -> List[int]:
        '''
        we can swap in place
        if there is an even number at odd index, we swap with odd number at even index
        since there exactly half even and half odd
        find pair that needs to be swapped
        we start by looking at even places
            when there is an error find even number at odd index
        but we need to maintain the invariant,
        everything less than i is correclty even placed, and everything less than j is correctly odd placed
        '''
        N = len(nums)
        odd = 1
        for even in range(0,N,2):
            if nums[even] % 2:
                while nums[odd] % 2:
                    odd += 2
                nums[even],nums[odd] = nums[odd], nums[even]
        
        return nums

#######################
#Split Linked List in Parts
#######################
#sheesh, well it works
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        '''
        return value is going to be an an array of arrays, where each element in the return
        array is a linked list of equal size
        each part should be N/k,excep that first N % k parts have an extra one
        earlier part in the array should be greater
        get length of linked list
        compute part sizes
        traverse the linked list using two pointers to tear apart the list accordingly
        '''
        #get size
        N = 0
        temp = head
        while temp:
            N += 1
            temp = temp.next
        
        #get parts array
        sizes = []
        for i in range(1,k+1):
            if i <= N % k:
                sizes.append((N // k) + 1)
            else:
                sizes.append(N//k)
        print(sizes)
        
        temp = head
        res = []
        #build out ans array
        for size in sizes:
            #empty size
            if size == 0:
                res.append([])
            else:
                curr_part = []
                if temp:
                    for i in range(size):
                        curr_part.append(temp.val)
                        temp = temp.next
                    res.append(curr_part)
                else:
                    continue
        print(res)
        
        #change types
        for i in range(len(res)):
            curr_part = res[i]
            if len(curr_part) == 0:
                res[i] = None
            if len(curr_part) > 0:
                curr_head = ListNode(val = curr_part[0])
                temp = curr_head
                for j in range(1,len(curr_part)):
                    temp.next = ListNode(val = curr_part[j])
                    temp = temp.next
                
                res[i] = curr_head
                
        return res
        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        '''
        let's first try to understand the hint, i.e the size of the parts in the array
        if there are N elements and we want k parts, each part should have at least N // k elements
        we want earlier parts to have a larger size than later parts
        (N % k) is reaminder of N // k
        N = k*(N//k) + (N % k)
        So if we have k segments of length N//k, the sum of segment lengths will be k * N//k which is less than N by exactly N % k.
        Anothe way: if N / k doesn't evenly divide, we have remainder in range of 1 to k-1
        in the case our reaminder is k-1, then we can add on more to all the k-1 parts, except that last k, or kth
        
        we can use a trick, 
        d, r = divmod(N, k)
        and if a part, i, in range k
        we can use (d + (i < r)-1)
        '''
        temp = head
        N = 0
        while temp:
            temp = temp.next
            N += 1
            
        width, remainder = divmod(N,k)
        res = []
        temp = head
        
        #now loop through each part
        ans = []
        cur = head
        for i in range(k):
            head = cur
            for j in range(width + (i < remainder) - 1):
                if cur: cur = cur.next
            if cur:
                cur.next, cur = None, cur.next
            ans.append(head)
        return ans

#could also make it so that way we make new lists than reconstruct
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        temp = head
        N = 0
        while temp:
            temp = temp.next
            N += 1
            
        width, remainder = divmod(N,k)
        res = []
        temp = head
        
        #now loop through each part
        ans = []
        cur = head
        for i in range(k):
            head = write = ListNode(None)
            for j in range(width + (i < remainder)):
                write.next = write = ListNode(cur.val)
                if cur: 
                    cur = cur.next
            ans.append(head.next)
        return ans

#just another way
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        '''
        without usig the trick
        '''
        #get size
        N = 0
        temp = head
        while temp:
            N += 1
            temp = temp.next
        
        part,left = N // k, N % k
        #keep two poiners one wit head and one at none
        cur = head
        prev = None
        
        res = []
        for _ in range(k):
            res.append(cur)
            for _ in range(part):
                #move current point in first entry
                if cur:
                    prev = cur
                    cur = cur.next
            #if we can afford one more part
            if left and cur:
                prev = cur
                cur = cur.next
                left -= 1
            #chop off
            if prev:
                prev.next = None
        
        return res


#######################################
# Maximum Size Subarray Sum Equals k
#######################################
#brute force
class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        '''
        i can use prefix sum array
        recall sum from i to j in nums would be pref[j+1] - pref[i]
        brute foce would be to examine all i:j subarrays
        '''
        N = len(nums)
        
        if N == 1:
            return 1 if nums[0] == k else 0
        pref = [0]
        
        ans = 0
        for num in nums:
            pref.append(pref[-1]+num)
        
        for i in range(N):
            for j in range(i+1,N):
                curr_sum = pref[j+1] - pref[i]
                if curr_sum == k:
                    if j-i > ans:
                        ans = j-i +1
        
        return ans

class Solution:
    def maxSubArrayLen(self, nums: List[int], k: int) -> int:
        '''
        recall in prefix sum, pref[i] - pref[j] is sum of subarray starting at i+1 and ending at j
        if i want sum from num[i:j]
        it would be pref[j] - pref[i-1]
        sum(nums[i:j]) = pref[j] - pref[i-1]
        this is a variation of Two Sum
        idea, store the previously seen prefix sum in hash map 
        then check if a specific value exists in the hash map as we iterate along prefix
        in this case we iterate from left to right along pref sum and if pref[i] - k has been seen, then we have found a pair
        we store previous seen pref sums in hash, with keys starting and end indices
        but we don't actually need to store the pref array
        we can use an integer variable to keep track of pref sum, and at each number store the pref sum up that number inclusive
        if we run in to duplicate, update hashmap to keep longest subarray length (marked by starting and ending indicies)
        one more thing, we need to watch out for cases where pref sum is k
        this is very similar to two sum, k - partial sum, gives complement which is still part of the subarray
        and could contribute to its length
        '''
        pref_sum = 0
        longest = 0
        
        #map of comp prefsums to indces
        mapp = {}
        for i, num in enumerate(nums):
            pref_sum += num
            if pref_sum == k:
                #remember we are going along pref sum left to right
                longest = i + 1
            #check comps
            if pref_sum - k in mapp:
                longest = max(longest,i - mapp[pref_sum-k])
            
            #put in mapp
            if pref_sum not in mapp:
                mapp[pref_sum] = i
        
        return longest

##################################
# Partition to K Equal Sum Subsets
##################################
#bleagh i tried
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        '''
        can i use freq counts instead to keep track of elements?
        well first check if sum(nums) // 4 can make even groups
        '''
        #check we can make even groups
        N = len(nums)
        SUM = sum(nums)
        if SUM % k != 0:
            return False
        
        #otherwise lets get target
        target = SUM // k
        #get freq count
        counts = Counter(nums)
        
        
        ans = False
        def rec(idx,subsets):
            #subsets ill be array of length k, idx is where we currently are
            if idx == N:
                if all([sub == target for sub in subsets]):
                    ans = True
                    return
            
            for sub in subsets:

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        '''
        lets just go throught the first three solutions, navie backtracking, optimized backtracking, optimized
        backtracking with memo
        1. Navie Backtracking
        first check fi we can partition nums into k subsets of equal sum
        this if just finding the taget sum
        if the array can be evenly divided into k parts, lets build these parts using backracking
        we will keep a currSu, var denoting sum of the current subsetset and one by one try to include it if it has not already been chosen
        we can use taken dp state array to see if we are currently taking this num
        if we reach a conidtion where currSum > targetSum, prune this path (or abandon)
        if we get to a condition, where currSum is tagetSum,  that means we made a valid subset and increment out count
        when count becomes k, we are done and return true
        also not when count becomes k-1, we are done, since the last kth part must be of the same sum, since we checked for this in the beginnign
        algo:
            1. get sum array and check if we can even do this
            2. for eac element not yet pick, 
                include in current subset and increment  currSum
                then make recusrive calls to find next element
                    if call returns true it means we are down a valid path
                    otherwsise discard the curr element and backtrack
            3. when currSum == targetSum, we have made one, so reset currSum and incrment count
            4. when count == k -1, we've amde k-1 subsets, and we know the last subset it valid
        '''
        SUM = sum(nums)
        N = len(nums)
        
        if SUM % k != 0:
            return False
        
        target_sum = SUM // k
        taken = [False]*N
        
        def backtrack(count,curr_sum):
            #k-1 subsets, we good
            if count == k - 1:
                return True
            #exceeding target
            if curr_sum > target_sum:
                return False
            
            #when we matched target, advance
            if curr_sum == target_sum:
                return backtrack(count+1,0)
            
            #pick element not taken
            for j in range(N):
                if taken[j] == False:
                    taken[j] = True
                    #take and backtrack
                    if backtrack(count,curr_sum + nums[j]):
                        return True
                    #backtrack
                    taken[j] = False
            return False
        
        return backtrack(0,0)

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        '''
        in the naive backtrackign apporach approach, when we recurse, we always started from the 0th position in nums
        even if the previous elements were already taken
        we can optimize by starting from the last index on which we made the decision to pick
        when a subset is completed, only then will we start the search from the 0th index
        as we can now include the previosuly skipped elements in enw subsets
        ALSO, we can fruther improve if we srot in decreasing order
        WHY?
        If we had sorted the array in ascending order (smaller values on the left side), 
        then there would be more recursion branches (recursive calls). 
        This is because when the change in subset-sum is small, more branches will be repeatedly created during the backtracking process.
        '''
        SUM = sum(nums)
        N = len(nums)
        
        if SUM % k != 0:
            return False
        
        target_sum = SUM // k
        taken = [False]*N
        
        nums.sort(reverse=True)
        
        def backtrack(index,count,curr_sum):
            #k-1 subsets, we good
            if count == k - 1:
                return True
            #exceeding target
            if curr_sum > target_sum:
                return False
            
            #when we matched target, advance
            if curr_sum == target_sum:
                return backtrack(0,count+1,0)
            
            #pick element not taken
            for j in range(index,N):
                if taken[j] == False:
                    taken[j] = True
                    #take and backtrack
                    if backtrack(index+1,count,curr_sum + nums[j]):
                        return True
                    #backtrack
                    taken[j] = False
            return False
        
        return backtrack(0,0,0)

#now adding memoization
class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        '''
        approach 3, we can finally try adding a memo to the recursion
        we can memoize the answer based on elements included in any of the subsets
        NOTE; sorting reduces the number of recursion branches in the main call
        we store that if we picked some number of elements, we will never be able to use them to make another k valid subset
        
        '''
        SUM = sum(nums)
        N = len(nums)
        
        if SUM % k != 0:
            return False
        
        target_sum = SUM // k
        taken = ['0']*N
        
        nums.sort(reverse=True)
        
        memo = {}
        
        def backtrack(index,count,curr_sum):
            #we need to hash taken state
            taken_str = "".join(taken)
            #k-1 subsets, we good
            if count == k - 1:
                return True
            #exceeding target
            if curr_sum > target_sum:
                return False
            
            #if we have seen this taken state
            if taken_str in memo:
                return memo[taken_str]
            
            #when we matched target, advance
            if curr_sum == target_sum:
                memo[taken_str] = backtrack(0,count+1,0)
                return memo[taken_str]
            
            #pick element not taken
            for j in range(index,N):
                if taken[j] == '0':
                    taken[j] = '1'
                    #take and backtrack
                    if backtrack(index+1,count,curr_sum + nums[j]):
                        return True
                    #backtrack
                    taken[j] = '0'
            memo[taken_str] = False
            return memo[taken_str]
        
        return backtrack(0,0,0)
