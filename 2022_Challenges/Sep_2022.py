###############################################
# 2271. Maximum White Tiles Covered by a Carpet
# 31AUG22
###############################################
#binary search coded out, we want upper mid
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        '''
        https://leetcode.com/problems/maximum-white-tiles-covered-by-a-carpet/discuss/2038177/PythonGreedy-%2B-prefix-sum-%2B-binary-search-easy-to-understand-with-explanation
        prefix sum and binary search to find the largest
        1. sort tiles by starting position
        2. build pref sum to store lenght sum of tiles
        3. traverse each tile, given its starting s, we know that the further postion the carpetn can cover is s + carepentLen-1
        4. having the furthest position, we binary search the idnex of the ending tile that the carpent can partially cover
            we greedily align the starting position of the carpet with teh starting position of the tile
        5. calculate the length of the ending tile that the carpet cannot partially cover, we need to subtract this from the last part
        '''
        #sort on starting
        tiles.sort(key = lambda x: x[0])
        #store starting position array
        startPos = [tile[0] for tile in tiles]
        #build pref sum array
        prefSum = [0]
        for s,e in tiles:
            prefSum.append(prefSum[-1] + (e - s) + 1)
            
        #helper function for binary search
        def binarySearch(arr,start,end,target):
            if start == end:
                if target == arr[start]:
                    return start
                else:
                    return end
            
            mid = start + (end - start) // 2
            if target <= arr[mid]:
                return binarySearch(arr,start,mid,target)
            else:
                return binarySearch(arr,mid+1,end,target)
            
        
        ans = 0
        N = len(tiles)
        for i in range(N):
            start,end = tiles[i]
            #if we can at least catch all tiles here, we done
            if end >= start + carpetLen - 1:
                return carpetLen
            #otherwise binary search the index of the ending tile that the carpetn can partially cover, we want the upper mid
            endIndex = binarySearch(startPos,0,len(startPos)-1,start+carpetLen-1)
            #get the length of the tiles the carpet cannot cover
            cantCover = 0
            if tiles[endIndex][1] > start + carpetLen - 1:
                cantCover = tiles[endIndex][1] - (start + carpetLen - 1)
            ans = max(ans,prefSum[endIndex+1] - prefSum[i] - cantCover)
        
        return ans
        

#using bisect right
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        # sort the tiles by the starting position
        tiles.sort(key = lambda x:x[0])
        # build the starting position array
        startPos = [tiles[i][0] for i in range(len(tiles))]
        # build the prefix sum array
        preSum = [0] * (len(tiles) + 1)
        for i in range(1, len(tiles) + 1):
            preSum[i] = preSum[i - 1] + (tiles[i-1][1]-tiles[i-1][0] + 1)
        
        res = 0
        for i in range(len(tiles)):
            s, e = tiles[i]
            # if the length of tile >= length of carpet, return carpetLen
            if e >= s + carpetLen - 1:
                return carpetLen
            # binary search the index of the ending tile that the carpet can partially cover
            endIdx = bisect_right(startPos, s + carpetLen - 1) - 1
            # calculate the length of the ending tile that the carpet cannot cover 
            compensate = 0
            if tiles[endIdx][1] > s + carpetLen - 1:
                compensate = tiles[endIdx][1] - s - carpetLen + 1
            # update the result
            res = max(res, preSum[endIdx+1] - preSum[i] - compensate)
            
        return res

#sort and sliding window with two cases
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        '''
        https://leetcode.com/problems/maximum-white-tiles-covered-by-a-carpet/discuss/2038674/Python-Explanation-with-pictures-sliding-window
        the trick is to realize (rather convince yourself the placing a carpet at the beignning of the range) will give you the maximum size
        we have two cases
        1. the right end is in the middle of some tiles
            covered = pref[j] - pref[i]
        2. the right end lies in section of tiles
            covered = pref[j+1] - prev[i] - (ends[j] - j)
        '''
        #sort on start
        tiles.sort(key = lambda x: x[0])
        #prefsum
        prefSum = [0]
        for s,e in tiles:
            prefSum.append(prefSum[-1] + (e - s) + 1)
            
        ends = [e for s,e in tiles]
        N = len(ends)
        ans = 0
        j = 0
        for i in range(N):
            #carpet start from the beginning of each range
            start,end = tiles[i]
            #the right most index have tiles is ends[-1]
            right_most = min(ends[-1],start+carpetLen-1)
            
            #while the whole current rang is covered by carpet
            while j < N and ends[j] < right_most:
                j += 1
            #two cases
            #first case, if the right end of the carept doesn't reach the jth rang
            if tiles[j][0] > right_most:
                ans = max(ans, prefSum[j] - prefSum[i])
            #the right end of the carpert covers parrt of it, there is some tiles that are left over
            else:
                ans = max(ans, prefSum[j+1] - prefSum[i] - ends[j] + right_most )
        
        return ans

############################
# 637. Average of Levels in Binary Tree (REVISITED)
# 02SEP22
############################
#without having to do second pass
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        '''
        us bfs then explore level by level
        '''
        averages = []
        q = deque([root])
        
        while q:
            N = len(q)
            level_sum = 0
            for _ in range(N):
                node = q.popleft()
                level_sum += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            
            averages.append(level_sum / N)
        
        return averages


################################
# 2187. Minimum Time to Complete Trips
# 02SEP22
################################
#TLE
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        '''
        if i'm givein the time array, example [1,2,3]
        t = 1
            [1,0,0], trips = 1
        t = 2
            [2,1,0], trips = 3
        t = 3
            [3,1,1], trips = 5
        answer = 3
        
        given a time t, i want to compute number of trips in the whole array
        then use binary search to find the lower bound where num trips for i just less than totalTrips
        O(len(times)) to compute trips for time t
        then log(maxtime) for getting the minimu time
        i'll maximize using 2**32 first
        '''
        def getNumTrips(t):
            trips = 0
            for i in range(len(time)):
                trips += (t // time[i])
            
            return trips
        
        start = 0
        
        while getNumTrips(start) < totalTrips:
            start += 1
        
        return start
    
#YAYYYY, you just needed to watch the upper bound
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        '''
        if i'm givein the time array, example [1,2,3]
        t = 1
            [1,0,0], trips = 1
        t = 2
            [2,1,0], trips = 3
        t = 3
            [3,1,1], trips = 5
        answer = 3
        
        given a time t, i want to compute number of trips in the whole array
        then use binary search to find the lower bound where num trips for i just less than totalTrips
        O(len(times)) to compute trips for time t
        then log(maxtime) for getting the minimu time
        i'll maximize using 2**32 first

        upper bound is min(time)*totalTrips
        in the worst caes we only use the fasts bus
        '''
        def getNumTrips(t):
            trips = 0
            for i in range(len(time)):
                trips += (t // time[i])
            
            return trips
        
        
        start = 0
        end = min(time)*totalTrips
        
        while start < end:
            mid = start + (end - start) // 2
            #guess
            guess = getNumTrips(mid)
            if guess >= totalTrips:
                #dont need to look beyond end anymore
                end = mid
            else:
                start = mid+1
        
        return start

#one liner solution just to show off
def minimumTime(self, time: List[int], totalTrips: int) -> int:
    return bisect_left(range(1, 10**14), totalTrips, key= lambda x: sum(x // t for t in time)) + 1
####################################
# 1207. Unique Number of Occurrences
# 02SEP22
####################################
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        '''
        get count mapp the second pass with hashset
        i could have also sorted the counts and check for repeated values
        '''
        counts = Counter(arr)
        seen = set()
        
        for c in counts.values():
            if c in seen:
                return False
            seen.add(c)

        return True
        
##############################
# 1929. Concatenation of Array
# 02SEP22
##############################
class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        '''
        preallocate the ans array and point two pointers
        one at i and the other at i+1
        '''
        N = len(nums)
        ans = [0]*(2*N)
        for i in range(N):
            ans[i] = nums[i]
            ans[N+i] = nums[i]
        
        return ans

############################################################
# 967. Numbers With Same Consecutive Differences (REVISITED)
# 03SEP22
############################################################
#backtracking, time complexity if O(2**n)
#from a node in the executino tree, we have at most 2 children, 
#num + k, and num -k
#binary stree with depth N-1, and 2 children nodes          
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        '''
        note the conditions, n is in the closed range [2,9]
        i can try building a digit
        for (n = 2, k = 0)
            [11,22,33,44,55,66,77,88,99]
        for (n=2, k = 1)
            [10,12,21,23,32,34,43,45,54,56,65,67,76,78,87,89,98]
            
        i need to try building a digit using backtracking, contraints are small enough to allow for this
        i can start the function offf using each number [0,9] then rebuild
        just try building the string for now, worry about optimizations later
        '''
        paths = []
        n -= 1
        
        def backtrack(n,path):
            if n == 0:
                paths.append(int("".join(path)))
                return
            last_digit = int(path[-1])
            for next_digit in range(0,10):
                if abs(last_digit - next_digit) == k:
                    path += str(next_digit)
                    backtrack(n-1,path)
                    path.pop()
        
        for i in range(1,10):
            backtrack(n,[str(i)])
        return paths

class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        '''
        dfs without backtracking, being explicity when to call
        '''
        if n == 1:
            return [i for i in range(10)]
        
        ans = []
        
        def dfs(n,num):
            if n == 0:
                ans.append(num)
                return
            
            last_digit = num % 10
            next_digits = set([last_digit+k,last_digit-k])
            
            for foo in next_digits:
                if 0 <= foo < 10:
                    new_num = num*10 + foo
                    dfs(n-1,new_num)
                    
        
        for num in range(1,10):
            dfs(n-1,num)
        
        return ans
            
#bfs
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        '''
        we can also use bfs and just generate the numbers layer by layer and returnt he final layer
        '''
        if n == 1:
            return [i for i in range(10)]
        
        q = [num for num in range(1,10)]
        
        #we have the first level done, so there are at least N-1 levels left
        for level in range(n-1):
            next_level = []
            for num in q:
                last_digit = num % 10
                next_digits = set([last_digit+k,last_digit-k])
            
                for foo in next_digits:
                    if 0 <= foo < 10:
                        new_num = num*10 + foo
                        next_level.append(new_num)
            
            q = next_level
        
        return q

###########################
# 124. Binary Tree Maximum Path Sum
# 04SEP22
############################
#closeeeeee
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        '''
        at every node, find the max sum path for its left side and the max sum path for its right side
        then we can just maximize globally at each function call
        max(left,right,left,right+1)
        but pass this up
        '''
        self.ans = float('-inf')
        
        def dp(node):
            if not node:
                return 0
            left = dp(node.left)
            right = dp(node.right)
            #update
            option1 = left + node.val
            option2 = right + node.val
            option3 = left + right + node.val
            option4 = node.val
            self.ans = max(self.ans, option1,option2,option3,option4)
            return max(option1,option2,option3,option4)
        
        
        dp(root)
        return self.ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        '''
        almost had it, we just need to either take the max(left,0) and max(right,0)
        then calculate adding a new path
        '''
        
        self.ans = float('-inf')
        
        def dp(node):
            if not node:
                return 0
            
            #get max of left and right
            left = max(dp(node.left),0)
            right = max(dp(node.right),0)
            
            #making a new path rooted at this node
            new_path = node.val + left + right
            
            #maximize
            self.ans = max(self.ans,new_path)
            
            #return the max
            return node.val + max(left,right)
        
        
        dp(root)
        return self.ans

#good write up
# https://leetcode.com/problems/binary-tree-maximum-path-sum/discuss/603423/Python-Recursion-stack-thinking-process-diagram
#for a tree with all negative numbers, the best path is in face the largest number in the tree, adding negatives only makes it smaller


#############################
# 678. Valid Parenthesis String 
# 04SEP22
#############################
#close one again....
class Solution:
    def checkValidString(self, s: str) -> bool:
        '''
        i can use the balacne method +1 for ( and -1 for )
        keep count of stars too, so we can use them to balance the string
        '''
        balance = 0
        count_stars = 0
        
        for ch in s:
            if ch == '(':
                balance += 1
            elif ch == ')':
                balance -= 1
            else:
                count_stars += 1
            
            if balance < 0:
                if count_stars > 0:
                    balance += 1
                    count_stars -= 1
                else:
                    return False
            
        return balance == 0

#dp
class Solution:
    def checkValidString(self, s: str) -> bool:
        '''
        dynamic programming
        if we let dp(i,j) be true only of s[i:j] is a valid expression
        then dp(i,j) is true if and only iff:
            s[i] == '*' and the interval s[i+1:j] is valid
            or s[i] can be made to be '(' and there is some k in [i+1,j] such that s[k] can be made to be ')' 
            plus the two itnervals cut by s[k] (s[i+1:k],s[k+1:j+1])
        
        
        '''
        if not s:
            return True
        
        lefty = '(*'
        righty = ')*'
        
        N = len(s)
        dp = [[False]*N for _ in range(N)]
        
        for i in range(N):
            #empty string is trivally True
            if s[i] == '*':
                dp[i][i] = True
            #can index into N and one after
            if i < N-1 and s[i] in lefty and s[i+1] in righty:
                dp[i][i+1] = True
        
        for size in range(2,N):
            #check all substrings in s for all sizes between 2 and N
            for i in range(N-size):
                #can we build up a valid parantheses expression
                if s[i] == '*' and dp[i+1][i+size] == True:
                    dp[i][i+size] = True
                elif s[i] in lefty:
                    #check in between i and j
                    for k in range(i+1,i+size+1):
                        if s[k] in righty and (k == i+1 or dp[i+1][k-1] == True) and (k == i+size or dp[k+1][i+size] == True):
                            dp[i][i+size] = True
        
        return dp[0][N-1]

#greedy
class Solution:
    def checkValidString(self, s: str) -> bool:
        '''
        we can still follow the balacing principle from previous valid parenthese problems
        but rather we need to keep track of minimum possible balance and the maximum possible balance
        for example
        for '(', the possible values for balance is [1]
        for '(*', could be [0,1,2]
        for '(***', could be [0,1,2,3,4]
        for '(***)' could be [0,1,2,3]
        
        rather, it can be proven that each of these states can form a contiguous interval
        
        whenever we see an ( incremeant both lo and hi
        whenever we see a ) decrement both
        when it's a start, we greedily increment the max, and decrement the min
        check if the max balance is 0, which means the string could not have been balanced in the first place
        then at the end, check that min == 0
        
    https://leetcode.com/problems/valid-parenthesis-string/discuss/543521/Java-Count-Open-Parenthesis-O(n)-time-O(1)-space-Picture-Explain        
        Case - 1:
if (cmax < 0) return false
and
Case - 2:
cmin = Math.max(cmin, 0)

This is what I could summarize this as:

Case - 1:
If cmax < 0, the number of ')' is lesser than 0. We immediately return false.
Why : Let's take an example "())", in this case, cmax would be less than 0 because we have two ' )' and only one '('. Now irrespective of how many '*' we have, this sequence is already invalid, hence we return false.

Case - 2:
cmin = Math.max(cmin, 0)

The way I got to wrap my head around this was:
Cmin and Cmax are both subtracted by 1, whenever we encounter a ")". Therefore, Case -1 covers the case in which we have more ")" than "(". Now the additional case we have to look at is, when we have extra ")", which we can account to the "*" [Since we do --cmin here].

However, we can just ignore the "*" as empty strings in this case.
Example: "( ) * * "
cmax = 1 0 1 2
cmin = 1 0 0 0 -> We don't want the last two to become 1 0 -1 -2

We can see that the cmin values would become -1 and -2 for the last two "". However this would mean we would be adding additional ")", which makes the sequence "()))". This is not a right sequence. Therefore, we must keep them as empty strings. Hence we do a max with 0, which implies that if we have additional "", we don't take them as ")", instead we treat them as empty strings.
        '''
        min_balance = 0
        max_balance = 0
        
        for ch in s:
            if ch == '(':
                min_balance += 1
                max_balance += 1
            elif ch == ')':
                min_balance -= 1
                max_balance -= 1
            elif ch == '*':
                max_balance += 1
                min_balance -= 1
            
            if max_balance < 0:
                return False
            min_balance = max(min_balance,0)
        
        return min_balance == 0

#official solution
class Solution(object):
    def checkValidString(self, s):
        lo = hi = 0
        for c in s:
            lo += 1 if c == '(' else -1
            hi += 1 if c != ')' else -1
            if hi < 0: break
            lo = max(lo, 0)

        return lo == 0

################################
# 1110. Delete Nodes And Return Forest
# 06SEP22
################################
#close one
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        '''
        similar to 814. Binary Tree Pruning
        use recursino to check if node and any descendant node is in to delete
        out to global list
        '''
        to_delete = set(to_delete)
        self.ans = []
        
        def needs_deletion(node):
            if not node:
                return False
            left = needs_deletion(node.left)
            right = needs_deletion(node.right)
            if left:
                node.left = None
            if right:
                node.right = None
            #add to list 
            self.ans.append(node)
            return left or right or node.val in to_delete
        
        needs_deletion(root)
        return self.ans


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        '''
        https://leetcode.com/problems/delete-nodes-and-return-forest/discuss/328854/Python-Recursion-with-explanation-Question-seen-in-a-2016-interview
        to remove a node, the child needs to notify the parent about its' exsitence
        to determine whether a node is a root node in the final forest, we need to know
            1. wheter the node is removed (pass down whether we delted)
            2. whteher its parent has been removed
            
        recursion intution:
            pass down deeper into the recurison by passinga arguments
            return up from below passing in return values
        '''
        to_delete = set(to_delete)
        ans = []
        
        def rec(node,parent_exist):
            #empty node
            if not node:
                return None
            #if we need to delte this node
            if node.val in to_delete:
                node.left = rec(node.left,False)
                node.right = rec(node.right,False)
                return None
            else:
                #if this node's parent didnt' exist, then it must have been deleted and this node is part of the forest
                if not parent_exist:
                    ans.append(node)
                node.left = rec(node.left,True)
                node.right = rec(node.right,True)
                return node
        
        rec(root,False)
        return ans

#another solution
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        '''
        another solution
        function that returns the appropiate left and right subtrees after deleting
        '''
        to_delete = set(to_delete)
        ans = []
        
        def rec(node,parent_exist):
            if not node:
                return None
            
            #build the left and right sides
            node.left = rec(node.left,node.val not in to_delete)
            node.right = rec(node.right,node.val not in to_delete)
            
            #check if we can add to forest
            #it is in not in delete, but parent is gone
            if node.val not in to_delete:
                if parent_exist == False:
                    ans.append(node)
                return node
            else:
                return None
            
            
        rec(root,False)
        return ans

#best one so far
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root: Optional[TreeNode], to_delete: List[int]) -> List[TreeNode]:
        '''
        another solution
        function that returns the appropiate left and right subtrees after deleting
        '''
        to_delete = set(to_delete)
        ans = []
        
        def delete(node):
            if not node:
                return None
            node.left = delete(node.left)
            node.right = delete(node.right)
            #if we need to delte
            if node.val in to_delete:
                #if there is a left, we want it to be a part of the first
                if node.left:
                    ans.append(node.left)
                if node.right:
                    ans.append(node.right)
                #we need to delete this node that we are one
                return None
            return node
        
        delete(root)
        #check root
        if root.val not in to_delete:
            ans.append(root)
        return ans
            
#iterative solutions using bfs and q
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def delNodes(self, root: TreeNode, to_delete: List[int]) -> List[TreeNode]:
        queue = collections.deque([(root, False)])
        res = []
        deletes = set(to_delete)
        
        while queue:
            node, hasParent = queue.popleft()
            if not hasParent and node.val not in deletes:
                res.append(node)
                
            hasParent = not node.val in deletes
            
            if node.left:
                queue.append((node.left, hasParent))
                if node.left.val in deletes:
                    node.left = None
            
            if node.right:
                queue.append((node.right, hasParent))
                if node.right.val in deletes:
                    node.right = None
        
        return res

###########################
# 94. Binary Tree Inorder Traversal (REVSITED)
# 08SPE22
###########################
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        '''
        inorder is left,node,right
        
        '''
        def dfs(node):
            if not node:
                return []
            
            left = dfs(node.left)
            left.append(node.val)
            right = dfs(node.right)
            for num in right:
                left.append(num)
            
            return left
        
        return dfs(root)

#################################################
# 1996. The Number of Weak Characters in the Game
# 08SEP22
#################################################
class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        '''
        we are given a prporties array, of length N 
        where propeties[i] represents the i'th character in the game
        we define a weak character if any other character has both attack and defences striclty greater than this i'th's attack and defense
        
        brute force is to just compare all i and j, which is super slow
        obvie sorting comes to mind, but how to sort it
        
        looked at the hints, sort by attack then group
        then check the next larget attack group for a character with a larger defense?
        
        think of the easier problem, if we were to only have the attack values
        the number of weak characters would be:
            number of characters - count(characters with highest attack), [1,2,3,4,5], there are 4 weak characters
        
        key:
            Now once we have the array sorted in ascending order of their attack value, we can iterate over the pairs from right to left keeping the maximum defense value achieved so far. If this maximum defense value is more than the defense value at the current index then it's a weak character.
            
        sort on acensding attack, and descending defense
        
        algo:
            1. sort on ascending attack and desending defense
            2. init max defense to 0
            3. iterate from right to left
                update max defense
                update weak charcters if defense is less than the current max defense
        
        '''
        properties.sort(key = lambda x: (x[0],-x[1]))
        max_defense = 0
        weak_chars = 0
        
        for attack,defense in properties[::-1]:
            if defense < max_defense:
                weak_chars += 1
            max_defense = max(max_defense,defense)
        
        return weak_chars

class Solution:
    def numberOfWeakCharacters(self, properties: List[List[int]]) -> int:
        '''
        intuition:
            for a pair (a,b) we can say it to be weak if the maximum defense value among all the pais with attack_value >  a is strictly greater than b
        
        so we can keep maximum defense value amon all the pairs with an attack value greater than x, for every value of x
        then the pair (a,b) will be weak of the maximum defense stored at (a+1) > b
        
        to find the maximum defense value, we first the max defense for a partifular attack, update max attack's max defense
        
        algo:
            1. Iterate over properties, and store the maximum defense value for attack values in the array maxDefense.
            2. Iterate over all the possible values of attack from the maximum possible attack value (100000) to 0. Keep the maximum value seen so far, maxDefense[i]                      will represent the maximum value in the suffix [i, maxAttack].
            3. Iterate over the properties for every pair (attack, defense), increment the counter weakCharacters if the value at maxDefense[attack + 1] is greater than defense.
            4. Return weakCharacters.
        '''
        max_attack = max([a for a,b in properties])
        
        #store max defnese for all attacks
        max_def_for_attack = [0]*(max_attack+2)
        for attack,defense in properties:
            max_def_for_attack[attack] = max(max_def_for_attack[attack],defense)
        
        #store maximum defense for attack greater >= to attack
        for i in range(max_attack-1,-1,-1):
            max_def_for_attack[i] = max(max_def_for_attack[i],max_def_for_attack[i+1])
        
        #verify and count weak chars
        weak_chars = 0
        for attack,defense in properties:
            if defense < max_def_for_attack[attack+1]:
                weak_chars += 1
        
        return weak_chars

#####################################################
# 188. Best Time to Buy and Sell Stock IV (REVISITED)
# 10SEP22
#####################################################
#take recursive solution from Stock III and iterate for all possible k
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        '''
        this is 0/1 knapsack problem
        we either take or we don't take
        let dp(i,j) be the max profit we can get after j transactions and after i days
        
        dp(i,j) = {
        max(dp(i-1,k) for k in range(num transactions))
        }
        '''
        memo = {}
        N = len(prices)
        
        def rec(day,own,k):
            if day >= N or k == 0:
                return 0
            if (day,own,k) in memo:
                return memo[(day,own,k)]
            #if i had own stack, i can sell or stay, and if sell use up transation
            #take means i do an action
            if own:
                #i can sell and go up in prices
                take = prices[day] + rec(day+1,0,k-1)
                #or just hold and go on to the next day
                no_take = rec(day+1,1,k)
            else:
                #i dont own, so i have to buy, but keep k
                take = -prices[day] + rec(day+1,1,k)
                #stay but keep k
                no_take = rec(day+1,0,k)
            
            memo[(day,own,k)] = max(take,no_take)
            return memo[(day,own,k)]
        
        ans = float('-inf')
        for t in range(k+1):
            ans = max(ans,rec(0,0,t))
        
        return ans

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        '''
        another recursive way
        
        '''
        memo = {}
        N = len(prices)
        def rec(day,transactionsLeft):
            if day >= N or transactionsLeft == 0:
                return 0
            if (day,transactionsLeft) in memo:
                return memo[(day,transactionsLeft)]

            #we can always choose to stay
            ans1 = rec(day+1,transactionsLeft)
            ans2 = 0
            #if we can buy
            buy = (transactionsLeft % 2 == 0)
            if buy:
                ans2 = -prices[day] + rec(day+1, transactionsLeft -1)
            else:
                ans2 = prices[day] + rec(day+1, transactionsLeft -1)

            memo[(day,transactionsLeft)] = max(ans1,ans2)
            return memo[(day,transactionsLeft)] 
        
        ans = 0
        for t in range(k+1):
            ans = max(ans,rec(0,t*2))
        
        return ans

#bottom up
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        
        def dp(k):
            N = len(prices)
            dp = [[0]*(2*k+1) for _ in range(N+1)]

            for day in range(N, -1,-1):
                for trans in range(2*k+1):
                    if day >= N or trans == 0:
                        dp[day][trans] = 0
                    else:
                        ans1 = dp[day+1][trans]
                        ans2 = 0
                        buy = (trans % 2 == 0)
                        if buy:
                            ans2 = -prices[day] + dp[day+1][trans -1]
                        else:
                            ans2 = prices[day] + dp[day+1][trans-1]

                        dp[day][trans] = max(ans1,ans2)

            return dp[0][-1]
        
        ans = 0
        for t in range(k+1):
            ans = max(ans,dp(t))
        
        return ans

#these solutions are reall O(n*k*k), because the original solution from StockIII was already O(nk)
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        '''
        well firs think of the brute force, a single transaction consists of a buy and sell event
        if we have k transactions, we need to pick 2*k points on the stock line, can pick at most
        we would need to find all combinations of 2*k points from length(proces) 
        N choose 2*k{
        n! / (2k)!(n-2k)!
        }
        special case is when 2k > N, just tak all increasing ones
        if we has 1 transaction, we just keep carying the previous max and try to maximize by adding the current profix
        dp(i) = max(dp(i-1) + profit_so_far + prices[i])
        
        we need to store three things: the day number, the number of used transactions, and the stock holding status
        exaples, dp(i,j,1) represents the max profits up to i, j remaining transactions, and if we are currently holding
        
        we start with dp[0][0][0] = 0 and dp[0][0][1] = -prices[i]
        
        1. keep holding stock:
            dp[i][j][1] = dp[i-1][j][1]
        2. keep not holding stock
            dp[i][j][0] = dp[i-1][j][0]
        3. buying when j > 0, we have transactions left
            dp[i][j][1] = dp[i-1][j-1][0] - prices[i]
        4. selling
            dp[i][j][0] = dp[i-1][j][1] + prices[i]
            
        we can consolidate these:
            dp[i][j][1]=max(dp[i−1][j][1],dp[i−1][j−1][0]−prices[i])

            dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1]+prices[i])dp[i][j][0]=max(dp[i−1][j][0],dp[i−1][j][1]+prices[i])
            
        the take the max at dp[n-1][j][0] for j in range(k+1)
        '''
        n = len(prices)

        # solve special cases
        if not prices or k==0:
            return 0

        if 2*k > n:
            res = 0
            for i, j in zip(prices[1:], prices[:-1]):
                res += max(0, i - j)
            return res

        # dp[i][used_k][ishold] = balance
        # ishold: 0 nothold, 1 hold
        dp = [[[-math.inf]*2 for _ in range(k+1)] for _ in range(n)]

        # set starting value
        dp[0][0][0] = 0
        dp[0][1][1] = -prices[0]

        # fill the array
        for i in range(1, n):
            for j in range(k+1):
                # transition equation
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1]+prices[i])
                # you can't hold stock without any transaction
                if j > 0:
                    dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i])

        #get max in last entry
        res = max(dp[n-1][j][0] for j in range(k+1))
        return res

################################
# 447. Number of Boomerangs
# 11SEP22
################################
#https://leetcode.com/problems/number-of-boomerangs/discuss/92868/Short-Python-O(n2)-hashmap-solution
#i had the right idea!
#its just a hashamp and combinations problem
class Solution:
    def numberOfBoomerangs(self, points: List[List[int]]) -> int:
        '''
        we are given a list of points in the 2d plane
        we define a bommerang as tuple (i,j,k) where dist(i,j) == dist(i,k)
        brute force would be to examine all possible tuples(i,j,k) and check distances
        500^3, too big, there must be an O(N^2) solution
        
        what if i mapped distances to points?
        
        for each point in point, make a count map for all distances
        then tracrse the count map and count the combinations
        '''
        boomerangs = 0
        for p in points:
            mapp = Counter()
            for q in points:
                x_comp = p[0] - q[0]
                y_comp = p[1] - q[1]
                dist = x_comp*x_comp + y_comp*y_comp
                mapp[dist] += 1
            #count up paths
            for d in mapp:
                #if we have k points that are distance d away from each other
                #then we need two of these points to make a boomerang
                #so the number of boomerans is counts[d]*(counts[d]-1) 
                #rather k*k-1
                #think permutations, since order matters to give unique set
                boomerangs += mapp[d]*(mapp[d] - 1)
        
        return boomerangs

################################
# 948. Bag of Tokens (REVISTED)
# 11SEP22
################################
#nice try with dp though :(
class Solution:
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        '''
        if current power >= tokens[i]:
            score += 1
            power -= tokens[i]
        if current power >= tokens[i]:
            score -= 1
            power += tokens[i]
        
        we want the largest score possible
        
        '''
        #try dp
        #dp(i,power) repersent the max scroe i get using up tokens[i:] and having power
        memo = {}
        
        def dp(i,power):
            if i < 0:
                return 0
            if (i,power) in memo:
                return memo[(i,power)]
            #play
            play = 1 + dp(i-1,power - tokens[i])
            no_play = dp(i-1,power+tokens[i]) - 1
            ans = max(play,no_play)
            memo[(i,power)] = ans
            return ans
        
        return dp(len(tokens))

class Solution:
    def bagOfTokensScore(self, tokens: List[int], power: int) -> int:
        '''
        if current power >= tokens[i]:
            score += 1
            power -= tokens[i]
        if current power >= tokens[i]:
            score -= 1
            power += tokens[i]
        
        we want the largest score possible
        two pointers, take from hi and low when we can
        '''
        tokens.sort()
        tokens = deque(tokens)
        
        max_score = 0
        curr_score = 0
        
        while tokens and (power >= tokens[0] or curr_score): #the second part just means while we have enough power or we still have a score
            while tokens and power >= tokens[0]:
                power -= tokens.popleft()
                curr_score += 1
            
            #update
            max_score = max(max_score,curr_score)
            
            if tokens and curr_score:
                power += tokens.pop()
                curr_score -= 1
        
        return max_score
            

##############################
# 894. All Possible Full Binary Trees
# 13SEP22
##############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        '''
        are these the catalan numbers? similar to it
        shoot, i actually need to generate the trees
        all powers of must return and empty list
       
        base case when n == 1
            return TreeNode(val = 0)
       
        if we had a recursive function dp, where dp(N) that return the the list of all possible binary trees
        and every full binary tree with 3 or more nodes has 2 children, call them left and right
        then dp(N) = [all trees with child from dp(left) and all trees with dp(right)]
        '''
        memo = {}
       
        def dp(n):
            if n == 0:
                return []
            if n == 1:
                return [TreeNode(0)]
            if n in memo:
                return memo[n]
            ans = []
            for x in range(n):
                y = n-x-1
                for left in dp(x):
                    for right in dp(y):
                        #made new node
                        curr = TreeNode(0)
                        curr.left = left
                        curr.right = right
                        ans.append(curr)
           
            memo[n] = ans
            return ans
       
        return dp(n)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def allPossibleFBT(self, N: int) -> List[Optional[TreeNode]]:
        '''
        bottom up
        https://leetcode.com/problems/all-possible-full-binary-trees/discuss/618779/Python-3-solution-(Recursion-Memoization-DP)-with-explantions
        '''
        if N % 2 == 0: return []
        dp = [[] for _ in range(N + 1)]
        dp[1].append(TreeNode(0))
        for n in range(3, N + 1, 2):
            for i in range(1, n, 2):
                j = n - 1 - i
                for left in dp[i]:
                    for right in dp[j]:
                        root = TreeNode(0)
                        root.left = left
                        root.right = right
                        dp[n].append(root)
        return dp[N]

#############################
# 393. UTF-8 Validation (REVISITED)
# 13SEP22
#############################
#using bit shifts instead of string manipulationg
class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        '''
        lets take a look at what parts of a byte cirrespond to an integer that we need to process
        1. if the starting byte for a UTF8 char, then we need to process the first N bits, where N is uppper bounded by 4
        anything more than that and we would have an invalid character
        2. in the case the byte is part of a utf8 character, then we simply need to check for the first two bits or the most significant bits
        the most significant bit needs to be a 1 and the second ost needs to be a zero
        
        mask = 1 << 7
        while mask & num:
            n_bytes += 1
            mask = mask >> 1
            
        to count the set bits in a UTF8 char
        
        mask1 = 1 << 7
        mask2 = 1 << 6

        if not (num & mask1 and not (num & mask2)):
            return False
            
        to check if the most sig bit is 1 and secnod most sig bit is 0
        '''
        # Number of bytes in the current UTF-8 character
        n_bytes = 0

        # Mask to check if the most significant bit (8th bit from the left) is set or not
        mask1 = 1 << 7

        # Mask to check if the second most significant bit is set or not
        mask2 = 1 << 6
        for num in data:

            # Get the number of set most significant bits in the byte if
            # this is the starting byte of an UTF-8 character.
            mask = 1 << 7
            if n_bytes == 0:
                while mask & num:
                    n_bytes += 1
                    mask = mask >> 1

                # 1 byte characters
                if n_bytes == 0:
                    continue

                # Invalid scenarios according to the rules of the problem.
                if n_bytes == 1 or n_bytes > 4:
                    return False
            else:

                # If this byte is a part of an existing UTF-8 character, then we
                # simply have to look at the two most significant bits and we make
                # use of the masks we defined before.
                if not (num & mask1 and not (num & mask2)):
                    return False
            n_bytes -= 1
        return n_bytes == 0     

##############################################################
# 1457. Pseudo-Palindromic Paths in a Binary Tree (Revisited)
# 14SEP22
##############################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        '''
        the hint gives the solution away
        if any permuation in a path forms a palindrome, then only one digit ni the path occurs an odd number of times
        use dfs and pass along a counter object by reference, then update this count object
        one we get to a leaf check the counter objects for only one digit have an odd occurrence
        
        must do at the end
        '''
        self.ans = 0
        
        def dfs(node,counts):
            if not node:
                return
            #leaf node
            if not node.left and not node.right:
                counts[node.val] += 1
                odd = 0
                for k,v in counts.items():
                    if v % 2 == 1:
                        odd += 1
                if odd <= 1:
                    self.ans += odd ==  1
                return
            dfs(node.left,counts)
            counts[node.val] += 1
            dfs(node.right,counts)
        
        
        dfs(root,Counter())
        return self.ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        '''
        the hint gives the solution away
        if any permuation in a path forms a palindrome, then only one digit ni the path occurs an odd number of times
        use dfs and pass along a counter object by reference, then update this count object
        one we get to a leaf check the counter objects for only one digit have an odd occurrence
        
        must do at the end
        '''
        self.ans = 0
        
        def dfs(node,counts):
            if not node:
                return
            #generate new count object
            new_counts = counts[:]
            new_counts[node.val] ^= 1
            if not node.left and not node.right and sum(new_counts) <= 1:
                self.ans += 1
            dfs(node.left,new_counts)
            dfs(node.right,new_counts)
        
        dfs(root,[0]*10)
        return self.ans

#real way is with bitshifting
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        '''
        the hint gives the solution away
        if any permuation in a path forms a palindrome, then only one digit ni the path occurs an odd number of times
        use dfs and pass along a counter object by reference, then update this count object
        one we get to a leaf check the counter objects for only one digit have an odd occurrence
        
        must do at the end
        '''
        self.ans = 0
        
        def dfs(node,counts):
            if not node:
                return
            counts = counts ^ (1 << node.val)
            if not node.left and not node.right:
                if counts & (counts -1) == 0:
                    self.ans += 1
            
            dfs(node.left,counts)
            dfs(node.right,counts)
        
        dfs(root,0)
        return self.ans
            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pseudoPalindromicPaths (self, root: Optional[TreeNode]) -> int:
        '''
        the hint gives the solution away
        if any permuation in a path forms a palindrome, then only one digit ni the path occurs an odd number of times
        use dfs and pass along a counter object by reference, then update this count object
        one we get to a leaf check the counter objects for only one digit have an odd occurrence
        
        must do at the end
        '''
        count = 0
        
        stack = [(root, 0) ]
        while stack:
            node, path = stack.pop()
            if not node:
                continue
            # compute occurences of each digit 
            # in the corresponding register
            path = path ^ (1 << node.val)
            # if it's a leaf, check if the path is pseudo-palindromic
            if node.left is None and node.right is None:
                # check if at most one digit has an odd frequency
                if path & (path - 1) == 0:
                    count += 1
            stack.append((node.right, path))
            stack.append((node.left, path))
        
        return count

###########################
# 2007. Find Original Array From Doubled Array
# 15SEP22
###########################
#close one
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        '''
        changed array comes from original array
        where every element in original array is doubled and appened, and then randomly shuffling the array
        its ok to return any permutation of the oriignal array
        
        i can traverse changed and its double to a seen set, maybe first sort?
        changed will always be an even number
        
        what if i initally counted the elements in the changed array
        then retravesre and remove the doubled ounts
        then grab only elements who's counts are 1
        '''
        N = len(changed)
        #odd length
        if N % 2 == 1:
            return []
        
        counts = Counter(changed)
        
        #re traverse and removed double, 
        #keep variable of original size to mainint deletion
        orig_size = N // 2
        for num in changed:
            if num*2 in counts and orig_size > 0:
                orig_size -= 1
                if counts[num*2] > 0:
                    counts[num*2] -= 1
                if counts[num*2] == 0:
                    del counts[num*2]
                
        
        ans = [k for k,v in counts.items()]
        #i could do aonther check agian??
        counts = Counter(changed)
        for num in ans:
            if num in counts and counts[num] > 0:
                counts[num] -= 1
                if counts[num] == 0:
                    del counts[num]
            #for num*2
            if num in counts and counts[num] > 0:
                counts[num] -= 1
                if counts[num] == 0:
                    del counts[num]
        
#counting and using hashmap
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        '''
        changed array comes from original array
        where every element in original array is doubled and appened, and then randomly shuffling the array
        its ok to return any permutation of the oriignal array
        
        i can traverse changed and its double to a seen set, maybe first sort?
        changed will always be an even number
        
        what if i initally counted the elements in the changed array
        then retravesre and remove the doubled ounts
        then grab only elements who's counts are 

        we need to sort and start with the smallest element
        '''
        N = len(changed)
        #odd length
        if N % 2 == 1:
            return []
        
        counts = Counter(changed)
        changed.sort()
        
        #re traverse and removed double, 
        original = []

        
        for num in changed:
            #remove ocurrences
            if num in counts and counts[num] > 0:
                counts[num] -= 1
                twiceNum = num*2
                if  twiceNum in counts and counts[twiceNum] > 0:
                    #pair up elements and lower count
                    counts[twiceNum] -= 1
                    #add this number
                    original.append(num)
                else:
                    return []
        
        return original

#counting sort
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        '''
        we can use counting sort
        the only diffetence is that for every evelemnt we wil be iterating once, howver there migh be multiple instances of it in the oginral array
        so we iterate over an element and we decrement the counter to reiterate it
        '''
        N = len(changed)
        #odd length
        if N % 2 == 1:
            return []

        max_number = max(changed)
        counts = [0]*(2*max_number+1)
        
        for num in changed:
            counts[num] += 1
        
        original = []
        #need to use while loop
        num = 0
        while num <= max_number:
            #first pairing
            if counts[num] > 0:
                counts[num] -= 1
                twiceNum = num*2
                #second pairing
                if counts[twiceNum] > 0:
                    counts[twiceNum] -= 1
                    original.append(num)
                    num -= 1
                else:
                    return []
            
            num += 1
        
        return original

#using heap
class Solution:
    def findOriginalArray(self, changed: List[int]) -> List[int]:
        '''
        using a a min heap
        sort changed and iterate over each element
        then check if this num and the number at the top of the heap is 2times that number
        '''
        changed.sort()
        heap, ans = [], []
        for x in changed:
            if heap and x==heap[0]*2:
                ans.append(heapq.heappop(heap))
            else:
                heapq.heappush(heap, x) 
        return ans if not heap else []

################################
# 159. Longest Substring with At Most Two Distinct Characters (REVISITED)
# 15SEP22
################################
#close one
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        '''
        sliding window with seen hash set problem
        
        '''
        ans = 0
        N = len(s)
        left = right = 0
        
        seen = set()
        while right < N:
            #expand window
            while len(seen) <= 2 and right < N:
                seen.add(s[right])
                right += 1
            ans = max(ans,right -left - 1)
            seen.discard(s[left])
            left += 1
        
        return ans
            
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        '''
        sliding window with seen hash set problem
        
        '''
        n = len(s)
        if n < 3:
            return n
        
        left = right = 0
        ans = 0
        
        mapp = defaultdict()
        
        while right < n:
            #add to window
            mapp[s[right]] = right
            right += 1
            
            #if we go over, 
            if len(mapp) == 3:
                #delete the character we haven't seen in a while
                last_recent = min(mapp.values())
                #delete
                del mapp[s[last_recent]]
                #move
                left = last_recent + 1
            
            ans = max(ans, right - left)
        
        return ans

################################
# 1770. Maximum Score from Performing Multiplication Operations
# 16SEP22
################################
#close one!!!!
#but we need to reduce to three states
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        '''
        we are only allowed m operations, where m is len(multipliers)
        lets say i knew the max answer already after taking some elements off the from the left and some off from the right
        call this dp(state), to get the max answer i need
        dp(state) + max(nums[left]*operation,nums[right]*operation)
        time complexity for these problems can be dervied by using master theorm and sub problme reduction
        Thus, the recurrence relation is T(M)=2T(M-1)+O(1)T(M)=2T(M−1)+O(1), which can be solved using Master Theorem and the result is O(2^M)O(2 
M
 ).


        '''
        memo = {}
        n = len(nums)
        m = len(multipliers)
        
        def dp(left,right,operations):
            if operations == m:
                return 0
            if (left,right,operations) in memo:
                return memo[(left,right,operations)]
            take_left = dp(left+1,right,operations+1) + nums[left]*multipliers[operations]
            take_right = dp(left,right -1,operations+1) + nums[right]*multipliers[operations]
            ans = max(take_left,take_right)
            memo[(left,right,operations)] = ans
            return ans

        return dp(0,n-1,0)

class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        '''
        we need to reduce from three states back
        it turns out that in the recursion tree, we visit repeated states but going down seperate paths
        recall in our last soluiton we had three states left,right, ops
        right = n  - 1 - (ops - left)
        
        rather if we have left, then there are ops-left on the right side
        
        dp(ops,left) stores the maxi score possible after we have done a total of operations using left numbers
        
        dp(left,ops) = max{
                left = dp(left+1,ops+1) + nums[left]*multipliers[ops]
                right = dp(left,ops+1) + nums[n - 1 - (ops - left)]*multipliers[ops]
        } when we have ops available
        '''
        # Number of Operations
        m = len(multipliers)

        # For Right Pointer
        n = len(nums)

        memo = {}

        def dp(op, left):
            if op == m:
                return 0

            # If already computed, return
            if (op, left) in memo:
                return memo[(op, left)]

            l = nums[left] * multipliers[op] + dp(op+1, left+1)
            r = nums[(n-1)-(op-left)] * multipliers[op] + dp(op+1, left)

            memo[(op, left)] = max(l, r)

            return memo[(op, left)]

        # Zero operation done in the beginning
        return dp(0, 0)

class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        '''
        tranlasting to bottom up
        '''
        m = len(multipliers)
        n = len(nums)
        
        #dp array will b m by m, since ops can very up to m and left can veary up to m
        dp = [[0]*(m+1) for _ in range(m+1)]
        
        #start with ops from m-1
        for ops in range(m-1,-1,-1):
            for left in range(ops,-1,-1):
                #if we were to start with left m-1
                #if ops >= left:
                l = nums[left] * multipliers[ops] + dp[ops+1][left+1]
                r = nums[(n-1)-(ops-left)] * multipliers[ops] + dp[ops+1][left]
                
                dp[ops][left] = max(l,r)
        
        return dp[0][0]
        
##############################
# 336. Palindrome Pairs (REVISITED)
# 17SEP22
##############################
class Solution:
    def palindromePairs(self, words: List[str]) -> List[List[int]]:
        '''
        another way of seeing this problem
        https://medium.com/@harycane/palindrome-pairs-46c5b8511397
        
        algo:
        1 Since we need return a list of lists, containing indices whose words if we combine shall result in a palindrome, it makes intuitive sense to have a mapping between words in the given array and their corresponding indices. Therefore lets populate a hash map with the words and their indices.

2 For each unique word in the words array, consider the word to be divided into two substrings, str1 and str2, with str1 progressively increasing from “” (empty) substring to the entire word while str2 assumes the remaining part of the word. Check if str1 is a palindrome and if so then there is a possibility of it functioning as the pivot around which a palindrome could be formed. In order to determine whether a palindrome could be indeed formed, determine whether the reverse of the str2 exists within the map and does not correspond to the current index in contention (as is the case in case str2 is “aa” in which case reverse of str2 is also “aa” and hence can correspond to the current index in the map), so as to function as a prefix to form a palindrome with str1 as the pivot.

3 If the reversed string is indeed present in the map and does not correspond to the current index, then you have got one pair of palindromes that can be added to the result list of lists. Create a temporary list, and add the prefix reversed string’s index to the temp list first, followed by current index i and add the temp list to the resultant list of lists.

4 Now like wise check if str2 is a palindrome, in which case it can function as a pivot around which a palindrome can be formed. Check if the str1’s reverse is present in the map and does not correspond to the current index. Also in order to consider the corner case of “” empty string being one of the words in the array, there is a need to iterate until the length of word[i] inclusive. But this may lead to empty string “” being considered in str2 as a duplicate in addition to being considered initially in str1. Therefore care must be taken to ensure that str2 is not equal to empty string to avoid duplicates. If all the above checks are satisfied, then create a temp list, add the prefix index i and the suffix index from the map corresponding to the reversed str1, and add the temp list to result list of lists. After iterating through the entire words list, return the result list of lists.
        '''
        ans = []
        N = len(words)
        
        if not words or len(words) < 2:
            return ans
        
        #store words to index
        mapp = {}
        for i,word in enumerate(words):
            mapp[word] = i
        
        #now go through the words and gerneate substrings of each word
        for i in range(N):
            for j in range(len(words[i])+1):
                #to generate all subtrings and empty string cases, since they could be present
                str1 = words[i][0:j]
                str2 = words[i][j:]
                
                #if str1 is a palindrom, i can make a pair if the reverse of str2 is in the mapp, and not the current index in contection
                if str1 == str1[::-1]:
                    str2_rev = str2[::-1]
                    if str2_rev in mapp and mapp[str2_rev] != i:
                        ans.append([mapp[str2_rev],i])
                
                #now the revesre case, if str2 is also a palindrome
                if str2 == str2[::-1]:
                    str1_rev = str1[::-1]
                    if str1_rev in mapp and mapp[str1_rev] != i and len(str2) != 0: #we may have already used the empty string on the first time when generating a candidate palindrom
                        ans.append([i,mapp[str1_rev]])
        
        return ans

#Trie solution
class TrieNode:
    def __init__(self):
        self.next = collections.defaultdict(TrieNode)
        self.ending_word = -1
        self.palindrome_suffixes = []

class Solution:
    def palindromePairs(self, words):

        # Create the Trie and add the reverses of all the words.
        trie = TrieNode()
        for i, word in enumerate(words):
            word = word[::-1] # We want to insert the reverse.
            current_level = trie
            for j, c in enumerate(word):
                # Check if remainder of word is a palindrome.
                if word[j:] == word[j:][::-1]:# Is the word the same as its reverse?
                    current_level.palindrome_suffixes.append(i)
                # Move down the trie.
                current_level = current_level.next[c]
            current_level.ending_word = i

        # Look up each word in the Trie and find palindrome pairs.
        solutions = []
        for i, word in enumerate(words):
            current_level = trie
            for j, c in enumerate(word):
                # Check for case 3.
                if current_level.ending_word != -1:
                    if word[j:] == word[j:][::-1]: # Is the word the same as its reverse?
                        solutions.append([i, current_level.ending_word])
                if c not in current_level.next:
                    break
                current_level = current_level.next[c]
            else: # Case 1 and 2 only come up if whole word was iterated.
                # Check for case 1.
                if current_level.ending_word != -1 and current_level.ending_word != i:
                    solutions.append([i, current_level.ending_word])
                # Check for case 2.
                for j in current_level.palindrome_suffixes:
                    solutions.append([i, j])
        return solutions

        
###############################
# 42. Trapping Rain Water (REVISITED)
# 18SEP22
###############################
class Solution:
    def trap(self, height: List[int]) -> int:
        '''
        lefts array and rights array problem
        '''
        #first find max values to the left
        N = len(height)
        #don't forget to fill in the base cases
        max_lefts = [0]*N
        max_lefts[0] = height[0]
        max_rights =[0]*N
        max_rights[-1] = height[-1]
        
        #when finding max lefts we start from the beginning of he array
        for i in range(1,N):
            #take max of current max or heights
            max_lefts[i] = max(max_lefts[i-1],height[i])
        
        #max to the rights, we start at tned
        for i in range(N-2,-1,-1):
            max_rights[i] = max(max_rights[i+1],height[i])
        
        #now we just take the min of left and right at each increment by height
        ans = 0
        for i in range(N):
            water_level = min(max_lefts[i],max_rights[i])
            if water_level > height[i]:
                ans += water_level - height[i]
        
        return 

class Solution:
    def trap(self, height: List[int]) -> int:
        '''
        insteaf of storing the largest bar up to an index, we can keep a montonic stack
        keep track of bards that are bounded by longer bars
        
        we keeps tack and iterate over the array, we then added the idnex of the bar to the stack if the bar is <= to the bar at the top of the stack
        which means the current bar is bounded by the previous bar in the stack
        if we found a bar logner than the one at the top, we are sure that the bar at the top of the stack is bounded by the cyrrent bar 
        so we can pop it and add the resulting trapper water to ans
        
        traverse the height array
        while stack and current bar > top of stack:
            pop the top
            find the distance between the current element and the element at the top of the stick, this is the distance to be filled
            find the bounded height: min(current bar, height[top]) - height top
            add trapped water to result
        '''
        
        ans = 0
        current = 0
        N = len(height)
        stack = []
        
        #keep track of index on the stack
        while current < N:
            while len(stack) > 0 and height[current] > height[stack[0]]:
                #we have a bigger bar
                top = stack[0]
                stack.pop()
                #causes stack to be empty
                if len(stack) == 0:
                    break
                distance = current - stack[0] - 1
                bounded_height = min(height[current],height[stack[0]] - height[top])
                ans += distance*bounded_height
            
            stack.append(current)
            current += 1
        
        return ans

#two pointers
class Solution:
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        areas = 0
        max_l = max_r = 0
        l = 0
        r = len(height)-1
        while l < r:
            if height[l] < height[r]:
                if height[l] > max_l:
                    max_l = height[l]
                else:
                    areas += max_l - height[l]
                l +=1
            else:
                if height[r] > max_r:
                    max_r = height[r]
                else:
                    areas += max_r - height[r]
                r -=1
        return areas


#https://leetcode.com/problems/trapping-rain-water/discuss/1374608/C%2B%2BJavaPython-MaxLeft-MaxRight-so-far-with-Picture-O(1)-space-Clean-and-Concise

#################################
# 609. Find Duplicate File in System (REVISTED)
# 19SEP22
#################################
#almost had it
class Solution:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        '''
        i can split on space
        and then use hashmap to group them by file contencts
        '''
        content_to_path = defaultdict(list)
        
        for path in paths:
            path = path.split(' ')
            #get the root path
            root_path = path[0]
            #get contents and firt sub root
            for file in path[1:]:
                content = file[file.find('(')+1:file.find(')')]
                #get path name for first part
                folder = file[:file.find('(')]
                full_path = root_path+'/'+folder
                #put in mapp
                content_to_path[content].append(full_path)
        
        
        ans = []
        for k,v in content_to_path.items():
            if len(v) > 1:
                ans.append(k)
        
        return 

################################
# 718. Maximum Length of Repeated Subarray (Revisited)
# 20SEP22
################################
#Recursive solution gets MLE, calling it too many times
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        from the hint lets use dp
        dp(i,j) represent the longest repeated subarray using nums1[i:] and nums2[j:]
        
        dp(i,j) = if nums1[i] == nums2[j]:
            1 + max(dp(i-1,j),dp(i))
        '''
        memo = {}
        M = len(nums1)
        N = len(nums2)
        
        def rec(i,j):
            if i < 0 or j < 0:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            if nums1[i] == nums2[j]:
                res = 1 + rec(i-1,j-1)
            else:
                res = 0
            memo[(i,j)] = res
            return res
        
        
        ans = 0
        for i in range(M):
            for j in range(N):
                ans = max(ans,rec(i,j))
        
        return ans

class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        M = len(nums1)
        N = len(nums2)
        
        dp = [[0]*(N+1) for _ in range(M+1)]
        ans = 0
        
        
        for i in range(1,M+1):
            for j in range(1,N+1):
                if nums1[i-1] == nums2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = 0
                
                ans = max(ans,dp[i][j])
        
        return ans

#cheeky
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        #convert to char string, better to use chr than str, to identify repated element of multiple digits
        strnum2 = ''.join([chr(x) for x in nums2])
        #we are going to build this string
        strmax = ''
        ans = 0
        for num in nums1:
            #add the new char character
            strmax += chr(num)
            #if its prefext, we have at least len(strmax)
            if strmax in strnum2:
                ans = max(ans,len(strmax))
            #if it's not, we need to move up this prefix
            else:
                strmax = strmax[1:]
        return ans

#rolling hash
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can use rolling hash
        algo:
            precompute the rolling hahs of nums1 and nums2
            when choosing base we pick a larger primer number bigger than the max
            mod we can choose the largest primer number that fits into a 32 signed bit
            then binary search to find the maximum size of subarray appearing both nums1 and nums2
            we just checl all subarrays for all sizes using hashmap
            we can get the hash of a subarray in O(1) time using the procomputations
        '''
        M = len(nums1)
        N = len(nums2)
        base = 101 #largest number in either of the two arrays is at least 100
        mod = 1_000_000_000_001
        hash1 = [0]*(M+1)
        hash2 = [0]*(N+1)
        POW = [1]*(max(M,N)+1)
        
        #compute powers for each base
        for i in range(max(M,N)):
            POW[i+1] = POW[i]*base % mod
        #compute hashing for nums1, this is not the hash per se
        for i in range(M):
            hash1[i+1] = (hash1[i]*base + nums1[i]) % mod
        #compute hashing values for nums2
        for i in range(N):
            hash2[i+1] = (hash2[i]*base + nums2[i]) % mod
        
        #function to get hash for a subarray, 0 based indexing and right inclusive
        
        def getHash(h,left,right):
            return (h[right+1] - h[left]*POW[right - left+1] % mod + mod) % mod
        
        #function to find a valid subarray, i.e whos'e hash1 == hash2
        def foundSubArray(size):
            seen = defaultdict(list)
            #get first hash
            for i in range(M - size + 1):
                h = getHash(hash1,i,i+size-1)
                #add the starting index of the subarray
                seen[h].append(i)
            #get the second hash
            for i in range(N - size+1):
                h = getHash(hash2,i,i+size-1)
                if h in seen:
                    for j in seen[h]:
                        #compare starting index for all j with the current i
                        if nums1[j:j+size] == nums2[i:i+size]:
                            return True
            return False
        
        #we can use binary search to find a workable solution for size
        left = 1
        right = min(M,N)
        ans = 0
        
        while left <= right:
            mid = left + (right - left) // 2
            if foundSubArray(mid):
                ans = mid
                #expand
                left = mid + 1
            else:
                #shrink
                right = mid - 1
        
        return ans

############################
# 985. Sum of Even Numbers After Queries
# 21SEP22
############################
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        '''
        first try simulating and see if you can spot patterns
        '''
        even_vals = []
        for val,index in queries:
            nums[index] += val
            #get sum of even values, this is the bottle neck here
            curr_sum = 0
            for num in nums:
                if num % 2 == 0:
                    curr_sum += num
            even_vals.append(curr_sum)
        
        return even_vals

#close one
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        '''
        i can first find the curr sum of even numbers
        then alter the sum after i apply the query
        
        there are multiple things that could happen
        the numbs[index] is curetnyl even or its currently odd
        then we need to reflect this on the current sum
        '''
        even_sums = []
        curr_sum = 0
        for num in nums:
            if num % 2 == 0:
                curr_sum += num
        
        
        for val,index in queries:
            if nums[index] % 2 == 0:
                #if adding makes it even
                cand = nums[index] + val
                if cand % 2 == 0:
                    curr_sum += cand - nums[index]
                    nums[index] = cand
                else:
                    curr_sum -= nums[index]
                    nums[index] = cand
            else:
                cand = nums[index] + val
                if cand % 2 == 0:
                    curr_sum += cand - nums[index]
                    nums[index] = cand
                else:
                    curr_sum -= nums[index]
                    nums[index] = cand
            
            even_sums.append(curr_sum+1)
        
        return even_sums

class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        '''
        when adding to nums[index], the rest of the values of nums remain the same
        lets remove nums[index] from the current sum if it even
        make the adjustment
        then add nums[index] + val back in if it is even
        
        think of a few cases
        If we have A = [2,2,2,2,2], S = 10, and we do A[0] += 4: we will update S -= 2, then S += 6. At the end, we will have A = [6,2,2,2,2] and S = 14.

        If we have A = [1,2,2,2,2], S = 8, and we do A[0] += 3: we will skip updating S (since A[0] is odd), then S += 4. At the end, we will have A = [4,2,2,2,2] and S = 12.

        If we have A = [2,2,2,2,2], S = 10 and we do A[0] += 1: we will update S -= 2, then skip updating S (since A[0] + 1 is odd.) At the end, we will have A = [3,2,2,2,2] and S = 8.

        If we have A = [1,2,2,2,2], S = 8 and we do A[0] += 2: we will skip updating S (since A[0] is odd), then skip updating S again (since A[0] + 2 is odd.) At the end, we will have A = [3,2,2,2,2] and S = 8.
        
        '''
        curr_sum = 0
        ans = []
        for num in nums:
            if num % 2 == 0:
                curr_sum += num
                
        for val,index in queries:
            #if even take it away
            if nums[index] % 2 == 0:
                curr_sum -= nums[index]
            #make the adjustment
            nums[index] += val
            #now if its even again, add it back in, remmeber we initally took it away
            if nums[index] % 2 == 0:
                curr_sum += nums[index]
            ans.append(curr_sum)
        
        return ans

#segment tree solution
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        '''
        we can use segment tree to answer the querie questino in log(len(nums)) time
        updates would also happend in log(len(nums)) time
        '''
        #use array of 4 times the size
        N = len(nums)
        segment_tree = [0]*(4*N)
        
        def build_tree(i,left,right):
            if left == right:
                if nums[left] % 2 == 0:
                    segment_tree[i] = nums[left]
                else:
                    segment_tree[i] = 0
                return
            
            else:
                mid = left + (right - left) // 2
                build_tree(i*2,left,mid)
                build_tree(i*2 + 1,mid+1,right)
                segment_tree[i] = segment_tree[i*2] + segment_tree[i*2 + 1]
        
        #build_tree(0,0,N-1)
        #print(segment_tree)
        
        def query_sum(v,tl,tr,l,r):
            #if we go past
            if l > r:
                return 0
            #hit target
            if l == tl and r == tr:
                return segment_tree[v]
            #subproblem summation
            t_mid = tl + (tr - tl) // 2
            left = query_sum(v*2, tl, t_mid, l, min(r,t_mid))
            right = query_sum(v*2 + 1, t_mid+1,tr,max(l,t_mid+1),r)
            return left + right
        
        #print(query_sum(0,0,len(segment_tree),0,N))
        
        def update(v,tl,tr,pos,new_val):
            if tl == tr:
                segment_tree[v] = new_val
            else:
                tm = tl + (tr - tl) // 2
                if pos <= tm:
                    update(v*2,tl,tm,pos,new_val)
                else:
                    update(v*2 +1,tm+1,tr,pos,new_val)
            
            segment_tree[v] = segment_tree[v*2] + segment_tree[v*2+1]
            
        ans = []
        build_tree(0,0,N-1)
        for val,index in queries:
            
            update(0,0,len(segment_tree),index,nums[index]+val)
            curr_sum = query_sum(0,0,len(segment_tree),0,N)
            ans.append(curr_sum)
        
        return ans
            

#we actually don't neet to query, just pull the root
class Solution:
    def sumEvenAfterQueries(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        
        
        seg = [0]*4*len(nums)
        def build(idx,lo,hi):
            if lo == hi:
                if nums[lo]%2 == 0:
                    seg[idx] = nums[lo]
                return
            mid = (lo+hi)//2
            build(2*idx+1,lo,mid)
            build(2*idx+2,mid+1,hi)
            seg[idx] = seg[2*idx+1]+seg[2*idx+2]
        def update(idx,lo,hi,i,val):
            if lo == hi:
                seg[idx] = val if val % 2 == 0 else 0
                return
            mid = (lo+hi)//2
            if i <= mid:
                update(2*idx+1,lo,mid,i,val)
            else:
                update(2*idx+2,mid+1,hi,i,val)
            seg[idx] = seg[2*idx+1]+seg[2*idx+2]
            
        
        #just for the sake of segment tree completeness
        def query(idx,lo,hi,l,r):
            if r < lo or l > hi:
                return 0
            if l >= lo and r <= hi:
                return seg[idx]
            mid = (lo+hi)//2
            return query(2*idx+1,lo,mid,l,r)+query(2*idx+2,mid+1,hi,l,r)
        stored = [0 for _ in range(len(nums))]
        ans = []
        build(0,0,len(nums)-1)
        for i,j in queries:
            val = nums[j]+i
            if val%2 == 0:
                update(0,0,len(nums)-1,j,val)
            else:
                update(0,0,len(nums)-1,j,0)
            ans += [seg[0]]
            nums[j] += i
        return ans

###############################
# 2161. Partition Array According to Given Pivot
# 21SEP22
###############################
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        '''
        problem sounds easier than written, 
        every element less than pivot appears before every element greater than pivot
        every elmeent equal to pivot appears in between the elements less than and greater than pivot
        
        example nums = [9,12,5,10,14,3,10], pivot = 10
        less than [9,5,3]
        greater than [12,10,10,14]
        '''
        equals = []
        less_than = []
        greater_than = []
        
        for num in nums:
            if num == pivot:
                equals.append(num)
            elif num < pivot:
                less_than.append(num)
            else:
                greater_than.append(num)
                
        
        return less_than + equals + greater_than

#count num of pivots
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        '''
        another way 
        first pass to find elements < pivot, and also record frequency of numbers that are == pivot
        then get numbs greater
        '''
        pivotfreq = 0
        ans = []
        
        for num in nums:
            if num < pivot:
                ans.append(num)
            elif num == pivot:
                pivotfreq += 1
        
        ans += [pivot]*pivotfreq
        
        #second pass
        for num in nums:
            if num > pivot:
                ans.append(num)
        
        return ans

#three pointers
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        '''
        we can travese from left to right
        and into a new results array place the elements at their right position using two poiners
        we can do this in one pass using i and ~i
        ~i = -i - 1
        
        or offset using N
        
        '''
        N = len(nums)
        ans = [pivot]*N
        left = 0
        right = N - 1
        
        for i in range(N):
            if nums[i] < pivot:
                ans[left] = nums[i]
                left += 1
            if nums[N-i-1] > pivot:
                ans[right] = nums[N-i-1]
                right -= 1
        
        return ans

##############################
# 557. Reverse Words in a String III (REVISTED)
# 22SEP22
##############################
#two pointer solution
class Solution:
    def reverseWords(self, s: str) -> str:
        s = list(s)
        last_space_idx = -1
        N = len(s)
        
        for i in range(N+1):
            #reached ending or space
            if i == N or s[i] == ' ':
                left = last_space_idx + 1
                right = i - 1
                while left < right:
                    s[left],s[right] = s[right],s[left]
                    left += 1
                    right -= 1
                
                last_space_idx = i
        
        return "".join(s)

#################################
# 622. Design Circular Queue(Revisited)
# 26SEP22
#################################
# Time Complexity: O(1)
# Space Complexity: O(N)
class MyCircularQueue:

    def __init__(self, k: int):
        # the queue holding the elements for the circular queue
        self.q = [0] * k
        # the number of elements in the circular queue
        self.cnt = 0
        # queue size
        self.sz = k
        # the idx of the head element
        self.headIdx = 0
        

    def enQueue(self, value: int) -> bool:
        # handle full case
        if self.isFull(): return False
        # Given an array of size of 4, we can find the position to be inserted using the formula
        # targetIdx = (headIdx + cnt) % sz
        # e.g. [1, 2, 3, _]
        # headIdx = 0, cnt = 3, sz = 4, targetIdx = (0 + 3) % 4 = 3
        # e.g. [_, 2, 3, 4]
        # headIdx = 1, cnt = 3, sz = 4, targetIdx = (1 + 3) % 4 = 0
        self.q[(self.headIdx + self.cnt) % self.sz] = value
        # increase the number of elements by 1
        self.cnt += 1
        return True

    def deQueue(self) -> bool:
        # handle empty case
        if self.isEmpty(): return False
        # update the head index
        self.headIdx = (self.headIdx + 1) % self.sz
        # decrease the number of elements by 1
        self.cnt -= 1
        return True

    def Front(self) -> int:
        # handle empty queue case
        if self.isEmpty(): return -1
        # return the head element
        return self.q[self.headIdx]
        
    def Rear(self) -> int:
        # handle empty queue case
        if self.isEmpty(): return -1
        # Given an array of size of 4, we can find the tail using the formula
        # tailIdx = (headIdx + cnt - 1) % sz
        # e.g. [0 1 2] 3
        # headIdx = 0, cnt = 3, sz = 4, tailIdx = (0 + 3 - 1) % 4 = 2
        # e.g. 0 [1 2 3]
        # headIdx = 1, cnt = 3, sz = 4, tailIdx = (1 + 3 - 1) % 4 = 3
        # e.g. 0] 1 [2 3
        # headIdx = 2, cnt = 3, sz = 4, tailIdx = (2 + 3 - 1) % 4 = 0
        return self.q[(self.headIdx + self.cnt - 1) % self.sz]

    def isEmpty(self) -> bool:
        # no element in the queue
        return self.cnt == 0

    def isFull(self) -> bool:
        # return True if the count is equal to the queue size
        # else return False
        return self.cnt == self.sz


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()

#using linkedList

class Node:
    def __init__(self, value, nextNode=None):
        self.value = value
        self.next = nextNode

class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.capacity = k
        self.head = None
        self.tail = None
        self.count = 0

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        if self.count == self.capacity:
            return False
        
        if self.count == 0:
            self.head = Node(value)
            self.tail = self.head
        else:
            newNode = Node(value)
            self.tail.next = newNode
            self.tail = newNode
        self.count += 1
        return True


    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        if self.count == 0:
            return False
        self.head = self.head.next
        self.count -= 1
        return True


    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        if self.count == 0:
            return -1
        return self.head.value

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        # empty queue
        if self.count == 0:
            return -1
        return self.tail.value
    
    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.count == 0

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.count == self.capacity


###########################################
# 272. Closest Binary Search Tree Value II
# 24SEP22
###########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def closestKValues(self, root: Optional[TreeNode], target: float, k: int) -> List[int]:
        '''
        brute force would be to traverse the whole tree and record the distanks from target
        then sort on their distances away from target and return the first k
        '''
        node_dists = []
        
        def dfs(node):
            if not node:
                return
            node_dists.append([abs(node.val - target),node.val])
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        #sort
        node_dists.sort()
        ans = []
        for i in range(k):
            ans.append(node_dists[i][1])
        
        return ans

#now using property of BSTs


###################################
# 990. Satisfiability of Equality Equations
# 26SEP22
###################################
#close one
class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        '''
        i can traverse the equations and make mappings for each thing
        two maps equals and not equals
        then we can just check the validaty of the string and try to make it to the end
        nope, its a graph problem
        
        dfs solution
            built adj list of colors, of unidretec graph where each variable equalts another
            then we dfs and color each equal, example a == b, b == c, then a == b, these three would be all the same color
            then we traverse the equations and check for a contradiction of x != y
        '''
        adj_list = defaultdict(list)
        for eq in equations:
            if eq[1] == '=':
                a = eq[0]
                b = eq[-1]
                adj_list[a].append(b)
                adj_list[b].append(a)
        
        
        #we only have 26 lower case chars, so lets try to color then
        #use dfs to mark groupings, color to nodes
        seen = set()
        groups = []
        
        def dfs(node,comp):
            if node not in seen:
                seen.add(node)
                comp.add(node)
                for neigh in adj_list[node]:
                    dfs(neigh,comp)
                    
        for node in adj_list:
            comp = set()
            dfs(node,comp)
            #if you have a group
            if len(comp) > 0:
                groups.append(comp)
        
        #now make new mapping
        node_to_color = {}
        for i in range(len(groups)):
            nodes = groups[i]
            for n in nodes:
                node_to_color[n] = i
        
        #check
        for eq in equations:
            if eq[1] == '!':
                a = eq[0]
                b = eq[-1]
                if node_to_color[a] == node_to_color[b]:
                    return False
        
        return True

#dfs using coloring
class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        '''
        we can use dfs and color each node along the way
        we really have an undirecte graph of connected components when we == each variable
        thenwe just need to check for a contradiction
        '''
        adj_list = defaultdict(list)
        for eq in equations:
            if eq[1] == '=':
                a = eq[0]
                b = eq[-1]
                adj_list[a].append(b)
                adj_list[b].append(a)
        
        node_to_color = defaultdict()
        
        def dfs(node,color):
            if node not in node_to_color:
                #mark
                if node not in node_to_color:
                    node_to_color[node] = color
                    for neigh in adj_list[node]:
                        dfs(neigh,color)
        
        #check for all variables between a and b
        for i in range(26):
            node = chr(ord('a') + i)
            color = i
            if node not in node_to_color:
                dfs(node,color)
        
        #check
        for eq in equations:
            if eq[1] == '!':
                a = eq[0]
                b = eq[-1]
                if node_to_color[a] == node_to_color[b]:
                    return False
        
        return True

#union find
#https://leetcode.com/problems/satisfiability-of-equality-equations/discuss/2625039/LeetCode-The-Hard-Way-Explained-Line-By-Line
class Solution:
    def equationsPossible(self, equations: List[str]) -> bool:
        '''
        we can use union find
        the idea is to put all the characters in the same group if they are equal
        
        intially each variable points to itself, when we haven == we need to join them using the uion method
        we build the graph, and then check for a contradiction again
        '''
        
        roots = [i for i in range(26)]
        
        def find(x):
            #keep searching until it points to it selft
            if x != roots[x]:
                roots[x] = find(roots[x])
            
            return roots[x]
        
        def union(x,y):
            x = find(x)
            y = find(y)
            roots[x] = y
            
        for eqn in equations:
            if eqn[1] == '=':
                x = ord(eqn[0]) - ord('a')
                y = ord(eqn[-1]) - ord('a')
                union(x,y)
        
        for eqn in equations:
            if eqn[1] == '!':
                x = ord(eqn[0]) - ord('a')
                y = ord(eqn[-1]) - ord('a')
                if find(x) == find(y):
                    return False
        return True

#using only find method
class Solution:
    # the idea is to put all characters in the same group if they are equal
    # in order to do that, we can use Disjoint Set Union (dsu) aka Union Find
    # for dsu tutorial, please check out https://wingkwong.github.io/leetcode-the-hard-way/tutorials/graph-theory/disjoint-set-union
    def equationsPossible(self, equations: List[str]) -> bool:
        # find the root of node x. 
        # here we are not using parent[x] 
        # because it may not contain the updated value of the connected component that x belongs to. 
        # Therefore, we walk the ancestors of the vertex until we reach the root.
        def find(x):
            # with path compress
            if parent[x] == x:
                return x
            parent[x] = find(parent[x])
            return parent[x]
            # without path compression
            #return x if parent[x] == x else find(parent[x])
        # at the beginning, put each character in its own group
        # so we will have 26 groups with one character each
        # i.e. 'a' in group 0, 'b' in group 1, ..., 'z' in group 25
        parent = [i for i in range(26)]
        for e in equations:
            if e[1] == '=':
                # e.g. a == b
                # then we group them together
                # how? we use `find` function to find out the parent group of the target character index
                # then update parent. a & b would be in group 1 (i.e. a merged into the group where b belongs to)
                # or you can also do `parent[find(ord(e[3]) - ord('a'))] = find(ord(e[0]) - ord('a'))`
                # i.e. b merged into the group where a belongs to
                parent[find(ord(e[0]) - ord('a'))] = find(ord(e[3]) - ord('a'))
        # handle != case
        for e in equations:
            # if two characters are not equal
            # then which means their parent must not be equal
            if e[1] == '!' and find(ord(e[0]) - ord('a')) == find(ord(e[3]) - ord('a')):
                return False
        return True

###############################
# 838. Push Dominoes (REVISTED)
# 27SEP22
###############################
#start writing solutions in C++
'''
class Solution {
public:
    string pushDominoes(string s) {
        //https://leetcode.com/problems/push-dominoes/discuss/2628923/C%2B%2B-or-Two-Pointer-or-Diagram-or-Related-Problems
        /*
        two pointer solution, initially set right pointer to -1
        1. if we encounter . in a string, we do nothing
        2. if we encounter an L in a string, we need to if index right is -1, and make everythng previous to L
        3.If we encounter L in string and there is some previous R index, then we simultaneously change string from left and right side till two pointers reach each other. After that right moves back to -1.
        4. If we encounter R in string, we see if the index of R is not -1, we make all the indices upto that index R.
        */
    int N = s.size(), right = -1;
    for (int i = 0; i < N; ++i) {
        if (s[i] == 'L') {
            if (right == -1) { 
                // Step 2
                for (int j = i - 1; j >= 0 && s[j] == '.'; --j) {
                  s[j] = 'L';  
                } 
            } else {
                // Step 8
                for (int j = right + 1, k = i - 1; j < k; ++j, --k) {
                    s[j] = 'R';
                    s[k] = 'L';
                } 
                right = -1;
            }
        } else if (s[i] == 'R') {
            if (right != -1) {
                for (int j = right + 1; j < i; ++j) s[j] = 'R';
            }
            right = i;
        }
    }
    if (right != -1) {
        for (int j = right + 1; j < N; ++j) s[j] = 'R';
    }
    return s;
};
};
'''
#python version
#https://leetcode.com/problems/push-dominoes/discuss/2628923/C%2B%2B-or-Two-Pointer-or-Diagram-or-Related-Problems
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        N = len(dominoes)
        dominoes = list(dominoes)
        right = -1
        for i in range(N):
            #if we hit a left
            if dominoes[i] == 'L':
                #if we are anchored right
                if right == -1:
                    #change everything in between
                    j = i -1
                    while j >=0 and dominoes[j] == '.':
                        dominoes[j] = 'L'
                        j -= 1
                #if we have found a closer right
                else:
                    j = right + 1
                    k = i - 1
                    while j < k:
                        dominoes[j] = 'R'
                        dominoes[k] = 'L'
                        j += 1
                        k -= 1
                #move anchor back
                right = -1
            #if we hit an R dominoe
            elif dominoes[i] == 'R':
                #if we don't have an earlier right
                if right != -1:
                    j = right + 1
                    while j < i:
                        dominoes[j] = 'R'
                        j += 1
                right = i
        
        #change last rights
        if right != -1:
            j = right + 1
            while j < N:
                dominoes[j] = 'R'
                j += 1
        
        return "".join(dominoes)

#https://leetcode.com/problems/push-dominoes/discuss/2628871/LeetCode-The-Hard-Way-Explained-Line-By-Line
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        d = list(dominoes)
        # l is the left pointer
        l, n = 0, len(dominoes)
        # r is the right pointer
        for r in range(n):
            if d[r] == '.':
                # case 1. meeting `.`, then skip it
                continue
            elif (d[r] == d[l]) or (d[l] == '.' and d[r] == 'L'):
                # case 2. both end is equal, i.e. d[r] == d[l]
                # then fill all the dots between both end 
                # e.g. L....L -> LLLLLL
                # e.g. R....R -> RRRRRR
                # case 2.1 if the left end is . and the right end is L, 
                # i.e. d[l] == '.' && d[r] == 'L'
                # then we need to fill them from `l` to `r` in this case
                for k in range(l, r):
                    # case 3. left end is L and right end is R
                    # e.g. L.....R
                    # then do nothing
                    d[k] = d[r]
            elif d[l] == 'L' and d[r] == 'R':
                # case 3. left end is L and right end is R
                # e.g. L.....R
                # then do nothing
                pass
            elif d[l] == 'R' and d[r] == 'L':
                # case 4. left end is R and right end is L
                # if we have odd number of dots between them (let's say m dots), 
                # then we can only add (m // 2) Ls and (m // 2) Rs. 
                # p.s // here is integer division. e.g. 3 // 2 = 1
                # e.g. R...L -> RR.LL 
                # if we have even number of dots between them (let's say m dots), 
                # then we can only add (m // 2) Ls and (m // 2) Rs. 
                # e.g. R....L -> RRRLLL
                m = (r - l - 1) // 2
                for k in range(1, m + 1):
                    d[r - k] = 'L'
                    d[l + k] = 'R'
            # update left pointer
            l = r
        
        # case 5. if the left dominoe is `R`, then fill all 'R' till the end
        # e.g. LL.R. -> LL.RR
        if d[l] == 'R':
            for k in range(l, n):
                d[k] = 'R'
                
        return ''.join(d)
                        
#forces revisted
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        '''
        we can define the net force for every domino
        we only care about close a dominos is to a leftward R and a rightward L, then the direction of the domino goes to the magnitude
        
        algo:
            going from left to right, our force decays every iteration by 1, or sets back to N when we hit an R
            samething going right to left, but we are not choosing the L dominoes
        '''
        N = len(dominoes)
        
        right_forces = [0]*N
        left_forces = [0]*N
        sum_forces = [0]*N
        
        #going left to right
        curr_force = 0
        for i in range(N):
            if dominoes[i] == 'R':
                curr_force = N
            elif dominoes[i] == 'L':
                curr_force = 0
            else:
                curr_force = max(curr_force -1, 0)
            
            right_forces[i] = curr_force
            
        
        #right to left
        curr_force = 0
        for i in range(N-1,-1,-1):
            if dominoes[i] == 'L':
                curr_force = N
            elif dominoes[i] == 'R':
                curr_force = 0
            else:
                curr_force = max(curr_force -1, 0)
            
            left_forces[i] = curr_force
            
        #subtract up vectors by element
        for i in range(N):
            sum_forces[i] = right_forces[i] - left_forces[i]
        
        ans = ""
        for f in sum_forces:
            if f > 0:
                ans += 'R'
            elif f < 0:
                ans += 'L'
            else:
                ans += '.'
        
        return ans
        
###################################
# 218. The Skyline Problem (REVISTED)
# 30SEP22
###################################
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        '''
        what's also hard about this problem is how to encode/represent the answer
        the answer should be sorted by their x-coordinate, where each key point is the left endpoint of some horizontal segment
        the last cooridnate should have a value of zero, which marks the edge of the right most building
        and ground in between should be part of hte countour and should have height of zero
        
        brute force 1
            intuion: if a building with height h covers indices from x_i to x_j, then all the indices from x_i to x_j, then the height between x_i and x_j is at least h
            for each building find the boundaries and store the max height at each
            finally traverse the updaed heights and output all the positions where height changes as skyline key points
        '''
        #sort unique positions of all the edges
        edges = set()
        for x,y,h in buildings:
            edges.add(x)
            edges.add(y)
        
        #sort
        edges = sorted(list(edges))
        
        #make mapp, {position:index}
        edgeIndexMap = {}
        for i,p in enumerate(edges):
            edgeIndexMap[p] = i
        
        #initilize heights to record max
        heights = [0]*len(edges)
        
        #pass over buildings
        for left,right,height in buildings:
            #for each buuilding find where it spans in the heights array
            #recall we mapped position to index
            left_idx = edgeIndexMap[left]
            right_idx = edgeIndexMap[right]
            
            #for this current height maximize it
            #notice we exlucde the right edge
            for i in range(left_idx,right_idx):
                heights[i] = max(heights[i],height)
        
        ans = []
        #iterate over the height array to record changes in height, this is the skyline
        for i in range(len(heights)):
            curr_height = heights[i]
            #get current edge, kinda like mapping heights to edgees
            curr_edge = edges[i]
            #add to answer
            if not ans or ans[-1][1] != curr_height:
                ans.append([curr_edge,curr_height])
        
        return ans

#line sweep
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        '''
        we can use line sweep with vertical line passing through each building and record the max at each edge
        remember the rightmost builindg doesn't count
        '''
        positions = sorted(list(set([edge for building in buildings for edge in building[:2]])))
        
        ans = []
        
        for p in positions:
            #store curretn max height for this line sweep
            max_height = 0
            for left,right,height in buildings:
                #update
                if left <= p < right:
                    max_height = max(max_height,height)
            
            #add in
            if not ans or max_height != ans[-1][-1]:
                ans.append([p,max_height])
        
        return ans


#line sweep with pq
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        '''
        line sweep with priority q
        recall for each edge we previously had to go through the whole buildings array to find the heighest height
        is there a way to get the height heigt more effeciently, we need to use q priority queue
        
        check tallest building in live is past the current building
        algo:
            iteratore over buildings and store each building's edges 
            sort the endges
            iterate over the sorted edges and for each edge/index:
                if buildings[0][0] == curr_x, meanings its left edge and tghe building[b] is live, we add (height,right) to live
                while the tallest building has beend passed, remove it from live
            
            once we finish handlgin the edges at the curr_x, move on to the next positions
        '''
        #pass over buildgins and for each building store [position,index]
        edges = []
        for i,b in enumerate(buildings):
            edges.append([b[0],i])
            edges.append([b[1],i])
        
        #sort edges
        edges.sort()
        
        ans = []
        live = []
        idx = 0
        
        #iterate over sorted edges
        while idx < len(edges):
            #we may have multiple edges at the same x
            curr_x = edges[idx][0]
            
            #while we are handlingg the edges at 'curr_x'
            while idx < len(edges) and edges[idx][0] == curr_x:
                #get the index of this building
                b = edges[idx][1]
                
                #if this is a left edge, building b, meaning we are past, add its (height,right) to live
                if buildings[b][0] == curr_x:
                    right = buildings[b][1]
                    height = buildings[b][2]
                    heapq.heappush(live,[-height,right])
            
                #if the tallest live building has been passed, remove it
                while live and live[0][1] <= curr_x:
                    heapq.heappop(live)
                idx += 1
            
            #get the maximum height from live
            max_height = -live[0][0] if live else 0
            
            #change in heights means part of skyling
            if not ans or max_height != ans[-1][1]:
                ans.append([curr_x,max_height])
        
        return ans

#two pq
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        '''
        instead of keeping track of the building'sindex to retrieve informatino indicating we have we past its right edge
        we can use to heaps, one for live and one for past
        whenever we meet the left edge of a building, we just add its height to live
        
        question? how do we know if some buildings apart from the top should be removed
            use another pq, called past to keep all thje buildings that should be removed from live but haven't been uet
            
        intuition:
            live works as a debt card
                once we are able to pay the debt, that is, when the top building in live == top in past
                we remove it from past
                since debt has been cleared, we will remove the top building from past as well
                
        we repeated remote top both live and past until:
            past is empty
            the top building in live is taller than than the top building in past
                in this case we may still have some buildings to remove
        '''
        # Iterate over the left and right edges of all the buildings, 
        # If its a left edge, add (left, height) to 'edges'.
        # Otherwise, add (right, -height) to 'edges'.
        # to help distinguish between left edge and right edge
        edges = []
        for left, right, height in buildings:
            edges.append([left, height])
            edges.append([right, -height])
        edges.sort()
        
        # Initailize two empty priority queues 'live' and 'past' 
        # for the live buildings and the past buildings.
        live, past = [], []
        answer = []
        idx = 0
        
        # Iterate over all the sorted edges.
        while idx < len(edges):
            # Since we might have multiple edges at same x,
            # Let the 'curr_x' be the current position.
            curr_x = edges[idx][0]
            
            # While we are handling the edges at 'curr_x':
            while idx < len(edges) and edges[idx][0] == curr_x:
                height = edges[idx][1]
                
                # If 'height' > 0, meaning a building of height 'height'
                # is live, push 'height' to 'live'. 
                # Otherwise, a building of height 'height' is passed, 
                # push the height to 'past'.
                if height > 0:
                    heapq.heappush(live, -height)
                else:
                    heapq.heappush(past, height)
                idx += 1
            
            # While the top height from 'live' equals to that from 'past',
            # Remove top height from both 'live' and 'past'.
            while past and past[0] == live[0]:
                heapq.heappop(live)
                heapq.heappop(past)
            
            # Get the maximum height from 'live'.
            max_height = -live[0] if live else 0
            
            # If the height changes at 'curr_x', we add this
            # skyline key point [curr_x, max_height] to 'answer'.
            if not answer or answer[-1][1] != max_height:
                answer.append([curr_x, max_height])
                
        # Return 'answer' as the skyline.
        return answer   












