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

