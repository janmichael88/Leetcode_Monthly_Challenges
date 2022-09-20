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
        
        return ans