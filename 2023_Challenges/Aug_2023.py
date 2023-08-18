##################################
# 77. Combinations (REVISTED)
# 01AUG23
##################################
#recurse, backtrack with taken array
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        '''
        number of combinations is given as nCk, i.e n choose k
        if we have some indices (i,j,k), any permuation of (i,j,k) is considered the same combination
        i.e combinations are unordered
        backtracking with taken array, and keep track if items taken, 
        when k == 0, we have a pair, then just bactrack
        '''
        ans = []
        taken = [False]*n
        nums = [i for i in range(1,n+1)]
        def backtrack(i,taken):
            if i == n:
                return
            if sum(taken) == k:
                curr_comb = []
                for j in range(n):
                    if taken[j] == True:
                        curr_comb.append(nums[j])
                
                ans.append(curr_comb)
                return
            
            for j in range(i,n):
                if not taken[j]:
                    taken[j] = True
                    backtrack(j,taken)
                    taken[j] = False
        
        
        backtrack(0,taken)
        return ans
    
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        '''
        back track, no taken array, but prune 
        '''
        ans = []
        def backtrack(first_num,path):
            if len(path) == k:
                ans.append(path[:])
                return
            
            nums_needed = k - len(path)
            remaining = n - first_num + 1
            nums_available = remaining - nums_needed
            #If we moved to a child outside of this range, like firstNum + available + 1, then we will run out of numbers to use before reaching a length of k.
            
            for next_num in range(first_num,first_num + nums_available + 1):
                path.append(next_num)
                backtrack(next_num+1,path)
                path.pop()
        
        backtrack(1,[])
        return ans
    
#########################################
# 46. Permutations (REVISITED)
# 02AUG23
########################################
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        '''
        use taken array but go through the nums n times in each loop
        '''
        ans = []
        N = len(nums)
        taken = [False]*N
        
        def backtrack(path,taken,nums):
            if len(path) == N:
                ans.append(path[:])
                return
            
            for i in range(N):
                if not taken[i]:
                    taken[i] = True
                    path.append(nums[i])
                    backtrack(path,taken,nums)
                    taken[i] = False
                    path.pop()
        
        backtrack([],taken,nums)
        return ans
    
##########################################
# 2361. Minimum Costs Using the Train Line
# 02AUG23
##########################################
#for some reason it doesn't working starting from the beginning 
#then stop until we hit the end of the array
class Solution:
    def minimumCosts(self, regular: List[int], express: List[int], expressCost: int) -> List[int]:
        '''
        two routes, of the same n+1 stops
        both arrays at index i describe cost from going i-1 to i using the route
        we can switch between the regular route and the express route at anytime but we pay expressCost
        expressCost represents the cost to transfer from regular to express
        
        no cost to transfer from express back to regular
        no extra cost to stay on expresss rout
        
        return 1-indexed array of length, where costs[i] is the min cost to reach stop i from stop 0
        
        if i'm at stop i with curr_cost i can
            - move to i+1 on regular route with curr_cost + regular[i]
            - move to express: expressCost + curr_cost + express[i]
            - move back to regular from express, which is the same as the first
            
        let dp(i) be the min cost getting to i from starting 0
            need index into routes, and whether i am on regular or not
        '''
        memo = {}
        N = len(regular)
        def dp(i,on_regular):
            if i == N:
                return 0
            if (i,on_regular) in memo:
                return memo[(i,on_regular)]
            #stay on regular and move to next one
            op1 = regular[i] + dp(i+1,on_regular)
            if on_regular:
                op2 = expressCost + express[i] + dp(i+1,False)
            else:
                op2 = express[i] + dp(i+1,False)
            
            ans = min(op1,op2)
            memo[(i,on_regular)] = ans
            return ans
        
        
        dp(0,True)
        print(memo)

class Solution:
    def minimumCosts(self, regular: List[int], express: List[int], expressCost: int) -> List[int]:
        '''
        two cases:
        1. if we are taking regular route, we need to spend regular[i] cost
        2. if we are taking the express route, we need to check what lane we're on
            switchin from reg to express = expressCost + express[i]
            if already on express, we just need to spend express[i]
        
        note; because we can move freely from regular to express, one could always arrive at stop o in express and move for free to regular
        this means that when it comes to reaching stop i, it can never be more expensive to be in the regular lang than to be in the expresss lane
        so the answer for each stop can be represented as the cost in the regular lane
        expressCost >= 1 
        '''
        N = len(regular)
        memo = {}
        
        def dp(i,lane):
            if i < 0:
                return 0
            if (i,lane) in memo:
                return memo[(i,lane)]
            regularLane = regular[i] + dp(i-1,1)
            expressLane = expressCost*(lane == 1) + express[i] + dp(i-1,0)
            ans = min(regularLane,expressLane)
            memo[(i,lane)] = ans
            return ans
        
        dp(N-1,True)
        ans = []
        for i in range(N):
            ans.append(memo[(i,1)])
        return ans
        print(memo)

#bottom up
#not too bad, just watch for the ordering ans additional base case when starting to expressLane first
class Solution:
    def minimumCosts(self, regular: List[int], express: List[int], expressCost: int) -> List[int]:
        '''
        bottom up
        '''
        N = len(regular)
        dp = [[0]*(2) for _ in range(N+1)]
        dp[0][0] = expressCost #cost to start swtiching to express lane incurs expressCost
        
        #base case fill, no need, already zero, start from 1
        for i in range(N):
            for lane in [1,0]: #must be one first then zero
                regularLane = regular[i] + dp[i-1][1]
                expressLane = expressCost*(lane == 1) + express[i] + dp[i-1][0]
                ans = min(regularLane,expressLane)
                dp[i][lane] = ans
        
        ans = []
        for i in range(N):
            ans.append(dp[i][1])
        
        return ans

#################################
# 139. Word Break (REVISITED)
# 04AUG23
################################
#TLE
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        '''
        just dfs for all possible paritions of s and return true if they are in wordsDict
        i need to break the word, that's that kicker
        i can discard each word in word dict recursively and return is its empty
        '''
        word_set = set(wordDict)
        self.got_to_end = False
        used = set()
        N = len(s)
        
        def dfs(i):
            if i == N:
                self.got_to_end = True
                return
            for j in range(i+1,N+1):
                word = s[i:j]
                if word in word_set:
                    used.add(word)
                    dfs(j)
        
        dfs(0)
        return self.got_to_end and len(used) >= 1
        
#make it dp
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        '''
        just dfs for all possible paritions of s and return true if they are in wordsDict
        i need to break the word, that's that kicker
        i can discard each word in word dict recursively and return is its empty
        
        need to check all and only advance when we can make a word
        '''
        word_set = set(wordDict)
        N = len(s)
        memo = {}
        
        def dp(i):
            if i == N:
                return True
            
            if i in memo:
                return memo[i]
            for j in range(i+1,N+1):
                word = s[i:j]
                if word in word_set and dp(j):
                    memo[i] = True
                    return True
            
            memo[i] = False
            return False
        
        return dp(0)
        
#bottom up
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        '''
        bottom up
        
        '''        
        word_set = set(wordDict)
        N = len(s)
        dp = [False]*(N+1)
        dp[N] = True
        
        for i in range(N-1,-1,-1):
            for j in range(i+1,N+1):
                word = s[i:j]
                if word in word_set and dp[j]:
                    dp[i] = True
                    break #early optimization
                    
        return dp[0]

#we can optimize the search for a word usint a Trie insteaf of checking all prefixes,
#we just check if it exits in the tree and that it is a word
class Node:
    def __init__(self,):
        self.isWord = False
        self.children = defaultdict()
        
class Trie:
    def __init__(self,):
        self.root = Node()
        self.children = self.root.children
        
    def addWord(self,word):
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = Node()
            curr = curr.children[ch]
        
        curr.isWord = True

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        '''
        bottom up
        
        '''
        word_Trie = Trie()
        for word in wordDict:
            word_Trie.addWord(word)
            
        N = len(s)
        dp = [False]*(N+1)
        dp[N] = True
        
        for i in range(N-1,-1,-1):
            curr = word_Trie
            for j in range(i+1,N+1):
                ch = s[j-1]
                if ch not in curr.children:
                    break
                curr = curr.children[ch]
                if curr.isWord and dp[j]:
                    dp[i] = True
                    break #stop search

                    
        return dp[0]
    
#there is also a BFS approach
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        '''
        we can also use BFS, if we imagine the indices of the words as states
        if we are at index i, it implies we have already found s[:i+1] up this point
        we just need to connect i+1 to an end, for all possible ends
        all possible ends would be [i+1,N+1]
        
        if we have gotten to the end of the word, return True
        otherwise, there isn't a path, return False
        '''
        wordDict = set(wordDict)
        seen = set()
        q = deque([0])
        N = len(s)
        
        while q:
            start = q.popleft()
            if start == N:
                return True
            
                
            for end in range(start+1,N+1):
                if s[start:end] in wordDict and end not in seen:
                    seen.add(end)
                    q.append(end)
        
        return False

####################################
# 95. Unique Binary Search Trees II
# 05AUG23
###################################
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        '''
        if we pick an i, all [1 to i-1] are on the left and all[i+1,n+1] are on the right
        need to keep pointers to start and end
        let dp(left,right) return a list of all unique BST
        then we split for all possible left and right
        '''
        memo = {}
        
        def dp(left,right):
            if left > right:
                return [None]
            if (left,right) in memo:
                return memo[(left,right)]
            
            res = []
            for i in range(left,right+1):
                left_trees = dp(left,i-1)
                right_trees = dp(i+1,right)
                
                for l_tree in left_trees:
                    for r_tree in right_trees:
                        curr_tree = TreeNode(i,l_tree,r_tree)
                        res.append(curr_tree)
            
            memo[(left,right)] = res
            return res
        
        
        return dp(1,n)

###################################
# 920. Number of Music Playlists
# 06AUG23
###################################
#fmlllll
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        '''
        we have n songs, we want to listen to goal songs
        constraints:
            every song is played at least once
            a song can only be played again if k other songs have been played, what do they mean k other songs?
            
        if goal == n, then no matter k is, the answer is just
            goalChooseN i.e the binomial coefficient
        
        norie how k < n <= goal
        menaing k never be n or goal
        
        keep track of number of songs played and also the current count of sounds played
        use set to keep track of player songs count
        '''
        memo = {}
        mod = 10**9 + 7
        curr_played = [False]*n
        
        def convert(arr):
            ans = ""
            for b in arr:
                if b == True:
                    ans += '1'
                else:
                    ans += '0'
            
            return ans
        
        def dp(count,curr_played,k):
            if count == goal:
                return 1
            
            if (count,convert(curr_played)) in memo:
                return memo[(count,convert(curr_played))]
            
            ans = 0
            for i in range(n):
                if curr_played[i] == False:
                    curr_played[i] = True
                    ans += dp(count+1,curr_played,k)
                    ans %= mod
                    
                if sum(curr_played) > k:
                    ans += dp(count+1,curr_played,k)
                    ans %= mod
                    

            
            ans %= mod
            memo[(count,convert(curr_played))] = ans
            return ans
        
        return dp(0,curr_played,k)
                    
#push dp, but multiply to get the count
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        '''
        welp, i didn't get this sone
        if we define dp(i,j) as the number of possible playlists of length i contains exactly j unique songs
        then we can solve dp(goal,n)
        if (i,j) == (0,0)
            this represents exactly 1 valid way
        
        for all i < j:
            dp(i,j) = 0, because we can make a playlist with length i, if i < j, we simply don't have enough songs
            
        
        transition rules:
            if we add a song we haven't added to the playlist, the length increases by 1
            and the number of unique songs to chose from decreses
            therefore a playlist of length  i with j unique songs can be formed by adding to each preivous state (i-1,j-1)
            but for here, how many songs can we choose from?
            at this point there are j-1 unique songs in the playlistn and there are n songs in total 
            which means we can chose from n - (j-1) = n -j + 1 songs to choose from
            but we can do this for each playlist, or rather we do this for the number of playlists given in state9 (i-1,j-1)
        
        first transition
            dp(i,j) = dp(i-1,j-1)*(n-j+1)
            
        now how about old songs?
            if we replay an old song, the list increases by 1 (from i-1 to i), but the number of unique songs remains the same, still k
            therfore the number of playlists if length i with j unique songs can be icnrease by replaying an old song in every playlist 
            from dp(i-1)
        
        if we have more j unique songs than k, we can play and extra song, but will having j unique songs
        then we can increment by dp(i-1,j)*(j-k) songs
        i.e for each of the playlists in dp(i-1,j), we can add any (j-k) songs
        '''
        mod = 10**9 + 7
        memo = {}
        
        def dp(i,j):
            if i == 0 and j == 0:
                return 1
            if i == 0 or j == 0:
                return 0
            
            if (i,j) in memo:
                return memo[(i,j)]
            ans = 0
            ans += dp(i-1,j-1)*(n-j+1)
            ans %= mod
            
            #if we have extra
            if j > k:
                ans += dp(i-1,j)*(j-k)
                ans %= mod
            
            memo[(i,j)] = ans
            return ans
        
        return dp(goal,n) % mod
    
#bottom up
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        '''
        bottom up
        '''
        mod = 1_000_000_007
        dp = [[0]*(n+1) for _ in range(goal+1)]
        
        #base case fill
        dp[0][0] = 1
        
        #start 1 away from bounary
        for i in range(1,n+1):
            for j in range(1,n+1):

                ans = 0
                ans += dp[i-1][j-1]*(n-j+1)
                ans %= mod
            
                #if we have extra
                if j > k:
                    ans += dp[i-1][j]*(j-k)
                    ans %= mod
                    
                dp[i][j] = ans

        
        return dp[goal][n] % mod
    
#just another way
#https://leetcode.com/problems/number-of-music-playlists/discuss/1879256/PYTHON-SOLUTION-oror-RECURSION-%2B-MEMO-oror-WELL-EXPLAINED-oror-HOW-TO-REACH-EXPLAINED-oror
'''
Understand some basics of the question
    1. We have to use all of the n songs in our playlist
    2. We can repeat a song only after k other songs
    
	Now for first point : We have to make sure that we will use all the n  songs
	To make this happend I have made a check whenever we are going to repeat a song
	
	reapeat only if the no. of songs left to complete playlist > no. unique of songs left to add to playlist
	
	For the seecond point : We have this above condition + we can only repeat if our  current size of playlist is greater than or equals to k + 1
	
	You can use recursion and keep tracking of actual playlist bit it will give TLE
	
	To remove this TLE we need to understand combinations
	
	For adding a new element: 
	        lets n = 5 goal  = 7 k = 2
			our playlist = [ 1 , 2 ]
			In how many ways can we add new item ?? ans is the no. of unique item left
			Exactly !! So we do not need to check for all of them one by one but we do check for one and multilply by the no. of unique items left
			
			
	For repeating a element
	    lets  n = 5 goal = 7 k = 2
		our playlist = [1 , 2 ,  3  , 4 , __ ]
		now we can repeat 1 or 2 at the blank position 
		So is there any mathematical way to find the no. of ways of repeatition too ??
		Yes , there is (I got this from my observation )
		
		See there are exactly 4 unique items in our playlist
		We know that k = 2 means we need atleast a gap of 2
		So we leave 2 unique items from the k=last( they can never be same and will always be unique as we have a gap of k for repeating )
		So yes we can repeat 2 items i.e. unique - k precisely
		
		if you say for n = 5 goal = 10 k  = 4
		our playlist = [ 1 , 2 ,3 , 4 , 5 , 1 , 2 , 3 , __ ]
		See the last 4 (k) items are unique and if we remove 4 from no. of unique (5)
		i.e 5-4 = 1 we know that we can repeat exactly 1 item i,.e 4
		
		So for any point find no. of ways we insert by repeating and by non repeating
		Find ans for both of them and boom question solved !!!!!
		
		In dp we store our key as ( unique , size ) and value as ans for the key
		

'''
class Solution:
    def numMusicPlaylists(self, n: int, goal: int, k: int) -> int:
        # size means current size without item we are going to add
        # unique is the count of unique items
        def solve(size , unique):
            if size == goal : return 1
            if (size,unique) in dp:return dp[(size,unique)]
            ans = 0
            repeat = 0
            # check if we can repeat
            if goal - size > n - unique and size >= k+1:repeat = unique - k
            non_repeat = n - unique
            if repeat > 0 :ans += solve(size+1,unique) * repeat
            ans += solve(size+1,unique+1) * non_repeat
            dp[(size,unique)] = ans
            return ans
        dp = {}
        return solve(0,0)%1000000007
    
##############################
# 2266. Count Number of Texts
# 03AUG23
##############################
#YESSS
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        '''
        same phone mapping, but to get the ith character, alice has to press that key i times
        no zero or one key used
        given a string of key pressings return the total number of possible text messages
        i.e, the number of valid ways to decode the string
        similar to decode ways
        
        if we are given a substring 222
        i can use 'aaa', 'ab', 'ba','c' = 4 ways
        now say if we have 222333. this is just 4*4 = 16 ways
        223: aad, bd
        single digits can only decode to one way
        
        let dp(i) be the number of ways to decode the string using string[:i]
        if im at i, then i need to find the number of ways to decode at i*dp(i+1)
        
        for subtrings of the same digit, let digit be d:
            d = 1
            dd = 2
            ddd = 4
            dddd = 8 #for keys 7 and 9 only
            
        need to reduce to 3 or 4
        '''
        #num ways to decode
        mapp = {'2':'abc',
                '3':'def',
                '4':'ghi',
                '5':'jkl',
                '6':'mno',
                '7':'pqrs',
                '8':'tuv',
                '9':'wxyz',
               }
        
        size_mapp = {1:1,
                    2:2,
                    3:4,
                    4:8}
        
        mod = 10**9 + 7
        
        
        threes = '234568'
        fours = '79'
        
        memo = {}
        N = len(pressedKeys)
        
        def dp(i):
            if i == N:
                return 1
            if i in memo:
                return memo[i]
            
            ans = 0
            if pressedKeys[i] in threes:
                j = i
                while j < N and pressedKeys[j] == pressedKeys[i] and j - i < 3:
                    #i need to accumlate the answers, its the sum of all j in the current substring that im on for the current subtree
                    j += 1
                    ans += dp(j) % mod
            
            else:
                j = i
                while j < N and pressedKeys[j] == pressedKeys[i] and j - i < 4:
                    j += 1
                
                    ans += dp(j) % mod
            
            memo[i] = ans
            return ans % mod
            
        return dp(0) % mod
    
#bottom up
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        '''
        bottom up
        '''
        
        mod = 1_000_000_007
        threes = '234568'
        fours = '79'
        
        N = len(pressedKeys)
        
        dp = [0]*(N+1)
        dp[N] = 1
        
        for i in range(N-1,-1,-1):
            ans = 0
            if pressedKeys[i] in threes:
                j = i
                while j < N and pressedKeys[j] == pressedKeys[i] and j - i < 3:
                    #i need to accumlate the answers, its the sum of all j in the current substring that im on
                    j += 1
                    ans += dp[j] % mod
            
            else:
                j = i
                while j < N and pressedKeys[j] == pressedKeys[i] and j - i < 4:
                    j += 1
                
                    ans += dp[j] % mod
            
            dp[i] = ans % mod

            
        return dp[0] % mod

#####################################
# 74. Search a 2D Matrix (REVISTED)
# 07AUG23
#####################################
#binary search by row is easy
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        '''
        brute force is just to search
        next best is binary search each row
        '''
        
        def bin_search(row,target):
            left = 0
            right = len(row)
            while left < right:
                mid = left + (right - left) // 2
                if row[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            
            return left
        
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        for row in matrix:
            idx = bin_search(row,target)
            print(idx)
            if idx <= cols - 1:
                if row[idx] == target:
                    return True
                if row[idx-1] == target:
                    return True
            if idx == cols and row[idx-1] == target:
                return True
        
        return False
                    
#convert m*n indices to cells => (i,j)
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        '''
        notice that if we concated the elements into a long array, we could jsut do binary search
        we just need to convert an index to an (i,j) point
        given an index idx
        row = idx // cols
        col = idx % cols
        '''
        
        rows = len(matrix)
        cols = len(matrix[0])
        
        if rows == 0:
            return False
        left = 0
        right = rows*cols - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            i = mid // cols
            j = mid % cols
            
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                right = mid - 1
            else:
                left = mid + 1
        
        return False
    

####################################
# 625. Minimum Factorization
# 07AUG23
####################################
class Solution:
    def smallestFactorization(self, num: int) -> int:
        '''
        find smallest positive integer x whose multilcation of each digit == num
        if i first find the factorization of num, 
        then just find the smallest number i can create using the prime factors
        48
        6*8  = 68
        2*3*4*2  = 2342
        2*3*2*2*2 = 23222
        of thes 48 is the smallest
        
        then it makes sens to use the largest numbers first
        
        just greedily divide by numbers 9 to 1
        '''
        if num < 2:
            return num
        
        res = 0
        mult = 1
        for i in range(9,1,-1):
            while (num % i == 0):
                num //= i
                res = mult*i + res
                mult  *= 10
        
        return res if num == 1 and res <= 2**31 - 1 else 0
    
################################################
# 2616. Minimize the Maximum Difference of Pairs
# 09AUG23
################################################
#FML
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        '''
        find p pairs of indices such that tht maximum difference among all pairs is minimzed
        is the absolute difference
        return minimum maximum difference among all pairs
        
        brute force would be to examine all (i,j) pairings, sort the pairings increasinly by their abs(nums[i] - nums[j]), grab the p smallest,
        making sure that the group i grab doesnt have intersecting indices
        
        hint says dp
        sort increasinly
        
        if i let dp(i,count_pair) be the answer to he min maximum difference
            then i can either not make a pair here in which case i move i+1
            or i make a pair, and since i sorted the next smallest min would just be nums[i+1] - nums[i]
            
        '''
        nums.sort()
        memo = {}
        N = len(nums)
        
        def dp(i,p):
            #since p if smaller than len(nums), we always hit the base case first
            #no pairs left, return a small number to max
            if p == 0:
                return 0
            #return a larg number in order to minimize
            if i >= N - 1:
                return float('inf')
            if (i,p) in memo:
                return memo[(i,p)]
            
            #dont make a pair
            no_pair = dp(i+1,p)
            #try making pair with next greater minimum and find the max for the p-1 pairs
            max_pair = max(nums[i+1] - nums[i],dp(i+2,p-1))
            #minimuze
            ans = min(no_pair,max_pair)
            memo[(i,p)] = ans
            return ans
        
        return dp(0,p)
    
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        '''
        this is binary search on workable solution paradigm, the crux of the problem is trying to figure what a workable solution is
        if i have n nums, how many pairs can i possibly make?
        i can make at most n//2 pairs
        
        when we select p pairs, the we want to minmize the maximum different among all the pairs
        say we pick a candidate maximum, call it curr_max
        we can sort the array and greedily try to build up pairs such that each pair isnt more than the curr_max
        now say we have k pairs with this curr_max as the upper bound
        if k >= p, then it means we can use this as the upper bound, so find a smaller one
        else curr_max += 1
        
        
        we can us dp to find such a workable solution given a treshhold
        what are the bounds
        well the highest could be max(nums) - min(nums)
        lowest could be 0
        '''
        
        nums.sort()
        N = len(nums)
        memo = {}
        
        #function to count pairs that are less than curr_max
        def dp(i,curr_max,memo):
            if i >= N-1:
                return 0
            if i in memo:
                return memo[i]
            ans = 0
            if nums[i+1] - nums[i] <= curr_max:
                ans = 1 + dp(i+2,curr_max,memo)
            else:
                ans = dp(i+1,curr_max,memo)
            
            memo[i] = ans
            return ans
        
        lo = 0
        hi = nums[-1] - nums[0]
        
        while lo < hi:
            curr_max = lo + (hi - lo) // 2
            #get candidate count
            memo = {}
            curr_count = dp(0,curr_max,memo)
            if curr_count >= p:
                hi = curr_max
            else:
                lo = curr_max + 1
        
        return lo
    
###################################
# 271. Encode and Decode Strings
# 09AUG23
####################################
#escaping
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        '''
        we use /: as our delimiter and / as our escapre character
        iterate over each string in the list and for each string is a character is a /, we another / to escape it
        if the character is not a slash we do nothing
        escaping means to escape the delimiter!
        exmple
        Wor/:ld, uses a delimter we can escape it if we we
        Wor//:ld, in which cas the second the /: is part of the string
        decode:
            is the current character is the escape character, we check the one next to it, and if its our delimeter we know we need to split
            otherwise its part of the actual string
        '''
        enc_str = ""
        for s in strs:
            #replace slash with // followed by delimter
            temp = s.replace('/','//') + '/:'
            enc_str += temp
        
        return enc_str
            

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        res = []
        curr_string = ""
        i = 0
        while i < len(s):
            if s[i:i+2] == '/:':
                res.append(curr_string)
                curr_string = ""
                i += 2
            elif s[i:i+2] == '//':
                #first / is part
                curr_string += '/'
                i += 2
            else:
                curr_string += s[i]
                i += 1
        
        return res
        


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))

#Chunked Transfer Encoding
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        '''
        we can use chunked trasfer encoding
        for each string in strs, we store len(str) + '/:' + str
        then we keep reading until we hit '/:'
        the number before '/:' is the amount we need to read
        '''
        enc_str = ""
        for s in strs:
            temp = str(len(s))+'/:'+s
            enc_str += temp
        
        return enc_str
        

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings.
        """
        res = []
        i = 0
        while i < len(s):
            #find first deleimier after current index i
            delim_idx = s.find('/:',i)
            #get length to read
            size = int(s[i:delim_idx])
            #get the actual string 
            curr_string = s[delim_idx+2:delim_idx + 2 + size]
            res.append(curr_string)
            i = delim_idx + 2 + size
        
        return res
        


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))

#another way
class Codec:
    def encode(self, strs: List[str]) -> str:
        # encode as: string length+string body. if you need more, use 4 bytes
        return ''.join(chr(len(s)) + s for s in strs)
    def decode(self, s: str) -> List[str]:
        i = 0
        res = []
        while i < len(s):
            res.append(s[i + 1:i + ord(s[i]) + 1])
            i += 1 + ord(s[i])
        return res

###################################################
# 81. Search in Rotated Sorted Array II (REVISTED)
# 10AUG23
###################################################
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        '''
        the other way is to linearly scan to find the pivot, we cannot simply use binary search to exclude side
        then binary search on the correct side
        '''
        N = len(nums)
        #look for pivot
        pivot = -1
        for i in range(N-1):
            if nums[i] > nums[i+1]:
                pivot = i
                break
            if nums[i] == target:
                return True
        
        
        #now apply binary search on the correct side
        def binarySearch(left_bound,right_bound,target):
            found = False
            
            while left_bound <= right_bound:
                mid = left_bound + (right_bound - left_bound) // 2
                if nums[mid] == target:
                    found = True
                if nums[mid] >= target:
                    right_bound = mid - 1
                else:
                    left_bound = mid + 1
                
            
            return found
        
        
        left_side = binarySearch(0,pivot,target)
        right_side = binarySearch(pivot+1, N-1,target)
        
        if left_side or right_side:
            return True
        return False
    
#####################################
# 638. Shopping Offers
# 10AUG23
#####################################
#TLE!
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        '''
        price[i] gives price for the ith item
        needs[i] gives units neede for ith item
        special[i][j] = number of pieces of the jth item in the ith offer 
        
        given an offer i special[i] is of length n + 1
        where i can pay special[i][-1] for special[i][j] units of the jth item, read the explanation for more on this
        
        return lowest price i have to pay to get exactly needs unies of each item
        we are not allowed to buy more items than we want, even if that would lower the overall price
        
        we can define a state, call it X, an array is size n, n == len(price)
        in X we have some counts of n items (a,b,c,d)
        
        call this state X optimal, meaning it contains the lowest prices with item counts (a,b,c,d) and with indicies (i,j,k,l)
        now we can move from this state for any number
        say for a, we can go to a+1 going up in prices[i]
        
        first 
            we can try ill going up by count 1 with indices (i,j,k,l)
                min(choices using prices up by 1), and go up in prices[(for all indicies in (i,j,k,l))]
            or we can try all speical offers
                min(prices and we ge up) by the last number of the offeres array
                then we just take the min
            
            base case,
                if any of the counts goes over the needs, return float('inf'), because we need to maximuze
            
            if the current state X == needs,
                return 0

        '''
        
        memo = {}
        n = len(price)
        starting = [0]*(n)
        
        def dp(state):
            if state == needs:
                return 0
            if any([a > b for a,b in zip(state,needs)]):
                return float('inf')
            
            if tuple(state) in memo:
                return memo[tuple(state)]
            
            #minimize by taking only a single item at a time
            ans = float('inf')
            for i in range(n):
                next_state = state[:]
                next_state[i] += 1
                ans = min(ans, price[i] + dp(next_state))
            
            #or we try using the offers array[][]
            for off in special:
                next_state = state[:]
                for i in range(n):
                    next_state[i] += off[i] #add counts
                
                ans = min(ans, off[-1] + dp(next_state))
            
            memo[tuple(state)] = ans
            return ans
        
        
        return dp(starting)
            
#imporvemnts, start from needs then check if we go to zeros
#also add pruning, but still TLE
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        '''
        but still TLE, getting there
        '''
        
        memo = {}
        n = len(price)
        ending = [0]*(n)
        
        def dp(state):
            if state == ending:
                return 0
            if tuple(state) in memo:
                return memo[tuple(state)]
            
            #minimize by taking only a single item at a time
            ans = float('inf')
            for i in range(n):
                next_state = state[:]
                if next_state[i] > 0:
                    next_state[i] -= 1
                    ans = min(ans, price[i] + dp(next_state))
            
            #or we try using the offers array[][]
            for off in special:
                next_state = state[:]
                broken = False
                for i in range(n):
                    if next_state[i] >= off[i]:
                        next_state[i] -= off[i] #add counts
                    else:
                        broken = True
                        break
                if not broken:
                    ans = min(ans, off[-1] + dp(next_state))
            
            memo[tuple(state)] = ans
            return ans
        
        
        return dp(needs)

#instead of increamting by 1 and price[i] each time, just buy them all for the given state!
class Solution:
    def shoppingOffers(self, price: List[int], special: List[List[int]], needs: List[int]) -> int:
        '''
        but still TLE, getting there
        '''
        
        memo = {}
        n = len(price)
        ending = [0]*(n)
        
        def dp(state):
            if state == ending:
                return 0
            if tuple(state) in memo:
                return memo[tuple(state)]
            
            ans = sum([p*c for p,c in zip(price,state)])
            '''
            for i in range(n):
                next_state = state[:]
                if next_state[i] > 0:
                    next_state[i] -= 1
                    ans = min(ans, price[i] + dp(next_state))
            '''
            #or we try using the offers array[][]
            for off in special:
                next_state = state[:]
                broken = False
                for i in range(n):
                    if next_state[i] >= off[i]:
                        next_state[i] -= off[i] #add counts
                    else:
                        broken = True
                        break
                if not broken:
                    ans = min(ans, off[-1] + dp(next_state))
            
            memo[tuple(state)] = ans
            return ans
        
        
        return dp(needs)
    
######################################
# 518. Coin Change II (REVISTED)
# 11AUG23
######################################
#one state with curr sum is no good
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        '''
        dp with in state, just the sum, why doesn't it work thoguh
        need to keep trak if the ith coun that im on as well as the curr sum
        '''
        memo = {}
        def dp(curr_sum):
            if curr_sum < 0:
                return 0
            if curr_sum == 0:
                return 1
            if (curr_sum) in memo:
                return memo[curr_sum]
            
            ans = 0
            for c in coins:
                ans += dp(curr_sum - c)
            
            memo[curr_sum] = ans
            return ans
        
        dp(amount)
        print(memo)
        return 0
    
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        '''
        dp with in state, just the sum, why doesn't it work thoguh
        need to keep trak if the ith coun that im on as well as the curr sum
        
        keep track if the ith coun and the curramount
        then we take or dont take and add them both
        '''
        memo = {}
        N = len(coins)
        
        def dp(i,curr_sum):
            if i == N:
                return 0
            if curr_sum == 0:
                return 1
            if (i,curr_sum) in memo:
                return memo[(i,curr_sum)]
            
            if coins[i] > curr_sum:
                no_take = dp(i+1,curr_sum)
                memo[(i,curr_sum)] = no_take
                return no_take
            else:
            
                take = dp(i,curr_sum - coins[i]) + dp(i+1,curr_sum)
                memo[(i,curr_sum)] = take
                return take
        
        
        return dp(0,amount)
    
#bottom up
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        '''
        dp with in state, just the sum, why doesn't it work thoguh
        need to keep trak if the ith coun that im on as well as the curr sum
        
        keep track if the ith coun and the curramount
        then we take or dont take and add them both
        '''
        
        N = len(coins)
        dp = [[0]*(amount+1) for _ in range(N+1)]
        #base case fill
        for i in range(N+1):
            for curr_sum in range(amount+1):
                if curr_sum == 0:
                    dp[i][curr_sum] = 1
        
        
        #start one away from base case
        for i in range(N-1,-1,-1):
            for curr_sum in range(1,amount+1):
                if coins[i] > curr_sum:
                    no_take = dp[i+1][curr_sum]
                    dp[i][curr_sum] = no_take
                else:
                    take = dp[i][curr_sum - coins[i]] + dp[i+1][curr_sum]
                    dp[i][curr_sum] = take
        
        
        
        return dp[0][amount]
    
#space save, we only look back at the previous i
#and we only need to check for the next valid coins, this is just a tiny prune i feel
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        '''
        reduce to 1 dp
        '''
        
        N = len(coins)
        dp = [0]*(amount+1)
        #base case fill
        dp[0] = 1
        
        
        #start one away from base case
        for i in range(N-1,-1,-1):
            for curr_sum in range(coins[i],amount+1):
                    take = dp[curr_sum - coins[i]]
                    dp[curr_sum] += take
        
        
        
        return dp[amount]


#########################################
# 63. Unique Paths II (REVISTED)
# 12AUG23
##########################################
#top down
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        '''
        this is just dp
        let dp(i,j) be the number of unique paths starting at (i,j) and getting to to (final_row,fina_col)
        base is case is when:
            (i,j) == (final_row,final_col)
            return 1
        
        out of bounds
            return 0
        
        then for neigh_x,neigh_y in neighbors of (i,j)
            if we can advcane to that neigh and its not an obstacle
            dp(i,j) += dp(neigh_x,neigh_y) for all nieghborng pairs
            and it can only go down and to the right
        '''
        
        rows = len(obstacleGrid)
        cols = len(obstacleGrid[0])
        
        
        #fucking corner case
        if obstacleGrid[rows-1][cols-1] == 1:
            return 0
        memo = {}
        
        
        def dp(i,j):
            if (i,j) == (rows-1,cols-1):
                return 1
            if (i < 0) or (i >= rows) or (j < 0) or (j >= cols):
                return 0
            
            if (i,j) in memo:
                return memo[(i,j)]
            
            if obstacleGrid[i][j] == 1:
                return 0
            
            ans = dp(i+1,j) + dp(i,j+1)
            memo[(i,j)] = ans
            return ans
        
        return dp(0,0)
    
#bottom up
#need to be carefule when adding into cells and for boundary conditions, otherwise we overwrite
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        '''
        this is just dp
        let dp(i,j) be the number of unique paths starting at (i,j) and getting to to (final_row,fina_col)
        base is case is when:
            (i,j) == (final_row,final_col)
            return 1
        
        out of bounds
            return 0
        
        then for neigh_x,neigh_y in neighbors of (i,j)
            if we can advcane to that neigh and its not an obstacle
            dp(i,j) += dp(neigh_x,neigh_y) for all nieghborng pairs
            and it can only go down and to the right
        '''
        
        rows = len(obstacleGrid)
        cols = len(obstacleGrid[0])
        
        
        #fucking corner case
        if obstacleGrid[rows-1][cols-1] == 1:
            return 0
        
        dp = [[0]*(cols+1) for _ in range(rows+1)]
        
        #base case fill
        dp[rows-1][cols-1] = 1
        #then fill for the first row and first col
        
        
        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                if obstacleGrid[i][j] == 1:
                    continue
                
                if i + 1 >= 0:
                    dp[i][j] += dp[i+1][j]
                
                if j + 1 >= 0:
                    dp[i][j] += dp[i][j+1]
        
        return dp[0][0]

#########################################################
# 2369. Check if There is a Valid Partition For The Array
# 13AUG23
#########################################################
class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        '''
        need to parition the array into one or more contiguous subarrays
        a valid array is, on that meets the requirments
            1. exactly two elemetns
            2. exactly 3 elements
            3. consecutive increasing by diff 1
        
        cant just linearly scan the array and greedily paritions when we find valid subarray
        the contig arrays can only be of length 2 or length 3
        so we can check from an index i for the conditions
        if any of these are true, we can make a valid partition
        
        i just don't know what to get when i get to the end of the array?
        return true
        '''
        N = len(nums)
        memo = {}
        
        def dp(i):
            if i == N:
                return True
            if i in memo:
                return memo[i]
            
            #two equal elements
            if i + 1 < N and nums[i] == nums[i+1]:
                #must be true here and for the other part
                ans = True or dp(i+2)
                memo[i] = ans
                return ans
            #three of a kind
            if i + 2 < N and nums[i] == nums[i+1] == nums[i+2]:
                ans = True or dp(i+3)
                memo[i] = ans
                return ans
            
            #diff 1
            if (i + 2 < N) and (nums[i+2] - nums[i+1] == 1) and (nums[i+1] - nums[i] == 1):
                ans = True or dp(i+3)
                memo[i] = ans
                return ans
            
            else:
                memo[i] = False
                return False
            
        
        return dp(0)


class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        '''
        need to parition the array into one or more contiguous subarrays
        a valid array is, on that meets the requirments
            1. exactly two elemetns
            2. exactly 3 elements
            3. consecutive increasing by diff 1
        
        cant just linearly scan the array and greedily paritions when we find valid subarray
        the contig arrays can only be of length 2 or length 3
        so we can check from an index i for the conditions
        if any of these are true, we can make a valid partition
        
        i just don't know what to get when i get to the end of the array?
        return true
        '''
        N = len(nums)
        memo = {}
        
        def dp(i):
            if i == N:
                return True
            if i in memo:
                return memo[i]
            
            ans = False
            #dont immediaatley return from these staments, but add to its possibiltiy
            #two equal elements
            if i + 1 < N and nums[i] == nums[i+1]:
                #must be true here and for the other part
                ans = ans or dp(i+2)
            #three of a kind
            if i + 2 < N and nums[i] == nums[i+1] == nums[i+2]:
                ans = ans or dp(i+3)
            
            #diff 1
            if (i + 2 < N) and (nums[i+2] - nums[i+1] == 1) and (nums[i+1] - nums[i] == 1):
                ans = ans or dp(i+3)
            
            memo[i] = ans
            return ans
            
        
        return dp(0)

#bottom up, fuck yeahhh!
class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        '''
        need to parition the array into one or more contiguous subarrays
        a valid array is, on that meets the requirments
            1. exactly two elemetns
            2. exactly 3 elements
            3. consecutive increasing by diff 1
        
        cant just linearly scan the array and greedily paritions when we find valid subarray
        the contig arrays can only be of length 2 or length 3
        so we can check from an index i for the conditions
        if any of these are true, we can make a valid partition
        
        i just don't know what to get when i get to the end of the array?
        return true
        '''
        N = len(nums)
        dp = [False]*(N+1)
        
        dp[N] = True
        
        for i in range(N-1,-1,-1):
            ans = False
            #two equal elements
            if i + 1 < N and nums[i] == nums[i+1]:
                #must be true here and for the other part
                ans = ans or dp[i+2]
            #three of a kind
            if i + 2 < N and nums[i] == nums[i+1] == nums[i+2]:
                ans = ans or dp[i+3]
            
            #diff 1
            if (i + 2 < N) and (nums[i+2] - nums[i+1] == 1) and (nums[i+1] - nums[i] == 1):
                ans = ans or dp[i+3]
            
            dp[i] = ans
        
        return dp[0]
    
#optimzed, only need to keep three entries in the array
class Solution:
    def validPartition(self, nums: List[int]) -> bool:
        '''
        we can optimzied space
        its called rolling index, use modular 3
        '''
        N = len(nums)
        dp = [False]*(3)
        
        dp[-1] = True
        
        for i in range(N-1,-1,-1):
            ans = False
            #two equal elements
            if i + 1 < N and nums[i] == nums[i+1]:
                #must be true here and for the other part
                ans = ans or dp[(i+2) % 3]
            #three of a kind
            if i + 2 < N and nums[i] == nums[i+1] == nums[i+2]:
                ans = ans or dp[(i+3) % 3]
            
            #diff 1
            if (i + 2 < N) and (nums[i+2] - nums[i+1] == 1) and (nums[i+1] - nums[i] == 1):
                ans = ans or dp[(i+3) % 3]
            
            dp[i % 3] = ans
        
        return dp[0]

#######################################
# 646. Maximum Length of Pair Chain
# 13AUG23
#######################################
#eeeh at least at works
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        '''
        for p in pairs, it will always be the case taht p[0] < p[1]
        this is similart to LIS
        i could sort on start, then just keep track of i and the current end of the chain
        
        '''
        N = len(pairs)
        pairs.sort()
        memo = {}
        
        def dp(i,curr_end):
            if i == N:
                return 0
            if (i,curr_end) in memo:
                return memo[(i,curr_end)]
            
            #we can extend from this i
            extend = 0
            if pairs[i][0] > curr_end:
                extend = 1 + dp(i+1,pairs[i][1])
            
            no_extend = dp(i+1,curr_end)
            ans = max(extend,no_extend)
            memo[(i,curr_end)] = ans
            return ans
        
        return dp(0,float('-inf'))
    
#we can speed up using binary search for the next greater start from i
#then we move up by that distance?
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        '''
        for p in pairs, it will always be the case taht p[0] < p[1]
        this is similart to LIS
        i could sort on start, then just keep track of i and the current end of the chain
        
        '''
        N = len(pairs)
        pairs.sort()
        memo = {}
        
        def binary_search(start,pairs,target):
            left = start + 1
            right = len(pairs) - 1
            
            while left <= right:
                mid = left + (right - left) // 2
                if pairs[mid][0] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            
            return left
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            
            #we can extend from this i
            next_index = binary_search(i,pairs,pairs[i][1])
            extend = 1 + dp(next_index)
            no_extend = dp(i+1)
            ans = max(extend,no_extend)
            memo[i] = ans
            return ans
        
        return dp(0)
    
#using for loop inside to find the next greater end
#need to do dp for all i
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        '''
        for p in pairs, it will always be the case taht p[0] < p[1]
        this is similart to LIS
        i could sort on start, then just keep track of i and the current end of the chain
        
        '''
        N = len(pairs)
        pairs.sort()
        memo = {}
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            
            #we can extend from this i
            ans = 1
            for j in range(i+1,N):
                if pairs[j][0] > pairs[i][1]:
                    ans = max(ans, 1 + dp(j))
            memo[i] = max(ans,dp(i+1))
            return ans
        
        
        ans = 0
        for i in range(N):
            ans = max(ans,dp(i))
        
        return ans
    
#################################################
# 215. Kth Largest Element in an Array (REVISTED)
# 14AUG23
#################################################
#quick select, make sure to remember this implementation, 
#its with lists insteaf of playing around with indices
#careful when framing the questn kth smalest or kth largest
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''
        quick select
        partition into three arrays, then compre with left and right
        start by picking ranom index
        '''
        def quick_select(nums,k):
            pivot = random.choice(nums)
            left, right,mid = [],[],[]
            
            for num in nums:
                if num > pivot:
                    right.append(num)
                elif num < pivot:
                    left.append(num)
                else:
                    mid.append(num)
                    
            if k <= len(left):
                return quick_select(left,k)
            
            if len(left) + len(mid) < k:
                return quick_select(right,k-len(left) - len(mid))
            
            return pivot
        
        
        return quick_select(nums,len(nums) - k + 1)
    
#inplace
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''
        do quick select in place
        kth largest means its position in sorted array would be 
        nums[len(nums) - k + 1]
        for problems like these, ask what would k be in an ascending array
        '''
        N = len(nums)
        
        def quick_select(left,right,k,nums):
            pivot = left + (right - left) // 2 #could also do random
            #move pivot to the end
            nums[pivot],nums[right] = nums[right],nums[pivot]
            store_idx = left
            for i in range(left,right):
                #verything greater than the pivot is to the left
                if nums[i] > nums[right]:
                    continue
                #swap
                nums[store_idx],nums[i] = nums[i], nums[store_idx]
                store_idx += 1
            
            #put pivot back at its place
            nums[store_idx],nums[right] = nums[right],nums[store_idx]
            
            if store_idx == k:
                return nums[store_idx]
            elif store_idx > k:
                return quick_select(left,store_idx-1,k,nums)
            else:
                return quick_select(store_idx+1,right,k,nums)
    
        
        return quick_select(0,N-1,N-k,nums)
    
#counting sort
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''
        we can use counting sort, with offsets
        need to offset by min value in order to allow for negative numbres mapping to arrays which are indexed by positive real numbers
        then rebuild the array
        '''
        N = len(nums)
        min_val = min(nums)
        max_val = max(nums)
        counts = [0]*(max_val - min_val + 1)
        
        for num in nums:
            counts[num - min_val] += 1
        
        #created sorted array
        start = 0
        for i,count in enumerate(counts):
            counts[i] = start
            start += count
        
        sorted_list = [0]*N
        for num in nums:
            sorted_list[counts[num - min_val]] = num
            counts[num - min_val] += 1
            
        
        return sorted_list[-k]
    
#counting sort without building the array
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        '''
        we can use counting sort, with offsets
        need to offset by min value in order to allow for negative numbres mapping to arrays which are indexed by positive real numbers
        then rebuild the array
        '''
        N = len(nums)
        min_val = min(nums)
        max_val = max(nums)
        counts = [0]*(max_val - min_val + 1)
        
        for num in nums:
            counts[num - min_val] += 1
        
        #using no sorted array, start from then end 
        for num in range(len(counts)-1,-1,-1):
            k -= counts[num]
            if k <= 0:
                return num + min_val
        
        return -1
    
##########################################
# 767. Reorganize String 
# 14AUG23
##########################################
#the corner cases absolutlte suck
class Solution:
    def reorganizeString(self, s: str) -> str:
        '''
        build a string so that any two adjacent characters are not the same
        use heap and alternate placing most common chars
        
        '''
        counts = Counter(s)
        max_heap = [(-count,char) for char,count in counts.items()]
        heapq.heapify(max_heap)
        ans = ""
        
        if len(counts) == 1:
            return ans
        
        while len(max_heap) > 1:
            first_count,first_char = heapq.heappop(max_heap)
            second_count,second_char = heapq.heappop(max_heap)
            ans += first_char
            ans += second_char
            first_count += 1
            second_count += 1
            
            if first_count < 0:
                heapq.heappush(max_heap, (first_count,first_char))
            if second_count < 0:
                heapq.heappush(max_heap, (second_count,second_char))
    
        
        
        
        if len(max_heap) == 1 and abs(max_heap[0][0]) == 1:
            ans += max_heap[0][1]
        else:
            return ""
            
        #check
        for i in range(len(ans) - 1):
            if ans[i] == ans[i+1]:
                return ""
        
        return ans
    
#aye yai yai
class Solution:
    def reorganizeString(self, s: str) -> str:
        '''
        build a string so that any two adjacent characters are not the same
        use heap and alternate placing most common chars
        
        '''
        counts = Counter(s)
        max_heap = [(-count,char) for char,count in counts.items()]
        heapq.heapify(max_heap)
        ans = []
        
        while max_heap:
            #check first
            first_count,first_char = heapq.heappop(max_heap)
            #alloed char
            if not ans or first_char != ans[-1]:
                ans.append(first_char)
                #make sure we have available
                if first_count + 1 < 0:
                    heapq.heappush(max_heap, (first_count + 1, first_char))
            
            #if we cant use the first char, then try the second char
            else:
                #if we cant use the first char, and there are nothing left, it canot be done
                if not max_heap:
                    return ""
                
                second_count,second_char = heapq.heappop(max_heap)
                ans.append(second_char)
                if second_count + 1 < 0:
                    heapq.heappush(max_heap, (second_count + 1, second_char))
                
                #push the first char back
                heapq.heappush(max_heap, (first_count,first_char))
                
        
        return "".join(ans)
                
class Solution:
    def reorganizeString(self, s: str) -> str:
        '''
        we can intelligently place chars at odd/even indices
        we put the largest chars at even indices first until we exhaut the largest
        then place the remaning at odd inices
        
        note:
            to guarantee an arrangement, we need to ensure that the most frequent letter doest not exceed len(s) // 2 + 1
            otherwise its impossible. why? becasue we need a gap of 1 between same chars, so if a char occupies more than half, its going to be place next to itself somewhere
            
        
        '''
        counts = Counter(s)
        N = len(s)
        max_count = 0
        max_count_char = ""
        for char,count in counts.items():
            if count > max_count:
                max_count = count
                max_count_char = char
                
        #check
        if max_count > (len(s) + 1) // 2:
            return ""
        
        ans = [""]*N
        i = 0
        
        #place most frequesnt letter
        while counts[max_count_char] > 0:
            ans[i] = max_count_char
            counts[max_count_char] -= 1
            i += 2
            
        #recall we may not hav gotten to the end and there may still be spots to place
        #place the rest
        for char,count in counts.items():
            while count > 0:
                if i >= N:
                    i = 1
                ans[i] = char
                count -= 1
                i += 2
        
        return "".join(ans)
            
#########################################
# 239. Sliding Window Maximum (REVISTED)
# 16AGU23
#########################################
#good idea but doesnt work
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        '''
        need to find max for each sliding window of size k
        initally store max in the first window,
        then during the update step update the max by doing max(nums[left+1],nums[right+1],min(kthprevious_window))
        there can only be N-k+1 windows givne some nums of length N
        '''
        N = len(nums)
       
        if k == 1:
            return nums
       
        ans = []
        #find first max
        curr_max = max(nums[:k+1])
        for left in range(N-k+1):
            #sub = nums[left:left+k]
            #print(nums[left],nums[left+k-1])
            #update
            curr_max = max(curr_max,nums[left],nums[left+k-1])
            print(nums[left],nums[left+k-1],curr_max)
            ans.append(curr_max)
            #need to find new max
            curr_max = max(float('-inf'),nums[left],nums[left+k-1])
       
        return ans

#using pq
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        '''
        i can try using a max heap
        initalliy i keep the first k largest elements in the window
        then when i go in and add the next element, i need to remove one of the elements
        i can also keep track of the indices for each num
        
        [1,3,-1,-3,5,3,6,7]
        3
        
        inital max_heap:
            [[3,1],[1,0],[-3,2]]
        
        largest so far is 3
        
        intution, the elemetns that come before the largest will never be selected as the largest of any future windows
        example
        [1,2,3,4,1]
        any window with the elements [1,2,3] would also include 4, but they are not the max anway! so we can disacrd them
        
        make a pq of nums[:k] and for each entry pair it with its index
        get the first max as the answer
        then we need to examine the next k to n elements
        also keep track of the number of elements after k we have examined
        '''
        #print(nums[:k],nums[k])
        N = len(nums)
        if N <= k:
            return max(nums)
        
        ans = []
        max_heap = [(-num,i) for i,num in enumerate(nums[:k])]
        heapq.heapify(max_heap)
        
        ans.append(-max_heap[0][0]) #first larget
        j = 0
        for i in range(k,N):
            j += 1
            #if the nums's index currently at the top is smaller than j, then this can't possibly be a max for the the current window
            #so we just clear it
            while max_heap and max_heap[0][1] < j:
                heapq.heappop(max_heap)
            
            #add new one to the window
            heapq.heappush(max_heap, (-nums[i],i))
            ans.append(-max_heap[0][0])
        
        return ans
    
#another way
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        #additional way
        N = len(nums)
        res = [0]*(N-k+1)
        #pushsh the first k-1 elements
        pq = []
        for i in range(k-1):
            heapq.heappush(pq, (-nums[i],i))
        
        for window_end in range(k-1,N):
            #add the new element to be considers
            heapq.heappush(pq, (-nums[window_end],window_end))
            while pq and window_end - k + 1 > pq[0][1]:
                heapq.heappop(pq)
            
            res[window_end - k + 1] = -pq[0][0]
        
        return res
    
#montonic queue
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        '''
        true solution is something called montinic deque
        take the array [1,2,3,4,1]
        any window including the the first three elements would always have a max of 4, if 4 were included in the array
        idea is that elements that come before the largest element will never be select as the largest or any future windows
        but we cannot ignore the items that follow the largest
        in the example, we cannot ignore the last 1, since there mayb be a winow from the fourhth inde to the tieth index where there is a larger eleemnt
        
        say for exmaple we are considereing some number x, and x > 1, we can discard 1 because will be the new max
        in general, when we encounter a new element x, we can discard any elment less than x
        
        Let's say we currently have [63, 15, 8, 3] and we encounter 12. 
        Any future window with 8 or 3 will also contain 12, so we can discard them. After discarding them and adding 12, we have [63, 15, 12]. 
        As you can see, we keep elements in descending order.
        
        when we add a new element x to the deque, we maintain the rep invariant be keeping it descending
        keep dq is useful indces in the current window
        we need the indices in order to detect when elements leave the window 
        
        largest elmeent must always be in the first elment in the deque
        i.e nums[dq[0]] is the max
        
        
        We initialize the first window with the first k elements. 
        Then we iterate over the indices i in the range [k, n - 1], and for each element, we add its index to dq while maintaining the monotonic property. 
        We also remove the first element in dq if it is too far to the left (dq[0] = i - k). After these operations, 
        dq will correctly hold the indices of all useful elements in the current window in decreasing order. Thus, we can push nums[dq[0]] to the answer.


        '''
        n = len(nums)
        q = deque([])
        ans = []
        for i in range(k):
            #we are storing indices
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
        
        #first answer
        ans.append(nums[q[0]])
        
        for i in range(k,n):
            #if the current eleement at the front is outside the window, remove it
            if q and i - k == q[0]:
                q.popleft()
            #maintin invaraiant
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            
            q.append(i)
            ans.append(nums[q[0]])
        
        return ans
    
############################
# 542. 01 Matrix (REVISITED)
# 24MAY23
#############################
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        '''
        brute force is to look up down left right for each (i,j) the the position of the zero
        if dp(i,j) represents the nearest distance of zero cell (i,j)
        
        and we cant to consider a new i,j
            it would be 1 + min(of all neighbors for i,j)
        '''
        rows = len(mat)
        cols = len(mat[0])
        memo1 = {}
        memo2 = {}
        
        dirrs = [(-1,0),(1,0),(0,-1),(0,1)]
        #need two dps, one getting min from going down and to the right
        #the other going up and the the left
        #then dp on both and take min
        
        def dp1(i,j,memo):
            #out of bounds
            if i < 0 or i >= rows or j < 0 or j >= cols:
                return rows + cols
            if (i,j) in memo:
                return memo[(i,j)]
            if mat[i][j] == 0:
                return 0
            
            ans = 1 + min(dp1(i+1,j,memo),dp1(i,j+1,memo))
            memo[(i,j)] = ans
            return ans
        
        def dp2(i,j,memo,memo2):
            #out of bounds
            if i < 0 or i >= rows or j < 0 or j >= cols:
                return rows + cols
            if (i,j) in memo:
                return memo[(i,j)]
            if mat[i][j] == 0:
                return 0
            
            #if there is a shorter min dsitnace from cming down and to the right, leave that answer at a minimum
            ans = min(memo2[(i,j)], 1 + min(dp2(i-1,j,memo,memo2),dp2(i,j-1,memo,memo2)))
            memo[(i,j)] = ans
            return ans
        
        
        
        ans = [[0]*cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                ans[i][j] = dp1(i,j,memo1)
        

        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                ans[i][j] = min(ans[i][j], dp2(i,j,memo2,memo1))
        
        return ans
    
#bottom up
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        '''
        we can also do bottom up,
        note, we just cant take the minimum from any of the 4 neighbors and add 1, becausee it depends on how we got to a previous neighbor
        so we start bottom going left, then up going right
        i.e we don't have a strict ordering on how we got to the current minimum and we don't know what way to sovle the subprblems
        '''
        
        rows = len(mat)
        cols = len(mat[0])
        
        dp = [row[:] for row in mat]
        
        #going down and right
        for row in range(rows):
            for col in range(cols):
                ans = float('inf')
                if mat[row][col] != 0:
                    #can go back
                    if row > 0:
                        ans = min(ans, dp[row-1][col] )
                    
                    if col > 0:
                        ans = min(ans, dp[row][col-1])
                        
                    dp[row][col] = ans + 1
                    
        #going up and left
        for row in range(rows-1,-1,-1):
            for col in range(cols-1,-1,-1):
                ans = float('inf')
                if mat[row][col] != 0:
                    if row < rows - 1:
                        ans = min(ans,dp[row+1][col])
                    if col < cols - 1:
                        ans = min(ans,dp[row][col+1])
                
                dp[row][col] = min(dp[row][col],ans+1)
    
        return dp

#bfs
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        '''
        this is multipoint bfs
        i can first load up into a queue, all the (i,j);s that correspong to a 0 
        also, pair with it its distance from 0, in this case its zero
        then bfs to its unseen neighbors and if this neighor is a 1, this distacnce must be the min to it
        '''
        seen = set()
        rows = len(mat)
        cols = len(mat[0])
        q = deque([])
        
        ans = [[0]*cols for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    q.append((i,j,0))
                    #need toa dd them before starting
                    seen.add((i,j))
                    #becase we look in directions, the could be added again
        
        dirrs = [(1,0),(-1,0),(0,-1),(0,1)]
        
        while q:
            x,y,dist = q.popleft()
            for dx,dy in dirrs:
                neigh_x = dx + x
                neigh_y = dy + y
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and (neigh_x,neigh_y) not in seen:
                    q.append((neigh_x,neigh_y,dist+1))
                    #we need to mark them zene here because fomr another neighbore, we might be able to reach it
                    #add the "going" to be seen neighbors
                    seen.add((neigh_x,neigh_y))
                    ans[neigh_x][neigh_y] = dist + 1
        
        return ans

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        '''
        we can just use -1 as the results matrix (i,j) entry marking as unvitised
        then just bfs from all zeros
        then retraverse and only take values where mat[i][j] == 1
        '''
        rows = len(mat)
        cols = len(mat[0])
        zeros_dist = [[0]*cols for _ in range(rows)]
        dirrs = [(1,0),(-1,0),(0,-1),(0,1)]
        
        q = deque([])
        seen = set([])
        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    q.append((i,j,0))
                    seen.add((i,j))
                    
        
        while q:
            x,y,dist = q.popleft()
            zeros_dist[x][y] = dist
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and (neigh_x,neigh_y) not in seen:
                    seen.add((neigh_x,neigh_y))
                    q.append((neigh_x,neigh_y,dist+1))
        
        
        return zeros_dist
    


#######################################
# 636. Exclusive Time of Functions
# 15AUG23
#######################################
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        '''
        we are given n jobs with logs
        logs[i] is read as 'id':'started/ended at':'time'
        
        functions exlcusive time is the sum of executing times for all functions in the program
        what if i keep a hashamp with entries
        stack of times for each function
        it a singel threaded CPU, so we can only be running the function that most receently start
        we keep an array and stack
        on the stack we keep track of the current running functinons start/end event time
        
        '''
        ans = [0]*n
        stack = []
        
        for log in logs:
            log = log.split(":")
            #new start time, addd to stack
            if log[1] == 'start':
                stack.append([int(log[2]),0]) #zero to keep track of the amouynt of time since this function start
                #entires are for started processed only (time when it start,curr_running_time)
            #otherwise we end
            else:
                #get the previous process and how long its been running
                start = stack.pop()
                #evalute time since it ran
                time = int(log[2]) - start[0] + 1
                ans[int(log[0])] += time - start[1] #add this running time to the function
                if stack:
                    #add the rest of the running time to the previous funcino that ran
                    stack[-1][1] += time
        
        return ans