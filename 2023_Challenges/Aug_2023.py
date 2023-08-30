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
#TLE
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        '''
        intuitino
            func1 calls func2, func2 calls func3
            func3 last, but ends first, and func1 starts first, but ends last (if there were a dependency chain)
        
        we can use stack, and just keep icnrementing a time count, until we get to the enxt time
        intiaiyll we push the first functino on top of thes stack, then we keep increment both timer and exclusing time for functino on stack
        untilwe get to the next time
            if its a start, add to stack and keep icnrmeing timers and exlcusivv time
            if its end, we need to increment the timer and exlusive time of the last function for this end time, thne pop
            
        #TLE
        '''
        stack = []
        ans = [0]*n
        
        #first function
        first = logs[0].split(":")
        #push indicies
        stack.append(int(first[0]))
        curr_time = int(first[2])
        
        for i in range(1,len(logs)):
            next_func = logs[i].split(':')
            #cinrement time intil we get to the the next time marker
            while curr_time < int(next_func[2]):
                curr_time += 1
                ans[stack[-1]] += 1
            
            #now we have moved up to this time
            #check start of new function
            if next_func[1] == 'start':
                stack.append(int(next_func[0]))
            #we are done
            else:
                ans[stack[-1]] += 1
                curr_time += 1
                stack.pop()
                
        
        return ans
    
#moving one at time
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
    
#jump to the end for the ith job
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        '''
        instead of going 1 step at a time for curr time, just move to it
        '''
        stack = []
        ans = [0]*n
        
        #first function
        first = logs[0].split(":")
        #push indicies
        stack.append(int(first[0]))
        prev = int(first[2])
        
        for i in range(1,len(logs)):
            next_func = logs[i].split(':')
            #now we have moved up to this time
            #check start of new function
            if next_func[1] == 'start':
                if stack:
                    ans[stack[-1]] += int(next_func[2]) - prev
                stack.append(int(next_func[0]))
                prev = int(next_func[2])
            #we are done
            else:
                ans[stack[-1]] += int(next_func[2]) - prev + 1
                prev = int(next_func[2]) + 1
                stack.pop()
                
        
        return ans
        

#############################
# 1615. Maximal Network Rank
# 18JUL23
#############################
class Solution:
    def maximalNetworkRank(self, n: int, roads: List[List[int]]) -> int:
        '''
        we are given n cities and roads connecting them
        network rank between two cities
            the total number of directly connected roads to either city
            if road is direclt connected to both cities, its counted once
        
        max network rank is max of all pairs
        make adjcent list, then coount the number of connecting rouds between an i and j
        (0,1) same
        (0,3)
        (1,0) same
        (0,2)
        (0,3)
        
        network rank is almost the sum of their degrees
        then we just check of the are connected using the adj list
        '''
        adj_list = defaultdict(set)
        for u,v in roads:
            adj_list[u].add(v)
            adj_list[v].add(u)
        
        ans = 0
        for i in range(n):
            for j in range(i+1,n):
                i_deg = len(adj_list[i])
                j_deg = len(adj_list[j])
                rank = i_deg + j_deg
                if j in adj_list[i] or i in adj_list[j]:
                    rank -= 1
                
                ans = max(ans,rank)
        
        return ans
    
#############################
# 490. The Maze (REVISTED)
# 18AUG23
#############################
#BFS
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        '''
        0s are spaces and 1s are walls, the ball can only go through spaces and must stop at a wall
        if i end up going back to the same spot, ive seen, return false
        bfs, but instead of step, go all the way in for directions
        '''
        rows = len(maze)
        cols = len(maze[0])
        seen = set()
        
        q = deque([start])
        while q:
            x,y = q.popleft()
            if [x,y] == destination:
                return True
            seen.add((x,y))
            #go down
            neigh_x,neigh_y = x,y
            while neigh_x - 1 >= 0 and maze[neigh_x-1][neigh_y] == 0:
                neigh_x -= 1
            #chceck that its not the same and not in vsitied
            if (neigh_x,neigh_y) != (x,y) and (neigh_x,neigh_y) not in seen:
                q.append((neigh_x, neigh_y))
            
            #go up
            neigh_x,neigh_y = x,y
            while neigh_x + 1 < rows and maze[neigh_x+1][neigh_y] == 0:
                neigh_x += 1
            #chceck that its not the same and not in vsitied
            if (neigh_x,neigh_y) != (x,y) and (neigh_x,neigh_y) not in seen:
                q.append((neigh_x, neigh_y))
                
            #go left
            neigh_x,neigh_y = x,y
            while neigh_y - 1 >= 0 and maze[neigh_x][neigh_y-1] == 0:
                neigh_y -= 1
            #chceck that its not the same and not in vsitied
            if (neigh_x,neigh_y) != (x,y) and (neigh_x,neigh_y) not in seen:
                q.append((neigh_x,  neigh_y))
                
            #go right
            neigh_x,neigh_y = x,y
            while neigh_y + 1 < cols and maze[neigh_x][neigh_y+1] == 0:
                neigh_y += 1
            #chceck that its not the same and not in vsitied
            if (neigh_x,neigh_y) != (x,y) and (neigh_x,neigh_y) not in seen:
                q.append((neigh_x, neigh_y))
        
#optimized using for loop
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        '''
        0s are spaces and 1s are walls, the ball can only go through spaces and must stop at a wall
        if i end up going back to the same spot, ive seen, return false
        bfs, but instead of step, go all the way in for directions
        '''
        rows = len(maze)
        cols = len(maze[0])
        seen = set()
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        q = deque([start])
        while q:
            x,y = q.popleft()
            if [x,y] == destination:
                return True
            seen.add((x,y))
            for dx,dy in dirrs:
                neigh_x,neigh_y = x,y
                while 0 <= neigh_x + dx < rows and 0 <= neigh_y + dy < cols and maze[neigh_x+dx][neigh_y+dy] == 0:
                    neigh_x += dx
                    neigh_y += dy
                #chceck that its not the same and not in vsitied
                if (neigh_x,neigh_y) != (x,y) and (neigh_x,neigh_y) not in seen:
                    q.append((neigh_x, neigh_y))
            

        
        return False
    
#dfs
class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        '''
        0s are spaces and 1s are walls, the ball can only go through spaces and must stop at a wall
        if i end up going back to the same spot, ive seen, return false
        bfs, but instead of step, go all the way in for directions
        '''
        rows = len(maze)
        cols = len(maze[0])
        seen = set()
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(x,y,seen):
            if [x,y] == destination:
                return True
            if (x,y) in seen:
                return False
            seen.add((x,y))
            for dx,dy in dirrs:
                neigh_x,neigh_y = x,y
                while 0 <= neigh_x + dx < rows and 0 <= neigh_y + dy < cols and maze[neigh_x+dx][neigh_y+dy] == 0:
                    neigh_x += dx
                    neigh_y += dy
                #chceck that its not the same and not in vsitied
                if dfs(neigh_x,neigh_y,seen):
                    return True
            
            return False
        
        return dfs(start[0],start[1],seen)
    
#########################################################################
# 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
# 19AUG23
##########################################################################
#jesus fuck
class UnionFind:
    def __init__(self,n):
        self.rank = [1]*n
        self.parent = [i for i in range(n)]
        #need to make sure we have n nodes
        self.nodes = 0
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
            
        return self.parent[x]
    
    def join(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        #same group
        if x_par == y_par:
            return False
        if self.rank[x_par] >= self.rank[y_par]:
            self.parent[y_par] = x_par
            self.rank[x_par] += self.rank[y_par]
            self.nodes = max(self.nodes,self.rank[x_par])
            self.rank[y_par] = 0
        else:
            self.parent[x_par] = y_par
            self.rank[y_par] += self.rank[x_par]
            self.nodes = max(self.nodes,self.rank[y_par])
            self.rank[x_par] = 0       
            
        return True
class Solution:
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        '''
        a critical edge is an edge whose deletion from the graph would cause the MST weight to increase
            note, this is from the perspective of the GRAPH and not an MST from the graph
            if we were to remove a critical edge from a graph, and the MST weight goes up, we know that this edge must be critical (i.e we need it to keep the MST at a minmum)
        a psuedo-critical edge is one that can appear in some MST's but not all
            i.e, this edge isn't really necessary to keep the MST weight at the minimum
        
        the algorithm is straight forward, i jutt dont know how to implement it
        notes on difference between kruskal and primm
        krukal and find an MST in a disconnected graph
            i.e for each disconnected componenet, find an MST
        primms only works on connected graph
        
        kruskals uses DSU to find MSt
        hints
            1. use kurskal algo to find the MST by sorting edges and picking esges from smaller weights
            2. use dsu to to avoid adding reducnt edges that result in cycle
            3. to find if edge is critical, delete that edge and re-run MST and see if weight incresed
            4. to find pseudo-critical (in any MST), include that edge to the accepted edge list and continue MST,
                then see if the resulting MST has the same wieght of the inital MST
        
        first step is to run kruskal to find the MST and the MST weight
        next is to identify the critical and non critical edges
        
        for critical
            rebuild the MST while dropping the edge, if this new MST's weight increases or an MST cannot be made, this edge must be critical
            given the inputs peforming kruskal should be fine
        
        for non critical
            first check that adge is not critical, which mean it is a candidate for being non critical
            then run kruskal again while forcing the edge to be part of three
            if the new MST weight remains the same, this edges is part of at least one MST ans is non critical (i.e psuedo critical)
        '''
        #make new edges paired with indices
        new_edges = []
        for i,edge in enumerate(edges):
            entry = (i,edge)
            new_edges.append(entry)
            
        #sort increasinly by weight
        new_edges.sort(key = lambda x: x[-1][-1])
        
        #find mst first
        mst = UnionFind(n)
        mst_weight = 0
        for i,entry in new_edges:
            u,v,w = entry
            #we if we can union them add to mst weight
            if mst.join(u,v):
                mst_weight += w
        
        #print(mst_weight,mst.nodes)
        #now check for critical and pseudo_critical
        critical = set()
        pseudo_critical = set()
        
        #first check critical
        for i,entry1 in new_edges:
            #igonre this ith edge and rebuild mst with the rest
            mst_critical = UnionFind(n)
            critical_edge_mstweight = 0
            for j,entry2 in new_edges:
                u,v,w = entry2
                if i != j and mst_critical.join(u,v):
                    critical_edge_mstweight += w
            
            if mst_critical.nodes < n or critical_edge_mstweight > mst_weight:
                critical.add(i)

        #now check pseudo critical
        for i,entry1 in new_edges:
            #any edege not in critical could be psuedo critical
            if i not in critical:
                #include this edge
                uf_pseudo = UnionFind(n)
                uf_pseudo.join(entry1[0],entry1[1])
                pseudo_edge_weight = entry1[-1]
                for j,entry2 in new_edges:
                    u,v,w = entry2
                    if i != j and uf_pseudo.join(u,v):
                        pseudo_edge_weight += w
                
                #print(i,pseudo_edge_weight,mst_weight)
                if pseudo_edge_weight == mst_weight and uf_pseudo.nodes == n:
                    pseudo_critical.add(i)
        
        return [critical,pseudo_critical]

############################################
# 1135. Connecting Cities With Minimum Cost
# 19AUG23
###########################################
class UnionFind:
    def __init__(self,n):
        self.rank = [1]*n
        self.parent = [i for i in range(n)]
        #need to make sure we have n nodes
        self.nodes = 0
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
            
        return self.parent[x]
    
    def join(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        #same group
        if x_par == y_par:
            return False
        if self.rank[x_par] >= self.rank[y_par]:
            self.parent[y_par] = x_par
            self.rank[x_par] += self.rank[y_par]
            self.nodes = max(self.nodes,self.rank[x_par])
            self.rank[y_par] = 0
        else:
            self.parent[x_par] = y_par
            self.rank[y_par] += self.rank[x_par]
            self.nodes = max(self.nodes,self.rank[y_par])
            self.rank[x_par] = 0
            
            
        return True
    
class Solution:
    def minimumCost(self, n: int, connections: List[List[int]]) -> int:
        '''
        kruskal, sort in min weights then use union find
        '''
        mst = UnionFind(n+1)
        #sort edges by weight
        connections.sort(key = lambda x: x[-1])
        mst_weight = 0
        for u,v,w in connections:
            if mst.join(u,v):
                mst_weight += w
        
        
        if mst.nodes == n:
            return mst_weight
        return -1
    
####################################################
# 1203. Sort Items by Groups Respecting Dependencies
# 19AUG23
#####################################################
class Solution:
    def sortItems(self, n: int, m: int, group: List[int], beforeItems: List[List[int]]) -> List[int]:
        '''
        we have n items, that belong to zero or one of m groups, i.e one to many from n to m
        a group can have no item belonging to it
        return sorted list such that
            * items belong to the same group are next to each other
            * itemss follow beforeItems, beforeItems[i] is a list contains all the items that should come before the ith item in the sorted array
            * essentially this gives a relative ordering
        
        hints
            top sort on the depency graph
            build two graphs, one for the grous and another for the items
            
        careful with shared indices between n items and m groups
        intuition
            do top sort using the item dependencies
            then do top sort in each group using item depenedncies in the group
            colect the items into each group
        
        the beforItems top sort is trivial
        when doing the groups, we check that the previtem is coming fron a different group
        
        so we need to assign a uniqe id for each of the groups
            groups can only be up to m, and if they are already in a group, we can skip them
        '''
        #trasform the group array to encode each item as a unique group
        uniq_group = m
        for i in range(n):
            if group[i] == -1:
                group[i] = uniq_group
                uniq_group += 1
                
        group_graph = defaultdict(list)
        group_indegree = [0]*uniq_group
        
        item_graph = defaultdict(list)
        item_indegree = [0]*n

        for i,come_before in enumerate(beforeItems):
            for item_before in come_before:
                item_graph[item_before].append(i)
                item_indegree[i] += 1
                
                #group depenedncy 
                if group[item_before] != group[i]:
                    group_graph[group[item_before]].append(group[i])
                    group_indegree[group[i]] += 1
                    
        #top dort
        def topSort(graph,indegree):
            ordering  = []
            q = deque([])
            #add in 0 inderrees
            for i in range(len(indegree)):
                if indegree[i] == 0:
                    q.append(i)
            
            while q:
                curr = q.popleft()
                ordering.append(curr)
                for neigh in graph[curr]:
                    indegree[neigh] -= 1
                    if indegree[neigh] == 0:
                        q.append(neigh)
            
            return ordering if len(ordering) == len(indegree) else []
        
        #top sort on both graphs
        item_ordering = topSort(item_graph,item_indegree)
        group_ordering = topSort(group_graph, group_indegree)
        
        if not item_ordering or not group_ordering:
            return []
        
        #items are sorts already, we just need to order within in groups
        ordered_groups = collections.defaultdict(list)
        for item in item_ordering:
            #find the group they belong too
            group_key = group[item]
            ordered_groups[group_key].append(item)
        
        #now use the group ordering, we already know the ordering of items in the group
        #no we just need to use the ordering of the groups
        ans = []
        for group_index in ordered_groups:
            ans += ordered_groups[group_index]
            
        return ans
    
#the actual solution
class Solution:
    def sortItems(self, n, m, group, beforeItems):
        # If an item belongs to zero group, assign it a unique group id.
        group_id = m
        for i in range(n):
            if group[i] == -1:
                group[i] = group_id
                group_id += 1
        
        # Sort all item regardless of group dependencies.
        item_graph = [[] for _ in range(n)]
        item_indegree = [0] * n
        
        # Sort all groups regardless of item dependencies.
        group_graph = [[] for _ in range(group_id)]
        group_indegree = [0] * group_id      
        
        for curr in range(n):
            for prev in beforeItems[curr]:
                # Each (prev -> curr) represents an edge in the item graph.
                item_graph[prev].append(curr)
                item_indegree[curr] += 1
                
                # If they belong to different groups, add an edge in the group graph.
                if group[curr] != group[prev]:
                    group_graph[group[prev]].append(group[curr])
                    group_indegree[group[curr]] += 1      
        
        # Tologlogical sort nodes in graph, return [] if a cycle exists.
        def topologicalSort(graph, indegree):
            visited = []
            stack = [node for node in range(len(graph)) if indegree[node] == 0]
            while stack:
                cur = stack.pop()
                visited.append(cur)
                for neib in graph[cur]:
                    indegree[neib] -= 1
                    if indegree[neib] == 0:
                        stack.append(neib)
            return visited if len(visited) == len(graph) else []

        item_order = topologicalSort(item_graph, item_indegree)
        group_order = topologicalSort(group_graph, group_indegree)
        
        if not item_order or not group_order: 
            return []
        
        # Items are sorted regardless of groups, we need to 
        # differentiate them by the groups they belong to.
        ordered_groups = collections.defaultdict(list)
        for item in item_order:
            ordered_groups[group[item]].append(item)
        
        # Concatenate sorted items in all sorted groups.
        # [group 1, group 2, ... ] -> [(item 1, item 2, ...), (item 1, item 2, ...), ...]
        answer = []
        for group_index in group_order:
            answer += ordered_groups[group_index]
        return answer
    
#######################################################
# 1180. Count Substrings with Only One Distinct Letter
# 21AUG23
#######################################################
class Solution:
    def countLetters(self, s: str) -> int:
        '''
        partition s into substrings that contain only one distinct character
        then cound number of substring we can get for that subtrain
        exmple, so we have
        'abc'
        a,ab,abc
        b,bc,
        c
        
        which is just sum of arithmetic sequences
        use sliding windwo to partition
        '''
        left = 0
        N = len(s)
        ans = 0
        for right in range(N+1):
            if s[min(right,N-1)] != s[left]:
                temp = s[left:right]
                #print(temp,right - left)
                substring_size = right - left
                num_substrings = (substring_size + 1)*(substring_size) // 2
                ans += num_substrings
                left = right
        
        #last one
        substring_size = right - left + 1
        num_substrings = (substring_size + 1)*(substring_size) // 2
        ans += num_substrings
        return ans
        
#extend to boundary condition when right get to the end
class Solution:
    def countLetters(self, s: str) -> int:
        '''
        partition s into substrings that contain only one distinct character
        then cound number of substring we can get for that subtrain
        exmple, so we have
        'abc'
        a,ab,abc
        b,bc,
        c
        
        which is just sum of arithmetic sequences
        use sliding windwo to partition
        '''
        left = 0
        N = len(s)
        ans = 0
        for right in range(N+1):
            if right == N or s[min(right,N-1)] != s[left]:
                temp = s[left:right]
                #print(temp,right - left)
                substring_size = right - left
                num_substrings = (substring_size + 1)*(substring_size) // 2
                ans += num_substrings
                left = right
        

        return ans
        
#dp, top down
class Solution:
    def countLetters(self, s: str) -> int:
        '''
        can use dp
        let dp(i) be the number of substrings with one unique letter ending at i
        if s[i] == s[i+1], then extend
        then do this for all i
        '''
        N = len(s)
        memo = {}
        
        def dp(i):
            if i == N-1:
                return 1
            if i in memo:
                return memo[i]
            if s[i] == s[i+1]:
                ans = 1 + dp(i+1)
                memo[i] = ans
                return ans
            memo[i] = 1
            return 1
        
        ans = 0
        for i in range(N):
            ans += dp(i)
        
        return ans
        
#bottom up, the answer is just the sum of the dp array
class Solution:
    def countLetters(self, s: str) -> int:
        '''
        bottom up
        '''
        N = len(s)
        dp = [0]*(N)
        dp[-1] = 1
        
        for i in range(N-2,-1,-1):
            if s[i] == s[i+1]:
                dp[i] = dp[i+1] + 1
            else:
                dp[i] = 1
        
        return sum(dp)
    
#####################################
# 533. Lonely Pixel II
# 21AUG23
#####################################
#bleagh, close one
class Solution:
    def findBlackPixel(self, picture: List[List[str]], N: int) -> int:
        '''
        return the number of black lonely pixels
        a black lonely pizel is a char 'B' at s specific (i,j) where
            i and j both contain exactly target black pixels
            for all rows that have a black pixel at column c, the hould be exactly the same row as r
            
        find all rows and cols that contain target black pixels
        then check
            for all rows that have a black pixel at column c, the should be exactly same as row r
            basically for all rows r (in candidate rows)
                all the c's in this row are black
            i.e for each candidate_row, there can only be black pixels at columns c
            
        the row should match,i.e their string singatures shuld match
        '''
        rows = len(picture)
        cols = len(picture[0])
        def countBsAtRow(row):
            count = 0
            for col in range(cols):
                count += picture[row][col] == "B"
            
            return count
        
        def countBsAtCol(col):
            count = 0
            for row in range(rows):
                count += picture[row][col] == "B"
            
            return count
        
        candidate_rows = []
        candidate_cols = []
        for i in range(rows):
            if countBsAtRow(i) == N:
                candidate_rows.append(i)
        
        for j in range(cols):
            if countBsAtCol(j) == N:
                candidate_cols.append(j)
        
        
        #get singatures of each rows and put into a count map
        counts = Counter()
        for row in picture:
            row = "".join(row)
            counts[row] += 1
        
        #for each signature, check that there are only B's at the candidate columns, 
        #and for each of these increment
        for row,c in counts.items():

class Solution:
    def findBlackPixel(self, picture: List[List[str]], target: int) -> int:
        '''
        for each row, check if row and N black pixels. if it does, store 2 things
            its row signauture in a count map
            and keep track of the of the number of pixels in each column
            
        then, for through the hashamp, and find singautres where we have N black pixels,
        then validate the cols for that row
        ans is count of that row signature times the number of B's in that column
        '''
        rows = len(picture)
        cols = len(picture[0])
        if rows == 0 or cols == 0:
            return 0
        
        counts = Counter()
        col_counts = [0]*cols #store count Bs in this column
        for i,row in enumerate(picture):
            if row.count('B') == target:
                counts["".join(row)] += 1
            for j in range(cols):
                if picture[i][j] == "B":
                    col_counts[j] += 1
        
        
        lonely_pixels = 0
        #goo through hashmap row signatures, and validate the rows
        for row,count in counts.items():
            #if the count of rows with target B == target, this might be a valid row
            if count == target:
                for j in range(cols):
                    if (row[j] == 'B' and col_counts[j] == target):
                        #there are target occurrences of lonely pixels
                        lonely_pixels += target
        
        return lonely_pixels
    
class Solution:
    def findBlackPixel(self, picture: List[List[str]], target: int) -> int:
        '''
        just another way,
        intuition, if a column is a valid, than it contributes N lonely pizels
        reduction: find number of valid columns
        
        valid columns are:
            1. it has target Bs
            2. first row with B insertsection col count == n
            3. 
        '''
        count_valid_cols = 0
        for col in zip(*picture):
            if col.count('B') != target:
                continue
            #find first index in row where there is a B
            first_row = picture[col.index('B')] #first row with B at this col
            #must have target B's
            if first_row.count('B') != target:
                continue
            
            #there must be N of these
            if picture.count(first_row) != target:
                continue
            
            count_valid_cols += 1
        
        return target*count_valid_cols

#############################################
# 2343. Query Kth Smallest Trimmed Number
# 22AUG23
#############################################
class Solution:
    def smallestTrimmedNumbers(self, nums: List[str], queries: List[List[int]]) -> List[int]:
        '''
        try brute force
        '''
        def trim_digit(dig,trim):
            return dig[-trim:]
        
        ans = []
        
        for k,trim in queries:
            temp = []
            for i,num in enumerate(nums):
                entry = (int(trim_digit(num,trim)),i)
                temp.append(entry)
            
            #sort
            temp.sort()
            ans.append(temp[k-1][1])
        
        return ans
    
####################
# 723. Candy Crush
# 23AUG23
#####################
class Solution:
    def candyCrush(self, board: List[List[int]]) -> List[List[int]]:
        '''
        this is similar to Advent of Code 2022 with the stone and chambers problem
        first loook through board for 3 in a row, horiz, and vert, then clear these with X
        then replace the zeros with numbers above them
        it shouldn't matter if i move down a cell at (i,j) where this cell (i,j) is abover another X cell (i',j') where (i',j') is below i,j
        
        
        '''
        rows = len(board)
        cols = len(board[0])
        #need extra board to store crush cells
        crush_cells = [[False]*cols for _ in range(rows)]
        
        def clear_crush_board(board):
            for i in range(rows):
                board[i] = [False]*cols
        
        def placeZeros(crushed_board,board):
            for i in range(rows):
                for j in range(cols):
                    if crushed_board[i][j] == True:
                        board[i][j] = 0
        
        def imputeCrushes(crushed_board,board):
            #locate crushes, and put as some char 'X'
            crush_found = False
            for i in range(rows):
                for j in range(cols):
                    #check cols at this row
                    if board[i][j] == 0:
                        continue
                    curr_col = j
                    next_col = j + 1
                    while (next_col < cols) and board[i][next_col] == board[i][j]:
                        next_col += 1
                    
                    #check we have at least three
                    if next_col - curr_col >= 3:
                        #impute
                        crush_found = True 
                        for k in range(curr_col,next_col):
                            crushed_board[i][k] = True
                        
                    #check rows doing down at this col
                    curr_row = i
                    next_row = i + 1
                    while (next_row < rows) and board[next_row][j] == board[i][j]:
                        next_row += 1
                    
                    #check
                    if next_row - curr_row >= 3:
                        crush_found = True
                        for k in range(curr_row,next_row):
                            crushed_board[k][j] = True
            
            return crush_found
        
        #implement drop col, pass in board values along col, and crush locations, then recreate a new array
        def crush_col(col,board):
            #imlement as swapping
            values = []
            for i in range(rows):
                values.append(board[i][col])
            lowest_zero = rows-1
            for i in range(rows-1,-1,-1):
                if values[i] == 0:
                    lowest_zero = max(lowest_zero,i)
                elif values[i] > 0:
                    values[i],values[lowest_zero] = values[lowest_zero],values[i]
                    lowest_zero -= 1
            
            #put back into board
            for i in range(rows):
                board[i][col] = values[i]
        
        
        while imputeCrushes(crush_cells,board):
            placeZeros(crush_cells,board)
            for col in range(cols):
                crush_col(col,board)
            clear_crush_board(crush_cells)
            
        return board

######################################
# 68. Text Justification
# 25AUG23
######################################
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        '''
        read each word line by line and try to fit
        so we are on some line, and we added words a,b,c but adding d exceeds the maxwidth, it means we can't add d on this line
        so we start a new line
        we then need to appropriately add spaces between the words so that they fit
        if the number of spaces does not evenly divide the words, the empty spots in the left will be assigned more spaces on the right
        last line should be left justified
        '''
        ans = []
        i = 0
        N = len(words)
        
        #first just try getting words that fit on line
        def getWords(i,words,maxWidth):
            curr_line = []
            curr_size = 0
            #ith word could be the last word, which doesn't require space, so we don't reflect adding a space here while we checl
            #we add space when adding its length to curr_size
            while i < len(words) and curr_size + len(words[i]) <= maxWidth:
                curr_line.append(words[i])
                curr_size += len(words[i]) + 1 #include spacce
                i += 1
            
            return curr_line,i
        
        
        def justifyLine(i,line,words,maxWidth):
            start_length = 0 #we are gonna add the each word's length + 1, 
            for word in line:
                start_length += len(word) + 1
            
            #remvove space from last word
            start_length -= 1
            extra_spaces = maxWidth - start_length
            
            #check if final line wor single word
            if len(line) == 1 or i == len(words):
                return " ".join(line) + " "*extra_spaces
            
            #otherwise distribute spaces in between, excep the last word
            space_words = len(line) - 1
            space_per_word = extra_spaces // space_words
            needs_extra_space = extra_spaces % space_words
            
            #add extra spaces
            for j in range(needs_extra_space):
                line[j] += " "
            
            #add the regular space
            for j in range(space_words):
                line[j] += " "*space_per_word
            
            return " ".join(line)
            
        i = 0
        ans = []
        while i < len(words):
            curr_line,next_i = getWords(i,words,maxWidth)
            #transform line
            transformed_line = justifyLine(next_i,curr_line,words,maxWidth)
            #print(transformed_line)
            ans.append(transformed_line)
            i = next_i
        
        return ans
                
#######################################
# 97. Interleaving String (REVISTED) 
# 25AUG23
#######################################
#nice try again
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        if len(s3) > len(s1) + len(s2)
        
        say we have indices (i,j,k) going into s1,s2, and s3
        and we know s3[:k] i.e up to this k is interleaving
        no we want to examin k+1
            we can only move to k+1 if s1[i] or s2[j] == s3[k]
            if there isn't a match we can't make it interleaving
            otherwise if there is a match, try both
        '''
        
        if len(s3) > len(s1) + len(s2):
            return False
        memo = {}
        
        def dp(i,j,k):
            if i >= len(s1):
                return k == len(s3) and j == len(s2)
            if j >= len(s2):
                return k == len(s3) and i == len(s1)
            if k >= len(s3):
                #gone all the way
                return i == len(s1) and j == len(s2)
            
            if (i,j,k) in memo:
                return memo[(i,j,k)]
            

            if s3[k] == s1[i] or s3[k] == s2[j]:
                ans = dp(i+1,j,k+1) or dp(i,j+1,k+1)
                memo[(i,j,k)] = ans
                return ans
            else:
                ans = dp(i,j,k+1)
                memo[(i,j,k)] = ans
                return ans
             
        return dp(0,0,0)
    
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        if len(s3) > len(s1) + len(s2)
        
        say we have indices (i,j,k) going into s1,s2, and s3
        and we know s3[:k] i.e up to this k is interleaving
        no we want to examin k+1
            we can only move to k+1 if s1[i] or s2[j] == s3[k]
            if there isn't a match we can't make it interleaving
            otherwise if there is a match, try both
            
        k is just the sum of i + j
        '''
        
        if len(s3) != len(s1) + len(s2):
            return False
        
        
        dp = [[0]*(len(s2) + 1) for _ in range(len(s1) + 1)]
        
        #base case fill
        for i in range(len(s1) + 1):
            for j in range(len(s2) +1):
                if i == len(s1):
                    dp[i][j] = s2[j:] == s3[i+j:]
                if j == len(s2):
                    dp[i][j] = s1[i:] == s3[i+j:]
    
    
        #start from base case but 1 away
        for i in range(len(s1)-1,-1,-1):
            for j in range(len(s2)-1,-1,-1):
                a = s1[i] == s3[i+j] and dp[i+1][j]
                b = s2[j] == s3[i+j] and dp[i][j+1]
                ans = a or b
                dp[i][j] = ans
        
        return dp[0][0]

##########################################
# 1128. Number of Equivalent Domino Pairs
# 26AUG23
##########################################
#two pass
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        '''
        this is just a hashmap problem, store domino and its index and count totla number of pairs
        divid the pairs by two
        key, (a,b) and (b,a) are the some dominoe
        '''
        counts = Counter()
        
        for a,b in dominoes:
            entry = tuple(sorted([a,b]))
            counts[entry] += 1
        
        ans = 0
        for k,v in counts.items():
            uniq_pairs = v*(v-1) // 2
            ans += uniq_pairs
        
        return ans
    
#count on the fly
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        '''
        this is just a hashmap problem, store domino and its index and count totla number of pairs
        divid the pairs by two
        key, (a,b) and (b,a) are the some dominoe
        '''
        counts = Counter()
        ans = 0
        
        for a,b in dominoes:
            entry = tuple(sorted([a,b]))
            if entry in counts:
                ans += counts[entry]
            counts[entry] += 1
        
        return ans

##################
# 403. Frog Jump
# 27AUG23
###################
#need to cache seen states
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        '''
        check for zero index
        '''
        N = len(stones)
        
        seen = set()
        #temp = bisect.bisect_right(stones,4)
        #print(stones[temp-1])
        q = deque([(0,0)]) #state is curr_stone,and prev_jump size
        while q:
            curr_stone, prev_jump = q.popleft()
            if curr_stone == N-1:
                return True
            
            
            neigh1 = stones[curr_stone] + (prev_jump - 1)
            neigh2 = stones[curr_stone] + (prev_jump)
            neigh3 = stones[curr_stone] + (prev_jump + 1)
            
            #try finding each of the stones in the array
            idx1 = bisect.bisect_right(stones,neigh1)
            #make sure we dont hit the samse stone
            if idx1 - 1 != curr_stone and stones[idx1-1] == neigh1 and (idx1-1,prev_jump - 1) not in seen:
                seen.add((idx1-1,prev_jump - 1))
                q.append([idx1-1,prev_jump - 1])
                
            idx2 = bisect.bisect_right(stones,neigh2)
            #make sure we dont hit the samse stone
            if idx2 - 1 != curr_stone and stones[idx2-1] == neigh2 and (idx2-1,prev_jump) not in seen:
                seen.add((idx2-1,prev_jump))
                q.append([idx2-1,prev_jump])  
            
            idx3 = bisect.bisect_right(stones,neigh3)
            #make sure we dont hit the samse stone
            if idx3 - 1 != curr_stone and stones[idx3-1] == neigh3 and (idx3-1,prev_jump+1) not in seen:
                seen.add((idx3-1,prev_jump+1))
                q.append([idx3-1,prev_jump+1])
            
        return False
    
#consolidate
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        '''
        check for zero index
        '''
        N = len(stones)
        
        seen = set()
        #temp = bisect.bisect_right(stones,4)
        #print(stones[temp-1])
        q = deque([(0,0)]) #state is curr_stone,and prev_jump size
        while q:
            curr_stone, prev_jump = q.popleft()
            if curr_stone == N-1:
                return True
            
            
            for k in [-1,0,1]:
                neigh = stones[curr_stone] + (prev_jump + k)
                #try finding each of the stones in the array
                idx = bisect.bisect_right(stones,neigh)
                #make sure we dont hit the samse stone
                if idx - 1 != curr_stone and stones[idx-1] == neigh and (idx-1,prev_jump + k) not in seen:
                    seen.add((idx-1,prev_jump + k))
                    q.append([idx-1,prev_jump + k])
            
        return False 
            
#top down
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        '''
        let dp(i) be the answer if we can reach the end
        then we just check of we can hit try for any neighbors or i
        pre process stones 
            mapp its location to its index
        '''
        mapp = {dist:i for (i,dist) in enumerate(stones)}
        memo = {}
        
        N = len(stones)
        #keep track of index and prev_jump
        def dp(i,prev_jump):
            if i == N-1:
                return True
            if (i,prev_jump) in memo:
                return memo[(i,prev_jump)]
            for k in [prev_jump-1,prev_jump,prev_jump+1]:
                next_stone = stones[i] + k
                if next_stone in mapp and mapp[next_stone] != i and dp(mapp[next_stone],k):
                    memo[(i,prev_jump)] = True
                    return True
            
            memo[(i,prev_jump)] = False
            return False
        
        
        
        return dp(0,0)
    
#one ans for child answers
class Solution:
    def canCross(self, stones: List[int]) -> bool:

        mapp = {dist:i for (i,dist) in enumerate(stones)}
        memo = {}
        
        N = len(stones)
        #keep track of index and prev_jump
        def dp(i,prev_jump):
            if i == N-1:
                return True
            if (i,prev_jump) in memo:
                return memo[(i,prev_jump)]
            ans = False
            for k in [prev_jump-1,prev_jump,prev_jump+1]:
                next_stone = stones[i] + k
                if next_stone in mapp and mapp[next_stone] != i and dp(mapp[next_stone],k):
                    ans = ans or True
            
            memo[(i,prev_jump)] = ans
            return ans
        
        return dp(0,0)
        
#bottom up, TLE though, need to do it differently
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        '''
        for bottom up we notice that the length of stones can only be no greater than 2000
        so we make dp array 2001 by 2001, the rest should be just the reverse of top down
        '''
        mapp = {dist:i for (i,dist) in enumerate(stones)}
        dp = [[False]*(2001) for _ in range(2001)]
        N = len(stones)
        
        #base case fill
        for i in range(N):
            for prev_jump in range(2001):
                if i == N-1:
                    dp[i][prev_jump] = True
                    
                    
        for i in range(N-2,-1,-1):
            for prev_jump in range(min(N,2000)):
                ans = False
                for k in [prev_jump-1,prev_jump,prev_jump+1]:
                    next_stone = stones[i] + k
                    if next_stone in mapp and mapp[next_stone] != i and dp[mapp[next_stone]][k] == True:
                        ans = ans or True

                dp[i][prev_jump] = ans

        
        
        return dp[0][0]
    
#bottom up AC
class Solution:
    def canCross(self, stones: List[int]) -> bool:
        '''
        just check if we can reach the jth stone from any previous stone i
        return the last dp entry
        '''
        #easy cases, pruning
        if len(stones) == 1:
            return True
        if stones[1] - stones[0] != 1:
            return False
        
        dp = [set() for _ in range(len(stones))]
        dp[1].add(1)
        for i in range(2, len(stones)):
            for j in range(i):
                if (
                    stones[i] - stones[j] + 1 in dp[j] or
                    stones[i] - stones[j] in dp[j] or
                    stones[i] - stones[j] - 1 in dp[j]
                ):
                    dp[i].add(stones[i] - stones[j])
        return dp[-1]


#####################################
# 979. Distribute Coins in Binary Tree
# 25AUG23
######################################
#wasn't a graph problem
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        '''
        its given that we have n coins and n nodes
        we can only move one coin at a time and we can move from parent to child or from child to parent
        turn tree into graph
        if a node a has k coins, and a node b is two steps away, it will alwasy take 2 moves
        
        multipoint BFS
        queue up those with non zero coins
        '''
        graph = defaultdict(list)
        
        def dfs(parent,child):
            if not child:
                return
            if parent:
                graph[parent].append(child)
                graph[child].append(parent)
            
            if child.left:
                dfs(child,child.left)
            if child.right:
                dfs(child,child.right)
        
        dfs(None,root)
        
    # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def distributeCoins(self, root: Optional[TreeNode]) -> int:
        '''
        if the leaf of a tree has 0 coins (which has an ecess of -1, for example if a left has 1 coin, there is no escess)
        then we should push a coin from its parent on to the left
        if a leaf has 4 coins (excess of +3) then we need to move 3 coins from the leaf to its parent
        so in total, the number of moves from tha leave to or from its parent is
            excess = abs(num_coins-1)
            
            afterwards we never have to consider this leaf again
            
        we define dp(node) as the number of coinds in the subbtree at or below this node
            i.e the number of coins in this subtree - number of nodes
        
        then the number of moves we make from this node to and from its children is
            abs(dp(node.left)) + abs(dp(node.right))
        
        after we have an excess of node.val + dp(node.left) + dp(node.right) - 1
        '''
        ans = [0]
        
        def dp(node):
            if not node:
                return 0
            left = dp(node.left)
            right = dp(node.right)
            ans[0] += abs(left) + abs(right) #add up moves fro each node
            return node.val + left + right - 1 #keep one coin for the root
    
    
        dp(root)
        return ans[0]
    
#detailed explanation
#https://leetcode.com/problems/distribute-coins-in-binary-tree/discuss/432210/Detail-Explanation-Plus-solution
class Solution:
    def distributeCoins(self, root):

        self.moves = 0
        def move(root):
            if root == None:
                return 0
            left = move(root.left)
            right = move(root.righot)
            total_coins = left + right + root.val
            self.moves += abs(total_coins - 1)
            return total_coins - 1
        move(root)
        return self.moves


##################################
# 1175. Prime Arrangements
# 28AUG23
##################################
import math
class Solution:
    def numPrimeArrangements(self, n: int) -> int:
        '''
        fix the prime numbers at their prime indices, 
        say for an array of size k, there are p primes
        that mean there are (k-p) non primes
        we are free to move these (k-p) non primes anywhere
        so its just p!*(k-p)!
        
        in order to this, we need to count the number of primes up to n,
        we can use seive
        '''
        primes = [True]*(n+1)
        
        i = 2
        while i*i <= n:
            if primes[i] == True:
                multiples_i = i*i
                while multiples_i <= n:
                    primes[multiples_i] = False
                    multiples_i += i
            
            i += 1
        
        
        count_primes = sum(primes[2:])
        return math.factorial(count_primes)*math.factorial(n - count_primes) % (10**9 + 7)

#######################################################
# 2483. Minimum Penalty for a Shop
# 29AUG23
#######################################################
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        '''
        customers is log, string of N and Y, where N means no customers come at ith hour, and Y means they cam
        len(customers) is total hours, could be in range 0 to len(customers)
        
        if shop closes at the jth hour, define pentalty as
            for every hour when shop is open and no customers came, penalty goes up by 1
            for every hour when shop is closed and  customers came, penalty goes up by 1
        
        return earliest hour at which shop mus be close to incur the smallest penalty
        this is just prefix sum and suffix sum
        for penalty, we need pref count of N and suff count of Y
        precompute and evalutae
        '''
        pref_count_Ns = [0]
        suff_count_Ys = [0]
        N = len(customers)
        
        for i in range(N):
            pref_count_Ns.append(pref_count_Ns[-1] + (customers[i] == 'N'))
            suff_count_Ys.append(suff_count_Ys[-1] + (customers[N-i-1] == 'Y'))
        
        #reverse Ys to get suffix
        suff_count_Ys = suff_count_Ys[::-1]
        #print(pref_count_Ns)
        #print(suff_count_Ys)
        
        min_penalty = float('inf')
        earliest_hour = -1
        for i in range(N+1):
            curr_penalty = pref_count_Ns[i] + suff_count_Ys[i]
            if curr_penalty < min_penalty:
                min_penalty = curr_penalty
                earliest_hour = i
        
        return earliest_hour
    
#two pass
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        '''
        try optimizing by not reversing the suff_count_Ys, 
        which means we need N+1 elements
        '''
        N = len(customers)
        pref_count_Ns = [0]*(N+1)
        suff_count_Ys = [0]*(N+1)
        
        for i in range(N):
            pref_count_Ns[i+1] = pref_count_Ns[i] +  (customers[i] == 'N')
            suff_count_Ys[(N+1 -i) - 2] = suff_count_Ys[(N+1 - i) - 1] + (customers[N-i-1] == 'Y')
        
        min_penalty = float('inf')
        earliest_hour = -1
        for i in range(N+1):
            curr_penalty = pref_count_Ns[i] + suff_count_Ys[i]
            if curr_penalty < min_penalty:
                min_penalty = curr_penalty
                earliest_hour = i
        
        return earliest_hour
    
#two pass constant space
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        '''
        if we pick an ith hour to close, then the penalty becomes 
            the number of Ns to the left (prefix Ns)
            the number of Ys to the right (suffix Ys)
            
        
        intution, this one is very substle
            notince that in two adjacent cases (if we wish to close at i, then look at i -1)
            the status of one hour has changed, where the status has been changed from closing hour to open hour
            this implies that we can record the overall penalty change by calculating the difference betweeen two adjacent cases
        
        explanation
            say we calculate the penalty for some time i, and we need to calculate the next penalty for i + 1
            we just need to record the differences in penalties to get the penality for i + 1
        
        idea: close at hour 0 and calculate initial penalty
        '''
        N = len(customers)
        curr_penalty = 0
        earliest_hour = 0
        
        #calculate penalty by closing at hour zero first
        for ch in customers:
            curr_penalty += ch == 'Y'
            
        min_penalty = curr_penalty
            
        #to find penalty at i, we update by decreasing if Y or increasing if zero
        #we check by closing at i + 1
        for i,ch in enumerate(customers):
            if ch == 'Y':
                curr_penalty -= 1
            else:
                curr_penalty += 1
                
            #updates
            if curr_penalty < min_penalty:
                min_penalty = curr_penalty
                earliest_hour = i + 1
                
        
        return earliest_hour
    
#one pass
#turns out we dont need to precompute penalty for closing at the 0th hour
class Solution:
    def bestClosingTime(self, customers: str) -> int:
        '''
        we dont need to precompute
        '''
        N = len(customers)
        curr_penalty = 0
        earliest_hour = 0
            
        min_penalty = curr_penalty
            
        #to find penalty at i, we update by decreasing if Y or increasing if zero
        #we check by closing at i + 1
        for i,ch in enumerate(customers):
            if ch == 'Y':
                curr_penalty -= 1
            else:
                curr_penalty += 1
                
            #updates
            if curr_penalty < min_penalty:
                min_penalty = curr_penalty
                earliest_hour = i + 1
                
        
        return earliest_hour


        
#########################################
# 655. Print Binary Tree
# 28AUG23
#########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def printTree(self, root: Optional[TreeNode]) -> List[List[str]]:
        '''
        the rules are not quite correct
        rows should be equal to the height of tree
        cols should be 2^(neight + 1) - 1
        root node should be in the top row 
            grid[0][(cols-2)//2] i.e the middle of the first row
            so any node below the root is offset (+ or - from the center)
        
        if we place a node at grid[r][c]
            left should be grid[r+1][c-2^(height - r - 1)]
            right should be grid[r+1][c+2^(height - r - 1)]
            
        review notes on binary tree
        max total numbers of nodes is 2**height - 1
        max number of nodes at each level (0 indexed) = 2**L
        
        if we places a node at (level,c), and if depth = height
        then we define offset as depth - level - 2
        then left is placed at (level + 1, c - 2**offet)
        then right is placed at (level + 1, c + 2**offet )
        its given in the instructions that this shoudl be the calse, but its off by one
        '''
        #find height
        def getHeight(node):
            if not node:
                return 0
            left = getHeight(node.left)
            right = getHeight(node.right)
            return max(left,right) + 1
        
        def dfs(node,depth,level,pos):
            if not node:
                return
            #place
            self.grid[level][pos] = str(node.val)
            #we need to offset only in column placement
            #follow the formula, instead of height - r - 1, do height - level - 2
            offset = depth - level - 2
            dfs(node.left,depth,level+1,pos - 2**(offset))
            dfs(node.right,depth,level+1,pos + 2**(offset))
        
        #get height
        height = getHeight(root)
        
        #grid, note how cols are 2**(height - 1) insteaf of whats in the problemd escription
        self.grid = [[""]*(2**height - 1) for _ in range(height)]
        
        #start at first row and middle column
        dfs(root,height,0,(2**height - 1) // 2)

        return self.grid

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def printTree(self, root: Optional[TreeNode]) -> List[List[str]]:
        '''
        we just always place the node at middle
        left = 0, right = (number of nodes at the deepest levelt) - 1
        then we place at (left + right) // 2
        '''
        #find height
        def getHeight(node):
            if not node:
                return 0
            left = getHeight(node.left)
            right = getHeight(node.right)
            return max(left,right) + 1
        
        def dfs(node,level,left,right):
            if not node:
                return
            mid = (left + right) // 2
            self.ans[level][mid] = str(node.val)
            dfs(node.left,level+1,left,mid-1)
            dfs(node.right,level+1,mid+1,right)
            
        height = getHeight(root)
        cols = 2**height - 1 #basically the number of nodes at the last level
        self.ans = [[""]*cols for _ in range(height)]
        dfs(root,0,0,cols-1)
        return self.ans

############################################
# 1183. Maximum Number of Ones
# 29AGU23
############################################
#yayyy!
class Solution:
    def maximumNumberOfOnes(self, width: int, height: int, sideLength: int, maxOnes: int) -> int:
        '''
        define a matrix , M of width by height
        and submatrix of dimensions sideLength*sideLength, and any submatrix in M, can have at most maxOnes
        return the number of ones M could have
        
        hints. 1. say you choose some cell (i,j) and set to i, then for all cells (x,y) such that i % sideLength == x % sideLength and
        j % sideLength == y % sideLength
        chose the cells with the max frequency
        '''
        counts = [[0]*width for _ in range(height)]
        counts_first_square_matrix = []
        
        
        for i in range(sideLength):
            for j in range(sideLength):
                count_replicated_Is = 0
                copy_i = i
                while copy_i < height:
                    count_replicated_Is += 1
                    copy_i += sideLength
                
                count_replicated_Js = 0
                copy_j = j
                while copy_j < width:
                    count_replicated_Js += 1
                    copy_j += sideLength
                
                total_replicated_cells = count_replicated_Is*count_replicated_Js
                counts_first_square_matrix.append(total_replicated_cells)
                #print(total_replicated_cells)
                
        counts_first_square_matrix.sort(reverse = True)
        #print(counts_first_square_matrix)
        return sum(counts_first_square_matrix[:maxOnes])
    

class Solution:
    def maximumNumberOfOnes(self, width: int, height: int, sideLength: int, maxOnes: int) -> int:
        '''
        define a matrix , M of width by height
        and submatrix of dimensions sideLength*sideLength, and any submatrix in M, can have at most maxOnes
        return the number of ones M could have
        
        hints. 1. say you choose some cell (i,j) and set to i, then for all cells (x,y) such that i % sideLength == x % sideLength and
        j % sideLength == y % sideLength
        chose the cells with the max frequency
        '''
        counts = [[0]*width for _ in range(height)]
        counts_first_square_matrix = []
        
        
        for i in range(sideLength):
            for j in range(sideLength):
                #for each (i,j) in the first square matrix, count the number of times this cell would be replicated in the other square matrix
                #the crux of the problem is counting the number of times (i,j) is tiled along Matix M
                count_Is_along_rows = 1 + (height - i - 1) // sideLength
                count_Js_along_cols = 1 + (width - j - 1) // sideLength 
                counts_first_square_matrix.append(count_Is_along_rows*count_Js_along_cols)
        
        #get the maxOnes largest
        #to maximimze the ones, just place them in the the ones have that have the largest
        #rather if maxOnes == k, get the k largest
        counts_first_square_matrix.sort(reverse = True)
        print(counts_first_square_matrix)
        return sum(counts_first_square_matrix[:maxOnes])
    
