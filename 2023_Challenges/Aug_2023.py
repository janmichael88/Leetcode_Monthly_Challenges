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
