#########################################
# 2707. Extra Characters in a String
# 02SEP23
#########################################
#YESSSS
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        '''
        break s into substrings in the mostt optimal way such that the number of leftover characters as minimum
        try dp
        if i'm at i, i can technially try all substrings from i to n+1
        try all j, from i+1 to n
        if i has gotten all the way to the end, it means there are i leftover characters
        
        what if a maximize the used characters, then its just N - max_used
        '''
        memo = {}
        dictionary = set(dictionary)
        N = len(s)
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            ans = dp(i+1)
            for j in range(i+1,N+1):
                if s[i:j] in dictionary:
                    ans = max(ans,(j-i) + dp(j))
            
            #ans = max(ans,dp(i+1))
            memo[i] = ans
            return ans
        
        
        return N - dp(0)

#bottom up
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        '''
        '''
        dictionary = set(dictionary)
        N = len(s)
        dp = [0]*(N+1)
        dp[N] = 0
        
        
        for i in range(N-1,-1,-1):
            ans = dp[i+1]
            for j in range(i+1,N+1):
                if s[i:j] in dictionary:
                    ans = max(ans,(j-i) + dp[j])
            
            dp[i] = ans

        
        return N - dp[0]

#for directly solving
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        '''
        for direclty solving, if we cant match a character we add 1
        otherwise take the minimum of the nexdt
        '''
        
        memo = {}
        dictionary = set(dictionary)
        N = len(s)
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            ans = dp(i+1) + 1
            for j in range(i+1,N+1):
                if s[i:j] in dictionary:
                    ans = min(ans,dp(j))
            

            memo[i] = ans
            return ans
        
        
        return dp(0)
    
#insteaf of using hahset build up a trie
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        root = self.buildTrie(dictionary)
        
        @cache
        def dp(start):
            if start == n:
                return 0
            # To count this character as a left over character 
            # move to index 'start + 1'
            ans = dp(start + 1) + 1
            node = root
            for end in range(start, n):
                if s[end] not in node.children:
                    break
                node = node.children[s[end]]
                if node.is_word:
                    ans = min(ans, dp(end + 1))
            return ans
        
        return dp(0)
    
    def buildTrie(self, dictionary):
        root = TrieNode()
        for word in dictionary:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.is_word = True
        return root
