####################################
#01_OCT_2021
#1143. Longest Common Subsequence
###################################
#recursive knapsack solution
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        '''
        i can use recursion, two pointers i and j,
        if i and j match, advance them both, if they dont, there are two options
        stay at i, move j
        stat at j, move i
        if either of these pointers got the end, we have a subsequence, which is just
        '''
        
        memo = {}
        def rec(i,j):
            #ending case
            if i == len(text1) or j == len(text2):
                return 0
            #retrieve
            if (i,j) in memo:
                return memo[(i,j)]

            #matching, 1 + advance both
            if text1[i] == text2[j]:
                res = 1 + rec(i+1,j+1)
                memo[(i,j)] = res
                return res
            
            #two options
            first = rec(i+1,j)
            second = rec(i,j+1)
            res = max(first,second)
            memo[(i,j)] = res
            return res
        
        return rec(0,0)

#dp solution, 2d array
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        '''
        dp solution, just translate from the top down recursive approach
        remember we work backwards
        '''
        M = len(text1)
        N = len(text2)
        
        dp = [[0]*(N+1) for _ in range(M+1)]
        #remember we padded the dp array with one extra col and row with zeros
        for i in range(M-1,-1,-1):
            for j in range(N-1,-1,-1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i+1][j+1]
                else:
                    dp[i][j] = max(dp[i+1][j],dp[i][j+1])
        
        return dp[0][0]
