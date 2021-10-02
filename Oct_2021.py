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

#dp solution space optimized
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        '''
        we can also optimize space 
        instead of using the whole dp array, just keep current and previous rows
        '''
        #first check if text1 doest not reference the shoortest string, swap them
        if len(text2) < len(text1):
            #swap
            text1,text2 = text2,text1
        
        M = len(text1)
        N = len(text2)
        
        prev = [0]*(N+1)
        curr = [0]*(N+1)
        #remember we padded the dp array with one extra col and row with zeros
        for i in range(M-1,-1,-1):
            for j in range(N-1,-1,-1):
                if text1[i] == text2[j]:
                    curr[j] = 1 + prev[j+1]
                else:
                    curr[j] = max(prev[j],curr[j+1])
            #update rwos
            prev,curr = curr,prev
        
        return prev[0]
                
##################################
# 01_OCT_2021
# 1428. Leftmost Column with at Least a One
#############################
#exhausted calls, 
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
#class BinaryMatrix(object):
#    def get(self, row: int, col: int) -> int:
#    def dimensions(self) -> list[]:

class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        '''
        we are given a row sorted binary matrix, so its in a non decreasing order along the rows
        we want the index of the leftmost columnwith a 1 in it
        can only access matrix using the API
        brute force way would be to access all elements in the object
        we want to go down columns starting with leftmost column, if when going down this row, we hit a 1, return that column index
        otherwise -1
        '''
        rows,cols = binaryMatrix.dimensions()
        for col in range(cols):
            for row in range(rows):
                if binaryMatrix.get(row,col) == 1:
                    return col
        return -1

#yessss
class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        '''
        repeatedly making calls to API won't work, cheeky way, cache calls in the constructor first
        then check matrix, but that's stupid
        i could binary search along each row for all rows
        '''
        rows,cols = binaryMatrix.dimensions()
        smallest_i = cols
        for row in range(rows):
            #binary seach along a row
            left = 0
            right = cols - 1
            while left < right:
                mid = left + (right - left) // 2
                val = binaryMatrix.get(row,mid)
                #if its a 1, i can still look in the lower half
                if val == 1:
                    right = mid
                else:
                    left = mid + 1
            if binaryMatrix.get(row,left) == 1:
                smallest_i = min(smallest_i,left)
        
        return smallest_i if smallest_i != cols else -1