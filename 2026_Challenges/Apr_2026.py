######################################################
# 3418. Maximum Amount of Money Robot Can Earn
# 02APR26
######################################################
#top down gets MLE
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        '''
        dp, states are (i,j,k)
        netrualize or don't netrualize
        '''
        memo = {}
        rows, cols = len(coins), len(coins[0])

        def dp(i, j, k):
            if (i, j) == (rows - 1, cols - 1):
                if coins[i][j] >= 0:
                    return coins[i][j]
                else:
                    if k > 0:
                        return 0
                    else:
                        return coins[i][j]

            if (i, j, k) in memo:
                return memo[(i, j, k)]

            ans = float('-inf')

            if coins[i][j] >= 0:
                if i + 1 < rows:
                    ans = max(ans, coins[i][j] + dp(i + 1, j, k))
                if j + 1 < cols:
                    ans = max(ans, coins[i][j] + dp(i, j + 1, k))
            else:
                # dont neutralize
                if i + 1 < rows:
                    ans = max(ans, coins[i][j] + dp(i + 1, j, k))
                if j + 1 < cols:
                    ans = max(ans, coins[i][j] + dp(i, j + 1, k))

                # option 2: neutralize if we can
                if k > 0:
                    if i + 1 < rows:
                        ans = max(ans, dp(i + 1, j, k - 1))
                    if j + 1 < cols:
                        ans = max(ans, dp(i, j + 1, k - 1))

            memo[(i, j, k)] = ans
            return ans

        return dp(0, 0, 2)
    
#yesssss
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        '''
        dp, states are (i,j,k)
        netrualize or don't netrualize
        '''
        rows, cols = len(coins), len(coins[0])
        dp = [[[float('-inf')]*3 for _ in range(cols)] for _ in range(rows)]

        #base case fill
        for i in range(rows):
            for j in range(cols):
                for k in range(3):
                    if (i, j) == (rows - 1, cols - 1):
                        if coins[i][j] >= 0:
                            dp[i][j][k] = coins[i][j]
                        else:
                            if k > 0:
                                dp[i][j][k] = 0
                            else:
                                dp[i][j][k] = coins[i][j]
        
        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                if (i, j) == (rows - 1, cols - 1):
                    continue
                for k in range(3):
                    ans = float('-inf')
                    if coins[i][j] >= 0:
                        if i + 1 < rows:
                            ans = max(ans, coins[i][j] + dp[i + 1][j][k])
                        if j + 1 < cols:
                            ans = max(ans, coins[i][j] + dp[i][j + 1][k])
                    else:
                        # dont neutralize
                        if i + 1 < rows:
                            ans = max(ans, coins[i][j] + dp[i + 1][j][k])
                        if j + 1 < cols:
                            ans = max(ans, coins[i][j] + dp[i][j + 1][k])

                        # option 2: neutralize if we can
                        if k > 0:
                            if i + 1 < rows:
                                ans = max(ans, dp[i + 1][j][k - 1])
                            if j + 1 < cols:
                                ans = max(ans, dp[i][j + 1][k - 1])

                    dp[i][j][k] = ans

     
        return dp[0][0][2]