###############################################
# 1536. Minimum Swaps to Arrange a Binary Grid
# 02MAR26
################################################
class Solution:
    def minSwaps(self, grid: List[List[int]]) -> int:
        '''
        grid is square
        we can only swap to adjacent rows
        for i in range(1,n):
            count_zeros_at_col(i) >= 1, there can be zeros below main daig, but need to make zeros above main diag
        count zeros for each row to the right, until we hit a wone
        '''
        n = len(grid)
        count_zeros = [0]*n

        for i in range(n):
            for j in range(n-1,-1,-1):
                if grid[i][j] == 0:
                    count_zeros[i] += 1
                else:
                    break
        
        #each row i, should have at least n-i zeros in ot
        #if it does, we are good, otherwise, find the the closest row to it with at least n-i zeros to the right
        ans = 0
        print(count_zeros)
        for i in range(n):
            j = i
            while j < n and count_zeros[j] < n - i - 1:
                j += 1
            if j == n:
                return -1
            ans += j - i
            #swap back up to i
            while j > i:
                count_zeros[j],count_zeros[j-1] = count_zeros[j-1],count_zeros[j]
                j -= 1
        
        return ans
