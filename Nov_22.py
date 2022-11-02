##############################
# 1706. Where Will the Ball Fall
# 01NOV22
##############################
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        '''
        for each ball dropped from the top of the 0th row and the jth column traces its path
        to find the path we need to examine the current cell as well as the cell to its right
        
        say we are at cell (i,j)
        and grid[i][j] == 1, we need to check grid[i][j+1]
        if grid[i][j] == grid[i][j+1]:
            we can advance down a row in that direction
        
        we can use dfs to track the state of the ball
        keep state (i,j)
        if grid at i,j == 1, it must move right
        so move ball down a row and right 1
        
        if grid at (i,j) == -1, it must move left
        so move ball down  and left
        
        each time we have to check for base cases
        if we got to the last row, return the column
        
        if we go outside the walls or hit a v, the ball can't make it down, return -1
        '''
        rows = len(grid)
        cols = len(grid[0])
        def dfs(row,col):
            #got to the bottom
            if row == rows:
                return col
            
            #going right
            if grid[row][col] == 1:
                #if we can at least check right and go right
                if col < cols-1 and grid[row][col] == grid[row][col+1]:
                    return dfs(row+1,col+1)
                else:
                    return -1
            #going left
            elif grid[row][col] == -1:
                if col > 0 and grid[row][col] == grid[row][col-1]:
                    return dfs(row+1,col-1)
                else:
                    return -1

            
        ans = []
        for i in range(cols):
            ans.append(dfs(0,i))
        
        return ans
            
#also could have done dfs iteratively
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        '''
        for each ball dropped from the top of the 0th row and the jth column traces its path
        to find the path we need to examine the current cell as well as the cell to its right
        
        say we are at cell (i,j)
        and grid[i][j] == 1, we need to check grid[i][j+1]
        if grid[i][j] == grid[i][j+1]:
            we can advance down a row in that direction
        
        we can use dfs to track the state of the ball
        keep state (i,j)
        if grid at i,j == 1, it must move right
        so move ball down a row and right 1
        
        if grid at (i,j) == -1, it must move left
        so move ball down  and left
        
        each time we have to check for base cases
        if we got to the last row, return the column
        
        if we go outside the walls or hit a v, the ball can't make it down, return -1
        '''
        rows = len(grid)
        cols = len(grid[0])
        def find_col(row,col):
            while row < rows and col < cols:
                #got to the bottom
                if row == rows:
                    return col
                #going right
                if grid[row][col] == 1:
                    #if we can at least check right and go right
                    if col < cols-1 and grid[row][col] == grid[row][col+1]:
                        row += 1
                        col += 1
                    else:
                        return -1
                #going left
                elif grid[row][col] == -1:
                	#if we can at least check left
                    if col > 0 and grid[row][col] == grid[row][col-1]:
                        row += 1
                        col -= 1
                    else:
                        return -1
            return col if row == rows else -1

            
        ans = []
        for i in range(cols):
            ans.append(find_col(0,i))
        
        return ans