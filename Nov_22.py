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

############################################################
# 1198. Find Smallest Common Element in All Rows (REVISTED)
# 02NOV22
###########################################################
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        for each element in the first row, binary search for it in the other rows
        
        '''
        to_search = mat[0]
        for num in to_search:
            check = 0
            for row in mat[1:]:
                #look for it
                idx = bisect_left(row,num)
                if (0 <= idx < len(row)) and num == row[idx]:
                    check += 1
            
            if check == len(mat) - 1:
                return num
        
        return -1


class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        since the numbers in the rows are sorted in strictly increasing order, we can just count up the numbers
        then traverse the the numbers in order and return the one that has count at least n
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        # know that largest number
        counts = [0]*(10**4 + 1)
        for i in range(rows):
            for j in range(cols):
                num = mat[i][j]
                counts[num] += 1
        
        for num in range(1, len(counts)):
            if counts[num] == rows:
                return num
        
        return -1

class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        we can speed things up by going down a column
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        # know that largest number
        counts = [0]*(10**4 + 1)
        for j in range(cols):
            for i in range(rows):
                num = mat[i][j]
                counts[num] += 1
                if counts[num] == rows:
                    return num
        
        return -1

#using merge k sorted lists
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        we can treat this like the merge k sorted problem
        for each we track the position of the current element starting from zero
        then we find the smallest element among all positions and advance the position for the row that we are on
        
        algo:
            1. init row positions, current max and counter (we are couting that have at least the maximum)
            2. for each row:
                increment the row position until the value is equal or greater than the current max
                    if we reach the end of a row, return -1
                if the value equals the current max, increase the counter
                    otherwise reset counter to 1 and update max
                
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        
        #store poisitions of each row int an array
        positions = [0]*rows
        curr_max = 0
        count = 0
        
        while True:
            for i in range(rows):
                #keep advancing along a row
                curr_pos = positions[i]
                while curr_pos < cols and mat[i][curr_pos] < curr_max:
                    curr_pos += 1
                positions[i] = curr_pos
                
                #if we have reached the end of a row
                if positions[i] >= cols:
                    return -1
                
                if mat[i][positions[i]] != curr_max:
                    count = 1
                    curr_max = mat[i][positions[i]]
                
                else:
                    count += 1
                    if count == rows:
                        return curr_max

#abusing set notation
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        we can also use sets
        
        '''
        #turn this into iter object
        mat = iter(mat)
        temp = set(next(mat)) #grabs the first row and advance mat
        for row in mat:
            temp = temp & set(row)
        
        return min(temp,default=-1)




#################################
# 951. Flip Equivalent Binary Trees
# 02NOV22
#################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        '''
        if there is a flip in a a node of tree
        then it should be the case that at the flipped node
        (node1.left = node2.right) == (node1.right == node2.left)
        problem is we can so some number of flips
        
        if there is say N number of flips, then the permuations in a level should be the same
        rather perm of a level in tree 1 should be prem of that same level in tree 2
        actually nope this case doesn't hold
        
        
        4 5 6 7
        5 4 6 7
        5 4 7 6
        4 5 7 6
        6 7 4 5
        7 6 4 5
        7 6 5 4
        6 7 4 5
        
        gahh, straigh up tricky recursion
        
        rules:
            root1 and root2 have the same root value, then we only need to check if their children are equal
            1. if root1 or root2 is null, then they are equivlanet if  and only if they are both null
            2. else if root1 and root have different values, they ar enot equivalent
            3. else, lets check whether the children of root1 are equivalent to the children of root 2
            thtere are two ways to pair these chidlren
        '''
        def dp(p,q):
            #base case both are none, this is trieivally true
            if not p and not q:
                return True
            #if only one is empty, and the values don't match, not true
            if not p or not q or p.val != q.val:
                return False
            #check the options
            op1 = dp(p.left,q.left)
            op2 = dp(p.right,q.right)
            op3 = dp(p.left,q.right)
            op4 = dp(p.right,q.left)
            
            return (op1 and op2) or (op3 and op4)
        
        return dp(root1,root2)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        '''
        there is actually a characteristic for binary trees where all the nodes have unique values
        flip each node such that the left child < right child
            call this the canonical representation
            all equivlant trees have eactly one
        
        preorder both trees, for empty nodes make value -1
        '''
        vals1 = []
        vals2 = []
        
        def dfs(node,vals):
            if node:
                vals.append(node.val)
                left = node.left.val if node.left else -1
                right = node.right.val if node.right else -1
            
                if left < right:
                    dfs(node.left,vals)
                    dfs(node.right,vals)
                else:
                    dfs(node.right,vals)
                    dfs(node.left,vals)
            else:
                vals.append(None)
                
        dfs(root1,vals1)
        dfs(root2,vals2)
        
        return vals1 == vals2