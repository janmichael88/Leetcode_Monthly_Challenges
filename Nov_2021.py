#####################################
# 01NOV21
# 130. Surrounded Regions
#####################################
#fuck yeah
#dfs solution
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        this is simlar to capturing islands
        note: surrounded regions should not be on the border
        dfs from all the O's on the borders and, since these cannot be captured, add  the zeros to the non capture group
        then re traverse the grid and if an O is not in the capture group change it
        '''
        rows = len(board)
        cols = len(board[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        non_capture_zeros = set()
        
        def dfs(r,c):
            #add to non_capture
            non_capture_zeros.add((r,c))
            for dx,dy in dirrs:
                neigh_r = r + dx
                neigh_c = c + dy
                #bounds check
                if 0 <= neigh_r < rows and 0 <= neigh_c < cols:
                    #is an 'O'
                    if board[neigh_r][neigh_c] == 'O': 
                        #is not seen
                        if (neigh_r,neigh_c) not in non_capture_zeros:
                            dfs(neigh_r,neigh_c)
            return
                        
        #now dfs along borders
        #first row
        for c in range(cols):
            if board[0][c] == "O":
                dfs(0,c)
                
        #first col
        for r in range(rows):
            if board[r][0] == "O":
                dfs(r,0)
        
        #last row
        for c in range(cols):
            if board[rows-1][c] == 'O':
                dfs(rows-1,c)
        #last col
        for r in range(rows):
            if board[r][cols-1] == 'O':
                dfs(r,cols-1)
                
        #now pass board and make sure this O is a non capture
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    if (r,c) not in non_capture_zeros:
                        board[r][c] = 'X'
                        
        return board
        print(non_capture_zeros)

#bfs
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        this is simlar to capturing islands
        note: surrounded regions should not be on the border
        dfs from all the O's on the borders and, since these cannot be captured, add  the zeros to the non capture group
        then re traverse the grid and if an O is not in the capture group change it
        '''
        rows = len(board)
        cols = len(board[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        non_capture_zeros = set()
        
        def bfs(r,c):
            #add to non_capture
            non_capture_zeros.add((r,c))
            q = deque([(r,c)])
            while q:
                curr_r,curr_c = q.popleft()
                for dx,dy in dirrs:
                    neigh_r = curr_r + dx
                    neigh_c = curr_c + dy
                    #bounds check
                    if 0 <= neigh_r < rows and 0 <= neigh_c < cols:
                        #is an 'O'
                        if board[neigh_r][neigh_c] == 'O': 
                            #is not seen
                            if (neigh_r,neigh_c) not in non_capture_zeros:
                                q.append((neigh_r,neigh_c))
                                non_capture_zeros.add((neigh_r,neigh_c))
                        
        #now dfs along borders
        #first row
        for c in range(cols):
            if board[0][c] == "O":
                bfs(0,c)
                
        #first col
        for r in range(rows):
            if board[r][0] == "O":
                bfs(r,0)
        
        #last row
        for c in range(cols):
            if board[rows-1][c] == 'O':
                bfs(rows-1,c)
        #last col
        for r in range(rows):
            if board[r][cols-1] == 'O':
                bfs(r,cols-1)
                
        #now pass board and make sure this O is a non capture
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    if (r,c) not in non_capture_zeros:
                        board[r][c] = 'X'
                        
        return board
        print(non_capture_zeros)

########################
# 01NOV21
# 1231. Divide Chocolate
########################