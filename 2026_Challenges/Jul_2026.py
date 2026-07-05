#####################################
# 3286. Find a Safe Walk Through a Grid
# 02JUL26
###########################################
class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        '''
        this is just 01 bfs/djiktras
        we need to store the beast
        need to minimize damage taken along the apth for djikstras
        '''
        rows, cols = len(grid), len(grid[0])

        dirs = [(0,1),(0,-1),(1,0),(-1,0)]

        dist = [[float("inf")] * cols for _ in range(rows)]
        dist[0][0] = grid[0][0]

        pq = [(grid[0][0], 0, 0)]

        while pq:
            damage, i, j = heapq.heappop(pq)

            if damage > dist[i][j]:
                continue

            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    nd = damage + grid[ni][nj]
                    if nd < dist[ni][nj]:
                        dist[ni][nj] = nd
                        heapq.heappush(pq, (nd, ni, nj))

        #also do
        #return health - dist[rows-1][cols-1] >= 1
        return dist[rows - 1][cols - 1] < health
    
#0/1 BFS
from collections import deque
from typing import List

class Solution:
    def findSafeWalk(self, grid: List[List[int]], health: int) -> bool:
        rows, cols = len(grid), len(grid[0])

        health -= grid[0][0]
        if health < 1:
            return False

        best = [[-1] * cols for _ in range(rows)]
        dq = deque([(0, 0, health)])
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while dq:
            i, j, hp = dq.popleft()

            if hp <= best[i][j]:
                continue
            best[i][j] = hp

            if (i, j) == (rows - 1, cols - 1):
                return True

            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    nhp = hp - grid[ni][nj]
                    if nhp < 1:
                        continue

                    if grid[ni][nj] == 0:
                        dq.appendleft((ni, nj, nhp))
                    else:
                        dq.append((ni, nj, nhp))

        return False
    
#################################################
# 3620. Network Recovery Pathways
# 04JUL26
################################################
class Solution:
    def findMaxPathScore(self, edges: List[List[int]], online: List[bool], k: int) -> int:
        '''
        we have a DAG
        need max path score
            score is min edge on path, and total path score does need exceed k
            can only use online nodes
        binary seach on ans using djikstras
        when doing dijsktrs only include edges >= mid
        '''
        def dijkstras(edges,online,k,min_edge):
            n = len(online)
            graph = defaultdict(list)
            for u,v,w in edges:
                #filter
                if online[u] and online[v] and w >= min_edge:
                    graph[u].append((v,w))
            
            dists = [float('inf')]*n
            dists[0] = 0
            pq = [(0,0)]

            while pq:
                d,u = heapq.heappop(pq)
                if d > dists[u]:
                    continue
                for v,w in graph[u]:
                    nd = d + w
                    if nd < dists[v]:
                        dists[v] = nd
                        heapq.heappush(pq, (nd,v))
                        
            if dists[n-1] == float('inf') or dists[n-1] > k:
                return False
            return True

        #edge case...

        left,right =  0, max([w for _,_,w in edges]) if edges else 0
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            can_do = dijkstras(edges,online,k,mid)
            if can_do:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans
    
######################################################
# 2492. Minimum Score of a Path Between Two Cities
# 04JUL24
########################################################
#revisited
#TLE
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        dfs from 1 to n
        there is a pathway always from 1 to n
        '''
        graph = defaultdict(list)
        for u,v,w in roads:
            graph[u].append((v,w))
            graph[v].append((u,w))
        
        ans = [float('inf')]
        def dfs(curr,parent,seen,d):
        
            seen.add(curr)
            for neigh,dist in graph[curr]:
                if neigh != parent and neigh not in seen:
                    ans[0] = min(ans[0],dist)
                    dfs(neigh,curr,seen,dist)
            seen.remove(curr)
        
        seen = set()
        dfs(1,-1,seen,float('inf'))
        return ans[0]
    
class Solution:
    def minScore(self, n: int, roads: List[List[int]]) -> int:
        '''
        dfs from 1 to n
        there is a pathway always from 1 to n
        find nodes on path , then take the smallest edges
        '''
        graph = defaultdict(list)
        for u,v,w in roads:
            graph[u].append((v,w))
            graph[v].append((u,w))

        on_path = set()
        seen = set()

        def dfs(curr,parent,seen):
            on_path.add(curr)
            seen.add(curr)
            for neigh,dist in graph[curr]:
                if neigh != parent and neigh not in seen:
                    dfs(neigh,curr,seen)
            

        dfs(1,-1,seen)
        ans = float('inf')
        for u,v,w in roads:
            if u in on_path or v in on_path:
                ans = min(ans,w)
        return ans
    
#######################################################
# 1301. Number of Paths with Max Score
# 05JUL26
########################################################
#TLE, close though
class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        '''
        this is just dp
        we can only go left or up, or left up
        '''
        n = len(board)
        dirrs = [(0,-1),(-1,0),(-1,-1)]
        mod = 10**9 + 7

        #find max score
        def dp1(i, j):
            if board[i][j] == 'E':
                return 0

            if (i, j) in memo:
                return memo[(i, j)]

            ans = -float("inf")

            for di, dj in dirrs:
                ii, jj = i + di, j + dj

                if (0 <= ii < n and 0 <= jj < n and board[ii][jj] != 'X'):
                    score = dp1(ii, jj)
                    if score == -float("inf"):
                        continue

                    if board[ii][jj].isdigit():
                        score += int(board[ii][jj])

                    ans = max(ans, score)

            memo[(i, j)] = ans
            return ans

        def dp2(i, j, curr_sum):
            if board[i][j] == 'E':
                return 1 if curr_sum == max_score else 0

            if (i, j, curr_sum) in memo:
                return memo[(i, j, curr_sum)]

            ways = 0

            for di, dj in dirrs:
                ii, jj = i + di, j + dj

                if (0 <= ii < n and 0 <= jj < n and board[ii][jj] != 'X'):
                    nxt = curr_sum
                    if board[ii][jj].isdigit():
                        nxt += int(board[ii][jj])

                    ways += dp2(ii, jj, nxt)

            memo[(i, j, curr_sum)] = ways % mod
            return memo[(i, j, curr_sum)]
                
        memo = {}
        max_score = dp1(n-1,n-1)
        if max_score == float("-inf"):
            return [0,0]
        memo = {}
        num_ways = dp2(n-1,n-1,0)
        return [max_score,num_ways]
    
#need to make into one function
class Solution:
    def pathsWithMaxScore(self, board: List[str]) -> List[int]:
        '''
        this is just dp
        we can only go left or up, or left up
        '''
        n = len(board)
        dirrs = [(0,-1),(-1,0),(-1,-1)]
        mod = 10**9 + 7

        #find max score
        def dp(i, j):
            if board[i][j] == 'E':
                return [0,1]

            if (i, j) in memo:
                return memo[(i, j)]

            best = -float("inf")
            ways = 0

            for di, dj in dirrs:
                ii, jj = i + di, j + dj

                if (0 <= ii < n and 0 <= jj < n and board[ii][jj] != 'X'):
                    child_score,child_ways = dp(ii, jj)
                    if child_score == -float("inf"):
                        continue
                    candidate = child_score
                    if board[ii][jj].isdigit():
                        candidate += int(board[ii][jj])

                    if candidate > best:
                        best = candidate
                        ways = child_ways
                    elif candidate == best:
                        ways += child_ways

            entry = [best, ways % mod]
            memo[(i, j)] = entry
            return entry
        
        memo = {}
        score,ways = dp(n-1,n-1)
        if score == float('-inf'):
            return [0,0]
        
        return [score,ways]

