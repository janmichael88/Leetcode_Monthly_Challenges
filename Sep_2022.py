###############################################
# 2271. Maximum White Tiles Covered by a Carpet
# 31AUG22
###############################################
#binary search coded out, we want upper mid
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        '''
        https://leetcode.com/problems/maximum-white-tiles-covered-by-a-carpet/discuss/2038177/PythonGreedy-%2B-prefix-sum-%2B-binary-search-easy-to-understand-with-explanation
        prefix sum and binary search to find the largest
        1. sort tiles by starting position
        2. build pref sum to store lenght sum of tiles
        3. traverse each tile, given its starting s, we know that the further postion the carpetn can cover is s + carepentLen-1
        4. having the furthest position, we binary search the idnex of the ending tile that the carpent can partially cover
            we greedily align the starting position of the carpet with teh starting position of the tile
        5. calculate the length of the ending tile that the carpet cannot partially cover, we need to subtract this from the last part
        '''
        #sort on starting
        tiles.sort(key = lambda x: x[0])
        #store starting position array
        startPos = [tile[0] for tile in tiles]
        #build pref sum array
        prefSum = [0]
        for s,e in tiles:
            prefSum.append(prefSum[-1] + (e - s) + 1)
            
        #helper function for binary search
        def binarySearch(arr,start,end,target):
            if start == end:
                if target == arr[start]:
                    return start
                else:
                    return end
            
            mid = start + (end - start) // 2
            if target <= arr[mid]:
                return binarySearch(arr,start,mid,target)
            else:
                return binarySearch(arr,mid+1,end,target)
            
        
        ans = 0
        N = len(tiles)
        for i in range(N):
            start,end = tiles[i]
            #if we can at least catch all tiles here, we done
            if end >= start + carpetLen - 1:
                return carpetLen
            #otherwise binary search the index of the ending tile that the carpetn can partially cover, we want the upper mid
            endIndex = binarySearch(startPos,0,len(startPos)-1,start+carpetLen-1)
            #get the length of the tiles the carpet cannot cover
            cantCover = 0
            if tiles[endIndex][1] > start + carpetLen - 1:
                cantCover = tiles[endIndex][1] - (start + carpetLen - 1)
            ans = max(ans,prefSum[endIndex+1] - prefSum[i] - cantCover)
        
        return ans
        

#using bisect right
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        # sort the tiles by the starting position
        tiles.sort(key = lambda x:x[0])
        # build the starting position array
        startPos = [tiles[i][0] for i in range(len(tiles))]
        # build the prefix sum array
        preSum = [0] * (len(tiles) + 1)
        for i in range(1, len(tiles) + 1):
            preSum[i] = preSum[i - 1] + (tiles[i-1][1]-tiles[i-1][0] + 1)
        
        res = 0
        for i in range(len(tiles)):
            s, e = tiles[i]
            # if the length of tile >= length of carpet, return carpetLen
            if e >= s + carpetLen - 1:
                return carpetLen
            # binary search the index of the ending tile that the carpet can partially cover
            endIdx = bisect_right(startPos, s + carpetLen - 1) - 1
            # calculate the length of the ending tile that the carpet cannot cover 
            compensate = 0
            if tiles[endIdx][1] > s + carpetLen - 1:
                compensate = tiles[endIdx][1] - s - carpetLen + 1
            # update the result
            res = max(res, preSum[endIdx+1] - preSum[i] - compensate)
            
        return res

#sort and sliding window with two cases
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        '''
        https://leetcode.com/problems/maximum-white-tiles-covered-by-a-carpet/discuss/2038674/Python-Explanation-with-pictures-sliding-window
        the trick is to realize (rather convince yourself the placing a carpet at the beignning of the range) will give you the maximum size
        we have two cases
        1. the right end is in the middle of some tiles
            covered = pref[j] - pref[i]
        2. the right end lies in section of tiles
            covered = pref[j+1] - prev[i] - (ends[j] - j)
        '''
        #sort on start
        tiles.sort(key = lambda x: x[0])
        #prefsum
        prefSum = [0]
        for s,e in tiles:
            prefSum.append(prefSum[-1] + (e - s) + 1)
            
        ends = [e for s,e in tiles]
        N = len(ends)
        ans = 0
        j = 0
        for i in range(N):
            #carpet start from the beginning of each range
            start,end = tiles[i]
            #the right most index have tiles is ends[-1]
            right_most = min(ends[-1],start+carpetLen-1)
            
            #while the whole current rang is covered by carpet
            while j < N and ends[j] < right_most:
                j += 1
            #two cases
            #first case, if the right end of the carept doesn't reach the jth rang
            if tiles[j][0] > right_most:
                ans = max(ans, prefSum[j] - prefSum[i])
            #the right end of the carpert covers parrt of it, there is some tiles that are left over
            else:
                ans = max(ans, prefSum[j+1] - prefSum[i] - ends[j] + right_most )
        
        return ans

############################
# 637. Average of Levels in Binary Tree (REVISITED)
# 02SEP22
############################
#without having to do second pass
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        '''
        us bfs then explore level by level
        '''
        averages = []
        q = deque([root])
        
        while q:
            N = len(q)
            level_sum = 0
            for _ in range(N):
                node = q.popleft()
                level_sum += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            
            averages.append(level_sum / N)
        
        return averages


################################
# 2187. Minimum Time to Complete Trips
# 02SEP22
################################
#TLE
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        '''
        if i'm givein the time array, example [1,2,3]
        t = 1
            [1,0,0], trips = 1
        t = 2
            [2,1,0], trips = 3
        t = 3
            [3,1,1], trips = 5
        answer = 3
        
        given a time t, i want to compute number of trips in the whole array
        then use binary search to find the lower bound where num trips for i just less than totalTrips
        O(len(times)) to compute trips for time t
        then log(maxtime) for getting the minimu time
        i'll maximize using 2**32 first
        '''
        def getNumTrips(t):
            trips = 0
            for i in range(len(time)):
                trips += (t // time[i])
            
            return trips
        
        start = 0
        
        while getNumTrips(start) < totalTrips:
            start += 1
        
        return start
    
#YAYYYY, you just needed to watch the upper bound
class Solution:
    def minimumTime(self, time: List[int], totalTrips: int) -> int:
        '''
        if i'm givein the time array, example [1,2,3]
        t = 1
            [1,0,0], trips = 1
        t = 2
            [2,1,0], trips = 3
        t = 3
            [3,1,1], trips = 5
        answer = 3
        
        given a time t, i want to compute number of trips in the whole array
        then use binary search to find the lower bound where num trips for i just less than totalTrips
        O(len(times)) to compute trips for time t
        then log(maxtime) for getting the minimu time
        i'll maximize using 2**32 first

        upper bound is min(time)*totalTrips
        in the worst caes we only use the fasts bus
        '''
        def getNumTrips(t):
            trips = 0
            for i in range(len(time)):
                trips += (t // time[i])
            
            return trips
        
        
        start = 0
        end = min(time)*totalTrips
        
        while start < end:
            mid = start + (end - start) // 2
            #guess
            guess = getNumTrips(mid)
            if guess >= totalTrips:
                #dont need to look beyond end anymore
                end = mid
            else:
                start = mid+1
        
        return start

#one liner solution just to show off
def minimumTime(self, time: List[int], totalTrips: int) -> int:
    return bisect_left(range(1, 10**14), totalTrips, key= lambda x: sum(x // t for t in time)) + 1
####################################
# 1207. Unique Number of Occurrences
# 02SEP22
####################################
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        '''
        get count mapp the second pass with hashset
        i could have also sorted the counts and check for repeated values
        '''
        counts = Counter(arr)
        seen = set()
        
        for c in counts.values():
            if c in seen:
                return False
            seen.add(c)

        return True
        
##############################
# 1929. Concatenation of Array
# 02SEP22
##############################
class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        '''
        preallocate the ans array and point two pointers
        one at i and the other at i+1
        '''
        N = len(nums)
        ans = [0]*(2*N)
        for i in range(N):
            ans[i] = nums[i]
            ans[N+i] = nums[i]
        
        return ans

############################################################
# 967. Numbers With Same Consecutive Differences (REVISITED)
# 03SEP22
############################################################
#backtracking, time complexity if O(2**n)
#from a node in the executino tree, we have at most 2 children, 
#num + k, and num -k
#binary stree with depth N-1, and 2 children nodes          
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        '''
        note the conditions, n is in the closed range [2,9]
        i can try building a digit
        for (n = 2, k = 0)
            [11,22,33,44,55,66,77,88,99]
        for (n=2, k = 1)
            [10,12,21,23,32,34,43,45,54,56,65,67,76,78,87,89,98]
            
        i need to try building a digit using backtracking, contraints are small enough to allow for this
        i can start the function offf using each number [0,9] then rebuild
        just try building the string for now, worry about optimizations later
        '''
        paths = []
        n -= 1
        
        def backtrack(n,path):
            if n == 0:
                paths.append(int("".join(path)))
                return
            last_digit = int(path[-1])
            for next_digit in range(0,10):
                if abs(last_digit - next_digit) == k:
                    path += str(next_digit)
                    backtrack(n-1,path)
                    path.pop()
        
        for i in range(1,10):
            backtrack(n,[str(i)])
        return paths

class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        '''
        dfs without backtracking, being explicity when to call
        '''
        if n == 1:
            return [i for i in range(10)]
        
        ans = []
        
        def dfs(n,num):
            if n == 0:
                ans.append(num)
                return
            
            last_digit = num % 10
            next_digits = set([last_digit+k,last_digit-k])
            
            for foo in next_digits:
                if 0 <= foo < 10:
                    new_num = num*10 + foo
                    dfs(n-1,new_num)
                    
        
        for num in range(1,10):
            dfs(n-1,num)
        
        return ans
            
#bfs
class Solution:
    def numsSameConsecDiff(self, n: int, k: int) -> List[int]:
        '''
        we can also use bfs and just generate the numbers layer by layer and returnt he final layer
        '''
        if n == 1:
            return [i for i in range(10)]
        
        q = [num for num in range(1,10)]
        
        #we have the first level done, so there are at least N-1 levels left
        for level in range(n-1):
            next_level = []
            for num in q:
                last_digit = num % 10
                next_digits = set([last_digit+k,last_digit-k])
            
                for foo in next_digits:
                    if 0 <= foo < 10:
                        new_num = num*10 + foo
                        next_level.append(new_num)
            
            q = next_level
        
        return q
