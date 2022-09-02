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