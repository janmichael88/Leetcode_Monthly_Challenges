#########################################
# 3660. Jump Game IX
# 07MAY26
#############################################
#cant just take max of left and right sides, bcause there could be paths leading to anothing index
class Solution:
    def maxValue(self, nums: List[int]) -> List[int]:
        '''
        if im at i, i can jump to j if
        * j > i and nums[j] < nums[i]
        * j < i and nums[j] > nums[i]
        '''
        ans = nums[:]
        n = len(nums)

        for i in range(n):
            best = nums[i]
            for j in range(i):
                if nums[j] > nums[i]:
                    best = max(best,nums[j])
            for j in range(i+1,n):
                if nums[j] < nums[i]:
                    best = max(best,nums[j])
            ans[i] = best
        return ans
    
class Solution:
    def maxValue(self, nums: List[int]) -> List[int]:
        '''
        notice that forward jumps lead to smaller values
        and backward jumps lead to larger values
        compute pref_max and suff min and look for the cuts
        '''
        n = len(nums)
        pref_max = nums[:]
        suff_min = nums[:]

        for i in range(1,n):
            pref_max[i] = max(pref_max[i-1],nums[i])

        for i in range(n-2,-1,-1):
            suff_min[i] = min(suff_min[i+1],nums[i])
        
        #find cuts
        ans = pref_max[:] 
        #we have at least pref_max for each ans[i]
        for i in range(n-2,-1,-1):
            if pref_max[i] > suff_min[i+1]:
                ans[i] = ans[i+1]
        return ans

#################################################
# 1824. Minimum Sideway Jumps
# 09MAY26
#################################################
class Solution:
    def minSideJumps(self, obstacles: List[int]) -> int:
        '''
        frog can onle jump from point i to i+1 on the same lane if there is not ob stacle on te aline at point i + 1
        to avoid obstalces, frog can side jump to another lane, even if they arent adjacent
        no obstalces at point 0 and point n
        '''
        n = len(obstacles)
        INF = float('inf')

        # dp[i][lane]
        # minimum jumps to reach position i in lane
        dp = [[INF] * 3 for _ in range(n)]

        # starting position
        dp[0][0] = 1   # lane 1
        dp[0][1] = 0   # lane 2
        dp[0][2] = 1   # lane 3

        for i in range(1, n):

            # first: move forward in same lane
            for lane in range(3):

                # obstacle blocks this lane
                if obstacles[i] == lane + 1:
                    continue

                dp[i][lane] = dp[i - 1][lane]

            # second: try side jumps at same position
            for lane in range(3):

                # can't stand on obstacle
                if obstacles[i] == lane + 1:
                    continue

                for other in range(3):
                    #could comment out this block
                    #
                    if lane == other:
                        continue
                    #
                    # other lane also must not be blocked
                    if obstacles[i] == other + 1:
                        continue

                    dp[i][lane] = min(dp[i][lane],dp[i][other] + 1)

        return min(dp[n - 1])
    
#############################################
# 1914. Cyclically Rotating a Grid
# 09MAY26
#############################################
class Solution:
    def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        '''
        each layer is an array
        rorate k times and assign them back in grid
        this problem is not fun
        '''
        rows, cols = len(grid), len(grid[0])
        layers = []
        curr_row, curr_col = 0, 0
        end_row, end_col = rows - 1, cols - 1

        while curr_row <= end_row and curr_col <= end_col:
            arr = []
            # go down
            for row in range(curr_row, end_row + 1):
                arr.append(grid[row][curr_col])
            # go right
            for col in range(curr_col + 1, end_col + 1):
                arr.append(grid[end_row][col])
            # go up
            if curr_col < end_col:
                for row in range(end_row - 1, curr_row - 1, -1):
                    arr.append(grid[row][end_col])
            # go left
            if curr_row < end_row:
                for col in range(end_col - 1, curr_col, -1):
                    arr.append(grid[curr_row][col])
            #rotate
            n = len(arr)
            rotated = arr[-(k%n):] + arr[:-(k%n)]
            layers.append(rotated)

            curr_row += 1
            curr_col += 1
            end_row -= 1
            end_col -= 1

        #one more time and put back in
        curr_row, curr_col = 0, 0
        end_row, end_col = rows - 1, cols - 1
        curr_layer = 0
        while curr_row <= end_row and curr_col <= end_col:
            arr = layers[curr_layer]
            idx = 0
            # go down
            for row in range(curr_row, end_row + 1):
                grid[row][curr_col] = arr[idx]
                idx += 1
            # go right
            for col in range(curr_col + 1, end_col + 1):
                grid[end_row][col] = arr[idx]
                idx += 1
            # go up
            if curr_col < end_col:
                for row in range(end_row - 1, curr_row - 1, -1):
                    grid[row][end_col] = arr[idx]
                    idx += 1
            # go left
            if curr_row < end_row:
                for col in range(end_col - 1, curr_col, -1):
                    grid[curr_row][col] = arr[idx]
                    idx += 1

            curr_row += 1
            curr_col += 1
            end_row -= 1
            end_col -= 1
            curr_layer += 1
        
        return grid

#############################################################
# 2770. Maximum Number of Jumps to Reach the Last Index
# 10MAY26
###########################################################
class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        '''
        dp
        '''
        memo = {}
        n = len(nums)

        def dp(i):
            if i == n-1:
                return 0
            if i in memo:
                return memo[i]
            ans = float('-inf')
            for j in range(i+1,n):
                if -target <= nums[j] - nums[i] <= target:
                    ans = max(ans, 1 + dp(j))
            memo[i] = ans
            return ans
        
        ans = dp(0)
        if ans == float('-inf'):
            return -1
        return ans
    
##################################################
# 2553. Separate the Digits in an Array
# 11MAY26
##################################################
class Solution:
    def separateDigits(self, nums: List[int]) -> List[int]:
        '''
        pop off
        '''
        ans = []
        for num in nums:
            digs = []
            while num > 0:
                digs.append(num % 10)
                num = num // 10
            
            ans.extend(digs[::-1])
        
        return ans