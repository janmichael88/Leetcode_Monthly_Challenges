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

##################################################
# 759. Employee Free Time
# 11MAY26
###################################################
"""
# Definition for an Interval.
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end
"""

class Solution:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        ints = []
        for sched in schedule:
            for interval in sched:
                ints.append([interval.start,interval.end])
        ints.sort(key = lambda x: x[0])
        mapp = Counter()
        for start,end in  ints:
            mapp[start] += 1
            mapp[end] -= 1
        arr = []
        for k in sorted(mapp):
            arr.append([k,mapp[k]])
        n = len(arr)
        for i in range(1,n):
            arr[i][1] += arr[i-1][1]
        common = []
        for i in range(1,n-1):
            if arr[i][1] == 0:
                common.append([arr[i][0],arr[i+1][0]])
        ans = []
        for start,end in common:
            ans.append(Interval(start,end))
        return ans
    
#########################################
# 1665. Minimum Initial Energy to Finish Tasks
# 12MAY26
#########################################
#binary search on answer
class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        '''
        sort on tasks then binary search
        minimum is bigger than actual
        i dont know why though...
        '''
        new_tasks = []
        for actual,minimum in tasks:
            new_tasks.append([actual,minimum,minimum - actual])
        

        #sort decesding in minimum - acctual
        new_tasks.sort(key = lambda x : -x[2])

        def can_do(arr,initial):
            energy = initial
            for actual,minimum,diff in arr:
                if energy < minimum:
                    return False
                energy -= actual
            return True

        left,right = 0,sum([minimum for _,minimum in tasks])
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            if can_do(new_tasks,mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans
    
class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        '''
        sort on tasks then binary search
        minimum is bigger than actual
        i dont know why though...
        '''
        #new_tasks = []
        #for actual,minimum in tasks:
        #    new_tasks.append([actual,minimum,minimum - actual])
        

        #sort decesding in minimum - acctual
        tasks.sort(key = lambda x : -(x[1] - x[0]))

        def can_do(arr,initial):
            energy = initial
            for actual,minimum in arr:
                if energy < minimum:
                    return False
                energy -= actual
            return True

        left,right = 0,sum([minimum for _,minimum in tasks])
        ans = -1
        while left <= right:
            mid = left + (right - left) // 2
            if can_do(tasks,mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans
    
#sort in increasing diff
#then just add energy, if the min enery is bigger than the energy we have used so far
#then we need to set that one as the new energy
class Solution:
    def minimumEffort(self, tasks: List[List[int]]) -> int:
        tasks.sort(key=lambda x: x[1] - x[0])
        ans = 0
        for task in tasks:
            ans = max(ans + task[0], task[1])
        return ans
    
####################################################
# 1674. Minimum Moves to Make Array Complementary
# 13MAY26
####################################################
class Solution:
    def minMoves(self, nums: List[int], limit: int) -> int:
        '''
        we are allowed to replace any num in nums with another number between 1 and limit + 1
        say we have array
        [a,b,c,d]
        we need a + d = b + c
        same as:
            a + d - b - c = 0
        [1,2,3,4]
            1 + 4 - 2 - 3 = 0
            0 = 0
        say we have [1,2,4,3]
            1 + 3 - 2 - 4 = 0
            -2 != 0
        
        1 + 3 ?= 2 + 4
        4 or 6, try 4 or 6?
        [1,2,2,1]
        1 + 1 = 2 + 2
        2 or 4
        if pairs are already equal, we need no change
        we either need to change i, or n-i-1, or change both
        each pair then requires 0,1,2 modifications
        notice that the numbers can only be from 1 to limit + 1
        '''
        n = len(nums)
        diff = [0]*(2*limit + 2)
        for i in range(n//2):
            left,right = nums[i],nums[n-i-1]
            a = min(left,right)
            b = max(left,right)

            diff[2] += 2
            diff[a + 1] -= 1
            diff[a + b] -= 1
            diff[a + b + 1] += 1
            diff[b + limit + 1] += 1
        
        min_ops = n
        curr_ops = 0
        for c in range(2,2*limit + 1):
            curr_ops += diff[c]
            min_ops = min(min_ops,curr_ops)
        return min_ops
    
###########################################
# 2784. Check if Array is Good
# 14MAY26
###########################################
class Solution:
    def isGood(self, nums: List[int]) -> bool:
        '''
        just check if it contains 1 to n+1 and has two occurnces of n+1
        '''
        n = len(nums)
        max_ = max(nums)
        mapp = Counter(nums)
        if max_ >= n:
            return False
        for i in range(1,n):
            if i == n-1:
                if mapp[i] != 2:
                    return False
            else:
                if mapp[i] != 1:
                    return False
        return True
    
#sort, and check the first n-1 elements
#then check the ends for the same
class Solution:
    def isGood(self, nums: List[int]) -> bool:
        nums.sort()
        n = len(nums) - 1
        for i in range(n):
            if nums[i] != i + 1:
                return False
        return nums[n] == n
    
class Solution:
    def isGood(self, nums: List[int]) -> bool:
        '''
        count and check on fly
        '''
        n = len(nums)
        counts = Counter()
        for num in nums:
            if num >= n:
                return False
            if num < n - 1 and counts[num] > 0:
                return False
            if num == n-1 and counts[num] > 1:
                return False
            counts[num] += 1
        
        return True
    
################################################
# 2856. Minimum Array Length After Pair Removals
# 16MAY26
#################################################
#cheeky
class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        '''
        if the max(count) of nums is <= len(nums) // 2, we can clear them all out based on parity
        if even, they all go away, if odd there is 1 remaining
        now if the count is bigger than len(nums) // 2
        we need to clear out the rest instead
        '''
        n = len(nums)
        counts = Counter(nums)
        max_count = max(counts.values())
        if max_count <= n // 2:
            if n % 2:
                return 1
            else:
                return 0
        
        remaining = n - max_count
        return max_count - remaining
    
###############################################
# 1340. Jump Game V
# 24MAY26
################################################
class Solution:
    def maxJumps(self, arr: List[int], d: int) -> int:
        '''
        from any index i, i can jump to (i + ii) or (i - ii) for ii in range(d+1) so long as i am in bounds
        not only that arr[i + ii] or arr[i-ii] both must be smaller than arr[i] and all indicies k
        i.e min (i,j) < k max(i,j), i,e all must be smaller than the current arr[i] i am on
        dp on all i
        '''
        n = len(arr)
        memo = {}

        def dp(i):
            if i < 0 or i >= n:
                return 0
            if i in memo:
                return memo[i]
            count = 1
            curr_max = arr[i]
            #try going to the right first
            for ii in range(1,d+1):
                if i + ii < n:
                    if arr[i + ii] < curr_max:
                        count = max(count, 1 + dp(i+ii))
                    else:
                        break
            #now left
            for ii in range(1,d+1):
                if i - ii >= 0:
                    if arr[i - ii] < curr_max:
                        count = max(count, 1 + dp(i-ii))
                    else:
                        break
            memo[i] = count
            return count
        
        ans = 0
        for i in range(n):
            ans = max(ans,dp(i))
            print(ans)
        
        return ans

##################################################
# 3121. Count the Number of Special Characters II
# 27MAY26
##################################################
class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        '''
        hash the characters and check if its lowercase appears before the first
        all lower case muse appear before uppercase
        there cannot be more than 26 speical chars
        '''
        mapp = defaultdict(list)
        for i,ch in enumerate(word):
            if ch not in mapp:
                mapp[ch] = i
            elif ch.islower():
                mapp[ch] = max(mapp[ch],i)
            elif ch.isupper():
                mapp[ch] = min(mapp[ch],i)
        
        special = 0
        #validate
        for i in range(26):
            ch = chr(ord('a') + i)
            if ch.lower() in mapp and ch.upper() in mapp:
                if mapp[ch.lower()] < mapp[ch.upper()]:
                    special += 1
        return special

###############################################
# 3093. Longest Common Suffix Queries
# 28MAY26
#################################################
#fucking ezzzzz dawggg
from collections import defaultdict


class Node:
    def __init__(self):
        self.children = {}
        self.length = float('inf')
        self.idx = float('inf')
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word, idx):
        curr = self.root
        n = len(word)

        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = Node()

            curr = curr.children[ch]

            # update criteria
            if n < curr.length:
                curr.length = n
                curr.idx = idx

            elif n == curr.length:
                curr.idx = min(curr.idx, idx)

        curr.is_end = True
    
    def search(self,word):
        curr = self.root

        for ch in word:
            if ch not in curr.children:
                return curr.idx
            curr = curr.children[ch]
        
        return curr.idx

class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        '''
        notice the longest common suffix, if there isnt one is the empty string ""
        build trie on reverse strings, then
        for each node, store the smallest length
                and its index earliest index in the
        the hard part is updating the nodes as we are desceding
        '''
        trie = Trie()
        n = len(wordsContainer)
        for i,word in enumerate(wordsContainer):
            trie.insert(word[::-1],i)
        
        #find index of shortest word that comes the earliest
        shortest = min([len(word) for word in wordsContainer])
        idx_shortest = 0
        for i,word in enumerate(wordsContainer):
            if len(word) == shortest:
                idx_shortest = i
                break
            
        ans = []
        for word in wordsQuery:
            idx = trie.search(word[::-1])
            if idx == float('inf'):
                ans.append(idx_shortest)
            else:
                ans.append(idx)
 
        return ans
    
####################################################
# 2126. Destroying Asteroids
# 31MAY26
####################################################
class Solution:
    def asteroidsDestroyed(self, mass: int, asteroids: List[int]) -> bool:
        '''
        we only stand to gain mass, try sorting in order and check
        '''
        asteroids.sort()
        for a in asteroids:
            if a > mass:
                return False
            mass += a
        
        return True


