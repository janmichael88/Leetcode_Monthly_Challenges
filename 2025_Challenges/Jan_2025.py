###########################################
# 2382. Maximum Segment Sum After Removals
# 01JAN25
##########################################
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.sum = {}
        self.max_sum_segment = 0


    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        #if we join segments just pass up sum to x
        x_par = self.find(x)
        y_par = self.find(y)

        if x_par == y_par:
            return
        self.sum[x_par] += self.sum[y_par]
        self.sum[y_par] = 0
        self.parent[y_par] = x_par
        self.max_sum_segment = max(self.max_sum_segment, self.sum[x_par])
    
    def add_merge(self, x, num):
        #add pointer to self first
        self.parent[x] = x
        self.sum[x] = num
        self.max_sum_segment = max(self.max_sum_segment, num)
        if x - 1 in self.parent:
            self.union(x,x-1)
        if x + 1 in self.parent:
            self.union(x,x+1)

    
class Solution:
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        '''
        removed just means the num is set to zero in nums
        after each removal find the maximm segment sum
        hint says to use sorted data structure and remove invalid segments from the structure
        need to maintain the maximum sum, at the start the maximum sum is sum(nums)
        we would eventually get sent to the zeros array, but the queries have an order, we cant greedily take the largest

        reverse union find, start from the end
        whnever we union to a group att the sum back in, whenever we join, we need to check left and right
        when we do join we need to update max sums
        for union find do we need to keep track of ranks for this problem???
        go in reverse order of removals
        say we add in back i
        we need to check (i-1) to the left and (i+1) to the right
        find is still the same, but for union, we need to pass up the sums

        need to to union find in order!, but in reverse
        '''
        n = len(nums)
        uf = UnionFind()
        ans = [0]*n

        for i in range(n-1,-1,-1):
            ans[i] = uf.max_sum_segment
            idx = removeQueries[i]
            uf.add_merge(idx, nums[idx])
        
        return ans

##########################
# 1871. Jump Game VII
# 02JAN25
#########################
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        '''
        dp(i) returns true if we can reach the end
        so dp will TLE
        it becomes O(N^2)
        hint says to use prefix sums
        '''
        memo = {}
        n = len(s)

        def dp(i):
            if i >= n-1:
                return True
            if i in memo:
                return memo[i]
            ans = False
            for j in range(i + minJump, min(i + maxJump + 1,n)):
                if s[j] == '0':
                    ans = ans or dp(j)
            
            memo[i] = ans
            return ans

        return dp(0)
    
########################################
# 2270. Number of Ways to Split Array
# 03JAN25
########################################
class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        '''
        prefix sum, then just check the left and right parts at each split
        we can only split into 2
        '''
        pref_sum = [0]
        for num in nums:
            pref_sum.append(pref_sum[-1] + num)
        
        n = len(nums)
        splits = 0
        for i in range(n-1):
            left = pref_sum[i+1]
            right = pref_sum[-1] - pref_sum[i+1]
            if left >= right:
                splits += 1
        
        return splits

class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        '''
        we can optimize if we keep track of pref sum and suff sum
        '''
        pref_sum = 0
        suff_sum = sum(nums)

        splits = 0
        for i in range(len(nums) - 1):
            pref_sum += nums[i]
            suff_sum -= nums[i]

            if pref_sum >= suff_sum:
                splits += 1
        
        return splits
    
#################################
# 1871. Jump Game VII
# 03JAN25
#################################
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        '''
        dp(i) returns true if we can reach the end
        we can use bfs, but we need to prune so we dont visit previous indices in the interval
        but we jump to the max index instead
        https://leetcode.com/problems/jump-game-vii/solutions/1224681/python3-thinking-process-no-dp-needed/
        '''
        q = deque()
        seen = set()
        q.append(0)
        seen.add(0)
        mx = 0

        while q:
            i = q.popleft()
            if i == len(s) - 1:
                return True
            for j in range(max(i + minJump, mx + 1), min(i + maxJump + 1, len(s))):
                if s[j] == '0' and j not in seen:
                    seen.add(j)
                    q.append(j)
            
            mx = i + maxJump
        
        return False

###################################
# 2381. Shifting Letters II
# 05JAN25
###################################
class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        '''
        accumulate the shifts and apply at the end!
        this is just line sweep
        its only in step up or one steo down
        '''
        n = len(s)
        up_shifts = [0]*(n+1)
        down_shifts = [0]*(n+1)
        for l,r,d in shifts:
            if d == 1:
                up_shifts[l] += 1
                up_shifts[r+1] -= 1
            else:
                down_shifts[l] += 1
                down_shifts[r+1] -= 1
        #accumulate
        for i in range(1,n+1):
            up_shifts[i] += up_shifts[i-1]
            down_shifts[i] += down_shifts[i-1]
        
        ans = []
        for i in range(n):
            curr_shift = up_shifts[i] - down_shifts[i]
            #apply shift
            new_idx = ((ord(s[i]) - ord('a')) + curr_shift) % 26
            new_char = chr(ord('a') + new_idx)
            ans.append(new_char)
        
        return "".join(ans)

class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        '''
        we can also accumulate on the fly
        '''
        n = len(s)
        up_shifts = [0]*(n+1)
        down_shifts = [0]*(n+1)
        for l,r,d in shifts:
            if d == 1:
                up_shifts[l] += 1
                up_shifts[r+1] -= 1
            else:
                down_shifts[l] += 1
                down_shifts[r+1] -= 1

        accum_up,accum_down = 0,0
        
        ans = []
        for i in range(n):
            accum_up += up_shifts[i]
            accum_down += down_shifts[i]
            curr_shift = accum_up - accum_down
            #apply shift
            new_idx = ((ord(s[i]) - ord('a')) + curr_shift) % 26
            new_char = chr(ord('a') + new_idx)
            ans.append(new_char)
        
        return "".join(ans)

#try bishop fenwich trees next

###################################################################
# 1769. Minimum Number of Operations to Move All Balls to Each Box
# 06JAN25
##################################################################
#brute force
class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        first of all you can solve this using brute force
        '''
        ans = []
        n = len(boxes)
        for i in range(n):
            moves = 0
            #go left
            for l in range(i-1,-1,-1):
                moves += (i - l) if boxes[l] == '1' else 0
            #go right
            for r in range(i+1,n):
                moves += (r - i) if boxes[r] == '1' else 0
            
            ans.append(moves)
        
        return ans


class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        first of all you can solve this using brute force
        need to store the number of moves in the left and right arrays
        '''
        ans = []
        n = len(boxes)
        for i in range(n):
            moves = 0
            for j in range(n):
                if j != i and boxes[j] == '1':
                    moves += abs(i-j)
            
            ans.append(moves)
        
        return ans

class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        first of all you can solve this using brute force
        need to store the number of moves in the left and right arrays
        build prefix array of moves
        we'll need at least two pref arrays, which should store the number of moves for to move each box to i
        need to track moves to the left and moves to the right

        idea is to accumulate balls and moves
        if we have k balls to the left of i
        then we need the move all k balls + the previous moves for (i-1)
        every time we move a step to left or right, all balls will need an additional move
        '''
        n = len(boxes)
        moves_to_left = [0]*n
        moves_to_right = [0]*n

        balls = int(boxes[0])
        for i in range(1,n):
            moves_to_left[i] = balls + moves_to_left[i-1]
            balls += int(boxes[i])
        
        balls = int(boxes[n-1])
        for i in range(n-2,-1,-1):
            moves_to_right[i] = balls + moves_to_right[i+1]
            balls += int(boxes[i])

        ans = [0]*n
        for i in range(n):
            ans[i] = moves_to_left[i] + moves_to_right[i]
        
        return ans

class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        we can change to one pass, just use accumulators for each one
        '''
        n = len(boxes)
        moves_to_left = 0
        moves_to_right = 0
        balls_to_left = 0
        balls_to_right = 0

        ans = [0]*n

        for l in range(n):
            ans[l] += moves_to_left
            balls_to_left += int(boxes[l])
            moves_to_left += balls_to_left

            r = n - l - 1
            ans[r] += moves_to_right
            balls_to_right += int(boxes[r])
            moves_to_right += balls_to_right

        return ans



