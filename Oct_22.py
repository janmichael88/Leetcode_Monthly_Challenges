############################################
# 1155. Number of Dice Rolls With Target Sum
# 02COT22
##########################################
#bactracking TLE
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        '''
        i can solve this using backtracking, but it might time out
        '''
        self.num_ways = 0
        
        def backtrack(num_die,curr_score):
            if num_die > n:
                return
            if curr_score > target:
                return
            if curr_score == target:
                self.num_ways += 1
                self.num_ways %= 10**9 + 7
                return
            for side in range(1,k+1):
                backtrack(num_die+1,curr_score + side)
    

        backtrack(0,0)
        return self.num_ways

#combine subproblems at a subroot
class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        '''
        top down
        '''
        memo = {}
        
        def backtrack(num_die,curr_score):
            if num_die == 0:
                if curr_score == 0:
                    return 1
                else:
                    return 0
            if (num_die,curr_score) in memo:
                return memo[(num_die,curr_score)]
            ans = 0
            for side in range(1,k+1):
                ans += backtrack(num_die-1,curr_score - side)
            ans %= 10**9 + 7
            #print(num_die,curr_score)
            memo[(num_die,curr_score)] = ans
            return ans
    

        return backtrack(n,target)

class Solution:
    def numRollsToTarget(self, n: int, k: int, target: int) -> int:
        '''
        bottom up dp
        '''
        memo = {}
        
        dp = [[0]*(target+1) for _ in range(n+1)]
        
        #base cases
        dp[0][0] = 1
        
        for num_die in range(1,n+1):
            for curr_score in range(1,target+1):
                ans = 0
                for side in range(1,k+1):
                    if num_die - 1 >= 0 and curr_score - side >= 0:
                        ans += dp[num_die-1][curr_score - side]
                        
                ans %= 10**9 + 7
                dp[num_die][curr_score] = ans

        return dp[n][target]

##########################################
# 1578. Minimum Time to Make Rope Colorful 
# 03OCT22
##########################################
#FUCKING EASSSSSY boyyyyy XDDD
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        i can scan to find the groupings of balloons
        then in each grouping, delete the ballones that take up the smallest needed time
        i can use two pointers and sliding window to find the groups
        '''
        N = len(colors)
        left,right = 0,0
        min_time = 0
        
        while left < N:
            #if we can expand the current grouping
            while right + 1 < N and colors[right+1] == colors[left]:
                right += 1
            
            #if we have a window
            if right > left:
                #get the max in this window
                max_ = max(neededTime[left:right+1])
                sum_ = sum(neededTime[left:right+1])
                min_time += sum_ - max_
            
            right += 1
            left = right
        
        return min_time
            
#actual solutions from write up
class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        the intuions are that we need to only keep at least one balloon in the group and that we keep the balloon with the maximuim
        algo:
            init totaltime, left, right and 0
            pass over balloons and for each group, record the total removal time
        '''
        total_time = 0
        i,j = 0,0
        
        N = len(colors)
        
        while i < N and j < N:
            curr_total = 0
            curr_max = 0
            
            #final all balongs have same color, and update max and totals
            while j < N and colors[i] == colors[j]:
                curr_total += neededTime[j]
                curr_max = max(curr_max,neededTime[j])
                j += 1
            
            #first pas is zero anyway
            total_time += curr_total - curr_max
            i = j
        
        return total_time

class Solution:
    def minCost(self, colors: str, neededTime: List[int]) -> int:
        '''
        we can do this in one pass using one pointer by adding the smaller removal times directly to the answer
        intuition:
            for each group, we always record the largest removal time (lets call it currMaxTime for convience)
            and add other smaller removal times to totalTime
            when we have another newly added removal time (t[i]) that belongs to the curret group, we compare t[i]
             with currMaxTime, add the samller one totoalTime, and leave the largerone as the currmaxtime
        '''
        total_time = 0
        curr_max_time = 0 #maxtime for current group
        N = len(colors)
        
        for i in range(N):
            #if this ballong is the first baollong of a new group, rest
            if i > 0 and colors[i] != colors[i-1]:
                curr_max_time = 0
                
            total_time += min(curr_max_time,neededTime[i])
            curr_max_time = max(curr_max_time,neededTime[i])
        
        return total_time

############################
# 112. Path Sum (REVISTED)
# 04OCT22
############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        '''
        just dfs carrying sum along the way
        '''
        self.ans = False
        
        def dfs(node,curr_sum):
            if not node:
                return
            dfs(node.left,curr_sum + node.val)
            if not node.left and not node.right and curr_sum + node.val == targetSum:
                self.ans = True
            dfs(node.right,curr_sum + node.val)
        
        
        dfs(root,0)
        return self.ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        '''
        let dp(node,sum) return whether or not we have a valid root to leaf == targetsum
        then dp(node,sum) = dp(node.left,sum+node.val) or dp(node.right,sum_node.val)
        base case, empty not cannot be an answer
        '''
        def dp(node,sum_):
            if not node:
                return False
            if not node.left and not node.right and sum_ + node.val == targetSum:
                return True
            left = dp(node.left,sum_ + node.val)
            right = dp(node.right,sum_ + node.val)
            return left or right
        
        return dp(root,0)


##############################
# 531. Lonely Pixel I
# 04OCT22
##############################
#count em up
class Solution:
    def findLonelyPixel(self, picture: List[List[str]]) -> int:
        '''
        be careful, we cant just check for neighboring
        a black lonely pixel is a character 'B' that is located at a specific positions where same row and col don't have any other black pixels
        i can use hash set for cols and rows and add the i,j part to them respectively
        then retraverse and check for each black pixel
        check that counts are 1
        '''
        rows = len(picture)
        cols = len(picture[0])
        
        row_counts = [0]*rows
        col_counts = [0]*cols
        
        for i in range(rows):
            for j in range(cols):
                if picture[i][j] == 'B':
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        lonely = 0
        for i in range(rows):
            for j in range(cols):
                if picture[i][j] == 'B':
                    if row_counts[i] == 1 and col_counts[j] == 1:
                        lonely += 1
        
        return lonely

#################################
# 623. Add One Row to Tree (REVISTED)
# 06OCT22
#################################
#close one....
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        '''
        dfs but pass in parent and pass in current depth
        '''
        #don't forget the depth == 1 case
        if depth == 1:
            newNode = TreeNode(val)
            newNode.left = root
            return newNode
        def dfs(node,parent,from_left,from_right,curr_depth):
            if not node:
                return
            if curr_depth == depth:
                newNode = TreeNode(val)
                if from_left:
                    newNode.left = node
                    parent.left = newNode
                if from_right:
                    newNode.right = node
                    parent.right = newNode
            
            dfs(node.left,node,True,False,curr_depth+1)
            dfs(node.right,node,False,True,curr_depth+1)
        
        dfs(root,None,False,False,1)
        return root

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
        '''
        dfs but pass in parent and pass in current depth
        '''
        #don't forget the depth == 1 case
        if depth == 1:
            newNode = TreeNode(val)
            newNode.left = root
            return newNode
        
        def insert(node,curr_depth):
            if not node:
                return
            #stop just before to add in the node
            if curr_depth == depth - 1:
                old_left = node.left
                old_right = node.right
                node.left = TreeNode(val)
                node.left.left = old_left
                node.right = TreeNode(val)
                node.right.right = old_right 
            else:
                insert(node.left,curr_depth+1)
                insert(node.right,curr_depth+1)
        
        insert(root,1)
        return root
                

##################################
# 981. Time Based Key-Value Store
# 06OCT22
################################
#dictionary of dictioanry
class TimeMap:

    def __init__(self):
        '''
        i can use a dictinoary of dictionarys
        '''
        self.key_time_map = {}
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.key_time_map:
            self.key_time_map[key] = {}
        #store
        self.key_time_map[key][timestamp] = value

    def get(self, key: str, timestamp: int) -> str:
        #return largest timestamp <= timestampe
        #if we can't find the key, empty string
        if key not in self.key_time_map:
            return ""
        #retrieve the closest one
        for curr_time in range(timestamp,0,-1):
            if curr_time in self.key_time_map[key]:
                return self.key_time_map[key][curr_time]
        #otherwise we have no time
        return ""

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

#we can use the sortedconatiner class from python
from sortedcontainers import SortedDict
class TimeMap:

    def __init__(self):
        self.key_time_mapp = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        #no key, make new sroteddict at this key
        if key not in self.key_time_mapp:
            self.key_time_mapp[key] = SortedDict()
        #store
        self.key_time_mapp[key][timestamp] = value

    def get(self, key: str, timestamp: int) -> str:
        #try to retreive
        if key not in self.key_time_mapp:
            return ""
        
        #search for upperbound
        idx = self.key_time_mapp[key].bisect_right(timestamp)
        
        #if the upper bound is the first element, them there is nothing to return
        if idx == 0:
            return ""
        
        idx -= 1
        #peek item method, grabs the entry at the inex, insorted order
        return self.key_time_mapp[key].peekitem(idx)[1]


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

#binary search
class TimeMap:
    '''
    we can also use binary seach, since we just adding timestamps in order
    '''

    def __init__(self):
        self.key_time = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.key_time:
            self.key_time[key] = []
        #add npw
        self.key_time[key].append([timestamp,value])
        

    def get(self, key: str, timestamp: int) -> str:
        #no mwatching key
        if key not in self.key_time:
            return ""
        #if we cannot pull a most recent value
        if timestamp < self.key_time[key][0][0]:
            return ""
        
        #binary search for upper bound
        lo = 0
        hi = len(self.key_time[key])
        
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if self.key_time[key][mid][0] <= timestamp:
                lo = mid + 1
            else:
                hi = mid
        #return the one just before
        return "" if hi == 0 else self.key_time[key][hi-1][1]
        

# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

#############################
# 732. My Calendar III
# 07OCT22
############################
#right idea
class MyCalendarThree:
    '''
    the crux of the problem lies in effeciently looking for the k in the book function
    catching intervals really
    brute force would be add start,end to list,
    then in book, add, sort, and check intersections
    '''

    def __init__(self):
        self.bookings = []

    def book(self, start: int, end: int) -> int:
        self.bookings.append([start,end])
        #sort, default is on first
        self.bookings.sort()
        k = 1
        first_start,first_end = self.bookings[0]
        for s,e in self.bookings[1:]:
            if first_start <= s < e or first_end <= e

# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)

#line sweep O(N)**2)
from sortedcontainers import SortedDict

class MyCalendarThree:
    '''
    make sure to revisit My Calendar II after this problem
    intution:
        when we assign a booking on the closed interval [start,end] we are essentieally increamenitng the count for every number in that range
        thus the final result of each book cal is exactly the max count of a single time in the whole range [1,e^9]
        sweep line, sweep through the range
    instead of keeping all values of counts in a traditional array,we can use a differential array to represent the change that occurs at each time point
    we increase the count by 1 at a point start, end decrease the count by 1 at point end
    after enumerating all booked events and updating the differential array,
        we can simulate scanning the differential array with a vertical swee from the origin time point 0 to the max number 1e^9 and obtain prefix sum at each time point t, (running sum)
        this is the event count of time t
        then we just need to find the maximum value of such counts when we scan the array
    
    if i were to have a closed array [1,10] i would need to keep a count at each number, then sweep
    faster to just increment at start, ane decreemnt at end, then record the max
    '''

    def __init__(self):
        self.diffs = SortedDict()
        

    def book(self, start: int, end: int) -> int:
        #increament start by 1
        self.diffs[start] = self.diffs.get(start,0) + 1
        #decrement by 1, we are free after this booking is done
        self.diffs[end] = self.diffs.get(end,0) - 1
        curr = res = 0
        for time,diff in self.diffs.items():
            curr += diff
            res = max(res,curr)
        
        return res


# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)

#segment tree but with lazy propgations
class MyCalendarThree:

    '''
    we can use segment tree to store the maxinum numb of bookings in the array
    rather when we query(start,end) this should return the maxnumber of k bookings between start and end
    we only need query start to end in book, but we update lazily
    at the leaves we in crement by 1, then we take the max of the left and right subtrees
    we need to update lazily because we are given the start and ends of an interval and not an actual interval
    https://www.geeksforgeeks.org/lazy-propagation-in-segment-tree/
    
    the idea behind lazy propogations
        recall when we update a node, we have to update the predecessor nodes too, and this can be very expensive because the segment tree could have many nodes
        When there are many updates and updates are done on a range, we can postpone some updates (avoid recursive calls in update) and do those updates only when required.
        come back to this problem later on
    '''
    def __init__(self):
        #instead of nodes, store them in a hashmap
        self.vals = Counter()
        self.lazy = Counter()
        
    def update(self,start:int, end:int, left:int = 0,right:int = 10**9,idx:int = 1) -> None:
        #out of bonuds
        if start > right or end <= left:
            return
        #if we require an update
        if start <= left <= right <= end:
            self.vals[idx] += 1
            self.lazy[idx] += 1
            
        #recurse left and right
        else:
            mid = left + (right - left) // 2
            self.update(start,end,left,mid,idx*2)
            self.update(start,end,mid+1,right,idx*2 + 1)
            #actual updates
            self.vals[idx] = self.lazy[idx] + max(self.vals[2*idx],self.vals[2*idx+1])
        

    def book(self, start: int, end: int) -> int:
        #when we book we need to update
        self.update(start,end-1)
        #return the value at the root
        return self.vals[1]
        
# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)

#using sortedlist
from sortedcontainers import SortedList
class MyCalendarThree:
    
    '''
    we can add the intervals into a sorted container, then when adding an interal we just increment the counts
    to keep all the intervals we can use sorted list
        with the time range beings [1,e^9]
    
    when we need to book a new event [start,end)
    1. Binary search all starting points in intervals to find the first interval [L1, R1) that has L1 >= start, then we split the interval into [L1, start) and [start, R1), keep the events in them the same as the origin interval [L1, R1), and put them back in intervals container.
    2. Similarly, perform a binary search to get the first [L2, R2) that satisfies L2 <= end, split it into[L2, start) and [start, R2) and inserting them into intervals.
    3. For all non-empty intervals between [start, R1) and [start, R1) inclusively in intervals, increase the events of them by 1 as we added a new event in time [start, end) just now. Because only the number of events in those intervals are updated, to get the max number of events now, we just need to compare the last max number of events with them.

    '''
    def __init__(self):
        # only store the starting point and count of events
        self.starts = SortedList([[0,0]])
        self.res = 0

    def split(self, x: int) -> None:
        idx = self.starts.bisect_left([x,0])
        if idx < len(self.starts) and self.starts[idx][0] == x:
            return idx
        self.starts.add([x,self.starts[idx-1][1]])

    def book(self, start: int, end: int) -> int:
        self.split(start)
        self.split(end)
        for interval in self.starts.irange([start,0], [end,0], (True,False)):
            interval[1] += 1
            self.res = max(self.res, interval[1])
        return self.res
        


# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)

#############################
# 716. Max Stack
# 09SEP22
#############################
#nope, dont'est work
#the problem is that if there is more than on maximum element, i need to remove the topmost one
class MaxStack:

    def __init__(self):
        '''
        probably one of the hardest design questions on LC
        i could use a stack and a max heap, but the problem would be with popMax
        i could combine hashmap, maxheap and stack
        
        '''
        self.stack = []
        self.max_heap = []
        self.mapp = Counter()
        

    def push(self, x: int) -> None:
        #add to each of them
        self.stack.append(x)
        heapq.heappush(self.max_heap, -x)
        self.mapp[x] += 1
        

    def pop(self) -> int:
        #only looking at stack part now
        ans = self.stack.pop()
        #clear
        

    def top(self) -> int:
        return self.stack[-1]
        

    def peekMax(self) -> int:
        
        

    def popMax(self) -> int:
        


# Your MaxStack object will be instantiated and called as such:
# obj = MaxStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.peekMax()
# param_5 = obj.popMax()

from sortedcontainers import SortedList
class MaxStack:
    '''
    we maintain two balanced trees, but for each push operation, increment counter and push the counter id to the stack and values balacned tree
    its a rather tricky implementation, but we need keep track of the opeations using a unique id
    on stack we keep elements as (id,val)
    on values we keep elements as (val,id)
    
    stack is done by pushing order
    values is done by value order
    '''

    def __init__(self):
        self.stack = SortedList()
        self.values = SortedList()
        self.count = 0
        

    def push(self, x: int) -> None:
        #add by pushing order
        self.stack.add((self.count,x))
        #add by values order
        self.values.add((x,self.count))
        #increment
        self.count += 1

    def pop(self) -> int:
        #retrieve from stack
        idx,val = self.stack.pop()
        #remove from values tree
        self.values.remove((val,idx))
        return val

    def top(self) -> int:
        return self.stack[-1][1]
        

    def peekMax(self) -> int:
        return self.values[-1][0]

    def popMax(self) -> int:
        val,idx = self.values.pop()
        self.stack.remove((idx,val))
        return val
        


# Your MaxStack object will be instantiated and called as such:
# obj = MaxStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.peekMax()
# param_5 = obj.popMax()

############################
# 1328. Break a Palindrome (REVISTED)
# 10OCT22
############################
#almost
class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        '''
        lexographically smallest means first in alphabetical order
        we meed to replace a character!
        brute force would be to try and place each letter starting with a to z and check not palindrome
        '''
        palindrome = list(palindrome)
        N = len(palindrome)
        def ispal(s):
            left, right = 0,len(s) - 1
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            
            return True
        
        #generate letters
        letters = [chr(num + ord('a') ) for num in range(26) ]
        
        #try
        cands = []
        for l in letters:
            for i in range(N):
                #cache the old one
                old = palindrome[i]
                palindrome[i] = l
                #check
                if ispal(palindrome) == False:
                    cands.append( "".join(palindrome))
                palindrome[i] = old
        
        cands.sort()
        if not cands:
            return ""
        return cands[0]

###############################
# 420. Strong Password Checker
# 10OCT22
###############################
#https://leetcode.com/problems/strong-password-checker/discuss/91008/Simple-Python-solution
#revisit this fucking problem
class Solution:
    def strongPasswordChecker(self, password: str) -> int:
        '''
        the two easy cases
            when len < 6, need to add
            when 6 <= len < 20, we need tp dp len - 20 deletions
            also we need to to a change for every three repeating characters
            
        for any repeasting sequences with len % 3 == 0, we can reduces one replacements by deleting one char
        
        for any repeating sequences with len % 3 == 1
            we can reduce one replacement by deleting to characters
        
        When we have > 20 chars, we absolutely must delete (can not insert). There is no way to NOT delete!
    However, we could use some of the deletions towards resolving 3-char rule violations.

    Case one: aaa ... violation can be resolved by deleting last 'a'. So, we subtract those deletions from the number of needed replacements (line: replace -= min(delete, one) )

    Case two: aaaa ... violation can only be resolved by deleting last 'aa' only. Since, we already used deletions for case one, we subtract those here.
    The formula min(max(delete - one, 0), two * 2) // 2 is a bit convoluted, but it means if there are still left some deletes max(delete - one, 0), let's use them to resolve the case two violations. Since we must delete 2 chars to resolve each violation, we can only use delete // 2 deletions. if delete//2 < two, we can resolve only delete//2 violations here (no delete's left thereafter). if delete//2 >= two, we can resolve two-number of violations (there will be some delete's left).

    Case 3: aaaaa ... violation can only be resolved by deleting last 'aaa' only. We must again account for case one and case two here: delete - one - 2 * two.
    So, we are subtracting the number of deletions that we used towards resolving the respective cases.
    Now, if there are any deletions left, let's use them towards case 3 here! max(delete - one - 2 * two, 0). Again, we can only resolve remaining_delete // 3 number of violations.

    Note that there are no more cases! aaaaaa is just the case one (length%3 == 0) plus case 3 again.

    So, the whole time, we have been reducing the number of change / repeat violations by resolving them with deletions.
    Once again: we are subtracting cases one and two from repeat/change, so that all the repeat runs would become case 3's. Then, we delete the triples.

    The final answer is delete + max(missing_type, repeat). Here's why:

    If there were enough required deletions, we would take # repeat violations to zero. We would still have missing_type violations to add (impossible to resolve by deleting)!
    However, If there were not enough deletions to take care of repeat violations, we could combine resolving the remaining repeat violations with
    missing_type violations by using replacement: max(missing_type, repeat) (as in <= 20).

    What happens when there are more deletes than repeats that are resolved (3 cases)? reapeat variable becomes < 0, deletes absorb all repeats, and we still have to resolve missing_type violations. When the number of deletes is less than repeats, either of 2 may happen: repeat-delete > missing_type (repeat-delete absorbs missing_type) OR repeat-delete < missing_type (missing_type absorbs repeat-delete).


        '''
        #first check for each of the types
        missing_types = 3
        #take off 1 for each missing type
        if any('a' <= c <= 'z' for c in password):
            missing_types -= 1
        if any('A' <= c <= 'Z' for c in password):
            missing_types -= 1
        if any(c.isdigit() for c in password):
            missing_types -= 1
        
        #now we need to check for repeating sequences
        change = 0 # of replacements to deal with three repeating characters
        one = 0 # of seqs that can be substituted with 1 deletions, (3k)-seqs
        two = 0 # of seqs that can be substituted with 2 deletions, (3k + 1)-seqs
        p = 2
        while p < len(password):
            #triplet
            if password[p] == password[p-1] == password[p-2]:
                length = 2
                #what the fuck is going on here
                while p < len(password) and password[p] == password[p-1]:
                    #advance
                    length += 1
                    p += 1
                change += length // 3 #'aaaaaaa' -> 'aaxaaxa'
                if length % 3 == 0:
                    one += 1
                elif length % 3 == 1:
                    two += 1
            else:
                p += 1
        
        #easy cases
        if len(password) < 6:
            return max(missing_types, 6 - len(password))
        elif len(password) <= 20:
            return max(missing_types,change)
        
        else:
            delete = len(password) - 20
            change -= min(delete,one)
            change -= min(max(delete - one,0),two*2) // 2
            change -= max(delete - one - 2*two,0) // 3
            return delete + max(missing_types,change)
        

############################
# 976. Largest Perimeter Triangle
# 12OCT22
############################
#FAIL T.T
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        '''
        i need to fix two of the largest sides
        then find the greatest third side that is first <= third <= second
        '''
        nums.sort()
        first = nums[-1]
        second = nums[-2]
        #greedy two pointer
        left = 0
        right = len(nums) - 3
        while left <= right:
            if left == right:
                third = nums[left]
                if first <= third <= second:
                    return first + second + third
            else:
                if right > first + second:
                    right -= 1
                else:
                    left += 1
        
        return 0
            
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        '''
        just check each triplet sucht that
        fist + second > third
        after sorting of course
        '''
        nums.sort()
        for i in range(len(nums) - 3, -1, -1):
            if nums[i] + nums[i+1] > nums[i+2]:
                return nums[i] + nums[i+1] + nums[i+2]
        return 0

#################################
# 921. Minimum Add to Make Parentheses Valid
# 12OCT22
#################################
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        '''
        a string is valid of:
            it is empty
            it can be written as AB (A concat with B) where A adn B are valid strings
            it can be written as (A) where A is a valid string
            
        can i just use stack to clear out current valid parenthese
        then return the lenght of the stack
        '''
        stack = []
        
        for ch in s:
            if stack and stack[-1] == '(' and ch == ')':
                stack.pop()
            else:
                stack.append(ch)
        
        return len(stack)

class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        #keeping track of balances
        ans = 0
        balance = 0
        for ch in s:
            balance += 1 if ch == '(' else -1
            if balance == -1:
                ans += 1
                balance += 1
        
        return ans + balance

        