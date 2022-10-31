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

###################################################################
# 889. Construct Binary Tree from Preorder and Postorder Traversal
# 13OCT22
###################################################################
# Definition for a binary tree node.
# class TreeNode:
# def __init__(self, val=0, left=None, right=None):
# self.val = val
# self.left = left
# self.right = right
class Solution:
	def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
		'''
		pre-order is node,left,right
		postorder is left,right,node
		the first element of pre and the last element of post should be the value of the root
		the second to last value of post should be the right child of the root
		so we need to find the index to split left and right children in pre
		if the length of post is larger than 1, we need to used post[-2]
		'''
		def build(pre,post):
			#base cases 
			if not pre or not post:
				return None
			#make the current root, could be from either pre or post
			root = TreeNode(pre[0])
			if len(post) == 1:
				return root
			#search for the index
			#so we need to find the index to split left and right children in pre
			idx = pre.index(post[-2])
			#print(pre,post)
			root.left = build(pre[1:idx],post[:(idx-1)])
			root.right = build(pre[idx:],post[(idx-1):-1])
			return root
		
		return build(preorder,postorder)

#doesnt quite work# Definition for a binary tree node.
# class TreeNode:
# def __init__(self, val=0, left=None, right=None):
# self.val = val
# self.left = left
# self.right = right
class Solution:
	def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
		'''
		same solution but let pass indices intstead of arrays
		doesn't quite work...
		'''
		def build(pre_start,pre_end,post_start,post_end):
			#base cases, pointers overlap
			if pre_start > pre_end or post_start > post_end:
				return None
			#make the current root, could be from either pre or post
			root = TreeNode(preorder[pre_start])
			if post_start == post_end:
				return root
			#search for the index
			#so we need to find the index to split left and right children in pre
			idx = preorder.index(postorder[post_end-2])
			#print(pre,post)
			
			root.left = build(pre_start,idx,post_start, idx-1)
			root.right = build(idx,pre_end, idx,post_end-2)
			
			
			#root.left = build(pre[1:idx],post[:(idx-1)])
			#root.right = build(pre[idx:],post[(idx-1):-1])
			return root
		
		return build(0,len(preorder)-1,0,len(postorder)-1)

#https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/discuss/748216/Python3-Solution-with-a-Detailed-Explanation-Construct-Binary-Tree-from
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        # read this: https://www.techiedelight.com/construct-full-binary-tree-from-preorder-postorder-sequence/
        def helper(pre,post):
            print('pre is: ', pre, 'post is: ', post)
            if not pre:
                return None
        
            if len(pre)==1:
                return TreeNode(post.pop())
        
        
            node=TreeNode(post.pop()) #3
            ind=pre.index(post[-1]) #4
        
            node.right=helper(pre[ind:],post) #1
            node.left=helper(pre[1:ind],post) #2
            return node
    
        return helper(pre,post)

#hashmap for fast lookup
class Solution:
	def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
    
    idx={}
    for i,j in enumerate(pre):
        idx[j]=i
    def helper(start,end,post):
        
        if end==start:
            return None
    
        if end-start==1:
            return TreeNode(post.pop())
    
        
        node=TreeNode(post.pop()) #3
        ind=idx[post[-1]] #4
        
        node.right=helper(ind,end,post) #1
        node.left=helper(start+1,ind,post) #2
        return node

    return helper(0,len(pre),post)

###############################################
# 2095. Delete the Middle Node of a Linked List
# 14OCT22
###############################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        two pass,
        get the length of the linked list and find the middle
        advance to just before the middle and set .next to .next.next
        '''
        N = 0
        curr = head
        
        while curr:
            curr = curr.next
            N += 1
        
        if N == 1:
            return None
        
        mid = N // 2
        curr = head
        for _ in range(mid-1):
            curr = curr.next
            
        curr.next = curr.next.next
        

        
        return head

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can sue slow and fast pointer trick
        '''
        if head.next == None:
            return None
        slow = head
        fast = head
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        
        prev.next = prev.next.next
        
        return head

#################################################
# 1431. Kids With the Greatest Number of Candies
# 14OCT22
#################################################
#brute force works
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        '''
        given candies array, where each ith kid has candies[i]
        return boolean array if after givine all extracandies, they would have the greatest
        
        well N is smalle so brutforce is ok
        '''
        ans = []
        
        for c in candies:
            #if we give them
            give_them = c + extraCandies
            ans.append(give_them >= max(candies))
        
        return ans

class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        '''
        pre compute the max before then check
        '''
        curr_max = max(candies)
        
        ans = []
        for c in candies:
            #give them
            give_them = c + extraCandies
            ans.append(give_them >= curr_max)
        
        return ans

####################################
# 1531. String Compression II
# 15OCT22
####################################
#close one with backtracking
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        '''
        we can delete at most k characters and we want to return s such that is has the min run length encoding
        example
        s = "aaabcccd", k = 2
        it doesnt make sense to delete an a or c, since this would get compressed anyway
        so we delete b and d to get a3b3 which has length 4
        
        s = "aabbaa", k = 2
        delete the two b's to get a4
        
        s = "aaaaaaaaaaa", k = 0
        just the length of the run length encoding
        a11, which has length 3
        
        n is small enough, to allow for ane expensive call (N^2) in the dp function
        if i let dp(i) be the min length of s if i delete s[i] only if i have deletions
        knapsack delete or not delete if i still have deletions left
        
        first lets track backtracking
        we need to keep track of the last letter and the last encoding, as well as the current length
        
        '''
        self.ans = float('inf')
        N = len(s)
        
        def backtrack(i,last_char,last_count,curr_len,k):
            if i == N:
                self.ans = min(self.ans,curr_len)
                return
            if k < 0:
                return
            #if i were to delete
            delete_last_char = last_char
            delete_last_count = last_count
            delete_curr_len = curr_len
            backtrack(i+1,delete_last_char,delete_last_count,delete_curr_len,k-1)
            #if i were to keep
            keep_last_char = s[i]
            keep_last_count = last_count + 1 if last_count == s[i] else 0
            keep_curr_len = curr_len + len(str(last_count))
            backtrack(i+1,keep_last_char,keep_last_count,keep_curr_len,k)
        
        backtrack(0,"",0,0,k)
        return self.ans-1

#top down
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        '''
        we can delete at most k characters and we want to return s such that is has the min run length encoding
        example
        s = "aaabcccd", k = 2
        it doesnt make sense to delete an a or c, since this would get compressed anyway
        so we delete b and d to get a3b3 which has length 4
        
        s = "aabbaa", k = 2
        delete the two b's to get a4
        
        s = "aaaaaaaaaaa", k = 0
        just the length of the run length encoding
        a11, which has length 3
        
        n is small enough, to allow for ane expensive call (N^2) in the dp function
        if i let dp(i) be the min length of s if i delete s[i] only if i have deletions
        knapsack delete or not delete if i still have deletions left
        
        we need to keep track of a few things in each caller
        idx: current index into s
        lastChar = last symbold we have in compression
        lastCharCount: number of lastChar we have considered
        k: remaining deletions
        
        at each step we either choose to delete or keep
        and notince the the length of the string only increase if the current count is 0,1,9,99
        recall we don't encode 1
        we can use memeo to keep track of all the states, only because in this problem we are visting repeated states
        
        '''
        memo = {}
        n = len(s)
        def dp(idx, last_char, last_char_count, k):
            if k < 0: 
                return float('inf')
            if idx == n: 
                return 0
            if (idx, last_char, last_char_count, k) in memo:
                return memo[(idx, last_char, last_char_count, k)]

            delete_char = dp(idx + 1, last_char, last_char_count, k - 1)
            if s[idx] == last_char:
                keep_char = dp(idx + 1, last_char, last_char_count + 1, k) + (last_char_count in [1, 9, 99])
            else:
                keep_char = dp(idx + 1, s[idx], 1, k) + 1

            ans = min(keep_char, delete_char)
            memo[(idx, last_char, last_char_count, k)] = ans
            return ans

        return dp(0, "", 0, k)

#another way
#https://leetcode.com/problems/string-compression-ii/discuss/2704470/LeetCode-The-Hard-Way-Explained-Line-By-Line
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        @cache
        def dp(i, prev, prev_cnt, k):
            # set it to inf as we will take the min later
            if k < 0: 
            	return inf
            # we delete all characters, return 0
            if i == len(s): 
            	return 0
            # here we can have two choices, we either
            # 1. delete the current char
            # 2. keep the current char
            # we calculate both result and take the min one
            delete = dp(i + 1, prev, prev_cnt, k - 1)
            if s[i] == prev:
                # e.g. a2 -> a3
                keep = dp(i + 1, prev, prev_cnt + 1, k)
                # add an extra 1 for the following cases
                # since the length of RLE will be changed
                # e.g. prev_cnt = 1: a -> a2
                # e.g. prev_cnt = 9: a9 -> a10
                # e.g. prev_cnt = 99: a99 -> a100 
                # otherwise the length of RLE will not be changed
                # e.g. prev_cnt = 3: a3 -> a4
                # e.g. prev_cnt = 8: a8 -> a9
                # alternative you can calculate `RLE(prev_cnt + 1) - RLE(cnt)`
                if prev_cnt in [1, 9, 99]:
                    keep += 1
            else:
                # e.g. a
                keep = dp(i + 1, s[i], 1, k) + 1
            return min(delete, keep)
        
        # dp(i, prev, prev_cnt, k) returns the length of RLE with k characters to be deleted
        # starting from index i 
        # with previous character `prev`
        # with `prev_cnt` times repeated so far
        return dp(0, "", 0, k)

class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        '''
        solutions inspired from this post
        https://leetcode.com/problems/string-compression-ii/discuss/757506/Detailed-Explanation-Two-ways-of-DP-from-33-to-100
        '''
        memo = {}
        N = len(s)
        def calcLen(length):
            if length == 0:
                return 0
            elif length == 1:
                return 1
            elif length < 10:
                return 2
            elif length < 100:
                return 3
            else:
                return 4
            
        def dp(i,ch,length,k):
            if i == N:
                return calcLen(length)
            if k < 0:
                return float('inf')
            if (i,ch,length,k) in memo:
                return memo[(i,ch,length,k)]
            delete_char = dp(i+1,ch,length,k-1)
            if s[i] == ch:
                keep_char = dp(i+1,ch,length+1,k)
            else:
                keep_char = dp(i+1,s[i],1,k) + 1
            
            ans = min(delete_char,keep_char)
            memo[(i,ch,length,k)] = ans
            return ans
        
        
        return dp(0,"",0,k)


#bottom up
class Solution:
    def getLengthOfOptimalCompression(self, s: str, k: int) -> int:
        def length(x):
            return 1 if x == 1 else 2 if 1 < x < 10 else 3 if 10 <= x < 100 else 4
        
        n = len(s)
        dp = [[float("inf")] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(min(i, k) + 1):
                if j:
                    dp[i][j] = dp[i - 1][j - 1]
                remove = count = 0
                for l in range(i, 0, -1):
                    if s[l - 1] == s[i - 1]:
                        count += 1
                    else:
                        remove += 1
                        if remove > j:
                            break
                    dp[i][j] = min(dp[i][j], dp[l - 1][j - remove] + length(count))
        
        return dp[-1][-1]

#############################################
# 1335. Minimum Difficulty of a Job Schedule
# 16OCT22
#############################################
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        we want to schedule a list of jobs in d days
        to work on the ith job, i need to finiish all jobs from 0 to i-1
        i need to do at least one task every day
        the difficulty of a job schedule is the sume of all difficulties for each day of the d days
        the difficulty on a day is the max of the jobs done on that day
        return the min difficult of a job schedule
        
        basically we want to parition dobs into d-1 partitions such that sum of the max of the paritions is minimized
        the paritions must be non empty
        
        advance pointer along and keep track of max of current paritiion and current difficulty
        then we have include a new job into the parition or make a new partition
        
        start with backtracking
        '''
        self.ans = float('inf')
        N = len(jobDifficulty)
        
        def backtrack(i,curr_max,curr_difficulty,d):
            #if i have gotten to the end and used up my days
            if i == N and d == 0:
                self.ans = min(self.ans,curr_difficulty+curr_max)
                return
            elif d < 0:
                return
            elif i == N:
                return
            
            #include
            backtrack(i+1,max(curr_max,jobDifficulty[i]),curr_difficulty,d)
            #don't include
            backtrack(i+1,jobDifficulty[i], curr_difficulty + curr_max,d-1)
        
        backtrack(0,0,0,d)
        return -1 if self.ans == float('inf') else self.ans

#yes!!!!
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        we want to schedule a list of jobs in d days
        to work on the ith job, i need to finiish all jobs from 0 to i-1
        i need to do at least one task every day
        the difficulty of a job schedule is the sume of all difficulties for each day of the d days
        the difficulty on a day is the max of the jobs done on that day
        return the min difficult of a job schedule
        
        basically we want to parition dobs into d-1 partitions such that sum of the max of the paritions is minimized
        the paritions must be non empty
        
        advance pointer along and keep track of max of current paritiion and current difficulty
        then we have include a new job into the parition or make a new partition
        
        start with backtracking
        '''

        memo ={}
        N = len(jobDifficulty)
        
        def dp(i,curr_max,d):
            #if i have gotten to the end and used up my days
            if d < 0:
                return float('inf')
            if i == N:
                if d == 0:
                    return curr_max
                else:
                    return float('inf')
            if (i,curr_max,d) in memo:
                return memo[(i,curr_max,d)]
            
            #include
            include = dp(i+1,max(curr_max,jobDifficulty[i]),d)
            #don't include
            dont_include = curr_max + dp(i+1,jobDifficulty[i],d-1)
            
            curr_difficulty = min(include,dont_include)
            memo[(i,curr_max,d)] = curr_difficulty
            return curr_difficulty
        
        ans = dp(0,0,d)
        return ans if ans != float('inf') else -1

#bottom up
#i fucking give up, i couldn't translate this stupid shit to bottom up
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        we are translating directly from out top down appaorach
        '''
        
        N = len(jobDifficulty)
        dp = [[[float('inf')]*d + [0] for _ in range(max(jobDifficulty)+1)] for i in range(N+1)]
        
        for i in range(N,-1,-1):
            for curr_max in range(max(jobDifficulty)+1,-1,-1):
                for rem in range(d,-1,-1):
                    #base cases
                    if i == N:
                        if rem == 0:
                            dp[i][curr_max][rem] = curr_max
                            
                    #transition cases
                    include = dp[i-1][max(curr_max,jobDifficulty[i])][rem]
                    dont_include = curr_max + dp[i-1][jobDifficulty[i]][rem-1]
                    curr_difficulty = min(include,dont_include)
                    dp[i][curr_max][rem] = curr_difficulty
        print(dp)

#actual solutions from LC
#two state instead of three state
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        let dp(i,d) represent the sub problem for finding the min difficulty starting with index 0 with d days remaning
        to solve dp(i,d) we need to solve dp(j,d-1) for all j > i then take the min
        dp(i,d) = {
        min(dp(j,d-1) for j in range(N to i +1))
        }
        when there is only 1 day left, we must comiplete all of the remaining jobs on the final day
        
        algo:
            state defintion:
                dp(i,d) refers to the minimum difficulty when starting at the ith job with d days left
                dp(0,d) will be the final answer since it represents the starting at the beginning of the job array and finish all jobs in exactly d days
                the job at idnex j will be the first task for the upcoming day, thereforce the jobs to be finished are all jobs with indices beteween i and j
                dp(i,d) = dp(j,d-1) + max(jobDifficulty[i:j] for all j > i)
            
            base cases:
                when there is exactly on day remaining, we need to finish all the unfinished jobs on that day
        
        edge cases
        One edge case that we must consider is if the number of days is more than the number of tasks, then we won't be able to arrange at least one job per day; in this case, we should return -1.
        '''
        memo = {}
        N = len(jobDifficulty)
        
        if N < d:
            return -1
        
        def dp(i,d):
            if d == 1:
                #finisn the remaining and take the max of the after i
                #this could be optimized
                return max(jobDifficulty[i:])
            if (i,d) in memo:
                return memo[(i,d)]
            
            res = float('inf')
            #find the max difficulty from [i to N - d]
            max_difficulty = 0
            for j in range(i,N-d+1):
                max_difficulty = max(max_difficulty,jobDifficulty[j] )
                res = min(res,max_difficulty + dp(j+1,d-1))
            
            memo[(i,d)] = res
            return res
        return dp(0,d)
        
#bottom up
class Solution:
    def minDifficulty(self, jobDifficulty: List[int], d: int) -> int:
        '''
        bottom up dp
        tips:
            when transferring states, we need to ensure there are enough tasks remaining to arrange at least one job per day
            There is no dependency between dp[d][i] and dp[d][j] if i != j, because the value of dp[d][i] only depends on the results of dp[d - 1][j] when j > i. By identifying such relationships, we can draw viable edges for state transfer between different cells in the DP matrix.


        '''
        n = len(jobDifficulty)
        # Initialize the min_diff matrix to record the minimum difficulty
        # of the job schedule        
        min_diff = [[float('inf')] * n + [0] for i in range(d + 1)]
        for days_remaining in range(1, d + 1):
            for i in range(n - days_remaining + 1):
                daily_max_job_diff = 0
                for j in range(i + 1, n - days_remaining + 2):
                    # Use daily_max_job_diff to record maximum job difficulty
                    daily_max_job_diff = max(daily_max_job_diff, jobDifficulty[j - 1])
                    min_diff[days_remaining][i] = min(min_diff[days_remaining][i],
                                                      daily_max_job_diff + min_diff[days_remaining - 1][j])
        if min_diff[d][0] == float('inf'):
            return -1
        return min_diff[d][0]

#using stack (most optimal)
#come back to this one

#################################
# 1832. Check if the Sentence Is Pangram
# 17OCT22
#################################
class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        '''
        put into a hashset all the lower case chars 
        then as we traverse the sentence remove the char
        and return whether or not we have an empty set
        '''
        to_be_removed = set()
        for i in range(26):
            to_be_removed.add(chr(ord('a') + i))
        
        for ch in sentence:
            to_be_removed.discard(ch)
        
        return len(to_be_removed) == 0

class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        return len(set(sentence)) == 26

class Solution:
    def checkIfPangram(self, sentence: str) -> bool:
        '''
        we can use bit mask of size 26
        and set bit with XOR
        then we just need to check if we have 26 set bits
        '''
        seen = 0
        for ch in sentence:
            #get inde into bit mask
            idx = ord(ch) - ord('a')
            #gut this bit positions
            curr_bit = 1 << idx
            #set bit
            seen |= curr_bit
        
        #if we haver all 26 set bits
        return seen == (1 << 26) - 1

#################################
# 800. Similar RGB Color (REVISTED)
# 17OCT22
##################################
class Solution:
    def similarRGB(self, color: str) -> str:
        '''
        for each pair in color try to find the closest shorthand
        conver to base 16 and compare absolute values
        '''
        cand = [str(i)*2 for i in range(10)] + [chr(ord('a') + i)*2 for i in range(6)]
        
        def find_closest(code):
            cands = []
            for c in cand:
                #find similarity
                sim = abs(int(c,16) - int(code,16))
                cands.append([sim,c])
            
            #sort on score
            cands.sort(key = lambda x: x[0])
            return cands[0][1]
        
        ans = '#'
        for i in range(1,len(color),2):
            pair = color[i:i+2]
            #find closest
            clos = find_closest(pair)
            ans += clos
        
        return ans

class Solution:
    def similarRGB(self, color: str) -> str:
        '''
        instead of using implicit hex conversion we can use a trick
        (AB)_{16} = 16*A + B
        (XX)_{16} = 16*X + X = 17X
        
        simliarty for a pair is then −(16⋅A+B−17⋅X)^2
        where X is any digit fomr 0 to 16
        '''
        def findClosest(code):
            min_diff = float('inf')
            ans = None
            
            for i in range(16):
                curr_diff = abs(int(code,16) - i*17)
                #could also use squred difference
                #curr_diff = abs(int(code,16) - i*17)**2
                if curr_diff < min_diff:
                    min_diff = curr_diff
                    ans = i
            
            return hex(ans)[-1]*2
        
        ans = "#"
        for i in range(1,6,2):
            ans += findClosest(color[i:i+2])
        
        return ans

#tricky
class Solution:
    def similarRGB(self, color: str) -> str:
        '''
        turnst out minimum is at num / 17
        
        Expanding f(i) we get:
        (num^2 -2*num*17*i + (17^2)* i^2)

        We take the derivative with respect to i, and set to the equation to 0
        0 - (2)*(17)*(num) + 2*(17^2)*i = 0

        Rearranging
        2*(17^2)*i = (2)*(17)*(num)

        Then solve for i:
        i = num / 17
        
        
        '''
        def findClosest(code):
            ans = round(int(code,16)/17)
            return hex(ans)[-1]*2
        
        ans = "#"
        for i in range(1,6,2):
            ans += findClosest(color[i:i+2])
        
        return ans

##############################
# 38. Count and Say (REVISTED)
# 18OCT22
##############################
class Solution:
    def countAndSay(self, n: int) -> str:
        '''
        the recursive defintion is already defined for us, we just need to translate it
        '''
        memo = {}
        
        def rec(n):
            if n == 1:
                return '1'
            if n in memo:
                return memo[n]
            #first retrieve the last casll
            res = ""
            ct = 1
            prev = rec(n-1)
            #say it out by counting, this is the bottle neck here because we need to traverse the whole strin
            for i in range(len(prev)):
                #if we reach the end or start a new streak
                #cool way to count the streaks
                if  i == len(prev) - 1 or prev[i] != prev[i + 1]:
                    res += str(ct) + prev[i]
                    ct = 1
                else:
                    ct += 1

            memo[n] = res
            return res
        
        
        return rec(n)
                
#bottom up
class Solution:
    def countAndSay(self, n: int) -> str:
        '''
        bottom up
        '''
        dp = ['1']
        
        for _ in range(n-1):
            prev = dp[-1]
            res = ""
            ct = 1
            #say it out by counting, this is the bottle neck here because we need to traverse the whole strin
            for i in range(len(prev)):
                #if we reach the end or start a new streak
                #cool way to count the streaks
                if  i == len(prev) - 1 or prev[i] != prev[i + 1]:
                    res += str(ct) + prev[i]
                    ct = 1
                else:
                    ct += 1
            
            dp.append(res)
        
        return dp[n-1]

#using two pointers, and updating prev and current strings
class Solution:
    def countAndSay(self, n: int) -> str:
        '''
        bottom up
        '''
        curr_string = '1'
        
        for _ in range(n-1):
            next_string = ""
            j = k = 0
            while j < len(curr_string):
                while k < len(curr_string) and curr_string[j] == curr_string[k]:
                    k += 1
                next_string += str(k-j) + curr_string[j]
                j = k
            
            curr_string = next_string
        
        return curr_string

##################################
# 692. Top K Frequent Words
# 19OCT22
##################################
#asy
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        counts = Counter(words)
        #put into list of lists for sorting
        counts = [[k,v] for k,v in counts.items()]
        
        #default sorting string is lexgraphical
        counts.sort(key = lambda x: (-x[1],x))
        
        ans = []
        for i in range(k):
            ans.append(counts[i][0])
        
        return ans

#i could actually use a heap (max_heap)
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        counts = Counter(words)
        
        return heapq.nsmallest(k,counts.keys(), key = lambda x: (-counts[x],x))

#min heap
class Pair:
    def __init__(self,word,freq):
        self.word = word
        self.freq = freq
        
    def __lt__(self,p):
        #compare to another pair object p
        return self.freq < p.freq or (self.freq == p.freq and self.word > p.word)
    

class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        '''
        we can use a min heap but we need to create the object to compare
        we need to define it in the __lt__ part of the class
        '''
        counts = Counter(words)
        min_heap = []
        for word,freq in counts.items():
            heapq.heappush(min_heap, Pair(word,freq))
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        
        return [p.word for p in sorted(min_heap,reverse = True)]


#bucket sort and trie
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        '''
        we can use bucket sort to sort the frequenies for each word
        but in each bucket we need to store a trie
        when we do a preoder traversal of the trie, we should get the lexographical order of the words
        '''
        N = len(words)
        counts = Counter(words)
        buckets = [{} for _ in range(N+1)]
        self.k = k
        
        #helper to add word
        def add_word(trie: Mapping, word:str) -> None:
            root = trie
            for c in word:
                if c not in root:
                    root[c] = {}
                root = root[c]
            
            #end of word
            root['#'] = {}
        
        #get words from current trie at the current frequency
        def get_words(trie: Mapping, prefix: str) -> List[str]:
            #bacase no k let
            if self.k == 0:
                return []
            res = []
            #end of word
            if '#' in trie:
                self.k -= 1
                res.append(prefix)
            #check all roots 
            for i in range(26):
                ch = chr(ord('a') + i)
                #if we ca find it, recurse
                if ch in trie:
                    res += get_words(trie[ch], prefix+ch)
            
            return res
        
        #build
        for word,freq in counts.items():
            add_word(buckets[freq],word)
            
        res = []
        #greates frequency first
        for i in range(N,0,-1):
            #no more k
            if self.k == 0:
                return res
            #if we have a bucket to process
            if buckets[i]:
                res += (get_words(buckets[i],""))
        
        return res

########################################
# 1239. Maximum Length of a Concatenated String with Unique Characters (REVISTED)
# 24OCT22
#########################################
#close one, aye yai yai
class Solution:
    def maxLength(self, arr: List[str]) -> int:
        '''
        seems to be ok to try all possible concatneations
        try all of them using dfs
        '''
        self.ans = 0
        N = len(arr)
        
        def dfs(i,path,taken):
            if i == N:
                self.ans = max(self.ans,len(path))
                return
            #take
            to_take = True
            for ch in arr[i]:
                if ch in taken:
                    to_take = False
                else:
                    taken.add(ch)
            
            if to_take:
                dfs(i+1,path+arr[i],taken | set(arr[i]))
            #bactrack on this part
            for ch in arr[i]:
                taken.discard(ch)
            dfs(i+1,path,taken)
        
        dfs(0,"",set())
        return self.ans

#backtring with countmapp
class Solution:
    '''
    backtracking with mapp
    
    '''
    def maxLength(self, arr: List[str]) -> int:
        # Use depth first search recursion through arr
        # with backracking and a map for results
        return self.backtracking(arr, 0, Counter())
    
    def backtracking(self, arr: List[str], pos: int, res_map: Counter[str]) -> int:
        # Check for duplicate characters
        if len(res_map) and res_map.most_common(1)[0][1] > 1:
            return 0

        # Recurse through each possible next option
        # and find the best answer
        best = len(res_map)
        for i in range(pos, len(arr)):
            # Check for duplicate characters in word
            # then add the current word to the result map
            # and recurse to the next position
            word_map = Counter(arr[i])
            if len(word_map) != len(arr[i]):
                 continue
            res_map.update(word_map)
            best = max(best, self.backtracking(arr, i + 1, res_map))
            
            # Backtrack the result map before continuing
            for c in word_map:
                if res_map[c] == word_map[c]:
                    del res_map[c]
                else:
                    res_map[c] -= word_map[c]
        return best

class Solution:
    def maxLength(self, arr: List[str]) -> int:    
        # Pre-process arr with an optimizing helper
        # which converts each word to its character bitset
        # and then uses a set to prevent duplicate results
        opt_set = set()
        for word in arr:
            self.word_to_bitset(opt_set, word)

        # Convert the set back to an array for iteration
        # then start up the recursive helper
        opt_arr = list(opt_set)
        return self.backtracking(opt_arr, 0, 0, 0)
        
    def word_to_bitset(self, opt_arr: Set[int], word: str) -> None:        
        # Initialize an empty int to use as a character bitset
        char_bitset = 0
        for c in word:            
            # If the bitset contains a duplicate character
            # then discard this word with an early return
            # otherwise add the character to the bitset
            mask = 1 << ord(c) - 97
            if char_bitset & mask:
                return
            char_bitset += mask
        
        # Store the length of the word in the unused space
        # then add the completed bitset to our optimized set
        opt_arr.add(char_bitset + (len(word) << 26))
        
    def backtracking(self, opt_arr: List[int], pos: int, res_chars: int, res_len: int) -> int:
        # Recurse through each possible next option
        # and find the best answer
        best = res_len
        for i in range(pos, len(opt_arr)):
            new_chars = opt_arr[i] & ((1 << 26) - 1)
            new_len = opt_arr[i] >> 26
            
            # If the two bitsets overlap, skip to the next result
            if new_chars & res_chars:
                continue
            
            # Add the current word to the result
            # and recurse to the next position
            res_chars += new_chars
            res_len += new_len
            best = max(best, self.backtracking(opt_arr, i + 1, res_chars, res_len))
            
            # Backtrack the result before continuing
            res_chars -= new_chars
            res_len -= new_len
        return best

#checky dp without bit masks
class Solution:
    def maxLength(self, arr: List[str]) -> int:        
        # Use depth first search recursion through arr
        # building from an initial empty string
        return self.dfs(arr, 0, "")
    
    def dfs(self, arr: List[str], pos: int, res: str) -> int:        
        # Use a set to check res for duplicate characters
        if len(res) != len(set(res)):
            return 0

        # Recurse through each possible next option
        # and find the best answer
        best = len(res)
        for i in range(pos, len(arr)):
            best = max(best, self.dfs(arr, i + 1, res + arr[i]))
        return best

#cheeky with masks
class Solution:
    def maxLength(self, arr: List[str]) -> int:    
        # Pre-process arr with an optimizing helper
        # which converts each word to its character bitset
        # and then uses a set to prevent duplicate results
        opt_set = set()
        for word in arr:
            self.word_to_bitset(opt_set, word)

        # Convert the set back to an array for iteration
        # then start up the recursive helper
        opt_arr = list(opt_set)
        return self.dfs(opt_arr, 0, 0)
        
    def word_to_bitset(self, opt_arr: Set[int], word: str) -> None:        
        # Initialize an empty int to use as a character bitset
        char_bitset = 0
        for c in word:            
            # If the bitset contains a duplicate character
            # then discard this word with an early return
            # otherwise add the character to the bitset
            mask = 1 << ord(c) - 97
            if char_bitset & mask:
                return
            char_bitset += mask
        
        # Store the length of the word in the unused space
        # then add the completed bitset to our optimized set
        opt_arr.add(char_bitset + (len(word) << 26))
        
    def dfs(self, opt_arr: List[int], pos: int, res: int) -> int:        
        # Separate the parts of the bitset res
        old_chars = res & ((1 << 26) - 1)
        old_len = res >> 26
        best = old_len
        
        # Iterate through the remaining results
        for i in range(pos, len(opt_arr)):
            new_chars = opt_arr[i] & ((1 << 26) - 1)
            new_len = opt_arr[i] >> 26
            
            # If the two bitsets overlap, skip to the next result
            if new_chars & old_chars:
                continue
            
            # Combine the two results and trigger the next recursion
            new_res = new_chars + old_chars + (new_len + old_len << 26)
            best = max(best, self.dfs(opt_arr, i + 1, new_res))
        return best

#################################
# 1662. Check If Two String Arrays are Equivalent (REVISITED)
# 25OCT22
#################################
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        '''
        what if we wanted to compare without using up space for integer conversion
        we would need to concat and compare
        '''
        word1_ptr = 0
        word2_ptr = 0
        
        string1_ptr = 0
        string2_ptr = 0
        
        while word1_ptr < len(word1) and word2_ptr < len(word2):
            #no match
            if word1[word1_ptr][string1_ptr] != word2[word2_ptr][string2_ptr]:
                return False
            else:
                #advance string pointers
                string1_ptr += 1
                string2_ptr += 1
            
            #if we have to move to the next pointer
            if string1_ptr == len(word1[word1_ptr]):
                word1_ptr += 1
                string1_ptr = 0
            
            if string2_ptr == len(word2[word2_ptr]):
                word2_ptr += 1
                string2_ptr = 0
                
        
        return word1_ptr == len(word1) and word2_ptr == len(word2)

##################################
# 1281. Subtract the Product and Sum of Digits of an Integer
# 25OCT22
##################################
class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        SUM = 0
        PROD = 1
        
        while n:
            digit = n % 10
            SUM += digit
            PROD *= digit
            n //= 10
        
        return PROD-SUM

################################
# 149. Max Points on a Line
# 25OCT22
################################
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        '''
        we can simplify the problem and search for the maximum number of points on a line passing through the point i
        
        notice that we only need to consider the next points i+1....N-1 because the maximum number of points containing the point i-1 was already found through point i-2
        
        intution:
            draw the lines passing through the point i and each of the next points
            save these lines in a hash table with counter = 2 (if we have a point and a next point, we can get the slope, and for this line we already have two points)
            then when we encounter another point i + k, and the points i, i+1, and i+k are on the same line,
            then at that slope we increment by 1
        
        how can we save a line?
            if line is horizontal, y == c, the other lines we can get their slopes
            problem is with caluclating slopes with floats (1.2 - 1.0)
            another is the divide by zero with two slopes
            we beed to use a pair of co-prime integers to get the unique slope, rather the gcd(between two integers must be 1)
            example, slopes (1/3), (2/6), (3/9) are all the same slope, but the greates common divisor between them is 1 and 3
            
        algo:
            initiate max number of points to at least 1
            iterate over all points i from 0 to N-2
                for each point i, find the maximum number of points max_count_i on a line passing through the point i
                    initiate the max number of points on a linse passing through the the point i: count = 1
                    iteratoe over the next points from j to (i+1,N-1)
                        if j is a duplicate of i, update a number of duplicates for poit i
                        else:
                            save the line passing through (i,j)
                            update count
                    
                    return max_count_i = count + duplicates
                
                update the result
        
        '''
        N = len(points)
        
        #edge case
        if N < 3:
            return N
        #helper function to get coprime integers to represent the state of a  line
        def coprime(p1,p2):
            x1,y1 = p1
            x2,y2 = p2
            
            delta_x = x1 - x2
            delta_y = y1 - y2
            
            #divide by zero case
            if delta_x == 0:
                return (0,0)
            #horizontal line
            elif delta_y == 0:
                return (float('inf'),float('inf'))
            #negative slope, keep delta_x positive
            elif delta_x < 0:
                delta_x = -delta_x
                delta_y = -delta_y
            
            gcd = math.gcd(delta_x,delta_y)
            slope = (delta_x / gcd, delta_y /gcd)
            return slope
        
        #add a line passing through points i and j
        #update max number of points on a line containin point i
        #update a number of duplicate i point
        #pass by reference for horizontal lines and lines
        def add_line(i,j,count,duplicates,horizontal_lines,lines):
            x1,y1 = i
            x2,y2 = j
            
            #add duplicate point
            if x1 == x2 and y1 == y2:
                duplicates += 1
            #horizontal line
            elif y1 == y2:
                horizontal_lines += 1
                count = max(horizontal_lines,count)
            #add a lines, only slope is needed for hashmap
            else:
                slope = coprime(i,j)
                lines[slope] = lines.get(slope,1) + 1
                count = max(count,lines[slope])
            
            return (count,duplicates)
        
        def get_max_points_for_i(i):
            #init lines for passing through point i
            lines = {}
            horizontal_lines = 1
            #one start with just one point on line
            count = 1
            #currently no duplicates
            duplicates = 0
            #compute lines passing through fixes point i and the next j
            #update the number of points on a line
            for j in range(i+1,N):
                count,duplicates = add_line(points[i],points[j],count,duplicates,horizontal_lines,lines)
            return count + duplicates
        
        max_count = 1
        for i in range(N-1):
            max_count = max(max_count, get_max_points_for_i(i))
        
        return max_count

#official solution
class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        def max_points_on_a_line_containing_point_i(i):
            """
            Compute the max number of points
            for a line containing point i.
            """
            def slope_coprime(x1, y1, x2, y2):
                """ to avoid the precision issue with the float/double number,
                    using a pair of co-prime numbers to represent the slope.
                """
                delta_x, delta_y = x1 - x2, y1 - y2
                if delta_x == 0:    # vertical line
                    return (0, 0)
                elif delta_y == 0:  # horizontal line
                    return (sys.maxsize, sys.maxsize)
                elif delta_x < 0:
                    # to have a consistent representation,
                    #   keep the delta_x always positive.
                    delta_x, delta_y = - delta_x, - delta_y
                gcd = math.gcd(delta_x, delta_y)
                slope = (delta_x / gcd, delta_y / gcd)
                return slope


            def add_line(i, j, count, duplicates):
                """
                Add a line passing through i and j points.
                Update max number of points on a line containing point i.
                Update a number of duplicates of i point.
                """
                # rewrite points as coordinates
                x1 = points[i][0]
                y1 = points[i][1]
                x2 = points[j][0]
                y2 = points[j][1]
                # add a duplicate point
                if x1 == x2 and y1 == y2:  
                    duplicates += 1
                # add a horisontal line : y = const
                elif y1 == y2:
                    nonlocal horizontal_lines
                    horizontal_lines += 1
                    count = max(horizontal_lines, count)
                # add a line : x = slope * y + c
                # only slope is needed for a hash-map
                # since we always start from the same point
                else:
                    slope = slope_coprime(x1, y1, x2, y2)
                    lines[slope] = lines.get(slope, 1) + 1
                    count = max(lines[slope], count)
                return count, duplicates
            
            # init lines passing through point i
            lines, horizontal_lines = {}, 1
            # One starts with just one point on a line : point i.
            count = 1
            # There is no duplicates of a point i so far.
            duplicates = 0
            # Compute lines passing through point i (fixed)
            # and point j (interation).
            # Update in a loop the number of points on a line
            # and the number of duplicates of point i.
            for j in range(i + 1, n):
                count, duplicates = add_line(i, j, count, duplicates)
            return count + duplicates
            
        # If the number of points is less than 3
        # they are all on the same line.
        n = len(points)
        if n < 3:
            return n
        
        max_count = 1
        # Compute in a loop a max number of points 
        # on a line containing point i.
        for i in range(n - 1):
            max_count = max(max_points_on_a_line_containing_point_i(i), max_count)
        return max_count

#another solution
#turns out python can handle floats, we could also use the coprimes as the keys
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) <= 2:
            return len(points)
        
        def find_slope(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            if x1-x2 == 0:
                return inf
            return (y1-y2)/(x1-x2)
        
        ans = 1
        for i, p1 in enumerate(points):
            slopes = defaultdict(int)
            for j, p2 in enumerate(points):
                if i != j:
                    slope = find_slope(p1, p2)
                    slopes[slope] += 1
                    ans = max(slopes[slope], ans)
        return ans+1

#we don't have to check all pairs, but just all pairs i with i+1
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) <= 2:
            return len(points)
        
        def find_slope(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            if x1-x2 == 0:
                return inf
            return (y1-y2)/(x1-x2)
        
        ans = 1
        for i, p1 in enumerate(points):
            slopes = defaultdict(int)
            for j, p2 in enumerate(points[i+1:]):
                slope = find_slope(p1, p2)
                slopes[slope] += 1
                ans = max(slopes[slope], ans)
        return ans+1

################################
# 523. Continuous Subarray Sum
# 26OCT22
################################
class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        '''
        we can use pref sum array here
        and pref_sum[0] = 0
        if we have indices l and r
        (pref_sum[r+1] - pref_sum[l]) % k == 0
        since the subarray must be a multiple of k
        then pref_sum[r+1] % k = pref_sum[l] % k
        
        we just need to check whether we have already seen a pref_sum % k
        
        for an index r, we do not need to check all possible indcies for l, such that l < r
        we are only intereted in the equality:
            pref_sum[l] % k = pref_sum[r+1] % k
        
        we can use a hasha map storing the leftmost index for each remainder of mod k
        in other words, the value for a key x is the leftmost index such that pref[i] = x
        '''
        mapp = {0:0} #prefsum storing pref_sum at this index
        curr_sum = 0
        
        for i in range(len(nums)):
            curr_sum += nums[i]
            #if we have seen the remainder the first time
            if curr_sum % k not in mapp:
                mapp[curr_sum % k] = i+1 #the right most part
            #otherwise try to find its left, which must be less than the right
            elif mapp[curr_sum % k] < i:
                return True
        
        return False

class Solution:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        '''
        another way to see it
        if k = 5, the 6 % k = 1
        if add 5 to this: (6 + 5) % 5 = 1
        we we just need to check if we have seen the remainder before
        
        if we had a contiguous subarray [i,j]
        then pref_sum[j] - pref_sum[i-1] = n*k
        
        then:
            (pref_sum[j] - pref_sum[i-1]) % k = (n*k) % k
            
        sum[j] / k - sum[i - 1] / k = n
        
        which means we only need to enumerate the right index and check wheter the left index i has appear before 
        '''
        N = len(nums)
        
        pref_sum = [0]*(N+1)
        for i in range(1,N+1):
            pref_sum[i] = pref_sum[i-1] + nums[i-1]
        
        seen = set()
        
        for i in range(2,N+1):
            seen.add(pref_sum[i-2] % k)
            #must have size at least 2
            if (pref_sum[i] % k) in seen:
                return True
        return False

class Solution():
    def checkSubarraySum(self, nums, k):
        '''
        Idea: if sum(nums[i:j]) % k == 0 for some i < j, then sum(nums[:j]) % k == sum(nums[:i]) % k. So we just need to use a dictionary to keep track of sum(nums[:i]) % k and the corresponding index i. Once some later sum(nums[:i']) % k == sum(nums[:i]) % k and i' - i > 1, we return True.

		Time complexity: O(n), space complexity: O(min(k, n)) if k != 0, else O(n).

		This helped me visualize the ans. In the example:
		[23,2,6,4,7] k =12

		(23)%12 = 11
		(23+2)%12=1
		(23+2+6)%12 = 7
		(23+2+6+4)%12=11 -> seen before, so return true because:

		(23+2+6+4) - (23) = 12 -> 12 % 12 = 0
        '''
        dic = {0:-1}
        summ = 0
        for i, n in enumerate(nums):
                summ = (summ + n) % k
            if summ not in dic:
                dic[summ] = i
            else:
                if i - dic[summ] >= 2:
                    return True
        return False

#############################
# 273. Integer to English Words
# 26OCT22
#############################
class Solution:
    def numberToWords(self, num: int) -> str:
        '''
        we can divide the number into sets of three and solve
        for example: 
        1234567890
        1.234.567.890
        1 billion 234 million 567 thousand 890
        ONE billion 2 hundred 34 5 hundred 67...
        
        we keep dividing
        
        problem is really jusy about edge cases and how to handle them in the right order
        
        '''
        def one(num):
            switcher = {
                1: 'One',
                2: 'Two',
                3: 'Three',
                4: 'Four',
                5: 'Five',
                6: 'Six',
                7: 'Seven',
                8: 'Eight',
                9: 'Nine'
            }
            return switcher.get(num)

        def two_less_20(num):
            switcher = {
                10: 'Ten',
                11: 'Eleven',
                12: 'Twelve',
                13: 'Thirteen',
                14: 'Fourteen',
                15: 'Fifteen',
                16: 'Sixteen',
                17: 'Seventeen',
                18: 'Eighteen',
                19: 'Nineteen'
            }
            return switcher.get(num)
        
        def ten(num):
            switcher = {
                2: 'Twenty',
                3: 'Thirty',
                4: 'Forty',
                5: 'Fifty',
                6: 'Sixty',
                7: 'Seventy',
                8: 'Eighty',
                9: 'Ninety'
            }
            return switcher.get(num)
        

        def two(num):
            if not num:
                return ''
            elif num < 10:
                return one(num)
            elif num < 20:
                return two_less_20(num)
            else:
                tenner = num // 10
                rest = num - tenner * 10
                return ten(tenner) + ' ' + one(rest) if rest else ten(tenner)
        
        def three(num):
            hundred = num // 100
            rest = num - hundred * 100
            if hundred and rest:
                return one(hundred) + ' Hundred ' + two(rest) 
            elif not hundred and rest: 
                return two(rest)
            elif hundred and not rest:
                return one(hundred) + ' Hundred'
        
        billion = num // 1000000000
        million = (num - billion * 1000000000) // 1000000
        thousand = (num - billion * 1000000000 - million * 1000000) // 1000
        rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000
        
        if not num:
            return 'Zero'
        
        result = ''
        if billion:        
            result = three(billion) + ' Billion'
        if million:
            result += ' ' if result else ''    
            result += three(million) + ' Million'
        if thousand:
            result += ' ' if result else ''
            result += three(thousand) + ' Thousand'
        if rest:
            result += ' ' if result else ''
            result += three(rest)
        return result

#################################
# 835. Image Overlap (REVISTED)
# 27OCT22
##################################
class Solution:
    def largestOverlap(self, img1: List[List[int]], img2: List[List[int]]) -> int:
        '''
        we try all posible shifts and try to find the largest overlap
        one thing to note:
            shifting a matrix in one direction is the same as shifiting the other matrix in the opposite direction
            
        helper function: shift_and_count(x_shift, y_shift,M,R)
            this shifts the matrix M with referernce to R with shitinf coordinate (x_shift,y_shit)
            then we count over lapping
        
        shifting up-left and up-right M wrt to R is the same is shifint down-left, and down-right R wrt M
        
        The algorithm is organized as a loop over all possible combinations of shifting coordinates (x_shift, y_shift).

More specifically, the ranges of x_shift and y_shift are both [0, N-1] where NN is the width of the matrix.

At each iteration, we invoke the function shift_and_count() twice to shift and count the overlapping zone, first with the matrix B as the reference and vice versa.


        '''
        def shift_and_count(x_shift,y_shift,M,R):
            """ 
            Shift the matrix M in up-left and up-right directions 
              and count the ones in the overlapping zone.
            M: matrix to be moved
            R: matrix for reference

            moving one matrix up is equivalent to
            moving the other matrix down
            """
            left_shift_count = 0
            right_shift_count = 0
            
            #index for row into R matrix
            rRow = 0
            for mRow in range(y_shift,len(M)):
                rCol = 0
                for mCol in range(x_shift,len(M)):
                    #shift up and to the left
                    if M[mRow][mCol] == 1 and M[mRow][mCol] == R[rRow][rCol]:
                        left_shift_count += 1
                    #up and to the right
                    if M[mRow][rCol] == 1 and M[mRow][rCol] == R[rRow][mCol]:
                        right_shift_count += 1
                    
                    rCol += 1
                
                rRow += 1
            
            return max(left_shift_count,right_shift_count)
        
        max_overlaps = 0
        for y_shift in range(0,len(img1)):
            for x_shift in range(0,len(img1)):
                #shifts can only be from [0,N-1]
                #print(y_shift,x_shift)
                
                #move img1 up right and up left
                max_overlaps = max(max_overlaps,shift_and_count(x_shift,y_shift,img1,img2))
                
                #move img2 up and to the right, which is the same as moving image1 down right and down left
                max_overlaps = max(max_overlaps,shift_and_count(x_shift,y_shift,img2,img1))
        
        return max_overlaps

class Solution:
    def largestOverlap(self, img1: List[List[int]], img2: List[List[int]]) -> int:
        '''
        the other prolem with examining all the shifts is that we would have to scan through zones that are filled with zeros over and over, and the zero zones are not of interest (think sparse matrix)
        
        we can filter out cells applying a linear transformatino to align the cells
        
        say we have two non zero cells in img1 (x1,y1) and in img2 (x2,y2)
        to align these would need a transformatino vector
        V_{12} = (x2 - x1, y2-y1) such that img1 + V_{12} = img2
        
        the key insight is that all the cells in the same overlapping zone would share the same linear transformation vector
        
        we can use the transformation vector as key to group the non zeor cells betwtween two matrics
        each group represents an overlapping zone
        
        first filter out all non zero cells in each matrix repsctively
        do cartesian products on the non zero cells
        for each pair of products, calculate the linear transformation vector
        then count the number of pairs that have the same transformation vector
        '''
        size = len(img1)
        
        img1_ones = []
        img2_ones = []
        vectors = defaultdict(int)
        ans = 0
        
        for i in range(size):
            for j in range(size):
                if img1[i][j] == 1:
                    img1_ones.append((i,j))
                if img2[i][j] == 1:
                    img2_ones.append((i,j))
        
        #for the cartesian product find the unqiue transformaiton vector
        for (x1,y1) in img1_ones:
            for (x2,y2) in img2_ones:
                vec = (x2 - x1,y2-y1)
                vectors[vec] += 1
                ans = max(ans,vectors[vec])
        
        return ans

#convolutions
class Solution:
    def largestOverlap(self, img1: List[List[int]], img2: List[List[int]]) -> int:
        '''
        turns out the number of overlapping 1's between two images is the convolution of the two images
        we pad im2 with zeros and for each kernel in the padded img2, find the convolution with img1
        take the max
        '''
        import numpy as np
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        dim = len(img1)
        
        #pad b
        img2_padded = np.pad(img2,dim, mode='constant', constant_values=(0, 0))
        
        max_overlaps = 0
        for x_shift in range(dim*2):
            for y_shift in range(dim* 2):
                # extract a kernel from the padded matrix
                kernel = img2_padded[x_shift:x_shift+dim, y_shift:y_shift+dim]
                # convolution between A and kernel
                non_zeros = np.sum(img1 * kernel)
                max_overlaps = max(max_overlaps, non_zeros)

        return max_overlaps

#############################
# 1365. How Many Numbers Are Smaller Than the Current Number
# 28OCT22
#############################
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        '''
        i could sort, find the position in the array, call it i
        then there must b i-1 smaller numbers than it
        '''
        N = len(nums)
        sorted_nums = sorted(nums)
        ans = [0]*N
        
        for i in range(N):
            curr_num = nums[i]
            #find it using binary serach
            index = bisect_left(sorted_nums,curr_num)
            ans[i] = index
        
        return ans

class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        '''
        use hashmap to stor number to index in sorted array
        '''
        mapp = {}
        for i,num in enumerate(sorted(nums)):
            if num not in mapp:
                mapp[num] = i
        
        return [mapp[num] for num in nums]

class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        '''
        bucket sort since we know the largest numbers
        '''
        buckets = [0]*102
        #record the number of ocurrences
        for num in nums:
            buckets[num+1] += 1
            
        #record number less than
        for i in range(1,len(buckets)):
            buckets[i] += buckets[i-1]
        return [buckets[num] for num in nums]

#################################
# 2136. Earliest Possible Day of Full Bloom
# 29OCT22
#################################
'''
JAVA solution
class Solution {
    public int earliestFullBloom(int[] plantTime, int[] growTime) {
        /*
        we have N flower seeds
        we are given plantTimes and blom times
        we can only plant one seed on a day, you do not have to plan the same seed on consecutive days
        while planting we cannot plant anothe seed, rather once we try planting a seed, we cannot plant another one until it is planted
        grow time is the number of days it takes a plant to bloom once planted
        return the earliest posible day all seeds are blooming
        
        hint 1:
            list planting diagram with rows
            a row i is above another row j if the last day planting seed i is ahead of the last day for seed j
        hint 2:
            it does not have any advantage to plant seed j before completing planting seed i
            but i could potentailly delay the completion of seed
        hint 3:
            sort seeds by their growTime descendingly
            bloomTime of the seed is the sum of plantTime of all seeds precedeing this seed play the growTime of this seed
        
        hint4:
            the sed to bloom last domniates the final answer
            exchanging the planting of this seed with another seed with a larger or smaller growTime will result in a potentially worse answer
            
        intuition:
            notice that once we plant a seed, we can continue to plant another seed (the concsecutive part)
            there always exists an optimal solution with seeds planted during consecutive days
        so it is always better to plant a seed with a longer growth time before the one with a shorter growtime 
            it takes time to grow, so we might as well us the time it takes to grow, to plant other seeds
        
        let's call the answer t and say we have just planted seed i
        then the seed has to begin growing no later than t - growTime[i]
        the larger the growTime the seed has, the sooner 
        
        */
        int N = growTime.length;
        List<List<Integer>> plant_grow_times = new ArrayList();
        
        //add to list as [grow,plant]
        for (int i = 0; i < N; ++i){
            //make entry
            List<Integer> entry =  new ArrayList();
            entry.add(growTime[i]);
            entry.add(plantTime[i]);
            //add
            plant_grow_times.add(entry);
                
        }
        
        //sort on growth times deceding, comparator with index
        Collections.sort(plant_grow_times, Comparator.comparing(x -> -x.get(0)));
        
        int ans = 0;
        int curr_plant_time = 0;
        
        for (List<Integer> pair: plant_grow_times){
            curr_plant_time += pair.get(1);
            ans = Math.max(ans,curr_plant_time + pair.get(0));

        }
        return ans;
        
        
    }
}
'''
class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        '''
        we have N flower seeds
        we are given plantTimes and blom times
        we can only plant one seed on a day, you do not have to plan the same seed on consecutive days
        while planting we cannot plant anothe seed, rather once we try planting a seed, we cannot plant another one until it is planted
        grow time is the number of days it takes a plant to bloom once planted
        return the earliest posible day all seeds are blooming
        '''
        curr_plant_time = 0
        res = 0
        indices = sorted(range(len(plantTime)), key = lambda x: -growTime[x])
        
        for i in indices:
            curr_plant_time += plantTime[i]
            res = max(res,curr_plant_time+growTime[i])
        
        return res

#another way to think is imagine all the plants start growing at the same time
class Solution:
    def earliestFullBloom(self, plantTime: List[int], growTime: List[int]) -> int:
        res = 0
        for grow, plant in sorted(zip(growTime, plantTime)):
            res = max(res, grow) + plant
        return res

#####################################################################
# 1293. Shortest Path in a Grid with Obstacles Elimination (REVISTED)
# 30OCT22
######################################################################
class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        '''
        we can use A*
            we priortize options using a given heurtistic (since we start upper left and need to go bottom right)
            we must give preference to the moves down and right
        
        informed search, or best first search
        minimizes the following function;
            f(n) = g(n) + h(n)
        
        n: is a specific step during the exploration
        g(n): the cost to reach the step n, here the cost referts to the distance traveled so far 
        h(n): a heuristic estimation on the cost to reach the destination from step n, in this case it is the distance remaining
        f(n): the estimated total cost to reach the desitination if on takes the step n
        
        function needs to be admissiable:
            it never overestimates the cost, otherwise it could not gurantee that thepath we find is the shortest one
        
        we can continue to use a deque and a seen set, but we make small modificatons
        
        1. we use priority qye to store the order of the visits, order of the visited is based on the estiamte total cost function f(n)
        2. for each element in q, we add one more pice:
            estimated total cost to reach the destination at each step
        3. add another heuristic condition that allows us to determine the length of the shortest path without exploration
            a. at any step, if the remainin quota to eliminate the obstalces is larger then the lgnth of the estimated shortets path, then the lenght of the remaining path is the manhattan distance (duh!)
            b. the condition can also be interprested as if we have sufficient capacity to remove any obstalce to the desitation
            c. we apply this check at the beginning of the loop
            
        A* is just BFS with pruning
        '''
        rows, cols = len(grid), len(grid[0])
        target = (rows - 1, cols - 1)

        def manhattan_distance(row, col):
            return target[0] - row + target[1] - col

        # (row, col, remaining_elimination)
        state = (0, 0, k)

        # (estimation, steps, state)
        # h(n) = manhattan distance,  g(n) = 0
        queue = [(manhattan_distance(0, 0), 0, state)]
        seen = set([state])

        while queue:
            estimation, steps, (row, col, remain_eliminations) = heapq.heappop(queue)

            # we can reach the target in the shortest path (manhattan distance),
            #   even if the remaining steps are all obstacles
            remain_min_distance = estimation - steps
            if remain_min_distance <= remain_eliminations:
                return estimation

            # explore the four directions in the next step
            for new_row, new_col in [(row, col + 1), (row + 1, col), (row, col - 1), (row - 1, col)]:
                # if (new_row, new_col) is within the grid boundaries
                if (0 <= new_row < rows) and (0 <= new_col < cols):
                    new_eliminations = remain_eliminations - grid[new_row][new_col]
                    new_state = (new_row, new_col, new_eliminations)

                    # if the next direction is worth exploring
                    if new_eliminations >= 0 and new_state not in seen:
                        seen.add(new_state)
                        new_estimation = manhattan_distance(new_row, new_col) + steps + 1
                        heapq.heappush(queue, (new_estimation, steps + 1, new_state))

        # did not reach the target
        return -1













