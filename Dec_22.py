####################################
# 1657. Determine if Two Strings Are Close (REVISTED)
# 02DEC22
####################################
#using counts and check counts of chars are the same
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        '''
        strings are close if we can obtain 1 from the other using the following transformations
        1. swap any two existing chars
        2. transform every occurrence of one char into another existing char
        
        the first case allows me to generate any permutation of the word
        for the first part we can just check if word1 is a perm of word2 or word2 is a perm of word1
        
        characters must be the same
        and the frequncies of characters must be the same
        
        the implication of rule 2 is that if we have u counts of char i
        and v counts of char j
        if we were to swap char i to char j, then the counts must also swap
        and if word1 is a transormation of word2, the freqeusnt of counts should be the same
        '''
        if set(word1) == set(word2):
            #check counts
            counts1 = Counter(word1)
            counts2 = Counter(word2)
            if Counter(counts1.values()) == Counter(counts2.values()):
                return True
            else:
                return False
        else:
            return False


class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        '''
        use integer to store presnce of char in word
        and array to store counts
        '''
        if len(word1) != len(word2):
            return False
        
        counts1 = [0]*26
        counts2 = [0]*26
        
        seen_chars1 = 0
        seen_chars2 = 0
        
        #they must be the same size at this point
        for i,j in zip(word1,word2):
            first = ord(i) - ord('a')
            second = ord(j) - ord('a')
            
            counts1[first] += 1
            seen_chars1 = seen_chars1 | (1 << first)
            
            counts2[second] += 1
            seen_chars2 = seen_chars2 | (1 << second)
        
        
        if seen_chars1 != seen_chars2:
            return False
        
        counts1.sort()
        counts2.sort()
        
        for i in range(26):
            if counts1[i] != counts2[i]:
                return False
        return True

###################################
# 2396. Strictly Palindromic Number
# 02DEC22
###################################
#this shit fucking works?! LMAOOOO
class Solution:
    def isStrictlyPalindromic(self, n: int) -> bool:
        '''
        just get base represenstation of a number
        in log n time
        then check if it is a palindrom
        '''
        def getBaseRep(n,b):
            bits = []
            while n:
                bits.append(n % b)
                n //= b
            return bits
        
        for b in range(2,n-2+1):
            bits = getBaseRep(n,b)
            if bits != bits[::-1]:
                return False
        
        return True

class Solution:
    def isStrictlyPalindromic(self, n: int) -> bool:
        '''
        turns out we just return false

        Intuition
		The condition is extreme hard to satisfy, think about it...
		for every base b between 2 and n - 2...
		4 is not strictly palindromic number
		5 is not strictly palindromic number
		..
		then the bigger, the more impossible.
		Just return false


		Prove
		4 = 100 (base 2), so 4 is not strictly palindromic number
		for n > 4, consider the base n - 2.
		In base n - 1, n = 11.
		In base n - 2, n = 12, so n is not strictly palindromic number.

		There is no strictly palindromic number n where n >= 4


		More
		I think it may make some sense to ask if there a base b
		between 2 and n - 2 that n is palindromic,
		otherwise why it bothers to mention n - 2?

		It's n - 2, not n - 1,
		since for all n > 1,
		n is 11 in base n - 2.
		(Because n = (n - 1) + (1))

		Then it's at least a algorithme problem to solve,
		instead of a brain-teaser.

		Maybe Leetcode just gave a wrong description.


		Complexity
		Time O(1)
	Space O(1)

        '''
        return False

###############################################
# 451. Sort Characters By Frequency (REVISTED)
# 03NOV22
##############################################
class Solution:
    def frequencySort(self, s: str) -> str:
        '''
        we can use a variant of bucket sort
        first find the max frequency among all the char frequencies
        then we have buckets for each of the frequencies up to and including the max freq
        in each bucket we just put the letter
        then re-traverse the buckets and grab each letter that number of times
        
        '''
        #find max frequqency
        counts = Counter(s)
        max_freq = max(counts.values())
        
        buckets = [[] for _ in range(max_freq + 1)]
        
        #each bucket represents a frequency up to max freq, and for each freq add the letter into it
        for k,v in counts.items():
            buckets[v].append(k)
        
        #traverse the buckets and add to ans
        ans = ""
        for i in range(len(buckets)):
            for char in buckets[i]:
                ans += char*i
        
        return ans[::-1]

#############################################
# 1823. Find the Winner of the Circular Game
# 03DEC22
#############################################
#yassss! lets fucking goooooo!
class Node:
    def __init__(self,val,next=None):
        self.val = val
        self.next = next
        
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        i could just simulate the game until there is only one player left
        i can mimic this with a linked list

        '''
        #first make the circular linked list
        dummy = Node(-1)
        curr = dummy
        for i in range(1,n+1):
            newNode = Node(i)
            curr.next = newNode
            curr = curr.next
        
        curr.next = dummy.next
        
        curr = dummy.next
        
        #now we can simulate
        lost = set()
        
        while len(lost) < n-1:
            prev = curr
            for _ in range(k-1):
                prev = curr
                curr = curr.next
            prev.next = curr.next
            lost.add(curr)
            curr = prev.next
            
        return curr.val

#with an array
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        we can also simulate with an array
        '''
        
        circle = [i for i in range(1,n+1)]
        last_idx = 0
        
        while len(circle) > 1:
            #move index
            last_idx = (last_idx + k - 1) % len(circle)
            #remove
            del circle[last_idx]
        
        return circle[0]

class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        we can also use a q
        we shuffle array by moving the first element back to the end for each step k
        then once are out of steps k, we remove the first element
        '''
        q = deque([i for i in range(1,n+1)])
        
        while len(q) > 1:
            x = k
            print(q)
            while x > 1:
                r = q[0]
                q.popleft()
                q.append(r)
                x -= 1
            q.popleft()
        
        return q[0]

#dp
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        this is the josephus problem
        to find an O(N) solution we first need to model the recurrent
        then we can build bottom up dp using constant space
        
        recurrence
        dp(n,k) = (dp(n-1,k) + (k-1)) % n +1
        
        After the first person (kth from the beginning) is killed, n-1 persons are left. Make recursive call for Josephus(n – 1, k) to get the position with n-1 persons. But the position returned by Josephus(n – 1, k) will consider the position starting from k%n + 1. So make adjustments to the position returned by Josephus(n – 1, k). 
        '''
        memo = {}
        def dp(n):
            if n == 1:
                return 1
            if (n) in memo:
                return memo[(n,k)]
            ans = (dp(n-1) + k - 1) % n + 1
            memo[(n)] = ans
            return ans
        
        return dp(n)


#constant space
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        constance space
        '''
        prev = 1
        curr = 0
        for x in range(n+1):
            curr = (prev + k -1) % n + 1
            prev = curr
        
        return curr


##################################
# 2256. Minimum Average Difference
# 04DEC22
##################################
class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:
        '''
        we need to calculate the average difference fore ach index i, and return the smallest one
        
        averae diferent for an index i is
            abs different between average of first i+1 elements and average of n-i-1 elements
        
        we can use prefix sum array to reduce over head needed then just return the minimum
        
        '''
        pref_sum = [0]
        for num in nums:
            pref_sum.append(pref_sum[-1] + num)
        
        #to get sum for nums between i and j, we want pref_sum[j+1] - pref_sum[i]
        
        avg_diff = float('inf')
        index = 0
        N = len(nums)
        
        for i in range(N):
            #need sum from i+1,N
            right_sum = pref_sum[N] - pref_sum[i+1]
            left_sum = pref_sum[i+1] - pref_sum[0]
            
            #calcualte average difference
            size_right = N-i-1
            size_left = i + 1
            
            #get avg diff
            right_avg_diff = right_sum // size_right if size_right != 0 else 0
            left_avg_diff = left_sum // size_left if size_left != 0 else 0
            
            local_avg_diff = abs(left_avg_diff - right_avg_diff)
            if local_avg_diff < avg_diff:
                avg_diff = local_avg_diff
                index = i
        
        return index

################################
# 942. DI String Match
# 05NOV22
################################
class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        '''
        build the string smartly by taking numbers from lo and hi ends
        i guessed i must alternating taking lo and hi because if i take the lowest low first or the highest hi first
        im guaranteed to have another avaialble smaller of higher number
        
        problem is with logic in taking the last number
        to s, append the opposite of the last digit
        '''
        s += 'I' if s[-1] == 'D' else 'I'
        N = len(s)
        nums = [i for i in range(N)]
        lo = 0
        hi = len(nums) - 1
        
        ans = []
        
        for ch in s:
            if ch == 'I':
                ans.append(nums[lo])
                lo += 1
            else:
                ans.append(nums[hi])
                hi -= 1
        
        return ans

###################################################################
# 2035. Partition Array Into Two Arrays to Minimize Sum Difference
# 05DEC22
####################################################################
#brute force TLE,build up one paritions and just check
class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        '''
        we want to get two subsequences if len(nums) // 2 such that the absolulte diffeence between the sums of the arrays is minimzed
        we want to return the minimum possible asbsolute difference
        
        n is small, whic allows for an expoenential time comleixty
        
        first try generating all paritions 
        
        if were to generate

        '''
        sum_nums = sum(nums)
        N = len(nums)
        
        self.ans = float('inf')
        
        def rec(i,path):
            if i == N:
                return
            if len(path) == N // 2:
                #get sum of first partition
                first = sum(path)
                second = sum_nums - first
                self.ans = min(self.ans,abs(first-second))
                return
            #take it
            rec(i+1,path +[nums[i]])
            rec(i+1,path)
        rec(0,[])
        
        return self.ans

class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        '''
        this is a very hard problem lets examine the 4 hints
        1. target sum for the partitions would be sum(nums) /2, obvie to minimize the absolute difference
        2. abritiraly divides nums into tow halves
        3. for both halves pre-calculate a 2d array where the kth index wils tore all possible sum values if only k element from this half are added
        4. For each sum of k elements in the first half, find the best sum of n-k elements in the second half such that the two sums add up to a value closest to the target sum from hint 1. These two subsets will form one array of the partition.
        
        https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/discuss/1515202/Replace-up-to-n-2-elements
        
        notes on why dp cannot be useful
            even with just 30 eleemnts, there are 155,117,520 ways of picking 15 elements out of 30 (15 chose 30)
            the vaue of range for each num is wide, and so we may not have repeating sums, and we would just be memozing everything
            
        inution:
            we can pick k elements from the first array and try to combine them with n - k elements from the second one
            this requires to caclcualte the sum of up to n-1 elemeents, and our solutions can contain only n/2 elements
            
        range for k?
            from 0 to n/2
            say we have only 11 elements in each array, replace 7 elemements in the first array with 7 elements in the second has the same effect as replacing the other 4 eleemnts
            the sums of the array will swap, but we don't care because their absolute difference will be the same
            
        algo:
            generate all possible sums for the frsit ands econd arrays
            then for k in range [1, n//2] we pick a sum from the first array and binarys earch for a compelement sum in the second array
            the complement value is calcualted to minimuze the difference betwen two arrays
                comp = (sum1 + sum2 ) / 2 - (nsum1 - s1_using_k)
            half of the array's sum is the ideal state when two array sums are the same and s1; is one of the generated sum of k elemenets in the first array
            
        subset generation (which is number of possible sums) is 2**N
        notes on updating the min
        (curleftSum + *it) is the sum of one side of our N/2 elements. Consider this as sum A.
The sum of other N/2 elements would be totalSum - A. Correct?
Absolute difference would be = abs((totalSum - A) - A) = abs(totalSum - 2*A) = abs(totalSum - 2 * (curleftSum + *it)), same as we want.
        '''
        #first generate sums of size N//2 for the left and right parts
        #for each part mapp a sum to using k elements
        
        #precomputations
        N = len(nums)
        left_sum = sum(nums[:N//2])
        right_sum = sum(nums[N//2:])
        all_sum = left_sum + right_sum
        
        left_sums = defaultdict(list)
        right_sums = defaultdict(list)
        
        res = float('inf')
        
        def dfs(start,end,k,curr_sum,memo):
            if start == end or k >= N//2:
                memo[k].append(curr_sum)
                return
            #don't use it
            dfs(start + 1,end,k,curr_sum,memo)
            #use it
            dfs(start+1,end,k+1,curr_sum + nums[start],memo)
        
        #leftside
        dfs(0,N//2,0,0,left_sums)
        #rightside
        dfs(N//2, N,0,0,right_sums)

        #for this part we are comparing all all pairs of left_sum and right_sum, regardless if any size is not N//2
        #it will eventually be minimum when left_sum is as close to right_sum as possible

        #sum using k elements
        for k in range(N//2 + 1):
            #start with a left sum and look for a right sum so as to minizise the abs diffeence
            right_sums[k].sort()
            #for each left sum using k elements
            for s1k in left_sums[k]:
                #complement is sum to seach for in right_sums
                comp = all_sum // 2 - (left_sum - s1k)
                #to find the actual absolute difference
                diff = left_sum - right_sum - s1k * 2
                j = bisect.bisect_left(right_sums[k], comp)
                #in the case it is the first sum in right_sums[k]
                #bisect left works fin even if the lower bound is zero
                #even at upper bounds we can still index into the right sums array
                if j < len(right_sums[k]):
                    res = min(res, abs(diff + right_sums[k][j] * 2))
        return res
        

#another, but using i and N//2 - i for using sums of left and righ respecitvely
class Solution:
    def minimumDifference(self, nums: List[int]) -> int:
        '''
        another way
        '''
        #first generate sums of size N//2 for the left and right parts
        #for each part mapp a sum to using k elements
        
        #precomputations
        N = len(nums)
        left_sum = sum(nums[:N//2])
        right_sum = sum(nums[N//2:])
        all_sum = left_sum + right_sum
        
        left_sums = defaultdict(list)
        right_sums = defaultdict(list)
        
        res = float('inf')
        
        def dfs(start,end,k,curr_sum,memo):
            if start == end or k >= N//2:
                memo[k].append(curr_sum)
                return
            #don't use it
            dfs(start + 1,end,k,curr_sum,memo)
            #use it
            dfs(start+1,end,k+1,curr_sum + nums[start],memo)
        
        #leftside
        dfs(0,N//2,0,0,left_sums)
        #rightside
        dfs(N//2, N,0,0,right_sums)

        #sum using k elements
        for k in range(N//2 + 1):
            #start with a left sum and look for a right sum so as to minizise the abs diffeence
            #if we are using i elements on the left, we must be using N//2 - i on the right
            #so get those sumes
            r = right_sums[N // 2 - k]
            #sort
            r.sort()
            
            for curr_left_sum in left_sums[k]:
                needSumFromRight = (all_sum) // 2 - curr_left_sum
                j = bisect.bisect_left(r,needSumFromRight)
                if j < len(r):
                    res = min(res,abs(all_sum - 2*(curr_left_sum + r[j])))
        
        return res

##############################
# 1528. Shuffle String
# 06DEC22
###############################
class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        '''
        just swap in place
        '''
        s = list(s)
        N = len(indices)
        
        ans = [0]*N
        
        for i in range(N):
            swap_idx = indices[i]
            ans[swap_idx] = s[i]
            
        return "".join(ans)

#sorting
class Solution:
    def restoreString(self, s: str, p: List[int]) -> str:
        return ''.join([v for (_,v) in sorted(zip(p,s))])


#cycle sort
class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        '''
        we can use cyclic sotring
        we keep swapping until we get back the original index
        
        we are essentialy sorting the index mapped to the letters!
        '''
        s = list(s)
        N = len(s)
        
        for i in range(N):
            #while the index doest point to itself
            #rather we keep swapping until we get back to ouriginal index
            while i != indices[i]:
                print(i,indices[i],s)
                
                #swap the characters in place
                s[i],s[indices[i]] = s[indices[i]],s[i]
                #swap the indices
                j = indices[i]
                indices[i], indices[j] = indices[j],indices[i]
            print(i,indices[i])
                
        print(indices)
        return "".join(s)


####################################
# 938. Range Sum of BST (REVISITED)
# 07DEC22
####################################
#bottom up not using BST propert
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        '''
        let dp(node) be the sum nodes whose values are in range
        dp(node) = dp(node.left) + dp(node.right)
        '''
        def dp(node):
            if not node:
                return 0
            
            to_add = node.val if (low <= node.val <= high) else 0
            left = dp(node.left)
            right = dp(node.right)
            
            ans = to_add + left + right
            return ans
        
        return dp(root)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        '''
        let dp(node) be the sum nodes whose values are in range
        dp(node) = dp(node.left) + dp(node.right)
        '''
        def dp(node):
            if not node:
                return 0
            
            if node.val < low:
                return dp(node.right)
            elif node.val > high:
                return dp(node.left)
            else:
                left = dp(node.left)
                right = dp(node.right)
                return left + right + node.val
            
        return dp(root)

############################
# 690. Employee Importance
# 07DEC22
############################
#close one....
"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        '''
        this is just a dfs problem
        we are given a list of employee classes and we want to return the importance for an employee and all his/her suboorindates
        it may be the case that the first employee does not much the id, in which case we keep dfsing
        
        keep global importance
        then dfs, once we havefound the id, mark as true, and dfs again
        if true keep incrementing the importance
        
        dang it, suborindates are a list of ints not employees
        '''
        #id's may not be in order, so from list of employees, turn into hashmap where (ID mapps to index in employees)
        mapp = defaultdict()
        
        for i,e in enumerate(employees):
            mapp[e.id]  = i
        
        self.ans = 0
        
        def dfs(emp,found):
            if found:
                self.ans += emp.importance
            #this is a list of ids
            for sub in emp.subordinates:
                #get the object
                sub_object = employees[mapp[sub]]
                dfs(sub_object,emp.id == id)
        
        for e in employees:
            if e.id == id:
                self.ans += e.importance
                dfs(e,False)
        
        return self.ans


"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        '''
        turns out we can just dfs from fromt the given id, and keep icrmenting self.ans
        '''
        mapp = defaultdict()
        
        for i,e in enumerate(employees):
            mapp[e.id]  = e
        
        self.ans = 0
        
        def dfs(emp_id):
            #get the employee
            curr_emp = mapp[emp_id]
            #add its importance
            self.ans += curr_emp.importance
            for sub in curr_emp.subordinates:
                dfs(sub)
                
        dfs(id)
        return self.ans

#bottome up
"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        '''
        turns out we can just dfs from fromt the given id, and keep icrmenting self.ans
        '''
        mapp = defaultdict()
        
        for i,e in enumerate(employees):
            mapp[e.id]  = e
        
        
        def dfs(emp_id):
            #get the employee
            curr_emp = mapp[emp_id]
            #add its importance
            curr_importance = curr_emp.importance
            for sub in curr_emp.subordinates:
                curr_importance += dfs(sub)
            
            return curr_importance
                
        return dfs(id)

"""
# Definition for Employee.
class Employee:
    def __init__(self, id: int, importance: int, subordinates: List[int]):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates
"""

class Solution:
    def getImportance(self, employees: List['Employee'], id: int) -> int:
        '''
        and we can use bfs
        '''
        mapp = defaultdict()
        
        for i,e in enumerate(employees):
            mapp[e.id]  = e
        
        ans = 0
        
        q = deque([id])
        
        while q:
            curr_emp = q.popleft()
            curr_emp = mapp[curr_emp]
            ans += curr_emp.importance
            for sub in curr_emp.subordinates:
                q.append(sub)
        
        return ans
        

##############################
# 872. Leaf-Similar Trees
# 08DEC22
##############################
#stupid way
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        '''
        both trees to get their leaves
        dump into aux array and check that arrays are euqal
        '''
        leaves1 = []
        leaves2 = []
        
        def dfs(node,leaves):
            if not node:
                return
            dfs(node.left,leaves)
            if not node.left and not node.right:
                leaves.append(node.val)
            dfs(node.right,leaves)
            
        dfs(root1,leaves1)
        dfs(root2,leaves2)
        
        return leaves1 == leaves2

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        '''
        don't forget the yeild function!
        '''
        def dfs(node):
            if node:
                if not node.left and not node.right:
                    yield node.val
                yield from dfs(node.left)
                yield from dfs(node.right)
        
        return list(dfs(root1)) == list(dfs(root2))

###################################################################
# 1026. Maximum Difference Between Node and Ancestor (REIVISTED)
# 09DEC22
###################################################################
#top down
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        '''
        we want to find the max value v for which:
            there exists two different nodes a nad b
            and v = abs(a.val - b.val)
        
        a node a is an ancestor of b if either any child of a == b 
        or any child of a is an ancestory of b
        
        the hint says that for each subtree, find the min value and max value of its descendatns
        if we know the max and mini values for a subtree
        we would only need to check abs(root.val - min) and abs(root.val - max)
        so for each subtree, find the max and the min
        '''
        self.ans = 0
        if not root:
            return 0
        
        def dp(node,curr_min,curr_max):
            if not node:
                return
            first = abs(node.val - curr_min)
            second = abs(node.val - curr_max)
            self.ans = max(self.ans,first,second)
            dp(node.left, min(curr_min,node.val),max(curr_max,node.val))
            dp(node.right, min(curr_min,node.val),max(curr_max,node.val))
        
        dp(root,root.val,root.val)
        return self.ans
            
#think about what to return up to the parent
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        '''
        for each subtree we want to know the min and the max then
        we just take the maximim absolute difference
        if we define
        dp(node,min,max) is the answer to the largeat maximum differenne for a subtree
        then
        dp(node,min,max) =
            left = dp(node.left, min(min,node.val),max(max,node.val))
            right = dp(node.right, min(min,node.val),max(max,node.val))
            
        base case is empty node
        so we just return max-min
        '''
        
        if not root:
            return 0
        
        def dp(node,curr_min,curr_max):
            if not node:
                return curr_max - curr_min
            
            left = dp(node.left,min(curr_min,node.val),max(curr_max,node.val))
            right = dp(node.right,min(curr_min,node.val),max(curr_max,node.val))
            return max(left,right)
        
        return dp(root,root.val,root.val)

#############################
# 124. Binary Tree Maximum Path Sum (REVISTED)
# 11DEC22
##############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        '''
        official writeup, turns out to be post order
        the problem is that a path does not have to be root to leaf,
        if the path were to include the root:
            1. it could start at the root and go down through the left child (but would go down to further)
            2. same for the right
            3. involces both left and right
            4. only involves the root itself
        
        we can say this about any node
        we need to find the gain in going left and the gain in going right
        it makes sense to only go in the direction with the largest gain
        we examine children first before the node
            so use post order
        
        but what is the path doesn't include this root (or doesn't include the current node for that matter)
        
        the path sum contributed by the subtre can be dervied from a path that cinludes at mose one child of the root
        we cannot include both children, because including both children would have to make a fork
        and since the root is already included
        path consists of at most one child of the oort
        
        base case, empty node return 0
        since there is nothing to contribute to the path
        '''
        self.ans = float('-inf')
        
        def dp(node):
            if not node:
                return 0
            
            #get the gain from the children
            left_gain = max(dp(node.left),0)
            right_gain = max(dp(node.right),0)
            
            #find the nex max path
            new_path = left_gain + right_gain + node.val
            
            #updat
            self.ans = max(self.ans,new_path)
            
            #we need to retun the max path for this node
            return node.val + max(left_gain,right_gain)
        
        dp(root)
        return self.ans

#########################################################################
# 323. Number of Connected Components in an Undirected Graph (REVISTED)
# 12DEC22
########################################################################
#just a dsu review
class DSU:
    def __init__(self,size):
        self.parent = [i for i in range(size)]
        self.size = [1 for _ in range(size)]
        
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        
        #assign to bigger grous
        if parent_x == parent_y:
            return
        if self.size[parent_x] > self.size[parent_y]:
            self.size[parent_x] += self.size[parent_y]
            self.parent[parent_y] = parent_x
            self.size[parent_y] = 0
        elif self.size[parent_y] > self.size[parent_x]:
            self.size[parent_y] += self.size[parent_x]
            self.parent[parent_x] = parent_y
            self.size[parent_x] = 0
        else:
            self.size[parent_x] += self.size[parent_y]
            self.parent[parent_y] = parent_x
            self.size[parent_y] = 0
            

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        ds = DSU(n)
        for a,b in edges:
            ds.union(a,b)
        
        ans = 0
        for size in ds.size:
            if size != 0: 
                ans += 1
        
        return ans

class UnionFind:
    def __init__(self, n: int) -> None:
        self.root = [i for i in range(n)]
        self.rank = [1] * n
        self.count = n
        
    def find(self, x: int) -> int:
        if x == self.root[x]:
            return x
        
        self.root[x] = self.find(self.root[x])
        return self.root[x]
    
    def union(self, x: int, y: int) -> None:
        rootx = self.find(x)
        rooty = self.find(y)
        if rootx != rooty:
            self.count -= 1
            if self.rank[rootx] > self.rank[rooty]:
                self.root[rooty] = rootx
            elif self.rank[rootx] < self.rank[rooty]:
                self.root[rootx] = rooty
            else:
                self.root[rooty] = rootx
                self.rank[rootx] += 1
        
        

class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        uf = UnionFind(n)
        for edge in edges:
            uf.union(*edge)
            
        return uf.count

######################################
# 1066. Campus Bikes II (REVISTED)
# 20DEC22
######################################
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> int:
        '''
        we can treat this is djikstras
        intuition:
            from the bottom up approach we built each new mask from 0 to 2^M
            insteaf of traversing mask in sequentnial order, we find masks that have the smallest total ditance
        
        to find next mask, we use a prioty queu (heap)
        
        algo;
            inital state (0,0), immplying empty mask has 0 for min dist
            pop pair from heap
                discard and continue to the next pair if currmask has already been visited
            ad next start and min dist
            return currdsit is workerIndex == n
        
        '''
        def getDist(worker,bike):
            x = abs(worker[0] - bike[0])
            y = abs(worker[1] - bike[1])
            return x + y
    
        #brian kernighan, counting ones in bitset
        def countOnes(mask):
            count = 0
            while mask:
                count += 1
                mask = mask & (mask-1)
            return count
        
        n = len(workers)
        m = len(bikes)
        
        heap = [(0,0)] #entry is going to be dist and (0,0)
        visited = set()
        #python is min heap
        
        while heap:
            dist,mask = heapq.heappop(heap)
            
            #if i've already seen this state
            if mask in visited:
                continue
            visited.add(mask)
            
            #next worker index is the number of 1's in the mask
            worker_index = countOnes(mask)
            
            #if we have mapped all workers
            if worker_index == n:
                return dist
            
            for bike in range(m):
                #check if we have taken this bike
                if not (mask & (1 << bike)):
                    #get the next min dist
                    next_min_dist = dist + getDist(workers[worker_index],bikes[bike])
                    #get the next maxk
                    next_mask = mask | (1 << bike)
                    #push bacnk on to heap
                    heapq.heappush(heap,(next_min_dist, next_mask))
        
        return -1