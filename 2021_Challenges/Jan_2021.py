###############################################
#Check Array Formation Through Concatenation
###############################################

class Solution(object):
    def canFormArray(self, arr, pieces):
        """
        :type arr: List[int]
        :type pieces: List[List[int]]
        :rtype: bool
        """
        '''
        this was a tricky problem lets go over each of the solution
        O(N^2)
        we can just match them one by one
        if we can't find a piece that starts with the ith arr, then return false
        then we need to examine the elements in the first matched piece
        one the pieces match just move i to the last mached picece
        then continue
        otherwise return False
        algo:
            1. init index t to the current matching index in arr
            2. iterate over pieces to find peice starting with arr[i], return false other wise
            3. use the matched piece to match arr's sublist with i, return false other wise
            4. move up i
            5. return true until we get to the end of the array
        '''

        N = len(arr)
        i = 0
        while i < N:
            for p in pieces:
                matched = None
                if p[0] == arr[i]:
                    matched = p
                    break
            if not matched:
                return False
            #now examine matched
            for num in p:
                if num != arr[i]:
                    return False
                #keep increameint our i
                else:
                    i += 1
        return True

class Solution(object):
    def canFormArray(self, arr, pieces):
        """
        :type arr: List[int]
        :type pieces: List[List[int]]
        :rtype: bool
        """
        '''
        and even faster way would be to sort the pieces on their starting element
        then we could do binary search
        which reduces to O(NlogN)
        algo:
            1. init an index i to record the current matching index in arr
            2. binary search to find the piece starting with arr[i], return false otherwise
            3. then match the matched p's sublist, return false otherwise
            4. incrment i
            5. repeat until i gets to the end
        '''
        #sort pierce
        N = len(arr)
        num_p = len(pieces)
        pieces.sort()
        
        i = 0
        while i < N:
            l,r = 0,num_p-1
            matched = None
            #binary serach
            while l <= r:
                mid = l + (r-l) // 2
                if pieces[mid][0] == arr[i]:
                    matched = pieces[mid]
                    break
                elif pieces[mid][0] > arr[i]:
                    r = mid - 1
                else:
                    l = mid + 1
            if matched == None:
                return False
            
            #now check mtached piece
            for num in matched:
                if num != arr[i]:
                    return False
                else:
                    i += 1
        return True
#O(N)
class Solution(object):
    def canFormArray(self, arr, pieces):
        """
        :type arr: List[int]
        :type pieces: List[List[int]]
        :rtype: bool
        """
        '''
        O(N) using hashing
        hash the pieces so for each p in pieces:
        hash is p[0] : p
        algo:
            init hash map to record peices firt elene and whole sublist
            init indiex to i
            find starting piece with arr[i] in mapping, return false if no match
            use amtche pice to match sublsit
        '''
        mapp = {p[0]: p for p in pieces}
        N = len(arr)
        i = 0
        while i < N:
            #check for the first occrurent
            if arr[i] not in mapp:
                return False
            matched = mapp[arr[i]]
            for num in matched:
                if num != arr[i]:
                    return False
                else:
                    i += 1
        return True

######################
#Palindrom Permutation
######################
class Solution(object):
    def canPermutePalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        well this is just counts of a palindrome
        where at most 1 digit can have an odd occurennce
        '''
        if not s:
            return True
        counts = Counter(s)
        odds = 0
        for v in counts.values():
            if v % 2 != 0:
                odds += 1
            if odds > 1:
                return False
        return True

class Solution(object):
    def canPermutePalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        #single pass hash
        #init count of evens to zero
        #once we update our hashp, we also check is the current count is even
        #its even, we we decreement our count, but increase otherwise
        mapp = {}
        N = len(s)
        odds = 0
        for i in range(N):
            if s[i] in mapp:
                mapp[s[i]] += 1
            else:
                mapp[s[i]] = 1
            
            if mapp[s[i]] % 2 == 0:
                odds -= 1
            else:
                odds += 1
        return odds <= 1

####################################################################
#Find a Corresponding Node of a Binary Tree in a Clone of That Tree
####################################################################
#reucrsive
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def getTargetCopy(self, original, cloned, target):
        """
        :type original: TreeNode
        :type cloned: TreeNode
        :type target: TreeNode
        :rtype: TreeNode
        """
        '''
        we need another pointer as mover through original
        move original until it machtes target but also move through cloned
        stack might be better
        '''
        
        def dfs(org,clnd,target):
            if not org:
                return
            if org.val == target.val:
                self.ans = clnd
            dfs(org.left,clnd.left,target)
            dfs(org.right,clnd.right,target)
        
        dfs(original,cloned,target)
        return self.ans

class Solution(object):
    def getTargetCopy(self, original, cloned, target):
        """
        :type original: TreeNode
        :type cloned: TreeNode
        :type target: TreeNode
        :rtype: TreeNode
        """
        '''
        we need another pointer as mover through original
        move original until it machtes target but also move through cloned
        stack might be better
        '''
        
        def dfs(org,clnd,target):
            if not org:
                return
            if org.val == target.val:
                return clnd
            left = dfs(org.left,clnd.left,target)
            right = dfs(org.right,clnd.right,target)
            return left or right

        
        return dfs(original,cloned,target)

#iterative with stack
class Solution(object):
    def getTargetCopy(self, original, cloned, target):
        """
        :type original: TreeNode
        :type cloned: TreeNode
        :type target: TreeNode
        :rtype: TreeNode
        """
        '''
        iterative solution with stack
        '''
        #we traverse both trees so we need to stacks
        stack_org, stack_clnd = [],[]
        #we need to give reference to our pointers
        node_org, node_clnd = original,cloned
        
        while stack_org or node_clnd:
            #alwasy go all the way left before visiting node
            while node_org:
                stack_org.append(node_org)
                stack_clnd.append(node_clnd)
                
                #dont forget to move
                node_org = node_org.left
                node_clnd = node_clnd.left
            #now we pop
            node_org = stack_org.pop()
            node_clnd = stack_clnd.pop()
            
            if node_org is target:
                return node_clnd
            
            #if we can't statisfy, we go right
            node_org = node_org.right
            node_clnd = node_clnd.right

########################
#Beautiful Arrangement
########################
#aye yai yai
#TLE THOUGH
class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        condtions for a beatiful arrangment
            1. nums[i] % i == 0 
            or
            2. i % nums[i] == 0
        well we could recursively generate all permutations and check if the arrangment is beautiful
        then increment a count and return the count
        
    
        '''
        nums = list(range(1,n+1))
        def build_perms(items):
            if len(items) == 1:
                return [items]
            permutations = []
            for i,a in enumerate(items):
                #get the remaining
                remaining = [num for j,num in enumerate(items) if i != j]
                #recurse
                perm = build_perms(remaining)
                for p in perm:
                    permutations.append([a]+p)
            return permutations
        perms = build_perms(nums)
        ways = 0
        for p in perms:
            valid = 0
            for i in range(len(p)):
                if p[i] % (i+1) == 0 or (i+1) % p[i] == 0:
                    valid += 1
                else:
                    break
            if valid == len(p):
                ways += 1
        return ways

#just an aside, creating perms but repeats
        def swap(nums,a,b):
            temp = nums[a]
            nums[a] = nums[b]
            nums[b] = temp
            
        perms = []
        def permute(nums, idx):
            if idx == len(nums) - 1:
                return
            for i in range(len(nums)):
                swap(nums,i,idx)
                permute(nums,idx+1)
                permuted = []
                for n in nums:
                    permuted.append(n)
                perms.append(permuted)
                swap(nums,i,idx)
        permute(list(range(1,n+1)),0)
        #print set((tuple(foo) for foo in perms))
        print perms

class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        let's walk through all three solutions to understand how to permute a string
        we can define a function permute(nums, curr_idx)
        the function taktes the index of the current element then it swapss with every other element in the array lying to the right
        once swapping has been done it makes another cal to permute, but we advance the index
        '''
        self.count = 0
        nums = list(range(1,n+1))
        def swap(nums,a,b):
            temp = nums[a]
            nums[a] = nums[b]
            nums[b] = temp
            
        perms = []
        def permute(nums, idx):
            if idx == len(nums) - 1:
                #check the perm for arrangement conditions
                for i in range(1,len(nums)+1):
                    if nums[i-1] % i != 0 and nums[i-1] != 0:
                        break
                if i == len(nums) + 1:
                    self.count += 1
                else:
                    return
            for j in range(len(nums)):
                swap(nums,j,idx)
                permute(nums,idx+1)
                swap(nums,j,idx)
        permute(nums,0)
        return self.count

class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        https://leetcode.com/problems/beautiful-arrangement/discuss/1000324/Python-Permutation-Solution
        the conditions are that either the number at index+1 is divisble by index +1
        or index+1 is disvisble by that number
        algo:
            1. generate the array of numbers that will be used to create perms (1 to N inclusive)
            2. iterate through all element in the list and compare to to i, which i 1 to avoid the index + 1 thing
            3. of the nmber is divisible by i or i divisible by the number, we can continute with the permuation, otherwise abandon
            4. if our i has move al the way through our nums list (i.e i == len(nums)+1) we made a valid result
            
        '''
        self.res = 0
        nums = [i for i in range(1, n+1)]
        
        def perm_check(nums,i):
            #start i off at 1
            if i == n + 1:
                self.res += 1
                return
            
            for j,num in enumerate(nums):
                if num % i == 0 or i % num == 0:
                    perm_check(nums[:j]+nums[j+1:],i+1)
        
        perm_check(nums,1)
        return self.res

#this one deson't seem to work,but its the right approach
class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        let's walk through all three solutions to understand how to permute a string
        we can define a function permute(nums, curr_idx)
        the function taktes the index of the current element then it swapss with every other element in the array lying to the right
        once swapping has been done it makes another cal to permute, but we advance the index
        '''
        self.count = 0
        nums = list(range(1,n+1))
        def swap(nums,a,b):
            temp = nums[a]
            nums[a] = nums[b]
            nums[b] = temp
            
        perms = []
        def permute(nums, idx):
            if idx == len(nums)-1:
                #check the perm for arrangement conditions
                for i in range(1,len(nums)+1):
                    if nums[i-1] % i != 0 or nums[i-1] % i != 0:
                        break
                if i == len(nums)-1:
                    self.count += 1
            return
            for j in range(len(nums)):
                swap(nums,j,idx)
                permute(nums,idx+1)
                swap(nums,j,idx)
        permute(nums,0)
        return self.count

#with pruning using boolean array

class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        back tracking variant, pruning search sapce
        we try to create all the permutations of numbers from 1 to N
        we can fix one number at a particular position and check for the divisibility criteria
        but we need to keep track of the numbers which have laready eeb consider easlier
        we can make use of visitied boolean array of size N
        here visited[i] refers to the ith number being already placed/not placed
        
        '''
        self.count = 0
        visited = [False]*(n+1)
        
        def calculate(n, pos):
            if pos > n: #meaning we have used all numbers and since we are pruning, it must be a path
                self.count += 1
            
            for i in range(1,n+1):
                if (visited[i] == False) and (pos % i == 0 or i % pos == 0 ):
                    visited[i] = True
                    calculate(n,pos+1)
                    #clear it again because we need to backtrack
                    visited[i] = False
        calculate(n,1)
        return self.count
#adding a memo
class Solution(object):
    def countArrangement(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        back tracking variant, pruning search sapce
        we try to create all the permutations of numbers from 1 to N
        we can fix one number at a particular position and check for the divisibility criteria
        but we need to keep track of the numbers which have laready eeb consider easlier
        we can make use of visitied boolean array of size N
        here visited[i] refers to the ith number being already placed/not placed
        #adding memoization
        '''
        visited = [False]*(n+1) # i need to pass this in now
        
        memo = {}
        
        def calculate(n, pos):
            if pos > n: #meaning we have used all numbers and since we are pruning, it must be a path
                return 1
            #retrieve
            if tuple(visited) in memo:
                return memo[tuple(visited)]
            valid = 0
            for i in range(1,n+1):
                if (visited[i] == False) and (pos % i == 0 or i % pos == 0 ):
                    visited[i] = True
                    valid += calculate(n,pos+1)
                    #clear it again
                    visited[i] = False
            #put back in memory
            memo[tuple(visited)] = valid
            return valid
        
        return calculate(n,1)
            
#######################
#Merge Two Sorted Lists
#######################
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        create dummy node and have to pointers move to each one
        keep adding to dummy.next the minimum of the two pointers
        '''
        dummy = ListNode() #return dummy.next
        d_ptr = dummy
        first = l1
        second = l2
        
        
        while first and second:
            #take first if small
            if first.val <= second.val:
                d_ptr.next = first
                first = first.next
            else:
                d_ptr.next = second
                second = second.next
            d_ptr = d_ptr.next
                
        while second:
            d_ptr.next = second
            second = second.next
        
        return dummy.next

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode()
        curr = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
            
        if l1:
            curr.next = l1
        if l2:
            curr.next = l2
        return dummy.next


#####Recursive approach
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        we can define a recursive relation ship for the two lists
        when list1[0] < list2[0] we can recurse and add the list1[0]:
            list1[0] + merge(list[1:],list2)
        otherwise take list2's current element
            list2[0] + merge(list[1],list2[1:])
        now we have to model the recurrsnece using a function
        if either l1 or l2 is null, there is no need for a emrge, so we simply return the non-null list
        otherwise, we determine which is l1 of l2 has the smaller head and recurse on the enxt value
        
        '''
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2
            
##########################################
# Remove Duplicates from Sorted List II
##########################################
#naive way
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        while we are at it, lets do this naively
        get all the items and create new list such that there are no duplicates ones then rebuild
        '''
        items = []
        cur = head
        while cur:
            items.append(cur.val)
            cur = cur.next
        
        counts = Counter(items)
        valid = [k for k,v in counts.items() if v == 1]
        
        dummy = ListNode()
        cur = dummy
        for i in valid:
            cur.next = ListNode(i)
            cur = cur.next
        return dummy.next


#damn, i couldn not figure out the logic on this onee....
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        naive way would be to just traverse the whole thing, clear duplicates and make a new node
        two pointers, one stays at current and we keep advancing our next so long it does not match the current
        once it doen't match, connect
        while loop invariant?
            so long as there is a next or next is not null
        '''
        dummy = ListNode()
        #dummy.next = head
        
        prev = dummy
        #we need to keep check for the existence of a node and its next
        while head:
            if head.next and head.val == head.next.val: #we need to make sure there is next node to check
                #we need to keep moving up 
                #we need to make sure there is a head because of the danling null pointer
                #if head.next exists, then head surely exsists
                while head.next and head.val == head.next.val:
                    #move up
                    head = head.next
                #we connect
                prev.next = head.next
            else:
                #if we didnt have to delte move the prev
                prev.next = head
                prev = prev.next
            #always move the head
            head = head.next
        
        return dummy.next

################
# Kth Missing Positive Number
########################
#well it works
class Solution(object):
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        '''
        this might be bad, but i could start off with all possible numbers in range 1 to 1000
        and remove each one as i pass the array
        then convert to a list and reeturn k
        '''
        possible = set(tuple(range(1,20010)))
        for num in arr:
            possible.remove(num)
        possible = list(possible)
        if k < len(possible):
            return possible[k-1]

class Solution(object):
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        '''
        we can solve the problem in O(N) time and O(1) space
        the number of missing eleemnts in the array, since its in range(1,wtv)
        would be arr[i+1] - arr[i] - 1
        algo:
            check if the kth missing number is less than the first element of the array, if it is return k
            decrease k by the number of positive intergers which are missing before the array starts: k -= arr[0]-1
            traverse the array
                at each step, compute the number of misssing positive integers in between i+1'th and  ith elements
                compare k to the curMissing
                if k <= cur missing then the nmber to reutrn is in between i+1 and i'th and return arr[i]+k
                otherwise decrese k by currMissing
        
            if we've passed the array, it means the missing is afte the last element
            so return arr[-1] + k
            
        '''
        #edge case, when k is smaller than the first element, well just return k
        if k <= arr[0]-1:
            return k
        #now it must lie in the array, so we can decrement k by the first eleemnt
        k -= arr[0] - 1
        for i in range(len(arr)-1):
            #find num missing
            num_missing = arr[i+1] - arr[i] -1
            if k <= num_missing:
                return arr[i] + k
            else:
                k -= num_missing
        #now the kth missing lies beyond the array
        return arr[-1] + k
        
#another way
#https://leetcode.com/problems/kth-missing-positive-number/discuss/1004535/Python-Two-solutions-O(n)-and-O(log-n)-explained
class Solution(object):
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        '''
        from the hint, keep track of the number of positves if missed
        keep a set of all numbers in the array
        and we check all numbers in range(1,k+len(arr)+1)
        everytime a number is not in our set, we decrement k 1
        only then hwen k is zero we return the number
        '''
        nums = set(arr)
        for missing in range(1,k+len(arr)+1):
            if missing not in nums:
                k -= 1
            if k == 0:
                return missing

#binary search
class Solution(object):
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        '''
        https://www.youtube.com/watch?v=Nfu-ubvJaZ0&ab_channel=AnishMalla
        we can use binary search because the array is strictly increasing
        ex 
        [2,3,4,7,11]
        if we have a regaulr array with no missing numbers
        [1,2,3,4,5]
        to do binary search we can ask, at a certain number, how many are missing to the left of it 
        or rather at an index how many are missing to the left of it
        num[idx] - 1 is num missing
        if there were no missing numbers, the number at the indx would be idx + 1
        and the number it is supposed to be is num[idx] - 1
        [2,3,4,7,11]
        [1,2,3,4,5]
        [1,1,1,3,6] #number missing to the left
        binary search to look for smallest number missing greater than k
        just return l pointer + k
        arr[left] - legntharray  - 1
        '''
        #edge case
        if k < arr[0]:
            return k
        l,r = 0,len(arr) - 1
        while l <= r:
            mid = l + (r-l) //2
            #if the number of positive intergers whic are missing before arr[mid] < k
            # we move up
            if arr[mid] - (mid + 1) < k:
                l = mid + 1
            else:
                r = mid - 1 #notice how go one less than the mid, because it can't be at the mid
            #at the end of the loop l = r +1 and the kth missing is ine between arr[r] and ar[l]
            #the number of integerts missing before arr[r] is arr[r] - r - 1
            #and so the number to return is 
            #arr[r] + k - (arr[r] - r - 1) = l + k
        return arr[r] + k - (arr[r] - r - 1) #or just l - k


class Solution(object):
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        skipped = 0 #the skipped idx
        i,j = 0,1 #all skipped numbers
        N = len(arr)
        
        while i < N:
            if arr[i] != j:
                skipped += 1
                if skipped == k:
                    return j
                j += 1
            else:
                i += 1
                j += 1
            
        #if i've gone beyong at the difference of k and skipped to the end of the array
        return arr[-1] + (k-skipped)

#################################################
#Longest Substring Without Repeating Characters
################################################
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        they really should just call it a contiguous array
        use two pointers
        and hashmap marking the last recently seen char
        keep updating the length so long as the set i
        '''
        #edge case
        if not s:
            return 0
        mapp = set()
        N = len(s)
        l,r = 0,0
        max_length = 0
        while l < N and r < N:
            #first move right to get a bigger length
            if s[r] not in mapp:
                mapp.add(s[l])
                r += 1
                #update to get max_length
                max_length = max(max_length,r-l)
            else:
                #keep advanving r but remove
                mapp.remove(s[l])
                r += 1
        
        return max_length

#worst case it is O(2N), because we may have to move l and r once every time for a lenth N


#brute force
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        lets just go over brute force for some extra practice
        we can examine all substrings and check tha they do not contain duplicates
        '''
        max_len = 0
        N = len(s)
        for i in range(N):
            for j in range(i+1,N+1):
                if len(s[i:j]) == len(set(s[i:j])):
                    #update
                    max_len = max(max_len,len(s[i:j]))
        return max_len
                    

#O(N)
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        O(2N) solution keeps moving left and right pointers
        we hash char to and index and just move it pas the last recently seen
        set our left pointer to 0 and pass through s
        whenver we've seen s[r] we have to move l to the most reent 
        then update our max and then update our s[r] index
        '''
        N = len(s)
        max_length = 0
        
        mapp = {}
        
        left,right = 0,0
        while right < N:
            if s[right] not in mapp:
                mapp[s[right]] = right
            else:
                #move left to the last recently seen s[right]
                left = max(mapp[s[right]],left)

            
            max_length = max(max_length, right - left+1)
            #update mapp
            mapp[s[right]] = right +1
            right += 1
        
        return max_length

######################
#Check if Two Strin Arrays are equivalent
######################
class Solution(object):
    def arrayStringsAreEqual(self, word1, word2):
        """
        :type word1: List[str]
        :type word2: List[str]
        :rtype: bool
        """
        '''
        uhhhhh?? just join all strings and see if they are euqal
        '''
        word1 = "".join(word1)
        word2 = "".join(word2)
        return word1 == word2
        

class Solution(object):
    def arrayStringsAreEqual(self, word1, word2):
        """
        :type word1: List[str]
        :type word2: List[str]
        :rtype: bool
        """
        '''
        O(1) space
        we can just maniputlate pointers and adnvae through both word1s at the same time
        '''
        i,j = 0,0
        ith_word, jth_word = 0,0
        while ith_word < len(word1) and jth_word < len(word2):
            char1 = word1[ith_word][i]
            char2 = word2[jth_word][j]
            
            if char1 != char2:
                return False
            #advane pointers
            i += 1
            j += 1
            
            #we need to check if we have gone
            if i >= len(word1[ith_word]):
                #reset
                ith_word += 1
                i = 0
            
            if j >= len(word2[jth_word]):
                #reset
                jth_word += 1
                j = 0
        
        #final check, we check that we have pushed i and j pointers to the end
        return (ith_word == len(word1)) and (jth_word == len(word2))

###########################
#Find Root of N-Ary Tree
###########################
#it was a good attempt lol
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution(object):
    def findRoot(self, tree):
        """
        :type tree: List['Node']
        :rtype: 'Node'
        """
        '''
        we do not need to build the tree
        we just need to find the node that is the root!
        could i not just find the node with the smallest node.val and just reutn that?
        '''

        node_vals = []
        for node in tree:
            node_vals.append(node.val)
        #find index of mid
        mini = float('inf')
        mini_idx = 0
        for i,val in enumerate(node_vals):
            if val < mini:
                mini = val
                mini_idx = i
        return tree[mini_idx]
            

##########################
#Find Root of N-ary Tree
##########################
#bad way but works lol
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution(object):
    def findRoot(self, tree):
        """
        :type tree: List['Node']
        :rtype: 'Node'
        """
        '''
        we do not need to build the tree
        we just need to find the node that is the root!
        could i not just find the node with the smallest node.val and just reutn that?
        the value to return must be in the array tree
        from the hint, the root as an indegree 0, meaning there are no nodes going to it,
        but only from it
        BFS!
        for each node in three, count up the number of edges TO IT!
        the one with zero is the root
        
        '''
        '''
        bad way
        '''
        all_children = set()
        for node in tree:
            if node.children:
                for child in node.children:
                    all_children.add(child.val)
        
        for node in tree:
            if node.val not in all_children:
                return node
        
class Solution(object):
    def findRoot(self, tree):
        """
        :type tree: List['Node']
        :rtype: 'Node'
        """
        '''
        we can solve this problem in constance space and linear time
        recall, that if we visit all the nodes and all the child nodes, then the root node would be the only node that we visit once and only once, the rest would be visited twice
        we can say this
        given a list of numbers where some of the numbers appear twice, find the number that appears only once
        the idea is to use a variable (lets call it value_sum) to keep track of the sum of node values
        example. look at the array [4,3,5,3,4]
        we cad pass the one time by adding nums we havent seen, and if we have seen that num, deduct it
        more specifically, we add the value of each node to valu_sum and we decut the value of each child node from value_sum
        at the end, the valu_sum wold be the value of the root node
        the rational is that the valaues of non-root nodes are cancelled out during the above additoin and dection operations (the value of a non root node is added onces as aprent not but never deducted)
        in order for this to work, all values must be unieu
        '''
        val_sum = 0
        for node in tree:
            val_sum += node.val
            for child in node.children:
                val_sum -= child.val
        
        for node in tree:
            if node.val == val_sum:
                return node


#####################
#Word Ladder
#####################
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        '''
        the idea would be to treat this like a graph problem, each node is only possible if it exsits in wordlist
        an edge only exsists if we can change a letter to the word and the word being the the dictinoary
        it will be an undirected and unwiehgted graph (each edge is 1)
        the problem then becomes a shortes path problem
        we need to preprocess each word
        EX : hot can be *ot, h*t, ho*
        allowing us to find the next node one letter away
        if we didn't, we would have to iterate over the entire words list and find words that differ by oneletter
        while doing BFS, we have to find the adjacent nodes for
        mapp transformations of each each word to a dict
        algo:
            1. pre process each word in wordlist showing each of the nodes possible from the word
            2. save these intermediate stes in a dict with key as the intermediate word and value as the list of wrds which have the same intermediate word
            3. push tuple containing the beginWord and 1 in a q. 1 represesent the number of edges
            4. use dictionary as visited set
            5. while q, get elemtns from q
            6. Find all generic transformations of the current_word and find out if any of these transformations is also a transformation of other words
            7. List of words from all aomcdict are all the words which have a common intermaeidate. Thesenew sets of words will be the adjacent nodes from current word
            7. for each word in the list, add the word and level + 1
            
        '''
        #edge case, check that being adn wen din word list
        if endWord not in set(wordList) or not endWord or not beginWord:
            return 0
        
        #all words are of the same length
        L = len(beginWord)
        
        #init dict to hold combinations of words that can be formed, from any given word by chaing one letter at time
        all_combinations = defaultdict(list)
        #get all possible nodes for each word in wordList
        for word in wordList:
            for i in range(L):
                #key is the generic word after swapping out the ith char
                #value is the list of words
                all_combinations[word[:i]+"*"+word[i+1:]].append(word)
        #print all_combinations
        
        #BFS
        q = deque([(beginWord,1)])
        #visited set
        visited_words = set()
        while q:
            current_word, length = q.popleft()
            for i in range(L):
                #generate generic word
                generic_word = current_word[:i]+"*"+current_word[i+1:]
                #find words with with next generic stat
                for word in all_combinations[generic_word]:
                    #condiiton met
                    if word == endWord:
                        return length + 1
                    if word not in visited_words:
                        visited_words.add(word)
                        q.append((word, length+1))
                
                #after looking at all the next words, insert the new genric word
                all_combinations[generic_word] = []
                
        return 0

#T:E
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        '''
        brute force
        we can examine each single char diff for each word and create an adjacent list from that
        before doing that we need to see if we need to see if we can get to that word from the current word
        then just do BFS until we reach the end word
        '''
        def one_away(w1,w2):
            if w1 == w2:
                return False
            diff = 0
            for i in range(len(w1)): # they are both the same
                if w1[i] != w2[i]:
                    diff += 1
            return diff == 1
        
        #build adjaceny list
        adj = defaultdict(list)
        for w1 in [beginWord]+wordList:
            for w2 in [beginWord]+wordList:
                if one_away(w1,w2):
                    adj[w1].append(w2)
        
        #now we can just bfs the normal way
        q = deque([(beginWord,0)])
        #visited set
        visited = set()
        while q:
            word, level = q.popleft()
            if word == endWord:
                return level + 1
            if word not in visited:
                visited.add(word)
                for nextt in adj[word]:
                    if nextt not in visited:
                        q.append((nextt,level+1))
                        
        return 0

class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        '''
        nuilding the adj list is M*M*N, this is bad
        insted of compared each char for each word against each word
        change one letter at a time and check if its a possible word
        '''
        possible_words = set([beginWord]+wordList)
        adj = defaultdict(list)
        for word in [beginWord]+wordList:
            for i in range(len(word)):
                for ch in range(ord('a'),ord('z')+1):
                    transformed = word[:i]+chr(ch)+word[i+1:]
                    if transformed in possible_words and transformed != word:
                        adj[word].append(transformed)
        #now we can just bfs the normal way
        q = deque([(beginWord,0)])
        #visited set
        visited = set()
        while q:
            word, level = q.popleft()
            if word == endWord:
                return level + 1
            if word not in visited:
                visited.add(word)
                for nextt in adj[word]:
                    if nextt not in visited:
                        q.append((nextt,level+1))
                        
        return 0

#bidrectional BFS
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        '''
        instead of doing BFS from only the start, we BFS from the end at the same time
        wherever we hit, the lenght if just sum of both final levels from the beginning and end
        #it all depends on the branching factor, but for the most part it would half the time
        algo:
            1. BFS from two sides
            2. we now need to visited dicts to keep track of nodes visited from the start and the beginning
            3. if we ever find a node/word which is in the visited dictionary of the parallel search, we terminate our serch since we have found the meeting point of this bidiretional serach
            4. the termination condition is finding a word which has already been seen by the parallel search
            5. and the size of the transformation sequence would just be the summ of the lvels
        '''
        from collections import defaultdict
class Solution(object):
    def __init__(self):
        self.length = 0
        # Dictionary to hold combination of words that can be formed,
        # from any given word. By changing one letter at a time.
        self.all_combo_dict = defaultdict(list)

    def visitWordNode(self, queue, visited, others_visited):
        current_word, level = queue.popleft()
        for i in range(self.length):
            # Intermediate words for current word
            intermediate_word = current_word[:i] + "*" + current_word[i+1:]

            # Next states are all the words which share the same intermediate state.
            for word in self.all_combo_dict[intermediate_word]:
                # If the intermediate state/word has already been visited from the
                # other parallel traversal this means we have found the answer.
                if word in others_visited:
                    return level + others_visited[word]
                if word not in visited:
                    # Save the level as the value of the dictionary, to save number of hops.
                    visited[word] = level + 1
                    queue.append((word, level + 1))
        return None

    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """

        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0

        # Since all words are of same length.
        self.length = len(beginWord)

        for word in wordList:
            for i in range(self.length):
                # Key is the generic word
                # Value is a list of words which have the same intermediate generic word.
                self.all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)


        # Queues for birdirectional BFS
        queue_begin = collections.deque([(beginWord, 1)]) # BFS starting from beginWord
        queue_end = collections.deque([(endWord, 1)]) # BFS starting from endWord

        # Visited to make sure we don't repeat processing same word
        visited_begin = {beginWord: 1}
        visited_end = {endWord: 1}
        ans = None

        # We do a birdirectional search starting one pointer from begin
        # word and one pointer from end word. Hopping one by one.
        while queue_begin and queue_end:

            # One hop from begin word
            ans = self.visitWordNode(queue_begin, visited_begin, visited_end)
            if ans:
                return ans
            # One hop from end word
            ans = self.visitWordNode(queue_end, visited_end, visited_begin)
            if ans:
                return ans

        return 0

##############################
#Create Sorted Array through Instructions
##############################
#TLE 50/65
class Solution(object):
    def createSortedArray(self, instructions):
        """
        :type instructions: List[int]
        :rtype: int
        """
        '''
        brute force
        keep insterting into nums and update the min every time
        conditions
        The number of elements currently in nums that are strictly less than instructions[i].
        The number of elements currently in nums that are strictly greater than instructions[i].
        '''
        nums = [] #this really does not need to be sorted right now
        cost = 0
        for i in range(len(instructions)):
            candidate = instructions[i]
            less_than = 0
            greater_than = 0
            for num in nums:
                if num > candidate:
                    greater_than += 1
                if num < candidate:
                    less_than += 1
                else:
                    continue
            #push candidate
            nums.append(candidate)
            #update cost
            cost += min(less_than,greater_than)
        
        return cost

#https://leetcode.com/articles/a-recursive-approach-to-segment-trees-range-sum-queries-lazy-propagation/
tree = []
array = []
def build_segment_tree(array, tree_idx, lo,hi):
    #takes in array and builds out tree
    if lo == hi:
        tree[tree_idx] = array[lo] #push left pointer into a leaf
        #this is the base condiiton 
        return
        mid = lo + (hi - lo) // 2
        #build left side
        build_segment_tree(array,2*tree_idx + 1, lo, mid)
        #build right
        build_segment_tree(array,2*tree_idx+2,mid+1,hi)


class Solution(object):
    def createSortedArray(self, instructions):
        """
        :type instructions: List[int]
        :rtype: int
        """
        '''
        basic way is just use empty array in nums
        binary search in binary nums and check where it is in nums
        check left and right side to get the cost, becaue there could be duplicates
        we can use a sortedlist, which is just a container that allows for seraching and insterting in LogN time
        
        '''
        from sortedcontainers import SortedList
        #insert/search log n, find where to insert
        
        #bisectleft and right, what is index number to insert into from the left and right
        #find the min update cost
        
        #add function
        
        s = SortedList()
        
        cost = 0
        for i in instructions:
            cost_left = s.bisect_left(i)
            cost_right = len(s) - s.bisect_right(i)
            cost += min(cost_left,cost_right)
            #update conatiner with i
            s.add(i)
            
        return cost % (10**9 + 7)

####################
#Merge Sorte Array
####################
#close one again, god dammit
#well this works, but only for legnths >= 3
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """

        
        ptr1 = 0
        ptr2 = 0
        
        while ptr1 < m:
            if nums2[ptr2] <= nums1[ptr1]:
                #insert and shift
                remaining = nums1[ptr1+1:]
                #insert
                nums1[ptr1+1] = nums2[ptr2]
                #insert remaining
                nums1[ptr1+2:] = remaining[:-1]
                #move on to the next element
                ptr2 += 1
            else:
                ptr1 += 1
        
        #remaining elements
        nums1[ptr1+1:] = nums2[ptr2:]


#better solution
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        #O(m+n) but using up M space
        #make copy of nums1
        nums1_copy = nums1[:m]
        #clear nums1
        nums1[:] = []
        
        #two pointers
        p1,p2 = 0,0
        while p1 < m and p2 < n:
            if nums1_copy[p1] < nums2[p2]:
                nums1.append(nums1_copy[p1])
                p1 += 1
            else:
                nums1.append(nums2[p2])
                p2 += 1
        
        #if there are remaining elments
        #rmember either m or n should have gone through all the way
        if p1 < m:
            #extend nums1
            nums1[p1+p2:] = nums1_copy[p1:]
        if p2 < n:
            nums1[p1+p2:] = nums2[p2:]

#two pointer O(m+n) and O(1) space
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        '''
        since the arrays are sorted in increasing order and we know that the remaining elements are to be added to the right, we start right!
        set p1 and p2 pointing to final non zero elements in arrays
        then we just add into the arra the larger of the two pointers and go backward
        
        '''
        #two pointers into arrays
        p1 = m - 1
        p2 = n - 1
        #set pointer for nums1
        p = m + n - 1
        
        #while there are elements to compare
        while p1 >= 0 and p2 >=0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                #move p2 down
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -= 1
            
        #add remaining elements from num2
        nums1[:p2+1] =nums2[:p2+1]

#https://www.youtube.com/watch?v=Mm9C9M8-BBA&t=1s&ab_channel=AnishMalla
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        '''
        we can start from the right and go backwards
        we take the greaer the of the two nmbers and add to the right most poistions in nums1
        then we just move our pointers
        we can access the right most just by calling m+n-1
        whenver we make a decision we decrement form the pointer in our array
        #special case
        if m is zero, we have to take from nums2
        
        '''
        while m > 0 and n > 0:
            if nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        
        #edge case, when we have nothing in m, and we need take all elements from nums2
        while n > 0:
            nums1[m+n-1] = nums2[n-1]
            n -= 1

#################
#Add Two Numbers
#################
#not the prettiest but it works
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        well the naive way would be to traverse both linked lists concating along the way
        recast as int then add then return
        lets do this for now, warm up
        
        '''
        num1 = ""
        while l1:
            num1 += str(l1.val)
            l1 = l1.next
            
        num2 = ""
        while l2:
            num2 += str(l2.val)
            l2 = l2.next
        #reverse
        result = str(int(num1[::-1]) + int(num2[::-1]))[::-1]
        #build
        temp = ListNode()
        dummy = temp
        for i,num in enumerate(result):
            dummy.val = int(num)
            if i == len(result) - 1:
                dummy.next = None
                dummy = dummy.next
            else:
                dummy.next = ListNode()
                dummy = dummy.next
                
        return temp

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        '''
        this is just textbook addition but i have the numbers reversed
        i can implement carry and dump them into a new list node object
        special cases
        when 1 is longe than the other, but this invaraint is held with this implementation
        when 1 list is null, again still held with or
        
        '''
        dummy = ListNode()
        temp = dummy #move through dummy, return dummy
        carry = 0
        while l1 or l2:
            #get nums
            num1 = l1.val if l1 else 0
            num2 = l2.val if l2 else 0
            #get the summ including carry
            summ = num1 + num2 + carry
            #mod 10 it
            digit = summ % 10
            #increment carry
            carry = summ // 10
            temp.next = ListNode(digit)
            temp = temp.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            
        #the finel carry
        if carry > 0:
            temp.next = ListNode(carry)
        
        return dummy.next
            


#####################
#Boats to Save People
#####################
#ordering does not work
class Solution(object):
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        '''
        i can do this greedily
        if i sort, then just take people up the limit
        keep doing this while there are people
        q pop
        keep checking i and i+1 person because i can only take two at a time
        '''
        people.sort()
        q = deque(people)
        
        boats = 0
        
        while len(q) >=2:
            if q[0]+q[1] <= limit:
                q.popleft()
                q.popleft()
                boats += 1
            else:
                q.popleft()
                boats += 1
        #now we are in the case where there can be 1 or two peole left
        if len(q) == 1:
            boats += 1
        if len(q) == 2:
            if sum(q) <= limit:
                boats += 1
            else:
                boats += 2
        
        print q,boats

class Solution(object):
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        '''
        what is i sort and use two pointers, trying to make a pair such that the sum <= limit
        keep the invaraint by going to the middle
        '''
        people.sort()
        N = len(people)
        
        boats = 0
        left,right = 0, N-1
        
        while left <= right:
            if people[left] + people[right] <= limit:
                boats += 1
                left += 1
                right -= 1
            else:
                boats += 1
                right -= 1
        return boats


#######################################
#Minimum Operations to Reduce X to Zero
#######################################
#fuck this problem.....
class Solution(object):
    def minOperations(self, nums, x):
        """
        :type nums: List[int]
        :type x: int
        :rtype: int
        """
        '''
        the numbers will always be positive
        twist on max subarray problem
        if i can reduce x to zero exactly using, then there existed an array in nums whos's sum was x
        if i find this array, return its length
        othersie it cannot be done and return -1
        what is if find the cumsums left to right
        and cumsums right to left
        
        '''
        #edge case
        if min(nums) > x:
            return -1
        N = len(nums)
        
        left_to_right = [0]*N
        right_to_left = [0]*N
        #seed
        left_to_right[0] = nums[0]
        right_to_left[0] = nums[-1]
        
        #left to right first
        for i in range(1,N):
            left_to_right[i] = left_to_right[i-1] + nums[i]
        
        #now right to left
        for i in range(1,N):
            right_to_left[i] = right_to_left[i-1]+ nums[::-1][i]
        
        #pass both arrays to see if i can get the sum
        lengths = []
        for i in range(N):
            if left_to_right[i] == x:
                lengths.append(i+1)
        
        for i in range(N):
            if right_to_left[i] == x:
                lengths.append(i+1)
                
        #now we need to examine the case where we take from both sides
        #keep going until cumsum exceeds x both both arrays
        p1, p2 = None,None
        for i in range(N):
            if left_to_right[i] <= x:
                p1 = i
            else:
                break
        
        for i in range(N):
            if right_to_left[i] <= x:
                p2 = i
            else:
                break
        
        if p1 and p2 and left_to_right[p1] + right_to_left[p2] == x:
            lengths.append(p1+p2+2)
        
        return min(lengths) if len(lengths) > 0 else -1


#indirectly calculating
class Solution(object):
    def minOperations(self, nums, x):
        """
        :type nums: List[int]
        :type x: int
        :rtype: int
        """
        '''
        this is another problem that we should have in our back pocket
        similar to maximum size subarray sum equals k (revisit another time)
        only works for array with nonngeative numbers
        indirect apporach determines which values reamin in the array as opposed which to remove from the arrays
        total should be sum of elements in nums
        example,
        given the array
        [4,3,2,3,5,1,7] and x =14
        we would take [4,3] and [7]
        but notice that the reaminaig elements [2,3,5,1], whose sum is 11
        is just total - x
        so we want to find the sub array who's sum is is total - x, if it exsists
        we then get the length of that subarray subrtaced from the length of the array and return it
        now how do we solve this problem?
        we can use two pointers left and right
        we move out right from the start of the end of nums and it each right we move left as far as possible until the subarray sum is equal to or less than the required total - x
        #NOTE: with this moving method, we can find the subarray [left..right] whose sum is closest to total - x but not greater than
        algo:
            1. cacualte the total of nums
            2. init wo pointers l,r and keep moving r, inint an int max to record max length that sms up to total - x
            3. iterater right 
                update current
                if current >= total - x move left
                if current === total - x upaate max length
        '''
        total = sum(nums)
        N = len(nums)
        max_length = -1
        l = 0 #start pointer to mark left boundary of subarray who's sume clostes to total - x
        current_sum = 0 #curren sum of candidate subarray
        
        #potential right ends
        for r in range(N):
            current_sum += nums[r]
            #change our candidate array
            while current_sum > total-x and l <= r:
                current_sum -= nums[l]
                l += 1
            #update lenght when we have found our total - z
            if current_sum == total - x:
                max_length = max(max_length, r-l + 1)
                
        #get the the number of reamining elmement after exlcuding length subarray
        return N - max_length if max_length != -1 else -1


class Solution(object):
    def minOperations(self, nums, x):
        """
        :type nums: List[int]
        :type x: int
        :rtype: int
        """
        '''
        we can also do this directly,
        this time, instead of marking the points of the subarray who's nume is total - x
        who find left and right sums of the complement subarray
        we need to iterate our right from start to end of nums and move left
        after we have found the the l and r pointers, what we need to do is record the min legnth and return it!
        algo:
            same as the indirect method,
            init two pointers l and r
            init count  to currnet to represent sum from nums[0] to nums[l-1]
            and nums[r+1] to end
            init minimum to record the min lenght
            iterate all possible rights
            update current:
                if current < x move left 1
                if currnet == x update length!
        '''
        total = sum(nums)
        N = len(nums)
        remaining = float('inf')
        l = 0
        
        for r in range(N):
            total -= nums[r]
            while total < x and l <= r: #no longer finding total - x
                #incremtn current
                total += nums[l]
                l += 1
            #now check the left and right sums
            if total == x:
                remaining = min(remaining, l + (N-r-1))
        return remaining if remaining != float('inf') else -1

class Solution(object):
    def minOperations(self, nums, x):
        """
        :type nums: List[int]
        :type x: int
        :rtype: int
        """
        '''
        another approach, we still want to find the contiguous subarray who's sum is sum(nums) - x
        but this time we use the cum sum array
        image we have the array
        [1,1,4,2,3] and x = 5
        [1,2,6,8,11]
        sum of the array is 11, and we need to find contig array who's sum is 11-5 = 6
        or we can say, we need to find two cum sums, one of them goal plus naother one
        we can keep a hash of indices into the pre fix array, so when we iterate num in dic, if num+goal in dic, then we can the lengt of hte window, which would just be dic[num+goal] - dict[num]
        using the prefix array 
        1 + 6 = 7 not in map
        2+ 6 = 8, in mapp
        mapp[8] - mapp[num] = 2
        len(nums) - 2 is the answer
        '''
        #generate cumsums
        N = len(nums)
        cumsums = [0]*N
        cumsums[0] = nums[0]
        for i in range(1,N):
            cumsums[i] = cumsums[i-1] + nums[i]
        
        cumsums = [0]+cumsums
        complement = sum(nums) - x
        max_length = -1 #length of subarray
        
        #hash prefix array {val:idx}
        mapp = {val:i for i,val in enumerate(cumsums)}
        for num in mapp:
            if num + complement in mapp:
                max_length = max(max_length, mapp[num+complement]-mapp[num])
        
        return N - max_length if max_length != -1 else -1


################################
#Get Maximum in Generated Array
################################
class Solution(object):
    def getMaximumGenerated(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        array = [0]*(n+1)
        array[1] = 1
        for i in range(1,len(array)//2):
            array[2*i] = array[i]
            array[(2*i)+1] = array[i] + array[i+1]
        
        return max(array)

#removing last O(N) loop to get answer
class Solution(object):
    def getMaximumGenerated(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        array = [0]*(n+1)
        array[1] = 1
        max_val = 1
        for i in range(1,len(array)//2):
            array[2*i] = array[i]
            array[(2*i)+1] = array[i] + array[i+1]
            max_val = max(max_val, array[(2*i)+1])
        
        return max_val

class Solution(object):
    def getMaximumGenerated(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        another way using dividing nums[i*2] == nums[i]
        so at i, it is just nums[i/2]
        and nums[(i*2)+1] = nums[i] + nums[i+1]
        which is if its odd its just nums[i] + nums[i+1]
        build the array outside the base case 
        rules
        nums[2*i] = nums[i]
        nums[(2*i)+1] = nums[i] + nums[i+1]
        '''
        if n == 0:
            return 0
        
        array = [0]*(n+1)
        array[1] = 1
        max_val = 1
        for i in range(2,n+1):
            if i % 2 == 0:
                array[i] = array[i//2]
            else:
                array[i] = array[i//2] + array[(i//2)+1]
            
            max_val = max(max_val, array[i])
        
        return max_val

#######################################
#215. Kth Largest Element in an Array
#######################################
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        naive way would be to set sort and return k
        '''
        nums.sort()
        return nums[-k]
#using a heap
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        we can us heap and keep adding eleemnts onto the heap one by one
        whenever the heap exceeds k, just pop it off
        inserting into a heap is logN and we do this N times O(N*logN)
        '''
        heap = []
        for n in nums:
            heappush(heap,n)
            if len(heap) > k:
                heappop(heap)
        return heappop(heap)

#quick select
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        this is a textbook algo called quick select
        to find the kth smallest/largest in an unsorted array in one pass without sorting
        otherwise known as Hoare's selection algo
        we can fram saying that the kth largest element would be N-kth smallest eleemnt
        this is similar to quicksort
        first one chooses a pivot and efines its positions in a sorted array in linear time
        parition algo: one moves along an array, compares each element iwth a pivot, and moves all elements smaller than pivot to the left of the pivot
        quick select in a nut shell
        first we parition the array by first generating a radm pivot number
        we then parition into elements that are greater than pivot, less than or equal
        give reference to the lenght of number of elements greater than pivot and == pivot
        if k <= L, we are certain the element resides in L
        if k > l+M its on the right sides
        
        '''
        def quick_select(nums,k):
            if not nums:
                return
            pivot = random.choice(nums)
            left = [x for x in nums if x > pivot]
            mid = [x for x in nums if x == pivot]
            right = [x for x in nums if x < pivot]
            
            L,M = len(left),len(mid)
            
            if k <= L:
                return quick_select(left,k)
            elif k > L + M:
                return quick_select(right, k - L - M)
            else:
                return mid[0]
        
        return quick_select(nums,k)

                

#################################
#339. Nested List Weight Sum
#################################
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution(object):
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        '''
        i can peforem dfs on each element on the list
        if the element is a list get the value and mulitple by its depth
        otherwise recurse on the new nested list
        '''
        def dfs(nestedList,depth):
            total = 0
            for nested in nestedList:
                if nested.isInteger():
                    total += nested.getInteger()*depth
                else:
                    total += dfs(nested.getList(),depth + 1)
            return total
        
        return dfs(nestedList,1)

#BFS solution
class Solution(object):
    def depthSum(self, nestedList):
        """
        :type nestedList: List[NestedInteger]
        :rtype: int
        """
        '''
        we can use bfs in O(N) time, the total number of nodes 
        space O(M), the maximum number of nodes in a nested list at any depth
        we process only an integer element at that depth
        if it is another nested listed, we push elemtns bacn on to the q and process at a later depth
        not we need to extend left,
        if we just appened right, we would be able to multiply at the right correct depth
        '''
        depth = 1
        total = 0
        
        q = deque(nestedList)
        while q:
            #trick, keep processing for the nuymber of elements in q
            for i in range(len(q)):
                current = q.pop()
                #if its an integer
                if current.isInteger():
                    total += current.getInteger()*depth
                else:
                    #add to the beginning
                    for nextt in current.getList(): #could also use extendleft, but this makes more sense
                        q.appendleft(nextt)
            depth += 1
        
        return total


#####################
#Counter Sorted Vowel Strings
##########################
#works but TLE
class Solution(object):
    def countVowelStrings(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        think recursion and backtracking
        its possible that values will depend on the value of the previour char because it need to be smaller
        function that counts the number  of  valid strings of length n and whose first chars are noot less than the previous char
        in the recursive funcition iteratre on the possible chars for  the  first chat, which will be  all the voles not less than the last  char and  for each possible value of  c  increase the answer  by count(n-1,c)
        say for n = 2
        aaaaaaaaa
        naive way would be build a permutation check  the perm in  lexographic
        
        '''
        vowels = ['a','e','i','o','u']
        strings = []
        def rec_build(n, build):
            if len(build) == n:
                if len(build) == 1:
                    strings.append(build)
                    return
                else:
                    for i in range(len(build)-1):
                        if ord(build[i+1]) < ord(build[i]):
                            return
                    if i+1 == len(build) - 1:
                        strings.append(build)
                        return
                return
            for v in vowels:
                rec_build(n,build+v)
                
        
        rec_build(n,"")
        return len(strings)

#brute force with caching
class Solution(object):
    def countVowelStrings(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        this is a long solution write up, so lets go over
        #Brute Force Using Backtracking
        keep building a string up to to length n, fix that and back track
        as we are picking up voles in alpha rder, we know the result will be lexographic
        algo:
            as we start with the first vowel a then e and so on, we need a way to determine the current voled in a recursive function. we can use an interger var for that purpose (just index )
            base case when n is 0, we used them up and have found a valid string
        '''
        memo = {}
        def recCount(n, vowels):
            if n == 0:
                return 1
            if (n,vowels) in memo:
                return memo[(n,vowels)]
            result = 0
            for i in range(vowels,5):
                #n-1 is the backtracking, and we try to make a comb with the next vowel
                result += recCount(n-1,i)
            memo[(n,vowels)] = result
            return result
        
        return recCount(n,0)

#top down recursion with memoization
class Solution(object):
    def countVowelStrings(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        we can optimize the recursion just a bit
        if we observe, the problem follows a specific 
        for any given n, and number of vowels, the number of combinations is always equal to:
            number of combs for previous n and same number of vowels
            number of combinations for vowels - 1 and same n
        
        say we have our function rec(n,vowels), where n if the lenght and vowels is are the first v vowels
        the recurrence is devines as:
        rec(n,vowels) = rec(n-1,vowels) + rec(n,vowels - 1)
        example N = 2, vowels = 3 {a,e,i}
        resutl 6
        aa,ae,ai,ee,ei,ii
        
        then take N = 1, vowels = 4
        a,e,i,o
        
        combing we get
        aa,ae,ai,ee,ei,ii + ao,ae,io,oo
        
        more succintly
        countVowelStrings (n, vowels) = isSum(n - 1, vowels) + isSum(n, vowels - 1)
        algo:
            we can start with n and the 5 voles
            recusrively canddiate the results for every n and nowels using the cureence
            backtack base on the base cases:
                if  n==1, we only have one position left, here we just return the current number of vowels
                i.e n = 1, and three vowels,well that's just that number of available vowels
                if vowels = 1, well duh thats just one

        '''
        memo = {}
        def recCount(n,vowels):
            if n == 1:
                return vowels
            if vowels == 1:
                return 1
            if (n,vowels) in memo:
                return memo[(n,vowels)]
            result = recCount(n-1,vowels) + recCount(n, vowels - 1)
            memo[(n,vowels)] = result
            return result
        
        return recCount(n,5)

#using the formula class Solution(object):
    def countVowelStrings(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        we can think of this is finding the number of combinatinos
        let k = 5
        and we are choosing from n vowels
        we want to fint he number of combinations using n vowels
        combWithRep(n,k) = ((k+n-1)!) / ((k-1)!n!)
        since our k iso constatnt we can simplfy the expreassion
        (5 + n - 1)! / (5-1)!n! = (n+4)! / 4! n!
        we really just get
        (n+4)*(n+3)*(n+2)*(n+1) / 24
        '''
        return (n + 4) * (n + 3) * (n + 2) * (n + 1) / 24

###########################
#Max Number of K-Sum Pairs
###########################
class Solution(object):
    def maxOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        pass the array once gett ing the complements of each num
        k - num[i]
        [1,2,3,4] and k = 5
        [4,3,2,1]
        get counts of nums
        get counts of k - num
        see if i can make a pair,
        then delete from complements
        algo:
            get count of occurences in the array
            pass the array looking back into the hash for counts and also look for its complement = k - num
            EDGE CAE, it is possible that the current elemnt is also taken before and paird with some other element, hen we check if both element of the pair cur and comp are present in hash, uf yes we form the pair and decrement by 1
            EDGE CASE again, if the value of current and comp are the same we need at least two occurences of that element present in the array, otherwise we cannot fomr the pair
            exapmle, k = 6, and we are at 3, we need two of them!
            every time we find a suitable pair of elements incremnt count
        '''
        #get occurnces of numbers
        counts = Counter(nums)
        pairs = 0
        for num in nums:
            #find the complement
            complement = k - num
            if complement in counts:
                #edge case 1, enough elements
                if counts[num] > 0 and counts[complement] > 0:
                    #edge case when they are the same and count at nums < 2
                    #i still don't really get why this edge case is holding
                    if num == complement and counts[num] < 2: #we need two of the same
                        continue
                    pairs += 1
                    counts[num] -= 1
                    counts[complement] -= 1
        return pairs
        
class Solution(object):
    def maxOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        single pass hash
        for every eleemn ni nums, we try to find its complement pait if it existss in the map
        if it does, there is no need to add the current elemtn to the map and simpile remove the comp from the hashmap
        if the comp does not exist in the map, we could add the current element to the map
        in this approach the hashmap would only hold those array elements for which we have not yet found a suitable pair with sum == k
        when the elements are paired up, we remove from the hash
        in one pass,we build tha map and also find the pairs
        algo:
            init hash to store elements traverse until now
            count var storing pairs
            traverse each element in the array nums and for each element,curr, calc comp and check if comp is n map
            if compe exists rmeove frmo map (note that we would not add the current element in the map here)
            otherwise add the current elemtn to the map so it can be apired with some other element in the future
        '''
        mapp = {}
        count = 0
        for num in nums:
            #check if complement in mapp so far
            if mapp.get(k-num,0) > 0: #it exsits
                #remove compe from map
                mapp[k-num] = mapp[k-num] - 1
                count += 1
            else:
                #it does not exist, put it in hash
                mapp[num] = mapp.get(num,0) + 1
        return count
        
class Solution(object):
    def maxOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        another way would be to sort nums and make pairs from left and right
        then move the pointers depedning on whether or not they are > or < k
        else move boht and count += 1
        '''
        nums.sort()
        l,r = 0,len(nums)-1
        count = 0
        while l < r:
            if nums[l] + nums[r] < k:
                l += 1
            elif nums[l] + nums[r] > k:
                r -= 1
            else:
                l += 1
                r -= 1
                count += 1
        return count

class Solution(object):
    def maxOperations(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        another way wouyld be to use a seen set and only take from the hashmap if we havent seen them
        the only edge case would be  if x == k - x
        in which just take the count // 2
        
        '''
        counts = Counter(nums)
        seen = set()
        pairs = 0
        
        for num in nums:
            if num not in seen and (k-num) in counts:
                #edge case when they are the same
                if num == k - num:
                    pairs += counts[num] // 2
                else:
                    pairs += min(counts[num],counts[k-num])
            
            seen.add(num)
            seen.add(k-num)
        
        return pairs


#############################
#Longest Palindromic Substring
##############################
#TLE obvie
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        well lets just code out a brute force solution
        examine all possible substrings and check if it is a palindrome
        if it is update the maxlength
        '''
        def isPalindrome(s):
            if not s:
                return True
            N = len(s)
            l, r = 0, N-1
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True
        
        max_length = 0
        answer = None
        N = len(s)
        for i in range(N):
            for j in range(i,N):
                substring = s[i:j+1]
                if isPalindrome(substring):
                    if len(substring) > max_length:
                        max_length = len(substring)
                        answer = substring
                    
        return answer
        

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        https://leetcode.com/problems/longest-palindromic-substring/discuss/900639/Python-Solution-%3A-with-detailed-explanation-%3A-using-DP
        this is the dp solution
        we can set up a 2d array where ij correspond to the slice of the string
        example babad dp[2][3] = s[2:3] = ba
        fill in diagnoal True, a single char is by itself a palindrome
        don't traverse bottom part of the diagnoal, since we cannot reverse slice
        1. iterate backwards starting from the right most botom ceel to the top
            start out loop i and inner loop j
        2. pick char from string base on position i and j, if the char matches you need to check two conditions
            a. if len(substring) == 1, its a palindrom
            b. if len(substring)  > 1:
                check inner also substring? How?
                go to the bottom left corner and check if it s True
                Eg. if dp[i][j]= cur_sub_string = 'ababa' --> True because dp[i+1][j-1] is True
                dp[i+1][j-1] = 'bab' = True, we're just shrinking the window
                .Howerver if dp[i][j]= cur_sub_string = 'abaca' --> False because dp[i+1][j-1] is False
                dp[i+1][j-1] = 'bac' = False --> not palindrom
            if dp[i+1][j-1] == True: its'a palindrome
            noe compre the length of the current aplindrome and previous substring and take he max
            else: just pass
            
        '''
        longest = ''
        dp = [[0]*len(s) for _ in range(len(s))]
        #fill out diagonal
        for i in range(len(s)):
            dp[i][i] = True
            longest = s[i]
            
        #fill out dp table
        for i in range(len(s)-1,-1,-1): #goin backwards
            #j starts to end end, and we only want upper
            for j in range(i+1,len(s)):
                #if chars match
                if s[i] == s[j]:
                    #if len slicied sub_string is just one letter if the characters are equal, we can say they are palindomr dp[i][j] =True 
                    #if the slicied sub_string is longer than 1, then we should check if the inner string is also palindrom (check dp[i+1][j-1]
                    if j-i == 1 or dp[i+1][j-1] == True:
                        dp[i][j] = True
                        #update max len
                        if len(longest) < len(s[i:j+1]):
                            longest = s[i:j+1]
                            
        return longest
            

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        the idea is to start from the center and keep expanding so long as the next candidate char makes it a palindrom
        '''
        N = len(s)
        #function to return end points of a palindrome if it is one
        def LP(l,r):
            while r < N and l >=0:
                if s[l] != s[r]:
                    break
                r+=1
                l-= 1
            return l,r
        
        #there are two cases we need to check for
        #odd length we start from the center
        #even length we start from center-1 and center + 1
        #we then check if the new ends are greater
        
        start,end = 0,0
        for i in range(N):
            cand_l, cand_r = LP(i,i)
            #check greater
            if cand_r - cand_l > end - start:
                start = cand_l
                end = cand_r
            #now check for the even case
            cand_l,cand_r = LP(i,i+1)
            if cand_r-cand_l > end-start:
                start = cand_l
                end = cand_r
        
        return s[start+1:end]


###################
#Valid Parentheses
###################
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        
        for char in s:
            #any opening
            if char == '(' or char == '{' or char == '[':
                stack.append(char)
            elif stack and char == ')' and stack[-1] == '(':
                stack.pop()
            elif stack and char == ']' and stack[-1] == '[':
                stack.pop()
            elif stack and char == '}' and stack[-1] == '{':
                stack.pop()
            else:
                return False
            
        return len(stack) == 0

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        load each char onto a stack only if its opening
        check of top is closing bracket and pop it off
        if its not closing, return False
        if we encounter an opening bracket we push onto the stack
        if we later encounter a closingbrackert then we check the top elemnt  is an opneindg bracket of the char we are on, if it is we pop it off, else its an invalid exprssion
        at the end were are left with items in stack which mean it is in invalid expression
        '''
        stack = []
        mapping = mapping = {")": "(", "}": "{", "]": "["}
        
        for ch in s:
            #check if in mapping
            if ch in mapping:
                #view the twop
                top = stack.pop() if stack else "#" #only pop if there is a stack
                #check closting
                if mapping[ch] != top:
                    return False
            else:
                stack.append(ch)
                    
        return not stack

        class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        load each char onto a stack only if its opening
        check of top is closing bracket and pop it off
        if its not closing, return False
        if we encounter an opening bracket we push onto the stack
        if we later encounter a closingbrackert then we check the top elemnt  is an opneindg bracket of the char we are on, if it is we pop it off, else its an invalid exprssion
        at the end were are left with items in stack which mean it is in invalid expression
        '''
        stack = []
        mapping = mapping = {")": "(", "}": "{", "]": "["}
        
        for ch in s:
            #check if in mapping
            if ch in mapping:
                if stack and mapping[ch] == stack[-1]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(ch)
        
        return len(stack) == 0

###########################
#Find the Most Competitive Subsequence
############################
class Solution(object):
    def mostCompetitive(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        '''
        a subsequence of nums, can be created by removing or removnig zero elements and having length k
        we define a more competitive subsequence as: given subseq a and b, the position where the differ first has the smallest value
        note we are not taking element from nums, we simplet delte and shift *not permuting*
        similar to remove k digits
        https://leetcode.com/problems/find-the-most-competitive-subsequence/discuss/1027495/Python-Stack-solution-explained
        we can traverse the original nums array and push canddiate values on to the stack
        if it happens that the new number is less than the top of your stack, we can still affour to delte another number
        left most number has priority*
        we pop from our stack and push the new umber
        example
        [1,4,5,3,2,8,7]
        stack progression
        [1]
        [1,4]
        [1,4,5]
        [1,4,3]
        [1,3]
        [1,2]
        but since we have deleted len(nums) - k numbers we cant delete anymore
        [1,2,8]
        [1,2,8,7]
        #check out remove k digits again LC 402
        '''
        deletions = len(nums) -  k #since we want a subesqeunce of len(k)
        stack = []
        
        for num in nums:
            while stack and num < stack[-1] and deletions > 0:
                stack.pop()
                deletions -= 1
            stack.append(num)
            
        return stack[:k] #for some cases we may end up adding the final num making Len(stack) > k
        #in which case we only want k elements

class Solution(object):
    def mostCompetitive(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        '''
        just a review from the offical LC
        the idead would be to created a increasing subsequence(in the first k elemenets)
        example
        nums = [3,2,5,4] and k = 3
        _,_,_
        3,_,_
        2,_,_
        2,5,_
        2,5,4 #we need at least length 3
        we can use deque, giving us access to the left and right ends
        we need a way to konw the number of elements that can be dropped from the array, len(num) -k
        now start passing through the array
            compare last element in 1 with curr
            if the curr element is less than the last element in q,continue to pop so long as we are greater than the number of elements that can be dropped
            when we pop decremtn count by 1
            
        '''
        q = deque()
        allowable = len(nums) - k
        
        for num in nums:
            while q and q[-1] > num and allowable > 0:
                q.pop()
                allowable -= 1
            
            q.append(num)
        result = []
        for i in range(k):
            result.append(q.popleft())
        return result

###################################
#Determine if Two Strings Are Close
###################################
#112/144
class Solution(object):
    def closeStrings(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: bool
        """
        '''
        strings are close if:
            1. can swap any two existing chars abcde -> aecdb
            2. transform every occurcne of an exiting char into another char: aacabb -> bbcbaa (a->b and b->a)
        operation 1 allows for freely swapping chars (set)
        operation 2 allows to freely assign char counts
        '''
        if set(word1) == set(word2):
            #check counts
            count1 = Counter(word1)
            count2 = Counter(word2)
            #in the counts, there needs to be at least 2 of the same count
            if len(set(count1.values()).intersection(set(count2.values()))) % 2 == 1:
                return True
            else:
                return False

class Solution(object):
    def closeStrings(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: bool
        """
        '''
        strings are close if:
            1. can swap any two existing chars abcde -> aecdb
            2. transform every occurcne of an exiting char into another char: aacabb -> bbcbaa (a->b and b->a)
        operation 1 allows for freely swapping chars (set)
        operation 2 allows to freely assign char counts
        '''
        if set(word1) == set(word2):
            #check counts
            count1 = Counter(word1)
            count2 = Counter(word2)
            #in the counts, now we need to check if the Counts of Counts in both are the same
            if Counter(count1.values()) == Counter(count2.values()):
                return True
            else:
                return False
        else:
            return False
        

##################
#One Edit Distance
#####################
#close one, but is ont dp :(
class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        '''
        return true of one edit distance apart
        we can either insert,delete, or replace
        this is just where levenstein == 1
        dp all the way down and check if bottom right == 1
            a   c   b    
        a   0   1   2         
        b   1   1   1
        dp solution
        fill up first row and first col
        
        '''
        #dp array rows x cols -> len(s) -> len(t)
        rows = len(s)
        cols = len(t)
        dp = [[0]*(cols) for _ in range(rows)]
        #fill in top row
        for i in range(cols):
            if s[0] == t[i] and i == 0:
                dp[0][i] == 0
            else:
                dp[0][i] == i+1
                
        #fil in first col
        for i in range(rows):
            

class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        '''
        we need to ensure that s and t are not too far away from one another
        if abs(len(s) - len(t)) > 2, they are not within one edit distance away
        
        we then assusme len(s) < len(t), if not we can just swap the inputs
        with the same length, we can pass along the strings and see where they are differennt
        
        if we get the len(s) finding that they are not different then:
            1. they are equal
            2. they are one edit distance away (we contrained > 2 from the first part)
        
        but if we did not get to len(s) and found s[i] != t[i]:
            if the strings are the same length, all next chars should be euqla to keep one away!
            if t is one char longer than s, the additional char t[i] should be fht eonly differnce between both strings. 
            
        example equal length
        s = 'abxcd'
        t = 'abycd'
        s[i:n] = t[i:n]
        
        example not equal length
        s = 'abcd'
        t = 'abxcd'
        s[i:n] = t[i+1:n+1]
        '''
        N_s = len(s)
        N_t = len(t)
        
        #ensure  s is shorter than 1
        if N_s > N_t:
            return self.isOneEditDistance(t,s)
        
        #check not more than 2
        if N_t - N_s > 2:
            return False
        
        #now pass the shorter strongs
        for i in range(N_s):
            #first mismatch
            if s[i] != t[i]:
                #equal length
                if N_s == N_t:
                    return s[i+1:] == t[i+1:]
                #not equal length, this case is trick
                else:
                    return s[i:] == t[i+1:]
        
        #now we are the the end of the string
        #check qual lenghts, since s was shorter and they are within 1, check
        return N_s + 1 == N_t


################################
#Sort The Matrix Diagonally
###############################
class Solution(object):
    def diagonalSort(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        there was a trick, to record the elements along a diagonal
        (1,0),(2,1) = diff is 1
        (0,1),(1,2),(2,3) diff is -1
        (0,0),(1,1),(2,2)
        all elements along a diagnoal have the same i-j
        dump the element into a hash for each diag
        sort each one 
        then put them back into the matrix
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        diags = defaultdict(list)
        for i in range(rows):
            for j in range(cols):
                diags[i-j].append(mat[i][j])
                
        #not sort in reverse order, we are going to pop from each each
        for k,v in diags.items():
            diags[k] = sorted(v,reverse=True)
        #now walk the matrix grabbing the right values after popping from the right hash
        for i in range(rows):
            for j in range(cols):
                if diags[i-j]:
                    mat[i][j] = diags[i-j].pop()
        return mat

#brute force
class Solution(object):
    def diagonalSort(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        the brute force way would be to walk the diagonals starting from the bottome left to the upper right
        '''
        rows = len(mat)
        cols = len(mat[0])
        diags = []
        
        #walk the diags start at the first col, stop at fiirst row
        for i in range(1,rows):
            curr_row,curr_col = rows - i, 0
            temp = []
            while curr_row < rows and curr_col < cols:
                temp.append(mat[curr_row][curr_col])
                curr_row += 1
                curr_col += 1
            diags.append(sorted(temp)[::-1])
            
        #now walk the diags start staying at the first row but moving through all the cols
        for i in range(cols):
            curr_row, curr_col = 0,i
            #down right until i cannot
            temp = []
            while curr_row < rows and curr_col < cols:
                temp.append(mat[curr_row][curr_col])
                curr_row += 1
                curr_col += 1
            diags.append(sorted(temp)[::-1])
            
        #now build the new matrix, just replace mat[i][j]
        #which was just walking the diags again
        #traverse our diags and take first element
        
        #reverse diags
        diags = diags[::-1]
        
         #repeat for rows
        for i in range(1,rows):
            curr_row,curr_col = rows - i, 0
            #get elemenets
            candidates = diags.pop()
            while curr_row < rows and curr_col < cols and candidates:
                mat[curr_row][curr_col] = candidates.pop()
                curr_row += 1
                curr_col += 1
        
        #repeat for cols
        for i in range(cols):
            curr_row, curr_col = 0,i
            #down right until i cannot
            #get elemenets
            candidates = diags.pop()
            while curr_row < rows and curr_col < cols and candidates:
                mat[curr_row][curr_col] = candidates.pop()
                curr_row += 1
                curr_col += 1
        return mat

#using a heap
class Solution(object):
    def diagonalSort(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[List[int]]
        """
        rows = len(mat)
        cols = len(mat[0])
        
        diags = defaultdict(list)
        for i in range(rows):
            for j in range(cols):
                heappush(diags[i-j],mat[i][j])
                
        #now walk the matrix grabbing the right values after popping from the right hash
        for i in range(rows):
            for j in range(cols):
                if diags[i-j]:
                    mat[i][j] = heappop(diags[i-j])
        return mat

#####################
#Merge K sorted Lists
#####################
#brute force solution
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        '''
        well the naive way would be to get all the elements, sort and recreate the list
        '''
        elements = []
        for linked in lists:
            while linked:
                elements.append(linked.val)
                linked = linked.next
        #sort
        elements.sort()
        dummy = ListNode()
        cur = dummy
        for num in elements:
            cur.next = ListNode(val=num)
            cur = cur.next
        return dummy.next

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        '''
        i cant megere linked lists in in pairs
        take the first one, merge with the first one, and make that new list
        then with the new list merge again with the second one
        '''
        #edge case
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]
        
        def mergeTwo(l1,l2):
            #mergine two at time and reutring the new one
            dummy = curr = ListNode()
            while l1 and l2:
                if l1.val < l2.val:
                    curr.next = ListNode(l1.val)
                    l1 = l1.next
                else:
                    curr.next = ListNode(l2.val)
                    l2 = l2.next
                curr = curr.next
            
            #left over
            if l1:
                curr.next = l1
            if l2:
                curr.next = l2
            return dummy.next
                
        result = lists[0]
        for linked in lists[1:]:
            result = mergeTwo(result, linked)
            
        return result

#using a heap
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        '''
        i could use a heap, prrioty queue and keep taking the min
        when i take the min move the pointer to the correspoding linked list
        '''
        heap = []
        
        for i in range(len(lists)):
            if lists[i]:
                heappush(heap,(lists[i].val,i))
        
        dummy = curr = ListNode()
        
        while heap:
            val, i = heappop(heap)
            #set the value
            curr.next = ListNode(val)
            #now move point
            if lists[i].next:
                #move pointer and push next
                lists[i] = lists[i].next
                heappush(heap,(lists[i].val,i))
            curr = curr.next
        return dummy.next

######################################################
#Check If All 1's Are at Least Length K Places Away
######################################################
#65 of 67
class Solution(object):
    def kLengthApart(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        '''
        each time i find a number 1, check that it is K or more places away from the next one
        special case when
        [1,0,0,0,1,0,0,1,0]
        2
        this shuld be true, it doenst matter
        '''
        for i in range(len(nums)):
            if nums[i] == 1 and i != len(nums)-1:
                j = i
                while nums[j+1] != 1 and j+1 < len(nums)-1:
                    #move right
                    j += 1
                #check if i got to the end and wheter it it zero or not
                if j == len(nums) -1 and nums[j]== 0:
                    return True
                else:
                    if j - i < k:
                        return False
        return True

class Solution(object):
    def kLengthApart(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        '''
        if we meet a zero, incrent distance by 1
        if we meet z 1, and currnet distance is >= k, then just update count
        if we meet 1 and current distance < k, return false
        if we got to the end, return true
        '''
        count = k
        
        for num in nums:
            if num == 0:
                count += 1
            elif num == 1 and count >=k:
                #update
                count = 0
            else:
                return False
        
        return True

class Solution(object):
    def kLengthApart(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        '''
        we can just see if the the next 1 and current 1 are within k
        to help with this assign last one idx as None
        '''
        last_one = None
        for i in range(len(nums)):
            if nums[i] == 1:
                if last_one != None and (i - last_one -1) < k:
                    return False
                last_one = i
                
        return True

#using bit wise operators
class Solution(object):
    def kLengthApart(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        '''
        we can used bit wise operators, and we can use the trick to remove trailing zeros
        algo: 
            1. convert binary array into interg x
            2. consider bases case x == 0 or k == 0, True
            3. remove trailing zeros in binary rep
            4. while x is greater than 1
                a. remove trailing 1-but with x >> = 1
                b. remove trialign zeros one by one and count them
                c. when we hit a one again, check count < k, 
        '''
        #convert binary array to int
        x = 0
        for num in nums:
            x = (x << 1) | num
            
        #base case
        if x == 0 or k == 0:
            return True
        
        #removeee trialing zeros
        while x & 1 == 0:
            x = x >> 1
            
        #now we need to keep popping of bits
        while x != 1:
            #first remove trailing 1 bit
            x = x >> 1
            #count trailing zeros
            count = 0
            while x & 1 == 0:
                x == x >> 1
                count ++ 1
            if count < k:
                return False
        return True

##########################
#Path With Minimum Effort
##########################
#TLE
class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """
        '''
        ok so i could do a dfs from from the upper left to bottomr right
        and explore each path and for each path record the effort by updating a max
        push this max effort val into a list
        and return the min value of the list
        '''
        rows = len(heights)
        cols = len(heights[0])
        
        self.efforts = []
        def dfs(x,y,effort):
            #ending call
            if x == rows - 1 and y == cols - 1:
                self.efforts.append(effort)
                return
            #boundary check while dfsing
            if x < 0 or x >= rows or y < 0 or y >= cols:
                return
            for dirr in [(1,0),(-1,0),(0,1),(0,-1)]:
                #boundary check again
                if x + dirr[0] < 0 or x+dirr[0] >= rows or y + dirr[1] < 0 or y +dirr[1] >= cols:
                    pass
                else:
                    dfs(x+dirr[0],y+dirr[1],max(effort,abs(heights[x][y] - heights[x+dirr[0]][y+dirr[1]])))
                    
        dfs(0,0,0)

class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """
        '''
        brute force using backtring
        ok so i could do a dfs from from the upper left to bottomr right
        and explore each path and for each path record the effort by updating a max
        push this max effort val into a list
        and return the min value of the list
        to make the algo more efficient, once we have found a path from start fo end, track max difference for the path (maxSoFar)
        if we have already found a path to the destination cell with maxSoFar, then we would only explore other paths if it takes less effort
        for a given x,y scan edjacent, 
        the max difference keeps tracking of the max abs diff so far, then we go in the direction of the smallest
        to back track we just mark the cell with a special character, in this case zero is acceptable
        '''
        row = len(heights)
        col = len(heights[0])
        self.max_so_far = float('inf')

        def dfs(x, y, max_difference):
            #max difference of the difference along our current path we are one
            if x == row-1 and y == col-1:
                self.max_so_far = min(self.max_so_far, max_difference)
                return max_difference
            current_height = heights[x][y]
            heights[x][y] = 0
            #we need the min effort to examine other neighbors
            min_effort = float('inf')
            for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                adjacent_x = x + dx
                adjacent_y = y + dy
                if 0 <= adjacent_x < row and 0 <= adjacent_y < col and heights[
                        adjacent_x][adjacent_y] != 0:
                    #now for each neghbor check if the new diffeence is greater
                    current_difference = abs(
                        heights[adjacent_x][adjacent_y]-current_height)
                    #take the next max difference
                    max_current_difference = max(
                        max_difference, current_difference)
                    #if going to the adjacent neighbor results in a smaller difference,recurse, but store the result
                    if max_current_difference < self.max_so_far:
                        result = dfs(adjacent_x, adjacent_y,
                                     max_current_difference)
                        #after we recurse, we update the min effort
                        min_effort = min(min_effort, result)
            #back track
            heights[x][y] = current_height
            #return call
            return min_effort

        return dfs(0, 0, 0)
    
'''
TimeComplexity:
let m be the number of rows and n cols
O(3^(m*n)), draw the tree to see this, in the worse case the depth of tree is the number of nodes m*n
Space O(m*n) for the call stack from each node
'''

#djistra's variations
class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """
        '''
        variations of Dikstra
        similart to finding the shortest path from a source tod esit
        shortest path is one with min abs difference
        difference between adjacent cells can be thought of as weighted edge
        algo:
            1. differene matrix row by col where each cell represents the min effort require to reach that cell from all possible pths, init all values to float('inf')
            2. after visiting each cell, the adjacent become avaiable. we update the abd diff between each cell and adja in the differene amtrix and at the same time push all adj cells in priorty q, the priortiy q holds all reachable cells sorted by value
            3. starting at 0,0 and push to q, visit each cell and put into q - the less is the diffeence of a cell is higher priorty
                BFS
                get top of q and for each of 4 adjacents calc the maxdifference
                if this value is greater than the max diffeercne, we updat that vlaue with maxdiffer
            #NOTE, for updating the priority q, we must delte the old value and reinsert with the new maxdifference val
        '''
        rows = len(heights)
        cols = len(heights[0])
        #diff matrx, min effort to reach that cell so far
        difference_matrix = [[float('inf')*cols for _ in range(rows)]]
        difference_matrix[0][0] = 0
        #visied mark
        visited = [[False]*cols for _ in range(rows)]
        q = [(0,0,0)] #difference,x,y
        while q:
            difference,x,y = heapq.heappop(q)
            #mark 
            visited[x][y] = True
            for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                adjacent_x = x + dx
                adjacent_y = y + dy
                #bounds check
                if 0 <= adjacent_x < rows and 0 <= adjacent_y < cols and visited[adjacent_x][adjacent_y] != False:
                    #get the difference
                    current_difference = abs(heights[adjacent_x][adjacent_y] - heights[x][y])
                    #find the max
                    max_difference = max(current_difference, difference_matrix[x][y])
                    #if the max diff if greater than what is currently in diff matrix we need to update
                    if difference_matrix[adjacent_x][adjacent_y] > max_difference:
                        difference_matrix[adjacent_x][adjacent_y] = max_difference
                        #update q
                        heapq.heappush(max_difference, adjacent_x, adjacent_y)
                        
        return difference_matrix[-1][-1]

#BFS Binary Search
class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """
        '''
        Binary Search Using BFS
        we know in our heights matrix, there is a max height and a minheight
        alludes to binary serach
        intuition:
            given the lowerbound as 0 and upper bound as 10^6 can binarys earch to find the mid
            if there exists a path form source cell tod est cell with effort less than mid, we know it lies in 0 mid
            else it liefes in mud 10^6
        algo:
            1. init the lower bound left as 0 10^6 as the max height in the array
            2. using BFS, check if there exists a pathf rom source to dest with less effort than the mid
            3. need helper bfs function which returns bool
            4. if there exists a path from source to cell, we must update the result val as a min of the current result and mid and contie the search
        '''
        rows = len(heights)
        cols = len(heights[0])
        
        def canreach(mid):
            visited = set()
            q = deque([(0,0)])
            while q:
                x,y = q.popleft()
                if x == rows -1 and y == cols -1:
                    return True
                visited.add((x,y))
                for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                    adjacent_x = x + dx
                    adjacent_y = y + dy
                    if 0 <= adjacent_x < rows and 0 <= adjacent_y < cols and (adjacent_x,adjacent_y) not in visited:
                        current_difference = abs(heights[adjacent_x][adjacent_y]-heights[x][y])
                        if current_difference <= mid:
                            #go keep going down the path
                            visited.add((adjacent_x,adjacent_y))
                            q.append((adjacent_x,adjacent_y))
        
        left = 0
        right = max([heights[row][col] for row in range(rows) for col in range(cols)])
        while left < right:
            mid = (left + right)//2
            if canreach(mid):
                right = mid
            else:
                left = mid + 1
        return left

#DFS Binary Search
class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """
        '''
        DFS binary search, similar to BFS
        we binary search in the space of all possible efforts
        and if we reach the dest cell from the source cell with that effort, well surely there must be one that is even smaller
        so we can cut the search space in half
        '''
        rows = len(heights)
        cols = len(heights[0])
        
        def canreach(x,y,mid):
            if x == rows -1 and y == cols -1:
                return True
            visited.add((x,y))
            for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                adjacent_x = x + dx
                adjacent_y = y + dy
                if 0 <= adjacent_x < rows and 0 <= adjacent_y < cols and (adjacent_x,adjacent_y) not in visited:
                    current_difference = abs(heights[adjacent_x][adjacent_y]-heights[x][y])
                    if current_difference <= mid:
                        #go keep going down the path
                        visited.add((adjacent_x,adjacent_y))
                        if canreach(adjacent_x,adjacent_y,mid):
                            return True
        left = 0
        right = max([heights[row][col] for row in range(rows) for col in range(cols)])
        while left < right:
            mid = (left + right)//2
            visited = set()
            if canreach(0,0,mid):
                right = mid
            else:
                left = mid + 1
        return left
        class Solution(object):
    def minimumEffortPath(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: int
        """
        '''
        using a heap, without the difference maxtrix
        always go fown the adjacent cell that has the smallest effort
        '''
        rows = len(heights)
        cols = len(heights[0])
        heap = [(0,0,0)] #min effort so far for current path, x,y
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        visited = set()
        output = 0
        
        while heap:
            min_effort,x,y = heappop(heap)
            #we never need to take the min because python is a min heap
            output = max(output,min_effort)
            if x == rows - 1 and y == cols - 1:
                return output
            visited.add((x,y))
            for dx,dy in dirs:
                new_x,new_y = x+dx,y+dy
                if 0 <= new_x < rows and 0 <= new_y < cols and (new_x,new_y) not in visited:
                    effort = abs(heights[x][y]-heights[new_x][new_y])
                    #push back into heap
                    heappush(heap,(effort,new_x,new_y))

##############################################
# Concatenation of Consecutive Binary Numbers
##############################################
#TLE
class Solution(object):
    def concatenatedBinary(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        i could generate the bits into a string
        then convert that to base 10        
        '''
        def recbinary(n):
            if n == 0:
                return '0'
            return recbinary(n//2) + str(n%2)
        
        string = ""
        for i in range(1,n+1):
            string += recbinary(i)[1:]
        
        output = 0
        N = len(string)
        for i in range(N):
            output += int(string[i])*(2**(N-i-1))
        return output % (10**9 + 7)

#the cheeky way
class Solution(object):
    def concatenatedBinary(self, n):
        """
        :type n: int
        :rtype: int
        """
        MOD = 10**9 + 7
        concatenation = "".join(bin(i)[2:] for i in range(n + 1))
        return int(concatenation, 2) % MOD

#using the recurrent relationship
class Solution(object):
    def concatenatedBinary(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        evidently there is a recursive formula and we can build this bottom up
        a(n) = a(n-1)*2^(1+floot(log_2(n))) + n
        '''
        import math
        sequence = [1]
        for i in range(2,n+1):
            next_num = (sequence[-1]*2**(1+math.floor(math.log(i,2)))+i) % (10**9 + 7)
            sequence.append(next_num)
        return int(sequence[-1])

class Solution(object):
    def concatenatedBinary(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        using bit shifts
        for example what happens when we add 3 (binary 11) to the previosu result 110
        what we do is shift the previous '110' two units to the left and add 3
        110
        shift left 2
        11000
        add number
        11011, we move left two becase 11 in binary has length 2
        to find out the length of the binary rep we can use log2 (or we can record the lenght and increast when we meet a power of 2)
        in conclusion, we can multiple the previous result by some power of 2 shift to the left and add the number to get the result
        algo:
            1. init an integr to store the final result
            2. iterater from 1 to n, for each number i
                find the length of the binary rep of the number length
                update rsult to 2**lenght + i
        '''
        mod = 10**9 + 7
        length = 0 #the lenght of the bit being added
        result = 0
        for i in range(1,n+1):
            #check power of 2, increase length
            if math.log(i,2).is_integer():
                length += 1
            #first exponenitl the result
            result <<= length
            #add the new number
            result += i 
            #mod it
            result %= mod
        return result
        
class Solution(object):
    def concatenatedBinary(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        evidently log(n) is log(n), and not O(1), think binary search trying to find the right exponent
        we can check if the number is power of 2 in O(1) time
        for example: 
        x == 4 is 100 in binary
        x - 1 == 3 011
        x & (x-1) to check power of two
        to 'concat' which can just use |
        '''
        mod = 10**9 + 7
        length = 0
        result = 0
        for i in range(1,n+1):
            #check power of two and increment length
            if i & (i-1) == 0:
                length += 1
            #shift
            result <<= length
            #concat
            result |= i
            result %= mod
        return result

############################################
#Smallest String With A Given Numeric Value
############################################
class Solution(object):
    def getSmallestString(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        '''
        hint 1, think greedily, 
        thank god it wasn't a recursion
        if we want the smallest one, then we would start with the the alpha that is the most beginning
        stack
        starting from the end
        only had the greatest value char if k-val(char) > n-1
        n = 3 and k = 27
        z   27 -26 = 1 so no
        y   27 - 25 = 2
        k = 27 - 25
        yaa
        
        '''
        string = ""
        while n > 0:
            n -=1
            start = 25
            while start >= 0:
                candidate = chr(ord('a')+start)
                if k - (ord(candidate) - ord('a')) < n:
                    start += 1
                    break
                start -= 1
            k -= ord(candidate)
            print candidate

#left to right
class Solution(object):
    def getSmallestString(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        '''
        building left to right
        just a few notes, if k==n, then every position would need to be an a
        k can be at most 26*n
        we need to keep track of 2 questions going left to right
            1. the remaining value of k at any given point
            2. the number of positions that are yet to be filled
            just note that lexographically sorted does not always mean alphabetical
            example dz, 4 + 26 = 30, is the same as ey, 5+25 = 30
        two scenarios
        k = 32, n = 4
        a,a,_,_
        the last spot is maxed to 26, and we would just fill in the reminder sport 5
        a,a,d,z
        k = 24, n = 4
        a,a,_,_
        we are note at k = 22,last would be 21, which is u, then put in a
        a,a,a,u
        rules, given a k value to be filled as positions left
            if k is greater than posL*26 we can reseve the max numerical valye z to that position
            otherwise we assing the msallest char at the current position
        algo:
            1. create char array to hold letters
            2. iterate from 1 to n filling each position. at each position find the positions left to be filled
                if the k > positoinsleft*26 we could revsere the last numeric char for each positions
                
        '''
        #result = [0]*n
        result = ""
        for i in range(n):
            positions_remaining = n -i - 1
            #case 1, k > positions_remaing*26
            if k > positions_remaining*26:
                #get the ASCII
                num_char = k - (positions_remaining*26)
                result += chr(ord('a')+num_char-1)
                k -= num_char
            #scenario 2, just add an a
            else:
                result += 'a'
                k -= 1
        return result

class Solution(object):
    def getSmallestString(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        '''
        initally start with all as
        go the last character and max it out, then decrement our k
        n = 3
        k = 27
        aaa
        24+1
        aay
        '''
        mapp = {}
        for i in range(26):
            mapp[i] = chr(ord('a')+i)
        #start off with all a's whicha re just ones
        result =[0]*n
        #we are doing this from right to left
        k -= n
        right = len(result) - 1
        
        while k >= 0 and right >=0:
            #if we more than 26 leftover, well max out the entry
            if k > 25:
                result[right] = 25
                #use it up
                k -= 25
            else:
                #use what's left
                result[right] = k
                #then make it a
                k = 0
            right -= 1
        return "".join(mapp[foo] for foo in result)

class Solution(object):
    def getSmallestString(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        '''
        the string is going to consist of some number of a's 
        some number of z's
        and on other number in between
        the number of z's is just the number of times the max goes into it
        '''
        if n*26 == k:
            return 'z'*n
        k -= n
        nz = k // 25
        na = n - nz -1 #one less to include the other characetr
        return 'a'*na+chr(ord('a')+(k%25))+'z'*nz

#another way
class Solution(object):
    def getSmallestString(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: str
        """
        '''
        building the number from right o left
        initally the positions start with all a
        if we have any k left, we try to reserve as much as possible for the ending positinos
        algo: 
            1. build result starting with number a
            2. starting from the end allocate the max possible value, as we have already allocated a at each positin with 1, the max additional value that we can add at each a is 25, 26-1
            3. caculate the addiontal value to be added given by add as minimu of 25 and k
        '''
        result = [1]*n
        #use up all the a's
        k -= n
        #starting backwards
        right = len(result) - 1
        while right >=0  and k > 0:
            #we either max out the last with 25 or use the left over k
            to_add = min(k,25)
            result[right] = result[right] + to_add
            k -= to_add
            right -= 1
        #recast each
        for i in range(len(result)):
            result[i] = chr(ord('a') - 1 + result[i])
        return "".join(result)


##########################################
#Vertical Order Traversal of a Binary Tree
##########################################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def verticalTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        '''
        descend the tree, in any order,
        but for each node pairs its node.val with the new x and y
        i can dump these values into a hash map 
        at the end just order the values by the hashmap
        '''
        self.mapp = defaultdict(list) #((x,y):val)
        
        def dfs(node,x,y):
            if node:
                self.mapp[(x,y)].append(node.val)
            if node.left:
                dfs(node.left,x-1,y-1)
            if node.right:
                dfs(node.right,x+1,y-1)
                
        dfs(root,0,0)
        #now i need to piece them together
        #group by xx
        #creat another mapp, this time by x coordinate and push a tuple (val, ycoord)
        x_mapp = defaultdict(list)
        for k,v in self.mapp.items():
            for foo in v:
                x_mapp[k[0]].append((foo,k[1]))
        #in each value, which is a list, i need to sort in the second element of the tuple
        x_coords = []
        for k,v in x_mapp.items():
            x_mapp[k] = sorted(x_mapp[k], key = lambda x: (x[1],-x[0]),reverse=True)
            #now grab only the first elements in each tuple
            values = []
            for val,coord in x_mapp[k]:
                values.append(val)
            x_mapp[k] = values
            x_coords.append(k)
        x_coords.sort()
        result = []
        for x in x_coords:
            result.append(x_mapp[x])
        return result

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def verticalTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        '''
        another way would be to push elements onto a heap
        '''
        mapp = defaultdict(list)
        
        def dfs(node,x,y):
            if not node:
                return
            #we want the highest y coordinate, since we push the negative value of it
            #y is always going to be negative, if we want the highest first negatte, 
            #python is a min heap
            heappush(mapp[x], (-y,node.val))
            dfs(node.left, x-1, y-1)
            dfs(node.right, x+1,y-1)
            
        dfs(root,0,0)
        output = []
        for k,v in sorted(mapp.items()):
            temp = []
            while v:
                element = heappop(v)
                temp.append(element[1])
            output.append(temp)
        
        return output

######################################
# 1675. Minimize Deviation in Array
######################################
class Solution(object):
    def minimumDeviation(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        there are a few appraoches here
        if element is even we divide by two
        if element is odd we multiply by two
        
        consequently, if an element is even we cannot increase it and if an element is odd we cannot decrease it
        
        we can try to increase all numbers that can be increased to their maximum and reduce them step by step
        recall deviation = max - minimum, and so there are onlt ways to decrease deviation, decrease max or increase min
        
        if we have increased all numbers to their max, then cannot increase the minimum, if we wan a smaller deviations, we can only decrease the max
        [4,10,2,6] deviation is 10 - 2= 8
        decrease the max
        [4,5,2,6] deviation is 5-2=3
        
        if the max is odd, then we can not decreas the maximum neither
        we need to find the maximum and minium after each modification, use a heap
        
        its OK to decrease tll numbers to their minimum and then increase them step by step
        algo:
            1. init max heap to evens,for an even number in num add into the heap, for and odd number multiply by 2 and put into eens
            2. maintain var for minimum
            3. take out the max number in evens and use the max and the maintiend min to update hte mind eiaviont. if max is even divie by two and push back into eens
            4. keep going intl the max number in heap is odd
        '''
        evens = []
        minimum = float('inf')
        for num in nums:
            if num % 2 == 0:
                heappush(evens,-num)
                minimum = min(minimum, num)
            else:
                heappush(evens,-num*2)
                minimum = min(minimum,num*2)
        
        min_deviation = float('inf')
        while evens:
            candidate = -heappop(evens)
            #update min deivation
            min_deviation = min(min_deviation, candidate - minimum)
            #we cannot increase any more, we can only decrease
            if candidate % 2 == 0:
                minimum = min(minimum,candidate //2)
                heappush(evens,-candidate//2)
            else:
                #if the next max is odd, we can only increase
                #which is not what we want ebcause we increased all possible to max first
                break
        return 

class Solution(object):
    def minimumDeviation(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/minimize-deviation-in-array/discuss/1041766/Python-Heap-solution-explained
        we can go through the array and push into a heap the possible ranges for each element, orderd by the minimum possible rangese
        '''
        heap = []
        for num in nums:
            current = num
            #if even we need to decrease it
            while current % 2 == 0:
                current //=2
            #push tuple of (min, max) back into the heap
            #if it was even, we had to keep reducing by two
            #otherise it was odd, we only multiplied by two ONCE
            heappush(heap,(current,max(num, current*2)))
        
        #in the first pass we brought everyything down to its minimum
        #the only we can minimize the deivation now is if the minimum can be increased
        
        #find maximum of lower limit allows
        max_lower_limit = max([i for i,j in heap])
        min_deviation = float('inf')
        
        while len(heap) == len(nums):
            current, limit = heappop(heap)
            min_deviation = min(min_deviation, max_lower_limit - current)
            if current < limit: 
                #push up to the limit stepwise
                heappush(heap, (current*2,limit))
                max_lower_limit = max(max_lower_limit,current*2)
        return min_deviation
        
        print heap


####################
#Next Permutation
###################
#brute force if something like this
class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        '''
        try the brute force approach first
        just generate all permutations and set it equal to the next permutation
        [1,2,3]
        [1,3,2]
        [2,1,3]
        [2,3,1]
        [3,1,2]
        [3,2,1]
        next permutation if can be reached is just swapping the last two elements
        '''
        nums.sort()
        perms = [p for p in itertools.permutations(nums)]
        print perms
        for i,p in enumerate(perms):
            if nums == p:
                nums[:] = list(perms[i+1])
                break
        if i == len(p) - 1:
            nums[:] = list(perms[0])

class Solution(object):
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        '''
        yeah ther was no way in hell i was gonna get this without seeing it first
        
        similar to the problem next greater element
        
        first we observe that for any given sequence, that is in descending order there is not next larger array possible, we need to find a pair of two successive numbers from the right, a[i] and a[i-1], where a[i] > a[i-1]
        
        now we can ask what rearrangment will produce the next larger one, i.e the next larger permutation. therefore we need to replace teh number a[i-1] with the number that is just larger than itself
        algo:
            1. from the right look for strictly decreasing sequence
            2. go one index after the start of hte decreasingn seuqnce, and go left again until we find the element just greater than this one
            3. swap them
            4. then reverse the found decreqsing sequence
        '''
        def reverse(s,start,end):
            while start < end:
                s[start],s[end] = s[end],s[start]
                start += 1
                end -= 1
        N = len(nums)
        right = N - 1
        while right >= 1 and nums[right] <= nums[right-1]:
            right -= 1
        
        #if we have gotten to zero, there is no decreasing subsequence
        #just reverse the whole thing
        if right != 0:
            j = right
            #find the next greater element
            while j+1 < N and nums[j+1] > nums[right -1]:
                j += 1
            nums[right-1],nums[j] = nums[j], nums[right-1]
            
        reverse(nums,right,N-1)