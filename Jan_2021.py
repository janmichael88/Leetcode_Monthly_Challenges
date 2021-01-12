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
            