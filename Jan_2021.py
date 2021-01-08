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
            if s[l] not in mapp:
                mapp.add(s[l])
                l += 1
                #update to get max_length
                max_length = max(max_length,l-r)
            else:
                #keep advanving r but remove
                mapp.remove(s[r])
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