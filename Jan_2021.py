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
                    #clear it again
                    visited[i] = False
        calculate(n,1)
        return self.count
            