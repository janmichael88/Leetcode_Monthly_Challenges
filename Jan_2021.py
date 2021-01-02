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