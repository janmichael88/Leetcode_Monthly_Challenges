#######################################
# 2022. Convert 1D Array Into 2D Array
# 01SEP24
######################################
class Solution:
    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        '''
        i can either precompute the 2d array and fill in the sports
        if im at cell (i,j) i can get its 1d index as i*
        '''
        if len(original) != m*n:
            return []
        
        ans = [[0]*n for _ in range(m)]
        for i in range(len(original)):
            #could also do:
            #row,col = divmod(i)
            row = i // n
            col = i % n
            ans[row][col] = original[i]
        
        return ans
        
#####################################################
# 1894. Find the Student that Will Replace the Chalk
# 02SEP24
#####################################################
class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        '''
        the sum of the whole array repeats itself
        find the point where it exceeds k
        its just modulu sum(chalk) for the remainder of chalk
        '''
        sum_chalk = sum(chalk)
        times_around = k // sum_chalk
        #chalk_left = max(0,k - sum_chalk*times_around)
        chalk_left = k % sum_chalk

        for i,student in enumerate(chalk):
            if student > chalk_left:
                return i
            chalk_left -= student
        
        return i
    
#binary search
class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        '''
        another solution is to do binary search on the pref_sum array of chalk
        the idea is to look for the index in pref_sum array that is < remaining chalk
        '''
        pref_chalk = [0]
        for c in chalk:
            pref_chalk.append(pref_chalk[-1] + c)
        
        chalk_left = k % pref_chalk[-1]
        #print(pref_chalk)
        #print(chalk_left)
        #look for upper bound
        left = 1
        right = len(pref_chalk) - 1
        ans = right
        while left < right:
            mid = left + (right - left) // 2
            if pref_chalk[mid] <= chalk_left:
                left = mid + 1
            else:
                ans = mid
                right = mid
                
        return ans - 1
    
##############################################
# 1945. Sum of Digits of String After Convert
# 03SEP24
#############################################
class Solution:
    def getLucky(self, s: str, k: int) -> int:
        '''
        generate starting number and repeat k times
        '''
        starting_number = 0
        for ch in s:
            pos = (ord(ch) - ord('a')) + 1
            if pos // 10 > 0:
                starting_number *= 100
            else:
                starting_number *= 10
            starting_number += pos
        
        #reapeat k times
        while k > 0:
            starting_number = self.sum_digits(starting_number)
            k -= 1
        
        return starting_number
    
    def sum_digits(self,num):
        ans = 0
        while num:
            ans += num % 10
            num = num // 10
        
        return ans
    
class Solution:
    def getLucky(self, s: str, k: int) -> int:
        '''
        instead of checking for multiple of 1 and 10, just grab the gitis
        '''
        starting_number = 0
        for ch in s:
            pos = (ord(ch) - ord('a')) + 1
            while pos > 0:
                starting_number = starting_number*10 + pos % 10
                pos = pos // 10
        
        #reapeat k times
        while k > 0:
            starting_number = self.sum_digits(starting_number)
            k -= 1
        
        return starting_number
    
    def sum_digits(self,num):
        ans = 0
        while num:
            ans += num % 10
            num = num // 10
        
        return ans
    
########################################################
# 1634. Add Two Polynomials Represented as Linked Lists
# 03SEP24
########################################################
# Definition for polynomial singly-linked list.
# class PolyNode:
#     def __init__(self, x=0, y=0, next=None):
#         self.coefficient = x
#         self.power = y
#         self.next = next

class Solution:
    def addPoly(self, poly1: 'PolyNode', poly2: 'PolyNode') -> 'PolyNode':
        '''
        fun linked list problem!
        they are in sorted order, so if the powers match, add upp the coeffcients and make a new node
        if the powers dont match, move the larger one and cappy the current node
        '''
        dummy = PolyNode(-1,-1)
        curr = dummy
        p1 = poly1
        p2 = poly2
        
        while p1 != None and p2 != None:
            #equal power, add coefficents
            if p1.power == p2.power:
                #drop zero coefs
                if p1.coefficient + p2.coefficient == 0:
                    p1 = p1.next
                    p2 = p2.next
                else:
                    new_node = PolyNode(p1.coefficient + p2.coefficient, p1.power)
                    curr.next = new_node
                    curr = curr.next
                    p1 = p1.next
                    p2 = p2.next
            elif p1.power > p2.power:
                curr.next = p1
                curr = curr.next
                p1 = p1.next
            elif p1.power < p2.power:
                curr.next = p2
                curr = curr.next
                p2 = p2.next

        
        if p1 == None:
            curr.next = p2
        else:
            curr.next = p1
        
        return dummy.next
                
# Definition for polynomial singly-linked list.
# class PolyNode:
#     def __init__(self, x=0, y=0, next=None):
#         self.coefficient = x
#         self.power = y
#         self.next = next

class Solution:
    def addPoly(self, poly1: 'PolyNode', poly2: 'PolyNode') -> 'PolyNode':
        '''
        if i had a hashamp that stored the powers and sum of their coefficents
        i could just go in order and build the polynomial
        '''
        mapp = defaultdict() #entry is (power,coefs)
        dummy = PolyNode(-1,-1)
        curr = dummy
        self.get_vals(poly1,mapp)
        self.get_vals(poly2,mapp)
        
        for key in sorted(mapp.keys(), reverse = True):
            curr.next = PolyNode(mapp[key],key)
            curr = curr.next
        
        return dummy.next
    
    def get_vals(self,poly,mapp):
        curr = poly
        #remember to omit coefficents with value 0
        while curr:
            curr_coef = mapp.get(curr.power,0) + curr.coefficient
            mapp[curr.power] = curr_coef
            if curr_coef == 0:
                del mapp[curr.power]
            curr = curr.next
        
########################################
# 874. Walking Robot Simulation
# 04SEP24
#########################################
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        '''
        simulate and just store the max (x,y) we get after doing the commands
        need to efficietnyl rotate curr_d
        make sure to hash obstacles
        '''
        new_obstacles = set([(x,y) for x,y in obstacles])
        #(curr_dir : [left,right])
        rotations = {
            (1,0) : [(0,1), (0,-1)],
            (-1,0) : [(0,-1), (0,1)],
            (0,1) : [(-1,0), (1,0)],
            (0,-1) : [(1,0),(-1,0)]
        }
        
        ans = float('-inf')
        curr_xy = [0,0]
        curr_r = (0,1)
        for c in commands:
            #left rotation
            if c == -2:
                curr_r = rotations[curr_r][0]
            elif c == -1:
                curr_r = rotations[curr_r][1]
            else:
                dx,dy = curr_r
                while (curr_xy[0] + dx, curr_xy[1] + dy) not in new_obstacles and c > 0:
                    c -= 1
                    curr_xy[0] += dx
                    curr_xy[1] += dy
                
                ans = max(ans, curr_xy[0]**2 + curr_xy[1]**2)
        
        return ans

#another way
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        '''
        instead of hashing directions, just think of cycle array
        north,east,south,west
        if we turn right, we just move to east, so move + 1
        if we turn left, we just look west, so + 1
        right = (+ 1 % 4)
        left = (+ 3 % 4)
        
        insteaf of hashing each (i,j) obstalcle, we can use our own hash function
        use next largest prime after largerst (i,j) cell -> 60013
        '''
        hash_mult = 60013
        new_obstacles = set([(x*hash_mult + y) for (x,y) in obstacles])
        #N,E,S,W
        dirrs = [(0,1),(1,0),(0,-1),(-1,0)]
        ans = float('-inf')
        curr_xy = [0,0]
        dir_ptr = 0
        for c in commands:
            #left rotation, + 3
            if c == -2:
                dir_ptr = (dir_ptr + 3) % 4
            elif c == -1:
                dir_ptr = (dir_ptr + 1) % 4
            else:
                dx,dy = dirrs[dir_ptr]
                for _ in range(c):
                    if ((curr_xy[0] + dx)*hash_mult + (curr_xy[1] + dy)) in new_obstacles:
                        break
                    curr_xy[0] += dx
                    curr_xy[1] += dy
                
                ans = max(ans, curr_xy[0]**2 + curr_xy[1]**2)
        
        return ans
        
#using complex numbers
from itertools import starmap

DIR = {
    -2: 1j,  # cos(90) + sin(90)i, left rotation multiply by 1j
    -1: -1j,  # cos(-90) + sin(-90)i right rotation multiply by -1j
}
class Solution:
    def robotSim(self, C: list[int], O: list[list[int]]) -> int:
        O = set(starmap(complex, O))
        #could also do
        seen = set()
        for coord in map(lambda x : complex(*x),O):
            seen.add(coord)
        #map(lambda x : func(X)) is similar to starmap
        cur_pos, cur_dir = 0 + 0j, 1j
        output = 0

        for c in C:
            if c < 0:
                cur_dir *= DIR[c]
            else:
                #walrus operator, instantiate and update
                while c > 0 and (next_pos := cur_pos + cur_dir) not in O:
                    cur_pos = next_pos
                    c -= 1

                output = max(output, self.distance(cur_pos))

        return output

    @staticmethod
    def distance(p: complex) -> int:
        x, y = int(p.real), int(p.imag)
        return x ** 2 + y ** 2

##############################################
# 2028. Find Missing Observations
# 05SEP24
###############################################
#dang it
class Solution:
    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        '''
        calculate sum to get to then intelligently get n numbers that add to needed sum
        so we have needed sum 7, with 6 numbers
        i can do [1,1,1,1,1,2] do i priortize smaller or larger first
        if i started with 6
        [6,1,] cant do it
        try using 6 and make it 
        '''
        curr_sum = sum(rolls)
        target_sum = mean*(len(rolls) + n)
        needed_sum = target_sum - curr_sum
        #check if even possible
        if needed_sum > 6*n:
            return []
        #build n numbers that get to needed_sum
        print("need_sum :", needed_sum)
        for dice in range(7,0,-1):
            print(dice, divmod(needed_sum,dice))
            num_dice,rem = divmod(needed_sum,dice)
            #can do evenly
            if num_dice == n and rem == 0:
                return [dice]*n
            #corner case
            elif num_dice == n:
                return [dice + 1] + [dice]*(n-1)
            
#trickyyyy
class Solution:
    def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
        '''
        calculate sum to get to then intelligently get n numbers that add to needed sum
        so we have needed sum 7, with 6 numbers
        i can do [1,1,1,1,1,2] do i priortize smaller or larger first
        if i started with 6
        [6,1,] cant do it
        try using 6 and make it 
        
        omg, just take the needed_sum // n, and distribute the remaidner to each of the mod elements
        '''
        curr_sum = sum(rolls)
        target_sum = mean*(len(rolls) + n)
        needed_sum = target_sum - curr_sum
        #check if even possible
        if needed_sum > 6*n or needed_sum < n:
            return []
        
        starting_die,rem = divmod(needed_sum,n)
        ans = [starting_die]*n
        for i in range(rem):
            ans[i] += 1
        
        return ans
            
###########################################################
# 3217. Delete Nodes From Linked List Present in Array
# 06SEP24
###########################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can use hashset and dummy head
        '''
        nums = set(nums)
        dummy = ListNode(-1)
        dummy_ptr = dummy
        curr = head
        
        while curr:
            if curr.val in nums:
                curr = curr.next
            else:
                dummy_ptr.next = ListNode(curr.val)
                dummy_ptr = dummy_ptr.next
                curr = curr.next
        
        return dummy.next
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can use hashset 
        and just check for next and next val
        '''
        nums = set(nums)
        
        #find first node we can include
        while head and head.val in nums:
            head = head.next
        
        if not head:
            return head
        
        curr = head
        while curr.next:
            if curr.next.val in nums:
                #making next pointer to next.next, deletes it
                curr.next = curr.next.next
            else:
                curr = curr.next
        
        return head
        
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can use hashset 
        and just check for next and next val
        must use sentinel node though
        '''
        nums = set(nums)
        dummy = ListNode(-1)
        dummy.next = head
        curr = dummy
        
        while curr.next:
            if curr.next.val in nums:
                curr.next = curr.next.next
            else:
                curr = curr.next
        
        return dummy.next
    
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can use hashset 
        and just check for next and next val
        must use sentinel node though
        '''
        nums = set(nums)
        dummy = ListNode(-1)
        dummy.next = head
        prev = dummy
        curr = head
        
        while curr:
            #delete the current node by making prev.next point to curr.next
            if curr.val in nums:
                prev.next = curr.next
                #sort of like a lagging pointer
            #otherwise just move prev to curr
            else:
                prev = curr
            curr = curr.next
        
        return dummy.next

#######################################
# 1367. Linked List in Binary Tree
# 07SEP24
#######################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        '''
        recursion,
        rec(treenode,listnode), if we get to the end of the listnode, there is a path
        we could have a partial path of head in root at an earlier depth
        but a cmoplete path later in discovery at a deeper depth
        need to two dfs functions, one to traverse the whole tree, the other to follow path
        '''
        def is_path(tree_node,list_node):
            if not list_node:
                return True
            if not tree_node:
                return False
            if tree_node.val != list_node.val:
                return False
            left = is_path(tree_node.left,list_node.next)
            right = is_path(tree_node.right,list_node.next)
            return left or right
        
        def dfs(tree_node,list_node):
            if not tree_node:
                return False
            if (tree_node.val == list_node.val):
                return is_path(tree_node,list_node)
            left = dfs(tree_node.left,list_node)
            right = dfs(tree_node.right,list_node)
            return left or right
        
        return dfs(root,head)

#i did it backwards
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        '''
        omg we need to dfs from every possible node!
        '''
        def dfs(tree_node,list_node):
            if not list_node:
                return True
            if not tree_node:
                return False
            if (tree_node.val != list_node.val):
                return False
            left = dfs(tree_node.left,list_node.next)
            right = dfs(tree_node.right,list_node.next)
            return left or right
        
        def is_path(tree_node,list_node):
            if not list_node:
                return True
            if not tree_node:
                return False
            if dfs(tree_node,list_node):
                return True
            left = is_path(tree_node.left,list_node)
            right = is_path(tree_node.right,list_node)
            return left or right
        
        return is_path(root,head)
    
#Rabin Karp, keep stack of prefix hashes
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        '''
        can either use KMP or Rabing Karp
        we are essentially tring to find a pattern (path of list node) in tree
        reduce to string matching, we can use dfs with backtracking to calculate the rolling hash at each point in the tree
        
        '''
        #powers array mod m
        base = 101
        mod = 10**9 + 7
        powers_array = [0]*2500 #pathological case, 2500 nodes going left/right
        #fill in
        powers_array[0] = 1
        for i in range(1,len(powers_array)):
            powers_array[i] = (base*powers_array[i-1]) % mod
        
        #compute pattern hash
        p_hash = 0
        p_len = 0
        curr = head
        while curr:
            p_hash = (p_hash + curr.val*powers_array[p_len]) % mod
            p_len += 1
            curr = curr.next
        
        pref_hashes = [0]
        
        return self.dfs(root,powers_array,pref_hashes,p_hash,p_len)
        
    def dfs(self,tree_node,powers,pref_hashes,p_hash,p_len):
        m = 10**9 + 7
        if not tree_node:
            return False
        
        #get curr pref hash
        curr_pref_hash = (pref_hashes[-1] + tree_node.val*powers[len(pref_hashes) - 1]) % m
        #if we have at a current path at least length pattern
        #determine if pattern hash is in the current prefix hash
        if len(pref_hashes) >= p_len:
            k = len(pref_hashes) - p_len
            #rolling hash
            curr_hash = (m + curr_pref_hash - pref_hashes[k]) % m
            ref_hash = (p_hash)*powers[k] % m
            if curr_hash == ref_hash:
                return True
        
        #add to current hashes
        pref_hashes.append(curr_pref_hash)
        left = self.dfs(tree_node.left,powers,pref_hashes,p_hash,p_len)
        right = self.dfs(tree_node.right,powers,pref_hashes,p_hash,p_len)
        ans = left or right
        pref_hashes.pop()
        return ans
            
            
##############################################
# 725. Split Linked List in Parts (REVISTED)
# 08SEP24
#############################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def splitListToParts(self, head: Optional[ListNode], k: int) -> List[Optional[ListNode]]:
        '''
        get size and partitions
        missing parts can just be the null pointer
        if there are n nodes 
        then there will be n//k nodes in each partition
        with the first n%k parts having onre more
        '''
        n = self.get_size(head)
        size, rem = divmod(n,k)
        
        ans = []
        curr = head
        #the first n % k have size + 1
        for _ in range(rem):
            ans_head = curr
            prev = None
            for _ in range(size+1):
                prev = curr
                curr = curr.next
            prev.next = None
            ans.append(ans_head)
        
        #the next parts
        while curr:
            ans_head = curr
            prev = None
            for _ in range(size):
                prev = curr
                curr = curr.next
            prev.next = None
            ans.append(ans_head)
            
        if len(ans) < k:
            return ans+[None]*(k - len(ans))
        return ans
        
            
    def get_size(self, node : ListNode) -> int:
        if not node:
            return 0
        return 1 + self.get_size(node.next)
    
####################################
# 2094. Finding 3-Digit Even Numbers
# 08SEP24
#####################################
class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        '''
        we can sort the array decreasingly
        then going right to left, fis the ones digit with an even number
        then ever number to its left, make a three digit number
        that would take too long
        check all numbers from 100 to 99
        should really be medium lol
        '''
        digits = Counter(digits)
        ans = []
        
        for num in range(100,1000,2):
            if num % 2 == 0:
                count_d = self.get_digits(num)
                diff = Counter()
                for k,v in count_d.items():
                    diff[k] = digits[k] - v
                
                if all([v >= 0 for _,v in diff.items()]):
                    ans.append(num)
        return ans
    
    
    def get_digits(self,num):
        ans = Counter()
        while num > 0:
            ans[num % 10] += 1
            num = num // 10
        
        return ans
    
#hashmap subtraction
class Solution:
    def findEvenNumbers(self, digits: List[int]) -> List[int]:
        '''
        we can sort the array decreasingly
        then going right to left, fis the ones digit with an even number
        then ever number to its left, make a three digit number
        that would take too long
        check all numbers from 100 to 99
        should really be medium lol
        '''
        digits = Counter(digits)
        ans = []
        
        for num in range(100,1000,2):
            count_d = self.get_digits(num)
            diff = count_d - digits
            #check remainder from counts digits, should only be zero
            if not diff:
                ans.append(num)
        return ans
    
    
    def get_digits(self,num):
        ans = Counter()
        while num > 0:
            ans[num % 10] += 1
            num = num // 10
        
        return ans

###################################
# 2326. Spiral Matrix IV (REVISTED)
# 09SEP24
###################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def spiralMatrix(self, m: int, n: int, head: Optional[ListNode]) -> List[List[int]]:
        '''
        instead of checking edges, rotate direction if we have an already filled cell
        
        '''
        mat = [[-1]*n for _ in range(m)]
        #East,South,West,Noth
        dirrs = [(0,1),(1,0),(0,-1),(-1,0)]
        curr_d = 0
        i,j = 0,0
        curr = head
        
        while curr:
            mat[i][j] = curr.val
            #find next cell
            ii,jj = dirrs[curr_d]
            if (0 <= i + ii < m) and (0 <= j + jj < n) and (mat[i + ii][j + jj] == -1):
                i = i + ii
                j = j + jj
            else:
                curr_d = (curr_d + 1) % 4
                ii,jj = dirrs[curr_d]
                i = i + ii
                j = j + jj
            
            curr = curr.next
        
        return mat
    
###########################################
# 2076. Process Restricted Friend Requests
# 09SEP24
###########################################
class DSU:
    def __init__(self,n):
        self.ranks = [1]*n
        self.parents = [i for i in range(n)]
        
    def find(self,x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        
        return self.parents[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)
        
        if x_par == y_par:
            return False
        #union swap
        if self.ranks[x_par] > self.ranks[y_par]:
            self.ranks[x_par] += self.ranks[y_par]
            self.ranks[y_par] = 0
            self.parents[y_par] = x_par
        else:
            self.ranks[y_par] += self.ranks[x_par]
            self.ranks[x_par] = 0
            self.parents[x_par] = y_par
        
        return True

class Solution:
    def friendRequests(self, n: int, restrictions: List[List[int]], requests: List[List[int]]) -> List[bool]:
        '''
        union find on friend connections
        say we have request (u,v)
        union in the restrictions first, then check if request is allowed
        we then need to check that for a request (u,v)
            u_par != v_par
        
        we need to check restirctions each time we do a request first!
        one of those wierd times where order matters for union find
        '''
        
        dsu = DSU(n)
        ans = []
        
        for u,v in requests:
            u_par,v_par = dsu.find(u),dsu.find(v)
            can_make = True
            for x,y in restrictions:
                x_par,y_par = dsu.find(x), dsu.find(y)
                #now checek if parents coincide
                parents1 = set([u_par,v_par])
                parents2 = set([x_par,y_par])
                if parents1 == parents2:
                    can_make = False
                    break
            
            ans.append(can_make)
            if can_make:
                dsu.union(u,v)
        
        return ans
                
#######################################################
# 2807. Insert Greatest Common Divisors in Linked List
# 10SEP24
#######################################################
class Solution:
    def insertGreatestCommonDivisors(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        same is insert into linked list
        but insert between every pair of nodes
        '''
        prev = None
        curr = head
        
        while curr.next:
            GCD = self.gcd(curr.val,curr.next.val)
            GCD_node = ListNode(GCD)
            next_ = curr.next
            curr.next = GCD_node
            GCD_node.next = next_
            curr = next_
        
        return head
    
    def gcd(self,a,b):
        if b == 0:
            return a
        return self.gcd(b,a % b)

##########################################
# 1229. Meeting Scheduler (REVISTED)
# 10SEP24
##########################################
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        '''
        no two intersecting time slots for a person!
        need earliest time slot the works for both
        sort both on start times, and two pointers, advance the smaller of the two start times
        both must be available for the duration
        '''
        slots1.sort(key = lambda x : x[0])
        slots2.sort(key = lambda x : x[0])
        
        i,j = 0,0
        
        while i < len(slots1) and j < len(slots2):
            interval_1 = slots1[i]
            interval_2 = slots2[j]
            #find common
            start,end = max(interval_1[0],interval_2[0]),min(interval_1[1],interval_2[1])
            if end - start >= duration:
                return [start,start + duration]
            #move the smaller
            if interval_1[0] < interval_2[0]:
                i += 1
            else:
                j += 1
        
        #check the ends
        while i < len(slots1):
            interval_1 = slots1[i]
            interval_2 = slots2[j-1]
            #find common
            start,end = max(interval_1[0],interval_2[0]),min(interval_1[1],interval_2[1])
            if end - start >= duration:
                return [start,start + duration]
            else:
                i += 1
        
        while j < len(slots2):
            interval_1 = slots1[i-1]
            interval_2 = slots2[j]
            #find common
            start,end = max(interval_1[0],interval_2[0]),min(interval_1[1],interval_2[1])
            if end - start >= duration:
                return [start,start + duration]
            else:
                j += 1
        return []

#######################################################
# 1953. Maximum Number of Weeks for Which You Can Work
# 10SEP24
#######################################################
#dang it
class Solution:
    def numberOfWeeks(self, milestones: List[int]) -> int:
        '''
        milestones gives list of projects, and miletsons[i] is the number of miletones for the ith project
        each week i can do one milestone, but i cannot work on the same project for two conseuctive weeks
        so the i for week1 != i for week2
        return max weeks to complete as many of the prokects! not milestones
        [3,1] = 3 -> [0,1,0]
        max number of weeks we can work
        say we have projects (u,v), idea is to weave milteons with each other
        if we have [5,4], i can do all 5 weeks if i have another project that is 4 weeks
        this is 9
        if i have [5,3], the most i can do is6
        when can i do the greatest, if the diff between the greatest and second greatest is 1
        otherwise we can't do it

        note, intution of taking top fails
        examine [100,100,100], annswer should be 300
        https://leetcode.com/problems/maximum-number-of-weeks-for-which-you-can-work/discuss/1375472/69-73-test-cases-passed-(priority_queue-HELP)
        '''
        n = len(milestones)
        if n == 1:
            return 1
        max_heap = [-m for m in milestones]
        heapq.heapify(max_heap)
        weeks = 0
        
        while len(max_heap) > 1:
            first_largest = -heapq.heappop(max_heap)
            second_largest = -max_heap[0]
            if first_largest - second_largest <= 1:
                weeks += first_largest + second_largest
                heapq.heappop(max_heap)
            else:
                #weave the smaller one as many times as we can
                weeks += (first_largest - second_largest) + second_largest
                heapq.heappop(max_heap)
                heapq.heappush(max_heap,-(first_largest - second_largest))
                
        
        return weeks
    
#heap doesn't hold
class Solution:
    def numberOfWeeks(self, milestones: List[int]) -> int:
        '''
        greedy heap, top 2 doest not work
        '''
        n = len(milestones)
        if n == 1:
            return 1
        max_heap = [-m for m in milestones]
        heapq.heapify(max_heap)
        weeks = 0
        
        while len(max_heap) > 1:
            first = -heapq.heappop(max_heap)
            second = -heapq.heappop(max_heap)
            weeks += second*2
            if (first - second) > 0:
                heapq.heappush(max_heap,-(first-second))
                
        if len(max_heap) > 0: #done of the last one
            return weeks + 1
        return weeks
    
class Solution:
    def numberOfWeeks(self, milestones: List[int]) -> int:
        '''
        we can complete all milestones, unless one project has more milestones then all others
        idea,
            if max elements is larger than the sum of the rest of the elemments, then the answer is
            2*(sum without max) + 1
                we can pick max, pick other, pick max, pick other
                which is just the rest of sum + 1
                    the plus one comes from the extra max milestone
                    its 2*rest because we are doing the rest but pick max to space it out
        
        first case, they are all the same, then we can do all of them each weeek (i.e one milestone per week)
        We can complete all milestones for other projects,
        plus same number of milestones for the largest project,
        plus one more milestone for the largest project.
        '''
        sum_ = sum(milestones)
        max_ = max(milestones)
        rest = sum_ - max_
        return min(sum_, rest*2 + 1)
    
#rewritten for easier undertanding
class Solution:
    def numberOfWeeks(self, milestones: List[int]) -> int:

        sum_ = sum(milestones)
        max_ = max(milestones)
        rest = sum_ - max_
        if max_ > rest:
            return rest*2 + 1
        return sum_
    
###########################################
# 2220. Minimum Bit Flips to Convert Number
# 11SEP24
###########################################
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        '''
        if there's a mismatch, we need to flip it
        if there isn't a mismatch leave it
        0001010
        1010010
        '''
        ans = 0
        while start > 0 and goal > 0:
            if (start % 2) != (goal % 2):
                ans += 1
            start = start >> 1
            goal = goal >> 1
        
        #we're in the leading zeros not, so it its a 1, add
        while goal > 0:
            if (goal % 2 == 1):
                ans += 1
            goal = goal >> 1
        
        while start > 0:
            if (start % 2) == 1:
                ans += 1
            start = start >> 1
        
        return ans
    
#consolidate to one
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        '''
        if there's a mismatch, we need to flip it
        if there isn't a mismatch leave it
        0001010
        1010010
        '''
        ans = 0
        while start > 0 or goal > 0:
            #could also do:
            #if (start & 1) != (goal & 1):
            if (start % 2) != (goal % 2):
                ans += 1
            start = start >> 1
            goal = goal >> 1
        
        return ans
    
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        '''
        could also do recursively
        '''
        def dp(start,goal):
            if start == 0 and goal == 0:
                return 0
            ans = 0
            if (start & 1) != (goal & 1):
                ans = 1
            
            return ans + dp(start >> 1, goal >> 1)
        
        return dp(start,goal)
    
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        '''
        XOR the numbers to get the matching mask
        1 if they dont match 0 if they do
        '''
        xor_mask = start ^ goal
        ans = 0
        while xor_mask > 0:
            ans += xor_mask & 1
            xor_mask = xor_mask >> 1
        
        return ans
    
#binar kernig hand n & n-1
#instead of moving 1 by 1
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        '''
        XOR the numbers to get the matching mask
        1 if they dont match 0 if they do
        '''
        xor_mask = start ^ goal
        ans = 0
        while xor_mask > 0:
            ans += 1
            xor_mask = xor_mask & (xor_mask - 1)
        
        return ans


#######################
# 855. Exam Room
# 12SEP24
#######################
from sortedcontainers import SortedList

class ExamRoom:

    def __init__(self, n: int):
        '''
        if we thinkg about it, its going to be left,rigth,middle,left middle, right middle
        if we keep adding, but the thing is people can leave and open up a seat
        when i add a person, i need to update the furhtest distnace
        need to binary search on left and right sides
        for any one seat, there is either a person to its left or to its right
        say we have seats [0,1,2,3,4,5]
                           X         X  
        for each seat, i would need to store the closes person to its left, and the closes person to its right
        well then i would have to check each seat every time wont work
        what if before adding a person, compute the best seat
        order will be 0,n-1, (n-1)/2, (n-1)/2/2, (n-1) - (n-1)/2
        n = 10
        [0,1,2,3,4,5,6,7,8,9]
         X
                            X
                 X
             X
                     X
        looks like brute force might work
        '''
        self.n = n
        self.rooms = SortedList([])

    def seat(self) -> int:
        if not self.rooms:
            self.rooms.add(0)
            return 0
        else:
            #find the room going left to right, we check i and i+1, then insort
            #need maximim dist
            max_dist = self.rooms[0]
            leftmost_seat = 0
            for i in range(len(self.rooms)-1):
                curr, nextt = self.rooms[i], self.rooms[i+1]
                curr_dist = (nextt - curr) // 2
                if curr_dist > max_dist:
                    max_dist = curr_dist
                    leftmost_seat = (curr + nextt) // 2
            #check if we need to add at the end
            if self.n - 1 - self.rooms[-1] > max_dist:
                self.rooms.add(self.n - 1)
                return self.n-1
            self.rooms.add(leftmost_seat)
            return leftmost_seat

    def leave(self, p: int) -> None:
        self.rooms.remove(p)


# Your ExamRoom object will be instantiated and called as such:
# obj = ExamRoom(n)
# param_1 = obj.seat()
# obj.leave(p)

#using insort works though??
import bisect
class ExamRoom:

    def __init__(self, n: int):
        self.n, self.room = n, []


    def seat(self) -> int:
        if not self.room: 
            ans = 0                           # sit at 0 if empty room 

        else:
            dist, prev, ans = self.room[0], self.room[0], 0 # set best between door and first student   

            for curr in self.room[1:]:                      # check between all pairs of students  
                d = (curr - prev)//2                        # to improve on current best

                if dist < d: 
                    dist, ans = d, (curr + prev)//2
                prev = curr

            if dist < self.n - prev-1: 
                ans = self.n - 1     # finally, check whether last seat is best

        bisect.insort(self.room, ans)                              # sit down in best seat

        return ans

    def leave(self, p: int) -> None:
        self.room.remove(p)

# Your ExamRoom object will be instantiated and called as such:
# obj = ExamRoom(n)
# param_1 = obj.seat()
# obj.leave(p)

#######################################
# 1310. XOR Queries of a Subarray
# 13SEP24
#######################################
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        '''
        prefix xor is the same as prefix sum
        say we have [a,b,c,d,e]
        if we did
        all_xor = a ^ b ^ c ^ d & e
        
        say we want (b ^ c ^ d) = (a ^ b ^ c ^ d) ^ a
        '''
        pref_xor = [0]
        for num in arr:
            pref_xor.append(pref_xor[-1] ^ num)
        
        ans = []
        for i,j in queries:
            val = pref_xor[j+1] ^ pref_xor[i]
            ans.append(val)
        
        return ans
    
#we can overwrite the array with prefxor
class Solution:
    def xorQueries(self, arr: List[int], queries: List[List[int]]) -> List[int]:
        '''
        prefix xor is the same as prefix sum
        say we have [a,b,c,d,e]
        if we did
        all_xor = a ^ b ^ c ^ d & e
        
        say we want (b ^ c ^ d) = (a ^ b ^ c ^ d) ^ a
        '''
        arr = [0] + arr
        for i in range(1,len(arr)):
            arr[i] ^= arr[i-1]
        
        ans = []
        for i,j in queries:
            val = arr[j+1] ^ arr[i]
            ans.append(val)
        
        return ans

#################################################
# 2419. Longest Subarray With Maximum Bitwise AND
# 14SEP24
#################################################
#ez, two pass
#can we do one pass?
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        '''
        bitwise AND of two different numbers will always be strictly less than the max of those two numbers
        say we have array, [3,4,5,6]
        [3,4] < [4]
        in order for the subarray to have max bitwise AND, it must include the maximum number
        count the longest streak for the maximum element!
        '''
        max_num = max(nums)
        longest_streak = 0
        curr_streak = 0
        
        for num in nums:
            if num == max_num:
                curr_streak += 1
            else:
                longest_streak = max(longest_streak,curr_streak)
                curr_streak = 0
        
        return max(curr_streak,longest_streak)

#yessss
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        '''
        we can find the max on the fly and update the streak
        '''
        max_num = 0
        longest_streak = 0
        curr_streak = 0
        
        for num in nums:
            if num > max_num:
                max_num = num
                curr_streak = 1
                longest_streak = 1
            elif num == max_num:
                curr_streak += 1
            else:
                curr_streak = 0
            
            longest_streak = max(longest_streak,curr_streak)
        
        return longest_streak

####################################################################
# 1371. Find the Longest Substring Containing Vowels in Even Counts
# 15SEP24
####################################################################
#brute force using pref_xor
class Solution:
    def findTheLongestSubstring(self, s: str) -> int:
        '''
        if i had pref_sum, i could check all (i,j) and find max meeting criteria
        that would take too long though
        i can use bit mask at each position, an even number of times could also be zero
        the bit mask should be zero if it has each vowel an even number of times
        after genering pref_xor find the longest subrray that is 0
        this is just subarray sum == k problem
        
        do brute force first
        '''
        pref_xor = [0]
        N = len(s)
        for i in range(N):
            char = s[i]
            if char in 'aeiou':
                pos = ord(char) - ord('a')
                new_mask = pref_xor[-1] ^ (1 << pos)
                pref_xor.append(new_mask)
            else:
                pref_xor.append(pref_xor[-1])
        
        ans = 0
        for i in range(N):
            #remember it can be a single element!
            for j in range(i,N):
                mask = pref_xor[j+1] ^ pref_xor[i]
                if mask == 0:
                    ans = max(j-i+1,ans)
        
        return ans
                    
#precompute pref_xor and check
class Solution:
    def findTheLongestSubstring(self, s: str) -> int:
        '''
        if i had pref_sum, i could check all (i,j) and find max meeting criteria
        that would take too long though
        i can use bit mask at each position, an even number of times could also be zero
        the bit mask should be zero if it has each vowel an even number of times
        after genering pref_xor find the longest subrray that is 0
        this is just subarray sum == k problem
        
        do brute force first
        '''
        pref_xor = [0]
        N = len(s)
        for i in range(N):
            char = s[i]
            if char in 'aeiou':
                pos = ord(char) - ord('a')
                new_mask = pref_xor[-1] ^ (1 << pos)
                pref_xor.append(new_mask)
            else:
                pref_xor.append(pref_xor[-1])
        
        ans = 0
        mapp = {}
        for i in range(len(pref_xor)):
            curr_xor = pref_xor[i]
            if curr_xor not in mapp:
                mapp[curr_xor] = i
            else:
                ans = max(ans, i - mapp[curr_xor])
        
        return ans
    
#compute on fly
class Solution:
    def findTheLongestSubstring(self, s: str) -> int:
        '''
        for one pass just keep the current mask
        subarray sum == k paradigm, seach for complement of if we've already seen this mask
        '''
        N = len(s)
        curr_mask = 0
        ans = 0
        mapp = {0:-1}
        for i in range(N):
            char = s[i]
            if char in 'aeiou':
                pos = ord(char) - ord('a')
                curr_mask = curr_mask ^ (1 << pos)
            #dont care if its not a vowel
            if curr_mask not in mapp:
                mapp[curr_mask] = i
            else:
                ans = max(ans, i - mapp[curr_mask])
        
        return ans

##########################################
# 2162. Minimum Cost to Set Cooking Time
# 16SEP24
###########################################
#ughhh, so annoying
class Solution:
    def minCostSetTime(self, startAt: int, moveCost: int, pushCost: int, targetSeconds: int) -> int:
        '''
        if we have to move from a digit, its cost moveCost
        doesn't matter where to move from and to
        how many ways are there to get targetSeconds? brute force??
        ideally we would like to keep moves and pressing to a minimum
        minutes and seconds can go up to 99, loop through numbers 0 to 9999, and get each cost
        make sure to prepend 0s, the thing is i can press 954 which will be interpreted as 0954
        or actually 0954, but it doesnt make sense to press 0954 if i can do 954 with one less press
        '''
        ans = float('inf')
        for code in range(10000):
            str_code = str(code)
            #prepend zeros
            str_code = '0'*(4-len(str_code))+str_code
            mins = str_code[0:2]
            seconds = str_code[2:4]
            if int(mins)*60 + int(seconds) == targetSeconds:
                print(str_code)
                ans = min(ans,self.cost(startAt,moveCost,pushCost,mins,seconds))
        
        return ans
    #get cost passing in string minutes and string seconds
    def cost(self,start,moveCost,pushCost,str_mins,str_secs):
        #remove leading zero now from str_mins
        str_code = str_mins + str_secs
        str_code = str_code.lstrip('0')
        #calculate cost
        curr_cost = 0
        presses = [num for num in str_code]
        if int(presses[0]) != start:
            curr_cost += moveCost
        for i in range(1,len(presses)):
            if presses[i] != presses[i-1]:
                curr_cost += moveCost
        curr_cost += len(presses)*pushCost
        return curr_cost
    
class Solution:
    def minCostSetTime(self, startAt: int, moveCost: int, pushCost: int, targetSeconds: int) -> int:
        '''
        cleaner way using format
        '''
    
        def get_cost(mm,ss):
            time = f'{mm // 10}{mm % 10}{ss // 10}{ss % 10}'
            time = time.lstrip('0')
            time = [int(ch) for ch in time ]
            cost = 0
            current = startAt
            for ch in time:
                if ch != current:
                    current = ch
                    cost += moveCost
                cost += pushCost
            
            return cost
        
        ans = float('inf')
        for mm in range(100):
            for ss in range(100):
                if mm*60 + ss == targetSeconds:
                    ans = min(ans,get_cost(mm,ss))
        
        return ans


####################################
# 1257. Smallest Common Region
# 17SEP24
####################################
class Solution:
    def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
        '''
        first entry in region is parent (i.e includes all other regions from 1 to n)
        also if region x contains y, x > y
        given two regions, find smallest region that contains both of them
        this is just LCA on an N-ary tree
        we can follow the leaves back up the tree
        need reverse graph
        then find paths, if its a tree, there will be a common ancestor in it, that's the LCA
        '''
        graph = defaultdict(list)
        for r in regions:
            parent,children = r[0], r[1:]
            for ch in children:
                graph[ch].append(parent)
        
        r1_path,r2_path = [],[]
        self.dfs(graph,region1,r1_path)
        self.dfs(graph,region2,r2_path)
        r1_path,r2_path = r1_path[::-1],r2_path[::-1]
        
        #follow path until divergence
        i,j = 0,0
        lca = None
        while i < len(r1_path) and j < len(r2_path) and r1_path[i] == r2_path[j]:
            lca = r1_path[i]
            i += 1
            j += 1
        return lca
    
    def dfs(self,graph,node,path):
        if node not in graph:
            path.append(node)
            return
        path.append(node)
        for neigh in graph[node]:
            self.dfs(graph,neigh,path)

class Solution:
    def findSmallestRegion(self, regions: List[List[str]], region1: str, region2: str) -> str:
        '''
        can also do iteratively, just follow parent pointers ups but we need child to parent pointer relationship
        we follow up as far as we can, and keep track of visited parents
        if we see a previsoly seen parent, thats our lca
        if we can't find a parent, swap regions
        '''
        graph = {}
        for r in regions:
            parent,children = r[0], r[1:]
            for ch in children:
                graph[ch] = parent
        
        prev_parents = set()
        curr_region = region1
        while curr_region:
            if curr_region in prev_parents:
                return curr_region
            prev_parents.add(curr_region)
            if curr_region in graph:
                curr_region = graph[curr_region]
            else:
                curr_region = region2