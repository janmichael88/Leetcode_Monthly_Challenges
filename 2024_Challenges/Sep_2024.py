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
        
        '''
        sum_ = sum(milestones)
        max_ = max(milestones)
        rest = sum_ - max_
        return min(sum_, rest*2 + 1)
    
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