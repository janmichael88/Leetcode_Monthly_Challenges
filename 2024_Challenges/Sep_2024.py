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

#check up to maxmins and add in remaining seconds
class Solution:
    def minCostSetTime(self, startAt: int, moveCost: int, pushCost: int, targetSeconds: int) -> int:
        '''
        we actually don't need to check all minutes and seconds
        max possible minutes is targetSeconds / 60
        '''
        def cost(mins,secs):
            presses = str(mins*100 + secs)
            curr = str(startAt)
            cost = 0
            for ch in presses:
                if ch == curr:
                    cost += pushCost
                else:
                    cost += (moveCost + pushCost)
                    curr = ch
            
            return cost
        
        ans = float('inf')
        max_mins = targetSeconds // 60
        for mins in range(max_mins + 1):
            secs = targetSeconds - mins*60
            #out of  bounds
            if secs > 99 or mins > 99:
                continue
            ans = min(ans, cost(mins,secs))
        
        return ans
    
class Solution:
    def minCostSetTime(self, startAt: int, moveCost: int, pushCost: int, targetSeconds: int) -> int:
        '''
        turns out we really only have two choices
        maxmins,secs
        maxmins - 1, secs + 60
        etierh press whats there if its a valid config between 0000 and 9999
        or try one less minute but carry over seconds
        '''
        def cost(mins,secs):
            if mins > 99 or secs > 99 or mins < 0 or secs < 0:
                return float('inf')
            presses = str(mins*100 + secs)
            curr = str(startAt)
            cost = 0
            for ch in presses:
                if ch == curr:
                    cost += pushCost
                else:
                    cost += (moveCost + pushCost)
                    curr = ch
            
            return cost
        
        mins,secs = divmod(targetSeconds,60)
        return min(cost(mins,secs), cost(mins-1,secs + 60))

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

################################################
# 2165. Smallest Value of the Rearranged Number
# 18SEP24
#################################################
class Solution:
    def smallestNumber(self, num: int) -> int:
        '''
        largest number will be 10**15
        will have at most 16 digits
        there will be 16! ways to make a new digit
        thats too big to try all
        need to start the number with the largest number that isn't zero
        '''
        if num == 0:
            return 0
        digits = []
        is_neg = True if num < 0 else False
        num = abs(num)
        while num > 0:
            digits.append(num % 10)
            num = num // 10
        
        if is_neg:
            digits.sort(reverse = True)
        else:
            digits.sort()
        
        #find largest non zero
        i = 0
        while i < len(digits) and digits[i] == 0:
            i += 1
        
        digits[0],digits[i] = digits[i],digits[0]
        ans = 0
        for num in digits:
            ans *= 10
            ans += num
        
        return ans*(-1 if is_neg else 1)
    
##################################
# 179. Largest Number (REVISTED)
# 18SEP24
##################################
from functools import cmp_to_key
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        '''
        what if i sort lexographically?
        convert each int to a string then sort?
        say we have like [3,31], the answer should be 331
        but if i sort just lexo i get [31,3], i need 3 to come first
        91
        98
        98
        9
        9
        need to compare the concatnettion of two numbers
        say we are are comparing 3 and 30, do we cant 330 or 303?
        we want 330
        '''
        def largest(a,b):
            if a + b > b + a:
                return -1
            elif a + b < b + a:
                return 1
            else:
                return 0
            
        nums = [str(n) for n in nums]
        nums.sort(key = cmp_to_key(largest))
        if nums[0] == '0':
            return '0'
        
        return "".join(nums)
    
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        '''
        good review on merge sort
        '''
        nums = self._merge_sort(nums, 0, len(nums) - 1)
        nums = [str(n) for n in nums]
        if nums[0] == '0':
            return '0'
        
        return "".join(nums)
    
    def _merge_sort(self,arr,left,right):
        if left >= right:
            return [arr[left]]
        
        mid = left + (right - left) // 2
        left_sorted = self._merge_sort(arr,left,mid)
        right_sorted = self._merge_sort(arr,mid+1,right)
        return self._merge(left_sorted,right_sorted)
    
    def _merge(self, left,right):
        sorted_halves = []
        i,j = 0,0
        
        while i < len(left) and j < len(right):
            if self._compare(left[i],right[j]):
                sorted_halves.append(left[i])
                i += 1
            else:
                sorted_halves.append(right[j])
                j += 1
        
        #add in the rest
        sorted_halves.extend(left[i:])
        sorted_halves.extend(right[j:])
        return sorted_halves
                
    
    def _compare(self, a,b):
        return str(a)+str(b) > str(b)+str(a)
            
#######################################################
# 241. Different Ways to Add Parentheses (REVISTED)
# 19SEP24
#######################################################
#recursion, but pass strings
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        '''
        well it doesn't make sense to just do this (2), don't self paranthesize, its still the same number
        this is matric chain multiplication [left,right], try everything in between left and right
        at each operator we split into left and right parts and see if we can make a result by adding the 
        results of the operations
        if empty string, return []
        if one, return numbers as list
        if two, and the first digit is number return the number as list
        its of the form dp(i,j) = some opertion on all k between (i,j)
            rather dp(i,j) = {
            for k in range(i,j):
                left_side = dp(i,k-1)
                right_side = dp(k+1,j)
                
            and then some operation on left and right
            }
        '''
        
        def rec(string):
            if not string:
                return []
            if len(string) == 1:
                return [int(string)]
            
            if len(string) == 2 and string[0].isdigit():
                return [int(string)]
            
            ans = []
            for i,ch in enumerate(string):
                if ch.isdigit():
                    continue
                #split left and right
                left = rec(string[:i])
                right = rec(string[i+1:])
                for l in left:
                    for r in right:
                        if ch == '+':
                            ans.append(l + r)
                        elif ch == '-':
                            ans.append(l - r)
                        elif ch == '*':
                            ans.append(l*r)
            
            return ans
        
        return rec(expression)
                            

#dp
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        '''
        well it doesn't make sense to just do this (2), don't self paranthesize, its still the same number
        this is matric chain multiplication [left,right], try everything in between left and right
        at each operator we split into left and right parts and see if we can make a result by adding the 
        results of the operations
        if empty string, return []
        if one, return numbers as list
        if two, and the first digit is number return the number as list
        its of the form dp(i,j) = some opertion on all k between (i,j)
            rather dp(i,j) = {
            for k in range(i,j):
                left_side = dp(i,k-1)
                right_side = dp(k+1,j)
                
            and then some operation on left and right
            }
        '''
        memo = {}
        
        def dp(i,j):
            if i >= j:
                return [int(expression[i])]
            
            if j - i == 1 and expression[i].isdigit():
                return [int(expression[i:j+1])]
            
            if (i,j) in memo:
                return memo[(i,j)]
            ans = []
            
            for k in range(i,j+1):
                if expression[k].isdigit():
                    continue
                ch = expression[k]
                #split left and right
                left = dp(i,k-1)
                right = dp(k+1,j)
                for l in left:
                    for r in right:
                        if ch == '+':
                            ans.append(l + r)
                        elif ch == '-':
                            ans.append(l - r)
                        elif ch == '*':
                            ans.append(l*r)
            memo[(i,j)] = ans
            return ans
        
        
        return dp(0,len(expression) - 1)
                            
#############################################################
# 1886. Determine Whether Matrix Can Be Obtained By Rotation
# 19SEP24
#############################################################
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        '''
        we just need to check 0,1,2,and 3 rotations of mat
        how can we rotate a matrix 90?
        for some (i,j) -> (j,N-1-i)
        '''
        for _ in range(4):
            mat = self.rotate_90(mat)
            if mat == target:
                return True
        
        return False
    
    def rotate_90(self,mat):
        n = len(mat)
        rotated_mat = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                rotated_mat[j][n-1-i] = mat[i][j]
        
        return rotated_mat
        
#using list zip hack
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        '''
        another way of seeinf it is that the rows become the reverse of the columns
        '''
        for _ in range(4):
            mat = self.rotate_90(mat)
            if mat == target:
                return True
        
        return False
    
    def rotate_90(self,mat):
        return [list(reversed(col)) for col in zip(*mat)]
        
#its just cols but in reverse each time
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        '''
        another way of seeinf it is that the rows become the reverse of the columns
        '''
        for _ in range(4):
            mat = self.rotate_90(mat)
            if mat == target:
                return True
        
        return False
    
    def rotate_90(self,mat):
        n = len(mat)
        rotated_mat = []
        for r in range(n):
            col = []
            for c in range(n):
                col.append(mat[c][r])
            
            rotated_mat.append(col[::-1])
        
        return rotated_mat
        
###########################################################
# 2232. Minimize Result by Adding Parentheses to Expression
# 19SEP24
############################################################
class Solution:
    def minimizeResult(self, expression: str) -> str:
        '''
        we can only add a pair of parenthese, the opening must be on the left
        and the closing must be on thr right
        find the indices where i can place ( and find indices where i can place  )
        '''
        opening = []
        closing = []
        found_op = False
        i = 0
        n = len(expression)
        while i < n:
            if expression[i].isdigit():
                if found_op:
                    closing.append(i)
                else:
                    opening.append(i)
                i += 1
            elif expression[i] == '+':
                found_op = True
                i += 1
        
        ans = expression
        min_val = eval(expression)
        for o in opening:
            for c in closing:
                subset = expression[:o]+"*1*"+'('+expression[o:c+1]+')'+"*1*"+expression[c+1:]
                subset = subset.lstrip('*')
                subset = subset.rstrip('*')
                if eval(subset) <= min_val:
                    ans = expression[:o]+'('+expression[o:c+1]+')'+expression[c+1:]
                    min_val = eval(subset)
        
        return ans

########################################
# 214. Shortest Palindrome (REVISTED)
# 20SEP24
########################################
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        '''
        i can easily make a palindrom by just adding the reverse of the string to the end
        in this case i can add the reverse to ther front
        cat -> taccat
        the problem is that the string could 'almost' be a palindrome
        notice that if we just do rev(s)+s, the answer should exist in the concatenation!
        largest_palin = s[::-1]+s
        print(largest_palin)
        test = "aaacecaaa"
        ind = largest_palin.find(test)
        print(largest_palin[ind:ind+len(test)])
        
        we can solve, find smallest palindrom in rev(s)+s, expanding from centers would take O(n^2) time
        need to reframe the problem is findin the longest palindromic substring starting from 0
        once we have found the longest palindromi substring (lps), we take the remaning part, reverse and append it to the oriiginal string
        solve brute force first
        '''
        #find longest lps, so start from end
        n = len(s)
        for i in range(n,-1,-1):
            pref = s[:i]
            if pref == pref[::-1]:
                remaining_part = s[i:]
                return remaining_part[::-1] + s
        
        return ""

#since we are findng prefixes, we can using rolling hashes, rabin karp
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        '''
        for rabin karp, we need to keep two hashes, 
        one for prefix, and one for suffix
        if they match, we know that this is the longest
        keep prefix hashes and suffix hashes
        can i get reverse pref from suffix hashes
        so we have some pref (i,j)
        its reverse is (j,i)
        i need to reverse s first, then i can build it up
        abcd
        dcba
        
        we can update hash in revesre!
        forward hash = (forwardHash * hashBase + (currentChar - 'a' + 1)) % modValue
        reverse haseh = (reverseHash + (currentChar - 'a' + 1) * powerValue) % modValue
        
        '''
        base = 29
        mod = 10**9 + 7
        pref_hashes = [0]
        rev_hashes = [0]
        power = 1
        n = len(s)
        
        #first precopmute pref hashses and suffix hashes
        for i in range(n):
            #prefix_hash
            forward_char = s[i]
            forward_hash = (pref_hashes[-1]*base + (ord(forward_char) - ord('a') + 1)) % mod
            pref_hashes.append(forward_hash)
            #suffix hash
            rev_char = s[i]
            rev_hash = rev_hashes[-1] + ((ord(rev_char) - ord('a') + 1)*power) % mod
            rev_hashes.append(rev_hash % mod)
            power = (power*base) % mod

            
        for i in range(n,-1,-1):
            if pref_hashes[i] == rev_hashes[i]:
                return s[i:][::-1] + s
        
        return -1
    
######################################################
# 3043. Find the Length of the Longest Common Prefix
# 21SEP24
#######################################################
class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        '''
        finding all prefixes is allowable
        '''
        prefixes1 = set()
        
        for num in arr1:
            for p in self.generate_prefixes(num):
                prefixes1.add(p)
        
        ans = 0
        for num in arr2:
            for p in self.generate_prefixes(num):
                if p in prefixes1:
                    ans = max(ans, len(str(p)))
        
        return ans
                    
    def generate_prefixes(self,num):
        if not num:
            return []
        
        return self.generate_prefixes(num // 10) + [num]
    
#trie solution
class TrieNode:
    def __init__(self):
        self.children = [None]*10
        
class Trie:
    def __init__(self,):
        self.root = TrieNode()
    
    def insert(self,num):
        node = self.root
        for d in str(num):
            idx = int(d)
            if not node.children[idx]:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        
    def find_longest(self,num):
        node = self.root
        length = 0
        
        for d in str(num):
            idx = int(d)
            if not node.children[idx]:
                break
            else:
                length += 1
                node = node.children[idx]
        
        return length

class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        #make trie
        trie = Trie()
        for num in arr1:
            trie.insert(num)
        
        ans = 0
        for num in arr2:
            ans = max(ans, trie.find_longest(num))
        
        return ans
        
###########################################
# 386. Lexicographical Numbers (REVISTED)
# 22SEP24
##########################################
class Solution:
    def lexicalOrder(self, n: int) -> List[int]:
        '''
        dfs, start with zero
        '''
        ans = []
        def dfs(i):
            if i > n:
                return
            ans.append(i)
            for j in range(10):
                if i*10 + j <= n:
                    dfs(i*10 + j)
        
        for i in range(1,10):
            dfs(i)
        
        return ans

##############################################
# 440. K-th Smallest in Lexicographical Order
# 23SEP24
#############################################
#efficient counting
class Solution:
    def findKthNumber(self, n: int, k: int) -> int:
        '''
        finally they set it up for us where we do the prereq proble before
        binary search on tree
        if n changes, does kth change?
            yes, but after a certain number
        for (n,k) -> (13,3)
            we get 11
        but for (130,3) we get 100
        
        the idea is to effectively traverse a 10-ary tree
        1 can have children [10,11,12,13...19]
        10 can have children [100,101...109]
        this is prefix tree
        intuition: we need to figure out how many numbers exists in the subtree rooted at some node
        count the number of nodes in curr and curr + 1
        the kth number is not in this subtree of the number (or number of steps) is <= k
            move to its next subling and subtract the nuber of stpes from k
        
        if the number of steps is larger than k, then it must be in this subtree
            move down, mulitpely by 10 (the next level)
            decrease k by 1 ebcaue we've taken on step closer into the tree
        
        count how many numbers are between some pref and pref + 1
        multiply by 10 to dive deeper into this root
        increment steps by Math.min(n + 1, prefix2) - prefix1, 
        we need to capp pref2 and n + 1 if pref2 is bigger than n

        deep dive:
            if steps <= k, we know we can move to curr + 1, and narrow down to k- steps
                we skipped all the numbers in this node, so its k- steps
            if steps > k, its beloww in the pre order travesal, so we cant jump to curr + 1
            we need the enxt predix which is just pref*10, then we ise 1-step to go down

        _count function
        if pref2 <= n, it means that pref1 right most node exsists, so we can add the number of nodes from n1 to n2
        if pref2 > n, it means n is on the path between n1 and n2, so add (n+1 - pref1)
        If pref2 steps is at least n, then i just need the number of steps from pref2 to pref1. If pref2 steps is more than n then i need the full n+1 steps less than the steps in root pref1
        '''
        #start with first
        curr = 1
        k -= 1 #we already havey at least one umber
        while k > 0:
            print(curr)
            steps = self._count(n,curr,curr + 1)
            if steps <= k:
                curr += 1
                k -= steps
            else:
                curr *= 10
                k -= 1
        
        return curr
    
    def _count(self, n, pref1, pref2):
        steps = 0
        while pref1 <= n:
            steps += min(n+1,pref2) - pref1
            #scale by 10
            pref1 *= 10
            pref2 *= 10
        
        return steps

################################################
# 2707. Extra Characters in a String (REVISTED)
# 23SEP24
################################################
class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        '''
        need to break s, so that substrings (the original string is also a substring) are in dictionary
        let dp(i) be the min extra character if breaking s[0:i] optimally
        '''
        dictionary = set(dictionary)
        memo = {}
        n = len(s)
        
        def dp(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            curr_word = ""
            #dont break here
            ans = 1 + dp(i+1)
            for j in range(i,n):
                curr_word += s[j]
                #if word is in there its a valid break
                if curr_word in dictionary:
                    #break_here = dp(j+1)
                    ans = min(ans,dp(j+1))
            memo[i] = ans
            return ans
        
        return dp(0)
                
#trie solution
class TrieNode:
    def __init__(self,):
        self.children = {}
        self.is_word = False
        
class Trie:
    def __init__(self,):
        self.root = TrieNode()
    
    def insert(self,word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        
        node.is_word = False

class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        
        trie = Trie()
        for word in dictionary:
            trie.insert(word)
        root = trie.root
        
        memo = {}
        n = len(s)
        
        def dp(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            #dont break here
            ans = 1 + dp(i+1)
            node = root
            for j in range(i,n):
                if s[j] not in node.children:
                    break
                node = node.children[s[j]]
                #if word is in there its a valid break
                if node.is_word:
                    #break_here = dp(j+1)
                    ans = min(ans,dp(j+1))
            memo[i] = ans
            return ans
        
        return dp(0)

###########################################
# 1130. Minimum Cost Tree From Leaf Values
# 23SEP24
###########################################
class Solution:
    def mctFromLeafValues(self, arr: List[int]) -> int:
        '''
        i hade to use the the hint though
        dp(i,j) variant
        try all k in between (i,j)
        '''
        memo = {}
        
        def dp(i,j):
            if i >= j:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = float('inf')
            for k in range(i,j):
                left_max = max(arr[i:k+1])
                right_max = max(arr[k+1:j+1])
                left = dp(i,k)
                right = dp(k+1,j)
                ans = min(ans, left_max*right_max + left + right)
            
            memo[(i,j)] = ans
            return ans
        
        return dp(0,len(arr)-1)

#####################################################
# 3043. Find the Length of the Longest Common Prefix
# 24SEP24
#####################################################
class TrieNode:
    def __init__(self,):
        self.children = {}
        self.is_word = False
        
class Trie:
    def __init__(self,):
        self.root = TrieNode()
    
    def add_word(self,word):
        root = self.root
        for ch in word:
            if ch not in root.children:
                root.children[ch] = TrieNode()
            root = root.children[ch]
        
        root.is_word = True
    
    def add_words(self,words):
        for w in words:
            self.add_word(w)

class Solution:
    def longestWord(self, words: List[str]) -> str:
        '''
        trie
        if a word had all the prefixes, then there would only be one path
        a word itself can also be a prefix
        checking each prefix in each word would take too long, even after making three
        i can make trie, and use dfs to find the longes path making sure that each node is a word
        i need global answer
        '''
        #make trie
        trie = Trie()
        trie.add_words(words)
        root = trie.root
        self.ans = ""
        
        #dfs for longest path, we dont want length, need the actual word
        #must be in lexographical order
        def dfs(node,path):
            #leaf is has no children
            if not node.children:
                return path
            ans = "" if not path else path
            for neigh in range(26):
                neigh_char = chr(ord('a') + neigh)
                if neigh_char in node.children and node.children[neigh_char].is_word:
                    child_ans = dfs(node.children[neigh_char],path + neigh_char)
                    if len(child_ans) > len(ans):
                        ans = child_ans
            
            return ans
        
        return dfs(root,"")
                    