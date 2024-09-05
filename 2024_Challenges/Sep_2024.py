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
            