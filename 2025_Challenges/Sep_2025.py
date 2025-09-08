############################################
# 1792. Maximum Average Pass Ratio (REVISTED)
# 01SEP25
############################################
class Solution:
    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        '''
        need to assign extraStudents somehwere so tha we can maximuze the average pass ration
        ration is pass/total
        when we assign 1 student, its gain is (pass + 1) / (total + 1) need delta
        
        max_heap with on gain, and add student to largest gain
        answer is just top of heap 
        '''

        max_heap = []
        for p,t in classes:
            delta = (((p+1) / (t+1))) - (p/t)
            entry = (-delta,p,t)
            max_heap.append(entry)
        
        heapq.heapify(max_heap)
        
        for k in range(extraStudents):
            curr_delta, p, t = heapq.heappop(max_heap)
            p += 1
            t += 1
            new_delta = (((p+1) / (t+1))) - (p/t)
            entry = (-new_delta, p,t)
            heapq.heappush(max_heap, entry)
        
        sum_ratios = 0
        for r,p,t in max_heap:
            sum_ratios += (p/t)
        
        return sum_ratios / len(max_heap)


#######################################################################
# 3584. Maximum Product of First and Last Elements of a Subsequence
# 01SEP25
#####################################################################
class Solution:
    def maximumProduct(self, nums: List[int], m: int) -> int:
        '''
        two pointers
        if take at nums[i], then we can take the last numbers from nums[i+m-1] to nums[n-1]
        the middle m-2 elements can be any
        for any pair (i,j), such that j - i + 1 >= m
            pick nums[i] as first
            pick nums[j] as last
        '''
        MAX = float('-inf')
        MIN = float('inf')
        ans = float('-inf')
        n = len(nums)
        for i in range(m-1,n):
            MAX = max(MAX,nums[i - m + 1])
            MIN = min(MIN,nums[i - m + 1])
            ans = max(ans, nums[i]*MIN, nums[i]*MAX)
        
        return ans


######################################################
# 3025. Find the Number of Ways to Place People I
# 02SEP25
######################################################
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        '''
        need to make a box with two points (a,b) such that the box has no other points in it (even on the borders)
        points could be a line too
        a needs to be on the upper left side of b
        '''
        count = 0
        n = len(points)

        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]

                # determine UL and BR
                if x1 <= x2 and y1 >= y2:
                    ul, br = (x1, y1), (x2, y2)
                elif x2 <= x1 and y2 >= y1:
                    ul, br = (x2, y2), (x1, y1)
                else:
                    continue  # not a valid UL-BR pair

                # check for any other point inside rectangle
                is_valid = True
                for k, (x, y) in enumerate(points):
                    if k in (i, j):
                        continue
                    if ul[0] <= x <= br[0] and br[1] <= y <= ul[1]:
                        is_valid = False
                        break

                if is_valid:
                    # print(ul, br)  # debug
                    count += 1

        return count
    
###################################################
# 3027. Find the Number of Ways to Place People II
# 04SEP25
###################################################
class Solution:
    def numberOfPairs(self, points: List[List[int]]) -> int:
        '''
        alice must be upper left corner
        bob must be lower right corner
        need to count number of (alice, bob) pairs of points such that they form a rectangle
        and no other points are in there
        we obvie cant fix ul and br corners to check
        so we need an effcient way to check if there are any points in between
        we can sort along x and along y and check for intersection
        track max y
        double sort and count
        '''
        #sort increasing on x, and decreasing on way
        #since we're sorted in increasing x, we know that next i+1, is bigger, so x is satified
        #now check
        n = len(points)
        points.sort(key = lambda x: (x[0],-x[1]))
        print(points)
        count = 0
        #this part is the same still
        for i in range(n):
            max_y = float('-inf')
            x1,y1 = points[i]
            #we know x2 is >= x1, so that's satisfied
            for j in range(i+1, n):
                x2,y2 = points[j]
                #y2 must be between the current max_y and y1
                #uptdate max_y to the current y2, since we sorted decreasling
                if max_y < y2 <= y1:
                    #valid pair with the current i
                    count += 1
                    max_y = y2

        return count
    
########################################
# 3516. Find Closest Person
# 04SEP25
########################################
class Solution:
    def findClosest(self, x: int, y: int, z: int) -> int:
        '''
        x and y move to z, we're on a number line
        
        '''
        if abs(x - z) < abs(y-z):
            return 1
        elif abs(x - z) > abs(y-z):
            return 2
        
        return 0
    
################################################################
# 2749. Minimum Operations to Make the Integer Zero (REVISTED)
# 06SEP25
################################################################
class Solution:
    def makeTheIntegerZero(self, num1: int, num2: int) -> int:
        '''
        hints
        1. if we want to make n == 0 by using only pwoers of 2 from n, we need at least the number of bits in binary rep of 2 and at most -n
        2. if its possible to make num1 == 0, then we need at most 60 opeations
        
        i.e if i can only subtract powers of 2, the mininum number of operations is just the number of set bit in num1
        the issue is that we are subtracting a power of 2 + num2
        say we have 3 and -2
        smallest value we can subtract depends on the sign
        if i == 0, we just subtract num2, but if we subtract a negative we move away
        
        if num2 is positive then we are always subtracting a postivie number
        if num2 is negative, then it depends when 2**i > abs(num2), for use to subtract
        
        get diff between steps
        (num1 - (2**(i+1) + num2)) - (num1 - (2**i + num2)) = diff
        num1 -(2**(i+1)) - num2 - num1 + 2**i + num2 = diff
            -(2**(i+1)) + 2**i = diff
            diff = -2**i

        think of it like this: if use steps in total we really are doing the subtraction part is:
        (num2 + 2^i1) + (num2 + 2^i2) ... + (num2 +2^ik)
        so we want num1 - (steps*num2 + sum(2^(i for each i up to k))) = 0
        we can rearrange to get
        num1 - steps*num2 = sum(2^(i for each i up to k))
        really is sum of powers of 2 used, this is just a binary rep of some number
        this means that:
            diff = num1 - steps*num2
            the diff must be represented as the smallest power of two!

        ie.
            Any integer diff can be written in binary.
            The minimum number of powers of two needed is the number of set bits (countBits(diff)).
            The maximum number of powers of two you can use is diff itself (worst case, use 1 a bunch of times).
            so condition is
            countBits(diff) <= steps <= diff
        
        for the sum of powers of two, i could use 2^1 + 2^1 + 2^1 = 6, using three steps
        or i could hae used 2^2 + 2^1 = 6 for two steps, 
        it's not worth repeating an ith power multiple times if i can get to another power of two in smaller steps
        '''
        #can't do it
        if num1 < num2:
            return -1
        #couting set bits
        def countBits(num):
            count = 0
            while num > 0:
                count += num & 1
                num >>= 1
            return count
        
        for steps in range(61):
            #this is diff using steps only
            diff = num1 - num2 * steps
            bits = countBits(diff)
            #need to make bits less then actual steps, and steps <= bits
            if bits <= steps <= diff:
                return steps

        return -1

#########################################################
# 3495. Minimum Operations to Make Array Elements Zero
# 06SEP25
##########################################################
#couldn't get it....
#sadness 
import math
class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        '''
        each query is range from numers [l....r]
        in one operation we can pick two integers in that range and replace them with floor(a/4) and floor(b/4)
        return min operations to reduce each query range to the zeros vector
        hint, for a number x, the number of floor operations to change to zero is floor(log4(x) + 1)
        always pair 2 numbers with maximuim /4 operations needed
        i cant expand the queries, because that take too long
        given this:
        test = [2,3,4,5,6]
        steps arrays is 
        [1,1,2,2,2]
        now this is pick two numbers (a,b) and reduce by min(a,b)
        keep going until all are zero
        '''
        ans = 0

        for l,r in queries:
            steps_l = math.floor(math.log(l,4)) + 1
            steps_r = math.floor(math.log(r,4)) + 1
            print(steps_l,steps_r)
            
        return ans

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        '''
        to reduce some number by zero we need to use log4 + 1 divisions
        [1,3] -> 1 division
        [4,15] -> 2 division
        [16,63] -> 3 visions
        code the check
        for num in range(1,64):
            #count divisions
            temp = num
            divisions = 0
            while temp > 0:
                divisions += 1
                temp = temp >> 2
            print(num,divisions)
        digit dp paradigm

        '''
        ans = 0
        for l,r in queries:
            divisions = 0
            prev = 1
            #Then, we slide the interval (prev = cur) --> d = 1 : [1, 3] -> d = 2 : [4, 15] -> d = 3 : [16, 63]...
            #check overlap in range
            for divisions in range(1,17):
                curr = prev*4
                start = max(l,prev)
                end = min(r,curr - 1)
                #complete overlap
                if end >= start:
                    divisions += (end - start + 1)*divisions
                #slide range
                prev = curr
            #ceiling would also work here too
            ans += (divisions + 1) // 2
        
        return ans

#################################################
# 3032. Count Numbers With Unique Digits II
# 08SEP25
#################################################
class Solution:
    def numberCount(self, a: int, b: int) -> int:
        '''
        brute force with hashset
        digit dip
        count nums that have unique digits less then some num, call it dp(num)
        we want
        dp(b) - do(a)
        '''
        ans = 0
        for num in range(a,b+1):
            digits = set(str(num))
            if len(digits) == len(str(num)):
                ans += 1
        return ans