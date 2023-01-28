######################################
# 1056. Confusing Number
# 01JAN23
######################################
class Solution:
    def confusingNumber(self, n: int) -> bool:
        '''
        confusing number is a number that when rotated becomes a different number with each digit valid
        
        first check that each digit is valid
        '''
        valid = {0:0,1:1,6:9,9:6,8:8}
        
        copy_n = n
        
        #first check all valid flippable numbers
        while copy_n:
            if copy_n % 10 not in valid:
                return False
            copy_n //= 10
        
        
        #build new number
        digits = []
        copy_n = n
        
        while copy_n:
            digits.append(valid[copy_n % 10])
            copy_n //= 10
        
        new_number = 0
        for num in digits:
            new_number *= 10
            new_number += num
        
        
        return new_number != n

class Solution:
    def confusingNumber(self, n: int) -> bool:
        '''
        confusing number is a number that when rotated becomes a different number with each digit valid
        
        we actually don't need to check, we can just terminate early
        '''
        valid = {0:0,1:1,6:9,9:6,8:8}
        
        copy_n = n
        new_number = 0
        
        #first check all valid flippable numbers
        while copy_n:
            if copy_n % 10 not in valid:
                return False
            #multipley first then add
            new_number = new_number*10 + valid[copy_n % 10]
            copy_n //= 10
        

        
        return new_number != n


###################################
# 818. Race Car
# 27DEC22
###################################
#not close at all in thiking....
class Solution:
    def racecar(self, target: int) -> int:
        '''
        car starts at position 0 and speed +1
        we can either A (acelerate) or R (reverse)
        
        when A
            position += speed
            speed *= 2
        
        when R:
            if speed is positive
                speed = -
            else:
                speed = 1
            
            position stays the same
            
        example
            pos = 0
            speed = 1
            
            first instruction:
            A
            
            pos = 1
            speed = 2
            
            A
            
            pos = 3
            speed = 4
            
            R
            
            pos = 3
            speed = - 1
            
        given target position, return length of shortest sequecne to get there
        
        this is dp
        the transitions are already there
        
        if dp(pos,speed) represents the smallest sequence of A and R to get to pos
        then dp(pos,speed) =  1 + min ({
            if previous was an A:
                dp(pos - speed//2, speed//2)
            if previous was an R
                dp(pos,1)
        })
        '''


#dijkstra's SSP
class Solution:
    def racecar(self, target: int) -> int:
        '''
        framework
        
        denote: A^k, be AAAA... some number of times (k times to be exact)
        sequence should not start with an R or end with an R
        rather it would be A^{k1 times}R A^{k2 times}.....
        
        final positions would be sum of series :
            \sum_{k=1}^{k=n} = (2*k -1)
            because our speed adoubles, and we have to go back again because of the R, speed becommes -1
            
        wlog, we can say the (k_i,i odd) is monotone decreasing and (k_i,i even is also monton decreasing)
        
        key claim:
            k_i is bounded by a + 1, where a is the smallest integer such that 2**a >= traget
            or a >= log_2(target), just the sequences of AAAAA.... (basically A any number of times)
            
        we can use dijkstra's ssp
        
        with some target we have different moves we can perform such as (k1 = 0,1,2...)
        
        we also have an uppder bound which we cannnot cross
            i.e just a series of A we would have reacred taraget in log(target) steps, or at least shoot past it
            
        from our current step, we could walk:
            walks = 2**k - 1 steps for a cost of (k+1)
        '''
        K = target.bit_length() + 1
        barrier = 1 << K
        
        pq = [(0,target)]
        distances = [float('inf')]*(2*barrier + 1)
        distances[target] = 0
        
        while pq:
            steps, node = heapq.heappop(pq)
            if distances[node] > steps:
                continue
            for k in range(K+1):
                walk = (1 << k) - 1 #cost in steps
                steps2 = steps + k + 1
                next_node = walk - node
                
                #no R command if already exact
                if walk == node:
                    steps2 -= 1
                    
                if abs(next_node) <= barrier and steps2 < distances[next_node]:
                    heapq.heappush(pq,(steps2,next_node))
                    distances[next_node] = steps2
        
        return distances[0]


#using deqsolution, and maintaing (pos,speed,and, size)
from collections import deque
class Solution(object):
    def racecar(self, target):
        q = deque()
        #starting (position,speed,length_of_commands)
        q.append((0, 1, 0))
        while q:
            pos, speed, n = q.popleft()
            if pos == target:
                return n
            #go forward
            q.append((pos+speed, speed*2, n+1))
            #if we have gone past target but have positive speed
            #we need an R command
            if speed > 0:
                if pos + speed > target:
                    q.append((pos, -1, n+1))
            #negative speed
            else:
                #and still not at target
                if pos + speed < target:
                    q.append((pos, 1, n + 1))

#another way to use bfs, just build up paths and explore the recursion tree
#https://leetcode.com/problems/race-car/discuss/124326/Summary-of-the-BFS-and-DP-solutions-with-intuitive-explanation
class Solution:
    def racecar(self, T: int) -> int:
        q = [[0,1,0,[]]]
        #to the bfs soluitons we can save repated states
        #if we have seen this state do nothing
        visited = set()
        while q:
            pos,speed,step,op = q.pop(0)
            if pos==T:
                print (op)
                return step
            if (pos,speed) not in visited:
                visited.add((pos,speed))
                q.append([pos+speed,speed*2,step+1,op+['A']])
                if (speed>0 and pos+speed>T):
                    q.append([pos,-1,step+1,op+['R']])
                if (speed<0 and pos+speed<T):
                    q.append([pos,1,step+1,op+['R']])
        return -1

#actual write up for top down dp
class Solution:
    def racecar(self, target: int) -> int:
        '''
        top down dp
        https://leetcode.com/problems/race-car/discuss/124326/Summary-of-the-BFS-and-DP-solutions-with-intuitive-explanation
        '''
        memo = {}
        
        def dp(i):
            #dp(i) returns number of steps remaning to close the gap, or reach i
            if i == 0:
                return 0
            if i in memo:
                return memo[i]
            
            result = float('inf')
            #case 1: try out all options when moving to the right then move left
            numStepsRight = 1
            distRight = 1
            
            while distRight < i:
                numStepsLeft = 0
                distLeft = 0
                
                while numStepsLeft < numStepsRight:
                    result = min(result, numStepsRight + 1 + numStepsLeft + 1 + dp(i - (distRight - distLeft)) )
                    numStepsLeft += 1
                    distLeft = (1 << numStepsLeft) - 1
                
                numStepsRight += 1
                distRight = (1 << numStepsRight) - 1
                
            #case 2: we go straight from the current positon to target positiosn
            if distRight == i:
                result = min(result,numStepsRight)
                
            # Case 3: We go from the current position beyond the target position (we step over it and will need to go in the other direction during the next recursive call).
            else:
                result = min(result, numStepsRight + 1 + dp(distRight - i))
            
            memo[i] = result
            return result
        
        return dp(target)


###################################
# 944. Delete Columns to Make Sorted
# 03JAN23
###################################
class Solution:
    def minDeletionSize(self, strs: List[str]) -> int:
        '''
        we just scan down columns and check if they are unsorted
        '''
        
        rows = len(strs)
        cols = len(strs[0])
        
        delete_cols = 0
        
        for c in range(cols):
            for r in range(1,rows):
                first_char = strs[r-1][c]
                second_char = strs[r][c]
                if first_char > second_char:
                    delete_cols += 1
                    break
        
        return delete_cols


#######################################
# 2244. Minimum Rounds to Complete All Tasks
# 04JAN23
#######################################
#fucknig edge cases
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        '''
        want the min number of rounds to complete the tasks
        it makes sense to complete tasks in three first, then two to get the min number of rounds
        if we get to a point where there is a count of 1 at all, we can't finish it
        
        [2,2,3,3,2,4,4,4,4,4]
        
        2:3
        3:2
        4:5
        
        then traverse then sor the counts and try t0 reduce by three, only if we have have 2 left over
        same thing with 2
        '''
        counts = Counter(tasks)
        sorted_counts = sorted([v for k,v in counts.items()],reverse = True)
        
        rounds = 0
        final_array = []
        
        for count in sorted_counts:
            #first reduce by three
            if count % 3 == 0:
                rounds += count // 3
                count = 0
            #use up 3, then 2
            elif count % 3 == 2:
                rounds += count // 3
                rounds += 1
                count = 0
            elif count % 2 == 0:
                rounds += count // 2
                count = 0
            final_array.append(count)
        
        if all([0 == num for num in final_array]):
            return rounds
        else:
            return -1

#finally
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        '''
        we don't need to sort, just check and use up as much as we can
        '''
        counts = Counter(tasks)
        
        rounds = 0

        
        for task,count in counts.items():
            if count == 1:
                return -1
            elif count % 3 == 2:
                rounds += (count //3) + 1
            elif count % 3 == 0:
                rounds += (count //3)
            else:
                #think o flike 4,7,and,10, we left with 2 groups of two
                count -= 4
                rounds += (count // 3) + 2
        
        return rounds


class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        '''
        we don't need to sort, just check and use up as much as we can
        the trick is to realize we need to priotirze reducing by // 3
        then we can just use the remaining round of 2
        #think of numbres like 4 and 7

        we can represent the counts of tasks as number n
        and n can be written as: 3x + 2y
        where x in the number of groups of 3 and y is the number of groups of 2
        to minimuze the sum of x + y, we want to maximize x
        
        if number is 3*k, we can make k groups of 3
        if numbers is 3*k + 1, we can make (k-1) groups of 3, and leave 4, which is just 2 groupds of two
        if number is 3*k + 2, just make k groups of 3, and 1 groupd of 2
        '''
        counts = Counter(tasks)
        
        rounds = 0
        
        for task,count in counts.items():
            if count == 1:
                return -1
            
            if count % 3 == 0:
                rounds += count // 3
            else:
                rounds += (count // 3) + 1
        
        return rounds

#dp using counter
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        '''
        we can also use dp to solve this, similar to the coin change problem
        count them up and try use a 2 or 3
        then take the min
        dp(i) represents the minunum number of roundsto reduce to 0
        dp(i) = 1 + min(dp(i-2),dp(i-3))
        '''
        counts = Counter(tasks)
        
        memo = {}
        
        def dp(i):
            if i == 2:
                return 1
            if i == 3:
                return 1
            if i == 1:
                return float('inf')
            if i in memo:
                return memo[i]
            ans = float('inf')
            if i >= 2:
                ans = min(ans,1 + dp(i-2))
            if i >= 3:
                ans = min(ans,1 + dp(i-3))
                
            memo[i] = ans
            return ans
        
        rounds = 0
        
        for task,count in counts.items():
            if count == 1:
                return -1
            min_rounds = dp(count)
            if min_rounds == float('inf'):
                return -1
            rounds += min_rounds
        
        return rounds

#dp using sorting
#quite painful to think about recursion for this one
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        '''
        we can use dynamic programming on the non decreasing sorted array
        '''
        tasks.sort()
        
        memo = {}
        
        def dp(i):
            #reach beyond end of array, nothing
            if i == len(tasks):
                return 0
            elif i in memo:
                return memo[i]
            elif i > len(tasks) or i == len(tasks) - 1:
                return float('inf')
            elif tasks[i] != tasks[i+1]:
                return float('inf')

            
            elif i + 2 < len(tasks) and tasks[i] == tasks[i+2]:
                ans = 1 + min(dp(i+2),dp(i+3))
            else:
                ans = 1 + dp(i+2)
            
            memo[i] = ans
            return ans
        
        ans = dp(0)
        return ans if ans != float('inf') else -1

#############################################################
# 452. Minimum Number of Arrows to Burst Balloons (REVISTED)
# 05JAN23
#############################################################
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        '''
        aye yai yai
        night mare problem
        
        if i want to minimize the number of arrows use to pop balloons
        i want to find the largest number of intersections for the balloong
        
        this is really just capturing intervals smartly
        for a close group of ballongs
        i want max(all starts) and min(all ends)
        
        sort ballons on start, then try to capture as many intervals as you can
        evidently sort by end
        
        no you could do by start
        '''
        points.sort()
        points.sort()
        curr_start,curr_end = points[0]
        ans = 0
        
        for start, end in points[1:]:
            if start <= curr_end:
                curr_start = max(curr_start, start)
                curr_end = min(curr_end, end)
            else:
                ans += 1
                curr_start,curr_end = start, end
        
        return ans + 1

###################################
# 233. Number of Digit One
# 05JAN23
###################################
#O(NlgN)
#TLE unfortunatley
class Solution:
    def countDigitOne(self, n: int) -> int:
        '''
        first lets try linear dp
        dp(i) be the number of ones for a number i
        dp(i) = dp(i-1) + number of ones contributed at i
        '''
        
        memo = {}
        
        def dp(i):
            if i == 0:
                return 0
            if i == 1:
                return 1
            if i in memo:
                return memo[i]
            
            #find contribution of ones for this current i
            #this is Olog(N)
            temp = i
            count_ones_at_i = 0
            while temp:
                if temp % 10 == 1:
                    count_ones_at_i += 1
                temp = temp // 10
            
            ans = dp(i-1) + count_ones_at_i
            memo[i] = ans
            return ans
        
        return dp(n)

#dp O(N)


class Solution:
    def countDigitOne(self, n: int) -> int:
        '''
        official solution first
        
        consider the number of 1's in the ones positions, tens positions, and so on...
        for the ones places
        for 
        0 to 10
        there can only be 1 one in the ones positions
        0 to 20
        there are two ones in the ones place
        0 to 30
        there are 3 ones in the one's place
        
        nuber of 1's at the ones positosn
        (n//10) + (n % 10 != 0)
        
        now how about the 10's place?
        up to 100, 10 ones at the 10's place
        up to 200, 20 ones at the 10's place
        
        1600, 160 one's
        1610, 161 one's
        
        number of 1's at the 10s place:
        (n/100)*10 + min(max(n % 100 - 10 + 1,0),10))
        
        for hundred's place
        (n/1000)*100 + min(max(n % 1000 - 100 + 1,0),100))
        
        count is formulated as (n/(i*10))*i
        
        we essentially just sum up the ones at each position using the formula
        it's not easy to come up with

        No of \text{'1'}’1’ in \text{ones}ones place = 1234/101234/10(corresponding to 1,11,21,...1221) + \min(4,1)min(4,1)(corresponding to 1231) =124124

No of \text{'1'}’1’ in \text{tens}tens place = (1234/100)*10(1234/100)∗10(corresponding to 10,11,12,...,110,111,...1919) +\min(21,10)min(21,10)(corresponding to 1210,1211,...1219)=130130

No of \text{'1'}’1’ in \text{hundreds}hundreds place = (1234/1000)*100(1234/1000)∗100(corresponding to 100,101,12,...,199) +\min(135,100)min(135,100)(corresponding to 1100,1101...1199)=200200

No of \text{'1'}’1’ in \text{thousands}thousands place = (1234/10000)*10000(1234/10000)∗10000 +\min(235,1000)min(235,1000)(corresponding to 1000,1001,...1234)=235235

Therefore, Total = 124+130+200+235 = 689124+130+200+235=689.
        '''
        count = 0
        i = 1
        while i <= n:
            #base at the radix for the 10's position
            divider = i*10
            #what we can only go up to for this bit position
            #but need to include the remainder, we cannot add less then zero
            #and it cannot be more than the current i
            count_at_pos = (n // divider)*i + min(max(n % divider - i + 1,0),i)
            #increment count
            count += count_at_pos
            #move to next bit posisiton
            i = i*10
        
        return count

#https://leetcode.com/problems/number-of-digit-one/discuss/64382/JavaPython-one-pass-solution-easy-to-understand
#bleagh
class Solution:
    def countDigitOne(self, n: int) -> int:
        '''
        the idea is to calculate the occurrence of 1 on every digit position
        
        if n = xyzdabc
        
        then there are theree scenarios
        
        (1) xyz * 1000                     if d == 0
        (2) xyz * 1000 + abc + 1           if d == 1
        (3) xyz * 1000 + 1000              if d > 1
        
        
        following the rules
        '''
        if n <= 0:
            return 0
        
        position = 1
        count = 0
        copy_n = n
        
        while n > 0:
            #find the d
            d = n % 10
            if d == 0:
                count += (copy_n // position*10)*position
            elif d == 1:
                count += (copy_n // position*10)*position + (n % position*10) + 1
            elif d > 1:
                count += (copy_n // position*10) + position
                
            position = position*10
            n = n // 10
            
        return count

class Solution:
    def countDigitOne(self, n: int) -> int:
        '''
        the idea is to calculate the occurrence of 1 on every digit position
        
        if n = xyzdabc
        
        then there are theree scenarios
        
        (1) xyz * 1000                     if d == 0
        (2) xyz * 1000 + abc + 1           if d == 1
        (3) xyz * 1000 + 1000              if d > 1
        
        running through a few examples
        
        For people who doesn't understand the author's explanations, just look at some examples:

        Let n = 4560000

        How many nums with "1" at the thousand's position?

        4551000 to 4551999 (1000)
        4541000 to 4541999 (1000)
        4531000 to 4531999 (1000)
        ...
        1000 to 1999 (1000)

        That's 456 * 1000

        What if n = 4561234?

        4561000-4561234 (1234+1)
        4551000 to 4551999 (1000)
        4541000 to 4541999 (1000)
        4531000 to 4531999 (1000)
        ...
        1000 to 1999 (1000)

        That's 456 * 1000 + 1234 + 1

        What if n = 4562345?
        4561000-4561999 (1000)
        4551000 to 4551999 (1000)
        4541000 to 4541999 (1000)
        4531000 to 4531999 (1000)
        ...
        1000 to 1999 (1000)

        That's 456*1000 + 1000

        Same for hundred's position.

        Let n = 4012

        How many nums with "1" at the hundred's position?

        3100-3999 (100)
        2100-2999 (100)
        1100-1999 (100)
        100 to 199 (100)
        That's 4 * 100

        Let n = 4111

        4100-4111 ( 11 + 1)
        3100-3999 (100)
        2100-2999 (100)
        1100-1999 (100)
        100 to 199 (100)
        That's 4 * 100 + 11 + 1

        Let n = 4211

        4100-4199 (100)
        3100-3999 (100)
        2100-2999 (100)
        1100-1999 (100)
        100 to 199 (100)
        That's 4 * 100 + 100

        Same for ten's digit

        Let n = 30
        How many nums with "1" at the ten's position?

        210-219 (10)
        110-119 (10)
        10-19 (10)

        That's 3 * 10

        Let n = 312

        310-312 (2 + 1)
        210-219 (10)
        110-119 (10)
        10-19 (10)
        That's 3 * 10 + 2 + 1

        Let n = 322

        310-319 (10)
        210-219 (10)
        110-119 (10)
        10-19 (10)
        That's 3 * 10 + 10

        Same for one's digit

        Let n = 30
        How many nums with "1" at the one's position?

        21 (1)
        11 (1)
        1(1)
        That's 3 * 1

        Let n = 31
        How many "1" are there at the one's position?
        31 (1)
        21 (1)
        11 (1)
        1 (1)
        That's 3 * 1 + 1

        Let n = 32
        How many "1" are there at the one's position?
        31 (10)
        21 (10)
        11 (10)
        1 (10)
        That's 3 * 1 + 1

        Let n = 3

        only 1 (10 of "1" at one's position)

        That's 0 * 1 + 1
        '''
        if n <= 0:
            return 0
        
        q = n
        radix = 1
        num_ones = 0
        
        while q > 0:
            digit = q % 10
            q //= 10
            
            #increment
            num_ones += q*radix
            
            #additionals
            if digit == 1:
                num_ones += n % radix + 1
            elif digit > 1:
                num_ones += radix
            
            radix = radix*10
        
        return num_ones

#digit dp
class Solution:
    def countDigitOne(self, n: int) -> int:
        '''
        really just digit dp
        https://www.geeksforgeeks.org/digit-dp-introduction/
        https://leetcode.com/problems/number-of-digit-one/discuss/908032/Easy-to-understand-Digit-DP-solution(C%2B%2B)
        '''
        memo = {}
        
        digits = []
        while n:
            digits.append(n % 10)
            n = n // 10
        
        digits = digits[::-1] #digits procreed from most sig bit to least sig bit
        
        #states
        #(pos,count,flag), pos = current position from the left side, count = count of 1s so far
        #flag = the number we are building has already become mssaller than the given number (0 = no, 1 = yes)
        
        
        N = len(digits)
        
        def dp(pos,count,flag):
            if pos == N:
                return count
            if (pos,count,flag) in memo:
                return memo[(pos,count,flag)]
            
            #if the number has already become smaller than the given number we can insert any digit we want
            if flag == 0:
                limit = digits[pos]
            else:
                limit = 9
            
            res = 0
            for i in range(limit+1):
                #if the digit is 1, we increase count by 1
                n_count = count + 1 if i == 1 else count
                #if the digit at the ith position is msaller than the digit at the ith position of the number
                #then the number has already become smaller, so we can set flag to 1
                n_flag = flag or i != limit
                #recurse
                res += dp(pos+1,n_count,n_flag)
            
            memo[(pos,count,flag)] = res
            return res
        
        
        return dp(0,0,0)

##################################
# 1833. Maximum Ice Cream Bars
# 06JAN23
##################################
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        '''
        uhhhh
        just sort and by the cheapest ones first

        i also could have sorted, then use pref sum
        the binary search for the upper bound
        still NlgN
        '''
        costs.sort()
        ice_cream = 0
        for c in costs:
            if c <= coins:
                ice_cream += 1
                coins -= c
            else:
                break
        
        return ice_cream

#binary search
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        '''
        sorthing then use binary search
        '''
        costs.sort()
        pref_sum = [0]
        for c in costs:
            pref_sum.append(c + pref_sum[-1])
        
        left = 0
        right = len(pref_sum)
        
        while left < right:
            mid = left + (right - left) // 2
            if pref_sum[mid] > coins:
                right = mid
            else:
                left = mid + 1
            
        return left - 1

#actual way is counting sort
class Solution:
    def maxIceCream(self, costs: List[int], coins: int) -> int:
        '''
        we can using counting sort
        make frequqne array showing number of cones at that cost
        then take as much as we can
        be careful when trying to take 1 cone at a time for a cost
        just use division
        '''
        n, icecreams = len(costs), 0
        m = max(costs)

        costsFrequency = [0] * (m + 1)
        for cost in costs:
            costsFrequency[cost] += 1

        for cost in range(1, m + 1):
            # No ice cream is present costing 'cost'.
            if not costsFrequency[cost]:
                continue
            # We don't have enough 'coins' to even pick one ice cream.
            if coins < cost:
                break
            
            # Count how many icecreams of 'cost' we can pick with our 'coins'.
            # Either we can pick all ice creams of 'cost' or we will be limited by remaining 'coins'.
            count = min(costsFrequency[cost], coins // cost)
            # We reduce price of picked ice creams from our coins.
            coins -= cost * count
            icecreams += count
            
        return icecreams

###########################
# 134. Gas Station (REVISTED)
# 07JAN23
###########################
#brute force
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        '''
        we want to to find the starting station's gas index if we can make a complete circuit
        interesting enough, if there is a complete circuit, it can only be started from a unique gas station
        
        brute force would be to try each gas station and make it across
        '''
        N = len(gas)
        for i in range(N):
            curr_station = i
            tank = gas[curr_station] - cost[curr_station]
            curr_station = (curr_station + 1) % N
            while tank > 0 and curr_station != i:
                #travel
                tank -= cost[curr_station]
                tank += gas[curr_station]
                curr_station = (curr_station + 1) % N
            
            if curr_station == i:
                return curr_station
        
        return -1

class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        '''
        need to realize that it is impossible to complete the route if sum(gas) < sum(cost)
        and that it is impossible to start at station i, if gas[i] - cost[i] < 0, because there is not enough in the tank to reach i + 1
        
        what can we conclude from these two things?
        we need to genearlize the second fact
        
        for each station i, stat with curr_tank 0, if it ever goes below negative, we can't start here at all
        our candidate answer will be 0, then we update tank
        if tank every drops bewlow 0, update tank to zero, and start new candidate answer to i + 1
        
        we could also just mimic a circular array and check that we can reach all (i.e i - curr_start >= N)

        https://leetcode.com/problems/gas-station/discuss/42568/Share-some-of-my-ideas.

        This is how I understood and may be it helps others to develop intuition on the O(N) solution.

        A -- x -- x --- x --- x -- B
        The proof says, let's say we start at A and B is the first station we can not reach. Then we can not reach B from all the stations between A and B. The way to think about it is like this, let's say there was a station C between A and B.

                        fuel >= 0
        A -- x -- x --- *C --- x -- B
        When we started from A we had enough fuel to get from A to C and then from C to a station before B. This means that when we reached from A to C we had at least 0 or more fuel in our tank. We refueled at C and then started onward trip.

         fuel = 0
                *C --- x -- B
        Now if we were to start at C with 0 capacity, we would not be any better in terms of fuel reserves than a trip that started at A. It's guaranteed that we'd fail to make it to B.

        Hence we start our search at i+1'th index.


        '''
        start = 0
        tank = 0
        N = len(gas)
        
        for i in range(2*N):
            tank += gas[i % N] - cost[i % N]
            
            #not enough gas
            if tank < 0:
                tank = 0
                start = i + 1
                
            #can rech
            if i - start == N:
                return start
        
        return -1

#################################
# 1854. Maximum Population Year
# 07JAN23
################################
#brute force
class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        '''
        similar to capturing intervals
        want the earliest year with max population
        
        brute force first just check all years using countmap
        then sort
        '''
        counts = Counter()
        for start,end in logs:
            for year in range(start,end):
                counts[year] += 1
        
        counts = [(k,v) for k,v in counts.items()]
        counts.sort(key = lambda x: (-x[1],x[0]))
        print(counts)
        return counts[0][0]

#range addition
#similar concept for my booking calendar problems
#https://leetcode.com/problems/maximum-population-year/discuss/1198978/JAVA-oror-O(n)-solution-With-Explanation-oror-Range-Addition
class Solution:
    def maximumPopulation(self, logs: List[List[int]]) -> int:
        '''
        we can use conept of range addition
        for example, given the long entry [1950,1961] it means there was +1 for every year from 1950 to 1960
        only at 1961, we -1
        then for every year after 1950 we can accumlate them
        we just travere the prex sum array
        then take the maximum, 
        we want the earliest year
        '''
        year_counts = [0]*2051
        
        for start,end in logs:
            year_counts[start] += 1
            year_counts[end] -= 1
            
        max_pop = year_counts[1950]
        min_year = 1950
        
        # we can find max as we build the prefsum sum array
        for i in range(1950,len(year_counts)):
            #accumulate
            year_counts[i] += year_counts[i-1]
            
            #maximum
            if year_counts[i] > max_pop:
                max_pop = year_counts[i]
                min_year = i
        
        return min_year



#######################################
# 149. Max Points on a Line (REVISTED)
# 08JAN23
#######################################
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        '''
        for each point, we need to compare it with all the other points
        then when comparing, find the unique signature for that line
        
        increment into a counter using signature
        then update the current max
        
        inf would only show up ince since we do it one point at a time
        python is is smart enough to handle long floats as unique hahses
        '''
        
        def get_slope(x1,y1,x2,y2):
            if x1 == x2:
                return float('inf')
            return (y1-y2)/(x1-x2)
        
        ans = 1 #we will always have at least 1 point
        N = len(points)
        
        for i in range(N):
            curr_max = 1
            slopes = Counter()
            for j in range(i+1,N):
                x1,y1 = points[i]
                x2,y2 = points[j]
                slope = get_slope(x1,y1,x2,y2)
                slopes[slope] += 1
                curr_max = max(curr_max, 1 + slopes[slope]) #because we need to include the other point
            #update globalle
            ans = max(ans,curr_max)
        
        return ans

class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        '''
        we can alsoe use the line equation is the signature and store the coefficients as tuple
        find the slope, y intercept, and x intercept
        '''
        n = len(points)
        def calLine(x1, y1, x2, y2): # calculate the line equation
            if x1 == x2:
                return (float('inf'), 0, x1)
            k = (y1 - y2) / (x1 - x2)
            b = y1 - k * x1
            #this is the slope, something, and y intercept
            return (k, b, x1)

        res = 1
        for i in range(n):
            mem = collections.defaultdict(set)
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                line = calLine(x1, y1, x2, y2)
                #add the current point and the next point to the set
                mem[line].add((x1, y1))
                mem[line].add((x2, y2))
                #maximize
                res = max(res, len(mem[line]))
        return res

#using acrtan trick
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        '''
        instead of using line equation and slopes, we can use arctan between two vectors
        two points are in the same line, if they have the same arctan, i.e there vectors are aligned
        i also could have use unit normal vectors too as theri signatures
        
        for using arc tans we need to check all of them
        '''
        N = len(points)
        ans = 1
        
        for i in range(N):
            arc_tans = Counter()
            for j in range(N):
                if i != j:
                    xcomp = points[j][0] - points[i][0]
                    ycomp = points[j][1] - points[i][1]

                    arc_tan = math.atan2(xcomp,ycomp)
                    arc_tans[arc_tan] += 1
                    ans = max(ans,arc_tans[arc_tan]+1)
        
        return ans

###################################
# 2214. Minimum Health to Beat Game
# 06JAN23
####################################
#YASSS, 
#trick was whether we need to decut armor or the damage at i
#take min
class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        '''
        notice the anser for the question:
            want min health needed at start of game to beat the game
            not the number of level we can complete with armor 4
        
        we can use armor which shields as from at most armor
        health must be greater then zero at all times
        
        if i have no armor, i need sum(damage) + 1
        find the point where we can use armor
        use armor at the highest 
        '''
        if armor == 0:
            return sum(damage) + 1
        
        #sort
        total_damage = sum(damage)
        damage.sort()
        N = len(damage)
        
        

        for i in range(N):
            if damage[i] >= armor:
                break
        
        return total_damage - min(armor,damage[i]) + 1


#fun problem
class Solution:
    def minimumHealth(self, damage: List[int], armor: int) -> int:
        '''
        official write up
        we want to use amore to block the largest damage
        either it sheilds us from all damage: where armore > maxDamage
        or it partially sheilds us, where armor < d: and we take d - armor damage
        '''
        total_damage = sum(damage)
        max_damage = max(damage)
        
        return total_damage - min(max_damage,armor) + 1

###############################################
# 1525. Number of Good Ways to Split a String
# 10JAN23
##############################################
#close one...
class Solution:
    def numSplits(self, s: str) -> int:
        '''
        brute force would be check all splits and check distinct counts
        i can use hashmap to store distinct counts for each part
        
        what if i started off with all distinct
        and just keep count
        '''
        ans = 0
        N = len(s)
        all_distinct = set(s)
        count_all = len(all_distinct)
        
        curr_distinct = set()
        curr_count_distinct = 0
        
        for i in range(N):
            char = s[i]
            if char not in curr_distinct:
                curr_distinct.add(char)
                curr_count_distinct += 1
            
            if char in all_distinct:
                all_distinct.remove(char)
                count_all -= 1
            
            if count_all == curr_count_distinct:
                ans += 1
        
        return ans

#yassss!
class Solution:
    def numSplits(self, s: str) -> int:
        '''
        almost had it
        we can keep an inital mapp/hashset of the left part
        then build up the right part, but take away from left as we move along
        
        this is really just a two hashmap problem
        
        '''
        right = Counter()
        left = Counter(s)
        
        ans = 0
        for ch in s:
            #increment
            right[ch] += 1
            #remove from left
            if ch in left:
                left[ch] -= 1
            if left[ch] == 0:
                del left[ch]
                
            if len(right) == len(left):
                ans += 1
        
        return ans

class Solution:
    def numSplits(self, s: str) -> int:
        '''
        we don't need to use two hashmaps for the left and the right
        we only just need to use one hashmap and a hash set
        
        first say that the inital string is entirely on the left
        '''
        right = set()
        left = Counter(s)
        
        ans = 0
        
        for ch in s:
            right.add(ch)
            #iniitally starting this char must have been accounted for
            left[ch] -= 1
            #if it dropes to zero
            if left[ch] == 0:
                del left[ch]
                
            if len(right) == len(left):
                ans += 1
        
        return ans

#####################################################
# 1443. Minimum Time to Collect All Apples in a Tree
# 11JAN23
#####################################################
#fuckkkkkkk
class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        '''
        each edge has cost of 1, so we can use BFS somehow, no relaxing
        that fact that i have to go back up an edge would mean i need to usse dfs to return to the parent caller
        the hint gave it away
            if a node u contains an apple, then all edges in the path from root to node u have to be used two times
        '''
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        seen = set()
        def dfs(node,seen):
            if node in seen:
                return 0
            if hasApple[node]:
                return 2
            #visit
            seen.add(node)
            ans = 0
            
            for neigh in graph[node]:
                ans += dfs(neigh,seen)
            seen.remove(node)
            
            return ans
        
        return dfs(0,seen)

class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        '''
        the idea it visit all children before returning back to the parent
        then we can just add them up back to the root
        to avoid going back on an edge we need to take to dfs, you can keep track of a parent and child node in the recursive function
        
        example
            say we are at p, and we have child node c1
                we first find the time it takes to collect all the pples in the subtree of c1, which we call t
                if t == 0, there are no apples and that tree, so don't go down that path, return 0
                
            otherwise we must visit the subtree and collect all the apples, which would be t + 2, because we have to back up
            
        dfs function has paramters node and child
            for each call we don't know the totalTime or childTime
            totalTime is time to collect all apples for this subtree
            childTime is time required to collect all apples for each immediate child of node
            
            check all children, and child is equal to parent, skip it (cool way of backtracking)
            if child is not equal to parent, recurse
            
            if child has an apple or there are any apples in he subtree which can be checked if childTime > 0, we must visit this child, which takes one unit of time
            
            if neither the child nor the subtree has apples, we don't need to include the time to visit this child
            as we will consier we never visited this child's subtree
        '''
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def dfs(node,parent):
            totalTime = 0
            childTime = 0
            for neigh in graph[node]:
                #don't go back
                if neigh == parent:
                    continue
                childTime = dfs(neigh,node)
                #get the mount of time is takes to get apples for this child
                if childTime > 0 or hasApple[neigh]:
                    totalTime += childTime + 2
                    
            return totalTime
        
        return dfs(0,-1)
            
#another way
class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        '''
        just another way 
        '''
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        def dp(node,parent):
            ans = 0
            #graph is a tree so there are no cycles, but the edges are unidrected
            for neigh in graph[node]:
                if neigh == parent:
                    continue
                #accumlate ansers
                else:
                    ans += dp(neigh,node)
            
            #if res is not 0, there are still apples down in the tree, so we need to add 2 
            #if this node has an appler, we add 2 to the result
            if ans > 0 or hasApple[node]:
                return ans + 2
            
            return ans
        
        
        ans = dp(0,-1) - 2 #when coming back to nodes 0. we over counted by 2 (recall this works when solving the subproblems, but for the root we dont need it)
        return max(ans,0)


#one more way using visited set
#https://leetcode.com/problems/minimum-time-to-collect-all-apples-in-a-tree/discuss/623686/Java-Detailed-Explanation-Build-Tree-%2B-DFS
class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        seen = set()
        
        def dp(node):
            #add node
            seen.add(node)
            totalTime = 0
            
            for neigh in graph[node]:
                if neigh in seen:
                    continue
                #get answers to subproblems
                totalTime += dp(neigh)
                
            #if there are apples further down the tree, or this node has an apples
            if (totalTime > 0) or hasApple[node]:
                totalTime += 2
            
            return totalTime
        
        
        return max(dp(0) -2,0)


#anothe way is to use kahn's algo
'''
First you trim the tree and remove leaf-nodes with edges without apples. In example #1 it will be 3,6. In example #2 - 4,3,6. Now all you need to do is calc remaining edges and x2.
'''
class Solution:
    def minTime(self, n: int, edges: List[List[int]], hasApple: List[bool]) -> int:
        def kahnsalgo():
            N = len(hasApple)
            E = len(edges)
            indegrees = [0] * N
            
            for u,v in edges:
                indegrees[v] += 1
                indegrees[u] += 1
            
            queue = deque()
            for i,ind in enumerate(indegrees):
                if ind == 1 and not hasApple[i]: # if there's only 1 in-edge and there's no apple -> empty leaf
                    queue.append(i) # add for removal
                    
            removed_nodes = 0
            while queue:
                node = queue.popleft()
                
                if node == 0: # we cannot remove root in any case
                    continue
                
                for nei in adj_list[node]:
                    adj_list[nei].discard(node)
                    indegrees[nei] -= 1 # remove edge from v,u
                    if indegrees[nei] == 1 and not hasApple[nei]: # another leaf? add to remove
                        queue.append(nei)
                    removed_nodes += 1 # count removed nodes
            
            return (E - removed_nodes) * 2 # (all edges - edges_to_empty_nodes) * 2
        
        adj_list = defaultdict(set)
        for u,v in edges: adj_list[u].add(v), adj_list[v].add(u)
        return kahnsalgo()
            
###########################################################
# 1519. Number of Nodes in the Sub-Tree With the Same Label
# 12JAN23
###########################################################
#close one
class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        '''
        we have unidrected edge lest, and each node has a label
        return array of length n where each interger in the array hold the number of occurences for that label
        
        leaves will return a value of 1 into the array
        
        very similar to what we did yesterday
        
        i need to return some vector of counts to each parent
        then for each call accumulate the counts for that character count
        '''
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
    
        
        seen = set()
        ans = [0]*n #the nodes are 0 indexed
        
        def dp(node):
            seen.add(node)
            count = Counter()
            #first add this node
            count[labels[node]] += 1
            
            for neigh in graph[node]:
                if neigh in seen:
                    continue
                #retrieve the child counts
                child_counts = dp(neigh)
                for char,c in child_counts.items():
                    if char == labels[node]:
                        count[char] += c
            
            #put into answer
            ans[node] = count[labels[node]]
            return count
        
        temp = dp(0)
        return ans

#phew! just accumlate all the letters
class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        '''
        we have unidrected edge lest, and each node has a label
        return array of length n where each interger in the array hold the number of occurences for that label
        
        leaves will return a value of 1 into the array
        
        very similar to what we did yesterday
        
        i need to return some vector of counts to each parent
        then for each call accumulate the counts for that character count
        '''
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
    
        
        seen = set()
        ans = [0]*n #the nodes are 0 indexed
        
        def dp(node):
            seen.add(node)
            count = Counter(labels[node])
            
            for neigh in graph[node]:
                if neigh in seen:
                    continue
                #retrieve the child counts
                child_counts = dp(neigh)
                for char,c in child_counts.items():
                    count[char] += c
            
            #put into answer
            ans[node] = count[labels[node]]
            return count
        
        temp = dp(0)
        return ans

#bfs, kahsn' prune edges and start from leaves
class Solution:
    def countSubTrees(self, n: int, edges: List[List[int]], labels: str) -> List[int]:
        '''
        we can also use BFS, but in that case we would need to start from the bottom up
        start traversal from all the leaf nodes, then move to their parents, then their parents' parents, and so on
        we can do this because we are given the edge list, and leaves have no out going edges (Kahn's algorithm)
        
        also need to keep a count array count[n][26], where count[i] stores the counts for each of its labels
        
        bfs traversal:
            pop from left, fetch parent
            add count of each lable in the subtree of node to that parent: count[parent] += count[node]
            then push parent back into the queue
        
        at the end re-traverse the count array for counta
        
        important, 
        how to figure out the parent node and how to compute the count of each label in its subtree using children
        leaf nodes will have have on node in the graph, which is just their immediate parent
        we bring count of leaf up to the parent
        
        then we delete leaf from pop, we can just pop, so we don't go back down then we eventually add the parent back to the queue
        
        Create a mapping adj where adj[X] contains a set of all the neighbors of node X.
        Initialize an array counts[26] for every node, storing the count of each label in the node's subtree. Initialize it with 0 for every node.
        Initialize a queue.
        Iterate over all the nodes and for each node mark counts[node][labels[node] - 'a'] = 1. Also, check if node is a leaf node. It is a leaf node, if node != 0 && adj[node].size() == 1. Push node into the queue.
        Then, while the queue is not empty:
        Dequeue the first node from the queue.
        Get the parent of the node from adj[node].
        For the parent, we remove node from adj[parent] to avoid traversing back to node from parent.
        Add counts[node] to counts[parent].
        If the size of adj[parent] == 1 && parent != 0 (root has no parent), which means we added the count of each label in all subtrees of its children and deleted the children. The node present in adj[parent] is its parent. In such a case, push the parent into the queue.
        Iterate over all the nodes and for each node return counts[node][labels[node] -a].
        
        
            '''
        graph = defaultdict(list)
        
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        counts = [[0]*26 for _ in range(n)]
        q = deque([])
        
        #store inital counts for each node
        for node in range(n):
            #get its character
            char = labels[node]
            counts[node][ord(char) - ord('a')] = 1
            #store leaf nodes in queue
            if node != 0 and len(graph[node]) == 1:
                q.append(node)
        
        #bfs
        while q:
            curr = q.popleft()
            #find parent, but we need to remove this so we don't come back down
            #we are essentially removing edges by removing nodes when going from the leaves to the root
            parent = graph[curr].pop()
            
            #from the current nodes accumlate into parent
            for i in range(26):
                counts[parent][i] += counts[curr][i]
            
            #if after remove edges, parent becomes a leaf, push back into q
            if parent != 0 and len(graph[parent]) == 1:
                q.append(parent)
        
        #get the anser
        ans = []
        for node in range(n):
            ans.append(counts[node][ord(labels[node]) - ord('a')])
        
        return ans

########################################################
# 2246. Longest Path With Different Adjacent Characters
# 13JAN23
#########################################################
#close one... :(
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        '''
        we want the longest path in tree such that no pair of adjacent nodes on the path have the same character assigned to them
        path length can be defined as number of nodes in path + 1
        
        notes:
            keeping parent pointers is another way to construct graph
        
        what if were to bfs from the root
        this is just dp after we get the graph
        
        dp(node) gives path for longest in that subtree
        do(node) = 1 + max(dp(childd) for child in graph[node] of s[child] != s[node])
        
        '''
        #make graph
        graph = defaultdict(list)
        n = len(parent)
        for i in range(1,n):
            u = i
            v = parent[i]
            graph[u].append(v)
            graph[v].append(u)
        
        
        seen = set()
        
        def dp(node):
            seen.add(node)
            max_path_from_child = 0
            
            for neigh in graph[node]:
                if neigh in seen:
                    continue
                
                if s[node] != s[neigh]:
                    max_path_from_child = max(max_path_from_child,1 + dp(neigh))
            
            return max_path_from_child
        
        return dp(0) + 1
                    

class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        '''
        almost had it
        for a node we want the first lognest chain, L1 and the second longest chain, L2
        we can do this for each node, do we need to update globally
        
        let dp(node) by longest path at this node
        then dp(node) = 1 + l1 + l2
        why do we have to maximize globally?
        each child is returning to its parent the longest path
            because we need to return only one chain
            if each child returned a chain, we could have instants where nodes have > 2 child, and > 2 chains is not a valid path
        
        same thing, by in dp function we need to find the frist longest and second longest
        '''
        #make graph
        graph = defaultdict(list)
        n = len(parent)
        for i in range(1,n):
            u = i
            v = parent[i]
            graph[v].append(u)
            graph[u].append(v)
            
        #don't use seen set, just keep passing parent and child
        self.ans = 1
        
        def dp(node,parent):
            first_longest = 0
            second_longest = 0
            
            for neigh in graph[node]:
                #don't go back up to parent
                if neigh == parent:
                    continue
                #get path for this child
                longest_from_child = dp(neigh,node)
                if s[node] != s[neigh]:
                    if longest_from_child >= first_longest:
                        second_longest = first_longest
                        first_longest = longest_from_child

                    elif longest_from_child > second_longest:
                        second_longest = longest_from_child
            #get this anyser
            curr_ans = first_longest + second_longest + 1
            self.ans = max(self.ans,curr_ans)
            #return the longest path, not the answer we wishing to maximize!!!
            return first_longest + 1
        
        temp = dp(0,-1)
        return self.ans

#bottom up bfs
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        '''
        we need to start from the children and work our way up
        we can adotop a similar appraoch using Kahn's algrothm
        then we just need to keep a memo for the two longest chains at each node
        
        '''
        #make graph
        n = len(parent)
        graph = defaultdict(list)
        n = len(parent)
        for i in range(1,n):
            u = i
            v = parent[i]
            graph[v].append(u)
            graph[u].append(v)
        
        ans = 1
        longest_two_per_node = [[0]*2 for _ in range(n)]
        #index[node][0] is longest, index[node][1] is the second longest
        
        #queue up leaves
        q = deque([])
        
        for node,children in graph.items():
            if len(children) == 1:
                #so far for leaves, we need an iniial anwer to build up our subpobrlme
                #longest chain for each is size 1
                longest_two_per_node[node][0] = 1
                q.append(node)
                
        while q:
            curr_node = q.popleft()
            #get parent, but remove so we don't visit again
            parent = graph[curr_node].pop()
            
            #subproblem
            longest_from_child = longest_two_per_node[curr_node][0]
            
            if s[curr_node] != s[parent]:
                #swap direclty in memo
                if longest_from_child >= longest_two_per_node[parent][0]:
                    longest_two_per_node[parent][1] = longest_two_per_node[curr_node][0]
                    longest_two_per_node[parent][0] = longest_from_child

                elif longest_from_child > longest_two_per_node[parent][1]:
                    longest_two_per_node[parent][1] = longest_from_child
            
            #updates, but we update from parent now
            curr_ans = longest_two_per_node[parent][0] + longest_two_per_node[parent][1] + 1
            ans = max(ans,curr_ans)
            
            #if after remove edges, parent becomes a leaf, push back into q
            if parent != 0 and len(graph[parent]) == 1:
                q.append(parent)
        
        return ans

######################################################
# 1061. Lexicographically Smallest Equivalent String
# 14JAN23
######################################################
#FUCK YES!
class DSU(object):
    def __init__(self):
        self.parents = [i for i in range(26)]
        
    def find(self,x):
        if self.parents[x] != x:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    
    def union(self,x,y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        
        #we want it to put to the smaller character
        if parent_x == parent_y:
            return
        if parent_x < parent_y:
            self.parents[parent_y] = parent_x
        if parent_y < parent_x:
            self.parents[parent_x] = parent_y

class Solution:
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        '''
        this is a graph problem
        we want the lexographicallt smallest equivlanet of baseStr using equivaleny information
        union find to get the groupings from the edgelist using s1 and s2
        
        then we want the lexogrphically smallest base string, which means we want the smallest character in that group
        in the DSU api, add a step that as we modify parent pointers, we will keep an extra array showing the smallest character so far in that group
        wait, instead of doing it by largest count for parent, do it by smallest character!
        

        '''
        dsu = DSU()
        for u,v in zip(s1,s2):
            #convert to index from ord('a')
            u = ord(u) - ord('a')
            v = ord(v) - ord('a')
            
            dsu.union(u,v)
        
        #now go through the baseStre and find the parent, which should be the smallest now
        ans = ""
        for ch in baseStr:
            ch = ord(ch) - ord('a')
            parent = dsu.find(ch)
            ans += chr(ord('a') + parent)
        
        return ans

#we can also just use DFS
class Solution:
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        '''
        we can just treat this a connected components, then just find the smallest in each connected component
        first build graph, then dfs to return the minimum in each group
        '''
        graph = defaultdict(list)
        for u,v in zip(s1,s2):
            graph[u].append(v)
            graph[v].append(u)
            
        def dfs(node,parent,min_char):
            min_char[0] = min(min_char[0],node)
            for child in graph[node]:
                if child == parent:
                    continue
                dfs(child,node,min_char)
        
        ans = ""
        
        for ch in baseStr:
            min_char = ['z']
            dfs(ch,'#',min_char)
            ans += min_char[0]
        
        return ans
            
class Solution:
    def smallestEquivalentString(self, s1: str, s2: str, baseStr: str) -> str:
        '''
        we can just treat this a connected components, then just find the smallest in each connected component
        first build graph, then dfs to return the minimum in each group
        '''
        graph = defaultdict(set)
        for u,v in zip(s1,s2):
            graph[u].add(v)
            graph[v].add(u)
        
        #need temp array to character mappings to minmum
        min_char_mappings = {}
        for i in range(26):
            min_char_mappings[chr(ord('a') + i)] = chr(ord('a') + i)
        
        visited = set()
        
        def dfs(source,visited,components,min_char):
            visited.add(source)
            components.append(source)
            min_char[0] = min(min_char[0],source)
            
            #dfs
            for i in range(26):
                char = chr(ord('a') + i)
                if char not in visited and char in graph[source]:
                    dfs(char,visited,components,min_char)
        
        for i in range(26):
            char = chr(ord('a') + i)
            if char not in visited:
                components = []
                min_char = ['z']
                dfs(char,visited,components,min_char)
                for ch in components:
                    min_char_mappings[ch] = min_char
        
        ans = ""
        for ch in baseStr:
            ans += min_char_mappings[ch][0]
        
        return ans
                
####################################
# 2421. Number of Good Paths
# 15JAN23
####################################
#phew
class DSU(object):
    def __init__(self,size):
        self.parent = [i for i in range(size)]
        self.size = [0]*size
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    #union by rank
    def union(self,x,y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        
        #same parent
        if parent_x == parent_y:
            return
        if self.size[parent_x] > self.size[parent_y]:
            self.size[parent_x] += 1
            self.parent[parent_y] = parent_x
        elif self.size[parent_y] > self.size[parent_x]:
            self.size[parent_y] += 1
            self.parent[parent_x] = parent_y
        else:
            self.size[parent_x] += 1
            self.parent[parent_y] = parent_x
            
            
        

class Solution:
    def numberOfGoodPaths(self, vals: List[int], edges: List[List[int]]) -> int:
        '''
        truly a hard problem, but not incredibly difficult
        we define a good path as:
            1. starting adn ending node are the same value
            2. all nodes in between starting node and ending node have values <= starting node
        
        note:
            a path and its reverse path are the same path
        
        singles nodes are fundamentally a good path, so ans must be >= number of nodes
        had to use hints 
        1. build graph with nodes from smallest to largest value
        2. union find
        
        brute force would be to start from a node, and visit is neighboring nodes that have a lower value (do this for all nodes)
        if we find anothr node with a similar value to the start, this counts as a good path
        
        intution:
            there is no point in traversing from a node to neigh of the neigh has a smaller value (i.e vals[neigh] > vals[node]), hints at sorting
            so we are at a node X in tree T, we can make a subgraph of all nodes <= vals[X], the new subgraph could be connected, or there could be multiple trees sperate from each other
            now in this subgraph, say we have trees c1 and c2, there are subgraphs of T
            imaginge we have 6 nodes with value X, call them a,b,c,d,e, we want to add all of them to the subgraph and find the number ofparths starting and ending with node X
            lets say a,b,and c connect with some nodes in c1
            nodes d and e connect with some nodes in c2, node f is by itsetlf
            c1 = [a,b,c]
            c2 = [d,e]
            c3 = [f] 
            and all these have value X
            
            in c1, starting with a, we have three good paths (a, ab, ac)
                    starting with b, we have (b,bc)
                    starting with c, we have (c)
                    6 in total
            formula for nodes with the same value X and that are connected:
                count(nodes) * count(nodes) + 1 // 2
                
        now we know how to get the paths with value of X, how can we compuet good paths start with value X+1, or X+2...and so on
        to the subgraph above, we add all the nodes having value X+1 and repeat the same 
        so we have compute the counts of all good pahts for val = X + 1, then we extend all those paths the counts of good paths for just X
        count(X+1) + count(X) + count(X-1) .....
        
        key: 
            We can extend the above to start with value 1 first, then add nodes with value 2, then add nodes with value 3, and so on, to the subgraph formed in the previous iteration. 
            we should begin from lowest and move to higher and higher values
            for each value we should add all the nodes with the same value to the exsiting subgraph and calculate the number of good paths formed in each componnet
            
        use unino find
        in this apporach, we map a value to all the nodes that have that value, then we can sort the map with respect to the values
        
        For each node in nodes[], we check all neighbor. If vals[node] >= vals[neighbor], neighbor is already covered, and it is a node in the subgraph. It can be used as an intermediate node in a good path if formed using it. We perform union (node, neighbor) to add node to the current subgraph. Otherwise, if vals[node] < vals[neighbor], the node is not added to the existing components, and it creates a new component with the node itself.
        
        The next step is to compute the number of nodes with the same value added to each component and use the above formula to count the number of good paths. We can do this by using a map, say groups where the key is the unique id of the component (or tree) and the value is the count of nodes from nodes[] in that component. Iterate over all the nodes[] and for each node, we increment group[find(node)] by one. This way, we have the count of nodes in each component.
        
        We iterate over the group map, and for each entry id, size, we add size * (size + 1) / 2 to the count of good paths.
        
        algo:
            1. make graph, for the edge list
            2. create map that groups all the nodes witht same value, we then need to traverse this map in order of by kyes
            3. implement Union Find
            4. iterate over each entry value,node in the value to node hahsmap
                for every node, get the neighbord
                for each neigh, if vals[node] >= vals[neighbor] we perform a union on the node and the neighbord
                after itearting throguh thn nodes, create map  called group
                group[A. contains the number of nodes (form nodes array) that belong to the same component
                for every node in nodes, finds its componenet, and increment the size of the that compoonenet by 1 ing roups
                groups[find(node)] = groups[find(node]] + 1
                iteratre throuhg all the entreis in the group and for each key get the value and add (size(size+1)/2)
        '''
        n = len(vals)
        graph = defaultdict(list)
        dsu = DSU(n)
        
        #make graph
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            
        #make mapping by value
        values_to_nodes = defaultdict(list)
        for i in range(n):
            value = vals[i]
            values_to_nodes[value].append(i)
        
        good_paths = 0
        
        #iterate over values to nodes in sorted order
        for value,nodes in sorted(values_to_nodes.items()):
            #for every nodes, combined neighbords if we can
            for node in nodes:
                for neighbor in graph[node]:
                    #only traverse neighbors with a smaller value
                    if vals[node] >= vals[neighbor]:
                        dsu.union(node,neighbor)
            #map to compute the number od nodes under observation
            group = Counter()
            for node in nodes:
                group[dsu.find(node)] += 1
            
            #the counting only workds if we go in order
            for g,size in group.items():
                good_paths += (size*(size+1) // 2)
                        
        
        return good_paths

###################################
# 57. Insert Interval (REVISITED)
# 16JAN23
####################################
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        '''
        insteaf of traversing the list to to find the insert position,
        we can use binary search
        but we still have to spend linear time doing the merge
        just keep updating the ends of the last interval
        
        '''
        # O(logN)
        position = bisect.bisect_left(intervals, newInterval)
        # O(N)
        intervals.insert(position, newInterval)

        answer = []
        # O(N)
        for i in range(len(intervals)):
            if not answer or intervals[i][0] > answer[-1][1]:
                answer.append(intervals[i])
            else:
                answer[-1][1] = max(answer[-1][1], intervals[i][1])

        return answer
        
####################################################
# 926. Flip String to Monotone Increasing (REVISTED)
# 17JAN23
#####################################################
#dp, knapsack
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        '''
        we either want all 1's, all 0's, or two groups with 0s on left and 1s on the right
        says its a DP problem
        
        if we let dp(i,prev) give the nin number of flips to make montonize increasein from s[:i-1]
        then dp(i,prev) = {
        if we can flip:
            flipped = 1 + dp(i+1,next_digit)
        if we cant flip
            no_flip = dp(i+1,prev)
            
        ans = min(flipped,no_flip)
        }
        '''
        
        memo = {}
        N = len(s)
        
        def dp(i,prev):
            #cant flip anything
            if i == N:
                return 0
            
            if (i,prev) in memo:
                return memo[(i,prev)]
            
            curr_digit = int(s[i])
            ans = float('inf')
            
            #if we can flip
            if 1 - curr_digit >= prev:
                ans = min(ans,1 + dp(i+1, 1 - curr_digit))
                
            #if we can't flip, but still monotone increasing
            if curr_digit >= prev:
                ans = min(ans,dp(i+1,curr_digit))
            
            memo[(i,prev)] = ans
            return ans
        
        return dp(0,0)

#bottom up
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        '''
        we either want all 1's, all 0's, or two groups with 0s on left and 1s on the right
        says its a DP problem
        
        if we let dp(i,prev) give the nin number of flips to make montonize increasein from s[:i-1]
        then dp(i,prev) = {
        if we can flip:
            flipped = 1 + dp(i+1,next_digit)
        if we cant flip
            no_flip = dp(i+1,prev)
            
        ans = min(flipped,no_flip)
        }
        '''

        N = len(s)
        dp = [[float('inf')]*2 for _ in range(N+1)]
        
        for i in range(N,-1,-1):
            if i == N:
                dp[i][0] = 0
                dp[i][1] = 0
                continue
            curr_digit = int(s[i])
            
            for prev in range(2):
                #if we can flip
                if 1 - curr_digit >= prev:
                    dp[i][prev] = min(dp[i][prev],1 + dp[i+1][1-curr_digit])
                #if we can't flip
                if curr_digit >= prev:
                    dp[i][prev] = min(dp[i][prev],dp[i+1][curr_digit])

        
        return dp[0][0]

#official solution
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        '''
        greedy way, dynamic windows
        either all of s is just 0s or 1s, or the left part is all 0s and the right parts is all 1s
        
        rather we initally start with an empty left window, and the right window contains all of s
        at each step, the left window's size
        
        we enumerate each left-right window configuration, the number of flips to make the string monotone increasing is the sum of the number of 1s in the left window
        add the number of 0's in the right window and save the smallest value
        
        if we let left1, be the number of 1's in the left window, and right0 be the number of 0's in the current right window
        when the left window increase and the right window shrinks by 1 chracter, it means we move a character from the right window to the left
        
        if c == 0, left is unchanged, and we decrease right by 1
        if c == 1, left increase by 1, and right will be unchanged
        '''
        left_ones = 0 #left is initially empty
        right_zeros = 0
        
        #right_zeros is the number of flips needed when the left window is empty and the right window is the whole string
        #we have to at most flip all these zeros to ones
        for ch in s:
            if ch == '0':
                right_zeros += 1
        
        ans = right_zeros
        
        for ch in s:
            if ch == '0': #we don't need to flip
                right_zeros -= 1
                #only update herem since this is when the sum drops
                ans = min(ans,right_zeros + left_ones)
            elif ch == '1':
                left_ones += 1
            

        
        return ans

#sinlge state dp,linear time
class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        '''
        bottom up, single dp array
        let dp[i] represent the minimum number of fllips to make the prefix of s of length i, rather s[:i] monton increasing
        dp[0] = 0, empty string is trivially monotone increasing
        
        if s[i-1] == '1', then dp[i] = dp[i-1], sincew we can always append a char of 1 to mianitn montone increasing, recall it was increasing already up to s[:i]
        
        if s[i-1] == '0'
            we need to flip or not flip
            
        Let number curr_ones be the number of character 1s in s' prefix of length i:
        dp[i] = dp[i - 1] if s[i - 1] = '1'
        dp[i] = min(num, dp[i - 1] + 1) otherwise.
        '''
        N = len(s)
        dp = [float('inf')]*(N+1)
        dp[0] = 0
        
        curr_ones = 0
        
        for i in range(N):
            if s[i] == '1':
                dp[i+1] = dp[i]
                curr_ones += 1
            else:
                dp[i+1] = min(curr_ones,dp[i] + 1)
        
        return dp[N]
            
###################################################
# 918. Maximum Sum Circular Subarray (REVISITED)
# 18JAN23
#######################################################
#fuck you...
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        '''
        i can concat the nums array twice and use kadanes algo, where i take the maximum
        then dp(i) = max(dp(i-1) + nums[i],nums[i])
        then just take the max of the dp array
        but once we get to N, we need to remove the first number from the sum since it can no longeg be part of the circular sum
        also need to keep track of rolling sum
        '''
        nums = nums + nums
        N = len(nums)
        dp = [0]*N
        
        curr_sum = nums[0]
        dp[0] = nums[0]
        for i in range(1,N):
            curr_sum += nums[i]
            dp[i] = max(nums[i],dp[i-1] + nums[i],curr_sum)
            #if we have gone over N, remove the first part
            if i >= N // 2:
                print(nums[i-N])
                curr_sum -= nums[i-N]
        
        return max(dp)


#using heap
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        '''
        almost had it, but we need keep memory of the prevSums so far
        we can use a minheap to store the previous maxsums and decremeenet when the size is more than n
        For each i in range [0..2*n-1], we try to find the maximum sum subarray which ends at i.
Let preSumSoFar is the prefix sum of subarray nums[0..i].
We need the minHeap to store previous prefix sums of sub array [0..j], where j < i. We need to store pair of (prefixSum, j) and we must to remove elements from minHeap when it's out of range n, which means when i-j > n.
We pick the minimum prefix sum previous preSumPrevious from the minHeap, so that we can achieve with maximum sum subarray ends at i by preSumSoFar - preSumPrevious.
Add the current pair of (preSumSoFar, i) to minHeap.
https://leetcode.com/problems/maximum-sum-circular-subarray/discuss/1348545/Python-3-solutions-Clean-and-Concise-O(1)-Space
        '''
        N = len(nums)
        prev_sums = [(0,-1)] #keep this trick in mind
        pref_sum_so_far = 0
        
        max_sum = nums[0]
        
        for i in range(2*N):
            #increment pref_sum so far
            pref_sum_so_far += nums[i % N]
            #we need to remove prefSums whos's lengths are bigger than N!
            #this prefix sums are no longser useable
            #we keep removing this prefix sum until i - j < N, since we are advancing i we are always guarenteed to find a next valid prefix sum
            while prev_sums and i - prev_sums[0][1] > N:
                heapq.heappop(prev_sums)
            #update answer by decrementing pref_sum_so_far by the smallest previous prefSum
            smallest_prev_sum = prev_sums[0][0]
            max_sum = max(max_sum,pref_sum_so_far - smallest_prev_sum)
            #push this on to the heap with the index
            heapq.heappush(prev_sums,(pref_sum_so_far,i))
        
        return max_sum

#using queue instead of heap
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        '''
        now instead of storing the previous prefix sums in using a min heap, we can store it in a deque of size N
        deque is in increaisng order
            the front of the queue is the minimum prefixSum, the end of the queue is the maximum prefsum
        
        now when adding a new prefix sum, into the q
            remove the font if its indx is out of range n
            while the max prefix sum >= the current prefix sum, we pop off the maximum pref sum, becase we want to keep a smaller prefSum and the lastest one smallest which is better than the old oness
            push back on to the front prefSumFar, i
            
        
        '''
        N = len(nums)
        prev_prefSums = deque([(0,-1)]) #current prefsum up to index i
        max_sum = nums[0]
        
        pref_sum_so_far = 0
        
        for i in range(2*N):
            pref_sum_so_far += nums[i % N]
            #remove minmum if prefsum is larger than n, this prefsum is unuseable
            if i - prev_prefSums[0][1] > N:
                prev_prefSums.popleft()
            #get answer by removing minimum prefSum
            curr_ans = pref_sum_so_far - prev_prefSums[0][0]
            max_sum = max(max_sum, curr_ans)
            #we need to maintain the rep invaraint for the increasing order queueu
            while prev_prefSums and prev_prefSums[-1][0] >= pref_sum_so_far:
                prev_prefSums.pop()
            #add back in 
            prev_prefSums.append((pref_sum_so_far,i))
        
        return max_sum

class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        '''
        official solution:
            in the typical maximum subarray problem, we deinfe the normal sum as the maximum sum subarray
            for the circulare problem we define a special sum as, the maximum sum of prefsum and a suffix sum where prefix Sum and suffix Sum do not overlap
             
        intuition:
            we can calulate both the normal and the special sum and return the larger one
            we can get the normal sum using Kadane's but how do we get the special sum
        
        len(nums) is n
        to calculate the special sum, we need to find the maximum sum of aprefix sum and a non overlapping suffix sum
        the idea is to enumerate a prefix with its sum and add the maximum suffix sum that starts after the prefix so that there is no overlap
        
        imagine an array suffixSum where suffixSum[i] represents the ssuffix sum starting startinf from index i
        rather suffixSum[i] = sum(nums[i:])
        we can construct an array rightMax, where rightMax[i] = max(suffixSum[i] for i in range(i,N))
        
        rather, rightMax[i] is the largest suffus sum of nums coming at or afet i
        i.e, the max circular sub array is sum of whole array minus min subarray m
        
        With rightMax, we can then calculate the special sum by looking at all prefixes. We can easily accumulate the prefix while iterating over the input, and at each index i, we can check rightMax[i + 1] to find the maximum suffix that won't overlap with the current prefix.
        
        two parts
        1. find speical sum
            create rightMax interger array of length n
            set rightMax[n-1] = nums[n-1]
            increemnt suffix sum and store max
        
        2. find normal sum
        3. take the max
        
        
        '''
        N = len(nums)
        rightMax = [0]*N
        rightMax[N-1] = nums[N-1]
        suffix_sum = nums[N-1]
        
        #getting max suffix sums
        for i in range(N-2,-1,-1):
            suffix_sum += nums[i]
            rightMax[i] = max(rightMax[i+1],suffix_sum)
        
        
        #kadanes algorithm to find the regular max sum subarray, since this could be an answer
        dp = [0]*N
        dp[0] = nums[0]
        for i in range(1,N):
            dp[i] = max(dp[i-1] + nums[i],nums[i])
        
        #find max sum subarray
        max_sum_subarray = max(dp)
        
        #now find special sum
        special_sum = nums[0]
        pref_sum = 0
        for i in range(N):
            pref_sum += nums[i]
            if i + 1 < N:
                special_sum = max(special_sum,pref_sum + rightMax[i+1])
        
        return max(special_sum,max_sum_subarray)
        

#if we just find the min sum subarray, then our answer is just max(max_sum_subarray, sum(nums) - min subarray)
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        def maximumSubArray(nums):
            ans = nums[0]
            sumSoFar = 0
            for num in nums:
                sumSoFar += num
                ans = max(ans, sumSoFar)
                if sumSoFar < 0:
                    sumSoFar = 0
            return ans
        
        def minimumSubArray(nums):  # the first element and the last element are exclusive!
            if len(nums) <= 2: return 0
            ans = nums[1]
            sumSoFar = 0
            for i in range(1, len(nums) - 1):
                sumSoFar += nums[i]
                ans = min(ans, sumSoFar)
                if sumSoFar > 0:
                    sumSoFar = 0
            return ans
        
        return max(maximumSubArray(nums), sum(nums) - minimumSubArray(nums))

############################################
# 974. Subarray Sums Divisible by K
# 19JAN23
############################################
#literally just modified the solutino from  560. Subarray Sum Equals K
#take modulo instead of difference
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        '''
        we need to borrow the idea from 560. Subarray Sum Equals K
        if the cum_sum up to two indices is the sum:
            i.e pref_sum[:i] == pref_sum[:j]
            then the sum(i:j) is zerp
            
        rather if pref_sum[:i] - pref_sum[:j] = k
        then the sum in between must be k
        so we just look up the complement counts pref_sum_so_far - k
        
        for the divisible by k problem, instead of difference, we can use modulo
        but what does this work?
        
        first:
            we can store prefix sums in a pref_sum array
            then we can find the subarray sum between(i,j), where i < j, and check for divisibility by k
            pref_sum[j] - pref_sum[i] % k == 0, retults in TLE
            
        second:
            examine the eqaulity for an i,j pair, where i < j
            pref_sum[j] - pref_sum[i] % k = 0
            pref_sum[j] % k = pref_sum[i] % k
        
        third:
            we can express any number as: number = divisor*quotient + remainder
            we can also express the prefix_sums in the forms:
                pref_sum[i] = A*k + R1
                pref_sum[j] = B*k + R2
                for some integers A,B,R1,and R2
        fourth:
            we can then right pref_sum[j] - pref_sum[i] = k*(B-A) + (R1 - R0), where the first time is also divisible by k
        
        fifth:
            for the entire espression to be divisble by k, the term (R1 - R0) must be divisible by k
            then we get R1 - R0 = C*k
            R1 = C*k + R0, the values of R0 and R1, will be in the rannge of 0 to k-1, R1 cannot be greater than k
        
        sixth:
            so the only possible value for C is 0, leading to R0 = R1
            If C > 0, then the RHS would be at least k, but as stated the LHS (R1) is between 0 and k - 1.
            
        examples:
            nums = [2,3,4,6,-1,2,3]
            say i = 1, and j = 4, k = 3
            prefSum[i] = 2 + 3 = 5
            prefSum[i] % k = 5 % 3 = 2
            
            prefSum[j] = 2 + 3 + 4 + 6 + -1 = 14
            prefSum[j] % k = 14 % 3 = 2
            
            we have the same prefSum modes, so the subaray from i+1 to j is also divisble by k
            
        '''
        N = len(nums)
        sum_mapp = Counter()
        sum_mapp[0] += 1
        
        count = 0
        curr_sum = 0
        
        for num in nums:
            curr_sum += num
            #find the modulo complement
            if curr_sum % k in sum_mapp:
                count += sum_mapp[curr_sum % k]
            #otherwise put in ther
            sum_mapp[curr_sum % k] += 1
        
        return count

#####################################
# 1025. Divisor Game
# 19JAN23
#####################################
#fuck 
class Solution:
    def divisorGame(self, n: int) -> bool:
        '''
        we can choose any x such that 0 < x < n and n % x == 0
        replace n with n - x
        in order for alice to win, alice must make a move such that bob has no turns the next move
        
        if alice can get two a 2
        
        base case if n == 1:
            return False
        if n == 2:
        return true
        '''
        memo = {}
        
        def dp(i):
            if i == 1:
                return False
            if i == 2:
                return True
            if i in memo:
                return memo[i]
            
            for play in range(1,i):
                if i % play == 0:
                    return dp(i-play)
            return False
                
        
        return dp(n)

#TLE

class Solution:
    def divisorGame(self, n: int) -> bool:
        '''
        say n = 4 alice
        try = [1,2,3], of these only two works
        becomes 2 bob
        try = [1]
        becomes 1 alice no moves
        at each one count the number of moves
        '''
        memo = {}
        
        def dp(i,turns):
            #alice wins if i == 2 and turns are evern
            if i == 2 and turns % 2 == 0:
                return True
            if i == 1 and turns % 2 == 0:
                return False
            
            if (i,turns) in memo:
                return memo[(i,turns)]
            
            ans = 0
            for play in range(1,i):
                if i % play == 0:
                    ans += dp(i-play,turns + 1)
            ans = ans >= 1
            memo[(i,turns)] = ans
            return ans
        
        return dp(n,0)

#correct answer
class Solution:
    def divisorGame(self, n: int) -> bool:
        '''
        say n = 4 alice
        try = [1,2,3], of these only two works
        becomes 2 bob
        try = [1]
        becomes 1 alice no moves
        at each one count the number of moves
        '''
        memo = {}
        
        def dp(i):
            if i == 1:
                return False
            if i == 2:
                return True
            if i in memo:
                return memo[i]
            
            ans = False
            
            for play in range(1,i):
                if i % play == 0:
                    ans = ans or not dp(i-play) #if alice doesn't lose
            
            memo[i] = ans
            return ans
                
        
        return dp(n)

###################################
# 491. Non-decreasing Subsequences
# 20JAN23
##################################
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        '''
        dp on subsets
        if i have mask encoding indices of nums to take from num, which already encoded a non-decreasing sequecen
        then i can add to this sequence of the num to add is greater than or equal to the last number in the sequence
        we dn't want count, we just want to return all possible non-decreasing sequences
       
        doing brute force for now
        '''
        paths = set()
        N = len(nums)
       
        def rec(i,path):
            if len(path) >= 2:
                paths.add(tuple(path[:]))
            #pruning
            if i >= N:
                return
           
            # i need to prune somwhere
            if not path or path[-1] <= nums[i]:
                rec(i+1,path + [nums[i]])
               
            rec(i+1,path)
       
       
        rec(0,[])
        return paths

#we can also use backtracking
#using set in C++ uses redblack tree
#time complexity is O(2^n * n ) or O(2^n * n*n)

class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        result = set()
        sequence = []

        def backtrack(index):
            # if we have checked all elements
            if index == len(nums):
                if len(sequence) >= 2:
                    result.add(tuple(sequence))
                return
            # if the sequence remains increasing after appending nums[index]
            if not sequence or sequence[-1] <= nums[index]:
                # append nums[index] to the sequence
                sequence.append(nums[index])
                # call recursively
                backtrack(index + 1)
                # delete nums[index] from the end of the sequence
                sequence.pop()
            # call recursively not appending an element
            backtrack(index + 1)
        backtrack(0)
        return result


#bit masking review 
#(maks >> position) & 1
class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        '''
        brute force is just subset generation, which will always be 2**n
        then we we just need to check decreasingly in the subset
        '''
        N = len(nums)
        res = set()
        #iterate for all bitmasks for 1 to 2^n - 1
        #we don't want the non empty set for a possible sequence
        for bitmask in range(1, 1 << N):
            #build up sequence
            sequence = []
            for i in range(N):
                #if this bit is set
                if (bitmask >> i) & 1 == 1:
                    sequence.append(nums[i])
            
            if len(sequence) >= 2:
                if all([sequence[j] <= sequence[j+1] for j in range(len(sequence)-1) ]):
                    res.add(tuple(sequence))
        
        return res

#####################################
# 93. Restore IP Addresses (REVISTED) 
# 22JAN23
######################################
#dfs
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        '''
        backtracking, keep track of index and valid number of digits
        we need at least four digits, such that each digit is between 0 and 255 and that there are no leading zero digits
        '''
        ans = []
        i = len(s)
        N = len(s)
        
        
        def backtrack(i,path):
            if i == N and len(path) == 4:
                print(path)
                ans.append(".".join(path)) #convert back to string after
                return
            #try to build digit
            for j in range(3):
                number = s[i:i+j+1]
                #careful for leading zeros
                if len(number) == 1:
                    backtrack(i+j+1,path + [number])
                elif len(number) >= 2 and 0 <= int(number) <= 255 and number[0] != '0':
                    backtrack(i+j+1,path + [number])
                    
        
        backtrack(0,[])
        return ans
    
#backtracking
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        '''
        we can also use backtracking
        go far down as possible, then backtrack
        '''
        
        ips = []
        N = len(s)
        
        def backtrack(start,path):
            if len(path) == 4 and start == N:
                ips.append(".".join(path))
                return
            for size in range(1,4):
                number = s[start:start+size]
                if len(number) > 1 and (number[0] == 0 or int(number) > 255):
                    continue
                if len(path) < 4:
                    path.append(number)
                    backtrack(start+size,path)
                    path.pop()
                    
        
        backtrack(0,[])
        return ips



############################################
# 1533. Find the Index of the Large Integer
# 16JAN23
############################################
# """
# This is ArrayReader's API interface.
# You should not implement it, or speculate about its implementation
# """
#class ArrayReader(object):
#    # Compares the sum of arr[l..r] with the sum of arr[x..y]
#    # return 1 if sum(arr[l..r]) > sum(arr[x..y])
#    # return 0 if sum(arr[l..r]) == sum(arr[x..y])
#    # return -1 if sum(arr[l..r]) < sum(arr[x..y])
#    def compareSub(self, l: int, r: int, x: int, y: int) -> int:
#
#    # Returns the length of the array
#    def length(self) -> int:
#


class Solution:
    def getIndex(self, reader: 'ArrayReader') -> int:
        '''
        if our array contains an even number of elements, we can split the array into two parts and get the sums
        one part should be larger than the other, we move to that part of the array
        
        the issue is when there are an odd number of elements in search, one part could be larger sum than the other, but not necessarily have the largest element
        instead we can keep one element separte then the remaining arrays will become even
        
        for a current seach space that has an odd length, we can break the array into 3 parts
        left,right and extra
        
        now there are three possbilites
        1. if the sum of all eleemnts on left is larger, we can eliminate right and extra
        2. if sume element on left smaller than right, we can elemnt left and extra
        3. if sum of all eleemnts left == right, then the extra element is the larger one
        
        algo:
        * set left = 0, length = reader.legnth
            left if the leftmost index our ours earch space
            the larger integer's index should always be in [left,left + length)
        while length > 1, if it hits zero, we know we haver found the largest index given by left
            reduce search space by half, length // 2
            set cmp = reader.compareSub(left, left + length - 1, left + length, left + length + length - 1)
            if comp = 0
                return left + legnth + lenth, which is the extra selement, case when search space if of odd length
                if cmp is =1, increase left by length, i.e go right
                if cmp = 1, don't do anything,since our left bound stays the same, we just want the lenght half
                    reduce search space by 2
                    i.e length // 2
        '''
        left = 0
        search_space = reader.length()
        
        #while we still hav espace to search
        while search_space > 1:
            search_space //= 2
            
            cmp = reader.compareSub(left, left + search_space - 1,left + search_space,left+2*search_space - 1)
            if cmp == 0:
                #return the extra eleemnt index
                return left + 2*search_space
            #go search right
            if cmp < 0:
                left += search_space
            #if 1, since our left stays the same, and we have reduce search space, we can do nothing on this case
            
        
        return left

#formal binary search
class Solution:
    def getIndex(self, rd: 'ArrayReader') -> int:
        l, r = 0, rd.length() - 1
        while l < r:
            h = (r - l + 1) // 2  # half, h * 2 <= r - l + 1
            if rd.compareSub(l, l + h - 1, l + h, l + h * 2 - 1) != 1:
                l = l + h
            else:
                r = l + h - 1
        return l


####################################
# 131. Palindrome Partitioning (REVISTED)
# 22JAN23
####################################
#barely passes
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        '''
        try all partitions and check that the substring is a palindrome,
        then proceed to add
        '''
        N = len(s)
        valids = []
        
        def backtrack(i,path):
            if i == N:
                valids.append(path[:])
                return
            for step in range(1,N+1):
                substring = s[i:i+step]
                #check
                if substring != '' and substring == substring[::-1]:
                    path.append(substring)
                    backtrack(i+step,path)
                    path.pop()
        
        backtrack(0,[])
        return valids

##################################
# 1592. Rearrange Spaces Between Words 
# 23JAN22
##################################
class Solution:
    def reorderSpaces(self, text: str) -> str:
        '''
        calculate the even spacing
        any extra spaces will be placed at the end
        
        '''
        words = []
        N = len(text)
        spaces = 0
        i = 0
    
        while i < N:
            #if there are spaces
            while i < N and text[i] == ' ':
                spaces += 1
                i += 1
            #if this is a word:
            if i < N and text[i] != ' ':
                j = i
                while j < N and text[j] != ' ':
                    j += 1
                words.append(text[i:j])
                i = j
                
        #calculate even spacing
        even_spaces = spaces //  (len(words)- 1) if len(words) > 1 else 0
        left_over = spaces - even_spaces*(len(words)-1)
        spacing = ""
        for _ in range(even_spaces):
            spacing += " "
        
        return spacing.join(words)+" "*left_over


##########################################
# 1088. Confusing Number II
# 24JAN23
###########################################
#bleaghhhh
class Solution:
    def confusingNumberII(self, n: int) -> int:
        '''
        hint implies backtracking to generate all, since the set of digit used to create is small
        highest number is 10**9, so at most the length of the digits is 9
        
        '''
        rotations = {'0':'0',
                     '1':'1',
                     '6':'9',
                     '9':'6',
                     '8':'8'
                    }
        
        self.ans = 0
        seen = set()
        
        def dp(path):
            if len(path) > 10:
                return
            if path and int("".join(path)) > n:
                return
            #check valid conufsing number
            confusing = []
            for ch in path:
                confusing.append(rotations[ch])
                confusing = confusing[::-1]
            if "".join(confusing) != "".join(path):
                if "".join(path) not in seen:
                    seen.add("".join(path))
                    self.ans += 1
                    return
            
            for possible in ['0','1','6','8','9']:
                path.append(possible)
                dp(path)
                path.pop()

                
        dp([])
        return self.ans

class Solution:
    def confusingNumberII(self, n: int) -> int:
        '''
        hint implies backtracking to generate all, since the set of digit used to create is small
        highest number is 10**9, so at most the length of the digits is 9
        
        we can build up a number, digit by digit using only the valid digits
        when the digit we are building has the same length as n, check valid and return 1
        else return 0
        
        in our recusrive helper keep track of variable smaller
        this represents whether the current number will always be less than n
        for example say the first digit that we are building is 8
        if we use 1 for the first digit, num will always be smaller then s, num < s
        so this number is currenlty smaller than s
        
        
        notes to memo
        we need to see often we query repeated subproblems for this to have any benefit
        '''
        rotations = {'0':'0',
                     '1':'1',
                     '6':'9',
                     '9':'6',
                     '8':'8'
                    }
        
        digits = ['0','1','6','8','9']

        
        s = str(n)
        
        def dfs(smaller,num):
            if len(s) == len(num):
                #count zeros
                i = 0
                while i < len(num) and num[i] == '0':
                    i += 1
                #removing leaidng 0's
                temp = num[i:]
                #check if valid
                for j in range(len(temp)):
                    if temp[j] != rotations[temp[len(temp) -j - 1]]:
                        return 1
                return 0
            
            
            #recursive case
            ans = 0
            for d in digits:
                #if we go over n stop
                if not smaller and d > s[len(num)]:
                    break
                num.append(d)
                ans += dfs(smaller or d < s[len(num)-1],num)
                #backtrack
                num.pop()

            return ans
        
        return dfs(False,[])

########################################
# 909. Snakes and Ladders
# 24JAN23
########################################
#closseeeeee
#idk why this doesn't pass
#but good enough
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        '''
        each move counts as 1, so we can use bfs and return the number of steps to get to the n**2 tile
        it might be easier to conver the board to its number tile and value
        then just use bfs
        
        hardest part is trying to genearte the board tiles in the boustrophedon style
        '''
        N = len(board)
        board_to_tile = {}
        
        tiles = N*N
        #we can just reverse all the odd numbered rows
        for i,row in enumerate(board):
            if i % 2 == 0:
                for space in row:
                    board_to_tile[tiles] = space
                    tiles -= 1
            else:
                row = row[::-1]
                for space in row:
                    board_to_tile[tiles] = space
                    tiles -= 1
        
        
        #bfs
        q = deque([(1,0)])
        seen = set()
        
        while q:
            tile, steps = q.popleft()
            if tile == N*N:
                return steps
            
            seen.add(tile)
            
            #otherwise neighboard search
            for neigh in range(tile + 1, min(tile + 6,N**2) + 1):
                #not visited
                if neigh not in seen:
                    #move to that tile
                    if board_to_tile[neigh] == -1:
                        q.append((neigh,steps+1))
                    #take the ladder
                    else:
                        if board_to_tile[neigh] not in seen:
                            q.append((board_to_tile[neigh],steps+1))
        
        return -1

#yessss
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        n = len(board)
        n_squared = n ** 2
        def convert_to_index(pos: int) -> tuple:
            # Needs input validation for an interview.
            pos -= 1 # Convert to zero-based.
            starts_left: bool = (((pos // n) % 2) == 0)
            i = (n_squared - pos - 1) // n
            j = (pos % n) if starts_left else n - (pos % n) - 1
            return (i, j)
        
                #bfs
        q = deque([(1,0)])
        seen = set([1])
        
        while q:
            tile, steps = q.popleft()
            if tile == n_squared:
                return steps

            #otherwise neighboard search
            for neigh in range(tile + 1, min(tile + 6,n**2) + 1):
                #not visited
                if neigh not in seen:
                    seen.add(neigh)
                    #get the corresponding index
                    i,j = convert_to_index(neigh)
                    if board[i][j] == -1:
                        q.append((neigh,steps+1))
                    else:
                        q.append((board[i][j],steps+1))
                        
        
        return -1

#keep dist array
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        
        '''
        instead of seen set, we can keep dist array for all cells (tiles)
        then just update the dist array
        and return dist array at destination
        
        '''
        n = len(board)
        n_squared = n ** 2
        def convert_to_index(pos: int) -> tuple:
            # Needs input validation for an interview.
            pos -= 1 # Convert to zero-based.
            #check if this positions starts left
            starts_left: bool = (((pos // n) % 2) == 0)
            #rows will be multiple of n
            i = (n_squared - pos - 1) // n
            #cols is just the remainder
            j = (pos % n) if starts_left else n - (pos % n) - 1
            return (i, j)
        
        dist = [-1]*(n_squared+1)
        #base case, we don't need to travel a distace at the start
        dist[1] = 0
        q = deque([1])
        
        
        while q:
            curr = q.popleft()
            for neigh in range(curr + 1, min(curr + 6,n**2) + 1):
                #convert to index
                i,j = convert_to_index(neigh)
                destination = board[i][j] if board[i][j] != -1 else neigh
                #havent gotten here in a short path
                if dist[destination] == -1:
                    dist[destination] = dist[curr] + 1
                    q.append((destination))
                    
        return dist[n_squared]
        
        

#dijkstras
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        '''
        we can also use Dijkstra's
        in the pq, keep (min_dist,to_cell)
        the only catch is that when we pop from the min_heap (priority queue) we first check that min_dist != dist[curr]
        if they are unequal the distance is outdated
        '''
        n = len(board)
        n_squared = n ** 2
        def convert_to_index(pos: int) -> tuple:
            # Needs input validation for an interview.
            pos -= 1 # Convert to zero-based.
            #check if this positions starts left
            starts_left: bool = (((pos // n) % 2) == 0)
            #rows will be multiple of n
            i = (n_squared - pos - 1) // n
            #cols is just the remainder
            j = (pos % n) if starts_left else n - (pos % n) - 1
            return (i, j)
        
        dist = [-1]*(n_squared + 1)
        dist[1] = 0
        
        pq = [(0,1)] #min dist, tile
        while pq:
            curr_dist, curr_tile = heapq.heappop(pq)
            if dist[curr_tile] < curr_dist:
                continue
            for neigh in range(curr_tile + 1, min(curr_tile + 6,n**2) + 1):
                #convert to index
                i,j = convert_to_index(neigh)
                destination = board[i][j] if board[i][j] != -1 else neigh
                #if not visited or is smaller
                if dist[destination] == -1 or dist[curr_tile] + 1 < dist[destination]:
                    dist[destination] = dist[curr_tile] + 1
                    heapq.heappush(pq,(dist[destination],destination))
        
        return dist[n*n]

#############################################
# 2359. Find Closest Node to Given Two Nodes
# 25JAN22
#############################################
#yasssss
class Solution:
    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
        '''
        we are given a directed graph (it may contain cycles)
        we want to retun the index of the node that can be reached from both node1 and node2, such that the maximum
        distance from node1 to that node AND from node2 to that node is minimized
        if there are multiple answers with the sam maximum, return the smallest index
        
        make the graph
        us bfs to find the shortest distances for both node1 and node2 to all the nodes in the graph
        then iterate over all the nodes to find the maximum distance
        '''
        #make graph
        graph = defaultdict(list)
        N = len(edges)
        
        for i in range(N):
            if edges[i] == -1:
                continue
            graph[i].append(edges[i])
            
        def bfs(start_node,graph):
            #peform bfs from start node and return distances array
            dist = [-1]*(N)
            #initialize
            dist[start_node] = 0
            #no need for dijkstra's since its only 1 step away
            q = deque([start_node])
            
            while q:
                curr_node = q.popleft()
                for neigh in graph[curr_node]:
                    if dist[neigh] == -1:
                        dist[neigh] = dist[curr_node] + 1
                        q.append((neigh))
            return dist
        
        node1_dists = bfs(node1,graph)
        node2_dists = bfs(node2,graph)
        
        #find index, this is probably the crux of the problem
        #sort after
        candidates = []
        for i in range(N):
            #this node cannot be reached from either node1 or node2
            if node1_dists[i] == -1 or node2_dists[i] == -1:
                continue
            #get min distance to this node
            min_distance = max(node1_dists[i],node2_dists[i])
            #pair with index
            entry = (min_distance,i)
            candidates.append(entry)
        
        #sort
        if not candidates:
            return -1
        candidates.sort(key = lambda x: (x[0],x[1]))
        return candidates[0][1]
            

#the last part is how to efficietly search ofr the maximum of the minimum
#just take the max of the distances and go in increasing order of index
class Solution:
    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
        '''
        we are given a directed graph (it may contain cycles)
        we want to retun the index of the node that can be reached from both node1 and node2, such that the maximum
        distance from node1 to that node AND from node2 to that node is minimized
        if there are multiple answers with the sam maximum, return the smallest index
        
        make the graph
        us bfs to find the shortest distances for both node1 and node2 to all the nodes in the graph
        then iterate over all the nodes to find the maximum distance
        '''
        #make graph
        graph = defaultdict(list)
        N = len(edges)
        
        for i in range(N):
            if edges[i] == -1:
                continue
            graph[i].append(edges[i])
            
        def bfs(start_node,graph):
            #peform bfs from start node and return distances array
            dist = [-1]*(N)
            #initialize
            dist[start_node] = 0
            #no need for dijkstra's since its only 1 step away
            q = deque([start_node])
            
            while q:
                curr_node = q.popleft()
                for neigh in graph[curr_node]:
                    if dist[neigh] == -1:
                        dist[neigh] = dist[curr_node] + 1
                        q.append((neigh))
            return dist
        
        node1_dists = bfs(node1,graph)
        node2_dists = bfs(node2,graph)
        
        #find index, this is probably the crux of the problem
        min_dist_node = -1
        min_allowed = float('inf')
        
        for i in range(N):
            #this node cannot be reached from either node1 or node2
            if node1_dists[i] == -1 or node2_dists[i] == -1:
                continue
            #if either of the max is smaller
            if max(node1_dists[i],node2_dists[i]) < min_allowed:
                min_dist_node = i
                min_allowed = max(node1_dists[i],node2_dists[i])
        
        return min_dist_node


#we can also use dfs twice
class Solution:
    def closestMeetingNode(self, edges: List[int], node1: int, node2: int) -> int:
        '''
        we can also use dfs
        the problem is unique in that each node has at most 1 outoing edge
        we only need to check the the single neighbor
        '''
        #make graph
        graph = defaultdict(list)
        N = len(edges)
        
        for i in range(N):
            if edges[i] == -1:
                continue
            graph[i].append(edges[i])
            
        def dfs(node,dist,seen,graph):
            #visit
            seen.add(node)
            neigh = graph[node]
            if not neigh:
                return
            if neigh[0] not in seen:
                dist[neigh[0]] = 1 + dist[node]
                dfs(neigh[0],dist,seen,graph)
                
        seen1 = set()
        seen2 = set()
        node1_dists = [-1]*N
        node2_dists = [-1]*N
        
        node1_dists[node1] = 0
        node2_dists[node2] = 0
        
        dfs(node1,node1_dists,seen1,graph)
        dfs(node2,node2_dists,seen2,graph)
        
        #find index, this is probably the crux of the problem
        min_dist_node = -1
        min_allowed = float('inf')
        
        for i in range(N):
            #this node cannot be reached from either node1 or node2
            if node1_dists[i] == -1 or node2_dists[i] == -1:
                continue
            #if either of the max is smaller
            if max(node1_dists[i],node2_dists[i]) < min_allowed:
                min_dist_node = i
                min_allowed = max(node1_dists[i],node2_dists[i])
        
        return min_dist_node

##################################################
# 787. Cheapest Flights Within K Stops (REVISTED)
# 26JAN22
###################################################
#dynamic programming, top down
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        '''
        first try dynamic programming
        i will be from j will be to
        
        let dp(i,j,k) be the cheapest price staring from node i, ending at node k, with k transactions
        
        so we want min(dp(src,target,k),dp(src,target,k-1)...dp(src,target,0))
        we could just loop k from 0 to k and take the min(dp,src,target,k)
        
        now for the state transition
        
        dp(i,j,k) = take the minimum edege
        
        wtf????
        
        first part was wrong
        we define dp(i,k) which give us the minimum cheaptest price for leaving node i with k tranactions
        if node == dst, we dont need to take any fares
        if we are out of k transactions, return larger number
        then we need to minimize
            we nee to tak the smallest cost for all neighbors from this need
        '''
        #make graph
        #for an i,j element, read as from, to and store the ticket price
        graph = [[0]*n for _ in range(n)]
        
        for u,v,price in flights:
            graph[u][v] = price
            
        memo = {}
        
        def dp(node,k): #returns cheapest fair with k transactions
            if node == dst:
                return 0
            if k < 0:
                return float('inf')
            
            if (node,k) in memo:
                return memo[(node,k)]
            
            ans = float('inf')
            #we want the smallest outgaing edge
            for neigh in range(n):
                if graph[node][neigh] == 0:
                    continue
                #get child anser, recurse to get the anwer
                child_ans = dp(neigh,k-1)
                #minimize
                ans = min(ans, graph[node][neigh] + child_ans)
            
            memo[(node,k)] = ans
            return ans
        
        ans = dp(src,k)
        return ans if ans != float('inf') else -1



#bfs
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        '''
        typically we cannot use BFS in an weighted graph
        recall the property of BFS is taht the first time a node si reached during the traversal, it must be the minimum
        but we cannot assume that the first node we visit is the minimum, which usually implies using dijkstras
        
        however, we are limited by the number of stops k,
        we can do bfs in k layers, and stop after k+1
        
        while doing bfs, only update the dist arrays when going down this node results in a cheaper fare
        '''
        #adj list node: [neigh,cost]
        adj_list = defaultdict(list)
        for u,v,price in flights:
            adj_list[u].append((v,price))
            
        costs = [float('inf')]*n
        costs[src] = 0
        
        q = deque([(src,0)])
        
        #while there is q and we still have k
        while q and k >= 0:
            m = len(q)
            #explore frontier
            for _ in range(m):
                curr, curr_price = q.popleft()
                #optimize
                #if costs[curr] < curr_price:
                #    continue
                for neigh,price in adj_list[curr]:
                    if curr_price + price < costs[neigh]:
                        costs[neigh] = curr_price + price
                        q.append((neigh,costs[neigh]))
            k -= 1
            
        return costs[dst] if costs[dst] != float('inf') else -1

#dijkstras
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        '''
        we can use dikstras, maintin min heap of distances
        and continue when we are out of transactions
        '''
        #adj list node: [neigh,cost]
        adj_list = defaultdict(list)
        for u,v,price in flights:
            adj_list[u].append((v,price))
            
        costs = [float('inf')]*n
        costs[src] = 0
        visited = set()
        
        pq = [(0,src,k)]
        
        #while there is q and we still have k
        while pq:
            curr_dist,curr_node,curr_k = heapq.heappop(pq)
            if k <= 0:
                continue
            if costs[curr_node] < curr_dist:
                continue
            #mark
            visited.add(curr_node)
            for neigh,price in adj_list[curr_node]:
                if neigh in visited:
                    continue
                new_dist = costs[curr_node] + price
                #update
                if new_dist < costs[neigh]:
                    costs[neigh] = new_dist
                    heapq.heappush(pq,(new_dist,neigh,curr_k - 1))
            
        return costs[dst] if costs[dst] != float('inf') else -1

#another dijkstras way
class Solution:
    def findCheapestPrice(self, n, flights, src, dst, k):
        visited = {}
        adj = defaultdict(list)
        for s, d, p in flights:
            adj[s].append((d, p))
        pq = [(0, 0, src)]
        while pq:
            cost, stops, node = heapq.heappop(pq)
            if node == dst and stops - 1 <= k:
                return cost
            if node not in visited or visited[node] > stops:
                visited[node] = stops
                for neighbor, price in adj[node]:
                    heapq.heappush(pq, (cost + price, stops + 1, neighbor))
        return -1

#####################################
# 472. Concatenated Words
# 27JAN23
#####################################
#brute force
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        '''
        input will never have more than 30 words
        the entire length of the words will not be more than 10**5
        
        brute force is just check if each word can be made from at least two shorter words
        use python contains, and check if it contains that word and that word is smaller
        
        just clear the replace the word with "" and see if we get the empty string
        '''
        ans = []
        
        for word in words:
            count = 0
            copy_word = word
            for other_word in words:
                if other_word != word:
                    if word.__contains__(other_word) and len(other_word) < len(word):
                        count += 1
                        copy_word = copy_word.replace(other_word,"")
            
            if count >= 2 or copy_word == "":
                ans.append(word)
        
        return answer

#fuck this god damn piece of shit....
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        '''
        this is a famous reachabilit problem, and it can be solve by DP or DFS/BFS
        if a given word can be created by concatenating the given words, we can split it into two parts,
        one prefix and suffix,
        
        prefix is a shorted word which can be reached by concatenating the given words and the suffix is the given word
        
        rather, word = (another shorter word that can be creatted by conat) + (a given word in the dictionary)
        we can enumerate the suffix and look it up in the dictionary, and prefix part is just a subproblem
        
        for the dp solution, this is just one step further from wordbreak
        adopt the dp strategy from word break
        
        for a word, we define dp(i) is whether word[:i] (the prefix) can be made through concaentation
        and where word[i+1:] is in the dictionary
        
        dp array should be length + 1, and base case is dp[0] = true, which means the empty string can be always be made into a prefix from no words in the dictionary
        
        we need to calculate do(i) for i in range(0,len(word))
        
        now consider dp[i] for i > 0
        
        if dp[i] is true, we can split the words into s[:i] and s[i+1:], where the first part can be created by the words in dictionary, and the maing suffix is just a single word in the dictionary
        
        dp[i] is true if and only if there is an integer j, such that 0 <= j < i and the word's substring (index range [j, i - 1]) is in the dictionary.
        
        corner case:
            when i == length, since we don't want to use the word in the dictionary directly, we should check 1 <= j < i instead.
            
        algo:
            Put all the words into a HashSet as a dictionary.
            Create an empty list answer.
            For each word in the words create a boolean array dp of length = word.length + 1, and set dp[0] = true.
            For each index i from 1 to word.length, set dp[i] to true if we can find a value j from 0 (1 if i == word.length) such that dp[j] = true and word.substring(j, i) is in the dictionary.
            Put word into answer if dp[word.length] = true.
            After processing all the words, return answer.
            
        '''
        all_words = set(words)
        res = []
        #integer break solutions
        def dp(i,word,memo):
            if i == 0:
                return True
            if i in memo:
                return memo[i]
            
            ans = False
            for j in range((i == N) and 1 or 0,i):
                if not dp(i-1,word,memo):
                    ans = dp(j,word,memo) and word[j:i] in all_words
            
            memo[i] = ans
            return ans
        
        for word in words:
            N = len(word)
            memo = {}
            print(dp(N,word,memo))

#dp class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        '''
        this is a famous reachabilit problem, and it can be solve by DP or DFS/BFS
        if a given word can be created by concatenating the given words, we can split it into two parts,
        one prefix and suffix,
        
        prefix is a shorted word which can be reached by concatenating the given words and the suffix is the given word
        
        rather, word = (another shorter word that can be creatted by conat) + (a given word in the dictionary)
        we can enumerate the suffix and look it up in the dictionary, and prefix part is just a subproblem
        
        for the dp solution, this is just one step further from wordbreak
        adopt the dp strategy from word break
        
        for a word, we define dp(i) is whether word[:i] (the prefix) can be made through concaentation
        and where word[i+1:] is in the dictionary
        
        dp array should be length + 1, and base case is dp[0] = true, which means the empty string can be always be made into a prefix from no words in the dictionary
        
        we need to calculate do(i) for i in range(0,len(word))
        
        now consider dp[i] for i > 0
        
        if dp[i] is true, we can split the words into s[:i] and s[i+1:], where the first part can be created by the words in dictionary, and the maing suffix is just a single word in the dictionary
        
        dp[i] is true if and only if there is an integer j, such that 0 <= j < i and the word's substring (index range [j, i - 1]) is in the dictionary.
        
        corner case:
            when i == length, since we don't want to use the word in the dictionary directly, we should check 1 <= j < i instead.
            
        algo:
            Put all the words into a HashSet as a dictionary.
            Create an empty list answer.
            For each word in the words create a boolean array dp of length = word.length + 1, and set dp[0] = true.
            For each index i from 1 to word.length, set dp[i] to true if we can find a value j from 0 (1 if i == word.length) such that dp[j] = true and word.substring(j, i) is in the dictionary.
            Put word into answer if dp[word.length] = true.
            After processing all the words, return answer.
            
        '''
        all_words = set(words)
        res = []
        #integer break solutions
        for word in words:
            N = len(word)
            dp = [False]*(N+1)
            dp[0] = True
            
            for i in range(1,N+1):
                for j in range((i ==N) and 1 or 0,i):
                    if not dp[i]:
                        dp[i] = dp[j] and word[j:i] in all_words
            
            if dp[N]:
                res.append(word)
        
        return res
            

#dfs
#we can just check prefix and suffix in the dictionary of all words
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        '''
        we can turn the problem into a reachability problem
        for eafh word, construct graph with all prefixes as nodes
        
        the graoh contains (len(word) + 1 nodes) for each word
        for edges, consider 2 prefixes (i,j) such that 0 <= i < j < len(word)
        if prefix j can be created by concatenatins prefix i and a word in the dictionary, we add an edge from node i to j
        
        When i = 0, we require j < word.length as there should be an edge from node 0 to node word.length. Determining whether a word can be created by concatenating 2 or more words in the dictionary is the same as determining whether there is a path from node 0 to node word.length in the graph.


        
        '''
        d = set(words)
        memo = {}

        #dfs returns whether a word can be formed from the concatneatino of other words in dictionary
        #if we split word into prefix and suffix (examine) all possible
        #if prefix and suffix can be formed, must be tree
        #if prefix can made and dp(suffix) must bet rue
        #if suffix can be made and dp(prefix) must also be true
        def dfs(word):
            if word in memo:
                return memo[word]
            memo[word] = False
            for i in range(1, len(word)):
                prefix = word[:i]
                suffix = word[i:]
                if prefix in d and suffix in d:
                    memo[word] = True 
                    break
                if prefix in d and dfs(suffix):
                    memo[word] = True 
                    break
                if suffix in d and dfs(prefix):
                    memo[word] = True 
                    break
            return memo[word] 
        return [word for word in words if dfs(word)] 


#another dfs solution
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        dictionary = set(words)
        result = []
        for word in words:
            visited = [False] * len(word)
            if self.dfs(word, 0, visited, dictionary):
                result.append(word)
        return result
        
    def dfs(self, word: str, length: int, visited: List[bool], dictionary: set) -> bool:
        if length == len(word):
            return True
        if visited[length]:
            return False
        visited[length] = True
        for i in range(len(word) - (1 if length == 0 else 0), length, -1):
            if word[length:i] in dictionary and self.dfs(word, i, visited, dictionary):
                return True
        return False

#another dfs solution
class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
        visited_indices: set[int] = set()
        words_set: set[str] = set(words)
        def dfs_other_words(word: str, start_idx: int):
            n = len(word)
            if start_idx == n:
                # Got past the end - the word is made of multiple words.
                return True
            if start_idx in visited_indices:
                # We already searched from this index.
                return False
            
            visited_indices.add(start_idx)
            # Go up til the last character in the word if we are starting
            # from the first index. Otherwise we'll mark single words as results.
            for i in range(start_idx + 1, (n + 1 if start_idx != 0 else n)):
                if (word[start_idx:i] in words_set
                    and dfs_other_words(word, i)):
                    return True
            
            return False

        result: list[str] = []
        for word in words:
            visited_indices.clear()
            if dfs_other_words(word, 0):
                result.append(word)
        
        return result



