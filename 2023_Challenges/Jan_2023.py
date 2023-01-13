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
