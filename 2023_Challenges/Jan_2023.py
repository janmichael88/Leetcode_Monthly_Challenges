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
        '''
        count = 0
        i = 1
        while i <= n:
            divider = i*10
            count_at_pos = (n // divider)*i + min(max(n % divider - i + 1,0),i)
            count += count_at_pos
            i = i*10
        
        return count

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
