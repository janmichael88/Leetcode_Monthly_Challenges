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