#################################################
# 2071. Maximum Number of Tasks You Can Assign
# 01MAY25
#################################################
class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        '''
        in order to complete task[i], workers[j] >= task[i]
        we can use pills only once per worker, and it increease this workes strength by strength
        we wou;ndt want to use a stronger work to finish a task, if there is a weaker worker that can do it
        so save the strongest workes, 
        sort workers and tasks increasinly?
        '''
        tasks.sort()
        workers.sort()
        i,j = 0,0
        print(tasks)
        print(workers)
        
        while i < len(tasks) and j < len(workers):
            #worker is strong enough
            if workers[j] >= tasks[i]:
                i += 1
                j += 1
            #not strong enough, but we have pill to use to make it strong enough
            elif pills > 0 and workers[j] + strength >= tasks[i]:
                pills -= 1
                i += 1
                j += 1
            #try using stonger worker
            else:
                j += 1
        
        return i
    
#gahhh
class Solution:
    def maxTaskAssign(self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        '''
        can't just pair smallest taks with smallest works, the reason is becaue of pill strength
        we can't priortize using a pill if a worker isn't strong enough
        so we shold save pills after using our strongest workers
        '''
        tasks.sort()
        workers.sort()
        i,j = 0,0

        #first pass, dont use pulls, then keep track of unised workers that can use pill
        unused_workers = [] #by index
        while i < len(tasks) and j < len(workers):
            #worker is strong enough
            if workers[j] >= tasks[i]:
                i += 1
                j += 1
            #try using stonger worker
            else:
                unused_workers.append(workers[j])
                j += 1
        unused_workers.sort(reverse = True)
        print(unused_workers)
        print(tasks[i:])
        #second pass?
        j = 0
        
        while i < len(tasks) and j < len(unused_workers):
            if pills > 0 and unused_workers[j] + strength >= tasks[i]:
                pills -= 1
                i += 1
                j += 1
            #try using stonger worker
            else:
                j += 1
        
        return i

from sortedcontainers import SortedList
class Solution:
    def maxTaskAssign(
        self, tasks: List[int], workers: List[int], pills: int, strength: int) -> int:
        '''
        binarry search, but maintain sortedset (list) when searching
        sortedlist with workers
        case 1:
            the worke with the highest available value is greater than or equal to the task value
            in this case, we do not need to use a pill
            we can assign this worker (with the maximum value) to task and remove from pool or workers
            why? since this is the most difficult task, any worker who can complete it, can comlete any otehr task
            so assign strongest worker available to the hargest task
        
        case 2:
            no woker can complete the task without a pill
            in this case we must use a pill
            look for the weakest worker who can complete the task with the pill (i.e worker with value >= t - strength)
            and remove them from pool
        
        iterate tasks and worrkers in increasing order
        '''
        n, m = len(tasks), len(workers)
        tasks.sort()
        workers.sort()

        def check(mid: int) -> bool:
            p = pills
            # Ordered set of workers
            #we need the m-mid worrkes from the end, for a given min
            ws = SortedList(workers[m - mid :])
            # Enumerate each task from largest to smallest
            for i in range(mid - 1, -1, -1):
                # If the largest element in the ordered set is greater than or equal to tasks[i]
                if ws[-1] >= tasks[i]:
                    ws.pop()
                else:
                    #no pills, cant doo
                    if p == 0:
                        return False
                    #use pill to find the worker that can do it
                    #ie look for tasks[i] - strength
                    rep = ws.bisect_left(tasks[i] - strength)
                    #can't do
                    if rep == len(ws):
                        return False
                    #use pill
                    p -= 1
                    #remove this worker
                    ws.pop(rep)
            #can do
            return True

        left, right, ans = 1, min(m, n), 0
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1

        return ans

#################################################
# 3343. Count Number of Balanced Permutations
# 12MAY25
################################################
import math
class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        '''
        count number and then distribute them between odd and even positions usinng dp
            need to check all numbes from 0 to 9, and add them up
        given number 000111122, 
        we have 4 even positions and 5 odd positions
        to have it balance, we need 8/2 == 4 for sum of each spot

        0/3:
            There are comb(5, 3) = 10 ways to put 3 zeros into 5 odd positions.
            The result is 10 * dp(0 + 1, 4 - 0, 5 - 3, 4 - 0).
        1/2:
            There are comb(4, 1) = 4 ways to put 1 zero into 4 even positions.
            There are comb(5, 2) = 10 ways to put 2 zeros into 5 odd positions.
            The result is 4 * 10 * dp(1, 3, 3, 4).
        2/1:
            There are comb(4, 2) = 6 ways to put 2 zeros into 4 even positions.
            There are comb(5, 1) = 5 ways to put 1 zero into 5 odd positions.
            The result is 6 * 5 * dp(1, 2, 4, 4).
        3/0:
            There are comb(4, 3) = 4 ways to put 3 zeros into 4 even positions.
            The result is 4 * dp(1, 1, 5, 4).
            We continue the recursion to distribute 1 and 2. After distributing all numbers, we return 1 if the balanced sum is zero, and 0 otherwise.


        '''
        cnt = Counter(int(ch) for ch in num)
        total = sum(int(ch) for ch in num)

        @cache
        def dfs(curr_digit, odd, even, balance):
            if odd == 0 and even == 0 and balance == 0:
                return 1
            #possbile pruning herre
            if curr_digit > 9 or odd < 0 or even < 0 or balance < 0:
                return 0
            res = 0
            for num_times in range(0, cnt[curr_digit] + 1):
                ways_odd = math.comb(odd,num_times)
                ways_even = math.comb(even, cnt[curr_digit] - num_times)
                #if we place odd with j occurence, we place cnt[i] - j occurrences
                #for balace, we using the current digit j times
                #go up to the next digit, reduce odd positions, reduce even positionss, update balance
                #state definition is (odd spots, even sports, number, and balance)
                prev_ways = dfs(curr_digit + 1, odd - num_times, even - (cnt[curr_digit] - num_times), balance - curr_digit * num_times)
                res += ways_even * ways_odd * prev_ways
            return res % 1000000007

        return 0 if total % 2 else dfs(0, len(num) - len(num) // 2, len(num) // 2, total // 2)