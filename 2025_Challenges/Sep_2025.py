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