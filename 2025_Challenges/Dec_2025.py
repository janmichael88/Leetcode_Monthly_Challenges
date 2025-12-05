############################################
# 3623. Count Number of Trapezoids I
# 02DEC25
#############################################
#TLE
#problem is picking two distinct groups
class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        '''
        group by y coordinate
        then chose two distinct gruos and from each group select two points to make a trapezoid
        remember the trapezoid coords need to be unique
        '''
        mapp = defaultdict(list)
        for x,y in points:
            mapp[y].append((x,y))
        
        groups = [k for k,v in mapp.items()]
        ans = 0
        mod = 10**9 + 7
        n = len(groups)
        for i in range(n):
            for j in range(i+1,n):
                #how many ways can i pick 2 points from groups[i], nCr
                ways1 = math.comb(len(mapp[groups[i]]),2)
                ways2 = math.comb(len(mapp[groups[j]]),2)
                ans += ways1*ways2
                ans %= mod
        
        return ans
    
class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        '''
        group by y coordinate
        then chose two distinct gruos and from each group select two points to make a trapezoid
        remember the trapezoid coords need to be unique
        can't pick two distinct groups, take too long, use sums
        oh whoops you can't just combine the groups into one

        this is a cool trick for computing pairwise products (which would be n*n)
        but in linear time
        suppose you have A: [x0, x1, x2]
res = x0 * x1 + x0 * x2 + x1 * x2 (with 2 nested for loops, O(n^2))

now in O(n)
init: res = 0, tot = 0
for i in 0...n
i = 0: res += tot * x0 ==> res = 0; tot += x0 ==> tot = x0;
i = 1: res += tot * x1 ==> res = x0 * x1; tot += x1 ==> tot = x0+x1;
i = 2: res += tot * x2 ==> res = x0 * x1 + (x0+x1) * x2 ==> res = x0 * x1 + x0 * x2 + x1 * x2
tot += x2 ==> tot = x0 + x1 + x2

I hope it's a little bit clear now

And for this problem, you can also use the suffix sum of number of paires within the same group if you don't know how to compute res in O(n)
        better way to see it
        res += x[i]*(sum of all previous values)
        
        '''
        mapp = Counter()
        n = len(points)
        for x,y in points:
            mapp[y] += 1
        
        ans = 0
        total_sum = 0
        mod = 10**9 + 7
        for k,v in mapp.items():
            first_group = v
            ways1 = math.comb(first_group, 2)
            ans += ways1*total_sum
            ans %= mod
            total_sum += ways1 % mod
        
        return ans % mod
    
class Solution:
    def countTrapezoids(self, points: List[List[int]]) -> int:
        '''
        group by y coordinate
        then chose two distinct gruos and from each group select two points to make a trapezoid
        remember the trapezoid coords need to be unique
        can't pick two distinct groups, take too long, use sums
        oh whoops you can't just combine the groups into one

        for linear trick
        say we have [a,b,c,d] and we want [a*b + a*c + a*d + b*c + b*d + c*d]
        so it just a*sum(b,c,d) + b*sum(c,d) * c*sum(d)
        the sums are just suffix sums!, we can also so it that way too
        '''
        mapp = Counter()
        n = len(points)
        for x,y in points:
            mapp[y] += 1
        
        ans = 0
        total_sum = 0
        mod = 10**9 + 7
        for k,v in mapp.items():
            first_group = v
            ways1 = math.comb(first_group, 2)
            ans += ways1*total_sum
            ans %= mod
            total_sum += ways1 % mod
        
        return ans % mod
    
#############################################
# 2211. Count Collisions on a Road
# 04DEC25
##############################################
#almost
class Solution:
    def countCollisions(self, directions: str) -> int:
        '''
        if R -> <- L
            thats two collisions, and they become S at the point of collision
            collisions += 1
        
        if R -> S
            thats one, and the stay there
        
        if S <- L
            thats one, and they stay there
        
        use stack
        '''
        ans = 0
        stack = []
        for ch in directions:
            if ch == 'R':
                stack.append('R')
            elif ch == 'L':
                if stack:
                    if stack[-1] == 'R':
                        ans += 2
                        stack.pop()
                        stack.append('S')
                    elif stack[-1] == 'S':
                        ans += 1
                        stack.pop()
                        stack.append('S')
                else:
                    stack.append('L')
            elif ch == 'S':
                if stack:
                    if stack[-1] == 'R':
                        ans += 1
                        stack.pop()
                        stack.append('S')
                else:
                    stack.append('S')
        
        #since we started going right, we might have SRRSS, we need to tally up these
        new_stack = []
        for ch in stack:
            if ch == 'S':
                while new_stack and new_stack[-1] == 'R':
                    ans += 1
                    new_stack.pop()
                new_stack.append(ch)
            else:
                new_stack.append(ch)
        print(new_stack)
        return ans
    
class Solution:
    def countCollisions(self, directions: str) -> int:
        '''
        travere left to right and use flag to record status of vehicls on left
        if there are no vehicles on left side or all vehilces on left are moving left, set to 1
        if a collision occuret on the left and vehicales stop flag is 0
        if there are consecutive vehiblces on left moving right, flag counts vehicles moving left
        '''
        res = 0
        flag = -1

        for c in directions:
            if c == "L":
                if flag >= 0:
                    res += flag + 1
                    flag = 0
            elif c == "S":
                if flag > 0:
                    res += flag
                flag = 0
            else:
                if flag >= 0:
                    flag += 1
                else:
                    flag = 1
        return res
    
class Solution:
    def countCollisions(self, directions: str) -> int:
        '''
        remove left moving on left side and right moving in rihgt side
        everything inside should collide exactly one
        '''
        directions = directions.lstrip("L").rstrip("R")
        return len(directions) - directions.count("S")