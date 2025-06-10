#############################################
# 2929. Distribute Candies Among Children II
# 01JUN25
############################################
class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        '''
        (0,0,0) as states, then try adding 1 at each step
        for one child i, that child can have [0 to min(lmit,n)]
        the second child j can get [0 to limit] and (i + 1) <= n
        the third child k will get (n-i-j) and should have 0 <= n - i - j <= limit
        for each i we know:
            max(0, n - i - limit) <= j <= min(limit, n - i) in order for the third child to have some
            so the number of solutions for this i (i.e the count for the first child is)
            min(limit, n - i) - max(0, n - i - limit)
        
        0 <= i <= min(limit, n)
        0 <= j <= limit
        i + j <= n
        0 <= n - i - j <= limit
        '''
        ways = 0
        for i in range(min(n,limit)+1):
            ways += max(min(limit, n - i) - max(0, n - i - limit) + 1, 0)
        
        return ways
    
class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        '''
        we try every possible number of candies i in range(min(n,limit))
        then we check
        n-i > limit*2, if this happens then either of the two remaining children will have more than limit candies
            this can't be
        if n-1 <= limit*2
            second child can have eat least max(0,n-i-limit) candies
            third child can have at most min(limit,n-i)
        '''
        ans = 0
        for i in range(min(limit, n) + 1):
            if n - i > 2 * limit:
                continue
            ans += min(n - i, limit) - max(0, n - i - limit) + 1
        return ans
    



class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        '''
        def cal(x):
            if x < 0:
                return 0
            return x * (x - 1) // 2
        '''
        return (
            self.cal(n + 2)
            - 3 * self.cal(n - limit + 1)
            + 3 * self.cal(n - (limit + 1) * 2 + 2)
            - self.cal(n - 3 * (limit + 1) + 2)
        )
    def cal(self,x):
        if x < 0:
            return 0
        return x * (x - 1) // 2
    
####################################################
# 2927. Distribute Candies Among Children III
# 02JUN25
####################################################
#combinatorics
class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        '''
        we can treat this as bars and starts first
        its just nCk
        where n is the number of candies and k is the number of dividers
        i.e n balls into k bins
        its just n+k-1 choose (k-1)
        here k == 2
        answer is:
            total_number_unrestricted - at_least_one_more_than_limit + at_least_two_more_than_limit - all_three_more_than_limit

        we need to add bacck in count for at_least_two 
        for no limit
            its the sam s placing two dividers among n cadies
            so its 2 choose(n+2)

        Total number of unrestricted distributions
        * distributing n candies among 3 children is equivalent to placing two dividers among n candies to split them into three groups.
        * 2 choose (n+2)

        At least one child receives more than limit candies: (add back in)
        * We give limit + 1 candies to one child first, reducing the problem to distributing n−(limit+1) candies 
        * among 3 children (with possible zeroes). There are 3 choices for which child gets the extra candies,
        3* 2 choose (n - limit + 1) + 2

        At least two children receive more than limit candies:
        * We give limit + 1 candies to any two children, reducing the problem to distributing 
        * n−2×(limit+1) candies among 3 children. There are 3 ways to choose the two children:
        3* 2 choose (n - 2*(limit + 1)) + 2

        All three children receive more than limit candies:
        We give limit + 1 candies to each child, so we're left with n−3×(limit+1) candies to distribute among 3 children.
        3 * 2 choose(n-3*(limit+1)) + 2
        '''
        return (
            self.cal(n + 2)
            - 3 * self.cal(n - limit + 1)
            + 3 * self.cal(n - (limit + 1) * 2 + 2)
            - self.cal(n - 3 * (limit + 1) + 2)
        )
    def cal(self,x):
        if x < 0:
            return 0
        return x * (x - 1) // 2
    
##############################################
# 1298. Maximum Candies You Can Get from Boxes
# 03JUN25
##############################################
#not quite
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        '''
        graph problem, theme this week is candy
        1 is opened 0 is closed
        we can open a box with the right key
        bfs, but only enqueu the boexes for which we have keys for
        mark boxes for which are open and for which we have keys for, then just sum them at the end
        when we have a key then change status and open
        '''
        n = len(status)
        have_box = [False]*n
        count = 0
        q = deque([])
        for b in initialBoxes:
            have_box[b] = True
            q.append(b)
        
        while q:
            curr_box = q.popleft()
            for next_box in containedBoxes[curr_box]:
                have_box[next_box] = True
                q.append(next_box)
                #now check keys
                for k in keys[next_box]:
                    status[k] = 1
            for k in keys[curr_box]:
                status[k] = 1

        for i in range(n):
            if have_box[i] and status[i] == 1:
                count += candies[i]
        return count
    
#this one is the update version
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        n = len(status)
        q = deque()
        visited = [False] * n
        for box in initialBoxes:
            q.append(box)
            visited[box] = True

        while q:
            topBox = q.popleft()
            for key in keys[topBox]:
                if topBox != key:
                    status[key] = 1
            for nextBox in containedBoxes[topBox]:
                if not visited[nextBox]:
                    q.append(nextBox)
                    visited[nextBox] = True

        ans = 0
        for i in range(n):
            if status[i] == 1 and visited[i]:
                ans += candies[i]
        return ans
    
#this is trick queue, not really bfs
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        '''
        queue of boxes, of box is open take candy
        then use keys to unlock new boxes
        if we can't open any, then requeeu
        we need a state to checek if we've opened any new boxes otherwise get out of search
        '''
        res = 0
        boxes = deque(initialBoxes)
        while boxes:
            updated = False    # To check if simulation stops or not
            N = len(boxes)
            for i in range(N) :
                box = boxes.popleft()   # process the leftMost box
                if status[box] == 1 :
                    res += candies[box]
                    updated = True
                    boxes.extend(containedBoxes[box])     # take Contained boxes of box
                    for key in keys[box]:                 # take keys from box
                        status[key] = 1
                #we need to keep requeing until we can open the box
                #this is just horriblyy worded
                else:
                    boxes.append(box)  # take it in queue back
            if not updated:
                break
        return res
            
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        '''
        queue of boxes, of box is open take candy
        then use keys to unlock new boxes
        if we can't open any, then requeeu
        we need a state to checek if we've opened any new boxes otherwise get out of search
        '''
        ans = 0
        q = deque(initialBoxes)

        while q:
            #need to keep trak that we can continue taking boxes
            N = len(q)
            for _ in range(N):
                curr_box = q.popleft()
                can_take = False
                #if open take candies
                if status[curr_box] == 1:
                    ans += candies[curr_box]
                    can_take = True
                    #add in boxes
                    for next_box in containedBoxes[curr_box]:
                        q.append(next_box)
                    
                    #use keys to open
                    for k in keys[curr_box]:
                        status[k] = 1
                else:
                    q.append(curr_box)
            if not can_take:
                break
        return ans
            
#using dfs
class Solution:
    def maxCandies(self, status: List[int], candies: List[int], keys: List[List[int]], containedBoxes: List[List[int]], initialBoxes: List[int]) -> int:
        '''
        we can use dfs too, just recurse on all outging boxes and on keys
        keep track of seen, and unlockable
        '''
        seen = set()
        can_look = set()

        def dfs(box):
            if box in seen:
                return 0

            if status[box] == 0:
                can_look.add(box)
                return 0

            seen.add(box)
            total = candies[box]

            for next_box in containedBoxes[box]:
                total += dfs(next_box)

            for next_box in keys[box]:
                status[next_box] = 1
                if next_box in can_look:
                    total += dfs(next_box)

            return total


        ans = 0
        for b in initialBoxes:
            ans += dfs(b)
        
        return ans
    
#################################################################
# 3403. Find the Lexicographically Largest String From the Box I
# 04JUN25
#################################################################
#TLE
class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        '''
        the say rounds to make you think you need to try all possible splits
        but really we  just want the largest string of all splits
        let n = len(word)
        there will be numFriends splits
        so the size of the string would be n - numFriends + 1
        so this for every index i
        
        '''
        if numFriends == 1:
            return word
        ans = ""
        n = len(word)
        k = n - numFriends + 1
        for i in range(n):
            sub = ""
            j = i
            while j < n and len(sub) < k:
                sub += word[j]
                j += 1
                ans = max(ans,sub)
        return ans

#jessus  
class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        '''
        the say rounds to make you think you need to try all possible splits
        but really we  just want the largest string of all splits
        let n = len(word)
        there will be numFriends splits
        so the size of the string would be n - numFriends + 1
        so this for every index i
        damn the later problems (like 3000+ are trickier)
        '''
        if numFriends == 1:
            return word
        ans = ""
        n = len(word)
        k = n - numFriends + 1
        for i in range(n):
            #we are starting at every i, and the substring can be <= k
            #so just take the minimum
            sub = word[i:min(i + k,n)]
            ans = max(sub,ans)
        return ans
    
######################################################
# 3474. Lexicographically Smallest Generated String
# 05JUN25
######################################################
#close... T.T
class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        '''
        if str1[i] is T, we need to use a substring from str2 here
        so whenver this is T, we need to fix this part, otherwise we are free to use any letter to make it smaller
        the first example makes sense
        the second one

        '''
        n = len(str1)
        m = len(str2)
        k = n + m - 1
        word = [""]*k
        #fix the first part
        for i in range(n):
            if str1[i] == 'T':
                #make == to st2
                for j in range(m):
                    word[i+j] = str2[j]
        #for an empty spots, fill with 'a'
        for i in range(k):
            if word[i] == '':
                word[i] = 'a'
        #then check at the true positions, it matches str2, if it doesn't return an empty string ""
        for i in range(n):
            if str1[i] == 'T':
                #validate
                substring = word[i:i+m]
                if "".join(substring) != str2:
                    return ""
        return "".join(word)
    
####################################################################
# 2434. Using a Robot to Print the Lexicographically Smallest String
# 06JUN25
####################################################################
#fucking hell man...
class Solution:
    def robotWithString(self, s: str) -> str:
        '''
        operatios are:
            remove first character from s and append to t (first s -> end to t)
            remove last character from t, and write on paper (end from t -> first to p)
        return smalles string that can be written
        this is weird because we can have an empty t too, we aren't just shuffling
            need to get the smallest possible char to the last place in t,
            then we can write to p
        
        how do i know when to write to p from t, i need to know before hand if there was something smaller
        work backwards from s?
        '''
        ans = []
        n = len(s)
        used = [False]*n
        counts = Counter(s)
        for i in range(26):
            ch = chr(ord('a') + i)
            not_curr_chars = []
            #need the last occurence of the current char
            for j in range(n):
                #no more of current char
                if counts[ch] == 0:
                    break
                #is current char
                elif not used[j] and s[j] == ch:
                    ans.append(ch)
                    counts[ch] -= 1
                    used[j] = True
                #not current char
                elif not used[j] and s[j] != ch:
                    not_curr_chars.append(s[j])
                    used[j] = True
            #write in reverse
            if not_curr_chars:
                for ch in not_curr_chars[::-1]:
                    ans.append(ch)
            print(ans)
        return "".join(ans)

class Solution:
    def robotWithString(self, s: str) -> str:
        '''
        im stupid, its just a stack, 
        we iterate s, and push to stack t
        we keep doing this until we find the smallest character remaining in s
        starting from a
        '''
        counts = Counter(s)
        t  = []
        p = []
        min_ch = 'a'

        for ch in s:
            #puch to t and decrement count
            t.append(ch)
            counts[ch] -= 1
            #maintain the current min char remaing in s
            while min_ch < 'z' and counts[min_ch] == 0:
                min_ch = chr(ord(min_ch) + 1)
            while t and t[-1] <= min_ch:
                p.append(t.pop())
        
        return "".join(p)

#############################################################
# 3170. Lexicographically Minimum String After Removing Stars
# 07JUN25
#############################################################
class Solution:
    def clearStars(self, s: str) -> str:
        '''
        iterate s, whenever we see a *, we need to delete it the smallest non star to its left
        do we delete one for multiple of the smallest
        its if there are ties for the smallest non * to the left
        aaba*
        1. a
        2. aa
        3. aab
        4. aaba
        5. aab
        but could we have made aba, no, since aab < aba
        the answer for cases like aaaaa*
        if just aaaa (star + 1 removes a)
        the problem is figuring out which smallest non * to remove 
        for each star * delete 1
            it doesnt mean delete multiple smallest non * for a singel *, just that there could be multiple *
            you need to delete the smallest non *, but priotrize deleting the right most ones
            store entries in heap as (char,index), use negative index
            then rebuild after
        '''
        n = len(s)
        min_heap = []
        for i,ch in enumerate(s):
            idx = -(i + 1)
            if ch == '*':
                heapq.heappop(min_heap)
            else:
                heapq.heappush(min_heap, (ch,idx))
        
        #sort and rebuild
        temp = []
        for ch,idx in min_heap:
            temp.append((ch,-idx))
        
        temp.sort(key = lambda x: x[1])
        ans = []
        for ch,idx in temp:
            ans.append(ch)
        return "".join(ans) 

class Solution:
    def clearStars(self, s: str) -> str:
        '''
        dont need to sort after, just replace index with ""
        '''
        n = len(s)
        ans = list(s)
        min_heap = []
        for i,ch in enumerate(s):
            if ch == '*' and min_heap:
                del_char, del_idx = heapq.heappop(min_heap)
                ans[-del_idx] = ""
                ans[i] = ""
            else:
                heapq.heappush(min_heap, (ch,-i))
        
        return "".join(ans) 

#instead of using min_heap look through all chars in order
#and keep hashmap to char
class Solution:
    def clearStars(self, s: str) -> str:
        '''
        dont need to sort after, just replace index with ""
        '''
        n = len(s)
        ans = list(s)
        mapp = defaultdict(list)
        for i,ch in enumerate(s):
            if ch == "*":
                for j in range(26):
                    check_ch = chr(ord('a') + j)
                    if len(mapp[check_ch]) > 0:
                        del_idx = mapp[check_ch].pop()
                        ans[del_idx] = ""
                        ans[i] = ""
                        break
            else:
                mapp[ch].append(i)

        return "".join(ans) 

########################################################
# 440. K-th Smallest in Lexicographical Order (REVISTED)
# 09JUN25
#########################################################
class Solution:
    def findKthNumber(self, n: int, k: int) -> int:
        '''
        need to traverse n-ary tree (10-ary tree) and keep count of the numbers seen at each node
        problem is that we can choose the endpoint n
        if the number of elements at some node exceeds n, we can't descend into that node
        for the count
        1
        ├── 10
        │   ├── 100
        │   ├── ...
        ├── 11
        ...
        curr is start of current prefix and next_ is sibling in level
        '''
        #print(self._count(11,1))
        curr = 1
        k -= 1
        while k > 0:
            #count number of nodes at this node for numbers between curr and n
            steps = self._count(n,curr)
            #if we have enough steps in this node, we go on to the next number in sequence
            #stay on this level, and use up steps on this level from k
            if steps <= k:
                curr += 1
                #used up step
                k -= steps
            else:
                #go down a level, curr*10 is the next node in order, i.e we are past this current level
                curr *= 10
                k -= 1
        
        return curr
    
    #count number of nums in pre-order at some noe
    #i.e, given some number curr in 10-ary tree, count numbers less then <= n
    def _count(self,n,curr):
        #counting in between curr and next_ up to n
        next_ = curr + 1
        count = 0
        while curr <= n:
            count += min(n + 1, next_) - curr
            curr *= 10
            next_ *= 10
        return count

###############################
# 2376. Count Special Integers
# 09JUN25
################################
#digit dp
class Solution:
    def countSpecialNumbers(self, n: int) -> int:
        '''
        this is digit dp, but we dont need to create a range and do inclusion exclusion
        https://leetcode.com/problems/count-special-integers/solutions/2422121/python-digit-dp/
        use n as string for its digits
        then we can use position i and check if we are tight to this bound at the position i in number n
        basic structure
        dp(pos, tight, leading_zero, other_states)
        pos = current digit position (from left to right)

        tight = whether the current number is tight to the upper bound (i.e., we haven’t exceeded N's prefix)

        leading_zero = whether we are still placing leading zeros

        other_states = problem-specific info (e.g., previous digit, sum of digits, etc.)

        Transition
        At each position, try all digits from 0 to 9 unless tight restricts you (you must stay ≤ N[pos]).

        for d in range(start, upper_bound + 1):
            if d >= prev_digit:
                dp(pos + 1, d, new_tight)

                new_tight becomes tight and (d == N[pos])

        for problem: "Count numbers from 1 to N where the digits are non-decreasing."
        from functools import lru_cache

        def count_non_decreasing(N: int) -> int:
            digits = list(map(int, str(N)))

            @lru_cache(None)
            def dp(pos: int, prev: int, tight: bool) -> int:
                if pos == len(digits):
                    return 1  # valid number

                limit = digits[pos] if tight else 9
                total = 0
                for d in range(prev, limit + 1):
                    total += dp(pos + 1, d, tight and (d == limit))
                return total

            return dp(0, 0, True)
        '''
        
        digits = [int(x) for x in str(n)] #use for digits
        memo = {}

        def dp(pos,tight,leading_zeros,mask):
            #valid number
            if pos == len(digits):
                return 1
            
            key = (pos,tight,leading_zeros,mask)
            if key in memo:
                return memo[(key)]
            
            if tight == 0:
                limit = digits[pos] #can only use up to this digit
            else:
                limit = 9 #no restriction

            count = 0
            for next_digit in range(limit+1):
                next_tight = tight or next_digit < limit
                next_leading = leading_zeros or (next_digit > 0)
                #leading zeros
                if next_leading:
                    #next digit is avaialble
                    if not mask & (1 << next_digit):
                        count += dp(pos + 1, next_tight, next_leading, mask | (1 << next_digit))
                else:
                    #skip
                    count += dp(pos + 1, next_tight, next_leading, mask)
            memo[key] = count
            return count
        
        return dp(0,0,0,0) - 1
