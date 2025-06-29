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

##############################################################
# 3442. Maximum Difference Between Even and Odd Frequency I
# 10JUN25
###############################################################
class Solution:
    def maxDifference(self, s: str) -> int:
        '''
        count of counts,
        find a1 and a2
        oh whoops need maximum difference, 
        '''
        counts = Counter(s)
        a1 = 0
        a2 = float('inf')
        for k,v in counts.items():
            if v % 2 == 1:
                a1 = max(a1,v)
            else:
                a2 = min(a2,v)
        return a1 - a2
    
################################################################
# 3445. Maximum Difference Between Even and Odd Frequency II
# 11JUN25
###############################################################
#copy paste T.T
class Solution:
    def maxDifference(self, s: str, k: int) -> int:
        '''
        char a must have odd frequency
        char b must have even frequency
        need to track pairs for '01234'
            we need to fix an a and b, and try them all
            when a appears odd times, we need its max occruence, and we need b's min occurence when even
            same in reverse
        https://leetcode.com/problems/maximum-difference-between-even-and-odd-frequency-ii/?envType=daily-question&envId=2025-06-11
        need to keep track of:
        1. occruence of a given pair of characters
        2. track occurence and difference
        3. retreive proper pairty pair to calculate max difference

        imagine instead of 4 characters, we only have 0 and 1
        we need to find the max difference when 0 has odd freq and 1 has even freq
        or when 1 has odd free and 0 has even freq
        
        we don't care about what the numbers are exactly, its just their parities

        for fixed a and b, find the maximum difference
        let frequence of char in sum substring[i...j] be count(char)
        so we have:
            freq[a] in s[i...j] = count(a, j) - count(a, i-1)
            freq[b] in s[i...j] = count(b, j) - count(b, i-1)

        we want to maximize freq[a] - freq[b]
            (count(a, j) - count(a, i-1)) - (count(b, j) - count(b, i-1))
        so we have
            (count(a, j) - count(b, j)) - (count(a, i-1) - count(b, i-1))
        
        so to maximize the total difference ending at k we need to find a starting point j-1, that minimizes the term (count(a, i-1) - count(b, i-1))
            store minimum on the left!
        
        parity difference
            odd - even = odd
            even - odd = odd
            odd - odd = even
            even - even = even

        So, for freq[a] to be odd, the parities of count(a, j) and count(a, i-1) must be different. 
        For freq[b] to be even, the parities of count(b, j) and count(b, i-1) must be the same.


        '''
        ans = float('-inf')
        for a in "01234": 
            for b in "01234": 
                if a != b: 
                    seen = defaultdict(lambda : inf)
                    #this is pref_sum of counts for an a and b
                    pa = [0]
                    pb = [0]
                    left = 0 
                    for right, ch in enumerate(s):
                        #keep track of pref_counts for the current a and b 
                        if ch == a: 
                            pa.append(pa[-1] + 1)
                            pb.append(pb[-1])
                        elif ch == b: 
                            pb.append(pb[-1] + 1)
                            pa.append(pa[-1])
                        else:
                            pa.append(pa[-1])
                            pb.append(pb[-1])
            
                        #sliding window loop invariant, while we have enough k, shrink
                        #check current pairty for start of string for a and b are different
                        #for count(a) to be oadd the aprities of count(a, at right) and count(a, left) must be differrent
                        #what would be the case if pa[left] == pa[right] and pb[left] == pb[right]?
                            #it means there's no count in between! so we need to move our left point
                            #there needs to be at least non zero occurence of a and b for this pref/substring
                        while right - left + 1 >= k and pa[left] != pa[-1] and pb[left] != pb[-1]:
                            #store minimum difference for the parity status on the left 
                            key = (pa[left] % 2, pb[left] % 2) 
                            diff = pa[left] - pb[left]
                            seen[key] = min(seen[key], diff)
                            left += 1
                        #complement look up, kinda like subarray sum == k
                        #don't store minimum, need to look for the opposite status of the current parity pair that's on the left
                        #check, loook for complement parity previously seen to satisfy constraint, i.e
                        #so we have parity (1,1) for right, if we look for (0,1) on the left, this would give us (1,0)
                        #if we have parity (0,1), we look for (1,1) on the left, this would give us (1,0)
                        #if we have parity (1,0), we look for (0,0) on the left, this would gives is (1,0) again
                        #if we have parity (0,0), we look for (1,0) on the left, this gives us (1,0)
                        key = (1 - pa[-1] % 2, pb[-1] % 2) 
                        if key in seen:
                            diff = pa[-1] - pb[-1]
                            ans = max(ans, diff - seen[key])
        return ans 
    
################################################
# 2566. Maximum Difference by Remapping a Digit
# 14JUN25
################################################
class Solution:
    def minMaxDifference(self, num: int) -> int:
        '''
        there must be a greedy solution
        we need to find the maximum digit bob can make by remapping a single digit
        and we need to find the minimum
        '''
        digits = str(num)
        n = len(digits)
        #find the first non nine digit
        first_non_nine = digits[0]
        for i in range(n):
            if digits[i] != '9':
                first_non_nine = digits[i]
                break
        
        first_non_zero = digits[0]
        for i in range(n):
            if digits[i] != '0':
                first_non_zero = digits[i]
                break
        
        max_number = 0
        for d in digits:
            max_number *= 10
            if d == first_non_nine:
                max_number += 9
            else:
                max_number += int(d)
        
        min_number = 0
        for d in digits:
            min_number *= 10
            if d == first_non_zero:
                min_number += 0
            else:
                min_number += int(d)
        
        return max_number - min_number
    
class Solution:
    def minMaxDifference(self, num: int) -> int:
        '''
        find the most siginicatn digit from the left that is 9 and change all its occruences to 9, this will make it maximum
        to make it mimum, take the frist non zero digit from the left and make all its occurences 0
        since the first digit isnt'a zero, use that occurences
        '''
        max_ = str(num)
        min_ = str(num)
        pos = 0

        while pos < len(max_) and max_[pos] == '9':
            pos += 1
        
        if pos < len(max_):
            max_ = max_.replace(max_[pos],'9')
        
        min_ = min_.replace(min_[0],'0')
        return int(max_) - int(min_)
    
###########################################################
# 1432. Max Difference You Can Get From Changing an Integer
# 16JUN25
###########################################################
class Solution:
    def maxDiff(self, num: int) -> int:
        '''
        we can pick a digit from num and change all its occurrences to another chosen digit
        do this to make the max number and min number, and return max diff difference
        there can be no leading zero's,
        sine there are only two digits, try all of them and take max
        '''
        num = str(num)
        max_num = int(num)
        min_num = int(num)
        for i in range(10):
            for j in range(10):
                changed = self.change(num,str(i),str(j))
                max_num = max(max_num,changed)
                min_num = min(min_num,changed)
        
        return max_num - min_num
    def change(slef,num,x, y):
        new_num = num.replace(x,y)
        if new_num.startswith('0'):
            return int(num)
        return int(new_num)

#same as previous problem, but careful with leading zeros, we need to make sure we have found at least one number
class Solution:
    def maxDiff(self, num: int) -> int:
        '''
        max num is easy, just make the most signigicant digit that isn't a 9 and make all its occruences 9
        for min number, we need to make we don't replace a number with a leading zero
        replace most signiciant bit digit with 1
        or find high-order digit that != the highest digit and replace with zero
        '''

        max_num = str(num)
        min_num = str(num)

        for i,ch in enumerate(max_num):
            if ch != '9':
                max_num = max_num.replace(ch,'9')
                break
        
        for i, digit in enumerate(min_num):
            if i == 0:
                if digit != "1":
                    min_num = min_num.replace(digit, "1")
                    break
            else:
                if digit != "0" and digit != min_num[0]:
                    min_num = min_num.replace(digit, "0")
                    break

        return int(max_num) - int(min_num)

########################################################
# 2016. Maximum Difference Between Increasing Elements
# 16JUN25
#######################################################
class Solution:
    def maximumDifference(self, nums: List[int]) -> int:
        '''
        for some num[i], we need to find the largest num[j] that is on its right
        similar to finding ramp
        keep track of min while traversing, and if its smaller than nums[i] we can do it
        '''
        ans = float('-inf')
        curr_min = float('inf')
        for num in nums:
            if num > curr_min:
                ans = max(ans, num - curr_min)
            curr_min = min(curr_min,num)
        
        if ans == float('-inf'):
            return -1
        
        return ans
    
######################################################################
# 3405. Count the Number of Arrays with K Matching Adjacent Elements
# 17JUN25
#######################################################################
import math
class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        '''
        this is a combinatorics problem, need number of arrays of size n,
        where each element is in between [1,m], and there k elements that are next wo each other and are equal
        need to be exactly k indices, where each index i, and arr[i-1] == arr[i]
            numbers dont need to be the same, just the indices, if we want the k - 1 indices to be the same
            we could just slide k numbers across the arrayy, and this can be done n - k times
            if we were to split up k, then we would neet at least 2 spots where i and i+1
        
        we need k indicies, and there are exactly k-1 adjacent indices to place the same number
        the first index, index 0, we no adjacent (i.e no 0 - 1), so it can be of any m numbers
        for the middle part, we have (n-1) potential spotrs to place k adjcanet indices
            this is comb(n-1,k)
        for the remaining (n - 1 - k) spots, we are free to chose any (m-1) numbers
            this is (m-1)**(n-k-1)
        
        its really just three parts
        [ways to pick first number] * [ways to place (n-1) numbers in k indices] * [ways to place (m-1) numbers in (n-1-k) spots]
        first part is m*(n-1)choose(x)*(the other parts that don't need to have arr[i-1] != arr[i])
        ans is just m*comb(n-1,k)*(m-1)**(n-k-1)
        '''
        mod = 10**9 + 7
        return m*math.comb(n-1,k)*self.fast_pow(m-1,n-k-1) % mod
    
    def fast_pow(self,base,power):
        if power == 0:
            return 1
        half_power = self.fast_pow(base,power//2)
        if power % 2 == 0:
            return half_power*half_power
        return base*half_power*half_power

import math
class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        '''
        pre-computing inverfactorial using modular multiplactive inverse
        '''
        self.MOD = 10**9 + 7
        self.MX = 10**5

        self.fact = [0] * self.MX
        self.inv_fact = [0] * self.MX
        
        def qpow(x, n):
            res = 1
            while n:
                if n & 1:
                    res = res * x % self.MOD
                x = x * x % self.MOD
                n >>= 1
            return res
        
        self.fact[0] = 1
        for i in range(1, self.MX):
            self.fact[i] = self.fact[i - 1] * i % self.MOD
        self.inv_fact[self.MX - 1] = qpow(self.fact[self.MX - 1], self.MOD - 2)
        for i in range(self.MX - 1, 0, -1):
            self.inv_fact[i - 1] = self.inv_fact[i] * i % self.MOD

        def comb(n, m):
            return self.fact[n] * self.inv_fact[m] % self.MOD * self.inv_fact[n - m] % self.MOD



        return comb(n - 1, k) * m % self.MOD * qpow(m - 1, n - k - 1) % self.MOD

#########################################################
# 3567. Minimum Absolute Difference in Sliding Submatrix
# 19JUN25
########################################################
class Solution:
    def minAbsDiff(self, grid: List[List[int]], k: int) -> List[List[int]]:
        '''
        this is enumerration
        for each (i,j) look at its k x k submatrix, and recorde the minimum absolute value between any two distance values
        issue is getting data in right format
        '''
        m,n = len(grid),len(grid[0])
        ans = [[float('-inf')]*(n - k + 1) for _ in range(m - k + 1)]
        for i in range(m-k+1):
            for j in range(n-k+1):
                #look at the submatrix
                #need two closet values, if theres only one, its zero
                #otherwise sort and scan adjacent elements abs diff
                values = set()
                for ii in range(i,i+k,1):
                    for jj in range(j,j+k,1):
                        values.add(grid[ii][jj])
                values = list(values)
                values.sort()
                if len(values) == 1:
                    ans[i][j] = 0
                else:
                    temp = float('inf')
                    for idx in range(1,len(values)):
                        temp = min(temp, values[idx] - values[idx-1])
                    ans[i][j] = temp
        return ans
    
##########################################################
# 2294. Partition Array Such That Maximum Difference Is K
# 19JUN25
###########################################################
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        '''
        we need to partition nums into subsequences, such that the difference between the max and min values in each subsequence is at most k
        return min number of subsequences
        what if we first sort
        [1, 2, 3, 5, 6], k = 2
        sort and check current min and max, if we go past 2, increment into a new subsequnece
        '''
        nums.sort()
        n = len(nums)
        ans = 1
        curr_min = nums[0]
        for i in range(1,n):
            if nums[i] - curr_min > k:
                ans += 1
                curr_min = nums[i]
        return ans
    
#####################################################
# 3443. Maximum Manhattan Distance After K Changes
# 20JUN25
###################################################
class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        '''
        we can only change at most k characters, 
            could be none at all
        check all directions if they live in NE,NE,SE,SW
        if a direction in s i in the checking quadrant, it contributes to the overall manahat distance
        otherwise if have k to negate it, we do
        if we don't we decreement
        '''
        max_dist = 0
        #check quads
        for quad in ['NE','NW','SE','SW']:
            curr_dist = 0
            curr_k = k
            for d in s:
                if d in quad:
                    curr_dist += 1
                else:
                    if curr_k > 0:
                        curr_dist += 1
                        curr_k -= 1
                    else:
                        curr_dist  -= 1
                max_dist = max(max_dist,curr_dist)
        
        return max_dist
    
class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        '''
        overll manhat distance from origin would be
        abs(count(N) - count(S)) + abs(count(E) - count(W))
        keep track of the counts of N,S,E,W
        then we need to figure out how to use k at each step
        for a direction, N <-> S or E <-> W, modify the smaller of them wiith k, this would extend in that direction
        then try doing it with E W
        '''
        max_dist = 0
        counts = Counter()
        for d in s:
            counts[d] += 1
            #these count the modifications, applied to each direcition
            #these add to the distance
            n_to_s_mods = min(counts['N'],counts['S'],k)
            e_to_w_mods = min(counts['E'],counts['W'], k - n_to_s_mods)
            #get new manhate dist
            new_n_to_s = abs(counts['N'] - counts['S']) + 2*n_to_s_mods
            new_e_to_w = abs(counts['E'] - counts['E']) + 2*e_to_w_mods
            max_dist = max(max_dist, new_n_to_s + new_e_to_w)
        
        return max_dist
        
###################################################
# 3085. Minimum Deletions to Make String K-Special
# 22JUN25
###################################################
class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        '''
        for all (i,j) indices in the string, such that (i,j)
        we need |freq(word[i]) - freq(word[j])| <= k 
        return minimum number of characters i need to delete to make k-special
        k can b <= 0
        start with the smallest character, then go through the counts and delete if we have to
        basically we need to make all abs_diff(pairwise counts) in word <= k 
        '''
        #check all a-z pairs
        counts = Counter(word)
        #fix character for all  charaterrs in counts,
        #and assume this to be the smallest, normalize to the minimum
        ans = float('inf')
        for char in counts:
            min_count = counts[char]
            deletions = 0
            for other_char in counts:
                if counts[other_char] < min_count:
                    #the current min_count is the smallest
                    deletions += counts[other_char]
                #reduce so that abs(counts[other_count] - min_count) <= k
                elif counts[other_char] - min_count > k:
                    deletions += counts[other_char] - min_count - k
            #assuming the current char wast the min to be reduced too
            ans = min(deletions,ans)
        
        return ans
    
###############################################
# 2138. Divide a String Into Groups of Size k
# 22JUN25
###############################################
class Solution:
    def divideString(self, s: str, k: int, fill: str) -> List[str]:
        '''
        just fill the ending with x
        '''
        n = len(s)
        ans = []
        for i in range(0,n,k):
            ans.append(s[i:i+k])
        
        ans[-1] += fill*(k - len(ans[-1]))
        
        return ans
    
########################################
# 2081. Sum of k-Mirror Numbers
# 23JUN25
#########################################
#generating palindromes in base 10 will TLE
class Solution:
    def kMirror(self, k: int, n: int) -> int:
        '''
        checkin base k is easy,just do modk and //k
        n is in between 1 and 30, so at must there can be 30 k-mirror numbers
        so generate them in order, first start with numbers [1,9]
        start with 1, we can do 11,
        crux of the problem is making palindromes in order
        we could introduce 12, becase on the next level we could get 121, but there are more base-10 palindromes < 121
        this is probably the harder of the two problems
        1,111,121,131....191
        2,212,222,232...292
        mirror half digits
        if we are given a d digit number, we can make a 2d digit number by doing num + num[::-1]
        we can get a 2d-1 digit number by doing num + num[-2::-1]
        '''
        #print(self.base_k(151,3))
        nums = []
        q = deque(list('0123456789'))
        #for each number mirror, then try playing digits 0 to 9, and half mirrow
        while q:
            N = len(q)
            is_broken = False
            for _ in range(N):
                curr = q.popleft()
                if curr == '0':
                    continue
                int_num = int(curr)
                if self.check(int_num,10) and self.check(int_num,k):
                    nums.append(int_num)
                    if len(nums) >= n:
                        is_broken = True
                        break
                #even palidrome
                even_pal = curr + curr[::-1]
                #print(even_pal)
                q.append(even_pal)
                #then fix digit and mirror half
                for d in '0123456789':
                    temp = curr + d
                    next_mirror = temp + temp[-2::-1]
                    q.append(next_mirror)
            if is_broken:
                break
        return sum(nums)

    def check(self,n,k):
        ans = []
        while n:
            ans.append(n % k)
            n //= k
        
        return ans == ans[::-1]
    
class Solution:
    def kMirror(self, k: int, n: int) -> int:
        '''
        checkin base k is easy,just do modk and //k
        n is in between 1 and 30, so at must there can be 30 k-mirror numbers
        so generate them in order, first start with numbers [1,9]
        start with 1, we can do 11,
        crux of the problem is making palindromes in order
        we could introduce 12, becase on the next level we could get 121, but there are more base-10 palindromes < 121
        this is probably the harder of the two problems
        1,111,121,131....191
        2,212,222,232...292
        mirror half digits
        if we are given a d digit number, we can make a 2d digit number by doing num + num[::-1]
        we can get a 2d-1 digit number by doing num + num[-2::-1]
        How can be sure that there will b n k-palindrome numbers in the range of base 10 numbers from [1 to 10**62]?
        '''
        def gen():
            '''
            generate for value with different length
            when i == 0: num：[1, 10)
            size of num: 1, 2 -> 1 or 11
            when i == 1: [10, 100)
            size of num: 3, 4 -> 10 or 101
            when i == 2: [100, 1000)
            size of num: 5, 6 -> 10001 or 100001
            
            we hae full coverage at 6, resulting in 1999998 numbers
            the last being: 999999999999
            but really this will take at most
                6*(10**6 - 10**5) time, which is allowable (10**6 upper bound by LC)
            
            '''
            #we have full coverage at 6
            for i in range(6):
                #odd
                for num in range(10**i, 10**(i+1)):
                    s = str(num) + str(num)[::-1][1:]
                    yield int(s)
                #even
                for num in range(10**i, 10**(i+1)):
                    s = str(num) + str(num)[::-1]
                    yield int(s)
        '''
        temp = []
        for num in gen():
            temp.append(num)
        print(len(temp))
        print(temp[-1])
        '''
        ans = 0
        for num in gen():
            if self.check(num,k):
                ans += num
                n -= 1
            if n == 0:
                break
        return ans

    def check(self,n,k):
        ans = []
        while n:
            ans.append(n % k)
            n //= k
        
        return ans == ans[::-1]
    
################################################
# 2200. Find All K-Distant Indices in an Array
# 23JUN25
#################################################
class Solution:
    def findKDistantIndices(self, nums: List[int], key: int, k: int) -> List[int]:
        '''
        we need to examine only the indices in nums, where a number is k
        for each index i where nums[i] == key,
            grab all the ones k away from it, sort
            i can grab the intervals
        '''
        intervals = []
        n = len(nums)
        for i,num in enumerate(nums):
            if num == key:
                left = max(0,i-k)
                right = min(n-1,i+k)
                if not intervals:
                    intervals.append([left,right])
                elif intervals[-1][0] <= left <= intervals[-1][1]:
                    intervals[-1][1] = max(intervals[-1][1],right)
                else:
                    intervals.append([left,right])
        ans = []
        for l,r in intervals:
            for k in range(l,r+1):
                ans.append(k)
        
        return ans
    
#########################################
# 683. K Empty Slots
# 24JUN25
#########################################
from sortedcontainers import SortedList
class Solution:
    def kEmptySlots(self, bulbs: List[int], k: int) -> int:
        '''
        brute force would be simulate the days, and searach in the binary array where  there are k bulbs turned off
        need an effcient way to check if there are k bublbs off between two bulbs that are on
        maintaine sortedlist and add in, meaning we've turned on the bulb
        then look to its neighbord left and right and check that there are  bulbs in between them
        '''
        on = SortedList([])
        for i,b in enumerate(bulbs):
            on.add(b)
            #find position in range
            pos = on.bisect_left(b)
            #pos = on.index(b)
            #either works
            if pos - 1 >= 0 and on[pos] - on[pos-1] == k+1:
                print(on[pos],on[pos-1])
                return i+1
            if pos + 1 < len(on) and on[pos+1] - on[pos] == k + 1:
                return i+1

        return -1
    
################################################
# 2040. Kth Smallest Product of Two Sorted Arrays
# 25JUN25
#################################################
#clever counting, and binary search
class Solution:
    def kthSmallestProduct(self, nums1: List[int], nums2: List[int], k: int) -> int:
        '''
        if all the numbers are positive, then we could use the merge two sorted arrays approacch
        pick the smaller of two, and we can advance it by len(nums1) or len(nums2)
        now if we have negative number, and we keep multiplying it by largest numbers (that are still postive)
        this makes it increasing
        four cases
        nums1 | nums2
         pos  | pos
         pos  | neg
         neg  | pos
         neg  | neg
        
        binary search on the answer between -10**5 and 10**5
        we need to count the number of elements <= some number, call it mid
        if count < k, need to move up
        if count > k, move number down
        so how can we count number of products (given nums[1] and nums[2]) that are less than a number mid
        fix number of num1s, call it n1
        if n1 >= 0, then the array of product with nums2 is increasing, so we can use binary search
        if n1 < 0, the array is decreasing, so we want from the end
        if n1 == 0, then all values in nums2*n1 are == 0,
            update only if mid >= 0

        clever counting
        '''
        left = -10**10
        right = 10**10
        while left <= right:
            mid = left + (right - left)//2
            #count
            count = self.count(nums1,nums2,mid)
            if count < k:
                left = mid + 1
            else:
                right = mid - 1
        return left
    
    #count number of products less than some mid
    def count(self,nums1,nums2,mid):
        total = 0
        for n1 in nums1:
            if n1 > 0: 
                #bisect or bisect_right works here
                #doing ceuling on both works
                total += bisect.bisect_right(nums2, ceil(mid//n1))
            if n1 < 0: 
                total += len(nums2) - bisect.bisect_left(nums2, ceil(mid/n1))
            if n1 == 0 and mid >= 0: 
                total += len(nums2)

        return total

#breaking up the two binary search variants, but it gets TLE
class Solution:
    def kthSmallestProduct(self, nums1, nums2, k):
        lo, hi = -10**10, 10**10
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if self.check(nums1, nums2, mid) >= k:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def check(self, nums1, nums2, target):
        res = 0
        for num in nums1:
            if num < 0:
                # Products are descending, count how many num * nums2 <= target
                res += self.find1(nums2, num, target)
            elif num > 0:
                # Products are ascending
                res += self.find2(nums2, num, target)
            else:
                res += len(nums2) if target >= 0 else 0
        return res

    #find leftmost index, where n1*nums2[mid] <= target
    def find1(self, nums2, n1, target):
        lo, hi = 0, len(nums2) - 1
        ans = len(nums2)  # Default to len(nums2) if no valid index found

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if n1 * nums2[mid] <= target:
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1

        return len(nums2) - ans

    #find rightmost index where n1*nums2[mid] <= target
    def find2(self, nums2, n1, target):
        lo, hi = 0, len(nums2) - 1
        ans = -1  # Default to -1 if no valid index found

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if n1 * nums2[mid] <= target:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return ans + 1  # Total count is index + 1

#################################################
# 2014. Longest Subsequence Repeated k Times
# 27JUN25
#################################################
class Solution:
    def longestSubsequenceRepeatedK(self, s: str, k: int) -> str:
        '''
        there's a limit on the longest longest subsequence cannot exceed len(s)//k times
        it must be repeated k times
        try all candidates in reverse lexogrphical order, and check that they are present in s k times
        we need to use backtracking and generate the candidates in order
        we can only use the characters in the string
        find characters that could be in the potential answer
            a char >= k can be use in the answer up to count(ch) // k times
        n could be 2000
        in which case k could be as large 2000//8 = 250
        so we can try all
        start from each base character and move up
        '''
        counts = Counter(s)
        n = len(s)
        largest_size = n // k
        possible_chars = []
        for ch,c in sorted(counts.items(), key = lambda x: x[0], reverse = True):
            if c >= k:
                possible_chars.append(ch)
        if not possible_chars:
            return ""
        #print(possible_chars)
        #print(self.is_sub_seq(s,'ttlttl'))
        #enumerate with q, starting from all single chars
        ans = ""
        q = deque(possible_chars) #build them one letter at a time
        while q:
            curr = q.popleft()
            if len(curr) > len(ans):
                ans = curr
            for next_ch in possible_chars:
                next_pattern = curr + next_ch
                if self.is_sub_seq(s,next_pattern*k):
                    q.append(next_pattern)
        return ans
    def is_sub_seq(self, s,t):
        #check t is sub seq of s
        i = 0
        for ch in s:
            if ch == t[i]:
                i += 1
            if i == len(t):
                return True
        return False
    
################################################################################
# 1498. Number of Subsequences That Satisfy the Given Sum Condition (REVISTED)
# 29JUN25
###############################################################################
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        '''
        sort the array and fix for each index i,
        we need to find the right most index j, such that nums[i] + nums[j] <= target
        oh sum of min and max <= target
        the size would be (j - i + 1) then count number of subsequences
        the answer is 2**k - 1 subsequences
        '''
        N = len(nums)
        mod = 10**9 + 7
        nums.sort()
        
        count = 0
        for left,num in enumerate(nums):
            #binary serach, look for the rightmost number that is <= target, but make sure the index is >= left
            right = self.binary_search(nums,num,target)
            if right >= left:
                #find size of array
                k = right - left
                count += 2**k
                count %= mod
        return count % mod

    def binary_search(self,nums,num,target):
        ans = -1
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if num + nums[mid] <= target:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans


#linear time solution, after sorting of course
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        '''
        we can just sort and user two pointers
        if nums[left] + nums[right] <= target, we can use the whole thing
        use the smaller sum
        '''
        nums.sort()
        mod = 10**9 + 7
        count = 0
        left = 0
        right = len(nums) - 1
        while left <= right:
            if nums[left] + nums[right] <= target:
                k = right - left
                count += 2**k
                count %= mod
                left += 1
            else:
                right -= 1
        
        return count