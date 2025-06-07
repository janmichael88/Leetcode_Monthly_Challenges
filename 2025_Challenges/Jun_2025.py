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
