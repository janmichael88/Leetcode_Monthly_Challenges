###########################################
# 2382. Maximum Segment Sum After Removals
# 01JAN25
##########################################
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.sum = {}
        self.max_sum_segment = 0


    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        #if we join segments just pass up sum to x
        x_par = self.find(x)
        y_par = self.find(y)

        if x_par == y_par:
            return
        self.sum[x_par] += self.sum[y_par]
        self.sum[y_par] = 0
        self.parent[y_par] = x_par
        self.max_sum_segment = max(self.max_sum_segment, self.sum[x_par])
    
    def add_merge(self, x, num):
        #add pointer to self first
        self.parent[x] = x
        self.sum[x] = num
        self.max_sum_segment = max(self.max_sum_segment, num)
        if x - 1 in self.parent:
            self.union(x,x-1)
        if x + 1 in self.parent:
            self.union(x,x+1)

    
class Solution:
    def maximumSegmentSum(self, nums: List[int], removeQueries: List[int]) -> List[int]:
        '''
        removed just means the num is set to zero in nums
        after each removal find the maximm segment sum
        hint says to use sorted data structure and remove invalid segments from the structure
        need to maintain the maximum sum, at the start the maximum sum is sum(nums)
        we would eventually get sent to the zeros array, but the queries have an order, we cant greedily take the largest

        reverse union find, start from the end
        whnever we union to a group att the sum back in, whenever we join, we need to check left and right
        when we do join we need to update max sums
        for union find do we need to keep track of ranks for this problem???
        go in reverse order of removals
        say we add in back i
        we need to check (i-1) to the left and (i+1) to the right
        find is still the same, but for union, we need to pass up the sums

        need to to union find in order!, but in reverse
        '''
        n = len(nums)
        uf = UnionFind()
        ans = [0]*n

        for i in range(n-1,-1,-1):
            ans[i] = uf.max_sum_segment
            idx = removeQueries[i]
            uf.add_merge(idx, nums[idx])
        
        return ans

##########################
# 1871. Jump Game VII
# 02JAN25
#########################
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        '''
        dp(i) returns true if we can reach the end
        so dp will TLE
        it becomes O(N^2)
        hint says to use prefix sums
        '''
        memo = {}
        n = len(s)

        def dp(i):
            if i >= n-1:
                return True
            if i in memo:
                return memo[i]
            ans = False
            for j in range(i + minJump, min(i + maxJump + 1,n)):
                if s[j] == '0':
                    ans = ans or dp(j)
            
            memo[i] = ans
            return ans

        return dp(0)
    
########################################
# 2270. Number of Ways to Split Array
# 03JAN25
########################################
class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        '''
        prefix sum, then just check the left and right parts at each split
        we can only split into 2
        '''
        pref_sum = [0]
        for num in nums:
            pref_sum.append(pref_sum[-1] + num)
        
        n = len(nums)
        splits = 0
        for i in range(n-1):
            left = pref_sum[i+1]
            right = pref_sum[-1] - pref_sum[i+1]
            if left >= right:
                splits += 1
        
        return splits

class Solution:
    def waysToSplitArray(self, nums: List[int]) -> int:
        '''
        we can optimize if we keep track of pref sum and suff sum
        '''
        pref_sum = 0
        suff_sum = sum(nums)

        splits = 0
        for i in range(len(nums) - 1):
            pref_sum += nums[i]
            suff_sum -= nums[i]

            if pref_sum >= suff_sum:
                splits += 1
        
        return splits
    
#################################
# 1871. Jump Game VII
# 03JAN25
#################################
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        '''
        dp(i) returns true if we can reach the end
        we can use bfs, but we need to prune so we dont visit previous indices in the interval
        but we jump to the max index instead
        https://leetcode.com/problems/jump-game-vii/solutions/1224681/python3-thinking-process-no-dp-needed/
        '''
        q = deque()
        seen = set()
        q.append(0)
        seen.add(0)
        mx = 0

        while q:
            i = q.popleft()
            if i == len(s) - 1:
                return True
            for j in range(max(i + minJump, mx + 1), min(i + maxJump + 1, len(s))):
                if s[j] == '0' and j not in seen:
                    seen.add(j)
                    q.append(j)
            
            mx = i + maxJump
        
        return False

###################################
# 2381. Shifting Letters II
# 05JAN25
###################################
class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        '''
        accumulate the shifts and apply at the end!
        this is just line sweep
        its only in step up or one steo down
        '''
        n = len(s)
        up_shifts = [0]*(n+1)
        down_shifts = [0]*(n+1)
        for l,r,d in shifts:
            if d == 1:
                up_shifts[l] += 1
                up_shifts[r+1] -= 1
            else:
                down_shifts[l] += 1
                down_shifts[r+1] -= 1
        #accumulate
        for i in range(1,n+1):
            up_shifts[i] += up_shifts[i-1]
            down_shifts[i] += down_shifts[i-1]
        
        ans = []
        for i in range(n):
            curr_shift = up_shifts[i] - down_shifts[i]
            #apply shift
            new_idx = ((ord(s[i]) - ord('a')) + curr_shift) % 26
            new_char = chr(ord('a') + new_idx)
            ans.append(new_char)
        
        return "".join(ans)

class Solution:
    def shiftingLetters(self, s: str, shifts: List[List[int]]) -> str:
        '''
        we can also accumulate on the fly
        '''
        n = len(s)
        up_shifts = [0]*(n+1)
        down_shifts = [0]*(n+1)
        for l,r,d in shifts:
            if d == 1:
                up_shifts[l] += 1
                up_shifts[r+1] -= 1
            else:
                down_shifts[l] += 1
                down_shifts[r+1] -= 1

        accum_up,accum_down = 0,0
        
        ans = []
        for i in range(n):
            accum_up += up_shifts[i]
            accum_down += down_shifts[i]
            curr_shift = accum_up - accum_down
            #apply shift
            new_idx = ((ord(s[i]) - ord('a')) + curr_shift) % 26
            new_char = chr(ord('a') + new_idx)
            ans.append(new_char)
        
        return "".join(ans)

#try bishop fenwich trees next

###################################################################
# 1769. Minimum Number of Operations to Move All Balls to Each Box
# 06JAN25
##################################################################
#brute force
class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        first of all you can solve this using brute force
        '''
        ans = []
        n = len(boxes)
        for i in range(n):
            moves = 0
            #go left
            for l in range(i-1,-1,-1):
                moves += (i - l) if boxes[l] == '1' else 0
            #go right
            for r in range(i+1,n):
                moves += (r - i) if boxes[r] == '1' else 0
            
            ans.append(moves)
        
        return ans


class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        first of all you can solve this using brute force
        need to store the number of moves in the left and right arrays
        '''
        ans = []
        n = len(boxes)
        for i in range(n):
            moves = 0
            for j in range(n):
                if j != i and boxes[j] == '1':
                    moves += abs(i-j)
            
            ans.append(moves)
        
        return ans

class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        first of all you can solve this using brute force
        need to store the number of moves in the left and right arrays
        build prefix array of moves
        we'll need at least two pref arrays, which should store the number of moves for to move each box to i
        need to track moves to the left and moves to the right

        idea is to accumulate balls and moves
        if we have k balls to the left of i
        then we need the move all k balls + the previous moves for (i-1)
        every time we move a step to left or right, all balls will need an additional move
        '''
        n = len(boxes)
        moves_to_left = [0]*n
        moves_to_right = [0]*n

        balls = int(boxes[0])
        for i in range(1,n):
            moves_to_left[i] = balls + moves_to_left[i-1]
            balls += int(boxes[i])
        
        balls = int(boxes[n-1])
        for i in range(n-2,-1,-1):
            moves_to_right[i] = balls + moves_to_right[i+1]
            balls += int(boxes[i])

        ans = [0]*n
        for i in range(n):
            ans[i] = moves_to_left[i] + moves_to_right[i]
        
        return ans

class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        '''
        we can change to one pass, just use accumulators for each one
        '''
        n = len(boxes)
        moves_to_left = 0
        moves_to_right = 0
        balls_to_left = 0
        balls_to_right = 0

        ans = [0]*n

        for l in range(n):
            ans[l] += moves_to_left
            balls_to_left += int(boxes[l])
            moves_to_left += balls_to_left

            r = n - l - 1
            ans[r] += moves_to_right
            balls_to_right += int(boxes[r])
            moves_to_right += balls_to_right

        return ans

###############################################
# 1408. String Matching in an Array (REVISTED)
# 08JAN25
##############################################
#using rabin karp
class Solution:
    def stringMatching(self, words: List[str]) -> List[str]:
        '''
        kmp or rabin karp
        good review on rabin karp
        then we need to run rabin karp for each word on any of the words
        '''
        ans = []
        for i in range(len(words)):
            for j in range(len(words)):
                if i == j:
                    continue
                if self.rabin_karp(words[i],words[j]):
                    ans.append(words[i])
                    break
        return ans

    def rabin_karp(self, s, t):
        p = 31
        m = int(1e9 + 9)
        S = len(s)
        T = len(t)
        
        # Compute powers of p modulo m
        p_pow = [1] * max(S, T)
        for i in range(1, len(p_pow)):
            p_pow[i] = (p_pow[i - 1] * p) % m

        # Compute hash values for all prefixes of t
        h = [0] * (T + 1)
        for i in range(T):
            h[i + 1] = (h[i] + (ord(t[i]) - ord('a') + 1) * p_pow[i]) % m

        # Compute the hash of the pattern s
        h_s = 0
        for i in range(S):
            h_s = (h_s + (ord(s[i]) - ord('a') + 1) * p_pow[i]) % m

        # Find occurrences of s in t
        occurrences = []
        for i in range(T - S + 1):
            cur_h = (h[i + S] + m - h[i]) % m
            if cur_h == h_s * p_pow[i] % m:
                return True
        return False

#using KMP, just review it
class Solution:
    def stringMatching(self, words: List[str]) -> List[str]:
        '''
        the idea with string matching brute force is that we can have partial matches of s in t, but not all the way
        do we need to just re start or can we use some kind of past information to match the string
        remember past prefixes
        lps array, longest prefix also a suffix
        remember prev prefix already a match, then shift
        proper prefix, any prefix not the string itself
        '''
        ans = []
        for i in range(len(words)):
            for j in range(len(words)):
                if i == j:
                    continue
                if self.kmp(words[i],words[j]):
                    ans.append(words[i])
                    break
        
        return ans
    
    def kmp(self, sub, main):
        #make lps array
        #lps reads as prefix of string that's also a suffix, but the longest
        lps = [0]*len(sub)
        i = 1
        length = 0

        while i < len(sub):
            if sub[i] == sub[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length > 0:
                    length = lps[length - 1] #go back the length of the previous longest prefix also a suffix
                else:
                    i += 1
        
        main_idx = 0
        sub_idx = 0

        while main_idx < len(main):
            if main[main_idx] == sub[sub_idx]:
                main_idx += 1
                sub_idx += 1
                if sub_idx == len(sub):
                    return True
            else:
                if sub_idx > 0:
                    #use lps to skip
                    sub_idx = lps[sub_idx - 1]
                else:
                    main_idx += 1
        
        return False

########################################
# 3042. Count Prefix and Suffix Pairs I
# 09JAN25
########################################
#brute force
class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        '''
        is this not just the lps array
        brute force works just fine,
        make sure to go back and look at KMP 
        '''
        pairs = 0
        n = len(words)
        for i in range(n):
            for j in range(i+1,n):
                curr = words[i]
                comp = words[j]
                if curr == comp[:len(curr)] and curr == comp[-len(curr):]:
                    pairs += 1
        
        return pairs

#trie review
class Node:
    def __init__(self,):
        self.children = defaultdict()

class Trie:
    def __init__(self,):
        self.root = Node()
    
    def insert(self,word):
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = Node()
            curr = curr.children[ch]
    
    def starts_with(self,pref):
        curr = self.root
        for ch in pref:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        
        return True


class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        '''
        other way is to make trie
        make two tries, pref trie, and suff trie
        '''
        n = len(words)
        ans = 0
        for i in range(n):
            pref_trie = Trie()
            suff_trie = Trie()
            #insert both
            pref_trie.insert(words[i])
            suff_trie.insert(words[i][::-1])
            for j in range(i):
                if len(words[j]) > len(words[i]):
                    continue
                
                if pref_trie.starts_with(words[j]) and suff_trie.starts_with(words[j][::-1]):
                    ans += 1
        
        return ans

####################################
# 916. Word Subsets (REVISTED)
# 10JAN25
#####################################
class Solution:
    def wordSubsets(self, words1: List[str], words2: List[str]) -> List[str]:
        '''
        just check counts
        but first count all of word2
        its just the max array of all words in words
        '''
        max_all_words_2 = Counter()
        for w in words2:
            count_w = Counter(w)
            for k,v in count_w.items():
                max_all_words_2[k] = max(max_all_words_2[k],v)
        
        ans = []

        for w in words1:
            count_w = Counter(w)
            if not (max_all_words_2 - count_w):
                ans.append(w)
        
        return ans

#######################################
# 1400. Construct K Palindrome Strings
# 12JAN25
#######################################
class Solution:
    def canConstruct(self, s: str, k: int) -> bool:
        '''
        to make a palindrome:
            odd length, fix center with any letter, then i can use smae letters left and right
                i.e the counts of letters used on left == right
            even, length
                no center, counts used left == counts used right

        if every char in string can be palindrome, we can make at least len(s) palindromes
        can't be bigger than k

        even freq chars don't need a center, we can distribute them across all palindromes
        the minimum number of palindroms we can make == the number of odd-frequency characets, 
        each odd-ref requires its own palindrome
        
        '''
        if len(s) < k:
            return False
        #count chars
        char_counts = Counter(s)
        #check off count chars
        odd_counts = 0
        for char,count in char_counts.items():
            if count % 2 == 1:
                odd_counts += 1
        
        #we are always left with pairs that can't be matched!
        return odd_counts <= k
    
#####################################################
# 2116. Check if a Parentheses String Can Be Valid
# 12JAN25
######################################################
class Solution:
    def canBeValid(self, s: str, locked: str) -> bool:
        '''
        we can only change s at indices is locked[i] == '0'
        we are free to pick open or close
        first try to balance using stack trick, but make sure to pair indices too
        anything left over on the stack, see if we can balanc it
        '''
        unbalanced = []
        for i,ch in enumerate(s):
            if ch == ')':
                if unbalanced and unbalanced[-1][1] == '(':
                    unbalanced.pop()
                else:
                    unbalanced.append((i,ch))
            else:
                unbalanced.append((i,ch))
        
        if not unbalanced:
            return True

        if len(unbalanced) % 2:
            return False
        #try to balance
        #balance in pairs
        for i in range(0,len(unbalanced),2):
            curr_idx,curr_ch = unbalanced[i]
            next_idx,next_ch = unbalanced[i+1]
            if curr_ch == ')':
                if next_ch == ')':
                    if locked[curr_idx] == '1':
                        return False
                if next_ch == '(':
                    if locked[curr_idx] == '1' and locked[next_idx] == '1':
                        return False
            elif curr_ch == '(':
                if next_ch == '(':
                    if locked[next_idx] == '1':
                        return False
        return True
    
class Solution:
    def canBeValid(self, s: str, locked: str) -> bool:
        '''
        idea is to keep track of opeb brackets and unlocked indices
        if we have any unmatched opening, we can close them with unlocked indices
        as we track, we keep track of any closed paranthesses and match them with any open ones

        if at any oint the number of closing exceeds opening, and there are no unlocked indices
        we cant balance them, retrun galse
        '''
        if len(s) % 2 == 1:
            return False
        
        open_brackets = []
        unlocked_brackets = []

        for i,ch in enumerate(s):
            #add index to unlocked
            if locked[i] == '0':
                unlocked_brackets.append(i)
            elif ch == '(':
                open_brackets.append(i)
            #if closing, match it with opend
            elif ch == ')':
                if open_brackets:
                    open_brackets.pop()
                elif unlocked_brackets:
                    unlocked_brackets.pop()
                else:
                    return False
        
        #no just clear them
        #but for any inlocked index, it must precede the open brackets index
        while open_brackets and unlocked_brackets and unlocked_brackets[-1] > open_brackets[-1]:
            open_brackets.pop()
            unlocked_brackets.pop()
        
        return not open_brackets
    
class Solution:
    def canBeValid(self, s: str, locked: str) -> bool:
        '''
        we can do two pass in constant space
        we count the violations
        if closed and locked, decreemnt
        if < 0 return false
        otherwise increment

        you can do this in the general case,
        must check balance left to right and right to left
        '''
        balance = 0
        n = len(s)
        for i,ch in enumerate(s):
            if locked[i] == '1' and ch == ')':
                balance -= 1
                if balance < 0:
                    return False
            else:
                balance += 1
        
        balance = 0
        for i in range(n-1,-1,-1):
            if locked[i] == '1' and s[i] == '(':
                balance -= 1
                if balance < 0:
                    return False
            else:
                balance += 1

        return n % 2 == 0
    
##################################################
# 3223. Minimum Length of String After Operations
# 14JAN24
##################################################
class Solution:
    def minimumLength(self, s: str) -> int:
        '''
        to keep deleting we would want to pick the index i such that counts of s[i] to left and to right of i are as large as possible
        heap of counts
        we need at least three occurences of a character
        '''
        counts = Counter(s)
        max_heap = [(-count,ch) for ch,count in counts.items()]
        heapq.heapify(max_heap)
        ans = len(s)

        while max_heap and abs(max_heap[0][0]) >= 3:
            curr_count, curr_char = heapq.heappop(max_heap)
            curr_count += 2
            ans -= 2
            heapq.heappush(max_heap, (curr_count,curr_char))
        
        return ans
        
#ezzzz
class Solution:
    def minimumLength(self, s: str) -> int:
        '''
        we actually dont need a heap
        count up the chars
        if chars have even parity use use all but one
        intution, if a char appears and off number of times, we can keep one instance of if and remove thest

        '''
        counts = Counter(s)
        removals = 0

        for k,v in counts.items():
            if v >= 3:
                #if its even we need to leave 2
                if v % 2 == 0:
                    removals += v - 2
                else:
                    removals += v - 1
                    
        
        return len(s) - removals
