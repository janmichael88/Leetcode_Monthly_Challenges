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

###################################################
# 2657. Find the Prefix Common Array of Two Arrays
# 14JAN25
###################################################
from collections import Counter
class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        '''
        arrays are permutations of each other and of equal length
        common prefix array is the array C, where C[i] == count of numbers present at or before the index\
        count object at and increment,
        then check counts
        '''
        curr_count = Counter()
        ans = []
        n = len(A)

        for i in range(n):
            curr_count[A[i]] += 1
            curr_count[B[i]] += 1
            #count the number of frequencies that == 2
            temp = Counter(curr_count.values())
            ans.append(temp[2])
        
        return ans

class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        '''
        the frequencies of each number can be at most 2
        we can use a a single hashset to keep track of the common numbers at each index i
        the ans is just (i+1)*2 - len(hashset)
        if the elements in the two sets are always different, there will be 2*(i+1) elements in the set and 0 common elements
        if there are K elements in common, then there are 2*(i+1) - K elements (the commone ones)
        at each index i, we increment by 2*(i+1) elements
        tricky math
        '''
        common_array = []
        intersection = set()
        for i in range(len(A)):
            intersection.add(A[i])
            intersection.add(B[i])
            common_array.append((i+1)*2 - len(intersection))
        
        return common_array
    
class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        '''
        we compute common on the fly

        '''
        ans = []
        counts = Counter()
        common = 0

        for i in range(len(A)):
            counts[A[i]] += 1
            if counts[A[i]] == 2:
                common += 1
            counts[B[i]] += 1
            if counts[B[i]] == 2:
                common += 1
            
            ans.append(common)
        
        return ans
    
#######################
# 2429. Minimize XOR
# 15JAN25
########################
#almost the right idea
#but we want to clear from the least signicant virst
class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        '''
        need x to have the same number of bits as num 2
        and x XOR 1 to be as small as possible
        we can count the bits in num2, and move them all to lower registers
        for some number x, if the set bits align with num1, they go to zero
        greddily make the number
        '''
        set_bits = self.count_bits(num2)
        #for each set bit in num1 (from the left) pair it with a bit
        #we need to clear out the most significan bits
        #any left over bits should be set to 1
        curr_set_bits = self.count_bits(num1)
        bits = self.get_bits(num1)
        i = len(bits) - 1
        #if we don't have enough bits
        while curr_set_bits < set_bits:
            if i < 0:
                bits = [0] + bits
                i = 0
            if bits[i] == 0:
                bits[i] = 1
                curr_set_bits += 1
            
            i -= 1
        
        #clear out from most sig
        while curr_set_bits > set_bits:
            if i < 0:
                bits = [0] + bits
                i = 0
            if bits[i] == 1:
                bits[i] = 0
                curr_set_bits -= 1
            i -= 1
        
        ans = 0
        for num in bits:
            ans = (ans << 1) | num
        return ans

    def count_bits(self,num):
        set_bits = 0
        while num:
            set_bits += 1
            num = num & (num - 1)
        
        return set_bits
    
    def get_bits(self,num):
        bits = []
        while num:
            bits.append(num & 1)
            num = num >> 1
        
        return bits[::-1]
    
#we can use bitwise opertors
class Solution:
    def minimizeXor(self, num1: int, num2: int) -> int:
        '''
        insted of using an array, we can use bitwise operators
        and change at the bit registers
        cant just clear out the most significant bits in num1
        because want to minimue the number x XOR num1
        '''
        ans = num1
        target_set_bits = self.count_set_bits(num2)
        curr_set_bits = self.count_set_bits(num1)

        i = 0

        while curr_set_bits < target_set_bits:
            if not self.is_set(ans,i):
                ans = self.set_bit(ans,i)
                curr_set_bits += 1
            i += 1
        
        while curr_set_bits > target_set_bits:
            if self.is_set(ans,i):
                ans = self.unset_bit(ans,i)
                curr_set_bits -= 1
            
            i += 1
        
        return ans

    def is_set(self, num, bit):
        return (num & (1 << bit)) != 0
    
    def set_bit(self,num,bit):
        return num | (1 << bit)
    
    #for unsetting invert the mask using ~(1 << i)
    def unset_bit(self,num,bit):
        return num & ~(1 << bit)
    
    def count_set_bits(self,num):
        bits = 0
        while num:
            bits += 1
            num = num & (num - 1)
        return bits
    
#######################################
# 2425. Bitwise XOR of All Pairings
# 16JAN25
########################################
class Solution:
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        count up the number of times each number appears in the pairing
        then only grab the odds and xor them
        if some number x apperas in nums1, it will appear len(nums2) times
        if some number y appears in num2, it will appear len(nums1) times
        then just cancel out even parities
        '''
        counts = Counter()
        for num in nums1:
            counts[num] += len(nums2)
        
        for num in nums2:
            counts[num] += len(nums1)
        
        xor = 0
        for k,v in counts.items():
            if v % 2 == 1:
                xor ^= k
        
        return xor
    
class Solution:
    def xorAllNums(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        bit maniuplations
        if len(nums2) is even, each element in nums1 will cancel out
        if len(nums2) is odd, each element will have one value (i.e a^ a^a) is just a
        store the registers in xor1 and xor2
        '''
        xor1, xor2 = 0,0

        if len(nums2) % 2 == 1:
            for num in nums1:
                xor1 = xor1 ^ num
        
        if len(nums1) % 2 == 1:
            for num in nums2:
                xor2 = xor2 ^ num
        
        return xor1 ^ xor2
    
###################################
# 2683. Neighboring Bitwise XOR
# 17JAN25
###################################
class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        '''
        we have an array dervied, that came from another array, orignla
        where dervied[i] = original[i] ^ original[i+1]
        if i == n - 1:
            dervied[i] = originla[i] ^ original[0]
        
        this means circular array
        if derived[i] = 1, then orig[i],orig[i+1] = [0,1] or [1,0]
        if dervied[i] = 0, then orig[i],orig[i+1] = [1,1] or [0,0]
        build the array on the fly and check for violation?
        say we have 
        orig = [a,b,c,d]
        derived = [a^b,b^c,c^d,d^a]
        if derived did indeed come from original,
        then we could xor all of them and get zero
        if it didnt, then dervied could not have possibly come from original
        '''
        check = 0
        for num in derived:
            check = check ^ num
        
        return check == 0
    

class Solution:
    def doesValidArrayExist(self, derived: List[int]) -> bool:
        '''
        in addition to XOR beeing commutative and associate there is another property
        if we have
        a ^ b = c
        a ^ b ^ b = c ^ b
        a ^ 0 = c ^ b
        a = c ^ b
        so we have 
        derived[i] = original[i] ^ original[i+1]
        we can find the original
        original[i] = dervied[i] ^ originl[i+1]
        or
        original[i+1] = derived[i] ^ original[i]
        we can actuall build the arrays, if we start with original[0] = 0 or 1
        one we have the oringall array, we need to check the cirular property in the original array
        '''
        #start with zero
        zero_first = [0]
        for num in derived:
            zero_first.append(zero_first[-1] ^ num)
        
        #start with one
        one_first = [1]
        for num in derived:
            one_first.append(one_first[-1] ^ num)
        
        return (zero_first[-1] == zero_first[0]) or (one_first[-1] == one_first[0])
    
################################################################
# 1368. Minimum Cost to Make at Least One Valid Path in a Grid
# 18JAN25
##################################################################
#good reivew on djisktras
class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        '''
        if i'm at cell (i,j) 
        i can do in the current direction with no cost or i can change it with cost + 1
            we can do this for all directions
        states are (i,j,direction)
        make graph where where (i,j) is connected to all other cells adjacnetl, who's weight is 1
        else 0 other wise, then do dp in graph
        it only works on DAGs, here we can have a back edge
        
        '''
        rows,cols = len(grid),len(grid[0])
        graph = defaultdict(list)
        dirrs = {1 : [0,1], 2: [0,-1], 3: [1,0], 4:[-1,0]}
        for i in range(rows):
            for j in range(cols):
                for d in dirrs:
                    if d == grid[i][j]:
                        di,dj = dirrs[d]
                        if (0 <= i + di < rows) and (0 <= j + dj < cols):
                            graph[(i,j)].append((0,i + di, j + dj))
                    else:
                        di,dj = dirrs[d]
                        if (0 <= i + di < rows) and (0 <= j + dj < cols):
                            graph[(i,j)].append((1,i + di, j + dj))
        #now do djikstras on graph
        dists = [[float('inf')]*cols for _ in range(rows)]
        dists[0][0] = 0
        pq = [(0,0,0)]
        visited = set()

        while pq:
            min_dist,i,j = heapq.heappop(pq)
            if dists[i][j] < min_dist:
                continue
            if (i,j) in visited:
                continue
            visited.add((i,j))

            for cost,di,dj in graph[(i,j)]:
                if (di,dj) in visited:
                    continue
                new_dist = dists[i][j] + cost
                if new_dist < dists[di][dj]:
                    dists[di][dj] = new_dist
                    heapq.heappush(pq, (new_dist, di,dj))
        
        return dists[rows-1][cols-1]

#0,1 bfs, 0 weight to front, 1 to the end
class Solution:
    def minCost(self, grid: List[List[int]]) -> int:
        '''
        0-1 bfs    
        '''
        rows,cols = len(grid),len(grid[0])
        graph = defaultdict(list)
        dirrs = {1 : [0,1], 2: [0,-1], 3: [1,0], 4:[-1,0]}
        for i in range(rows):
            for j in range(cols):
                for d in dirrs:
                    if d == grid[i][j]:
                        di,dj = dirrs[d]
                        if (0 <= i + di < rows) and (0 <= j + dj < cols):
                            graph[(i,j)].append((0,i + di, j + dj))
                    else:
                        di,dj = dirrs[d]
                        if (0 <= i + di < rows) and (0 <= j + dj < cols):
                            graph[(i,j)].append((1,i + di, j + dj))
        #now do djikstras on graph
        dists = [[float('inf')]*cols for _ in range(rows)]
        dists[0][0] = 0
        q = deque([])
        q.append((0,0))

        while q:
            i,j = q.popleft()
            for cost,ii,jj in graph[(i,j)]:
                new_dist = dists[i][j] + cost
                if dists[ii][jj] > new_dist:
                    dists[ii][jj] = new_dist
                    if cost == 1:
                        q.append((ii,jj))
                    else:
                        q.appendleft((ii,jj))

        return dists[rows-1][cols-1]

#i wonder why top down dp don't work

########################################
# 407. Trapping Rain Water II (REVISTED)
# 20JAN25
########################################
class Solution:
    def trapRainWater(self, heightMap: List[List[int]]) -> int:
        '''
        min heap and fill the smallest ones first
        if i'm at some cell (i,j), look at its neighbors
        the amount of water that can be filled is just the differnce of the largest of its neighbors and the heigh of (i,j)
        we start from the outer walls then work in
            we need to fill from the boundaries
            for a cell to trap water, it must not exceed the smallest height of its neigbors
            when we add water to this cell, we add it back to the boundary
            boundary will be the min heap, this prevents water from splling over the boundary
        
        now if the cell's height is >= to the boundary height, no water can be trapped above it
        but it can still be use to trap water to its neighbords

        '''
        m, n = len(heightMap), len(heightMap[0])
        heap = []
        visited = [[0]*n for _ in range(m)]
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]

        # Push all the block on the border into heap
        for i in range(m):
            for j in range(n):
                if i == 0 or j == 0 or i == m-1 or j == n-1:
                    heapq.heappush(heap, (heightMap[i][j], i, j))
                    visited[i][j] = 1
        
        result = 0
        while heap:
            #get smalest from heap
            height, i, j = heapq.heappop(heap)    
            for di, dj in dirrs:
                x = i + di
                y = j + dj
                #neighbores and in bounds check
                if 0 <= x < m and 0 <= y < n and not visited[x][y]:
                    #we can store water with differeunt, up to though
                    result += max(0, height-heightMap[x][y])
                    next_height = max(heightMap[x][y],height)
                    heapq.heappush(heap, (next_height, x, y))
                    #mark
                    visited[x][y] = 1
        return result
    
##############################################
# 2661. First Completely Painted Row or Column
# 20JAN25
###############################################
class Solution:
    def firstCompleteIndex(self, arr: List[int], mat: List[List[int]]) -> int:
        '''
        simulate an mark each index i in arr to the (i,j) where mat[i][j] = i
        after we paint we need to check all rows and columns for any t hat are complete filled
        return the index

        then validate row counts filled and col counts filled
        when i fill a cell (i,j)
        but then checking takes additional time
        '''
        #rrevese mapp of mat
        rows,cols = len(mat),len(mat[0])
        mapp = {}
        for i in range(rows):
            for j in range(cols):
                mapp[mat[i][j]] = (i,j)
        
        row_counts = [0]*rows
        col_counts = [0]*cols
        for idx,i in enumerate(arr):
            curr_row,curr_col = mapp[i]
            row_counts[curr_row] += 1
            col_counts[curr_col] += 1
            #after incrementing check
            if row_counts[curr_row] == cols or col_counts[curr_col] == rows:
                return idx
        
        return -1

#########################################
# 1730. Shortest Path to Get Food
# 20JAN25
##########################################
class Solution:
    def getFood(self, grid: List[List[str]]) -> int:
        '''
        multipoint bfs from all food cells to where i'm standing
        '''
        rows,cols = len(grid),len(grid[0])
        dists = [[float('inf')]*cols for _ in range(rows)]
        q = deque([])
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '*':
                    dists[i][j] = 0
                    q.append((0,i,j))
            
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        while q:
            curr_dist,i,j = q.popleft()
            for di,dj in dirrs:
                ii = i + di
                jj = j + dj
                if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] in ('O','#'):
                    neigh_dist = curr_dist + 1
                    #can be improved
                    if dists[ii][jj] > neigh_dist:
                        dists[ii][jj] = neigh_dist
                        q.append((neigh_dist, ii,jj))
        ans = float('inf')
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == '#':
                    ans = min(ans,dists[i][j])
        if ans == float('inf'):
            return -1
        return ans
        