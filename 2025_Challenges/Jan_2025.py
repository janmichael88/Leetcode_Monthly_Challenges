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
    
##########################
# 2017. Grid Game
# 21JAN25
##########################
#not quite
class Solution:
    def gridGame(self, grid: List[List[int]]) -> int:
        '''
        minimizing the opposing robot score is the same as maximizing its current score
        robots can only move right and down
        the first robot makes all its moves first
        the second robot then goes
        we could do bfs, but we'd need to record the optimal path for robot 1
        then replace those values with zero and do bfs on the second robot
        '''
        #first path for robot 1
        path = self.dijkstra(grid)
        #now drop in zeros
        for i,j in path:
            grid[i][j] = 0
        for r in grid:
            print(r)
    
    def dijkstra(self,grid):
        '''
        record with path and get maximum score
        '''
        dirrs = [(1,0),(0,1)]
        rows,cols = len(grid),len(grid[0])
        dists = [[float('-inf')]*cols for _ in range(rows)]
        dists[0][0] = - grid[0][0]
        visited = set()
        prev = {}
        max_heap = [(-grid[0][0], 0,0)]
        
        while max_heap:
            max_weight, i,j = heapq.heappop(max_heap)
            if dists[i][j] > max_weight:
                continue
            visited.add((i,j))
            for di,dj in dirrs:
                ii = i + di
                jj = j + dj
                if (ii,jj) in visited:
                    continue
                if 0 <= ii < rows and 0 <= jj < cols:
                    neigh_weight = -max_weight + grid[ii][jj]
                    if neigh_weight > dists[ii][jj]:
                        dists[ii][jj] = -neigh_weight
                        prev[(ii,jj)] = (i,j)
                        heapq.heappush(max_heap, (-neigh_weight, ii,jj))
                    elif neigh_weight == dists[ii][jj]:
                        prev[(ii,jj)] = (i,j)
        print('Dists are: ')
        for r in dists:
            print(r)
        #get path
        path = []
        curr = (rows - 1, cols - 1)
        while curr != (0,0):
            path.append(curr)
            curr = prev[curr]
        path.append(curr)
        return path[::-1]

#minimizing  the number of points collected by second robot on robot's first path is different than finding the maximum path
#for robot 1

#not quite, you need to max and min at the same time
class Solution:
    def gridGame(self, grid: List[List[int]]) -> int:
        '''
        using hints, alludes to pref_sum
        there are n choice for when the first robot moves to second row
        the robot can only move down and to the right
        the aren'y playing move against move, first robot makes all moves in the beginning
        one robot moves down one time, he can no longer move up
        find where robot moves down one time optmially
        then replace zeros, then do robot two
        '''
        rows,cols = len(grid),len(grid[0])
        pref_grid = [[0]*(cols + 1) for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                pref_grid[i][j+1] = pref_grid[i][j] + grid[i][j]
        
        #find down index
        down_idx = -1
        first_score = float('-inf')
        for i in range(1,cols+1):
            left_total = pref_grid[0][i]
            right_total = pref_grid[1][-1] - pref_grid[1][i-1]
            #print(left_total, right_total)
            if left_total + right_total > first_score:
                first_score = left_total + right_total
                down_idx = i-1
        
        #replace with zeros in original grid up to the down index
        curr_row = 0
        for col in range(down_idx+1):
            grid[curr_row][col] = 0
        curr_row += 1
        for col in range(down_idx, cols):
            grid[curr_row][col] = 0
        
        #now do robot 2
        pref_grid = [[0]*(cols + 1) for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                pref_grid[i][j+1] = pref_grid[i][j] + grid[i][j]
        
        ans = 0
        for i in range(1,cols+1):
            left_total = pref_grid[0][i]
            right_total = pref_grid[1][-1] - pref_grid[1][i-1]
            ans = max(ans, left_total + right_total)
        
        return ans

#max-mini or mini-maxi paradigm
class Solution:
    def gridGame(self, grid: List[List[int]]) -> int:
        '''
        at each turn calculate what robot 2 ccould have gotten
        assumuing that once robot 1 makes its turn, the cells have been zero'd

        inuition:
            a robot has n possible turning conditions
            after robot1 moves, its top row becomes zero up to the index, and the bottom row from that turn index to the end becomes zero
        
        if after some turn i, robot 2 must  make his move
        from robot 2's choise we want to maximize his score
        robot 2 can take the remaining in the first row to the right
        or robot2 can take the reaming in the second row to the left
        robot 2's choices = max(op1,op2)
        at the same time, robot1 needs to minimize thise

        the goal is to reduce the highest points the second robot can collect
        '''
        rows,cols = len(grid),len(grid[0])
        pref_grid = [[0]*(cols + 1) for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                pref_grid[i][j+1] = pref_grid[i][j] + grid[i][j]
        
        ans = float('inf')
        for i in range(1,cols+1):
            right_total = pref_grid[0][-1] - pref_grid[0][i]
            left_total = pref_grid[1][i-1]
            ans = min(ans, max(right_total,left_total))
        
        return ans


###########################################
# 1765. Map of Highest Peak
# 22JAN25
###########################################
class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        '''
        binary matrix (i,j)
        0 is land, 1 is water
        heights must be non-negative, if water, leave at zero
        any two adjacent cells must have abs height diff at most 1 (up,dowm,left,right)
        find assignemnt of heights so that the matrix is maximized
        bfs from all water cells, and find the min distance from each water cell to its nerest water cell
        reframe the problem as finding the min dist for each water cell (1) to its near land cell (0)
        this allows heights to increae smoothly form water cells 0/1 matrix
        '''
        rows = len(isWater)
        cols = len(isWater[0])
        zeros_dist = [[0]*cols for _ in range(rows)]
        dirrs = [(1,0),(-1,0),(0,-1),(0,1)]
        
        q = deque([])
        seen = [[False]*cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if isWater[i][j] == 1:
                    q.append((i,j,0))
                    seen[i][j] = True
                    
        while q:
            x,y,dist = q.popleft()
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols and not seen[neigh_x][neigh_y]:
                    seen[neigh_x][neigh_y] = True
                    zeros_dist[neigh_x][neigh_y] = dist + 1
                    q.append((neigh_x,neigh_y,dist+1))
        
        
        return zeros_dist
    
#dp solution is more insightful
class Solution:
    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        '''
        for the dp solution
        we find the min dist going down and to the right first
        then we find the min dist going up and to the left next
        alternate trick is just to hash the array, and check for neighbors in the hashmapp, just to easily check boundary conditions

        '''
        rows = len(isWater)
        cols = len(isWater[0])
        heights = [[float('inf')]*cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if isWater[i][j] == 1:
                    heights[i][j] = 0
        
        #down and to the right
        for i in range(rows):
            for j in range(cols):
                dirrs = [[-1,0],[0,-1]]
                min_dist = heights[i][j]
                for di,dj in dirrs:
                    if 0 <= i + di < rows and 0 <= j + dj < cols:
                        min_dist = min(min_dist,heights[i+di][j+dj] + 1)
                heights[i][j] = min(heights[i][j],min_dist)
        
        #up and to the left
        for i in range(rows - 1,-1,-1):
            for j in range(cols -1,-1,-1):
                dirrs = [[1,0],[0,1]]
                min_dist = heights[i][j]
                for di,dj in dirrs:
                    if 0 <= i + di < rows and 0 <= j + dj < cols:
                        min_dist = min(min_dist,heights[i+di][j+dj] + 1)
                heights[i][j] = min(heights[i][j],min_dist)
        
        return heights
    
#########################################
# 1267. Count Servers that Communicate
# 23JAN25
#########################################
class Solution:
    def countServers(self, grid: List[List[int]]) -> int:
        '''
        servers can communicate if the are on the same row or column
        just keep hahset of of servers
        '''
        rows,cols = len(grid), len(grid[0])
        servers = set()
        
        #go down rows
        for i in range(rows):
            along_row = []
            for j in range(cols):
                if grid[i][j] == 1:
                    along_row.append((i,j))
            if len(along_row) > 1:
                for k in along_row:
                    servers.add(k)
        
        #go down cols
        for j in range(cols):
            along_col = []
            for i in range(rows):
                if grid[i][j] == 1:
                    along_col.append((i,j))
            
            if len(along_col) > 1:
                for k in along_col:
                    servers.add(k)
        
        return len(servers)

#counting
class Solution:
    def countServers(self, grid: List[List[int]]) -> int:
        '''
        another way would be to just store row_counts and col_counts
        then check if a server at some (i,j) has row count and col count > 1, it can communicate
        '''
        rows,cols = len(grid),len(grid[0])
        row_counts = [0]*rows
        col_counts = [0]*cols

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    row_counts[i] += 1
                    col_counts[j] += 1
        
        ans = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    if row_counts[i] > 1 or col_counts[j] > 1:
                        ans += 1
        return ans

############################################
# 802. Find Eventual Safe States (REVISTED)
# 24JAN25
############################################
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        '''
        need to find all safe nodes
        a node is safe if every possible path starting from that node leads to a terminal node
        any node with outgoing edges are is a sage node
        brute force would be to first find all terminal nodes
        then for every node that isn't terminal check if every path to see if it leads to a terminal node
        terminal nodes are safe
        can't do dfs on each node
        if we know we have to end at terminal nodes, then let's start from them, reverse the edges and start from terminal nodes
        any node in a cycle cannot be be safe, and node visited during traversal while kahn is safe
        '''
        n = len(graph)
        degree = [0]*n
        #reverse graph
        reverse_graph = defaultdict(list)
        for i,neighs in enumerate(graph):
            for neigh in neighs:
                reverse_graph[neigh].append(i)
                degree[i] += 1
   
        q = deque([])
        #start with leaves
        for i in range(n):
            if degree[i] == 0:
                q.append(i)
  
        safe = [False]*n
        while q:
            curr = q.popleft()
            safe[curr] = True
            for neigh in reverse_graph[curr]:
                degree[neigh] -= 1
                if degree[neigh] == 0:
                    q.append(neigh)
        
        return [i for i in range(n) if safe[i]]

#TLE checking from each node, for all nodes
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        '''
        we can also use dfs to find cycles
        '''
        n = len(graph)
        ans = []
        seen = set()
        for i in range(n):
            if not self.has_cycle(i,seen,graph):
                ans.append(i)
        return ans
        
    def has_cycle(self,node,seen,graph):
        if node in seen:
            return True
        seen.add(node)
        for neigh in graph[node]:
            if self.has_cycle(neigh,seen,graph):
                return True
        seen.remove(node)
            
        return False

####################################################################
# 2948. Make Lexicographically Smallest Array by Swapping Elements
# 25JAN25
####################################################################
class DSU:
    def __init__(self, n):
        self.n = n
        self.size = [1]*n
        self.parent = [i for i in range(n)]
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)

        if x_par == y_par:
            return
        elif self.size[x_par] >= self.size[y_par]:
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
            self.parent[y_par] = x_par
        else:
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
            self.parent[x_par] = self.parent[y_par]

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        '''
        we can only swap indices (i,j) is abs(nums[i] - nums[j]) <= limit
        if we have a valid (i,j) then to make it currentl smaller put the smaller of the two to the left and bigger to the right
        union find on groups after sorting array
        if numbers are connected by limit, we are free to swap all of them.
        imagine [4,3,2,1], limit = 1, we can get [1,2,3,4]
        [3,4,1,2]
        [2,4,1,3]
        [1,4,2,3]
        [1,3,2,4]
        [1,2,3,4]
        for any group of numbers in a connected componented, we can sort them lexographically
        since we sorted, we have access to the current minimum value in the group
        i need the orignial array
        need to maintain the oringal sorted array
        '''
        #maintain indices
        nums_sorted = sorted([(num,i) for i,num in enumerate(nums)])
        n = len(nums)
        dsu = DSU(n)
        for i in range(n-1):
            if nums_sorted[i+1][0] - nums_sorted[i][0] <= limit:
                dsu.union(nums_sorted[i+1][1],nums_sorted[i][1])
        
        groups = defaultdict(list)
        for i in range(n):
            groups[dsu.find(i)].append(nums[i])
        
        #sort within groups
        for k,v in groups.items():
            v_sorted = sorted(v)
            groups[k] = v_sorted
        
        ans = []
        for i in range(n):
            ans.append(groups[dsu.find(i)].pop(0))
        
        return ans
    
#using union  find with no so, do union find on numbers instead! not indices
class DSU:
    def __init__(self, nums):
        self.n = len(nums)
        self.size = defaultdict(int)
        self.parent = defaultdict(int)
        for e in nums:
            self.parent[e] = e
            self.size[e] = 1
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)

        if x_par == y_par:
            return
        elif self.size[x_par] >= self.size[y_par]:
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
            self.parent[y_par] = x_par
        else:
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
            self.parent[x_par] = self.parent[y_par]

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        '''
        dont maintain indices while sorting
        do union find on values of nums instead of their indices
        '''
        #maintain indices
        nums_sorted = sorted(nums)
        n = len(nums)
        dsu = DSU(nums)
        for i in range(n-1):
            if nums_sorted[i+1] - nums_sorted[i] <= limit:
                dsu.union(nums_sorted[i+1],nums_sorted[i])
        
        groups = defaultdict(deque)
        for e in nums_sorted:
            groups[dsu.find(e)].append(e)
        
        ans = []
        for e in nums:
            ans.append(groups[dsu.find(e)].popleft())
        
        return ans

#sort and hashmap
class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        '''
        there is a no union find solution
        key, swapping indices within a limit is transisivve
        i.e if swap(a,b) and swap(b,c) we can swap(a,c)
        find numbers that are all swappable and put into groups
        then order the groups increasingly
        rearrragnment of any permutation can be dont
        sort numbers and add to groups
        we keep adding to the same group if within the limit
        otherwise we make a new group
        need to mainting smallest numbers, so use deque
        '''
        nums_sorted = sorted(nums)

        curr_group = 0
        num_to_group = {}
        num_to_group[nums_sorted[0]] = curr_group

        group_to_list = {}
        group_to_list[curr_group] = deque([nums_sorted[0]])

        #need to init with first number
        for i in range(1, len(nums)):
            if abs(nums_sorted[i] - nums_sorted[i - 1]) > limit:
                # new group
                curr_group += 1

            # assign current element to group
            num_to_group[nums_sorted[i]] = curr_group

            # add element to sorted group deque
            if curr_group not in group_to_list:
                group_to_list[curr_group] = deque()
            group_to_list[curr_group].append(nums_sorted[i])

        # iterate through input and overwrite each element with the next element in its corresponding group
        for i in range(len(nums)):
            num = nums[i]
            group = num_to_group[num]
            nums[i] = group_to_list[group].popleft()

        return nums

###########################################
# 1462. Course Schedule IV
# 27JAN25
###########################################
#bfs
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        '''
        we need to check if some course u is a preq of another course v
        if courses are in a cycle, each is a prequiste of the aother
        bfs from each course and keep track in array
        reachable[i][j], where j is reachable from i
        us array to check the queires given
        note in queries, classes will not point to themselves
        '''
        graph = defaultdict(list)
        for u,v in prerequisites:
            graph[u].append(v)
        
        can_reach = [[False]*numCourses for _ in range(numCourses)]
        for i in range(numCourses):
            self.bfs(graph,i,can_reach)
        
        ans = []

        for u,v in queries:
            ans.append(can_reach[u][v])
        
        return ans
    
    def bfs(self,graph,start,can_reach):
        seen = set()
        q = deque([])
        q.append(start)

        while q:
            curr = q.popleft()
            if curr != start:
                can_reach[start][curr] = True
            if curr in seen:
                continue
            seen.add(curr)
            for neigh in graph[curr]:
                if neigh not in seen:
                    q.append(neigh)

#dfs
class Solution:
    def checkIfPrerequisite(self, numCourses: int, prerequisites: List[List[int]], queries: List[List[int]]) -> List[bool]:
        '''
        we need to check if some course u is a preq of another course v
        if courses are in a cycle, each is a prequiste of the aother
        bfs from each course and keep track in array
        reachable[i][j], where j is reachable from i
        us array to check the queires given
        note in queries, classes will not point to themselves
        '''
        graph = defaultdict(list)
        for u,v in prerequisites:
            graph[u].append(v)
        
        can_reach = [[False]*numCourses for _ in range(numCourses)]
        for i in range(numCourses):
            seen = set()
            self.dfs(graph,i,i,can_reach,seen)
        
        ans = []

        for u,v in queries:
            ans.append(can_reach[u][v])
        
        return ans
    
    def dfs(self,graph,start,curr,can_reach,seen):
        if start != curr:
            can_reach[start][curr] = True
        seen.add(curr)

        for neigh in graph[curr]:
            if neigh not in seen:
                self.dfs(graph,start,neigh,can_reach,seen)

#floyd warshall
class Solution:
    def checkIfPrerequisite(
        self,
        numCourses: int,
        prerequisites: List[List[int]],
        queries: List[List[int]],
    ) -> List[bool]:
        isPrerequisite = [[False] * numCourses for _ in range(numCourses)]

        for edge in prerequisites:
            isPrerequisite[edge[0]][edge[1]] = True

        for intermediate in range(numCourses):
            for src in range(numCourses):
                for target in range(numCourses):
                    # If there is a path src -> intermediate and intermediate -> target, then src -> target exists as well
                    isPrerequisite[src][target] = isPrerequisite[src][
                        target
                    ] or (
                        isPrerequisite[src][intermediate]
                        and isPrerequisite[intermediate][target]
                    )

        answer = []
        for query in queries:
            answer.append(isPrerequisite[query[0]][query[1]])

        return answer
    
#############################################
# 2658. Maximum Number of Fish in a Grid
# 28JAN25
##############################################
class Solution:
    def findMaxFish(self, grid: List[List[int]]) -> int:
        '''
        land cell is grid[i][j] = 0
        water cell is grid[i][j] > 0
        fisher can start at and water cell (i,j) and do opertions any number of times
            * catch all fick at cell (r,c)
            * move to any adjcant water cell

        dfs/bfs/union find on connected componsnets and take the max
        '''
        rows,cols = len(grid),len(grid[0])
        seen = set()
        ans = 0

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] > 0 and (i,j) not in seen:
                    score = [0]
                    self.dfs((i,j),grid,seen,score)
                    ans = max(ans,score[0])
        return ans
    
    def dfs(self,curr,grid,seen,score):
        seen.add(curr)
        i,j = curr
        score[0] += grid[i][j]
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        rows,cols = len(grid),len(grid[0])

        for di,dj in dirrs:
            ii = i + di
            jj = j + dj
            if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] > 0 and (ii,jj) not in seen:
                self.dfs((ii,jj),grid,seen,score)

#bfs
class Solution:
    def findMaxFish(self, grid: List[List[int]]) -> int:
        '''
        land cell is grid[i][j] = 0
        water cell is grid[i][j] > 0
        fisher can start at and water cell (i,j) and do opertions any number of times
            * catch all fick at cell (r,c)
            * move to any adjcant water cell

        dfs/bfs/union find on connected componsnets and take the max
        '''
        rows,cols = len(grid),len(grid[0])
        seen = set()
        ans = 0

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] > 0 and (i,j) not in seen:
                    s = self.bfs((i,j),grid,seen)
                    ans = max(ans,s)
        return ans
    
    def bfs(self,curr,grid,seen):
        score = 0
        q = deque([])
        seen.add(curr)
        q.append(curr)
        
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        rows,cols = len(grid),len(grid[0])
        while q:
            i,j = q.popleft()
            score += grid[i][j]
            seen.add((i,j))

            for di,dj in dirrs:
                ii = i + di
                jj = j + dj
                if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] > 0 and (ii,jj) not in seen:
                    q.append((ii,jj))
                    seen.add((ii,jj))
        return score

#union find
class DSU:
    def __init__(self,n,grid):
        self.parent = [i for i in range(n)]
        self.size = [1]*n
        self.num_fish = [0]*n
        rows, cols = len(grid),len(grid[0])
        for i in range(rows):
            for j in range(cols):
                self.num_fish[i*cols + j] = grid[i][j]
    
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        x_par = self.find(x)
        y_par = self.find(y)

        if x_par == y_par:
            return
        if self.size[x_par] >= self.size[y_par]:
            self.size[x_par] += self.size[y_par]
            self.size[y_par] = 0
            self.parent[y_par] = x_par
            self.num_fish[x_par] += self.num_fish[y_par]

        else:
            self.size[y_par] += self.size[x_par]
            self.size[x_par] = 0
            self.parent[x_par] = y_par
            self.num_fish[y_par] += self.num_fish[x_par]
    
    def get_max(self,):
        return max(self.num_fish)

class Solution:
    def findMaxFish(self, grid: List[List[int]]) -> int:
        '''
        we can do union find,
        but first we need to represent and index for (i,j)
        (i,,j) - > i*cols + j
        '''
        rows, cols = len(grid),len(grid[0])
        dsu = DSU(rows*cols,grid)
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] > 0:
                    for di,dj in dirrs:
                        ii = i + di
                        jj = j + dj
                        if 0 <= ii < rows and 0 <= jj < cols and grid[ii][jj] > 0:
                            #neigh_idx = ii*cols + jj
                            dsu.union(i*cols + j, ii*cols + jj ) 
        
        return dsu.get_max()

######################################################
# 2127. Maximum Employees to Be Invited to a Meeting
# 28JAN25
######################################################
class Solution:
    def maximumInvitations(self, favorite: List[int]) -> int:
        '''
        also important to note there is only one favorite, not multiple favorites
            how would it work if there are multiple favorites??
            interesting that the table is round
        a person will only attend if they can sit next to their favorite person at the table
        hint says to make graph where there is a directed edges from fav[i] to i
        then there will be a combindatino of cycles and acyclic edges
        we can choose employees by :
        1. selecting a cycle of the raph
            the employees that do not lie in the cycle can cnever se seated (unless cycle is of length 2)
        1. combining acylic chains
            at most two chains can be combined by a cycle of lenght 2, where each chain ends on one of the emlpyees in the cycle
        idea is that cycles can be seated together, and we can add to the cycles with chains
        and chains can link other cycles
        need to find the longest possible extended paths for each endpoint of a 2-cycle and combine
        thrree steps
        1. cycle detection
        2. finding longest path
            single cycle > 2, find largest
            multiple 2 cycles, we can link them
            max possible length for any group is the sum of the longest paths from both endpoints + 1
        3. final comparison
            we take the larger of the two, max length form extended paths or size of the largest cycle
        '''
        num_people = len(favorite)
        reversed_graph = [[] for _ in range(num_people)]

        # Build the reversed graph where each node points to its admirers
        for person in range(num_people):
            reversed_graph[favorite[person]].append(person)

        longest_cycle = 0
        two_cycle_invitations = 0
        visited = [False] * num_people

        # Find all cycles in the graph
        for person in range(num_people):
            if not visited[person]:

                # Track visited persons and their distances
                visited_persons = {}
                current_person = person
                distance = 0
                while True:
                    if visited[current_person]:
                        break
                    visited[current_person] = True
                    visited_persons[current_person] = distance
                    distance += 1
                    next_person = favorite[current_person]

                    # Cycle detected
                    if next_person in visited_persons:
                        cycle_length = distance - visited_persons[next_person]
                        longest_cycle = max(longest_cycle, cycle_length)

                        # Handle cycles of length 2
                        if cycle_length == 2:
                            visited_nodes = {current_person, next_person}
                            two_cycle_invitations += (2 + self.bfs(next_person, visited_nodes, reversed_graph) 
                            + self.bfs(current_person,visited_nodes,reversed_graph))
                        break
                    current_person = next_person

        return max(longest_cycle, two_cycle_invitations)
    
    # Calculate the maximum distance from a given start node
    def bfs(self, start_node: int, visited_nodes: set, reversed_graph: List[List[int]]) -> int:
        # Queue to store nodes and their distances
        q = deque([])
        q.append((start_node,0))
        max_distance = 0
        while q:
            current_node, current_distance = q.popleft()
            for neighbor in reversed_graph[current_node]:
                if neighbor in visited_nodes:
                    continue  # Skip already visited nodes
                visited_nodes.add(neighbor)
                q.append((neighbor, current_distance + 1))
                max_distance = max(max_distance, current_distance + 1)
        return max_distance

#######################################
# 684. Redundant Connection
# 29JAN25
#######################################
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        '''
        drop and edge one at a time from the end if the input, and then dfs to see if we can touch the whole tree
        that's n^3, wont work
        if a node is a leaf, its edge cannot be removed, otherwise we wouldn't have a tree
        find the cycle nodes, then check the edges that are in the cycle, if they're in the cycle they can be removed
        this is just kahsn starting from the leaf node
        '''
        n = len(edges)
        graph = defaultdict(list)
        indegree = [0]*(n+1)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
            indegree[v] += 1
            indegree[u] += 1
        
        incycle = [True]*(n+1)
        q = deque([])
        for i in range(1,n+1):
            if indegree[i] == 1:
                q.append(i)
        
        while q:
            curr = q.popleft()
            incycle[curr] = False
            for neigh in graph[curr]:
                indegree[neigh] -= 1
                if indegree[neigh] == 1:
                    q.append(neigh)
        
        for u,v in edges[::-1]:
            if incycle[u] and incycle[v]:
                return [u,v]
        return [-1]
        
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        '''
        intuition:
        we can discard en edge if it connected two noes that were already part of the same connected componenet
            dfs/union find
        before adding the edge to the graph, we see if we can first reach u to v
        if we can that edge is a redundcant connection
        '''
        graph = defaultdict(list)
        for u,v in edges:
            visited = set()
            if self.dfs(u,v,visited,graph):
                return [u,v]
            
            graph[u].append(v)
            graph[v].append(u)

        return -1
    
    def dfs(self,curr,target,visited,graph):
        visited.add(curr)
        if curr == target:
            return True
        
        for neigh in graph[curr]:
            if neigh not in visited:
                if self.dfs(neigh,target,visited,graph):
                    return True
        return False
    
#using dfs to track cycle nodes
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        '''
        we can also use dfs to find the cycle nodes
        once we find a node in the cycle, we can just follow the parent pointers back up
        if we encounter a node that has already been visited and the node we are coming from is different
        from its parent, we can conclude the node is in the cycle
        '''
        graph = defaultdict(list)
        n = len(edges)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = set()
        parent = defaultdict(lambda : -1)
        cycle_start = [-1]
        self.dfs(1,visited,graph,parent,cycle_start)

        #start from cycle start, since its in the cycle and follow parent
        curr = cycle_start[0]
        cycle_nodes = defaultdict(lambda : -1)
        while True:
            cycle_nodes[curr] = 1
            curr = parent[curr]
            if curr == cycle_start[0]:
                break
        
        for u,v in edges[::-1]:
            if cycle_nodes[u] == 1 and cycle_nodes[v] == 1:
                return [u,v]
        return []

    def dfs(self,curr,visited,graph,parent,cycle_start):
        visited.add(curr)
        for neigh in graph[curr]:
            if neigh not in visited:
                parent[neigh] = curr
                self.dfs(neigh,visited,graph,parent,cycle_start)
            #if neode is is visited, check if differnt from parent, if differ from parent we can start the cycle here
            elif neigh in visited and neigh != parent[curr] and cycle_start[0] == -1:
                cycle_start[0] = neigh
                parent[neigh] = curr

########################################################
# 2493. Divide Nodes Into the Maximum Number of Groups
# 30JAN25
#######################################################
#dammit
class Solution:
    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        '''
        need to divide nodes into m groups such that
        * each node in graph belongs to one group
        * for every edge connection [u,v] if u belongs to group with index and b belongs to group with index y, then |y-x| = 1
        basically the groups need to be in ascending order
        hints, if graph is bipartite, we can't do it
        then we solve the porblme for each connected component indepdnetly
        for a connected component, the maximum number of groups is just the depth in a bfs tree after rooting at node v
        bipartitie solutino, just color the nodes a different color
        
        the maximum number of groups in a componenets is determined by the longest shortest path between any pair of nodes in that componenet
        this is similar to finding the heigh of the componenet if it were structures like at tree
        with differnt nodes as potential roots
        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        #check biparite condition, good review
        seen = {}
        for i in range(n):
            if i not in seen:
                if not self.dfs(graph,i,0,seen):
                    return -1

        #graph is not bipartite, first get conected compoennts
        seen = set()
        comps = []

        for i in range(1,n):
            if i not in seen:
                curr_comp = []
                self.dfs2(i,graph,curr_comp,seen)
                comps.append(curr_comp)
        
        #now solve the problem indivudally for each comp
        groups = 0
        for c in comps:
            temp = self.bfs(c,graph)
            groups += temp
        
        return groups
    #utility to check if bipartite
    def dfs(self,graph,curr,color,seen):
        if curr in seen:
            if seen[curr] != color:
                return False
            return True
        seen[curr] = color
        for neigh in graph[curr]:
            if not self.dfs(graph,neigh,1-color,seen):
                return False
        return True
    
    def dfs2(self,curr,graph,curr_comp,seen):
        seen.add(curr)
        curr_comp.append(curr)
        for neigh in graph[curr]:
            if neigh not in seen:
                self.dfs2(neigh,graph,curr_comp,seen)

    #find longest shortest path
    def bfs(self,comp,graph):
        ans = 0
        for c in comp:
            dists = defaultdict(lambda : float('inf'))
            dists[c] = 0
            seen = set()
            q = deque([])
            q.append(c)
            while q:
                curr = q.popleft()
                if curr in seen:
                    continue
                seen.add(curr)
                for neigh in graph[curr]:
                    if neigh not in seen:
                        neigh_dist = dists[curr] + 1
                        if dists[neigh] > neigh_dist:
                            dists[neigh] = neigh_dist
                            q.append(neigh)
            
            ans = max(ans, max(dists.values()))
        
        return ans + 1
    
class Solution:
    def magnificentSets(self, n: int, edges: List[List[int]]) -> int:
        '''
        first check bipartite criteria
        then for each node, get the longest shortest path if using each node as starting

        '''
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)
        #check biparite condition, good review
        seen = {}
        for i in range(n):
            if i not in seen:
                if not self.dfs(graph,i,0,seen):
                    return -1
            
        #for each node, get the longest shortest path
        distances = []
        for i in range(1,n+1):
            distances.append(self.longest_shortest(graph,i,n+1))
        
        #calculate max number of groups for all components
        max_groups = 0
        visited = [False]*(n+1)
        for node in range(1,n+1):
            if visited[node]:
                continue
            curr_groups = self.count_groups(graph,node,distances,visited)
            max_groups += curr_groups
        
        return max_groups

    #utility to check if bipartite
    def dfs(self,graph,curr,color,seen):
        if curr in seen:
            if seen[curr] != color:
                return False
            return True
        seen[curr] = color
        for neigh in graph[curr]:
            if not self.dfs(graph,neigh,1-color,seen):
                return False
        return True
    
    def longest_shortest(self, graph,start,n):
        q = deque([start])
        visited = [False]*n
        visited[start] = True
        dist = 0

        while q:
            N = len(q)
            for _ in range(N):
                curr = q.popleft()
                for neigh in graph[curr]:
                    if not visited[neigh]:
                        q.append(neigh)
                        visited[neigh] = True
            dist += 1
        
        return dist
    #recursion/dp here
    def count_groups(self,graph,curr,distances,visited):
        #start with distance as max nodes
        max_num_nodes = distances[curr - 1]
        visited[curr] = True
        for neigh in graph[curr]:
            if not visited[neigh]:
                max_num_nodes = max(max_num_nodes, self.count_groups(graph,neigh,distances,visited))
        return max_num_nodes








