#######################################
# 121. Best Time to Buy and Sell Stock
# 01FEB22
#######################################
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        we are given only 1 transactions
        brute force is check all
        '''
        max_prof = 0
        for i in range(len(prices)):
            for j in range(i+1,len(prices)):
                prof = prices[j] - prices[i]
                if prof > max_prof:
                    max_prof = prof
        return max_prof

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        we want to keep track of the smalest price and the cucrnet price
        '''
        min_price = float('inf')
        max_prof = 0
        N = len(prices)
        
        for i in range(N):
            min_price = min(min_price,prices[i])
            max_prof = max(max_prof, prices[i] - min_price)
            
        
        return max_prof

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        we could also treat this like Kadane's algo
        and just find the sum of the largest cnotig subarray using consecutive diffs
        '''
        max_prof = 0
        N = len(prices)
        curr_diff = 0
        
        for i in range(1,N):
            curr_diff += prices[i] - prices[i-1]
            if curr_diff < 0:
                curr_diff = 0
            max_prof = max(max_prof, curr_diff)
        
        return max_prof

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        i can do this recursively, if i had the differencs array
        than just frame it as dp
        dp(i) is max sum at this index
        '''
        N = len(prices)
        
        #edge case
        if N == 1:
            return 0
        diffs = [prices[i] - prices[i-1] for i in range(1,N)]
        
        memo = {}
        
        def dp(i):
            if i < 0:
                return 0
            if i in memo:
                return memo[i]
            take = dp(i-1) + diffs[i]
            no_take = diffs[i]
            ans = max(take,no_take)
            memo[i] = ans
            return ans
        
        dp(len(diffs)-1)
        #return profit at least 0
        return max(0,max(memo.values()))
            
#########################################################
# 1101. The Earliest Moment When Everyone Become Friends
# 01FEB22
########################################################
class UnionFind:
    def __init__(self,size):
        self.group = [i for i in range(size)]
        self.size = [0]*size
        
    #recursively follow parent poitners
    def find(self,person):
        if self.group[person] != person:
            self.group[person] = self.find(self.group[person])
        return self.group[person]
    #union
    def union(self,a,b):
        #return true if and b are not connected and connect
        #otherwise return False
        parent_a = self.find(a)
        parent_b = self.find(b)
        is_merged = False
        
        if parent_a == parent_b:
            return is_merged
        is_merged = True
        #merge, make the lower rank group pointer to the higher rank group
        if self.size[parent_a] > self.size[parent_b]:
            self.group[parent_b] = parent_a
        elif self.size[parent_a] < self.size[parent_b]:
            self.group[parent_a] = parent_b
        else:
            self.group[parent_a] = parent_b
            self.size[parent_b] += 1
        
        return is_merged

class Solution:
    def earliestAcq(self, logs: List[List[int]], n: int) -> int:
        '''
        we are given an array of logs 
        we have n-1 people [0, n-1]
        and logs[i][0] respresents the time stam when logs[i][1] and logs[i][2] become friends
        friend ship is symmetric, meaning that if a if friends with b, then b is friends with a
        also a is acquainted with b if a is friends with b or a is a friend of someone acruqnated with b
        return the earliest time for which every perosn became acquainted with every other person
        if there is no such earlier time, return -1
        
        into, this probably screams union find, we need to write out the API
            find(a) finds the group a belongs to
            union(a) mergees the two and sets is parent
                union(a,b) == True when a and b are merged
                union(a,b) == False, when they are already in the same group
            we merger the a,b in order of log time
            each log adds more connections among the individuals, a connection is usefil if the two individals are separated (disjoint) or redudnant if the two individuals are conected via other indivuduals
            intially we treat individuals as a seperate group, the number of groups decrease along the useful mergeing oeprations
            the moment the number of groups is reduced to 1 one, is the earliest momemetn
        
        inutition for union find
            nodes are people, and edges are the logs
            edges indicated friends becoming acquainted
            really we want the earliest edge that connects all the people
            keep connecting until we have one big friend group!
        
        notes on time complexity:
            O(M/alpha(N)) where \alpha(N) is the inverse Ackerman Function
            let N be the number of people and M be the number of operations
            O(N + MlogM + M\alphaN)
            
        '''
        #sort by logs
        logs.sort(key = lambda x: x[0])
        
        #initially each person is its own group
        uf = UnionFind(n)
        
        count_group = n
        for t,a,b in logs:
            if uf.union(a,b):
                count_group -= 1
                
            #check if we have 1 group
            if count_group == 1:
                return t
        return -1

######################################
# 521. Longest Uncommon Subsequence I
# 01FEB22
#######################################
#brute force
class Solution:
    def findLUSlength(self, a: str, b: str) -> int:
        '''
        we cab defube an uncommon subsequence between two strings as a string that is a subsequence of one by not the other
        we define a subsequence of string s that can be obtained after deleting anu numberof chars
        we can brute force by generating all subsequence for a and b
        then count them up and return the length which has frequency of one
        '''
        counts = Counter()
        def rec(string,i,path):
            if len(path) > 0:
                counts["".join(path)] += 1
            if i < len(string):
                rec(string,i+1,path+[string[i]])
                rec(string,i+1,path)
            else:
                return
        
        rec(a,0,[])
        rec(b,0,[])
        ans = 0
        for k,v in counts.items():
            if v == 1:
                ans = max(ans,len(k))
        
        return ans if ans != 0 else -1

#tricky way
class Solution:
    def findLUSlength(self, a: str, b: str) -> int:
        '''
        if a is b, then there is no uncommon subsequence
        if a is not b, then the longest uncommon subsequcne if the max(length a,length b)
        '''
        if a == b: return -1
        else: return max(len(a),len(b))

###################################
# 438. Find All Anagrams in a String
# 02FEB22
####################################
#brute force
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        '''
        brute force wouold be to check each window of size p
        '''
        size_s = len(s)
        size_p = len(p)
        
        #edge case, when s is smaller than p, i cant make any anagram of p
        if size_s < size_p:
            return []
        
        starting_idxs = []
        #iterate  i from 0 to the last window of size p
        for i in range(0,len(s)-len(p)+1):
            #get window
            window = s[i:len(p)+i]
            #we'll delete each char p from this window
            p_list = list(p)
            for j in range(len(window)):
                if window[j] in p_list:
                    p_list.remove(window[j])
            
            if len(p_list) == 0:
                starting_idxs.append(i)
        
        return starting_idxs

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        '''
        given two strings s and p, return the start indices of p's anagrams in s
        i cant use a sliding windo and just check each slice is an anagram
        when i advance, add char and coung too right and remove frmot left
        '''
        size_s = len(s)
        size_p = len(p)
        
        #edge case, when s is smaller than p, i cant make any anagram of p
        if size_s < size_p:
            return []
        
        count_p = Counter(p)
        count_s = Counter()
        res = []
        
        for i in range(size_s):
            #add char and count to cuont_s
            count_s[s[i]] += 1
            #clear letter once we have size_p
            if i >= size_p:
                #removoe occurences
                if count_s[s[i-size_p]] == 1:
                    del count_s[s[i-size_p]]
                else:
                    count_s[s[i-size_p]] -= 1
            #check that they match
            if count_p == count_s:
                res.append(i - size_p + 1)
        
        return res

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        '''
        we can also do this with a while loop
        initally conver p and s to coounters
        use s to to be the frist slice of len(p)
        '''
        size_s = len(s)
        size_p = len(p)
        
        #edge case, when s is smaller than p, i cant make any anagram of p
        if size_s < size_p:
            return []
        
        p_count = Counter(p)
        curr_window = Counter(s[0:len(p)-1])
        
        left = 0
        right = len(p)
        ans = []
        
        while right <= len(s):
            #keep expanding right, but ooffset by 1
            #add in char
            curr_window[s[right-1]] += 1
            if p_count == curr_window:
                ans.append(left)
            #remove levt
            curr_window[s[left]] -= 1
            if curr_window[s[left]] == 0:
                del curr_window[s[left]]
            
            left += 1
            right += 1
        
        return ans

###############################
# 146. LRU Cache
# 02FEB22
###############################
'''
we can used orderdict class and inherit into LRU cache
same as a dict put keeps order in which key:values were added
'''
from collections import OrderedDict

class LRUCache(OrderedDict):
    '''
    we need to store the capcity
    get returns the value of the key otherwise return -1
    put updates the value of they key if the key exitis
        otherwise add k-valur pair to cache
        if the number of keys exceeds cap from this operation, evict the least recently used key
    '''

    def __init__(self, capacity: int):
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self: #this is call, we can just call self, since its an OrdereDict
            return -1
        #when accessing from LRU, it becomes recently used, and not least recently used
        #if we can retrieve it, move to end and return it
        self.move_to_end(key)
        return self[key]

    def put(self, key: int, value: int) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        #evict LRU if we exceed cap
        if len(self) >self.capacity:
            self.popitem(last = False)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

#the real way would be to use a doubly linked list and hash map
'''
we can use a doubly linkedin list
and ate ach node store a hashmap with key and value

the add and remove nodes are operations for inserting and deleting into a doubly linbked list
we need to move back to head, when we tocuh a node
and move to tail when we get too big
'''

class D_LINKED_NODE():
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None
        
class LRUCache:
    def _add_node(self,node):
        #always add the new node right after head
        #this is sort of like inserting into a doubly linked next
        node.prev = self.head
        node.next = self.head.next
        
        #reconnect head to node
        self.head.next.prev = node
        self.head.next = node
        
    def _remove_node(self,node):
        prev = node.prev
        new = node.next
        
        prev.next = new
        new.prev = prev
    
    def _move_to_head(self,node):
        #move certain node in between to the head
        self._remove_node(node)
        self._add_node(node)
        
    def _pop_tail(self):
        res = self.tail.prev
        self._remove_node(res)
        return res
        

    def __init__(self, capacity: int):
        self.cache = {}
        self.size = 0 
        self.capacity = capacity
        self.head = D_LINKED_NODE()
        self.tail = D_LINKED_NODE()
        
        #initially connect head to tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        node = self.cache.get(key,None)
        if not node:
            return -1
        #otherwise return, but not ebfore moving node back to head
        self._move_to_head(node)
        return node.value
        

    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key, None)
        if not node:
            newNode = D_LINKED_NODE()
            newNode.key = key
            newNode.value = value
            
            self.cache[key] = newNode
            self._add_node(newNode)
            self.size += 1
            
            #evict if we are too big
            if self.size > self.capacity:
                #pop from the tail
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1
        else:
            #update value
            node.value = value
            self._move_to_head(node)
        
        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

########################
# 454. 4Sum II
# 03FEB22
########################
#got ncubed!
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        '''
        the lengnth of  the arrays is small enough to allow cubic solutiono
        i can find the sum in the first three arrays
        then check if the comolement is in nums4
        i can dump nums4 into a mapp into a count
        '''
        #first mapp the last array
        counts = Counter(nums4)
        N = len(nums1)
        
        ans = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    #find complement
                    comp = -(nums1[i]+nums2[j]+nums3[k])
                    if comp in counts:
                        ans += counts[comp]
        
        return ans

class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        '''
        looks like they want an O(N^2) solution
        first observe that nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
        this implies that
        nums1[i] + nums2[j] = -(nums3[k] + nums4[l])
        we can put into a hashmap the sums of the first i and j
        then check for complements of k and l on second pass
        '''
        counts = Counter()
        ans = 0
        N = len(nums1)
        
        for i in range(N):
            for j in range(N):
                SUM = nums1[i] + nums2[j]
                counts[SUM] += 1
        
        for k in range(N):
            for l in range(N):
                COMP = -(nums3[k] + nums4[l])
                if COMP in counts:
                    ans += counts[COMP]
        
        return ans

#don't forget the .get for dicts
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        cnt = 0
        m = {}
        for a in A:
            for b in B:
                m[a + b] = m.get(a + b, 0) + 1
        for c in C:
            for d in D:
                cnt += m.get(-(c + d), 0)
        return cnt

########################################
# 241. Different Ways to Add Parentheses
# 03FEB22
########################################
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        '''
        we first have to notice the recurrence
        we define our operators as '+-*'
    We will call +/-/* symbols.

    (base) case 1: if there is 0 symbol, 
        e.g. 1, we simply return it as a list, [1].

    case 2: if there is 1 symbol, 
        e.g., 1 + 2, we divide it by symbol and add parenthesis in both sides: (1) + (2), (1) and (2) reduce to case 1, and we add the results of both sides and return as a list, [3].

    case 3: if there are 2 symbols, 
        e.g., 1 + 2 + 3, we divide it by symbol and add parenthesis in both sides: (1) + (2 + 3) or (1 + 2) + (3).
        Take (1) + (2 + 3) for example, (1) reduces to case1, (2 + 3) reduces to case 2 (then case 1), finally we add the results of both sides and add the final ressult to the list, [6].
        
        If there are n symbols, for each symbol, we divide it into two parts and add parenthesis in both sides, each side is reduced to a subproblem which can be solved recursively, finally we combine the results of both sides by the symbol and add to the final result list.

...
        '''
        memo = {}
        
        def rec(expression):
            #base case
            if all([op not in expression for op in '+-*']):
                return [int(expression)]
            res = []
            for i,v in enumerate(expression):
                #if its an operation, we need to apply to the left and right
                if any([op in  v for op in '+-*']):
                    left = rec(expression[:i])
                    right = rec(expression[i+1:])
                    for left_ans in left:
                        for right_ans in right:
                            #apply opertions
                            if v == '+':
                                res.append(left_ans + right_ans)
                            elif v == '-':
                                res.append(left_ans - right_ans)
                            else:
                                res.append(left_ans*right_ans)
            memo['expression'] = res
            return res
        
        return rec(expression)

#now lets try top down
#try using i,j instead of expres

class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        '''
        instead of passing a string expression, lets pass indices i and j
        '''
        memo = {}
        def rec(i,j):
            #if we get to a point where expression[i:j] has no op, return this value
            if all([op not in expression[i:j] for op in '+-*']):
                return [int(expression[i:j])]
            res = []
            for k in range(i,j):
                v = expression[k]
                if any([op in  v for op in '+-*']):
                    left = rec(i,k)
                    right = rec(k+1,j)
                    for left_ans in left:
                        for right_ans in right:
                            #apply opertions
                            if v == '+':
                                res.append(left_ans + right_ans)
                            elif v == '-':
                                res.append(left_ans - right_ans)
                            else:
                                res.append(left_ans*right_ans)
            memo[(i,j)] = res
            return res
        
        return rec(0,len(expression))

#it's something like this...
class Solution:
    def diffWaysToCompute(self, expression: str) -> List[int]:
        '''
        now lets try doing this bottom up
        '''
        N = len(expression)
        dp = [[[] for _ in range(N)] for _ in range(N)]
        #fill in base cases
        for i in range(N):
            for j in range(i,N):
                if all([op not in expression[i:j+1] for op in '+-*']):
                    dp[i][j] += [int(expression[i:j+1])]
                    
        #fill in rest
        for i in range(N):
            for j in range(N):
                for k in range(i,j):
                    v = expression[k]
                    if any([op in  v for op in '+-*']):
                        #print(v,dp[i][k-1],dp[k+1][j-1])
                        left = dp[i][k-1]
                        right = dp[k+1][j]
                        res = []
                        for left_ans in left:
                            for right_ans in right:
                                print(left,right)
                                #apply opertions
                                if v == '+':
                                    res.append(left_ans + right_ans)
                                elif v == '-':
                                    res.append(left_ans - right_ans)
                                else:
                                    res.append(left_ans*right_ans)
                        dp[i][j] = res
        
        print(dp)

#actual answer.......fuck, why can't i get this...
import operator, re
ops = {'+': operator.add, '-': operator.sub, '*': operator.mul}

class Solution(object):
    def diffWaysToCompute(self, s):
        nums = [int(x) for x in re.findall(r'[0-9]+', s)]
        opers = re.findall(r'\+|\-|\*', s)
        n, DP = len(nums), {}
        for i in range(n):
            DP[i, i] = [nums[i]]
        for i in range(n - 1):
            DP[i, i+1] = [ops[opers[i]](nums[i], nums[i + 1])]
        for k in range(3, n + 1):
            for i in range(n - k + 1):
                j = i + k - 1
                DP[i, j] = []
                for v in range(i, j):
                    left = DP[i, v]
                    right = DP[v + 1, j]
                    for e1 in left:
                        for e2 in right:
                            DP[i, j].append(ops[opers[v]](e1, e2))
        return DP[0, n - 1]

#############################
# 04FEB21
# 525. Contiguous Array
#############################
#using hashmap
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        '''
        for an array between i and j
        we can get count ones using sum(nums[i:j]) 
        we can get sum zeros by using j-i+1 - sum(nums[i:j])
        notice that for a contiguous subarray, it obeys the proprerty  that if adding 1  and zero is -1
        the  balance is  zeroo
        if a  balance is  is zero, we  know that from the start  of the  array we  have  a contiguouos  subarray
        mark  the positions  of  where  the balance is zero, we can find another  sub array
        points in the  subarray that have balance 0 to the  next are also contiguoours sub array
        more  see if we see  that balance appears again, then the array between thoose  balance poionts
        has  the  same number  of zeroos
        '''
        #make curr_balance mapp
        balance_mapp = {}
        balance = 0
        N = len(nums)
        ans = 0
        
        for i in range(N):
            balance += 1 if nums[i] == 1 else -1
            #check for anoother balance point
            if balance == 0:
                ans = max(ans, i+1)
            if balance in  balance_mapp:
                ans = max(ans, i - balance_mapp[balance])
            else:
                balance_mapp[balance] = i
        
        return ans

#can  also use array, but  need to accuont for negative numbers
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        '''
        '''
        #make curr_balance mapp
        N = len(nums)
        balance_mapp =[0]*(2*N + 1)
        balance = 0
        ans = 0
        
        for i in range(N):
            balance += 1 if nums[i] == 1 else -1
            #check for anoother balance point
            #conovert too index in array
            if balance == 0:
                ans = max(ans, i+1)
            if balance_mapp[balance + N] >= 0:
                ans = max(ans, i - balance_mapp[balance +  N]-1)
            else:
                balance_mapp[balance + N] = i 
        
        return ans

####################################
# 245. Shortest Word Distance III
# 04FEB22
#####################################
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        mapp = collections.defaultdict(list)
        for i,word in enumerate(wordsDict):
            mapp[word].append(i)
            
        #get their posisitions
        pos1 = mapp[word1]
        pos2 = mapp[word2]
        #case 1: when they are not the same word, treat like the first
        if word1 != word2:
            ans = float('inf')
            i,j = 0,0
            #we don't need to traverse the entirey of both lists, just stop when the we done with the shortes one
            #why? if we got to the end of the shorter list, and the second list must be increasing, we 
            #would only every increase the the diff
            while i < len(pos1) and j < len(pos2):
                ans = min(ans, abs(pos1[i]-pos2[j]))
                if pos1[i] < pos2[j]:
                    i += 1
                else:
                    j += 1

            return ans
        
        else:
            #pass one of the indicies arrays and record the min diff
            ans = float('inf')
            for i in range(1,len(pos1)):
                ans = min(pos1[i] - pos1[i-1],ans)
            
            return ans

class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        '''
        we can just do this in one pass, similar to the first problem in this series
        keep track of a last pointer
        check for word1 or word1
        set the last pointer to the last seen index of either word1 or word2
        then just update
        '''
        last_seen_idx = None
        ans = float('inf')
        N = len(wordsDict)
        same = word1 == word2
        for i in range(N):
            #if this word either matches word1 or word2
            if wordsDict[i] == word1  or wordsDict[i] == word2:
                #if we have last seen either of the words
                if last_seen_idx != None:
                    #if they are the same, or if the last seen idx is not the samme is current
                    #we only could have got here if we matched word1 or word2
                    #or that they are they are the same
                    if same or wordsDict[last_seen_idx] != wordsDict[i]:
                        ans = min(ans, i - last_seen_idx)
                #save last_seen
                last_seen_idx = i

        return ans


