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


#################################
# 23. Merge k Sorted Lists
# 05FEB22
#################################
#brute force solution
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        '''
        well the naive way would be to get all the elements, sort and recreate the list
        '''
        elements = []
        for linked in lists:
            while linked:
                elements.append(linked.val)
                linked = linked.next
        #sort
        elements.sort()
        dummy = ListNode()
        cur = dummy
        for num in elements:
            cur.next = ListNode(val=num)
            cur = cur.next
        return dummy.next

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        '''
        i cant megere linked lists in in pairs
        take the first one, merge with the first one, and make that new list
        then with the new list merge again with the second one
        '''
        #edge case
        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]
        
        def mergeTwo(l1,l2):
            #mergine two at time and reutring the new one
            dummy = curr = ListNode()
            while l1 and l2:
                if l1.val < l2.val:
                    curr.next = ListNode(l1.val)
                    l1 = l1.next
                else:
                    curr.next = ListNode(l2.val)
                    l2 = l2.next
                curr = curr.next
            
            #left over
            if l1:
                curr.next = l1
            if l2:
                curr.next = l2
            return dummy.next
                
        result = lists[0]
        for linked in lists[1:]:
            result = mergeTwo(result, linked)
            
        return result

#using a heap
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        '''
        i could use a heap, prrioty queue and keep taking the min
        when i take the min move the pointer to the correspoding linked list
        '''
        heap = []
        
        for i in range(len(lists)):
            if lists[i]:
                heappush(heap,(lists[i].val,i))
        
        dummy = curr = ListNode()
        
        while heap:
            val, i = heappop(heap)
            #set the value
            curr.next = ListNode(val)
            #now move point
            if lists[i].next:
                #move pointer and push next
                lists[i] = lists[i].next
                heappush(heap,(lists[i].val,i))
            curr = curr.next
        return dummy.next

##############################################
# 80. Remove Duplicates from Sorted Array II
# 06FEB22
##############################################
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        '''
        we need to remove duplicates in the integer array so that their multiplicty is at most 2
        we can push the duplicated elements to the end of the array (it does not matter beyond this)
        finally we want to return k, which would be the number of elements that have been mulitpicty > 2
        brute force would be to pass the array and check current occurences of each eleemnts
        
        algo:
            1. lets create two pointers, one pointing into the array, and the othe recording the count
            2. lower bound for count will always be 1
            3. if we find the curr element is the same as prev element, we increase the count by 1
                if count > 2, this is repeated elment and we delete it fom the array,
                we can also decrease the current pointer by 1
            4. if curr elment is not teh same as prev, reset count to 1
            5. return the number of duplicteates, which is just the length of the array
        '''
        i = 1
        count = 1
        while i < len(nums):
            #same element
            if nums[i] == nums[i-1]: 
                count += 1
                
                #greater than 2
                if count > 2:
                    nums.pop(i)
                    
                    #lower pointer 
                    i -= 1
            else:
                count = 1
                
            i += 1
        
        return len(nums)

#linear time, but using up space
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        '''
        anothe way would be to use up space and record indices of elements that have been duplicated
        then just take the elements not in thos indcies and rebuild the array
        this is not 0(1)
        '''
        idxs = set()
        i = 1
        count = 1
        while i < len(nums):
            if nums[i] == nums[i-1]:
                count += 1
                
                if count > 2:
                    idxs.add(i)
            
            else:
                count = 1
            
            i += 1
        
        new_nums = []
        for i in range(len(nums)):
            if i in idxs:
                continue
            new_nums.append(nums[i])
        
        nums[:len(new_nums)] = new_nums
        return len(new_nums)

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        '''
        in the last approach, we modified in place, but also has to return the len of the array
        we can just return how many number of elements that have multiplcity > 2
        we need to use two pointers here:
        
        algo: 
            1. keep two pointers i and j; i is curr poitner, j is potision where we can place unawanted
            2. keep count variable
            3. if curr elment == prev element, incrematn count
                if count > 2, then we have an un wanted eleemnt, in this case we keep moving forward
            4. if count <= 2, we can move the element from index i to index j
            5. if we ecnoutner new element, resetcount to 1, and move this element to j
        '''
        next_available = 1
        count = 1
        
        for i in range(1,len(nums)):
            
            #same element
            if nums[i] == nums[i-1]:
                count += 1
            else:
                count = 1
            
            #for a count <= 2, we can copy the elemnt over from i to j
            if count <= 2:
                nums[next_available] = nums[i]
                next_available += 1
        
        return next_available

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        '''
        another way wouyld be to always wap 2 away
        if we can't swap move up the
        '''
        if len(nums) <= 2:
            return len(nums)
        
        j = 2
        for i in range(2,len(nums)):
            nums[j] = nums[i]
            if nums[j] != nums[j-2]:
                #because they can only be duplicated, we need maintain the criteria ofr the array
                j += 1
        
        return j
        
#################################
# 389. Find the Difference
# 07FEB22
#################################
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        '''
        i can use a counter to find the count of chars in s
        then pass t and decrement along
        '''
        counts = Counter(s)
        for ch in t:
            #not in counts
            if ch not in counts:
                return ch
            #must be in counts
            else:
                counts[ch] -= 1
                if counts[ch] < 0:
                    return ch
        
        print(counts)

#sorting
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        '''
        another way is to just sort
        then return the unmatching char
        otherwise it's at the end
        '''
        s = sorted(s)
        t = sorted(t)
        
        i = 0
        while i < len(s):
            if s[i] != t[i]:
                return t[i]
            
            i += 1
        
        return t[i]

#bit manipulation
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        '''
        we can use xor
        recall xor is true if only one of the bits is true
        if we xor all elements in s
        then x all elements in t
        the final is what is different
        '''
        # Initialize ch with 0, because 0 ^ X = X
        # 0 when XORed with any bit would not change the bits value.
        ch = 0

        # XOR all the characters of both s and t.
        for char_ in s:
            ch ^= ord(char_)

        for char_ in t:
            ch ^= ord(char_)

        # What is left after XORing everything is the difference.
        return chr(ch)

#########################################
# 530. Minimum Absolute Difference in BST
# 07FEB22
#########################################
#close one
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        '''
        i would only ever have to compare parent to child nodes, never up along a path because that would result in a greater differene
        '''
        self.ans = float('inf')
        
        def dfs(node,parent_value):
            if not node:
                return
            
            local_min = abs(node.val - parent_value)
            #update
            self.ans = min(self.ans,local_min)
            dfs(node.left,node.val)
            dfs(node.right,node.val)
            
        dfs(root,float('inf'))
        return self.ans

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        '''
        we can just do in order, and find the min of the first order differences
        
        '''
        L = []
        def dfs(node):
            if node.left: 
                dfs(node.left)
            L.append(node.val)
            if node.right: 
                dfs(node.right)
        dfs(root)
        
        return min(b - a for a, b in zip(L, L[1:]))


#####################################
# 247. Strobogrammatic Number II
# 07FEB22
#####################################
class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        '''
        a strobrogrammatic number is a number that reads the same when inverted 180 degrees
        for an even length there is no center
        for an odd length there is a center
        the digits that can be strobgrammic are
        0:0
        1:1
        8:8
        6:9
        9:6
        unflipped:flipped
        for odd length, the digit at the center cannot change, so the centers must be 0,1,8, if odd
        when two positions interchange, we can use both types of digits which change into one another or we put two same digits which remain the same when rotated 
        
        approach 1 recursion:
            casework, look for patterns tryin to create with n = 1 up to n = 4
            n = 1:
                0,1,8
            n = 2:
                11, 88,69,96
                note, we cannot do 00, since this is not a valid two digit number
            n = 3,
                index 1 is the center, and can be 0,1,8
                suppose we had all the 1-digit strobrogrammtic numbers, to find the 3 digits numbers, we just need to append one extra digit at th ebgning and end
                i.e fix at the center, every 1-digit strobrogammitc number, and use the reversible pairs
            n = 4
                for this, index 0 and 3 will interhcnage, and index 1 and 2 will interchange
                to find the 4 digit stobrorammitc numbers, we just need to append 1 extra digit at the begnning or end
            
        to find n digit strobrogrammitc numbers,we first need to find the solutions to n-2 digits, and thena ppend reversible digits to beginning and end
        
        the transition function becomes:
            generateStroboNumbers(N) = List("digit1" + "number" + "digit2"
                                for each number in generateStroboNumbers(N - 2)
                                for each (digit1, digit2) in reversiblePairs
                               )
        base cases:
            when n = 0, base case is the results of the emptry string, n = [""]
            when n = 1, base case is jsut ['0','1','8']
        
        note: we use base case for n = 0, instead of n = 2, becaue we cannot include '00' as a valid number
        if we did n = 1, we would alwasy reduce to n = 0, or n = 1, then we could try bulding using '0'
        we can contorl for this by passing in final length as paramter

    algo:
    Initialize a data structure reversiblePairs, which contains all pairs of reversible digits.

    Call and return the recursive function, generateStroboNumbers(n, finalLength), where the first argument indicates that the current call will generate all n-digit strobogrammatic numbers. The second argument indicates the length of the final strobogrammatic numbers that we will generate and will be used to check if we can add '0' to the beginning and end of a number.

    Create a function generateStroboNumbers(n, finalLength) which will return all strobogrammatic numbers of n-digits:

    Check for base cases, if n == 0 return an array with an empty string [""], otherwise if n == 1 return ["0", "1", "8"].
    Call generateStroboNumbers(n - 2, finalLength) to get all the strobogrammatic numbers of (n-2) digits and store them in subAns.
    Initialize an empty array currStroboNums to store strobogrammatic numbers of n-digits.
    For each number in prevStroboNums we append all reversiblePairs at the beginning and the end except when the current reversible pair is '00' and n == finalLength (because we can't append '0' at the beginning of a number) and push this new number in ans.
    At the end of the function, return all the strobogrammatic numbers, i.e. currStroboNums.
        '''
        pairs = [['0','0'],['1','1'],['6','9'],['8','8'],['9','6']]
        
        def rec(n,final_length):
            #base cases
            if n == 0:
                return ['']
            if n == 1:
                return ['0','1','8']
            
            prev_nums = rec(n-2,final_length)
            curr_nums = []
            for prev in prev_nums:
                for pair in pairs:
                    #we cannot add zeros to the front
                    #nor can we add any digits to then of we have finishted the current length
                    if pair[0] != '0' or n != final_length:
                        curr_nums.append(pair[0]+prev+pair[1])
            
            return curr_nums
        
        return rec(n,n)

class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        '''
        since the recusive call is tree, we can do a level order traversal 
        at each root, we just calculate all the strobrorammtic numbers
        we keep all strobrogrammtic numbers of N-2 in the queue and then append all reversible pairs to all the numbers in the queue to get strob. number of N digits
        we also know we can't append the '00' pair at the beginning of the number
        so here we use one varibale currStringLenth which denots the length of the strob number at this level
        when currStringLength == n, the added number will be the first in postiion
        
        algo:
        Initialize:

        reversiblePairs as a data structure that stores all pairs of reversible digits.
        q as a queue for doing level order traversal.
        In javascript and python, we will use an array as a queue for easier implementation.
        currStringsLength as 0 if n is even or 1 if n is odd, to denote the number of digits in each string, in the current level strobogrammatic numbers.
        If n is even, we push [""] in the queue, and initialize currStringsLength with 0, because n will decrease till 0. Thus for this case, our starting case will be 0-digit strobogrammatic numbers.
        Otherwise, if n is odd, we push ["0", "1", "8"] in the queue and initialize currStringsLength with 1, because for odd n the starting case will be 1-digit strobogrammatic numbers.

        We will iterate over one whole level stored in the queue until currStringsLength becomes equal to n.

        In each level, we will append two characters, thus increasing currStringsLength by 2.
        For each number in the current level (present in the queue) we pop it and, append all reversiblePairs at the beginning and the end except when the current reversible pair is '00' and currStringsLength == n (because we can't append '0' at the beginning of a number) and push this new number in the queue again.
        After traversing all levels, the queue will contain all n digit strobogrammatic numbers, thus we push them in a stroboNums array and return it.
        '''
        reversible_pairs = [
            ['0', '0'], ['1', '1'], 
            ['6', '9'], ['8', '8'], ['9', '6']
        ]

        # When n is even (n % 2 == 0), we start with strings of length 0 and
        # when n is odd (n % 2 == 1), we start with strings of length 1.
        curr_strings_length = n % 2
        
        q = ["0", "1", "8"] if curr_strings_length == 1 else [""]
        
        while curr_strings_length < n:
            curr_strings_length += 2
            next_level = []
            
            for number in q:
                for pair in reversible_pairs:
                    if curr_strings_length != n or pair[0] != '0':
                        next_level.append(pair[0] + number + pair[1])
            q = next_level
            
        return q

###########################
# 258. Add Digits
# 08FEB22
###########################
class Solution:
    def addDigits(self, num: int) -> int:
        '''
        just mod 10 and // 10 to pop off last digit
        but we need to make sure we don't terminate the loop before summing them up
        define recursively
        '''
        def rec(n):
            print(n)
            if n // 10 == 0:
                return n
            new_n = 0
            while n:
                new_n += n % 10
                n = n // 10
            return rec(new_n)
            
        return rec(num)
            

class Solution:
    def addDigits(self, num: int) -> int:
        '''
        just mod 10 and // 10 to pop off last digit
        but we need to make sure we don't terminate the loop before summing them up
        define recursively
        then find the pattern
        
        the answer is always in the range [1,9]
        only when the num is 0, the answer is zero
        or when the number is divisble by 9, return 9
        '''
        def rec(n):
            if n // 10 == 0:
                return n
            new_n = 0
            while n:
                new_n += n % 10
                n = n // 10
            return rec(new_n)
            
        '''
        for i in range(1000):
            print(i,rec(i))
        '''
        if num == 0:
            return 0
        if num % 9 == 0:
            return 9
        return num % 9

#iterative version
class Solution:
    def addDigits(self, num: int) -> int:
        digital_root = 0
        while num > 0:
            digital_root += num % 10
            num = num // 10
            
            if num == 0 and digital_root > 9:
                num = digital_root
                digital_root = 0
                
        return digital_root

        '''
        just some notes in the math explanation, the value we are asked  to compute is  called the  digital root
        if we have a  k digit number, we can demopcoose it  to:
        n  = d0 + d1*10^1 + d2*10^2 + ... + dk*10^k
        we can rewrtie base  10 as:
        10^k = 9*(1 k times) + 1
        
        we eventually get to n mod 9 = (d0 + d1 + d2 + ... + dk) mod 9
        
        the cases are:
            n == 0, then return 0
            n % 9 == 0, then return 9
            else, return n % 9
        '''

########################################
# 1274. Number of Ships in a Rectangle
# 08FEB22
########################################
# """
# This is Sea's API interface.
# You should not implement it, or speculate about its implementation
# """
#class Sea(object):
#    def hasShips(self, topRight: 'Point', bottomLeft: 'Point') -> bool:
#
#class Point(object):
#   def __init__(self, x: int, y: int):
#       self.x = x
#       self.y = y

class Solution(object):
    def countShips(self, sea: 'Sea', topRight: 'Point', bottomLeft: 'Point') -> int:
        '''
        i can use divide and conquer 
        i can divide the seach space into 4 recatangles at every call
        but be careful in cases where it is not a perfect square
        i can dvide the squre into 4 parts
        and on each part recurse
        rec(sq) = sum(rec(smaller_sq) for sq in [4 squares])
        but how do we define this
        
        the original size of the rectable will be no greater than 1000**2
        using recursion would result in the largest level to be log4(1000^2) = 10, 
        if there were a ship at every point, we would need a million calles to has ship, but we are limited to only 400
        the problem statement says that there can be at most 10 ships in a singel call
        
        there can be at most 10 ships in the rectangle, i.e, the last level can contain no more than 10 calls to the recursive function
        so the max number of subrectagnles at each can be nore more than 40
        if we log4(1000*1000) ~ 10, then that last level would contain 10*40 = 400 calls, which is our limit
        
        division of rectangles:
            for each rectangle, defined by bottomLeft and upperRight, we cand define bounds for 4 smaller rectangles
            find mid_x and mid_y
            then the reactanlges can be defined by new bottomleft and upperright cooridnates
        base case:
            if i return false for has ships, there are no ships in the rectangle, return 0
            if the corners pass each other, there possible couldn't be a ship there, return 0
            the recursive case (dived into 4 and sum them up)
            
        notes on time complexity:
            Hi @sud01 we can simplify the time complexity analysis a little, it equals O(1) (the time required for a single recursive call) times the number of recursive calls.

In the third picture, we see that there can be at most S sub-rectangles (one per ship, so at most 10) that contain a ship. And for each subrectangle, we will make 4 recursive calls (one with a ship, and 3 without a ship). Thus, we have at most 4 * S recursive calls per layer.

Now the question is, how many layers can there be? Since with each recursive call we divide our search space by 2 in the X and Y direction, the maximum depth of the recursion tree will be log2max(M, N).

Multiplying the width * height of the recursion tree, we get (4 * S) * (log2max(M, N)) = O(S * log2max(M, N)). Hope this helps!
        '''
        #if current the corners pass each other, there possible couldn't be a ship
        if bottomLeft.x > topRight.x or bottomLeft.y > topRight.y:
            return 0
        #if there isn't a ship to be found in this square
        if not sea.hasShips(topRight,bottomLeft):
            return 0
        #otherwise we must have collapsed to a point where there is a ship
        if (bottomLeft.x == topRight.x) and (bottomLeft.y == topRight.y) and (sea.hasShips(topRight,bottomLeft)):
            return 1
        
        num_ships = 0
        mid_x = (bottomLeft.x + topRight.x) // 2
        mid_y = (bottomLeft.y + topRight.y) // 2
        
        #we have 4 sqaures to search bottomleft (BL), upperleft (UL), upperRight (UR), bottomright (BR)
        #not i could have gotten creative here with a for loop
        BL = self.countShips(sea,Point(mid_x,mid_y), bottomLeft)
        UL = self.countShips(sea,Point(mid_x,topRight.y), Point(bottomLeft.x,mid_y+1))
        UR = self.countShips(sea,Point(topRight.x,topRight.y), Point(mid_x+1,mid_y+1))
        BR = self.countShips(sea,Point(topRight.x,mid_y), Point(mid_x+1,bottomLeft.y))
        return BL + UL + UR + BR

#there's also an interative version
# 48 ms, faster than 46.38%
class Solution(object):
    def countShips(self, sea: 'Sea', topRight: 'Point', bottomLeft: 'Point') -> int:
        res = 0
        right, top = topRight.x, topRight.y
        left, bottom = bottomLeft.x, bottomLeft.y
        q = [(right, top, left, bottom)]
        while len(q) > 0:
            right, top, left, bottom = q.pop(0)
            
            if left > right or bottom > top:
                continue
            
            hasShip = sea.hasShips(Point(right, top), Point(left, bottom))
            if hasShip == False:
                continue
            if left == right and top == bottom:
                res += 1
            else:
                midX = (left + right) // 2
                midY = (top + bottom) // 2
                q.append((right, top, midX+1, midY+1))  # top right
                q.append((right, midY, midX+1, bottom))  # bottom right
                q.append((midX, midY, left, bottom))    # bottom left
                q.append((midX, top, left, midY+1))     # top left
        return res

#################################
# 532. K-diff Pairs in an Array
# 09FEB22
#################################
class Solution:
    def findPairs(self, nums: List[int], k: int) -> int:
        '''
        we can define a k diff pair, as:
            given indices i,j and 0 <= i < j < len(nums)
            abs(nums[i] - nums[j]) == k
        
        its the absoulute difference in their values
        we want the total number of unique pairs
        brute force is trivial, what is use a hashmap, and check its complement is in there
        its complement would be k more than the current value
        
        we can create a count of nums in the table
        then for each num in count, check if num + k in table
        if k == 0, we are checking for multiplicty
        '''
        counts = Counter(nums)
        
        pairs = 0
        
        for num in counts:
            #if k > 0
            if k > 0:
                #if in mapp
                if num + k in counts:
                    pairs += 1
            #if k is 0, we are just looking for multiplicty
            else:
                if counts[num] > 1:
                    pairs += 1
        
        return pairs
        
#########################
# 251. Flatten 2D Vector
# 09FEB22
#########################
class Vector2D:

    def __init__(self, vec: List[List[int]]):
        '''
        i can just flatten intially the check with points
        '''
        self.flattened = []
        for l in vec:
            for num in l:
                self.flattened.append(num)
        self.cap = len(self.flattened)
        self.ptr = 0

    def next(self) -> int:
        ans = self.flattened[self.ptr]
        self.ptr += 1
        return ans

    def hasNext(self) -> bool:
        return self.ptr < self.cap


# Your Vector2D object will be instantiated and called as such:
# obj = Vector2D(vec)
# param_1 = obj.next()
# param_2 = obj.hasNext()

class Vector2D:

    def __init__(self, vec: List[List[int]]):
        '''
        frist approach involved flattening out in constructor
        we can maintin two pointers, one pointing to each nested list, and another pointing in elements in the pointed to nested lists
        then just advance
        note, when the outer becomes equal to the length of the 2D vector, it means there are nore more inner vectors 
        so there are no numbers next
        
        we need to define an advanceToNext() helper function that checks if the current inner and outer values point to an int
        if they don't move it forard until they point to an int
        if outer == len(vec) there are no more vectors left
        
        both next and hasnext make call to advanve to next to ensure inner and out point to an int or that outer is at tis stop
        
        it is important to note that calling the hasNext() method will only cause the pointers to move if the don't point to an interger
        once they point to an integer, repeated calls to hasnext() will not move them further
        
        '''
        self.vec = vec
        self.element_ptr = 0
        self.list_ptr = 0
    
    def move_to_next_list(self):
        #if eleemnt ptr and list ptr point to an int, the method does nothing
        #otherwise both are advanced until they point to an int
        #if there are no more ints, the list point == len(vec), and so this termiantes
        while self.list_ptr < len(self.vec) and self.element_ptr == len(self.vec[self.list_ptr]):
            self.list_ptr += 1
            self.element_ptr = 0
        

    def next(self) -> int:
        #ensure positions pointsers are moved so that they point to an int
        self.move_to_next_list()
        #return result
        res = self.vec[self.list_ptr][self.element_ptr]
        self.element_ptr += 1
        return res
        
    def hasNext(self) -> bool:
        #advance to int
        self.move_to_next_list()
        return self.list_ptr < len(self.vec)

# Your Vector2D object will be instantiated and called as such:
# obj = Vector2D(vec)
# param_1 = obj.next()
# param_2 = obj.hasNext()

#############################
# 560. Subarray Sum Equals K
# 10FEB22
#############################
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        '''
        i can get the pref sum array and then just check all i,j in that pref sum array
        '''
        N = len(nums)
        pref_sum = [0]
        for num in nums:
            pref_sum.append(num + pref_sum[-1])
        
        #now just check all i,j indexing back into pref_sum
        ans = 0
        for i in range(N):
            for j in range(i+1,N+1):
                #watch for the +1 and -1 
                #off by one index
                sub_sum = pref_sum[j] - pref_sum[i]
                if sub_sum == k:
                    ans += 1
        return ans
        
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        '''
        we can also save space but nopt pre computing the pref sum array
        '''
        N = len(nums)
        
        #now just check all i,j indexing back into pref_sum
        ans = 0
        for i in range(N):
            sub_sum = 0
            for j in range(i,N):
                sub_sum += nums[j]
                if sub_sum == k:
                    ans += 1
        return ans
        
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        '''
        we can use a hashap storing the occurrences of the rolling sums
        then we just check if the complement of this exists.
        why does this work?
        if we have two indices, call them j and k, sucht hat j < k
        and sum(nums[i:j]) == sum(nums[i:k])
        then the sum(nums[j:k]) must be zero
        
        extending if the cum sum up two two indices, call them i and j, is at a difference of k
        sum(nums[i:j]) - sum(nums[i:k]) = k, then sum(nums[i:j]) = k
        
        if we have subarray sum == k
        then sum_end = sum_start + k
        implying that sumstart = sumend -k
        '''
        N = len(nums)
        sum_mapp = Counter()
        sum_mapp[0] += 1
        count = 0
        sub_sum = 0
        for num in nums:
            #find current rolling summ
            sub_sum += num
            #if its complement is in mapp
            #then it must mean there was another subarray sum == k, with multipliacity
            if sub_sum - k in sum_mapp:
                count += sum_mapp[sub_sum - k]
            sum_mapp[sub_sum] += 1
        
        return count

#################################
# 254. Factor Combinations
# 10FEB22
##################################
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        '''
        we can only include factors in the range [2,n-1]
        i need to use dfs to build up a path
        for a number n, i can check if it is diviible by [2,sqrt(n)]
        for each of this candidates add to the path, and reduce the number by that candidate
        since, i can reduce that number by the candidate, add it to global ans
        '''
        if n == 1:
            return []
        res = []
        
        def dfs(num,start,path):
            if len(path) > 0:
                res.append(path + [num])
            #start off at next candidate
            for cand in range(start,int(num**0.5 + 1)):
                if num % cand == 0:
                    dfs(num // cand,cand,path + [cand])
                    
                    
        dfs(n,2,[])
        return res

#iterative
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        if n == 1:
            return []
        res = []
        
        stack = [(n,2,[])]
        
        while stack:
            num,start,path = stack.pop()
            
            if len(path) > 0:
                res.append(path + [num])
            #start off at next candidate
            for cand in range(start,int(num**0.5 + 1)):
                if num % cand == 0:
                    stack.append((num // cand,cand,path + [cand]))
                    
                    
        return res

##################################
# 567. Permutation in String
# 11FEB22
##################################
#brute force, generate perms of smaller, and check if subtring of larger
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        brute force would be to generate all permutations of the shorter string
        and check if it is a substring of the longer string
        this would be O(N)
        '''
        s1 = list(s1)
        self.ans = False
        
        
        def permute(string,size):
            #finished permuting
            if size == len(string):
                #is substring
                if  "".join(string) in s2:
                    self.ans = True
                    return
                
            for i in range(size,len(string)):
                string[i],string[size] = string[size],string[i]
                permute(string,size+1)
                string[i],string[size] = string[size],string[i]
                
        permute(s1,0)
        return self.ans
            
#sorting
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        we can sort; one string will be a permutation of another string only if both of them containt the same chars the same number of times
        one string x is a permutation of another string y, only if sorted(x) = sorted(y)
        '''
        s1 = sorted(s1)
        for i in range(len(s2) - len(s1)+1):
            sub = s2[i:i+len(s1)]
            if s1 == sorted(sub):
                return True
        return False

#hashmap counting
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        this stems for the facts that one string will be a permutation of another string if boht contain the same chars with frequency
        we can consider of possible substring of 2 of length s1 and check frequency counts match
        '''
        counts_s1 = Counter(s1)
        for i in range(len(s2) - len(s1)+1):
            sub = s2[i:i+len(s1)]
            if counts_s1 == Counter(sub):
                return True
        
        return False

#sliiding window, hashmap, check zero counts after reducing
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        this stems from the fact the we don't neeed to generate a new hashmap for every substring in s2
        we can create the hashmap once for the first window in s2
        then later when we slide in the window, we remove a char and add a char
        
        tricks:
            we can loop over the entire string s2 and check if current char in counter
            then we need to make sure the last character is removed
                we do this by first checking if we have at least len(s1), then decrement
                as soon as all the counts hit zero, we have a valid perm in s2
        '''
        counts_s1 = Counter(s1)
        
        for i in range(len(s2)):
            #remove
            if s2[i] in counts_s1:
                counts_s1[s2[i]] -= 1
            #add back in
            if i >= len(s1) and  s2[i-len(s1)] in counts_s1:
                counts_s1[s2[i-len(s1)]] += 1
            
            if all([counts_s1[i] == 0 for i in counts_s1]):
                return True
        
        return False

#optimzied,check zero counts
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        we can slightly optimize by using an extra vairable to count the number of chars whose freqeucne gets to zero
        during the sliding
        this helps to avoid iterating over the entire hashmap
        '''
        cntr, w, match = Counter(s1), len(s1), 0     

        for i in range(len(s2)):
            if s2[i] in cntr:
                if not cntr[s2[i]]: match -= 1
                cntr[s2[i]] -= 1
                if not cntr[s2[i]]: match += 1

            if i >= w and s2[i-w] in cntr:
                if not cntr[s2[i-w]]: match -= 1
                cntr[s2[i-w]] += 1
                if not cntr[s2[i-w]]: match += 1

            if match == len(cntr):
                return True

        return False

##########################
# 127. Word Ladder
# 12FEB22
###########################
#TLE, the problem is generateing the grapch
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        '''
        shortest transformation sequence screams shortest paths
        we need tp make a graph for each word in the wordList
        the outgoing edge would be if we can get from one word to another with a single change
        once we have the graph, we can just do bfs
        
        this gets TLE, because graph generation takes way too long
        '''
        #edge cases
        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0
        #add beginword to wordList
        wordList += [beginWord]

        #build graph
        graph = defaultdict(set)
        N = len(wordList)
        for i in range(N):
            for j in range(N):
                if i != j :
                    start = wordList[i]
                    neigh = wordList[j]
                    for k in range(len(beginWord)):
                        curr = start[:k]+'*'+start[k+1:]
                        cand = neigh[:k]+'*'+neigh[k+1:]
                        if curr == cand:
                            graph[start].add(neigh)
                            
        #bfs
        q = deque([(beginWord,1)])
        seen = set()
        seen.add(beginWord)
        
        while q:
            curr,length_so_far = q.popleft()
            if curr == endWord:
                return length_so_far
            for neigh in graph[curr]:
                if neigh not in seen:
                    seen.add(neigh)
                    q.append((neigh,length_so_far + 1))
        
        return 0

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        '''
        new graph generation, instead of trying to find all possible one aways from each word
        we can find its generate state
        i.e hot can be *ot,h*t,ho*
        these genearics state will all point to hot
        
        if we don't this, we would have to iterate fore very word, finds its wildcard, and pair it, which would take way too long
        '''
        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0

        # Since all words are of same length.
        L = len(beginWord)

        # Dictionary to hold combination of words that can be formed,
        # from any given word. By changing one letter at a time.
        all_combo_dict = defaultdict(list)
        for word in wordList:
            for i in range(L):
                # Key is the generic word
                # Value is a list of words which have the same intermediate generic word.
                all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)


        # Queue for BFS
        queue = collections.deque([(beginWord, 1)])
        # Visited to make sure we don't repeat processing same word.
        visited = {beginWord: True}
        while queue:
            current_word, level = queue.popleft()
            for i in range(L):
                # Intermediate words for current word
                intermediate_word = current_word[:i] + "*" + current_word[i+1:]

                # Next states are all the words which share the same intermediate state.
                for word in all_combo_dict[intermediate_word]:
                    # If at any point if we find what we are looking for
                    # i.e. the end word - we can return with the answer.
                    if word == endWord:
                        return level + 1
                    # Otherwise, add it to the BFS Queue. Also mark it visited
                    if word not in visited:
                        visited[word] = True
                        queue.append((word, level + 1))
                #don't forget to clear the candidate after exaining the next word
                all_combo_dict[intermediate_word] = []
        return 0

#bi directinol bfs
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        '''
        we can also a a bi-directinal search, where we start from both being word and end word
        the only difference is that we terminate, once we have found a word in either parallel search
        
        to code this out better, we build out a helper function to carry out only one iteration (one level of bfs)
        then we can all this from beign adna dn
        
        the shortes transformation sequence is the sum of lvles othe meeting point node from both ends
        or the number of steps progresgressed in bfs from begin and end
        '''
        self.N = len(beginWord)
        self.graph = defaultdict(list)
        
        #edge cases
        if endWord not in wordList or not endWord or not beginWord or not wordList:
            return 0
        
        for word in wordList:
            for i in range(self.N):
                state = word[:i]+'*'+word[i+1:]
                self.graph[state].append(word)
        
        #set up for bi-driection bfs
        q_begin = deque([beginWord])
        q_end = deque([endWord])
        
        #visited sets, we can't use a set here, because we want to maintain the leveks
        visited_begin = {beginWord : 1}
        visited_end = {endWord : 1}
        ans = None
        
        while q_begin and q_end:
            #progress with bi-directional search but start with smaller one first
            if len(q_begin) < len(q_end):
                ans = self.invokeBFS(q_begin,visited_begin,visited_end)
            else:
                ans = self.invokeBFS(q_end,visited_end, visited_begin)
            if ans:
                return ans
        
        return 0
    
    def invokeBFS(self,q,visited,others_visited):
        q_size = len(q)
        for _ in range(q_size):
            current_word = q.popleft()
            for i in range(self.N):
                #intermediate words
                state = current_word[i:]+'*'+ current_word[i+1:]
                #now check neigbors
                for neigh in self.graph[state]:
                    #if the state has already been visited from the other parallel search, this mean we found our answer
                    if neigh in others_visited:
                        return visited[neigh] + others_visited[neigh]
                    #if its not, mark as visited, but not before increaming depth size
                    if neigh not in visited:
                        visited[neigh] = visited[current_word] + 1
                        q.append(neigh)
        return None

########################
# 78. Subsets
# 13FEB22
########################
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        '''
        routine recusrive path generation
        each recursive call add currenet element then try to incldue in new path
        we need to recusrive that we create to start at one of the elemetns
        then we need to call this for each eleemnt
        '''
        paths = []
        N = len(nums)
        
        #single recursive call for and we need to invoke at each index
        def rec(i,path,size):
            if len(path) == size:
                paths.append(path[:])
                return
            for j in range(i,N):
                path.append(nums[j])
                rec(j+1,path,size)
                path.pop()
                
        
        for size in range(N+1):
            rec(0,[],size)
        return paths

# using global variable and not calling on each one
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        '''
        we can also contain the recursive call to invoke on each element anyway
        rather than recusring for each index
        '''
        self.paths = []
        N = len(nums)
        def rec(start,path):
            #always add in the new path
            print(path)
            self.paths += [path]
            for i in range(start,N):
                rec(i+1,path + [nums[i]])
        
        rec(0,[])
        return self.paths

#iterative, cascading
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        '''
        we can seen an initital ouput with and empty list
        then add each num to it
        '''
        subsets = [[]]
        
        for num in nums:
            #to store new subsets
            new_subsets = []
            for sub in subsets:
                new_subsets.append(sub +[num])
            
            for new_subset in new_subsets:
                subsets.append(new_subset)
        
        return subsets

#we can use the bit masking trick from Knuth
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        '''
        we can use the bit track, where we take an element at an index or we dont
        '''
        N = len(nums)
        subsets = []
        for num in range(2**N):
            bit_mask = []
            for _ in range(N):
                #mod 2 of numbers gives us LSB
                bit = num % 2
                bit_mask.append(bit)
                #reduce number
                num = num >> 1
            #reverse mask 
            bit_mask = bit_mask[::-1]
            #generate new subse
            subset = []
            for i in range(N):
                if bit_mask[i]:
                    subset.append(nums[i])
            
            subsets.append(subset)
            
        return subsets
        
################################
# 104. Maximum Depth of Binary Tree
# 14FEB22
################################
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        '''
        bottom up return directly
        recurrnce is:
            rec(node) = max(rec(node.left),rec(node.right)) + 1
            bottom case is the empty node
        '''
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            return max(left,right) + 1
        
        return dfs(root)
        
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        self.ans = 0
        
        def dfs(node,depth):
            if not node:
                return
            dfs(node.left,depth+1)
            self.ans = max(self.ans,depth)
            dfs(node.right,depth+1)
        
        dfs(root,1)
        return self.ans

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        ans = 0
        
        
        stack = [(root,1)]
        
        while stack:
            node,depth = stack.pop()
            if not node:
                continue
            if node.left:
                stack.append((node.left,depth+1))
            
            ans = max(ans,depth)
            
            if node.right:
                stack.append((node.right,depth+1))
        
        return ans

################################
# 506. Relative Banks
# 14FEB22
#################################
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        '''
        top three scores should get the 'Gold Medal', 'Silver Medal', 'Bronze Medal'
        the rest get their positions
        hashmap score maps to its value
        '''
        sorted_score = sorted(score, reverse = True)
        mapp = {}
        for i,s in enumerate(sorted_score):
            if i == 0:
                mapp[s] = 'Gold Medal'
            elif i == 1:
                mapp[s] = 'Silver Medal'
            elif i == 2:
                mapp[s] = 'Bronze Medal'
            else:
                mapp[s] = str(i+1)
            
        ans = []
        for s in score:
            ans.append(mapp[s])
        
        return ans

#just a another cool way
class Solution:
    def findRelativeRanks(self, nums):
        rank = {n:i>2 and str(i+1) or ["Gold","Silver","Bronze"][i]+' Medal' for i,n in enumerate(sorted(nums,reverse=True))}
        return [rank[num] for num in nums]

#########################
# 507. Perfect Number
# 14FEB22
##########################
class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        '''
        a perfect number is a positive integers == to sum of divisors, excluding the numberitself
        
        find all divisors of num
        sum them and check
        '''
        sum_divisors = 0
        i = 1
        
        #we will examine only up sqrt(num)
        while i*i <= num:
            if num % i == 0:
                sum_divisors += i
                #check its complement, must coonsider num // i, i i divides num
                #but only if i != sqrt(num)
                if i*i != num:
                    sum_divisors += num // i
            
            i += 1
        #we sum up all such factors and check if given num is preft
        #we need to subtract num frmo the sum, why? becasue we consindered 1 as factor, which also forced 
        #us to consider num as a factor, we double counted here
        #we need to remove it from the sum and check sum - num is num
        return sum_divisors - num == num

################################
# 136. Single Number
# 15FEB22
#################################
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        '''
        hash map is trivial
        if i have numbers a,b,c,d
        and a,b,c are in the array twice
        a + a + b + b + c + c + d
        how can we find d?
        d = (a + a + b + b + c + c + d) - (a + a + b + b + c + c)
        d = sum(array) - 2*(a + b + c)
        turns out to be just 2*sum(nums) - sum(nums)
        '''
        return 2*sum(set(nums)) - sum(nums)

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        '''
        turns out we can use XOR
        recall XOR is true only if 1 element is true
        but more importnatly it has the following two properties
        num XOR 0 = num
        num XOR num = 0
        
        if we have a XOR b XOR a
        is the same as
        a XOR a XOR b
        0 XOR B
        b
        '''
        ans = 0
        for num in nums:
            ans = ans ^ num
        
        return ans
        
#############################################################
# 1100. Find K-Length Substrings With No Repeated Characters
# 15FEB22
#############################################################
class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        '''
        brute force would be to examine each substring of length k
        '''
        #edge case, if there the lenght of the substring is more than the unique number of chars
        if k > 26:
            return 0
        ans = 0
        N = len(s)
        
        for i in range(N-k+1):
            counts = Counter(s[i:i+k])
            if all([v == 1 for _,v in counts.items()]):
                ans += 1
        
        return ans

class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        '''
        brute force would be to examine each substring of length k
        '''
        #edge case, if there the lenght of the substring is more than the unique number of chars
        if k > 26:
            return 0
        ans = 0
        N = len(s)
        
        for i in range(N-k+1):
            counts = [0]*26
            #found = False
            '''
            recall the break fall through in python for loops
            '''
            for j in range(i,i+k):
                curr_char = ord(s[j]) - ord('a')
                counts[curr_char] += 1
                if counts[curr_char] > 1:
                    #found = True   
                    break
            else:
                ans += 1
        
        return ans
            
#close one....
class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        '''
        examining all k lenght substrings would result len(s) times k
        which is too much, we want to do this in linear time
        we can store a char counts into an inital counter
        then just add in the next char and delete the last char
        we maintain a size variable, and only increment when adding new char
        '''
        size = 0
        counts = Counter()
        N = len(s)
        ans = 0
        
        #edge cases
        if N < k:
            return 0
        
        
        #adding initial counts
        for i in range(k):
            if s[i] in counts:
                counts[s[i]] += 1
            else:
                counts[s[i]] = 1
                size += 1
                
        for i in range(k+1,N):
            #checking size contstaint
            if size == k:
                ans += 1
            #remove leftmost char
            if s[i-k] in counts:
                if counts[s[i-k]] == 0:
                    del counts[s[i-k]]
                else:
                    counts[s[i-k]] -= 1
                size -= 1
            #add in new char
            if s[i] in counts:
                counts[s[i]] += 1
            else:
                counts[s[i]] = 1
                size += 1
        
        if size == k:
            ans += 1
        
        return ans

#sliding window
class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        '''
        we can use a slidigin and we keep advancing the right pointer as long as we are <= k
        and the next character we are addining isn't repeated
        if it is repeated, we need to shrink the left pointer
        whenver right - left + 1 == k, we found a valid substring
        '''
        if k > 26:
            return 0
        ans = 0
        N = len(s)
        left = right = 0
        #note we also could have used a array count here
        counts = Counter()
        
        while right < N:
            #load up the right most char
            counts[s[right]] += 1
            
            #if the current char appears more than one in the hash, keep shrinking left unitl
            #we break out of this invariant
            while counts[s[right]] > 1:
                counts[s[left]] -= 1
                left += 1
            
            #check size requirment
            if right - left + 1 == k:
                ans += 1
                
                #finally reduce the size by one more, becae it can't be bigger than k
                counts[s[left]] -= 1
                left += 1
            
            right += 1
        
        return ans

class Solution:
    def numKLenSubstrNoRepeats(self, s: str, k: int) -> int:
        count = 0
        seen = set()
        i = j = 0
        N = len(s)
        
        while j < N:
            if len(seen) == k:
                count += 1
                seen.remove(s[i])
                i += 1
            if s[j] in seen:
                seen.remove(s[i])
                i += 1
            else:
                seen.add(s[j])
                j += 1
                
        #final check after going throught hem all
        if len(seen) == k:
            count += 1
        
        return count