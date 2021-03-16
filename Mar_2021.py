##################
#Distribute Candies
###################
class Solution(object):
    def distributeCandies(self, candyType):
        """
        :type candyType: List[int]
        :rtype: int
        """
        '''
        count the types of candies,
        then traverse the counts and just take the max when you can
        hashset can only be so large, we can keep adding candies until we are in the acorrect size
        then just return the size of the hashset
        '''
        N = len(candyType)
        numAllowed = N // 2
        
        num_types = set()
        
        for c in candyType:
            if len(num_types) >= numAllowed:
                break
            num_types.add(c)
        return len(num_types)
        

        #could also just
        return min(len(candType)//2, len(set(candyType)))

#####################
#Single Row Keyboard
#####################
class Solution(object):
    def calculateTime(self, keyboard, word):
        """
        :type keyboard: str
        :type word: str
        :rtype: int
        """
        '''
        mapp the keyboard then just add it up
        '''
        N = len(word)
        mapp = {k:v for k,v in zip(keyboard,range(26))}
        steps = mapp[word[0]]
        
        for i in range(1,N):
            steps += abs(mapp[word[i]] - mapp[word[i-1]])
        return steps


#######################
# Set Mismatch
####################
#close one!
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        since they are all unique, the ith position should be the ith number
        it it is not, then that number would either have been duplicated or not inlcuded
    
        '''
        #wellp the may not be sorted
        nums.sort()
        #first find the doubled number
        counts = Counter()
        doubled = None
        for num in nums:
            counts[num] += 1
            if counts[num] == 2:
                doubled = num
            
        #then find the delted number
        deleted = None
        for i in range(len(nums)):
            if nums[i] != i + 1:
                deleted = i
        
        return [doubled,deleted+1]

#LETS GOOOOOO
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        what if i add to a hashset 1 by 1
        if the size of the hashset does not increase, i know that was the doubled number
        i can do this i
        using hash avoid the need to sort
        '''
        #first pass, find the doubled number
        first_set = set()
        size = len(first_set)
        
        doubled = None
        for i in range(len(nums)):
            first_set.add(nums[i])
            if len(first_set) == size:
                doubled = nums[i]
            size = len(first_set)
        
        #second pass, i've already hashed everyyint in the first_set
        #go backwards for n
        N = len(nums)
        
        deleted = None
        while N >= 1:
            if N in first_set:
                first_set.remove(N)
            else:
                deleted = N
                break
            N -= 1
        
        return [doubled,deleted]

#brute force
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        again lets go over some of the approaches
        bruteforce
        examine all elements 1 to N and count them up
        '''
        duplicated, missing = -1,-1
        for i in range(1,len(nums)):
            count = 0
            for j in range(0,len(nums)):
                if nums[j] == i:
                    count += 1
            if count == 2:
                duplicated = i
            elif count == 0:
                missing = i
            #to termiante the search stop if found
            if duplicated > 0 and missing > 0:
                break
        return [duplicated,missing]

#sorting
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        using sorting
        we can check missing by looking if consecutive elements are the same
        '''
        duplicated,missing = -1,-1
        nums.sort()
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                duplicated = nums[i]
            elif nums[i] > nums[i-1] + 1: #no longer consective increase by 1
                missing = nums[i-1] + 1
        if nums[len(nums)-1] != len(nums):
            missing = len(nums) #wtf is going on here
        return [duplicated,missing]

class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        using hash map
        and then counting
        '''
        duplicated,missing = -1,-1
        counts = Counter(nums)
        for i in range(1,len(nums)+1):
            if i in counts:
                if counts[i] == 2:
                    duplicated = i
            else:
                missing = i
        
        return [duplicated,missing]

class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        using count array instead of hashmap
        
        '''
        count_array = [0]*len(nums)
        duplicated,missing = -1,-1
        
        for i in range(len(nums)):
            count_array[nums[i]-1] += 1
        
        for i in range(len(count_array)):
            if count_array[i] == 0:
                missing = i + 1
            elif count_array[i] == 2:
                duplicated = i + 1
        return [duplicated,missing]

#constant space
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        the big one! using constant space
        we can save space regarding eh duplicacy of an element or absence of by using he nums array it self
        we know that all the elelmetns in the nums array are postive, and lie in the range(1,N)
        so we can pick an ellment i from nums, then we can invert the elment at the index abs(i)
        by doing so, if one of the elemnts j occurs twice, when this number is encountered a second time, the element nums[abs(i)] wel be found to be negative!!!!
        doing inversions we can check if a number if already negative to find the duplicate number
        After the inversions have been done, if all the elements in numsnums are present correctly, the resultant numsnums array will have all the elements as negative now. But, if one of the numbers, jj is missing, the element at the j^{th}j 
th
  index will be positive. This can be used to determine the missing number.
        '''
        duplicated,missing = -1,-1
        for num in nums:
            if nums[abs(num)-1] < 0:
                duplicated = abs(num)
            else:
                nums[abs(num)-1] *= -1
        
        for i in range(len(nums)):
            if nums[i] > 0:
                missing = i + 1
        return [duplicated,missing]

#more mathy solutions
def findErrorNums(self, nums):
	'''
	[1,2,2,4]
	[1,2,3,4]

	to find duplicated, just sum(nums) - sum(set(nums))
	[1,2,2,4] - [1,2,4], that just leaves 2

	to find missing
	sum(1 to N + 1) - sum(set(nums))

	[1,2,3,4] - [1,2,4] = 3
	'''
    return [sum(nums) - sum(set(nums)), sum(range(1, len(nums)+1)) - sum(set(nums))]

class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        [1,2,2,4] #act array
        [1,2,3,4] #exp array
        
        sum(act) - sum(exp) = rep - mis
        prod(act) / prod(exp) = rep / mis
        
        rep = (prod(act) / pro(exp))*mis
        mis = sum(act) - sum(exp) - rep
        '''
        N = len(nums)
        sumAct = sum(nums)
        sumExp =  (N*(N+1)) / 2
        prodAct = reduce(mul,nums)
        prodExp = reduce(mul,range(1,N+1))
        
        mis = round((sumAct - sumExp) / ((prodAct / prodExp) - 1))
        rep = (sumAct - sumExp)+mis
        
        print [rep,mis]

####################
#Missing Number
#####################
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        i could hash the nums and then check each number between 0 to N in hash
        if not that the missing number
        '''
        N = len(nums)
        
        hashed = set(nums)
        for i in range(N+1):
            if i not in hashed:
                return i

#Gauss Strikes again
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        just get the expected sum
        and deduct from that the current sum
        why? because 1 is missing from the the sum 0 to N
        '''
        N = len(nums)
        expSum = N*(N+1) // 2
        currSum = sum(nums)
        
        return expSum - currSum

####################################
#Intersection of Two Linked Lists
####################################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        just pass the contents one one list into a hash
        then check if any fothe nodes in b are in the hash, if so return the node
        '''
        seen = set()
        while headA:
            seen.add(headA)
            headA = headA.next
        
        while headB:
            if headB in seen:
                return headB
            headB = headB.next

#brute force
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        lets just code out the brute force solution
        for every node in A travers B and check
        '''
        while headA is not None:
            dummy_B = headB
            while dummy_B is not None:
                if headA == dummy_B:
                    return headA
                dummy_B = dummy_B.next
            headA = headA.next
        
        return None

#O(1) space
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        if the lengths of headA and headB were the same, then we could just use tow pointers
        and check if the equaled each other, if they didn't adavance both
        varaiant 1:
            get the length of the lists
            set the start pointer for the longer list
            step the pointer though the list togets
        '''
        #get sizeA
        sizeA = 0
        tempA = headA
        while tempA:
            sizeA += 1
            tempA = tempA.next
        #get sizeB
        sizeB = 0 
        tempB = headB
        while tempB:
            sizeB += 1
            tempB = tempB.next
        
        #now move the longer pointer until it gets to shorter
        longer,startLong,startShort = sizeA,headA,headB
        if sizeA > sizeB:
            longer, startLong,startShort = sizeA, headA,headB
        else:
            longer, startLong,startShort = sizeB, headB,headA
        
        #advance longer by abs(dist)
        toAdvance = abs(sizeA-sizeB)
        while toAdvance > 0:
            startLong = startLong.next
            toAdvance -= 1
        
        #now compare
        while startLong:
            if startLong == startShort:
                return startLong
            startLong = startLong.next
            startShort = startShort.next
        return None


#0(1) not getting the lengths
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        now using two pointers
        instead of getting the lengths of both lists
        start pointer both off at their respective heads
        the first node that gets to the end first gets sent back to the opposite list it was on
        same thing with the second node
        eventuall they start off at the same and it becomes the same thing as if they had the same length
        why this works? well it ensure both pointers travel the same number of nodes
        '''
        pA = headA
        pB = headB
        
        while pA != pB:
            if pA is None:
                pA = headB
            else:
                pA = pA.next
            if pB is None:
                pB = headA
            else:
                pB = pB.next
        
        return pB

###################################
# Average of Levels in Binary Tree
####################################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        '''
        level order BFS
        once i finish a level, dump the average
        '''
		averages = []

		q = deque([root])

		while q:
		    #get the current size of the level
		    size = len(q)
		    sumLevel = 0
		    for i in range(size):
		        node = q.popleft()
		        sumLevel += node.val
		        if node.left:
		            q.append(node.left)
		        if node.right:
		            q.append(node.right)
		    levelAvg = float(sumLevel) / float(size)
		    averages.append(levelAvg)

		return averages
                
#recursive approach is ugly
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        '''
        now how to do this recursively?
        i can use hashmap to store the node vals at each level, which can be done recusrisvely
        then traverse the hash map
        '''
        mapp = defaultdict(list)
        
        def dfs(node,level):
            if not node:
                return
            mapp[level].append(node.val)
            dfs(node.left,level+1)
            dfs(node.right,level+1)
            
        #invoke
        dfs(root,0)
        
        #i dont' wannt sort
        #it just makes sense passing the mapp once to get the max level
        max_level = 0
        for k in mapp.keys():
            max_level = max(max_level,k)
        
        averages = [0]*(max_level+1)
        for k,v in mapp.items():
            averages[k] = float(sum(v))/float(len(v))
        
        return averages

##########################
# Short Encoding of Words
##########################
#oh boy this one was tough, neeeded to use a Trie
class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        we can reduce the length of the encoding if the ith word is a subsequence of any of the i-n words
        the indices of the refernce string mark the start and ends of the words (sorta)
        start of by making a hash
        word : [sebequence of words with the word]
        '''
        mapp = defaultdict(set)
        N = len(words)
        #helper function for subsequence
        def is_sub(word,sub):
            M = len(word)
            N = len(sub)
            i,j = 0,0
            while i < M and j < N:
                if word[i] == sub[j]:
                    j += 1
                i += 1
            return j == N
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    if is_sub(words[i],words[j]):
                        mapp[words[i]].add(words[j])
        #if there are no intersections, then the ecnoding is just each word sepearte by#
        if len(mapp) == 0:
            return 2*len(words)
        
        #now there exists some intersections with the words
        #i can do bfs and increment by the lenghts
        distance = 0
        for w in words:
            if w in mapp:
                
        print mapp

#O(\sum(w_{i}))
#where w_i is the length of word i
class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        im so fucking dumd, just check if a word is suffix of one of the words in it already
        if its a suffix, then that word does not need to be included in the length of the encoding
        '''
        
        setWords = set(words)
        for w in words:
            for i in range(1,len(w)):
                if w[i:] in setWords:
                    setWords.remove(w[i:])
        size = 0
        for w in setWords:
            size += len(w) + 1
        
        return size

#dfs using Trie
class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/short-encoding-of-words/discuss/1096077/Python-Trie
        first build a Trie structure with letters in reverse
        then dfs on the tree getting the total lengths of all root to leaf paths + 1
        this is summed up
        '''
        trie = {}
        for word in words:
            curr_node = trie
            for char in reversed(word):
                if char not in curr_node:
                    curr_node[char] = {}
                curr_node = curr_node[char]
        
        def dfs(trie,level):
            #keep dfsing down the path getting the sum
            size = 0
            if not trie:
                return level
            for child in trie.values():
                size += dfs(child, level+1)
            return size
        return dfs(trie,1)

#
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode) #dict of TrieNodes, which is just nested dict

class Trie():
    #idk why we made a class for this, series of nested dicts would have been fine
    #fuck itttt
    def __init__(self):
        self.root = TrieNode()
        self.ends = []
    
    def insert(self,word):
        root = self.root
        for char in word:
            root = root.children[char]
        self.ends.append((root,len(word)+1))

class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/short-encoding-of-words/discuss/1095858/Python-Trie-solution-explained
        using a trie, but put them in reverse
        why? if we reverse words and put them in a trie, than for the words ["time", "me", "bell"]
        we get  emit,em,lleb, so if one words was suffix of another, it is now a prefx
        tries are normally done to store prefixes!
        
        in addition to this classical trie structure, we want to keep a list of ends so we can have quick access tot them (in normal tries, we'd have a special char indicating the end of the trie)
        
        '''
        trie = Trie()
        size = 0
        for word in set(words):
            trie.insert(word[::-1])
        
        for node,depth in trie.ends:
            if len(node.children) == 0:
                size += depth
        
        return size

#another Trie
class TrieNode():
    def __init__(self):
        self.children = {}

class Trie():
    def __init__(self):
        self.root = TrieNode()
        self.ends = [] #tuple(TrieNode(),length)
    def insert(self,word):
        root = self.root
        for char in word:
            if char not in root.children: #not at this level
                #make it
                root.children[char] = TrieNode()
            root = root.children[char]
        #we've now move through the root
        self.ends.append((root,len(word)+1))

class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        words = list(set(words))
        trie = Trie()
        for w in words:
            trie.insert(w[::-1])
        size = 0
        for node,depth in trie.ends:
            if len(node.children) == 0:
                size += depth
        return size

####################
#Design Hash Map
#####################
#approach 1
'''
inution:
    one of the most common implentatinos is to has the modulo opeartare where the base is the size of the hash
    common to use primes,2069
    in the case of a collision, we use a bucket to hold all values that have the same hash
    bucket can be modeld using an array or linked list

algo:
    *for a given key value, we first apply the hash function to geneate the hash key
    *which corresponds to the address in our main stroage, with this hash key we would find the bucket where the value should be stored
    *now that we have found the bucket, we check if the (key,value) exists
'''

#make bucket object
class Bucket:
    def __init__(self):
        self.bucket = []
    
    def get(self,key):
        for k,v in self.bucket:
            if k == key:
                return v
        return -1
    
    def update(self,key,value):
        #initaliit not found
        self.isFound = False
        for i,kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key,value)
                self.isFound = True
                break
        if not self.isFound:
            self.bucket.append((key,value))
    def remove(self,key):
        for i,kv in enumerate(self.bucket):
            if key == kv[0]:
                del self.bucket[i]
        

class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.space = 2069
        self.hash_table = [Bucket() for i in range(self.space)]

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: None
        """
        hash_key = key % self.space
        self.hash_table[hash_key].update(key,value)
       
    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        hash_key = key % self.space
        return self.hash_table[hash_key].get(key)

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: None
        """
        hash_key = key % self.space
        self.hash_table[hash_key].remove(key)

# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

#an addtional way using linked lists for chaning
#we can also just used a linked list of chaining
class ListNode:
    def __init__(self,key,val):
        self.item = (key,val)
        self.next = None


class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        #this can vary, but typically use a prime
        #https://cs.stackexchange.com/questions/11029/why-is-it-best-to-use-a-prime-number-as-a-mod-in-a-hashing-function
        #explaining the need for using a prime
        #if the distribution of items to be hashed is uniformly distributed, the size of the hash does not matter so much
        self.size = 2069
        self.hash_table = [None]*self.size
        

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: None
        """
        idx = key % self.size
        if self.hash_table[idx] == None:
            self.hash_table[idx] = ListNode(key,value)
        else:
            curr = self.hash_table[idx]
            while curr:
                #traverse this bucket
                if curr.item[0] == key:
                    curr.item = (key,value)
                    return
                if curr.next == None:
                    return
                curr = curr.next
            #should now be None if we have gone through the whole thing
            curr.next = ListNode(key,value) 

    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        idx = key % self.size
        curr = self.hash_table[idx]
        while curr:
            if curr.item[0] == key:
                return curr.item[1]
            else:
                curr = curr.next
        return -1 #because it now Nonea fter not finding the key in the linekd list
        

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: None
        """
        idx = key % self.size
        curr = prev = self.hash_table[idx]
        if not curr:
            return
        if curr.item[0] == key:
            self.hash_table[idx] = curr.next
        else:
            while curr:
                if curr.item[0] == key:
                    prev.next = curr.next
                    return
                else:
                    curr = curr.next
                    prev = prev.next

# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

####################################
# Remove Palindromic Subsequences
####################################
class Solution(object):
    def removePalindromeSub(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        the key observation is that any seuwnce of one unique char is indeed a plaind drom
        because we only have two unique lettters we can always solve the porblme with at most two steps
        1. remove all as as s single 
        2. remove all b's a s single
        
        remmember this is a subsequence
        if there isn't a palindrome at first
        then we can always just remove the a's and then the b's
        '''
        if not s:
            return 0
        if s == s[::-1]:
            return 1
        return 2
        
class Solution(object):
    def removePalindromeSub(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        using two pointers to check palindrom
        '''
        def is_palindrome(s):
            lo, hi = 0,len(s) -1
            while lo < hi:
                if s[lo] != s[hi]:
                    return False
                lo += 1
                hi -= 1
            return True
        if not s:
            return 0
        if is_palindrome(s):
            return 1
        else:
            return 2
        
############################
# Strobogrammatic Number
############################
class Solution(object):
    def isStrobogrammatic(self, num):
        """
        :type num: str
        :rtype: bool
        """
        '''
        there are only 4 numbers ther when flipped give another number
        0,6,8,9
        if the set of the numbers contains any number out of this, immedialt return false
        '''
        notSB = ['2','3','4','5','7']
        mapp = {'1':'1','0':'0','6':'9','9':'6','8':'8'}
        num = list(num)
        set_num = set(num)
        for n in notSB:
            if n in set_num:
                return False
        #otherwise try and build it
        flipped = []
        for n in reversed(num):
            flipped.append(mapp[n])
        return "".join(flipped) == "".join(num)

class Solution(object):
    def isStrobogrammatic(self, num):
        """
        :type num: str
        :rtype: bool
        """
        '''
        we can treat this like a palindrom problem and use two pointers
        be sure to check the middle as well is stobragmmtic since it gets flipped
        '''
        mapp = {'1':'1','0':'0','6':'9','9':'6','8':'8'}
        
        left,right = 0,len(num)-1
        
        while left <= right:
            if num[left] not in mapp or num[right] not in mapp or mapp[num[left]] != num[right]:
                return False
            left += 1
            right -= 1
        return True

#######################
#Add One Row to Tree
######################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        #edge cae
        if d == 1:
            newNode = TreeNode(v)
            newNode.left = root
            return newNode
        
        def insert(val,node,depth,n):
            #val is the value to be inserted
            #node if the current node
            #depth is the current depth of the current node
            #n is the height at which to insert thew new nodes
            if not node:
                return
            if depth == n-1:
                temp = node.left
                node.left = TreeNode(val)
                node.left.left = temp
                temp = node.right
                node.right = TreeNode(val)
                node.right.right = temp
            else:
                insert(val,node.left,depth+1,n)
                insert(val,node.right,depth+1,n)
        insert(v,root,1,d)
        return root

#iterative stack
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Node():
    def __init__(self,node,depth):
        self.node = node
        self.depth = depth
        
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        '''
        iterative solution with stack
        but we need a new strcture here, Node, to keep track of the depth of the current node along with its value
        algo:
            push Node on to stack
            for every popped node, check if  its depth == the one prior at which the new node needs to be inserted
            if yes, insert the new nodes appropriately
            if not, push both the left and right  child Node(val+dpeth)
        '''
        #edge case
        if  d == 1:
            temp = TreeNode(v)
            temp.left = root
            return temp
        stack  = [Node(root,1)]
        while stack:
            curr_node = stack.pop()
            if not curr_node.node:
                continue
            if curr_node.depth == d-1:
                temp = curr_node.node.left
                curr_node.node.left = TreeNode(v)
                curr_node.node.left.left = temp
                temp = curr_node.node.right
                curr_node.node.right = TreeNode(v)
                curr_node.node.right.right = temp
            else:
                stack.append(Node(curr_node.node.left, curr_node.depth+1))
                stack.append(Node(curr_node.node.right,curr_node.depth+1))
        return root

#BFS approach
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        #using bfs
        if d == 1:
            temp = TreeNode(v)
            temp.left = root
            return temp
        q = deque([(root,1)])
        while q:
            size = len(q)
            for i in range(size):
                #if we get the depth one befroe we need to insert
                curr,level = q.popleft()
                if level == d-1:
                    old_left,old_right = curr.left,curr.right
                    curr.left,curr.right = TreeNode(v),TreeNode(v)
                    curr.left.left, curr.right.right = old_left,old_right
                else:
                    if curr.left:
                        q.append((curr.left,level+1))
                    if curr.right:
                        q.append((curr.right,level+1))
            if level == d-1:
                return root
        return None

#compact dfs
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        def dfs(node,depth,d):
            if not node:
                return None
            elif d == 1:
                temp = TreeNode(v,left = node)
                return temp
            elif depth == d - 1:
                node.left = TreeNode(val=v,left=node.left)
                node.right = TreeNode(val=v,right = node.right)
                return node
            dfs(node.left,depth+1,d)
            dfs(node.right,depth+1,d)
            return node
        return dfs(root,1,d)

                    
#####################
#Integer to Roman
#####################
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        '''
        i can't believe this shit worked LOL
        just some quikc notes
        the set of all roman ints is constant [1,3999] inclusive
        for the mapp lets  inlcude the other numbers ending in 4 and 9
        we need to check if the largest numeral can go into the nnume
        i would need to start with the greatest roman numberal
        greedily use the largest roman numeral
        '''
        numbers = [1,4,5,9,10,40,50,90,100,400,500,900,1000]
        letters = ['I','IV','V','IX','X','XL','L','XC','C','CD', 'D','CM','M']
        
        output = ""
        start = len(numbers)-1
        while num > 0:
            while num // numbers[start] >= 1:
                output += letters[start]
                num -= numbers[start]
            start -= 1
        return output

class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        '''
        just another way of doing this using a for loop in order
        order the roman chars and their int values in decreasing order
        then use the divmod function to see how much times it goes and how much is left
        '''
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        letters  =  ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
        
        output = ""
        for i in range(len(values)):
            if num <=0:
                break
            while values[i] <= num:
                num  -= values[i]
                output += letters[i]
            '''
         output = ""
        	for i in range(len(values)):
            if num <=0:
                break
            #could also just use divmod
            count,num = divmod(num,values[i])
            output += letters[i]*count
        
        return output
        '''
        
        return output

############################
# Coin Change
############################
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        if we are being mathy about this then the optimization problem becomes:
        min_x \sum_{i=0}^{n-1} x_{i} subject to the constraint \sum_{i=0}^{n-1} x_{i} \times c_{i} = S
        where S is the amount, x_{i} is the count of coing with denomination c_{i}
        a trivial solution would be to enumerate all subsets of coin frequenceices [x_{0}..x{n-1}]
        
        we can recusrively use back tracking for this problem
        to the count array [x_{0}...x{n-1}] where each elemnt can be in the range [0,S/c_{i}]
        another way of thinking about this would be to brute force the count array suject to the above constraints
        think about the case we had coins [1] and amount [10]
        we'd go all the way through generating 10 coints total
        then idx falls out of the index and we get to end of the for loop case
        minCost has beenn updated so we reurn it
        if at any point we didn't update, we'd return -1
        
        '''
        
        def calcChange(idx,coins,amount):
            #bottom case
            if amount == 0:
                return 0
            #bonndarycheck
            if idx < len(coins) and amount > 0:
                maxVal = amount // coins[idx] #max coins with this denomination
                minCost = 2**31
                for i in range(maxVal+1):
                    if amount >= i*coins[idx]:
                        result = calcChange(idx+1,coins,amount -i*coins[idx])
                        if result != -1:
                            minCost = min(minCost,result+i)
                if minCost == 2**31:
                    return -1
                else:
                    return minCost
            return -1
        
        return calcChange(0,coins,amount)
        
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        we can use top down recusrive for this problem
        we define a function F(S), which returns the number of coins neeed to make channge for amount S using coins[c_{0}...c_{n-1}]
        we can split the problem into sub problems
        lets say we new the problem F(S) and now we are adding a new coin S, the recurrence can be written as
        we can san F(S) = F(S-C) + 1,  less C plus using up that coin C
        what we don't konw if the denomination of the last coin C,
        we compute F(S-c_{i}) for each possible denom c_0...c_1...c_{n-1} and chose the minimum amog them
        #draw out the recursion tree
        the actual recurren ise:
         * F(S) = min_{i=0}^{i=n-1} F(S-c_{i}) + 1, subject to S - c_{i} >= 0
         * base, F(S) = 0, when S = 0, F(S) = -1, when n = 0
         and of course we can cache along the way
         
        '''
        #special case
        if amount < 1:
            return 0
        memo = [0]*amount
        def calcChange(coins,remaining):
            if remaining < 0:
                return -1 #we've used too many coins
            if remaining == 0:
                return 0 #just enough so we can't add anymore
            if memo[remaining-1] != 0:
                return memo[remaining-1]
            mini = 2**31
            for c in coins:
                result = calcChange(coins,remaining - c)
                if result >=0 and result < mini:
                    mini = 1 + result
            memo[remaining-1] = -1 if mini == 2**31 else mini
            return memo[remaining -1]
        return calcChange(coins,amount)


#another top down memo offset with -1
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount < 1:
            return 0
        memo = [0]*(amount+1)
        def calcChange(coins,remaining):
            #the path is not valid
            if remaining < 0:
                return -1
            #nothing to contirbute
            if remaining == 0:
                return 0
            if memo[remaining] != 0:
                return  memo[remaining]
            
            minCount = amount+1
            for c in coins:
                count = calcChange(coins,remaining-c)
                if count == -1:
                    continue
                minCount = min(minCount,count+1)
            
            if minCount == amount+1:
                memo[remaining] = -1
            else:
                memo[remaining] = minCount
            return memo[remaining]
        return calcChange(coins,amount)

#bottom up dp araay, i still don't really get it
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        bottom up dp solution
        for the iterative solution we think in a bottom up manner
        before getting F(i)
        we have to compute all the minimum counts for amounts up to i 
        on each itetation of i of the algorithm F(i) is computd as min_{j=0...n-1} F_{i-c_{j}} + 1
        example:
        coins = [1,2,3]
        amount = 3
        F(3) = min(F(3-c1),F(3-c2),F(3-c1)) + 1
        = min(F(3-1),F(3-2),F(3-3)) + 1
        = min(F(2),F(1),F(0)) + 1
        = min(1,1,0) + 1
        '''
        #1d array holding the min coins needed for that amount
        max_coins = amount + 1
        dp = [2**31]*max_coins
        #zero amount uses no coins
        dp[0] = 0
        for i in range(1,amount+1):
            for j in range(len(coins)):
                #if i can use the jth coin for this ith amount
                if coins[j] <= i:
                    #use it up and take min
                    #at the ith amount or ith - jth coin amoint + plus the for the single jth coint
                    dp[i] = min(dp[i],dp[i-coins[j]]+1) 
        if dp[amount] == 2**31:
            return -1
        else:
            return dp[amount]

#bfs
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        using bfs
        the trick now is how to keep the visited set
        we check if we have not seen this amount yet, if we haven't we flipped it to visite again
        '''
		if not amount:
		    return 0
		visited = [False]*(amount+1)
		visited[0] = True

		q = deque([(0,0)])

		while q:
		    curr_amount, coinsUsed = q.popleft()
		    #take coin
		    coinsUsed += 1
		    for c in coins:
		        #incremtn amount by the count
		        next_amount = curr_amount + c
		        if next_amount == amount:
		            return coinsUsed
		        #if we can still keep going
		        if next_amount < amount:
		            #check we havent seen this amount yet
		            if visited[next_amount] == False:
		                visited[next_amount] = True
		                q.append((next_amount,coinsUsed))
		return -1

##########################################################
# Check If a String Contains All Binary Codes of Size K
##########################################################
#probably not the best way but it works
class Solution(object):
    def hasAllCodes(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: bool
        """
        '''
        brute force would be to find all unqiue binary codes of length k in string s
        then pass the string again to see that all binary codes are in it
        try that way first
        '''
        #make hash of all binary codes of length k
        allK = set()
        for i in range(2**k):
            allK.add(bin(i)[2:].zfill(k))
        #traverse s with size k
        count = 0
        for i in range(len(s)-k+1):
            if s[i:i+k] in allK:
                allK.remove(s[i:i+k])
        return len(allK) == 0 

#not generating all binary codes, and checking on the fly
class Solution(object):
    def hasAllCodes(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: bool
        """
        '''
        instead of creating all binary codes of length k
        just check every substring of length k untilwe get all possible binaruy cods
        we can keep the size of the hash equal to 2**k, since there are onlty 2**k substrings
        this way we can add a varaint of early stopping
        '''
        need = 1 << k
        seen = set()
        for i in range(k,len(s)+1):
            temp = s[i-k:i]
            if temp not in seen:
                seen.add(temp)
                need -= 1
            if need == 0:
                return True
        return False
