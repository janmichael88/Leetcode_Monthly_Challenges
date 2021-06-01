########################
#Prefix and Suffix Search
########################
#wohoo brute force TLE
class WordFilter(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        '''
        well the dumb way would be to just dump the words in a list and check the start and ends of word
        '''
        self.bucket = []
        for w in words:
            self.bucket.append(w)

    def f(self, prefix, suffix):
        """
        :type prefix: str
        :type suffix: str
        :rtype: int
        """
        output = -1 #max this on the fly
        for i in range(len(self.bucket)):
            w = self.bucket[i]
            if w[0:len(prefix)] == prefix and w[len(w)-len(suffix):] == suffix:
                output = max(output,i)
        return output

class WordFilter(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        '''
        well the best way would be to hash the words by suffix and prefix,
        do this for all possible suffixes and prefixes
        then for each suffix and prefix generates, dump into hash pointing to idx
        DO THIS FIRST, then Trie
        '''
        self.dict = {}
        
        for i in range(len(words)):
            N = len(words[i])
            forwards = words[i]
            rev = words[i][::-1]
            for j in range(N+1): # we are slices, +1 to get the end of the window
                for k in range(N+1):
                    self.dict[(forwards[:j],rev[:k])] = i #this will carry the largest i upon update!
    def f(self, prefix, suffix):
        """
        :type prefix: str
        :type suffix: str
        :rtype: int
        """
        suffix = suffix[::-1] #reverse this
        if (prefix,suffix) not in self.dict:
            return -1
        else:
            return self.dict[(prefix,suffix)]


# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(prefix,suffix)

#Trie solution. two Tries
class WordFilter(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        '''
        use two Tries, not really Tries
        for suffixes and prefixes, for each prefix add to word,
        but keep track of largest weight
        https://leetcode.com/problems/prefix-and-suffix-search/discuss/110053/Python-few-ways-to-do-it-with-EXPLANATIONS!-U0001f389
        '''
        #init
        self.prefs = defaultdict(set)
        self.suffs = defaultdict(set)
        self.weights = {}
        for i,word in enumerate(words):
            prefix = ''
            suffix = ''
            #prefx
            for ch in ['']+list(word):
                prefix += ch
                self.prefs[prefix].add(word)
            #suffixes
            for ch in ['']+list(word[::-1]):
                suffix += ch
                self.suffs[suffix[::-1]].add(word)
            self.weights[word] = i
        

    def f(self, prefix, suffix):
        """
        :type prefix: str
        :type suffix: str
        :rtype: int
        """
        weight = -1
        #find candidate words, intersection of sets
        cand_words = self.prefs[prefix] & self.suffs[suffix]
        for w in cand_words:
            weight = max(weight, self.weights[w])
        return weight


#using the hint exactly,TLE, but its another way directly using the hint
class WordFilter(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        '''
        suffix#prefix
        apple#apple
        our query becomes
        e#a
        '''
        self.dict = {}
        for i,word in enumerate(words):
            self.dict[word+'#'+word] = i

    def f(self, prefix, suffix):
        """
        :type prefix: str
        :type suffix: str
        :rtype: int
        """
        ps = suffix+'#'+prefix
        seen_idxs = [-1]
        for k,v in self.dict.items():
            if ps in k:
                seen_idxs.append(v)
        return max(seen_idxs)

#single Trie, think about threadind the words
class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.index = -1
        
class Trie(object):
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self,word,index):
        root = self.root
        root.index = index
        for char in word:
            if char not in root.children:
                root.children[char] = TrieNode()
            root = root.children[char]
            root.index =index #word belongs to this index
            
    def find(self,prefix,suffix):
        ps = suffix+'#'+prefix
        root = self.root
        for char in ps:
            if char not in root.children:
                return -1
            root = root.children[char]
        return root.index

class WordFilter(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        #bild the Trie
        self.T = Trie()
        for idx,word in enumerate(words):
            N = len(word)
            temp = word+'#'+word
            for i in range(N):
                self.T.insert(temp[i:],idx)

        

    def f(self, prefix, suffix):
        """
        :type prefix: str
        :type suffix: str
        :rtype: int
        """
        return self.T.find(prefix,suffix)

#########################################################
#Number of Connected Components in an Undirected Graph
########################################################
#barely passed, maybe there is a better way
class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        '''
        thank god i've done something like this before
        make the adj list, remember its undirectd, then dfs on each node in n
        then add the nodes visited to a set, frozen set,
        then add them up
        
        '''
        adj_list = defaultdict(list)
        for start,end in edges:
            adj_list[start].append(end)
            adj_list[end].append(start)
        
        def dfs(node):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh)
                    
        all_comps = set()
        for i in range(n):
            seen = set()
            dfs(i)
            all_comps.add(frozenset(seen))
        return len(all_comps)

class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        '''
        instead of dfsing for every node,
        dfs on the nodes we haven't visited!
        then every time we have a new invocation, increment a count
        algo:
            1. make adjlist
            2. init visited set, hashmap or array of size n
            3 inint counter
            4. only dfs on nodes we haven't seen yet
        '''
        adj_list = defaultdict(list)
        for start,end in edges:
            adj_list[start].append(end)
            adj_list[end].append(start)
        
        def dfs(node):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh)
        count = 0
        seen = set()
        for i in range(n):
            if i not in seen:
                dfs(i)
                count += 1
        return count

#using usinon find
#make the DSU class, with find, and uiond
class DSU:
    def __init__(self,n):
        #initiall all nodes point to themseleves and have rank 0
        self.parent = [i for i in range(n)]
        self.rank = [0]*n
    #find representative member of n using path compression, RECURSE
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    #union method, for joining a dis joint set(take from adj_list)
    def union(self,x,y):
        #find current parents
        x_parent = self.find(x)
        y_parent = self.find(y)
        #if they are both pointing to the same paren
        if x_parent == y_parent:
            return
        #if rank on left > right, make left the parent
        if self.rank[x_parent] > self.rank[y_parent]:
            self.parent[y_parent] = self.parent[x_parent]
        #if ranke on left < right, make right the parent
        elif self.rank[x_parent] < self.rank[y_parent]:
            self.parent[x_parent] = self.parent[y_parent]
        else:
            #they must be equal, in which case right gets precedence
            #still dont' really get the +1 part, think of rank as the depth
            self.parent[x_parent] = self.parent[y_parent]
            self.rank[y_parent] += 1

class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        '''
        this is a good example of dis joint set union find
        imagine we have a graph with N vertices and 0 edges....
        the number of connected comps will be
        first assume each node is separate
        then add in edges, then adding in an edge reduces componenets by 1
        3 - 1 = 2
        adding in another one
        2 - 1 = 1
        algo:
            init variable count, with the number of vertices in the input
            traverse all of the edges one by one, performing the union find method,
            combine on each edge
            if the endpoints are already in the same set, then keep traversing, if they are not,then decrement count by 1
            after traversing all of the edges, the variable count will contain the number of componenets
        https://www.youtube.com/watch?v=ID00PMy0-vE&t=68s&ab_channel=TusharRoy-CodingMadeSimple 
        good review on DSU
        '''
        dsu = DSU(n)
        for start,end in edges:
            dsu.union(start,end)
        #print dsu.parent
        
        #return len(set(dsu.parent))
        #i could have inited the parents in ranks of the method in the solution Class
        #then create new functinos find and union
        #invoke and then build
        
        parents = set()
        for i in range(n):
            #find the current parent
            parent = dsu.find(i)
            parents.add(parent)
        return len(parents)

####################
#Course Schedule III
####################
#recursion with memo,TLE
class Solution(object):
    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        '''
        brute force, divide we could enumerate all possible permutations of courses
        and check the number of courses taken in each permutations
        apporach 1: recursion with memoization
        
        say for example are looking at two course (a,x) and (b,y) and assume y>x
        case 1:
            if (a+b) <= x, we can take both
        case 2:
            if (a+b) > x, and a > b, we can only take both if i take a before b
            i.e a + b <= y
        case 3:
            if (a+b) > x, and b > a, can only take both if i take b before a
            i.ie a+b <= x
        case 4:
            if (a+b) > y, i can take any the two courses in any order
        we can conclude it it alwasy profitable to take the course with a smllaer end day prior to a course with a larger end day
        this is becaue the course with a smaller duration, if taken, can surely be take only if it is taken prior to a course with a larger day
        SORT on their end days
        make use of a recursive function:
            schedule(courses, i, time) which return the max number of coruse that can be taken starting from the ith course(starting from 0), given that time already consuemd by ther other courses is time
            now for each functino call, we try to include the current course in the ttaken course, but this can be done only if time + duration_{i} < end_day_{i}
            if the course can be taken, icrement the number of courses taken and obtain the number of courses that can be taken by passing the update time and indx.i.e schedule(course,i+1,time+duration_{i})
            stor the number of courses that can be takenby taking the current course in taken vairable
            further, for every current course, we also leave the current course and find the number of courses taht can be taken theorf
            we do not need to update the tiume, but update the coruses index. i.e schedule(course, i+1,time)
            then return the max of taken or not taken
        '''
        courses.sort(key = lambda x: x[1])
        memo = {}
        
        def schedule_time(courses,idx,time):
            #base case
            if idx == len(courses):
                return 0
            if (idx,time) in memo:
                return memo[(idx,time)]
            #initally nothing take
            taken = 0
            if (time+courses[idx][0] <= courses[idx][1]):
                #take the course
                taken = 1 + schedule_time(courses,idx+1,time+courses[idx][0])
            #don't take
            not_taken = schedule_time(courses,idx+1,time)
            memo[(idx,time)] = max(taken,not_taken)
            return memo[(idx,time)]
        
        return schedule_time(courses,0,0)

#TLE again
class Solution(object):
    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        '''
        from the top down recursion w/ memo approach, we saw that we priotrize a smaller end day first over a larger end day, onyl after sorting on their end day
        we consider coursses in ascending order of their end days
        for a course being considered, we try to take it
        BUT, in order to take it the current course should end before its correspdoing end day (time + duration_{i} > end_day_{i})
        if we can take the coruse, we update time with the ith duration
        but if we are not able to take the current course, (i.e time + ith duration > ith end day) we cant ry to take this course by removing some other course from amongst the courses already taken
        the current course in questino can be fit in by removing some other course only if the duration if the jth course (i.e all course taken such that j < i - 1) and only if the jth duration > ith druation
        KEY: if such a course is found, max_i, we remove this course currently being considers amd the current course as taken
        '''
        courses.sort(key = lambda x: x[1])
        time = 0
        taken = 0
        for i in range(len(courses)):
            if (time + courses[i][0] <= courses[i][1]): #if taken course < end
                time += courses[i][0]
                taken += 1
            else: #we can't take it
                max_i = i #set the max as the current class under consdierations
                for j in range(i):
                    #find the longer duration looking back
                    if courses[j][0] > courses[max_i][0]:
                        #set the max
                        max_i = j
                #now check if max found beats our i
                if courses[max_i][0] > courses[i][0]:
                    #we have a savings of the difference
                    time += courses[i][0] - courses[max_i][0]
                #mark couse so not to take again
                courses[max_i][0] = -1
        return taken

class Solution(object):
    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        '''
        the last approach gets TLE, because we have to look back in the array for a class with a longer duration than the current class we are on
        we optimize by searching among only those courses which has been taken 
        how?
        as we pass the courses, we also keep updating, such that the first count number of elements in this array now correspong to only thsoe count number of courses which have been taken until now
        whenever we update the count to indicate that one more course has been taken, we also update the courses[count] to refleft the current count that has just been taken
        whenver we find a coruse for which time + duration[i] > end_day[i], we canf find max_i from only amonst these first count number courses in the courses array, which indicate the corues that been taken until now
        we do this using an extra list
        
        '''
        courses.sort(key = lambda x: x[1])
        valid_courses = []
        time = 0
        for i in range(len(courses)):
            if (time + courses[i][0] <= courses[i][1]): #if taken course < end
                time += courses[i][0]
                valid_courses.append(courses[i])
            else: #we can't take it
                max_i = 0 #set the max as the current class under consdierations
                for j in range(len(valid_courses)):
                    #find the longer duration looking back
                    if valid_courses[j][0] > valid_courses[max_i][0]:
                        #set the max
                        max_i = j
                #now check if max found beats our i
                if valid_courses and valid_courses[max_i][0] > courses[i][0]:
                    #we have a savings of the difference
                    time += courses[i][0] - valid_courses[max_i][0]
                    #mark couse so not to take again
                    valid_courses[max_i]= courses[i]
                else:
                    continue
        return len(valid_courses)

class Solution(object):
    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        '''
        normally i'd expecte to down recurison TLE or iterative dp to pass
        but they are both pretty long 
        if there are N courses N*N for recursion
        N*len(taken) for iterative dp, even after sorting!
        we can save more time by using a heap
        we pass through courses and whenvere the current time + current duration <= current end, we push on to the heap
        only then, when we can't take class, we take the class of the longest duration so far, and reclaim more time
        '''
        courses.sort(key = lambda x: x[1])
        heap = []
        time = 0
        for duration,end in courses:
            if time + duration <= end:
                time += duration 
                heappush(heap,-duration) #max heap, we want to try getting more time back
            else:
                if len(heap) > 0 and -heap[0] > duration:
                    time += duration - -heappop(heap)
                    heappush(heap,-duration)
        return len(heap)

##########################
#Running Sum of 1d Array
##########################
class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        this is just the prefix sum array
        '''
        N = len(nums)
        dp = [0]*N
        dp[0] = nums[0]
        for i in range(1,N):
            dp[i] = dp[i-1] + nums[i]
        return dp

class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        or you could just do this in place
        '''
        for i in range(1,len(nums)):
            nums[i] += nums[i-1]
        return nums

####################
#Non-decreasing Array
####################
#320/333
#there's an edge case going on here, somwhere
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        non decreasing array means increassing array but with plataeus, that's why they call it non-decreasing
        criteria:
            nums[i] <= nums[i-1] for i in the range [0,n-1]
        [1,2,3,4,4,4,5,6,7]
        if there is a spike, we can remove the spike bike making it smaller than the its i+1 element
        but it must be >= the current smallest element
        count increasing and decreasing intervals
        '''
        smallest = min(nums)
        largest = max(nums)
        N = len(nums)
        
        increasing = 0
        decreasing = 0
        for i in range(1,N):
            if nums[i-1] <= nums[i]:
                increasing += 1
            else:
                decreasing += 1
        if decreasing == 0:
            return True
        elif decreasing > 1:
            return False
        else:
            return True

#this one is tricky....
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        official LC solution
        initially i thought just count up the discpancies and if there is more than 1
        we can't change
        the problem is that if we change the elemene  at the index we affect the order just beofer that index
        we need to keep track of i, i-1, and i-2
        the first time we encounter a violation: nums[i-1] > nums[i] we make that hcnage
        (we actually don't need to make a change, but see if making change changes the array)
        then we continue through the rest of the array
        we we see another violateion, we can return false
        but can we even make that change the changes the the array to non-drecreasing
        [3,4,5,3,6,8]
        [5,3] is the violation
        but we want to consider [4,5,3], no matter what chane we do we cannot make this non-decreasing
        if nums[i-1] > num[i], there is no point in updates nums[i-1] to nums[i]
        why?
        we already know nums[i-1] >= nums[i-2] since the first violation we found was the index 'i'
        SO:
        if nums[i-2] > nums[i]:
            nums[i] = nums[i-1] #it smmaller
        else:
            nums[i-1] = nums[i]
        after making the modification we expect the rest of the array to besorted (at least >=)
        if after we see another violation after making the change, then we cannot make the nums array non-decreasing with onlye one change
        takeaways:
            actually make a change, making sure that we can
            then check the rest of the array to see if its ok
        algo:
            1. we iterate over the array until we reach the end oor find a violation
            2. if we get through the whole array, return True
            3. otherwise, when we find a violation we consider i,i-1,i-2
                * if the violation is at the index 1, we wont have i-1, in that case sets nums[i-1] to nums[i]
                * otherise we check if nus[i-2] <= nums[i-1] in which case we set nums[i-1] = nums[i]
                * else nums[i] = nums[i-1], make sure to go over this
        '''
        violations = 0
        N = len(nums)
        for i in range(1,N):
            #find violation
            if nums[i-1] > nums[i]:
                #check if we have more than one
                if violations == 1:
                    return False
                violations += 1
                #otherwise we are free to make a change
                if i < 2 or nums[i-2] <= nums[i]:
                    #make the switch
                    nums[i-1] = nums[i]
                else:
                    #i cant make it non -decreasing
                    nums[i] = nums[i-1]
        return True
        
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        https://leetcode.com/problems/non-decreasing-array/discuss/1190780/Python-Easy-one-pass-and-two-pass-O(N)-solutions
        we can do passes with this solution
        intution:
            keep track of the largest max seen so far at every point in the traversal
            then whenever the nums[i] < curremtn max, increase a count
            if we that count exceeds 1, we cant fix the array with one change
        BUT THERE IS AN EDGE
        100,1,2,3,4
        if we did it this way, we'd return False, because 100 stays
        but we could have just made it 1,1,2,3,4
        to adjust for this, just go backwards, but correct for the min
        '''
        max_so_far,min_so_far = float('-inf'),float('inf')
        num_ups, num_downs = 0,0
        N = len(nums)
        
        #first pass, check less than max so far
        for n in nums:
            if n < max_so_far:
                num_ups += 1
            max_so_far = max(n, max_so_far)
            
        #second pass, go backwards still check for increasing, but greater than min so far
        for n in reversed(nums):
            if n > min_so_far:
                num_downs += 1
            min_so_far = min(n,min_so_far)
        
        return min(num_ups,num_downs) <= 1

#even faster
class Solution(object):
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        '''
        refercne this article for the the algo
        https://leetcode.com/problems/non-decreasing-array/discuss/1190763/JS-Python-Java-C%2B%2B-or-Simple-Solution-w-Visual-Explanation
        always examine i-i and i, if nums[i-1] < nums[i]
        case 1:
            if we have nums[i-2] > nums[i-1] > nums[i], we can immediatler return false
            why? two violations present
        case2:
            if nums[i-2] < nums[i-1] > nums[i] < nums[i+1], we can move nums[i-1] and nums[i], only one of them, to make this segment non=decreasing
        case3:
            same as case 2, but we cannot fix it with on cange
        '''
        errors = 0
        for i in range(1,len(nums)):
            #violation
            if nums[i-1] > nums[i]:
                if errors == 1 or (i > 1 and i < len(nums)-1 and nums[i-2] > nums[i] and nums[i-1] > nums[i+1] ):
                    return False
                errors += 1
        return True
     
########################
#Jump Game II
########################
#this gets TLE
#edited, NOTE: add to seen before adding to q
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        N = len(nums)
        seen = set()
        
        q = deque([(0,0)]) # tuple is (spot,jumps)
        
        while q:
            current_spot,curr_jumps = q.popleft()
            #made it to the end
            if current_spot == N-1:
                return curr_jumps
            #othweise i have spots i can go to via the jumps allowed at that sport
            #crap i can also jump backwards
            for jump in range(1,nums[current_spot]+1):
                next_spot = current_spot + jump
                if next_spot not in seen:
                    q.append((next_spot,curr_jumps+1)) #disjance of jump, does not mean number of jups
                    seen.add(next_spot)
#another BFS
#takeaway, mayybe add to visited after checking neighbords, not right away when popping
#also process length og q, and global result
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        for some reason, my BFS get's TLE....
        https://leetcode.com/problems/jump-game-ii/discuss/1192310/Python-Solution
        lets try it anonther way:
        KEY: we keep processinng the length of the q every time
        NOTE: FOR SOME REASONO ITS BETTER to add seen later
        also, better  to return global variable for jumps, instead of pairinng as tuple in  the  q
        
        '''
        N = len(nums)
        if N == 1:
            return 0
        q = deque([0])
        seen = set([0])
        jumps = 0
        while q:
            jumps  += 1
            for i in range(len(q)):
                curr_spot = q.popleft()
                max_jump = nums[curr_spot]
                for j in range(1,max_jump+1):
                    next_spot = curr_spot + j
                    if next_spot == N-1:
                        return  jumps
                    if next_spot not in seen:
                        seen.add(next_spot)
                        q.append(next_spot)

#greedy solution
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        make sure to revisit the LC 55: Jump Game for the back tracking approach
        enumerating all paths and checking the min is O(2^n) time
        we can instead find the shortest jumps by making locally optimal choices at each index which leads to a globally optimal solution
        first thought:
            if im at i, the next subarry available to me is the array [nums[i+1],nums[nums[i]]], now ask myself? do i greedily take the largest jump available, or do i jump the next sport that alllows me mores sports to jump after i make htat jump
        intuition:
            us proof by contacdiction with a few thoght examples?
            break the decision into two people, A and B
            where A, jumps only 1 and B takes the largest jump
            you see that A has more options and thus it must be tree, because if i followed B, it would be a contradictions
        algo:
            init three int vars:
                jumps, count number of jumps
                currJumpEnd, marks the end of the range
                farthest, marks the farthest place we can jump to
            traverse the jumps, excludin the last (it is guarnteed we can reach the last)
            update farthest to i + nums[i] if the latter is largers
            if we reach the currJumpEnd, it means e finished teh current jump and can beign checking the next jump by seeting currJumpEnd = fartherst
            return jumps
        TAKEAWAY: continuously find the farthest position we can be at
        check if we can make, and use up a jump, but also update farthest
        '''
        jumps = 0
        currJumpEnd = 0
        farthest = 0
        for i in range(len(nums)-1):
            #find fatheest always
            farthest = max(farthest, i+nums[i])
            #if we have gotten to the end, jump and update the furthest
            if i == currJumpEnd:
                jumps += 1
                currJumpEnd = farthest
        return jumps

#recusrive memoization top down, but starting from the end
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        lets try the resurvie backtracking apporach
        base case, the min number of jumps to get to i == 0 is zero
        so if i == 0:
            return 0
        recursive function
        rec(i) returns the min number of jumps to get to i
        for each possible jump at nums[i]:
            the minimum for the spot is the min(MAX,rec(jump)+1)
        '''
        memo = {}
        def rec_jump(i):
            if i == 0:
                return 0
            if i in memo:
                return memo[i]
            result = float('inf')
            for j in range(i):
                if nums[j] >= i -j:
                    result = min(result, rec_jump(j)+1)
            memo[i] = result
            return result
        
        return rec_jump(len(nums)-1)

#dp
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        another dp solution
        for each position in nums, try to find the smallest number of jumps it could have taken to get ther
        
        '''
        N = len(nums)
        dp = [float('inf') for _ in range(N)]
        #i know i can get to the first spot
        dp[0] = 0
        
        #now examine every spot after the frist
        for i in range(1,N):
            #all apossible jump positions from current i
            #meand i could have gotten to i from j
            for j in range(i):
                if nums[j] + j >= i: 
                    #i can reach it on this jump, and keep taking min
                    dp[i] = min(dp[i],dp[j]+1)
        return dp[-1]

#good article showing all the approaches
#https://leetcode.com/problems/jump-game-ii/discuss/1192401/Easy-Solutions-w-Explanation-or-Optimizations-from-Brute-Force-to-DP-to-Greedy-BFS
        
#we can memoize
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        just go over the recursive approaches first
        brute force recursive TLE
        start at index 0 and see if we an reach the end
        at each position i, the jump size is in the range [1,nums[i]]
        define recurisve function
        rec(i) = min(curr_minjumps,1+rec(i +jth jump)
        ADDING MEMEO
        '''
        memo = {}
        def rec_jump(pos):
            #if we have gone past the array, no more jumps are need
            if pos >= len(nums)-1:
                return 0
            if pos in memo:
                return memo[pos]
            min_jumps = 1001 #remember the size of the array is 1000
            #get possible jump ranges
            for jump_size in range(1,nums[pos]+1):
                min_jumps = min(min_jumps,1+rec_jump(pos+jump_size))
            memo[pos] = min_jumps
            return memo[pos]
        
        return rec_jump(0)


###############################
#Convert Sorted List to Binary Search Tree
###############################
#aye yai yai
#so this workds but it doesnlt make it balanced

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        '''
        height balanced means for every subtree of a node, the depths of the subtrees differ by no more than one
        i can pull the head apart, and make it an array
        then make a BST from inorder
        recursion, find middle of list and make is the root
        recursively do the same for heft left half and the right half
            left half : get middle, and make it left child of the root 
            right halt, get the middle and make it the right child of the root
        '''
        if not head:
            return None
        array = []
        temp = head
        while temp:
            array.append(temp.val)
            temp = temp.next
        
        def build_tree(inorder):
            if len(inorder) == 0:
                return None
            max_idx = inorder.index(max(inorder))
            root = TreeNode(inorder[max_idx])
            #if this is the only element in inorder, return it
            root.left = build_tree(inorder[:max_idx])
            root.right = build_tree(inorder[max_idx+1:])
            return root
        
        return build_tree(array)


#using up space to get the in order array
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        '''
        height balanced means for every subtree of a node, the depths of the subtrees differ by no more than one
        i can pull the head apart, and make it an array
        then make a BST from inorder
        recursion, find middle of list and make is the root
        recursively do the same for heft left half and the right half
            left half : get middle, and make it left child of the root 
            right halt, get the middle and make it the right child of the root
        '''
        if not head:
            return None
        array = []
        temp = head
        while temp:
            array.append(temp.val)
            temp = temp.next
        
        def build_tree(inorder,start,end):
            if start > end:
                return None
            mid = (start + end) // 2
            root = TreeNode(inorder[mid])
            #when the pointers meet up, return the node created
            if start == end:
                return root
            #if this is the only element in inorder, return it
            root.left = build_tree(inorder,start,mid-1)
            root.right = build_tree(inorder,mid+1,end)
            return root
        
        return build_tree(array, 0,len(array)-1)

#i could also just find the middle of the linkeind list every time
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        '''
        instead of building out the array, use the linked list, but find the middle
        if we always root the middle value, recusrively, then the heights of each tree will differ by more than one, imagine the case for even , then odd
        algo:
            when finding the middle of the linkeind list, we use two slow and fast pointers
            the slow marks the middle, but we need to break it!
            we keep another pointer to slow, and when the loop invarirant ends, we keep this pointer to slow, and make its next be none
            we pass the halves left and right recusrively
        '''
        def get_middle(head):
            prev = None
            slow = fast = head
            while fast and fast.next:
                prev = slow
                slow = slow.next
                fast = fast.next.next
            if prev:
                prev.next = None
            return slow
        def build_tree(head):
            #nothing to return
            if not head:
                return
            mid = get_middle(head)
            root = TreeNode(mid.val)
            #when there is only one element make it a node
            if head == mid:
                return root
            root.left = build_tree(head)
            root.right = build_tree(mid.next)
            return root
        return build_tree(head)

#inorder simulation
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        '''
        https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/discuss/1194284/Python-Optimal-O(n)-inorder-traversal-explained
        we know that an inorder traversal gives the elements in ascending order
        if we traverse our linked list at the same time we do our inorder traversal, we will end up with a binary tree
        kinda confusing...but i have to keep going
        '''
        #get the size
        size = 0
        temp = head
        while temp:
            temp = temp.next
            size += 1
        self.head = head #need to keep head global
        
        def build_tree(start,end):
            if start > end:
                return None
            mid = (start + end) // 2
            left = build_tree(start,mid-1)
            #make the root with the curent heads val
            root = TreeNode(self.head.val)
            #we need to move the linked list as we move through our inorder
            self.head = self.head.next
            #now make the left
            root.left = left
            #make right
            root.right = build_tree(mid+1,end)
            return root
        return build_tree(0,size-1)

###################################
#Deletion Operations for Two Strings
##################################
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        '''
        go through all the approaches...
        in order to determine the minimum number of delete operations needed, we can make use of the length of the longest common subseqeucne within the two strings (LCS)
        if we know the length of this, we can easily determing the required results as m+n -2*lcs
        but why is it m+n -2*lcs,
        if the two strings cannot be euqliazed, then number of deletinos of just m+n
        now if there is a common subsequence, which has length lcs, well we just need to do lcs less operations on both strings
        the problem now degenerats to how to find LCS:
        FINDING LCS:
            in order to find the length of the longest common subsequence, we use recurison
            function lcs(s1,s2,i,j)
            for evaluting the function, we cheeck if the chars s1[m-1] and s[n-1] are queal
            if they match we can consider the corresponding strings up to 1 less lengthers since the last chars have already been considered and add 1 to the result
            if they don't match we go back 1 index for each string and take the max
        '''
        memo = {}
        def lcs(s1,s2,i,j):
            #base case, zero index i and j, no matching, nothing to contribute
            if i == 0 or j == 0:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            res = None
            if s1[i-1] == s2[j-1]:
                res = 1 + lcs(s1,s2,i-1,j-1)
            else:
                take_first = lcs(s1,s2,i,j-1)
                take_second = lcs(s1,s2,i-1,j)
                res = max(take_first,take_second)
            memo[(i,j)] = res
            return memo[(i,j)]
        LCS = lcs(word1,word2,len(word1),len(word2))
        return len(word1)+len(word2) -2*LCS

#DP solution O(MN) time and O(MN) space
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        '''
        now that we have the recurrence down, lets try solving it using a dp array
        '''
        M = len(word1)
        N = len(word2)
        dp = [[0 for _ in range(N+1)] for _ in range(M+1)]
        for i in range(M+1):
            for j in range(N+1):
                if i == 0 or j == 0:
                    continue
                #first case, both match
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                #they dont matche, take last of first and second last of second, vice versa, max
                else:
                    dp[i][j] = max(dp[i-1][j],dp[i][j-1])
        print dp,dp[M][N]
        return M+N-(2*dp[M][N])

#using dp directly to get the number of deletions at each index
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        '''
        instead of using dp to find the LCS, we can use dp to find the number of deletions needed to equalize the string at each index
        if the chars match, we replicate the previous entry [i-1][j-1], it doen't need to be delted if it matches
        if they don't match, we need to delete either the currne char of s1 or s2
        we also need to increment by 1
        '''
        M = len(word1)
        N = len(word2)
        dp = [[0 for _ in range(N+1)] for _ in range(M+1)]
        for i in range(M+1):
            for j in range(N+1):
                if i == 0 or j == 0:
                    dp[i][j] = i + j
                #first case, both match
                elif word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                #they dont matche, take last of first and second last of second, vice versa, max
                else:
                    dp[i][j] = 1+min(dp[i-1][j],dp[i][j-1])
        return dp[M][N]

#finally we can reduce the space consumption by noticing we only need the previous row
#create row, iteratre, thenre assign trick
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        '''
        saving on space
        '''
        M = len(word1)
        N = len(word2)
        dp = [0 for _ in range(N+1)]
        for i in range(M+1):
            temp = [0 for _ in range(N+1)]
            for j in range(N+1):
                if i == 0 or j == 0:
                    temp[j] = i + j
                #first case, both match
                elif word1[i-1] == word2[j-1]:
                    temp[j] = dp[j-1]
                #they dont matche, take last of first and second last of second, vice versa, max
                else:
                    temp[j] = 1+min(dp[j],temp[j-1]) #in the hold one dp[i-1] == temp
            dp = temp
        return dp[N]

class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        '''
        we can only use recursion again, but this time we use recursion to find the number of deletions
        instead of finding the LCS!
        index pointers i,j
        if i == len(word1) or j == len(word2), one of the strings is empty, so just sum the remaining
        when word1[i] == word2[j], no need to delte so just move up the indexes
        when word1[i] != word2[j], we need one more deletion and we skip one of the chars in the words and take the min
        '''
        memo = {}
        def dp(i,j):
            if (i,j) in memo:
                return memo[(i,j)]
            if i == len(word1) or j == len(word2):
                ans = len(word1) + len(word2) - i - j
            elif word1[i] == word2[j]:
                ans = dp(i+1,j+1)
            else:
                ans = 1 + min(dp(i+1,j),dp(i,j+1))
            memo[(i,j)] = ans
            return memo[(i,j)]
        
        return dp(0,0)

#####################
#Super Palindromes
#####################
#welp, this gets TLE
#hard problem, be happy you got a solution for this
class Solution(object):
    def superpalindromesInRange(self, left, right):
        """
        :type left: str
        :type right: str
        :rtype: int
        """
        '''
        i cant just search for all super palindromes in range [1,10^18]
        i can flip it, instead of checking each number for palindrom and sqaure is also palindrome
        check if number is palindrome, and square root is plaindrome
        since we are squaring, changes the bounds to checkf rom sqrt(left) to sqrt(right)
        then for every number in that his range check is number is palindrome and squared number is palindrom
        '''
        #helper
        def isPal(n):
            return str(n) == str(n)[::-1]
        
        smallest = int(int(left)**0.5)
        largest = int(int(right)**0.5)
        
        count = 0
        for num in range(smallest, largest+1):
            if isPal(num) and isPal(num**2):
                count += 1
        return count

class Solution(object):
    def superpalindromesInRange(self, left, right):
        """
        :type left: str
        :type right: str
        :rtype: int
        """
        '''
        call super palindrom P
        P = R**2, where R is also palindrome
        the first half of digits R must match the last half of digits R, depending on odd or even length
        P <= 10^18
        implies R <= 10^9
        and since R is a palindrome, the first half must be the same as the last half 
        call the first k digits of R, and the last half as k'
        k+k' = len(R),
        which implies k < 10^(9/2)
        which implies k < 10^5, inclusive 4.5
        '''
        low,high = int(left),int(right)
        highestPalindrome = 10**5
        
        #cool way of reversing instread of recasting as string
        def reverse(x):
            ans = 0
            while x:
                ans =  10*ans + x % 10 #popping of last digit and adding next multiple of 10
                x //= 10 #removing last digit
            return ans
        
        def isPal(x):
            return x == reverse(x)
        
        count = 0
        
        #odd length palindrom
        for k in range(highestPalindrome):
            k = str(k)
            k = k + k[-2::-1] #could also so k = k + k[:-1][::-1]
            cand = int(k)**2
            if cand > high:
                break
            if cand >= low and isPal(cand):
                count += 1
        #for even length
        for k in range(highestPalindrome):
            k = str(k)
            k = k + k[::-1]
            cand = int(k)**2
            if cand > high:
                break
            if cand >= low and isPal(cand):
                count += 1
        return count

class Solution(object):
    
    def __init__(self):
        self.highestPalindrome = 10**5
        self.allSupers = []
        for i in range(1, self.highestPalindrome):
            odd = int((str(i))+str(i)[:-1][::-1])**2
            even = int(str(i)+str(i)[::-1])**2
            
            if self.isPal(odd):
                self.allSupers.append(odd)
            if self.isPal(even):
                self.allSupers.append(even)
    
    #revers functino, instead of creating the string and checking its revers
    def reverse(self,x):
        ans = 0
        while x:
            ans =  10*ans + x % 10 #popping of last digit and adding next multiple of 10
            x //= 10 #removing last digit
        return ans
    
    def isPal(self,x):
        return x == self.reverse(x)
        
    def superpalindromesInRange(self, left, right):
        """
        :type left: str
        :type right: str
        :rtype: int
        """
        '''
        instead of doing it this way, we can generate all super palindromes in the constructor
        using the same strategy
        then just find the super palindromes in the method call in the constraint
        '''
        count = 0
        for num in self.allSupers:
            if int(left) <= num <= int(right):
                count += 1
        return count
###################################
#Put Boxes Into the Warehouse I
##################################
#so close!!! 96/97
#brute force gets TLE
class Solution(object):
    def maxBoxesInWarehouse(self, boxes, warehouse):
        """
        :type boxes: List[int]
        :type warehouse: List[int]
        :rtype: int
        """
        '''
        there are few constrains we need to pay attentino too
        you cannot stack boxes, 
        can only slide a box in from left to right
        you can rearrange in the insertion order of boxes
        if height of box is less than the current room,it can fit
        sort boxes increasingly
        i can keep a an array indicating if that room number is occupied
        and just keep sliding in, after going through all the boxes return the number of Trues on the array
        '''
        rooms = len(warehouse)
        filled = [False for _ in range(rooms)]
        boxes.sort()
        for i in range(len(boxes)):
            curr_box = boxes[i]
            j = 0
            #sliding
            while j < rooms and curr_box <= warehouse[j] and filled[j] == False:
                j += 1
            #current box couldn't fit
            if j == 0:
                continue
            else:
                filled[j-1] = True
        return sum(filled)

class Solution(object):
    def maxBoxesInWarehouse(self, boxes, warehouse):
        """
        :type boxes: List[int]
        :type warehouse: List[int]
        :rtype: int
        """
        '''
        intuition, there's only one opening and i have a bunch of boxes of different sizes
        if we are trying to fit as many boxes in the warehouse, it makes sens to do all the small ones first!
        a larger box could indeed fit, but it would block other smaller boxes from fitting in
        intuition:
            say for example we have a box of height h and we want to push it in as far as we can
            the point where we can't push is when the h is greater than the current room, so we stop
            we first pre process the warehouse array, keeping in mind that the limiting factor for each position is the minimum height that comes before it, we update the height for each position so that it is no height than this minimum, make the warehouse weakly decreasing array, we trunacte the size of the room to the height of the smallest room left of it
            LOWER THE ROOMS USABLE HEIGHT!!!!
            then sort the boxes increasingly
        '''
        #alter the usable sizes of the rooms in the warhouse
        for i in range(1,len(warehouse)):
            warehouse[i] = min(warehouse[i],warehouse[i-1]) #this is fucking clever
        
        #sort
        boxes.sort()
        
        count = 0
        
        #start from the smallest 'bottlenecked room', and try to fit in the smallest box
        for i in range(len(warehouse)):
            #going right to left in rooms
            curr_room = warehouse[len(warehouse)-i-1]
            if count < len(boxes) and boxes[count] <= curr_room:
                count += 1 #note, this also moves up our index if we can use that room
            #other wise we can't use that room but we still remain on the next box
        return count

class Solution(object):
    def maxBoxesInWarehouse(self, boxes, warehouse):
        """
        :type boxes: List[int]
        :type warehouse: List[int]
        :rtype: int
        """
        '''
        in the last we appoach we modified the input, but we might not always have that luxury in an interview
        again we can be greedy, but we be greedy in a different way
        we sort deceasinly and try to fit the largest box first in the left most array, think on this
        one we fill in that room, we go on to the next room and try to fill it with the next box
        for each position, we discard boxes that are too tal to find in the current room, because they won't fid in any rooms further to the right
        algo:
            sort boxes from largest to smallest
            try to fit the largest box to the left most room
        '''
        boxPtr = 0
        count = 0
        boxes.sort(reverse = True)
        
        #now try to fit the largest box going left to right in warehouse
        for i in range(len(warehouse)):
            curr_room = warehouse[i]
            #keep moving to the enxt smalelst box
            while boxPtr < len(boxes) and boxes[boxPtr] > curr_room:
                boxPtr += 1
            if boxPtr == len(boxes): #i.e we've gone through all the boxes
                return count
            count += 1
            boxPtr += 1
        return count

######################################
#Construct Target Array With Multiple Sums
######################################
#TLE
class Solution(object):
    def isPossible(self, target):
        """
        :type target: List[int]
        :rtype: bool
        """
        '''
        i always start with the ones array of len(target)
        one operation consists of choosing an index in the starting array and make the index element be the curretn sum
        return True of i can make the target
        the sum always increase by sum(A)
        what if we start from target and see if we can get to the ones array
        [9 3 5] sum is 17, prvious sum is 17 - 9 = 8, subtract 8 from the largest
        [1 3 5] sum is 9, previous sum is 9- 5 = 4, subtract 4 from largest
        [1 3 1] sum is 5, previous sum is 5 -3 = 2, subtract 2 from largest
        [1 1 1], final sum is 3
        simulate reducing
        '''
        size = len(target)
        #when we reduce target, if we can, it shold be all ones, with sum == size
        #make max heap tuple (maxvalue, index)
        heap = []
        for idx,num in enumerate(target):
            heappush(heap,(-num,idx))
        
        #simulate
        while sum(target) > size:
            #get current sum
            curr_sum = sum(target)
            largest_val,idx = heappop(heap)
            #get the difference and deduct from the largest at the index
            diff = curr_sum - -(largest_val)
            target[idx] -= diff
            #push back into heap
            heappush(heap,(-target[idx],idx))
        return sum(target) == size

#TLE once again
class Solution(object):
    def isPossible(self, target):
        """
        :type target: List[int]
        :rtype: bool
        """
        '''
        almost had it, but was getting TLE on cases like 9 9 9 9, 8 8 8 8, etc
        takeaways,
            always reduce the largest element in the array by the current sum - previous sum
            and check if we can get to the 1's array, so long as we have a valid array of length >= 2 or > 1
            i.e we reduce the largeget element to x: x = largest - (curr_sum - largest)
        the problem with this solution is thate for cases with one smalle and one really large [1,100000000]
        it will keep reducing by one, which is slow
        orer doesn't matter
        then we keep simulating until the element at the top is > 1,
        one the max element is equal to 1, everything else should be one too, why?
        we keep reducing the largest until it is 1, once the largest is 1,everything else should be 1
        '''
        #edge case of length 1
        if len(target) == 1:
            return target == [1]
        
        total_sum = sum(target)
        heap = []
        #we don't care about indices, we only care about the largest element at any one time
        for num in target:
            heappush(heap,-num) #max heap
        
        #simulate until the max eleemnt is 1
        while -heap[0] > 1:
            largest = -heappop(heap)
            #get the new reduced element
            x = largest - (total_sum - largest)
            #if we go below zero when reducing, we can't make the ones array
            if x < 1:
                return False
            #update total sum
            total_sum = total_sum - largest + x
            heappush(heap,-x)
        return True

#finally
class Solution(object):
    def isPossible(self, target):
        """
        :type target: List[int]
        :rtype: bool
        """
        '''
        weve dont great so far, but we fail on cases [1,10000000]
        lets see what happens when we do a few
        target = [1,4,998]
        [1,4,993]
        [1,4,988]
        [1,4,983]
        notice we reduce by five each time, which is the sum of the remaning numbers
        it would eventually get down to
        [1,4,3] , whic was 998 % 5
        KEY:
            instead of doing x = largest - (total_sum - largest)
            we do x = largest % (total_sum - largest), take the MOD, or whats leftover
        TAKEAWAYS:
            the largest is always  at least half the total sum_sum
            the largest is always replaces with avalue at most half of itself
        lets create a variable rest, which is just the curr sum less largest, or simply the sum exlucding the largest
        rest = total_sum - largest
        this implies largest is ALWAYS greater than rest, if it weren't the case, the reduced x value would go neagtive and we can return False
        because largest > rest, we know that x is at most largest - rest,
        if rest is at least half the size of largest, then this will clearly chop largest in half
        '''
        #edge case of length 1
        if len(target) == 1:
            return target == [1]
        
        total_sum = sum(target)
        heap = []
        #we don't care about indices, we only care about the largest element at any one time
        for num in target:
            heappush(heap,-num) #max heap
        
        #simulate until the max eleemnt is 1
        while -heap[0] > 1:
            largest = -heappop(heap)
            #get the new reduced element
            rest = total_sum - largest
            #careful when taking moduls of 1, when == 2
            if rest == 1:
                return True
            x = largest % rest
            #ix is now zero or invalud or didn't change return False
            if x == largest or x == 0:
                return False
            #update total sum
            total_sum = total_sum - largest + x
            heappush(heap,-x)
        return True

###########################
#Count Primes
##########################
#got O(sqrt(N)) time, looks like they want us to do better
#use Seive
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        i just check whether the number divivides into every number up to sqrt(n)
        which is better than n and sqrt(n)
        '''
        def isPrime(n):
            for i in range(2,int(n**.5)+1):
                if n % i == 0:
                    return False
            return True
        
        #edge cases 
        if n == 0 or n == 1:
            return 0
        count = 0
        for i in range(2,n):
            if isPrime(i):
                count += 1
        return count

#good job on the review!
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        we can use sieve of Eratosthenes, we just keep markinf of multiples in a 1d array
        we start off with 2, and eliminate all muliplies of 2
        then 3, and eliminate all multiples of three,
        but when we get to 4, we should have  already elinated its multiples because we passed 2
        therefore if we are at number of p, we  can mark of numbers in multiples if  n: p**2 + n*p
        we mark as non primes for every number up to  sqrt(n), but how do we mark efficientely
        KEY:
            we can start with the smalest prime nuber 2, and mark all of its multiples up to 'n' as non primes,
            then we repeate the same process for the next available number in the array that is not marked as composite and so on
            outer loop is bounded by sqrt(n)
        WHYS?
        lets examin numbers less than 36
        6 * 1 = 6 = 1 * 6
        6 * 2 = 12 = 2 * 6
        6 * 3 = 18 = 3 * 6
        6 * 4 = 24 = 2 * 12
        6 * 5 = 30 = 5 * 6
        6 * 6 = 36 > 30
        notice that every multiple of 6 was adressed with some mutiple less thant sqrt(36), this is super fucking important!
        how about the inner loop,
            we will INVARIANTLY piick the nextre prime number available (we pass if eleement in array is True)
            say if we pick prime i, then we loop through i**2 in increment of i but only up to n
        why do we start at i^2? why not 2*i?, fundamental theorem of arithmetic,
            all previous multiples would have already been covered by a previous prime
            
        Let's assume that n is 50 (a value greater than 7*7) to demonstrate this claim. 

        7 * 2 = 14 = 2 * 7
        7 * 3 = 21 = 3 * 7
        7 * 4 = 28 = 2 * 2 * 7 = 2 * 14
        7 * 5 = 35 = 5 * 7
        7 * 6 = 42 = 2 * 3 * 7 = 2 * 21
        general algo:
            create list of consecutive intergs frmo 2 to n
            out loop candidate prime p frm 2 to sqrt(n)
            add to this canddiate p*p + some multiple of p, which should also be prime
            find the smallest number in the list greater than p that is not marked, if there was no such number, stop,
            otherwise let p now equal this new number (which is the next prime), 
            when the algo terminates, all of the remaning numbers not marked are prime
            
        for loops won't work for the boolean array solution
        '''
        #init array, all numbers up to n are initially assumed to be prime
        isPrime = [True]*n
        i =2
        while i*i < n:
            #found occruence of prime
            if isPrime[i] == True:
                #check all multiples of this primes i,e, prime**2 + prime for prime in prime
                j = i*i
                while j<n:
                    isPrime[j] = False
                    j += i
            i += 1
        
        count = 0
        for i in range(2, n):
            if isPrime[i] == True:
                count += 1
        return count

#using hashset with Python
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return 0
        
        numbers = set()
        for i in range(2, int(sqrt(n))+1):
            if i not in numbers:
                for j in range(i*i,n,i):
                    numbers.add(j)
        return n - len(numbers) - 2

#array slicing with spaces and answer imputation
#this is actually a really cool trick
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        instead of using an array, we can use a dictionary,
        if a number is not in the dictionary it is prime by definition
        the dictionary holds all non prime numbers less than n
        note the solution does not execute in the time limit
        '''
        if n < 3:
            return 0
        primes = [True] * n
        primes[0] = primes[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if primes[i]:
                primes[i * i: n: i] = [False] * len(primes[i * i: n: i])
        return sum(primes)

#readings on proof for sum of reciprocal primes
#https://en.wikipedia.org/wiki/Divergence_of_the_sum_of_the_reciprocals_of_the_primes
#http://www.cs.umd.edu/~gasarch/BLOGPAPERS/sump.pdf

######################################
#  Maximum Points You Can Obtain from Cards
######################################
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        '''
        i can only take cards from the left and right up to k times
        return the max
        well the hint gave it away
        get the total sum of the  array as total points
        removing k element from left and right means removing a subarray of length n-k
        hint 2:
            keep window size of n-k over the array, the answer is max(answe, total_pts - sumCurrentwind)
        im stil trying to figue out why this works
        the problem degenerts to removing a subarray of length n-k from the array
        examin all subarrays of that length and max points obtained would be the the difference of the total - the the smallest sum subarray of length k-n, but we can max on the fly
        PROOVE:
        [1,2,3,4,5] if k = 2
        case 1: all from left
        [3,4,5]
        case 2: all from right
        [1,2,3]
        case 3: alternationg
        [2,3,4]
        in every case we  end  up with a sub array of length n  - k
        i wonder if not using a sliding window causes TLE
        yep, TLE, need to use  sliding window
        '''
        totalPoints = sum(cardPoints)
        N = len(cardPoints)
        size = N - k
        ans = 0
        for i in range(N-size+1):
            ans = max(ans, totalPoints -sum(cardPoints[i:i+size]))
        return ans
        
#hacky sliding window, after correcting
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        totalPoints = sum(cardPoints)
        N = len(cardPoints)
        size = N - k
        ans = 0
        curr_sum = None
        for i in range(N-size+1):
            #getting the first window
            if curr_sum  == None:
                curr_window = cardPoints[i:i+size]
                curr_sum = sum(curr_window)
            else:
                curr_sum = curr_sum - cardPoints[i-1] + cardPoints[i+size-1]
                
            ans = max(ans, totalPoints- curr_sum)
        return ans

#well i got the recursion working
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        '''
        brute force would be examine all possible combinations of left and right takes
        base case would be when k cars are chose or when no cards are left to be selected
        but this gets TLE, for fun lets try coding this out
        '''
        self.highest = 0
        
        def rec(left,right,points,taken):
            if taken == k:
                self.highest = max(self.highest,points)
                return
            
            rec(left+1,right,points+cardPoints[left],taken+1)
            rec(left,right-1,points+cardPoints[right],taken+1)
        rec(0,len(cardPoints)-1,0,0)
        return self.highest

class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        '''
        brute force would be examine all possible combinations of left and right takes
        base case would be when k cars are chose or when no cards are left to be selected
        but this gets TLE, for fun lets try coding this out
        '''
        memo = {}
        
        def rec(left,right,k):
            #out of numbers/ nothing to take
            if left > right and k < 0:
                return 0
            if k == 0:
                return 0
            if (left,right) in memo:
                return memo[(left,right)]
            left_take = cardPoints[left] + rec(left+1,right,k-1)
            right_take = cardPoints[right] + rec(left,right-1,k-1)
            memo[(left,right)] = max(left_take,right_take)
            return memo[(left,right)]
        return rec(0,len(cardPoints)-1,k)
            
#dp
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        '''
        using dynamic programming
        when we take (could be all left, all from right, or a mix), we are just taking subarrays
        KEY:
            if we take i cards from the left, then we must take k-i cards from the right, if k-i > 0
            we can simualte this by going fowards and then backwards, putting our results into two arrays
            then we use these to check each possible way of slecting i cars from left and k-i cards from the right
            get prefix sum for first k values, and then for each of the last k values
        algo:
            init two arrays of size k + 1, call them front and read
            We calculate the prefix sum (sum of 0 <= i <= k cards) for the first k cards frontSetOfCards[i + 1] = frontSetOfCards[i] + cardPoints[i] and the last k cards rearSetOfCards[i + 1] = cardPoints[n - i - 1] + rearSetOfCards[i]
            init max score to zero
            Iterate from i = 0 -> k. At each iteration, we determine the possible score by selecting i cards from the beginning of the array and k - i cards from the end (currentScore). If this score is greater than the maxScore then we update it.
            
        '''
        N = len(cardPoints)
        front = [0]*(k+1)
        rear = [0]*(k+1)
        for i in range(k):
            #prefix sums
            front[i+1] = cardPoints[i] + front[i]
            rear[i+1] = cardPoints[N-i-1] + rear[i]
        #print front,rear
        
        SCORE = 0
        
        #using the arrays take i from the beginnin and k-i from the end
        for i in range(k+1):
            begCards = front[i]
            endCards = rear[k-i]
            SCORE = max(SCORE,begCards+endCards)
        return SCORE

#dp collpasing the array
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        '''
        we can collapse the dp array to get constant space, although its not like the usual collapse with other dp problems
        instead of precopmuting the arrays, we calculate them on the fly, and store the total score in two vars
        algo:
            init two vars front and rear, remember we select i from front or k-i from rear
            front is init to sum of first k cars in array
            inint maxScore to front score
            now go backwards, and at each iteration, we calculate the score by selecting i cards from the beginning of the array and k-i from the end
        '''
        front = sum(cardPoints[:k])
        rear = 0
        N = len(cardPoints)
        
        maxscore = front #just assume we take from the beginning, but we'll simulate taking k-i end and see if that makes our score any better
        for i in range(k-1,-1,-1):
            #incrment rear scroe
            rear += cardPoints[N-(k-i)]
            #if i took this rear card, i need to get rid of a front car
            front -= cardPoints[i]
            #this is the new score
            currscore = rear+front
            #update
            maxscore = max(maxscore,currscore)
            #print i,N-(k-i)
        
        return maxscore

#the above was already similar to a sliding window solution
#here's the actual one
#i think i pretty much had it
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        N = len(cardPoints)
        best = currSum = sum(cardPoints[:k])
        #now we remove from the start and two from the end
        for i in range(k-1,-1,-1):
            #increment by end, decremnt off front
            #print i, i+N-k
            currSum += cardPoints[i+N-k] - cardPoints[i]
            best = max(best,currSum)
        return best

#another way, but using sliding window for find the min subarray, intstead of maxing taken subarray
class Solution(object):
    def maxScore(self, cardPoints, k):
        """
        :type cardPoints: List[int]
        :type k: int
        :rtype: int
        """
        N = len(cardPoints)
        size = N-k
        totalSum = sum(cardPoints)
        minSubSum = subSum = sum(cardPoints[:size])
        for i in range(k): #we have k subarrays of size k
            subSum -= cardPoints[i]
            subSum += cardPoints[i+size]
            minSubSum = min(minSubSum,subSum)
        return totalSum - minSubSum

######################################
#Range Sum Query 2D - Immutable
########################################
#TLE??? wtf?!?!
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        #inint the matrix
        self.matrix = matrix
        

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        #now we just sum for start is indexed by (row1,col1) upper left
        #end is lower right (row2,col2)
        SUM = 0
        for i in range(row1,row2+1):
            for j in range(col1,col2+1):
                SUM += self.matrix[i][j]
        return SUM
# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)

#i need to resue past queries to make the it faste
'''
lets step aside for a minute and try to figure out how to get sums of a subaraary using pref sum
[1,2,3,4,5,6] is our array
[1, 3, 6, 10, 15, 21] is our prefix array
now if we wanted the sum fomr index 0 to index we know thats just 6
now how about index 2 to 4, we know this is 12
but we can also get this by taking 15 - 3, or prefix[4] - prefix[1]
i.e sum from i to j is just pefix[j] -prefix[i-1]
good job! now that you finall figured out the 2d prefix sum
'''
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        '''
        we can accumlate the rows, in our constructor, the constructors now thas O(mn) times
        but the method can be done in O(m)
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        self.cache = [[0]*(cols+1) for _ in range(rows)]
        #accumalte rows
        for i in range(rows):
            for j in range(cols):
                self.cache[i][j+1] = self.cache[i][j] + matrix[i][j]
        

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        SUM = 0
        for i in range(row1,row2+1):
            SUM += self.cache[i][col2+1] - self.cache[i][col1]
        return SUM

#now for the crafty way
#i don't really understand the sub problem and the final return answer
#but now the return is O(1), but constructors is O(mn)
class NumMatrix(object):

    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        '''
        we can write the sum of the bounded region as the sum of three other regions
        starting from upper left and going clockewise, label the corners of the square ABCD with origin marked 0
        we will call sum(XY) of the sums of rectangular regions marked sum(upperLEFT,lowerRIGHT)
        we have the following sums
        OD
        OB
        OC
        OA (notince we 0 is our pivot and we just slide the boundaries over)
        our region then sum(ABCD) can be written as:
        OB  + OC - OA + ABCD = OD
        solving for ABCD
        ABCD = OD - OB - OC + OA
        but how do we translate this cumaltively?
        each cell in our dp array dp(i,j) answers the subpobrlme:
            what is the area of the rectanle formed by upper left at (0,0) and lower right at (i-1,j-1)
            we know for each sub problem we can break down sum of that subproblem into the four aforementioned parts
            this is reflected in the final subproblem for the return value
            but what about the sub problem?
            we can say dp(i+1,j+1) = area(left) + area(above)+current matrix element - upper right - accumalted upper left 
            we subtract the last value beause of double counting
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        if rows == 0 or cols == 0:
            return
        self.dp = [[0]*(cols+1) for _ in range(rows+1)]
        for i in range(rows):
            for j in range(cols):
                self.dp[i+1][j+1] = self.dp[i+1][j] + self.dp[i][j+1] + matrix[i][j] - self.dp[i][j]

    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        #now just return the formula we dervied from the first part
        return self.dp[row2 + 1][col2 + 1] - self.dp[row1][col2 + 1] - self.dp[row2 + 1][col1] + self.dp[row1][col1]

#also check this out
#https://leetcode.com/problems/range-sum-query-2d-immutable/discuss/1204168/js-python-java-c-easy-4-rectangles-dp-solution-w-explanation/935307
   
#####################################
#Ambiguous Coordinates
#####################################
#edge cases are killing me...
class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        '''
        the fact that the size of the string is small suggets recursion w/o memo build
        i.e brute force backtracking
        things to keep track of from the problem statement
        NO LEADING ZEROS
        a decimal can prefix an int of there the decial is prefixed by another int
        in the return answer, it can be any order
        there must be a space after each comma and each point is represented as (x, y)
        this might actually be easier iterativel
        '''
        #try splitting on only commas first
        #idea split iteratively, then recurse? well i dont need to recurse
        results = []
        N = len(s)
        for i in range(2,N-1):
            #split on comma
            comma_split = s[:i]+", "+s[i:]
            #check is current split is valid answer, no leading zero
            if comma_split[1] != "0" and comma_split[i+1] != "0":
                results.append(comma_split)
            
            #otherwise we need to try make a result using decimal;
            #now try splitting on decimal, remember the decimal constaints
            #i can move the decimal through the comma split and check
            #but we also need to check non leading zeros
            
            #print(results)
            for j in range(2,len(comma_split)-1):
                decimal_split = comma_split[:j]+"."+comma_split[j:]
                #edge case here is gonna give me nightmars
                if decimal_split[j-1].isdigit() and decimal_split[j-2] != "0" and decimal_split[j+1].isdigit():
                    results.append(decimal_split)
                #print(decimal_split,decimal_split[j])
        #print(results)
        return results

#aye yai yai.....
class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        '''
        well the official solutions uses itertools and python bultins,
        lets just try to understand it
        we try adding a comma for all possible palces in the string, the make sense
        then for each framgment, where can we put a decimal
        KEY: because of extanneous zeros, we should ignore possibilites where the part of the framgent to the left of the decinaal starts with a zero and ignore possibilies where the part of the fragment to teh right of the decimal ends with 0
        
        '''
        def make(frag):
            N = len(frag)
            for d in range(1,N+1):
                left = frag[:d]
                right = frag[d:]
                if (left.startswith('0') == False or left == '0') and (right.endswith('0') == False):
                    yield(left +("." if d != N else '')+right)
        s = s[1:-1]
        results = []
        for i in range(1,len(s)):
            for cand in itertools.product(make(s[:i]),make(s[i:])):
                results.append("({}, {})".format(*cand))
        return(results)

class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        '''
        good solution here
        https://leetcode.com/problems/ambiguous-coordinates/discuss/934654/Python3-valid-numbers
        we just need to watch for leading zeros, and lagging zeros
        1. if a string has length 1 return it
        2. if a strings with length larger than 1 and starts and eids with 0 zero, it's not valid
        3. if string starts with 0, return 0.xxxx
        4. if a strings ends with 0, return xxxx0
        5. otherwsie, put a decinamls points in hte n-1 places
        then we just take the cartesian product of the possible fragments
        
        #notes on time complexity
        https://leetcode.com/problems/ambiguous-coordinates/discuss/123851/C%2B%2BJavaPython-Solution-with-Explanation
        '''
        s = s[1:-1]
        def helper(s):
            #returns valid entries from s
            if len(s) == 1:
                return([s])
            if s.startswith("0") and s.endswith("0"):
                return([])
            if s.startswith("0"):
                return([s[:1]+"."+s[1:]]) #skipping zeros
            if s.endswith("0"):
                return([s])
            else:
                temp = []
                for i in range(1,len(s)):
                    temp.append(s[:i]+"."+s[i:])
                temp.append(s)
                return temp
        #print(helper(s[:len(s)]))
        #print(helper(s[len(s):]))
        results = []
        for i in range(1,len(s)):
            for a in helper(s[:i]):
                for b in helper(s[i:]):
                    results.append(f"({a}, {b})")
        return(results)

class Solution:
    def ambiguousCoordinates(self, s: str) -> List[str]:
        '''
        just another way
        try to use a comma to split the original string
        then build up pairs from that comma split string
        but before building up pairs, we want to generate more candiates for the tuples by using a decimal
        '''
        s = s[1:-1]
        N = len(s)
        output = []
        
        def unpack(x):
            possible = []
            N = len(x)
            #first, no decimal
            if x[0] != "0" or x == "0":
                possible.append(x)
            #now place the decimal
            for i in range(1,N):
                #is it possible to even place a decimal?
                if (x[:i] == "0" or x[0] != "0") and x[-1] != "0":
                    cand = x[:i]+"."+x[i:]
                    possible.append(cand)
            return(possible)
        
        for i in range(1,N):
            left = s[:i]
            right = s[i:]
            #now for each left and right, try to build up possible canddiates
            for first in unpack(left):
                for second in unpack(right):
                    pair = (first,second)
                    output.append("({}, {})".format(first,second))
        return(output)

#####################################
# Flatten Binary Tree to Linked List
#####################################
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        easy way, dump contents into an extra array and build the linked list from there
        preoder is root,left,right
        '''
        if not root:
            root = TreeNode()
        def get_preorder(node):
            if not node:
                return []
            output = []
            output = [node.val]
            output += get_preorder(node.left)
            output += get_preorder(node.right)
            return(output)
        preorder = get_preorder(root)
        #now build the tree, always go right, and make left none
        temp = root
        for num in preorder[:-1]:
            temp.val = num
            temp.left = None
            temp.right = TreeNode()
            temp = temp.right
        temp.val = preorder[-1]

#using recursion directly
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        if we look at the problem again, we really just want to turn the binary tree into a right skewed tree
        suppose we use recursion to flatten the left and right subtrees  fromo a node,
        all we wouod have to is make the left child the right and current right child to the child of the left
        key take aways from articles:
            we assume that recursiono tranforms our left and right subtrees
            now lets call the pieces left,right, and left tail
            the idea woud be to make the lefttail.right = right
            and then make node.right = left
            and then noode.left = None
        once recursion does the hard work for us and flattens uot the subtrees, we will essentially get  the two linked lists and we need the tail end of the left one to attach it to the right one
        things we need:
            _node_ : current node
            _leftChild_: left child of current node
            _rightChild_: right child  of  current node
            _lefttail_ : tail node of flattend left subtree
            _righttail_ tail node  of flattened rightsubtre
        algo:
            define seperate funcionn for flattening out a subtree
            for each node, recursively flatten left and right, and store their left  and right  tails
            then we reconnect!
                leftTail.right = node.right
                 node.right = node.left
                 node.left = None
        '''
        def flattenTree(node):
            #base cases where there isn't  a ndoe
            if not node:
                return(None)
            #if leat, return the node
            if not node.left and not node.right:
                return(node)
            #get the subpoblems
            leftTail = flattenTree(node.left)
            rightTail = flattenTree(node.right)
            #if there is a left tree we need too reconnect
            if leftTail:
                leftTail.right = node.right
                node.right = node.left
                node.left = None
            #we nneed to return the right mode noode after we are done writing the new connetionns
            return rightTail if rightTail else leftTail
        flattenTree(root)

#using a stack
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        we can use a stack for this problem, but if we do, we need to control the states the nodes are in
        we push onto our stack (node,state)
        there are two states:
            START: means we have not processed the node yet and so we will try to process left if left else right
            END: means we are done processing this subtree and we can set up our reconnections as in the recursive solution
        #be sure to check out the pictures for this solution
        if the recursion state of a popped node is START, we will check if the node has left child or not
        if it does, we add the node back to the stack with the END recursion state and also add the left child with the START recursion state
        if there is no left child, then we add the right childonly with START
        if a node popped from the stack is END, it implies we finished processing left and that we have a tail node ready for a reconnection
        we reconnect, and push the right child back on to the stack
        finally for a node that is a leaf, we set out tailnode
        '''
        if not root: 
            return None
        #start is 0, end is 1
        tail = None
        stack = [(root,0)]
        while stack:
            curr, state = stack.pop()
            #if lead node, this is the new tail
            if not curr.left and not curr.right:
                tail = curr
                continue
            #if node is in start state, it means we NEED to process it
            if state == 0:
                #if the curr node has left, we add it to the stack AFTER adding the curr node again, but with end state
                if curr.left:
                    stack.append((curr,1))
                    stack.append((curr.left,0))
                #if only a right
                elif curr.right:
                    #append right and start
                    stack.append((curr.right,0))
            #otherwise the node is in END state and we need to reconnect
            else:
                #we processed this nodes left if it existed, else right
                right = curr.right
                #if there was a left child, there must have been a leaf node, and so we have set the tail node
                if tail:
                    #reconnect
                    tail.right = curr.right
                    curr.right = curr.left
                    curr.left = None
                    right = tail.right
                if right:
                    stack.append((right,0))


#great now do it without using extra space
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        the O(1) space is a Morris Traversal, infact, doing a morris traversal of tree modifies the structure of the tree such that it becomes right skewed
        recall preorder is node,left,right,the tree becomes threaded
        the entire left tree becomes placed betweent thr current node and its right
        algo:
            located the lat node in the left subtree, we go left once if we wan, then right all the way if we can
            NOTE: the last node of pre order tree can be found by moving right as many times from the root
            whenever we find a left subtree, we dispatch a runner to finds its lat node, then stitch together both ends of the left subtree into the right path of curr, then sever the left connection
            once that's doen we move curr right looking for the next left subtree
            when we can't go right, the tree should be flattened
        '''
        curr = root
        while curr:
            #if there is a left, go left once
            if curr.left:
                runner = curr.left
                #go right as far as we can
                while runner.right:
                    runner = runner.right
                #now we reconnect
                runner.right = curr.right
                curr.right = curr.left
                #sever
                curr.left = None
            #if there isn't a left, we can only go right
            curr = curr.right

#be sure to check these link out
#https://leetcode.com/problems/flatten-binary-tree-to-linked-list/discuss/1207642/JS-Python-Java-C%2B%2B-or-Simple-O(1)-Space-and-Recursive-Solutions-w-Explanation
#another way
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        there is also an O(1) space iterative that does not use a morris traversal, but at the cost of being O(N^2)
        in order to properly connect the linked list, we need to work from the bottom and work up
        which means we need revers pre order
        NOTE: reverse pre order is not the same as post order
        pre is  NLR
        post is LRN
        revere pre is RLN
        in order to complete this solution in O(1) space we wont be able to conviently backtrack, so we have to retreat all the way back up to the root each time we hit a leaft
        we'll want to first set up head and curr to keep track of the head of the linked list we're building and the current node we're visiting
        we'll know once we're finished head == root
        to follow the reverse pre order, we'll first attempt to go right then left
        since we're backtracking to root, we'll eventually run back inot the same node that weve set as head doing this
        To prevent this, we'll stop before moving to the head node and sever the connection.

        Now that we can't run into already-completed territory, 
        we can be confident that any leaf we move to must be the next value for head, 
        so we should connect it to the old head, update head, and reset back to the root.
        '''
        head = None
        curr = root
        while head != root:
            #reverse preorder, RLN
            if curr.right == head:
                curr.right = None
            if curr.left == head:
                curr.left = None
            if curr.right:
                curr = curr.right
            elif curr.left:
                curr = curr.left
            else:
                curr.right = head
                head = curr
                curr = root

#another recursino
#https://leetcode.com/problems/flatten-binary-tree-to-linked-list/discuss/1207839/Python-Intuitive-solution-explained
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        define a helper nodes, that returns the head and tail of a flattened subtree from a node for the lieft and right
        then we just have a few cases
        1. if there no left or no right, then just reuturn the node
        2.if there is no left, but a right, we make the connection for the righ sides node->b2 .. -> e2
        3. if there is left and not right, then node -> b1
        4. if there is  a left and a right, node -> b2...e1->b2
        '''
        def helper(node):
            if not node:
                return(None,None)
            leftHead,leftTail = helper(node.left)
            rightHead,rightTail = helper(node.right)
            node.left = None #sever the connection
            
            #if we don't have any tails
            if not leftTail and not rightTail:
                return(node,node)
            #if we only have a right tail
            if not leftTail and rightTail:
                node.right = rightHead
                return (node,rightTail) #make a new head and get the end
            #if we only have a left tail
            if leftTail and not rightTail:
                node.right = leftHead
                return (node,leftTail)
            #we have both a left and right
            node.right =leftHead
            leftTail.right = rightHead
            return (node,rightTail)
        
        helper(root)

############################
# Valid Number
#############################
class Solution:
    def isNumber(self, s: str) -> bool:
        '''
        there is just way too many edge cases for this one
        it's not a super hard problem, but the edge cases mess me up...
        lets just carefully read through the solution
        approach 1, follow the rules:
        intuitions:
            both decimals and intergers must conain at least 1
            if there is a + or - it should be first
            exponenets are optional, but if it does have one it must be after a decimal and followed by an integer
            a dot "." there should be only one
        rules:
            1. there must be at least one digit, lets keep seenDigit variable to indicate if we've seen one
            2. signs. if there is a sign, it should be at the font, OR jsut before the e
            3. exponents, there can be more than one, seenExpoenent, and it must appear after a decimal or an int, so if we have seen an exonenet we MUST have seen a digit
            4. dots: there cannot be more than one, and since integers are not allowed after an exponent, so there cannot be more than one decimal number. if we see a dot appear after an exponent, the number is not valid, because integers cannot have dots
        algo:
            1. declare three variables seenDigit, seenExp, and seenDot all to false
            2. pass the string
            3. if the char is a sign, check if it is either the the first char of the input or if the char before it is an exp, if not return False
            4. if char is a digit, set seen digit to False
            5. if char is exp, first check if we have already seen an exp OR if we have not yet seen a digit. if either is True, return False
                Otherwise, set seenExpo = True and seenDigit = false
                we reset seen digit because after and expoenent would could make a new int without a decimal
            6. if char is a dot, first check if we have already seen either a dot or an exp, if so return flase ors et seenDot = True
            7. if char is anything else return False
            8. at the end, return seenDigit (this is one reason why we have to reset seenDigit afte seeing an expo, otherwise an input like "21e" would be valid)
        FML, i'm never gonna make it to FAANG
        '''
        seenDigit = seenExp = seenDot = False
        for i,c in enumerate(s):
            #check digit
            if c.isdigit():
                seenDigit = True
            #check sign
            elif c in ["+","-"]:
                if i > 0 and s[i-1] != "e" and s[i-1] != "E":
                    return False #should be after e and E
            elif c in ["E","e"]:
                #if ive seen an exp alrady or haven't seen digit, its bunk
                if not seenDigit or  seenExp:
                    return False
                seenExp = True
                #newdigit
                seenDigit = False
            elif c == ".":
                if seenDot or seenExp: #if ive seen a dot already or it comes after an exp, we're also bunk
                    return False
                seenDot = True
            else:
                return False
        return seenDigit

#could also do try
class Solution:
    def isNumber(self, s: str) -> bool:
        '''
        could also try
        '''
        try:
            float(s)
        except:
            return False
        if 'nf' in s:
            return False
        return True

class Solution:
    def isNumber(self, s: str) -> bool:
        digit, dec, exp, symbol = False,False, False, False
        for c in s:
            if c in "0123456789":
                digit = True
            elif c in "+-":
                if digit or symbol or dec:
                    return False
                else:
                    symbol = True
            elif c in "Ee":
                if not digit or exp:
                    return False
                else:
                    exp = True
                    digit = False
                    symbol = False
                    dec = False
            elif c == ".":
                if dec or e:
                    return False
                else:
                    dec = True
            else:
                return False
        
        return digit
        

#DFA
class Solution:
    def isNumber(self, s: str) -> bool:
        '''
        just a brief review
        deterministic finite automatons are  very smilar to trees
        differences:
            DFS can represent and infinite sequence of inputs
            can loop back between states
            it is directed
        coming up with the graph is pretty tough,
        but once you have the states, go through thes tring and get the currnt state
        the current state must represent a valid string, and its its not return False
        '''
        dfa = [
            {"digit": 1, "sign": 2, "dot": 3},
            {"digit": 1, "dot": 4, "exponent": 5},
            {"digit": 1, "dot": 3},
            {"digit": 4},
            {"digit": 4, "exponent": 5},
            {"sign": 6, "digit": 7},
            {"digit": 7},
            {"digit": 7}
        ]
        
        current_state = 0
        for c in s:
            if c.isdigit():
                group = "digit"
            elif c in ["+", "-"]:
                group = "sign"
            elif c in ["e", "E"]:
                group = "exponent"
            elif c == ".":
                group = "dot"
            else:
                return False

            if group not in dfa[current_state]:
                return False
            
            current_state = dfa[current_state][group]

#######################
# Minimum Knight Moves
#######################
#niceeee
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        '''
        this is just bfs
        can only go in increments and combinations of +-1 or +-2
        the chess board is infinite, so we don't need to check for in bounds
        just check seen
        '''
        moves = [[2,1],[1,2],[-2,1],[-1,2],[-2,-1],[-1,-2],[1,-2],[2,-1]]
        seen = set()
        seen.add((0,0))
        q = deque([((0,0),0)]) #tuple (coord),moves

        while q:
            coord,steps = q.popleft()
            if coord == (x,y):
                return steps
            for dx,dy in moves:
                new_x = coord[0] + dx
                new_y = coord[1] + dy
                if (new_x,new_y) not in seen:
                    seen.add((new_x,new_y))
                    q.append(((new_x,new_y),steps+1))

#bidrectional bfs
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        '''
        another way would be to use bi-directional BFS with pruning
        for bidiretional BFS we start from boths ends, and check whthere the curent cell is found in the opposite set 
        we use two seen sets and two qs
        we can also prunce the search space taking into consideration symmetry
        (x, y), (-x, y), (x, -y), (-x, -y)
        these will all have the same results
        pruning, instead of searching for all 8 positions from where the knight currently is
        we can bound the search spaces from [-1,abs(x)+2]
        i understand it doens't make sense going out of bounds, so we can limit it to the first quadrant
        but why -1???
        '''
        x,y = abs(x),abs(y) #symmetry
        q_origin = deque([(0,0,0)]) #(x,y,moves)
        q_target = deque([(0,0,0)])
        moves = [(1,2),(2,1),(1,-2),(2,-1),(-1,2),(-2,1),(-1,-2),(-2,-1)]
        
        #seen
        seen_origin = {(0,0):0} 
        seen_target = {(x,y):0} #whether weve seen it or not and the min steps
        
        while True:
            #origin first
            origin_x,origin_y,origin_steps = q_origin.popleft()
            if (origin_x,origin_y) in seen_target:
                return seen_target[(origin_x,origin_y)] + origin_steps
            #target next
            target_x,target_y,target_steps = q_target.popleft()
            if (target_x,target_y) in seen_origin:
                return seen_origin[(target_x,target_y)] + target_steps
            for dx,dy in moves:
                #generate neighbors for origin and target
                new_origin_x = origin_x + dx
                new_origin_y = origin_y + dy
                if (new_origin_x,new_origin_y) not in seen_origin and -1 <= new_origin_x <= x+2 and -1 <= new_origin_y <= y+2:
                    q_origin.append((new_origin_x,new_origin_y,origin_steps+1))
                    seen_origin[(new_origin_x,new_origin_y)] = origin_steps + 1
                
                #target side
                new_target_x = target_x + dx
                new_target_y = target_y + dy
                if (new_target_x,new_target_y) not in seen_target and -1 <= new_target_x <= x+2 and -1 <= new_target_y <= y+2:
                    q_target.append((new_target_x,new_target_y,target_steps+1))
                    seen_target[(new_target_x,new_target_y)] = target_steps + 1

#dfs with memo
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        '''
        we can take advantge of symmetery and just abs value x and y
        since we are restricted to the first quadrant
        also with DFS approach, we start backwards and try to reach the origin
        since we are starting backwards, there are only two directions that will bring us closer to the origin
        the moves [-1,-2] and [-2,-1]
        the rest take as away
        we define the immediate neighborhood of the origin as points of the form (x,y) where x + y <= 2
        if we can get to a point whose x,y repsentation is that we are done
        our dfs call then becomes dfs(x,y) = min(dfs(abs(x-2),abs(y-1)),dfs(abs(x-1),abs(y-2))) + 1
        base cases:
            when x = 0 or y = 0, we have reached the origin return 0
            when x+y == 2, we are at an immeidate neighbor, so it takes two more steps 
        '''
        memo = {}
        def dfs(x,y):
            if x+y == 0:
                return 0
            elif x + y == 2:
                return 2
            elif (x,y) in memo:
                return memo[(x,y)]
            else:
                first_step = dfs(abs(x-1),abs(y-2))
                second_step = dfs(abs(x-2),abs(y-1))
                res = min(first_step,second_step) + 1
                memo[(x,y)] = res
                return memo[(x,y)]
        
        return dfs(abs(x),abs(y))

#######################
# Binary Tree Cameras
########################
#good idea
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        '''
        a camera keeps track of the parent, itselt and its children
        if when we get to a node, we came from another one, and at this node, we have at least one child we need a camera
        keep a variable cameFrom, initally set to False, when we invoke we set that to True
        and check of cameFrom is tree and node has children
        '''
        self.cameras = 0
        def dfs(node,cameFrom = False):
            if not node:
                return 
            if (node.left or node.right) and cameFrom == True:
                self.cameras += 1
            dfs(node.left, cameFrom = True)
            dfs(node.right, cameFrom = True)
        dfs(root)

class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        '''
        we define three states:
        0, node has a camera
        1, node is covered by a camera
        2, node needs covering
        '''
        self.cameras = 0
        
        def dfs(node):
            #base case, if there is not a node, its already coverd
            if not node:
                return 1
            
            #we need to see the left and right results from a node
            left = dfs(node.left)
            right = dfs(node.right)
            
            #if any of the node's children is NOT covered, we need to cover it
            #after covering, place a camera, and return the that this node has been placed by a camera
            if left == 2 or right == 2:
                self.cameras += 1
                return 0
            #if any of the children are already covered, return that it is covered
            if left == 0 or right == 0:
                return 1
            #if none of the children is covering the current node, then askits parent to cover, which is jsut the node we are already one
            else:
                return 2
        
        #now when we invoke, if the root is not covered, we need to place a camera at the root
        if dfs(root) == 2:
            self.cameras += 1
        return self.cameras

class Solution:
    def minCameraCover(self, root: TreeNode) -> int:
        self.cameras = 0
        def dfs(node):
            #this answers if this nodes is monitored and has a camera 
            #camera, monitored
            #base case, leaf, it does not need a camera but is parent can monitor it
            if not node:
                return False, True
            c1,m1 = dfs(node.left)
            c2,m2 = dfs(node.right)
            
            #initallay there is nothing
            camera, monitor = False, False
            if c1 or c2:
                monitor = True
            if not m1 or not m2:
                camera = True
                self.cameras += 1
                monitor = True
            return camera, monitor
        
        c,m = dfs(root)
        if not m:
            return self.cameras + 1
        return self.cameras
        
#######################
# Longest String Chain
#######################
#mmmmmm close one 46/78
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        '''
        well the order has to be increasing
        i.e a predecessor needs to be smaller then the next word
        what if i first sort by length? but do this in reverse
        then try to make a chain with each ending going backwards
        
        '''
        longest = 0
        #sort
        words.sort(key = lambda x: len(x), reverse = True)
        N = len(words)
        for i in range(N):
            curr_longest = 1
            curr_word = words[i]
            for j in range(i+1,N):
                cand_word = words[j]
                if len(curr_word) - len(cand_word) == 1:
                    curr_word = cand_word
                    curr_longest += 1
            longest = max(longest,curr_longest)
        
        return longest

 #reucrsive, but still not correct
 #FUCKKKK
 class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        '''
        welp ita recursion....
        still sort by increasing length
        we need take the  current  longest word, and check if the next word is smaller by at least one
        dfs function will take in a poointer to a word
        base case is when pointer moves through the whole list we return the max chain
        omg, it must be a  new  letter...
        '''
        words.sort(key = lambda x: len(x), reverse = True)
        N = len(words)
        
        self.longest = 0
        
        def recurse(idx,chain):
            if idx+1 == N:
                self.longest = max(self.longest,chain)
                return
            if len(words[idx]) - len(words[idx+1]) == 1:
                recurse(idx+1,chain+1)
            else:
                recurse(idx+1,chain)

        recurse(0,1)
        return self.longest

#god dammit
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        '''
        predecessor must be of shorter length than the next word, and when we place the next char, 
        it must not mess with the order of the words in word2
        therefore it is possible for particular word to have more than one predecessor
        but we want to find the largest chain
        we are essentailly finding the longest path in a trie, sort of
        because we can have repeate calcs, we use memo, but for different reason, an invocation of the function (with diffeent inputs) could have the same answer as another invocation with the same inputs
        whenever we encounter a new word, we find all possible sequences with this words as the last words in the sequence,
        then store the length of the longest possible seqeucne that ends with this word
        STARTINF FOMR REVESE
        our map will store the ending word in the chain as a key and the longest chain so far
        so the next time we get this words, we can retreive from our memo and keep getting max (no need to call it again)
        '''
        memo = {}
        words = set(words) #fast lookup
        
        def dfs(currWord):
            #retrieve current max answer
            if currWord in memo:
                return memo[currWord]
            #otherwise recurse and get the max
            
            #initial maxlength
            maxLength = 1
            #now create all possible strings taking one char out at a time
            for i in range(len(currWord)):
                #delete and check
                temp = currWord[:i]+currWord[i+1:]
                if temp in words:
                    currLength = 1 + dfs(temp)
                    maxLength = max(maxLength,currLength)
            
            memo[(currWord)] = maxLength
            return maxLength
        
        ans = 0
        for word in words:
            ans = max(ans,dfs(word))
        return ans

class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        '''
        dp version of this
        we sort the words in ascending length and try to make the longest chain
        as we are making a chain, whenver we find a word, we update the longest chain made ending in that word into a hash map, for easy retrieval
        then we just max on the fly
        '''
        mapp = {}
        words.sort(key = lambda x: len(x))
        
        longest = 1 #it will always be 1
        for word in words:
            currLength = 1
            for i in range(len(word)):
                #get a predecessor, if it exisits
                pred = word[:i]+word[i+1:]
                if pred in mapp:
                    prevLength = mapp[pred]
                else:
                    prevLength = 0
                #update for the pred
                currLength = max(currLength,prevLength+1)
            
            mapp[word] = currLength
            longest = max(longest,currLength)
        
        return longest

#another recursive approach
class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        words = set(words)
        memo = {}
        
        def dfs(word):
            if word not in words:
                return 0
            if word in memo:
                return memo[word]
            max_length = 0
            for i in range(len(word)):
                pred = word[:i]+word[i+1:]
                max_length = max(max_length,dfs(pred)+1)
            
            memo[word] = max_length
            return max_length
        
        LONGEST = 0
        for word in words:
            LONGEST = max(LONGEST,dfs(word))
        return LONGEST

###############################
# Find Duplicate File in System
###############################
#this problem is very practical
class Solution:
    def findDuplicate(self, paths: List[str]) -> List[List[str]]:
        '''
        what if i make a mapp by the content
        mapp would be 'content' :list of files that have this content
        '''
        mapp = defaultdict(list)
        for path in paths:
            #need to slplit content and and file name
            split = path.split(" ")
            currDir = split[0]
            files = split[1:]
            for f in files:
                #get content and file name
                split2 = f.split("(")
                fileName = split2[0]
                content = split2[1][:-1]
                #join filName with curDir
                mapp[content].append("/".join([currDir,fileName]))

        results = []
        for k,v in mapp.items():
            #only want groups
            if len(v) > 1:
                results.append(v)
        return results

################################
# Minimum Moves to Equal Array Elements II
################################
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        '''
        so this doesnt always give the minimum number
        bring each element to the target of sum(nums) / len(nums)
        '''
        SUM = sum(nums)
        target = SUM / len(nums)
        return int((sum([abs(num - target) for num in nums])))

#TLE
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        '''
        ok wait a minute,
        brute force, generate all arrays that have the same element repeated n times
        i don't need to generate the array, just the number to bring it up or down to
        then take the diff for each one, sum them up, and min on the fly
        don't check all of them just check in the range between min and max
        
        '''
        moves = float('inf')
        MIN = min(nums)
        MAX = max(nums)
        for cand in range(MIN,MAX+1):
            curr_moves = sum([abs(num - cand) for num in nums])
            moves = min(curr_moves,moves)
        return moves

class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        '''
        better brute foce, we only need to consider when bring an element up or down
        all k in nums
        KEY TAKEAWAY:
            when bring an element in the array down to a target,
            we only need to consider the numbers in nums
        the proof is actually very tricky, be sure to take a look at this 
        proof takeaway:
            the number of moves required to settle to some abritray number present inteh array is less less than the number of moves required to settle down to some abritray number
            and we want the min, if the its always less, then we don't need to check all numbers in the range MIN to MAX
        '''
        cands = set(nums)
        moves = float('inf')
        for cand in cands:
            curr_moves = sum([abs(num - cand) for num in nums])
            moves = min(curr_moves,moves)
        return moves

#using sorting and the counting
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        '''
        using sorting
        when we sort, we can say that the number of moves requirted to "RAISE" all elements smaller than k to k is given by
        (k*countBeforek) - (sumBeforek)
        and the num of moves required to bring all elements greater than k to k is given by
        (sumAfterK) - (k*countBeforek)
        and so the number of move in total would be the sum of the two moves (less and greater)
        
        lets say that the index of the element corresponding to the element k by index s
        insteadof passing over the array for cclcuating sumBeforek and sumAfterl, we can keep on calcuint hem while traversing the array since the array is sorted!
        we first get the total of the array
        then star off with sumBeforeK = 0 and sum after k = total
        to get sumBeforek, we just add the element at i to the previous sumbefore k
        to calculation sumAfter k, we subtract the element k from pev sumAfter K
        then and them up and min on the fly
        '''
        nums.sort()
        moves = float('inf')
        N = len(nums)
        
        sumBefore_k = 0
        sumAfter_k = sum(nums)
        
        for i in range(N):
            curr_moves = (nums[i]*i - sumBefore_k) + (sumAfter_k - (nums[i]*(N-i)))
            moves = min(moves,curr_moves)
            sumBefore_k += nums[i] #this bit here is finicky, we add to sumbefore K the current i, but decretment after
            sumAfter_k -= nums[i]
            
        return moves

#turns outs our k is just the median!
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        '''
        it turns out that finding the number k to which we bring an element in nums to is just the median of the array
        recall from approach 3:
        we have the two terms:
        1. k*CountBeforeK - sumBeforek
        2. sumAfterK - k*CountAfterK
        we then add these up, for each candidate k in the array, and take the min
        to find the minimum, we can take the deriative of this expression with respect k
        numMoves(k)= k*CountBeforeK - sumBeforek + sumAfterK - k*CountAfterK
        countBeforeK - sumBeforeK = 0
        this property is only satisfied using the median
        half of the elements are less than k, half greater than k
        '''
        #sort
        nums.sort()
        N = len(nums)
        #find the median
        median = nums[int(N /2)]
        moves = 0
        for num in nums:
            moves += abs(num - median)
        return moves

#without loooking for median
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        '''
        instead of seeking the median, we can take advantage of the sorted properties of the array
        if we have our maxEle and minELe
        we know we to bring this down to k
        so moves = maxEle - k + k - minEle
        moves = maxEle - minELE
        
        moves = \sum_{i=0}^{\ceil(n/2) -1} abs(nums[n-i] - nums[i])
        '''
        nums.sort()
        N = len(nums)
        moves = 0
        left,right = 0, N-1
        while left < right:
            moves += nums[right] - nums[left]
            left += 1
            right -= 1
        return moves

#using quick select
class Solution:
    def minMoves2(self, nums: List[int]) -> int:
        '''
        we can take an idea from quick sort using quick select
        we use the paritition function, to divide the array into two parts where the left part is less than the pivot and the right is greater
        if when paritiontions our pivot ends up at the middle index, call it k, we know that k is the median
        then we're done
        
        '''
        def partition(nums,left,right):
            pivot = nums[right] #pivot is right
            i = left
            for j in range(left,right+1):
                if nums[j] < pivot:
                    #swap
                    nums[j],nums[i] = nums[i],nums[j]
                    i += 1
            return i
        
        def select(nums,left,right,mid):
            if left == right:
                return nums[left]
            
            potential_mid = partition(nums,left,right)
            if potential_mid == mid:
                return nums[mid] #found
            elif mid < potential_mid:
                return select(nums,left,potential_mid-1,mid)
            else:
                return select(nums, potential_mid+1,right,mid)
            
        median = select(nums,0,len(nums)-1,(len(nums)//2))
        moves = 0
        for num in nums:
            moves += abs(num - median)
        return moves

###########################
# Binary Tree Level Order Traversal
###########################
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        '''
        intuitively just use bfs
        '''
        if not root:
            return []
        levels = []
        q = deque([root])
        
        while q:
            curr_level = []
            N = len(q)
            for i in range(N):
                curr_node = q.popleft()
                curr_level.append(curr_node.val)
                if curr_node.left:
                    q.append(curr_node.left)
                if curr_node.right:
                    q.append(curr_node.right)
            levels.append(curr_level)
        
        return levels

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        '''
        recusrively, i could dump mapp each level to hashmapp, then add the nodes from left to right on each level, but then you would have to order the levels again
        well just take the max along the way
        '''
        if not root:
            return []
        self.max_lvl = 0
        mapp = defaultdict(list)
        
        def dfs(node,depth):
            if not node:
                return
            self.max_lvl = max(self.max_lvl,depth)
            mapp[depth].append(node.val)
            dfs(node.left,depth+1)
            dfs(node.right,depth+1)
        
        dfs(root,0)
        res = []
        for i in range(self.max_lvl+1):
            res.append(mapp[i])
        return res

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        '''
        dfs instead of mapping we can just use a list
        and add a node to the left at its' index depth
        if we have reached the point  where our depth meets the  length leves, we need to add a new one
        
        '''
        levels = []
        if not root:
            return levels
        def dfs(node,depth):
            if not node:
                return
            #add  new level when
            if len(levels) == depth:
                levels.append([])
            levels[depth].append(node.val)
            dfs(node.left,depth+1)
            dfs(node.right,depth+1)
        
        dfs(root,0)
        return levels

#stack
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        '''
        dfs instead of mapping we can just use a list
        and add a node to the left at its' index depth
        if we have reached the point  where our depth meets the  length leves, we need to add a new one
        
        '''
        levels = []
        if not root:
            return levels
        
        stack = [(root,0)]
        while stack:
            curr,depth = stack.pop()
            if not curr:
                continue
            if len(levels) == depth:
                levels.append([])
            levels[depth].append(curr.val)
            stack.append((curr.right,depth+1))
            stack.append((curr.left,depth+1))
        
        return levels

#########################
# Find and Replace Pattern
#########################
class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        '''
        notes, a word matches the pattern if there exusts a permutation of letters p so that after
        replacing x in the pattern with p(x), we get the desired word
        len(words[i]) == len(pattern)
        i need to first make a bijection of the pattern the words[i]
        i need two maps!
        '''
        def helper(word,pattern):
            N = len(pattern)
            pattern_word = {}
            word_pattern = {}
            for i in range(N):
                if pattern[i] not in pattern_word:
                    pattern_word[pattern[i]] = word[i]
                if word[i] not in word_pattern:
                    word_pattern[word[i]] = pattern[i]
                #get the bijections from the word and pattern and see if they match
                if (pattern_word[pattern[i]],word_pattern[word[i]]) != (word[i],pattern[i]):
                    return False
            return True
                    
        
        matches = []
        N = len(words)
        for i in range(N):
            if helper(words[i],pattern) == True:
                matches.append(words[i])
        return matches

class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        '''
        there is also a one mapp solution
        recall we mapped pattern to char and word to pattern, and then checked for the bijection
        however instead of keeping track of the reverse map m2, we could simply make sure that
        every value in map1 in the codomain is reached at least one
        
        '''
        def match(word,pattern):
            word_pattern = {}
            N = len(pattern)
            for i in range(N):
                if word[i] not in word_pattern:
                    word_pattern[word[i]] = pattern[i]
                if word_pattern[word[i]] != pattern[i]:
                    return False
            #now check valeus of mapp to see if we've mapped to repeats!
            #think of the case with the word ccc and pattern abb
            seen = set()
            for ch in word_pattern.values():
                if ch in seen:
                    return False
                seen.add(ch)
            return True
        
        #lets use the filter function
        return filter(lambda word: match(word,pattern),words)

#there's also this very cheeky solutino here
b = pattern
def is_iso(a):
    return len(a) == len(b) and len(set(a)) == len(set(b)) == len(set(zip(a, b)))
return filter(is_iso, words)

#another one, just translating
class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        '''
        try converting them to numbers
        '''
        def get_pattern(s):
            lookup = {}
            output = []
            i = 0
            for ch in s:
                if ch in lookup:
                    output.append(lookup[ch])
                else:
                    i += 1
                    lookup[ch] = i
                    output.append(i)
            return output
        
        #now just compare lists
        matches = []
        p = get_pattern(pattern)
        for word in words:
            if get_pattern(word) == p:
                matches.append(word)
        return matches

###############################
#  N-Queens
################################
#its about time....
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        '''
        brute force would be to enumerate all possible board states with N queens
        for an N x N board, and Q queens
        N^2! times Q queens
        but Q can be up to N^2
        i still don't get whys its O(N^{2N})
        its just N squared for N queens, with N being less than equal N squared
        NOTE, THIS IS FOR NOT PLACING 1 to N queens
        THE PROBLEMS ASK PLACING N queens on an N x N chess board
        inution, if we place a queen one a time, we can eliminate spots on the board that causes conflict
        EXAMPLE:
            if i place queesn at (0,0),(0,1), i can't out queens anywhere else because the first two queens attack each other
            this is a good example of backtracking
        given a board state, and a possible placement for a qeeun, we need smart way to determin where the next queen can be palce
        queen cannot be places in same column, row, diagonal of another queen
        KEY:
            backtrack function take in another board state
        backtrack function:
            only place 1 queen per row, and we place one queen during each call
            whenever we place a queen, we'll move onto the next row calling backtrack with row + 1
            we place only 1 queen per column, and in the same invoke backtrack
        diagonal properties:
            TRICK: remember each i,j elemnt on the sam diag has the same (i,j) difference
            anti-diags have the same (i+j)
        every time we place a queen we should calculate the diagonal and anti-diagonal it belongs to
        we can use a set to show what (i,j) positions we cannot put a queen in
        algo:
            backtrack function needs the following arguments:
                first param is the row we're going to place the next queen,
                followed by three sets for rows,cols,diags that cause conflict for the next queen to placed
                remember each invocation places a queen!
                finally, we need to store the board in order to return a solution - paramter passing
            
            1. if the current row we are considering is equal to n, then have a solution, and we can add the current board to our results -> helper function here
            2. move through the board, at each columns attemp to palce a queen
                * it is here we check if we can place queen using our sets, rember cannot add to row, col, diag, anti diag that another queen can reach
                * if we cant place a queen, skip
            3. if we can place a queen, place, update out sets, board, and call function on row + 1
            4. The function call made in step 3 explores all valid board states with the queen we placed in step 2. 
            Since we're done exploring that path, backtrack by removing the queen from the square - this includes removing the values we added to our sets on top of removing the "Q" from the board.


        '''
        solutions = []
        empty_board = [["."]*n for _ in range(n)]
        
        #create board solution helper, to get board in wirte formate, list is board, and eachelemen in list a list, with rows as strings
        def create_board(state):
            board = []
            for row in state:
                board.append("".join(row))
            return board
        
        def back_track(row,diags,anti_diags,cols,state):
            #base case, when we have n queens
            if row == n:
                solutions.append(create_board(state))
                return
            #now we traverse column wise (placing a queen elimnates the next row below it, and we are backtracking)
            for col in range(n):
                #get diags and anti diags
                curr_diag = row - col
                anti_diag = row + col
                #if queen is placeable
                if col in cols or curr_diag in diags or anti_diag in anti_diags:
                    continue
                #otheriwse continue to place queen
                #mark sets
                cols.add(col)
                diags.add(curr_diag)
                anti_diags.add(anti_diag)
                #mark the current state
                state[row][col] = "Q"
                
                #recurse
                back_track(row+1,diags,anti_diags,cols,state)
                
                #FUCKIGN BACKTRACK, remove the last placed queen to examin a new sate
                cols.remove(col)
                diags.remove(curr_diag)
                anti_diags.remove(anti_diag)
                state[row][col] = "."
        back_track(0,set(),set(),set(),empty_board)
        return solutions

#typical dfs, with constrain programming
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        '''
        we actually don't need to back track
        we know that if place a queen at row i
        i cannot place another queeen at row i
        same goes for col j
        whenver we place  a queen keep track of its col index
        i wonder why we can't use sets, 
        i mean for this problems its the sizes are small so its ok
        '''
        q_col_idx = []
        def dfs(queens,diags,anti_diags):
            row = len(queens)
            if row == n:
                q_col_idx.append(queens) #these are list of indices (col) for each row in order
                return
            for col in range(n):
                if col not in queens and (row - col) not in diags and (row + col) not in anti_diags:
                    dfs(queens+[col],diags+[row-col],anti_diags+ [row+col])
        
        dfs([],[],[])
        return [ ["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in q_col_idx]

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        '''
        just a review of this problem instead of finding the excluding diags, actuall walk
        when we recurse, mark spors we cannot go down into
        '''
        output = []
        
        def dfs(solution,excluded):
            #solution is a list of strings
            row = len(solution)
            #we can still go down the board
            if row < n:
                #dfs
                #we are always goind down a row, so go across cols
                for col in range(n):
                    if (row,col) in excluded:
                        continue
                    #we can place a queen at this col
                    new_excluded = set()
                    curr_row_solution = "."*col+"Q"+"."*(n-col-1) #place a queen
                    #now go down rows
                    for r in range(row,n):
                        new_excluded.add((r,col))
                    #down daig right
                    row_diag = row_anti_diag = row
                    col_diag = col_anti_diag = col
                    
                    while col_diag < n:
                        row_diag += 1
                        col_diag += 1
                        new_excluded.add((row_diag,col_diag))
                        
                    #anti diag
                    while col_anti_diag > 0:
                        row_anti_diag += 1
                        col_anti_diag -= 1
                        new_excluded.add((row_anti_diag,col_anti_diag))
                    
                    dfs(solution+[curr_row_solution],excluded | new_excluded)
                    
            else:
                output.append(solution)
        dfs([],set())
        return output

################################
# Design Tic-Tac-Toe
################################
class TicTacToe:

    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        '''
        a game is one when there are n X's or O's in any of the rows,cols,diag, or anti diag
        just do the N^2 solution first
        i can amend the board with an extra row showing the sum of a row
        when I place an X, increase sum +=1,when i place a zero -0
        if on any row, cols or diag we have -n or n, we have a dinner
        '''
        self.n = n
        self.row_sums = [0]*n
        self.col_sums = [0]*n
        self.diags = 0
        self.anti_diags = 0
        self.spots = n*n
        
        

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """
        n = self.n
        #moves are always unique, so need to check if spot is occupied
        if player == 1:
            #on diag
            if row - col == 0:
                self.diags += 1
            #on anti diags
            if row + col == n -1:
                self.anti_diags += 1
            #rows
            self.row_sums[row] += 1
            #cols
            self.col_sums[col] += 1
            self.spots -= 1
        
        if player == 2:
            #on diag
            if row - col == 0:
                self.diags -= 1
            #on anti diags
            if row + col == n - 1:
                self.anti_diags -= 1
            #rows
            self.row_sums[row] -= 1
            #cols
            self.col_sums[col] -= 1
            self.spots -= 1

        #now we just check for winner status each time
        if self.diags == n:
            return 1
        if self.diags == -n:
            return 2
        if self.anti_diags == n:
            return 1
        if self.anti_diags == -n:
            return 2
        #check row sums
        for r in self.row_sums:
            if r == n:
                return 1
            if r == -n:
                return 2
        #check col sums
        for c in self.col_sums:
            if c == n:
                return 1
            if c == -n:
                return 2
        #draw
        if self.spots == 0:
            return 0
        else:
            return 0

#brute force
class TicTacToe:
    '''
    lets just review the brute force
    make the board, make the moves, make the checks!
    '''

    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        self.board = [[0]*n for _ in range(n)]
        self.n = n
        

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """
        #markit
        self.board[row][col] = player
        #check it
        if self.checkDiags(player) or self.checkAntiDiags(player) or self.checkCol(col,player) or self.checkRow(row,player):
            return player
        return 0
    def checkDiags(self,player:int) -> bool:
        for i in range(self.n):
            if self.board[i][i] != player:
                return False
        return True
    
    def checkAntiDiags(self,player:int) -> bool:
        for i in range(self.n):
            if self.board[i][self.n-i-1] != player:
                return False
        return True
    
    def checkCol(self,col: int,player:int) -> bool:
        for i in range(self.n):
            if self.board[i][col] != player:
                return False
        return True

    def checkRow(self,row:int,player:int) -> bool:
        for i in range(self.n):
            if self.board[row][i] != player:
                return False
        return True

#optimized
class TicTacToe:

    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        self.n = n
        self.rows = [0]*n
        self.cols = [0]*n
        self.diags = 0
        self.antiDiags =0
        

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """
        '''
        the trick was to conver player number to 1 or -1
        '''
        playerPce = 1 if player == 1 else -1
        #increment sums
        self.rows[row] += playerPce
        self.cols[col] += playerPce
        #increment diags
        if row == col:
            self.diags += playerPce
        #increment anti diags
        if row + col == self.n-1:
            self.antiDiags +=  playerPce
        #check if winner
        if abs(self.rows[row]) == self.n or abs(self.cols[col]) == self.n or abs(self.diags) == self.n or abs(self.antiDiags) == self.n:
            return player
    
        return 0

#################################
#943. Find the Shortest Superstring
####################################
#evidently this was a variant of the traveline salesman problem
'''
we have to put the words into a row, where each word "may" overlap the previous words
it is also sufficient to maximize the total overlap of the woords
say we have put some words down in our row, ending with word A[i]
now when we put down a word, A[j] as the next word, where j hasn't been put down yet
the overlap increase by overlap(A[i],A[j])
we can model the recurrence using dynamic programming
FUNCTION:
let dp(mask,i) be the total overlap after putting some words  down, represented by a bit mask  mask
for which A[i] was the last word put down? WTF?F?F
then the key recurison becomes dp(mask^{1<<j},j) = max(overlap(A[i],A[j]) + dp(mask,i)
where the jth bit is not  set in mask and i ranges over all bits set in  mask
j is not set, i is set

of course, this only tells us what the maximum overlap is for each set of words
we also need to remember each choice along the  way  (i.e the specific i that made dp(mask^(1<<j)),j)
achive a minimum

algo,
	1. precompute overlap(A[i],A[j]) for  all possible  i,j
	2. calculate dp[mask][i], keeping track of the partent ,i, for each j as described  above
	3. reconstruct the answer using parent information


'''
class Solution:
    def shortestSuperstring(self, words: List[str]) -> str:
        N = len(words)
        #first find the overlap
        #overlaps is N by N, overlap with i and j word
        #if an overlap exists between word i and word j, this marks the size of the overlap
        
        overlap = [[0]*N for _ in range(N)]
        for i,x in enumerate(words):
            for j,y in enumerate(words):
                if i!=j:
                    #we can only compare chars for the smallest lenght of the words
                    for ans in range(min(len(x),len(y)),-1,-1):
                        if x.endswith(y[:ans]):
                            #print(x,y,y[:ans],ans)
                            overlap[i][j] = ans
                            break
        #dp[mask][i] answers the question, what is the most overlap with mask ending with the ith element
        #for a size N, there are 1<< N maskes
        dp = [[0]*N for _ in range(1<<N)]
        parent = [[None]*N for _ in range(1 << N)]
        #go through all masks
        for mask in range(1,1<<N):
            #each bit in the mask
            for bit in range(N):
                #get the bit
                if (mask >> bit) & 1:
                    #try to find dp[mask][bit],previously we had a collection of items represented by pmask
                    pmask = mask ^ (1 << bit)
                    if pmask == 0:
                        continue
                    for i in range(N):
                        if (pmask >> i) & 1:
                            #for each bit i in pmask, calculate the value if we ended with word i, then added word bit
                            value = dp[mask][i] + overlap[i][bit]
                            if value > dp[mask][bit]:
                                dp[mask][bit] = value
                                parent[mask][bit] = i
        
        #answer will have length sum(len(A[i] for i )-mas(dp[-1]))
        perm = []
        mask = (1 << N) - 1
        i = max(range(N), key = dp[-1].__getitem__)
        while i is not None:
            perm.append(i)
            mask, i = mask ^ (1<<i), parent[mask][i]
        
        #reverse path to get fowards directions, and all remainin words
        perm = perm[::-1]
        seen = [False]*N
        for x in perm: 
            seen[x] = True
        perm.extend([i for i in range(N) if not seen[i]])
        
        #reconstruct answer given perm - wrods indicies in left to right order
        ans = [A[perm[0]]]
        for i in xrange(1, len(perm)):
            overlap = overlaps[perm[i-1]][perm[i]]
            ans.append(A[perm[i]][overlap:])

        return "".join(ans)

class Solution:
    def shortestSuperstring(self, words: List[str]) -> str:
        '''
        we can think of this as the traveling salesman problem
        each string is a node and we want to find the minimum distance that reaches every nodes
        in the NP hard case, we would enumerate all possible orders so n! states
        steps: 
            1. create strucute showing overlap between each i and j words, thiss tores the minimum overkap, to save space
        '''
        #create overlaps dp array
        N = len(words)
        cost = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                for k in range(min(len(words[i]),len(words[j])),-1,-1):
                    #we check if there is overlap from the end of word i to the beginning of word j
                    if words[i][-k:] == words[j][:k]:
                        cost[i][j] = k
                        break #we want the minium overlap
        
        #how do we want to the the miniumum superstring
        #brute force would be to enumerate all possible paths for every i,j pair, create the string, and return the string with the min length, this is why is like the TSP problem
        #use bit maskt to represetns state
        #mask represents the order in which we add word to string
        #better yet, 1 represents the word we 'just added'
        #mask dp[1011010] = [len(str),"string so far", 0,0,0]
        dp = [[(float('inf'),"")]*N for _ in range(1<<N)]
        #base cases for each node
        for i in range(N):
            dp[1 << i][i] = (len(words[i]),words[i])
        
        #now we need to find the paths
        for bitmask in range(1<<N): #basically all possibile path ordergins, this is just subset generation
            bits = [j for j in range(N) if bitmask & (1 << j)]
            #bits represents all possible ordering of adding to super string
            for add,src in permutations(bits,2):
                cand = dp[bitmask^(1<<add)][src][1] + words[add][cost[src][add]:]
                dp[bitmask][add] = min(dp[bitmask][add],(len(cand),cand))
        return(min(dp[-1])[1])

#another good solution
#https://leetcode.com/problems/find-the-shortest-superstring/discuss/195077/Clean-python-DP-with-explanations
class Solution:
    def shortestSuperstring(self, A):
        """
        :type A: List[str]
        :rtype: str
        """
        # construct a directed graph
        #   node i => A[i]
        #   weights are represented as an adjacency matrix:
        #   shared[i][j] => length saved by concatenating A[i] and A[j]
        n = len(A)
        shared = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                for k in range(min(len(A[i]), len(A[j])), -1, -1):
                    if A[i][-k:] == A[j][:k]:
                        shared[i][j] = k
                        break

        # The problem becomes finding the shortest path that visits all nodes exactly once.
        # Brute force DFS would take O(n!) time.
        # A DP solution costs O(n^2 2^n) time.
        # 
        # Let's consider integer from 0 to 2^n - 1. 
        # Each i contains 0-n 1 bits. Hence each i selects a unique set of strings in A.
        # Let's denote set(i) => {A[j] | j-th bit of i is 1}
        # dp[i][k] => shortest superstring of set(i) ending with A[k]
        #
        # e.g. 
        #   if i = 6 i.e. 110 in binary. dp[6][k] considers superstring of A[2] and A[1].
        #   dp[6][1] => the shortest superstring of {A[2], A[1]} ending with A[1].
        #   For this simple case dp[6][1] = concatenate(A[2], A[1])
        dp = [[''] * 12 for _ in range(1 << 12)]
        for i in range(1 << n):
            for k in range(n):
                # skip if A[k] is not in set(i) 
                if not (i & (1 << k)):
                    continue
                # if set(i) == {A[k]}
                if i == 1 << k:
                    dp[i][k] = A[k]
                    continue
                for j in range(n):
                    if j == k:
                        continue
                    if i & (1 << j):
                        # the shortest superstring if we remove A[k] from the set(i)
                        s = dp[i ^ (1 << k)][j]
                        s += A[k][shared[j][k]:]
                        if dp[i][k] == '' or len(s) < len(dp[i][k]):
                            dp[i][k] = s

        min_len = float('inf')
        result = ''

        # find the shortest superstring of all candidates ending with different string
        for i in range(n):
            s = dp[(1 << n) - 1][i]
            if len(s) < min_len:
                min_len, result = len(s), s
        return result

# 
		


############################
#  To Lower Case
############################
#idk if its the difference betten lower and upper
class Solution:
    def toLowerCase(self, s: str) -> str:
        '''
        for an upper case number, its lower is 32 away
        we only want to change numbers after 'z'
        '''
        res = []
        N = len(s)
        for ch in s:
            #if ch is uppercases
            if 'A' <= ch <= 'Z':
                res.append(chr(ord(ch)+32)) #could also use bitwuse or, '|'
            else:
                res.append(ch)
        return "".join(res)

#make the hashmap
class Solution:
    def toLowerCase(self, s: str) -> str:
        '''
        we also could just map
        '''
        upper = string.ascii_uppercase
        lower = string.ascii_lowercase
        
        mapp = dict(zip(upper,lower))
        
        return "".join([mapp[ch] if ch in mapp else ch for ch in s])

##########################
# Evaluate Reverse Polish Notation
##########################
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        '''
        reverse polish notation is just postfix
        keep pushing on a stack until we hit and opertion
        then apply the operation stack[-2],stack[-1], 
        push the result back on the stack
        '''
        N = len(tokens)
        stack = []
        for ch in tokens:
            if stack and ch in "+-*/":
                num_right = stack.pop()
                num_left = stack.pop()
                if ch == "+":
                    stack.append(num_left+num_right)
                elif ch == "-":
                    stack.append(num_left-num_right)
                elif ch == "*":
                    stack.append(num_left*num_right)
                elif ch == "/":
                    stack.append(int(num_left/num_right))
            else:
                stack.append(int(ch))
        
        return stack[0]

###############################################################
# Partitioning Into Minimum Number Of Deci-Binary Numbers
############################################################
class Solution:
    def minPartitions(self, n: str) -> int:
        '''
        we need binary numbers of len(n) bits
        that add up to n
        for each 1 in the string, i need to add n binary numbers up to that
        i need at least n binary numbers to get to the any digit
        to get to the number n at least, i need n binary numbers as for the max digit
        are you kidding me.....
        '''
        return max([int(foo) for foo in n])


##########################################
# Maximum Product of Word Lengths
##########################################
#surpsingly it works
class Solution:
    def maxProduct(self, words: List[str]) -> int:
        '''
        brute force would be to check each pair, check there are no matching
        then multiply
        '''
        def isCommon(word1,word2):
            if len(set(word1).intersection(set(word2))) > 0:
                return True
            return False

        ans = 0
        N = len(words)
        for i in range(N):
            for j in range(i+1,N):
                if isCommon(words[i],words[j]) == False:
                    ans = max(ans,len(words[i])*len(words[j]))
        return ans

class Solution:
    def maxProduct(self, words: List[str]) -> int:
        '''
        there are a few ways to increase the runtime of the brute force
        1. decrease the number of word comparisons
        2. optimize the no commonleters functino using bitmasks and precomputations
        i.e Among all the strings with the same set of letters (abab, aaaaabaabaaabbaaaaabaabaaabb, bbabbabbabbabbabba) it's enough to keep the longest one (aaaaabaabaaabbaaaaabaabaaabb).
        approach 1, optimize noCommonLetters Functino using bitmaks + precomp
            1. naive lengthwise comparion is L1 X L2, using bitmasks we can get that down to L1+L2
            since the words contia only lowe case letters, there are 26 bits to represent a word
            we set a bit equal to 1 if char == a
            standard bitwise trick:
                nth bit = 1 << n
            standard bitwise mask:  
                iterarte over the words leter by letter and compute the corresponding nth bit letter
                n = ord(ch) 0 ord('a')
                nth bit = 1 << n
                bitmask = bitmask | nth bit
            we can use the mask to compare bit bitwise operators in L1+L2
        algo:
            1. we cache all bitmasks once, for each word, allowing comparion in O(1) time
            2. compute bitmaasks for all words and svae them in an array 
            3. compare each word with all the following words one by one, and update ne product if the have no common letters
            
        '''
        def no_common_letters(s1,s2):
            '''
            using bit masks to compare words in linear time
            '''
            bit_number = lambda ch: ord(ch) - ord('a')
            
            bitmask1 = bitmask2 = 0
            for ch in s1:
                bitmask1 |= 1 << bit_number(ch)
            for ch in s2:
                bitmask2 |= 1 << bit_number(ch)
            return bitmask1 & bitmask == 0
        
        N = len(words)
        masks = [0]*N
        lens = [0]*N
        bit_number = lambda ch: ord(ch) - ord('a')
        
        #compute bitmasks for each word and legnths
        for i in range(N):
            bitmask = 0
            for ch in words[i]:
                bitmask |= 1 << bit_number(ch)
            #cache
            masks[i] = bitmask
            lens[i] = len(words[i])
        
        ans = 0
        #compare maskes
        for i in range(N):
            for j in range(i+1,N):
                if masks[i] & masks[j] == 0:
                    ans = max(ans, lens[i]*lens[j])
        return ans


class Solution:
    def maxProduct(self, words: List[str]) -> int:
        '''
        approach 2:
        we can avoid comparion in N^2 every time by using a hashmap 
        i.e  Among all the strings with the same set of letters (abab, aaaaabaabaaabbaaaaabaabaaabb, bbabbabbabbabbabba) it's enough to keep the longest one (aaaaabaabaaabbaaaaabaabaaabb)
        1. we use a hasmap, that maps the words bit mask to its length
        2. we update this result so that it gets the longest one
        '''
        
        mapp = defaultdict(int)
        bit_number = lambda ch: ord(ch) - ord('a')
        
        #compute bitmasks for each word and legnths
        for word in words:
            bitmask = 0
            for ch in word:
                bitmask |= 1 << bit_number(ch)
            #cache
            mapp[bitmask] = max(mapp[bitmask],len(word))
        ans = 0
        for word1 in mapp:
            for word2 in mapp:
                if word1 & word2 == 0:
                    ans = max(ans, mapp[word1]*mapp[word2])
        return ans
      
################################
#Maximum Erasure Value
################################
class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        '''
        subarray must be continuous and can be of any length
        and the sub array can contain only unique elements
        elements are unique
        we want to retunr the maximum score by erasing exactly one subarray
        similar to longest no repating substriong
        two pointers
        '''
        N = len(nums)
        seen = set()
        l,r = 0, 0
        max_score = 0
        curr_score = 0
        while l < N and r < N:
            #if i havent seen number at r yet
            if nums[r] not in seen:
                seen.add(nums[r])
                curr_score += nums[r]
                max_score = max(max_score,curr_score)
                r += 1
            else:
                seen.remove(nums[l])
                curr_score -= nums[l]
                l += 1
        return max_score

class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        '''
        using a boolean array to mark unique values instead adding and removing from set
        there is a littler bit of overhead add/remove items from set
        same is before, but mark unique boolean array True False
        we can do this because we know the elements reach an upper bound
        we could also use a count array/map and check if element has multiplicty > 1
        '''
        N = len(nums)
        unique = [False]*10001
        l,r = 0, 0
        max_score = 0
        curr_score = 0
        while l < N and r < N:
            #if i havent seen number at r yet
            if unique[nums[r]] == False:
                unique[nums[r]] = True
                curr_score += nums[r]
                max_score = max(max_score,curr_score)
                r += 1
            else:
                unique[nums[l]] = False
                curr_score -= nums[l]
                l += 1
        return max_score

class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        '''
        using count array
        '''
        N = len(nums)
        count = [0]*10001
        max_score = 0
        curr_score = 0
        start = 0
        for end in range(0,N):
            curr_element = nums[end]
            count[curr_element] += 1
            curr_score += curr_element
            while start < N and count[curr_element] > 1:
                count[nums[start]] -= 1
                curr_score -= nums[start]
                start += 1
            max_score = max(max_score, curr_score)
        
        return max_score

class Solution:
    def maximumUniqueSubarray(self, nums: List[int]) -> int:
        '''
        we can use the prefix sum array and map
        prefix sum gives us the sums of all subarrays
        our map gives use the last seen index position if the curr element we are on
        prefix sum always len(n) + 1 to control for edge cases
        We could use this map to know the last index of currentElement. 
        If the last occurrence of currentElement is between the start and end pointer, its last index would be greater than the start index.
        We can use this trick to determine if the currentElement is unique in the current subarray or not.
        start = max(start, lastIndexMap.get(currentElement) + 1)
        '''
        N = len(nums)
        last_idx = defaultdict(int)
        prefixSum = [0]*(N+1)
        
        start = 0
        max_score = 0
        for end in range(0,N):
            curr_element = nums[end]
            prefixSum[end+1] = prefixSum[end] + curr_element
            if curr_element in last_idx:
                start = max(start, last_idx[curr_element]+1)
            max_score = max(max_score, prefixSum[end+1]-prefixSum[start])
            #not in mapp
            last_idx[curr_element] = end
        
        return max_score

###############################
# N-Queens II
###############################
class Solution:
    def totalNQueens(self, n: int) -> int:
        '''
        this is the same as the first one, but instead of getting all solutions
        just return the number
        '''
        def backtrack(row,cols,diags,anti_diags):
            if row == n:
                return 1
            num_solutions = 0
            #place queen each row
            for col in range(n):
                curr_diag = row - col
                anti_diag = row + col
                if col in cols or curr_diag in diags or anti_diag in anti_diags:
                    continue
                cols.add(col)
                diags.add(curr_diag)
                anti_diags.add(anti_diag)
                #recurse
                num_solutions += backtrack(row+1,cols,diags,anti_diags)
                #backtrack
                cols.remove(col)
                diags.remove(curr_diag)
                anti_diags.remove(anti_diag)
            return num_solutions
        
        return backtrack(0,set(),set(),set())

##################################
# Minimum Cost to Connect Sticks
#################################
#easssssy
class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        '''
        can i just sort the sticks in order and just merge them piece by piece
        replacing the merge with the merged two sticks?
        i need heap!
        python is min heap
        [1,8,3,5]
        sorted
        [8,5,3,1]
        [8,5,4] cost = 4
        [8,9] cost 4 + 9
        [17] cost 4 + 9 + 17
        '''
        #edge case
        if len(sticks) == 1:
            return 0
        
        heapify(sticks)
        cost = 0
        while len(sticks) > 1:
            stick1 = heappop(sticks)
            stick2 = heappop(sticks)
            new_stick = stick1 + stick2
            cost += new_stick
            heappush(sticks,new_stick)
        return cost

###########################
# Maximum Gap
###########################
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        '''
        just use comparison sort
        
        '''
        if len(nums) < 1:
            return 0
        nums.sort()
        res = 0
        N = len(nums)
        for i in range(N-1):
            res = max(res, nums[i+1]-nums[i])
        return res

#using radix sort
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        '''
        we can use radix sort, where we use counting sort as sub routine
        couting sort runs in O(n+k), where there are n digits of base k
        our system is base 10 so its O(n+k)
        radix sort, time complexity is O(d*(time complexity of stable sort))
        d is the number of digits for the longes number in numbes
        O(d*(n+k))
        https://www.geeksforgeeks.org/radix-sort/
        '''
        if not nums or len(nums) < 2:
            return 0
        
        N = len(nums)
        maxVal = max(nums)
        exp = 1
        radix = 10
        
        def counting_sort():
            #ouptut array after sorting
            output = [0]*N
            #digits in base radix
            counts = [0]*radix
            #store occruences
            for i in range(N):
                idx = nums[i] // exp
                counts[idx % 10] += 1
            #accumulate occurnecs
            for i in range(1,radix):
                counts[i] += counts[i-1]
            #buiild the output array
            i = N-1
            while i >= 0:
                idx = nums[i] // exp
                output[counts[idx % 10]-1] = nums[i]
                counts[idx % 10] -= 1
                i -= 1
            #write back to nums
            for i in range(N):
                nums[i] = output[i]
                
        #radix sort
        while maxVal // exp > 0:
            counting_sort()
            exp *= 10
        
        res = 0
        for i in range(1,N):
            res = max(res, nums[i] - nums[i-1])
        return res

#bucket sort
class Bucket:
    def __init__(self):
        self.used = False
        self.minval = float('inf')
        self.maxval = float('-inf')

class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        '''
        recall comparison sorts are costly, the idea is to use bucket sort
        pigeon hole prinicple, if there n items to put in m containers and n > m, at least one container must have multiple items
        the idea if to figure out the bucket size
        if there are n items, there  n - 1 gaps of size t
        t = (max - min)  / (n-1) gives us the gap size
        this t is the smallest value that can give n-1 gaps for n elements, no proof in the pudding this time
        
        digression:
            start with uniform gap array and try to reuce the gap between any two adjacent elements
        
        idea is to compare buckets and not within buckets, if the latter we would do no better than using a stable nlgn sorting solution
        key: we need to ensure that the gap between the buckets itself repreent the maximual gap in the input array
        We could do that just by setting the buckets to be smaller than t = (max - min)/(n-1)t=(max−min)/(n−1)
        algo:
            we choose a bucket size to be b = ⌊(max−min)/(n−1)⌋
            so all the n elements would be divided amont k = k=⌈(max−min)/b⌉ 
            hence the i^{th}i th bucket would hold the range of values: \bigg [min + (i-1) * b, \space min + i*b \bigg )[min+(i−1)∗b, min+i∗b)
            we can access the bucket an element belongs to using
            ⌊(num−min)/b⌋, if there are k buckets
            once all n elements have been distribute, we compare k-1 adjacent buckets pairs to find the max gap
            
        '''
        if not nums or len(nums) < 2:
            return 0
        mini = min(nums)
        maxi = max(nums)
        N = len(nums)
        
        bucketSize = max(1,(maxi-mini) // (N-1))
        numBuckets = (maxi - mini) // (bucketSize) + 1
        
        #make buckets
        buckets = [Bucket() for _ in range(numBuckets)]
        
        for num in nums:
            #get the indxex of the corresponding num to its bucket using the transition fomrulate
            idx = (num - mini) // bucketSize
            #place into bucket
            buckets[idx].used = True
            buckets[idx].minval = min(num, buckets[idx].minval)
            buckets[idx].maxval = max(num, buckets[idx].maxval)
        
        #now compare min val of current buckt to prev
        prevBucketMax = mini
        maxGap = 0
        for b in buckets:
            if b.used == False:
                continue
            maxGap = max(maxGap, b.minval - prevBucketMax)
            prevBucketMax = b.maxval
        
        return maxGap

#just another way
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        '''
        just another way of doing bucket sort
        build buckets into a mapp
        then go across all the buckets 
        for each bucket maintain the current min and max
        '''
        N = len(nums)
        if N < 2:
            return 0
        lo,hi = min(nums),max(nums)
        buckets = defaultdict(list)
        #put into buckets
        for num in nums:
            #edge case, highest number to last bucket, 
            if num == hi:
                idx = N-1
            else:
                #get bucket index
                idx = (abs(lo-num)*(N-1) // (hi-lo))
            #put into bucket
            buckets[idx].append(num)
        
        #pull the min and maxs for each bucket
        ranges = []
        for i in range(N):
            if buckets[i]:
                ranges.append((min(buckets[i]),max(buckets[i])))
            
        #now compare the min if the previous bucket to the max of the current bucket
        maxGap = 0
        for i in range(1,len(ranges)):
            maxGap = max(maxGap,abs(ranges[i-1][-1]-ranges[i][0]) )
        return maxGap

##########################
# Search Suggestions System
###########################
#close one
#its the edge cases that are thorwing me off

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        '''
        trying doing brute force with binary seach first
        sort the products first by alphabetical order and then by lenght
        binary seach to find the answer
        '''
        products.sort(key = lambda x : (x,len(x)))
        
        results = []
        N = len(searchWord)
        for i in range(N):
            prefix = searchWord[:i+1]
            #binary search the list of products
            lo,hi = 0, len(products) - 1
            while lo <= hi: #there are no repated elements
                mid =  lo + (hi - lo) // 2
                if prefix <= products[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            results.append(products[lo:lo+3])
        return results

#using bisect
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        '''
        we can just the bisect left helper
        returns the left pointer, i.e the index to the left after binary seraching
        '''
        products.sort()
        results = []
        N = len(searchWord)
        lower_bound = 0
        for i in range(N):
            prefix = searchWord[:i+1]
            lower_bound = bisect.bisect_left(products,prefix,lower_bound)
            #append only if the first matches the prefix
            temp = []
            for word in products[lower_bound:lower_bound+3]:
                if word.startswith(prefix):
                    temp.append(word)
            results.append(temp)
        return results
        
#defining binary search
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        '''
        we can just the bisect left helper
        returns the left pointer, i.e the index to the left after binary seraching
        '''
        def binary_search(array,target):
            lo = 0
            hi = len(array) -1
            while lo < hi:
                mid = lo + (hi- lo) // 2
                if array[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            return lo
        
        products.sort()
        results = []
        N = len(searchWord)
        for i in range(N):
            prefix = searchWord[:i+1]
            lower_bound = binary_search(products,prefix)
            #append only if the first matches the prefix
            temp = []
            for word in products[lower_bound:lower_bound+3]:
                if word.startswith(prefix):
                    temp.append(word)
            results.append(temp)
        return results

#rebuilding the products array
class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        '''
        we can just generate each prefix and match,
        then reassign products after taking only the first three
        '''
        products.sort()
        res = []
        
        for i,ch in enumerate(searchWord):
            temp = []
            for p in products:
                if i < len(p):
                    if (ch == p[i]):#we are recreating products for each matche char and we must also check for lenth
                        temp.append(p)
            res.append(temp[:3])
            products = temp
        
        return res

#Trie solution
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.suggestions = []
    
    def add_suggestion(self,product):
        if len(self.suggestions) < 3:
            self.suggestions.append(product)

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        '''
        just go over Trie solution
        build trie, but at each node, add in at least thee sugessionts
        '''
        products.sort()
        root = TrieNode()
        
        #build True
        for p in products:
            node = root
            for ch in p:
                node = node.children[ch]
                node.add_suggestion(p)
        
        #build results
        res = []
        node = root
        for ch in searchWord:
            node = node.children[ch]
            res.append(node.suggestions)
        return res






























