####################################################
# 2300. Successful Pairs of Spells and Potions
# 02APR23
###################################################
#ezzzzzz
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        '''
        a successful spell and potion pair is when the product of their strengths is at least success

        n = len(spells)
        m = len(potions)
        
        want an array of length n, call it pairs
        where pairs[i] = num potions that make successful pair with the ith spell
        
        the values are all positive, sort potions, and binary search to find the lower bound in potions
        the answer for pairs[i] = len(potions) - index of lowwer bound
        '''
        n = len(spells)
        m = len(potions)
        pairs = [0]*n
        #sort
        potions.sort()
        
        for i in range(n):
            spell_strength = spells[i]
            #look for potion that would give us success
            #carefull for exact, because of at least
            if (success % spell_strength) == 0:
                look_for = success // spell_strength
            else:
                look_for = (success // spell_strength) + 1
            #find lower bound
            idx = bisect.bisect_left(potions,look_for)
            pairs[i] = m - idx
        
        return pairs

#we can fix look for and direcly code the lower bound search
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        '''
        a successful spell and potion pair is when the product of their strengths is at least success

        n = len(spells)
        m = len(potions)
        
        want an array of length n, call it pairs
        where pairs[i] = num potions that make successful pair with the ith spell
        
        the values are all positive, sort potions, and binary search to find the lower bound in potions
        the answer for pairs[i] = len(potions) - index of lowwer bound
        '''
        n = len(spells)
        m = len(potions)
        pairs = [0]*n
        #sort
        potions.sort()
        
        for i in range(n):
            spell_strength = spells[i]
            #look for potion that would give us success
            #carefull for exact, because of at least
            look_for,remainder = divmod(success,spell_strength)
            look_for += 1 if remainder != 0 else 0
            #find lower bound
            left = 0
            right = m
            while left < right:
                mid = left + (right - left) // 2
                if potions[mid] >= look_for:
                    right = mid
                else:
                    left = mid + 1
                    
            pairs[i] = m - left
        
        return pairs
#don't forget the celing function, incremenrt numerator by denom by just 1 multiplicty then less 1, interger division
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        '''
        a successful spell and potion pair is when the product of their strengths is at least success

        n = len(spells)
        m = len(potions)
        
        want an array of length n, call it pairs
        where pairs[i] = num potions that make successful pair with the ith spell
        
        the values are all positive, sort potions, and binary search to find the lower bound in potions
        the answer for pairs[i] = len(potions) - index of lowwer bound
        '''
        n = len(spells)
        m = len(potions)
        pairs = [0]*n
        #sort
        potions.sort()
        
        for i in range(n):
            spell_strength = spells[i]
            #look for potion that would give us success
            #carefull for exact, because of at least
            #look_for,remainder = divmod(success,spell_strength)
            #look_for += 1 if remainder != 0 else 0
            #cieling function ceil(success,spell) = (success + spell - 1) // spell
            look_for = (success + spell_strength - 1) // spell_strength
            #find lower bound
            left = 0
            right = m
            while left < right:
                mid = left + (right - left) // 2
                if potions[mid] >= look_for:
                    right = mid
                else:
                    left = mid + 1
                    
            pairs[i] = m - left
        
        return pairs

#we can also use two pointers
class Solution:
    def successfulPairs(self, spells: List[int], potions: List[int], success: int) -> List[int]:
        '''
        if we were to sort both spells and potions increasingly
        say we have spell 'a' and minPotion 'b', such that a*b = success
        then for any spell bigger than a, call it bigger_a, 
            bigger_a*b >= success
        
        we keep two poitners, one pointing to the smallest spell and one pointing to the largest point,
        then for every spell, we try to find the smallest postion, such that it is at least success
        
        for the spell[i], spell[i+1] >= spell[i], which means all potions used to get a success from spell[i] will certainly work with spell[i+1]
        
        we need to keep track of the spell with its index, because the answer array wants a spell potion pair from the original ordering in the input
        
        
        '''
        #keep spell with sorted index
        sorted_spells = [(spell,i) for i,spell in enumerate(spells)]
        #sort on spells
        sorted_spells.sort()
        #sort potions
        potions.sort()
        
        n = len(spells)
        m = len(potions)
        ans = [0]*m
        
        
        potion_ptr = m - 1
        for spell,index in sorted_spells:
            while potion_ptr > 0 and (spell*potions[potion_ptr] >= success):
                potion_ptr -= 1
            ans[index] = m - (potion_ptr + 1)
        
        return ans

#############################################
# 245. Shortest Word Distance III (REVISTED)
# 02APR23
############################################
#TLE
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        '''
        dump into hashmap for both words, then get minimum pair wise distance
        '''
        mapp = defaultdict(list)
        for i,w in enumerate(wordsDict):
            if w == word1 or w == word2:
                mapp[w].append(i)
        
        ans = len(wordsDict)
        list1 = mapp[word1]
        list2 = mapp[word2]
        
        for i in list1:
            for j in list2:
                if j != i:
                    ans = min(ans, abs(i-j))
        
        return ans

class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        '''
        store indices, then use two pointers to compare
        '''
        mapp = defaultdict(list)
        for i,word in enumerate(wordsDict):
            mapp[word].append(i)
            
        idxs_1 = mapp[word1]
        idxs_2 = mapp[word2]
        
        #if they are the same, get consective differecnes
        if word1 == word2:
            ans = float('inf')
            for i in range(1,len(idxs_1)):
                ans = min(ans, idxs_1[i] - idxs_2[i-1])
            
            return ans
        #not the same use two pointer and advance the samller index minimizing along the way
        else:
            ans = float('inf')
            i,j = 0,0
            while i < len(idxs_1) and j < len(idxs_2):
                ans = min(ans, abs(idxs_1[i] - idxs_2[j]))
                if idxs_1[i] < idxs_2[j]:
                    i += 1
                else:
                    j += 1
            
            return ans

#bianry search
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        '''
        we can use binary search
        say we have an index i, for all indices that have word1
        we want to find the shortest distance to all indices j for which wordsDict[j] == word2
        if these j indices are in sorted order, we can use binary search to find the closes index over all indices
        we want to find the upper bound
            upper bound returns the first index in indices2, that is just greater than the current index
        '''
        mapp = defaultdict(list)
        for i,word in enumerate(wordsDict):
            mapp[word].append(i)
            
        idxs_1 = mapp[word1]
        idxs_2 = mapp[word2]
        
        ans = float('inf')
        
        for i in idxs_1:
            look_for = bisect.bisect_right(idxs_2,i)
            #not pointing to the last element at idxs_2
            if look_for != len(idxs_2):
                ans = min(ans, idxs_2[look_for] - i)
            #if x > 0, find the difference between index with idxs_2[look_for -1], because we could have picked an indices that are the same
            if look_for != 0 and idxs_2[look_for - 1] != i:
                ans = min(ans,i - idxs_2[look_for - 1])
        
        return ans

#binary search, writing out upper bound
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        '''
        we can use binary search
        say we have an index i, for all indices that have word1
        we want to find the shortest distance to all indices j for which wordsDict[j] == word2
        if these j indices are in sorted order, we can use binary search to find the closes index over all indices
        we want to find the upper bound
            upper bound returns the first index in indices2, that is just greater than the current index
        '''
        mapp = defaultdict(list)
        for i,word in enumerate(wordsDict):
            mapp[word].append(i)
            
        idxs_1 = mapp[word1]
        idxs_2 = mapp[word2]
        
        ans = float('inf')
        
        #writing out upper bound
        def upper_bound(array,target):
            left = 0
            right = len(array)
            
            while left < right:
                mid = left + (right - left) // 2
                if array[mid] <= target:
                    left = mid + 1
                else:
                    right = mid
                    
            return left
        
        for i in idxs_1:
            look_for = upper_bound(idxs_2,i)
            #not pointing to the last element at idxs_2
            if look_for != len(idxs_2):
                ans = min(ans, idxs_2[look_for] - i)
            #if x > 0, find the difference between index with idxs_2[look_for -1], because we could have picked an indices that are the same
            if look_for != 0 and idxs_2[look_for - 1] != i:
                ans = min(ans,i - idxs_2[look_for - 1])
        
        return ans

#merging two lists
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        '''
        we can merge the two lists as entries:
            (index, is_word1)
            (index, is_word2)
        
        then check consective pairs, making sure the indices are not the same, and that they belong to different words
        '''
        indices = []
        for i,word in enumerate(wordsDict):
            if word == word1:
                indices.append((i,0))
            if word == word2:
                indices.append((i,1))
                
        ans = float('inf')
        for i in range(1,len(indices)):
            u = indices[i-1]
            v = indices[i]
            
            if u[0] != v[0] and u[1] != v[1]:
                ans = min(ans, v[0] - u[0])
        
        return ans

#another cool way
class Solution:
    def shortestWordDistance(self, wordsDict: List[str], word1: str, word2: str) -> int:
        w1i = w2i = -len(wordsDict)
        min_dist = float(inf)
        
        for i, w in enumerate(wordsDict):
            if w == word1:
                w1i = i
            if word1 == word2 and w1i != w2i:
                min_dist = min(min_dist, abs(w1i - w2i))
            if w == word2:
                w2i = i
            if w1i != w2i:
                min_dist = min(min_dist, abs(w1i - w2i))
            
        return min_dist


##################################
# 860. Lemonade Change
# 03APR23
##################################
#fuck
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        '''
        customers can only pay in denominatinos of 5,10,20
        each transactinos if 5, and may or may not warrant change
        return true if i can get through the whole array by providing every custome exact change
        
        no change on 5s
        change must be 5, or 15, i must always have the ability to give out a 5 or 15
        
        5, can do we count5s > 0 
        15, count5s == 3, count5s == 1 and count10s = 1
        x = number of 5s 
        y = number of 10s
        z = number of 20s
        
        x*5 + 0*10 + 0*20 = 5 (1,0,0)
        3*5 + 0*10 + 0*20 = 15 (3,0,0),
        1*5 + 1*10 + 0*20 = 15 (1,1,0)
        
        
        '''
        state = [0,0,0]
        for b in bills:
            if b == 5:
                state[0] += 1
            if b == 10:
                if state[0] == 0:
                    return False
                else:
                    state[0] -= 1
                    state[1] += 1
            
            if b == 20:
                if state[0] > 0 and state[1] > 0:
                    state[0] -= 1
                    state[1] -= 1
                elif state[0] >= 3:
                    state[0] -= 3
                else:
                    return False
        return True
                
#just keep track of fives and tens
class Solution(object): #aw
    def lemonadeChange(self, bills):
        five = ten = 0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if not five: return False
                five -= 1
                ten += 1
            else:
                if ten and five:
                    ten -= 1
                    five -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False
        return True

##################################
# 868. Binary Gap
# 03APR23
##################################
#bleaghh
class Solution:
    def binaryGap(self, n: int) -> int:
        '''
        we want the largest gap between between 1's in the binary form
        largest number is 10**9, biggest number of positions would be log_2(10**9) = 32
        
        slidigin windows, must start with and end with 1
        '''
        positions = []
        while n:
            positions.append(n & 1)
            n //= 2
            
        
        N = len(positions)
        ans = 0
        left = 0
        while left < N and positions[left] != 1:
            left += 1
            right = left + 1
            while right < len(positions) and positions[right] != 1:
                right += 1
            if positions[left] == 1 and positions[right] == 1:
                ans = max(ans,right - left + 1)
            
            left = right
    
        return ans
            
class Solution:
    def binaryGap(self, n: int) -> int:
        '''
        store the indices of 1s in the array, and take the lagest difference
        '''
        one_idxs = []
        curr_idx = 0
        
        while n:
            if n & 1 != 0:
                one_idxs.append(curr_idx)
            curr_idx += 1
            n //= 2
        
        ans = 0
        for i in range(1,len(one_idxs)):
            ans = max(ans, one_idxs[i] - one_idxs[i-1])
        
        return ans
    
class Solution:
    def binaryGap(self, n: int) -> int:
        '''
        we don't need all the idxs of ones' just keep the last 1 index seen
        '''
        last_ones_idx = None
        curr_idx = 0
        ans = 0
        
        while n:
            if n & 1 != 0:
                if last_ones_idx is not None:
                    ans = max(ans,curr_idx - last_ones_idx)
                last_ones_idx = curr_idx
            curr_idx += 1
            n //= 2
        

        return ans

#for loop
class Solution(object):
    def binaryGap(self, N):
        last = None
        ans = 0
        for i in xrange(32):
            if (N >> i) & 1:
                if last is not None:
                    ans = max(ans, i - last)
                last = i
        return ans

#######################################
# 2034. Stock Price Fluctuation
# 03APR23
#######################################
#fuckkk
from sortedcontainers import SortedList
class StockPrice:

    def __init__(self):
        #max and min calls are easy
        #notice how there is only an update, no delete
        self.prices = SortedList([])
        self.times = SortedList([])
        self.mapp = {}

    def update(self, timestamp: int, price: int) -> None:
        if timestamp not in self.mapp:
            self.mapp[timestamp] = price
            self.prices.add(price)
            self.times.add(timestamp)
        else:
            old_price = self.mapp[timestamp]
            self.prices.discard(old_price)
            self.times.discard(timestamp)
            self.times.add(timestamp)
            self.mapp[timestamp] = price
            
    def current(self) -> int:
        latest_time = self.times[-1]
        return self.mapp[latest_time]
        

    def maximum(self) -> int:
        return self.prices[-1]
        

    def minimum(self) -> int:
        return self.prices[0]


# Your StockPrice object will be instantiated and called as such:
# obj = StockPrice()
# obj.update(timestamp,price)
# param_2 = obj.current()
# param_3 = obj.maximum()
# param_4 = obj.minimum()

from sortedcontainers import SortedDict
class StockPrice:
    '''
    we are given prices for a stack at current time stamps, but times may not come in order
    for exmple if we have a list as [2,1,5,2,10,18,3]
    min price is 1, and max price is 18, and most recent is 3
    but we could update at time 1, 
    1 becomes 19
    our max becomes 19 now, but mins stays the same
    
    we can use hahsmap to store time -> price
    but how can we get the min and max, we can sort again, but deleteing in place is time consuming, and the number of method calls i larger
    
    use regular hashmap for mapping timestamp to price
    but use sorted map to store price frequencies, but in this case, the sorting invariatn on the sorted map will be on increasing price
    
    insert:
        insert (timestamp,price) into hashmap
        increase the price count, in sorted map
        
    update:
        update (timestamp,price) in hashmap
        decrement the onld stock price, and del if zero
    
    current:
        max to get latest variable
    
    max/min;
        get at the ends of the sorted map
    
    #notes, we have to import in python
    '''

    def __init__(self):
        self.latest_time = 0
        self.price_map = {}
        self.price_freq = SortedDict() #essentially a dictionary combined with a binary search tree
        

    def update(self, timestamp: int, price: int) -> None:
        #not in mapp
        if timestamp not in self.price_map:
            self.price_map[timestamp] = price
            
            #not in frequencies
            if price not in self.price_freq:
                self.price_freq[price] = 1
            else:
                self.price_freq[price] += 1
        else:
            old_price = self.price_map[timestamp]
            if old_price in self.price_freq:
                self.price_freq[old_price] -= 1

                if self.price_freq[old_price] == 0:
                    del self.price_freq[old_price]

            #don't forget to put
            self.price_map[timestamp] = price
        
        #update latest time
        self.latest_time = max(self.latest_time,timestamp)

    def current(self) -> int:
        return self.price_map[self.latest_time]
        

    def maximum(self) -> int:
        #retrieve from end of sorted dict
        return self.price_freq.peekitem(-1)[0]

    def minimum(self) -> int:
        #retriece from beginning of sorted dict
        return self.price_freq.peekitem(0)[0]


# Your StockPrice object will be instantiated and called as such:
# obj = StockPrice()
# obj.update(timestamp,price)
# param_2 = obj.current()
# param_3 = obj.maximum()
# param_4 = obj.minimum()

#using max heap and min heap
class StockPrice:
    '''
    instead of using sortedicts we can use min and max heaps, this is probably the more important of the two solutions
    how do we update effecitently?
        we'd have to keep popping until the old price comes out on top, then push back all the prices back in, which would be very costly
    
    idea: every time we get a new price, push into heap
    and only whule getting the top element we need to verify if the price is outdated
    
    how we know its outdates
    for both hepas store (price,timestamp)
        if timestamp already exists in the hashmap, overwrite the price
    
    when finding max or min, we check to if (price,timesamp) pair on top agrees with the price lists for the timestamp in the hashmap
    
    idea
        we update the most recent stock price given a time
        when we retrive from either of the heaps, we need to validate if its outdated or not
        since the true timestamp already exsits in the hashmap, we keep popping until the timestamp matches the price as the timesamp in the heaps
        doing so liberates outdated items on the heaps
        we are guarnteed to get the max and mins stil because they are heaps
        
        intution; 
            we have all prices seen in history on heaps, we just need to filter until it matches whats in the hashmap
    '''

    def __init__(self):
        self.latest_time = 0
        self.timePriceMap = {}
        
        self.min_heap = []
        self.max_heap = []
        

    def update(self, timestamp: int, price: int) -> None:
        self.timePriceMap[timestamp] = price
        self.latest_time = max(self.latest_time, timestamp)
         
        #add to heaps
        heapq.heappush(self.min_heap, (price,timestamp))
        heapq.heappush(self.max_heap, (-price,timestamp))

    def current(self) -> int:
        return self.timePriceMap[self.latest_time]
        

    def maximum(self) -> int:
        price,timestamp = self.max_heap[0]
        
        while -price != self.timePriceMap[timestamp]:
            price,timestamp = heapq.heappop(self.max_heap)
        
        return -price
        

    def minimum(self) -> int:
        price,timestamp = self.min_heap[0]
        
        while price != self.timePriceMap[timestamp]:
            price,timestamp = heapq.heappop(self.min_heap)
        
        return price


# Your StockPrice object will be instantiated and called as such:
# obj = StockPrice()
# obj.update(timestamp,price)
# param_2 = obj.current()
# param_3 = obj.maximum()
# param_4 = obj.minimum()


##################################
# 2405. Optimal Partition of String
# 04APR23
##################################
#sliding window
class Solution:
    def partitionString(self, s: str) -> int:
        '''
        we have a string s, and we want to partitions into 1 or more SUBSTRINGS such that the chars in each substring are unique
        return the minimum number of partitions
        
        if we split individually on each char, the answer can be no more than len(s), where each partition is just the single character
        
        sliding window? greedy
        abacaba
        
        ab ac ab a
        
        sliding window, where each window has unique chars, once we violate this condition, increment paritino count
        '''
        count = 0
        chars_in_window = set()
        N = len(s)
        
        left = 0
        while left < N:
            right = left
            while right < N and s[right] not in chars_in_window:
                chars_in_window.add(s[right])
                right += 1
            
            #no longer viable
            count += 1
            #reset
            chars_in_window = set()
            left = right
        
        return count
    
#we can save space, intead just keep track of the index of the last seen char
class Solution:
    def partitionString(self, s: str) -> int:
        '''
        since the characters are only lower case, we can keep track of the last seen index
        say we are at s[i], which is 'a'
        we can include this in our parition if haven't seen this yes
        we also need to include the index of the start of the current substring
        '''
        last_seen = [-1]*26 #could use bit bit mask
        count = 1 #in the case we go through the whole string
        left = 0 #start of current substring
        N = len(s)
        
        for i in range(N):
            idx = ord(s[i]) - ord('a')
            #if ive seen this char already after the current left
            if last_seen[idx] >= left:
                #make new window
                count += 1
                left = i
            #upate
            last_seen[idx] = i
        
        return count

#####################################
# 1146. Snapshot Array 
# 04APR23
#####################################
#TLE
class SnapshotArray:

    def __init__(self, length: int):
        '''
        the array can have different states at a certain point in time
        save arrays into a hashamp
        '''
        self.curr_array = [0]*length
        self.snapped_arrays = {}
        self.snap_id = 0

    def set(self, index: int, val: int) -> None:
        self.curr_array[index] = val

    def snap(self) -> int:
        self.snapped_arrays[self.snap_id] = self.curr_array[:]
        self.snap_id += 1
        return self.snap_id -1
        

    def get(self, index: int, snap_id: int) -> int:
        return self.snapped_arrays[snap_id][index]


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)

#fuck my life
class SnapshotArray:

    def __init__(self, length: int):
        '''
        hint1: use list of lists adding both element and snap id to each index
        then use binary search to find the snap_id's value
        '''
        self.container = [[0,0] for _ in range(length)] #at the index we have (snap_id,val)
        self.curr_snap = 0
        

    def set(self, index: int, val: int) -> None:
        #first get the entry
        entry = self.container[index]
        #updates
        if entry[0] == self.curr_snap:
            self.container[index][0] = self.curr_snap
            self.container[index][1] = val
        else:
            #prepare
            entry = self.container[index]
            new_entry = [entry[0]+1,entry[1]]
            self.container[index].append(new_entry)
    def snap(self) -> int:
        self.curr_snap += 1
        return self.curr_snap - 1
        

    def get(self, index: int, snap_id: int) -> int:
        return self.container[index][snap_id]


# Your SnapshotArray object will be instantiated and called as such:
# obj = SnapshotArray(length)
# obj.set(index,val)
# param_2 = obj.snap()
# param_3 = obj.get(index,snap_id)

##################################
# 2439. Minimize Maximum of Array
# 05APR23
##################################
#brute force
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        '''
        brute force solution, find the maximum number in the array
        then reduce it, then find max, then reduce it again
        we stop when the number before it decreaes
        [3,7,1,6] max is 7
        [4,6,1,6] max is 6
        [5,5,2,5] max is 5
        
        if i do this again
        [6,4,3,4] max is 6, woops, we can't do this
        
        '''
        prev_max = max(nums)
        curr_max = None
        N = len(nums)
        
        while True:
            found_max = False
            for i in range(1,N):
                if nums[i] > nums[i-1] and nums[i] == prev_max:
                    nums[i] -= 1
                    nums[i-1] += 1
                    found_max = True
            if not found_max:
                return max(nums)
            #print(nums)
            
            #check new max:
            curr_max = max(nums)
            if  curr_max < prev_max:
                prev_max = curr_max
                curr_max = None
            
            #no change
            elif curr_max > prev_max:
                return curr_max

#optimized brute force
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        '''
        instead of going in steps of 1, go right to the middle of the difference
        
        '''
        prev_max = max(nums)
        curr_max = None
        N = len(nums)
        
        while True:
            found_max = False
            for i in range(1,N):
                if nums[i] > nums[i-1] and nums[i] == prev_max:
                    diff = nums[i] - nums[i-1]
                    diff = diff // 2 if (diff % 2 == 0) else diff // 2 + 1
                    nums[i] -= diff
                    nums[i-1] += diff
                    found_max = True
            if not found_max:
                return max(nums)
            #print(nums)
            
            #check new max:
            curr_max = max(nums)
            if  curr_max < prev_max:
                prev_max = curr_max
                curr_max = None
            
            #no change
            elif curr_max > prev_max:
                return curr_max
        
#binary search
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        '''
        binary search on prefix sum
        [3,7,1,6] for an index i we can just increment and decrement by 1
        [4,6,1,6]
        [5,5,1,6]
        [5,5,2,5]
        [5,5,3,4]
        [5,5,4,3] we are sorta trying to distribute the sum evenly across all indices
        
        in the process we can decrease any index i, at the expense of icnreasing any index less than i
        we can try raising all indices to a value x
        and this sum would be x*[0:i], if prefix sum up to i > x*[0:i], then this x value cannot be possible
        
        let n = [a,b,c,d,e]
        
        [a,b,c,d,e] choose b
        [a+1,b-1,c,d,e] choose c
        [a+1,b,c-1,d,e] choose d
        [a+1,b,c,d-1,e] choose e
        [a+1,b,c,d,e-1]
        
        in all cases sum is still a + b + c + d + e
        now replace 1 with some number k, which reprsents any number of moves
        n = [a,b,c,d,e] choose b
        [a+k,b-k,c,d,e] choose c
        [a+k,b,c-k,d,e] choose c
        [a+k,b+k,c-2k,d,e] choose d in steps of three
        [a+k,b+k,c+k,d-3k,e] choose e in steps of 4
        [a+k,b+k,c+k,d+k,e-4k]
        we can call this
        [a+k,b+k,c+k,d+k,e-(len(n)-1)*k]
        
        this implies that we are free to raise any number by whatever k we want to, except the current i
        rather for an index i, we can raise n[:i-1]'s elements to any number x, except the last one
        find the lowerbound for the largest x
        
        
        '''
        def valid_maximum(candidate,nums):
            pref_sum = 0
            for i in range(len(nums)):
                pref_sum += nums[i]
                #if pref_sum is bigger than the sum of the array using candidate, we can't possible reach this sum
                if pref_sum > candidate*(i+1):
                    return False
            return True
        
        
        left = min(nums)
        right = max(nums)
        
        while left < right:
            mid = left + (right - left) // 2
            if valid_maximum(mid,nums):
                right = mid
            else:
                left = mid + 1
        
        return left

#pref_sum
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        '''
        given array of non negtative
        in one operation:
            choose an integer i such that 1 <= i < n and nums[i] > 0
            decrease nums[i] by i
            or decrease nums[i-1] by 1
        
        return minimum value of the maximum integer of nums after peforming ANY number of operations
        
        for each element in nums
        we can either increment at nums[i] += 1
        we can either decrement at nums
        
        wrong, this must happen at the same time for and index i
        [3,7,1,6]
        pick index 1
        [4,6,1,6]
        pick index 3
        [4,6,2,5]
        pick index 1
        [5,5,2,5]
        
        so i one move, we just move the value 1 over from index i to i-1
        
        intuion
            notice that the sum won't change
            what if we made each number equal in height
        
        sum of current array is [3,7,1,6] = 17
        if we were to equalize each height
        17 // 4 = 4.25
        [4,4,4,5]
        
        but this is only the case if we can move values to left and right,
        here we can only move a value left
        
        think of the trapping rain water problem
        nums = [3,7,1,3], ceil(14/4) = make new one [4,4,3,3]
        
        intution;
            while traversing the array keep track of the current prefix sum up to index i
            this prefix sum up i mean we can distribute the average to all indices up to i
            
        we updadte : answer = max(answer, ceil(prefixSum / (i + 1)))
        
        from binary search solution were eseentially solving
        pref_sum(i) <= (i+1)*x

        solving for x we get (pref_sum(i) / (i+1)) <= x
        we want to maximize this, so we try all i
        
        Without loss of generality, let's say that after reaching nums[i], we have obtained the minimum maximum value as answer_i and the prefix sum prefixSum_i. 
        Now we take into account the following number nums[i + 1], according to the operation, it can only increase the prefix sum prefixSum_i as well as the answer_i. 
        Therefore, the newly added number can't reduce the minimum maximum value, so we can't take the smaller one between answer and ceil(prefixSum / (i + 1))
        '''
        answer = 0
        pref_sum = 0
        for i in range(len(nums)):
            pref_sum += nums[i]
            #its rather unintuitve as to why this is max, note cases like [10,0,x,x]
            answer = max(answer,math.ceil(pref_sum / (i+1)))
        
        return answer
                

#can also do without ceiling function
class Solution:
    def minimizeArrayValue(self, nums: List[int]) -> int:
        answer = 0
        pref_sum = 0
        for i in range(len(nums)):
            pref_sum += nums[i]
            answer = max(answer,(pref_sum + i) // (i+1))
        
        return answer