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


#binary search
class SnapshotArray:

    def __init__(self, length: int):
        '''
        hint1: use list of lists adding both element and snap id to each index
        '''
        self.container = [[[0,0]] for _ in range(length)] #at the index we have (snap_id,val)
        self.curr_snap = 0
        

    def set(self, index: int, val: int) -> None:
        #check most recent for updating
        if self.container[index][-1][0] == self.curr_snap:
            self.container[index][-1][1] = val
            return
        #otherise append
        self.container[index].append([self.curr_snap,val])

    def snap(self) -> int:
        self.curr_snap += 1
        return self.curr_snap - 1

        
    def get(self, index: int, snap_id: int) -> int:
        #binary search to find the snap id
        curr_list = self.container[index]
        left = 0
        right = len(curr_list)
        while left < right:
            mid = left + (right - left) // 2
            if curr_list[mid][0] >= snap_id:
                right = mid
            else:
                left = mid+1
        if right == len(curr_list):
            return 0
        return curr_list[right][1]


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

####################################
# 883. Projection Area of 3D Shapes
# 06APR23
####################################
class Solution:
    def projectionArea(self, grid: List[List[int]]) -> int:
        '''
        v = grid[i][j] menas a tower of v cubes at (i,j)
        [[1,2],[3,4]]
        
        1 cube at (0,0)
        2 cubes at (0,1)
        3 cubes at (1,0)
        4 cubes at (1,1)
        
        we just need to look at the projects along xy, xz, and yz
        then sum them up, we don't want the total area, just the area of the projections
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        #for xy count up bases
        xy = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 0:
                    xy += 1
        
        
        #just take the largest
        xz = 0
        for r in range(rows):
            largest = 0
            for c in range(cols):
                largest = max(largest,grid[r][c])
            
            xz += largest
        
        yz = 0
        for c in range(cols):
            largest = 0
            for r in range(rows):
                largest = max(largest,grid[r][c])
            
            yz += largest
        
        return xy + xz + yz
            
############################################
# 1254. Number of Closed Islands (REVISTED)
# 06APR23
#############################################
#dfs
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        '''
        dfs for each cell, 
        and if we dfs on a cell that hits the bounary, this possible cannot be a closed island
        keep flag variable indicating if we hit a cell at any point in the dfs that touchs a boundary
        '''
        
        rows = len(grid)
        cols = len(grid[0])
        dirrs  = [(1,0),(-1,0),(0,1),(0,-1)]
        seen = set()
        count = 0
        
        def dfs(i,j,seen,contains_edge):
            seen.add((i,j))
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                contains_edge[0] = True
                
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #is zero and not seen
                    if grid[neigh_x][neigh_y] == 0 and (neigh_x,neigh_y) not in seen:
                        dfs(neigh_x,neigh_y,seen,contains_edge)
                        
        for i in range(rows):
            for j in range(cols):
                contains_edge = [False]
                if grid[i][j] == 0 and (i,j) not in seen:
                    dfs(i,j,seen,contains_edge)
                

                    if not contains_edge[0]:
                        count += 1
                    contains_edge[0] = False
        
        return count
    
#we can also just  use bfs, with boolean sean array
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        dirrs  = [(1,0),(-1,0),(0,1),(0,-1)]
        seen = [[False]*cols for _ in range(rows)]
        count = 0
        
        def bfs(i,j,seen,contains_edge):
            seen[i][j] = True
            q = deque([(i,j)])
            while q:
                (i,j) = q.popleft()
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    contains_edge = True

                for dx,dy in dirrs:
                    neigh_x = i + dx
                    neigh_y = j + dy
                    #bounds
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        #is zero and not seen
                        if grid[neigh_x][neigh_y] == 0 and not seen[neigh_x][neigh_y]:
                            q.append((neigh_x,neigh_y))
                            seen[neigh_x][neigh_y] = True
            
            return contains_edge
                        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0 and not seen[i][j]:
                    contains_edge = False
                    temp = bfs(i,j,seen,contains_edge)
                
                    if not temp:
                        count += 1
        
        return count
    
##################################
# 1020. Number of Enclaves
# 07APR23
##################################
#dfs
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        '''
        flood fill starting from all the boundaries where grid[i][j] == 1
        then traverse the grid again, each 1 now should be able to hold a move so count them up
        '''
        rows = len(grid)
        cols = len(grid[0])
        seen = [[False]*cols for _ in range(rows)]
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        
        
        def dfs(i,j,seen):
            #mutaute grid in place, fuck that aux space shit
            seen[i][j] = True
            grid[i][j] = 0
            
            for dx,dy in dirrs:
                neigh_x = dx + i
                neigh_y = dy + j
                
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #if 1 and not seen
                    if grid[neigh_x][neigh_y] == 1 and not seen[neigh_x][neigh_y]:
                        dfs(neigh_x,neigh_y,seen)
        
        #top row
        for c in range(cols):
            if grid[0][c] == 1 and not seen[0][c]:
                dfs(0,c,seen)
        
        #bottom row
        for c in range(cols):
            if grid[rows-1][c] == 1 and not seen[rows-1][c]:
                dfs(rows-1,c,seen)
        
        #left side
        for r in range(rows):
            if grid[r][0] == 1 and not seen[r][0]:
                dfs(r,0,seen)
                
        #right side
        for r in range(rows):
            if grid[r][cols-1] == 1 and not seen[r][cols-1]:
                dfs(r,cols-1,seen)
                
        #count
        count = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    count += 1
        
        return count

#bfs, with consolidating the extra two for loops
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        '''
        flood fill starting from all the boundaries where grid[i][j] == 1
        then traverse the grid again, each 1 now should be able to hold a move so count them up
        '''
        rows = len(grid)
        cols = len(grid[0])
        seen = [[False]*cols for _ in range(rows)]
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        
        
        def bfs(i,j,seen):
            
            q = deque([(i,j)])
            
            while q:
                i,j = q.popleft()
                #mutaute grid in place, fuck that aux space shit
                seen[i][j] = True
                grid[i][j] = 0
            
                for dx,dy in dirrs:
                    neigh_x = dx + i
                    neigh_y = dy + j

                    #bounds
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        #if 1 and not seen
                        if grid[neigh_x][neigh_y] == 1 and not seen[neigh_x][neigh_y]:
                            q.append((neigh_x,neigh_y,seen))
        
        #top row and bottom rows
        for r in [0,rows-1]:
            for c in range(cols):
                if grid[r][c] == 1 and not seen[r][c]:
                    bfs(r,c,seen)
        

        #left side and right sides
        for c in [0,cols-1]:
            for r in range(rows):
                if grid[r][c] == 1 and not seen[r][c]:
                    bfs(r,c,seen)

                
        #count
        count = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    count += 1
        
        return count

###########################################
# 133. Clone Graph (REVISTED)
# 09APR23
###########################################
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        '''
        another clone copy, need to use dfs
        map nodes to node copies using aux space
        
        
        '''
        #first try exploring all nodes in graph
        seen = set()
        
        def dfs(node):
            if node in seen:
                return
            seen.add(node)
            print(node.val)
            for neigh in node.neighbors:
                dfs(neigh)
        
        #dfs(node)
        cloned_mapp = {}
        
        if not node:
            return node
        
        def clone(node):
            if node in cloned_mapp:
                return cloned_mapp[node]
            #otherwise its not in there
            new = Node(val= node.val, neighbors = [])
            #make copy
            cloned_mapp[node] = new
            
            #if there are neighbors recurse
            for neigh in node.neighbors:
                #get answer
                new.neighbors.append(clone(neigh))
            
            #otherwise there isn't
            cloned_mapp[node] = new
            return new
        
        return clone(node)
    
################################################
# 1857. Largest Color Value in a Directed Graph
# 09APR23
################################################
#brute force, all path enumeration, graph could be disconnected
#dfs each node, get all paths, get counts of chars, get max
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        '''
        we are given n nodes, labeld 0 to n-1
        we have directed edge lists
        node[i] colored by colors[i]
        valid path, must be increasing 
        x1 -> x2 -> x3 -> ... -> xk such that there is a directed edge from xi to xi+1 for every 1 <= i < k
        the last part stems from the fact that the graph is connected
        
        color value of that most is the most freuntly occuring color along the path
        
        return largest color of any value in the graph, or is it contains a cycel, return -1 
        
        dfs along a paths, but keep color counts in the path,
        when i cant dfs anymore, update global maximum color value

        '''
        N = len(colors)
        graph = defaultdict(list)
        all_paths = []
        for u,v in edges:
            graph[u].append(v)
            
        self.ans = 0
        seen = set()
        
        #how would i print all paths
        seen = set()
        def dfs(node,path):
            #can't go anywhere else
            if len(graph[node]) == 0:
                all_paths.append(path[:])
                return
            #mark
            seen.add(node)
            for neigh in graph[node]:
                if neigh not in seen:
                    dfs(neigh,path + [neigh])
                if neigh in seen:
                    return -1
            seen.remove(node)
        
        for i in range(N):
            contains_cycle = dfs(i,[i])
        #the graph could be disconnected
        
        if contains_cycle == -1:
            return contains_cycle
        
        #count up paths and get maximum
        ans = 0
        for path in all_paths:
            counts = Counter()
            for i in path:
                counts[colors[i]] += 1
            
            ans = max(ans, max(counts.values()))
        
        return ans
            
        
#dp, top down first, jesus fucking christ
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        '''
        from the hints, we define the dp transtion as dp(u,c) be the maximum coount with color c, starting from any vertex u
        
        the answer dp(u,c) = max(of all colors c at u)
        
        the transition would be
        dp(u,c) = {
            for all neighbors v for u:
                for all colors c
                    child_ans = max(ans,dp(v,c))
        }
                
        intution:
            we use the maximum frequencies of all colors across all paths that begin with v to form the maximum frequencies for paths that begin wth u
            we can update the frequency of colors similar to topsort / kahns algo
            
        notes in dp function:
            for each nodes we start the traversal
            this returns the maximum frequency of the color of the node that we cang et across all the paths starting from node
            we aren't in a cycle we try ti update teh frequencies of all the colors stored for node by using 
            node -> neighbors edge
            we perform the maximization for all colors in the child ans
            
            #important here
            after we have processed all the outgoing edges od node, we need to increment the count for this nodes' color by 1
            because we haven't done this yet during the traversal 
        '''
        N = len(colors)
        graph = defaultdict(list)
        for u,v in edges:
            graph[u].append(v)
            
        
        memo = {} #answers (u,c) to max colors (use index and actual color)
        seen = set()
        
        def dp(node):
            #if there is a cycle while we dfsing, return -1
            if node in seen:
                return -1
            if (node) in memo:
                return memo[node]
            
            #visit
            seen.add(node)
            
            curr_count = Counter() #need to return count object for each child answer
            for neigh in graph[node]:
                #get the next neigh_color from the child
                child_ans = dp(neigh)
                #cycle
                if child_ans == -1:
                    return -1
                #from the child answer, update the currount count obkect
                for color,count in child_ans.items():
                    #update max counts
                    curr_count[color] = max(curr_count[color],count)
            #backtrack
            seen.remove(node)
            #increment this curr count color by 1
            curr_count[colors[node]] += 1
            
            #store the count object
            memo[node] = curr_count
            return curr_count
        
        ans = 0
        for i in range(N):
            contains_cycle = dp(i)
            if contains_cycle == -1:
                return -1
            ans = max(ans,max(contains_cycle.values()))
            
        return ans
    
#keeping dp memo as list of lists and adjacent list as list of lists 
#so count objects will be size (26), representing a through z
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        '''
        dp function returns max frequencye of the color of the current node we are one
        '''
        N = len(colors)
        graph = [[] for _ in range(N)]
        for u,v in edges:
            graph[u].append(v)
        
        seen = [False]*N
        memo = [[0]*26 for _ in range(N)]
        
        def dp(node):
            if seen[node] == True:
                return -1
            #dont need memo retreival anymore since memo is stored with all zeros
            #visit
            seen[node] = True
            
            for neigh in graph[node]:
                if dp(neigh) == -1:
                    return -1
                #otherwie maximuze in memo
                for i in range(26):
                    memo[node][i] = max(memo[node][i],memo[neigh][i])
                    
            #don't forget to increment teh current node
            #after all the incomind edges to nodes are prcossed, we increment the count of the color for this node itself
            curr_color = colors[node]
            color_number = ord(curr_color) - ord('a')
            memo[node][color_number] += 1
            
            #backtrack
            seen[node] = False
            return memo[node][color_number]
        
        ans = 0
        for i in range(N):
            temp = dp(i)
            if temp == -1:
                return -1
            ans = max(ans,temp)
        
        return ans

#kahns
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        '''
        we can also use kahns algorithm
        intuition:
            if we know the maximum frequency of all the colors for path ending at u
            we can use it to calculate the frequency of all colors for paths that use out edges from you
            i.e if ther eis an edge u->v, the path ending at v will have the same color ending at us, except incremented by one with color v
            
            if we do this for all the nodes that have an incmoing edge to v and take the max freq of each color acorss these edges
            we will have the max freq of all the colors for paths ending at v
            afte covering all the edges into v, we can use the max freq of all color stored in v for edges out of v
        
       notice that for each edge u->v we must first obtain the maximum frequencey of all the colors for paths ending until u
       and only then we can from the paths ending at v
       
       this leads to a top sort
       
       briefly:
        top sort is order of a directed grpah, we can use kahns algo to get the ordering
        keeps track of all the number of incoming edges into each node, 
        repeatedly visitng the nodes with indegreee of zero and deleting the edges associted with it leaving to a decrement
        of indegree for the ndoes whose incoming edges are deleted
        
        if there is a cycle, the nodes will remain unvisited, this solves the cycle part, what about the counting color part?
        
        keep count array [n by 26]
        during kahns
            a popped out node indicatate that all of its incomnig edges have been processed, and it can no be used to iterate over all out going edges
            i.e its a leaf, the path ends here so we can start counting. but how do we count
            So for each node -> neighbor edge, we use count[neighbor][i] = max(count[neighbor][i], count[node][i]) 
            (we use max here instead of just setting it because there could be multiple ways to reach the neighbor) for all colors i.
            we also need to increment the count of the color of the node we are one
            when we pop a ndoe, we incrment the count by 1
        
        do kahns for topsort and cycle deteection
        while doign kahns solve the problems:
            for a node ending at u, what is the max color for all colors c
            answer is just the max of he counts matrix if have visited all the nodes
        '''
        N = len(colors)
        graph = [[] for _ in range(N)]
        counts = [[0]*26 for _ in range(N)] #gives the max counts for each color with paths starting at node
        in_degree = [0]*N
        visited_nodes = 0 #we must be able to visit all nodes, otherwise there is a cycle
        
        for u,v in edges:
            graph[u].append(v)
            #u going into v
            in_degree[v] += 1
        
        #note these are not leaf nodes, but rather root nodes
        #we start from nodes that have nothing going into them, can only go out
        q = deque([])
        for i in range(N):
            if in_degree[i] == 0:
                q.append(i)
        
        while q:
            curr_node = q.popleft()
            curr_color = ord(colors[curr_node]) - ord('a')
            #update counts
            counts[curr_node][curr_color] += 1
            visited_nodes += 1
            
            for neigh in graph[curr_node]:
                for i in range(26):
                    #update the max frequency for the outgoing edges for this node
                    counts[neigh][i] = max(counts[curr_node][i],counts[neigh][i])
                #use up an edge
                in_degree[neigh] -= 1
                #no more edges, we q up
                if in_degree[neigh] == 0:
                    q.append(neigh)
                    
        if visited_nodes < N:
            return -1
        ans = 0
        for i in range(N):
            ans = max(ans,max(counts[i])) #we also could have maximized during kahns
        
        return ans

#########################################
# 20. Valid Parentheses (REVSITED)
# 10APR23
#########################################
class Solution:
    def isValid(self, s: str) -> bool:
        '''
        we need to use a stack
        push open exrpessions on to stack and try to pop them off when we get to a closing one
        '''
        stack = []
        mapp = {")": "(",
               "]":"[",
                "}":"{",
               }
    
        
        for ch in s:
            #add in opening
            if ch not in mapp:
                stack.append(ch)
            
            #otherwise we need to close and clear
            else:
                if stack and mapp[ch] != stack[-1]:
                    return False
                elif not stack:
                    return False
                else:
                    stack.pop()
        
        return not stack
            
#i thought this was also a cool way
class Solution:
    def isValid(self, s: str) -> bool:
        
        while ("()" in s) or ("{}" in s) or ("[]" in s):
            s = s.replace("()","")
            s = s.replace("{}", "")
            s = s.replace("[]","")
        
        return s == ""

#########################################
# 2332. The Latest Time to Catch a Bus
# 06APR23
#########################################
class Solution:
    def latestTimeCatchTheBus(self, buses: List[int], passengers: List[int], capacity: int) -> int:
        '''
        i have a list of buses with the times that they leave
        i have list of passengers with times that they arrive
        each bus can hold capacitty people (at most)
        
        when passenger arrives, they wait in line for next available bus
        
        i can get on bus that departs at time x if i arrive at time y, where y <= x, and bus is not full
        passengers with earliest arrival time get on bus first
        
        ground rules:
            i need to catch a bus
            my options can be between min(buses) and max(buses), if when i get there, there are not more than capacity people waiting
            at those times
            
            binary search, try to find a workable time??
            
        try all times between min(buses) and max(buses)
        if this time works, then anytime after should work 
        '''
        #sort
        buses.sort()
        passengers.sort()
        
        m = len(buses)
        n = len(passengers)
        
        #smallest ans can be at time 1
        ans = 1
        curr_cap = 0
        
        #pointer into passengers
        j = 0
        
        for i in range(m):
            #while there are passengers and curr cap is available and this passenger can tak
            #keep adding people to busses, there are two cases when we can have the latest time
            while j < n and curr_cap < capacity and passengers[j] <= buses[i]:
                #we could have several latest answers, so long as the current passenger - 1 is > the previous passenger in line
                #If one arrives 1 minute earlier than last added (given that the time slot is not occupied by an other passenger), he can get on the bus
                if passengers[j] - 1 != passengers[j-1]:
                    #we sorted increasingly, so keep getting the lastest answer
                    
                    #the one with the earllier time is the latest i can catch the base
                    #we can use the most recent passenger waiting for the bus, less one second
                    ans = passengers[j] - 1
                
                #advance in cinrement capacity
                j += 1
                curr_cap += 1
            
            #another time we can get is from the buses
            # If the last one added, his arrival is less than bus departure, check capacity, and if not full, we can arrive at the bus departure time.

            if curr_cap < capacity and (j == 0 or passengers[j-1] < buses[i]):
                ans = buses[i]
            
            #reset cap
            curr_cap = 0
        
        return ans
    
class Solution:
    def latestTimeCatchTheBus(self, buses: List[int], passengers: List[int], capacity: int) -> int:
        '''
        another solution just to hit the concept home
        https://leetcode.com/problems/the-latest-time-to-catch-a-bus/discuss/2259200/Python-Just-calculate-(faster-than-100.00)
        
        The line best = time if cap > 0 else passengers[cur - 1] is inside the for loop,
        If we have max capacity number of passengers before the i-th bus,
        The time we choose if we want to catch this bus is at max (passengers[cur - 1] -1). So we try to decrease from passengers[cur - 1].

        If we have empty slot for the best ,we should try to decrease from time, which the latester time to catch the i-th bus
        
        We have just exited the for loop, so time refers to the departure time of the bus that is last to leave, while cap refers to the remaining capacity on that bus.

So what the code is essentially saying is:

if there are still seats on that last bus, then all I need to do is to arrive the same time it leaves! (note that there may be existing people who are arriving at that time as well, so you can't return directly, e.g, [2], [2], 2).
if there are no more seats on the last bus, who is the last guy that managed to get a seat? (note that we still need to check for the above, because it is only when there is no more seat on the last bus, that this piece of information becomes valuable, e.g. [3], [2, 4], 2)
        '''
        buses.sort()
        passengers.sort()
        
        #print(buses)
        #print(passengers)
        
        m = len(buses)
        n = len(passengers)
        
        ptr_passengers = 0
        
        #try to fit as many people on a bus to find the absolute latest time we can leave
        for time in buses:
            curr_cap = capacity
            while ptr_passengers < n and passengers[ptr_passengers] <= time and curr_cap > 0:
                curr_cap -= 1
                ptr_passengers += 1
            
        #try to reduce best time so far
        #we either take the last bus, if we can fit on it
        #or we be the last person in passengers to take this bus
        best_time = time if curr_cap > 0 else passengers[ptr_passengers -1]
        
        #see if we can go one less?
        passengers = set(passengers)
        
        while best_time in passengers:
            best_time -= 1
        
        return best_time
        
        #print(time)

###################################
# 457. Circular Array Loop
# 10APR23
###################################
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        '''
        convert to graph and detect cycle
        cycle has a defintion
            for every index in a cycles, all numes[index] must be positive or all negative
            cycle must be greater than 1, i.e size of cycle = k, and k > 1
        '''
        N = len(nums)
        graph = defaultdict(list)
        
        for i in range(N):
            #positive direction
            if nums[i] >= 0:
                u = i
                v = (i + nums[i]) % N
                graph[u].append(v)
            else:
                #careful with negatives and their modulo
                u = i
                #still works with negatives, but we want the magnitude devoid of direction
                v = (i-(-1*nums[i])) % N
                graph[u].append(v)
        
        #cycle detection, with a specific defintino of cycle
        #keep track of size and counts positive and negatives
        
        def dfs(node,seen):
            if node in seen:
                return -1
            seen.add(node)
            for neigh in graph[node]:
                dfs(neigh,seen)
                
                
        seen = set()
        contains_cycle = None
        for i in range(N):
            contains_cycle = dfs(i,seen)
            if contains_cycle:
                return True
        
        return False
    

#close one 35/44
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        '''
        convert to graph and detect cycle
        cycle has a defintion
            for every index in a cycles, all numes[index] must be positive or all negative
            cycle must be greater than 1, i.e size of cycle = k, and k > 1
        '''
        N = len(nums)
        graph = defaultdict(list)
        
        for i in range(N):
            #positive direction
            if nums[i] >= 0:
                u = i
                v = (i + nums[i]) % N
                graph[u].append(v)
            else:
                #careful with negatives and their modulo
                u = i
                #still works with negatives, but we want the magnitude devoid of direction
                v = (i-(-1*nums[i])) % N
                graph[u].append(v)
        
        #cycle detection, with a specific defintino of cycle
        #keep track of size and counts positive and negatives
        
        def dfs(node,seen,size,count_pos,count_neg):
            seen.add(node)
            size[0] += 1
            if nums[node] > 0:
                count_pos[0] += 1
            elif nums[node] < 0:
                count_neg[0] += 1

            #no neigbors
            for neigh in graph[node]:
                if neigh not in seen:
                    dfs(neigh,seen,size,count_pos,count_neg)
                if neigh in seen:
                    return [size[0],count_pos[0],count_neg[0]]
            seen.remove(node)
        
        seen = set()
        for i in range(N):
            temp = dfs(i,seen,[0],[0],[0])
            size,pos,neg = temp
            if (size > 1 and pos == size) or (size > 1 and neg == size):
                return True
        
        return False
    
#we can use the cyclet detectino algorith, with three states, unvisite, visited, visited during recursion
#in addition, when making the graph, join edges with the same sign
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        '''
        convert to graph and detect cycle
        cycle has a defintion
            for every index in a cycles, all numes[index] must be positive or all negative
            cycle must be greater than 1, i.e size of cycle = k, and k > 1
        '''
        N = len(nums)
        graph = defaultdict(list)
        
        for i in range(N):
            u  = i
            v = i + nums[i]
		    #In case of x crosses the range [0,n-1]
            v = v % N 
			#Making sure self edges do not form and edges are formed only between elements of same sign
            if v != i and nums[u]*nums[v]>=0: 
                graph[u].append(v)

        

        #cycle detection, with a specific defintino of cycle
        #keep track of size and counts positive and negatives
        #in visited we need to keep three sateates
        #0 mean not visited yet, 1 means visited, -1 means visited on this current traversal
        # In visit list, 0 means unvisited, 1 means visited, -1 means we are currently recursing and encountered this element, 
	    # so we set -1 to 1 after completely searching through all possibilites from that element without finding a cycle else
	    # If we encounter an element with -1 value in visit, then it implies we have found a cycle.
        visited = [0]*N
        def dfs(curr,visited): #returns whether or not there is a cycle from this curr node
            if visited[curr] == -1:
                return True
            if visited[curr] == 1:
                return False
            #mark
            visited[curr] = -1
            for neigh in graph[curr]:
                if dfs(neigh,visited):
                    return True
            #complete
            visited[curr] = 1
            return False
        
        for i in range(N):
            if dfs(i,visited):
                return True
        
        return False 
    
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        '''
        we dont need to compute the graph before hand
        '''
        N = len(nums)
        visited = [0]*N
        def dfs(curr,visited): #returns whether or not there is a cycle from this curr node
            if visited[curr] == -1:
                return True
            if visited[curr] == 1:
                return False
            #mark
            visited[curr] = -1
            neigh = curr + nums[curr]
            neigh = neigh % N
            
            if neigh != curr and nums[neigh]*nums[curr] >= 0:
                if dfs(neigh,visited):
                    return True
            #complete
            visited[curr] = 1
            return False
        
        for i in range(N):
            if dfs(i,visited):
                return True
        
        return False 
            
#we can also do this iteratvely without using recursion, 
class Solution:
    def circularArrayLoop(self, nums: List[int]) -> bool:
        '''
        we dont need to compute the graph before hand
        but we need to different seen sets, one for the current traversal, and one to indicate whethr or not to start the traversal
        '''
        N = len(nums)
        visited = set()
        
        for i in range(N):
            curr = i
            if curr not in visited:
                local_visited = set()
                while True:
                    if curr in local_visited:
                        return True
                    if curr in visited:
                        break
                    #mark both
                    visited.add(curr)
                    local_visited.add(curr)
                    neigh = (curr + nums[curr]) % N
                    if neigh != curr and nums[neigh]*nums[curr] >= 0:
                        curr = neigh
                    else:
                        break
        
        return False
                    
#cycle detection, slow and fast pointers, 
#note, doesn't pass all cases
class Solution:
    
    def get_next(self,i: int, nums: List[int],):
        #moves pointer ahead one interation
        N = len(nums)
        #first advance
        i += nums[i]
        #if we had gone in the reverse direction
        if i < 0:
            i += N
        #gone out side the range after adding N
        elif (i > N - 1):
            i %= N
        
        return i
            
    def circularArrayLoop(self, nums: List[int]) -> bool:
        '''
        we can use a slow and faster pointers, and return true if the cycle lenght is greater than one
            i.e fast == slow
        if we meetn element with different directions, then the search fails
        while traveling we set all elemnts along the way to seen
        '''
        if not nums or len(nums) < 2:
            return False
        
        N = len(nums)
        #check eery possbiles tart locations, we can find a short loop, but the array may have a valid loop
        for i in range(N):
            #element sin vsited are known non loop paths, so if we've seen this, we know it doesnt have a lop
            if nums[i] == 0:
                continue
            #otherwise its not visited or we don't know it it has a loop from here,
            #we need to find out!
            slow = i
            fast = self.get_next(slow,nums)
            
            #whether i is positive or negative defines our direction, so i
            #we just need to make sure the signs are the same when advancing
            while (nums[i]*nums[fast] > 0) and (nums[i]*nums[self.get_next(fast,nums)] > 0): #same sign
                if slow == fast:
                    #one elemnt loop check
                    if (slow == self.get_next(slow,nums)):
                        break
                    return True
                
                #advance
                slow = self.get_next(slow,nums)
                fast = self.get_next(self.get_next(fast,nums),nums)
                
            #if we are here we didn't find a loop, so we know this path doesn't have a loop
            #so we retravesre it untilwe reverse directions or ecnounts it in loops
            #during the traverse add to seen
            slow = i
            sign = nums[i]
            while (sign*nums[slow] > 0):
                temp = self.get_next(slow,nums)
                nums[slow] = 0
                slow = temp
        
        return False
    
class Solution:
    def __init__(self):
        self.__visited = lambda x: not x # a cell i is visited when nums[i] = 0
        
    def __next(self, nums, idx, direction):
        if idx == -1: # To handle the case of next(next(fast)) = next(-1) = -1
            return -1

        elif (nums[idx] > 0) != direction: # check the direction
            return -1

        next_idx = (idx + nums[idx]) % len(nums)
        if next_idx < 0:
            next_idx += len(nums)

        return -1 if next_idx == idx else next_idx

    def circularArrayLoop(self, nums: List[int]) -> bool:

            for i in range(len(nums)):
                if self.__visited(nums[i]):
                    continue

                direction = nums[i] > 0

                # 1. Check if there is a cycle starting from i
                slow = fast = i
                while not (self.__visited(nums[slow]) or self.__visited(nums[fast])):

                    slow = self.__next(nums, slow, direction)
                    fast = self.__next(nums, self.__next(nums, fast, direction), direction)

                    if slow == -1 or fast == -1:
                        break

                    elif slow == fast:
                        return True

                # 2. Mark visited all cells that belong to the path starting from i
                slow = i
                while self.__next(nums, slow, direction) != -1:
                    nums[slow], slow = 0, self.__next(nums, slow, direction)

            return False

##################################
# 2390. Removing Stars From a String
# 11APR23
##################################
#stack
class Solution:
    def removeStars(self, s: str) -> str:
        '''
        we have a string s, which contains *
        in one operation:
            choose a star in s
            remove the closes non=start char to its left, as well as the star itself
        
        return the string after all stars have been removed
        that is i use a stack, then just pop the top of the stack on the stars
        
        [l,e,c,o,e]
        '''
        stack = []
        for ch in s:
            if ch != '*':
                stack.append(ch)
            else:
                if stack:
                    stack.pop()
        
        return "".join(stack)
    
#on strings
class Solution:
    def removeStars(self, s: str) -> str:
        '''
        we also don't need to use a stack, we can just keep two pointers,
        one on the string, and one for the char to take
        we are essential overwriting the s input
        '''
        s_list = list(s)
        
        j = 0
        for ch in s:
            if ch == '*': # we cant use what j i currently pointing to
                j -= 1
            else:
                #we can use current j and move up
                s_list[j] = ch
                j += 1
    
        #grab the firstt j chars
        ans = ""
        for i in range(j):
            ans += s_list[i]
        
        return ans

###########################
# 464. Can I Win
# 12APR23
###########################
#jesus fuck...
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        '''
        in the original game 100 two players take turns adding intergs from to 10 
        first person to reach of exceed 100 wins (not we can replace numbers)
        
        what if we cannot re-use inters?
        
        return true if the first person can win assuming both play optimally
        dp on bit masks to show what numbers haven't been taken 
        states:
            (p1_score, p2_score,nums_available_to_take)
            
        base case
            
        
        '''
        memo = {}
        
        def dp(p1,p2,nums_to_take):
            if p1 == desiredTotal:
                return True
            
            if p2 > desiredTotal or p2 > desiredTotal:
                return False
            if (p1,p2,nums_to_take) in memo:
                return memo[(p1,p2,nums_to_take)]
            
            #player 1
            player_1_number = 0
            for j in range(maxChoosableInteger):
                num = 1 << j
                #if we haven't taken it
                if nums_to_tak & num == 0:
                    nums_to_take = nums_to_take |  num
                    player_1_number = j
                    break
            
            #player 2
            player_2_number = 0
            for j in range(maxChoosableInteger):
                num = 1 << j
                #if we haven't taken it
                if nums_to_tak & num == 0:
                    nums_to_take = nums_to_take |  num
                    player_2_number = j
                    break
                    
            ans = dp()

#finally
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        '''
        dp states:
            (current score, who's move, numbers taken)
            we try all moves and backtrack in the process
            
            base case, when desied total <= 0
        easy edge cases
            if sum of digits < desiredTotal, no one can win
            if desiredTotal <= maxChooseableInterger, player one just picks on the first move
            
        
            for currs_score, if we get to zero, it means the other peron must have won on the previous turn
        '''
        if desiredTotal <= maxChoosableInteger:
            return True
        #get sum using Gauss trick
        if ((maxChoosableInteger)*(1+maxChoosableInteger)) / 2 < desiredTotal:
            return False
        
        memo = {}
        
        def dp(curr_score,curr_move,nums_taken):
            if curr_score <= 0:
                return False
            
            if (curr_score,curr_move,nums_taken) in memo:
                return memo[(curr_score,curr_move,nums_taken)]
            
            #player 1
            if curr_move == 0:
                for i in range(1,maxChoosableInteger+1):
                    #check if taken
                    if nums_taken & (1 << i):
                        continue
                    #otherwise take
                    nums_taken = nums_taken | (1 << i)
                    
                    #can't win from here
                    if not dp(curr_score - i,1,nums_taken):
                        memo[(curr_score,curr_move,nums_taken)] = True
                        #put the number back
                        nums_taken = nums_taken ^ (1 << i)
                        return True
                    
                nums_taken = nums_taken ^ (1 << i)
                memo[(curr_score,curr_move,nums_taken)] = False
                return False
            
            #other players move, but swap
            else:
                for i in range(1,maxChoosableInteger+1):
                    #check if taken
                    if nums_taken & (1 << i):
                        continue
                    #otherwise take
                    nums_taken = nums_taken | (1 << i)
                    
                    #can't win from here
                    if not dp(curr_score - i,0,nums_taken):
                        memo[(curr_score,curr_move,nums_taken)] = True
                        #put the number back
                        nums_taken = nums_taken ^ (1 << i)
                        return True
                    
                nums_taken = nums_taken ^ (1 << i)
                memo[(curr_score,curr_move,nums_taken)] = False
                return False
            

        
        return dp(0,0,1 << 22)
                

class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        '''
        player move is a redudant, we dont need it
            
            
        '''
        if desiredTotal <= maxChoosableInteger:
            return True
        #get sum using Gauss trick
        if ((maxChoosableInteger)*(1+maxChoosableInteger)) / 2 < desiredTotal:
            return False
        
        memo = {}
        
        def dp(curr_score,nums_taken):
            if curr_score <= 0:
                return False
            
            if (curr_score,nums_taken) in memo:
                return memo[(curr_score,nums_taken)]
            

            for i in range(1,maxChoosableInteger+1):
                #check if taken
                if (nums_taken & (1 << i)) == 0:
                    nums_taken = nums_taken | (1 << i)
                    #can't win from here
                    if not dp(curr_score - i,nums_taken):
                        memo[(curr_score,nums_taken)] = True
                        nums_taken = nums_taken ^ (1 << i)
                        return True
            nums_taken = nums_taken ^ (1 << i)
            memo[(curr_score,nums_taken)] = False
            return False
            
            
        return dp(0,1 << 22)
    
#this is a better solution
class Solution:
    def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
        
        memo = {}
        def can_player_win_on(target, taken):
            
            # Input: target number, current state of available call numbers for current player
            
            # Ouput: Return True if current player can win with given parameter.
            #        Otherwise, return False.

            if target <= 0:
				## Base case:
                # Opponent has reach target and won the game on previous round.
                return False
            
            if (target,taken) in memo:
                return memo[(target,taken)]
            
			## General cases:
            # Current player use all available call number, and try to make a optimal choice
            for number in range(1, maxChoosableInteger+1):
                
                if (taken & (1 << number)): 
                    # Players cannot reuse the same call number, defined by game rule
                    continue

                # opponent makes next optimal move after current player
                # update target and bitflag for opponent
                if not can_player_win_on(target - number, taken | (1 << number) ):
                    
                    # current player can win if opponent lose
                    memo[(target,taken)] = True
                    return True

            # current player lose, failed to find a optimal choice to win
            memo[(target,taken)] = False
            return False
        
        
        # total number sum = 1 + 2 + ... + max call number = n * (1 + n) // 2
        S = maxChoosableInteger * (maxChoosableInteger+1) // 2
        
        if S < desiredTotal:
            # max call number is too small, can not reach desired final value
            return False
        
        elif desiredTotal <= 0:
            # first player already win on game opening
            return True
        
        elif S == desiredTotal and maxChoosableInteger % 2 == 1:
            # first player always win, because she/he can choose last remaining number from 1 ~ maxChoosableInteger on final round
            return True
        
        return can_player_win_on(desiredTotal, 0 )
    
############################################
# 946. Validate Stack Sequences (REVISTED)
# 13APR23
###########################################
#sheesh, i tried too hard on this one
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        '''
        simulate and see if we can clear both pushed and popped stacks
        reverse both
        [5,4,3,2,1]
        [1,2,3,5,4]
        
        
        '''
        pushed = pushed[::-1]
        popped = popped[::-1]
        
        temp = []
        
        while pushed and popped:
            #empty temp or we can't pop
            while len(temp) == 0 or (len(pushed) > 0 and popped[-1] != temp[-1]):
                temp.append(pushed.pop())
            
            #keep popping
            while len(temp) > 0 and len(popped) > 0 and temp[-1] == popped[-1]:
                popped.pop()
                temp.pop()
                
        
        return len(pushed) == 0 and len(popped) == 0
            

#####################################
# 888. Fair Candy Swap
# 13APR23
#####################################
class Solution:
    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        '''
        find sums of alice and bob,
        then for each size in alice, decrement it by it size, and try to find bobs new size to swap
        so that their sums are equal
        
        let S_A and S_B be the sums of alice and bob
        let x and y be the sizes we can swap
        S_A - x + y = S_B - y + x
        2*(x-y) = (S_A - S_B)
        x - y = (S_A - S_B) / 2
        x = y + (S_A - S_B) / 2
        '''
        alice_total = sum(aliceSizes)
        bob_total = sum(bobSizes)
        
        #sort
        aliceSizes.sort()
        bobSizes.sort()
        
        for num in aliceSizes:
            #store the swap for alice
            alice_swap = num
            left = 0
            right = len(bobSizes)
            while left < right:
                mid = left + (right - left) // 2
                bob_swap = bobSizes[mid]
                #update both
                alice_change = alice_total - alice_swap + bob_swap
                bob_change = bob_total - bob_swap + alice_swap
                #bigger
                if alice_change >= bob_change:
                    #this swapping value was too big or it worked
                    right = mid
                else:
                    left = mid + 1
            
            #checks
            #boundary condition
            if left == len(bobSizes):
                left -= 1
            #check valid swap
            if alice_total - alice_swap + bobSizes[left] == bob_total - bobSizes[left] + alice_swap:
                return [alice_swap,bobSizes[left]]
            
        
        #no answer, but we can't be here anway because there is one
        return -1
    
class Solution:
    def fairCandySwap(self, aliceSizes: List[int], bobSizes: List[int]) -> List[int]:
        '''
        find sums of alice and bob,
        then for each size in alice, decrement it by it size, and try to find bobs new size to swap
        so that their sums are equal
        
        let S_A and S_B be the sums of alice and bob
        let x and y be the sizes we can swap
        S_A - x + y = S_B - y + x
        2*(x-y) = (S_A - S_B)
        x - y = (S_A - S_B) / 2
        x = y + (S_A - S_B) / 2
        
        fix alices scores as x, then find y in bobs scores, using binarys search
        this is an important paradigm
            find equation, fix one varibale then vary the other
            
        '''
        alice_total = sum(aliceSizes)
        bob_total = sum(bobSizes)
        
        diff = (alice_total - bob_total) / 2

        #sort
        aliceSizes.sort()
        bobSizes.sort()
        
        for num in aliceSizes:
            target = num + diff
            left = 0
            right = len(bobSizes)
            while left < right:
                mid = left + (right - left) // 2
                #bigger
                if bobSizes[mid] >= target:
                    #this swapping value was too big or it worked
                    right = mid
                else:
                    left = mid + 1
            print(left,right,target)
            if left == len(bobSizes):
                left -= 1
            if target == bobSizes[left]:
                return [num,bobSizes[left]]
                
        
        return -1
                
#######################################
# 516. Longest Palindromic Subsequence
# 14APR23
#######################################
#fuckkkk
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        '''
        keep in mind this is a subsequence, not a substring
        let dp(i,j) be the answer to the longest subsequence using s from i to j
        
        if i == j, we just have the single char, so return 1
        
        say we are trying to compute dp(i,j)
        we can extend this to j+1 or we can skeep and extend to j+2
        if j+1 == i, retreieve the previous answer and increment
        do the same with j+2
        if they are different, there is no valid subsequence
        try all i 
        '''
        memo = {}
        N = len(s)
        
        def dp(i,j):
            if i == j:
                return 1
            if j > 0:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            
            option1 = 0
            option2 = 0
            if s[i] == s[j]:
                option1 = 1 + dp(i,j-1)
                
            if s[i] == s[j-1]:
                option2 = 1 + dp(i,j-2)
            
            elif s[i] != s[j]:
                memo[(i,j)] = 0
                return 0
                
            ans = max(ans,option1,options2)
            memo[(i,j)] = ans
            return ans
        
        #try all i
        ans =  0
        for i in range(N):
            for j in range(i+1,N):
                ans = max(ans,dp(i,j))
        
        return ans
    
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        '''
        keep in mind this is a subsequence, not a substring
        let dp(i,j) be the answer to the longest subsequence using s from i to j
        
        if i == j, we just have the single char, so return 1
        
        say we are trying to compute dp(i,j)
        we can extend this to j+1 or we can skeep and extend to j+2
        if j+1 == i, retreieve the previous answer and increment
        do the same with j+2
        if they are different, there is no valid subsequence
        try all i 
        '''
        memo = {}
        N = len(s)
        
        def dp(i,j):
            if i == j:
                return 1
            if i > j:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            
            #we can extent
            if s[i] == s[j]:
                ans = 2 + dp(i+1,j-1)
                memo[(i,j)] = ans
                return ans
            else:
                #move one or the other and take the max
                first = dp(i+1,j)
                second = dp(i,j-1)
                ans = max(first,second)
                memo[(i,j)] = ans
                return ans
        
        #try all i
        ans =  0
        for i in range(N):
            for j in range(i,N):
                ans = max(ans,dp(i,j))
        
        return ans
    
#this is just knapsack
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        '''
        if s[i] == s[j], then we can extend the answer dp(i+1,j-1) by 2
        if they aren't the same we need to the max of dp(i+1,j) or dp(i,j-1)
        '''
        memo = {}
        N = len(s)
        
        def dp(i,j):
            if i == j:
                return 1
            if i > j:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            
            #we can extent
            if s[i] == s[j]:
                ans = 2 + dp(i+1,j-1)
                memo[(i,j)] = ans
                return ans
            else:
                #move one or the other and take the max
                first = dp(i+1,j)
                second = dp(i,j-1)
                ans = max(first,second)
                memo[(i,j)] = ans
                return ans
        
        #try all i
        return dp(0,N-1)

#bottom up
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        '''
        translating to bottom up
        '''
        N = len(s)
        dp = [[0]*(N+1) for _ in range(N+1)] #N+1 becase there are N states we need to index into, and there are N-1 index positions
        #base cases
        for i in range(N):
            for j in range(N):
                if i == j:
                    dp[i][j] = 1
                if i > j:
                    dp[i][j] = 0
        
        
        #fill dp array starting from bottom right
        for i in range(N-1,-1,-1):
            #i already have bases cases for when i == j, so don't start at i
            for j in range(i+1,N):
                if s[i] == s[j]:
                    dp[i][j] = 2 + dp[i+1][j-1]
                else:
                    first = dp[i+1][j]
                    second = dp[i][j-1]
                    dp[i][j] = max(first,second)
                    
        
        return dp[0][N-1]
    
##########################################
# 2218. Maximum Value of K Coins From Piles
# 15APR23
###########################################
#damn it, close one
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        '''
        each list in piles is a stack of coins in a stack ordered from top to bottom
        we are given an integer k
        we want to take from any of the piles to maximize our total value using k transactions
        dp obvie
        states:
            when k hits 0, we can't take anything, so return 0
        
        hint 1, 
            for each i pile, what will be the total value of coins we can collect if we choose the first j coins
        
        hint 2,
            dp, DUH, combine results from different piles in the most optimal mannder
            we need the largest prefix sum from all the piles!
            
        lets say for example i don't what the state is
        (state), but in this state i have the maximum possible value so far
        how do i know what coin to take next?
        well, i want the maximum coin avilable that i can take!
        in order to answer this, i would need to know what coins are currently at the top of the pile
        (state) = know coins at the tops of the pile
        then take the max and move that state!
        
        let state be tuple of pointers, initally all are are zero
        and keeep track of k


        this is no different the brute force really, albeit it's not totally right
        in order the state to work, i would have needed to already be at a maximum, but that isn't the case

        '''
        memo = {}
        N = len(piles)
        
        def dp(state,k):
            #no more k
            if k == 0:
                return 0
            if (state,k) in memo:
                return memo[(state,k)]
            
            #unpack states first
            unpacked_state = list(state)
            #find the max coin
            curr_max,pile_of_max = 0,0
            #local max
            local_max = 0
            for i in range(N):
                #bounds condition in pile, can't take coin
                if unpacked_state[i] == len(piles[i]):
                    continue
                #new max
                
                if piles[i][unpacked_state[i]] > curr_max:
                    #store max and pointer to pile
                    curr_max = piles[i][unpacked_state[i]] 
                    pile_of_max = i

                    #move pointer to get new state
                    unpacked_state[pile_of_max] += 1
                    local_max = max(local_max,curr_max + dp(tuple(unpacked_state),k-1))
            #cache
            memo[(state,k)] = local_max
            return local_max
        
        starting = tuple([0]*N)
        
        return dp(starting,k)
            
            
#dp, top down
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        '''
        let dp(i,k) be the answer to getting the max value using k coins
        using the piles[:i]
        
        for example if dp(4,7) is the max total value when one takes at most seven coins using piles[:4]
        if the 4 piles have seven coins, we just take them all
        
        base case is when i == 0, no piles to take from
        and when k == 0, no more coins are allowed to be taken
        
        since we are optimal at dp(i,k), we look back at dp(i-1,k-c) for c being in coins used up in the range [0,k]
        
        one may not take any coins from the (i-1) pile and take at most coins from th leftmost (i-1) piles, 
            i.e don't take from the i-1 pile and take the from the remaining piles
        
        rather we take c coins from i-1 and k-c from the rest
            where c and be in range [0,k]
        
        of these choices, we maximuze on each state
        so dp(i,k) = max{
            dp(i-1,k-c) for c in range(0,k)
        }
        
        when we choose some number of coins (c) to take from the i-1 pile, we must optimall choose at most coins - c
        from the rest of ht epiles
        
        on the current i-1 pile, we keep tracking to take a coins from this pile and incremnt a runing sum
        
        When the value of currentCoins is optimal, dp[i][coins] = dp[i - 1][coins - currentCoins] + currentSum, 
        because dp[i - 1][coins - currentCoins] gives the optimal answer to the smaller subproblem of size i - 1.
        
        There are two constraints for currentCoins: first, one cannot take more coins from the (i - 1)-th pile than the amount of coins the pile has (piles[i - 1].length); 
        and second, we cannot take more coins than we are allowed, so currentCoins must not exceed coins.
        
        Combining these two constraints, one concludes that all values of currentCoins between 0 and min(piles[i - 1].length, coins) inclusively are feasible. 
        We try all these values to find the optimal one.
        '''
        memo = {}
        N = len(piles)
        
        def dp(i,k):
            if i == 0:
                return 0
            if k == 0:
                return 0
            if (i,k) in memo:
                return memo[(i,k)]
            curr_sum = 0
            ans = 0
            #keep trying take coins adding to the previous problem
            upper_bound = min(len(piles[i-1]),k)
            for current_coins in range(upper_bound+1):
                #if we are allowed to take
                if current_coins > 0:
                    curr_sum += piles[i-1][current_coins-1]
                    
                ans = max(ans, dp(i-1,k-current_coins) + curr_sum)
            
            memo[(i,k)] = ans
            return ans
        
        
        return dp(N,k)
    
#dp bottom up
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        '''
        bottom up translation
        '''
        N = len(piles)
        
        dp = [[0]*(k+1) for _ in range(N+1)]
        
        #we alredy have base cases filled out, so start from i = 1 and k = 1
        #finally!
        for i in range(1,N+1):
            for j in range(k+1):
                curr_sum = 0
                ans = 0
                
                upper_bound = min(len(piles[i-1]),j)
                for current_coins in range(upper_bound+1):
                    #if we are allowed to take
                    if current_coins > 0:
                        curr_sum += piles[i-1][current_coins-1]

                    ans = max(ans, dp[i-1][j-current_coins]+ curr_sum)
                
                dp[i][j] = ans

        return dp[N][k]

#another recursive way
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        '''
        here's another way
        think of this is not taking from a pile
        for keep trying to take from a pile
        '''
        N = len(piles)
        memo = {}
        
        def dp(i,k):
            if i < 0 or k == 0:
                return 0
            if (i,k) in memo:
                return memo[(i,k)]
            
            #we can either not take, which is always an option
            not_take = dp(i-1,k)
            #we need to see how many times we can take from this pile i
            coins_we_can_take = min(k,len(piles[i]))
            
            #store sum we can take from this current pile
            curr_sum = 0
            local_ans = 0 #the anwer for this subproblem that we are trying to maxmize
            for taken_coins in range(coins_we_can_take):
                curr_sum += piles[i][taken_coins]
                #maximiz
                local_ans = max(local_ans,not_take, dp(i-1,k-taken_coins-1)+curr_sum)
            
            memo[(i,k)] = local_ans
            return local_ans
        
        
        return dp(N-1,k)
    

#using prefix sum, in the search for fiding the maximum answer, we are accumulating sums
#we can precompute the sum of the piles by generating prefix sums
class Solution:
    def maxValueOfCoins(self, piles: List[List[int]], k: int) -> int:
        '''
        we can use prefix sum on piles to get the sum at any index in piles[i] for i in range(len(piles))
        '''
        N = len(piles)
        memo = {}
        
        #precomputes piles running sums,
        #store in temp array
        pref_sum_piles = []
        for i in range(N):
            pref_sum = [piles[i][0]]
            for j in range(1,len(piles[i])):
                pref_sum.append(pref_sum[-1] + piles[i][j])
            
            pref_sum_piles.append(pref_sum)
            

        
        def dp(i,k):
            if i < 0 or k == 0:
                return 0
            if (i,k) in memo:
                return memo[(i,k)]
            
            #we can either not take, which is always an option
            not_take = dp(i-1,k)
            #we need to see how many times we can take from this pile i
            coins_we_can_take = min(k,len(pref_sum_piles[i]))
            
            #store sum we can take from this current pile
            local_ans = 0 #the anwer for this subproblem that we are trying to maxmize
            for taken_coins in range(coins_we_can_take):
                curr_sum = pref_sum_piles[i][taken_coins]
                #maximiz
                local_ans = max(local_ans,not_take, dp(i-1,k-taken_coins-1)+curr_sum)
            
            memo[(i,k)] = local_ans
            return local_ans
        
        return dp(N-1,k)


###################################################################
# 1639. Number of Ways to Form a Target String Given a Dictionary
# 16APR23
####################################################################
#brute force
#MAIN TAKEWAY, when think of number of ways, this of the product rule for counting the number of ways to do something
class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        '''
        given list of words, want to make target using the following words
            1. target should be formed from left to right
            2. to form the ith char of target, you can choose a character k from any string in wrods if
                target[i] = words[j][k]
            3. once you use the kth char of words[j] i can no loger use the xth char of any string in words
            wher x<=k
                i.e all the charcters ti the left of index k become unusable for every string
            4. repeat until we get target
        
        can use multiple character from the same string in words
        return number of way to form target to words
        
        graph problem,
            number of way is number of paths
            
        all words have the same length
        picking a word[j] and using characater word[j][k], means we can't use any character used at index <= k
        words = [
        acca
        bbbb
        caca
        ]
        
        brute force is to use dfs, keep advancing pointeers until we get to the end
        need pointer to index we are currently on in path
        for each index, check all words in the dictionary
        '''
        #do brute force first
        num_words = len(words)
        N = len(words[0])
        M = len(target)
        self.ways = 0
        mod = 10**9 + 7
        
        def rec(start_pos, pos_in_target,path):
            #got to the end and have made target, abandon path and increment
            if path == target:
                self.ways += 1
                self.ways %= mod
                return
            #no more chacters to use, all words have the same size
            if start_pos == N:
                return
            
            #loop over words
            for other_word in range(num_words):
                #loop over positions
                for next_pos in range(start_pos,N): #from the starting position to the size of words 
                    if words[other_word][next_pos] == target[pos_in_target]:
                        #recurse
                        rec(next_pos + 1, pos_in_target + 1, path+words[other_word][next_pos])
        
        rec(0,0,"")
        return self.ways % mod
    
#better brute force, use pointers into strings and memoize pointers as states
class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        '''
        adding memoe and storing string stats with expression
        '''
        #do brute force first
        num_words = len(words)
        N = len(words[0])
        M = len(target)
        memo = {}
        mod = 10**9 + 7
        
        def dp(start_pos, pos_in_target,):
            #got to the end and have made target, abandon path and increment
            if pos_in_target == M:
                return 1
            #no more chacters to use, all words have the same size
            if start_pos == N:
                return 0
            #retreive
            if (start_pos,pos_in_target) in memo:
                return memo[(start_pos,pos_in_target)]
            
            #answer for this tree is sum of all other states
            num_ways = 0
            #loop over words
            for other_word in range(num_words):
                #loop over positions
                for next_pos in range(start_pos,N): #from the starting position to the size of words 
                    if words[other_word][next_pos] == target[pos_in_target]:
                        #recurse
                        num_ways += dp(next_pos + 1, pos_in_target + 1)
                        num_ways %= mod
            
            #cache
            memo[(start_pos,pos_in_target)] = num_ways
            return num_ways
        
        return dp(0,0)
    
#using count map to get frequency counts for char at each column
class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        '''
        improving the time complextiy for solving the subproblems in DP is a hard on hard problem
        but it wont pass unless we can reduce the O(num_words*num_positions)
        remember this paradigm on hards going forward
        we need to store a count mapp, where counts[c,j] stores the count of character c in the jth column of the matrix

        An example:
        target:
        abc
        input words:
        1.a...
        2.a...
        3.a...
        4.a...
        ...
        100.a...

        more formally
        let dp(i,j) be the number of ways we can build prefix of target of length i using the words[:j] #prefix dp
        base case, when j == 0, there are no columns and so we can't make any prefix if taget
        rather there is only '1' way to get the null string, so return 1
        
        state dp(i,j), for j < k means that
            we are not considering the jth column of the mateix
            if i < m, we need to add character target[i] to his current prefix
            
        we also store counts
        counts(c,j) stores the frequency of character c at this column j
        when we chose a character target[i] from the jth column and add it to the prefix (which has length i) it now becomes i + 1
        after this we cannot use the jth column anymore and move on the j+1 column
        so we can move from dp(i,j) to dp(i+1,j+1)
        
        how many ways can move in this transition?
        its just the number of times target[i] appears at column j!
        dp(i+1,j+1) = counts(target[i],j)*dp(i,j)
        
        when we skip the jth column, we do not need to add anything to the current prefix. why?
        because there isn't a valid way to get here anyway through target[i]
        i.e there isn't a transisiont, so just move the current number of ways to the next state
        move dp(i,j) to dp(i+1,j)
        rather, it represents ONE way to not choose any character fro this column j
        
        time complexit is O(n*k + m*k)
        
        '''
        num_words = len(words)
        N = len(words[0])
        M = len(target)
        memo = {}
        mod = 10**9 + 7
        
        #pre process data by counting the counts for each char at each column j
        count_chars = Counter() 
        for word in words:
            for col,ch in enumerate(word):
                count_chars[(col,ch)] += 1
        
        
        def dp(start_pos, pos_in_target,):
            #got to the end and have made target, abandon path and increment
            if pos_in_target == M:
                return 1
            #no more chacters to use, all words have the same size
            if start_pos == N:
                return 0
            #retreive
            if (start_pos,pos_in_target) in memo:
                return memo[(start_pos,pos_in_target)]
            
            #skip this current word's index
            num_ways = dp(start_pos+1,pos_in_target)
            #get counts by multiplying multiplicity of next answer
            curr_char = target[pos_in_target]
            #to this current number of ways we try to increment by the number of ways we can make a path from start_pos to start_pos + 1
            num_ways += dp(start_pos+1,pos_in_target+1)*count_chars[(start_pos,curr_char)]
            num_ways %= mod
            
            #cache
            memo[(start_pos,pos_in_target)] = num_ways
            return num_ways
        
        return dp(0,0) % mod

#bottom up
class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        '''
        bottom up
        '''
        #do brute force first
        num_words = len(words)
        N = len(words[0])
        M = len(target)
        memo = {}
        mod = 10**9 + 7
        
        #pre process data by counting the counts for each char at each column j
        count_chars = Counter() 
        for word in words:
            for col,ch in enumerate(word):
                count_chars[(col,ch)] += 1
        
        dp = [[0]*(M+1) for _ in range(N+1)]
        
        #fill in base cases
        for start_pos in range(N+1):
            for pos_in_target in range(M+1):
                if pos_in_target == M:
                    dp[start_pos][pos_in_target] = 1
        
        
        for start_pos in range(N-1,-1,-1):
            for pos_in_target in range(M-1, -1,-1):
                #skip this current word's index
                num_ways = dp[start_pos+1][pos_in_target]
                #get counts by multiplying multiplicity of next answer
                curr_char = target[pos_in_target]
                #to this current number of ways we try to increment by the number of ways we can make a path from start_pos to start_pos + 1
                num_ways += dp[start_pos+1][pos_in_target+1]*count_chars[(start_pos,curr_char)]
                num_ways %= mod
                
                dp[start_pos][pos_in_target] = num_ways
        
        return dp[0][0] % mod
    
#########################################
# 1548. The Most Similar Path in a Graph
# 17APR23
#########################################
class Solution:
    def mostSimilar(self, n: int, roads: List[List[int]], names: List[str], targetPath: List[str]) -> List[int]:
        '''
        we have an undirected graph with n-1 nodes
        we are given the names of nodes in order from 0 to n-1
        we are given a string array targetPath 
        need to find a path in graph of same length and with minimum edit distance to targetpath
        
        define editDist as {
            targetPath = List[string]
            myPath = List[string]
            
            if len(targetPath) != len(myPath):
                return 10000000000 #or just a really big number
                
            dist = 0
            for i in range(len(myPath)):
                of targetPath[i] != myPath[i]:
                    dist += 1
                    
            return 1
        }
        
        basically just count the differentes at each position in the path
        hint:
            dp(i,j) represents the min edit distance for the path starting at node i and compared to index j of the targetPath
            we are at a minimum here, for any neighbor of i, the out going edge lead to state dp(neigh,j+1)
            take the minimum of all neighbors
            i.e let min_cost be the min_cost for dp(i,j):
                for all neighbors in graph[i]:
                    if the nodes differe cost += 1
                min(all choices)
        https://leetcode.com/problems/the-most-similar-path-in-a-graph/discuss/799773/Python-Top-down-and-bottom-up-DP-both-commented
        1. build graph from roads
        2. let dp(i,u) return a tuple (min_edit_dsit, best_sub_path)
            where the this is the min_edit dist up to targetPath[:i]
            bust subpath is some path that is the min edit distance so far, that differs minimally by 'edit_distance_nodes'
        3. u is the i-1th node we just chose immediately before the best sub_path
        4. dp(i,u) = dp(i+1,v) + (1 if names[v] != targetPath[i] else 0)
        
        '''
        graph = defaultdict(list)
        for u,v in roads:
            graph[u].append(v)
            graph[v].append(u)
            
        memo = {}
            
        def dp(i,u):
            #base case, when we reach target length, return 0 and empty path
            if i == len(targetPath):
                return 0,[]
            #cache
            if (i,u) in memo:
                return memo[(i,u)]
            
            #the min cost and subpath we are currently trying to find
            #i.e we are trying to minimize the min cost, and find the best path with min_cost
            #the dist function in this case is just the min_edit_distance function
            min_cost = float('inf')
            best_path = []
            
            
            #if 'u' is the node we've just chosen fro the bestt path, we then simply add on to this best_path for some neighbor v of u
            #since we dont which one of the neighbors of u gives the min difference, we need to try them all
            #if u is None, ti means weve not chosen a previous node, so we need to try all n nodes
            #this mean v is going to the frist node in the best path
            #we'll have to start with every node in the graph and see who leads to the best overall path
            if u == None:
                neighbors = [i for i in range(n)]
            else:
                neighbors = graph[u]
                
            for v in neighbors:
                child_cost,child_path = dp(i+1,v)
                child_cost += names[v] != targetPath[i] #increment 1 if different, else no difference
                
                #update
                if child_cost < min_cost:
                    min_cost = child_cost
                    best_path = [v] + child_path
                    
            memo[(i,u)] = (min_cost,best_path)
            return (min_cost,best_path)
        
        cost,path = dp(0,None)
        return path
                
#bottom up
class Solution:
    def mostSimilar(self, n: int, roads: List[List[int]], names: List[str], targetPath: List[str]) -> List[int]:
        '''
        bottom up, but this time the recurrences id dp(i) = d(i-1) and return dp(len(targetPath-1))
        we could have also made is so that we we still return dp(0), but we need to start at len(targetPath-1)
        
        define dp[i][u] to be the min edit distance of the best path in graph with i+1 nodes
        where u is the -th node in the best path found so far
        dp[i][u] = dp[i-1][v] + (1 if names[u] != targetPath[i] else 0) for all v coming out of u
        '''
        adj = collections.defaultdict(list)
        for e in roads:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
            
        dp = [[0]*n for _ in range(len(targetPath))]
        #base cases, initlize with 1 is there is match with the targetPath
        for v in range(n):
            #min edit distance for i == 0, can only be 1 or 0 depending on the first node we take
            #and if it maches targetPath[0]
            #store min edit distance and current node in path, simialr to prev pointer so we can rebuild path
            dp[0][v] = (1,v) if names[v] != targetPath[0] else (0,v)
            
        #traversal, one away from base case
        for i in range(1,len(targetPath)):
            #check all
            for v in range(n):
                #inital ansswer is empty and has highest min cost
                dp[i][v] = (float('inf') , "")
                #use is the prev hop on min edit dist path (best path) to v
                for u in adj[v]:
                    #update cost
                    cost = dp[i-1][u][0] + (1 if names[v] != targetPath[i] else 0)
                    if cost < dp[i][v][0]:
                        dp[i][v] = (cost,u)
        
        #find min_edit distance path, as well as the last pointer inpath
        min_cost = float('inf')
        best_hop = None
        for u in range(n):
            if dp[-1][u][0] < min_cost:
                min_cost = dp[-1][u][0]
                best_hop = u
                
        best_path = [best_hop]
        for i in range(len(targetPath) - 1, 0, -1):
            prev_hop = dp[i][best_hop][1]
            best_path.append(prev_hop)
            best_hop = prev_hop
        return best_path[::-1]
    
#official LC solution
class Solution:
    def mostSimilar(self, n: int, roads: List[List[int]], names: List[str], targetPath: List[str]) -> List[int]:
        '''
        let targetPath[:i] be the prefix of the targetPath ending at the ith element
        we can define dp[i][v] as the minimum edit dsitance bewteen targetPath[:i] and a path eneindg at the vertex v
        which means this path has length i+1
        
        example:
            targetPath = ["LAX", "ABC", "LAX", "DEF", "HTU", "XYZ"]
            v = 1
            what is the answer to dp[3][1] if we are looking for length of 3+1 prefix
            of all the paths ending at v = 1, with lenght 3+1, the min edit distance is 2
            
        base case is when i = 0 (path length 1) for any v
        how do we find dp[0][v]
        i = 0 corresponds to targetPath[0:0] or just the single node at i = 0
        path must be just [v] and edit distance would be names[0] != targetpath[0]
        
        dp[0][v] is then by definition equal toeditDistance([targetPath[0]], [v]). Therefore:
        dp[0][v] = 0 if names[v] = targetPath[0]
        dp[0][v] = 1 if names[v] != targetPath[0]
        for all v in the graph.
        
        for i > 0 we want to calculate dp[i][v] from dp[i-1][v]
        recall dp[i][v] is the min edit distance(targetPath[0:i], path)
        we just want a neighbor v
        
        we iterate over all possible canddiates of u for the second to last vertex of the path
        update dp[i][v] with dp[i-1][u]
             if names[v] != targetpath[i] then the current vertex v is a mismatch so we add 1
             
        One can imagine the transition from dp[i - 1][u] to dp[i][v] as appending an element targetPath[i] to the target path and appending the vertex v to the current path.
        
        dp[i][v] = mismatch + min(dp[i - 1][u]), where u is a neighbor of v and mismatch = 0 if names[v] = targetPath[i], or 1 otherwise.
        
        but we want the actual path. remember the technique from djikstras where we keep prev pointers?
        maintins array p[i][v] which represents the previous vertex before v in the optimal path
        '''
        dp = [[float('inf')]*n for _ in range(len(targetPath))]
        p = [[-1]*n for _ in range(len(targetPath))]
        
        adj = collections.defaultdict(list)
        for e in roads:
            adj[e[0]].append(e[1])
            adj[e[1]].append(e[0])
        
        #base cases
        for i in range(n):
            dp[0][i] = 1 if targetPath[0] != names[i] else 0
        
        for i in range(1,len(targetPath)):
            for v in range(n):
                for u in adj[v]:
                    #find min edit distance answer
                    cur = dp[i-1][u] + (names[v] != targetPath[i])
                    if cur < dp[i][v]:
                        dp[i][v] = cur
                        #to get to v we have to go through u
                        p[i][v] = u
        
        #find min edit distance in last row
        min_edit = min(dp[-1])
        #find last vertx in min edit dsit path
        # the last vertex in the path
        v = dp[-1].index(min_edit)
        ans = [v]
        #follow prev pointers
        for i in range(len(targetPath) - 1, 0, -1):
            # the previous vertex in the path
            v = p[i][v]
            ans.append(v)
        return ans[::-1]
    
#using dijsktras
#https://leetcode.com/problems/the-most-similar-path-in-a-graph/discuss/1011732/Python-dijakstra-with-heap-and-set
class Solution:
    def mostSimilar(self, n: int, roads: List[List[int]], names: List[str], targets: List[str]) -> List[int]:
        graph = defaultdict(list)
        heap = []
        
		# building graph with adjencency list 
        for x, y in roads:      # O(n+e) time, space
            graph[x] += [y]
            graph[y] += [x]
        
		# adding each node as starting to min heap (by default, all operations done on heapq module (our heap) support min heap)
		# (0, 0, i, [i]) means (edit_cost, node_nb_in_targets/level, city_nb, [path_so_far])
        for i in range(n):     # O(n) time
            heapq.heappush(heap, (0, 0, i, [i]))     # O(logn) time, O(n) space
			
        #seen set to prevent recomputing the same entires multiple times
        seen = set()
        while heap:    # O(len(t) * n) time, space - for every node in targets we have n-1 ~ n neighbours at most
            ed, i, city_nb, path = heapq.heappop(heap) O(logn)
			
			# -i because when adding to out min heap, we will be using negative indexing. Why? To extract the most possible path with
			# min edit distanceas soon as possible. Consider 3 following entries in heap:
			#    (1,-3, _, _)
			#    (1, -2, _, _)
			#    (2, -2, _, _)
			# Entry 1 and2 have the best edit distance so far. They are on 3rd (-3) and 2nd (-2) levels (nodes in target). 
			# That means we want to process (1,-3, _, _) first because it went further down the target path with the same
			# edit distance as (1, -2, _, _), so the chance of getting the best result (in our case min edit distance) are
			# higher in first, rather than second case. Since we use min heap, negative indexing is needed for the best result.
			
            if -i == len(targets):
                return path[:-1] #one last element in path is always redundant as it was added one level before
				
			#  if current city we are considering is different that city in targets at corresponding level, edit dist has to be increased
            if names[city_nb] != targets[-i]: 
                ed += 1
            
            for nbr in graph[city_nb]: 
			    # if we did not process a node 'nbr' on 'i'th-1 level with 'ed' edit distance, let's add it as a possibilty for a path 
                if (ed, i-1, nbr) not in seen:
                    heapq.heappush(heap, (ed, i-1, nbr, path + [nbr])) # add current ed, one level deeper (negative indexing, remember!), neighbouring node, and path so far
                    seen.add((ed, i-1, nbr))
        return 0

##############################
# 214. Shortest Palindrome
# 14APR23
##############################
#this was kinda a weird way
#https://leetcode.com/problems/shortest-palindrome/discuss/2513124/Shortest-and-Easiest-Solution-in-Python
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        '''
        easy case, just check if palindrom already
        "abcd"
        for this one just add s[1:] to the front
        but will this always work?
        what if we almost have a palindrome
        
        s = aacecaaa
        this is almost a palindrome
        we just a to the front
        
        we dont know what letter to add, so lets just call it ?
        ?aacecaaa
        '''
        
        #easy case
        if s == s[::-1]:
            return s
        
        N = len(s)
        #find all prefixes (starting wit the longest prefix first)
        #must be true prefix, i.e any prefix cannot equal to original string x
        for i in range(N-1,-1,-1):
            #get all prefixes for this string
            a = s[:i+1]
            #get the reverse of this current prefix
            b = a[::-1]
            #if they are the same, it means we can form a palindrome around s[i]
            if a == b:
                print(a,s[i],b)
                #get this suffix after i
                c = s[i+1:]
                print(i,c)
                #reverse it
                c = c[::-1]
                return c + a + c[::-1]
            
        #otherwise we can't use any prefxis to maked it a palindrome
        #then just take the string from index 1 to the end, reverse and put it in front of c
        s1 = s[1:]
        return s1[::-1] +s

class Solution:
    def shortestPalindrome(self, s: str) -> str:
        '''
        another way
        
        Example: s = dedcba. Then r = abcded and I try these overlays (the part in (...) is the prefix I cut off, I just include it in the display for better understanding):
        
          s          dedcba
          r[0:]      abcded    Nope...
          r[1:]   (a)bcded     Nope...
          r[2:]  (ab)cded      Nope...
          r[3:] (abc)ded       Yes! Return abc + dedcba
          
         recall, we are only allowed to add characters to the beignning of the string
         so we can find the laregst segment from the beginning that is a palindrome
         then we can take the end, reverse it and add it to the front
         exmple:
         abcbabcab
         
         longest prefix that is palindrome
         abcba
         
         remainign segment is bcab
         then we reverse and add it to the front
         bacb abcba bcab
        '''
        r = s[::-1]
        for i in range(len(s) + 1):
            #if beginning part is palindrome
            if s[:len(s)-i]==r[i:]:
                return r[:i] + s
            
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        '''
        in brute force we found laregst palindrom substring in s which takes O(len(s)) times
        we could make the process easier if we could reduce the size of thes tring to search for the substring without checeking the complete
        substring each time
         
        example
        take string abcbabcaba
        fix 2 poitners, i and j, set i = 0 and move j from n-1 to 0, and increment i if s[i] == s[j]
        now we justt need to seach in range [0,i)
        we are just reducing the size of the string to search for the largest palindrome
        the range [0,i) must always contain the largest palindrom substring
        proof:
            say that i was a perfect palindrome, in this case i would moved all the way to n (n times exactly)
        '''
        N = len(s)
        i = 0
        for j in range(N-1,-1,-1):
            if s[i] == s[j]:
                i += 1
        
        #complete palindrome
        #we are just saving time skipping the substring check for palindrome, which is O(N)
        if i == N:
            return s
        
        remaining_reversed = s[i:][::-1]
        #the middle part is just the longest palindrome part from approach 1 (betwen 0 and i, non inclusive)
        return remaining_reversed + self.shortestPalindrome(s[:i])+s[i:]
    
#KMP
class Solution:
    def shortestPalindrome(self, s: str) -> str:
        '''
        we can use KMP, which uses the pi table, also known as proper prefix that is also a suffix
        idea:
            say we have proper prefix that is also a suffix
            if we are trying to match a text string for the past prefix and we have matched the first s positions, but whenw e fail
            the value of the lookup table for s is the lgonest prefix taht could possibly match
            so we don't need to start all over again
            
        lps array
        f(0) = 0
        for (i = 1; i < n; i++) {
            t = f(i-1)
            while (t > 0 && b[i] != b[t]) {
                t = f(t-1)
            }
            if (b[i] == b[t]) {
                ++t
            }
            f(i) = t
        }
        
        in approache 1 , we stored reverse s
        then we tierated over from 0 to n-1 and checked s[:n-i] == rev[i:]
        make s as s + s[::-1]
        then look for the longest prefix that is also the suffix
        
        algo:
        We use the KMP lookup table generation
        Create new_s as s + "#" + reverse(s) and use the string in the lookup-generation algorithm
        The "#" in the middle is required, since without the #, the 2 strings could mix with each ther, producing wrong answer. For example, take the string 
        "aaaa"
        "aaaa". Had we not inserted "#" in the middle, the new string would be 
        "aaaaaaaa"
        "aaaaaaaa" and the largest prefix size would be 7 corresponding to "aaaaaaa" which would be obviously wrong. Hence, a delimiter is required at the middle.
        Return reversed string after the largest palindrome from beginning length(given by 
        f[n_new-1]
        n−f[n_new-1]) + original string
        '''
        N = len(s)
        rev_s = s[::-1]
        new_s = s + "#" + rev_s
        new_N = len(new_s)
        lps = [0]*new_N
        
        for i in range(1,new_N):
            t = lps[i-1]
            while t > 0 and new_s[i] != new_s[t]:
                t = lps[t-1]
            if new_s[i] == new_s[t]:
                t += 1
            lps[i] = t
        
        return rev_s[:N - lps[new_N-1]]+s


#############################################
# 1372. Longest ZigZag Path in a Binary Tree
# 18APR23
#############################################
#fuckkkk
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        '''
        just dfs on the tree and keep track of current direction and current size in path
        update global max
        keep gone left and gone right
        i cant reverse the direction here, update global max and reset count
        '''
        self.ans = 0
        
        def dfs(node,length,went_left,went_right):
            self.ans = max(self.ans,length)
            if not node:
                return
            if not went_left and not went_right:
                dfs(node.left,length+1,True,False)
                dfs(node.right,length+1,False,True)
            
            if went_left:
                #go right and advance
                dfs(node.right,length+1,False,True)
                #go left and reset
                dfs(node.left,0,True,False)
            
            if went_right:
                #go left and advance
                dfs(node.left,length+1,True,False)
                #go right and reset
                dfs(node.right,0,False,True)
                            
        dfs(root,0,None,None)
        return self.ans
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        '''
        just keep track of the current direction we can go into, lets just check left
        if we can go left, then we incremenet the current path length and in the next call, we can't fo left
            and also go right but reset the new count
        if we can't go left,
            then we proceeed to go left but stsrt a new count
            but it also could have been that we go right, so we need to extend the chain
        
        update max on the fly
        on null nodes we just return
        '''
        self.ans = 0
        
        def dfs(node,canLeft,path):
            if not node:
                return
            
            #otherwise update
            self.ans = max(self.ans,path)
            
            #if we can go left
            if canLeft:
                #extend
                dfs(node.left,False,path+1)
                #reset
                dfs(node.right,True,1)
            
            #can only go right while extending
            else:
                #reset
                dfs(node.left,False,1)
                #reset
                dfs(node.right,True,path+1)
        
        #start twice, once allowing left and once not allowing left
        dfs(root,False,0)
        dfs(root,True,0)
        return self.ans
    
#another way
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        '''
        another way is just keep both pathlengths going left and right in the caller
        in the function we keep variables left and right
        left gives us the length of longest zigzag path at that point in the path 
        same thing with the right
        we just need to swap the path lengths at each call
        '''
        self.ans = 0
        
        def dp(node,left,right):
            if not node:
                return 
            self.ans = max(self.ans, max(left,right))
            
            #if we are allowed go left
            if node.left != None:
                dp(node.left,right+1,0) #swap and reset
            
            if node.right != None:
                dp(node.right,0,left+1)
        
        dp(root,0,0)
        return self.ans
    

#another way, but just swap in place
class Solution:
    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        self.mx = 0
        def dfs(node):
            if node is None:
                return (0, 0)
            left = dfs(node.left)[0] 
            right = dfs(node.right)[1]
            self.mx = max(self.mx, left, right)
            return (right + 1, left + 1)
        dfs(root)
        return self.mx

#####################################
# 662. Maximum Width of Binary Tree
# 20APR23
######################################
#DFS solution, BFS is trivial
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        '''
        to do DFS, keep record of current depth, and index of node
        into a hashmap, record the depth with this current index
        
        if this depth isn't seen yet, add the current index
        otherwise we've seen this depth before, so find the width
        because we always go left, we are guaranteed that when we first see a depth, it must be the first index on this depth
        
        '''
        depth_to_index = {}
        self.ans = 0
        
        def dfs(node,depth,index):
            if not node:
                return 0
            if depth not in depth_to_index:
                depth_to_index[depth] = index
                
            #otherwise max it
            self.ans = max(self.ans, index - depth_to_index[depth] + 1)
            
            dfs(node.left,depth+1,index*2)
            dfs(node.right,depth+1,index*2 + 1)
        
        dfs(root,0,0)
        return self.ans

##############################################
# 467. Unique Substrings in Wraparound String
# 20APR23
##############################################
class Solution:
    def findSubstringInWraproundString(self, s: str) -> int:
        '''
        there are only so many viable substrings in s that are in the wrap around base
        example
        zab, 
        is already a substring of the wrap around base, which means all substrinsg of zab are also going to be in wraparound base
        
        say we are at z
        z is substring we have 1
        now we consider za
        a is a substring
        but so is za
        
        if we let dp(i) be the number of unique substrings using s[:i]
        then dp(i+1) = dp(i) + number of substrings if s[:i+1] make a susbtring
        z = 1
        za = 2 + 1 = 3
        zab = 3 + 3 = 6
        zabc = 6 + 4 = 10
        zabce = zabce not as substring so take 10 + 1
        
        the problem is checking if s[:i+1] is a unique substring of s
        and substrings must be unique
        
        store the number of substrings ending with char s
        abcd
        ending with a, a
        ending wih b, ab,b
        ending with c, abc, bc, c
        ending with d, abcd, bcd, cd, d
        
        keep track of the logngest contiguous string we can make, must be at least 1 away, for in the case za, they are 26 away
        must be ending in char, be are testing the next substring
        coud you do it with starting with a char
        '''
        counts = {}
        longest_streak = 0
        N = len(s)
        for i in range(N):
            if (i > 0) and ((ord(s[i]) - ord(s[i-1]) == 1) or (ord(s[i-1]) - ord(s[i]) == 25)):
                longest_streak += 1
            else:
                longest_streak = 1
            
            #put in coutns and store lognest
            counts[s[i]] = max(longest_streak,counts.get(s[i],0))
    
        return sum(counts.values())