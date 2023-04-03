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



