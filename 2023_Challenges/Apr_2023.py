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