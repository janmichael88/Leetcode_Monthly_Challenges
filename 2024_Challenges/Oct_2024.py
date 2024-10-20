###############################################
# 1497. Check If Array Pairs Are Divisible by k
# 01OCT24
###############################################
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        '''
        we need pairs of the form
        (a_i + a_j) % k == 0
        destribute
        (a_i % k + a_j % k) %k = 0
        we can just find each number's % k and storm them
        (mod_i + mod_j) % k = 0
        
        this implies that 
        mod_j = k - mod_i
        
        if mod is 0, we need to pair with another elements that is also mod = 0
        first we store numbers % k
        compute each number num as 
            mod = ((num % k) + k) % k to handle negative numbers
            
        tricky math problem
        '''
        counts = Counter()
        for num in arr:
            comp = ((num % k) + k) % k
            counts[comp] += 1
        
        for num in arr:
            comp = ((num % k) + k) % k
            if comp == 0:
                if counts[comp] % 2 == 1:
                    return False
            #mods and mods comps must have equal counts
            elif counts[comp] != counts[k - comp]:
                return False
        
        return True
    
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        '''
        in python we don't need to worry about taking the mod of negative numbers
        we can also keep an array of size k
        instead of checking all numbers in arr
        just check all mod_counts from 1 to k

        example
        if a number leaves a remainder of 2 when divided by k, then the other number in the pair 
        should leave a remainder that 'completes' the pair - usually, that means its remainder should be k - 2
        i.e remainder, and k - remainder
        '''
        mod_count = [0]*k
        for num in arr:
            mod_count[num % k] += 1
        
        #special case for mod k == 0
        if mod_count[0] % 2 == 1:
            return False
        
        for i in range(1,k):
            if mod_count[i] != mod_count[k-i]:
                return False
        
        return True
    
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        # Custom comparator to sort based on mod values.
        arr = sorted(arr, key=lambda x: (k + x % k) % k)

        # Assign the pairs with modulo 0 first.
        start = 0
        end = len(arr) - 1
        while start < end:
            if arr[start] % k != 0:
                break
            if arr[start + 1] % k != 0:
                return False
            start = start + 2

        # Now, pick one element from the beginning and one element from the
        # end.
        while start < end:
            if (arr[start] + arr[end]) % k != 0:
                return False
            start += 1
            end -= 1

        return True

########################################
# 1590. Make Sum Divisible by P (REVISTED)
# 04OCT24
########################################
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        '''
        we need to remove a subarray to make sum divivibsle by p
        say we have subarray [a,b,c,d,e]
        we want (a+b+c+d+e) % p == 0
        
        if (a+b+c+d+e) % p == k, where k != 0
        then we need to find the smallest subarray == k where sum(subarray) % p == k
        
        more fomrally:
        we need a subarray who's sum % p == sum(nums) % p
        we can say: 
        let k = sum(nums) % p
        (prefSum_i - prefSum_j) % p = k
        k is some multiple of pf
        prefSum_i - prefSum_j = k + m*p
        prefSum_i = prefSum_j + k + m*p
        prefSum_i = (prefSum_j - k) % p
        
        to correct for negatvies we just add p and most it again
        
        '''
        SUM = sum(nums)
        if SUM % p == 0:
            return 0
        N = len(nums)
        k = SUM % p
        ans = N
        
        mapp = {}
        mapp[0] = -1
        curr_sum = 0
        for i,num in enumerate(nums):
            curr_sum = (curr_sum + num + p) % p
            #find its complement
            comp = (curr_sum - k + p) % p
            if comp in mapp:
                ans = min(ans, i - mapp[comp])
            
            mapp[curr_sum] = i
        
        return -1 if ans == N else ans
        
#################################################
# 2491. Divide Players Into Teams of Equal Skill
# 04OCT24
#################################################
#phew
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        '''
        need to divide into len(skill) / 2 teams
        such that the skill of total skill of each team is equal
        '''
        SUM = sum(skill)
        N = len(skill)
        teams = N // 2
        
        #check that we can't make teams of even skill
        if SUM % teams != 0:
            return -1
        
        skill_of_team = SUM // teams
        counts = Counter(skill)
        
        chemistry = 0
        for i in range(1,(skill_of_team // 2)+1):
            #same i and skill - i
            if i == skill_of_team - i and counts[i] != 0 and counts[skill_of_team - i] != 0:
                if counts[i] % 2 != 0:
                    return -1
                chemistry += ((i)*(skill_of_team - i))*(counts[i] // 2)
                counts[i] = 0
                counts[skill_of_team - i] = 0
            elif counts[i] == counts[skill_of_team - i] and counts[i] != 0 and counts[skill_of_team - i] != 0:
                if counts[i] != counts[skill_of_team - i]:
                    return -1
                chemistry += (i*(skill_of_team - i))*counts[i]
                counts[i] = 0
                counts[skill_of_team - i] = 0
        
        #validate after getting answer
        for k,v in counts.items():
            if v != 0:
                return -1
        return chemistry
    
#sort solution, each pair should add up to the precompute score
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        '''
        need to divide into len(skill) / 2 teams
        such that the skill of total skill of each team is equal
        '''
        SUM = sum(skill)
        N = len(skill)
        teams = N // 2
        
        #check that we can't make teams of even skill
        if SUM % teams != 0:
            return -1
        
        skill_of_team = SUM // teams
        skill.sort()
        chemistry = 0
        left = 0
        right = N - 1
        
        while left < right:
            if skill[left] + skill[right] != skill_of_team:
                return -1
            chemistry += skill[left]*skill[right]
            left += 1
            right -= 1
        
        return chemistry
    
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        '''
        we can use a count array,
        then retraverse the skill array and find compelements and add accordingly
        '''
        SUM = sum(skill)
        N = len(skill)
        teams = N // 2
        
        #check that we can't make teams of even skill
        if SUM % teams != 0:
            return -1
        
        skill_of_team = SUM // teams
        counts = [0]*1001
        for s in skill:
            counts[s] += 1
        
        chemistry = 0
        for s in skill:
            comp = skill_of_team - s
            if counts[comp] == 0:
                return -1 #cant do it
            chemistry += s*comp
            counts[comp] -= 1
        
        return chemistry // 2 #we are only counting pairs, but we double counted
    
#############################################
# 567. Permutation in String (REVISTED)
# 05OCT24
#############################################
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        hashamp validation
        if we had a countmap of s1's chars, then there needs to be a subtring in s2, that contains all the chars of s1's count mapp
        we can use sliding window
        '''
        counts_s1 = Counter(s1)
        M = len(s1)
        N = len(s2)
        
        if N < M:
            return False
        
        count1 = Counter()
        count2 = Counter()
        
        for i in range(M):
            count1[s1[i]] += 1
            count2[s2[i]] += 1
        
        for i in range(N - M):
            #if they match, return
            if count1 == count2:
                return True
            #add to new char to s2, which is ust i + M
            count2[s2[i + M]] += 1
            #remove the leftmost char, which is at i
            count2[s2[i]] -= 1
        
        return count1 == count2
    
########################################
# 3163. String Compression III
# 05OCT24
########################################
class Solution:
    def compressedString(self, word: str) -> str:
        '''
        we can only compress up to size 9
        '''
        comp = ""
        curr_char = ""
        curr_count = 0
        
        for ch in word:
            if not curr_char:
                curr_char = ch
                curr_count += 1
            elif ch == curr_char and curr_count < 9:
                curr_count += 1
            else:
                comp += str(curr_count)+curr_char
                curr_char = ch
                curr_count = 1
        if curr_count:
            comp += str(curr_count)+curr_char 
            
        return comp
    
#another cool way, advance in steps of length
class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            current_char = word[i]
            length = 0
            while i + length < len(word) and word[i + length] == current_char and length < 9:
                length += 1
            comp += str(length) + current_char
            i += length
        return comp

##########################################
# 1813. Sentence Similarity III
# 06OCT24
##########################################
class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        '''
        the idea is that sentence1 must have a prefix and or suffix of sentence2 only!
        we also need to check that sentence1 has a prefix and/or suffix of setence 1
        we can use queue for both and check that they are non empty
        
        i.e one of the sentences must be empty after clearing the queue
        '''
        q1 = deque(sentence1.split(" "))
        q2 = deque(sentence2.split(" "))
        
        while q1 and q2 and q1[0] == q2[0]:
            q1.popleft()
            q2.popleft()
            
        while q1 and q2 and q1[-1] == q2[-1]:
            q1.pop()
            q2.pop()
            
        return not q1 or not q2
        
class Solution:
    def areSentencesSimilar(self, s1: str, s2: str) -> bool:
        '''
        this problem fucking sucks....
        i need to pick the smaller of the two strings and compare to the larger one
        assume s1 is the larger of the two strings
        '''

        s1_words = s1.split(" ")
        s2_words = s2.split(" ")
        start, ends1, ends2 = 0, len(s1_words) - 1, len(s2_words) - 1

        # If words in s1 are more than s2, swap them and return the answer.
        if len(s1_words) > len(s2_words):
            return self.areSentencesSimilar(s2, s1)

        # Find the maximum words matching from the beginning.
        while start < len(s1_words) and s1_words[start] == s2_words[start]:
            start += 1

        # Find the maximum words matching in the end.
        while ends1 >= 0 and s1_words[ends1] == s2_words[ends2]:
            ends1 -= 1
            ends2 -= 1

        # if ends1 crosses, it means prefixes and suffix match so return true
        return ends1 < start
    
########################################################
# 2696. Minimum String Length After Removing Substrings
# 07OCT24
########################################################
class Solution:
    def minLength(self, s: str) -> int:
        '''
        stack problem,
        whenever we have stack of length 2 or greater, remove AB
        '''
        stack = []
        for ch in s:
            stack.append(ch)
            while len(stack) >= 2 and (stack[-2:] == ['A','B'] or stack[-2:] == ['C','D']):
                stack.pop()
                stack.pop()
            
        
        return len(stack)
    
class Solution:
    def minLength(self, s: str) -> int:
        '''
        we can use two pointers and temporary char array
        we read each char s into write ptr
        and if there's a match, we move write ptr back
        '''
        char_arr = list(s)
        write = 0
        
        for read in range(len(char_arr)):
            char_arr[write] = char_arr[read]
            if write - 1 >= 0 and (char_arr[write - 1 : write + 1] == ['A', 'B'] or char_arr[write - 1 : write + 1] == ['C','D']):
                write -= 1
            else:
                write += 1
        return write
    
###########################################
# 2559. Count Vowel Strings in Ranges
# 07OCT24
###########################################
class Solution:
    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        '''
        need a way to access count of vowels fast
        prefix sum!
        keep array that stores number of strings staring or ending with vowel
        '''
        pref_counts = [0]
        vowels = 'aeiou'
        
        for w in words:
            if w[0] in vowels and w[-1] in vowels:
                pref_counts.append(pref_counts[-1] + 1)
            else:
                pref_counts.append(pref_counts[-1])
        
        ans = []
        for l,r in queries:
            ans.append(pref_counts[r+1] - pref_counts[l])
        
        return ans

########################################################################
# 1963. Minimum Number of Swaps to Make the String Balanced (REVISITED)
# 08OCT24
#########################################################################
class Solution:
    def minSwaps(self, s: str) -> int:
        '''
        keep track of opening and closing
        try closing valid bracket pairs if we can,
        antyhign left over needs to be swapped
        
        only works becaue the original can be balanced!
        
        '''
        stack = []
        for ch in s:
            if ch == '[':
                stack.append(ch)
            else:
                if stack:
                    stack.pop()
                else:
                    stack.append(ch)
        
        return len(stack)//2
        
class Solution:
    def minSwaps(self, s: str) -> int:
        '''
        keep track of balance
        whenver we are unbalanced, we can swap
        then the new balance becomes 1
        '''
        balance = 0
        swaps = 0
        
        for ch in s:
            if ch == '[':
                balance += 1
            else:
                balance -= 1
            
            #rebalancing leaves it as 1
            #because we made up the balance, but not we swapped, so its unbalanced by 1
            if balance < 0:
                swaps += 1
                balance = 1
        
        return swaps
    
############################################################
# 1541. Minimum Insertions to Balance a Parentheses String
# 08OCT24
############################################################
#fuckkk
class Solution:
    def minInsertions(self, s: str) -> int:
        '''
        we can only clear if we have ['(' ,')'] in the stack
        clear as much as we can and add after
        ")"
        '''
        stack = []
        for ch in s:
            if ch == ')':
                if len(stack) >= 2 and stack[-2] == '(' and stack[-1] == ')':
                    stack.pop()
                    stack.pop()
                else:
                    stack.append(ch)
            else:
                stack.append(ch)
        
        ans = 0
        while stack:
            if stack[-1] == '(':
                ans += 2
                stack.pop()
            elif len(stack) >= 2:
                if stack[-2] == ')' and stack[-1] == ')':
                    ans += 1
                    stack.pop()
                    stack.pop()
                elif stack[-2] == '(' and stack[-1] == ')':
                    ans += 1
                    stack.pop()
                    stack.pop()
                elif stack[-2] == '(' and stack[-1] == '(':
                    ans += 1
                    stack.pop()
                    stack.pop()
                elif stack[-2] == ')' and stack[-1] == ')':
                    ans += 1
                    stack.pop()
                    stack.pop()
            elif stack[-1] == ')':
                ans += 2
                stack.pop()

        return ans
    
class Solution:
    def minInsertions(self, s: str) -> int:
        '''
        maintain stack of open paranteh
        if i and i+1 in bounds and )), advance it
        if open just add to stack
        
        if it not we increment count
        
        whats left on the stack are the opening that need ))
        '''
        stack = []
        i = 0
        count = 0
        
        while i < len(s):
            if s[i] == '(':
                stack.append(s[i])
            
            else:
                if i + 1 < len(s) and s[i+1] == ')':
                    i += 1
                else:
                    count += 1
                
                if stack:
                    stack.pop()
                else:
                    count += 1
            
            i += 1
    
        return count + len(stack)*2
    
##################################################
# 2838. Maximum Coins Heroes Can Collect
# 09OCT24
##################################################
class Solution:
    def maximumCoins(self, heroes: List[int], monsters: List[int], coins: List[int]) -> List[int]:
        '''
        brute force would be to check each ith hero against its jth monster,  and if power of ith hero >= jth monster, aqcuqre jth coins
        i can pair monsters with coins value, sort on increasing health
        make pref_sum on this array, then binary search to find the upper bound of the hero to gets it total coins
        '''
        
        pairs = [(m,c) for (m,c) in zip(monsters,coins)]
        pairs.sort(key = lambda x : x[0])
        sorted_monsters = [0] + [m for (m,_) in pairs]
        pref_coins = [0]
        for _,c in pairs:
            pref_coins.append(pref_coins[-1] + c)
        
        ans = []
        print(pref_coins)
        for h in heroes:
            idx = self.binary_search(sorted_monsters,h)
            ans.append(pref_coins[idx])
        
        return ans
            
    def binary_search(self,arr,target):
        left = 0
        right = len(arr) - 1
        ans = 0
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] <= target:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans
        
#two pointer soln
class Solution:
    def maximumCoins(self, heroes: List[int], monsters: List[int], coins: List[int]) -> List[int]:
        '''
        another way to sort heros and monsters 
        keeping track of indicies, then if hero can beat this monster, advance and update coins
        if hero cant move to next hero but keep pointer positions
        '''
        heroes_sorted = sorted([(i,h) for i,h in enumerate(heroes)], key = lambda x: x[1])
        monsters_sorted = sorted([(i,m) for i,m in enumerate(monsters)], key = lambda x : x[1])
        
        ans = [0]*len(heroes)
        curr_coins = 0
        curr_monster = 0
        
        for i,h in heroes_sorted:
            while curr_monster < len(monsters) and monsters_sorted[curr_monster][1] <= h:
                #get actual monster index
                idx = monsters_sorted[curr_monster][0]
                curr_coins += coins[idx]
                curr_monster += 1
            
            #update ans
            ans[i] = curr_coins
        
        
        return ans
        
#############################################
# 962. Maximum Width Ramp (REVISTED)
# 10OCT24
#############################################
#almost optimal!
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        '''
        cand just do monostack, because i would clear the stack and there might be a j that is <= after clearing
        which would extend the ramp
        what if we sorted, but paired its indicies with its num value
        then we can use monostack on the indicies!
        '''
        pairs = [(i,num) for i,num in enumerate(nums)]
        #sort on num value
        pairs.sort(key = lambda x : x[1])
        #monostack on increasing indicies
        stack = []
        
        ans = 0
        for i,num in pairs:
            #monostack
            while stack and stack[-1][0] > i:
                ans = max(ans, stack[-1][0] - stack[0][0])
                stack.pop()
            
            stack.append((i,num))
        
        if stack:
            ans = max(ans, stack[-1][0] - stack[0][0])
        return ans
    
#straight soring without stack, keep track of the min index encounterd so far
#then update max and min
class Solution:
    def maxWidthRamp(self, nums: List[int]) -> int:
        pairs = [(i,num) for i,num in enumerate(nums)]
        #sort on num value
        pairs.sort(key = lambda x : x[1])
        n = len(pairs)
        ans = 0
        min_index = n
        for i,num in pairs:
            min_index = min(min_index,i)
            ans = max(ans, i - min_index)
        
        return ans

#####################################################
# 1942. The Number of the Smallest Unoccupied Chair
# 11OCT24
####################################################
class Solution:
    def smallestChair(self, times: List[List[int]], targetFriend: int) -> int:
        '''
        keep heap of available chairs, 
        then process the times in order of start
        need to pull apart times [index,time,start,leave]
        #there could be ties to arrive and leave, we need to make sure arrive happens before leave
        '''
        n = len(times)
        available_chairs = list(range(len(times)))
        heapq.heapify(available_chairs)
        taken_chairs = [-1]*n
        
        #pull apart times
        new_times = []
        for i,(arrive,leave) in enumerate(times):
            new_times.append((i,arrive,'arrive'))
            new_times.append((i,leave,'leave'))
        
        new_times.sort(key = lambda x: (x[1],x[2] == 'arrive'))
        
        #firs time person arrives, his taken_chair will be -1
        for person,time,_ in new_times:
            #check if no assignment
            if taken_chairs[person] == -1:
                if person == targetFriend:
                    return heapq.heappop(available_chairs)
                else:
                    taken_chairs[person] = heapq.heappop(available_chairs)
            else:
                #this must be a departure
                chair = taken_chairs[person]
                taken_chairs[person] = -1
                heapq.heappush(available_chairs, chair)

#brute force is actually kinda trickier
class Solution:
    def smallestChair(self, times: List[List[int]], targetFriend: int) -> int:
        '''
        just for fun, brute force 
        sort on times, then assign a leaving time for each chair
        then for each (arrival,start) find the left most available chair
            i.e if chair is available
        '''
        target_time = times[targetFriend][0]
        times.sort()
        n = len(times)
        chair_available = [0]*n
        
        for start,leave in times:
            for i in range(n):
                if chair_available[i] <= start:
                    chair_available[i] = leave
                    if start == target_time:
                        return i
                    break #found left most
        
        return -1
                
#########################################################
# 2406. Divide Intervals Into Minimum Number of Groups
# 12OCT24
########################################################
#bleagh
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        '''
        sort on starts and tie break on the earliest ends
        then we need to check i and i + 1 and try to capture as many groups as we can
        '''
        intervals.sort(key = lambda x : (x[0],x[1]))
        groups = 0
        i = 0
        n = len(intervals)
        
        while i < n:
            j = i + 1
            while j < n and not (intervals[i][0] <= intervals[j][0] <= intervals[i][1] or \
                                 intervals[i][0] <= intervals[j][1] <= intervals[i][1] or \
                                 intervals[j][0] <= intervals[i][0] <= intervals[i][0] or  \
                                 intervals[j][0] <= intervals[i][0] <= intervals[j][1]):
                j += 1
            
            groups += 1
            i = j
        
        return groups

#check all intervals, and find one with largest counts
#TLE
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        '''
        if intervals overlap, they need to be in sepeate groups,
        so the first algo i tried won't work for all cases, 
            i.e i could keep expanding the j pointer, but then there'd be another one that overlapped
        
        for example [3,7], [5,6], [1,8] all overlap at 5,6
            so these should be in sepearte groups
        
        so if we have k overlapping intervals,we need at least K groups
        so if we find the intervals for which the sections are overlapping
        
        for example, we can add the intervals [10,12] and [2,3]  to some of the groups without conflict
        '''
        earliest = 1
        latest = 1
        for s,e in intervals:
            latest = max(latest,e)
        times = [0]*(latest+1)
        for s,e in intervals:
            for t in range(s,e+1):
                times[t] += 1
        
        return max(times)

#line sweep
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        '''
        line, sweep, do -1 after (end + 1)
        then roll up like prefix sum to accumlate changes for each time t
        '''
        earliest = float('inf')
        latest = 0
        for s,e in intervals:
            earliest = min(earliest,s)
            latest = max(latest,e)
        
        times = [0]*(latest + 2)
        for s,e in intervals:
            times[s] += 1
            times[e+1] -= 1
        
        #roll them up
        for t in range(earliest,latest + 2):
            times[t] += times[t-1]
        

        return max(times)
    
#sorting and prefix sum
class Solution:
    def minGroups(self, intervals: List[List[int]]) -> int:
        '''
                if intervals overlap, they need to be in sepeate groups,
        so the first algo i tried won't work for all cases, 
            i.e i could keep expanding the j pointer, but then there'd be another one that overlapped
        
        for example [3,7], [5,6], [1,8] all overlap at 5,6
            so these should be in sepearte groups
        
        so if we have k overlapping intervals,we need at least K groups
        so if we find the intervals for which the sections are overlapping
        
        for example, we can add the intervals [10,12] and [2,3]  to some of the groups without conflict
        split events into (start_event, count) and (end_event, -1)
            like meeting rooms2
        
        then just get prefix sum to find the counts
        (start,+1), (end + 1, -1)
        menaing at this start, count goes up 1 for event, but after time end + 1, we are freed up
        if we go in order can track the number of intersections at each time
        '''
        events = []
        for s,e in intervals:
            events.append((s,1))
            events.append((e+1,-1))
        
        #sort on start, and then on events (-1, ending first)
        events.sort(key = lambda x : (x[0],x[1]))
        max_overlapping = 0
        curr_overlapping = 0
        for time,count in events:
            curr_overlapping += count
            max_overlapping = max(max_overlapping,curr_overlapping)
        
        return max_overlapping
    
###########################################################
# 2530. Maximal Score After Applying K Operations
# 14OCT24
###########################################################
import heapq
class Solution:
    def maxKelements(self, nums: List[int], k: int) -> int:
        '''
        heap them all and pop,take,and push k times
        '''
        score = 0
        max_heap = [-num for num in nums]
        heapq.heapify(max_heap)
        score = 0
        
        for _ in range(k):
            temp = -heapq.heappop(max_heap)
            score += temp
            heapq.heappush(max_heap, -math.ceil(temp / 3))
        
        return score
            
##########################################
# 1733. Minimum Number of People to Teach
# 14OCT24
##########################################
#close one
class Solution:
    def minimumTeachings(self, n: int, languages: List[List[int]], friendships: List[List[int]]) -> int:
        '''
        we would want to teach the one language the covers as many friendships as possible
        we need to make sure that friendships can be satisfied
        if it can't then teach the one language that minimuze the number of users i need to teach
        '''
        #mapp users to languages
        langs_usrs = defaultdict(set)
        for user,langs in enumerate(languages):
            for l in langs:
                langs_usrs[user + 1].add(l)
        
        #check each language and find smallest number we need to teach
        ans = len(languages)
        for l in range(1,n+1):
            #count
            curr_count = 0
            for u,v in friendships:
                #if there isn't a comman language, then teach it
                u_langs = langs_usrs[u]
                v_langs = langs_usrs[v]
                common = u_langs  & v_langs
                if len(common) == 0:
                    curr_count += 1
            
            ans = min(ans,curr_count)
        
        return ans
    
class Solution:
    def minimumTeachings(self, n: int, languages: List[List[int]], friendships: List[List[int]]) -> int:
        '''
        first find those who 'cant' communicate with each other
        among these, find the most popular spoken language
        then teach that language to the minosrity whoe cannot
            i.e teach the language to the one who don't speak the most frequqent one
        '''
        langs = [set(l) for l in languages]
        
        cant_speak = set()
        #find users who can't speak
        for u,v in friendships:
            #if there is a common lang, no need to do anything
            if langs[u-1] & langs[v-1]:
                continue
            cant_speak.add(u-1)
            cant_speak.add(v-1)
        
        if not cant_speak:
            return 0
        
        lang_counts = Counter()
        for user in cant_speak:
            for l in langs[user]:
                lang_counts[l] += 1
        
        
        #teach the minorities any one language
        return len(cant_speak) - max(lang_counts.values())
    
#another way
class Solution:
    def minimumTeachings(self, n: int, languages: List[List[int]], friendships: List[List[int]]) -> int:
        languages = [set(x) for x in languages]
        
        users = set()
        for u, v in friendships: 
            if not languages[u-1] & languages[v-1]: 
                users.add(u-1)
                users.add(v-1)
        
        freq = {}
        for i in users: 
            for k in languages[i]:
                freq[k] = 1 + freq.get(k, 0)
        return len(users) - max(freq.values(), default=0)
    
############################################
# 2938. Separate Black and White Balls
# 15OCT24
#############################################
class Solution:
    def minimumSteps(self, s: str) -> int:
        '''
        need min steps to move ones to the right and moves zeros to the left
        but we are only allow to swap some i and i + 1
        the hint gave it away T.T
        '''
        ans = 0
        count_zeros = 0
        N = len(s)
        for i in range(N-1,-1,-1):
            if s[i] == '0':
                count_zeros += 1
            else:
                ans += count_zeros
        
        return ans
        
class Solution:
    def minimumSteps(self, s: str) -> int:
        '''
        if we have a one at some position i, we need to move to the left i spots
        we can use an anchor pointer, starting at index 0
        if a char is one, it needs to move i - anchort pointer times
        '''
        white = 0
        swaps = 0
        N = len(s)
        for i,ch in enumerate(s):
            #if its a white ball, move it to the left i - white_ptr swaps
            if ch == '0':
                swaps += i - white
                white += 1
        
        return swaps
    
####################################
# 1405. Longest Happy String
# 16OCT24
####################################
#i hate these rule based heap problems....
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        '''
        we can only use a,aa,b,bb,c,cc to make the string
        it can ony contain at most: a count of a
                            b count of b
                            c count of c
        build intelligently
        use largest avilable letter, then break with another
        you dont need to use all of them, just limited by i
        '''
        ans = ""
        max_heap = [(-a,'a'),(-b,'b'),(-c,'c')]
        heapq.heapify(max_heap)
        
        while max_heap:
            first_count,first_letter = heapq.heappop(max_heap)
            if not max_heap:
                continue
            elif max_heap:
                second_count,second_letter = heapq.heappop(max_heap)
            
            #try adding first
            first_addition = min(2,-first_count)
            ans += first_addition*(first_letter)
            print(first_count,first_addition)
            first_count += first_addition
            if first_count < 0:
                heapq.heappush(max_heap, (first_count,first_letter))
                
            #do second
            second_addition = min(2,-second_count)
            ans += second_addition*(second_letter)
            second_count += second_addition
            if second_count < 0:
                heapq.heappush(max_heap, (second_count,second_letter))
        
        return ans
    
#need to add in one at a time....
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        '''
        we can only use a,aa,b,bb,c,cc to make the string
        it can ony contain at most: a count of a
                            b count of b
                            c count of c
        build intelligently
        use largest avilable letter, then break with another
        you dont need to use all of them, just limited by i
        for python its always more efficient to append to a list, then join at the end
        instead of concatenating a string
        '''
        pq = []
        if a > 0:
            pq.append((-a,'a'))
        if b > 0:
            pq.append((-b,'b'))
        if c > 0:
            pq.append((-c,'c'))
        
        heapq.heapify(pq)
        result = []
        
        while pq:
            count,letter = heapq.heappop(pq)
            #negate
            count = -count
            #check if same chars are are the ends
            #we need to check last two first
            if len(result) >= 2 and result[-1] == letter and result[-2] == letter:
                if not pq:
                    break
                #addd
                second_count,second_char = heapq.heappop(pq)
                result.append(second_char)
                if second_count + 1 < 0:
                    heapq.heappush(pq, (second_count + 1, second_char))
                #push back the original
                heapq.heappush(pq, (-count,letter))
            #add in one occurence of the largest:
            else:
                count -= 1
                result.append(letter)
                if count > 0:
                    heapq.heappush(pq, (-count,letter))
        
        return "".join(result)
        

#############################
# 1952. Three Divisors
# 17OCT24
#############################
class Solution:
    def isThree(self, n: int) -> bool:
        '''
        try all divisors from 1 to n
        '''
        divisors = 0
        for i in range(1,n+1):
            if n % i == 0:
                divisors += 1
        
        return divisors == 3
    
class Solution:
    def isThree(self, n: int) -> bool:
        '''
        a number can only have three divisors when it is a sqaure of a prime
        1,sqrt(num),and sqrt(num)*sqrt(num)
        generate all primes up sqrt(10**4)
        '''
        primes = [True]*(10**4 + 1)  
        start = 2
        
        while start*start <= 10*4:
            if primes[start] == True:
                for i in range(start*start,10**4 + 1,start):
                    primes[i] = False
            
            start += 1
        
        prime_nums = []
        for i in range(2,102):
            if primes[i]:
                prime_nums.append(i)
                
        if int(math.sqrt(n))**2 == n and int(math.sqrt(n)) in prime_nums:
            return True
        
        return False
    
#####################################
# 670. Maximum Swap (REVISTED)
# 17OCT24
#####################################
class Solution:
    def maximumSwap(self, num: int) -> int:
        '''
        brute force and save largest ans
        '''
        ans = num
        digits = []
        while num:
            digits.append(num % 10)
            num //= 10
        
        digits.reverse()
        N = len(digits)
        for i in range(N):
            for j in range(i+1,N):
                digits[i],digits[j] = digits[j],digits[i]
                ans = max(ans,self.get_num(digits))
                digits[i],digits[j] = digits[j],digits[i]
        
        return ans
                
    def get_num(self,arr):
        ans = 0
        for d in arr:
            ans = ans*10 + d
        
        return ans
    
#need max to left, then try swapping at each index it max
class Solution:
    def maximumSwap(self, num: int) -> int:
        '''
        brute force and save largest ans
        '''
        num = list(str(num))
        N = len(num)
        right_maxes = [[0,0] for _ in range(N)]
        right_maxes[N-1] = [int(num[-1]),N-1]
        
        for i in range(N-2,-1,-1):
            number = int(num[i+1])
            #update
            if number > right_maxes[i+1][0]:
                right_maxes[i] = [number,i+1]
            #carry left
            else:
                right_maxes[i] = right_maxes[i+1]
        
        
        for i in range(N):
            number = int(num[i])
            #if we can find a larger number swap it
            if number < right_maxes[i][0]:
                #swap
                num[i], num[right_maxes[i][1]] = num[right_maxes[i][1]],num[i]
                return int("".join(num))
            
        
        return int("".join(num))
    

###############################################################
# 2275. Largest Combination With Bitwise AND Greater Than Zero
# 18OCT24
##############################################################
class Solution:
    def largestCombination(self, candidates: List[int]) -> int:
        '''
        check at each each bit position if all bits for each num are set
        '''
        ans = 0
        for i in range(25):
            subset_size = 0
            mask = 1 << i
            for num in candidates:
                if num & mask:
                    subset_size += 1
            
            ans = max(ans,subset_size)
        
        return ans
    
############################################
# 1545. Find Kth Bit in Nth Binary String
# 19OCT24
############################################
#simulation works
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        '''
        we have the recurrence relation, but generating the actual string would take too long
        0
        0 + 1 + 1 = 011
        011 + 1 + 001 = 0111001
        simulate works i guess
        '''
        curr = [0]
        while n:
            next_ = curr + [1] + [1 - num for num in curr][::-1]
            curr = next_[:]
            n -= 1
        
        return str(curr[k-1])
    
#recursive approach
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        '''
        we have the recurrence relation, but generating the actual string would take too long
        0
        0 + 1 + 1 = 011
        011 + 1 + 001 = 0111001
        simulate works i guess
        
        string length doubles every time + 1
        1, 2*2 - 1, 2*2*2 - 1
        2^i - 1 for i >= 1
        1,3,7,15
        we just need to check if its on the left side or the right side
        left side the digit will remain the same, right sight, it will swap
        recursve approach, pass in length n and k
        if n == 1, it can only be '0'
        if k is in the firs half, we can call its on n-1, and k
        if its exactly in the middle, we can return 1 -> defined bu recurrense
        if its in the right half, then we know its going to be the kth om the end
        which would be length of the next string - k, but then need to return the flipped bit in the next call
        there are no repeated subproblems for this one
        '''
        def rec(n,k):
            if n == 1:
                return '0'
            
            #get the length of the current string
            size = 1 << n
            if k == size // 2:
                return '1'
            #if in the first half
            if k < size // 2:
                return rec(n-1,k)
            else:
                ans = rec(n-1,size-k)
                return '0' if ans == '1' else '1'
        
        return rec(n,k)
                
#iterative solution
class Solution:
    def findKthBit(self, n: int, k: int) -> str:
        '''
        for the iterative appaorch we start from the longest string -> 2**n - 1
        and we keep checking if k is in the first half or last half
        we also need to keep track of the number of time we inert a bit
        then we know we need to flip this bit depending on the parity of invert ont
        '''
        inverts = 0
        size = (1 << n) - 1
        
        while k > 1:
            #its is in the middel, return 1 but flip depending on parity
            if k == (size // 2) + 1:
                return '1' if inverts % 2 == 0 else '0'
            
            #if k is in right half, incement invert count
            if k > size // 2:
                inverts += 1
                #but now look at k from the end
                k = size - k + 1
            
            size = size  // 2
        
        return '0' if inverts % 2 == 0 else '1'

    
#########################################
# 1106. Parsing A Boolean Expression
# 20OCT24
##########################################
class Solution:
    def parseBoolExpr(self, expression: str) -> bool:
        '''
        its a valid expression
        indicators are ! -> negate
        & -> and
        | -> or
        i dont think we will have stuff like this !(t,t,t)
        use stack, and whenever we hit a closing, we need to evaluate whats one the stack
        '''
        stack = []
        for ch in expression:
            if ch == ',':
                continue
            if stack and ch == ')':
                vals = []
                while stack and stack[-1] != '(':
                    vals.append(stack.pop())
                
                stack.pop()
                operator = stack.pop()
                #eval here and push back
                eval_ans = self.evaluate(operator,vals)
                #print(operator,vals,eval_ans)
                stack.append(eval_ans)
            else:
                if ch == 'f':
                    stack.append(False)
                elif ch == 't':
                    stack.append(True)
                else:
                    stack.append(ch)
        
        return stack[0]
        
    def evaluate(self,op,vals):
        if op == '!':
            return not vals[0]
        elif op == '&':
            first = vals[0]
            for v in vals[1:]:
                first = first and v
            return first
        else:
            first = vals[0]
            for v in vals[1:]:
                first = first or v
            
            return first