#############################################
# 1065. Index Pairs of a String
# 01MAY23
############################################
class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        '''
        for each word in words, check if in text
        '''
        ans = []
        
        for word in words:
            N = len(word)
            for i in range(0,len(text) - N + 1):
                check = text[i:i+N]
                if check == word:
                    ans.append([i, i+N-1])
        
        ans.sort(key = lambda x: (x[0],x[1]))
        return ans
    
#hashmap
class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        '''
        hashset each word and just check all substrings
        '''
        words = set(words)
        N = len(text)
        ans = []
        for i in range(N):
            for j in range(i,N):
                substring = text[i:j+1]
                if substring in words:
                    ans.append([i,j])
        
        return ans
    
########################################
# 1822. Sign of the Product of an Array
# 01MAY23
#######################################
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        '''
        just check pairty of + signs, -signs, and count zeros
        if there is a zero:
            return 0
        positive signes don't matter
        if there are an even number of negative signes, its positive
        otherwise negative
        '''
        count_zeros = 0
        count_negatives = 0
        
        for num in nums:
            count_zeros += num == 0
            count_negatives += (num < 0)
        
        if count_zeros:
            return 0
        elif count_negatives % 2 == 0:
            return 1
        else:
            return -1
        
class Solution(object):
    def arraySign(self, nums):
        sign = 1
        for num in nums:
            if num == 0:
                return 0
            if num < 0:
                sign = -1 * sign

        return sign
    
##################
# 475. Heaters
# 02APR23
##################
#YASSSS
#careful with the max and min bounds
#binary search using workable solution
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        '''
        we are given arrays of houses and heaters,
        return the minimum heater radius so that all heaters can cover all the houses
        
        houses = [1,2,3,4]
        heaters = [1,4]
        
        let N = len(houses) - 1 = 3
        start with the first heater 1 advance all the way so that works
        try smaller, 2
        
        smallest answer can be 0
        largest answer can be max(houses) - min(houses)
        
        i can use binary search to find a workable solution, but I need an O(N) solution to check if the current workable solution works
        function to check if the current heater radius works
        '''
        left = 0
        right = max(max(houses),max(heaters)) - min(min(houses),min(heaters))
        #sort
        houses.sort()
        heaters.sort()
        
        def check_radius(heaters,houses,radius):
            i = 0
            for j in range(len(heaters)):
                curr_heater = heaters[j]
                #advance if within radius
                while i < len(houses) and abs(curr_heater - houses[i]) <= radius:
                    i += 1
            
            return i == len(houses)
        
        while left < right:
            #guess
            mid = left + (right - left) // 2
            if check_radius(heaters,houses,mid):
                right = mid
            else:
                left = mid + 1
        
        return right
                
#another binary search
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        '''
        sort housea and heaters in ascending order
        then for each house find the closest heater to the left and to the right (lower and upper bounds)
        if the house if to left of all the heaters, well this house has to use the first heater, heaters[0]
        if house is to the right of all the heaters, use the end
            in these cases, we need to take the maximum
        otherwise a house is in between two heaters, so take the minimum, then the max
        
        we turn a minimum problem to a maximum problem
        summary:
            for each house find the two closest heaters
            for this house the local minimum is just the close of the two heaters, and although it will work for this house
            it might not work for other houses, but we know that it cannot be any smaller than this current mimum
        '''
        #sort heaters
        heaters.sort()

        # Initialize the result to 0
        result = 0

        # For each house, find the closest heater to the left and the right
        for house in houses:
            left = bisect.bisect_right(heaters, house) - 1
            right = bisect.bisect_left(heaters, house)

            # If the house is to the left of all heaters, use the closest heater to the left
            if left < 0:
                result = max(result, heaters[0] - house)
            # If the house is to the right of all heaters, use the closest heater to the right
            elif right >= len(heaters):
                result = max(result, house - heaters[-1])
            # If the house is between two heaters, use the closer of the two
            else:
                result = max(result, min(house - heaters[left], heaters[right] - house))

        # Return the result
        return result
    
#one more
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        houses.sort()
        heaters.sort()
        res = 0
        heater = 0
        
        for h in houses:
            while heater + 1 < len(heaters) and heaters[heater + 1] == heaters[heater]:                  # Avoid duplicates
                heater += 1
            while heater + 1 < len(heaters) and abs(heaters[heater + 1] - h) < abs(heaters[heater] - h): # If using next heater is more efficient
                heater += 1                                                                              # Then use next heater
            
            res = max(res, abs(heaters[heater] - h))        # Update its range to house
        
        return res

#############################################
# 649. Dota2 Senate
# 04MAY23
############################################
#almost 72/80
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        '''
        game theory problem
        two parties, R and D,
        each senator has one of two rights:
            ban, make sentator lose all his rights and this following round
            announce victory, if senator found that the remaining senators are still from the same party, he can immediately announce the winnder
        
        each round consists of moving through the whole senate, and who every is left goes on to the next round
        
        get initailly counts, then q up,
        if there is a majority for this current senator's party, accnoune winnder, otherwise just ban an opposing member of the senator
        but who do they ban? does it make sense just to ban the next one coming in?, banning different senators will can give different final answers
        we need to ban the next closes opposing senator
        
        how many 
        '''
        count_radiant = 0
        count_dire = 0
        for vote in senate:
            count_radiant += vote == 'R'
            count_dire += vote == 'D'
        
        
        if count_radiant > count_dire:
            return "Radiant"
        elif count_dire > count_radiant:
            return "Dire"
        else:
            if senate[0] == 'R':
                return "Radiant"
            else:
                return "Dire"

#ummmm the brute force still passes
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        '''
        game theory problem
        two parties, R and D,
        each senator has one of two rights:
            ban, make sentator lose all his rights and this following round
            announce victory, if senator found that the remaining senators are still from the same party, he can immediately announce the winnder
        
        each round consists of moving through the whole senate, and who every is left goes on to the next round
        
        get initailly counts, then q up,
        if there is a majority for this current senator's party, accnoune winnder, otherwise just ban an opposing member of the senator
        but who do they ban? does it make sense just to ban the next one coming in?, banning different senators will can give different final answers
        we need to ban the next closes opposing senator
        why? when we ban the next closes opposing senator, we are restricting one senator from the other party
        think about it what if we were to ban the opposing senator at the end, we leave the other previous senators to ban senators on our party. so, it doesn't make sense to ban the last opposing senator, what about the second to last? same reason as the last
        
        how many rounds?
        for each senator, we can ban another sentor, so number of senators goes down by N/2, N/4....until N = 0, or N == 1, log_2(N) rounds
        
        how many votes? rather how many actions...
        N/2 actions, N/4 actions....
        sum_{i = 0}^{N} N*(1/2)^i convergent, geometric series to N
        
        approach 1: greedy
        * strategy is to best the next closes opposing senator
        * function ban(toBan, startAt), bans the closes oposing senator, starting at startAt, keep track if we looped round the whole seantor as a round
        * keep track of turn variable, which will keep track of current senator
        * while we have senators we keep banning
        * banning decrements count of senators
        * if senator was banned before  this index, it means the senator having the turn will be the senator at the same index, decremtn turn by 1
        * you can ban any senator, senate is a circular array
        
        '''
        senate = list(senate)
        counts = Counter(senate)
        
        def ban(to_ban, start_at):
            #ban starting at next index, we looped around the whole senate, it means the the next trun will be at the senator that started it
            loop = False
            ptr = start_at
            
            while True:
                if ptr == 0:
                    loop = True
                if senate[ptr] == to_ban:
                    senate.pop(ptr)
                    break
                ptr = (ptr + 1) % len(senate)
            
            return loop
        
        curr_senator = 0
        
        #while we have senators
        while counts['R'] > 0 and counts['D'] > 0:
            #ban phase
            if senate[curr_senator] == 'R':
                banned_senator = ban('D', (curr_senator + 1) % len(senate))
                counts['D'] -= 1
            else:
                banned_senator = ban('R', (curr_senator + 1) % len(senate))
                counts['R'] -= 1
            
            if banned_senator:
                curr_senator -= 1
            curr_senator = (curr_senator + 1) % len(senate)
        
        if counts['D'] == 0:
            return 'Radiant'
        else:
            return 'Dire'
        
#approach 2: boolean array
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        '''
        insteaf of looping around senatore, and keeping turn variable, keep boolean array of senators that are not banned
        
        '''
        N = len(senate)
        senate = list(senate)
        counts = Counter(senate)
        
        banned = [False]*N
        
        
        def ban(to_ban, start_at):
            #keep banning in ptr array
            ptr = start_at
            
            while True:
                #the one we need to ban and not banned, we ban them
                if senate[ptr] == to_ban and banned[ptr] == False:
                    banned[ptr] = True
                    break
                ptr = (ptr + 1) % len(senate)
        
        #we only need to keep track of current senator
        curr_senator = 0
        
        #while we have senators
        while counts['R'] > 0 and counts['D'] > 0:
            #ban phase
            if not banned[curr_senator]:
                if senate[curr_senator] == 'R':
                    ban('D', (curr_senator + 1) % len(senate))
                    counts['D'] -= 1
                else:
                    ban('R', (curr_senator + 1) % len(senate))
                    counts['R'] -= 1

            curr_senator = (curr_senator + 1) % len(senate)
        
        if counts['D'] == 0:
            return 'Radiant'
        else:
            return 'Dire'
        
#approach 3: binary search
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        '''
        we can use binary search to find the earliet opposing senator to ban
        since we always ban the next closest opposing, if we pick an index that has not been banned, we know that
        everything to the right, must not have been banned yet, so we discard the right
        same thing with the left
        
        maintin to two lists of eligible senators, to ban from
        then when we call, pass by reference the array we want to ban from,
        then binary search on that array
            in the case we cannot find the next senator to ban using binary search, we have to loop around and ban the first elibile senator
        
        keep banned boolean array, but for the eligible ones
        keep only the indices we want to ban from, for seperate radiant and dire
        
        then we just keep doing the rounds until we run out of eligible senators, no need to count
        '''
        N = len(senate)
        senate = list(senate)
        banned = [False]*N
        
        eligible_radiants = [i for i in range(N) if senate[i] == "R"]
        eligible_dires = [i for i in range(N) if senate[i] == 'D']
        
        def ban(eligible, start_at):
            #find next opposing senator to ban
            #i.e the inserction point one greater than the current idnex
            to_ban = bisect.bisect_left(eligible,start_at)
            
            #if we have gone around
            if to_ban == len(eligible):
                to_ban = eligible.pop(0)
                banned[to_ban] = True
            else:
                to_ban = eligible.pop(to_ban)
                banned[to_ban] = True

        #we only need to keep track of current senator
        curr_senator = 0
        
        #while we have senators
        while eligible_radiants and eligible_dires:
            #ban phase
            if not banned[curr_senator]:
                if senate[curr_senator] == 'R':
                    ban(eligible_dires, curr_senator)
                else:
                    ban(eligible_radiants,curr_senator)

            curr_senator = (curr_senator + 1) % len(senate)
        
        if eligible_radiants:
            return 'Radiant'
        else:
            return 'Dire'
        
#approch 4, two queues
class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        '''
        recall we stored the eligible sentor indices in two arrays, which we sorted
        at any one time, current turn would have been the minimum of these, and on that turn we would ban the next senator of the opposing party
        we can insteaf keep two queues for raditn and dire
        we popleft from both, the current turn is the minimum, 
        this senator cannot currently be banned, but would be banned in later rounds, so we put him back in the respective queue
        but + n, its next turn will be + n after moving through senate
        to create the effect of banning, we simly don't enque the larger to the two indices
        not en-queuing means this senator becomes banned
        '''
        N = len(senate)
        eligible_radiants = deque([])
        eligible_dires = deque([])
        
        for i,senator in enumerate(senate):
            #queue up the indices
            if senator == 'R':
                eligible_radiants.append(i)
            else:
                eligible_dires.append(i)
                
        while eligible_radiants and eligible_dires:
            curr_radiant = eligible_radiants.popleft()
            curr_dire = eligible_dires.popleft()
            
            #the curr turn is the min turn
            if curr_radiant < curr_dire:
                eligible_radiants.append(curr_radiant + N)
            else:
                eligible_dires.append(curr_dire + N)
                
        
        if not eligible_radiants:
            return "Dire"
        else:
            return "Radiant"
        
################################################################
# 1456. Maximum Number of Vowels in a Substring of Given Length
# 05MAY23
################################################################
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        '''
        careful here, k could == len(s)
        for the window keep count of the current number of vowels
        i can keep count mapp of current vowels in window (sum in constant time)
        then slide one at a time
        '''
        vowel_counts = {'a' : 0, 'e' : 0, 'i' : 0, 'o': 0, 'u': 0}
        #populate first substring k
        for i in range(k):
            if s[i] in vowel_counts:
                vowel_counts[s[i]] += 1
        
        ans = sum(vowel_counts.values())
        #mas this along the way
        left = 0
        
        for right in range(k,len(s)):
            if s[left] in vowel_counts:
                vowel_counts[s[left]] -= 1
            left += 1
            
            if s[right] in vowel_counts:
                vowel_counts[s[right]] += 1
            
            ans = max(ans, sum(vowel_counts.values()))
        
        return ans
    
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        '''
        we dont need to keep count map, just count of current chars in this window
        we don't need aux pointer for left, just do i -k
        '''
        vowels = {'a','e','i','o','u'}
        
        count_vowels = 0
        for i in range(k):
            count_vowels += s[i] in vowels
        
        
        ans = count_vowels
        
        for right in range(k,len(s)):
            count_vowels += s[right] in vowels
            count_vowels -= s[right - k] in vowels
            ans = max(ans,count_vowels)
        
        return ans
    
######################################################################
# 1498. Number of Subsequences That Satisfy the Given Sum Condition
# 06MAY23
#####################################################################
#edge cases suck on this one
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        '''
        return number of subsequences such that the sum of the min and max element in it <= target
        sor the nums array, then fix index i, for all possible left bounds of a subsequence
        use binary search to find the index j, such that nums[i] + nums[j] <= target
        the number of subsequences between i and j is just:
        say we have subarray 
        [1,2,3,4,5] k = 6
        bounds are at 1 and 5
        which means all subsequences starting at 1 should work!
        question now becomes, after finding a subarray that works, how many subsequence exist such that min(subarray) + max(subarray) <= target
        say we have [1,2,3], k = 4
        [1], [1,2], [1,2,3], [1,3]
        [2]
        we must include the left bound, but the array could be anysubset of left+1 right
        number of subsets would be 2^^[right - left + 1] + 1
        
        '''
        N = len(nums)
        nums.sort()
        ans = 0
        mod = 10**9 + 7
        
        for i in range(N):
            left = i
            right = len(nums)
            while left < right:
                mid = left + (right - left) // 2
                if nums[i] + nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            
            
            ans += int(2**(left - i - 1))
            ans %= mod
        
        return ans % mod
    
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        '''
        we need to find the insection for the largest right bound that we can find
        '''
        N = len(nums)
        mod = 10**9 + 7
        nums.sort()
        
        ans = 0
        
        for left in range(N):
            #find inserction point for target - nums[left]
            right = bisect.bisect_right(nums,target - nums[left])
            right -= 1
            
            if right >= left:
                ans += pow(2,right - left,mod)
                
        return ans % mod
    
#writing out binary search
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        '''
        we need to find the insection for the largest right bound that we can find
        '''
        N = len(nums)
        mod = 10**9 + 7
        nums.sort()
        
        ans = 0
        
        def upper_bound(array,look_for):
            left = 0
            right = len(array)
            
            while left < right:
                mid = left + (right - left) // 2
                if nums[mid] <= target:
                    left = mid + 1
                else:
                    right = mid - 1
            
            return left
        
        for left in range(N):
            #find inserction point for target - nums[left]
            right = upper_bound(nums,target - nums[left])
            if nums[left] + nums[right-1] <= target:
                if right - 1 >=left:
                    ans += pow(2,(right-1) - (left),mod)
        
        return ans
    
class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int:
        '''
        we dont need to check every fixed left bound for a canddiate subarray
        we can just use two pointers, and whenver we have a valid subarray, get the number of subsequences
        then advance the left pointer
        otherwise shift the right pointer
        '''
        n, mod = len(nums), 10 ** 9 + 7
        nums.sort()
        
        answer = 0
        left, right = 0, n - 1

        while left <= right:
            if nums[left] + nums[right] <= target:
                answer = (answer + pow(2, right - left, mod)) % mod
                left += 1
            else:
                right -= 1
        return answer
    
##################################################################
# 1964. Find the Longest Valid Obstacle Course at Each Position
# 07MAY23
#################################################################
#cant greedily build the subseqyuence
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        '''
        we are given obstacles if length n, where obstacles[i] describes the height
        for every index i in range(n) find longest obstalce in obstacles such that:
            * choose any between (0,i)
            * including i
            * put chosen obstalces in same order (lexographic) as obstacles
            * every obstalce (except first) is >= as the one before it
            
        return ans array where ans[i] is the height of the obstalce course for index i, as desriberd abover
        
        for each index i
            ans[i] is just a montonic subsequence of obstalces (except the first)
            
        brute force is easy, just build the obstalce course and return its length, start with that
        but i would need to build the course in a reverse, and make it motonic decreasing, except for the first
        longest increasing subsequence, i can't greddily build it
        
        really this is just longest decreasing subsequence
        ''' 
        N = len(obstacles)
        ans = [1]*N
        
        for i in range(N):
            #build in reverse
            curr_course = [obstacles[i]]
            j = i - 1
            while j >= 0:
                if obstacles[j] <= curr_course[-1]:
                    curr_course.append(obstacles[j])
                j -= 1
            #print(curr_course)
            ans[i] = len(curr_course)
        
        return ans
    
#naive LIS (longest increasing subsequence), TLE
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        '''
        we are given obstacles if length n, where obstacles[i] describes the height
        for every index i in range(n) find longest obstalce in obstacles such that:
            * choose any between (0,i)
            * including i
            * put chosen obstalces in same order (lexographic) as obstacles
            * every obstalce (except first) is >= as the one before it
            
        return ans array where ans[i] is the height of the obstalce course for index i, as desriberd abover
        
        for each index i
            ans[i] is just a montonic subsequence of obstalces (except the first)
            
        brute force is easy, just build the obstalce course and return its length, start with that
        but i would need to build the course in a reverse, and make it motonic decreasing, except for the first
        longest increasing subsequence, i can't greddily build it
        
        really this is just longest decreasing subsequence, but if i apply LIS (reverse), that problem runs in O(N^2) times
        just build along the way you idiot!
        
        reframe problem is longest decreasing subsequence using obstalces[i]
        ''' 
        N = len(obstacles)
        ans = [1]*N
        
        memo = {}
        
        def dp(i):
            if i < 0:
                return 0
            if i in memo:
                return memo[i]
            ans = 0
            for j in range(i-1,-1,-1):
                if obstacles[j] <= obstacles[i]:
                    ans = max(ans,1+dp(j))
            
            memo[i] = ans
            return ans
        
    
        for i in range(N):
            ans[i] = 1+dp(i)
        
        return ans
    
#we need to use the binary seach solution from LIS in order for this to pass
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        '''
        recall the solution from LIS where we intelligently build a subsequence, in the brute force solution
        we keep greedidly adding to our current subsequence until we can't 
        then we have to scan the array starting from the smallest element and replace the first element that is greater than or equal to num with hum
        class Solution:
            def lengthOfLIS(self, nums: List[int]) -> int:
                sub = [nums[0]]

                for num in nums[1:]:
                    if num > sub[-1]:
                        sub.append(num)
                    else:
                        # Find the first element in sub that is greater than or equal to num
                        i = 0
                        while num > sub[i]:
                            i += 1
                        sub[i] = num

                return len(sub)
                
        then it was just binary search to find the upper bound
        
        class Solution:
            def lengthOfLIS(self, nums: List[int]) -> int:
                sub = []
                for num in nums:
                    i = bisect_left(sub, num)

                    # If num is greater than any element in sub
                    if i == len(sub):
                        sub.append(num)

                    # Otherwise, replace the first element in sub greater than or equal to num
                    else:
                        sub[i] = num

                return len(sub)
        in short, the longest course ending at index i depends on the ending before index i
        we need to store all the previous obstalce courses we have met before index i
        then for the obstalce at index i, we can choose any course that had a final obtacle <= obstacles[i] and add in obstacles[i]
        we greedily choose the longest one out of them to make the longest course
        
        problem? there could be many sequences with the same length and its impracticel to store all of them
        we keep the course with the obstacle that has the smaller ending height
        we don't need to keep track of the the path, just the heights of the last obstacle being added in
        keep an array to record the height of the shortest ending obstacle for course 
        lis[i] is the heigh of the shortest ending obstalce of length i+1
        lis[4] = 7 means the lowest end of a course with length 4 we have met so far is 7
        
        answer at i is the index at  position idx, the idx where we peformed binary search
        we need to find the upper bound, the last possible solution we could put an obstacle in, and its insert position would be 1 less than that
        
        '''
        N = len(obstacles)
        ans = [1]*N
        
        lis = [] #store the smallest heights (list[i]) for longest increasing subsequences with length i+1
        
        for i, height in enumerate(obstacles):
            #find right most position
            idx = bisect.bisect_right(lis,height)
            #if its larger than everything else
            if idx == len(lis):
                lis.append(height)
            #otherwise, we can use a smaller heigh
            else:
                lis[idx] = height
            #longest has to be using this height
            ans[i] = idx + 1
        
        return ans
    
#brute force with linear scan
class Solution:
    def longestObstacleCourseAtEachPosition(self, obstacles: List[int]) -> List[int]:
        piles = []
        res = []
        for n in obstacles:
            found_pile = False
            for i in range(len(piles)):
                if piles[i][-1] > n:
                    piles[i].append(n)
                    res.append(i + 1)
                    found_pile = True
                    break
            if not found_pile:
                piles.append([n])
                res.append(len(piles))
        print(piles)
        return res
