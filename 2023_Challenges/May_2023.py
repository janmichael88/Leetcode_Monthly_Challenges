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

########################################
# 1572. Matrix Diagonal Sum
# 08MAY23
########################################
#welp it works, but i dont like it
class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        '''
        walk the diagonals, the problem is that the intersect
        '''
        rows = len(mat)
        cols = len(mat[0])
        seen = set()
        ans = 0
        #staring top left, and going down
        i,j = 0,0
        while i < rows and j < cols:
            ans += mat[i][j]
            print(mat[i][j])
            seen.add((i,j))
            i += 1
            j += 1
        
        #starting bottom left
        i = rows -1
        j = 0
        
        while i >= 0 and j < cols:
            if (i,j) not in seen:
                ans += mat[i][j]
                #print(mat[i][j])
                seen.add((i,j))
            i -= 1
            j += 1
        
        return ans
    
class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        '''
        what if we want to save space
        if there an even number rows/cols there is no double count
        if there is an odd number we have a double count
        '''
        rows = len(mat)
        cols = len(mat[0])
        ans = 0
        #staring top left, and going down
        i,j = 0,0
        while i < rows and j < cols:
            ans += mat[i][j]
            i += 1
            j += 1
        
        #starting bottom left
        i = rows -1
        j = 0
        
        while i >= 0 and j < cols:
            ans += mat[i][j]
            i -= 1
            j += 1
        
        if rows % 2 == 1:
            ans -= mat[rows//2][cols//2]
        return ans
        
class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        '''
        primary diagnols lay in (i,i) for i in range(len(mat))
        secondary diagonals lay in (n-i-1,i) for i in ranget(len(mat))
        '''
        n = len(mat)
        ans = 0

        for i in range(n):
            # Add elements from primary diagonal.
            ans += mat[i][i]
            # Add elements from secondary diagonal.
            ans += mat[n - i - 1][i]
        # If n is odd, subtract the middle element as its added twice.
        if n % 2 != 0:
             ans -= mat[n // 2][n // 2]
        
        return ans

###################################
# 311. Sparse Matrix Multiplication
# 08MAY23
###################################
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        inputs allow for naive implementation
            dot product between row and column
        '''
        m = len(mat1)
        n = len(mat1[0]) #which is also == len(mat2[0])
        r = len(mat2[0])
        
        ans = [[0]*r for _ in range(m)]
        
        for i in range(m):
            for j in range(r):
                ij_entry = 0
                for k in range(n):
                    ij_entry += mat1[i][k]*mat2[k][j]
                
                ans[i][j] = ij_entry
        
        return ans
    
#skipping on zeros whil traversing and adding directly to output matrix
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        inputs allow for naive implementation
            dot product between row and column
        '''
        m = len(mat1)
        n = len(mat1[0]) #which is also == len(mat2[0])
        r = len(mat2[0])
        
        ans = [[0]*r for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                if mat1[i][j] != 0:
                    #distribute over rows in the output
                    for k in range(r):
                        ans[i][k] += mat1[i][j]*mat2[j][k]

        return ans
    
#list of lists
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        for sparse matrix compress in format list of lists
        lists[i] is at row i ands stores (value,column) of non zero elements
        any element with index (row1,col1) in mat1 is multiplied with all elements of the col1'th row of mat2
        '''
        def compress_matrix(matrix):
            rows = len(matrix)
            cols = len(matrix[0])
            comp_mat = [[] for _ in range(rows)]
            for row in range(rows):
                for col in range(cols):
                    if matrix[row][col] != 0:
                        comp_mat[row].append([matrix[row][col],col])
            
            return comp_mat
        
        M = len(mat1)
        K = len(mat1[0])
        N = len(mat2[0])
        
        ans = [[0]*N for _ in range(M)]
        
        #compress
        A = compress_matrix(mat1)
        B = compress_matrix(mat2)
        
        
        for mat1_row in range(M):
            for element_1,mat1_col in A[mat1_row]:
                for element_2,mat2_col in B[mat1_col]:
                    ans[mat1_row][mat2_col] += element_1*element_2
        
        return ans
    
#sparse row and sparse col
class SparseMatrix:
    def __init__(self, matrix: List[List[int]], col_wise: bool):
        self.values, self.row_index, self.col_index = self.compress_matrix(matrix, col_wise)

    def compress_matrix(self, matrix: List[List[int]], col_wise: bool):
        return self.compress_col_wise(matrix) if col_wise else self.compress_row_wise(matrix)

    # Compressed Sparse Row
    def compress_row_wise(self, matrix: List[List[int]]):
        values = []
        row_index = [0]
        col_index = []

        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if matrix[row][col]:
                    values.append(matrix[row][col])
                    col_index.append(col)
            row_index.append(len(values))

        return values, row_index, col_index

    # Compressed Sparse Column
    def compress_col_wise(self, matrix: List[List[int]]):
        values = []
        row_index = []
        col_index = [0]

        for col in range(len(matrix[0])):
            for row in range(len(matrix)):
                if matrix[row][col]:
                    values.append(matrix[row][col])
                    row_index.append(row)
            col_index.append(len(values))

        return values, row_index, col_index
    

class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        yale format, compressed sparse row (CSR), compressed sparse col (CSC)
        three arrays
            values: array containing all non zero eleements
            colIndex: array containg column index of all non zero elements
            rowIndex: array storing valuesf of startindex of each rows
        
        len(values) == len(colIndex) === number of nonzero elements in matrix
        rowIndex = num_rows + 1, where 
            rowIndex[i] to rowIndex[i+1] (non inclusive) gives us the range index where the ith row element of matrix ar stored
            
        we compress mat1 as CSR, and mat2 as CSC, then we can easily find an element in row i of mat1 and in col j of mat2
        we can use the two pointer tricker and we keep advnacing until row i == cold j
        '''
        A = SparseMatrix(mat1,False)
        B = SparseMatrix(mat2,True)
        
        ans = [[0] * len(mat2[0]) for _ in range(len(mat1))]

        for row in range(len(ans)):
            for col in range(len(ans[0])):

                # Row element range indices
                mat1_row_start = A.row_index[row]
                mat1_row_end = A.row_index[row + 1]

                # Column element range indices
                mat2_col_start = B.col_index[col]
                mat2_col_end = B.col_index[col + 1]

                # Iterate over both row and column.
                while mat1_row_start < mat1_row_end and mat2_col_start < mat2_col_end:
                    if A.col_index[mat1_row_start] < B.row_index[mat2_col_start]:
                        mat1_row_start += 1
                    elif A.col_index[mat1_row_start] > B.row_index[mat2_col_start]:
                        mat2_col_start += 1
                    # Row index and col index are same so we can multiply these elements.
                    else:
                        ans[row][col] += A.values[mat1_row_start] * B.values[mat2_col_start]
                        mat1_row_start += 1
                        mat2_col_start += 1
    
        return ans

#############################################
# 59. Spiral Matrix II
# 10MAY23
############################################
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        '''
        we are given n and want to generate the matrix in spiral order
        allocate an empty matrix for the answer then walk the spiral using code from spiral matrix1
        '''
        ans = [[0]*n for _ in range(n)]
        numbers = [i for i in range(1,n*n + 1)]
        ptr = 0
        
        rows, columns = len(ans), len(ans[0])
        up = left = 0
        right = columns - 1
        down = rows - 1
        
        while ptr < len(numbers):
            # Traverse from left to right.
            for col in range(left, right + 1):
                ans[up][col] = numbers[ptr]
                ptr += 1
                

            # Traverse downwards.
            for row in range(up + 1, down + 1):
                ans[row][right] = numbers[ptr]
                ptr += 1

            # Make sure we are now on a different row.
            if up != down:
                # Traverse from right to left.
                for col in range(right - 1, left - 1, -1):
                    ans[down][col] = numbers[ptr]
                    ptr += 1

            # Make sure we are now on a different column.
            if left != right:
                # Traverse upwards.
                for row in range(down - 1, up, -1):
                    ans[row][left] = numbers[ptr]
                    ptr += 1

            left += 1
            right -= 1
            up += 1
            down -= 1
        
        return ans
    
#no need to keep array of numbers, just increment
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        '''
        we are given n and want to generate the matrix in spiral order
        allocate an empty matrix for the answer then walk the spiral using code from spiral matrix1
        '''
        ans = [[0]*n for _ in range(n)]
        ptr = 1
        
        rows, columns = len(ans), len(ans[0])
        up = left = 0
        right = columns - 1
        down = rows - 1
        
        while ptr < n*n + 1:
            # Traverse from left to right.
            for col in range(left, right + 1):
                ans[up][col] = ptr
                ptr += 1
                

            # Traverse downwards.
            for row in range(up + 1, down + 1):
                ans[row][right] = ptr
                ptr += 1

            # Make sure we are now on a different row.
            if up != down:
                # Traverse from right to left.
                for col in range(right - 1, left - 1, -1):
                    ans[down][col] = ptr
                    ptr += 1

            # Make sure we are now on a different column.
            if left != right:
                # Traverse upwards.
                for row in range(down - 1, up, -1):
                    ans[row][left] = ptr
                    ptr += 1

            left += 1
            right -= 1
            up += 1
            down -= 1
        
        return ans
    
#the boundary conditions freaking suck
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        '''
        traverse in layers. how many layers are there for a given n?
        n = 1, layer = 1
        n = 2, layer = 1
        n = 3, layer = 2
        n = 4, layer = 2
        n = 5, layer = 3
        n = 6, layer = 3
        
        layers = (n + 1) // 2 
        for n in range(1,7):
            print(n,(n+1)//2)
            
        Direction 1: From top left corner to top right corner.
            row remains constant as layer and column in [layer,n-layer-1]
        
        Direction 2: From top right corner to the bottom right corner.
            col remains as n - layer - 1 and row moves [layer+1,n-layer]
        
        Direction 3: From bottom right corner to bottom left corner.
            row remains as n-layer-1 and col moves [n-layer-2,layer]
            
        Direction 4: From bottom left corner to top left corner.
            col remains as layer and column decrements [n-layer-2,layer+1]
        '''
        matrix = [[0]*n for _ in range(n)]
        curr_num = 1
        for layer in range(0,(n+1)//2,1):
            #direction 1
            for col in range(layer,n-layer):
                matrix[layer][col] = curr_num
                curr_num += 1
            
            #direction 2
            for row in range(layer+1,n-layer):
                matrix[row][n-layer-1] = curr_num
                curr_num += 1
                
            #direction 3
            for col in range(n-layer-2,layer-1,-1):
                matrix[n-layer-1][col] = curr_num
                curr_num += 1
                
            #direction 4
            for row in range(n-layer-2,layer,-1):
                matrix[row][layer] = curr_num
                curr_num += 1
        
        for r in matrix:
            print(r)
        
        return matrix

###################################
# 486. Predict the Winner
# 09MAY23
###################################
#yessssss!! holy shit ballz! 
#game theory is cool!
#dont ever forget the negation of winning on the opposite turn in line 966
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        '''
        similar to can i win, state space exploration using recursion
        if player one can win this state
        game ends when there are no more elements in the array
        if there are no more elements in the array
        player 1 wins if player1_score >= player2_score
        
        try brute force, keep two pointers at the ends left and right, then shrink them keeping track of scores
        memo along the way
        
        '''
        memo = {}
        
        def dp(left,right,curr_turn,p1_score,p2_score): #left and right will be 0 and len(nums) - 1
            #pointer cross over
            if left > right:
                return p1_score >= p2_score
            
            if (left,right,curr_turn,p1_score,p2_score) in memo:
                return memo[(left,right,curr_turn,p1_score,p2_score)]
            
            #p1 turn, can win from this state by either taking left or taking right
            if curr_turn == 1:
                #try both
                take_left = dp(left+1,right,2,p1_score + nums[left],p2_score)
                take_right = dp(left,right-1,2,p1_score + nums[right],p2_score)
                #can win from here
                if take_left or take_right:
                    ans = True
                else:
                    ans = False
                memo[(left,right,curr_turn,p1_score,p2_score) ] = ans
                return ans
            
            #p2 turn else, if player 2 wins on this turn by taking left or right, player 1 will always win on the next turn
            else:
                take_left = dp(left+1,right,1,p1_score,p2_score + nums[left])
                take_right = dp(left,right-1,1,p1_score,p2_score + nums[right])
                if take_left and take_right:
                    ans = True
                else:
                    ans = False
                
                memo[(left,right,curr_turn,p1_score,p2_score)] = ans
                return ans
            
            
        return dp(0,len(nums)-1,1,0,0)
    
#offical LC solution
#top down
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        '''
        solving indirectly
        the problem isn't asking for what the final scores will be, but only if player 1 will have a score >= player 2
        so we insteaf focus on the score difference and as along a player1 >= player2, player1 wins
        
        dp(left,right) returns the the maximum score difference usings nums[left;right] >= 0
        if we want to find the differencce in scores by taking nums[left]
        it would be nums[left] - dp(left+1,right)
        similar for right it would be nums[right] - dp(left,right-1)
        then we take the max of these two
        
        and we want to to ensure that dp(left,right) >= 0
        
        dp is the max score difference for the current player, is the firstt call dp(0,len(nums)-1) is from player1
        but when we advance dp(left+1,right) or dp(left,righ-1) it will be from the persepctive of the opposite player
        to get the difference for player 1, we take nums[left] or nums[right], then find the the differnce on the smaller subproblem
        we then maximize the choices
        '''
        memo = {}
        
        def dp(left,right):
            if left == right:
                return nums[left]
            if (left,right) in memo:
                return memo[(left,right)]
            take_left = nums[left] - dp(left+1,right)
            take_right = nums[right] - dp(left,right-1)
            ans = max(take_left,take_right)
            memo[(left,right)] = ans
            return ans
        
        return dp(0,len(nums)-1) >= 0
    

#bottom up
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        '''
        bottom up
        '''
        
        n = len(nums)
        dp = [[0]*(n+1) for _ in range(n+1)]
        
        #base cases
        for left in range(n):
            for right in range(n):
                if left == right:
                    dp[left][right] = nums[left]
                    
        #we need to start to try all allowable [left,right] boundaries
        #gaps between left and right essentiall
        for gap in range(1,n):
            for left in range(n-gap):
                right = left + gap
                take_left = nums[left] - dp[left+1][right]
                take_right = nums[right] - dp[left][right-1]
                ans = max(take_left,take_right)
                dp[left][right] = ans
        
        return dp[0][n-1] >= 0
    
#solving directly top down
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        '''
        for player 1, with choosbale (left,right)
        1. if player picks nums[left],second player can pick from nums[left+1,right]
            if second player picks nums[left+1], player 1 on the next turn can get nums[left+2,right]
            if second picks nums[right], player1 on the enxt turn can choose from nums[left+1,right-1]
            since second plays to maximize score, 1 is left to choose
                nums[left] + min(dp(left+2,right), dp(left+1,right-1))
            
                
        2. is just the reverse if player picks for the right
        we only look ahead to the next move for plaeyaer 1
        '''
        memo = {}
        
        def dp(left,right):
            if left > right:
                return 0
            
            if (left,right) in memo:
                return memo[(left,right)]
            
            choose_left = nums[left] + min(dp(left+2,right),dp(left+1,right-1))
            choose_right = nums[right] + min(dp(left,right-2),dp(left+1,right-1))
            
            ans = max(choose_left,choose_right)
            memo[(left,right)] = ans
            return ans
        
        total_score = sum(nums)
        player1_score = dp(0,len(nums)-1)
        
        return player1_score >= total_score - player1_score
    
#bottom up
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        '''
        bottom up
        '''
        total_score = sum(nums)
        n = len(nums)
        
        dp = [[0]*(n+2) for _ in range(n+2)]
        for gap in range(n):
            for left in range(n-gap):
                right = left + gap
                
                choose_left = nums[left] + min(dp[left+2][right],dp[left+1][right-1])
                choose_right = nums[right] + min(dp[left][right-2],dp[left+1][right-1])
                ans = max(choose_left,choose_right)
                dp[left][right] = ans
        
        p1_score = dp[0][len(nums)-1]
        return p1_score >= total_score - p1_score


#############################################
# 1035. Uncrossed Lines (REVISTED)
# 11MAY23
############################################
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can draw lines from nums1[i] to nums2[j] only iff they are == and we are not intersecting another line
        return max number of connecting lines this way
        dp(i,j) gives the max number number of uncrosssed lines using nums1[i:] and nums2[j:]
        
        if i >= len(nums1) or j >= len(nums2):
            return 0
        
        we want the answer to dp(0,0)
        if nums1[i] == nums2[j]:
            #we can add to the maximum
            1 + dp(i+1,j+1)
        else:
            #we have to carry over the maximum
            max(dp(i+1,j),dp(i,j+1))
        '''
        memo = {}
        
        def dp(i,j):
            if i >= len(nums1) or j >= len(nums2):
                return 0
            
            if (i,j) in memo:
                return memo[(i,j)]
            if nums1[i] == nums2[j]:
                ans = 1 + dp(i+1,j+1)
                memo[(i,j)] = ans
                return ans
            else:
                ans = max(dp(i+1,j),dp(i,j+1))
                memo[(i,j)] = ans
                return ans
        
        return dp(0,0)
        
#bottom up
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        bottom up
        '''
        dp = [[0]*(len(nums2)+1) for _ in range(len(nums1)+1)]
        
        for i in range(len(nums1)-1,-1,-1):
            for j in range(len(nums2)-1,-1,-1):
                if nums1[i] == nums2[j]:
                    ans = 1 + dp[i+1][j+1]
                    dp[i][j] = ans
                else:
                    ans = max(dp[i+1][j],dp[i][j+1])
                    dp[i][j] = ans
        
        return dp[0][0]

#bottom up space save
class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        bottom up
        we only need top and bottom rows at any one times
        '''
        dp = [[0]*(len(nums2)+1) for _ in range(2)]
        
        for i in range(len(nums1)-1,-1,-1):
            for j in range(len(nums2)-1,-1,-1):
                if nums1[i] == nums2[j]:
                    ans = 1 + dp[1][j+1]
                    dp[0][j] = ans
                else:
                    ans = max(dp[1][j],dp[0][j+1])
                    dp[0][j] = ans
            
            #swap rows
            dp[1] = dp[0][:]
        return dp[0][0]
    
####################################
# 908. Smallest Range I
# 11MAY23
#####################################
#fuck me
class Solution:
    def smallestRangeI(self, nums: List[int], k: int) -> int:
        '''
        lots of dislikes on this problem
        i can change any number, nums[i] to num[i] + x
        where x in in the range [-k,k]
        i can do this operation at most once for each index i
        
        score is defined as the diff between max and min elements
        
        to minimze the score, we want to maximize the min and minimuze the max
        [1,3,6] k = 3
        [-2,4],[0,6],[3,9]
        but i can only choose 1 in each of the ranges
        
        if i can send all the numbers to the same number, the answer if zero
        what if added the numbers new array?
        '''
        #easy case, size 1, return 0
        if len(nums) == 1:
            return 0
        
        smallest = min(nums)
        largest = max(nums)
        
        score = float('inf')
        
        for num in nums:
            #shift as far as we can
            upshift = num + k
            downshift = num - k
            
            smallest = min(smallest,upshift,downshift)
            largest = max(largest,upshift,downshift)
            
            score = min(score,largest-smallest)
        
        return score
    
class Solution:
    def smallestRangeI(self, nums: List[int], k: int) -> int:
        '''
        If min(A) + K < max(A) - K, then return max(A) - min(A) - 2 * K
        If min(A) + K >= max(A) - K, then return 0
        
        idea, if we raise the smallest number, in nums, i.e min(nums) + K and it can be greater than max(nums) - K
        then any number larger than min nums can also be brought to any number in the range [max(nums)-k,max_nums]
        so we strive to make all nums in this range
        if we cant, then we need to try our best to minimze the score, the only way to do that would be to raise the min as far as we can
        and bring down the max as far as we can
        [ max(nums) - k ] - [ min(nums) + k ]
        
        max(nums) - min(nums) - k - k
        '''
        
        if min(nums) + k >= max(nums) - k:
            return 0
        
        else:
            return max(nums) - min(nums) - k - k
        
#############################################
# 2140. Solving Questions With Brainpower
# 12MAY23
############################################
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        '''
        0-1 knapsack
        must go in order of left to right for questions[i]
        i can choose to solve question[i] and earn questions[i][0] points
        but i have to skip questions[i+j] for j in range(i +questions[i][1])
        
        let dp(i) be the max points i can get starting from questions i
        '''
        
        memo = {}
        
        def dp(i):
            if i >= len(questions):
                return 0
            if i in memo:
                return memo[i]
            
            #answer question
            answer = questions[i][0] + dp(i+questions[i][1]+1) #careful with the index bounds here!
            dont_answer = dp(i+1)
            ans = max(answer,dont_answer)
            memo[i] = ans
            return ans
        
        return dp(0)
    
#bottom up, boundary conditions suckkkk
class Solution:
    def mostPoints(self, questions: List[List[int]]) -> int:
        '''
        bottom up, the only part is trying to access the last spot
        in cases where the skips can be very larger
        precompute skipped
        '''
        N = len(questions)
        dp = [0]*(N+1)
        
        for i in range(N-1,-1,-1):
            skipped = 0 if (i + questions[i][1] + 1) >= N  else dp[i+questions[i][1] + 1]
            answer = questions[i][0] + skipped
            dont_answer = dp[i+1]
            ans = max(answer,dont_answer)
            dp[i] = ans
        
        return dp[0]

#########################################
# 2466. Count Ways To Build Good Strings
# 13MAY23
#########################################
#brute force, TLE
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        '''
        zero is the number of times we can add a zero to the string
        one is the number of times we can a one to the string
        a good string s:
            low <= len(s) <= high
        
        give number of different strings that an be constructed
        brute force:
            keep track of string length then apply the operations
        '''
        self.num_ways = 0
        
        def rec(curr_length):
            if curr_length > high:
                return
            if low <= curr_length <= high:
                self.num_ways += 1
                self.num_ways %= 10**9 + 7
            rec(curr_length+zero)
            rec(curr_length+one)
        
        rec(0)
        return self.num_ways

#top down
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        '''
        now try memozing states
        
        '''
        memo = {}
        mod = 10**9 + 7
        
        def rec(curr_length):
            if curr_length > high:
                return 0
            if curr_length in memo:
                return memo[(curr_length)]
            ans = 0
            if low <= curr_length <= high:
                ans += 1

            try_zero = rec(curr_length+zero)
            try_one = rec(curr_length+one)
            ans += try_zero + try_one
            memo[(curr_length)] = ans % mod
            return ans % mod
        
        return rec(0) % mod
    
#bottom up, again be careful with the boundary check
class Solution:
    def countGoodStrings(self, low: int, high: int, zero: int, one: int) -> int:
        '''
        bottom up
        curr_length cannot be more than heigh, just watch the boundary conditions
        '''
        
        mod = 10**9 + 7
        dp =[0]*(high + max(one,zero)+1)
        
        #start from curr_length == high
        for curr_length in range(high,-1,-1):
            ans = 0
            if low <= curr_length <= high:
                ans += 1
            
            try_zero = dp[curr_length+zero]
            try_one = dp[curr_length+one]
            ans += try_zero + try_one
            dp[curr_length] = ans % mod
            
        return dp[0] % mod

#########################################
# 1799. Maximize Score After N Operations
# 14MAY23
#########################################
#brute force TLE
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        '''
        given nums array of length 2*n:
        len(nums) = n
        on the ith operation, we can choose two eleemnts (x,y)
        increment score by i*gcd(x,y)
        remove x and y from nums
        
        n is small, 1 <= n <= 7
        len(nums) is always even, so we can always do an operation
        
        hint 1, find every way split array until n groups of n, brute force recursion is acceptable
        keep track of curr_score and mask
        '''
        def gcd(x,y):
            if y == 0:
                return x
            else:
                return gcd(y,x % y)
            
        
        self.ans = 0
        n = len(nums)//2
        
        def rec(op_number,nums_taken,taken):
            #all are taken
            if op_number == n:
                curr_score = 0
                curr_step = 1
                for i in range(0,len(nums),2):
                    curr_score += curr_step*gcd(nums_taken[i],nums_taken[i+1])
                    curr_step += 1
                
                self.ans = max(self.ans,curr_score)
                return
            
            for i in range(len(nums)):
                for j in range(i+1,len(nums)):
                    if taken[i] or taken[j]:
                        continue
                    
                    #otherwise take
                    taken[i] = True
                    taken[j] = True
                    nums_taken.append(nums[i])
                    nums_taken.append(nums[j])
                    rec(op_number+1,nums_taken,taken)
                    #backtrack?
                    taken[i] = False
                    taken[j] = False
                    nums_taken.pop()
                    nums_taken.pop()
        
        rec(0,[],[False]*len(nums))
        return self.ans
    
#caching states, using arrays
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        '''
        we can pass a mask of indices taken
        base case is that we return 0 when we have no numbers to pick
        '''
        def gcd(x,y):
            if y == 0:
                return x
            else:
                return gcd(y,x % y)
            
        memo = {}
        n = len(nums) // 2
        def dp(operations,taken):
            if operations == n:
                return 0
            
            if (operations,tuple(taken)) in memo:
                return memo[(operations,tuple(taken))]
            
            #maximize this current state
            max_score = 0
            for i in range(len(nums)):
                for j in range(i+1,len(nums)):
                    #if we cant take a pair
                    if taken[i] or taken[j]:
                        continue
                        
                    #take them
                    taken[i] = True
                    taken[j] = True
                    
                    #get score
                    curr_score = (operations+1)*gcd(nums[i],nums[j])
                    max_score = max(max_score,curr_score + dp(operations+1,taken))
                    taken[i] = False
                    taken[j] = False
                
            
            memo[(operations,tuple(taken))] = max_score
            return max_score
        
        
        taken = [False]*len(nums)
        return dp(0,taken)
    
#if we use a bit mask, we don't need to untake it
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        '''
        we can pass a mask of indices taken
        base case is that we return 0 when we have no numbers to pick
        '''
        def gcd(x,y):
            if y == 0:
                return x
            else:
                return gcd(y,x % y)
            
        memo = {}
        n = len(nums) // 2
        def dp(operations,taken):
            if operations == n:
                return 0
            
            if (operations,taken) in memo:
                return memo[operations,taken]
            
            #maximize this current state
            max_score = 0
            for i in range(len(nums)):
                for j in range(i+1,len(nums)):
                    #if we cant take a pair
                    if (taken >> i) & 1 == 1 or (taken >> j) & 1 == 1:
                        continue
                        
                    #take them
                    taken = taken | (1 << i)
                    taken = taken | (1 << j)
                    
                    #get score
                    curr_score = (operations+1)*gcd(nums[i],nums[j])
                    max_score = max(max_score,curr_score + dp(operations+1,taken))
                    
                    #un take them
                    taken = taken ^ (1 << i)
                    taken = taken ^ (1 << j)
                
            
            memo[(operations,taken)] = max_score
            return max_score
        
        
        taken = 0
        return dp(0,taken)
    
#instead of untaking them just make a new mask
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        '''
        we can pass a mask of indices taken
        base case is that we return 0 when we have no numbers to pick
        '''
        def gcd(x,y):
            if y == 0:
                return x
            else:
                return gcd(y,x % y)
            
        memo = {}
        n = len(nums) // 2
        def dp(operations,taken):
            if operations == n:
                return 0
            
            if taken in memo:
                return memo[taken]
            
            #maximize this current state
            max_score = 0
            for i in range(len(nums)):
                for j in range(i+1,len(nums)):
                    #if we cant take a pair
                    if (taken >> i) & 1 == 1 or (taken >> j) & 1 == 1:
                        continue
                        
                    #take them
                    new_taken = taken | (1 << i)
                    new_taken = new_taken | (1 << j)
                    
                    #get score
                    curr_score = (operations+1)*gcd(nums[i],nums[j])
                    max_score = max(max_score,curr_score + dp(operations+1,new_taken))

            
            memo[taken] = max_score
            return max_score
        
        
        taken = 0
        return dp(0,taken)
    
'''
time complexity
let m be the number of elements in nums, len(nums) == m == 2*n
let A be the largest elements in numbs
O(2^(2n)*(2n)^2 * log A)
logA comes from the gcd function
2^m calls in total for each mask (using memo)
inner call is m^2 times logA

space complexity = O(n + 2^(2n)) = O(4^n)


'''
#bottom up
#careful about accessing only valid states in bottom up\
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        '''
        we can pass a mask of indices taken
        base case is that we return 0 when we have no numbers to pick
        '''
        def gcd(x,y):
            if y == 0:
                return x
            else:
                return gcd(y,x % y)
            
        #brian kernighan trick
        def count_ones(n):
            count = 0
            while n:
                count += 1
                n = n & (n-1)
            return count
            
        
        states = 1 << len(nums) # 2^(nums array size)

        # 'dp[i]' stores max score we can get after picking remaining numbers represented by 'i'.
        dp = [0] * states

        # Iterate on all possible states one-by-one.
        for state in range(states-1, -1, -1):

            numbersTaken = count_ones(state)
            pairsFormed = numbersTaken // 2
            # States representing even numbers are taken are only valid.
            if numbersTaken % 2:
                continue

            # We have picked 'pairsFormed' pairs, we try all combinations of one more pair now.
            # We iterate on two numbers using two nested for loops.
            for firstIndex in range(len(nums)):
                for secondIndex in range(firstIndex + 1, len(nums)):
                    # We only choose those numbers which were not already picked.
                    if (state >> firstIndex & 1) == 1 or (state >> secondIndex & 1) == 1:
                        continue
                    currentScore = (pairsFormed + 1) * gcd(nums[firstIndex], nums[secondIndex])
                    stateAfterPickingCurrPair = state | (1 << firstIndex) | (1 << secondIndex)
                    remainingScore = dp[stateAfterPickingCurrPair]
                    dp[state] = max(dp[state], currentScore + remainingScore)

        # Returning score we get from 'n' remaining numbers of array.
        return dp[0]

####################################
# 513. Find Bottom Left Tree Value
# 15MAY23
#####################################
#bfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        '''
        bfs and if there is a row, replace the first element with with the current left
        '''
        curr_left = -1
        q = deque([root])
        
        while q:
            N = len(q)
            for i in range(N):
                curr = q.popleft()
                if i == 0:
                    curr_left = curr.val
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
        
        return curr_left

#dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        '''
        just use bfs keeping track of the depth, in dfs, when we go to a new depth, it must have been the left most
        on this depth 
        update answe as well as max depth
        '''
        self.ans = root.val
        self.max_depth = 0
        
        def dfs(node,curr_depth):
            if not node:
                return
            if curr_depth > self.max_depth:
                self.ans = node.val
                self.max_depth = curr_depth
            
            dfs(node.left,curr_depth+1)
            dfs(node.right,curr_depth+1)
        
        dfs(root,0)

###########################################
# 515. Find Largest Value in Each Tree Row
# 15MAY23
##########################################
#bfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        '''
        bfs and just take max on each row
        '''
        if not root:
            return []
        ans =  []
        
        q = deque([root])
        
        while q:
            curr_max = float('-inf')
            N = len(q)
            for _ in range(N):
                curr = q.popleft()
                curr_max = max(curr_max,curr.val)
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
            
            
            ans.append(curr_max)
        
        return ans
            
#dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def largestValues(self, root: Optional[TreeNode]) -> List[int]:
        '''
        dfs, again when accessing a new depth, this msut have been the first time we were there, so make a new entry
        add entry
        '''
        if not root:
            return []
        ans = [root.val]
        
        def dfs(node,curr_depth):
            if not node:
                return
            if curr_depth == len(ans):
                ans.append(node.val)
            ans[curr_depth] = max(node.val,ans[curr_depth])
            dfs(node.left,curr_depth + 1)
            dfs(node.right,curr_depth + 1)
        
        dfs(root, 0)
        return ans

#######################################
# 265. Paint House II (REVISTED)
# 16MAY23
#######################################
#top down O(N*k*k)
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        let dp(i,color) be the answer to painting the all houses up to i, ending with color c
        if i go to the next house i+ 1, i need to make sure its the opposite color and take the minimum
        '''
        n = len(costs)
        k = len(costs[0])
        
        memo = {}
        
        def dp(i,c):
            if i >= n:
                return 0
            if (i,c) in memo:
                return memo[(i,c)]
            
            ans = float('inf')
            for next_color in range(k):
                if next_color == c:
                    continue
                ans = min(ans,dp(i+1,next_color))
            
            ans += costs[i][c]
            memo[(i,c)] = ans
            return ans
        
        ans = float('inf')
        for c in range(k):
            ans = min(ans,dp(0,c))
        
        return ans
    
#bottom up O(N*k*k)
#using (N*k) space
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        bottom up, using n*k space
        '''
        n = len(costs)
        k = len(costs[0])
        
        dp = [[0]*k for _ in range(n+1)]
        
        for i in range(n-1,-1,-1):
            for color in range(k):
                ans = float('inf')
                for next_color in range(k):
                    if next_color == color:
                        continue
                    ans = min(ans,dp[i+1][next_color])

                ans += costs[i][color]
                dp[i][color] = ans
                
                
        ans = float('inf')
        for c in range(k):
            ans = min(ans,dp[0][c])
        
        return ans

#bottom up using k space
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        bottom up, using k space
        just allocate 2 rows and swap
        '''
        n = len(costs)
        k = len(costs[0])
        
        dp = [[0]*k for _ in range(2)]
        
        for i in range(n-1,-1,-1):
            for color in range(k):
                ans = float('inf')
                for next_color in range(k):
                    if next_color == color:
                        continue
                    ans = min(ans,dp[1][next_color])

                ans += costs[i][color]
                dp[0][color] = ans
            
            #swap
            dp[1] = dp[0][:]
                
                
        ans = float('inf')
        for c in range(k):
            ans = min(ans,dp[0][c])
        
        return ans

#constant space, we just overwrite the input space
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:
        '''
        bottom up, using k space
        just allocate 2 rows and swap
        '''
        n = len(costs)
        k = len(costs[0])
        
        for i in range(n-2,-1,-1):
            for color in range(k):
                ans = float('inf')
                for next_color in range(k):
                    if next_color == color:
                        continue
                    ans = min(ans,costs[i+1][next_color])

                ans += costs[i][color]
                costs[i][color] = ans

        ans = float('inf')
        for c in range(k):
            ans = min(ans,costs[0][c])
        
        return ans
    
#the actual hard part of the problem is reducine it to O(Nk)
class Solution:
    def minCostII(self, costs: List[List[int]]) -> int:

        n = len(costs)
        if n == 0: return 0
        k = len(costs[0])

        for house in range(1, n):
            # Find the colors with the minimum and second to minimum
            # in the previous row.
            min_color = second_min_color = None
            for color in range(k):
                cost = costs[house - 1][color]
                if min_color is None or cost < costs[house - 1][min_color]:
                    second_min_color = min_color
                    min_color = color
                elif second_min_color is None or cost < costs[house - 1][second_min_color]:
                    second_min_color = color
            # And now update the costs for the current row.
            for color in range(k):
                if color == min_color:
                    costs[house][color] += costs[house - 1][second_min_color]
                else:
                    costs[house][color] += costs[house - 1][min_color]

        #The answer will now be the minimum of the last row.
        return min(costs[-1])


#############################################
# 2130. Maximum Twin Sum of a Linked List
# 17MAY23
############################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        '''
        if i had the array, then just find the max twin sum for all (i,n-1-i) pairs
        '''
        nums = []
        curr = head
        
        while curr:
            nums.append(curr.val)
            curr = curr.next
        
        ans = 0
        n = len(nums)
        for i in range(n//2):
            ans = max(ans,nums[i]+nums[n-1-i])
        
        return ans
    
#without converting to array
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def pairSum(self, head: Optional[ListNode]) -> int:
        '''
        find middle,
        reverse second half
        two pointers starting head of first, and head of second
        max out the pointer sums
        '''
        def reverse(node):
            prev = None
            curr = node
            while curr:
                next_node = curr.next
                curr.next = prev
                prev = curr
                curr = next_node
            
            return prev
        
        def rec_reverse(node):
            if not node or not node.next:
                return node
            #revsre the rest
            temp = rec_reverse(node.next)
            node.next.next = node
            temp.next = None
            return temp
                
        def get_middle(node):
            slow = node
            fast = node
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            
            return slow
        
        front = head
        middle = get_middle(head)
        #revese middle
        rev_middle = reverse(middle)
        ans = 0
        
        while rev_middle:
            ans = max(ans, rev_middle.val + front.val)
            rev_middle = rev_middle.next
            front = front.next
        
        return ans
    
#######################################################
# 1557. Minimum Number of Vertices to Reach All Nodes
# 18MAY23
####################################################
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        '''
        we have a directed graph, we want to include the minimum vertices needed to touch all nodes
        say we have the graph
        a node that does not have any incoming edges, can only be reached by itself
        any other node with incoming edges can be reched from some other node
        count nodes with 0 incoing edges

        if a node has an inward going edge to it, we don't need to include this node, because we could have reached it from another noder
        if a node doesn't have an inward edge going into it, it means we can't reach it from another node
        so we must include it in the min set!
        '''

        indirection = [0]*(n)
        for u,v in edges:
            indirection[v] += 1
        
        ans = []
        for i in range(n):
            if indirection[i] == 0:
                ans.append(i)
        
        return ans
    
#other ways
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        #return set(range(n)) - set(v for _,v in edges)
        return {e[0] for e in edges} - {e[1] for e in edges}

#######################################
# 914. X of a Kind in a Deck of Cards
# 18MAY23
######################################
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        '''
        if we have k groups, each group should have the same count
        we can try every possible X in the [2,N+1]
        if X can divide N evenly, then check that all counts can be evdenly divided by X
        '''
        counts = Counter(deck)
        N = len(deck)
        for X in range(2,N+1):
            #if this X can divide
            if N % X == 0:
                if all([v % X == 0 for k,v in counts.items()]):
                    return True
        
        return False
    
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        '''
        say there are count_i for car i
        all of these count_i must be dviisble for some X to work
        i.e all count_i % X == 0 for all i
        
        so X must divide all count_i, this mean X must be the GCD of count_i
        '''
        def gcd(x,y):
            if y == 0:
                return x
            return gcd(y, x % y)
        
        
        counts = Counter(deck)
        #get all counts
        counts = list(counts.values())
        if len(counts) == 1:
            return counts[0] >= 2
        
        first_gcd = gcd(counts[0],counts[1])
        for i in range(2,len(counts)):
            first_gcd = gcd(first_gcd,counts[i])
            
        return first_gcd >= 2
    
#using reduce function
class Solution:
    def hasGroupsSizeX(self, deck: List[int]) -> bool:
        '''
        say there are count_i for car i
        all of these count_i must be dviisble for some X to work
        i.e all count_i % X == 0 for all i
        
        so X must divide all count_i, this mean X must be the GCD of count_i
        '''
        def gcd(x,y):
            if y == 0:
                return x
            return gcd(y, x % y)
        
        
        counts = Counter(deck)
        #get all counts
        counts = list(counts.values())
        return reduce(gcd,counts) >= 2
    
#######################################################
# 785. Is Graph Bipartite?
# 19MAY23
####################################################
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        '''
        dfs typical two color
        color a node blue if it is part of the firt set, else red
        we should be able to greedily color the graph iff it is bipartitie
        
        '''
        colors = {}
        n = len(graph)
        
        def dfs(node,curr_color):
            if node in colors:
                if colors[node] == curr_color:
                    return False
                return True
            
            colors[node] = curr_color
            for neigh in graph[node]:
                colors[node] = curr_color ^ 1
                if dfs(neigh,curr_color ^ 1) == False:
                    return False
                
            return True
        
        for i in range(n):
            if i not in colors:
                if dfs(i,0) == False:
                    return False
        
        return True
    
#stack, dfs
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        '''
        iterative
        '''
        colors = {}
        N = len(graph)
        
        for i in range(N):
            if i not in colors:
                stack = [i]
                colors[i] = 0
                
                while stack:
                    node = stack.pop()
                    for neigh in graph[node]:
                        if neigh not in colors:
                            colors[neigh] = colors[node] ^ 1
                            stack.append(neigh)
                        elif colors[neigh] == colors[node]:
                            return False
                        
        
        return True

#########################################
# 399. Evaluate Division (REVISTED)
# 20MAY23
##########################################
#bfs
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        '''
        this is just a graph problem
        so really we have
            a->b = values[i]
            b->a = 1 / values[i]
        
        values cannot be 0, so no division by zero error
        '''
        graph = defaultdict(list)
        for eq,val in zip(equations,values):
            u = eq[0]
            v = eq[1]
            forwards = val
            backwards = 1/val
            graph[u].append((v,forwards))
            graph[v].append((u,backwards))
        
        
        def bfs(start,end):
            
            q = deque([(start,1.0)])
            visited = set()
            
            while q:
                curr,path = q.popleft()
                if curr not in graph:
                    return -1.0
                if curr == end:
                    return path
                #mark
                visited.add(curr)
                for neigh,weight in graph[curr]:
                    if neigh not in visited:
                        q.append((neigh,path*weight))
            
            return -1.0
        
        array = []
        for start,end in queries:
            ans = bfs(start,end)
            array.append(ans)
        
        return array

#dfs, no backtracking
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        '''
        this is just a graph problem
        so really we have
            a->b = values[i]
            b->a = 1 / values[i]
        
        values cannot be 0, so no division by zero error
        '''
        graph = defaultdict(list)
        for eq,val in zip(equations,values):
            u = eq[0]
            v = eq[1]
            forwards = val
            backwards = 1/val
            graph[u].append((v,forwards))
            graph[v].append((u,backwards))
        
        
        def rec(start,end,path,seen):
            if start == end:
                return path
            seen.add(start)
            for neigh,weight in graph[start]:
                if neigh not in seen:
                    #first got the child answer
                    child = rec(neigh,end,path*weight,seen)
                    if child != -1.0:
                        return child
            
            return -1.0
        
        array = []
        for start,end in queries:
            if start not in graph or end not in graph:
                array.append(-1.0)
            else:
                seen = set()
                array.append(rec(start,end,1,seen))

#dfs with backtracking
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        '''
        this is just a graph problem
        so really we have
            a->b = values[i]
            b->a = 1 / values[i]
        
        values cannot be 0, so no division by zero error
        '''
        graph = defaultdict(list)
        for eq,val in zip(equations,values):
            u = eq[0]
            v = eq[1]
            forwards = val
            backwards = 1/val
            graph[u].append((v,forwards))
            graph[v].append((u,backwards))
        
        
        def rec(start,end,path,seen):
            seen.add(start)
            ret = -1.0
            if start == end:
                ret = path
            else:
                for neigh,weight in graph[start]:
                    if neigh in seen:
                        continue
                    ret = rec(neigh,end,path*weight,seen)
                    if ret != -1.0:
                        break
            #backtrack
            seen.remove(start)
            return ret
        
        array = []
        for start,end in queries:
            if start not in graph or end not in graph:
                array.append(-1.0)
            else:
                seen = set()
                array.append(rec(start,end,1,seen))
        
        return array
    
#union find
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        '''
        we can use union find for this, but we need to edit the canonical union find methids
        we define key as -> (group_id,weight)
        given two nodes, a and b, with entries (a_group_id,a_weight) and (b_group_id,b_weight)
        we can just perform a_weight/b_weight to get the answer to the query
        
        if a_group_id == b_group_id, there is a path between them 
        a_weight / b_weight, this is the answer to the query
        
        intially each nodes point to itslef with weight of 1, DUH!
        for each equation we can do union operation, but also update the weights
        the find operations, will update the weight if there is a path between
        
        union attaches group of dividend to the divsor, if they are not in the same group   
        also, it udpates the weight of the dividens variable accordinlgy, so that the ration and divisor isr espected
        
        time complexity if log*
        O(M⋅log ∗N),
        let N be the number of input equations
        let M be the number of queries
        First we iterate through each input equations and invoke unions O(N*log_start(N))
        i.e nubmber of times we iterate log for it to be equal to one
        
        well both union and find update the wieghts
        '''
        gid_weight = {}

        def find(node_id):
            #recall we initailize these parents pointeres to itself and weights to 1 in the constructor
            if node_id not in gid_weight:
                gid_weight[node_id] = (node_id, 1)
            #this would be the actual find call
            group_id, node_weight = gid_weight[node_id]
            if group_id != node_id:
                # found inconsistency, trigger chain update
                new_group_id, group_weight = find(group_id)
                gid_weight[node_id] = (new_group_id, node_weight * group_weight)
            return gid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)
            if dividend_gid != divisor_gid:
                # merge the two groups together,
                # by attaching the dividend group to the one of divisor
                gid_weight[dividend_gid] = (divisor_gid, divisor_weight * value / dividend_weight)

        # Step 1). build the union groups
        for (dividend, divisor), value in zip(equations, values):
            union(dividend, divisor, value)

        results = []
        # Step 2). run the evaluation, with "lazy" updates in find() function
        for (dividend, divisor) in queries:
            if dividend not in gid_weight or divisor not in gid_weight:
                # case 1). at least one variable did not appear before
                results.append(-1.0)
            else:
                dividend_gid, dividend_weight = find(dividend)
                divisor_gid, divisor_weight = find(divisor)
                if dividend_gid != divisor_gid:
                    # case 2). the variables do not belong to the same chain/group
                    results.append(-1.0)
                else:
                    # case 3). there is a chain/path between the variables
                    results.append(dividend_weight / divisor_weight)
        return results
    
#check this out https://leetcode.com/problems/evaluate-division/discuss/270993/Python-BFS-and-UF(detailed-explanation)
#OO implementation
#intution: dividend points to parent divisor
'''
root[x] is of the form (root[x],ratio), and if x == root(x), then ratio is 1
find(x):
    we have root[x] = (p,x/p) #if x==p, x/p == 1
    p is the parent node x, and not necessariy the root (i.e the id of the group in typical Union Find)
    but with path compression we update
    we want find(p) to return
        root[p] = (root(p),p/root(p))
        root[x] should be updated to (root(x), x/root(x)) = (root(p), x/p * p/root(p)) = (root[p][0], root[x][1] * root[p][1])

union(x,y)L
     in equations processing, we make root(root(x)) = root(y) as mentiond previously. 
     And for root[root(x)]'s ratio, as root(y) is root(x)'s new root, 
     we update it to root(x)/root(y) = (x/y) * (y/root(y)) / (x/root(x)) = x/y * root[y][1] / root[x][1]. 
     x/y is the provided equation outcome value.

For union(x, y) in queries, we can just simply return x/y = (x/root(x)) / (y/root(y)) = root[x][1]/root[y][1].
'''
class UnionFind:

    def __init__(self, size):
        self.root = [0]*size
        for i in range(size):
            self.root[i] = (i, 1.0)

    def find(self, x):
        p, xr = self.root[x]
        if x!=p:
            r, pr = self.find(p)
            self.root[x] = (r, pr*xr)
        return self.root[x]

    def union(self, x, y, ratio):
        px, xr= self.find(x)
        py, yr = self.find(y)
        if not ratio:
            return xr / yr if px==py else -1.0
        if px!=py:
            self.root[px] = (py, yr/xr*ratio)


class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        variables = {}
        count = 0
        for a, b in equations:
            if a not in variables:
                variables[a]=count
                count+=1
            if b not in variables:
                variables[b]=count
                count+=1
        n = len(variables)
        uf = UnionFind(n)

        for (a, b), v in zip(equations, values):
            uf.union(variables[a], variables[b], v)

        return [uf.union(variables[a], variables[b], 0) \
                if (a in variables) and (b in variables) else -1 \
                for a, b in queries ]

######################
# 934. Shortest Bridge
# 21MAY23
######################
#TLE???
#O(N^4)
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        '''
        given binary matrix of 0s and 1s there are exactly two islands
        0 is water, 1 is land, return the smllest number of 0's i can flip to 1's to connect the islands
        i.e find the shortest bridge
        
        first find the islands using bfs
        then for each cell in island 1, use bfs to find the a cell in island 2
        the answer is the minimum
        '''
        rows = len(grid)
        cols = len(grid[0])
        island_1 = set()
        island_2 = set()
        seen = set()
        first_found = False
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def find_island(x,y,seen,curr_island):
            q = deque([(x,y)])
            
            while q:
                curr_x,curr_y = q.popleft()
                seen.add((curr_x,curr_y))
                curr_island.add((curr_x,curr_y))
                for dx,dy in dirrs:
                    neigh_x = curr_x + dx
                    neigh_y = curr_y + dy
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        if (neigh_x,neigh_y) not in seen and grid[neigh_x][neigh_y] == 1:
                            q.append((neigh_x,neigh_y))
                            
        first_i = None
        first_j = None
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    first_i = i
                    first_j = j
                    break
        #do first island
        find_island(first_i,first_j,seen,island_1)
        #any other 1 belongs to second island
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1 and (i,j) not in island_1:
                    island_2.add((i,j))
        
        ans = float('inf')
        for cell_1 in island_1:
            for cell_2 in island_2:
                x1,y1 = cell_1
                x2,y2 = cell_2
                dist = abs(x1 - x2) + abs(y1 -y2)
                ans = min(ans,dist)
        
        return ans - 1
    
#comparing manaht is O(N^4), after getting islands do bfs one more time to find the smalelst distance
#uhhhh??
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        '''
        given binary matrix of 0s and 1s there are exactly two islands
        0 is water, 1 is land, return the smllest number of 0's i can flip to 1's to connect the islands
        i.e find the shortest bridge
        
        first find the islands using bfs
        then for each cell in island 1, use bfs to find the a cell in island 2
        the answer is the minimum
        '''
        rows = len(grid)
        cols = len(grid[0])
        island_1 = set()
        island_2 = set()
        seen = set()
        first_found = False
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def find_island(x,y,seen,curr_island):
            q = deque([(x,y)])
            
            while q:
                curr_x,curr_y = q.popleft()
                seen.add((curr_x,curr_y))
                curr_island.add((curr_x,curr_y))
                for dx,dy in dirrs:
                    neigh_x = curr_x + dx
                    neigh_y = curr_y + dy
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        if (neigh_x,neigh_y) not in seen and grid[neigh_x][neigh_y] == 1:
                            q.append((neigh_x,neigh_y))
                            
        first_i = None
        first_j = None
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    first_i = i
                    first_j = j
                    break
        #do first island
        find_island(first_i,first_j,seen,island_1)
        #any other 1 belongs to second island
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1 and (i,j) not in island_1:
                    island_2.add((i,j))
        
        #q up from island 1
        q = deque([])
        for i,j in island_1:
            q.append((i,j,0))
        
        while q:
            curr_x,curr_y,dist = q.popleft()
            if (curr_x,curr_y) in island_2:
                return dist - 1
            for dx,dy in dirrs:
                neigh_x = curr_x + dx
                neigh_y = curr_y + dy
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    if (neigh_x,neigh_y) not in seen:
                        seen.add((neigh_x,neigh_y))
                        q.append((neigh_x,neigh_y,dist+1))

    

#######################################################
# 2542. Maximum Subsequence Score
# 24MAY23
####################################################
#bleaghhhh
class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        '''
        len(nums1) == len(nums2)
        we want to choose a subsequence of k indicies from nums1
        we need k indicies
        we define score as:
            let indices be an array of k indices
            score = sum(nums1[i] for i in indices)*min(nums2[i] for i indinces)
        
        there are two ways to maximize score
            1. maximize the sum
            2. maximize the minimum
            
        all nums are positive, so it only makes sense to keep including numbers in the subsequcne of nums1 
        we need to somehow sort
        hint2, try sorting the two arrays based on the second array
        hint3, loop through nums2 and compute the max product given the min is nums2[i]
        '''
        
        #sort nums2
        min_pairs = [(num,i) for i,num in enumerate(nums2)]
        #sort increasinly
        min_pairs.sort(key = lambda x: -x[0])
        #sort nums1
        nums1_sorted = []
        for num,i in min_pairs:
            nums1_sorted.append(nums1[i])
        
        ans = 0
        curr_sum = 0
        
        for i,num in enumerate(nums1_sorted):
            curr_sum += num
            cand_ans = curr_sum*(min_pairs[i][0])
            ans = max(ans,cand_ans)
        
        return ans
    
#heap
class Solution:
    def maxScore(self, nums1: List[int], nums2: List[int], k: int) -> int:
        '''
        note, that finding subsets of size k, is going to be in factorial time
        if we pick nums2[i] as the the minimum, then it menas we are free to select the other k-1 elements froms nums1
        in order to maximize the remaning score, we need to pick the next k-1 largest
        we can change the relative ordering of nums1 wrt to nums2, but we are free to choose an subset of indice in nums1
        it makes sense to store pairs as (nums1[i],nums2[i]) then sort decreasingly
            we need to pair the numbers, why?
            [1,2,3]
            [7,8,9] sort decreasingly
            
            [9,8,7], but the top becomes
            [3,2,1] (i.e we just want the indices to line up)
        
        if we fix the current minimum with nums2[i], after sorting
        we can mizmie the total score by select the maximum k elements from nums1
        this can be done by maintaing a min heap of size k that always contains the k largest
        whenver we pick a new nums2[i] as the min we need to remove one element from theap
            this represents removing a nums1 number and add nums[i] to it
            no the heap contains the largest k element including nums1[i] again,
            the current score == the sum of this heap times nums2[i]
        
        '''
        #sort decreasingly on nums2 and pair with nums1
        pairs = [(a,b) for a,b in zip(nums1,nums2)]
        #sort decreasingly on nums2
        pairs.sort(key = lambda x: -x[1])
        
        curr_top_k = [a for a,b in pairs[:k]]
        heapq.heapify(curr_top_k)
        curr_top_k_sum = sum(curr_top_k)
        
        #current ans
        curr_min = pairs[k-1][1]
        ans = curr_top_k_sum*curr_min
        
        for i in range(k,len(pairs)):
            #updates
            #remove smallest integer from previouis top k, and put it new one
            curr_top_k_sum -= heapq.heappop(curr_top_k)
            curr_top_k_sum += pairs[i][0]
            #add back in
            heapq.heappush(curr_top_k,pairs[i][0])
            
            #new potential answer
            curr_min = pairs[i][1]
            ans = max(ans,curr_top_k_sum*curr_min)
        
        return ans

##################
# 837. New 21 Game
# 24MAY23
##################
#aye yai yai
#nice try though
class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        '''
        game:
            alice starts with 0 points, and draws while she hass less than k points
            during each draw, she gains an integer number of points randomly [1,maxPts] where maxPts is an integer
            each draw is independent
            alice stops drawing when she gets k or more points
        
        return probabilty that Alice has n or fewer points
        for point in [1,maxPts]:
            alice can get point with probability 1/(maxPts)
        
        we want probibilty that alice has <= n points
        we want sum dp(i) to dp(n)
        '''
        memo = {}
        
        def dp(curr_points):
            if curr_points == k:
                return 1
            if curr_points > k:
                return 0
            if curr_points in memo:
                return memo[(curr_points)]
            ans = 0
            for next_point in range(1,maxPts+1):
                curr_prob = (1/maxPts)*dp(curr_points+next_point)
                ans += curr_prob
            
            memo[(curr_points)] = ans
            return ans
        
    
        
        return 1 - dp(-1)

#YESSS, but gets TLE
class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        '''
        game:
            alice starts with 0 points, and draws while she hass less than k points
            during each draw, she gains an integer number of points randomly [1,maxPts] where maxPts is an integer
            each draw is independent
            alice stops drawing when she gets k or more points
        
        return probabilty that Alice has n or fewer points
        for point in [1,maxPts]:
            alice can get point with probability 1/(maxPts)
        
        we want probibilty that alice has <= n points
        we want sum dp(i) to dp(n)
        '''
        memo = {}
        
        def dp(curr_points):
            if curr_points > n:
                return 0.0
            if curr_points >= k:
                return 1.0
            if curr_points in memo:
                return memo[(curr_points)]
            ans = 0
            for next_point in range(1,maxPts+1):
                curr_prob = (1/maxPts)*dp(curr_points+next_point)
                ans += curr_prob
            
            memo[(curr_points)] = ans
            return ans
        
        return dp(0)
