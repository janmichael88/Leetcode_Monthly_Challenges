#################
# Gray Code
################
#back tracking
class Solution:
    def grayCode(self, n: int) -> List[int]:
        '''
        we are given n, we want array of size 2**n
        first is always zero, each integer is disinct
        adjacent integers differ by exactly one bit
        first and last differ by one bit
        i can check if bit is same/differnet using ^ (1 << n places)
        we keep adding integers that are 1 bit off from one another
        '''
        present = set([0])
        result = [0]
        
        def recurse():
            if len(result) == 1 << n:
                return True
            current = result[-1]
            #toggle bits
            for i in range(n):
                possible = current ^ (1 << i)
                if possible not in present:
                    result.append(possible)
                    present.add(possible)
                    #backtrack
                    if recurse():
                        return True
                    present.remove(possible)
                    result.pop()
            return False
        recurse()
        return result

#recursion, pattern finding
class Solution:
    def grayCode(self, n: int) -> List[int]:
        '''
        we can obtain the n gray sequence from the n- 1 grqy sequence
        the sequence for n = 2:
        [00, 01, 11, 10]
        and for n = 3
        [000, 001, 011, 010, 110, 111, 101, 100]
        add 0 the n-1th position
        reverse the sequence and add 1 : (1 << n-1)
        concatetnate
        '''
        result = []
        def recurse(n):
            if n == 0:
                result.append(0)
                return
            recurse(n-1)
            mask = 1 << (n-1)
            for i in range(len(result)-1,-1,-1):
                result.append(result[i] | mask)
        recurse(n)
        return result

#non global return
class Solution:
    def grayCode(self, n: int) -> List[int]:
        '''
        we can obtain the n gray sequence from the n- 1 grqy sequence
        the sequence for n = 2:
        [00, 01, 11, 10]
        and for n = 3
        [000, 001, 011, 010, 110, 111, 101, 100]
        add 0 the n-1th position
        reverse the sequence and add 1 : (1 << n-1)
        concatetnate
        '''
        def recurse(n):
            if n == 0:
                return [0]
            result = recurse(n-1)
            mask = 1 << (n-1)
            for i in range(len(result)-1,-1,-1):
                result += [(result[i] | mask)]
            return result
        return recurse(n)
 
#iterative
class Solution:
    def grayCode(self, n: int) -> List[int]:
        '''
        we can obtain the n gray sequence from the n- 1 grqy sequence
        the sequence for n = 2:
        [00, 01, 11, 10]
        and for n = 3
        [000, 001, 011, 010, 110, 111, 101, 100]
        add 0 the n-1th position
        reverse the sequence and add 1 : (1 << n-1)
        concatetnate
        '''
        res = [0]
        
        for i in range(1,n+1):
            prev_size = len(res)
            mask = 1 << (i-1)
            for j in range(prev_size-1,-1,-1):
                res.append(res[j] | mask)
        return res

###################
# Find K Closest Elements
####################
#sort by distance
#then sort again
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        '''
        we can just resort the array based on distance from x
        '''
        arr = sorted(arr, key = lambda num: abs(x- num))
        res = arr[:k]
        res.sort()
        return res


#$binary seacrh and then sliding window
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        '''
        binary search to find x, then get k closest
        the problem is if x is outside the bound
        '''
        MIN = arr[0]
        MAX = arr[-1]
        N = len(arr)
        if N == k:
            return arr
        #edge cases x < min
        if x < MIN:
            return arr[:k]
        #x > max
        if x > MAX:
            return arr[-k:]
        #binary seacrh to find the elemnt nearest to x
        left,right = 0, N-1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] > x:
                right = mid - 1
            elif arr[mid] < x:
                left = mid + 1
            else:
                break
        #sliding window to find the right subarray
        left = max(0,mid-1)
        right = left + 1
        
        #sliding window < k
        while right - left - 1 < k:
            #don't go out of bounds
            if left == -1:
                right += 1
                continue
            #expand twoards side with number closer to x
            if right == len(arr) or abs(arr[left] - x) <= abs(arr[right]-x):
                left -= 1
            else:
                right += 1
        return arr[left+1:right]

#usin bisect
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        '''
        binary search to find x, then get k closest
        the problem is if x is outside the bound
        '''
        MIN = arr[0]
        MAX = arr[-1]
        N = len(arr)
        if N == k:
            return arr
        #edge cases x < min
        if x < MIN:
            return arr[:k]
        #x > max
        if x > MAX:
            return arr[-k:]
        #binary seacrh to find the elemnt nearest to x
        left = bisect_left(arr,x) - 1
        right = left + 1
        
        #sliding window < k
        while right - left - 1 < k:
            #don't go out of bounds
            if left == -1:
                right += 1
                continue
            #expand twoards side with number closer to x
            if right == len(arr) or abs(arr[left] - x) <= abs(arr[right]-x):
                left -= 1
            else:
                right += 1
        return arr[left+1:right]

class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        '''
        we know the arr is ordered and the sub array need to be of size k
        if we fix the bounds, then we can binry search the sliding windwo
        upper bound for left index would be len(arr) - k
        upper bound for right would be len(arr)
        we need to find the left bounds that get use close to our answer as possible
        key:
            if element at arr[mid] is closer to x than arr[mid+k], then arr[mid+k] and
            every point to the right can never be the answer
             if arr[mid + k] is closer to x, then move the left pointer.

        '''
        N = len(arr)
        left, right = 0,N-k
        
        #binary search to find the left bound
        while left < right:
            mid = left + (right - left) //2
            if x - arr[mid] > arr[mid+k]-x:
                left = mid + 1
            else:
                right = mid
        return arr[left:left+k]

################
# Find Leaves of Binary Tree
################
#close one, keep for reference
class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        '''
        while loop and dfs
        dfs to leaves add them to global result
        once i've mode it to a leaf, make them none
        '''
        def dfs(node):
            if not node.left and not node.right:
                return [node.val]
            result = []
            result += dfs(node.left)
            result += dfs(node.right)
            #but we also need to delett
            return result
        
        def dfs_delete(node):
            if not node.left and not node.right:
                return node
            else:
                return None
            left = dfs(node.left)
            right = dfs(node.right)
            if left and right:
                node.left = None
                node.right = None
            if right and not left:
                node.left = None
            if left and not right:
                node.right = None
            return None
        res = []
        while root:
            res += dfs(root)
            dfs_delete(root)
        
        return res
        
#wohhhooo
class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        '''
        leaves have a depth of 1
        how about i traverse tree once, and for each node gets is level and value
        i can then dump these valueds to a hash map and then build the answer
        '''
        mapp = defaultdict(list)
        self.max_level = 0
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            height = max(left,right) + 1
            mapp[height].append(node.val)
            self.max_level = max(self.max_level,height)
            return height
        dfs(root)
        ans = []
        for level in range(1,self.max_level+1):
            ans.append(mapp[level])
        return ans

#one pass
class Solution:
    def findLeaves(self, root: TreeNode) -> List[List[int]]:
        res = []
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            height = max(left,right) + 1
            #new height
            if len(res) < height:
                res.append([])
            res[height-1].append(node.val)
            return height
        dfs(root)
        return res

####################
# Max Sum of Rectangle No Larger Than K
####################
from sortedcontainers import SortedList
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        '''
        first start with the 1d problem, at every index i, we stored the prefix sum (summ ending at i)
        in a sorted container, to max find possible sum <= k we want to find a possible sub array ending at i == k
        if not, we check for k-1
        using binary seacrh we check for largest sub array sum <=  running_sum - k
        when getting a prewfix sum in sorted manner, we do it one by one and check in soreted array running_sum - k <= k
        if we've found one, we update our max possibel answer
        now we just extend to the 2d problem:
        we start finding bounds of height of rectanle for (r1,r2) for all combinations < len(matrix[0])
        then we accumlate accross cols for each r1r2 and solve the 1d problem taking max when we can
        running_sum - k = X
        find X closes <= k
        https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/discuss/1313721/JavaPython-Sub-problem%3A-Max-Sum-of-Subarray-No-Larger-Than-K-Clean-and-Concise
        '''
        def ceiling(sortedList,key):
            idx = sortedList.bisect_left(key)
            #in bounds
            if idx < len(sortedList):
                return sortedList[idx]
            return None
        
        def maxSumSubArray(arr,n,k):
            right = 0 #curr running sum
            seen = SortedList([0])
            ans = float('-inf')
            for i in range(n):
                right += arr[i]
                left = ceiling(seen, right - k) #largest subarray sum <= k so dfar
                if left != None:
                    ans = max(ans, right - left)
                seen.add(right)
            return ans
        m,n = len(matrix), len(matrix[0])
        ans = float('-inf')
        
        #solve to the 2d
        for r1 in range(m):
            arr = [0]*n
            for r2 in range(r1,m):
                #accumulate
                for c in range(n):
                    arr[c] += matrix[r2][c]
                ans = max(ans, maxSumSubArray(arr,n,k))
        return ans

######################
# Count Vowels Permutation
######################
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        '''
        we can do this recursively
        first what are the relations
        a --> e
        e --> (a,i)
        i --> (a,e,o,u)
        o --> (i,u)
        u --> (a)
        
        if we had a string of length n, we should just add all the solutions of n-1 
        using the rules
        rec(n,v) = sum rec(n-1,v) for v in relations
        
        rec(n,v) answers the question number of strings length n ending at v
        n is the the number of strings ending in vowel v
        to get n, go back to n-1
        '''
        mod = 10**9 + 7
        #recode reltaions 0-4
        relations = [[1],[0,2],[0,1,3,4],[2,4],[0]]
        memo = {}
        def rec(n,v):
            if n == 1:
                return 1
            if (n,v) in memo:
                return memo[(n,v)]
            res = 0
            for neigh in relations[v]:
                res += rec(n-1,neigh)
                res %= mod
            memo[(n,v)] = res
            return memo[(n,v)]
        
        #invoke for each vowlel
        count = 0
        for i in range(5):
            count += rec(n,i)
            count %= mod
        return count

class Solution:
    def countVowelPermutation(self, n: int) -> int:
        '''
        just another recursino way
        go back to only possible letters that leads to current vowel
        then add them up
        rec(n,'a') = rec(n-1,'e') + rec(n-1,'i') + rec(n-1,'u')
        rec(n,'e') = rec(n-1,'a') + rec(n-1,'i')
        rec(n,'i') = rec(n-1,'e') + rec(n-1,'o')
        rec(n,'o') = rec(n-1,'i')
        rec(n,'u') = rec(n-1,'i') + rec(n-1,'o')
        '''
        mod = 10**9 + 7 
        memo = {}
        
        def rec(n,v):
            res = None
            if n == 1:
                return 1
            elif (n,v) in memo:
                return memo[(n,v)]
            elif v == 'a':
                res = rec(n-1,'e') + rec(n-1,'i') + rec(n-1,'u')
            elif v == 'e':
                res = rec(n-1,'a') + rec(n-1,'i')
            elif v == 'i':
                res = rec(n-1,'e') + rec(n-1,'o')
            elif v == 'o':
                res = rec(n-1,'i')
            else:
                res = rec(n-1,'i') + rec(n-1,'o')
            memo[(n,v)] = res
            return res
        
        count = 0
        for v in 'aeiou':
            count += rec(n,v)
            count %= mod
        return count

#bottom up
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        '''
        top down,
        just starts with all 1, then repeat n times
        '''
        a, e, i, o, u, M = 1, 1, 1, 1, 1, 10**9 + 7
        for _ in range(n-1):
            a, e, i, o, u = (e + i + u)%M, (a + i)%M, (e + o)%M, i%M, (i + o)%M
        
        return (a + e + i + o + u)%M 

#power of matrix solution
#good explanation
import numpy as np
class Solution:
    def countVowelPermutation(self, n: int) -> int:
        '''
        what this questions asks really is just the number of paths of length N in a
        directed graph
        get the ajd matrix
        then matrix multiply
        '''
        def power(mat,n,M):
            res = np.eye(len(mat),dtype = int)
            while n > 0:
                if n % 2:
                    res = np.dot(res,mat) % M
                mat = np.dot(mat,mat) % M
                n //= 2
            return res
        
        M = 10**9 + 7
        #adj matrix
        mat = np.matrix([[0,1,0,0,0], [1,0,1,0,0], [1,1,0,1,1], [0,0,1,0,1], [1,0,0,0,0]])  
        return np.sum(power(mat,n-1,M)) % M

######################
# Reshape the Matrix
######################
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        '''
        one way could be to flatten the array
        then pull from the array in order and dump into matrix
        q or stack then revverse stack and pop
        first check if possible 
        flatten the oriignal
        and put back into new matrix
        '''
        M = len(mat)
        N = len(mat[0])
        
        if M*N != r*c:
            return mat
        #flatten
        temp = []
        for i in range(M):
            for j in range(N):
                temp.append(mat[i][j])
        
        #allocate new matrix
        new_mat = [[0]*c for _ in range(r)]
        
        #remap index
        #
        for i in range(len(temp)):
            new_mat[i//c][i%c] = temp[i]
        return new_mat

class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        '''
        getting directly from matrix
        '''
        M = len(mat)
        N = len(mat[0])
        
        if M*N != r*c:
            return mat

        #allocate new matrix
        new_mat = [[0]*c for _ in range(r)]
        
        for i in range(M):
            for j in range(N):
                #get element 
                elem = mat[i][j]
                #get indeix
                idx = N*i + j
                #put into new mat
                new_mat[idx//c][idx%c] = elem
        return new_mat

##################
# Reduce Array Size to The Half
##################
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        '''
        get the length of the array
        get counts
        now try to get half the size by chekcing lenght - count <= legnth //2
        but we want the minimum
        
        get counts
        push counts to max heap 
        keep taking max freq counts until we get below half
        '''
        N = len(arr)
        half_N = N //2
        counts = Counter(arr)
        heap = []
        for k,v in counts.items():
            heappush(heap,(-v,k))
        
        min_set = set()
        while N > half_N:
            count,num = heappop(heap)
            min_set.add(num)
            N += count
        return len(min_set)

class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        '''
        this was a really cool way of getting a count array
        sort,
        get counts
        reverse counts, keep taking freq counts until we are at > halfn
        '''
        arr.sort()
        counts = []
        curr_count = 1
        for i in range(1,len(arr)):
            if arr[i] == arr[i-1]:
                curr_count += 1
                continue
            counts.append(curr_count)
            curr_count = 1
        #final count
        counts.append(curr_count)
        #reverde sort
        counts.sort(reverse = True)
        N = len(arr)
        half_N = N//2
        set_size = 0
        for count in counts:
            set_size += 1
            N -= count
            if N <= half_N:
                break
        return set_size

#we can also try to use bucket sort
class Solution:
    def minSetSize(self, arr: List[int]) -> int:
        '''
        we can also use bucket sort
        generate the counts array
        then genrate the buckets of counts
        start from the last bucket and keep taking
        O(max(n,largest count))
        '''
        counts = Counter(arr)
        max_value = max(counts.values())
        
        #make buckets of counts
        buckets = [0]*(max_value + 1)
        for count in counts.values():
            buckets[count] += 1
        
        half = len(arr) // 2
        curr_freq = max_value
        size = 0
        removed = 0
        while removed < half:
            size += 1
            while buckets[curr_freq] == 0:
                curr_freq -= 1 #keep droping free
            #update num times removed
            removed += curr_freq
            buckets[curr_freq] -= 1
        return size
            
#this is beter way, 
#remember its a count of counts
class Solution:
	def minSetSize(self, arr: List[int]) -> int:
	        
	    # In Python, we can use the built-in Counter class.
	    counts = collections.Counter(arr)
	    max_value = max(counts.values())
	            
	    # Put the counts into buckets.
	    buckets = [0] * (max_value + 1)
	    
	    for count in counts.values():
	        buckets[count] += 1
	        
	    # Determine set_size.
	    set_size = 0
	    arr_numbers_to_remove = len(arr) // 2
	    bucket = max_value
	    while arr_numbers_to_remove > 0:
	        max_needed_from_bucket = math.ceil(arr_numbers_to_remove / bucket)
	        set_size_increase = min(buckets[bucket], max_needed_from_bucket)
	        set_size += set_size_increase
	        arr_numbers_to_remove -= set_size_increase * bucket
	        bucket -= 1
	        
	    return set_size

##########################################
# Kth Smallest Element in a Sorted Matrix
#########################################
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        '''
        cheese way would be to dump into a list and return k
        after sorting
        '''
        N = len(matrix)
        M = len(matrix[0])
        
        temp = []
        for i in range(N):
            for j in range(M):
                temp.append(matrix[i][j])
        temp.sort()
                
        return temp[k-1]

#keeping min heap
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        '''
        think about the problem of finding the kth smallest element in two sorted lists
        two pointers, advance the smaller one, do while k > 0
        but the problem is that we need to keep n pointers
        we only need to examine min(N,K) elements
        example, if we had a 100 rows, and were looking the 5th smallest element
        we can discard eveyrhint below the sixth row
        algo:
            maintain heap, where each element is a tuple (value,row,col)
            note: heap contains min(N,K) elements
            when we pop the min we advance column by 1
            and repeat up to k times
        '''
        rows = len(matrix)
        heap = []
        #add the heap the first element of each row up to min(N,K)
        for i in range(min(rows,k)):
            heappush(heap,(matrix[i][0],i,0))
        
        #keep moving pointers k times
        while k > 0:
            curr,row,col = heappop(heap)
            #advance in row if we can
            if col < rows -1:
                heappush(heap,(matrix[row][col+1],row,col+1))
            #use up a k when we advance
            k -= 1
        return curr

class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        '''
        we can also treat this is findsing the n-kth largest and keep a max heap of size k
        we add the largest element to heap and when > k, we pop
        then we return largest on heap
        '''
        m = len(matrix)
        n = len(matrix[0])
        heap = []
        for i in range(m):
            for j in range(n):
                heappush(heap,-matrix[i][j])
                if len(heap) > k:
                    heappop(heap)
        return -heappop(heap)

#binary search
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        '''
        imagine flattening the matrix, and finding the 'actual mid' of upper left and lower right
        to find the kth smallest we would first chekc if length of mid - 0 < k
        and return left[k-1] 
        if mid - 0 > k, it must be on the right, and we return right[k-len(left)]
        but how large are left and right?
        key: 
            count the number of elements in the left half of the number range of which we know the middle element and the two extremes as well.
            if an element in the cell [i, j] is smaller than our middle element, then it means that all the elements in the column "j" before this element i.r. (i-1) other elements in this column are also going to be less than the middle element. 
        https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/discuss/1322101/C%2B%2BPython-Binary-Search-Picture-Explain-Clean-and-Concise
        algo:
            * get mins and max of matrix
            * get mid (may not be an element)
            * if countlessorqual(mid) >= k, keep curr ans as mid, and try to find smaller value on left, else right
            * Since ans is the smallest value which countLessOrEqual(ans) >= k, so it's the k th smallest element in the matrix.
        counting elements eficinelty
            *  two pointers, one points to the rightmost column c = n-1, and one points to the lowest row r = 0
            * If matrix[r][c] <= x then the number of elements in row r less or equal to x is (c+1) (Because row[r] is sorted in ascending order, so if matrix[r][c] <= x then matrix[r][c-1] is also <= x). Then we go to next row to continue counting.
            * Else if matrix[r][c] > x, we decrease column c until matrix[r][c] <= x (Because column is sorted in ascending order, so if matrix[r][c] > x then matrix[r+1][c] is also > x).
            * check LC 74
            
        '''
        M = len(matrix)
        N = len(matrix[0])
        
        #count elemnts to get size functions
        def countLessEqualMid(x):
            count = 0
            c = N - 1 #right most column
            for r in range(M):
                while c >= 0 and matrix[r][c] > x:
                    #go down in column
                    c -= 1
                count += (c + 1) #num elements less than mid
            return count
        
        left,right = matrix[0][0],matrix[-1][-1]
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if countLessEqualMid(mid) >= k:
                #keep finding smaller and left
                ans = mid
                right = mid - 1
            #other on the righ
            else:
                left = mid + 1
        
        return ans
        
#####################
# Maximum Length of Repeated Subarray
#####################
#brute force
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        brute force would be to examine all subarrays
        '''
        ans = 0
        Bstarts = collections.defaultdict(list)
        for j, y in enumerate(nums2):
            Bstarts[y].append(j)

        for i, x in enumerate(nums1):
            for j in Bstarts[x]:
                k = 0
                while i + k < len(nums1) and j + k < len(nums2) and nums1[i + k] == nums2[j + k]:
                    k += 1
                ans = max(ans, k)
        return ans


#close one....
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        recursion
        if im at a sub array ending at nums1[i] and ending at nums2[j]
        and nums1[i] == nums2[j]:
            then its just 1 + rec(i-1,j-1)
        '''
        memo = {}
        
        def rec(i,j):
            if i < 0 or j < 0:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            res = 0
            if nums1[i] == nums2[j]:
                res = 1 + rec(i-1,j-1)
            else:
                res = max(rec(i-1,j),rec(i,j-1))
            memo[(i,j)] = res
            return res
        
        rec(len(nums1)-1,len(nums2)-1)
        return max(memo.values())

#recursion i still TLE
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can treat this as longest common substring
        we do bottom up dp startin from end
        '''
        memo = {}
        
        def rec(i,j,count):
            if i <= 0 or j <= 0:
                return count
            if (i,j,count) in memo:
                return memo[(i,j,count)]
            take_both = count
            if nums1[i-1] == nums2[j-1]:
                take_both = rec(i-1,j-1,count+1)
            take_first = rec(i-1,j,0) #reset counts to zero, 
            take_second = rec(i,j-1,0)
            res = max(take_both,max(take_first,take_second))
            memo[(i,j,count)] = res
            return res
        
        return rec(len(nums1),len(nums2),0)


#dp is slow
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we can treat this as longest common substring
        we do bottom up dp startin from end
        '''
        ans = 0
        dp = [[0]*(len(nums2)+1) for _ in range(len(nums1)+1)]
        for i in range(len(nums1)-1,-1,-1):
            for j in range(len(nums2)-1,-1,-1):
                if nums1[i] == nums2[j]:
                    dp[i][j] = dp[i+1][j+1] + 1
                    ans = max(ans, dp[i][j])
        
        return ans

#rolling hash
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        we and pre compute the rollin hash of nums1 and nums2
        base should be prime number = 101
        mod should be primer number (a larger one)
        binary search to find max size o sub array
        irstly, we iterate i = 0..m-size+1 to get all hashing values of subarray of size size in the nums1, and put them into HashMap, let name it seen.
Secondly, we iterate i = 0..n-size+1 to get all hashing values of subarray of size size in the nums2, if any hash is already found in seen, then it's a possible there is a subarray which exists in both nums1 and nums2. To make sure, we can do double-checking for it.
Return TRUE, if we found a valid subarray which exists in both nums1 and nums2, else return FALSE.
        '''
        m,n = len(nums1),len(nums2)
        BASE, MOD = 101, 1_000_000_000_001
        hash1 = [0]*(m+1)
        hash2 = [0]*(n+1)
        POW = [1]*(max(m,n)+1)
        #compute POW of base
        for i in range(max(m,n)):
            POW[i+1] = POW[i]*BASE % MOD
        #compute hash nums1
        for i in range(m):
            hash1[i+1] = (hash1[i]*BASE + nums1[i]) % MOD
        #compute hash nums2
        for i in range(n):
            hash2[i+1] = (hash2[i]*BASE + nums2[i]) % MOD
        
        #now we can get hash in O(1) time after pre computing
        def getHash(h, left, right):  # 0-based indexing, right inclusive
            return (h[right + 1] - h[left] * POW[right - left + 1] % MOD + MOD) % MOD
        
        #now try to find subarray with given size
        def foundSubArray(size):
            seen = defaultdict(list)
            for i in range(m - size + 1):
                h = getHash(hash1, i, i + size - 1)
                seen[h].append(i)
            for i in range(n - size + 1):
                h = getHash(hash2, i, i + size - 1)
                if h in seen:
                    for j in seen[h]:  # Double check - This rarely happens when collision occurs -> No change in time complexity
                        if nums1[j:j + size] == nums2[i:i + size]:
                            return True
            return False
        
        #binary search to get larger or smaller answer
        left, right, ans = 1, min(m, n), 0
        while left <= right:
            mid = (left + right) // 2
            if foundSubArray(mid):
                ans = mid  # Update answer
                left = mid + 1  # Try to expand size
            else:
                right = mid - 1  # Try to shrink size
        return ans

##################
# Longest Increasing Subsequence
###################
#TLE recursion
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        '''
        i can use recusion
        rec(i) represents the largest length subsequen ending at i
        if if nums[i+1] > nums[i]:
            1 + rec(i-1)
        '''
        memo = {}
        
        def rec(curr,prev): #these are indices
            if curr == len(nums):
                return 0
            if (curr,prev) in memo:
                return memo[(curr,prev)]
            count1 = 0
            if prev == -1 or nums[curr] > nums[prev]:
                count1 = 1 + rec(curr+1,curr)
            count2 = rec(curr+1,prev)
            res = max(count1,count2)
            memo[(curr,prev)] = res
            return res
        return rec(0,-1) 

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        '''
        i can use recusion
        rec(i) represents the largest length subsequen ending at i
        if if nums[i+1] > nums[i]:
            1 + rec(i-1)
        '''
        memo = {}
        N = len(nums)
        
        def rec(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            SUM = 0
            for j in range(i+1,N):
                if nums[j] > nums[i]:
                    SUM = max(SUM,1+rec(j))
            memo[i] = SUM
            return SUM
        
        #invoke for all elemnts in nums
        ans = 0
        for i in range(N):
            ans = max(ans,1+rec(i))
        return ans

#dp
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        '''
        iterative dp
        '''
        N = len(nums)
        dp = [1]*(N+1)
        ans = 1
        for i in range(1,N):
            #look at all elemnts before
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i],dp[j]+1)
                    ans = max(ans,dp[i])
        return ans

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        '''
        this is important for understandin the NlgN osluion
        we can buiild a subqeuqne by adddingf whne nums > last elemnt addeding
        if not, go through the bulided sub so far to find a smaller element than num
        then replace it
        '''
        sub = [nums[0]]
        for num in nums:
            if num > sub[-1]:
                sub.append(num)
            else:
                i = 0
                while num > sub[i]:
                    i += 1
                sub[i] = num
        return len(sub)

#Nlg(N)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        '''
        we can use binary search to find the left bound of the elemnt in our potential subqsequence
        
        '''
        def binarySearch(sub, val):
            lo, hi = 0, len(sub)-1
            while(lo <= hi):
                mid = lo + (hi - lo)//2
                if sub[mid] < val:
                    lo = mid + 1
                elif val < sub[mid]:
                    hi = mid - 1
                else:
                    return mid
            return lo
        
        sub = []
        for val in nums:
            pos = binarySearch(sub, val)
            if pos == len(sub):
                sub.append(val)
            else:
                sub[pos] = val
        return len(sub)

###################
# Decode Ways II
#####################
class Solution:
    def numDecodings(self, s: str) -> int:
        '''
        rec(i) give number of ways to decode with string ending at  i
        then just get rec(i-1) and add the additional ways
        but there are rules for getting the ways
        rules:
            if we are at *, 9*rec(i-1) but at most
            if 1*, 9*rec(i-2), 11-19
            2*, gives 21*26
            now, what about *** or *****, etc  multiple stars
            a 1* coule be anyhing 11-19, irrespective of all decoding previous to rec(i-1)
            9*ways(s,i-2)
            similarily 2*, could be 21-26
            if is 1-9 it can contribute to previous decodings, equal to the number of decodings up to i-1
            if 0 can only contirbute if 1 or 2
            , if the current digit is lesser than 7, again this pairing could add decodings with count equal to the ones possible by using the digits upto the (i-2)^{th}(i−2) th index only
            f the previous digit happens to be a *, the additional number of decodings depend on the current digit again i.e. If the current digit is greater than 6, this * could lead to pairings only in the range 17-19(
            
        '''
        mod = 10**9 + 7
        memo = {}
        def num_ways(i):
            #we will invoke from the end of the array
            if i < 0:
                return 1
            if i in memo:
                return memo[i]
            #if star,
            if s[i] == '*':
                res = (9*num_ways(i-1)) %  mod
                #rule are befor star, 1*
                if (i > 0) and s[i-1] == '1':
                    res += (9*(num_ways(i-2))) % mod
                elif (i > 0) and s[i-1] == '2':
                    res += (6*(num_ways(i-2))) % mod
                elif (i > 0) and s[i-1] == '*':
                    res += (15*(num_ways(i-2))) % mod
                memo[i] = res
                return res

            #not star rules
            res = num_ways(i-1) if s[i] != 0 else 0
            if (i > 0) and s[i-1] == '1':
                res += (num_ways(i-2)) % mod
            elif (i > 0) and s[i-1] == '2' and s[i] <= '6':
                res += (num_ways(i-2)) % mod
            elif (i > 0) and s[i-1] == '*':
                if s[i] <= '6':
                    res += (2*num_ways(i-2)) % mod
                else:
                    res += (num_ways(i-2)) % mod
            memo[i] = res
            return res
        return num_ways(len(s)-1)

#starting from the beginning
class Solution:
    def numDecodings(self, s: str) -> int:
        memo = dict()
        def backtrack(index):
            if index == len(s):
                return 1
            if index in memo:
                return memo[index]
            if s[index] == '0':
                return 0
            res = 0
            mult = 9 if s[index] == '*' else 1
            res += mult*backtrack(index+1)
            if index < len(s)-1:
                nextt = s[index+1]
                opts = 0
                if s[index] == '*':
                    if nextt == '*':
                        opts = 15 # for ** we have 15 options 11->19 21-26
                    else:
                        if nextt <= '6':
                            opts = 2 # the first * could be 1 or 2
                        else:
                            opts = 1 # # the first * could be just 1
                elif s[index] < '3':
                    if nextt == '*':
                        if s[index] == '1':
                            opts = 9 # 11-19
                        else:
                            opts = 6 # 21-26
                    else:
                        #regular two numbers
                        if int(s[index:index+2]) <= 26:
                            opts = 1
                if opts:
                    res += opts*backtrack(index+2)
            memo[index] = res%(10**9 + 7)
            return memo[index]
        return backtrack(0)

#using dp
class Solution:
    def numDecodings(self, s: str) -> int:
        '''
        we can use the same rules from top down recursion using boyttom up
        we only care about i-2 and i-2 previous decodings
        '''
        mod = 10**9 + 7
        N = len(s)
        dp = [0]*(N+1)
        dp[1] = 9 if s[0] == '*' else 1
        for i in range(1,N):
            if s[i] == '*':
                dp[i+1] = 9*dp[i] % mod
                if s[i-1] == '1':
                    dp[i+1] = (dp[i+1] + 9*dp[i-1]) % mod
                elif s[i-1] == '2':
                    dp[i+1] = (dp[i+1] + 6*dp[i-1]) % mod
                elif s[i-1] == '*':
                    dp[i+1] = (dp[i+1] + 15*dp[i-1]) % mod
            else:
                dp[i+1] = dp[i] if s[i] != '0' else 0
                if s[i-1] == '1':
                    dp[i+1] = (dp[i+1]+dp[i-1]) % mod
                elif s[i-1] == '2' and s[i] <= '6':
                    dp[i+1] = (dp[i+1]+dp[i-1]) % mod
                elif dp[i-1] == '*':
                    if s[i] <= '6':
                        dp[i+1] = (dp[i+1] + 2*dp[i-1]) % mod
                    else:
                        dp[i+1] = (dp[i+1] + dp[i-1]) % mod
        return dp[-1]

#constant space
#https://leetcode.com/problems/decode-ways-ii/discuss/1328138/Python-dp-solution-explained
class Solution(object):
    def numDecodings(self, S):
        mod, dp = 10**9 + 7, [1, 0, 0]
        for c in S:
            dp_new = [0,0,0]
            if c == '*':
                dp_new[0] = 9*dp[0] + 9*dp[1] + 6*dp[2]
                dp_new[1] = dp[0]
                dp_new[2] = dp[0]
            else:
                dp_new[0]  = (c > '0') * dp[0] + dp[1] + (c <= '6') * dp[2]
                dp_new[1]  = (c == '1') * dp[0]
                dp_new[2]  = (c == '2') * dp[0]
            dp = [i % mod for i in dp_new]
        return dp[0]

################################
# Reverse Words in a String II
################################
# nice try
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        '''
        try to allocate space first, then just re reference
        just put the words into a list of lists
        then go backwards, and re assign to s after building new onw
        '''
        if len(s) == 1:
            return
        words = []
        curr_word = []
        for char in s:
            if char.isalpha():
                curr_word.append(char)
            else:
                curr_word.append(char)
                words.append(curr_word)
                curr_word = []
        words.append(curr_word)
        #delete last space of first word
        del words[0][-1]
        #add space end of word
        words[-1].append(' ')
        print(words)
        #now go backwords
        i = 0
        for word in words[::-1]:
            for ch in word:
                s[i] = ch
                i += 1

class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        '''
        reverse the whole thing
        then just reverse the words
        '''
        if len(s) == 1:
            s = s[::-1]
        else:
            #first reverse
            left,right = 0,len(s)-1
            while left < right:
                s[left],s[right] = s[right],s[left]
                left += 1
                right -= 1

            #mark space index
            spaces = [0]
            for i,ch in enumerate(s):
                if ch == ' ':
                    spaces.append(i)
            #first flip
            s[0:spaces[1]] = s[0:spaces[1]][::-1]

            #middle
            for i in range(1,len(spaces)-1):
                s[spaces[i]+1:spaces[i+1]] = s[spaces[i]+1:spaces[i+1]][::-1]
            #last
            s[spaces[-1]+1:] = s[spaces[-1]+1:][::-1]

#write out as functions
class Solution:
    def reverseWords(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        def reverse(l,left,right):
            while left < right:
                l[left],l[right] = l[right],l[left]
                left += 1
                right -= 1
                
        reverse(s,0,len(s)-1)
        
        #reverse each word
        N = len(s)
        start = 0
        end = 0
        
        while start < N:
            while end < N and s[end] != " ":
                end += 1
            reverse(s,start,end-1)
            start = end + 1
            end += 1
        
####################
#  Find Median from Data Stream
#####################
#if i can use a a sortedLisr, then its easy
#simlar to insertion sort using binary search
from sortedcontainers import SortedList
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        '''
        use sorted container and keep track of size of array
        '''
        self.sortedList = SortedList()
        self.size = 0
        

    def addNum(self, num: int) -> None:
        self.size += 1
        self.sortedList.add(num)
        

    def findMedian(self) -> float:
        if self.size % 2 == 1:
            return float(self.sortedList[self.size//2])
        else:
            middle = self.size //2
            median = float(sum(self.sortedList[middle-1:middle+1]))
            return float(median/2)
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

#two heap solution
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        '''
        heap solution
        two heaps, one for lower half, one for upper half
        max heap, min heap
        and we always maintain prpoert that one if bigger than the other
        top of the max heap always holds the median
        if when equal, average the tops of both
        '''
        self.max_heap = []
        self.min_heap = []

    def addNum(self, num: int) -> None:
        #add to max
        heappush(self.max_heap,-num)
        #add to min
        heappush(self.min_heap,-heappop(self.max_heap))
        #maintain property
        if len(self.max_heap) < len(self.min_heap):
            heappush(self.max_heap, - heappop(self.min_heap))
        

    def findMedian(self) -> float:
        #max heap always has meadin when not the same size
        if len(self.max_heap) != len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) /2
        
###############
#   Isomorphic Strings
###############
#finally
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        '''
        just check for bijection
        '''
        mapp1 = {}
        mapp2 = {}
        for ch1,ch2 in zip(s,t):
            if ch1 not in mapp1:
                mapp1[ch1] = ch2
            if ch2 not in mapp2:
                mapp2[ch2] = ch1
            if mapp1[ch1] != ch2 or mapp2[ch2] != ch1:
                return False
        return True

class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        '''
        we can encode a string and check if they are the same
        add -> 122
        egg -> 122
        '''
        def encode(s):
            mapp = {}
            ans = []
            i = 0
            for ch in s:
                if ch not in mapp:
                    mapp[ch] = i
                    i += 1
                ans.append(mapp[ch])
            return ans
        
        return encode(s) == encode(t)

##################
# Find Peak Element
##################
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        '''
        n time is easy, add -float(infs) to ends and check neighbords
        '''
        nums = [float('-inf')] + nums + [float('-inf')]
        for i in range(1,len(nums)-1):
            if nums[i-1] < nums[i] and nums[i] > nums[i+1]:
                return i-1

class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        '''
        just check increasing from i to i+1
        '''
        for i in range(len(nums)-1):
            if nums[i] > nums[i+1]:
                return i

#using biary search
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        '''
        we can use binary search, there is a peak guaranteeds to be found
        we comparfe mid to mid + 1
        if the sequence mid, mid+1 is increase, the peak is on the right
        if mid,mid+1 is decreasing, peak is on the left
        '''
        def rec_binary(left,right):
            if left == right:
                return left
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid+1]:
                #it must be on the left side
                return rec_binary(left,mid)
            else:
                return rec_binary(mid+1,right)
        
        return rec_binary(0,len(nums)-1)

###################
# Custom Sort String
###################
class Solution:
    def customSortString(self, order: str, str: str) -> str:
        '''
        merge sort with defined mapping
        
        first try using for the sort functino
        '''
        mapp1 = {}
        mapp2 = {}
        for i,ch in enumerate(order):
            mapp1[ch] = i
            mapp2[i] = ch
        
        needSorting = []
        noSorting = []
        
        for ch in str:
            if ch in mapp1:
                needSorting.append(ch)
            else:
                noSorting.append(ch)
        #code to numbers
        for i in range(len(needSorting)):
            needSorting[i] = mapp1[needSorting[i]]
        #sorting
        needSorting.sort()
        #recode
        for i in range(len(needSorting)):
            needSorting[i] = mapp2[needSorting[i]]
        res = needSorting + noSorting
        return "".join(res)

#count and write
class Solution:
    def customSortString(self, order: str, str: str) -> str:
        '''
        omg this is so clever
        first count occurences in str
        then write to ne answer in order but use count, 
        then add in the rest
        '''
        counts = Counter(str)
        res = ""
        
        for ch in order:
            res += ch*counts[ch]
            counts[ch] = 0
        
        for ch in counts:
            res += ch*counts[ch]
        
        return res

#################
#  Valid Triangle Number
#################
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        '''
        we can use binary search the find the number of elements greater than nums[i] + nums[j]
        then add that to the count
        if i have two numbers i,j in sorted array
        find the smallest number k such that i + j >= k
        key:  Thus, the countcount of elements satisfying the inequality will be given by (k-1) - (j+1) + 1 = k - j - 1(k−1)−(j+1)+1=k−j−1.
        use binary search to find the right limit of k, then increase count
        '''
        def binarySearch(nums,left,right,bound):
            while right >= left and right < len(nums):
                mid = (left + right) // 2
                if nums[mid] >= bound: #find smaller one
                    right = mid -1
                else:
                    left = mid + 1
            #return the smallest k sucht tghat i + j > k
            return left
        
        count = 0
        for i in range(len(nums)-2):
            k = i+2
            j = i+1
            while j < len(nums) -1 and nums[i] != 0:
                k = binarySearch(nums,k,len(nums)-1,nums[i]+nums[j])
                #increment the count
                count += max(0,k-j-1)
                j += 1
        return count


#using bisect left to find smallest k
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        for i in range(n):
            for j in range(i+1, n):
                k = bisect_left(nums, nums[i] + nums[j])
                ans += max(0, k - 1 - j)
        return ans

class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        '''
        we can use binary search the find the number of elements greater than nums[i] + nums[j]
        then add that to the count
        if i have two numbers i,j in sorted array
        find the smallest number k such that i + j >= k
        key:  Thus, the countcount of elements satisfying the inequality will be given by (k-1) - (j+1) + 1 = k - j - 1(k−1)−(j+1)+1=k−j−1.
        use binary search to find the right limit of k, then increase count
        
        #https://leetcode.com/problems/valid-triangle-number/discuss/1339340/C%2B%2BJavaPython-Two-Pointers-Picture-Explain-Clean-and-Concise-O(N2)
        
        we can do this in O(N^2)
        recall after sorting nums[i] <= nums[j] <= nums[k] and i < j < k
        let i be the smallest element, j be the middle, and we search for k
        then we can just check nums[i] + nums[j] > nums[k]
         we fix k by iterating [2,n-1] and the answer is the total number of pairs (nums[i],nums[j]) for each nums[k] (i<j<k) so that nums[i] + nums[j] > k
         always start with i = 0,j = k-1 
         
         If nums[i] + nums[j] > nums[k] then:
There are j-i valid pairs, because in that case, when nums[k] and nums[j] are fixed, moving i to the right side always causes nums[i] + nums[j] > nums[k].
Try another nums[j] by decreasing j by one, so j -= 1.
Else if nums[i] + nums[j] <= nums[k] then:
Because nums[k] is fixed, to make the inequality correct, we need to increase sum of nums[i] + nums[j].
There is only one choice is to increase nums[i], so i += 1.
        '''
        nums.sort()
        N = len(nums)
        count = 0
        for k in range(2,N):
            left = 0
            right = k - 1
            while left < right:
                if nums[left] + nums[right] > nums[k]:
                    #verything in between is a triplet with fixed left and k
                    count += right - left
                    right -= 1
                else:
                    left += 1
        return count

###############
# 4Sum
##############
#oh shit! wooohoooo!
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        '''
        what if i use the idea from 3sum with two pointers
        sort
        fix i and j (all possible i and j)
        get their sum, and use the two pointe three sum trick
        '''
        N = len(nums)
        res = set()
        nums.sort()
        for i in range(N-3):
            for j in range(i+1,N-2):
                curr_sum = nums[i] + nums[j]
                left = j+1
                right = len(nums) - 1
                while left < right:
                    #match
                    if nums[left] + nums[right] == target-curr_sum:
                        res.add((nums[i],nums[j],nums[left],nums[right]))
                    if nums[left] + nums[right] > target-curr_sum:
                        right -= 1
                    else:
                        left += 1
        return res

#we can also avoid sorting and use hashet
#https://leetcode.com/problems/4sum/discuss/1341213/C%2B%2BPython-2-solutons-Clean-and-Concise-Follow-up%3A-K-Sum
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        '''
        we can avoid sorting and use a hashset
        given indices i,j,k and i < j < k, try all i,j,k but we can find the last element using
        lastnum = target - nums[i] - nums[j] - nums[k]
        '''
        N = len(nums)
        res = set()
        complements = set()
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    lastNum = target - nums[i] - nums[j] - nums[k]
                    if lastNum in complements:
                        #we have an answer
                        arr = sorted([nums[i],nums[j],nums[k],lastNum])
                        res.add((arr[0],arr[1],arr[2],arr[3]))
            complements.add(nums[i])
        
        return res

##################
# Three Equal Parts
##################
#close one
class Solution:
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        '''
        we are given an array of only zeros and ones
        return indices [i,j] such that
        i + 1 < j
        and arr[:i+1], arr[i+1:j], arr[j:] each reprsent the same binary number
        if not return [-1,-1]
        brute force would be get all possible i,j splits and check the above partitions
        print(arr[:0+1])
        print(arr[1:2])
        print(arr[2:])
        
        '''
        def convert(s):
            try:
                return int(s,2)
            except:
                return None
        ans = [-1,-1]
        #conver to str
        N = len(arr)
        for i in range(N):
            arr[i] = str(arr[i])
        arr = "".join(arr)
        for i in range(N):
            for j in range(i+1,N):
                if convert(arr[:i+1]) == convert(arr[i+1:j]) == convert(arr[j:]):
                    ans[0],ans[1] = i,j
        return ans

#this was a doozy
class Solution:
    def threeEqualParts(self, arr: List[int]) -> List[int]:
        '''
        the problem really just becomes: partition into three parts, remove zeros, and check
        each part contains same number of ones
        digits following the first 1 should be identical
        count number of ones in each part, and they should be the same
        sum(arr) should also be multiple of 3
        and so the first 1 in the left part is 1'th
        first 1 in middle part is numOnespart + 1'th
        first 1 in left part if numOnespart*2 + 1th
        then we compare the rest part digirt by digiit and we hope the pointer int ehr ight partr will reach len(arr)
        corner case for all zeros in A
        https://leetcode.com/problems/three-equal-parts/discuss/250203/Logical-Thinking
        '''
        countOnesUpToIdx = dict()
        count = 0
        for i,num in enumerate(arr):
            if num == 1:
                count += 1
                countOnesUpToIdx[count] = i
        #edge cases
        if count % 3 != 0:
            return [-1,-1]
        if count == 0:
            return [0,len(arr)-1]
        
        numOnesPerPart = count // 3
        #now allocate pointers
        left = countOnesUpToIdx[1]
        mid = countOnesUpToIdx[1+numOnesPerPart]
        right = countOnesUpToIdx[1+numOnesPerPart*2]
        #move all three and check
        while left < mid < right < len(arr) and arr[left] == arr[mid] == arr[right]:
            left += 1
            mid += 1
            right += 1
        if right == len(arr):
            return [left-1,mid]
        return [-1,-1]

##################
#  Reverse Nodes in k-Group
###################
#fucking edge cases
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        '''
        well the dumb way would be to pull them all out into an array
        reverse for every k, then make new list
        start with this
        '''
        temp = []
        dummy = head
        while dummy:
            temp.append(dummy.val)
            dummy = dummy.next
        
        left, right = 0,0
        while right < len(temp):
            if right - left == k:
                temp[left:right] = temp[left:right][::-1]
                left = right
            right += 1
        
        #rebuild
        dummy = ListNode()
        dummy2 = dummy
        for num in temp:
            dummy.next = ListNode(val=num)
            dummy = dummy.next
        return dummy2.next

#recursive
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        '''
        lets go over the recusrive solution first
        inution:
            linked lists are recursive structures, a sublist is also a linkeind list
            head, rev_head, list are pointers, move head through ptr
            attach curr ptr to rev_head, then reverse
            if there are < k nodes left, leave them, only reverse k nodes
        algo:
            * get reverse function for linked lists, only touching k nodes
            * each call count nodes, and break at k
            * if < k nodes left, do nothing
            * if we have hit k nodes, invoke
            * we need to return the head of the linked list
        So, in every recursive call, we first reverse k nodes, then recurse on the rest of the linked list. 
        When recursion returns, we establish the proper connections.    
        '''
        def reverse(head,k):
            #reverse LL of size k, and return head of reversed
            new_head = None
            ptr = head
            while k:
                #get next node
                next_node = ptr.next
                #reverse
                ptr.next = new_head
                new_head = ptr
                #on to next
                ptr = next_node
                #use up k
                k -= 1
            return new_head
        
        def rec_reverse(head,k):
            count = 0
            ptr = head
            #first check under k nodes
            while count < k and ptr is not None:
                ptr = ptr.next
                count += 1
            #reverse
            if k == count:
                #get head of revesed
                rev_head = reverse(head,k)
                #and now we recurse, assuming answer from call is already reversed
                head.next = rec_reverse(ptr,k)
                return rev_head
            return head
        
        #invoke
        return rec_reverse(head,k)

#O(1) space iterative
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        '''
        in additino to the rev and rev_head, we also need to know the tail of prev k nodes
        recursive approach reverse k nodes left to right, but  connects from right to left back to front
        algo:
            * we have same reverse function as before
            * keep head,revHead,ktail,newHead
            * same as before, count k and reverse if we can
            * check if we have k tail or not, it should not be set before first k nodes
            * then attach ktail.next = revhead
            * ktail = head
        '''
        def reverse(head,k):
            #reverse LL of size k, and return head of reversed
            new_head = None
            ptr = head
            while k:
                #get next node
                next_node = ptr.next
                #reverse
                ptr.next = new_head
                new_head = ptr
                #on to next
                ptr = next_node
                #use up k
                k -= 1
            return new_head
        
        #main part
        ptr,ktail = head,None
        #head of final
        new_head = None
        
        while ptr:
            count = 0
            #send out ptr form head
            ptr = head
            while count < k and ptr:
                ptr = ptr.next
                count += 1
            #k nodes
            if count == k:
                #reverse
                revHead = reverse(head,k)
                #new head is now head
                if not new_head:
                    new_head = revHead
                #check tail
                if ktail:
                    ktail.next = revHead
                ktail = head
                head = ptr
                
        #final unmet k
        if ktail:
            ktail.next = head
        
        return new_head if new_head else head

#########################
# Lowest Common Ancestor of a Binary Search Tree
########################
#similar problem to just binary tree one
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node):
            if node == None:
                return None
            if node == p or node == q:
                return node
            left = dfs(node.left)
            right = dfs(node.right)
            #if there something to return on left or right, this must be the LCA
            if left != None and right != None:
                return node
            #if we returned nothing, retuing nothing
            if left == None and right == None:
                return None
            return left if left else right
        
        return dfs(root)

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        recursing from the parent node
        if both nodes p and q are in the right, keep looking right
        if both nodes p and q are on the left, keep looking left
        if we cant find them, then return the root
        '''
        def dfs(node,p,q):
            #both are on the right
            if p.val > node.val and q.val > node.val:
                return dfs(node.right,p,q)
            elif p.val < node.val and q.val < node.val:
                return dfs(node.left,p,q)
            #else we cant find it, return node
            else:
                return node
        return dfs(root,p,q)

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        recursing from the parent node
        if both nodes p and q are in the right, keep looking right
        if both nodes p and q are on the left, keep looking left
        if we cant find them, then return the root
        '''
        while root:
            #both are on the right
            if p.val > root.val and q.val > root.val:
                root = root.right
            elif p.val < root.val and q.val < root.val:
                root = root.left
            #else we cant find it, return node
            else:
                return root

################
# Shuffle an Array
################
class Solution:
    '''
    brute force, put nums in a 'hat' and draw them out one by one
    copy array into a new one before removing/overwirting
    notes on time complexity
        for an elmenent e, its probability of getting chosen at the kth iteration is 1 / (n-k)
        the probabilty of it not being chosen is 1 - 1 / (n-k)
        or:
            prod_{i=1}^{k} \frac{n-i}{n-k+i}
            after expandidng we get \frac{1}{n-k}
            and we eventually get that each element has a 1/n chance of being chosen
    '''

    def __init__(self, nums: List[int]):
        #cache original
        self.nums = nums
        

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        return self.nums
        

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        '''
        shuffle self.nums
        '''
        aux = self.nums[:]
        temp = []
        for i in range(len(aux)):
            rm_idx = random.randrange(len(aux))
            temp.append(aux.pop(rm_idx))
        return temp

#Fisher Yates
class Solution:
    '''
    fisher yates, for each index i in the array, randomnly draw number from (idx,len(nums))
    and swap
    '''
    def __init__(self, nums: List[int]):
        self.nums = nums
        

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        return self.nums
        

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        temp = self.nums[:]
        for i in range(len(temp)):
            swap_idx = random.randrange(i,len(temp))
            temp[i],temp[swap_idx] = temp[swap_idx],temp[i]
        return temp

####################
# Push Dominoes
####################
#yikes i have no idea
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        '''
        the ends are easy, just move them in the directino they are falling
        for sequences like ...R.....L....
        the domino in the middle of R and L should be up right (if there are odd dots in between)
        otherwise half are R and half are L
        i need to mark indexes of consectuive RL, with separtion of wtv
        and take care of those first
        then just forward fill their directions
        
        use min q;s for right and left occruences
        '''
        doms = list(dominoes)
        #topple R..L sequences
        i = 0
        while i < len(doms):
            if doms[i] == 'R':
                j = i + 1
                while j < len(doms) and doms[j] != 'L':
                    j += 1
                #if we've reached an L, change between i and j
                if doms[j] == 'L':
                    for k in range(i,(j-i)//2):
                        doms[k] = 'R'
                    for k in range(j,(j-i)//2,-1):
                        doms[k] = 'L'
                #all right
                else:
                    for k in range(i,j):
                        doms[k] = 'R'
                #move i to j
                i += 1
            else:
                i += 1
        return "".join(doms)
            
class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        '''
        two passes, from left to right, get distance to curr ids to previous 'R'
        save to distances array
        second pass, right to left, count distance of current index to previous 'L' then:
            if Ldist < rdit (dist[i]) curr cell should be 'L'
            if ldist == rdist, curr cell should be '.'
        '''
        doms = list(dominoes)
        nearest_R = [0]*len(doms)
        curr_R = None
        #first pass, R distances
        for i,val in enumerate(doms):
            if val == 'R':
                curr_R = 0
            elif val == 'L':
                curr_R = None
            elif curr_R != None:
                curr_R += 1
                nearest_R[i] = curr_R
                doms[i] = 'R'
        
        #second pass, get dist from L
        curr_L = None
        for i in range(len(doms)-1,-1,-1):
            if doms[i] == 'L':
                curr_L = 0
            elif doms[i] == 'R':
                curr_L = None
            elif curr_L != None:
                curr_L += 1
                if curr_L < nearest_R[i] or doms[i] == ".":
                    doms[i] = 'L'
                elif curr_L == nearest_R[i]:
                    doms[i] = '.'
        
        return "".join(doms)

class Solution:
    def pushDominoes(self, dominoes: str) -> str:
        '''
        a cool way would be to calculate the forces going left to right
        and getting forces right to left
        going left to right, we decrease our force 1, but resent to n again with right
        but right to left we detemine by closness to f
        forces with zero are '.'
        '''
        N = len(dominoes)
        force = [0] * N

        # Populate forces going from left to right
        f = 0
        for i in range(N):
            if dominoes[i] == 'R': f = N
            elif dominoes[i] == 'L': f = 0
            else: f = max(f-1, 0)
            force[i] += f

        # Populate forces going from right to left
        f = 0
        for i in range(N-1, -1, -1):
            if dominoes[i] == 'L': f = N
            elif dominoes[i] == 'R': f = 0
            else: f = max(f-1, 0)
            force[i] -= f

        return "".join('.' if f==0 else 'R' if f > 0 else 'L'
                       for f in force)


#####################
# Range Addition
#####################
#TLE if i do what it says
class Solution:
    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        '''
        i start with the zeros array
        and for each entry in updates, we need to update all elements in arr[updates[0]:updates[1]] by updates[2]
        '''
        arr = [0]*length
        for update in updates:
            for i in range(update[0],update[1]+1):
                arr[i] += update[2]
        return arr
        
class Solution:
    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        '''
        i start with the zeros array
        and for each entry in updates, we need to update all elements in arr[updates[0]:updates[1]] by updates[2]
        hint says only to updat first and end element
        we only update the first and last elemtns in thw array
        then take the cumsum as the final answer
        '''
        arr = [0]*(length+1)
        for update in updates:
            arr[update[0]] += update[2]
            arr[update[1]+1] -= update[2]
        for i in range(1,length+1):
            arr[i] += arr[i-1]
        return arr[:-1]

###########################################
# Partition Array into Disjoint Intervals
############################################
#almost
class Solution:
    def partitionDisjoint(self, nums: List[int]) -> int:
        '''
        the problem really just comes down to max(left array) <= min(right array)
        find all the maxes for the left side
        find all the mins for the right side
        compare the two
        '''
        N = len(nums)
        max_lefts = [0]*N
        min_right = [0]*N
        
        #find maxes
        MAX = nums[0]
        for i in range(N):
            MAX = max(MAX,nums[i])
            max_lefts[i] = MAX
        
        #finds mins right
        MIN = nums[-1]
        for i in range(N-1,-1,-1):
            MIN = min(MIN,nums[i])
            min_right[i] = MIN
        
        for i in range(1,N):
            if max_lefts[i-1] <= min_right[i]:
                return i

#################
# Binary Tree Pruning
###################
#hell if i kknow, close one though
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        '''
        we need to remove all parts of subtrees that does not have a 1
        '''
        ans = root
        
        def dfs(node):
            if not node:
                return
            if node:
                if node.val == 0 and node.left.val == 0 and node.right.val == 0:
                    return True
            left = dfs(node.left)
            right = dfs(node.right)
            if left:
                node.left = None
            if right:
                node.right = None
            return node
        
        dfs(root)


class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        def has_ones(node):
            if not node:
                return False
            left = has_ones(node.left)
            right = has_ones(node.right)
            
            if left == 0:
                node.left = None
            if right == 0:
                node.right = None
            
            return node.val or left or right
        
        return root if has_ones(root) else None

#just another way
# https://leetcode.com/problems/binary-tree-pruning/discuss/1356558/C%2B%2BPython-DFS-Post-Order-Clean-and-Concise-O(N)
class Solution:
    def pruneTree(self, root: TreeNode) -> TreeNode:
        '''
        if current subtree  ontains 1 (i.e root.val == 1 or left contains 1 or right continas 1) return root
        otherwise return None
        Set root.left = null if its leftSubtree doesn't contains 1, otherwise keep root.left as its original node.
Do the same thing with root.right: Set root.right = null if its rightSubtree doesn't contains 1, otherwise keep root.right as its original node.
There is a simillar problem with this concept, you can try: 1110. Delete Nodes And Return Forest
        '''
        def prune(node):
            if not node:
                return node
            node.left = prune(node.left)
            node.right = prune(node.right)
            if node.left != None or node.right != None or node.val == 1:
                return node
            return None
        
        return prune(root)

###############
# Word Ladder II
###############
#close one again, nice try though
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        '''
        i could generate all paths, and pick the shortest from them
        all words have the same length
        i need to make adj list for each word, where i can swap one char and get to the next word
        
        '''
        wordListSet = set(wordList)
        #can't get to end
        if endWord not in wordListSet:
            return []
        
        #add begin word to wordList
        wordList = [beginWord] + wordList
        
        #make adj list {word:[words a change away]}
        adj_list = defaultdict(list)
        size = len(beginWord)
        N = len(wordList)
        for i in range(size):
            for j in range(N):
                for k in range(N):
                    if j != k:
                        #get words
                        word1 = wordList[j]
                        word2 = wordList[k]
                        if word1[:i]+"*"+word1[i+1:] == word2[:i]+"*"+word2[i+1:]:
                            adj_list[word1].append(word2)
        
        #we want paths, but beginWord might not be in wordList
        #we need to find the start,then we can dfs        
        #generate all paths and pick smallest
        self.paths = []
        def dfs(curr,path,seen):
            if curr == endWord:
                path.append(curr)
                self.paths.append(path)
                return
            path.append(curr)
            seen.add(curr)
            for neigh in adj_list[curr]:
                if neigh not in seen:
                    dfs(neigh,path,seen)
            return
                    
        dfs(beginWord,[],set())
        print(self.paths)

#layer by layer BFS
#https://leetcode.com/problems/word-ladder-ii/discuss/490116/Three-Python-solutions%3A-Only-BFS-BFS%2BDFS-biBFS%2B-DFS
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        '''
        BFS to find shortest paths, but we keep BFS'ing for each char in len(beginWord)
        in this case we need to keep doing BFS at each level
        a word along a path can be used again in another path, so we need a way to control for this
        '''
        #corner cases
        if not endWord or not beginWord or not wordList or endWord not in wordList or beginWord == endWord:
            return []
        
        L = len(beginWord)
        #build adjlist to hold all possible paths from next word
        adj_list = defaultdict(list)
        for word in wordList:
            for i in range(L):
                key = word[:i]+"*"+word[i+1:]
                adj_list[key].append(word)
        
        #shortest path bfs
        paths = []
        q = deque([(beginWord,[beginWord])])
        visited = set([beginWord])
        
        while q and not paths:
            size = len(q)
            #to allow for words to be on a different path
            localVisited = set()
            for _ in range(size):
                word,path = q.popleft()
                for i in range(L):
                    for neigh in adj_list[word[:i]+"*"+word[i+1:]]:
                        #ending
                        if neigh == endWord:
                            paths.append(path+[endWord]) #dont forget to add in the final end
                        if neigh not in visited:
                            localVisited.add(neigh)
                            q.append((neigh,path+[neigh]))
            #once im done with this layer adjust visited
            visited |= localVisited
        return paths

#bfs to buiild layers, then dfs to find shortest paths
def findLadders(self, beginWord, endWord, wordList):
	"""
	:type beginWord: str
	:type endWord: str
	:type wordList: List[str]
	:rtype: List[List[str]]
	"""
	if not endWord or not beginWord or not wordList or endWord not in wordList \
		or beginWord == endWord:
		return []

	L = len(beginWord)

	# Dictionary to hold combination of words that can be formed,
	# from any given word. By changing one letter at a time.
	all_combo_dict = collections.defaultdict(list)
	for word in wordList:
		for i in range(L):
			all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)

	# Build graph, BFS
	# ans = []
	queue = collections.deque()
	queue.append(beginWord)
	parents = collections.defaultdict(set)
	visited = set([beginWord])
	found = False 
	depth = 0
	while queue and not found:
		depth += 1 
		length = len(queue)
		# print(queue)
		localVisited = set()
		for _ in range(length):
			word = queue.popleft()
			for i in range(L):
				for nextWord in all_combo_dict[word[:i] + "*" + word[i+1:]]:
					if nextWord == word:
						continue
					if nextWord not in visited:
						parents[nextWord].add(word)
						if nextWord == endWord:    
							found = True
						localVisited.add(nextWord)
						queue.append(nextWord)
		visited = visited.union(localVisited)
	# print(parents)
	# Search path, DFS
	ans = []
	def dfs(node, path, d):
		if d == 0:
			if path[-1] == beginWord:
				ans.append(path[::-1])
			return 
		for parent in parents[node]:
			path.append(parent)
			dfs(parent, path, d-1)
			path.pop()
	dfs(endWord, [endWord], depth)
	return ans

#bidirectinal BFS
## Solution 3
def findLadders(self, beginWord, endWord, wordList):
	"""
	:type beginWord: str
	:type endWord: str
	:type wordList: List[str]
	:rtype: List[List[str]]
	"""
	if not endWord or not beginWord or not wordList or endWord not in wordList \
		or beginWord == endWord:
		return []

	L = len(beginWord)

	# Dictionary to hold combination of words that can be formed,
	# from any given word. By changing one letter at a time.
	all_combo_dict = collections.defaultdict(list)
	for word in wordList:
		for i in range(L):
			all_combo_dict[word[:i] + "*" + word[i+1:]].append(word)

	# Build graph, bi-BFS
	# ans = []
	bqueue = collections.deque()
	bqueue.append(beginWord)
	equeue = collections.deque()
	equeue.append(endWord)
	bvisited = set([beginWord])
	evisited = set([endWord])
	rev = False 
	#graph
	parents = collections.defaultdict(set)
	found = False 
	depth = 0
	while bqueue and not found:
		depth += 1 
		length = len(bqueue)
		# print(queue)
		localVisited = set()
		for _ in range(length):
			word = bqueue.popleft()
			for i in range(L):
				for nextWord in all_combo_dict[word[:i] + "*" + word[i+1:]]:
					if nextWord == word:
						continue
					if nextWord not in bvisited:
						if not rev:
							parents[nextWord].add(word)
						else:
							parents[word].add(nextWord)
						if nextWord in evisited:    
							found = True
						localVisited.add(nextWord)
						bqueue.append(nextWord)
		bvisited = bvisited.union(localVisited)
		bqueue, bvisited, equeue, evisited, rev = equeue, evisited, bqueue, bvisited, not rev
	# print(parents)
	# print(depth)
	# Search path, DFS
	ans = []
	def dfs(node, path, d):
		if d == 0:
			if path[-1] == beginWord:
				ans.append(path[::-1])
			return 
		for parent in parents[node]:
			path.append(parent)
			dfs(parent, path, d-1)
			path.pop()
	dfs(endWord, [endWord], depth)
	return ans

#just and additional way
#https://leetcode.com/problems/word-ladder-ii/discuss/1359027/C%2B%2BPython-BFS-Level-by-Level-with-Picture-Clean-and-Concise
class Solution:
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        '''
        another way of doing it level by level with BFS
        using a yeild function to make it go faster
        BFS level by level, but for each BFS, we can keep storing the paths
        '''
        wordSet = set(wordList)
        #drop beginWord if its in there
        wordSet.discard(beginWord)
        
        def neighbors(word):
            for i in range(len(word)):
                for ch in range(26):
                    newWord = word[:i]+chr(97+ch)+word[i+1:]
                    if newWord in wordSet:
                        yield newWord
        #layerwise BFS
        layer = {}
        layer[beginWord] = [[beginWord]]
        while layer:
            nextLayer = defaultdict(list)
            for word,paths in layer.items():
                if word == endWord:
                    return paths
                for neigh in neighbors(word):
                    for path in paths:
                        nextLayer[neigh].append(path+[neigh])
            wordSet -= set(nextLayer.keys())
            layer = nextLayer
        
        return []

###################################################
# Non-negative Integers without Consecutive Ones
###################################################
#close one, good enough for interview i feel,TLE
class Solution:
    def findIntegers(self, n: int) -> int:
        '''
        brute force would be to generate the binary reps
        check for consectuvie ones, and increment count if so
        i can use the bit shift to check
        (inter << n) & 1, gives me whether or not this is bit value is on
        at most this would run log_2(10**9) about 30, which isn't bad!
        '''
        #function to show has consectutive ones
        def consec_ones(num):
            one_found = False
            while num:
                if one_found and num & 1:
                    return True
                elif num & 1:
                    one_found = True
                else:
                    one_found = False
                num = num >> 1
            return False
        
        count = 0
        for i in range(n+1):
            if not consec_ones(i):
                count += 1
        return count

class Solution:
    def findIntegers(self, n: int) -> int:
        def check(num):
            i = 31
            while i > 0:
                if (num & (1 << i)) != 0 and (num & (1 << i-1)) != 0:
                    return False
                i -= 1
            return True
        
        count = 0
        for i in range(n+1):
            if check(i):
                count += 1
        return count

#recusrive solution
class Solution:
    def findIntegers(self, n: int) -> int:
        '''
        we make use of a recursive function
        rec(num,limit), which returns the number of valid integers that do not contain consecutive bits
        if the bit ended with a 1, we can only add a zero to is, which is the same as shifting over 1 place
        if it doesn't end in 1, we can append a zero or 1
        '''
        memo = {}
        def validCounts(num,limit):
            if num > limit:
                return 0
            if (num) in memo:
                return memo[num]
            res = None
            if num & 1:
                res = 1 + validCounts((num << 1) | 0,limit)
                memo[num] = res
                return res
            else:
                add_zero = validCounts((num << 1) | 0,limit)
                add_one = validCounts((num << 1) | 1, limit)
                res = 1 +  add_zero + add_one
                memo[num] = res
                return res
            return res
        
        return 1+validCounts(1,n)

#dp, think fibonacci
class Solution:
    def findIntegers(self, n: int) -> int:
        '''
        https://leetcode.com/problems/non-negative-integers-without-consecutive-ones/discuss/1361794/Python3-Official-Solution-Explained-Simply-with-Diagrams
        in reality this is similar to the fibnoccai sequence
        we aren't allowed consectuive bits of 11, but we are allowed 00,01,10
        say we have 4 bits to place
        we can have 0XXX and 10XXX
        the ans(xxxx) = ans(xxx) + ans(xx), which is the the int shifted one time
        and the int shifted twice or ans[i] = ans[i-1] + ans[i-2]
        which is the fibonnaci sequence
        '''
        f = [1,2]
        for i in range(2,30):
            f.append(f[-1]+f[-2])
        #last_seen tells us if there was a 1 right before
        count = 0
        last_seen = 0
        for i in range(30-1,-1,-1):
            if (1 << i) & n:
                count += f[i]
                if last_seen:
                    count -=1
                    break
                last_seen = 1
            else:
                last_seen = 0
        return count + 1

####################
# Convert Sorted Array to Binary Search Tree
####################
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        '''
        just use recursion and keep going until the middle of the array
        it will always be a height balacned treee
        '''
        N = len(nums)
        def build_tree(left,right):
            if left > right:
                return None
            mid = (right + left) //2
            root = TreeNode(nums[mid])
            root.left = build_tree(left,mid-1)
            root.right = build_tree(mid+1,right)
            return root
        
        return build_tree(0,N-1)

#we also could have chosen right middle
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:        
        def helper(left, right):
            if left > right:
                return None
            
            # always choose right middle node as a root
            p = (left + right) // 2 
            if (left + right) % 2:
                p += 1 

            # preorder traversal: node -> left -> right
            root = TreeNode(nums[p])
            root.left = helper(left, p - 1)
            root.right = helper(p + 1, right)
            return root
        
        return helper(0, len(nums) - 1)

######################
# 3Sum Closest
######################
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        '''
        i can sort the array and fix an i
        then try picking two elements using the two pointer technique
        and update the anser when candidates abs(target - i - j - k) > current candiates answer
        '''
        N = len(nums)
        curr_diff = float('inf')
        #sort
        nums.sort()
        ans = [0,0,0]
        
        for i in range(N):
            j = i+1
            k = N-1
            while j < k:
                diff = abs(target - nums[i] - nums[j] - nums[k])
                if diff < curr_diff:
                    curr_diff = diff
                    ans[0],ans[1],ans[2] = nums[i],nums[j],nums[k]
                j += 1
                k -= 1
        
        return sum(ans)

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        '''
        i can sort the array and fix an i
        then try picking two elements using the two pointer technique
        and update the anser when candidates abs(target - i - j - k) > current candiates answer
        '''
        N = len(nums)
        diff = float('inf')
        #sort
        nums.sort()
        ans = [0,0,0]
        
        for i in range(N):
            j = i+1
            k = N-1
            while j < k:
                SUM = nums[i] + nums[j] + nums[k]
                if abs(target - SUM) < abs(diff):
                    diff = target - SUM
                if SUM < target:
                    j += 1
                else:
                    k -= 1
            if diff == 0:
                break
        return target - diff
               
#####################
# Alien Dictionary
#####################
#dammit, close one
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        '''
        words are sorted leoxgrphically in the alien language
        return the chars in lexographic order
        if there are ties, we take the smaller length string to be first
        similar to radix sort
        put words into a column, then add to list, keeping seen char set
        
        watch edge cases where len(str[i]) < max legnth of longest string
        '''
        #first make mapping of all chars, intially set to zero
        seen = set() #clear every time
        longest = 0
        for w in words:
            longest = max(longest,len(w))
        letters = [] 

        start = 0
        idx = 1
        while longest > 0:
            #go in order of wrod starting fromt left most char
            for w in words:
                #edge case
                if start >= len(w):
                    continue 
                if w[start] not in seen:
                    letters.append(w[start])
                    seen.add(w[start])
            start += 1
            longest -= 1
        
        #pass letters to check order
        ans = "".join(letters)
        
        #but no we need to check using the mapping we made
        mapp = {char:idx for idx,char in enumerate(ans) }
        
        N = len(words)
        for i in range(N-1):
            first = words[i]
            second = words[i+1]
            for j in range(min(len(first),len(second))):
                if mapp[first[j]] > mapp[second[j]]:
                    return ""
        return ans
                
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        '''
        lets go through this while solution, there are three key insights
        1. getting as much representations about the alpha bet
        2. represent infor in meniangful way
        3. get valid ordering
        1.
            we can relativer order just by going left to right, then de duplicate
            for each two adjacvent words, look for the first difference between them
            this tells us the relative order between two letters
            now we have relations
        2. 
            it is tempting to just chain the relations in order in the way we found them
            but what happens? some letters are used more than once
            we draw graph from relations!
        3.
            if in the graph we have nodes have no indriection, then these ones could be in order, but must come before any of the the other letters
            remove the nodes with indirection of zero from graph 
            we keep removing and add no indirection letters in order
            #fun fact, we could also get the total number of orders using the number of no indfriectinos nodes, ie.num factorial
        algo:
            first two parts are easy, but how do we efficiently find nodes with indirection zero and remove them
            we can keep two adj_lists, one for reverse and one for forward
            but we can better! when building the adjList keep mapp for size of indegree for each letter
            removing and edge means decrement 1, and 0 is one of the letters in the first iteration
            bfs to find reachable letters
            fter that, we check whether or not all letters were put in the output list.
            if there is a cycle, we return ""
            edge case, if word is followed by prefix, like z x z
        '''
        #build adj_list
        adj_list = defaultdict(set)
        #build in degree
        in_degree = defaultdict()
        for word in words:
            for c in word:
                in_degree[c] = 0
        #build adj_list
        for word1,word2 in zip(words[:-1],words[1:]):
            for c,d in zip(word1,word2):
                if c != d:
                    #add edge
                    if d not in adj_list[c]:
                        adj_list[c].add(d)
                        #indcrment indegree to d
                        in_degree[d] += 1
                    #stop after first mismatch
                    break
            else:
                #prefix edge case
                if len(word2) < len(word1):
                    return ""
        
        #bfs, load in zeros, remove edge and find new zero indegrees
        order = []
        #add in zero in degree nodes
        q = deque([c for c in in_degree if in_degree[c] == 0])
        #BFS
        while q:
            #get node
            c = q.popleft()
            #add back in
            order.append(c)
            #remove an edge from c
            for d in adj_list[c]:
                in_degree[d] -= 1
                #if zero add to list
                if in_degree[d] == 0:
                    q.append(d)
        #one more edge case, if there was a cycle
        if len(order) < len(in_degree):
            return ""
        return "".join(order)

#####################
# Beautiful Array
#####################
class Solution:
    def beautifulArray(self, n: int) -> List[int]:
        '''
        what this really means is that the array must be arithmetic free
        we can only choose [1,2,3...n]
        this must be satisfied using (a+b, a+2*b, a+3*b....a+(n-1)*b)
        if we divide and conquer, we have to parts, left and right that are arithmetic free
        we only want a triple from both pats not arithmetic
        left must have all even elements and right must have all odd elements
        Another way we could arrive at this is to try to place a number in the middle, like 5. We will have 4 and 6 say, to the left of 5, and 7 to the right of 6, etc. 
        We see that in general, odd numbers move towards one direction and even numbers towards another direction.
        Looking at the elements 1, 2, ..., N, there are (N+1) / 2 odd numbers and N / 2 even numbers.

        We solve for elements 1, 2, ..., (N+1) / 2 and map these numbers onto 1, 3, 5, .... 
        Similarly, we solve for elements 1, 2, ..., N/2 and map these numbers onto 2, 4, 6, ....

        We can compose these solutions by concatenating them, since an arithmetic sequence never starts and ends with elements of different parity.

        We memoize the result to arrive at the answer quicker
        
        '''
        N = n
        memo = {1: [1]}
        def f(N):
            if N not in memo:
                odds = f((N+1)//2)
                evens = f(N//2)
                memo[N] = [2*x-1 for x in odds] + [2*x for x in evens]
            return memo[N]
        return f(N)

class Solution:
    def beautifulArray(self, n: int) -> List[int]:
        '''
        https://leetcode.com/problems/beautiful-array/discuss/186679/Odd-%2B-Even-Pattern-O(N)
        just another way of doing this, we can divide the N ints into left and right parts
        [1,N//2] and [N//2 +1,N] partss but really we want to divide into odd and even parts
        we notice partterns for N = 5 example
        [2, 1, 4, 5, 3]
        [3, 1, 2, 5, 4]
        [3, 5, 4, 1, 2]
        [4, 5, 2, 1, 3]
        Saying that an array is beautiful,
        there is no i < k < j,
        such that A[k] * 2 = A[i] + A[j]
        2. Addition
        If we have A[k] * 2 != A[i] + A[j],
        (A[k] + x) * 2 = A[k] * 2 + 2x != A[i] + A[j] + 2x = (A[i] + x) + (A[j] + x)

        E.g: [1,3,2] + 1 = [2,4,3].
        3. Multiplacation
        If we have A[k] * 2 != A[i] + A[j],
        for any x != 0,
        (A[k] * x) * 2 = A[k] * 2 * x != (A[i] + A[j]) * x = (A[i] * x) + (A[j] * x)

        E.g: [1,3,2] * 2 = [2,6,4]
        
        Explanation
        With the observations above, we can easily construct any beautiful array.
        Assume we have a beautiful array A with length N

        A1 = A * 2 - 1 is beautiful with only odds from 1 to N * 2 -1
        A2 = A * 2 is beautiful with only even from 2 to N * 2
        B = A1 + A2 beautiful array with length N * 2
        '''
        array = [1]
        while len(array) < n:
            odds = [num*2 - 1 for num in array]
            evens = [num*2 for num in array] 
            array = odds + evens
        return [num for num in array if num <= n]

#######################
# 01 Matrix
#######################
#bfs from each cell
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        '''
        just bfs from each ij cell
        '''
        matrix = mat
        rows = len(matrix)
        cols = len(matrix[0])
        def bfs(node):
            #node is a tuple
            i,j = node
            q = deque([(i,j,0)]) #eleemnts is (x,y,dist), of course starting from zero
            directions = [(1,0),(-1,0),(0,1),(0,-1)]
            visited = set()
            
            while q:
                x,y,d = q.popleft()
                #we keep going until we hit a zero
                if matrix[x][y] == 0:
                    return d
                #add to visited 
                visited.add((x,y))
                #otherwise bfs
                for dirr in directions:
                    newx,newy = x+dirr[0],y+dirr[1]
                    #boundary check
                    if 0 <= newx < rows and 0 <= newy < cols:
                        #not visited gain
                        if (newx,newy) not in visited:
                            q.append((newx,newy,d+1))
            return -1
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 1:
                    d = bfs((i,j))
                    matrix[i][j] = d
        return matrix

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        '''
        we could also start searching from all zeros
        q up all (i,j) that are zero
        then bfs from each cell in the q, but since they were all zero, we know that they are one away
        
        '''
        dirrs = [(0,1),(0,-1),(1,0),(-1,0)]
        q = deque()
        rows = len(mat)
        cols = len(mat[0])
        seen = set()
        for i in range(rows):
            for j in range(cols):
                if mat[i][j] == 0:
                    seen.add((i,j))
                    q.append((i,j))
                    
        #now bfs from all these zeros
        while q:
            x,y = q.popleft()
            for dx,dy in dirrs:
                new_x = x + dx
                new_y = y + dy
                #should be a 1 now if we havent seen it
                if 0 <= new_x < rows and 0 <= new_y < cols and (new_x,new_y) not in seen:
                    mat[new_x][new_y] = mat[x][y] + 1
                    seen.add((new_x,new_y))
                    q.append((new_x,new_y))
        return mat

#################
# Binary Tree Longest Consecutive Sequence II
#################
#fuckkkk
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        '''
        brute force would be to generate all paths and check
        but we need to be careful for both increasing and decreasing sequences
        we could just traverse the tree twice! and compare the depths
        two dfs calls, one checks increasing, one checks decreasing
        '''
        #increasing first
        def dfs_1(node,path):
            if not node:
                return path
            if node.left:
                if node.left.val > node.val:
                    dfs_1(node.left,path+1)
            if node.right:
                if node.right.val > node.val:
                    dfs_1(node.right,path+1)
            return path
        
        print(dfs_1(root,0))

#nope


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: TreeNode) -> int:
        '''
        note the path can be a child, parent, child
        not just descneding
        for each call from a node, we can return an array [inc,dec]
        which is the the size of the increasing path and decreasing path from that node
        both inc and dec from each node start off at 1
        whenever we go left, check left and right children of inc and dec
        and add 1 accordingling
        same for the right path
        once we have borth answer we can take the max
        key, at each node we return the max length path so far
        
        '''
        self.max_length = 0
        
        def dfs(node):
            if not node:
                return [0,0]
            #alwasy 1,1 from a node
            inc = 1
            dec = 1
            
            #left side
            if node.left:
                left = dfs(node.left)
                #decreasing
                if node.val == node.left.val+1:
                    dec = left[1] + 1
                #increasing
                elif node.val == node.left.val -1:
                    inc = left[0] + 1
            
            #right side
            if node.right:
                right = dfs(node.right)
                if node.val == node.right.val + 1:
                    dec = max(dec,right[1]+1)
                elif node.val == node.right.val - 1:
                    inc = max(inc,right[0]+1)
            
            #global upate asnwert
            self.max_length = max(self.max_length, inc+dec-1)
            return [inc,dec]
        
        dfs(root)
        return self.max_length

##################
# Map Sum Pairs
##################
#brute force
class MapSum:
    '''
    i can just hash map everything
    then check for a prefix in the mapp
    '''
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.mapp = {}
        
        

    def insert(self, key: str, val: int) -> None:
        self.mapp[key] = val
        

    def sum(self, prefix: str) -> int:
        res = 0
        for key,val in self.mapp.items():
            if key.startswith(prefix):
                res += val
        return res
            
#store all possible prefixes
class MapSum:
    '''
    we can store all possible prefs when we insert
    don't forget to update a new pref
    '''

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keys = {}
        self.answers = Counter()
        

    def insert(self, key: str, val: int) -> None:
        #get the current offset
        delta = val - self.keys.get(key,0)
        #load in curent value
        self.keys[key] = val
        #now put int all possible prefixes
        for i in range(len(key)+1):
            prefix = key[:i]
            self.answers[prefix] += delta
        

    def sum(self, prefix: str) -> int:
        return self.answers[prefix]

#Trie
class TrieNode:
    def __init__(self):
        self.child = defaultdict(TrieNode)
        self.sum = 0
        

class MapSum:
    '''
    Trie Solution
    when we call insert, the key string will go through all the chars in the Trie
    and we increase the sum value of each node by diff
    where diff = va; - self.map[key]
    why? because we need to update each key val as well as all of keys prefixes because it gets overriddenr
    that's why we need a hashmap and a Trie
    '''
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        self.map = defaultdict(int)
        

    def insert(self, key: str, val: int) -> None:
        diff = val - self.map[key]
        curr = self.root
        for ch in key:
            #move through trie
            curr = curr.child[ch]
            curr.sum += diff
        #put into mapp
        self.map[key] = val
        

    def sum(self, prefix: str) -> int:
        curr = self.root
        for ch in prefix:
            if ch not in curr.child:
                return 0
            curr = curr.child[ch]
        return curr.sum

#######################
# Trapping Rain Water
#######################
#brute force
class Solution:
    def trap(self, height: List[int]) -> int:
        '''
        for each element in the array
        we find the max level of water it can trap after rain
        which is miniumum of max height of of bars on both sides minus its own
        example
        [1,0,1]
        '''
        ans = 0
        N = len(height)
        for i in range(N):
            left_max = 0
            right_max = 0
            #find left max going from current to beginning
            for j in range(i,-1,-1):
                left_max = max(left_max,height[j])
            #find right max, curr element to end
            for j in range(i,N):
                right_max = max(right_max,height[j])
            ans += min(left_max,right_max) - height[i]
        return ans

class Solution:
    def trap(self, height: List[int]) -> int:
        '''
        https://www.youtube.com/watch?v=ZkrXxi5ay80&ab_channel=SaiAnishMalla
        idea is we reaally care about max value to the left of a cell
        and what is the larger value to the right
        thinkg baout hthe example [5,4,3,2,1,2,3,4,5]
        dp array holding left maxes and right maxes
        '''
        #first find max values to the left
        N = len(height)
        max_lefts = [0]*N
        max_rights =[0]*N
        
        #when finding max lefts we start from the beginning of he array
        for i in range(1,N):
            max_lefts[i] = max(max_lefts[i-1],height[i-1])
        
        #max to the rights, we start at tned
        for i in range(N-2,-1,-1):
            max_rights[i] = max(max_rights[i+1],height[i+1])
        
        #now we just take the min of left and right at each increment by height
        ans = 0
        for i in range(N):
            water_level = min(max_lefts[i],max_rights[i])
            if water_level > height[i]:
                ans += water_level - height[i]
        
        return ans
            