#!/usr/local/bin/python3

###############################################
# 744. Find Smallest Letter Greater Than Target
# 02AUG22
###############################################
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        '''
        just binary search and return the right boung
        '''
        index = bisect.bisect_right(letters,target)
        if index == len(letters):
            return letters[0]
        else:
            return letters[index]

#can also use modulo
class Solution(object):
    def nextGreatestLetter(self, letters, target):
        index = bisect.bisect(letters, target)
        return letters[index % len(letters)]


class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        '''
        just binary search and return the right boung
        '''
        
        # if the number is out of bound
        if target >= letters[-1] or target < letters[0]:
            return letters[0]
        
        
        left = 0
        right = len(letters) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            #we need to find the one just greater than the target
            if letters[mid] <= target:
                left = mid + 1
            else:
                right = mid - 1
        
        return letters[left % len(letters)]


###################################
# 760. Find Anagram Mappings
# 02AUG22
####################################
class Solution:
    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        '''
        for nums2, mapp nums to index, then retreive idnex for nums1
        '''
        nums2_mapp = {}
        for i,num in enumerate(nums2):
            nums2_mapp[num] = i
        
        ans = []
        for num in nums1:
            ans.append(nums2_mapp[num])
        
        return ans

############################
# 378. Kth Smallest Element in a Sorted Matrix (REVISTED)
# 02JUL22
#############################
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        '''
        really we can reframe the porblem as finding the Kth smallest element among N sorted lists
        think about finding the kth smallest in two sorted lists
            advance two pointers increasinly, then return the min of the two pointers
            
        IMPORTANT:
            say our matrix has 100 rows, and we want to find the kth smallest
            if the matrix is sorted increasingly left to right, then up to down
            then we would only need to look in the first 5 rows of the matrix
            O(min(rows,k)); overall time complexity would be
            O(min(rows,k)) + K*min(rows,K)
        
        algo:
            maintain min heap, where entry is (value,row,column)
            we extrat smallest then keep advancing along column
                note: we only advance along a row, if we can stay in the current row
        '''
        N = len(matrix)
        
        #minheap
        heap = []
        for r in range(min(N,k)):
            heap.append((matrix[r][0],r,0))
            
        #heapify
        heapq.heapify(heap)
        
        #keep extracing min and advancing along a row
        while k:
            value,row,col = heapq.heappop(heap)
            #if we can advance along row
            if col < N - 1:
                heapq.heappush(heap, (matrix[row][col+1],row,col+1))
                
            #use up k
            k -= 1
        
        return value

class Solution:
    
    def countLessEqual(self, matrix, mid, smaller, larger):
        
        count, n = 0, len(matrix)
        row, col = n - 1, 0
        
        while row >= 0 and col < n:
            if matrix[row][col] > mid:
               
                # As matrix[row][col] is bigger than the mid, let's keep track of the
                # smallest number greater than the mid
                larger = min(larger, matrix[row][col])
                row -= 1
                
            else:
                
                # As matrix[row][col] is less than or equal to the mid, let's keep track of the
                # biggest number less than or equal to the mid
                
                smaller = max(smaller, matrix[row][col])
                count += row + 1
                col += 1

        return count, smaller, larger
    
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        
        n = len(matrix)
        start, end = matrix[0][0], matrix[n - 1][n - 1]
        while start < end:
            mid = start + (end - start) / 2
            smaller, larger = (matrix[0][0], matrix[n - 1][n - 1])

            count, smaller, larger = self.countLessEqual(matrix, mid, smaller, larger)

            if count == k:
                return smaller
            if count < k:
                start = larger  # search higher
            else:
                end = smaller  # search lower

        return start