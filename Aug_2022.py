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

############################
# 427. Construct Quad Tree
# 03AUG22
############################
"""
# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""

class Solution:
    def construct(self, grid: List[List[int]]) -> 'Node':
        '''
        a quad tree is a tree struture where each inteal node has four children
        each node has two attributes
            val: True if the node representas a grid of 1's or False if node represents a grid of 0s
            isLeaf: True if the node is a leaf node or False if the nodes had four children
            
        algo:
            if the current grid has the same values (all 1's or all 0's) set isLeaf to True and val to to be the value of the grid and set children to None
            if the current grid has different values, set isLeaf to false and set val to any value and divide the grid into four
            recurse on the four
            
        Start with the full grid and keep on diving it four parts.
        Once we have a grid of size 1 then start with merging.
        Compare if all the leaf nodes,
        if yes then merge all into a single node.
        else, return all four nodes separately.
        
        each call will define the grid as the top left corner
        '''
        
        def build(row,col,grid_length):
            #base case, single element
            if grid_length == 1:
                val = grid[row][col]
                node = Node(val,True,None,None,None,None)
                return node
            
            #otherwise recurse, dividing them
            #represent each grid defined by coordinate in top left
            topLeft = build(row,col,grid_length //2)
            topRight = build(row,col + grid_length // 2, grid_length//2)
            bottomLeft = build(row + grid_length // 2, col, grid_length//2)
            bottomRight = build(row + grid_length //2, col + grid_length //2, grid_length//2)
            
            #check all leaves
            if topLeft.isLeaf == topRight.isLeaf == bottomLeft.isLeaf == bottomRight.isLeaf == True:
                #check all valuess are the same
                if topLeft.val == topRight.val == bottomLeft.val == bottomRight.val:
                    #build
                    node = Node(topLeft.val,True,None,None,None,None)
                    return node
            
            #make false node
            node = Node(-1,False,topLeft,topRight,bottomLeft,bottomRight)
            return node
        
        return build(0,0,len(grid))

#####################
# 729. My Calendar I
# 03AUG22
#####################
class MyCalendar:

    def __init__(self):
        '''
        we can use binary search to search for the position in the array who's start is just smaller then the start we want to add
        then we must check that this end is works
        '''
        self.intervals = []
    def book(self, start: int, end: int) -> bool:
        i = self.binarySearch(start)
        #i is where we want to insert
        #check the one previous to it and make sure there is no overlap
        if i > 0 and self.intervals[i-1][1] > start:
            return False
        #check starts
        if i < len(self.intervals) and end > self.intervals[i][0]:
            return False
        self.intervals.insert(i,[start,end])
        return True
        
    def binarySearch(self,target):
        left = 0
        right = len(self.intervals)
        while left < right:
            mid = left + (right - left) //2
            if self.intervals[mid][0] >= target:
                right = mid
            else:
                left = mid + 1
        return left


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)

'''
using sorted container
keep track all 2n points
then binary search twice
1. make sure that q1 == q2, which is the point we need to insert start and end
2. also check for q1 % 2 == 0
    i.e, it lies at an even index, because we alaways need to insert two pointts
    i the index were odd, it means we over lap on an end
'''
from sortedcontainers import SortedList

class MyCalendar:
    def __init__(self):
        self.arr = SortedList()
        
    def book(self, start, end):
        #we need to find the index that is just greater than start, and just smaller than end
        q1 = SortedList.bisect_right(self.arr, start)
        q2 = SortedList.bisect_left(self.arr, end)
        if q1 == q2 and q1 % 2 == 0:
            self.arr.add(start)
            self.arr.add(end)
            return True
        return False

# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)


#using balanced binary tree
'''
we can use a self balacing binary tree, i don't think you'd be expected
and build the insertion method recursively
'''

class Node:
    def __init__(self,start,end):
        self.start = start
        self.end = end
        self.left = None
        self.right = None
    
    def insert(self,start,end):
        #if the end we want to insert is bigger then our start
        if self.start >= end:
            #put to its left
            if self.left is None:
                self.left = Node(start,end)
                return True
            else:
                return self.left.insert(start,end)
        elif self.end <= start:
            if self.right is None:
                self.right = Node(start,end)
                return True
            else:
                return self.right.insert(start,end)
        else:
            return False

class MyCalendar:

    def __init__(self):
        self.root = None
        

    def book(self, start: int, end: int) -> bool:
        if not self.root:
            self.root = Node(start,end)
            return True
        else:
            return self.root.insert(start,end)
        


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)

###########################
# 
#
###########################
