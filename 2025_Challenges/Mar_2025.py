#########################################
#2460. Apply Operations to an Array
# 02MAR25
#######################################
class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        '''
        just follow the rules and apply on nums
        then move zeros to the end
        '''
        n = len(nums)
        ans = []
        for i in range(n-1):
            if nums[i] == nums[i+1]:
                nums[i] *= 2
                nums[i+1] = 0
            else:
                continue
        
        for num in nums:
            if num:
                ans.append(num)
        
        return ans + [0]*(n-len(ans))

#inplace two pass
class Solution:
    def applyOperations(self, nums: List[int]) -> List[int]:
        '''
        in place two pass
        first modify the array
        '''
        n = len(nums)
        for i in range(n-1):
            if nums[i] == nums[i+1]:
                nums[i] *= 2
                nums[i+1] = 0
        
        place_idx = 0
        for num in nums:
            if num:
                nums[place_idx] = num
                place_idx += 1
        #the rest are zeros
        while place_idx < len(nums):
            nums[place_idx] = 0
            place_idx += 1
        
        return nums
    
##############################################
# 2570. Merge Two 2D Arrays by Summing Values
# 02MAR25
###############################################
class Solution:
    def mergeArrays(self, nums1: List[List[int]], nums2: List[List[int]]) -> List[List[int]]:
        '''
        hashmap problem, how to return in sorted order after merging?
        the arrays are already in ascneding order,
        advance the smaller of the two, merge part of merge sort
        '''
        ans = []
        i,j = 0,0
        while i < len(nums1) and j < len(nums2):
            if nums1[i][0] < nums2[j][0]:
                ans.append(nums1[i])
                i += 1
            elif nums2[j][0] < nums1[i][0]:
                ans.append(nums2[j])
                j += 1
            #equal id
            else:
                entry = []
                entry.append(nums1[i][0])
                entry.append(nums1[i][1] + nums2[j][1])
                ans.append(entry)
                i += 1
                j += 1
        while i < len(nums1):
            ans.append(nums1[i])
            i += 1
        while j < len(nums2):
            ans.append(nums2[j])
            j += 1
        
        return ans
    
#############################################################
# 2161. Partition Array According to Given Pivot (REVISITED)
# 03MAR25
#############################################################
class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        '''
        im sure there is an in place algo, but we can use quick sort partition scheme
        three arrays
            lesss,than,equal, everything else
        '''
        less_than = []
        equal_to = []
        remaining = []

        for num in nums:
            if num < pivot:
                less_than.append(num)
            elif num == pivot:
                equal_to.append(num)
            else:
                remaining.append(num)
        
        return less_than + equal_to + remaining