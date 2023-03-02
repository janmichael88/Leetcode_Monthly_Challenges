####################################
# 912. Sort an Array (REVISTED) 
# 01MAR23
###################################
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''
        merge sort using extra space
        '''
        return self.merge_sort(nums)
    
    def merge(self, left_list: List[int], right_list: List[int]) -> List[int]:
        merged = []
        
        i,j = 0,0
        while i < len(left_list) and j < len(right_list):
            if left_list[i] < right_list[j]:
                merged.append(left_list[i])
                i += 1
            else:
                merged.append(right_list[j])
                j += 1
        
        #the rest
        while i < len(left_list):
            merged.append(left_list[i])
            i += 1
        
        while j < len(right_list):
            merged.append(right_list[j])
            j += 1
        
        #could also have done
        #merged.extend(left_list[i:])
        #merged.extend(right_list[j:])
        
        return merged
        
    def merge_sort(self, array : List[int]) -> List[int]:
        if len(array) == 1:
            #single element, just return the array
            return array
        
        mid = len(array) // 2
        left = self.merge_sort(array[0:mid])
        right = self.merge_sort(array[mid:])
        return self.merge(left,right)


#inplace merge sort
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        '''
        if we want in place merger sort, we need to use a temp array of len(nums)
        then we ferry elements from nums into temp
        then merge back into nums in sorted order
        '''
        self.temp = [0]*len(nums)
        self.nums = nums

        #invoke
        self.merge_sort(0,len(nums)-1)
        return self.nums
    def merge_sort(self,left:int,right:int):
        if left >= right:
            return
        mid = left + (right - left) // 2
        self.merge_sort(left,mid)
        self.merge_sort(mid+1,right)
        self.merge(left,mid,right)
    
    def merge(self,left:int, mid:int, right:int):
        #find starts and ends
        #we need access to the indices back into nums
        
        start1 = left
        start2 = mid + 1
        end1 = mid - left + 1
        end2 = right - mid
        
        #copy elements for both halves into temp array
        for i in range(end1):
            self.temp[start1 + i] = self.nums[start1 + i]
        
        for i in range(end2):
            self.temp[start2 + i] = self.nums[start2 + i]
            
        #merge in sorted order back inot numes
        i,j,k = 0,0,left #k is the insert posittion where we start at this current recursive call
        while i < end1 and j < end2:
            if self.temp[start1 + i] < self.temp[start2 + j]:
                self.nums[k] = self.temp[start1 + i]
                i += 1
            else:
                self.nums[k] = self.temp[start2 + j]
                j += 1
            
            k += 1
        
        #the rest
        while i < end1:
            self.nums[k] = self.temp[start1 + i]
            i += 1
            k += 1
        
        while j < end2:
            self.nums[k] = self.temp[start2 + j]
            j += 1
            k += 1
        

###################################
# 443. String Compression (REVISTED)
# 02MAR23
###################################
class Solution:
    def compress(self, chars: List[str]) -> int:
        '''
        i can use two pointers with a sliding window, once i'm done with the window just over write them with another pointer
        it wont matter anyway since we adavnce pointers
        '''
        N = len(chars)
        ptr = 0 #pointer to modify array
        curr_count = 0 #count of curr char
        left,right = 0,0
        
        while right < N:
            #expand
            while right < N and chars[right] == chars[left]:
                right += 1
                curr_count += 1
            
            #i need to modify the input array now
            #print(chars[left],curr_count)
            #curr_count = 0
            #left = right
        
            if curr_count == 1:
                chars[ptr] = chars[left]
                ptr += 1
            else:
                #get the digits to be splace in the array
                chars[ptr] = chars[left]
                ptr += 1
                for digit in str(curr_count):
                    chars[ptr] = digit
                    ptr += 1
            
            curr_count = 0
            left = right
        
        return ptr
        
        