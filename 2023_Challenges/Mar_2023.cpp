class Solution {
public:
    long long countSubarrays(vector<int>& nums, int minK, int maxK) {
        //we first find interals [j,i) that conatain intervals in the rang [mink,maxK]
        //then for each j, track the closest positions (p1) for minK and (p2) for maxK
        //the number of valid subarrays from j to i-1 is = i - max(p1,p2)
        
        long long res = 0, n = nums.size();
        //find subarrays meeting requirement
        for (int i = 0,j=0; i <= n; ++i){
            //int temp[2] = {i,j};
            /* print an array
            std::copy(std::begin(temp),
          std::end(temp),
          std::ostream_iterator<int>(std::cout, "\n"));
          */
            //end of the array or outside the range
            if (i == n || nums[i] < minK || nums[i] > maxK){
                //find the most recent index
                for (int p1 = j, p2 = j; j <= i; ++j){
                    while (p1 < i && (p1 < j || nums[p1] != minK)){
                        ++p1;
                    }
                    while (p2 < i && (p2 < j || nums[p2] != maxK)){
                        ++p2;
                    }
                    res += i - max(p1,p2);
                }
            }
        }

        return res;
        
    }
};
///////////////////////////////////////////////////
// 1539. Kth Missing Positive Number (REVISTED)
// 06MAR23
////////////////////////////////////////////////////
class Solution {
public:
    int findKthPositive(vector<int>& arr, int k) {
        /*
        we can count the number of missing in bettwen for any i as arr[i+1] - arr[i] - 1
        first check if k is less than the first number
        */
        if (k <= arr[0] - 1){
            return k;
        }
        //decrement k
        k -= arr[0] - 1;
        
        int N = arr.size();
        
        for (int i = 0; i < N-1; ++i){
            //get missing in between
            int missing = arr[i+1] - arr[i] - 1;
            //if kth missing is beetween, return k past it
            if (k <= missing){
                return arr[i] + k;
            }
            k -= missing;
        }
        
        //missing is greater than the last one
        return arr[N-1] + k;
            
    }
};

class Solution {
public:
    int findKthPositive(vector<int>& arr, int k) {
        //from the hint we can track the number of positives
        unordered_set<int> nums(arr.begin(), arr.end());
        int N = arr.size();
        for (int missing = 1; missing <= arr.size() + k; ++missing ){
            if (nums.count(missing) == 0 ){
                k -= 1;
            }
            if (k == 0){
                return missing;
            }
        }
        return arr[N-1] +  k;
    }
};

//binary search
class Solution {
public:
    int findKthPositive(vector<int>& arr, int k) {
        /*
        trick;
            say our array is [2,3,4,7,11]
            if we take the increaing array [1,2,3,4,5]
            the number of missing elements up to index i would be
            arr[i] - nums[i]
            rather, the number of misisng owuld be: arr[i] - (i + 1), which is just arr[i] - i - 1
        
        using binary search:
            choose pivot as middle of the array
            if the number of positive intergers which are missing before arr[pivot] is less than k, continue to search on the right
            otherwsie search on the left side
            
        Initialize search boundaries: left = 0, right = arr.length - 1.

        While left <= right:

        Choose the pivot index in the middle: pivot = left + (right - left) / 2. Note that in Java we couldn't use straightforward pivot = (left + right) / 2 to avoid the possible overflow. In Python, the integers are not limited, and we're fine to do that.

        If the number of positive integers which are missing before is less than k arr[pivot] - pivot - 1 < k - continue to search on the right side of the array: left = pivot + 1.

        Otherwise, continue to search on the left: right = pivot - 1.

        At the end of the loop, left = right + 1, and the kth missing number is in-between arr[right] and arr[left]. The number of integers missing before arr[right] is arr[right] - right - 1. Hence, the number to return is arr[right] + k - (arr[right] - right - 1) = k + left.
        
        really its:
            num in between (arr[right+1] and arr[right])
            arr[right+1] - (right + 1) - 1 - (arr[right] - right - 1)
            arr[right+1] - right - arr[right] + right + 1
        */
        
        int left = 0;
        int right = arr.size() - 1;
        
        while (left <= right){
            int pivot = left + (right - left) / 2;
            //if number of positive intergers mising are less than k, then we are good on the left side
            //other search leaf
            if (arr[pivot] - pivot - 1 < k){
                left = pivot + 1;
            } 
            else{
                right = pivot - 1;
            }
        }
        
        return left + k; //or just k after the lower bound
    }
};

///////////////////////////////////////////////
// 812. Largest Triangle Area
// 07MAR23
///////////////////////////////////////////////
#include <algorithm>
class Solution {
public:
    double largestTriangleArea(vector<vector<int>>& points) {
        //just try all points?
        //points i,j,i
        //i can get the area of a triangle using 1/2 cross product of v1 and v2
        float ans = INT_MIN;
        int N = points.size();
        for (int i = 0; i < N; ++i){
            for (int j = 0; j < N; ++j){
                for (int k = 0; k < N; ++k){
                    if (i != j and i != k and j != k){
                        //do ij and ik as vectors
                        int ij[] = {points[j][0] - points[i][0], points[j][1] - points[i][1]};
                        int ik[] = {points[k][0] - points[i][0], points[k][1] - points[i][1]};
                        float area = ij[0]*ik[1] - ik[0]*ij[1];
                        area = area / 2.0;
                        ans = max(ans,area);
                    }
                }
            }
        }
        
        return ans;
    }
};

////////////////////////////////////////
// 2187. Minimum Time to Complete Trips
// 07MAR23
////////////////////////////////////////
class Solution {
public:
    
    //functino definition outside, must be in under public
    long long get_time (vector<int>& time, long long test_time){
            long long num_trips = 0;
            for (auto t : time){
                num_trips += test_time / t;
            }
            return num_trips;
        }
    
    long long minimumTime(vector<int>& time, int totalTrips) {
        //for a unit of time t, each bus i, can make time[i] // t trips
        //if we a given a unit of time, we can calculate how many total trips we can make
        //if we start with a time k, and we can make get all these stops, then we know anything larger than k will work too
        //so we don't need to look on that side
        //binary search
        //min should be 0, and max should be, the slowest bus, time number of trips
        
        long long start = 0;
        long long end = 1LL * *min(time.begin(),time.end())*totalTrips;
        
        
        while (start < end){
            long long mid = start + (end - start) / 2;
            if (get_time(time,mid) >= totalTrips){
                end = mid;
            }
            else{
                start = mid + 1;
            }
        }

        return start;
    }
};

///////////////////////////////////////////
// 875. Koko Eating Bananas (REVISTED)
// 08MAR23
///////////////////////////////////////////
class Solution {
public:
    
    int computeHours(vector<int>& piles, int k){
        int totalHours = 0;
        for (int b: piles){
            if (b < k){
                totalHours += 1;
            }
            else{
                totalHours += (b / k) + 1;
            }
        }
        
        return totalHours;
    }
    int minEatingSpeed(vector<int>& piles, int h) {
        /*
        koko can choose her k, where she eats k banas per hour
        if the pile is < k, she eats all them and doesn't eat anymore
        we can try all possible k
            if koko can eat all possible bannans with the current k, then it would mean anything bigger than k works as well
            so we can stop seaching at afterk
        what would be the min and max bounds for the search?
            min of bananes if left bound
            
        */
        int left = 1;
        int right = *max(piles.begin(),piles.end());
        
        while (left < right){
            int mid = left + (right - left) / 2;
            if (computeHours(piles,mid) <= h){
                right = mid;
            }
            else{
                left = mid + 1;
            }
        }
        
        return right;
    }
};