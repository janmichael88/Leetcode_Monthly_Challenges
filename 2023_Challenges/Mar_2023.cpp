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

/////////////////////////////////////
// 142. Linked List Cycle II (REVISTED)
// 09MAR23
//////////////////////////////////////
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    
    ListNode *getIntersect(ListNode *node){
        ListNode *fast = node;
        ListNode *slow = node;
        
        while (fast != NULL and fast->next != NULL){
            slow = slow->next;
            fast = fast->next->next;
            
            if (slow == fast){
                return slow;
            }
        }
        
        return NULL;
    }
    ListNode *detectCycle(ListNode *head) {
        if (head == NULL){
            return NULL;
        }
        
        ListNode *intersect = getIntersect(head);
        if (intersect == NULL){
            return NULL;
        }
        
        ListNode *p1 = head;
        ListNode *p2 = intersect;
        
        while (p1 != p2){
            p1 = p1->next;
            p2 = p2->next;
        }
        
        return p1;
    }
};

//////////////////////////////////////
// 11MAR23
// 382. Linked List Random Node (REVISTED)
//////////////////////////////////////
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    //outside scope of solution
    //hashamp, of int valure to the acutal node
    unordered_map<int,ListNode*> nums;
    int i = 0;
    
    Solution(ListNode* head) {
        //naive wawy would be to pull ints and pull random index
        
        ListNode* temp = head;
        
        while (temp){
            nums[i] = temp;
            temp = temp->next;
            i += 1;
        }
    }
    
    int getRandom() {
        return nums[rand() % i]->val;
        
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(head);
 * int param_1 = obj->getRandom();
 */

/////////////////////////////////////////
// 382. Linked List Random Node (REVISITED)
// 11MAR23
/////////////////////////////////////////
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    //we cab use reservour sampling using size = 1
    //then we just choost to replace the sample with probbailty 1 / curr_size
    /*
    pseudo code for resevoior sampling
    # S has items to sample, R will contain the result
def ReservoirSample(S[1..n], R[1..k])
  # fill the reservoir array
  for i := 1 to k
      R[i] := S[i]

  # replace elements with gradually decreasing probability
  for i := k+1 to n
    # randomInteger(a, b) generates a uniform integer
    #   from the inclusive range {a, ..., b} *)
    j := randomInteger(1, i)
    if j <= k
        R[j] := S[i]

    */
    ListNode *HeadNode;
    Solution(ListNode* head) {
        HeadNode = head;
    }
    
    int getRandom() {
        ListNode* curr = HeadNode;
        int res;
        int size = 1;
        
        while (curr){
            float prob = (float) rand() / RAND_MAX;
            if (prob < (float) 1 / size){
                res = curr->val;
            }
            size += 1;
            curr = curr->next;
        }
        return res;
    }
};

/**
 * Your Solution object will be instantiated and called as such:
 * Solution* obj = new Solution(head);
 * int param_1 = obj->getRandom();
 */

//another way, but if the random number == size, just replace it
class Solution {
public:
    //Note : head is guaranteed to be not null, so it contains at least one node.
    ListNode* HeadNode;
    Solution(ListNode* head) {
       HeadNode = head;
    }
    //returns value of a random node
    int getRandom() {
        int res, len = 1;
        ListNode* x = HeadNode;
        while(x){
            if(rand() % len == 0){
                res = x->val;
            }
            len++;
            x = x->next;
        }
        return res;
    }
};

/////////////////////////////////////////////////////////////
// 109. Convert Sorted List to Binary Search Tree (REVISTED)
// 11MAR23
/////////////////////////////////////////////////////////////

//unpack into array and salve the problem on the array
 /**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    //easiest way would be dump elements into a list
    //then rebuild recursively
    //these are just variable defintions, looks like i can't do anything inside the constcutor
    vector<int> nums;
    TreeNode* temp = new TreeNode();
    
    //recursively build from nums
    TreeNode* buildTree(vector<int>& nums,int left,int right){
        if (left > right){
            return nullptr;
        }
        int mid = left + (right - left) / 2;
        TreeNode* curr_node = new TreeNode();
        curr_node->left = buildTree(nums,left,mid-1);
        curr_node->val = nums[mid];
        curr_node->right = buildTree(nums,mid+1,right);
        return curr_node;

    }

    
    TreeNode* sortedListToBST(ListNode* head) {
        ListNode* curr = head;
        while (curr){
            nums.push_back(curr->val);
            curr = curr->next;
        }
        
        int left = 0;
        int right = nums.size();
        
        TreeNode* ans = buildTree(nums,left,right-1);
        
        
     

        return ans;

    }
};

//find middle each time
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
    /*
    find middle of linked list 
    */
public:
    //notes on using new
    //https://stackoverflow.com/questions/679571/when-to-use-new-and-when-not-to-in-c
    //when recuring and building new objects, or if i need them, just call new
    //TreeNode* temp =  new TreeNode();
    TreeNode* sortedListToBST(ListNode* head) {
        //empty head
        if (head == nullptr){
            return nullptr;
        }
        
        ListNode* mid = getMiddle(head);
        
        //make new TreeNode
        TreeNode* node =  new TreeNode();
        node->val = mid->val;
        
        //base case, when mid is just the head
        if (mid == head){
            return node;
        }
        
        node->left = sortedListToBST(head);
        node->right = sortedListToBST(mid->next);
        return node;
    }
    
    
    ListNode* getMiddle(ListNode* head){
        ListNode* prev;
        ListNode* slow = head;
        ListNode* fast = head;
        
        while (fast != nullptr && fast->next != nullptr){
            prev = slow;
            slow = slow->next;
            fast = fast->next->next;
            
        }
        
        //handling case when slow == head
        if (prev != nullptr){
            prev->next = nullptr;
        }
        
        return slow;
    }
    
    
};

////////////////////////////////////////////
// 23. Merge k Sorted Lists
// 12MAR23
////////////////////////////////////////////
//using max heap
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    /*
    in C++ i need to use a custom comparator if i wanted a pq of vectors
    make a struct, also PQ in C++ requires three arugments
    */
    
    struct my_comparator
    {
        // queue elements are vectors so we need to compare those
        bool operator()(vector<int>& a, vector<int>& b) 
        {

            return a[0] > b[0];
        }
    };


    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<vector<int>, vector<vector<int>>, my_comparator> pq;
        int N = lists.size();
        
        ListNode* dummy = new ListNode(-1);
        ListNode* curr = dummy;
        
        for (int i = 0; i < N; ++i){
            if (lists[i] != nullptr){
                vector<int> entry = {lists[i]->val,i};
                pq.push(entry);
            }
        }
        
        while (!pq.empty()){
            //remove entry
            vector<int> entry = pq.top();
            pq.pop();
            
            //make new node
            ListNode* node = new ListNode(entry[0]);
            curr->next = node;
            curr = curr->next;
            
            //update
            if (lists[entry[1]]->next != nullptr){
                lists[entry[1]] = lists[entry[1]]->next;
                vector<int> entry2 =   {lists[entry[1]]->val,entry[1]};
                pq.push(entry2);
            }
        }
        
        return dummy->next;
        
    }
};

//merge one by one
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    
    //merger functino
    ListNode* mergeLists(ListNode* a, ListNode* b){
        ListNode* dummy = new ListNode();
        ListNode* curr = dummy;
        
        while (a != nullptr and b != nullptr){
            if (a->val < b->val){
                ListNode* node = new ListNode(a->val);
                curr->next = node;
                a = a->next;
            }
            else{
                ListNode* node = new ListNode(b->val);
                curr->next = node;
                b = b->next;
            }
            curr = curr->next;
        }
        
        if (a != nullptr){
            curr->next = a;
            
        }
        
        if (b != nullptr){
            curr->next = b;
        }
        
        return dummy->next;
    }
    
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.size() == 0){
            return nullptr;
        }
        
        if (lists.size() == 1){
            return lists[0];
        }
        //merge
        ListNode* res = lists[0];
        
        for (int i = 1; i < lists.size(); ++i){
            res = mergeLists(res,lists[i]);
        }
        
        return res;
    }
};

///////////////////////////////////////
// 605. Can Place Flowers (REVISTED)
// 20MAR23
////////////////////////////////////////
class Solution {
public:
    bool canPlaceFlowers(vector<int>& flowerbed, int n) {
        //just check eveyrthing in between, careful for the edge cases on the beginning and at the end
        int count = 0;
        for (int i = 0; i < flowerbed.size(); ++i){
            //variables for left and right
            if (flowerbed[i] == 0){
                //if at the beginning, its left is trivially empty
                //same thing with the end
                bool empty_left = (i == 0) || (flowerbed[i-1] == 0);
                //i can index out of bounrds if the previous statement evals to something
                bool empty_right = (i == flowerbed.size() - 1) || (flowerbed[i+1] == 0);
                
                if (empty_left && empty_right){
                    count += 1;
                    flowerbed[i] = 1;
                }
            }
        }
        
        return count >= n;
    }
};


/////////////////////////////////////
// 1236. Web Crawler
// 23MAR23
//////////////////////////////////////
//you can next functions in C++ in the form of lambdas
/**
 * // This is the HtmlParser's API interface.
 * // You should not implement it, or speculate about its implementation
 * class HtmlParser {
 *   public:
 *     vector<string> getUrls(string url);
 * };
 */

class Solution {
public:
    vector<string> crawl(string startUrl, HtmlParser htmlParser) {
        function<string(string)> getHost = [](string url){
            int pos = min(url.size(),url.find('/',7));
            
            return url.substr(7,pos-7);
        };
        
        string hostName = getHost(startUrl);
        //keep visited set 
        unordered_set<string> seen;
        
        function<void(string)> dfs = [&](string url){
            seen.insert(url);
            for (string neigh : htmlParser.getUrls(url)){
                if (getHost(neigh) == hostName && !seen.count(neigh)){
                    dfs(neigh);
                }
            }
        };
        
        dfs(startUrl);
        vector<string> ans;
        for (string url: seen){
            ans.push_back(url);
        }
        
        return ans;
    }
    
};

////////////////////////////////////////////////////////
// 1319. Number of Operations to Make Network Connected
// 23MAR23
////////////////////////////////////////////////////////
class Solution {
public:
    void dfs(int node, vector<vector<int>>& adj, vector<bool>& visit) {
        visit[node] = true;
        for (int neighbor : adj[node]) {
            if (!visit[neighbor]) {
                dfs(neighbor, adj, visit);
            }
        }
    }

    int makeConnected(int n, vector<vector<int>>& connections) {
        if (connections.size() < n - 1) {
            return -1;
        }

        vector<vector<int>> adj(n);
        for (auto& connection : connections) {
            adj[connection[0]].push_back(connection[1]);
            adj[connection[1]].push_back(connection[0]);
        }

        int numberOfConnectedComponents = 0;
        vector<bool> visit(n);
        for (int i = 0; i < n; i++) {
            if (!visit[i]) {
                numberOfConnectedComponents++;
                dfs(i, adj, visit);
            }
        }

        return numberOfConnectedComponents - 1;
    }
};