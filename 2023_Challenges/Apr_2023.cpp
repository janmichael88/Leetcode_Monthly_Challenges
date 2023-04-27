/////////////////////////////////////////
// 2034. Stock Price Fluctuation
// 04APR23
//////////////////////////////////////////
class StockPrice {
    int latestTime;
    // Store price of each stock at each timestamp.
    unordered_map<int, int> timestampPriceMap;
    // Store stock prices in increasing order to get min and max price.
    map<int, int> priceFrequency;
    
public:
    StockPrice() {
        latestTime = 0;
    }
    
    void update(int timestamp, int price) {
        // Update latestTime to latest timestamp.
        latestTime = max(latestTime, timestamp);
        
        // If same timestamp occurs again, previous price was wrong. 
        if (timestampPriceMap.find(timestamp) != timestampPriceMap.end()) {
            // Remove previous price.
            int oldPrice = timestampPriceMap[timestamp];
            priceFrequency[oldPrice]--;
            
            // Remove the entry from the map.
            if (priceFrequency[oldPrice] == 0) {
                priceFrequency.erase(oldPrice);
            }
        }
        
        // Add latest price for timestamp.
        timestampPriceMap[timestamp] = price;
        priceFrequency[price]++;
    
    }
    
    int current() {
        // Return latest price of the stock.
        return timestampPriceMap[latestTime];
    }
    
    int maximum() {
        // Return the maximum price stored at the end of sorted-map.
        return priceFrequency.rbegin()->first;
    }
    
    int minimum() {
        // Return the maximum price stored at the front of sorted-map.
        return priceFrequency.begin()->first;
    }
};

class StockPrice {
    int latestTime;
    // Store price of each stock at each timestamp.
    unordered_map<int, int> timestampPriceMap;
    // Store stock prices in sorted order to get min and max price.
    // difference in type delcaration
    priority_queue<pair<int, int>> maxHeap;
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int,int>>> minHeap;
    
public:
    StockPrice() {
        latestTime = 0;
    }
    
    void update(int timestamp, int price) {
        // Update latestTime to latest timestamp.
        latestTime = max(latestTime, timestamp);
        
        // Add latest price for timestamp.
        timestampPriceMap[timestamp] = price;
        minHeap.push({ price, timestamp });
        maxHeap.push({ price, timestamp });
    }
    
    int current() {
        // Return latest price of the stock.
        return timestampPriceMap[latestTime];
    }
    
    int maximum() {
        pair<int, int> top = maxHeap.top();
        // Pop pairs from heap with the price doesn't match with hashmap.
        while (timestampPriceMap[top.second] != top.first) {
            maxHeap.pop();
            top = maxHeap.top();
        }
        
        return top.first;
    }
    
    int minimum() {
        pair<int, int> top = minHeap.top();
        // Pop pairs from heap with the price doesn't match with hashmap.
        while (timestampPriceMap[top.second] != top.first) {
            minHeap.pop();
            top = minHeap.top();
        }
        
        return top.first;
    }
};

///////////////////////////////////////////
// 1146. Snapshot Array
// 06APR23
////////////////////////////////////////////
class SnapshotArray {
    vector<vector<pair<int,int>>> mapp;
    int curr_snap = 0;
public:
    SnapshotArray(int length) {
        mapp = vector<vector<pair<int,int>>>(length);
    }
    
    void set(int index, int val) {
        //empty container, or most snap !== currnsap
        if (mapp[index].empty() || mapp[index].back().first != curr_snap){
            mapp[index].push_back({curr_snap,val});
        }
        else{
            mapp[index].back().second = val;
        }
    }
    
    int snap() {
        
        return curr_snap++;
        
    }
    
    int get(int index, int snap_id) {
        //we don't always snap, and so if we are trying to retrieve an index for a snap we don't have, we just get the most recent
        /*
        we trying to find the snap that goes after snap_id. This is because we may or may not have the exact snap_id for a given index. Then we go one step backwards to get the most recent value before the snap that goes after snap_id.

Let's say we have snaps 3, 5, 7, 10 and we are looking for snap 8. upper_bound will point to snap 10. The most recent value before snap 10 is the value for snap 7.
Now, say we are looking for snap 7. upper_bound will also point to snap 10. And the most recent value before snap 10 is the value for snap 7 as well.
What if we are looking for snap 12? upper_bound will point to end, and prev(end) is the value for snap 10.
Finally, if we are looking for snap 2, upper_bound will return the position of snap 3. Now, since the position of snap 3 is the first one in the array (begin), that means that we did not have value for snap 2 and we need to return 0.
        */
        auto idx = upper_bound(begin(mapp[index]), end(mapp[index]), pair<int,int>(snap_id,INT_MAX));
        if (idx == begin(mapp[index])){
            return 0;
            
        }
        else{
            return prev(idx)->second; //this is a pointer
        }
            
    }
};

/**
 * Your SnapshotArray object will be instantiated and called as such:
 * SnapshotArray* obj = new SnapshotArray(length);
 * obj->set(index,val);
 * int param_2 = obj->snap();
 * int param_3 = obj->get(index,snap_id);
 */

///////////////////////////////////////
// 883. Projection Area of 3D Shapes
// 06APR23
///////////////////////////////////////
//from the python version we can reduce the number of loops
class Solution {
public:
    int projectionArea(vector<vector<int>>& grid) {
        int N = grid.size();
        
        int ans = 0;
        
        for (int i = 0; i < N; ++i){
            int largest_row = 0;
            int largest_col = 0;
            
            for (int j = 0; j < N; ++j){
                //xy
                if (grid[i][j] > 0){
                    ans++;
                }
                //other two sides
                largest_row = max(largest_row,grid[i][j]);
                largest_col = max(largest_col,grid[j][i]);
            }
            
            ans += largest_row + largest_col;
        }
        return ans;
    }
};

///////////////////////////////////////// 
// 888. Fair Candy Swap
// 13APR23
////////////////////////////////////////
class Solution {
public:
    vector<int> fairCandySwap(vector<int>& aliceSizes, vector<int>& bobSizes) {
        //use the equation directly and solve
        //use hashset to check for complement in constantime
        
        int S_A = 0;
        int S_B = 0;
        
        for (int num : aliceSizes){
            S_A += num;
        }
        
        for (int num: bobSizes){
            S_B += num;
        }
        
        int delta = (S_A - S_B) / 2;
        //hashbob
        set<int> setB;
        for (int num: bobSizes){
            setB.insert(num);
        
        }
        
        for (int num: aliceSizes){
            bool contains = setB.find(num + delta) != setB.end();
            if (contains == true){
                return {num,num + delta};
            }
        }
        
        return {0,0};
    }
};

///////////////////////////////////
// 727. Minimum Window Subsequence
// 26APR23
//////////////////////////////////
class Solution {
    /*
        greedy
        intution, we can try all values of start, and for each of them find the smallest end value such that s2 is a subsequence
        of s1[start:end]
        if m == len(s2), we want to find a sequence of indices in s1, i_0,i_1,..i_{m-1}
        such that start - 1 < i_0 < i_1 < ... i_{m-1}
        and s1[i_0] == s2[0], s1[i_1] == s2[1].....s1[i_{m-1}] == s2[m-1]
        
        we can greedily find the indices one bye one
        find the first index i_0, that has property start -1 < i_0
        then find i_1, where i_0 < i_1....and so on
        
        preprocess:
        for each char in s1, keep motonic array of indices
        example: s1 = "abcdebdde", indices['d'] = [3, 6, 7], because s1[3] = s1[6] = s1[7] = 'd'.
        
        finding the smallest i_0 > start such that s1[i_0] = s2[0] is the same as finding the smallest element >= start in the array
        indices[s2[0]]
        
        let ind_0 denote the index of the element at i_0 in s1
        indices[s2[0]][ind_0] = i_0
        
        since indices[s2[0]] is increasing, we can also say indices[s2[0]][ind_0] > start - 1
        
        similarily we find the smallest i_1 > i_0 in the array indices[s2[1]] by finding the smallest ind1 such that indices[s2[0]][ind_1] > i_0
        
        same for i_2....i_{m-1}
        
        Algorithm
        1. Let n be the length of s1 and m be the length of s2.
        2. Initialize the answer with an empty string.
        3. For each letter c, find all its occurrences in the string s1 and add them to indices[c].
        4. Initialize an array ind of size m with zeros.
        5. Iterate start from 0 to n - 1.
            Initialize prev with start - 1.
            Iterate j from 0 to m - 1. This variable is iterating over the characters of s2.
                Let curIndices be indices[s2[j]]. These are all the indices where the current character in s2 appears in s1.
                While ind[j] < len(curIndices) and curIndices[ind[j]] <= prev
                    Increment ind[j].
                If ind[j] = len(curIndices) (we could not find an element greater than last), there is no valid window starting at start, we can immediately return the answer, since all future values of start will be greater and there will also be no valid window.
                Set prev to curIndices[ind[j]].
            At this point, prev = end. Our candidate string is s1[start..prev]. If answer is empty or this candidate has a shorter length, update answer with the candidate.
        6. Return the answer
    */
public:
    string minWindow(string s1, string s2) {
        int n = s1.size();
        int m = s2.size();
        
        string ans = "";
        
        //char tp indx map
        unordered_map<char, vector<int>> chars_to_idx;
        
        for (int i = 0; i < n; i++){
            chars_to_idx[s1[i]].push_back(i);
        }
        vector<int> indices(m);
        
        //try all starts
        for (int start = 0; start < n; start++){
            int prev = start - 1;
            for (int j = 0; j < m; j++){
                if (!chars_to_idx.count(s2[j])){
                    return "";
                }
                //reference but just lock it
                const vector<int>& curr_indices = chars_to_idx[s2[j]];
                //try to find the next i_0
                while (indices[j] < curr_indices.size() && curr_indices[indices[j]] <= prev){
                    indices[j]++;
                }
                
                //if we have gotten to the end of all the indices for this current char, we can do no better than the current answer
                if (indices[j] == curr_indices.size()){
                    return ans;
                }
                
                prev = curr_indices[indices[j]];
                
            }
            
            if (ans == "" || prev - start + 1 < ans.size()){
                ans = s1.substr(start,prev-start+1);
            }
        }
        return ans;
    }
};