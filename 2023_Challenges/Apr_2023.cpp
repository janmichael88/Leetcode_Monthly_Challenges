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
