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