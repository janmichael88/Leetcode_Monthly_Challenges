//////////////////////////////////
//1065. Index Pairs of a String
//01MAY23
////////////////////////////////
struct TrieNode{
    bool is_end = false;
    map<char,TrieNode*> children;
};

struct Trie{
    TrieNode* root;
    
    Trie(){
        root = new TrieNode();
    }
    
    void insert(string word){
        TrieNode* curr = root;
        for (int i = 0; i < word.length(); ++i){
            //check if char is in the current children
            if (curr->children.count(word[i]) == 0){
                curr->children.insert(make_pair(word[i], new TrieNode()));
            }
            
            //otherwise its in there
            curr = curr->children[word[i]];
        }
        //mark end
        curr->is_end = true;
    }
    
    bool find(string word){
        TrieNode* curr = root;
        for (int i = 0; i < word.length(); ++i){
            if (curr->children.count(word[i]) == 0){
                return false;
            }
            //otherwise its in there
            curr = curr->children[word[i]];
        }
        
        return curr->is_end;
    }
    
    
};

class Solution {
public:
    vector<vector<int>> indexPairs(string text, vector<string>& words) {
        vector<vector<int>> ans;
        
        Trie trie;
        
        for (const string& word:words){
            trie.insert(word);
        }
        
        //try all prefixes
        for (int i = 0; i < text.size(); ++i){
            for (int j = i; j < text.size(); ++j){
                string check = text.substr(i,j-i+1);
                if (trie.find(check)){
                    ans.push_back({i,j});
                }
            }
        }
        return ans;
    }
};

//////////////////////////////////////////
// 1603. Design Parking System
// 29MAY23
//////////////////////////////////////////
class ParkingSystem {
public:
    
    // Parking limit for each type of car
    int bigLimit, mediumLimit, smallLimit;

    // Create an Array to store parked cars
    int* parkingArray;

    ParkingSystem(int big, int medium, int small) {
        
        // Parking limit for each type of car
        this->bigLimit = big;
        this->mediumLimit = medium;
        this->smallLimit = small;

        // Create an Array to store parked cars. 
        this->parkingArray = (int*)malloc((big + medium + small) * sizeof(int));
        for (int i = 0; i < big + medium + small; i++) {
            this->parkingArray[i] = -1;
        }
    }

    bool addCar(int carType) {

        // Depending on carType, store the limit for the type of car
        int limit = 0;
        if (carType == 1) {
            limit = this->bigLimit;
        } else if (carType == 2) {
            limit = this->mediumLimit;
        } else {
            limit = this->smallLimit;
        }

        // Traverse linearly through the array from the left
        int count = 0;
        for (int i = 0; i < this->bigLimit + this->mediumLimit + this->smallLimit; i++) {
            // Count the number of cars parked in the system of that type
            if (this->parkingArray[i] == carType) {
                count++;
            }

            // Stop if the count becomes equal to the limit
            if (count == limit) {
                return false;
            }

            // If the count is less than the limit, then add the car
            // to the first available empty slot from the left
            if (this->parkingArray[i] == -1) {
                this->parkingArray[i] = carType;
                return true;
            }
        }

        // If no empty slot is found, then return False.
        // However, this line will never be executed if count < limit
        // because slot will be found before count becomes equal to limit
        return false;
    }
};

//////////////////////////////////////////////
// 961. N-Repeated Element in Size 2N Array
// 30MAY23
/////////////////////////////////////////////
//the only element repeated n times is the one repetead more than once,
//use hashetset
class Solution {
public:
    int repeatedNTimes(vector<int>& nums) {
        set<int> seen;
        
        for(int x: nums){
            auto found = seen.find(x);
            if (found != seen.end()){
                return x;
            }
            seen.insert(x);
        }
        
        return 0;
    }
};