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