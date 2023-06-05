#include <iostream>
#include <vector>
#include <sstream>
#include <queue>
using namespace std;

class TreeNode {
 public:
   int val;
   TreeNode *left;
   TreeNode *right;
   TreeNode() : val(0), left(nullptr), right(nullptr) {}
   TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
   TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 };

void rightLeafSum(TreeNode* root, int& sum) {
    if (!root) {return;}
    if (root->right && !root->right->left && !root->right->right) {
            sum += root->right->val;
    }
 
    rightLeafSum(root->left, sum);
    rightLeafSum(root->right, sum);
}

void sumOfRightLeaves(TreeNode* root) {
    int sum{};
    if (!root) {return;}
    rightLeafSum(root, sum); 
    cout << sum << endl;
}




int main() {
    TreeNode* lr = new TreeNode(12);
    TreeNode* l = new TreeNode(9, nullptr, lr);
    TreeNode* rl = new TreeNode(15);
    TreeNode* rr = new TreeNode(7);
    TreeNode* r = new TreeNode(20, rl, rr);
    TreeNode* root = new TreeNode(3, l, r);
    sumOfRightLeaves(root);

}