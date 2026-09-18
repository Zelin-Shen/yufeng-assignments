/*
 * Binary Tree Isomorphism Check
 *
 * Algorithm:
 * - Recursively check if two trees are isomorphic
 * - For each node pair, try both:
 *   1. No swap: left-left, right-right
 *   2. Swap: left-right, right-left
 * - Base cases: both null (true), one null (false)
 *
 * Time Complexity: O(n) where n is number of nodes
 * Space Complexity: O(h) for recursion stack, h is height
 */

#include <iostream>
#include <vector>
#include <unordered_map>
using namespace std;

struct Node
{
    int left, right;
};

unordered_map<long long, int> memo;

// Encode two integers into one long long for hashing
inline long long encode(int a, int b)
{
    return ((long long)a << 32) | (unsigned int)b;
}

bool isIsomorphic(int a, int b, const vector<Node> &tree1, const vector<Node> &tree2)
{
    // Both null - isomorphic
    if (a == -1 && b == -1)
        return true;

    // One null - not isomorphic
    if (a == -1 || b == -1)
        return false;

    // Check memo
    long long key = encode(a, b);
    auto it = memo.find(key);
    if (it != memo.end())
        return it->second;

    // Try without swapping
    bool noSwap = isIsomorphic(tree1[a].left, tree2[b].left, tree1, tree2) &&
                  isIsomorphic(tree1[a].right, tree2[b].right, tree1, tree2);

    if (noSwap)
    {
        memo[key] = 1;
        return true;
    }

    // Try with swapping
    bool withSwap = isIsomorphic(tree1[a].left, tree2[b].right, tree1, tree2) &&
                    isIsomorphic(tree1[a].right, tree2[b].left, tree1, tree2);

    memo[key] = withSwap;
    return withSwap;
}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;

    while (T--)
    {
        int n;
        cin >> n;

        vector<Node> tree1(n), tree2(n);

        for (int i = 0; i < n; i++)
        {
            cin >> tree1[i].left >> tree1[i].right;
        }

        for (int i = 0; i < n; i++)
        {
            cin >> tree2[i].left >> tree2[i].right;
        }

        memo.clear();

        cout << (isIsomorphic(0, 0, tree1, tree2) ? "yes\n" : "no\n");
    }

    return 0;
}