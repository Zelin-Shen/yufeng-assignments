/*
 * Bidirectional Linked List with Cycle Detection
 *
 * Key points:
 * - split/link operations output yes/no
 * - visit operations directly output the traversal result
 * - Cycle detection: stop when returning to start node
 */

#include <iostream>
#include <vector>
#include <string>
using namespace std;

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<int> prev(n + 1, 0), next(n + 1, 0);

    while (m--)
    {
        string op;
        cin >> op;

        if (op == "split_succ")
        {
            int x;
            cin >> x;

            if (next[x] == 0)
            {
                cout << "no\n";
            }
            else
            {
                int y = next[x];
                next[x] = 0;
                prev[y] = 0;
                cout << "yes\n";
            }
        }
        else if (op == "split_prev")
        {
            int x;
            cin >> x;

            if (prev[x] == 0)
            {
                cout << "no\n";
            }
            else
            {
                int y = prev[x];
                prev[x] = 0;
                next[y] = 0;
                cout << "yes\n";
            }
        }
        else if (op == "link")
        {
            int x, y;
            cin >> x >> y;

            if (next[x] != 0 || prev[y] != 0)
            {
                cout << "no\n";
            }
            else
            {
                next[x] = y;
                prev[y] = x;
                cout << "yes\n";
            }
        }
        else if (op == "visit_succ")
        {
            int x;
            cin >> x;

            int start = x;
            cout << x;
            x = next[x];

            while (x != 0 && x != start)
            {
                cout << ' ' << x;
                x = next[x];
            }
            cout << '\n';
        }
        else
        { // visit_prev
            int x;
            cin >> x;

            int start = x;
            cout << x;
            x = prev[x];

            while (x != 0 && x != start)
            {
                cout << ' ' << x;
                x = prev[x];
            }
            cout << '\n';
        }
    }

    return 0;
}