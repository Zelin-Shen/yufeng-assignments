#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    cin >> q;

    unordered_map<int, long long> freq;
    freq.reserve(q);
    vector<pair<int, long long>> seq;
    seq.reserve(q);

    long long max_freq = 0;
    int mode_val = -1;
    bool mode_valid = false;

    auto recalc_mode = [&]()
    {
        if (freq.empty())
        {
            mode_val = -1;
            max_freq = 0;
            mode_valid = true;
            return;
        }

        max_freq = 0;
        mode_val = 2e9;

        for (auto &p : freq)
        {
            if (p.second > max_freq || (p.second == max_freq && p.first < mode_val))
            {
                max_freq = p.second;
                mode_val = p.first;
            }
        }
        mode_valid = true;
    };

    while (q--)
    {
        int op;
        cin >> op;

        if (op == 1)
        {
            long long k;
            int x;
            cin >> k >> x;

            if (!seq.empty() && seq.back().first == x)
            {
                seq.back().second += k;
            }
            else
            {
                seq.push_back({x, k});
            }

            long long new_freq = (freq[x] += k);

            if (!mode_valid)
            {
                recalc_mode();
            }
            else if (new_freq > max_freq || (new_freq == max_freq && x < mode_val))
            {
                max_freq = new_freq;
                mode_val = x;
            }
        }
        else
        {
            long long k;
            cin >> k;

            while (k > 0 && !seq.empty())
            {
                int val = seq.back().first;
                long long count = seq.back().second;
                long long remove = min(k, count);

                k -= remove;
                long long new_freq = (freq[val] -= remove);

                if (new_freq == 0)
                {
                    freq.erase(val);
                    if (val == mode_val)
                        mode_valid = false;
                }
                else if (val == mode_val && new_freq < max_freq)
                {
                    mode_valid = false;
                }

                if (remove >= count)
                {
                    seq.pop_back();
                }
                else
                {
                    seq.back().second -= remove;
                }
            }

            if (!mode_valid)
            {
                recalc_mode();
            }
        }

        cout << mode_val << '\n';
    }

    return 0;
}