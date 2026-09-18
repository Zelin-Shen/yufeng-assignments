"""
This program reads one line of text and counts the frequency of each English letter.
It ignores letter case, so 'a' and 'A' are treated as the same letter.

Algorithm:
1. Create an array of size 26, where index 0..25 corresponds to 'A'..'Z'.
2. Scan each character in the input line:
   - If it is a lowercase letter, convert it to uppercase by ASCII offset.
   - If it is an uppercase letter, count it directly.
   - Ignore all non-letter characters.
3. Output only letters that appear at least once, in alphabetical order.
   Format: "LETTER: count"

Complexity:
- Time: O(n), where n is the length of the input line.
- Space: O(1), fixed 26 counters.
"""

text = input()
cnt = [0] * 26

for ch in text:
    code = ord(ch)
    if 97 <= code <= 122:  # 'a' to 'z'
        code -= 32          # convert to uppercase
    if 65 <= code <= 90:    # 'A' to 'Z'
        cnt[code - 65] += 1

for i in range(26):
    if cnt[i] > 0:
        print(chr(i + 65) + ": " + str(cnt[i]))