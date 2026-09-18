"""
This program computes the phone call duration in seconds from two timestamps t1 and t2.

Input format:
- One line with two times separated by a space:
  HH:MM:SS HH:MM:SS
- Date information is omitted, and each part uses two digits.

Key points:
- Convert each timestamp into total seconds from 00:00:00.
- Because date is not given, if t2 is earlier than t1, it means the call crossed midnight.
- Duration = s2 - s1 (or s2 + 24*3600 - s1 when crossing midnight).

The problem guarantees:
- Valid time fields.
- Final duration does not exceed 12 hours.

Complexity:
- Time: O(1)
- Space: O(1)
"""

def to_seconds(t: str) -> int:
    # Parse "HH:MM:SS" into total seconds.
    h = (ord(t[0]) - 48) * 10 + (ord(t[1]) - 48)
    m = (ord(t[3]) - 48) * 10 + (ord(t[4]) - 48)
    s = (ord(t[6]) - 48) * 10 + (ord(t[7]) - 48)
    return h * 3600 + m * 60 + s


line = input().strip()
t1, t2 = line.split()

s1 = to_seconds(t1)
s2 = to_seconds(t2)

if s2 >= s1:
    print(s2 - s1)
else:
    print(s2 + 24 * 3600 - s1)