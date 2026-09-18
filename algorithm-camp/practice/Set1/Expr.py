expression = input().replace(" ", "")

operator_pos = -1
for i, char in enumerate(expression):
    if char in "+-*":
        operator_pos = i
        break

num1 = int(expression[:operator_pos])
num2 = int(expression[operator_pos + 1:])
operator = expression[operator_pos]

if operator == "+":
    result = num1 + num2
elif operator == "-":
    result = num1 - num2
elif operator == "*":
    result = num1 * num2

print(result)