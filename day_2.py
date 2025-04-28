"""def min(values):
    min_value = values[0]
    for i in range(len(values)):
        if values[i] < min_value:
            min_value = values[i]
    return min_value
a=int(input("Enter number of elements in list: "))
l=[]
for i in range(a):
    b=int(input("Enter numbers in list: "))
    l.append(b)
print("The minimum value is: ",min(l))
"""
"""def fact(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * fact(n - 1)
number = int(input("Enter a number: "))
result = fact(number)
print(f"The factorial of {number} is: {result}")"""

def is_palindrome(s):
    return s == s[::-1]
number = input("Enter a number: ")
if is_palindrome(number):
    print(f"{number} is a palindrome.")
else:
    print(f"{number} is not a palindrome.")
