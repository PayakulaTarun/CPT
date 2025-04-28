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

"""
def is_palindrome(s):
    return s == s[::-1]
number = input("Enter a number: ")
if is_palindrome(number):
    print(f"{number} is a palindrome.")
else:
    print(f"{number} is not a palindrome.")
    """

"""
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
number = int(input("Enter a number: "))
if is_prime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")
"""
"""
def func(n):
    if n>50:
        return n-5
    else:
        return func(n+5)
number = int(input("Enter a number: "))
print("The result is: ",func(number))
"""

def sfact(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * sfact(n - 1)
number = int(input("Enter a number: "))
result = sfact(number)
print(f"The factorial of {number} is: {result}")