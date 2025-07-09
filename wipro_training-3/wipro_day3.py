import sys
import argparse
# print("script name:", sys.argv[0])
# print("all arguments:", sys.argv[1:])
# print("Number of arguments:", len(sys.argv))
# if len(sys.argv) > 1:
#     print("first argument:", sys.argv[1])
# else:
#     print("no arguments provided")

# clt+/ will comment multiple lines

# num1=int(sys.argv[1])
# num2=int(sys.argv[2])
# num3=int(sys.argv[3])
# print("num1xnum2xnum3:", num1*num2*num3)

# if len(sys.argv) !=3:
#     print("usage:python wiproday3.py l b")
# else:
#     l=float(sys.argv[1])
#     b=float(sys.argv[2])
#     print("area of rectangle:", l+b)

# num1 = int(sys.argv[1])
# print("factorial of", num1, "is", end=" ")
# def factorial(n):
#     if n == 0 or n == 1:
#         return 1
#     else:
#         return n * factorial(n - 1)
# print(factorial(num1))

# if len(sys.argv) <2:
#     print("usage: python wiproday3.py n1 n2 n3 ...")
# numbers = [int(num) for num in sys.argv[1:]]
# total = sum(numbers)
# print("numbers:", numbers)
# print("number of arguments:", len(numbers))
# print("sum of numbers:", total)

# list append using command line arguments
# if len(sys.argv) < 2:
#     print("usage: python wiproday3.py n1 n2 n3 ...")
# else:
#     numbers = [int(num) for num in sys.argv[1:]]
#     print("numbers:", numbers)
#     print("number of arguments:", len(numbers))
#     print("sum of numbers:", sum(numbers))
#     print("list after appending:", end=" ")
#     print(numbers)
parser = argparse.ArgumentParser(description="Add two numbers")
parser.add_argument("num1", type=int, help="First number")
parser.add_argument("num2", type=int, help="Second number")