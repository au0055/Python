a=4
b=0

try:
    k= a//b
    print(k)

except ZeroDivisionError:
    print("A number cannot be divided by 0")

finally:
    print("zero division")

assert b!=0, "Divivsion by zero"
print(a/b)


