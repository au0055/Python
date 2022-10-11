#method 1

def maximum(a,b):
    
    if(a>b):
        return a
    else:
        return b

a=int(input())
b=int(input())
print(maximum(a,b))

#method 2

c=2
d=8
maxi = maximum(c,d)

print(maxi)

#method 3(ternary op)

e=1
f=9

print(e if e>=f else f)

# method 4(python code to find maximum of two numbers)

a=2;b=4
maximum = lambda a,b:a if a > b else b
print(f'{maximum(a,b)} is a maximum number')

# method 5 (list comprihension)
a=2;b=4
x=[a if a>b else b]
print("maximum number is:",x)
