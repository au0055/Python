def factorial(n):
    return 1 if(n==0 or n==1) else n* factorial(n-1)

a=7
print(factorial(a))