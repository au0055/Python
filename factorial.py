# method 1

def factorial(n):
	if n < 0:
		return 0
	elif n == 0 or n == 1:
		return 1
	else:
		p= 1
		while(n > 1):
			p*= n
			n -= 1
		return print(p)
	
def Main():
	a=5
	factorial(a)

# Driver Code
if __name__=="__main__":
	Main()





