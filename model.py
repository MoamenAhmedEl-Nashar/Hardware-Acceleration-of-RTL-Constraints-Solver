'''
you shoul mofify the model function implementation as follows :
  *adjust range of variables an add more lines for each variable
  *modify the eqn as products of f()*g() according to your input constraints
  *modify print to output all variables
you should modify the main code as follows:
  *modify number of variables and number of iterations

'''
import math
import random
def f(x):#return 1 if x>=0   , return 0 if x<0
    t=math.floor(x/((x**2)+1))
    return 1+t

def g(x):#return 1 if x>0    , return 0 if x<=0
    t=math.floor(-x/((x**2)+1))
    return -t



def model(n_variables,n_iterations):#add ranges , eqn

    for i in range(n_iterations):
        x=random.randint(0,10)
        y=random.randint(0,10)
        eqn=f(100-x**2-y**2)
        if eqn==1:
            print(str(x)+','+str(y))

    print('done')

#main code (testBench)
n_variables=2
n_iterations=100
model(n_variables,n_iterations)
