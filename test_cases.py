from sampler import *
from datetime import datetime

########################################## seed choosing ##################################################################
SEED=1
random.seed(SEED)
np.random.seed(SEED)
###########################################################################################################################

def main():

    T = 1
    pls0 = 1
    sampler = Sampler(T,pls0)
    dt = datetime.now()
    before = dt.microsecond
    sampler.sample()
    print("ms =", (datetime.now().microsecond-before ) / 1000.)

if __name__ == "__main__":
    main()
'''
-------------- Input example should be a text file named "input.txt" -------------------------
constraints on input format
- must be more than one constraint
- and constraints only
- integral variables only
- input is in the form "y0 , y1, ..." with the same order should start with y0
- if we want a negative coeff it should be - 1 not -1 for the coeff of a variable
Ex:
2
1 y0 + 1 y1 <= 10
and
1 y0 <= 10
and
1 y1 <= 10
'''