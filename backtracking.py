from datetime import datetime
from constraint import *
problem = Problem()
problem.addVariable("a", list(range(10000)))
problem.addVariable("b", list(range(10000)))
problem.addConstraint(lambda a, b: a+b <= 15,("a", "b"))
problem.addConstraint(lambda a: a <= 10,("a"))
problem.addConstraint(lambda b: -b <= -10,("b"))
dt = datetime.now()
before = dt.microsecond
print(problem.getSolution())
print("ms =", (datetime.now().microsecond-before ) / 1000.)
