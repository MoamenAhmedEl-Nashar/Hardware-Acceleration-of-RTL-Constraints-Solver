from Solver import *

myClause = Clause(2,3)
myClause.add_integer_literal([1,1,1,4])
myClause.add_integer_literal([11,22,33,44])
myClause.add_boolean_literal(0,0)
#myClause.print_clause()

mySolver = Solver()
mySolver.formula.append(myClause)

myClause1 = Clause(2,3)
myClause1.add_integer_literal([10,20,30,40])
myClause1.add_integer_literal([100,200,300,400])
myClause1.add_boolean_literal(0,0)
#myClause1.print_clause()

mySolver.formula.append(myClause1)

mySolver.formula[0].print_clause()
print(mySolver.reduce_literal(0,0,1))