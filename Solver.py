"""
-> Hyper Parameters

1) T (temperature) : The temperature T provides a means to control the balance between the time spent
 visiting unsatisfying assignments and the ability to move between separated regions of solutions.
  When the temperature is high, the distribution is smooth and we move easily between solution regions,
   but the total probability of non-solution states can be large compared to the total probability of solutions.
 Conversely, a low temperature biases the distribution toward solutions at the cost of effectively disconnecting solution regions

- in our application(no need to have a good target distribution since we generate only ONE SAMPLE per RUN) we need low T as possible

-------------------------------------------------------------

2) pls : probability of taking local search move,  (1-pls) -> metropolis move
Although our local-search moves are based on the same distributions as our MetropolisHastings moves, they do not use the acceptance rule,
 so they distort the stationary distribution away from the target distribution.
"""



############################################ importing packages ##########################################################
import numpy as np
import random
import copy
import math
###########################################################################################################################


######################################## defining classes to represent constraints#########################################

# boolean literal can be x^k or !x^k
# boolean literal is not needed in this version of code

class Clause:
    number_of_integer_literal = int
    boolean_literals = []
    integer_literals = []
    number_of_boolean_variables = int
    number_of_integer_variables = int

    def __init__(self, number_of_boolean_variables, number_of_integer_variables):
        self.number_of_boolean_variables = number_of_boolean_variables
        self.number_of_integer_variables = number_of_integer_variables
        self.boolean_literals = [ [0,0] for i in range(0, self.number_of_boolean_variables) ]
        self.number_of_integer_literal = 0

    def add_integer_literal(self, coefficients):
        self.integer_literals.append(coefficients)
        self.number_of_integer_literal = self.number_of_integer_literal + 1

    def add_boolean_literal(self, index, coeff):
        if coeff:
            self.boolean_literals[index] = [1,1]
        else:
            self.boolean_literals[index] = [1,0]

    def print_clause(self):
        #print(self.number_of_integer_literal)
        #print(self.integer_literals)
        for i in range(0, self.number_of_integer_literal):
            for j in range (0, self.number_of_integer_variables):
                print(self.integer_literals[i][j], "Y", j, " + ", sep=' ', end='', flush=True)
            print(self.integer_literals[i][self.number_of_integer_variables]," <= 0")
            if i != self.number_of_integer_literal - 1:
                print("OR")

        for j in range(0, self.number_of_boolean_variables):
            if self.boolean_literals[j] == 1:
                print("OR")
                print("! X", j)

            elif self.boolean_literals[j] == 3:
                print("OR")
                print("X", j)

class Indicator:
    _from = int
    _to = int
    _type = int

    
class Distribution:
    indicators = [Indicator]

#encoding
UNIFORM=3
EXP_UP=2
EXP_DOWN=1
#########

class Solver:
    clauses = []
    current_values_int = []
    current_values_boolean = []
    number_of_clauses = int
    number_of_boolean_variables = int
    number_of_integer_variables = int
    pls0 = float
    pls = float
    bit_width_for_int_variables = [32]*number_of_integer_variables # assume that all int variables is length 32 bit


    def __init__(self, number_of_boolean_variables, number_of_integer_variables,number_of_clauses,pls0):
        self.number_of_boolean_variables = number_of_boolean_variables
        self.number_of_integer_variables = number_of_integer_variables
        self.number_of_clauses = number_of_clauses
        self.clauses = [Clause(number_of_boolean_variables, number_of_integer_variables) for i in range(0,number_of_clauses)]
        self.pls0 = pls0

    def check_bool_literal(self,clause_num, bool_literal_num):
        clause = self.clauses[clause_num]
        bool_literal = clause.boolean_literals[bool_literal_num]

        # if not exist
        if bool_literal[0] == 0:  # not exist bool literal
            return -1

        elif self.current_values_boolean[bool_literal_num] == bool_literal[1]:
            return True
        else:
            return False

    def check_int_literal(self, clause_num, int_literal_num):

        clause = self.clauses[clause_num]
        int_literal = clause.integer_literals[int_literal_num]
        count = 0
        for j in range(self.number_of_integer_variables):
            count += int_literal[j] * self.current_values_int[j]
        count += int_literal[-1]  # bias
        if count <= 0:
            return True

        return False

    def check_clause(self,clause_num):  # from 0 to NUM_OF_CLAUSES-1
        clause = self.clauses[clause_num]


        for j in range(clause.number_of_boolean_variables):
            if self.check_bool_literal(clause_num, j) == True:
                return True

        for k in range(clause.number_of_integer_variables):
            if self.check_int_literal(clause_num, k) == True:
                return True
        return False

    def check_formula(self):
        for i in range(self.number_of_clauses):
            if self.check_clause(i) == False:
                return False
        return True

    def compute_pls(self,iteration_number):
        self.pls = self.pls0 * math.exp(1 - iteration_number)
        if (self.pls > 1):
            self.pls = 1
        elif (self.pls < 0):
            self.pls = 0

    def make_random_assignment_int(self):
        for i in range(self.number_of_integer_variables):
            maximum = math.pow(2, self.bit_width_for_int_variables[i] - 1) - 1
            minimum = -math.pow(2, self.bit_width_for_int_variables[i] - 1) - 1
            self.current_values_int.append(random.randint(minimum, maximum))

    def make_random_assignment_bool(self):
        for i in range(self.number_of_boolean_variables):
            self.current_values_boolean.append(random.randint(0, 1))
    

    def reduce_literal(self, clause_num, int_literal_num, index_variable_to_be_unchanged):  # from 0 to NUM_OF_INT_VARIABLES-1
        #self.current_values_int =[1, 1, 1]
        clause = self.clauses[clause_num]
        int_literal = clause.integer_literals[int_literal_num]
        reduced_int_literal = int_literal

        # this variable is not found in that int literal
        if reduced_int_literal[index_variable_to_be_unchanged] == 0:
            return False

        new_bias = 0
        bias_updating = False
        for i in range(clause.number_of_integer_variables):
            if i != index_variable_to_be_unchanged:
                if int_literal[i] != 0:
                    bias_updating = True
                new_bias += self.current_values_int[i] * int_literal[i]
                reduced_int_literal[i] = 0

        # updating the bias
        if bias_updating == True:
            reduced_int_literal[-1] = int_literal[-1] + new_bias

        # +ve value coeff
        if (reduced_int_literal[index_variable_to_be_unchanged] > 0):
            reduced_int_literal[-1] /= int_literal[index_variable_to_be_unchanged]
            reduced_int_literal[index_variable_to_be_unchanged] = 1

        # -ve value coeff
        else:
            reduced_int_literal[-1] /= (-1 * int_literal[index_variable_to_be_unchanged])
            reduced_int_literal[index_variable_to_be_unchanged] = -1

        # should be something like the form of y1 + 10 <= 0 or -1 y1 +5 <= 0
        return reduced_int_literal

    def get_active_clauses(self, index_variable_to_be_unchanged):  # return active formula

        active_formula =[]
        for i in range(self.number_of_clauses):
            skip = False
            active_clause = Clause

            for j in range(self.number_of_boolean_variables):
                if self.check_bool_literal(i, j) == True:
                    skip = 1  # the clause is satisfied

            if skip == 0:  # the clause is not satisfied
                for k in range(self.clauses[i].number_of_integer_literal):
                    reduced_int_literal = self.reduce_literal(i, k, index_variable_to_be_unchanged)
                    if reduced_int_literal == False:  # this variable is not in that int literal
                        if self.check_int_literal(i, k) == True:
                            skip = 1

            if skip == 0:  # the clause is not satisfied
                for k in range(self.clauses[i].number_of_integer_literal):
                    reduced_int_literal = self.reduce_literal(i, k, index_variable_to_be_unchanged)
                    if reduced_int_literal != False:  # this variable is in that int literal
                        active_clause.add_integer_literal(reduced_int_literal)
            active_formula.append(active_clause)
        return active_formula

    def get_indicator(self,reduced_int_literal,index_variable_to_be_unchanged): #[1 0 0 5] or [0 -1 0 3]
        
        maximum=math.pow(2,self.bit_width_for_int_variables[index_variable_to_be_unchanged]-1)-1
        minimum=-math.pow(2,self.bit_width_for_int_variables[index_variable_to_be_unchanged]-1)-1

        indicator= Indicator()       
        distribution = Distribution() 
        if reduced_int_literal[index_variable_to_be_unchanged]==1:
            # minimum --> -bias
            indicator._from = minimum
            indicator._to = -reduced_int_literal[-1]   
            indicator._type = UNIFORM
            distribution.indicators.append(indicator)
            # -bias --> maximum
            indicator._from = -reduced_int_literal[-1]
            indicator._to = maximum 
            indicator._type = EXP_DOWN
            distribution.indicators.append(indicator)

        if reduced_int_literal[index_variable_to_be_unchanged]==-1:
            # minimum --> bias
            indicator._from = minimum
            indicator._to = reduced_int_literal[-1] 
            indicator._type = EXP_UP
            distribution.indicators.append(indicator)
            # bias --> maximum
            indicator._from = reduced_int_literal[-1]
            indicator._to = maximum
            indicator._type = UNIFORM
            distribution.indicators.append(indicator)

        return distribution

    def get_segments_from_active_clause(self,active_clause,index_variable_to_be_unchanged):
        case1=[]
        case2=[]
        for literal in active_clause.integer_literals:
                      
            if literal[index_variable_to_be_unchanged] > 0:
                case1.append(get_indicator(int_literal))
            else:
                case2.append(get_indicator(int_literal)

myClause = Clause(2,3)
myClause.add_integer_literal([1,1,1,4])
myClause.add_integer_literal([11,22,33,44])
myClause.add_boolean_literal(0,0)
#myClause.print_clause()
'''
mySolver = Solver()
mySolver.clauses.append(myClause)

myClause1 = Clause(2,3)
myClause1.add_integer_literal([10,20,30,40])
myClause1.add_integer_literal([100,200,300,400])
myClause1.add_boolean_literal(0,0)
#myClause1.print_clause()

mySolver.clauses.append(myClause1)

mySolver.clauses[0].print_clause()

print(mySolver.reduce_literal(0,0,1))'''