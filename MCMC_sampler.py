import random
import copy
#################################
random.seed(2)



###classes difinition###



# boolean literal can be x^k or !x^k
class BooleanLiteral:
    number: int  # which is  k
    value: int  # means x^k or !x^k
    

    def set_literal(self, number: int, value: int):
        self.number = number
        self.value = value

    def get_literal(self):
        return [self.number, self.value]

    def print_boolean_literal(self):
        if self.value == 1:
            print(" x^", self.number, sep=' ', end='', flush=True)
        else:
            print(" !x^", self.number, sep=' ', end='', flush=True)

# integral literal is in form :
# "coefficient[0] * y^0 + coefficient[1] * y^1+.....+ bias <= 0"

class IntegerLiteral:
    no_of_int_variables: int
    variable_names : [str]
    coefficient: [int]

    def set_literal(self, no_of_int_variables: int,variable_names : [str], coefficient: [int]):
        self.no_of_int_variables = no_of_int_variables  # number of int_variables  [y1,y2,....,y3]
        self.variable_names =variable_names #y2,y1,y3,y5
        self.coefficient = coefficient  # coefficient is a set of integers

    def get_no_of_variables(self):
        return self.no_of_int_variables
    def get_coefficient(self):
        return self.coefficient
    def get_variable_names(self):
        return self.variable_names

    def print_int_literal(self):
        for i in range(0, self.no_of_int_variables+1):
            if i != self.no_of_int_variables:
                if self.coefficient[i]!=0:
                    print(self.coefficient[i], self.variable_names[i], " + ", sep=' ', end='', flush=True)
            else:  # it"s the bias
                print(self.coefficient[i], sep=' ', end='', flush=True)
        print(" <= 0 ")

    def reduce_literal(self, current_assignment: [int], variable_to_be_unchanged):
        if variable_to_be_unchanged in self.variable_names:
            reduced_integer_literal = IntegerLiteral()
            reduced_integer_literal = copy.deepcopy(self)
            new_bias = 0
            i: int
            for i in range(0, self.no_of_int_variables ):
                if self.variable_names[i] != variable_to_be_unchanged:
                    new_bias += current_assignment[i] * self.coefficient[i]
                    reduced_integer_literal.coefficient[i] = 0
            reduced_integer_literal.coefficient[reduced_integer_literal.no_of_int_variables] = self.coefficient[self.no_of_int_variables] + new_bias
            return reduced_integer_literal
        else :
            return False  #this variable is not found in that literal

# clause is in form
# "integer_literal[0] | integral_literal[1] | ... | boolean_literal[0] | boolean_literal[1] | ... "


class Clause:
    no_of_boolean_literals: int
    no_of_int_literals: int
    boolean_literals: [BooleanLiteral]  # is a set if literals
    int_literals = [IntegerLiteral]  # is a set of literals

    def set_clause(self, no_of_boolean_literals: int, no_of_int_literals: int, boolean_literals: [BooleanLiteral], int_literals: [IntegerLiteral]):
        self.no_of_boolean_literals = no_of_boolean_literals
        self.no_of_int_literals = no_of_int_literals
        self.boolean_literals = boolean_literals  # is a set if literals
        self.int_literals = int_literals  # is a set of literals

    def get_no_of_int_literals(self):
        return self.no_of_int_literals

    def get_no_of_boolean_literals(self):
        return self.no_of_boolean_literals

    def get_boolean_literals(self):
        return self.boolean_literals

    def get_int_literals(self):
        return self.int_literals

    def print_clause(self):
        print(" [ ", sep=' ', end='', flush=True )
        for i in range(0, self.no_of_int_literals):
            self.int_literals[i].print_int_literal()
            print("|", sep=' ', end='', flush=True)

        for i in range(0, self.no_of_boolean_literals):
            self.boolean_literals[i].print_boolean_literal()
            if i != self.no_of_boolean_literals - 1:
                print(" | ", sep=' ', end='', flush=True)
            else:
                print(" ] ", sep=' ', end='', flush=True)

# MBINF formula is in form
# " clause[0] & clause[1] & clause[2] & ... & clause[m] "


class MBINF:
    no_of_integer_variables: int
    no_of_boolean_variables: int
    integer_variable_names :[str]
    boolean_variable_names :[str]
    no_of_clauses: int
    clauses = [Clause]

    def set_formula(self, no_of_boolean_variables, no_of_integer_variables,boolean_variable_names,integer_variable_names, no_of_clauses, clauses):
        self.no_of_integer_variables = no_of_integer_variables
        self.no_of_boolean_variables = no_of_boolean_variables
        self.integer_variable_names=integer_variable_names
        self.boolean_variable_names=boolean_variable_names
        self.no_of_clauses = no_of_clauses
        self.clauses = clauses

    def print_formula(self): # to be edited

        print("Y=", self.no_of_integer_variables, "X=", self.no_of_boolean_variables,"C=", self.no_of_clauses)
        print("MBINF formula = ", sep=' ', end='', flush=True)

        for i in range(0, self.no_of_clauses):
            self.clauses[i].print_clause()
            if i != self.no_of_clauses-1:
                print("&", sep=' ', end='', flush=True)
    
    def get_clauses(self):
        return self.clauses
    def get_integer_variable_names(self):
        return self.integer_variable_names
    def get_boolean_variable_names(self):
        return self.boolean_variable_names
    def get_no_of_integer_variables(self):
        return self.no_of_integer_variables
    def get_no_of_boolean_variables(self):
        return self.no_of_boolean_variables


class Sampler:
    formula: MBINF
    temperature: float
    pls: float
    # variables
    current_values_boolean = []
    current_values_integer = []
    

    def __init__(self, formula: MBINF,T,pls):
        self.formula = formula
        self.temperature=T
        self.pls=pls

    def set_pls(self, pls):
        self.pls = pls

    def set_temperature(self, t):
        self.temperature = t

    def make_random_assignment_integer(self):
        n=self.formula.get_no_of_integer_variables()
        for i in range(0,n):
            self.current_values_integer.append(random.randint(0,1000)) # assume 1000 is the maximum number
        

    def make_random_assignment_boolean(self):
        n=self.formula.get_no_of_boolean_variables()
        for i in range(0,n):
            self.current_values_boolean.append(random.randint(0,1)) 
        

    #def set_values(self, values: [[int], [int]]):
        #self.current_values = values

    def check_satisfiability (self):
        pass
        ##########################

    # this function computes pls (the probability that determines which move to make"
    def compute_pls(self): 
         pass

    # this function computes temperature (the value that controls the probability p computed in metropolis move)

    def compute_temperature(self):
        pass

    def metropolis_move(self):

        no_of_integer_variables=self.formula.get_no_of_integer_variables()
        no_of_boolean_variables=self.formula.get_no_of_boolean_variables()
        integer_variable_names=self.formula.get_integer_variable_names()
        boolean_variable_names=self.formula.get_boolean_variable_names()
        #select variable boolean or integer
        random_index_is_int_or_bool=random.randint(0,1)    ## 1 --> int     0-->  boolean
        if random_is_int_or_bool == 1:
            v_index=randint(0,no_of_int_variables)
        else :
            v_index=randint(0,no_of_boolean_variables)

        if random_index_is_int_or_bool == 0 :  ## boolean variable
            current_values[0][v_index]=(current_values[0][v_index]-1)%2  # flip
        else : ## integer variable
            current_values[1][v_index]=gibbs(self)

    def gibbs (self,v_index):
        pass

    def project(self,v_index):
        new_clause=clause()
        old_clauses=formula.get_clauses() 
        for old_clause in old_clauses:
            old_int_literals=old_clause.get_int_literals()
            for old_int_literal in old_int_literals:
                pass
        return new_clause
        
         

    def sample_from(new_clause):
        pass
            
        
            
        

    def local_move(self): 
       pass



### test ###
integer_variable_names=['y1','y2']

L1 = IntegerLiteral()
L1.set_literal(2, integer_variable_names,[1, 1, -10])

L2 = IntegerLiteral()
L2.set_literal(1, integer_variable_names[0],[1, -10])

L3 = IntegerLiteral()
L3.set_literal(1, integer_variable_names[1],[1, -10])

C1 = Clause()
C1.set_clause(0, 3, [], [L1,L2,L3])

formula = MBINF()
formula.set_formula(0 , 2 , [] ,integer_variable_names , 1, [C1])
sampler=Sampler(formula,1,1)
sampler.make_random_assignment_integer()
sampler.make_random_assignment_boolean()
#formula.print_formula()

#L1.print_int_literal()
#new=L1.reduce_literal([11,11],'y1')
#new.print_int_literal()
