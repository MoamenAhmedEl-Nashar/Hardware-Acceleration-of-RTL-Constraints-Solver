import copy

# boolean literal can be x^k or !x^k


class boolean_literal:
    var_number: int  # which is  k
    var_value: int  # means x^k or !x^k

    def set_literal(self, var_number: int, var_value: int):
        self.var_number = var_number
        self.var_value = var_value

    def get_literal(self):
        return [self.var_number, self.var_value]

    def print_boolean_literal(self):
        if self.var_value == 1:
            print(" x^", self.var_number, sep=' ', end='', flush=True)
        else:
            print(" !x^", self.var_number, sep=' ', end='', flush=True)

# integral literal is in form :
# "coefficient[0] * y^0 + coefficient[1] * y^1+.....+ bias <= 0"

class integer_literal:
    no_of_int_variables: int
    coefficient: [int]

    def set_literal(self, no_of_int_variables: int, coefficient: [int]):
        self.no_of_int_variables = no_of_int_variables  # int_variables are [y^0,y^1,....,y^m]
        self.coefficient = coefficient  # coefficient is a set of integers

    def get_no_of_variables(self):
        return self.no_of_int_variables

    def get_coefficient(self):
        return self.coefficient

    def print_int_literal(self):
        for i in range(0, self.no_of_int_variables+1):
            if i != self.no_of_int_variables:
                print(self.coefficient[i], " y^", i, " + ", sep=' ', end='', flush=True)
            else:  # it"s the bias
                print(self.coefficient[i], sep=' ', end='', flush=True)
        print(" <= 0 ")

    def reduce_literal(self, current_assignment: [int], index_to_be_unchanged: int):
        temp = integer_literal()
        temp = copy.deepcopy(self)
        new_bias = 0
        i: int
        for i in range(0, self.no_of_int_variables ):
            if i != index_to_be_unchanged:
                new_bias += current_assignment[i] * self.coefficient[i]
                temp.coefficient[i] = 0
        temp.coefficient[temp.no_of_int_variables] = self.coefficient[self.no_of_int_variables] + new_bias
        return temp

# clause is in form
# "integer_literal[0] | integral_literal[1] | ... | boolean_literal[0] | boolean_literal[1] | ... "


class clause:
    no_of_boolean_literals: int
    no_of_int_literals: int
    boolean_literals: [boolean_literal]  # is a set if literals
    int_literals = [integer_literal]  # is a set of literals

    def set_clause(self, no_of_boolean_literals: int, no_of_int_literals: int, boolean_literals: [boolean_literal], int_literals: [integer_literal]):
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
    no_of_int_variables: int
    no_of_boolean_variables: int
    no_of_clauses: int
    clauses = [clause]

    def set_formula(self, no_of_int_variables, no_of_boolean_variables, no_of_clauses, clauses):
        self.no_of_int_variables = no_of_int_variables
        self.no_of_boolean_variables = no_of_boolean_variables
        self.no_of_clauses = no_of_clauses
        self.clauses = clauses

    def print_formula(self):

        print("Y=", self.no_of_int_variables, "X=", self.no_of_boolean_variables,"C=", self.no_of_clauses)
        print("MBINF formula = ", sep=' ', end='', flush=True)

        for i in range(0, self.no_of_clauses):
            self.clauses[i].print_clause()
            if i != self.no_of_clauses-1:
                print("&", sep=' ', end='', flush=True)


class Sampler:
    formula: MBINF
    current_values = [[int], [int]]
    temperature: float
    pls: float

    def __init__(self, formula: MBINF):
        self.formula = formula

    def set_pls(self, pls):
        self.pls = pls

    def set_temperature(self, t):
        self.temperature = t

    def set_values(self, values: [[int], [int]]):
        self.current_values = values

    def check_satisfiability (self): {}
        ##########################

    # this function computes pls (the probability that determines which move to make"
    def compute_pls(self): {}
         #####################

    # this function computes temperature (the value that controls the probability p computed in metropolis move)
    def compute_temperature(self): {}
    ########################

    def metropolis_move(self): {}
       ###########################

    def local_move(self): {}
       #############################



### test ###
L11 = boolean_literal(0, 1)
L12 = boolean_literal(1, 0)
L13 = integer_literal(2, [1, 1, 4])
L21 = boolean_literal(0, 0)
L22 = boolean_literal(1, 1)
L23 = integer_literal(2, [1, 1, 2])
C1 = clause()
C1.set_clause(2, 1, [L11, L12], [L13])
C2 = clause()
C2.set_clause(2, 1, [L21, L22], [L23])
MBINF1 = MBINF()
MBINF1.set_formula(2, 2, 2, [C1, C2])
MBINF1.print_formula()





