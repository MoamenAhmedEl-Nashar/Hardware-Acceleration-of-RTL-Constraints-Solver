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