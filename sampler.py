import numpy as np
import random
import copy
import math

#################################
random.seed(1)


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
    variable_names: [str]
    coefficient: [int]

    def set_literal(self, no_of_int_variables, variable_names, coefficient: [int]):
        self.no_of_int_variables = no_of_int_variables  # number of int_variables  [y1,y2,....,y3]
        self.variable_names = variable_names  # y2,y1,y3,y5
        self.coefficient = coefficient  # coefficient is a set of integers

    def get_no_of_variables(self):
        return self.no_of_int_variables

    def get_coefficient(self):
        return self.coefficient

    def get_variable_names(self):
        return self.variable_names

    def get_bias(self):
        return self.coefficient[-1]

    def print_int_literal(self):
        for i in range(0, self.no_of_int_variables + 1):
            if i != self.no_of_int_variables:
                if self.coefficient[i] != 0:
                    print(self.coefficient[i], self.variable_names[i], " + ", sep=' ', end='', flush=True)
            else:  # it"s the bias
                print(self.coefficient[i], sep=' ', end='', flush=True)
        print(" <= 0 ")

    def reduce_literal(self, current_assignment: [int], variable_to_be_unchanged):
        reduced_integer_literal = IntegerLiteral()
        reduced_integer_literal = copy.deepcopy(self)
        if len(self.variable_names) == 1 and variable_to_be_unchanged in self.variable_names:
            return reduced_integer_literal
        elif variable_to_be_unchanged in self.variable_names:
            new_bias = 0
            i: int
            for i in range(0, self.no_of_int_variables):
                if self.variable_names[i] != variable_to_be_unchanged:
                    new_bias += current_assignment[i] * self.coefficient[i]
                    reduced_integer_literal.coefficient[i] = 0
            reduced_integer_literal.coefficient[reduced_integer_literal.no_of_int_variables] = self.coefficient[
                                                                                                   self.no_of_int_variables] + new_bias
            return reduced_integer_literal
        else:
            return False  # this variable is not found in that literal


# clause is in form
# "integer_literal[0] | integral_literal[1] | ... | boolean_literal[0] | boolean_literal[1] | ... "


class Clause:
    no_of_boolean_literals: int
    no_of_int_literals: int
    boolean_literals: [BooleanLiteral]  # is a set if literals
    int_literals = [IntegerLiteral]  # is a set of literals

    def set_clause(self, no_of_boolean_literals: int, no_of_int_literals: int, boolean_literals: [BooleanLiteral],
                   int_literals: [IntegerLiteral]):
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
        print(" [ ", sep=' ', end='', flush=True)
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
    integer_variable_names: [str]
    boolean_variable_names: [str]
    no_of_clauses: int
    clauses = [Clause]

    def set_formula(self, no_of_boolean_variables, no_of_integer_variables, boolean_variable_names,
                    integer_variable_names, no_of_clauses, clauses):
        self.no_of_integer_variables = no_of_integer_variables
        self.no_of_boolean_variables = no_of_boolean_variables
        self.integer_variable_names = integer_variable_names
        self.boolean_variable_names = boolean_variable_names
        self.no_of_clauses = no_of_clauses
        self.clauses = clauses

    def print_formula(self):  # to be edited

        print("Y=", self.no_of_integer_variables, "X=", self.no_of_boolean_variables, "C=", self.no_of_clauses)
        print("MBINF formula = ", sep=' ', end='', flush=True)

        for i in range(0, self.no_of_clauses):
            self.clauses[i].print_clause()
            if i != self.no_of_clauses - 1:
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
    pls0: float
    # variables
    current_values_boolean = []
    current_values_integer = []

    def __init__(self, formula: MBINF, T, pls):
        self.formula = formula
        self.temperature = T
        self.pls = pls

    def set_pls(self, pls):
        self.pls = pls

    def set_temperature(self, t):
        self.temperature = t

    def make_random_assignment_integer(self):
        n = self.formula.get_no_of_integer_variables()
        for i in range(0, n):
            self.current_values_integer.append(
                random.randint(0, 1000))  # assume 1000 is the maximum number 'make an enhancement' !

    def make_random_assignment_boolean(self):
        n = self.formula.get_no_of_boolean_variables()
        for i in range(0, n):
            self.current_values_boolean.append(random.randint(0, 1))

    # def set_values(self, values: [[int], [int]]):
    # self.current_values = values

    def check_satisfiability(self):
        i: int
        j: int
        k: int
        m: int
        for i in range(0, self.formula.no_of_clauses):
            for j in range(0, self.formula.clauses[i].no_of_int_literals):
                count = 0
                for k in range(0, self.formula.no_of_integer_variables-1):
                    count += self.formula.clauses[i].int_literals[j].coefficient[k]*self.current_values_integer[k]
                count -= self.formula.clauses[i].int_literals[j].coefficient[self.formula.no_of_integer_variables]
                if count > 0:
                    return False
            for m in range(0, self.formula.clauses[i].no_of_boolean_literals):
                if self.current_values_boolean[self.formula.clauses[i].boolean_literals[m].number] != self.formula.clauses[i].boolean_literals[m].value:
                    return False
        return True
    
        def find_number_of_unsatisfied_clauses(self):
        i: int
        j: int
        k: int
        m: int
        no_of_unsatisfied_clauses = 0
        for i in range(0, self.formula.no_of_clauses):
            flag = 0
            for j in range(0, self.formula.clauses[i].no_of_int_literals):
                count = 0
                for k in range(0, self.formula.no_of_integer_variables - 1):
                    count += self.formula.clauses[i].int_literals[j].coefficient[k] * self.current_values_integer[k]
                count -= self.formula.clauses[i].int_literals[j].coefficient[self.formula.no_of_integer_variables]
                if count > 0:
                    no_of_unsatisfied_clauses += 1
                    flag = 1
                    break
            if flag == 1:
                continue

            for m in range(0, self.formula.clauses[i].no_of_boolean_literals):
                if self.current_values_boolean[self.formula.clauses[i].boolean_literals[m].number] != \
                        self.formula.clauses[i].boolean_literals[m].value:
                    no_of_unsatisfied_clauses += 1
                    break
        return no_of_unsatisfied_clauses

    # this function computes pls (the probability that determines which move to make"
    def compute_pls(self, iteration_number: int):
        self.pls = self.pls0 * math.exp(1-iteration_number)

    # this function computes temperature (the value that controls the probability p computed in metropolis move)

    def compute_temperature(self):
        pass

    def propose(self, selected_integer_variable):
        no_of_integer_variables = self.formula.get_no_of_integer_variables()
        no_of_boolean_variables = self.formula.get_no_of_boolean_variables()
        integer_variable_names = self.formula.get_integer_variable_names()
        boolean_variable_names = self.formula.get_boolean_variable_names()
        # get the proposed value of this selected integer varaible
        clauses = self.formula.get_clauses()
        first_bias = 1000
        second_bias = 1001
        for clause in clauses:
            literals = clause.get_int_literals()
            for literal in literals:
                reduced_literal = literal.reduce_literal(self.current_values_integer, selected_integer_variable)
                if reduced_literal != False:
                    bias = reduced_literal.get_bias()
                    if bias <= 0:  # -ve bias
                        first_bias = min(bias, first_bias)
                    elif bias > 0:  # +ve bias
                        second_bias = min(bias, second_bias)
        # construct weights for probability distribution segments(3 segments)
        c1 = -first_bias  # y<c1  ,  y> c2
        c2 = second_bias
        if c1 > c2:  # case 1  (exp1 uniform2 exp3)
            w1 = 1 / (1 - math.exp(-1))  # 1.58
            w2 = c1 - c2 + 1
            w3 = 1 / (1 - math.exp(-1))  # 1.58
            sum_w = w1 + w2 + w3
            p1 = w1 / sum_w
            p2 = w2 / sum_w
            p3 = w3 / sum_w
            # select segment according to the normalized propabilities p1,p2,p3
            segments_numbers = [1, 2, 3]
            probabilities = [p1, p2, p3]
            selected_segment_number = np.random.choice(segments_numbers, p=probabilities)
            # update current assignment with this new proposed value
            if selected_segment_number == 2:  # uniform2
                proposed_value = random.randint(c2, c1)
            if selected_segment_number == 1:  # exp1
                theta = random.uniform(0, w1)
                d = math.ceil(-1 - math.log(1 - theta * (1 - math.exp(-1))))
                proposed_value = c2 - d
            if selected_segment_number == 3:  # exp3
                theta = random.uniform(0, w3)
                d = math.ceil(-1 - math.log(1 - theta * (1 - math.exp(-1))))
                proposed_value = c1 + d
        if c1 <= c2:  # case 2  (exp1 exp2)
            mid = int((c1 + c2) / 2)
            w1 = 1 / (1 - math.exp(-1))  # 1.58
            w2 = 1 / (1 - math.exp(-1))  # 1.58
            sum_w = w1 + w2
            p1 = w1 / sum_w
            p2 = w2 / sum_w
            # select segment according to the normalized propabilities p1,p2
            segments_numbers = [1, 2]
            probabilities = [p1, p2]
            selected_segment_number = np.random.choice(segments_numbers, p=probabilities)
            # update current assignment with this new proposed value
            if selected_segment_number == 1:  # exp1
                theta = random.uniform(0, w1)
                d = math.ceil(-1 - math.log(1 - theta * (1 - math.exp(-1))))
                proposed_value = mid - d
            if selected_segment_number == 2:  # exp2
                theta = random.uniform(0, w2)
                d = math.ceil(-1 - math.log(1 - theta * (1 - math.exp(-1))))
                proposed_value = mid + d

        return proposed_value

    def metropolis_move(self):

        no_of_integer_variables = self.formula.get_no_of_integer_variables()
        no_of_boolean_variables = self.formula.get_no_of_boolean_variables()
        integer_variable_names = self.formula.get_integer_variable_names()
        boolean_variable_names = self.formula.get_boolean_variable_names()

        # select variable boolean or integer
        if len(boolean_variable_names) == 0:
            random_variable_is_int_or_bool = random.randint(1, 1)
        else:
            random_variable_is_int_or_bool = random.randint(0, 1)  ## 1 --> int     0-->  boolean
        if random_variable_is_int_or_bool == 0:
            # select boolean variable
            selected_boolean_variable = random.choice(boolean_variable_names)
            # flip the value of this selected boolean variable
            self.current_values_boolean[boolean_variable_names.index(selected_boolean_variable)] = (
                                                                                                               self.current_values_boolean[
                                                                                                                   boolean_variable_names.index(
                                                                                                                       selected_boolean_variable)] - 1) % 2
        else:
            # select integer variable
            selected_integer_variable = random.choice(integer_variable_names)
            index_of_selected_integer_variable = integer_variable_names.index(selected_integer_variable)
            proposed_value = self.propose(selected_integer_variable)
            # update current integer assignment
            self.current_values_integer[index_of_selected_integer_variable] = proposed_value
            # Q calculating

            # U calculating

            return self.current_values_integer

    def local_move(self):
        pass



# test #
L11 = BooleanLiteral()
L11.set_literal(0, 1)
L12 = BooleanLiteral()
L12.set_literal(1, 0)
L13 = IntegerLiteral()
L13.set_literal(4, ["Y0", "Y1", "Y2", "Y3"], [1, 3, 4, 1, 4])
L13.print_int_literal()
L21 = BooleanLiteral()
L21.set_literal(2, 0)
L22 = BooleanLiteral()
L22.set_literal(3, 1)
L23 = IntegerLiteral()
L23.set_literal(4, ["Y0", "Y1", "Y2", "Y3"], [1, 1, 0, 0, 2])
C1 = Clause()
C1.set_clause(2, 1, [L11, L12], [L13])
C2 = Clause()
C2.set_clause(2, 1, [L21, L22], [L23])
MBINF1 = MBINF()
MBINF1.set_formula(4, 4, ["X0", "X1", "X2", "X3"], ["Y0", "Y1", "Y2", "Y3"], 2, [C1, C2])
MBINF1.print_formula()
S1 = Sampler(MBINF1, 0, 0)
S1.current_values_boolean = [1, 0, 0, 1]
S1.current_values_integer = [1, 1, 0, 0]
print(S1.check_satisfiability())









### test ###
integer_variable_names=['y1','y2']

L1 = IntegerLiteral()
L1.set_literal(2, integer_variable_names,[1, 1, -10])

L2 = IntegerLiteral()
L2.set_literal(1, ['y1'],[1, -10])

L3 = IntegerLiteral()
L3.set_literal(1, ['y2'],[1, -10])

C1 = Clause()
C1.set_clause(0, 3, [], [L1,L2,L3])

formula = MBINF()
formula.set_formula(0 , 2 , [] ,integer_variable_names , 1, [C1])
sampler=Sampler(formula,1,1)
sampler.make_random_assignment_integer()
sampler.make_random_assignment_boolean()



for i in range(0,99):
    current=sampler.metropolis_move()
    if current[0]<=10 and current[1]<=10 and current[0]+current[1]<=10:
        print(current)
        print('iteration:',i)
        break
#formula.print_formula()
#print(L1.get_bias())
#L1.print_int_literal()
#new=L1.reduce_literal([11,11],'y1')
#new.print_int_literal()
