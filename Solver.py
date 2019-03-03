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
#encoding
UNIFORM=3
EXP_UP=2
EXP_DOWN=1
#########
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



class Solver:
    formula = []
    current_values_int = []
    current_values_boolean = []
    number_of_clauses = int
    number_of_boolean_variables = int
    number_of_integer_variables = int
    pls0 = float
    pls = float
    temperature = float
    bit_width_for_int_variables = [] # assume that all int variables is length 32 bit


    def __init__(self, number_of_boolean_variables, number_of_integer_variables,number_of_clauses,pls0,temperature=1):
        self.number_of_boolean_variables = number_of_boolean_variables
        self.number_of_integer_variables = number_of_integer_variables
        self.number_of_clauses = number_of_clauses
        self.formula = [Clause(number_of_boolean_variables, number_of_integer_variables) for i in range(0,number_of_clauses)]
        self.pls0 = pls0
        self.temperature=temperature
        self.bit_width_for_int_variables=[32] * number_of_integer_variables# assume that all int variables is length 32 bit

    def check_bool_literal(self,clause_num, bool_literal_num):
        clause = self.formula[clause_num]
        bool_literal = clause.boolean_literals[bool_literal_num]

        # if not exist
        if bool_literal[0] == 0:  # not exist bool literal
            return -1

        elif self.current_values_boolean[bool_literal_num] == bool_literal[1]:
            return True
        else:
            return False

    def check_int_literal(self, clause_num, int_literal_num):

        clause = self.formula[clause_num]
        int_literal = clause.integer_literals[int_literal_num]
        count = 0
        for j in range(self.number_of_integer_variables):
            count += int_literal[j] * self.current_values_int[j]
        count += int_literal[-1]  # bias
        if count <= 0:
            return True

        return False

    def check_clause(self,clause_num):  # from 0 to NUM_OF_CLAUSES-1
        clause = self.formula[clause_num]


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

    def find_number_of_unsatisfied_clauses(self,assignme_to_check_on_it):
        buffer=self.current_values_int
        self.current_values_int=assignme_to_check_on_it
        count=0
        for i in range(self.number_of_clauses):
           if self.check_clause(i) == False:
               count=count+1
        self.current_values_int=buffer
        return count

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
        clause = self.formula[clause_num]
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

    def construct_active_formula(self, index_variable_to_be_unchanged):  # return reduced active formula 
        active_formula =[]
        for i in range(self.number_of_clauses):
            skip = False
            active_clause = Clause

            for j in range(self.number_of_boolean_variables):
                if self.check_bool_literal(i, j) == True:
                    skip = 1  # the clause is satisfied

            if skip == 0:  # the clause is not satisfied
                for k in range(self.formula[i].number_of_integer_literal):
                    reduced_int_literal = self.reduce_literal(i, k, index_variable_to_be_unchanged)
                    if reduced_int_literal == False:  # this variable is not in that int literal
                        if self.check_int_literal(i, k) == True:
                            skip = 1

            if skip == 0:  # the clause is not satisfied
                for k in range(self.formula[i].number_of_integer_literal):
                    reduced_int_literal = self.reduce_literal(i, k, index_variable_to_be_unchanged)
                    if reduced_int_literal != False:  # this variable is in that int literal
                        active_clause.add_integer_literal(reduced_int_literal)

            active_formula.append(active_clause)

        return active_formula

    def propose_value(self,reduced_active_formula,index_variable_to_be_unchanged):
         #assuming that there is only one integer Literal in any clause
        
        case1_biases=[] # case one -> y + b < 0
        case2_biases=[] # case two -> -y + b <0
        for clause in reduced_active_formula:
            if clause.integer_literals[0][index_variable_to_be_unchanged] > 0: # clause.integer_literals[0][index_variable_to_be_unchanged] -> returns the coeff. of the the chosed variable to be sambled
                
                case1_biases.append(clause.integer_literals[0][-1])
            
            elif clause.integer_literals[0][index_variable_to_be_unchanged] < 0:

                case2_biases.append(clause.integer_literals[0][-1])

        c1=-1*(min(case1_biases))
        c2=min(case2_biases)
        ## From the old code "sampler.py"
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

        #need to add case 3 for the OR case
        return proposed_value

    def flib(self,index_variable_to_be_changed):

        if self.current_values_boolean[index_variable_to_be_changed]==1:
            self.current_values_boolean[index_variable_to_be_changed]=0
        else:
            self.current_values_boolean[index_variable_to_be_changed]=1

    def pick_variable_to_be_sampled(self):
        x=random.randint(0,1)
        _type=""
        chosen_var=0
        if x == 1 :
            _type="bool"
            chosen_var = random.choice(range(self.number_of_boolean_variables))
        else:
            _type="integer"
            chosen_var = random.choice(range(self.number_of_integer_variables))
        return chosen_var,_type
    def calculate_proposed_distribution(self):
        # we shoud here calculate Q ˜ p(x,y) in the thesis page 18
        # but right now we hard coded just to 0.5
        Q=0.5
        return Q
    
    def metropolis_move(self):
        chosen_var,_type = self.pick_variable_to_be_sampled()

        if _type =="bool":
            self.flib(chosen_var)
        else: #_type == "integer"

            unsatisfied_clauses_before_the_move=self.find_number_of_unsatisfied_clauses(self.current_values_int)

            active_formula=self.construct_active_formula(chosen_var)
            proposed_value=self.propose_value(active_formula,chosen_var)

            # Update the proposed move to be equals the old assignment except the chosen_variable 
            proposed_move = self.current_values_int
            proposed_move[chosen_var]=proposed_value
            
            proposed_distribution = self.calculate_proposed_distribution() # proposed_distribution = 'Q' in the thises

            unsatisfied_clauses_after_the_move=self.find_number_of_unsatisfied_clauses(proposed_move)

            #calculate target distribution = Q*e^(U`-U) with the notation of the thesis 
            target_distribution=proposed_distribution * (math.exp(-((unsatisfied_clauses_after_the_move - unsatisfied_clauses_before_the_move) / self.temperature)))
            pr_do_change = min(1,target_distribution )
            pr_stay = 1 - pr_do_change

            choice = np.random.choice(['do_change', 'stay'], p=[pr_do_change, pr_stay])

            if choice == 'do_change':
               self.current_values_int = proposed_move
            
            return

    def Local_search_move(self):
        # dont return a value just update the current assimnt if needed, just like the metropolis 
        pass


    def solve(self):
        #make random assignments
        self.make_random_assignment_int()
        print('random',self.current_values_int)
        self.make_random_assignment_bool()
        #metropolis
        self.metropolis_move()
        counter=1
        print(counter,self.current_values_int)

        while self.check_formula()==False: #problem here
            counter+=1
            self.compute_pls(counter)
            choice=np.random.choice(['local','metropolis'], p=[self.pls,1-self.pls])

            if choice == 'local':
                print("local")
                self.local_move()

            elif choice == 'metropolis':
                print("metropolis")
                self.metropolis_move()
            print(counter,self.current_values_int)











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
