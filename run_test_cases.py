from sampler import *




def main():

    ########################################## seed choosing ##################################################################
    SEED = 1
    random.seed(SEED)
    np.random.seed(SEED)
    ###########################################################################################################################


    '''
    integer_variable_names = ['y1', 'y2']

    L1 = IntegerLiteral(integer_variable_names,2)
    L1.set_literal( [-2, 4, -10])
    L1.print_int_literal()
    Lnew = L1.reduce_literal([6,3],'y1')
    Lnew.print_int_literal()

    '''


    formula = get_input()
#    formula.print_formula()

    T = 1
    pls0 = 0
    sampler = Sampler(T,pls0)
    sampler.sample()

#    print("------------------")




#    sampler = Sampler(1,1) # Temp = 1 , pls =1

    '''
    clauses = sampler.get_active_clauses("y1")

    for clause in clauses:
        print(clause)


    '''




    '''
    ### test cases ### no boolean , each clause contains one literal
    ### test case 1 #### y1<=10 , y2<=10 , y1+y2<=10 ,
    integer_variable_names = ['y1', 'y2']

    L1 = IntegerLiteral()
    L1.set_literal(2, integer_variable_names, [2, 4, -10])
    L1.print_int_literal()
    Lnew = L1.reduce_literal([6,3],'y1')
    Lnew.print_int_literal()



    L2 = IntegerLiteral()
    L2.set_literal(1, ['y1'], [1, -10])

    L3 = IntegerLiteral()
    L3.set_literal(1, ['y2'], [1, -10])

    C1 = Clause()
    C1.set_clause(0, 1, [], [L1])
    C2 = Clause()
    C2.set_clause(0, 1, [], [L2])
    C3 = Clause()
    C3.set_clause(0, 1, [], [L3])

    formula = MBINF()
    formula.set_formula(0, 2, [], integer_variable_names, 3, [C1, C2, C3])

    sampler = Sampler(formula, 1, 1)
    #sampler.sample()
    
    


    Test case 2 (for active clauses)
    
      [2 ≤ y1]
    ^ [y1 <= 16]
    ^ [0 <= y2] 
    ^ [0 <= y3] 
    ^ [y1+y2 <=12] 
    ^ [(y1 + y2 <= 16) ∨ (y1-y2 >=0)] 
    ^ [(y1-y2 <= 18) ∨ (y2 + 2y3 <= 20)]
    
    

    integer_variable_names = ['y1', 'y2','y3']
    L1 = IntegerLiteral()
    L1.set_literal(1, ['y1'], [-1, 2])
    C1 = Clause()
    C1.set_clause(0, 1, [], [L1])
   # C1.print_clause()

    L2 = IntegerLiteral()
    L2.set_literal(1,['y1'],[1,-16])
    C2 = Clause()
    C2.set_clause(0, 1, [], [L2])
   # C2.print_clause()

    L3 = IntegerLiteral()
    L3.set_literal(1,['y2'],[-1,0])
    C3 = Clause()
    C3.set_clause(0, 1, [], [L3])
   # C3.print_clause()


    L4 = IntegerLiteral()
    L4.set_literal(1,['y3'],[-1,0])
    C4 = Clause()
    C4.set_clause(0, 1, [], [L4])
   # C4.print_clause()



    L5 = IntegerLiteral()
    L5.set_literal(2,['y1','y2'],[1,1,-12])
    C5 = Clause()
    C5.set_clause(0, 1, [], [L5])
   # C5.print_clause()


    L6 = IntegerLiteral()
    L6.set_literal(2,['y1','y2'],[1,1,-16])
    L7 = IntegerLiteral()
    L7.set_literal(2,['y1','y2'],[-1,1,0])
    C6 = Clause()
    C6.set_clause(0, 2, [], [L6,L7])
   # C6.print_clause()


    L8 = IntegerLiteral()
    L8.set_literal(2,['y1','y2'],[1,-1,-18])
    L9 = IntegerLiteral()
    L9.set_literal(2,['y2','y3'],[1,2,-20])
    C7 = Clause()
    C7.set_clause(0, 2, [], [L8,L9])
   # C7.print_clause()


    formula = MBINF()
    formula.set_formula(0, 3, [], integer_variable_names, 7, [C1, C2, C3,C4,C5,C6,C7])


    formula.print_formula()

    print('------------------')

    sampler = Sampler(formula, 1, 1)

    clauses = sampler.get_active_clauses("y1")

    for clause in clauses:
        clause.print_clause()

    '''







if __name__ == "__main__":
    main()

    '''  sequence of input and output
    input Formula , T , pls
    random initialization 
    pls -> local
    1-pls -> metropolis
    
    check -> valid
    output 
    '''