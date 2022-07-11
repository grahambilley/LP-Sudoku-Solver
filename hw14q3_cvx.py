# imports
import pandas as pd
import numpy as np
import cvxpy as cp

# define 9x9x9 matrix of binary decision variables
# note: cvxpy does not allow for more than 2 dimensions
# use a dictionary to circumvent this issue
x = {}
for i in range(9):
    x[i] = cp.Variable(shape=(9,9), boolean=True)

# definitions to simplify summation code
l = list(range(9))

# define (4) 9x9 matrices for row constraints, col constraints, square constraints, and box constraints
rowConstr = [[x]*9 for x in [None]*9] # each num appears once per row
colConstr = [[x]*9 for x in [None]*9] # each num appears once per col
sqrConstr = [[x]*9 for x in [None]*9] # each square can only contain a single num
boxConstr = [[x]*9 for x in [None]*9] # each 3x3 box contains one of each num
fixedBoxes = [None]*21 # each num pre-filled 
# constraints
constraints = []

# exactly one entry in each square
for row in l:
    for col in l:
        sqrConstr[row][col] = (1 == cp.sum([x[i][row,col] for i in l]))

# each row includes one of each number
for row in l:
    for num in l:
        rowConstr[num][row] = (1 == cp.sum([x[num][row, i] for i in l]))

# each col includes one of each number
for col in l:
    for num in l:
        colConstr[num][col] = (1 == cp.sum([x[num][i, col] for i in l]))

# each box includes all numbers
for num in l:
    boxConstr[0][num] = (1 == cp.sum(x[num][0:3, 0:3])) # top left
    boxConstr[1][num] = (1 == cp.sum(x[num][0:3, 3:6])) # top mid
    boxConstr[2][num] = (1 == cp.sum(x[num][0:3, 6:9])) # top right
    boxConstr[3][num] = (1 == cp.sum(x[num][3:6, 0:3])) # mid left
    boxConstr[4][num] = (1 == cp.sum(x[num][3:6, 3:6])) # mid mid
    boxConstr[5][num] = (1 == cp.sum(x[num][3:6, 6:9])) # mid right
    boxConstr[6][num] = (1 == cp.sum(x[num][6:9, 0:3])) # bot left
    boxConstr[7][num] = (1 == cp.sum(x[num][6:9, 3:6])) # bot mid
    boxConstr[8][num] = (1 == cp.sum(x[num][6:9, 6:9])) # bot right

# fixed values
fixedBoxes[0] = (x[7][0, 0] == 1)
fixedBoxes[1] = (x[2][1, 2] == 1)
fixedBoxes[2] = (x[5][1, 3] == 1)
fixedBoxes[3] = (x[6][2, 1] == 1)
fixedBoxes[4] = (x[8][2, 4] == 1)
fixedBoxes[5] = (x[1][2, 6] == 1)
fixedBoxes[6] = (x[4][3, 1] == 1)
fixedBoxes[7] = (x[6][3, 5] == 1)
fixedBoxes[8] = (x[3][4, 4] == 1)
fixedBoxes[9] = (x[4][4, 5] == 1)
fixedBoxes[10] = (x[6][4, 6] == 1)
fixedBoxes[11] = (x[0][5, 3] == 1)
fixedBoxes[12] = (x[2][5, 7] == 1)
fixedBoxes[13] = (x[0][6, 2] == 1)
fixedBoxes[14] = (x[5][6, 7] == 1)
fixedBoxes[15] = (x[7][6, 8] == 1)
fixedBoxes[16] = (x[7][7, 2] == 1)
fixedBoxes[17] = (x[4][7, 3] == 1)
fixedBoxes[18] = (x[0][7, 7] == 1)
fixedBoxes[19] = (x[8][8, 1] == 1)
fixedBoxes[20] = (x[3][8, 6] == 1)

# append constraints
for i in fixedBoxes:
    constraints.append(i)
for row in l:
    for col in l:
        constraints.append(sqrConstr[row][col])
        constraints.append(rowConstr[col][row])
        constraints.append(colConstr[col][row])
        constraints.append(boxConstr[row][col])

# define objective function as sum of all decision variables
prob = cp.Problem(cp.Minimize(sum([cp.sum(x[i]) for i in l])), constraints)

prob.solve()
print("Model fitting complete\n----------------------\nSolution:")
solution = np.zeros(shape=(9,9))
for i in l:
    sol_mask = np.round(x[i].value) > 0.99
    solution[sol_mask] = i+1
print(solution)
print("----------------------\nEnd")

# Model fitting complete
# ----------------------
# Solution:
# [[8. 1. 2. 7. 5. 3. 6. 4. 9.]
#  [9. 4. 3. 6. 8. 2. 1. 7. 5.]
#  [6. 7. 5. 4. 9. 1. 2. 8. 3.]
#  [1. 5. 4. 2. 3. 7. 8. 9. 6.]
#  [3. 6. 9. 8. 4. 5. 7. 2. 1.]
#  [2. 8. 7. 1. 6. 9. 5. 3. 4.]
#  [5. 2. 1. 9. 7. 4. 3. 6. 8.]
#  [4. 3. 8. 5. 2. 6. 9. 1. 7.]
#  [7. 9. 6. 3. 1. 8. 4. 5. 2.]]
# ----------------------
# End