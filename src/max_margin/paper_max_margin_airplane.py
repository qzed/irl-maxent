import numpy as np
from scipy.linalg import block_diag
from scipy import optimize
import cvxopt

# Actions: 
# 0: insert tail wing in body
# 1: screw tail wing to body
# 2: insert main wing in body
# 3: screw main wing to body
# 4: insert wing tip in main wing
# 5: screw propeller to base
# 6: screw propeller cap to base 
# 7: screw base to body
# 8: attach bombs to wingtip
act = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Feature matrices (rewards):
phi_p = np.array([[1.0, 1.0, 1.0, 1.0, 0.9, 0.0, 0.0, 1.0, 0.9],
                  [1.0, 1.0, 1.0, 1.0, 0.9, 0.0, 0.0, 1.0, 0.9],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.9],
                  [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.9],
                  [0.9, 0.9, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                  [0.9, 0.9, 0.9, 0.9, 1.0, 0.0, 0.0, 0.0, 1.0]]) # part
phi_t = np.array([[1, 0, 1, 0, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 1, 0, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 1, 0, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 1, 0, 0],
                  [0, 1, 0, 1, 0, 1, 1, 0, 0],
                  [1, 0, 1, 0, 1, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 0, 1, 1]]) # tool
phi_m = np.array([[1, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 1, 1, 0],
                  [1, 0, 1, 0, 1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0, 1, 1, 1, 0],
                  [0, 1, 0, 1, 0, 1, 1, 1, 0],
                  [0, 1, 0, 1, 0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1]]) # motion
phi_l = np.array([[1.0, 1.0, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.8],
                  [1.0, 1.0, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.8],
                  [0.8, 0.8, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0],
                  [0.8, 0.8, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0],
                  [0.8, 0.8, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0],
                  [0.5, 0.5, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 0.3],
                  [0.5, 0.5, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 0.3],
                  [0.5, 0.5, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 0.3],
                  [0.8, 0.8, 1.0, 1.0, 1.0, 0.3, 0.3, 0.3, 1.0]]) # location
phi_e = np.array([[1.0, 0.8, 1.0, 0.8, 1.0, 0.2, 0.8, 1.0, 1.0],
                  [0.8, 1.0, 0.8, 1.0, 0.8, 0.4, 1.0, 0.8, 0.8],
                  [1.0, 0.8, 1.0, 0.8, 1.0, 0.2, 0.8, 1.0, 1.0],
                  [0.8, 1.0, 0.8, 1.0, 0.8, 0.4, 1.0, 0.8, 0.8],
                  [1.0, 0.8, 1.0, 0.8, 1.0, 0.2, 0.8, 1.0, 1.0],
                  [0.2, 0.4, 0.2, 0.4, 0.2, 1.0, 0.4, 0.2, 0.2],
                  [0.8, 1.0, 0.8, 1.0, 0.8, 0.4, 1.0, 0.8, 0.8],
                  [1.0, 0.8, 1.0, 0.8, 1.0, 0.2, 0.8, 1.0, 1.0],
                  [1.0, 0.8, 1.0, 0.8, 1.0, 0.2, 0.8, 1.0, 1.0]]) # effort

# Preconditions (transitions)
# T = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 1, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0]])
T = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Demonstration
demo = [0, 2, 4, 1, 3, 5, 6, 7, 8]
# demo = [1, 2, 3, 4, 5, 9, 6, 7, 8]
num_d = len(demo) - 1

# Max margin
A = []
S = []
for i in range(num_d):
    prev = demo[i]
    next = demo[i+1]
    candidates = demo[i+2:]
    for other in candidates:
        t = np.argwhere(T[other,:])
        if t.size == 0 or t in demo[:i+1]:
            a = [-phi_p[prev,next]+phi_p[prev,other],
                 -phi_t[prev,next]+phi_t[prev,other],
                 -phi_m[prev,next]+phi_m[prev,other],
                 -phi_l[prev,next]+phi_l[prev,other],
                 -phi_e[prev,next]+phi_e[prev,other]]
            s = np.zeros(num_d-1)
            s[i] = -1
            A.append(a)
            S.append(s)

A = np.array(A)
S = np.array(S)        


_, n_w = A.shape
_, n_b = S.shape
W = np.hstack((-1*np.eye(n_w), np.zeros((n_w, n_b))))
A = np.hstack((A, S))
# MATLAB % [A_new, ia, ic] = unique(A, 'rows', 'stable');
# MATLAB % S_new = S(ia, :);
# MATLAB % A = [A_new, S_new];
n_con, n_x = A.shape
C = 3.5
H = np.eye(5)
Hs = 2*C*np.eye(num_d-1)
H = block_diag(H, Hs)
f = np.zeros((1, n_x))
b = -1*np.ones((n_con, 1))
b_W = np.zeros((n_w, 1))
# MATLAB % x = quadprog(H,f,A,b)
# MATLAB x = quadprog(H,f,[A; W],[b; b_W]) % uses 'interior-point-convex' algorithm by default (https://www.mathworks.com/help/optim/ug/quadprog.html)

b_stack = np.vstack((b, b_W))
A_stack = np.vstack((A, W))

# # Doesn't work, gives all zero result
# x0 = np.random.randn(n_x,1)
# def fun(x):
#     return 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(f, x)
# cons = [{'type':'ineq', 'fun':lambda x: b_stack[i] - np.dot(A_stack[i], x)}
#         for i in range(b_stack.shape[0])]
# result = optimize.minimize(fun, x0, constraints=cons)
# x = result['x']

# Using interior-point algorithms (http://cvxopt.org/documentation/index.html#technical-documentation)
cvxopt.solvers.options['show_progress'] = False
x = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f.T), cvxopt.matrix(A_stack), cvxopt.matrix(b_stack))['x']
x = np.array(x)

print(x)


# Predict
w = x[:5]
candidates = set(act)
pred = []
prev = 0
candidates.remove(prev)
while not len(candidates)==0:
    pred.append(prev)
    r_max = -100
    for other in candidates:
        t = np.argwhere(T[other,:])  # precondition of candidate
        if t.size==0 or t in pred:
            a = [phi_p[prev,other],
                 phi_t[prev,other],
                 phi_m[prev,other],
                 phi_l[prev,other],
                 phi_e[prev,other]]
            r = np.dot(a, w)
            if r > r_max:
                r_max = r
                next = other
    candidates.remove(next)
    prev = next

pred.append(prev)

print(pred)