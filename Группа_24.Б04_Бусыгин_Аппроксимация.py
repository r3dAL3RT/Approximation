#Бусыгин Станислав Михайлович, вариант-3

from numpy import *
import matplotlib.pyplot as plt 
def f(x):
    return sqrt(x+1) - cos(x)

delta = 0.07

#DATA
X = linspace(-1,1,50)
y_data = []; x_data = []
for x in X:
    y  = f(x)
    y1 = y + random.uniform(-delta, delta, 3)
    x_data.append(x); x_data.append(x); x_data.append(x)
    y_data.extend(y1)

#нормальные уравнения
def norm(x,y,n):
    E = vander(x,n+1,increasing=True)
    ls = E.T @ E
    rs = E.T @ y

    coef = linalg.solve(ls,rs)
    return coef

#ортогональные полиномы
def ort(X, y, n):
    Q = zeros((n+1,len(X)))
    alpha = zeros(n+1)
    beta = zeros(n)
    Q[0] = 1
    alpha[1] = sum(X)/len(X)
    Q[1] = X - alpha[1]
    

    for i in range(1,n):
        alpha[i+1] = sum(X*Q[i]**2)/sum(Q[i]**2)
        beta[i] = sum(X*Q[i]*Q[i-1])/sum(Q[i-1]**2)

        Q[i+1] = X*Q[i] - alpha[i+1]*Q[i] - beta[i]*Q[i-1]
    
    coef = zeros(n+1)
    for k in range(n+1):
        coef[k] = sum(Q[k]*y)/sum(Q[k]**2)
    return Q, coef, alpha, beta

#вычисление значений полинома
def eval_pol(x_new, coef, alpha, beta, n):
    val = zeros_like(x_new)
    if n >= 0:
        q_prev2 = ones_like(x_new)  # q_0
        val += coef[0] * q_prev2
        
    if n >= 1:
        q_prev1 = x_new - alpha[1]     # q_1
        val += coef[1] * q_prev1
        
        for k in range(2, n + 1):
            q_curr = x_new * q_prev1 - alpha[k] * q_prev1 - beta[k-1] * q_prev2
            val += coef[k] * q_curr
            q_prev2, q_prev1 = q_prev1, q_curr
            
    return val

for n in range(1,6):
    
    c_norm = norm(x_data, y_data, n)
    y_pred_norm = polyval(c_norm[::-1], x_data)  # polyval ожидает [a_n ... a_0]
    S_norm = sum((y_data - y_pred_norm)**2)
    
    # Ортогональные полиномы
    _, coeffs_ort, alpha, beta = ort(x_data, y_data, n)
    y_pred_ort = eval_pol(x_data, coeffs_ort, alpha, beta, n)
    S_ort = sum((y_data - y_pred_ort)**2)
    
    print(f"{n:<4} | {S_norm:<18.6f} | {S_ort:<18.6f}")
    
    # Построение графика
    x_plot = linspace(-1, 1, 300)
    plt.figure(figsize=(8,5))
    plt.scatter(x_data, y_data, s=10, alpha=0.5, label='Эксперимент')
    plt.plot(x_plot, polyval(c_norm[::-1], x_plot), 'r-', label='Норм. ур.')
    plt.plot(x_plot, eval_pol(x_plot, coeffs_ort, alpha, beta, n), 'b--', label='Орт. полиномы')
    plt.title(f'Аппроксимация, степень n={n}')
    plt.legend(); plt.grid(True); plt.show()




