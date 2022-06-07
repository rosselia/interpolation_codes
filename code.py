from re import M
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# Newton's Divided Differences Polynomial Method

x = [20, 21, 24, 26, 27, 28]
y = [18, 17, 15, 14, 16, 15]


def getNDDCoeffs(x, y):
    n = np.shape(y)[0]
    pyramid = np.zeros([n, n])
    pyramid[::, 0] = y
    for j in range(1, n):
        for i in range(n-j):
            pyramid[i][j] = (pyramid[i+1][j-1] - pyramid[i]
                             [j-1]) / (x[i+j] - x[i])

    return pyramid[0]


coeff_vector = getNDDCoeffs(x, y)

final_pol = np.polynomial.Polynomial([0.])
n = coeff_vector.shape[0]
for i in range(n):
    p = np.polynomial.Polynomial([1.])
    for j in range(i):
        p_temp = np.polynomial.Polynomial([-x[j], 1.])
        p = np.polymul(p, p_temp)
    p *= coeff_vector[i]
    final_pol = np.polyadd(final_pol, p)

p = np.flip(final_pol[0].coef, axis=0)

x_axis = np.linspace(20, 28)
y_axis = np.polyval(p, x_axis)

plt.title('Newtons Divided Differences Polynomial Method')
plt.plot(x_axis, y_axis)
plt.grid()
plt.show()


print()
print('# Newtons Divided Differences Polynomial Method')
print('v(23) = ', np.polyval(p, 23))

print('-------------------------------------------------')

# Lagrange's interpolation

x = np.array([20, 21, 24, 26, 27, 28])
y = np.array([18, 17, 15, 14, 16, 15])

xs = np.linspace(np.min(x), np.max(x), 100)

inter = interpolate.lagrange(x, y)
ys = inter(xs)


def Lagrange(x, y, xi):
    n = np.size(x)
    p = 0
    for i in range(n):
        z = y[i]
        for j in range(n):
            if i != j:
                z = z*(xi-x[j])/(x[i]-x[j])
        p = p+z
    return p


days = 23
deaths = Lagrange(x, y, days)
print()
print('# Lagranges interpolation')
print('v(23) = ', deaths)

print('-------------------------------------------------')

fig = plt.figure()
plt.plot(xs, ys, '-', label='Lagrange')
plt.plot(x, y, '.', label='Datos')
plt.plot(days, deaths, 's', label='Interpolacion')
plt.text(days, deaths, '  deaths='+str(deaths))


plt.title('Lagrange Interpolation Polynomial')
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.grid()
plt.show()

# Direct Method Interpolation

requested_date = 23

m = np.ones((6, 6), dtype=int)
m[:, 1] = x

square = np.square(x)
m[:, 2] = square

third_power = [3, 3, 3, 3, 3, 3]
tm = np.power(x, third_power)
m[:, 3] = tm

fourth_power = [4, 4, 4, 4, 4, 4]
fm = np.power(x, fourth_power)
m[:, 4] = fm

fifth_power = [5, 5, 5, 5, 5, 5]
ffm = np.power(x, fifth_power)
m[:, 5] = ffm

x = np.linalg.solve(m, y)

result = (x[0])+(x[1] * (requested_date))+(x[2]*(requested_date)**2) + \
    (x[3]*(requested_date)**3) + \
    (x[4]*(requested_date)**4)+(x[5]*(requested_date)**5)

print()
print('# Direct Method Interpolation')
print('v(23) = ', result)
print()
x = [20, 21, 24, 26, 27, 28]
fig = plt.figure()
plt.plot(xs, ys, '-')
plt.plot(x, y, '.')
plt.plot(days, deaths, 's')
plt.text(days, deaths, '  deaths='+str(result))


plt.title('Direct Method')
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.grid()
plt.show()
