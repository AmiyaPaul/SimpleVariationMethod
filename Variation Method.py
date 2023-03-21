import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

a = 0
b = 10
h = 0.00001
X = np.arange(a, b, h)

def realpsi(x, l=b-a):
    return np.sqrt(2 / l) * np.sin(np.pi * x / l)



def psi1_func(x, l=b-a):
    pp = []
    for i in range(len(x)):
        if b < x[i] < a:
            pp.append(0)
        else:
            k = x[i] * (l - x[i])
            pp.append(k)
    return np.array(pp)


def psi2_func(x, l=b):
    pp = []
    for i in range(len(x)):
        if b < x[i] < a:
            pp.append(0)
        else:
            k = x[i] ** 2 * (l - x[i]) ** 2
            pp.append(k)
    return np.array(pp)


def hamiltonian(x, psi):
    dpsi = []
    k1 = -0.5 * (psi[1] - 2 * psi[0]) / h ** 2
    dpsi.append(k1)
    for i in range(1, len(x) - 1):
        fph = psi[i + 1]
        fmh = psi[i - 1]
        fx = psi[i]

        dd = -0.5 * (fph + fmh - 2 * fx) / h ** 2
        dpsi.append(dd)
    k2 = -0.5 * (psi[-2] - 2 * psi[-1]) / h ** 2
    dpsi.append(k2)
    return np.array(dpsi)


def integration(x, fnA, fnB):
    fA = fnA[0] * fnB[0]
    fB = fnA[-1] * fnB[-1]
    fodd = 0
    feve = 0
    for i in range(1, len(x) - 1):
        if i % 2 == 0:
            feve += 2 * fnA[i] * fnB[i]
        else:
            fodd += 4 * fnA[i] * fnB[i]

    intg = (fA + fB + fodd + feve) * h / 3

    return intg


psi1 = psi1_func(X)
psi2 = psi2_func(X)
dpsi1 = hamiltonian(X, psi1)
dpsi2 = hamiltonian(X, psi2)

H11 = integration(X, psi1, dpsi1)
S11 = integration(X, psi1, psi1)

H12 = integration(X, psi1, dpsi2)
S12 = integration(X, psi1, psi2)

H22 = integration(X, psi2, dpsi2)
S22 = integration(X, psi2, psi2)


def energy_cal_func(e):
    return (H11 - e * S11) * (H22 - e * S22) - (H12 - e * S12) ** 2


def energy_cal_def(e):
    return 2 * S12 * (H12 - e * S12) - S11 * (H22 - e * S22) - S22 * (H11 - e * S11)



# Energy Problem
xold = 0
for i in range(50):
    xnew = xold - (energy_cal_func(xold) / energy_cal_def(xold))

    if abs(xnew - xold) < 1E-7:
        break

    xold = xnew

en = xold

print(f"The energy is {en}")

aa = H11 - en * S11
ab = H12 - en * S12
bb = H22 - en * S22

ks = -ab / bb

c1 = np.sqrt(1 / (1 + ks**2))
c2 = 1 - c1**2

print(f"C1 {c1}", f"C2 {c2}")

psiTotal = c1 * np.sqrt(1 / S11) * psi1 + c2 * np.sqrt(1 / S22) * psi2


plt.plot(X, psiTotal, '-r')
plt.plot(X, realpsi(X), '-b')
plt.show()




