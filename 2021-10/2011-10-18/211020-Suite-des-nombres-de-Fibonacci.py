# Amusez vous bien !
# 0 , 1, 1, 2, 3, 5, 8, 13, 21, 34,

def fibonacci(n, arbre):
    res = 0
    arbre.append("calcul de " + str(n))
    if (n <= 1):
        res = n
    else:
        res = fibonacci(n - 1, arbre) + fibonacci(n - 2, arbre)
    return res

# 0 , 1, 1, 2, 3, 5, 8, 13, 21, 34,

arbre = []
n = 5
#fibo = fibonacci(n, arbre)
for idx in range(0, n):
    fibo = fibonacci(idx, arbre)
    print(fibo)

print("--------------------------------------------")
for idx in range(0, n):
    nb = arbre.count("calcul de " + str(idx))
    print("la suite fibo de", idx, "a été calculé", nb, "fois")