import sympy as sp
#zwracana jest macierz uzupełniona wartościoami poszczególnych pochodnych cząstkowych po podstawieniu
#gradient to macierz w której są pochodne cząstkowe funkcji w kolejności od x1 do x5

def get_grad(rozmiar, gradient, podstaw):
    grad03 = 0
    grad04 = 0
    grad05 = 0

    # obliczenie wartości poszczególnych pochodnych cząstkowych
    grad01 = gradient[0].subs(podstaw).evalf()
    grad02 = gradient[1].subs(podstaw).evalf()
    if rozmiar > 2:
        grad03 = gradient[2].subs(podstaw).evalf()
    if rozmiar > 3:
        grad04 = gradient[3].subs(podstaw).evalf()
    if rozmiar > 4:
        grad05 = gradient[4].subs(podstaw).evalf()

    # uzupełnienie zwracanej macierzy wartościami
    if rozmiar == 2:
        grad0 = sp.Matrix([grad01, grad02])
    if rozmiar == 3:
        grad0 = sp.Matrix([grad01, grad02, grad03])
    if rozmiar == 4:
        grad0 = sp.Matrix([grad01, grad02, grad03, grad04])
    if rozmiar == 5:
        grad0 = sp.Matrix([grad01, grad02, grad03, grad04, grad05])
    return grad0
