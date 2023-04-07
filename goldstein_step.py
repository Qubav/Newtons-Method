import sympy as sp
from get_grad import get_grad

def goldstein(funkcja, punkt_poczatkowy, kierunek, test_beta, tauR, epsilon, logger):
    rozmiar = sp.shape(punkt_poczatkowy)[0] # rozmiar uzależniamy od ilości współrzędnych punktu poczatkowego
    zmienne = []
    gradient = []
    parser = sp.parse_expr(funkcja)
    
    for i in range (0,rozmiar):
        zmienne.append(sp.symbols("x"+ str(i+1))) # zgodnie z tą konwencją wsyzstkie zmienne muszą być wprowadzane jako xn

    for i in range(0, rozmiar):
        gradient.append(sp.Derivative(parser, zmienne[i], evaluate=True))  # Tworzę listę z kolejnych pochodnych cząstkowych
        
    Temp = [("x"+str(i), punkt_poczatkowy[i-1])for i in range(1, rozmiar +1)] # tworzenie zmiennej co przechowuje 1 zmienna czastkowa
    
    grad0 = get_grad(rozmiar,gradient,Temp) # oblicznie gradientu w punkcie zerowym dal kazdego wymiaru

    punkt2 = punkt_poczatkowy + tauR*kierunek
    Temp2 = [("x"+str(i), punkt2[i-1])for i in range(1, rozmiar +1)]
    
    """ p = ((parser.subs(Temp2).evalf()-parser.subs(Temp).evalf())/tauR) """
    
    # krok 1 punkt 1
    p = (grad0.T * kierunek)[0] # na koncu [0] poniewaz iloczyn maciezy zawsze jest macierza a my chcemy by byla skalarem 
    # p - pochodna w kierunku
    
    #krok 1 punkt 2
    tauL = 0
    
    Temp_tauR = [("x" + str(i), (punkt_poczatkowy[i - 1] + tauR * kierunek[i - 1])) for i in range(1, rozmiar + 1)]
    fXzero = parser.subs(Temp).evalf() # evalf okresla ilosc liczb po przecinku natywnie 15 miejsc
    
    if parser.subs(Temp_tauR).evalf() < fXzero:
        return tauR
    
    
    # krok 2
    k = 0
    while k < 10000:

        tau = 1 / 2 * (tauL + tauR)
        if k < 1:
            tau = tauR
        Temp_tau = [("x" + str(i), punkt_poczatkowy[i - 1] + tau * kierunek[i - 1]) for i in range(1, rozmiar + 1)]
        fXtaud = parser.subs(Temp_tau).evalf() # f(x0 + tau*d)
        k += 1

        # krok 3
        if fXtaud < fXzero + (1 - test_beta) * p * tau:
            tauL = tau

        # krok 4
        elif fXtaud > fXzero + test_beta * p * tau:
            tauR = tau

        else:
            return tau

    return 0



