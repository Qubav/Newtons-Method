import sympy as sp
from krok_goldstein import goldstein
from hess import podwyznaczniki_hess
from hess import hesjan
from get_grad import get_grad
from sympy.parsing.sympy_parser import (parse_expr,
standard_transformations, implicit_application)
transformations = standard_transformations + (implicit_application,)
from sympy import *
init_printing

def Newton(punkt_poczatkowy, eps1, eps2, eps3, L, beta, tau0, funkcja, logger):
    parser = sp.parse_expr(funkcja) # parser - funkcja po przetworzeniu parserem
    xk = punkt_poczatkowy           # wprowadzony punkt początkowy przyjęty jako xk
    rozmiar = sp.shape(punkt_poczatkowy)[0]
    zmienne = []
    gradient = []
    for i in range(1, rozmiar + 1):
        zmienne.append(sp.symbols("x" + str(i)))    #utworzenie symboli i wypełnienie nimi macierzy
    podstaw = [("x" + str(i), xk[i - 1]) for i in range(1, rozmiar + 1)]    #macierz zmiennych wykorzystywanych przy podstawianiu
    tau = tau0
    lista_xk = []
    lista_xk.append(xk)
    nic = 0
    epsG = 10 ** (-5)

    # rozpoczęcie pętli
    for k in range (1, L + 1):
        for i in range(0, rozmiar):
            gradient.append(sp.Derivative(parser, zmienne[i], evaluate = True)) # zapełnienie macierzy/vektora gradient pochodnymi funkcji po kolejnych zmiennych w niej występujących
        if k > 1:
            podstaw = [("x" + str(i), xk[i - 1]) for i in range(1, rozmiar + 1)]    # aktualizacja wartości podstawianych
            stare_xk = xk
        wartosc_gradient = get_grad(rozmiar, gradient, podstaw)

        # procedura badania pierwszego kryterium stop
        gradXgrad = (wartosc_gradient.T * wartosc_gradient)[0]   # mnożenie gradientu przez siebie
        if gradXgrad <= eps1:
            # komunikacja spełnienia warunku ...
            hess_pod = hesjan(rozmiar, funkcja)                  #stworzenie hesjanu
            czy_min = podwyznaczniki_hess(hess_pod, xk)          # sprawdzenie czy osiągnięte został minimum przy pomocy kryterium Sylwestra
            # tu jeszcze jakieś akcje dodać
            wart_fuk = parser.subs(podstaw).evalf()
            if rozmiar == 2:
                logger(
                    "Został osiągnięty warunek stop dotyczący wartości iloczynu skalarnego gradientu w punkcie\n "
                    + str(round(xk[0],6)) + ", " + str(round(xk[1], 6))
                    + "\nLiczba iteracji wynosi: "
                    + str(k)
                    + "\nWartość kryterium stopu wynosi:\n"
                    + str(round(gradXgrad,6))
                    + "\nWartość funkcji w punkcie wynosi:\n"
                    + str(round(wart_fuk, 6))
                )
            elif rozmiar == 3:
                logger(
                "Został osiągnięty warunek stop dotyczący wartości iloczynu skalarnego gradientu w punkcie\n "
                + str(round(xk[0],6)) + ", " + str(round(xk[1],6)) + ", " + str(round(xk[2],6))
                + "\nLiczba iteracji wynosi: "
                + str(k)
                + "\nWartość kryterium stopu wynosi:\n"
                + str(round(gradXgrad,6))
                + "\nWartość funkcji w punkcie wynosi:\n"
                + str(round(wart_fuk, 6))
            )
            elif rozmiar == 4:
                logger(
                "Został osiągnięty warunek stop dotyczący wartości iloczynu skalarnego gradientu w punkcie\n "
                + str(round(xk[0],6)) + ", " + str(round(xk[1],6)) + ", " + str(round(xk[2],6)) + ", " + str(round(xk[3],6))
                + "\nLiczba iteracji wynosi: "
                + str(k)
                + "\nWartość kryterium stopu wynosi:\n"
                + str(round(gradXgrad,6))
                + "\nWartość funkcji w punkcie wynosi:\n"
                + str(round(wart_fuk, 6))
            )
            elif rozmiar == 5:
                    logger(
                "Został osiągnięty warunek stop dotyczący wartości iloczynu skalarnego gradientu w punkcie\n "
                + str(N(xk[0],6)) + ", " + str(N(xk[1],6)) + ", " + str(N(xk[2],6)) + ", " + str(N(xk[3],6)) + ", " + str(N(xk[4],6))
                + "\nLiczba iteracji wynosi: "
                + str(k)
                + "\nWartość kryterium stopu wynosi:\n"
                + str(round(gradXgrad,6))
                + "\nWartość funkcji w punkcie wynosi:\n"
                + str(round(wart_fuk, 6))
            )
            if czy_min != True:
                logger(
                    "\nZnaleziony punkt nie jest minimum. Hesjan w punkcie nie jest dodatnio określony.\n"
                )
            return{"punkty": lista_xk, "warunek": 1, "czy_min": czy_min}
        
        # procedura wyliczania kolejnego punktu
        hess_pod = hesjan(rozmiar, funkcja)
        odwr_hes = hess_pod.inv() 
        wartosci_hess = odwr_hes.subs(podstaw) # uzupełnienie hesjanu wartościami współrzędnych dla xk
        # odwrócenie hesjanu(xk)
        dk = -wartosci_hess * wartosc_gradient           # obliczenie kierunku
        tau = goldstein(funkcja, xk, dk, beta, tau, epsG, nic)   # obliczenie wartości kroku przy pomocy algortytmu goldsteina
        xk = xk + tau * dk                          # poprzednie xk nazwane stare_xk, więc nowe xk spełnia rolę xk+1
        lista_xk.append(xk)

        # kolejne kryteria stopu
        if k > 1:
            # procedura badania drugiego kryterium
            odleglosc_punktow = 0   # odległość punktów między sobą
            for i in range(0, rozmiar):
                odleglosc_punktow = odleglosc_punktow + (xk[i] - stare_xk[i]) ** 2  # sumowanie kwadratów różnic wartości poszczególnych współrzędnych pukntu xk+1 i xk 
            odleglosc_punktow = sp.sqrt(odleglosc_punktow)  # odległość wyliczana jest poprzez pierwiastek z wcześniej obliczonej sumy
            if odleglosc_punktow < eps2:
                # tu jakieś akcje
                hess_pod = hesjan(rozmiar, funkcja)                  #stworzenie hesjanu
                czy_min = podwyznaczniki_hess(hess_pod, xk)          # sprawdzenie czy osiągnięte został minimum przy pomocy kryterium Sylwestra
                wart_fuk = parser.subs(podstaw).evalf()
                if rozmiar == 2:
                    logger(
                        "Został osiągnięty warunek stop dotyczący odległości międy kolejnymi punktami dla punktu:\n"
                        + str(round(xk[0], 6)) + ", " + str(round(xk[1], 6))
                        + "\nLiczba iteracji wynosi: "
                        + str(k)
                        + "\nWartość kryterium stopu wynosi:\n"
                        + str(round(odleglosc_punktow,6))
                        + "\nWartość funkcji w punkcie wynosi:\n"
                        + str(round(wart_fuk, 6))
                    )
                elif rozmiar == 3:
                    logger(
                        "Został osiągnięty warunek stop dotyczący odległości międy kolejnymi punktami dla punktu:\n"
                        + str(round(xk[0], 6)) + ", " + str(round(xk[1], 6)) + ", " + str(round(xk[2], 6))
                        + "\nLiczba iteracji wynosi: "
                        + str(k)
                        + "\nWartość kryterium stopu wynosi:\n"
                        + str(round(odleglosc_punktow,6))
                        + "\nWartość funkcji w punkcie wynosi:\n"
                        + str(round(wart_fuk, 6))
                    )
                elif rozmiar == 4:
                    logger(
                        "Został osiągnięty warunek stop dotyczący odległości międy kolejnymi punktami dla punktu:\n"
                        + str(round(xk[0], 6)) + ", " + str(round(xk[1], 6)) + ", " + str(round(xk[2], 6)) + ", " + str(round(xk[3], 6))
                        + "\nLiczba iteracji wynosi: "
                        + str(k)
                        + "\nWartość kryterium stopu wynosi:\n"
                        + str(round(odleglosc_punktow,6))
                        + "\nWartość funkcji w punkcie wynosi:\n"
                        + str(round(wart_fuk, 6))
                    )
                elif rozmiar == 5:
                    logger(
                        "Został osiągnięty warunek stop dotyczący odległości międy kolejnymi punktami dla punktu:\n"
                        + str(round(xk[0], 6)) + ", " + str(round(xk[1], 6)) + ", " + str(round(xk[2], 6)) + ", " + str(round(xk[3], 6)) + ", " + str(round(xk[4], 6))
                        + "\nLiczba iteracji wynosi: "
                        + str(k)
                        + "\nWartość kryterium stopu wynosi:\n"
                        + str(round(odleglosc_punktow,6))
                        + "\nWartość funkcji w punkcie wynosi:\n"
                        + str(round(wart_fuk, 6))
                    )
                if czy_min != True:
                    logger(
                        "\nZnaleziony punkt nie jest minimum. Hesjan w punkcie nie jest dodatnio określony.\n"
                    )
                return{"punkty": lista_xk, "warunek": 2, "czy_min": czy_min}


            roznica_wartosci = 0
            podstaw_nowe = [("x" + str(i), xk[i - 1]) for i in range(1, rozmiar + 1)]
            roznica_wartosci = abs(parser.subs(podstaw_nowe)-parser.subs(podstaw))
            if roznica_wartosci < eps3:
                hess_pod = hesjan(rozmiar, funkcja)                  #stworzenie hesjanu
                czy_min = podwyznaczniki_hess(hess_pod, xk)          # sprawdzenie czy osiągnięte został minimum przy pomocy kryterium Sylwestra
                wart_fuk = parser.subs(podstaw_nowe).evalf()
                if rozmiar == 2:
                    logger(
                        "Został osiągnięty warunek stop dotyczący różnicy wartości funkcji w kolejnych punktach dla punktu:\n"
                        + str(round(xk[0],6)) + ", " + str(round(xk[1], 6))
                        + "\nLiczba iteracji wynosi: "
                        + str(k)
                        + "\nWartość kryterium stopu wynosi:\n"
                        + str(round(roznica_wartosci,6))
                        + "\nWartość funkcji w punkcie wynosi:\n"
                        + str(round(wart_fuk, 6))
                    )
                elif rozmiar == 3:
                    logger(
                    "Został osiągnięty warunek stop dotyczący różnicy wartości funkcji w kolejnych punktach dla punktu:\n"
                    + str(round(xk[0], 6)) + ", " + str(round(xk[1], 6)) + ", " + str(round(xk[2], 6))
                    + "\nLiczba iteracji wynosi: "
                    + str(k)
                    + "\nWartość kryterium stopu wynosi:\n"
                    + str(round(roznica_wartosci,6))
                    + "\nWartość funkcji w punkcie wynosi:\n"
                    + str(round(wart_fuk, 6))
                )
                elif rozmiar == 4:
                    logger(
                    "Został osiągnięty warunek stop dotyczący różnicy wartości funkcji w kolejnych punktach dla punktu:\n"
                    + str(round(xk[0], 6)) + ", " + str(round(xk[1], 6)) + ", " + str(round(xk[2], 6)) + ", " + str(round(xk[3], 6))
                    + "\nLiczba iteracji wynosi: "
                    + str(k)
                    + "\nWartość kryterium stopu wynosi:\n"
                    + str(round(roznica_wartosci,6))
                    + "\nWartość funkcji w punkcie wynosi:\n"
                    + str(round(wart_fuk, 6))
                )
                elif rozmiar == 5:
                    logger(
                    "Został osiągnięty warunek stop dotyczący różnicy wartości funkcji w kolejnych punktach dla punktu:\n"
                    + str(round(xk[0], 6)) + ", " + str(round(xk[1], 6)) + ", " + str(round(xk[2], 6)) + ", " + str(round(xk[3], 6)) + ", " + str(round(xk[4], 6))
                    + "\nLiczba iteracji wynosi: "
                    + str(k)
                    + "\nWartość kryterium stopu wynosi:\n"
                    + str(round(roznica_wartosci,6))
                    + "\nWartość funkcji w punkcie wynosi:\n"
                    + str(round(wart_fuk, 6))
                )
                if czy_min != True:
                    logger(
                        "\nZnaleziony punkt nie jest minimum. Hesjan w punkcie nie jest dodatnio określony.\n"
                    )
                return{"punkty": lista_xk, "warunek": 3, "czy_min": czy_min}
        
        # tu wypisywanie po każdym okrążeniu
        podstaw_nowe = [("x" + str(i), xk[i - 1]) for i in range(1, rozmiar + 1)]
        wartosc_gradient = get_grad(rozmiar, gradient, podstaw_nowe)
        gradXgrad = (wartosc_gradient.T * wartosc_gradient)[0]
        wart_fuk = parser.subs(podstaw_nowe).evalf()
        if rozmiar == 2:
            logger(
                "Iteracja: "
                + str(k)
                + "\nPunkt: "
                + str(round(xk[0], 6))
                + ", "
                + str(round(xk[1], 6))
                + "\nWartość funkcji: "
                + str(round(wart_fuk, 6))
                + "\nWartość pierwszego kryterium stopu: "
                + str(round(gradXgrad, 6))
                + "\n"
            )
        elif rozmiar == 3:
            logger(
                "Iteracja: "
                + str(k)
                + "\nPunkt: "
                + str(round(xk[0], 6))
                + ", "
                + str(round(xk[1], 6))
                + ", "
                + str(round(xk[2], 6))
                + "\nWartość funkcji: "
                + str(round(wart_fuk, 6))
                + "\nWartość pierwszego kryterium stopu: "
                + str(round(gradXgrad, 6))
                + "\n"
            )
        elif rozmiar == 4:
            logger(
                "Iteracja: "
                + str(k)
                + "\nPunkt: "
                + str(round(xk[0], 6))
                + ", "
                + str(round(xk[1], 6))
                + ", "
                + str(round(xk[2], 6))
                + ", "
                + str(round(xk[3], 6))
                + "\nWartość funkcji: "
                + str(round(wart_fuk, 6))
                + "\nWartość pierwszego kryterium stopu: "
                + str(round(gradXgrad, 6))
                + "\n"
            )
        elif rozmiar == 5:
            logger(
                "Iteracja: "
                + str(k)
                + "\nPunkt: "
                + str(round(xk[0], 6))
                + ", "
                + str(round(xk[1], 6))
                + ", "
                + str(round(xk[2], 6))
                + ", "
                + str(round(xk[3], 6))
                + ", "
                + str(round(xk[4], 6))
                + "\nWartość funkcji: "
                + str(round(wart_fuk, 6))
                + "\nWartość pierwszego kryterium stopu: "
                + str(round(gradXgrad, 6))
                + "\n"
            )
    # jak się nie udało
    logger("\nNie udało się odnaleźć minimum w " + str(k) + " iteracjach.\n")
    return{"punkty": lista_xk, "warunek": 4, "czy_min": czy_min}
    

            



            

            












