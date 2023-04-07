from tkinter import Y
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr,
standard_transformations, implicit_application)
transformations = standard_transformations + (implicit_application,)
from sympy import *
init_printing

def hesjan(rozmiar,funkcja):

    zmienne = []
    gradient = []
    druga_pochodna_po_x1=  []
    druga_pochodna_po_x2 = []
    druga_pochodna_po_x3 = []
    druga_pochodna_po_x4 = []
    druga_pochodna_po_x5 = []
    
    for i in range (0,rozmiar):
        zmienne.append(sp.symbols("x"+ str(i+1))) 

    parser = parse_expr(funkcja, transformations=transformations)

    for i in range(0, rozmiar):
        gradient.append(sp.Derivative(parser, zmienne[i], evaluate=True)) 
        
    for i in range(0, rozmiar):
            druga_pochodna_po_x1.append((sp.Derivative(gradient[i], zmienne[0], evaluate=True)))
            if rozmiar > 1:
                druga_pochodna_po_x2.append((sp.Derivative(gradient[i], zmienne[1], evaluate=True)))
            if rozmiar > 2:
                druga_pochodna_po_x3.append((sp.Derivative(gradient[i], zmienne[2], evaluate=True)))
            if rozmiar > 3:
                druga_pochodna_po_x4.append((sp.Derivative(gradient[i], zmienne[3], evaluate=True)))
            if rozmiar > 4:
                druga_pochodna_po_x5.append((sp.Derivative(gradient[i], zmienne[4], evaluate=True)))
            
    if rozmiar == 2:
        hess2 = sp.Matrix(
            [
                [druga_pochodna_po_x1[0], druga_pochodna_po_x2[0]],
                [druga_pochodna_po_x1[1], druga_pochodna_po_x2[1]]
                ]
            )
        return hess2  
    
    if rozmiar == 3:
        hess3 = sp.Matrix(
            [
                [druga_pochodna_po_x1[0], druga_pochodna_po_x2[0], druga_pochodna_po_x3[0]],
                [druga_pochodna_po_x1[1], druga_pochodna_po_x2[1], druga_pochodna_po_x3[1]],
                [druga_pochodna_po_x1[2], druga_pochodna_po_x2[2], druga_pochodna_po_x3[2]]
            ]
            )
        return hess3
    
    if rozmiar == 4:
        hess4 = sp.Matrix(
            [
                [druga_pochodna_po_x1[0], druga_pochodna_po_x2[0], druga_pochodna_po_x3[0], druga_pochodna_po_x4[0]],
                [druga_pochodna_po_x1[1], druga_pochodna_po_x2[1], druga_pochodna_po_x3[1], druga_pochodna_po_x4[1]],
                [druga_pochodna_po_x1[2], druga_pochodna_po_x2[2], druga_pochodna_po_x3[2], druga_pochodna_po_x4[2]],
                [druga_pochodna_po_x1[3], druga_pochodna_po_x2[3], druga_pochodna_po_x3[3], druga_pochodna_po_x4[3]]
                ]
            )
        return hess4
    
    if rozmiar == 5:
        hess5 = sp.Matrix(
            [
                [druga_pochodna_po_x1[0], druga_pochodna_po_x2[0], druga_pochodna_po_x3[0], druga_pochodna_po_x4[0], druga_pochodna_po_x5[0]],
                [druga_pochodna_po_x1[1], druga_pochodna_po_x2[1], druga_pochodna_po_x3[1], druga_pochodna_po_x4[1], druga_pochodna_po_x5[1]],
                [druga_pochodna_po_x1[2], druga_pochodna_po_x2[2], druga_pochodna_po_x3[2], druga_pochodna_po_x4[2], druga_pochodna_po_x5[2]],
                [druga_pochodna_po_x1[3], druga_pochodna_po_x2[3], druga_pochodna_po_x3[3], druga_pochodna_po_x4[3], druga_pochodna_po_x5[3]],
                [druga_pochodna_po_x1[4], druga_pochodna_po_x2[4], druga_pochodna_po_x3[4], druga_pochodna_po_x4[4], druga_pochodna_po_x5[4]]
                ]
            )
        return hess5
    
def podwyznaczniki_hess(hess,punkty):
    
    rozmiar = punkty.shape[0]
    punkt = [("x" + str(i), punkty[i - 1]) for i in range(1, rozmiar + 1)]
    
    if hess[0].subs(punkt).evalf() <= 0:# tu za kazdym razem sprawdzamy 1 miejsce w tabeli
        return False
    
    if rozmiar > 1 and hess.subs(punkt).det().evalf() <=0:# w przypadku gdy hesian jest wiekszy niż 1 na 1 obliczamy jego cały wyznacznik
        return False
      
    if rozmiar > 2: # gdy rozmiar hess 3x3 sprawdzamy czy 2x2 tez jest dodatni
        hess_dwa_na_dwa = sp.Matrix(
            [
                [hess[0,0],hess[0,1]],
                [hess[1,0],hess[1,1]]
            ]
        )
        if hess_dwa_na_dwa.subs(punkt).det().evalf() <=0:
            return False
    if rozmiar > 3 : # tu dodajemy by sprawdzal jeszcze 3 na 3 bo jest po drodze
        hess_3_na_3 = sp.Matrix(
            [
                [hess[0,0],hess[0,1],hess[0,2]],
                [hess[1,0],hess[1,1],hess[1,2]],
                [hess[2,0],hess[2,1],hess[2,2]]
            ]
        )
        if hess_3_na_3.subs(punkt).det().evalf() <=0:
            return False
        
    if rozmiar > 4 :
        hess_4_na_4 = sp.Matrix(
            [
                [hess[0,0],hess[0,1],hess[0,2],hess[0,3]],
                [hess[1,0],hess[1,1],hess[1,2],hess[1,3]],
                [hess[2,0],hess[2,1],hess[2,2],hess[2,3]],
                [hess[3,0],hess[3,1],hess[3,2],hess[3,3]]
            ]
        )
        if hess_4_na_4.subs(punkt).det().evalf() <=0:
            return False
        
    return True
    
def podstawienie_hess(hess,punkty): 
     
    rozmiar = punkty.shape[0]
    punkt = [("x" + str(i), punkty[i - 1]) for i in range(1, rozmiar + 1)]
    if rozmiar == 2:
        hess1 = hess.subs(punkt).det()
    if rozmiar == 3:
        hess1 = hess.subs(punkt).det()
    if rozmiar == 4:
        hess1 = hess.subs(punkt).det()
    if rozmiar == 5:
        hess1 = hess.subs(punkt).det()
        
    return hess1