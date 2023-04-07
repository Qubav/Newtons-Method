import sympy as sp

def function_string_parser(function_string: str):
    x1, x2 = sp.symbols("x1,x2")
    f = sp.lambdify([x1, x2], function_string)
    return f
