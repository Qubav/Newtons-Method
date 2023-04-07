import matplotlib
import numpy as np
import math

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Canvas(FigureCanvas):
    def __init__(self,parent = None, fun=None, width=5, height=5, dpi=1000, x2=None, y2=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)

        rozmiar = len(x2)
        wymX = math.fabs(x2[0] - x2[rozmiar - 1])
        wymY = math.fabs(y2[0] - y2[rozmiar - 1])

        if wymX > wymY:
            wymiar = wymX
        elif wymX < wymY:
            wymiar = wymY
        elif wymX == wymY:
            wymiar = wymX

        wymiar = wymiar * 1.5

        if x2[rozmiar - 1] - wymiar < 0:
            sX = (-1) * math.fabs(x2[rozmiar - 1] - wymiar) # start X
        else:
            sX = math.fabs(x2[rozmiar - 1] - wymiar) # start X
        if x2[rozmiar - 1] + wymiar < 0:
            kX = (-1) * math.fabs(x2[rozmiar - 1] + wymiar) # koniec X
        else:
            kX = math.fabs(x2[rozmiar - 1] + wymiar) # koniec X
        if y2[rozmiar - 1] - wymiar < 0:
            sY = (-1) * math.fabs(y2[rozmiar - 1] - wymiar)
        else:
            sY = math.fabs(y2[rozmiar - 1] - wymiar)
        if y2[rozmiar - 1] + wymiar < 0:
            kY = (-1) * math.fabs(y2[rozmiar - 1] + wymiar)
        else:
            kY = math.fabs(y2[rozmiar - 1] + wymiar)

        x1 = np.linspace(sX, kX, 100)
        y1 = np.linspace(sY, kY, 100)
        
        try:
            X, Y = np.meshgrid(x1, y1)
            Z = fun(X, Y)
            self.ax.contourf(X, Y, Z, 50, cmap="RdYlGn")
            self.ax.plot(x2, y2, "o-k", linewidth=0.3, markersize=2)
            self.ax.plot(x2[-1], y2[-1], "ok", markersize=5, markerfacecolor="white")
        except:
            pass
        super(Canvas, self).__init__(fig)
