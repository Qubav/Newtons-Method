import imp
from UIMW import Ui_MainWindow
import sys
from newtons_method import Newton
import sympy as sp
from functionStringParser import function_string_parser
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5 import QtWidgets
from canvas import Canvas

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Metoda Newtona")
        self.czykliknol()
    
    def start(self):
        
        beta = self.ui.Beta.text()
        Eps1 = self.ui.Eps1.text()
        Eps2 = self.ui.Eps2.text()
        Eps3 = self.ui.Eps3.text()
        tau = self.ui.tau.text()
        pp = self.ui.punkty_p.text()
        funkcja = self.ui.funkcja.text()
        L = self.ui.L.text()
        try:
            start_vec = sp.Matrix(list(map(lambda x: float(x), list(pp.split(", ")))))
        except:
            self.addLog("Blad we wprowadzanych parametrach")
            return
        self.ui.Logi.setText("")
        result = None
        try:
            result = Newton(
                start_vec, 
                float(Eps1),
                float(Eps2),
                float(Eps3),
                int(L), 
                self.teksst(beta), 
                float(tau), 
                funkcja,
                self.addLog,
            )
        except:
            self.addLog("Blad w parametrach wejsciowych")
            return

        rozmiar = sp.shape(start_vec)[0]
        if rozmiar == 2:
            points = list(map(lambda point: [point[0], point[1]], result["punkty"]))
            x2 = list(map(lambda point: point[0], points))
            y2 = list(map(lambda point: point[1], points))
            
            try:
                self.addCanvas(funkcja, x2, y2)
            except:
                self.addLog("Wymiar problemu > 2")
        else:
            self.addLog("Wymiar problemu > 2")
        
    def czykliknol(self):
        self.ui.oblicz.clicked.connect(self.start)  
          
    def addLog(self, log: str):
        self.ui.Logi.append("\n" + log)
        
    def teksst(self,test: str):
        return float(sp.parse_expr(test))  
    
    def addCanvas(self, line_func: str, x2, y2):
        func = function_string_parser(line_func)   
        sc = Canvas(self, fun=func, width=5, height=4, dpi=100, x2=x2, y2=y2)
        toolbar = NavigationToolbar(sc, self)
        layout = self.ui.box.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout()

        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().deleteLater()

        layout.addWidget(toolbar)
        layout.addWidget(sc)
        self.ui.box.setLayout(layout)
            
if __name__ == "__main__":
    import sys    
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(win)
    win.show()
    sys.exit(app.exec_())