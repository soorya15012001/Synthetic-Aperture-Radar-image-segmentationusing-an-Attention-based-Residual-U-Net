#In terminal type - set tuiview_allow_nogeo=yes
#Run from terminal

import sys
from PyQt5.QtWidgets import QApplication
# noinspection PyUnresolvedReferences
from tqdm import tqdm
from tuiview import geolinkedviewers
import os

os.popen("set TUIVIEW_ALLOW_NOGEO=YES")

app = QApplication(sys.argv)

X = "./data_final/data/train_full/img/"
Y = "./data_final/data/train_full/mask/"

X_png = "./data_final/data/png_full/img/"
Y_png = "./data_final/data/png_full/mask/"

def convert(X, X_png):
    for i in tqdm(os.listdir(X)):
        imgFile = X+i
        outFile = X_png+i[:-4]+'.png'
        app.setApplicationName('tuiview')
        app.setOrganizationName('TuiView')
        viewers = geolinkedviewers.GeolinkedViewers()
        viewer = viewers.newViewer()
        viewer.addRasterInternal(imgFile)
        viewer.resizeForWidgetSize(869, 869)
        viewer.saveCurrentViewInternal(outFile)
        # break

convert(X, X_png)
convert(Y, Y_png)