import glob
import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QApplication
import ui.mainui as mainui
import shutil
import string
import random
import os
path=''
from speechbrain.pretrained import SpeakerRecognition

class MatchThread(QThread):
    setprogress = Signal(int)
    setlog = Signal(str)
    setMatch = Signal(str)
    def __init__(self, parent=None):
        super(QThread, self).__init__()
    def run(self):
        global  path
        print(path)
        count=0
        for filename in glob.glob(os.path.join('./upload/', '*.wav')):
            count=count+1
        self.setprogress.emit(0)
        perslot=100/count
        pbarValue=0
        highest=0
        highestFilename = 0
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                       savedir="pretrained_models/spkrec-ecapa-voxceleb")
        for filename in glob.glob(os.path.join('./upload/', '*.wav')):
            pbarValue = pbarValue+perslot

            print(path,filename.replace("\\", "/"))
            score, prediction = verification.verify_files(path,
                                                          filename.replace("\\", "/"))
            # Different Speakers
            if(highest<score[0]):
                highest=score[0]
                highestFilename=filename
            self.setprogress.emit(int(pbarValue))
            self.setlog.emit(filename)
        self.setMatch.emit(highestFilename)
    def stop(self):
        print('stop')

class mainWindow(QMainWindow,mainui.Ui_MainWindow ):

    def __init__(self, parent=None):
        super(mainWindow, self).__init__(parent)
        self.setupUi(self)
        self.search.clicked.connect(self.defMatch)
        self.upload.clicked.connect(self.defCopy)
        self.threadmatch = MatchThread()
        self.threadmatch.setprogress.connect(self.defsetprogress)
        self.threadmatch.setlog.connect(self.defsetlog)
        self.threadmatch.setMatch.connect(self.defsetMatch)

    def defCopy(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Audio files (*.wav )")
        # start the thread
        global path
        path = fname[0]
        print(path)
        N = 7
        res = str(''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k=N)))
        print(res)
        shutil.copyfile(path, './upload/'+res+'.wav')
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Wav uploaded")
        msg.setInformativeText("")
        msg.setWindowTitle("MessageBox")
        msg.setDetailedText("Wav file upload Complete")
        msg.show()
    def defMatch(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Audio files (*.wav )")
        # start the thread
        global path
        path = fname[0]
        self.threadmatch.start()
    def defsetprogress(self,value):
        self.progressBar.setValue(value)
    def defsetlog(self,txt):
        self.log.addItem(txt)
    def defsetMatch(self,txt):
        self.Matched.setText(txt)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    form = mainWindow()
    form.show()
    app.exec_()