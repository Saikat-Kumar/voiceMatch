import glob
import json
import sys

import sqlite3
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
import torchaudio
from speechbrain.pretrained import EncoderClassifier

path=''
name=''
import sqlite3
conn = sqlite3.connect('DB.db')
sql = 'create table if not exists voice  (name VARCHAR(255) NOT NULL,embedding TEXT NOT NULL)'
conn.execute(sql)
from speechbrain.pretrained import SpeakerRecognition
class UploadThread(QThread):
    setlogUpload = Signal(str)
    def __init__(self, parent=None):
        super(QThread, self).__init__()
    def run(self):
        global  path
        print(path)
        conn = sqlite3.connect('DB.db')
        self.setlogUpload.emit('Path ='+path[0])
        self.setlogUpload.emit('classifier Load ')
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="pretrained_models/spkrec-ecapa-voxceleb")
        classifier.hparams.label_encoder.ignore_len()
        self.setlogUpload.emit('Audio Load ')
        # Compute embeddings
        signal, fs = torchaudio.load(path[0])
        self.setlogUpload.emit('Embedding Start....')
        embeddings = classifier.encode_batch(signal)
        self.setlogUpload.emit('Embedding Complete')
        listEm=embeddings[0][0].tolist()
        sql_as_text = json.dumps(listEm)

        global name
        try:
            conn.execute("INSERT INTO voice (name,embedding) VALUES (?, ?)",(name, sql_as_text))
        except sqlite3.Error as er:
            print('SQLite error: %s' % (' '.join(er.args)))

        self.setlogUpload.emit('Database Saved')



    def stop(self):
        print('stop')
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
        self.setprogress.emit(0)
        perslot=100/count
        pbarValue=0
        highest=0
        highestFilename = 0
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                       savedir="pretrained_models/spkrec-ecapa-voxceleb")
        conn = sqlite3.connect('DB.db')
        conn.execute('SELECT * FROM voice')
        # for row in conn:
        #     self.setlog.emit(row)
        # for filename in glob.glob(os.path.join('./upload/', '*.wav')):
        #     pbarValue = pbarValue+perslot
        #
        #     print(path,filename.replace("\\", "/"))
        #     score, prediction = verification.verify_files(path,
        #                                                   filename.replace("\\", "/"))
        #     # Different Speakers
        #     if(highest<score[0]):
        #         highest=score[0]
        #         highestFilename=filename
        #     self.setprogress.emit(int(pbarValue))
        #     self.setlog.emit(filename)
        self.setMatch.emit(highestFilename)
    def stop(self):
        print('stop')

class mainWindow(QMainWindow,mainui.Ui_MainWindow ):

    def __init__(self, parent=None):
        super(mainWindow, self).__init__(parent)
        self.setupUi(self)
        self.search.clicked.connect(self.defMatch)
        self.upload.clicked.connect(self.defCopy)
        self.submitUpload.clicked.connect(self.defsubmitUplaod)
        self.threadupload = UploadThread()
        self.threadupload.setlogUpload.connect(self.defsetlogupload)
        self.threadmatch = MatchThread()
        self.threadmatch.setprogress.connect(self.defsetprogress)
        self.threadmatch.setlog.connect(self.defsetlog)
        self.threadmatch.setMatch.connect(self.defsetMatch)

    def defCopy(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file','', "Audio files (*.wav )")
        global path
        path=fname
    def defsubmitUplaod(self):
        global name
        name = self.textEdit.toPlainText()
        self.threadupload.start()
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
    def defsetlogupload(self,txt):
        self.logupload.addItem(txt)

if __name__ == "__main__":

    app = QApplication(sys.argv)
    form = mainWindow()
    form.show()
    app.exec_()