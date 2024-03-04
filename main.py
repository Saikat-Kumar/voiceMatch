import glob
import json
import sys
import numpy as np
from numpy.linalg import norm
import sqlite3

import torch
import torchaudio
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

from speechbrain.inference.classifiers import EncoderClassifier

path=''
name=''
import sqlite3
conn = sqlite3.connect('DB.db')

cur = conn.cursor()
sql = 'create table if not exists voice  (name VARCHAR(255) NOT NULL,embedding TEXT NOT NULL)'
cur.execute(sql)
conn.commit()
class UploadThread(QThread):
    setlogUpload = Signal(str)
    def __init__(self, parent=None):
        super(QThread, self).__init__()
    def run(self):
        global  path
        print(path)
        conn = sqlite3.connect('DB.db')
        cur = conn.cursor()
        self.setlogUpload.emit('Path ='+path[0])
        self.setlogUpload.emit('classifier Load ')
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="pretrained_models/spkrec-ecapa-voxceleb")
        classifier.hparams.label_encoder.ignore_len()
        self.setlogUpload.emit('Audio Load ')
        signal, fs = torchaudio.load(path[0])
        # Compute embeddings
        chunk_duration = 2
        chunk_samples = int(chunk_duration * fs)
        numberOfSample = int(signal.shape[1] / chunk_samples)
        self.setlogUpload.emit('Total Chunk:'+str(numberOfSample))
        self.setlogUpload.emit('Embedding Start....')
        beg = 0
        end = chunk_samples
        centroid = torch.zeros((1, 1, 192), dtype=torch.float32)
        for i in range(0, numberOfSample, 1):
            chunk = signal[0][beg:end]
            embeddings = classifier.encode_batch(chunk)
            beg = end + 1
            end = end + chunk_samples
            centroid = torch.cat((centroid, torch.Tensor(embeddings)), 0)
            self.setlogUpload.emit('Embedding:'+str(i))
        centroid = centroid[1:, :]
        centroid = centroid.mean(dim=0)
        self.setlogUpload.emit('Embedding Complete')
        listEm=embeddings[0][0].tolist()
        sql_as_text = json.dumps(listEm)

        global name
        try:
            cur.execute("INSERT INTO voice (name,embedding) VALUES (?, ?)",(name, sql_as_text))
        except sqlite3.Error as er:
            print('SQLite error: %s' % (' '.join(er.args)))
        conn.commit()
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
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="pretrained_models/spkrec-ecapa-voxceleb")
        # classifier.hparams.label_encoder.ignore_len()
        conn = sqlite3.connect('DB.db')
        self.setlog.emit('Audio Load ')
        print(path)
        signal, fs = torchaudio.load(path)

        # Compute embeddings
        chunk_duration = 2
        chunk_samples = int(chunk_duration * fs)
        numberOfSample = int(signal.shape[1] / chunk_samples)
        self.setlog.emit('Total Chunk:' + str(numberOfSample))
        self.setlog.emit('Embedding Start....')
        beg = 0
        end = chunk_samples
        centroid = torch.zeros((1, 1, 192), dtype=torch.float32)
        for i in range(0, numberOfSample, 1):
            chunk = signal[0][beg:end]
            embeddings = classifier.encode_batch(chunk)
            beg = end + 1
            end = end + chunk_samples
            centroid = torch.cat((centroid, torch.Tensor(embeddings)), 0)
            self.setlog.emit('Embedding:' + str(i))
        centroid = centroid[1:, :]
        centroid = centroid.mean(dim=0)
        self.setlog.emit('Embedding Complete')
        embeding = embeddings[0][0].tolist()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM voice')
        results = cursor.fetchall()
        # Iterating over the results
        for row in results:
            row = json.loads(row[1])
            cosine = np.dot(embeding, row) / (norm(embeding) * norm(row))
            self.setlog.emit('Similarity '+str(cosine))

        # self.setMatch.emit(highestFilename)
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