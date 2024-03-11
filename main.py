import glob
import json
import sys

import faiss
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numpy.linalg import norm


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
import yt_dlp as youtube_dl
import subprocess
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
path=''
name=''
url=''

class UploadThread(QThread):
    setlogUpload = Signal(str)
    startsUp = Signal(int)
    stopsUp = Signal(int)
    def __init__(self, parent=None):
        super(QThread, self).__init__()
    def run(self):
        global  path
        print(path)
        self.startsUp.emit(1)
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
        embedding = centroid.mean(dim=0)
        self.setlogUpload.emit('Embedding Complete')
        namenp = np.load("name.npy")
        embeddingsnp = np.load("embeddings.npy")
        indexFaiss = faiss.read_index("my_index.index")
        self.setlogUpload.emit('Databas Loaded')
        indexFaiss.add(embedding)
        global name
        namenp = np.append(namenp, name)

        embeddingnp=np.append(embeddingsnp, embedding)

        faiss.write_index(indexFaiss, "my_index.index")

        # Save the embeddings and their corresponding IDs
        np.save("embeddings.npy", embeddingnp)
        np.save("name.npy", namenp)
        self.setlogUpload.emit('Databas Saved')
        self.stopsUp.emit(1)


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

        self.setlog.emit('Audio Load ')

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
        embedding = centroid.mean(dim=0)
        self.setlog.emit('Embedding Complete')
        indexF = faiss.read_index("my_index.index")

        # Load embeddings and IDs
        namenp = np.load("name.npy")
        embeddingsnp = np.load("embeddings.npy")
        k = 10  # Number of nearest neighbors to retrieve
        distances, indices = indexF.search(embedding, k)

        print("\nNearest neighbors:")
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"ID: {namenp[idx]}, Distance: {distance}")

        # self.setMatch.emit(highestFilename)
    def stop(self):
        print('stop')
class YoutubeThread(QThread):
    setfigure1 = Signal(str)
    def __init__(self, parent=None):
        super(QThread, self).__init__()
    def run(self):
        global url


    def stop(self):
        print('stop')
class mainWindow(QMainWindow,mainui.Ui_MainWindow ):

    def __init__(self, parent=None):
        super(mainWindow, self).__init__(parent)
        self.setupUi(self)
        self.movie = QMovie("./ui/assets/loader.gif")
        self.loaderup.setMovie(self.movie)
        self.getAudio.clicked.connect(self.defAudioget)
        self.search.clicked.connect(self.defMatch)
        self.upload.clicked.connect(self.defCopy)
        self.submitUpload.clicked.connect(self.defsubmitUplaod)
        self.threadupload = UploadThread()
        self.threadupload.setlogUpload.connect(self.defsetlogupload)
        self.threadupload.startsUp.connect(self.defstartUp)
        self.threadupload.stopsUp.connect(self.defstopUp)
        self.threadmatch = MatchThread()
        self.threadmatch.setprogress.connect(self.defsetprogress)
        self.threadmatch.setlog.connect(self.defsetlog)
        self.threadmatch.setMatch.connect(self.defsetMatch)
        self.threadYoutube = YoutubeThread()

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
    def defAudioget(self):
        global url
        # url = self.url.toPlainText()
        # output_file = 'download'
        # ydl_opts = {
        #     'format': 'bestaudio/best',
        #     'outtmpl': output_file + '.%(ext)s',
        #     'postprocessors': [{
        #         'key': 'FFmpegExtractAudio',
        #         'preferredcodec': 'wav',
        #         'preferredquality': '192',
        #     }],
        #     'ignoreerrors': True,  # Add this option to ignore errors
        # }
        # with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        #     ydl.download([url])
        signal, fs = sf.read('download.wav')
        Time = np.linspace(0, len(signal) / fs, num=len(signal))

        color = "tab:blue"
        start_time = 0 / fs
        end_time = start_time + (len(signal) / fs)


        plt.show()
        print(f'Audio saved as ')
        Time = np.linspace(0, len(signal) / fs, num=len(signal))

        color = "tab:blue"
        start_time = 0 / fs
        end_time = start_time + (len(signal) / fs)

        scene = QtWidgets.QGraphicsScene()
        fig = Figure(figsize=(7,1))
        ax = fig.add_subplot(111)
        ax.plot(np.linspace(start_time, end_time, len(signal)), signal, color=color)

        canvas = FigureCanvas(fig)
        scene.addWidget(canvas)
        self.figure1.setScene(scene)
    #loader
    def defstartUp(self,value):
        self.loaderup.show()
        self.movie.start()
    def defstopUp(self,value):
        self.movie.stop()
        self.loaderup.hide()
if __name__ == "__main__":

    app = QApplication(sys.argv)
    form = mainWindow()
    form.show()
    app.exec_()