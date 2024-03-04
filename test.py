import glob
import json
import sys

import sqlite3
import numpy as np
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

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="pretrained_models/spkrec-ecapa-voxceleb")
classifier.hparams.label_encoder.ignore_len()

signal, fs = torchaudio.load('./upload/115EVYW.wav')
print(signal.shape)


chunk_duration = 2
chunk_samples = int(chunk_duration *fs)
numberOfSample = int(signal.shape[1]/chunk_samples)
beg=0
end=chunk_samples
centroid = torch.zeros((1,1, 192), dtype=torch.float32)
for i in range(0,numberOfSample,1):
    chunk=signal[0][beg:end]
    embeddings = classifier.encode_batch(chunk)
    beg=end+1
    end=end+chunk_samples
    centroid = torch.cat((centroid, torch.Tensor(embeddings)), 0)
print(centroid.shape)

centroid = centroid[1:, :]
centroid = centroid.mean(dim=0)
embeding=embeddings[0][0].tolist()


conn = sqlite3.connect('DB.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM voice')
results = cursor.fetchall()
# Iterating over the results
for row in results:
    print(row[1])
    row=json.loads(row[1])
    cosine = np.dot(embeding, row) / (norm(embeding) * norm(row))
    print('Similarity ' + str(cosine))