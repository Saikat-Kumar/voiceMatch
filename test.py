import glob
import json
import sys

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

from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                    savedir="pretrained_models/spkrec-ecapa-voxceleb")
classifier.hparams.label_encoder.ignore_len()

signal, fs = torchaudio.load('./upload/115EVYW.wav')
print(signal.shape)
# Compute embeddings
chunk_duration = 2
chunk_samples = int(chunk_duration *fs)
print(chunk_samples)
print(signal[0][0:0 + chunk_samples].shape)
chunks = [signal[0][i:i + chunk_samples] for i in range(0, len(signal), chunk_samples)]

centroid = torch.zeros((1,1, 192), dtype=torch.float32)
for i, chunk in enumerate(chunks):
            print(chunk.shape)
            embeddings = classifier.encode_batch(chunk)
            print(embeddings.shape)
            centroid = torch.cat((centroid, torch.Tensor(embeddings)), 0)

centroid = centroid[1:, :]
centroid = centroid.mean(dim=0)
listEm=embeddings[0][0].tolist()