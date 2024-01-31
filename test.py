import glob
import json
import shutil
from speechbrain.pretrained import SpeakerRecognition
import sqlite3
import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="pretrained_models/spkrec-ecapa-voxceleb")
classifier.hparams.label_encoder.ignore_len()
# Compute embeddings
signal, fs = torchaudio.load('D:\VoiceData\dev-clean\84/1.wav')
embeddings = classifier.encode_batch(signal)
listEm=embeddings[0][0].tolist()
sql_as_text = json.dumps(listEm)
conn = sqlite3.connect('DB.db')
conn.execute(
            "INSERT INTO voice (name,embedding) VALUES (?, ?)",
            ('test', sql_as_text))
print('done')