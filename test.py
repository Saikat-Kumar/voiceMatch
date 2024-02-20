import librosa
import soundfile as sf
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

audio_file = "C:/Users/Saikat Kumar/Downloads/Akash Prasad _ Sankrail  BHQ.wav"
y1, sr1 = librosa.load(audio_file, sr=None)
y, sr = torchaudio.load(audio_file)
print(y.shape,sr)
print(len(y1),sr)
# chunk duration 2 seconds
# chunk_duration = 2
#
#
# chunk_samples = int(chunk_duration * sr)
#
#
# chunks = [y[i:i + chunk_samples] for i in range(0, len(y), chunk_samples)]
#
# centroid = torch.zeros((1, 192), dtype=torch.float32)
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
#                                                     savedir="pretrained_models/spkrec-ecapa-voxceleb")
# classifier.hparams.label_encoder.ignore_len()
# for i, chunk in enumerate(chunks):
#
#     embeddings = classifier.encode_batch(chunk)
#     print(embeddings.shape)
#     #centroid = torch.cat((centroid, torch.Tensor(embeddings)), 0)
#     print(i)
# centroid = centroid[1:, :]
# centroid = centroid.mean(dim=0)