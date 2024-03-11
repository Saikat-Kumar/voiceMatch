# import faiss
# import numpy as np
# import torch
# import torchaudio
# from speechbrain.inference import EncoderClassifier
#
# embeddingnp = np.array([]) # Example embeddings, replace with your own
# name = np.array([])
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
#                                                     savedir="pretrained_models/spkrec-ecapa-voxceleb")
# classifier.hparams.label_encoder.ignore_len()
#
# signal, fs = torchaudio.load('./upload/115EVYW.wav')
# nameVoice='test'
# # Compute embeddings
# chunk_duration = 2
# chunk_samples = int(chunk_duration * fs)
# numberOfSample = int(signal.shape[1] / chunk_samples)
#
# beg = 0
# end = chunk_samples
# centroid = torch.zeros((1, 1, 192), dtype=torch.float32)
# for i in range(0, numberOfSample, 1):
#             chunk = signal[0][beg:end]
#             embeddings = classifier.encode_batch(chunk)
#             beg = end + 1
#             end = end + chunk_samples
#             centroid = torch.cat((centroid, torch.Tensor(embeddings)), 0)
#
# centroid = centroid[1:, :]
# embedding = centroid.mean(dim=0)
# # Build the index
# index = faiss.IndexFlatL2(embedding.shape[1])
# index.add(embedding)
# name = np.append(name, nameVoice)
#
# embeddingnp=np.append(embeddingnp, embedding)
# # Save the index
# faiss.write_index(index, "my_index.index")
#
# # Save the embeddings and their corresponding IDs
# np.save("embeddings.npy", embeddingnp)
# np.save("name.npy", name)
#
import faiss
import numpy as np
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load the index
index = faiss.read_index("my_index.index")
colors = np.array(
    [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
)
# Load embeddings and IDs
name = np.load("name.npy")
embeddings = np.load("embeddings.npy")

signal, fs = torchaudio.load('./upload/115EVYW.wav')
nameVoice='test'
# Compute embeddings
chunk_duration = 5
chunk_samples = int(chunk_duration * fs)
numberOfSample = int(signal.shape[1] / chunk_samples)
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                     savedir="pretrained_models/spkrec-ecapa-voxceleb")
classifier.hparams.label_encoder.ignore_len()
beg = 0
end = chunk_samples
centroid = torch.zeros((1, 1, 192), dtype=torch.float32)
prev=897768
segstart=0
segend=0
plt.figure(figsize=(5,2))
matchindex=0
for i in range(0, numberOfSample, 1):
            chunk = signal[0][beg:end]
            embedding = classifier.encode_batch(chunk)
            print(embedding[0][0].shape)
            distances, indices = index.search(embedding[0], 1)
            print(indices[0][0])
            if(i==0):
                prev=indices[0][0]
            if(prev!=indices[0][0] and i!=0):
                print('here')
                segend=end
                speech = signal[0][segstart:segend]
                color = colors[matchindex]

                linelabel = "Speaker {}".format(nameVoice[indices[0][0]])
                plt.plot(
                    np.linspace(segstart, segend, len(speech)),
                    speech,
                    color=color,
                    label=linelabel,
                )
                segstart=segend+1
                matchindex=matchindex+1

            prev = indices[0][0]

            beg = end + 1
            end = end + chunk_samples

speech = signal[0][segstart:end]
print(segstart,end,len(speech))
color = colors[0]

linelabel = "Speaker {}"
plt.plot(
                    np.linspace(segstart, end, len(speech)),
                    speech,
                    color=color,
                    label=linelabel,
                )
plt.tight_layout()
plt.show()


