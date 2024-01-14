import glob
import shutil
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")
score, prediction = verification.verify_files("D:/VoiceData/WBSWAN/Amir_Mayureswar1_Bhq_Birbhum.wav","./upload/115EVYW.wav") # Different Speakers
print(score[0])