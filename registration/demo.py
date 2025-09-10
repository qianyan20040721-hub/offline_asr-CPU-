from registration.audio_utils import  get_embedding_model
from registration.speaker_ops import add_speaker,delete_speaker,match_speaker
wav_path_1=r"D:\python_poj\3D_speaker\funasr_speaker_offline\wav\住户male_1.mp3"
wav_path_2=r"D:\python_poj\3D_speaker\funasr_speaker_offline\wav\物业.mp3"
embedding_model, feature_extractor=get_embedding_model(device="cpu")
index_1= add_speaker(wav_path_1,embedding_model,feature_extractor,spk_name="男住户-1",)
index_2= add_speaker(wav_path_2,embedding_model,feature_extractor,spk_name="物业")

match_speaker(r"D:\python_poj\3D_speaker\funasr_speaker_offline\wav\物业经理.mp3",embedding_model, feature_extractor)
# delete_speaker("物业")