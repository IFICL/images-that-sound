import torch
import torch.nn as nn
import soundfile as sf
import librosa
from lightning import seed_everything
from transformers import AutoProcessor, ClapModel

class CLAPEvaluator(nn.Module):
    def __init__(
        self, 
        repo_id="laion/clap-htsat-unfused",
        **kwargs
    ):
        super(CLAPEvaluator, self).__init__()
        self.repo_id = repo_id

        # create model and load pretrained weights from huggingface
        self.model = ClapModel.from_pretrained(self.repo_id)
        self.processor = AutoProcessor.from_pretrained(self.repo_id)
    
    def forward(self, text, audio, sampling_rate=16000):
        return self.calc_score(text, audio, sampling_rate=sampling_rate)

    def calc_score(self, text, audio, sampling_rate=16000):
        if sampling_rate != 48000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=48000)
        audio = librosa.util.normalize(audio)
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", sampling_rate=48000, padding=True).to(self.model.device)
        outputs = self.model(**inputs)
        logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        score = logits_per_audio / self.model.logit_scale_a.exp()
        score = score.squeeze().item()
        score = max(score, 0.0)
        return score

    
if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    seed_everything(2024, workers=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    text = 'Birds singing sweetly'
    audio_path = '/home/czyang/Workspace/images-that-sound/data/audios/audio_02.wav'
    # import pdb; pdb.set_trace()
    audio, sr = sf.read(audio_path)

    clap = CLAPEvaluator().to(device)
    score = clap.calc_score(text, audio, sampling_rate=sr)
    print(score)




