import os
import sys
import argparse
from PIL import Image
import shutil

import numpy as np
import torch
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import torchaudio

import soundfile as sf
import librosa

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.components.auffusion_converter import mel_spectrogram, normalize_spectrogram, denormalize_spectrogram



def wav2spec(audio, sr):
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)    
    spec = mel_spectrogram(audio, n_fft=2048, num_mels=256, sampling_rate=16000, hop_size=160, win_size=1024, fmin=0, fmax=8000, center=False)
    spec = normalize_spectrogram(spec)
    return spec


def griffin_lim(mel_spec, ori_audio):
    mel_spec = denormalize_spectrogram(mel_spec)
    mel_spec = torch.exp(mel_spec)

    audio = librosa.feature.inverse.mel_to_audio(
        mel_spec.numpy(),
        sr=16000,
        n_fft=2048,
        hop_length=160,
        win_length=1024,
        power=1,
        center=True,
        # length=ori_audio.shape[0]
    )

    length = ori_audio.shape[0]

    if audio.shape[0] > length:
        audio = audio[:length]
    elif audio.shape[0] < length:
        audio = np.pad(audio, (0, length - audio.shape[0]), mode='constant')

    audio = np.clip(audio, a_min=-1, a_max=1)
    return audio 

def inverse_stft(mel_spec, ori_audio):
    mel_spec = denormalize_spectrogram(mel_spec)
    mel_spec = torch.exp(mel_spec)

    n_fft = 2048
    hop_length = 160
    win_length = 1024
    power = 1
    center = False

    spec_mag = librosa.feature.inverse.mel_to_stft(mel_spec.numpy(), sr=16000, n_fft=n_fft, power=power)
    spec_mag = torch.tensor(spec_mag).float()

    audio_length = ori_audio.shape[0]
    ori_audio = torch.tensor(ori_audio)
    ori_audio = ori_audio.unsqueeze(0)   
    ori_audio = torch.nn.functional.pad(ori_audio.unsqueeze(1), (int((n_fft-hop_length)/2), int((n_fft-hop_length)/2)), mode='reflect')
    ori_audio = ori_audio.squeeze(1)
    vocoder_spec = torch.stft(ori_audio, n_fft, hop_length=hop_length, win_length=win_length, window=torch.hann_window(win_length), center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
    # Get the phase from the complex spectrogram
    vocoder_phase = torch.angle(vocoder_spec).float()
    # import pdb; pdb.set_trace()

    # Combine the new magnitude with the original phase
    # We use polar coordinates to transform magnitude and phase into a complex number
    reconstructed_complex_spec = torch.polar(spec_mag.unsqueeze(0), vocoder_phase)

    # Perform the ISTFT to convert the spectrogram back to time domain audio signal
    reconstructed_audio = torch.istft(
        reconstructed_complex_spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        center=True,
        normalized=False,
        onesided=True,
        length=audio_length  # Ensure output audio length matches original audio
    )

    reconstructed_audio = reconstructed_audio.squeeze(0).numpy()
    reconstructed_audio = np.clip(reconstructed_audio, a_min=-1, a_max=1)
    
    return reconstructed_audio


# python src/utils/consistency_check.py --dir "logs/soundify-denoise/colorization/bell_example_005"
# python src/utils/consistency_check.py --dir "logs/soundify-denoise/colorization/tiger_example_002"
# python src/utils/consistency_check.py --dir "logs/soundify-denoise/colorization/dog_example_06"
# python src/utils/consistency_check.py --dir "logs/soundify-denoise/debug/results/example_015"



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=False, type=str, default="logs/soundify-denoise/colorization/bell_example_29")

    args = parser.parse_args()
    save_dir = f"logs/consistency-check/{args.dir.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    
    # import pdb; pdb.set_trace()

    # audio from vocoder to spectrogram
    audio_path = os.path.join(args.dir, "audio.wav")
    audio_data, sampling_rate = sf.read(audio_path)
    spec = wav2spec(audio_data, sampling_rate)

    save_path = os.path.join(save_dir, f"respec-hifi.png")
    save_image(spec, save_path, padding=0)

    # spectrogram to audio using griffin-lim
    spec_path = os.path.join(args.dir, "spec.png")
    spec = Image.open(spec_path)
    spec = TF.to_tensor(spec)

    save_path = os.path.join(save_dir, f"spec.png")
    shutil.copyfile(spec_path, save_path)

    audio_istft = inverse_stft(spec, audio_data)
    save_audio_path = os.path.join(save_dir, f"audio-istft.wav")
    sf.write(save_audio_path, audio_istft, samplerate=16000)
    spec_istft = wav2spec(audio_istft, sampling_rate)

    save_path = os.path.join(save_dir, f"respec-istft.png")
    save_image(spec_istft, save_path, padding=0)


    audio_gl = griffin_lim(spec, audio_data)
    save_audio_path = os.path.join(save_dir, f"audio-griffin-lim.wav")
    sf.write(save_audio_path, audio_gl, samplerate=16000)

    # audio_data, sampling_rate = sf.read(save_audio_path)

    spec_gl = wav2spec(audio_gl, sampling_rate)

    save_path = os.path.join(save_dir, f"respec-gl.png")
    save_image(spec_gl, save_path, padding=0)





