import os
import glob
import librosa
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader, TensorDataset

AUDIO_DIR = 'TransientDataset\\Input'
TARGET_DIR = 'TransientDataset\\Target'
SAMPLE_RATE = 44100
NUM_SAMPLES = SAMPLE_RATE * 32

class TransientDataset(Dataset):

    def __init__(self, audio_dir, target_dir, transformation, target_sr, num_samples, device):
        self.audio_dir = audio_dir
        self.target_dir = target_dir
        self.transformation = transformation
        self.target_sr = target_sr
        self.num_samples = num_samples
        self.device = device

        self.audio_files_list = []
        self.target_files_list = []
        self.list_audio_files(self.audio_dir, self.audio_files_list)
        self.list_audio_files(self.target_dir, self.target_files_list)

    def list_audio_files(self, audio_dir, listt):
        for file_name in glob.glob(os.path.join(audio_dir, "*.wav")):
            listt.append(file_name)

    def __getitem__(self, index):
        input_sample_dir = self.audio_files_list
        input_sample_path = input_sample_dir[index]

        target_sample_dir = self.target_files_list
        target_sample_path = target_sample_dir[index]

        input_signal, sr = torchaudio.load(input_sample_path)
        target_signal, sr = torchaudio.load(target_sample_path)

        self.signal_manipulation(input_signal, sr)
        self.signal_manipulation(target_signal, sr)

        input_signal = input_signal[:, :NUM_SAMPLES]
        target_signal = target_signal[:, :NUM_SAMPLES]

        return input_signal, target_signal

    def _get_duration(self, signal, sr):
        num_samples = signal.shape[1]
        duration = num_samples / sr
        return duration

    def signal_manipulation(self, signal, sr=44100):
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        dur = self._get_duration(signal, sr)
        #print(f"Duration = {dur}")
        signal = self.transformation(signal)
        return signal

    def __len__(self):
        return len(self.audio_files_list)

    def _cut_if_necessary(self, signal): # <------------------------
           if signal.shape[1] > self.num_samples:
               signal = signal[:, :self.num_samples]
           return signal
    def _right_pad_if_necessary(self, signal):
           signal_length = signal.shape[1]
           if signal_length < self.num_samples:
               missing_samples = self.num_samples - signal_length
               last_dim_padding = (0, missing_samples)
               signal = torch.nn.functional.pad(signal, last_dim_padding)
           return signal

    def _resample_if_necessary(self, signal, sr):
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)
                signal = resampler(signal)
            return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def load_audio(self, file_path, sr=44100):
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=64
    ).to(device)

    tds = TransientDataset( AUDIO_DIR,
                            TARGET_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    signal, target = tds[0]

    #print("tds[0] = ", tds[0])
    #print("tds[0][1] = ", tds[0][1])

    print(signal.shape)
    print(target, target.shape)