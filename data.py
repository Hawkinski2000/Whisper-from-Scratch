from datasets import load_dataset
import os
import numpy as np
import torchaudio.transforms as transforms
import tiktoken
import torch

# ----------------------------------------------------------------------------
# ---- Load Dataset and Create "data" Directory for Audio and Text ----

# Load dataset in streaming mode
ds = load_dataset(
    "openslr/librispeech_asr",
    "clean", split="test",
    streaming=True,
    trust_remote_code=True
)

# Create directory for audio and text
local_dir = "data"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# ---- Prepare Audio and Transcriptions ----

audio_samples = []
transcriptions = []

enc = tiktoken.get_encoding("gpt2")

eot = enc._special_tokens['<|endoftext|>'] # End of text token
enc._special_tokens['<|startoftranscript|>'] = eot + 1
sot = enc._special_tokens['<|startoftranscript|>'] # Start of transcript token
enc._special_tokens['<|pad|>'] = sot + 1
pad = enc._special_tokens['<|pad|>'] # Pad token

for sample in ds:
    # Get the audio sample and sampling rate
    audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
    sampling_rate = sample["audio"]["sampling_rate"]
    # Convert to PyTorch tensor
    waveform = torch.tensor(audio_array).unsqueeze(0) # [1, Time]
    # Pad or truncate the audio sample to 30 seconds
    target_length = 30 * sampling_rate # 30 seconds sampling at 16000 Hz
    waveform_length = waveform.shape[-1]
    if waveform_length < target_length:
        # Pad the waveform with zeros (at the end)
        padding = target_length - waveform_length
        waveform = torch.cat([waveform, torch.zeros(1, padding)], dim=-1)
        # Pad or truncate the transcriptions so they're 128 tokens long
        tokens = enc.encode_ordinary(sample["text"]) 
        if len(tokens) < 126: # Pad with special pad token if shorter than 126
            # Add sot token at start and eot at end of each transcription
            transcription = torch.tensor(
                [sot] + tokens + [eot] + [pad] * (126 - len(tokens))
            ).unsqueeze(0)
        else: # Truncate if longer than 126
            transcription = torch.tensor(
                [sot] + tokens[:126] + [eot]
            ).unsqueeze(0)
        transcriptions.append(transcription)
    elif waveform_length > target_length:
        # Skip the audio sample if it's longer than 30 seconds
        continue

    # Compute Mel Spectrogram
    sample_rate = 16000
    n_fft = int(25 / 1000 * sample_rate)  
    hop_length = int(10 / 1000 * sample_rate)
    n_mels = 80  
    mel_spectrogram_transform = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    mel_spectrogram = mel_spectrogram_transform(waveform) # [1, 80, Time]
    # Convert to log scale
    log_mel_spectrogram = torch.log10(mel_spectrogram + 1e-6)
    audio_samples.append(log_mel_spectrogram)

# ----------------------------------------------------------------------------
# ---- Save Spectrograms and Tokenized Text to "data" Directroy as .pt ----

audio_samples_tensor = torch.cat(audio_samples)
audio_path = os.path.join(DATA_CACHE_DIR, "audio.pt")
torch.save(audio_samples_tensor, audio_path)

transcriptions_tensor = torch.cat(transcriptions)
text_path = os.path.join(DATA_CACHE_DIR, "text.pt")
torch.save(transcriptions_tensor, text_path)