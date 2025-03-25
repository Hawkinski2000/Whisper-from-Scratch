""""
TODO:
 - Implement test/val sets.
"""
from datasets import load_dataset
import tiktoken
import numpy as np
import torch
import torchaudio.transforms as transforms
import os
import multiprocessing as mp
from tqdm import tqdm


# Up to 12k spectrograms or transcriptions per shard, total of 10 shards
shard_size = int(12e3)
duration = 10 # Seconds of audio in each spectrogram
transcription_length = 32 # Tokens per transcription

# ----------------------------------------------------------------------------
# ---- Load Dataset and Create "data" Directory for Audio and Text ----

# https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
# Must login to Hugging Face Hub with "huggingface-cli login" and access token
ds = load_dataset("mozilla-foundation/common_voice_17_0", 
                  "en",
                  split="train",
                  streaming=True)

# Create "data" directory to hold "audio" and "text"
data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)
# Create "audio" and "text" subfolders inside "data"
audio_dir = os.path.join(data_dir, "audio")
text_dir = os.path.join(data_dir, "text")
# Create the "audio" and "text" directories if they don't exist
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)

# ----------------------------------------------------------------------------
# ---- Prepare Spectrograms and Transcriptions ----

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # End of text token
enc._special_tokens['<|startoftranscript|>'] = eot + 1
sot = enc._special_tokens['<|startoftranscript|>'] # Start of transcript token
enc._special_tokens['<|pad|>'] = sot + 1
pad = enc._special_tokens['<|pad|>'] # Pad token

def tokenize(sample):
    # Get the audio sample and sampling rate
    audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
    sampling_rate = sample["audio"]["sampling_rate"]

    # Convert to PyTorch tensor
    waveform = torch.tensor(audio_array).unsqueeze(0) # [1, Time]

    # Convert stereo to mono
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample from 48kHz to 16kHz
    target_sampling_rate = 16000
    resampler = transforms.Resample(orig_freq=sampling_rate,
                                    new_freq=target_sampling_rate)
    waveform = resampler(waveform)

    # Pad/truncate the audio sample to specified number of seconds at 16 kHz
    target_length = duration * target_sampling_rate 
    waveform_length = waveform.shape[-1]
    if waveform_length < target_length:
        # Pad the waveform with zeros (at the end)
        padding = target_length - waveform_length
        waveform = torch.cat(
            [waveform, torch.full((1, padding), 0)], dim=-1)

        # Pad/truncate transcriptions so they're transcription_length tokens
        tokens = enc.encode_ordinary(sample["sentence"])
        # Pad with special pad token if < (transcription_length - 2)
        if len(tokens) < (transcription_length - 2):
            # Add sot token at start and eot at end of each transcription
            transcription = torch.tensor(
                [sot] + tokens + [eot] +
                [pad] * ((transcription_length - 2) - len(tokens)),
                dtype=torch.uint16)

        else: # Truncate if longer than (transcription_length - 2)
            transcription = torch.tensor(
                [sot] + tokens[:(transcription_length - 2)] + [eot],
            dtype=torch.uint16)

    elif waveform_length > target_length:
        # Skip the audio sample and its transcription if longer than 30 secs
        return None, None

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
    # audio_samples.append(log_mel_spectrogram)

    return log_mel_spectrogram, transcription

# ----------------------------------------------------------------------------
# ---- Create Spectrogram and Transcription Shards and Save to File ----

def main():
    nprocs = max(1, os.cpu_count() // 2) # Use half of CPU cores
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        example_count = 0
        progress_bar = None
        spectrograms = []
        transcriptions = []

        # Apply "tokenize()" to each row (spectrogram/transcription) in ds
        for log_mel_spectrogram, transcription in pool.imap(tokenize,
                                                            ds,
                                                            chunksize=16):
            if log_mel_spectrogram == None:
                continue
            # Create a progress bar for this pair of shards
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size,
                                    unit=" Examples",
                                    desc=f"Shard {shard_index}")
            spectrograms.append(log_mel_spectrogram)
            transcriptions += transcription
            example_count += 1
            # update progress bar
            progress_bar.update(1)

            # Save the pair of shards if they have been filled
            if example_count == shard_size:
                # Save spectrograms shard to "audio" directory
                audio_tensor = torch.cat(spectrograms)
                audio_path = os.path.join(
                    audio_dir, f"train_audio_{shard_index:02d}.pt")
                torch.save(audio_tensor, audio_path)

                # Save transcriptions shard to "text" directory
                transcriptions_tensor = torch.tensor(transcriptions)
                transcriptions_path = os.path.join(
                    text_dir, f"train_text_{shard_index:02d}.pt")
                torch.save(transcriptions_tensor, transcriptions_path)

                progress_bar.update(1)
                progress_bar = None
                shard_index += 1
                spectrograms = []
                transcriptions = []

if __name__ == '__main__':
    main()