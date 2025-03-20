""""
Todo:
    - Padding attention mask for encoder?
    - Dropout?
    - More heads/layers?
    - More data?
"""
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

class SelfAttention(nn.Module):

    def __init__(self, config, is_causal):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # Dropout after attention
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # Dropout after projection
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.is_causal = is_causal

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=self.is_causal) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.attn_dropout(y) # dropout after attention
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y) # dropout after projection
        return y
    
class CrossAttention(nn.Module):

    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        assert encoder_config.n_embd % encoder_config.n_head == 0
        assert decoder_config.n_embd % decoder_config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.encoder_c_attn = nn.Linear(encoder_config.n_embd, 2 * encoder_config.n_embd)
        self.decoder_c_attn = nn.Linear(decoder_config.n_embd, decoder_config.n_embd)
        # output projection
        self.c_proj = nn.Linear(decoder_config.n_embd, decoder_config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(encoder_config.attn_pdrop)  # Dropout after attention
        self.resid_dropout = nn.Dropout(encoder_config.resid_pdrop)  # Dropout after projection
        # regularization
        self.n_head = decoder_config.n_head
        self.n_embd = decoder_config.n_embd

    def forward(self, encoder_x, decoder_x):
        encoder_B, encoder_T, encoder_C = encoder_x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        decoder_B, decoder_T, decoder_C = decoder_x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        q = self.decoder_c_attn(decoder_x)
        kv = self.encoder_c_attn(encoder_x)
        k, v = kv.split(self.n_embd, dim=2)
        q = q.view(decoder_B, decoder_T, self.n_head, decoder_C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(encoder_B, encoder_T, self.n_head, encoder_C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(encoder_B, encoder_T, self.n_head, encoder_C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # flash attention
        y = y.transpose(1, 2).contiguous().view(decoder_B, decoder_T, decoder_C) # re-assemble all head outputs side by side
        y = self.attn_dropout(y) # dropout after attention
        # output projection
        y = self.c_proj(y)
        y = self.resid_dropout(y) # dropout after projection
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.fc_dropout = nn.Dropout(config.mlp_pdrop)
        self.proj_dropout = nn.Dropout(config.mlp_pdrop)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.fc_dropout(x) # Dropout after activation
        x = self.c_proj(x)
        x = self.proj_dropout(x) # Dropout after projection
        return x
    
class EncoderBlock(nn.Module):

    def __init__(self, encoder_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(encoder_config.n_embd)
        self.self_attn = SelfAttention(encoder_config, is_causal=False)
        self.ln_2 = nn.LayerNorm(encoder_config.n_embd)
        self.mlp = MLP(encoder_config)

    def forward(self, encoder_x, attn_mask=None):
        encoder_x = encoder_x + self.self_attn(self.ln_1(encoder_x), attn_mask)
        encoder_x = encoder_x + self.mlp(self.ln_2(encoder_x))
        return encoder_x

class DecoderBlock(nn.Module):

    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(decoder_config.n_embd)
        self.self_attn = SelfAttention(decoder_config, is_causal=True)
        self.ln_2 = nn.LayerNorm(decoder_config.n_embd)
        self.cross_attn = CrossAttention(encoder_config, decoder_config)
        self.ln_3 = nn.LayerNorm(decoder_config.n_embd)
        self.mlp = MLP(decoder_config)

    def forward(self, encoder_x, decoder_x, attn_mask=None):
        decoder_x = decoder_x + self.self_attn(self.ln_1(decoder_x), attn_mask)
        decoder_x = decoder_x + self.cross_attn(self.ln_2(encoder_x), self.ln_2(decoder_x))
        decoder_x = decoder_x + self.mlp(self.ln_3(decoder_x))
        decoder_x = decoder_x + self.mlp(self.ln_2(decoder_x))
        return decoder_x

@dataclass
class EncoderConfig:
    block_size: int = 3001 # max sequence length (spectrograms)
    vocab_size: int = 50259 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 4 # number of layers
    n_head: int = 4 # number of heads
    n_embd: int = 512 # embedding dimension
    n_mels: int = 80
    n_ctx: int = (block_size) // 2 + 1 # sequence length after 2x Conv1D layers
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1

@dataclass
class DecoderConfig:
    block_size: int = 128 # max sequence length (transcriptions)
    vocab_size: int = 50259 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 4 # number of layers
    n_head: int = 4 # number of heads
    n_embd: int = 512 # embedding dimension
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    mlp_pdrop: float = 0.1

def sinusoids(length, channels, max_timescale=10000):
    # Returns sinusoids for positional embedding
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class Whisper(nn.Module):

    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        encoder_pe = sinusoids(self.encoder_config.n_ctx, self.encoder_config.n_embd)
        self.register_buffer("positional_embedding", encoder_pe)
        self.transformer = nn.ModuleDict(dict(
            encoder_conv1 = nn.Conv1d(self.encoder_config.n_mels, self.encoder_config.n_embd, kernel_size=3, padding=1),
            encoder_conv2 = nn.Conv1d(self.encoder_config.n_embd, self.encoder_config.n_embd, kernel_size=3, stride=2, padding=1),
            decoder_wte = nn.Embedding(self.decoder_config.vocab_size, self.decoder_config.n_embd),
            decoder_wpe = nn.Embedding(self.decoder_config.block_size, self.decoder_config.n_embd),
            encoder_h = nn.ModuleList([EncoderBlock(self.encoder_config) for _ in range(self.encoder_config.n_layer)]),
            decoder_h = nn.ModuleList([DecoderBlock(self.encoder_config, self.decoder_config) for _ in range(self.decoder_config.n_layer)]),
            ln_f = nn.LayerNorm(self.decoder_config.n_embd),
        ))
        self.lm_head = nn.Linear(self.decoder_config.n_embd, self.decoder_config.vocab_size, bias=False)

        # weight sharing scheme (keep encoder separate since it's a different "type" of sequence?)
        self.transformer.decoder_wte.weight = self.lm_head.weight

        self.to(device)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.decoder_config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, spectrogram, transcription, decoder_mask=None, targets=None): # Padding attention mask for decoder is optional becuase it won't be provided during inference
        # idx is of shape (B, T)
        encoder_T = spectrogram.size()[2]
        B, decoder_T = transcription.size()
        assert encoder_T <= self.encoder_config.block_size, f"Cannot forward sequence of length {encoder_T}, block size is only {self.encoder_config.block_size}"
        assert decoder_T <= self.decoder_config.block_size, f"Cannot forward sequence of length {decoder_T}, block size is only {self.decoder_config.block_size}"
        # forward the spectrograms through the Conv1D + GELU layers
        encoder_x = F.gelu(self.transformer.encoder_conv1(spectrogram))
        encoder_x = F.gelu(self.transformer.encoder_conv2(encoder_x))
        encoder_x = encoder_x.permute(0, 2, 1)
        encoder_x = (encoder_x + self.positional_embedding).to(encoder_x.dtype) # (64, 512, 1501)
        # forward the token and posisition embeddings
        decoder_pos = torch.arange(0, decoder_T, dtype=torch.long, device=transcription.device) # shape (decoder_T)
        decoder_pos_emb = self.transformer.decoder_wpe(decoder_pos) # position embeddings of shape (decoder_T, n_embd)
        decoder_tok_emb = self.transformer.decoder_wte(transcription) # token embeddings of shape (B, decoder_T, n_embd)
        decoder_x = decoder_tok_emb + decoder_pos_emb

        for encoder_block in self.transformer.encoder_h:
            encoder_x = encoder_block(encoder_x)

        # Forward pass through decoder blocks
        for i, decoder_block in enumerate(self.transformer.decoder_h):
            if i == 0:  # Only pass decoder_mask to the first block
                decoder_x = decoder_block(encoder_x, decoder_x, decoder_mask)
            else:
                decoder_x = decoder_block(encoder_x, decoder_x)

        # forward the final layernorm and the classifier
        decoder_x = self.transformer.ln_f(decoder_x)
        logits = self.lm_head(decoder_x) # (B, decoder_T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")

eot = enc._special_tokens['<|endoftext|>'] # End of text token
enc._special_tokens['<|startoftranscript|>'] = eot + 1
sot = enc._special_tokens['<|startoftranscript|>'] # Start of transcript token
enc._special_tokens['<|pad|>'] = sot + 1
pad = enc._special_tokens['<|pad|>'] # Pad token

def load_tokens(filename):
    ptt = torch.load(filename, weights_only=True).to(device)
    return ptt

def create_transcription_masks(transcriptions):
    # Create a mask tensor of ones with the same shape as transcriptions
    masks = torch.ones_like(transcriptions, dtype=torch.float32)
    masks = masks.to(device)
    # Find the locations where transcriptions are equal to the pad token
    mask_pad = (transcriptions == pad)
    # Set the corresponding positions in masks to -inf where there's padding
    masks[mask_pad] = float('-inf')
    return masks

class DataLoader:
    def __init__(self, B, encoder_T, decoder_T, process_rank, num_processes, split):
        self.B = B  # Number of examples per batch
        self.encoder_T = encoder_T  # Length of spectrograms (3001)
        self.decoder_T = decoder_T  # Tokens per transcription (128)
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.audio_file_path = r"data/audio.pt"
        self.text_file_path = r"data/text.pt"
        self.split = split

        assert os.path.exists(self.audio_file_path), f"File {self.audio_file_path} not found"
        assert os.path.exists(self.text_file_path), f"File {self.text_file_path} not found"
        
        self.reset()

    def reset(self):
        self.spectrograms = load_tokens(self.audio_file_path) # [2611, 80, 3001]   
        self.transcriptions = load_tokens(self.text_file_path) # [2611, 128]
        self.spectrograms_position = self.B * self.process_rank
        self.transcriptions_position = self.B * self.decoder_T * self.process_rank

        self.transcription_masks = create_transcription_masks(self.transcriptions)
        self.transcription_masks_position = self.B * self.decoder_T * self.process_rank

    def next_batch(self):
        # Encoder
        encoder_x = self.spectrograms[self.spectrograms_position : self.spectrograms_position+B] # [64, 80, 3001]
        # Decoder
        transcriptions_buf = self.transcriptions[self.transcriptions_position : self.transcriptions_position+B*self.decoder_T+1] # [64, 129]
        x = transcriptions_buf[:-1].view(self.B, self.decoder_T)
        y = transcriptions_buf[1:].view(self.B, self.decoder_T)

        decoder_mask = self.transcription_masks[self.transcription_masks_position : self.transcription_masks_position+B*self.decoder_T].view(self.B, self.decoder_T)
        decoder_mask = decoder_mask[:, None, None, :]
        decoder_mask = decoder_mask.expand(self.B, DecoderConfig.n_head, self.decoder_T, self.decoder_T)

        # advance the position in the spectrograms and transcriptions tensors
        self.spectrograms_position += self.B * self.num_processes
        self.transcriptions_position += self.B * self.num_processes * self.decoder_T
        # if loading the next batch would be out of bounds, reset to the start of the tensors
        if self.spectrograms_position + (self.B * self.num_processes + 1) > len(self.spectrograms):
            self.spectrograms_position = self.B * self.process_rank
        if self.transcriptions_position + (self.B * self.num_processes * self.decoder_T + 1) > len(self.transcriptions):
            self.transcriptions_position = self.B * self.process_rank * self.decoder_T

        # advance the position in the mask tensors
        self.transcription_masks_position += self.B * self.num_processes * self.decoder_T
        # if loading the next batch would be out of bounds, reset to the start of the tensors
        if self.transcription_masks_position + (self.B * self.num_processes * self.decoder_T + 1) > len(self.transcription_masks):
            self.transcription_masks_position = self.B * self.process_rank * self.decoder_T
        return encoder_x, x, y, decoder_mask 

# -----------------------------------------------------------------------------
# simple launch:
# python trainWhisper.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 trainWhisper.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

# if master_process:
#     choice = input("Would you like to train or generate? (Enter t or g)")
#     if choice.lower().strip() == "t":
#         mode = "train"
#     elif choice.lower().strip() == "g":
#         mode = "generate"
mode = "train"
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
encoder_T = EncoderConfig.block_size # Spectrogram sequence length (3001)
decoder_T = DecoderConfig.block_size # Transcription sequence length (128)
assert total_batch_size % (B * decoder_T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * decoder_T * ddp_world_size)
if master_process and mode == "train":
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=B, encoder_T=encoder_T, decoder_T=decoder_T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
# val_loader = DataLoader(B=B, T=T_input, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = Whisper(EncoderConfig(vocab_size=50304), DecoderConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 18e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 1000 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=18e-4, device_type=device_type)

# Load the checkpoint if it exists, otherwise the model will train from scratch 
checkpoint_path = "checkpoints/checkpoint_1000.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load model parameters
    state_dict = checkpoint['model_state_dict']
    raw_model_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("module."):
            new_key = key[len("module."):]  # strip the prefix
        raw_model_state_dict[new_key] = value
    raw_model.load_state_dict(raw_model_state_dict)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Retrieve additional training state (e.g., step count)

    step = checkpoint['step']
    start_step = step + 1

    if master_process:
        print(f"Checkpoint loaded successfully! Resuming from step {step}.")
else:
    start_step = 0  # Starting from scratch if no checkpoint is found
    print("No checkpoint found. Initializing model from scratch.")

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

def generate(spectrogram, transcription, max_length=128, num_return_sequences=1):
    model.eval()
    text = transcription
    while text[-1] == 50258: # Check if the last token is padding
        text = text[:-1]
    text = enc.decode(text.tolist()[1:-1])
    input(f"The model should generate the following transcription: \"{text}\"")
    decoder_tokens = [sot]
    decoder_tokens = torch.tensor(decoder_tokens, dtype=torch.long)
    decoder_tokens = decoder_tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = decoder_tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(spectrogram.unsqueeze(0), xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length]
        tokens = tokens[(tokens != sot) & (tokens != eot) & (tokens != pad)].tolist()
        decoded = enc.decode(tokens).replace("\n", "")
        print(f"rank {ddp_rank} sample {i}: {decoded}")

def train():
    train_losses = []
    local_dir = "train_loss"
    LOSS_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(LOSS_DIR, exist_ok=True)
    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        # if step % 250 == 0 or last_step:
        #     model.eval()
        #     val_loader.reset()
        #     with torch.no_grad():
        #         val_loss_accum = 0.0
        #         val_loss_steps = 20
        #         for _ in range(val_loss_steps):
        #             x, y = val_loader.next_batch()
        #             x, y = x.to(device), y.to(device)
        #             with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        #                 logits, loss = model(x, y)
        #             loss = loss / val_loss_steps
        #             val_loss_accum += loss.detach()
        #     if ddp:
        #         dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        #     if master_process:
        #         print(f"validation loss: {val_loss_accum.item():.4f}")
        #         with open(log_file, "a") as f:
        #             f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # Saving model and optimizer state at a checkpoint
        def save_checkpoint(model, optimizer, step, loss, checkpoint_dir="checkpoints"):
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at step {step} to {checkpoint_path}")

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            encoder_x, x, y, decoder_mask = train_loader.next_batch()
            encoder_x, x, y, decoder_mask = encoder_x.to(device), x.to(device), y.to(device), decoder_mask.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(encoder_x, x, decoder_mask, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        if (master_process and (step > 0 and step % 50 == 0) or last_step):
            save_checkpoint(model, optimizer, step, loss_accum.item())

            train_loss_tensor = torch.tensor(train_losses, dtype=torch.long)
            train_loss_path = os.path.join(LOSS_DIR, "train_loss1.pt")
            torch.save(train_loss_tensor, train_loss_path)
            plt.plot(train_losses)
            plt.xlabel("Steps")
            plt.ylabel("Train Loss")
            plt.title("Training Loss")
            plt.savefig("training_loss_curve1.png")

        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.decoder_T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
            train_losses += [loss_accum.item()]

if mode == "train":
    train()
elif mode == "generate":
    if master_process:
        val_audio = torch.load(r"data/val_audio.pt", weights_only=True).to(device)
        val_text = torch.load(r"data/val_text.pt", weights_only=True).to(device)
        for spectrogram, transcription in zip(val_audio, val_text):
            generate(spectrogram, transcription)

if ddp:
    destroy_process_group()