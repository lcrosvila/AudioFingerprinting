import os
import glob
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import scipy.signal
import librosa
import sys

# --- PyTorch Lightning Imports ---
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Import h5py directly here for preloading
try:
    import h5py
except ImportError:
    h5py = None
    print("Warning: h5py not installed. H5 files will be skipped.")

# Configuration
CONFIG = {
    'sample_rate': 8000,          
    'duration': 4.0,              
    'n_mels': 64,                 
    'batch_size': 64,             
    'learning_rate': 1e-3,        
    'epochs': 20,
    'output_dim': 128,            
    'margin': 0.5,                
    'project_name': "audio-fingerprinting-lightning",
    'num_workers': 4,             
    
    # Paths
    'source_dir': '../aimir/lastfm/audio/', 
    'noise_dirs': [
        'data/chatter_sounds', 
        'data/gaming_sounds'
    ],
    'ir_root': 'data/IR'
}

def preload_h5_data(h5_files):
    """
    Loads all H5 IRs into a Python dictionary in RAM.
    Returns: dict { 'filename': numpy_array_of_shape(sources, mics, samples) }
    """
    if not h5_files or h5py is None:
        return {}

    print(f"--- Preloading {len(h5_files)} H5 files into RAM (This may take a minute) ---")
    cache = {}
    total_mb = 0
    
    for path in h5_files:
        try:
            fname = os.path.basename(path)
            with h5py.File(path, 'r') as f:
                # Based on MIRACLE structure: data -> impulse_response
                # We load the WHOLE tensor into RAM. 
                # Shape is usually (Source, Mic, Time)
                if 'data' in f and 'impulse_response' in f['data']:
                    data = f['data']['impulse_response'][:]
                    cache[fname] = data.astype(np.float32)
                    
                    size_mb = data.nbytes / (1024 * 1024)
                    total_mb += size_mb
                    print(f"Loaded {fname}: {data.shape} ({size_mb:.1f} MB)")
                elif 'impulse_response' in f:
                     # Some files might be flatter
                    data = f['impulse_response'][:]
                    cache[fname] = data.astype(np.float32)
                    print(f"Loaded {fname} (flat): {data.shape}")
        except Exception as e:
            print(f"Failed to load H5 {path}: {e}")

    print(f"--- H5 Preload Complete. Total Size: {total_mb:.1f} MB ---")
    return cache

class RobustFingerprintDataset(Dataset):
    def __init__(self, source_files, noise_files, ir_wav_files, ir_pickle_data=None, ir_h5_cache=None):
        self.source_files = source_files
        self.noise_files = noise_files
        
        # We separate file paths by type
        self.ir_wav_files = ir_wav_files
        self.ir_pickle_data = ir_pickle_data 
        self.ir_h5_cache = ir_h5_cache # Dict: {filename: data_array}
        self.h5_keys = list(ir_h5_cache.keys()) if ir_h5_cache else []
        
        self.sample_rate = CONFIG['sample_rate']
        self.num_samples = int(CONFIG['duration'] * self.sample_rate)
        
        # Audio Transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=CONFIG['n_mels'],
            n_fft=1024,
            hop_length=256,
            power=2.0
        )
        self.db_transform = T.AmplitudeToDB()

    def _load_audio_segment(self, path):
        try:
            info = torchaudio.info(path)
            total_frames = info.num_frames
            sr = info.sample_rate
            needed_frames = int(CONFIG['duration'] * sr)
            
            if total_frames <= needed_frames:
                waveform, _ = torchaudio.load(path)
                pad_amt = needed_frames - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_amt))
            else:
                start_frame = random.randint(0, total_frames - needed_frames)
                waveform, _ = torchaudio.load(path, frame_offset=start_frame, num_frames=needed_frames)
            
            if sr != self.sample_rate:
                waveform = T.Resample(sr, self.sample_rate)(waveform)
                
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            return waveform
        except Exception:
            return torch.zeros(1, self.num_samples)

    def _get_random_ir(self):
        """
        Efficiently retrieves an IR from RAM (Pickle/H5) or Disk (WAV).
        """
        # Determine which source to use (WAV, Pickle, or H5)
        # We assign probabilities based on availability
        options = []
        if self.ir_wav_files: options.append('wav')
        if self.ir_pickle_data: options.append('pickle')
        if self.ir_h5_cache: options.append('h5')
        
        if not options: return None
        
        choice = random.choice(options)
        
        ir_sig = None
        sr_ir = 44100 # Default assumption, corrected below if needed

        try:
            if choice == 'h5':
                # Fast Dictionary Lookup
                fname = random.choice(self.h5_keys)
                data_block = self.ir_h5_cache[fname]
                
                # data_block is (Sources, Mics, Samples) or (Angles, Samples)
                dims = data_block.ndim
                if dims == 3:
                    s = random.randint(0, data_block.shape[0]-1)
                    m = random.randint(0, data_block.shape[1]-1)
                    ir_sig = data_block[s, m, :]
                elif dims == 2:
                    r = random.randint(0, data_block.shape[0]-1)
                    ir_sig = data_block[r, :]
                else:
                    return None
                
                # Miracle dataset is usually 48kHz, but let's assume 44.1 or 48.
                # Since we rely on resampling anyway, 44100 is a safe base unless specified.
                sr_ir = 48000 
                
            elif choice == 'pickle':
                # Fast List Lookup
                record_idx = random.randint(0, len(self.ir_pickle_data) - 1)
                # Index 43 is rirData in the pickle list
                ir_sig = np.array(self.ir_pickle_data[record_idx][43], dtype=np.float32)
                sr_ir = 44100
                
            elif choice == 'wav':
                # Slow Disk Load (but OS caching helps)
                path = random.choice(self.ir_wav_files)
                ir_sig, sr_ir = librosa.load(path, sr=None, mono=True)
                
            return ir_sig, sr_ir
            
        except Exception:
            return None

    def _augment(self, waveform):
        # 1. Reverb (50%)
        if random.random() > 0.5:
            ir_data = self._get_random_ir()
            if ir_data is not None:
                ir_sig, sr_ir = ir_data
                if sr_ir != self.sample_rate:
                    # Calculate new length
                    num_s = int(len(ir_sig) * self.sample_rate / sr_ir)
                    # Optimize: If IR is huge, truncate it before resampling to save CPU
                    if num_s > self.num_samples * 2: 
                        num_s = self.num_samples * 2
                        
                    ir_sig = scipy.signal.resample(ir_sig, num_s)
                
                audio_np = waveform.numpy().flatten()
                
                # FFT Convolve
                convolved = scipy.signal.fftconvolve(audio_np, ir_sig, mode='full')
                convolved = convolved[:self.num_samples]
                
                max_val = np.max(np.abs(convolved))
                if max_val > 0: convolved /= max_val
                waveform = torch.from_numpy(convolved).float().unsqueeze(0)

        # 2. Noise (50%)
        if random.random() > 0.5 and self.noise_files:
            noise_path = random.choice(self.noise_files)
            noise_wave = self._load_audio_segment(noise_path)
            
            if noise_wave.size(1) != waveform.size(1):
                if noise_wave.size(1) < waveform.size(1):
                    repeats = 1 + waveform.size(1) // noise_wave.size(1)
                    noise_wave = noise_wave.repeat(1, repeats)
                noise_wave = noise_wave[:, :waveform.size(1)]
            
            snr_db = random.uniform(0, 15)
            sig_p = waveform.pow(2).mean()
            noise_p = noise_wave.pow(2).mean()
            
            if noise_p > 0:
                scale = torch.sqrt(sig_p / (10**(snr_db/10)) / noise_p)
                waveform = waveform + (noise_wave * scale)

        return waveform

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        path = self.source_files[idx]
        clean_wave = self._load_audio_segment(path)
        
        anchor_wave = self._augment(clean_wave.clone())
        positive_wave = self._augment(clean_wave.clone())
        
        anchor_spec = self.db_transform(self.mel_spectrogram(anchor_wave))
        positive_spec = self.db_transform(self.mel_spectrogram(positive_wave))
        
        return anchor_spec, positive_spec


class FingerprintLightningModule(pl.LightningModule):
    def __init__(self, output_dim=128, margin=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Architecture
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim)
        
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        anchor, positive = batch
        negative = torch.roll(positive, shifts=1, dims=0)
        
        a_emb = self(anchor)
        p_emb = self(positive)
        n_emb = self(negative)
        
        loss = self.triplet_loss(a_emb, p_emb, n_emb)
        
        dist_pos = F.pairwise_distance(a_emb, p_emb)
        dist_neg = F.pairwise_distance(a_emb, n_emb)
        accuracy = (dist_pos < dist_neg).float().mean()
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        anchor, positive = batch
        negative = torch.roll(positive, shifts=1, dims=0)
        
        a_emb = self(anchor)
        p_emb = self(positive)
        n_emb = self(negative)
        
        loss = self.triplet_loss(a_emb, p_emb, n_emb)
        
        dist_pos = F.pairwise_distance(a_emb, p_emb)
        dist_neg = F.pairwise_distance(a_emb, n_emb)
        accuracy = (dist_pos < dist_neg).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)


def scan_files(config):
    """Helper to find all files."""
    print("Scanning for files...")
    sources = glob.glob(os.path.join(config['source_dir'], '**', '*.mp3'), recursive=True) + \
              glob.glob(os.path.join(config['source_dir'], '**', '*.wav'), recursive=True)
    
    noises = []
    for d in config['noise_dirs']:
        noises.extend(glob.glob(os.path.join(d, '*')))
    
    # Scan for specific types
    ir_wavs = glob.glob(os.path.join(config['ir_root'], '**', '*.wav'), recursive=True)
    ir_h5s = glob.glob(os.path.join(config['ir_root'], '**', '*.h5'), recursive=True)
    ir_pickles = glob.glob(os.path.join(config['ir_root'], '**', '*.pickle.dat'), recursive=True)
    
    print(f"Sources: {len(sources)} | Noises: {len(noises)}")
    print(f"IRs: {len(ir_wavs)} WAVs, {len(ir_h5s)} H5s, {len(ir_pickles)} Pickles")
    
    return sources, noises, ir_wavs, ir_h5s, ir_pickles

def main():
    # 1. Prepare Data
    sources, noises, ir_wavs, ir_h5s, ir_pickles = scan_files(CONFIG)
    if not sources:
        print("No sources found!")
        return
        
    random.shuffle(sources)
    split_idx = int(len(sources) * 0.9)
    train_sources = sources[:split_idx]
    val_sources = sources[split_idx:]
    print(f"Training Samples: {len(train_sources)} | Validation Samples: {len(val_sources)}")

    # 2. Preload Data into RAM
    # Load Pickles
    pickle_data = None
    if ir_pickles:
        print(f"Preloading {len(ir_pickles)} pickle files...")
        pickle_data = []
        for pf in ir_pickles:
            try:
                with open(pf, 'rb') as f:
                    pickle_data.extend(pickle.load(f))
            except Exception: pass
            
    # Load H5s (NEW)
    h5_cache = preload_h5_data(ir_h5s)

    # 3. Dataset & Loader
    # We pass the cache to the dataset
    train_dataset = RobustFingerprintDataset(train_sources, noises, ir_wavs, pickle_data, h5_cache)
    val_dataset = RobustFingerprintDataset(val_sources, noises, ir_wavs, pickle_data, h5_cache)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'],
        persistent_workers=True, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        persistent_workers=True,
        pin_memory=True
    )

    # 4. Init Model
    model = FingerprintLightningModule(
        output_dim=CONFIG['output_dim'], 
        margin=CONFIG['margin'], 
        lr=CONFIG['learning_rate']
    )

    # 5. WandB Logger
    wandb_logger = WandbLogger(project=CONFIG['project_name'], config=CONFIG)
    
    # 6. Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator="auto",      
        devices=1,
        precision="16-mixed",    
        logger=wandb_logger,
        callbacks=[ModelCheckpoint(monitor="val_loss", save_top_k=3, filename="fingerprint-{epoch:02d}-{val_loss:.2f}")],
        log_every_n_steps=10
    )

    print("Starting Training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()