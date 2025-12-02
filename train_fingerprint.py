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
    'num_workers': 8,
    'cache_dir': './dataset_cache_v2',  # Changed folder to ensure fresh cache
    'samples_per_track': 10,            # NEW: How many variations to generate per song
    
    # Paths
    'source_dir': '../aimir/lastfm/audio/', 
    'noise_dirs': [
        'data/chatter_sounds', 
        'data/gaming_sounds'
    ],
    'ir_root': 'data/IR'
}

def resample_array(array, orig_sr, target_sr):
    """Helper to resample numpy arrays efficiently."""
    if orig_sr == target_sr:
        return array
    num_samples = int(len(array) * target_sr / orig_sr)
    return scipy.signal.resample(array, num_samples)

def preload_ir_data(ir_wavs, ir_h5s, ir_pickles, target_sr=8000):
    """
    Loads ALL IRs into a unified list of float32 arrays at 8kHz.
    """
    print(f"\n--- Preloading and Resampling IRs to {target_sr}Hz ---")
    ir_cache = []
    
    # 1. Load WAVs
    if ir_wavs:
        print(f"Processing {len(ir_wavs)} WAV IRs...")
        for path in ir_wavs:
            try:
                y, _ = librosa.load(path, sr=target_sr, mono=True)
                ir_cache.append(y.astype(np.float32))
            except Exception: pass

    # 2. Load Pickles
    if ir_pickles:
        print(f"Processing {len(ir_pickles)} Pickle IRs...")
        for path in ir_pickles:
            try:
                with open(path, 'rb') as f:
                    data_list = pickle.load(f)
                    for record in data_list:
                        try:
                            raw = np.array(record[43], dtype=np.float32)
                            resampled = resample_array(raw, 44100, target_sr)
                            ir_cache.append(resampled)
                        except: pass
            except Exception: pass

    # 3. Load H5s
    if ir_h5s and h5py:
        print(f"Processing {len(ir_h5s)} H5 IRs...")
        for path in ir_h5s:
            try:
                with h5py.File(path, 'r') as f:
                    data = None
                    if 'data' in f and 'impulse_response' in f['data']:
                        data = f['data']['impulse_response'][:]
                    elif 'impulse_response' in f:
                        data = f['impulse_response'][:]
                    
                    if data is not None:
                        flat_data = data.reshape(-1, data.shape[-1])
                        if flat_data.shape[0] > 500:
                            indices = np.random.choice(flat_data.shape[0], 500, replace=False)
                            flat_data = flat_data[indices]

                        for row in flat_data:
                            resampled = resample_array(row, 48000, target_sr)
                            ir_cache.append(resampled.astype(np.float32))
            except Exception as e:
                print(f"Error loading H5 {path}: {e}")

    print(f"--- Cached {len(ir_cache)} Total Impulse Responses at {target_sr}Hz ---")
    return ir_cache

class RobustFingerprintDataset(Dataset):
    def __init__(self, source_files, noise_files, ir_cache, samples_per_track=CONFIG['samples_per_track']):
        """
        Pre-computes spectrograms during initialization to avoid audio loading during training.
        """
        self.source_files = source_files
        self.noise_files = noise_files
        self.ir_cache = ir_cache 
        
        self.sample_rate = CONFIG['sample_rate']
        self.num_samples = int(CONFIG['duration'] * self.sample_rate)
        
        # Transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=CONFIG['n_mels'],
            n_fft=1024,
            hop_length=256,
            power=2.0
        )
        self.db_transform = T.AmplitudeToDB()

        # --- PRE-COMPUTATION LOOP ---
        self.data = []
        if len(source_files) > 0:
            print(f"Pre-computing dataset: {len(source_files)} tracks x {samples_per_track} pairs...")
            
            for path in source_files:
                # Generate N variations per song
                for _ in range(samples_per_track):
                    try:
                        # 1. Load random segment
                        clean_wave = self._load_audio_segment(path)
                        if clean_wave.sum() == 0: continue # Skip failed loads
                        
                        # 2. Augment twice
                        anchor_wave = self._augment(clean_wave.clone())
                        positive_wave = self._augment(clean_wave.clone())
                        
                        # 3. Compute Spectrograms
                        a_spec = self.db_transform(self.mel_spectrogram(anchor_wave))
                        p_spec = self.db_transform(self.mel_spectrogram(positive_wave))
                        
                        # 4. Store Tensors (FP32) to RAM
                        self.data.append((a_spec, p_spec))
                        
                    except Exception as e:
                        pass
                        
            print(f"Finished pre-computing {len(self.data)} total samples.")

    def _load_audio_segment(self, path):
        try:
            waveform, sr = torchaudio.load(path)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.size(1) < self.num_samples:
                pad_amt = self.num_samples - waveform.size(1)
                waveform = F.pad(waveform, (0, pad_amt))
            elif waveform.size(1) > self.num_samples:
                start = random.randint(0, waveform.size(1) - self.num_samples)
                waveform = waveform[:, start:start+self.num_samples]
                
            return waveform
        except Exception:
            return torch.zeros(1, self.num_samples)

    def _augment(self, waveform):
        # 1. Reverb (50%) - Uses cached 8k IRs
        if self.ir_cache and random.random() > 0.5:
            ir_sig = random.choice(self.ir_cache)
            audio_np = waveform.numpy().flatten()
            
            # Fast convolution (signals are small now)
            convolved = scipy.signal.fftconvolve(audio_np, ir_sig, mode='full')
            convolved = convolved[:self.num_samples]
            
            max_val = np.max(np.abs(convolved))
            if max_val > 0: convolved /= max_val
            waveform = torch.from_numpy(convolved).float().unsqueeze(0)

        # 2. Noise (50%)
        if self.noise_files and random.random() > 0.5:
            noise_path = random.choice(self.noise_files)
            noise_wave = self._load_audio_segment(noise_path)
            
            snr_db = random.uniform(0, 15)
            sig_p = waveform.pow(2).mean()
            noise_p = noise_wave.pow(2).mean()
            
            if noise_p > 0:
                scale = torch.sqrt(sig_p / (10**(snr_db/10)) / noise_p)
                waveform = waveform + (noise_wave * scale)

        return waveform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extremely fast: just return the tensor from RAM
        return self.data[idx]


class FingerprintLightningModule(pl.LightningModule):
    def __init__(self, output_dim=128, margin=0.5, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
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
    print("Scanning for files...")
    sources = glob.glob(os.path.join(config['source_dir'], '**', '*.mp3'), recursive=True) + \
              glob.glob(os.path.join(config['source_dir'], '**', '*.wav'), recursive=True)
    
    noises = []
    for d in config['noise_dirs']:
        noises.extend(glob.glob(os.path.join(d, '*')))
    
    ir_wavs = glob.glob(os.path.join(config['ir_root'], '**', '*.wav'), recursive=True)
    ir_h5s = glob.glob(os.path.join(config['ir_root'], '**', '*.h5'), recursive=True)
    ir_pickles = glob.glob(os.path.join(config['ir_root'], '**', '*.pickle.dat'), recursive=True)
    
    print(f"Sources: {len(sources)} | Noises: {len(noises)}")
    print(f"IRs: {len(ir_wavs)} WAVs, {len(ir_h5s)} H5s, {len(ir_pickles)} Pickles")
    
    return sources, noises, ir_wavs, ir_h5s, ir_pickles

def main():
    # Ensure cache dir exists
    os.makedirs(CONFIG['cache_dir'], exist_ok=True)
    
    train_pt_path = os.path.join(CONFIG['cache_dir'], 'train_dataset.pt')
    val_pt_path = os.path.join(CONFIG['cache_dir'], 'val_dataset.pt')

    # --- Check Cache ---
    if os.path.exists(train_pt_path) and os.path.exists(val_pt_path):
        print(f"Loading datasets from cache: {CONFIG['cache_dir']}")
        train_dataset = torch.load(train_pt_path)
        val_dataset = torch.load(val_pt_path)
        print("Datasets loaded successfully.")
    else:
        # 1. Prepare Data from scratch
        sources, noises, ir_wavs, ir_h5s, ir_pickles = scan_files(CONFIG)
        if not sources:
            print("No sources found!")
            return
            
        random.shuffle(sources)
        split_idx = int(len(sources) * 0.9)
        train_sources = sources[:split_idx]
        val_sources = sources[split_idx:]
        print(f"Training Samples: {len(train_sources)} | Validation Samples: {len(val_sources)}")

        # 2. Preload AND Resample IRs
        ir_cache = preload_ir_data(ir_wavs, ir_h5s, ir_pickles, target_sr=CONFIG['sample_rate'])

        # 3. Create Datasets
        # This will now TRIGGER the pre-computation loop
        train_dataset = RobustFingerprintDataset(train_sources, noises, ir_cache)
        val_dataset = RobustFingerprintDataset(val_sources, noises, ir_cache)
        
        # 4. Save to Cache
        print(f"Saving datasets to {CONFIG['cache_dir']}...")
        torch.save(train_dataset, train_pt_path)
        torch.save(val_dataset, val_pt_path)
        print("Datasets saved.")
    
    # --- Loaders ---
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

    # Init Model
    model = FingerprintLightningModule(
        output_dim=CONFIG['output_dim'], 
        margin=CONFIG['margin'], 
        lr=CONFIG['learning_rate']
    )

    # WandB Logger
    wandb_logger = WandbLogger(project=CONFIG['project_name'], config=CONFIG)
    
    # Trainer
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