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

# Import loading logic from convolution_reverb
try:
    from convolution_reverb import load_ir_from_h5
except ImportError:
    print("Warning: convolution_reverb.py not found. H5 loading will fail.")
    load_ir_from_h5 = None

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

class RobustFingerprintDataset(Dataset):
    """
    Same robust dataset logic as before, optimized for random access.
    """
    def __init__(self, source_files, noise_files, ir_files, ir_pickle_data=None):
        self.source_files = source_files
        self.noise_files = noise_files
        self.ir_files = ir_files
        self.ir_pickle_data = ir_pickle_data 
        
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
        if not self.ir_files: return None
        ir_path = random.choice(self.ir_files)
        
        try:
            if ir_path.endswith('.h5') and load_ir_from_h5:
                s_idx, m_idx = random.randint(0, 2), random.randint(0, 2)
                ir_sig, sr_ir = load_ir_from_h5(ir_path, source_idx=s_idx, mic_idx=m_idx)
                
            elif ir_path.endswith('.pickle.dat') and self.ir_pickle_data:
                record_idx = random.randint(0, len(self.ir_pickle_data) - 1)
                raw_data = self.ir_pickle_data[record_idx][43]
                ir_sig, sr_ir = np.array(raw_data, dtype=np.float32), 44100
                    
            elif ir_path.endswith('.wav'):
                ir_sig, sr_ir = librosa.load(ir_path, sr=None, mono=True)
            else:
                return None
                
            return ir_sig, sr_ir
        except Exception:
            return None

    def _augment(self, waveform):
        # 1. Reverb (50%)
        if random.random() > 0.5:
            ir_data = self._get_random_ir()
            if ir_data:
                ir_sig, sr_ir = ir_data
                if sr_ir != self.sample_rate:
                    num_s = int(len(ir_sig) * self.sample_rate / sr_ir)
                    ir_sig = scipy.signal.resample(ir_sig, num_s)
                
                audio_np = waveform.numpy().flatten()
                convolved = scipy.signal.fftconvolve(audio_np, ir_sig, mode='full')[:self.num_samples]
                
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
        
        # Create robust pairs
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
        
        # Hard Negative Mining (Shift Positive by 1)
        negative = torch.roll(positive, shifts=1, dims=0)
        
        a_emb = self(anchor)
        p_emb = self(positive)
        n_emb = self(negative)
        
        loss = self.triplet_loss(a_emb, p_emb, n_emb)
        
        # Compute Accuracy
        dist_pos = F.pairwise_distance(a_emb, p_emb)
        dist_neg = F.pairwise_distance(a_emb, n_emb)
        accuracy = (dist_pos < dist_neg).float().mean()
        
        # Log training metrics
        # on_step=True logs every step (noisy but detailed)
        # on_epoch=True logs the average at the end of the epoch
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Calculates performance on unseen data."""
        anchor, positive = batch
        negative = torch.roll(positive, shifts=1, dims=0)
        
        a_emb = self(anchor)
        p_emb = self(positive)
        n_emb = self(negative)
        
        loss = self.triplet_loss(a_emb, p_emb, n_emb)
        
        dist_pos = F.pairwise_distance(a_emb, p_emb)
        dist_neg = F.pairwise_distance(a_emb, n_emb)
        accuracy = (dist_pos < dist_neg).float().mean()
        
        # Log validation metrics (Only on epoch)
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
    
    irs = []
    irs.extend(glob.glob(os.path.join(config['ir_root'], '**', '*.wav'), recursive=True))
    irs.extend(glob.glob(os.path.join(config['ir_root'], '**', '*.h5'), recursive=True))
    irs.extend(glob.glob(os.path.join(config['ir_root'], '**', '*.pickle.dat'), recursive=True))
    
    print(f"Sources: {len(sources)} | Noises: {len(noises)} | IRs: {len(irs)}")
    return sources, noises, irs

def main():
    # 1. Prepare Data
    sources, noises, irs = scan_files(CONFIG)
    if not sources:
        print("No sources found!")
        return
        
    # --- NEW: Split Train/Val ---
    random.shuffle(sources)
    split_idx = int(len(sources) * 0.9)
    train_sources = sources[:split_idx]
    val_sources = sources[split_idx:]
    print(f"Training Samples: {len(train_sources)} | Validation Samples: {len(val_sources)}")

    # Preload Pickle
    pickle_data = None
    pickle_files = [f for f in irs if f.endswith('.pickle.dat')]
    if pickle_files:
        print("Preloading pickle data...")
        pickle_data = []
        for pf in pickle_files:
            try:
                with open(pf, 'rb') as f:
                    pickle_data.extend(pickle.load(f))
            except Exception: pass

    # Dataset & Loader
    train_dataset = RobustFingerprintDataset(train_sources, noises, irs, pickle_data)
    val_dataset = RobustFingerprintDataset(val_sources, noises, irs, pickle_data)
    
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

    # 2. Init Model
    model = FingerprintLightningModule(
        output_dim=CONFIG['output_dim'], 
        margin=CONFIG['margin'], 
        lr=CONFIG['learning_rate']
    )

    # 3. WandB Logger
    wandb_logger = WandbLogger(project=CONFIG['project_name'], config=CONFIG)
    
    # 4. Trainer
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
    # Pass both loaders to fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()