import os
import glob
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Added for distance calculation
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import scipy.signal
import librosa
import sys

# --- NEW: Import WandB ---
import wandb

# Import loading logic from your existing script
# Ensure convolution_reverb.py is in the same directory
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
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 20,
    'output_dim': 128,            
    'margin': 0.5,                
    'project_name': "audio-fingerprinting-robust", # WandB Project Name
    
    # Paths (Relative to where you run the script)
    'source_dir': '../aimir/lastfm/audio/', 
    'noise_dirs': [
        'data/chatter_sounds', 
        'data/gaming_sounds'
    ],
    'ir_root': 'data/IR'
}

class FingerprintNet(nn.Module):
    """
    A simple CNN that maps a Spectrogram to a 128-d fingerprint vector.
    """
    def __init__(self):
        super(FingerprintNet, self).__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, CONFIG['output_dim'])

    def forward(self, x):
        # x shape: [batch, 1, n_mels, time]
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 Normalize embeddings so they lie on a hypersphere
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

class RobustFingerprintDataset(Dataset):
    def __init__(self, source_files, noise_files, ir_files, ir_pickle_data=None):
        self.source_files = source_files
        self.noise_files = noise_files
        self.ir_files = ir_files
        self.ir_pickle_data = ir_pickle_data # Cache for the pickle file
        
        self.sample_rate = CONFIG['sample_rate']
        self.num_samples = int(CONFIG['duration'] * self.sample_rate)
        
        # Audio Transforms
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=CONFIG['n_mels'],
            n_fft=1024,
            hop_length=256
        )
        self.db_transform = T.AmplitudeToDB()

    def _load_audio_segment(self, path):
        """Loads a random segment from the audio file."""
        try:
            # Get info first to pick a random start time
            info = torchaudio.info(path)
            total_frames = info.num_frames
            sr = info.sample_rate
            
            needed_frames = int(CONFIG['duration'] * sr)
            
            if total_frames <= needed_frames:
                waveform, _ = torchaudio.load(path)
                # Pad if too short
                pad_amt = needed_frames - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_amt))
            else:
                start_frame = random.randint(0, total_frames - needed_frames)
                waveform, _ = torchaudio.load(path, frame_offset=start_frame, num_frames=needed_frames)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # Mix to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            return waveform
            
        except Exception as e:
            # print(f"Error loading {path}: {e}")
            return torch.zeros(1, self.num_samples)

    def _get_random_ir(self):
        """Pick a random IR from the diverse dataset structure."""
        if not self.ir_files:
            return None
            
        ir_path = random.choice(self.ir_files)
        ir_sig = None
        sr_ir = 44100
        
        try:
            if ir_path.endswith('.h5') and load_ir_from_h5:
                # Random source/mic index for MIRACLE dataset
                s_idx = random.randint(0, 2) 
                m_idx = random.randint(0, 2)
                ir_sig, sr_ir = load_ir_from_h5(ir_path, source_idx=s_idx, mic_idx=m_idx)
                
            elif ir_path.endswith('.pickle.dat') and self.ir_pickle_data:
                # Random record from the loaded pickle
                record_idx = random.randint(0, len(self.ir_pickle_data) - 1)
                try:
                    raw_data = self.ir_pickle_data[record_idx][43] # Index 43 is rirData
                    ir_sig = np.array(raw_data, dtype=np.float32)
                    sr_ir = 44100 
                except:
                    return None
                    
            elif ir_path.endswith('.wav'):
                # Standard WAV
                ir_sig, sr_ir = librosa.load(ir_path, sr=None, mono=True)
                
        except Exception as e:
            return None
            
        return ir_sig, sr_ir

    def _augment(self, waveform):
        """Apply Room Impulse Response and Background Noise."""
        # 1. Convolution (Reverb)
        # 50% chance to apply reverb
        if random.random() > 0.5:
            ir_data = self._get_random_ir()
            if ir_data is not None:
                ir_sig, sr_ir = ir_data
                
                # Resample IR if needed 
                if sr_ir != self.sample_rate:
                    num_samples = int(len(ir_sig) * float(self.sample_rate) / sr_ir)
                    ir_sig = scipy.signal.resample(ir_sig, num_samples)
                
                # FFT Convolve 
                audio_np = waveform.numpy().flatten()
                convolved = scipy.signal.fftconvolve(audio_np, ir_sig, mode='full')
                
                # Crop to original length
                convolved = convolved[:self.num_samples]
                
                # Normalize
                max_val = np.max(np.abs(convolved))
                if max_val > 0:
                    convolved = convolved / max_val
                    
                waveform = torch.from_numpy(convolved).float().unsqueeze(0)

    
        # 2. Add Noise (Chatter/Gaming)
        # 50% chance to apply noise
        if random.random() > 0.5 and self.noise_files:
            noise_path = random.choice(self.noise_files)
            noise_wave = self._load_audio_segment(noise_path)
            
            # Ensure length match
            if noise_wave.size(1) > waveform.size(1):
                noise_wave = noise_wave[:, :waveform.size(1)]
            elif noise_wave.size(1) < waveform.size(1):
                # Repeat noise if too short
                repeats = 1 + waveform.size(1) // noise_wave.size(1)
                noise_wave = noise_wave.repeat(1, repeats)[:, :waveform.size(1)]
            
            # Random SNR between 0dB and 15dB
            snr_db = random.uniform(0, 15)
            
            # Calculate power
            sig_power = waveform.pow(2).mean()
            noise_power = noise_wave.pow(2).mean()
            
            if noise_power > 0:
                target_noise_power = sig_power / (10 ** (snr_db / 10))
                scale = torch.sqrt(target_noise_power / noise_power)
                waveform = waveform + (noise_wave * scale)

        return waveform

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        path = self.source_files[idx]
        
        # Load clean anchor
        clean_waveform = self._load_audio_segment(path)
        
        # Create Positive Sample (Same audio + Robust Augmentation)
        positive_waveform = self._augment(clean_waveform.clone())
        
        # Create Anchor (Can be clean or slightly augmented, let's augment it too)
        anchor_waveform = self._augment(clean_waveform.clone())
        
        # Convert to Spectrograms
        anchor_spec = self.db_transform(self.mel_spectrogram(anchor_waveform))
        positive_spec = self.db_transform(self.mel_spectrogram(positive_waveform))
        
        return anchor_spec, positive_spec

def scan_files(config):
    """Walks the directory trees to find all relevant files."""
    print("Scanning for files...")
    
    # Sources
    sources = glob.glob(os.path.join(config['source_dir'], '**', '*.mp3'), recursive=True) + \
              glob.glob(os.path.join(config['source_dir'], '**', '*.wav'), recursive=True)
    
    # Noise
    noises = []
    for d in config['noise_dirs']:
        noises.extend(glob.glob(os.path.join(d, '*'))) 
    
    # IRs
    irs = []
    irs.extend(glob.glob(os.path.join(config['ir_root'], '**', '*.wav'), recursive=True))
    irs.extend(glob.glob(os.path.join(config['ir_root'], '**', '*.h5'), recursive=True))
    irs.extend(glob.glob(os.path.join(config['ir_root'], '**', '*.pickle.dat'), recursive=True))
    
    print(f"Found {len(sources)} source files.")
    print(f"Found {len(noises)} noise files.")
    print(f"Found {len(irs)} IR files.")
    
    return sources, noises, irs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # --- WandB Init ---
    wandb.init(project=CONFIG['project_name'], config=CONFIG)

    # 1. Scan Data
    sources, noises, irs = scan_files(CONFIG)
    if not sources:
        print("No source audio found. Check path in CONFIG.")
        return

    # 2. Preload Pickle Data
    pickle_data = None
    pickle_files = [f for f in irs if f.endswith('.pickle.dat')]
    if pickle_files:
        print(f"Pre-loading {len(pickle_files)} pickle files into memory...")
        pickle_data = []
        for pf in pickle_files:
            try:
                with open(pf, 'rb') as f:
                    data = pickle.load(f)
                    pickle_data.extend(data) 
            except Exception as e:
                print(f"Failed to load pickle {pf}: {e}")
        
    # 3. Setup Dataset & Loader
    dataset = RobustFingerprintDataset(sources, noises, irs, ir_pickle_data=pickle_data)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)

    # 4. Model & Loss
    model = FingerprintNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    triplet_loss = nn.TripletMarginLoss(margin=CONFIG['margin'], p=2)

    # --- WandB Watch ---
    wandb.watch(model, log="all")

    # 5. Training Loop
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (anchor, positive) in enumerate(dataloader):
            anchor, positive = anchor.to(device), positive.to(device)
            
            # Hard Negative: Shift positive by 1
            negative = torch.roll(positive, shifts=1, dims=0)
            
            optimizer.zero_grad()
            
            # Forward
            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)
            
            loss = triplet_loss(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            # --- Calculate Accuracy ---
            # Accuracy: Fraction of triplets where dist(a, p) < dist(a, n)
            with torch.no_grad():
                dist_pos = F.pairwise_distance(a_emb, p_emb, p=2)
                dist_neg = F.pairwise_distance(a_emb, n_emb, p=2)
                correct = (dist_pos < dist_neg).float().sum().item()
                
                total_correct += correct
                total_samples += anchor.size(0)

                # Log Batch Metrics
                wandb.log({
                    "batch_loss": loss.item(), 
                    "batch_accuracy": correct / anchor.size(0)
                })
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Step [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")

        # Epoch Metrics
        avg_loss = total_loss / len(dataloader)
        avg_acc = total_correct / total_samples
        
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")
        
        # Log Epoch Metrics
        wandb.log({
            "epoch": epoch + 1,
            "epoch_loss": avg_loss,
            "epoch_accuracy": avg_acc
        })
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"fingerprint_model_epoch_{epoch+1}.pth")

    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()