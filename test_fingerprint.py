import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import os
import glob
import random
import numpy as np
import scipy.signal
import argparse
from tqdm import tqdm
import librosa

# Import model and config from your training script
try:
    from train_fingerprint import FingerprintLightningModule as FingerprintNet, CONFIG, load_ir_from_h5
except ImportError:
    print("Error: Could not import from train_fingerprint.py. Make sure it's in the same directory.")
    exit(1)

# Ensure reproducibility for testing
random.seed(42)
torch.manual_seed(42)

class AudioFingerprinter:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = FingerprintNet().to(device)
        
        # Load Checkpoint
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")
            
        self.model.eval()
        
        # Audio Processing (Must match training exactly)
        self.sample_rate = CONFIG['sample_rate']
        self.duration = CONFIG['duration']
        self.num_samples = int(self.duration * self.sample_rate)
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=CONFIG['n_mels'],
            n_fft=1024,
            hop_length=256
        ).to(device)
        self.db_transform = T.AmplitudeToDB().to(device)

    def process_audio(self, waveform):
        """Prepares audio for the model (Resample, Mix, Crop/Pad)."""
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample
        if waveform.size(1) != self.num_samples:
            resampler = T.Resample(orig_freq=waveform.size(1)//self.sample_rate, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        # Pad or Crop to fixed duration
        if waveform.size(1) < self.num_samples:
            pad_amt = self.num_samples - waveform.size(1)
            waveform = F.pad(waveform, (0, pad_amt))
        elif waveform.size(1) > self.num_samples:
            # Let's center-crop
            mid = waveform.size(1) // 2
            half = self.num_samples // 2
            start = max(0, mid - half)
            waveform = waveform[:, start:start+self.num_samples]
            
        return waveform.to(self.device)

    def compute_fingerprint(self, waveform):
        """Generates the 128-d vector."""
        with torch.no_grad():
            processed = self.process_audio(waveform)
            spectrogram = self.db_transform(self.mel_transform(processed))
            spectrogram = spectrogram.unsqueeze(0) # Add batch dim
            embedding = self.model(spectrogram)
        return embedding

def build_database(fingerprinter, source_files):
    """
    Creates a dictionary: { file_path: embedding_vector }
    """
    print("\n--- Building Reference Database ---")
    database = {}
    
    for path in tqdm(source_files):
        try:
            waveform, sr = torchaudio.load(path)
            if sr != fingerprinter.sample_rate:
                resampler = T.Resample(sr, fingerprinter.sample_rate)
                waveform = resampler(waveform)
            
            # Compute fingerprint of the CLEAN source
            emb = fingerprinter.compute_fingerprint(waveform)
            database[path] = emb
            
        except Exception as e:
            # print(f"Skipping {path}: {e}")
            pass
            
    print(f"Indexed {len(database)} tracks.")
    return database

def distort_audio(waveform, sample_rate, noise_files, ir_files):
    """
    Applies Reverb + Noise to create a test query.
    Similar to training logic but applied on-demand.
    """
    # 1. Reverb
    if ir_files and random.random() > 0.3: # 70% chance of reverb
        ir_path = random.choice(ir_files)
        # Simplified IR loading for test script (ignoring H5 complexity for brevity, or reusing logic)
        try:
            # Standard wav load for speed, or add H5 logic here if strictly needed
            if ir_path.endswith('.wav'):
                ir_sig, sr_ir = librosa.load(ir_path, sr=sample_rate, mono=True)
                audio_np = waveform.numpy().flatten()
                convolved = scipy.signal.fftconvolve(audio_np, ir_sig, mode='full')
                convolved = convolved[:waveform.size(1)]
                # Normalize
                max_val = np.max(np.abs(convolved))
                if max_val > 0:
                    convolved = convolved / max_val
                waveform = torch.from_numpy(convolved).float().unsqueeze(0)
        except:
            pass

    # 2. Noise
    if noise_files and random.random() > 0.3: # 70% chance of noise
        noise_path = random.choice(noise_files)
        try:
            noise_wave, n_sr = torchaudio.load(noise_path)
            if n_sr != sample_rate:
                noise_wave = T.Resample(n_sr, sample_rate)(noise_wave)
            if noise_wave.shape[0] > 1:
                noise_wave = torch.mean(noise_wave, dim=0, keepdim=True)
                
            # Loop/Cut noise
            if noise_wave.size(1) < waveform.size(1):
                repeats = 1 + waveform.size(1) // noise_wave.size(1)
                noise_wave = noise_wave.repeat(1, repeats)
            noise_wave = noise_wave[:, :waveform.size(1)]
            
            # Heavy noise for testing (SNR 5-10dB)
            snr_db = random.uniform(5, 10)
            sig_power = waveform.pow(2).mean()
            noise_power = noise_wave.pow(2).mean()
            
            if noise_power > 0:
                scale = torch.sqrt(sig_power / (10**(snr_db/10)) / noise_power)
                waveform = waveform + (noise_wave * scale)
        except:
            pass
            
    return waveform

def run_evaluation(fingerprinter, database, source_files, noise_files, ir_files, num_tests=100):
    """
    Runs N tests where we distort a file and see if we can find it in the DB.
    """
    print(f"\n--- Running Evaluation ({num_tests} queries) ---")
    correct = 0
    
    # Pre-stack database for fast matrix multiplication
    # db_keys: list of paths
    # db_tensor: tensor of shape [N_tracks, 128]
    db_keys = list(database.keys())
    db_tensor = torch.cat(list(database.values()), dim=0) 
    
    for i in tqdm(range(num_tests)):
        # 1. Pick random target
        target_path = random.choice(source_files)
        
        # 2. Load and Distort
        waveform, sr = torchaudio.load(target_path)
        if sr != fingerprinter.sample_rate:
            waveform = T.Resample(sr, fingerprinter.sample_rate)(waveform)
            
        distorted_wave = distort_audio(waveform.clone(), fingerprinter.sample_rate, noise_files, ir_files)
        
        # 3. Compute Query Fingerprint
        query_emb = fingerprinter.compute_fingerprint(distorted_wave)
        
        # 4. Search (Cosine Similarity)
        # normalize query matches DB normalization
        # Cosine Similarity is basically Dot Product on normalized vectors
        scores = torch.mm(query_emb, db_tensor.t())
        
        # Get Top Match
        best_idx = torch.argmax(scores, dim=1).item()
        predicted_path = db_keys[best_idx]
        
        if predicted_path == target_path:
            correct += 1
            
    accuracy = (correct / num_tests) * 100
    print(f"\nResults:")
    print(f"Total Tests: {num_tests}")
    print(f"Correct Matches: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Test Audio Fingerprinting Model")
    parser.add_argument("--model", type=str, default="fingerprint_model_epoch_20.pth", help="Path to .pth checkpoint")
    parser.add_argument("--tests", type=int, default=100, help="Number of test queries to run")
    args = parser.parse_args()

    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fingerprinter = AudioFingerprinter(args.model, device)
    
    # 2. Get File Lists (Reusing scan_files from train script logic)
    # We do a quick glob manually here to be safe
    source_dir = CONFIG['source_dir']
    noise_dirs = CONFIG['noise_dirs']
    ir_root = CONFIG['ir_root']
    
    source_files = glob.glob(os.path.join(source_dir, '**', '*.mp3'), recursive=True) + \
                   glob.glob(os.path.join(source_dir, '**', '*.wav'), recursive=True)
                   
    noise_files = []
    for d in noise_dirs:
        noise_files.extend(glob.glob(os.path.join(d, '*')))
        
    ir_files = glob.glob(os.path.join(ir_root, '**', '*.wav'), recursive=True)
    
    if not source_files:
        print("No source files found.")
        return

    # 3. Build DB
    db = build_database(fingerprinter, source_files)
    
    # 4. Evaluate
    run_evaluation(fingerprinter, db, source_files, noise_files, ir_files, num_tests=args.tests)

if __name__ == "__main__":
    main()