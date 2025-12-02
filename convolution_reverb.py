"""
Audio Convolution Script
------------------------
This script convolves a dry audio signal with an impulse response (IR) to simulate reverb.
It supports standard audio files (.wav, .mp3) and specific research datasets (.h5, .pickle).

Usage Examples:

1. Standard WAV/MP3 File:
   python convolution_reverb.py input.wav impulse_response.wav output.wav

2. MIRACLE Dataset (.h5):
   (Uses --source for source index and --mic for microphone index)
   python convolution_reverb.py input.wav data/A1.h5 output.wav --source 2 --mic 5

3. GTU Dataset (.pickle.dat):
   (Uses --source as the record index in the list. e.g. for the 10th record:)
   python convolution_reverb.py input.wav data/RIR.pickle.dat output.wav --source 9

Dependencies:
pip install numpy scipy soundfile librosa h5py
"""
import numpy as np
import scipy.signal
import soundfile as sf
import librosa
import argparse
import sys
import os
import pickle

# Try importing h5py for the MIRACLE dataset format
try:
    import h5py
except ImportError:
    h5py = None

def load_ir_from_h5(file_path, source_idx=0, mic_idx=0):
    """
    Loads a specific impulse response from the MIRACLE .h5 dataset.
    Structure: data/impulse_response -> (ni, no, nt)
    """
    if h5py is None:
        print("Error: 'h5py' library is required to load .h5 files.")
        print("pip install h5py")
        sys.exit(1)

    print(f"Opening H5 dataset: {file_path}")
    with h5py.File(file_path, 'r') as f:
        try:
            sr_ir = f['metadata']['sampling_rate'][()]
        except KeyError:
            sr_ir = f['metadata']['sampling_rate'][()]
        
        ds = f['data']['impulse_response']
        total_sources, total_mics, total_samples = ds.shape
        
        print(f"Dataset Shape: {total_sources} sources, {total_mics} mics, {total_samples} samples")
        
        if source_idx >= total_sources:
            print(f"Error: Source index {source_idx} out of bounds (Max: {total_sources-1})")
            sys.exit(1)
        if mic_idx >= total_mics:
            print(f"Error: Mic index {mic_idx} out of bounds (Max: {total_mics-1})")
            sys.exit(1)
            
        ir_raw = ds[source_idx, mic_idx, :]
        ir_raw = ir_raw.astype(np.float32)
        
    return ir_raw, int(sr_ir)

def load_ir_from_pickle(file_path, record_index=0):
    """
    Loads an IR from the custom RIR.pickle.dat format.
    The file contains a list of lists. Index 43 is the IR data.
    Sampling rate is hardcoded to 44100 based on user spec.
    """
    print(f"Opening Pickle dataset: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    with open(file_path, 'rb') as f:
        # Load the entire list structure
        rir_data_list = pickle.load(f)

    total_records = len(rir_data_list)
    print(f"Total Records in Pickle: {total_records}")

    if record_index >= total_records:
        print(f"Error: Record index {record_index} out of bounds (Max: {total_records-1})")
        sys.exit(1)

    # Extract the specific row
    dataline = rir_data_list[record_index]
    
    # Based on provided field mapping: "rirData": 43
    try:
        raw_ir = dataline[43]
    except IndexError:
        print("Error: The selected record does not have an index 43 (rirData).")
        sys.exit(1)

    # Convert to numpy array
    ir_sig = np.array(raw_ir, dtype=np.float32)
    
    # Hardcoded based on user class definition
    sr_ir = 44100 
    
    return ir_sig, sr_ir

def convolve_audio(dry_signal_path, ir_path, output_path, mix_ratio=1.0, source_idx=0, mic_idx=0):
    """
    Convolves a dry audio signal with an impulse response.
    Supports: 
      - Standard audio files (wav, mp3)
      - MIRACLE .h5 datasets (use --source and --mic)
      - GTU .pickle.dat datasets (use --source as the record index)
    """
    # 1. Load Dry Audio
    print(f"Loading dry audio: {dry_signal_path}")
    dry_sig, sr_dry = librosa.load(dry_signal_path, sr=None, mono=False)
    
    # 2. Load Impulse Response based on extension
    print(f"Loading Impulse Response: {ir_path}")
    
    ext = ir_path.lower()
    if ext.endswith('.h5') or ext.endswith('.hdf5'):
        ir_sig, sr_ir = load_ir_from_h5(ir_path, source_idx, mic_idx)
    elif ext.endswith('.dat') or ext.endswith('.pickle') or ext.endswith('.pkl'):
        # For pickle files, we use source_idx as the main list index
        ir_sig, sr_ir = load_ir_from_pickle(ir_path, source_idx)
    else:
        # Standard wav/flac/mp3 load
        ir_sig, sr_ir = librosa.load(ir_path, sr=None, mono=False)

    # 3. Resample IR if necessary
    if sr_ir != sr_dry:
        print(f"Resampling IR from {sr_ir}Hz to {sr_dry}Hz...")
        ir_sig = librosa.resample(ir_sig, orig_sr=sr_ir, target_sr=sr_dry)
        sr_ir = sr_dry

    # 4. Standardize Dimensions to (Channels, Samples)
    if dry_sig.ndim == 1:
        dry_sig = dry_sig[np.newaxis, :]
    if ir_sig.ndim == 1:
        ir_sig = ir_sig[np.newaxis, :]

    dry_channels, dry_length = dry_sig.shape
    ir_channels, ir_length = ir_sig.shape
    
    print(f"Processing Rate: {sr_dry} Hz")
    print(f"Dry Signal: {dry_channels} ch, {dry_length} samples")
    print(f"IR Signal: {ir_channels} ch, {ir_length} samples")

    # 5. Convolution
    conv_length = dry_length + ir_length - 1
    wet_sig = np.zeros((dry_channels, conv_length))

    print("Convolving...")
    
    if ir_channels == 1 and dry_channels >= 1:
        for i in range(dry_channels):
            wet_sig[i] = scipy.signal.fftconvolve(dry_sig[i], ir_sig[0], mode='full')
            
    elif ir_channels == dry_channels:
        for i in range(dry_channels):
            wet_sig[i] = scipy.signal.fftconvolve(dry_sig[i], ir_sig[i], mode='full')
            
    elif ir_channels > 1 and dry_channels == 1:
        wet_sig = np.zeros((ir_channels, conv_length))
        for i in range(ir_channels):
            wet_sig[i] = scipy.signal.fftconvolve(dry_sig[0], ir_sig[i], mode='full')
    else:
        print("Error: Channel mismatch that cannot be automatically resolved.")
        sys.exit(1)

    # 6. Normalize
    peak = np.max(np.abs(wet_sig))
    if peak > 0:
        wet_sig = wet_sig / peak
        
    # Apply Mix
    wet_sig = wet_sig * mix_ratio
    final_output = wet_sig 
    
    # Final Safety Clamp
    max_val = np.max(np.abs(final_output))
    if max_val > 1.0:
        final_output = final_output / max_val
        print("Signal normalized to avoid clipping.")

    # 7. Save
    print(f"Saving output to: {output_path}")
    sf.write(output_path, final_output.T, sr_dry)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolve audio with an impulse response (WAV, MIRACLE H5, or Pickle dataset).")
    parser.add_argument("dry_file", help="Path to the input dry audio file (wav, mp3, etc.)")
    parser.add_argument("ir_file", help="Path to the impulse response file (.h5, .wav, .dat, .pickle)")
    parser.add_argument("output_file", help="Path for the output audio file")
    parser.add_argument("--mix", type=float, default=1.0, help="Mix ratio (0.0 to 1.0), default 1.0")
    parser.add_argument("--source", type=int, default=0, help="Source index (H5) OR Record index (Pickle), default 0")
    parser.add_argument("--mic", type=int, default=0, help="Microphone index (H5 only), default 0")

    args = parser.parse_args()

    convolve_audio(args.dry_file, args.ir_file, args.output_file, args.mix, args.source, args.mic)