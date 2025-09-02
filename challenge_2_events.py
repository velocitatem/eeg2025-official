"""
Challenge 2: Event-Driven EEG Processing for P-Factor Prediction
================================================================

This script demonstrates how to use the new event-driven EEG architecture
for predicting p-factor from EEG recordings. Instead of treating EEG as 
continuous time series, we convert each recording into a sequence of 
meaningful neural interaction events.

This approach is inspired by Universal Behavior Profile (UBP) modeling 
from recommender systems, where we treat brain regions as entities that
interact with cognitive stimuli through discrete events.
"""

import math
import os
import random
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.nn.functional import l1_loss
from eeg_events import EEGEventExtractor, EEGEventSequenceDataset, EEGEventTransformer


class MockEEGDataset(Dataset):
    """
    Mock EEG dataset for testing the event-driven pipeline.
    
    This simulates the structure of the actual EEG dataset without requiring
    the full EEGDash installation and data download.
    """
    
    def __init__(self, n_samples=100, n_channels=128, n_timepoints=200, sfreq=100):
        self.n_samples = n_samples
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.sfreq = sfreq
        
        # Generate synthetic EEG data and labels
        np.random.seed(42)
        self.eeg_data = []
        self.labels = []
        self.metadata = []
        
        for i in range(n_samples):
            # Create synthetic EEG with realistic characteristics
            base_signal = np.random.randn(n_channels, n_timepoints) * 50  # Microvolts
            
            # Add some frequency-specific activity
            t = np.linspace(0, n_timepoints/sfreq, n_timepoints)
            
            # Add alpha activity (8-13 Hz) in occipital regions
            alpha_channels = list(range(96, 128))  # Occipital
            for ch in alpha_channels:
                alpha_freq = 8 + np.random.rand() * 5  # 8-13 Hz
                base_signal[ch] += 20 * np.sin(2 * np.pi * alpha_freq * t + np.random.rand() * 2 * np.pi)
            
            # Add theta activity (4-8 Hz) in frontal regions
            theta_channels = list(range(0, 32))  # Frontal
            for ch in theta_channels:
                theta_freq = 4 + np.random.rand() * 4  # 4-8 Hz
                base_signal[ch] += 15 * np.sin(2 * np.pi * theta_freq * t + np.random.rand() * 2 * np.pi)
            
            # Add some noise
            base_signal += np.random.randn(n_channels, n_timepoints) * 5
            
            self.eeg_data.append(base_signal)
            
            # Generate synthetic p-factor (correlated with some EEG characteristics)
            alpha_power = np.mean([np.var(base_signal[ch]) for ch in alpha_channels])
            theta_power = np.mean([np.var(base_signal[ch]) for ch in theta_channels])
            p_factor = 0.3 * (alpha_power / 1000) - 0.2 * (theta_power / 1000) + np.random.randn() * 0.1
            
            self.labels.append(p_factor)
            self.metadata.append({
                'subject': f'sub_{i:03d}',
                'session': 'session_1',
                'task': 'resting_state',
                'age': 20 + np.random.randint(0, 40),
                'sex': 'M' if i % 2 else 'F'
            })
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        X = self.eeg_data[idx]  # Shape: (n_channels, n_timepoints)
        y = self.labels[idx]
        metadata = self.metadata[idx]
        
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), metadata


def create_event_dataset(raw_dataset, event_extractor):
    """Convert raw EEG dataset to event sequences"""
    event_sequences = []
    labels = []
    
    print("Converting EEG data to event sequences...")
    
    for i, (eeg_data, label, metadata) in enumerate(raw_dataset):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{len(raw_dataset)}")
        
        # Convert tensor to numpy
        eeg_np = eeg_data.numpy()
        
        # Extract events
        context = {
            'task': metadata['task'],
            'subject': metadata['subject'], 
            'age': metadata['age'],
            'sex': metadata['sex']
        }
        events = event_extractor.extract_events(eeg_np, context)
        
        event_sequences.append(events)
        labels.append(float(label))
    
    print(f"Extracted {len(event_sequences)} event sequences")
    print(f"Average events per sequence: {np.mean([len(seq) for seq in event_sequences]):.1f}")
    
    return EEGEventSequenceDataset(event_sequences, labels)


def train_event_model(dataset, device, epochs=5):
    """Train the event-driven EEG model"""
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model
    model = EEGEventTransformer(
        n_regions=len(dataset.region_vocab),
        n_event_types=len(dataset.event_type_vocab),
        n_freq_bands=len(dataset.freq_band_vocab),
        d_model=128,
        n_heads=8,
        n_layers=4,
        max_seq_len=dataset.max_sequence_length,
        dropout=0.1
    ).to(device)
    
    print(f"Model architecture:")
    print(f"- Regions vocabulary size: {len(dataset.region_vocab)}")
    print(f"- Event types vocabulary size: {len(dataset.event_type_vocab)}")
    print(f"- Frequency bands vocabulary size: {len(dataset.freq_band_vocab)}")
    print(f"- Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (event_sequences, labels) in enumerate(dataloader):
            event_sequences = event_sequences.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(event_sequences)
            
            # Compute loss
            loss = l1_loss(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
    
    return model


def compare_with_traditional_approach(raw_dataset, event_dataset, device):
    """Compare event-driven approach with traditional EEGNetv4"""
    
    print("\n=== Comparison with Traditional Approach ===")
    
    try:
        # Try to import EEGNetv4 (might not work without braindecode)
        from braindecode.models import EEGNetv4
        
        # Traditional model
        traditional_model = EEGNetv4(
            n_chans=128, 
            n_outputs=1, 
            n_times=200
        ).to(device)
        
        print(f"Traditional EEGNetv4 parameters: {sum(p.numel() for p in traditional_model.parameters()):,}")
        
    except ImportError:
        print("Braindecode not available - skipping traditional model comparison")
        return
    
    # Create simple traditional dataset wrapper
    class TraditionalDataset(Dataset):
        def __init__(self, raw_dataset):
            self.raw_dataset = raw_dataset
            
        def __len__(self):
            return len(self.raw_dataset)
            
        def __getitem__(self, idx):
            X, y, _ = self.raw_dataset[idx]
            return X, y
    
    traditional_dataset = TraditionalDataset(raw_dataset)
    traditional_loader = DataLoader(traditional_dataset, batch_size=8, shuffle=True)
    
    # Train traditional model for comparison
    optimizer = optim.AdamW(traditional_model.parameters(), lr=0.001)
    traditional_model.train()
    
    print("Training traditional EEGNetv4...")
    for epoch in range(2):  # Just a couple epochs for comparison
        for batch_idx, (X, y) in enumerate(traditional_loader):
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            predictions = traditional_model(X)
            loss = l1_loss(predictions, y)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 5 == 0:
                print(f"Traditional - Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")


def main():
    """Main execution function"""
    
    print("=== EEG Event-Driven Architecture Demo ===")
    print("Inspired by Universal Behavior Profile (UBP) modeling from RecSys")
    print()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create mock dataset (in real scenario, this would be the EEGChallengeDataset)
    print("Creating mock EEG dataset...")
    raw_dataset = MockEEGDataset(n_samples=50, n_channels=128, n_timepoints=200)
    print(f"Created dataset with {len(raw_dataset)} samples")
    
    # Initialize event extractor
    print("Initializing event extractor...")
    event_extractor = EEGEventExtractor(
        sfreq=100,
        window_size=0.5,  # 0.5 second windows
        overlap=0.25,     # 0.25 second overlap
        similarity_threshold=0.8
    )
    
    # Convert to event sequences
    event_dataset = create_event_dataset(raw_dataset, event_extractor)
    
    # Display some statistics
    print(f"\n=== Event Dataset Statistics ===")
    print(f"Number of sequences: {len(event_dataset)}")
    print(f"Max sequence length: {event_dataset.max_sequence_length}")
    print(f"Region vocabulary: {list(event_dataset.region_vocab.keys())}")
    print(f"Event type vocabulary: {list(event_dataset.event_type_vocab.keys())}")
    print(f"Frequency band vocabulary: {list(event_dataset.freq_band_vocab.keys())}")
    
    # Train the event-driven model
    print(f"\n=== Training Event-Driven Model ===")
    trained_model = train_event_model(event_dataset, device, epochs=3)
    
    # Save the model
    torch.save(trained_model.state_dict(), "event_model_weights_challenge_2.pt")
    print(f"Model saved to event_model_weights_challenge_2.pt")
    
    # Compare with traditional approach
    compare_with_traditional_approach(raw_dataset, event_dataset, device)
    
    print(f"\n=== Demo Complete ===")
    print("Key advantages of event-driven approach:")
    print("- Better interpretability (events have cognitive meaning)")
    print("- Leverages RecSys sequential modeling techniques")
    print("- Sparse representation (fewer parameters for same information)")
    print("- Better transfer learning potential across tasks/subjects")
    

if __name__ == "__main__":
    main()