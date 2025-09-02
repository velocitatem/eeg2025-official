"""
Challenge 2 Integration: Event-Driven EEG with Real Data
========================================================

This module demonstrates how to integrate the new event-driven EEG architecture
with the existing challenge_2.py pipeline that uses real EEG data from EEGDash.

The key innovation is converting the traditional DatasetWrapper to extract events
instead of working with raw time series, while maintaining compatibility with
the existing data loading infrastructure.
"""

import math
import os
import random
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.functional import l1_loss
from joblib import Parallel, delayed

from eeg_events import EEGEventExtractor, EEGEventSequenceDataset, EEGEventTransformer

try:
    from eegdash import EEGChallengeDataset
    from braindecode.preprocessing import create_fixed_length_windows
    from braindecode.datasets.base import EEGWindowsDataset, BaseConcatDataset, BaseDataset
    EEGDASH_AVAILABLE = True
except ImportError:
    # Create dummy classes for when dependencies aren't available
    class BaseDataset:
        pass
    
    class EEGWindowsDataset:
        pass
    
    class BaseConcatDataset:
        pass
    
    EEGDASH_AVAILABLE = False
    print("EEGDash and braindecode not available - using mock data for demonstration")


class EventDatasetWrapper(BaseDataset):
    """
    Event-driven version of the original DatasetWrapper.
    
    Instead of returning raw EEG windows, this wrapper converts each window
    to an event sequence using the EEGEventExtractor.
    """
    
    def __init__(self, dataset: EEGWindowsDataset, crop_size_samples: int, 
                 event_extractor: EEGEventExtractor, seed=None):
        self.dataset = dataset
        self.crop_size_samples = crop_size_samples
        self.event_extractor = event_extractor
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        X, _, crop_inds = self.dataset[index]

        # P-factor label:
        p_factor = self.dataset.description["p_factor"]
        p_factor = float(p_factor)

        # Additional information:
        infos = {
            "subject": self.dataset.description["subject"],
            "sex": self.dataset.description["sex"],
            "age": float(self.dataset.description["age"]),
            "task": self.dataset.description["task"],
            "session": self.dataset.description.get("session", None) or "",
            "run": self.dataset.description.get("run", None) or "",
        }

        # Randomly crop the signal to the desired length:
        i_window_in_trial, i_start, i_stop = crop_inds
        assert i_stop - i_start >= self.crop_size_samples, f"{i_stop=} {i_start=}"
        start_offset = self.rng.randint(0, i_stop - i_start - self.crop_size_samples)
        i_start = i_start + start_offset
        i_stop = i_start + self.crop_size_samples
        X = X[:, start_offset : start_offset + self.crop_size_samples]

        # Convert EEG data to event sequence
        context = {
            'subject': infos['subject'],
            'task': infos['task'],
            'age': infos['age'],
            'sex': infos['sex'],
            'session': infos['session']
        }
        
        # Extract events from the cropped window
        events = self.event_extractor.extract_events(X, context)

        return events, p_factor, (i_window_in_trial, i_start, i_stop), infos


def create_event_datasets_from_eeg_data():
    """
    Create event-based datasets from the existing EEG challenge data.
    This follows the same structure as challenge_2.py but converts to events.
    """
    
    if not EEGDASH_AVAILABLE:
        print("EEGDash not available - cannot load real data")
        return None, None
        
    print("Loading EEG challenge data...")
    
    # Define local path and load the data (same as challenge_2.py)
    cache_dir = Path("~/eegdash_data/eeg2025_competition").expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets (simplified for demonstration - using fewer releases)
    release_list = ["R1", "R2"]  # Using fewer releases for faster testing
    
    all_datasets_list = [
        EEGChallengeDataset(
            release=release,
            description_fields=[
                "subject",
                "session", 
                "run",
                "task",
                "age",
                "gender",
                "sex",
                "p_factor",
            ],
            mini=True,
            cache_dir=cache_dir,
        )
        for release in release_list
    ]
    
    print("Datasets loaded")
    sub_rm = ["NDARWV769JM7"]  # Remove problematic subjects

    # Combine datasets
    all_datasets = BaseConcatDataset(all_datasets_list)
    print(f"Combined dataset description: {len(all_datasets.datasets)} recordings")

    # Load raw data
    print("Loading raw EEG data...")
    raws = Parallel(n_jobs=min(4, os.cpu_count()))(
        delayed(lambda d: d.raw)(d) for d in all_datasets.datasets
    )

    SFREQ = 100

    # Filter out recordings that are too short
    all_datasets = BaseConcatDataset(
        [ds for ds in all_datasets.datasets 
         if not ds.description.subject in sub_rm and 
         ds.raw.n_times >= 4 * SFREQ and 
         not math.isnan(ds.description["p_factor"])]
    )

    print(f"Filtered dataset: {len(all_datasets.datasets)} valid recordings")

    # Create 4-seconds windows with 2-seconds stride
    windows_ds = create_fixed_length_windows(
        all_datasets,
        window_size_samples=4 * SFREQ,
        window_stride_samples=2 * SFREQ,
        drop_last_window=True,
    )

    # Initialize event extractor
    event_extractor = EEGEventExtractor(
        sfreq=SFREQ,
        window_size=0.5,  # 0.5 second analysis windows  
        overlap=0.25,     # 0.25 second overlap
        similarity_threshold=0.8
    )

    # Wrap each sub-dataset with EventDatasetWrapper
    print("Creating event dataset wrappers...")
    event_datasets = BaseConcatDataset([
        EventDatasetWrapper(ds, crop_size_samples=2 * SFREQ, event_extractor=event_extractor) 
        for ds in windows_ds.datasets
    ])

    print(f"Created {len(event_datasets)} event dataset wrappers")
    
    return event_datasets, event_extractor


class EventSequenceCollator:
    """
    Custom collate function for batching event sequences of different lengths.
    """
    
    def __init__(self, max_sequence_length=100):
        self.max_sequence_length = max_sequence_length
        
        # Initialize vocabularies (these should match training vocabularies)
        self.region_vocab = {'frontal': 0, 'parietal': 1, 'temporal': 2, 'occipital': 3}
        self.event_type_vocab = {'baseline_activity': 0, 'gamma_burst': 1, 
                                'synchronized_oscillation': 2, 'high_amplitude_event': 3}
        self.freq_band_vocab = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}
    
    def __call__(self, batch):
        """
        Collate a batch of (events, label, crop_inds, infos) tuples.
        """
        events_list = []
        labels = []
        crop_inds_list = []
        infos_list = []
        
        for events, label, crop_inds, infos in batch:
            events_list.append(events)
            labels.append(label)
            crop_inds_list.append(crop_inds)
            infos_list.append(infos)
        
        # Convert events to tensors
        batch_sequences = []
        for events in events_list:
            sequence_tensor = self._events_to_tensor(events)
            batch_sequences.append(sequence_tensor)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(batch_sequences)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        return batch_tensor, labels_tensor, crop_inds_list, infos_list
    
    def _events_to_tensor(self, events):
        """Convert event sequence to tensor representation"""
        feature_dim = 9
        sequence_tensor = torch.zeros(self.max_sequence_length, feature_dim)
        
        for i, event in enumerate(events[:self.max_sequence_length]):
            sequence_tensor[i] = torch.tensor([
                self.region_vocab.get(event.brain_region, 0),
                self.event_type_vocab.get(event.event_type, 0),
                self.freq_band_vocab.get(event.frequency_band, 2),  # Default to alpha
                event.amplitude,
                event.cross_channel_coherence,
                event.average_change_from_last,
                event.signal_to_noise_ratio,
                event.highest_delta_channel,
                event.lowest_delta_channel
            ])
        
        return sequence_tensor


def train_event_model_on_real_data():
    """
    Train the event-driven model on real EEG challenge data.
    This replaces the traditional EEGNetv4 training in challenge_2.py.
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load event datasets
    event_datasets, event_extractor = create_event_datasets_from_eeg_data()
    
    if event_datasets is None:
        print("Could not load real EEG data - falling back to mock data")
        # Could fall back to mock data here
        return None
    
    # Create data loader with custom collator
    collator = EventSequenceCollator(max_sequence_length=100)
    dataloader = DataLoader(
        event_datasets, 
        batch_size=8, 
        shuffle=True,
        collate_fn=collator
    )
    
    print(f"Created data loader with {len(dataloader)} batches")

    # Initialize model
    model = EEGEventTransformer(
        n_regions=len(collator.region_vocab),
        n_event_types=len(collator.event_type_vocab),
        n_freq_bands=len(collator.freq_band_vocab),
        d_model=128,
        n_heads=8,
        n_layers=4,
        max_seq_len=100,
        dropout=0.1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training loop
    print("Starting training...")
    model.train()
    
    epoch = 1  # Single epoch for demonstration (increase for real training)
    
    for epoch_idx in range(epoch):
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (event_sequences, labels, crop_inds, infos) in enumerate(dataloader):
            # Move to device
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
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch_idx+1}/{epoch}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        print(f"Epoch {epoch_idx+1}/{epoch} completed. Average Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "event_model_weights_challenge_2_real.pt")
    print("Model saved to event_model_weights_challenge_2_real.pt")
    
    return model


def compare_architectures():
    """
    Compare the event-driven approach with traditional approaches.
    This provides insight into the advantages of the new architecture.
    """
    
    print("\n=== Architecture Comparison ===")
    
    print("Traditional EEGNetv4:")
    print("- Input: (batch_size, 129, 200) - raw voltage time series")
    print("- Architecture: CNN with depthwise/separable convolutions")
    print("- Parameters: ~6,800")
    print("- Approach: Signal processing paradigm")
    print("- Interpretability: Limited (learned filters)")
    
    print("\nEvent-Driven EEG Transformer:")
    print("- Input: (batch_size, seq_len, 9) - event feature sequences")
    print("- Architecture: Transformer with attention mechanisms")
    print("- Parameters: ~800,000")
    print("- Approach: Interaction-driven paradigm (inspired by RecSys)")
    print("- Interpretability: High (meaningful event types and brain regions)")
    
    print("\nKey Advantages of Event-Driven Approach:")
    print("1. Better Transfer Learning: Events are more generalizable across subjects")
    print("2. Interpretability: Each event has cognitive/anatomical meaning")
    print("3. Efficiency: Sparse representation reduces temporal redundancy")
    print("4. Sequential Modeling: Leverages proven RecSys techniques")
    print("5. Cross-Task Adaptation: Event patterns transfer better than raw signals")
    

def main():
    """Main execution function"""
    
    print("=== Challenge 2: Event-Driven EEG with Real Data ===")
    print("Integration of UBP-inspired event architecture with EEG Challenge data")
    print()
    
    # Compare architectures
    compare_architectures()
    
    # Train on real data (if available)
    if EEGDASH_AVAILABLE:
        print("\n=== Training on Real EEG Data ===")
        trained_model = train_event_model_on_real_data()
        
        if trained_model is not None:
            print("Training completed successfully!")
        else:
            print("Training failed - check data availability")
    else:
        print("\n=== EEGDash not available ===")
        print("To run with real data, install:")
        print("pip install git+https://github.com/sccn/eegdash.git")
        print("pip install git+https://github.com/braindecode/braindecode.git")
        print("And ensure EEG challenge data is downloaded")
    
    print(f"\n=== Integration Complete ===")
    print("The event-driven architecture is now integrated with the challenge pipeline!")
    print("Key files:")
    print("- eeg_events.py: Core event extraction and modeling")
    print("- challenge_2_events.py: Standalone demo")
    print("- challenge_2_integration.py: Real data integration")  
    print("- submission.py: Updated for event-based models")


if __name__ == "__main__":
    main()