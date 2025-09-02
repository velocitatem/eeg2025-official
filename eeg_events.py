"""
Event-Driven EEG Architecture
============================

This module implements an event-driven approach to EEG processing, inspired by 
Universal Behavior Profile (UBP) modeling from RecSys. Instead of treating EEG 
as continuous time series, we convert each scan into a sequence of meaningful 
neural interaction events.

Key concepts:
- Brain regions as entities that interact with stimuli/tasks
- Events represent discrete neural interactions rather than continuous signals
- Sequential patterns similar to user-item interactions in RecSys
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd
from scipy import signal
from scipy.stats import zscore


@dataclass
class EEGEvent:
    """Represents a single neural interaction event"""
    timestamp: float
    brain_region: str  # Electrode cluster or anatomical region
    event_type: str  # Type of neural event (e.g., 'spike', 'oscillation', 'phase_coupling')
    highest_delta_channel: int
    lowest_delta_channel: int 
    average_change_from_last: float
    amplitude: float
    frequency_band: str  # 'delta', 'theta', 'alpha', 'beta', 'gamma'
    cross_channel_coherence: float
    signal_to_noise_ratio: float
    context: Dict  # Task, attention state, etc.


class EEGEventExtractor:
    """
    Converts raw EEG data (N×128) into sequences of neural interaction events.
    
    This is the core component that transforms traditional signal processing
    into an event-driven architecture.
    """
    
    def __init__(self, 
                 sfreq: int = 100,
                 window_size: float = 0.5,  # seconds
                 overlap: float = 0.25,  # seconds overlap
                 similarity_threshold: float = 0.8):
        """
        Args:
            sfreq: Sampling frequency
            window_size: Size of analysis window in seconds
            overlap: Overlap between consecutive windows in seconds  
            similarity_threshold: Threshold for pruning similar consecutive events
        """
        self.sfreq = sfreq
        self.window_size = window_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.window_samples = int(window_size * sfreq)
        self.hop_samples = int((window_size - overlap) * sfreq)
        
        # Define anatomical brain regions (simplified mapping)
        self.brain_regions = {
            'frontal': list(range(0, 32)),      # Frontal electrodes
            'parietal': list(range(32, 64)),    # Parietal electrodes  
            'temporal': list(range(64, 96)),    # Temporal electrodes
            'occipital': list(range(96, 128)),  # Occipital electrodes
        }
        
        # Frequency bands for analysis
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def extract_events(self, 
                      eeg_data: np.ndarray, 
                      context: Dict = None) -> List[EEGEvent]:
        """
        Extract event sequence from raw EEG data.
        
        Args:
            eeg_data: Shape (n_channels, n_timepoints), raw EEG data
            context: Additional context information (task, subject info, etc.)
            
        Returns:
            List of EEGEvent objects representing neural interactions
        """
        if context is None:
            context = {}
            
        events = []
        n_channels, n_timepoints = eeg_data.shape
        
        # Handle case where we have 129 channels (including reference)
        if n_channels == 129:
            eeg_data = eeg_data[:128]  # Remove reference channel
            n_channels = 128
            
        # Create sliding windows
        for start_idx in range(0, n_timepoints - self.window_samples, self.hop_samples):
            end_idx = start_idx + self.window_samples
            window_data = eeg_data[:, start_idx:end_idx]
            timestamp = start_idx / self.sfreq
            
            # Extract events from this window
            window_events = self._extract_window_events(window_data, timestamp, context)
            events.extend(window_events)
        
        # Prune similar consecutive events
        pruned_events = self._prune_similar_events(events)
        
        return pruned_events
    
    def _extract_window_events(self, 
                              window_data: np.ndarray, 
                              timestamp: float,
                              context: Dict) -> List[EEGEvent]:
        """Extract events from a single time window"""
        events = []
        
        # Analyze each brain region
        for region_name, channels in self.brain_regions.items():
            region_data = window_data[channels]
            
            # Calculate power in different frequency bands
            band_powers = {}
            for band_name, (low_freq, high_freq) in self.freq_bands.items():
                # Use Welch's method for power spectral density
                freqs, psd = signal.welch(region_data, fs=self.sfreq, nperseg=min(64, region_data.shape[1]))
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.mean(psd[:, band_mask], axis=1)
                band_powers[band_name] = np.mean(band_power)
            
            # Find dominant frequency band
            dominant_band = max(band_powers.keys(), key=lambda k: band_powers[k])
            
            # Calculate channel-specific metrics
            channel_means = np.mean(region_data, axis=1)
            highest_delta_channel = channels[np.argmax(np.abs(channel_means))]
            lowest_delta_channel = channels[np.argmin(np.abs(channel_means))]
            
            # Calculate average change from previous window (if available)
            average_change = np.mean(np.std(region_data, axis=1))
            
            # Calculate cross-channel coherence (simplified)
            coherence_values = []
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    coh = np.corrcoef(region_data[i], region_data[j])[0, 1]
                    if not np.isnan(coh):
                        coherence_values.append(abs(coh))
            cross_channel_coherence = np.mean(coherence_values) if coherence_values else 0.0
            
            # Calculate signal-to-noise ratio
            signal_power = np.mean(np.var(region_data, axis=1))
            noise_estimate = np.mean(np.var(np.diff(region_data, axis=1), axis=1))
            snr = signal_power / (noise_estimate + 1e-8)
            
            # Determine event type based on characteristics
            if band_powers['gamma'] > np.mean(list(band_powers.values())) * 1.5:
                event_type = 'gamma_burst'
            elif cross_channel_coherence > 0.7:
                event_type = 'synchronized_oscillation'  
            elif snr > 10:
                event_type = 'high_amplitude_event'
            else:
                event_type = 'baseline_activity'
            
            # Create event
            event = EEGEvent(
                timestamp=timestamp,
                brain_region=region_name,
                event_type=event_type,
                highest_delta_channel=highest_delta_channel,
                lowest_delta_channel=lowest_delta_channel,
                average_change_from_last=average_change,
                amplitude=signal_power,
                frequency_band=dominant_band,
                cross_channel_coherence=cross_channel_coherence,
                signal_to_noise_ratio=snr,
                context=context
            )
            
            events.append(event)
        
        return events
    
    def _prune_similar_events(self, events: List[EEGEvent]) -> List[EEGEvent]:
        """Remove events that are too similar to their neighbors"""
        if len(events) <= 1:
            return events
            
        pruned_events = [events[0]]  # Always keep first event
        
        for i in range(1, len(events)):
            current_event = events[i]
            previous_event = pruned_events[-1]
            
            # Calculate similarity based on multiple features
            similarity = self._calculate_event_similarity(current_event, previous_event)
            
            # Only add event if it's sufficiently different
            if similarity < self.similarity_threshold:
                pruned_events.append(current_event)
        
        return pruned_events
    
    def _calculate_event_similarity(self, event1: EEGEvent, event2: EEGEvent) -> float:
        """Calculate similarity between two events (0-1 scale)"""
        # Binary features
        binary_similarity = 0.0
        if event1.brain_region == event2.brain_region:
            binary_similarity += 0.25
        if event1.event_type == event2.event_type:
            binary_similarity += 0.25  
        if event1.frequency_band == event2.frequency_band:
            binary_similarity += 0.25
            
        # Continuous features (normalized)
        amplitude_sim = 1.0 - abs(event1.amplitude - event2.amplitude) / (event1.amplitude + event2.amplitude + 1e-8)
        coherence_sim = 1.0 - abs(event1.cross_channel_coherence - event2.cross_channel_coherence)
        change_sim = 1.0 - abs(event1.average_change_from_last - event2.average_change_from_last) / (abs(event1.average_change_from_last) + abs(event2.average_change_from_last) + 1e-8)
        
        continuous_similarity = (amplitude_sim + coherence_sim + change_sim) / 3 * 0.25
        
        return binary_similarity + continuous_similarity


class EEGEventSequenceDataset:
    """
    PyTorch dataset wrapper for event sequences.
    
    Converts event sequences into tensor format suitable for sequential models
    like those used in RecSys (e.g., SASRec, GRU4Rec).
    """
    
    def __init__(self, 
                 event_sequences: List[List[EEGEvent]],
                 labels: List[float],
                 max_sequence_length: int = 100):
        """
        Args:
            event_sequences: List of event sequences, one per EEG recording
            labels: Target labels (e.g., p-factor values)
            max_sequence_length: Maximum length for padding/truncation
        """
        self.event_sequences = event_sequences
        self.labels = labels
        self.max_sequence_length = max_sequence_length
        
        # Create vocabularies for categorical features
        self._build_vocabularies()
        
    def _build_vocabularies(self):
        """Build vocabularies for categorical features"""
        all_regions = set()
        all_event_types = set()
        all_freq_bands = set()
        
        for sequence in self.event_sequences:
            for event in sequence:
                all_regions.add(event.brain_region)
                all_event_types.add(event.event_type)
                all_freq_bands.add(event.frequency_band)
        
        self.region_vocab = {region: idx for idx, region in enumerate(sorted(all_regions))}
        self.event_type_vocab = {event_type: idx for idx, event_type in enumerate(sorted(all_event_types))}
        self.freq_band_vocab = {band: idx for idx, band in enumerate(sorted(all_freq_bands))}
    
    def __len__(self):
        return len(self.event_sequences)
    
    def __getitem__(self, idx):
        """Convert event sequence to tensor format"""
        events = self.event_sequences[idx]
        label = self.labels[idx]
        
        # Truncate or pad sequence
        if len(events) > self.max_sequence_length:
            events = events[:self.max_sequence_length]
        
        # Convert events to tensor format
        sequence_tensor = self._events_to_tensor(events)
        
        return sequence_tensor, torch.tensor(label, dtype=torch.float32)
    
    def _events_to_tensor(self, events: List[EEGEvent]) -> torch.Tensor:
        """Convert event sequence to tensor representation"""
        # Feature vector per event: [region_id, event_type_id, freq_band_id, 
        #                           amplitude, coherence, change, snr, highest_ch, lowest_ch]
        feature_dim = 9
        sequence_length = len(events)
        
        # Initialize tensor
        sequence_tensor = torch.zeros(self.max_sequence_length, feature_dim)
        
        for i, event in enumerate(events):
            if i >= self.max_sequence_length:
                break
                
            sequence_tensor[i] = torch.tensor([
                self.region_vocab[event.brain_region],
                self.event_type_vocab[event.event_type], 
                self.freq_band_vocab[event.frequency_band],
                event.amplitude,
                event.cross_channel_coherence,
                event.average_change_from_last,
                event.signal_to_noise_ratio,
                event.highest_delta_channel,
                event.lowest_delta_channel
            ])
        
        return sequence_tensor


class EEGEventTransformer(nn.Module):
    """
    Transformer model for processing EEG event sequences.
    
    Inspired by sequential recommendation models but adapted for neural events.
    Similar to SASRec but with EEG-specific features and objectives.
    """
    
    def __init__(self,
                 n_regions: int = 4,
                 n_event_types: int = 10, 
                 n_freq_bands: int = 5,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 max_seq_len: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layers for categorical features
        self.region_embedding = nn.Embedding(n_regions, d_model // 4)
        self.event_type_embedding = nn.Embedding(n_event_types, d_model // 4)
        self.freq_band_embedding = nn.Embedding(n_freq_bands, d_model // 4)
        
        # Linear projection for continuous features
        self.continuous_projection = nn.Linear(6, d_model // 4)  # 6 continuous features
        
        # Positional encoding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, 1)  # For regression
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, event_sequences):
        """
        Args:
            event_sequences: Tensor of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Predictions: Tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = event_sequences.shape
        
        # Extract features
        regions = event_sequences[:, :, 0].long()  # Brain regions
        event_types = event_sequences[:, :, 1].long()  # Event types
        freq_bands = event_sequences[:, :, 2].long()  # Frequency bands
        continuous_features = event_sequences[:, :, 3:9]  # Continuous features
        
        # Create embeddings
        region_emb = self.region_embedding(regions)
        event_type_emb = self.event_type_embedding(event_types)
        freq_band_emb = self.freq_band_embedding(freq_bands)
        continuous_emb = self.continuous_projection(continuous_features)
        
        # Combine embeddings
        combined_emb = torch.cat([region_emb, event_type_emb, freq_band_emb, continuous_emb], dim=-1)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=event_sequences.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Final input embeddings
        input_emb = combined_emb + pos_emb
        input_emb = self.dropout(input_emb)
        
        # Create attention mask (mask padding tokens)
        mask = (regions == 0) & (event_types == 0)  # Assume 0 indicates padding
        
        # Apply transformer
        transformer_output = self.transformer(input_emb, src_key_padding_mask=mask)
        
        # Global average pooling over sequence dimension (ignoring padding)
        mask_expanded = mask.unsqueeze(-1).expand_as(transformer_output)
        transformer_output = transformer_output.masked_fill(mask_expanded, 0)
        pooled_output = transformer_output.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float()
        
        # Apply layer norm and get final prediction
        pooled_output = self.layer_norm(pooled_output)
        prediction = self.output_projection(pooled_output)
        
        return prediction