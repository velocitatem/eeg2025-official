# ##########################################################################
# # Example of submission files
# # ---------------------------
# my_submission.zip
# ├─ submission.py
# ├─ weights_challenge_1.pt
# ├─ weights_challenge_2.pt
# ├─ event_model_weights_challenge_2.pt
# └─ eeg_events.py

try:
    from braindecode.models import EEGNetv4
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False
    
import torch
import torch.nn as nn
import numpy as np
from eeg_events import EEGEventExtractor, EEGEventTransformer


class EventBasedEEGWrapper(nn.Module):
    """
    Wrapper that converts raw EEG input to event sequences and processes them
    with the event-driven transformer model. This maintains compatibility with
    the existing evaluation pipeline while using the new event architecture.
    """
    
    def __init__(self, event_model, event_extractor, device):
        super().__init__()
        self.event_model = event_model
        self.event_extractor = event_extractor
        self.device = device
        
    def forward(self, X):
        """
        Args:
            X: Raw EEG tensor of shape (batch_size, n_channels, n_timepoints)
            
        Returns:
            Predictions: Tensor of shape (batch_size, 1)
        """
        batch_size = X.shape[0]
        predictions = []
        
        for i in range(batch_size):
            # Convert single sample to event sequence
            eeg_sample = X[i].cpu().numpy()  # Shape: (n_channels, n_timepoints)
            
            # Handle 129 channels case
            if eeg_sample.shape[0] == 129:
                eeg_sample = eeg_sample[:128]
            
            # Extract events
            events = self.event_extractor.extract_events(eeg_sample, context={})
            
            # Convert to tensor format (simplified for inference)
            if len(events) == 0:
                # If no events extracted, return default prediction
                predictions.append(torch.tensor([0.0], device=self.device))
                continue
                
            # Create simplified event tensor for inference
            max_events = 100
            event_tensor = torch.zeros(max_events, 9, device=self.device)
            
            for j, event in enumerate(events[:max_events]):
                # Simplified encoding for inference
                region_id = {'frontal': 0, 'parietal': 1, 'temporal': 2, 'occipital': 3}.get(event.brain_region, 0)
                event_type_id = 0  # Simplified
                freq_band_id = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3, 'gamma': 4}.get(event.frequency_band, 2)
                
                event_tensor[j] = torch.tensor([
                    region_id,
                    event_type_id,
                    freq_band_id,
                    event.amplitude,
                    event.cross_channel_coherence,
                    event.average_change_from_last,
                    event.signal_to_noise_ratio,
                    event.highest_delta_channel,
                    event.lowest_delta_channel
                ], device=self.device)
            
            # Get prediction from event model
            event_input = event_tensor.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                pred = self.event_model(event_input)
                predictions.append(pred.squeeze(0))
        
        return torch.stack(predictions)


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        if BRAINDECODE_AVAILABLE:
            model_challenge1 = EEGNetv4(
                in_chans=129, n_classes=1, input_window_samples=int(2 * self.sfreq)
            ).to(self.device)
            # checkpoint_1 = torch.load("weights_challenge_1.pt", map_location=self.device)
            # model_challenge1.load_state_dict(...)
            return model_challenge1
        else:
            # Fallback: simple linear model
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(129 * 200, 1)
                    
                def forward(self, x):
                    return self.linear(x.view(x.size(0), -1))
            
            return SimpleModel().to(self.device)

    def get_model_challenge_2(self):
        """
        Returns the event-driven EEG model for Challenge 2.
        This uses the new event-based architecture instead of traditional EEGNetv4.
        """
        try:
            # Initialize event extractor
            event_extractor = EEGEventExtractor(
                sfreq=self.sfreq,
                window_size=0.5,
                overlap=0.25,
                similarity_threshold=0.8
            )
            
            # Initialize event transformer model
            event_model = EEGEventTransformer(
                n_regions=4,
                n_event_types=1,  # Will be expanded as more event types are detected
                n_freq_bands=5,
                d_model=128,
                n_heads=8,
                n_layers=4,
                max_seq_len=100,
                dropout=0.1
            ).to(self.device)
            
            # Load trained weights if available
            try:
                checkpoint_2 = torch.load("event_model_weights_challenge_2.pt", map_location=self.device)
                event_model.load_state_dict(checkpoint_2)
                print("Loaded event-based model weights for Challenge 2")
            except FileNotFoundError:
                print("Event model weights not found, using randomly initialized model")
            
            # Wrap in compatibility layer
            wrapped_model = EventBasedEEGWrapper(event_model, event_extractor, self.device)
            return wrapped_model
            
        except Exception as e:
            print(f"Failed to load event-based model, falling back to traditional approach: {e}")
            
            # Fallback to traditional approach if event model fails
            if BRAINDECODE_AVAILABLE:
                model_challenge2 = EEGNetv4(
                    in_chans=129, n_classes=1, input_window_samples=int(2 * self.sfreq)
                ).to(self.device)
                try:
                    checkpoint_2 = torch.load("weights_challenge_2.pt", map_location=self.device)
                    model_challenge2.load_state_dict(checkpoint_2)
                except FileNotFoundError:
                    pass
                return model_challenge2
            else:
                # Simple fallback model
                class SimpleModel(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.linear = nn.Linear(129 * 200, 1)
                        
                    def forward(self, x):
                        return self.linear(x.view(x.size(0), -1))
                
                return SimpleModel().to(self.device)


# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission
#
# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()

# warmup_loader_challenge_1 = DataLoader(HBN_R5_dataset1, batch_size=BATCH_SIZE)
# final_loader_challenge_1 = DataLoader(secret_dataset1, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:  # and final_loader later
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_1.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score1 = compute_score_challenge_1(y_true, y_preds)
# del model_1
# gc.collect()

# model_2 = sub.get_model_challenge_2()
# model_2.eval()

# warmup_loader_challenge_2 = DataLoader(HBN_R5_dataset2, batch_size=BATCH_SIZE)
# final_loader_challenge_2 = DataLoader(secret_dataset2, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:  # and final_loader later
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_2.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score2 = compute_score_challenge_2(y_true, y_preds)
# overall_score = compute_leaderboard_score(score1, score2)
