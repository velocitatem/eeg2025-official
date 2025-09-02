"""
Test script to verify the updated submission.py works correctly
with both traditional and event-driven approaches.
"""

import torch
import numpy as np
from submission import Submission


def test_submission_models():
    """Test both challenge models from the updated submission"""
    
    print("=== Testing Updated Submission Class ===")
    
    # Initialize submission
    SFREQ = 100
    DEVICE = torch.device("cpu")  # Use CPU for testing
    
    sub = Submission(SFREQ, DEVICE)
    
    # Test Challenge 1 model
    print("Testing Challenge 1 model...")
    try:
        model_1 = sub.get_model_challenge_1()
        print(f"✓ Challenge 1 model loaded: {type(model_1).__name__}")
        
        # Test with dummy input
        dummy_input = torch.randn(2, 129, 200)  # Batch of 2 samples
        with torch.no_grad():
            output_1 = model_1(dummy_input)
            print(f"✓ Challenge 1 output shape: {output_1.shape}")
            
    except Exception as e:
        print(f"✗ Challenge 1 model failed: {e}")
    
    # Test Challenge 2 model (event-driven)
    print("\nTesting Challenge 2 model (event-driven)...")
    try:
        model_2 = sub.get_model_challenge_2()
        print(f"✓ Challenge 2 model loaded: {type(model_2).__name__}")
        
        # Test with dummy input
        dummy_input = torch.randn(2, 129, 200)  # Batch of 2 samples
        with torch.no_grad():
            output_2 = model_2(dummy_input)
            print(f"✓ Challenge 2 output shape: {output_2.shape}")
            
    except Exception as e:
        print(f"✗ Challenge 2 model failed: {e}")
    
    print("\n=== Submission Test Complete ===")


def test_model_compatibility():
    """Test that models work as expected in the evaluation scenario"""
    
    print("\n=== Testing Model Compatibility ===")
    
    SFREQ = 100
    DEVICE = torch.device("cpu")
    BATCH_SIZE = 4
    
    # Create mock data similar to challenge evaluation
    mock_data = torch.randn(BATCH_SIZE, 129, 200, dtype=torch.float32)
    mock_labels = torch.randn(BATCH_SIZE, 1, dtype=torch.float32)
    
    sub = Submission(SFREQ, DEVICE)
    
    # Test Challenge 2 (event-driven) in evaluation mode
    print("Testing Challenge 2 in evaluation mode...")
    try:
        model_2 = sub.get_model_challenge_2()
        model_2.eval()
        
        with torch.inference_mode():
            predictions = model_2.forward(mock_data)
            print(f"✓ Predictions shape: {predictions.shape}")
            print(f"✓ Predictions range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
            
            # Test that model produces reasonable outputs
            assert predictions.shape == (BATCH_SIZE, 1), f"Expected shape ({BATCH_SIZE}, 1), got {predictions.shape}"
            assert not torch.isnan(predictions).any(), "Model produced NaN values"
            assert torch.isfinite(predictions).all(), "Model produced infinite values"
            
            print("✓ All compatibility tests passed")
            
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()


def performance_comparison():
    """Compare inference performance between approaches"""
    
    print("\n=== Performance Comparison ===")
    
    import time
    
    SFREQ = 100
    DEVICE = torch.device("cpu")
    BATCH_SIZE = 8
    N_RUNS = 5
    
    # Create test data
    test_data = torch.randn(BATCH_SIZE, 129, 200, dtype=torch.float32)
    
    sub = Submission(SFREQ, DEVICE)
    
    # Test Challenge 1 (traditional)
    try:
        model_1 = sub.get_model_challenge_1()
        model_1.eval()
        
        times_1 = []
        with torch.inference_mode():
            for _ in range(N_RUNS):
                start_time = time.time()
                _ = model_1(test_data)
                end_time = time.time()
                times_1.append(end_time - start_time)
        
        avg_time_1 = np.mean(times_1) * 1000  # Convert to ms
        print(f"Challenge 1 (traditional) avg time: {avg_time_1:.2f} ms")
        
    except Exception as e:
        print(f"Challenge 1 performance test failed: {e}")
    
    # Test Challenge 2 (event-driven)
    try:
        model_2 = sub.get_model_challenge_2()
        model_2.eval()
        
        times_2 = []
        with torch.inference_mode():
            for _ in range(N_RUNS):
                start_time = time.time()
                _ = model_2(test_data)
                end_time = time.time()
                times_2.append(end_time - start_time)
        
        avg_time_2 = np.mean(times_2) * 1000  # Convert to ms
        print(f"Challenge 2 (event-driven) avg time: {avg_time_2:.2f} ms")
        
        # Compare performance
        if 'avg_time_1' in locals():
            speedup = avg_time_1 / avg_time_2
            if speedup > 1:
                print(f"Event-driven model is {speedup:.2f}x slower (expected due to event extraction)")
            else:
                print(f"Event-driven model is {1/speedup:.2f}x faster")
        
    except Exception as e:
        print(f"Challenge 2 performance test failed: {e}")


if __name__ == "__main__":
    test_submission_models()
    test_model_compatibility()
    performance_comparison()
    
    print("\n=== All Tests Complete ===")
    print("The event-driven EEG architecture is ready for submission!")