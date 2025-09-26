#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from spice.resources.rnn import BaseRNN, PolynomialLibraryModule

def test_polynomial_library():
    """Test the polynomial library functionality"""

    # Test parameters
    input_size = 3  # current state + 2 inputs
    embedding_size = 32
    degree = 2
    batch_size = 2
    n_actions = 2

    # Create polynomial library module
    poly_lib = PolynomialLibraryModule(input_size=input_size, embedding_size=embedding_size, degree=degree)

    # Test input tensor: (batch_size, n_actions, input_size + embedding_size)
    test_input = torch.randn(batch_size, n_actions, input_size + embedding_size)

    # Forward pass
    try:
        output = poly_lib(test_input)
        print(f"✓ PolynomialLibraryModule forward pass successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Number of polynomial terms: {poly_lib.n_terms}")
        print(f"  Polynomial terms: {poly_lib.polynomial_terms[:10]}...")  # Show first 10 terms

        # Check output shape
        expected_shape = (batch_size, n_actions, 1)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"  ✓ Output shape correct")

    except Exception as e:
        print(f"✗ PolynomialLibraryModule forward pass failed: {e}")
        return False

    return True

def test_setup_library():
    """Test the BaseRNN setup_library method"""

    # Create a simple RNN class for testing
    class TestRNN(BaseRNN):
        init_values = {'x_test': 0.0}

        def __init__(self, n_actions=2, n_participants=5, embedding_size=32):
            super().__init__(n_actions=n_actions, n_participants=n_participants, embedding_size=embedding_size)

    # Create test RNN
    rnn = TestRNN()

    # Test setup_library
    try:
        input_size = 3
        embedding_size = 32
        degree = 2

        library_module = rnn.setup_library(input_size=input_size, embedding_size=embedding_size, degree=degree)

        print(f"✓ setup_library method successful")
        print(f"  Module type: {type(library_module)}")
        print(f"  Input size: {input_size}")
        print(f"  Embedding size: {embedding_size}")
        print(f"  Degree: {degree}")
        print(f"  Number of terms: {library_module.n_terms}")

        return True

    except Exception as e:
        print(f"✗ setup_library method failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Polynomial Library Functionality")
    print("=" * 50)

    success = True

    print("\n1. Testing PolynomialLibraryModule...")
    success &= test_polynomial_library()

    print("\n2. Testing setup_library method...")
    success &= test_setup_library()

    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")

    sys.exit(0 if success else 1)