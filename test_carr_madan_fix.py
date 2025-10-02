#!/usr/bin/env python3
"""
Test script to verify that the Carr-Madan fix eliminates zeros in integrated variance.
Run this script to check if the fix is working properly.
"""

from utils import GeneratePathsHestonES
import numpy as np
import matplotlib.pyplot as plt

def test_carr_madan_fix():
    """Test the Carr-Madan fix with various parameter sets."""
    
    print("Testing Carr-Madan fix for integrated variance zeros...")
    print("=" * 60)
    
    # Test case 1: Original notebook parameters
    print("\nTest 1: Original notebook parameters")
    np.random.seed(3)
    paths1 = GeneratePathsHestonES(
        NoOfPaths=4, NoOfSteps=50, T=1.0, r=0.1, S_0=100.0,
        kappa=0.5, gamma=0.4, rho=-0.9, vbar=0.2, v0=0.2,
        nr_expansion=100, L=10, recovery_method='carr_madan', N=2**12
    )
    
    V_int1 = paths1['Vint']
    min_val1 = np.min(V_int1)
    zero_count1 = np.sum(V_int1 <= 1e-10)
    
    print(f"Min integrated variance: {min_val1:.10f}")
    print(f"Values <= 1e-10: {zero_count1}/{V_int1.size}")
    print(f"All positive: {np.all(V_int1 > 0)}")
    
    # Test case 2: Quick test parameters
    print("\nTest 2: Quick test parameters")
    np.random.seed(42)
    paths2 = GeneratePathsHestonES(
        NoOfPaths=3, NoOfSteps=10, T=0.5, r=0.05, S_0=100,
        kappa=0.5, gamma=0.4, rho=-0.7, vbar=0.2, v0=0.2,
        nr_expansion=50, L=8, recovery_method='carr_madan', N=2048
    )
    
    V_int2 = paths2['Vint']
    min_val2 = np.min(V_int2)
    zero_count2 = np.sum(V_int2 <= 1e-10)
    
    print(f"Min integrated variance: {min_val2:.10f}")
    print(f"Values <= 1e-10: {zero_count2}/{V_int2.size}")
    print(f"All positive: {np.all(V_int2 > 0)}")
    
    # Test case 3: Stress test with more paths
    print("\nTest 3: Stress test (more paths)")
    np.random.seed(123)
    paths3 = GeneratePathsHestonES(
        NoOfPaths=10, NoOfSteps=25, T=1.0, r=0.05, S_0=100,
        kappa=1.0, gamma=0.6, rho=-0.8, vbar=0.15, v0=0.25,
        nr_expansion=80, L=12, recovery_method='carr_madan', N=4096
    )
    
    V_int3 = paths3['Vint']
    min_val3 = np.min(V_int3)
    zero_count3 = np.sum(V_int3 <= 1e-10)
    
    print(f"Min integrated variance: {min_val3:.10f}")
    print(f"Values <= 1e-10: {zero_count3}/{V_int3.size}")
    print(f"All positive: {np.all(V_int3 > 0)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    total_zeros = zero_count1 + zero_count2 + zero_count3
    total_values = V_int1.size + V_int2.size + V_int3.size
    
    print(f"Total zero values across all tests: {total_zeros}/{total_values}")
    
    if total_zeros == 0:
        print("✅ SUCCESS: No zeros found! The fix is working correctly.")
    else:
        print("❌ FAILURE: Zeros still present. Need further investigation.")
        
        # Debug info for failures
        print("\nDEBUG INFO:")
        if zero_count1 > 0:
            zero_indices1 = np.where(V_int1 <= 1e-10)
            print(f"Test 1 zeros at indices: {zero_indices1}")
        if zero_count2 > 0:
            zero_indices2 = np.where(V_int2 <= 1e-10)
            print(f"Test 2 zeros at indices: {zero_indices2}")
        if zero_count3 > 0:
            zero_indices3 = np.where(V_int3 <= 1e-10)
            print(f"Test 3 zeros at indices: {zero_indices3}")
    
    return total_zeros == 0

if __name__ == "__main__":
    success = test_carr_madan_fix()
    exit(0 if success else 1)
