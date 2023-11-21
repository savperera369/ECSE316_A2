import argparse
import numpy as np
import matplotlib
import math
import cmath

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--mode', type=int, default=1)
# parser.add_argument('-i', '--image', type=str, default="./moonlanding.png")

# args = parser.parse_args()

def dft_naive_oneD(arrNums):

    dft_coefficients = []
    arr = np.array(arrNums)

    for k in range(len(arr)):
        kth_coefficient = 0

        for n in range(len(arr)):
            exponent_expression = cmath.exp((-1j*2*np.pi*k*n)/len(arr))
            kth_coefficient += (arr[n]*exponent_expression)

        dft_coefficients.append(kth_coefficient)
    
    return dft_coefficients

def dft_naive_oneD_inverse(arrNums):

    original_data = []
    arr = np.array(arrNums)

    for k in range(len(arr)):
        kth_original_data = 0

        for n in range(len(arr)):
            exponent_expression = cmath.exp((1j*2*np.pi*k*n)/len(arr))
            kth_original_data += (arr[n]*exponent_expression)

        kth_original_data = kth_original_data/len(arr)

        original_data.append(kth_original_data)

    return original_data

#compute X_subk
def fast_fourier_transform(arrNums):
    
    N = len(arrNums)
    arr = np.array(arrNums)

    even_indices = []
    odd_indices = []

    if N <= 16:
        return dft_naive_oneD(arr)
    
    for i in range(N):
        if i % 2 == 0:
            even_indices.append(arr[i])
        else:
            odd_indices.append(arr[i])

    #recursive calls
    fft_even = fast_fourier_transform(even_indices)
    fft_odd = fast_fourier_transform(odd_indices)

    fft_result = [0 for index in range(N)]

    for k in range(N//2):
        fft_even_result = fft_even[k]
        exponent_expression = cmath.exp((-1j*2*np.pi*k)/N)
        fft_odd_result = exponent_expression*fft_odd[k]

        fft_result[k] = fft_even_result + fft_odd_result
        fft_result[k + N//2] = fft_even_result - fft_odd_result

    return fft_result

if __name__ == "__main__":

    arrTest = [0 for i in range(64)]
    for i in range(64):
        arrTest[i] = i

    ft_naive = dft_naive_oneD(arrTest)
    ft_fast = fast_fourier_transform(arrTest)
    
    print(ft_naive)

    print("\n\n\nNaive\n")

    print(ft_fast)