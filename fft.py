import argparse
import numpy as np
import matplotlib
import math
import cmath

# parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--mode', type=int, default=1)
# parser.add_argument('-i', '--image', type=str, default="./moonlanding.png")

# args = parser.parse_args()

def dft_naive_oneD(arrNums, isInverse):
    exponent_expression = 0
    dft_coefficients = []
    N = len(arrNums)

    for k in range(N):
        kth_coefficient = 0

        for n in range(N):
            if isInverse == True:
                exponent_expression = cmath.exp((1j*2*np.pi*k*n)/N)
            else:
                exponent_expression = cmath.exp((-1j*2*np.pi*k*n)/N)

            kth_coefficient += (arrNums[n]*exponent_expression)

        dft_coefficients.append(kth_coefficient)
    
    return dft_coefficients

# def dft_naive_oneD_inverse(arrNums):

#     original_data = []
#     N = len(arrNums)

#     for k in range(N):
#         kth_original_data = 0

#         for n in range(N):
#             exponent_expression = cmath.exp((1j*2*np.pi*k*n)/N)
#             kth_original_data += (arrNums[n]*exponent_expression)

#         original_data.append(kth_original_data/N)
    
#     return original_data

#compute X_subk
def fast_fourier_transform(arrNums):
    
    N = len(arrNums)

    even_indices = []
    odd_indices = []

    if N <= 16:
        return dft_naive_oneD(arrNums, False)
    
    for i in range(N):
        if i % 2 == 0:
            even_indices.append(arrNums[i])
        else:
            odd_indices.append(arrNums[i])

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

def fast_fourier_transform_inverse(arrNums):
    
    N = len(arrNums)

    even_indices = []
    odd_indices = []

    if N <= 16:
        return dft_naive_oneD(arrNums, True)
    
    for i in range(N):
        if i % 2 == 0:
            even_indices.append(arrNums[i])
        else:
            odd_indices.append(arrNums[i])

    #recursive calls
    fft_even = fast_fourier_transform(even_indices)
    fft_odd = fast_fourier_transform(odd_indices)

    fft_result = [0 for index in range(N)]

    for k in range(N//2):
        fft_even_result = fft_even[k]
        exponent_expression = cmath.exp((1j*2*np.pi*k)/N)
        fft_odd_result = exponent_expression*fft_odd[k]

        fft_result[k] = (fft_even_result + fft_odd_result)/N
        fft_result[k + N//2] = (fft_even_result - fft_odd_result)/N

    return fft_result

def fast_fourier_transform_twoD(twoDArr):
    fft_twoD_rows = []
    fft_twoD_columns = []

    twoDArr_vTwo = np.array(twoDArr)

    for i in range(len(twoDArr_vTwo)):
        rowArr = twoDArr[i]
        resultRow = fast_fourier_transform(rowArr)
        fft_twoD_rows.append(resultRow)

    twoDArr_vTwo_transpose = np.transpose(fft_twoD_rows)

    for i in range(len(twoDArr_vTwo_transpose)):
        colArr = twoDArr_vTwo_transpose[i]
        colRow = fast_fourier_transform(colArr)
        fft_twoD_columns.append(colRow)

    fft_twoD_result = np.transpose(fft_twoD_columns)

    return fft_twoD_result

if __name__ == "__main__":

    arrTest = [0 for i in range(16)]
    for i in range(16):
        arrTest[i] = i

    arrTwoD = []
    for i in range(16):
        arrTwoD.append(arrTest)

    fft_twoD = fast_fourier_transform_twoD(arrTwoD)
    print(fft_twoD)
    
    #ft_naive = dft_naive_oneD(arrTest, False)
    #ft_fast = fast_fourier_transform(arrTest)
    # inverse_naive = dft_naive_oneD_inverse(ft_naive)
    #inverse_fft = fast_fourier_transform_inverse(ft_fast)

    
    # print("Naive \n")

    # print(ft_naive)

    # print("\n\n\nFFT\n")

    # print(ft_fast)

    # print("\n\n\nInverse Naive\n")

    # print(inverse_naive)

    # print("\n\n\nInverse FFT\n")

    # print(inverse_fft)