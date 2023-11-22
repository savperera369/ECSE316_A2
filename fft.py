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
                exponent_expression = (cmath.exp((1j*2*np.pi*k*n)/N))/N
            else:
                exponent_expression = cmath.exp((-1j*2*np.pi*k*n)/N)

            kth_coefficient += (arrNums[n]*exponent_expression)

        dft_coefficients.append(kth_coefficient)
    
    return dft_coefficients

#compute X_subk
def fast_fourier_transform(arrNums, isInverse):
    exponent_expression = 0
    N = len(arrNums)

    even_indices = []
    odd_indices = []

    if N <= 16:
        returnedData = []
        if isInverse == True:
            returnedData = dft_naive_oneD(arrNums, True)
        else:
            returnedData = dft_naive_oneD(arrNums, False)

        return returnedData
    
    for i in range(N):
        if i % 2 == 0:
            even_indices.append(arrNums[i])
        else:
            odd_indices.append(arrNums[i])

    #recursive calls
    fft_even = fast_fourier_transform(even_indices, isInverse)
    fft_odd = fast_fourier_transform(odd_indices, isInverse)

    fft_result = [0 for index in range(N)]

    for k in range(N//2):
        if isInverse == True:
            exponent_expression = cmath.exp((1j*2*np.pi*k)/N)
            fft_even_result = fft_even[k]
            fft_odd_result = (exponent_expression*fft_odd[k])
            fft_result[k] = (fft_even_result + fft_odd_result)/2
            fft_result[k + N//2] = (fft_even_result - fft_odd_result)/2
        else:
            exponent_expression = cmath.exp((-1j*2*np.pi*k)/N)
            fft_even_result = fft_even[k]
            fft_odd_result = (exponent_expression*fft_odd[k])
            fft_result[k] = fft_even_result + fft_odd_result
            fft_result[k + N//2] = fft_even_result - fft_odd_result

    return fft_result

def fft_inverse(arrNums):
    return fast_fourier_transform(arrNums, True)

def fft_twoD(twoDArr, isInverse):
    fft_twoD_rows = []
    fft_twoD_columns = []

    twoDArr_vTwo = np.array(twoDArr)

    for i in range(len(twoDArr_vTwo)):
        rowArr = twoDArr[i]
        if isInverse == True:
            resultRow = fast_fourier_transform(rowArr, True)
            fft_twoD_rows.append(resultRow)
        else:
            resultRow = fft_inverse(rowArr)
            fft_twoD_rows.append(resultRow)

    twoDArr_vTwo_transpose = np.transpose(fft_twoD_rows)

    for i in range(len(twoDArr_vTwo_transpose)):
        colArr = twoDArr_vTwo_transpose[i]
        if isInverse == True:
            colRow = fast_fourier_transform(colArr, True)
            fft_twoD_columns.append(colRow)
        else:
            colRow = fft_inverse(colArr)
            fft_twoD_columns.append(colRow)

    fft_twoD_result = np.transpose(fft_twoD_columns)

    return fft_twoD_result

def fft_twoD_inverse(twoDArr):
    return fft_twoD(twoDArr, True)

if __name__ == "__main__":

    arrTest = [0 for i in range(32)]
    for i in range(32):
        arrTest[i] = i

    # arrTwoD = []
    # for i in range(32):
    #     arrTwoD.append(arrTest)

    # testfft_twoD = fft_twoD(arrTwoD, False)
    # testfft_twoD_inverse = fft_twoD_inverse(testfft_twoD)
    # print(testfft_twoD_inverse)

    ft_naive = dft_naive_oneD(arrTest, False)
    ft_fast = fast_fourier_transform(arrTest, False)
    inverse_fft = fft_inverse(ft_fast)
    
    print("Naive \n")

    print(ft_naive)

    print("\n\n\nFFT\n")

    print(ft_fast)

    print("\n\n\nInverse FFT\n")

    print(inverse_fft)