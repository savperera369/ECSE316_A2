import numpy as np
import cmath

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

def dft_naive_twoD(twoDArr):
    dft_twoD_rows = []
    dft_twoD_columns = []

    twoDArr_vTwo = np.array(twoDArr)

    for row in twoDArr_vTwo:
        rowArr = dft_naive_oneD(row.tolist(), False)
        dft_twoD_rows.append(rowArr)

    outer_sum = np.transpose(np.array(dft_twoD_rows))

    for col in outer_sum:
        colArr = dft_naive_oneD(col.tolist(), False)
        dft_twoD_columns.append(colArr)

    dft_final = (np.transpose(np.array(dft_twoD_columns))).tolist()

    return dft_final

#compute X_subk
def fast_fourier_transform(arrNums, isInverse):
    exponent_expression = 0
    N = len(arrNums)

    even_indices = []
    odd_indices = []

    if N <= 8:
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

def fft_twoD(twoDArr):
    fft_twoD_rows = []
    fft_twoD_columns = []

    twoDArr_vTwo = np.array(twoDArr)

    for row in twoDArr_vTwo:
        rowArr = fast_fourier_transform(row.tolist(), False)
        fft_twoD_rows.append(rowArr)

    outer_sum = np.transpose(np.array(fft_twoD_rows))

    for col in outer_sum:
        colArr = fast_fourier_transform(col.tolist(), False)
        fft_twoD_columns.append(colArr)

    fft_final = (np.transpose(np.array(fft_twoD_columns))).tolist()

    return fft_final

def fft_twoD_inverse(twoDArr):
    
    original_rows = []
    original_cols = []

    inner_arr = np.array(twoDArr)

    for row in inner_arr:
        rowArr = fft_inverse(row.tolist())
        original_rows.append(rowArr)

    outer_arr = np.transpose(np.array(original_rows))

    for col in outer_arr:
        colArr = fft_inverse(col.tolist())
        original_cols.append(colArr)

    original_arr = (np.transpose(np.array(original_cols))).tolist()

    return original_arr