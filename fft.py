import argparse
import matplotlib
from fourier_transform import *

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=int, default=1)
parser.add_argument('-i', '--image', type=str, default="./moonlanding.png")

args = parser.parse_args()

if __name__ == "__main__":

    arrTest = [0 for i in range(16)]
    for i in range(16):
        arrTest[i] = i

    arrTwoD = []
    for i in range(2):
        arrTwoD.append(arrTest)

    print(arrTwoD)
    print("\n\n")
    testfft_twoD = fft_twoD(arrTwoD)
    print(testfft_twoD)
    print("\n\n")
    testfft_twoD_inverse = fft_twoD_inverse(testfft_twoD)
    print(testfft_twoD_inverse)

    # ft_naive = dft_naive_oneD(arrTest, False)
    # ft_fast = fast_fourier_transform(arrTest, False)
    # inverse_fft = fft_inverse(ft_fast)
    
    # print("Naive \n")

    # print(ft_naive)

    # print("\n\n\nFFT\n")

    # print(ft_fast)

    # print("\n\n\nInverse FFT\n")

    # print(inverse_fft)