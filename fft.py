import argparse
import cv2
import numpy as np
from fourier_transform import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

def padImage(imageStr):
    image = cv2.imread(imageStr)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgList = gray_image.tolist()
    numRows = len(imgList)
    numCols = len(imgList[0])

    n=0
    while 2**n <= numCols:
        n = n + 1

    paddingPixelsCol = (2**n) - numCols
    
    modImage = []

    for rowOneD in imgList:
        for i in range(paddingPixelsCol):
            rowOneD.append(0)
        
        modImage.append(rowOneD)

    m = 0
    while 2**m <= numRows:
        m=m+1

    paddingPixelsRow = (2**m) - numRows

    for i in range(paddingPixelsRow):
        padList = [0] * 1024
        modImage.append(padList)
    
    modFftImage = fft_twoD(modImage)
    built_in_fft = np.fft.fft2(np.array(modImage))
    builtInFft = built_in_fft.tolist()

    return modFftImage, builtInFft, gray_image, paddingPixelsRow, paddingPixelsCol

def unpadImage(paddedImage, paddingPixelsRow, paddingPixelsCol):
    # Use array slicing to remove the padded region
    for col in paddedImage:
        for i in range(paddingPixelsCol):
            col.pop(-1)

    for i in range(paddingPixelsRow):
        paddedImage.pop(-1)

    return np.array(paddedImage)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=int, default=1)
    parser.add_argument('-i', '--image', type=str, default="moonlanding.png")

    args = parser.parse_args()

    if args.mode == 1:
        
        ftImage, builtInFtImage, imgDisplay, padPixelRow, padPixelsCol = padImage(args.image)

        fftImage = np.array(ftImage)
        fftImageBuiltIn = np.array(builtInFtImage)

        absFftImage = np.abs(fftImage)
        absBuiltInFft = np.abs(fftImageBuiltIn)

        print(np.allclose(absFftImage, absBuiltInFft, rtol=1e-5, atol=1e-8))
        # Create a 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image without LogNorm
        im1 = axes[0].imshow(imgDisplay, cmap='gray')
        axes[0].set_title('Original Image')

        
        im2 = axes[1].imshow(absFftImage, cmap='viridis', norm=LogNorm(), alpha=0.3)  # Use alpha for overlapping channels
        axes[1].set_title('Fourier Transform')

        # Add colorbars
        cb1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        cb2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to prevent clipping of colorbars
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    elif args.mode == 2:

        ftImage, builtInFtImage, imgDisplay, padPixelsRow, padPixelsCol = padImage(args.image)

        keep_fraction = 0.1
        mod_fftImage = np.array(ftImage).copy()
        height,width = mod_fftImage.shape
        mod_fftImage[int(height*keep_fraction):int(height*(1-keep_fraction))] = 0
        mod_fftImage[:, int(width*keep_fraction):int(width*(1-keep_fraction))] = 0
        ft_modified = mod_fftImage.tolist()
        
        ft_denoised = fft_twoD_inverse(ft_modified)

        fft_denoised_final = unpadImage(ft_denoised, padPixelsRow, padPixelsCol)
        fftRealDenoised = np.real(fft_denoised_final)

        # Create a 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image without LogNorm
        im1 = axes[0].imshow(imgDisplay, cmap='gray')
        axes[0].set_title('Original Image')

        im2 = axes[1].imshow(fftRealDenoised, cmap = 'gray')  # Use alpha for overlapping channels
        axes[1].set_title('Denoised Image')

        # Add colorbars
        cb1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        cb2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to prevent clipping of colorbars
        plt.tight_layout()

        # Show the plot
        plt.show()

    elif args.mode == 3:

        ftImage, builtInFtImage, imgDisplay, padPixels = padImage(args.image)
        ftNumpy = np.array(ftImage)

        keep_fraction = 0.01
        numZeroes = 0
        ft_modifiedOne = []

        for twoDArr in ftImage:
            im_fft2 = np.array(twoDArr).copy()
            threshold = keep_fraction * np.max(np.abs(im_fft2))
            # Set coefficients below the threshold to zero
            im_fft2[np.abs(im_fft2) < threshold] = 0
            numZeroes = numZeroes + np.count_nonzero(im_fft2 == 0)
            ft_modifiedOne.append(im_fft2.tolist())

        ft_denoisedOne = []

        for twoDArr in ft_modifiedOne:
            inverse_row = fft_twoD_inverse(twoDArr)
            ft_denoisedOne.append(inverse_row)

        fft_denoised_finalOne = unpadImage(ft_denoisedOne, padPixels)
        fftRealDenoisedOne = np.real(fft_denoised_finalOne)
        print(f"Number of zeroes after thresholding: {numZeroes}")

        # #99.9% are zeroes

        keep_fractionSix = 0.001
        numZeroes = 0
        ft_modifiedSix = []

        for twoDArr in ftImage:
            im_fft2 = np.array(twoDArr).copy()
            threshold = keep_fractionSix * np.max(np.abs(im_fft2))
            # Set coefficients below the threshold to zero
            im_fft2[np.abs(im_fft2) < threshold] = 0
            numZeroes = numZeroes + np.count_nonzero(im_fft2 == 0)
            ft_modifiedSix.append(im_fft2.tolist())

        ft_denoisedSix = []

        for twoDArr in ft_modifiedSix:
            inverse_row = fft_twoD_inverse(twoDArr)
            ft_denoisedSix.append(inverse_row)

        fft_denoised_finalSix = unpadImage(ft_denoisedSix, padPixels)
        fftRealDenoisedSix = np.real(fft_denoised_finalSix)
        print(f"Number of zeroes after thresholding: {numZeroes}")

        # Create a 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image without LogNorm
        im1 = axes[0].imshow(imgDisplay, cmap='gray')
        axes[0].set_title('Original Image')

        for i in range(3):  # Loop over RGB channels
            im2 = axes[1].imshow(fftRealDenoisedOne[:,:,i], cmap = 'gray')  # Use alpha for overlapping channels
        axes[1].set_title('Denoised Image')

        # Add colorbars
        cb1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        cb2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to prevent clipping of colorbars
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    elif args.mode == 4:
        # n=5
        # random_array = np.random.rand(2**n, 2**n)
        # listForm = random_array.tolist()
        # dftResult = dft_naive_twoD(listForm)
        #dftNumpy = np.abs(np.array(dftResult))
        # fftResult = fft_twoD(listForm)
        #fftNumpy = np.abs(np.array(fftResult))
        # builtInFft = np.fft.fft2(np.array(listForm))
        # builtNumpy = np.abs(builtInFft)
        # print(np.allclose(dftNumpy, fftNumpy, rtol=1e-5, atol=1e-8))
        # print(np.allclose(dftNumpy, builtNumpy, rtol=1e-5, atol=1e-8))
        # print(np.allclose(fftNumpy, builtNumpy, rtol=1e-5, atol=1e-8))
        runtimesDft = [[],[],[],[],[]]
        runtimesFft = [[],[],[],[],[]]
        sizes = [32*32, 64*64, 128*128, 256*256, 512*512]
        n=5
        for i in range(3):
            for j in range(5):
                random_array = np.random.rand(2**n, 2**n)
                listForm = random_array.tolist()

                start_dft = time.time()
                dftResult = dft_naive_twoD(listForm)
                #dftNumpy = np.abs(np.array(dftResult))
                end_dft = time.time()
                runtime_dft = end_dft - start_dft
                runtimesDft[j].append(runtime_dft)

                start_fft = time.time()
                fftResult = fft_twoD(listForm)
                #fftNumpy = np.abs(np.array(fftResult))
                end_fft = time.time()
                runtime_fft = end_fft - start_fft
                runtimesFft[j].append(runtime_fft)

                n = n + 1

                if j==4:
                    n = 5

        runTimeDftAve = []
        runTimeFftAve = []
        runTimeDftStd = []
        runTimeFftStd = []

        for runTime in runtimesDft:
            npRuntime = np.array(runTime)
            dftMean = np.mean(npRuntime)
            runTimeDftAve.append(dftMean)
            dftStd = np.std(npRuntime)
            runTimeDftStd.append(dftStd)

        for runTime in runtimesFft:
            npRuntime = np.array(runTime)
            fftMean = np.mean(npRuntime)
            runTimeFftAve.append(fftMean)
            fftStd = np.std(npRuntime)
            runTimeFftStd.append(fftStd)

        sizeProblem = np.array(sizes)
        dftMeans = np.array(runTimeDftAve)
        dftStd = np.array(runTimeDftStd)
        fftMeans = np.array(runTimeFftAve)
        fftStd = np.array(runTimeFftStd)

        # Plotting with error bars
        plt.errorbar(sizeProblem, dftMeans, yerr=dftStd, label='Naive', marker='o')
        plt.errorbar(sizeProblem, fftMeans, yerr=fftStd, label='FFT', marker='s')

        # Adding labels and title
        plt.xlabel('Problem Size')
        plt.ylabel('Runtime in Seconds')
        plt.title('Average Runtimes of 2D Naive DFT vs 2D Fast Fourier Transform')

        # Adding legend
        plt.legend()

        # Display the plot
        plt.show()

        # builtInFft = np.fft.fft2(np.array(listForm))
        # builtNumpy = np.abs(builtInFft)
        # print(np.allclose(dftNumpy, fftNumpy, rtol=1e-5, atol=1e-8))
        # print(np.allclose(dftNumpy, builtNumpy, rtol=1e-5, atol=1e-8))
        # print(np.allclose(fftNumpy, builtNumpy, rtol=1e-5, atol=1e-8))
        
    








        


        



