import argparse
import cv2
import numpy as np
from fourier_transform import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

def padImage(imageStr):
    image = cv2.imread(imageStr)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgList = img.tolist()
    numRows = len(imgList[0])

    n=0
    while 2**n <= numRows:
        n = n + 1

    paddingPixels = (2**n) - numRows
    
    modImage = []

    for rowTwoD in imgList:
        for i in range(paddingPixels):
            rowTwoD.append([0,0,0])
        
        modImage.append(rowTwoD)

    modFftImage = []
    builtInFft = []

    for twoDRow in modImage:
        ft_row = fft_twoD(twoDRow)
        modFftImage.append(ft_row)
        built_in_fft_row = np.fft.fft2(np.array(twoDRow))
        builtInFft.append(built_in_fft_row.tolist())

    return modFftImage, builtInFft, img, paddingPixels

def unpadImage(paddedImage, paddingPixels):
    # Use array slicing to remove the padded region
    return np.array(paddedImage)[:, :-paddingPixels, :]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=int, default=1)
    parser.add_argument('-i', '--image', type=str, default="moonlanding.png")

    args = parser.parse_args()

    if args.mode == 1:
        
        ftImage, builtInFtImage, imgDisplay, padPixels = padImage(args.image)

        fftImage = np.array(ftImage)
        fftImageBuiltIn = np.array(builtInFtImage)

        absFftImage = np.abs(fftImage)
        absBuiltInFft = np.abs(fftImageBuiltIn)

        print(np.allclose(absFftImage, absBuiltInFft, rtol=1e-5, atol=1e-8))
        # Create a 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image without LogNorm
        im1 = axes[0].imshow(imgDisplay, cmap='viridis')
        axes[0].set_title('Original Image')

        # Plot the second image with LogNorm for each RGB channel
        for i in range(3):  # Loop over RGB channels
            im2 = axes[1].imshow(absFftImage[:,:,i], cmap='viridis', norm=LogNorm(), alpha=0.3)  # Use alpha for overlapping channels
        axes[1].set_title('Fourier Transform')

        # Add colorbars
        cb1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        cb2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to prevent clipping of colorbars
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    elif args.mode == 2:

        ftImage, builtInFtImage, imgDisplay, padPixels = padImage(args.image)
        
        ft_modified = []
        keep_fraction = 0.1

        for twoDArr in ftImage:
            im_fft2 = np.array(twoDArr).copy()
            r,c = im_fft2.shape
            im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
            im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
            ft_modified.append(im_fft2.tolist())
        
        ft_denoised = []

        for twoDArr in ft_modified:
            inverse_row = fft_twoD_inverse(twoDArr)
            ft_denoised.append(inverse_row)

        fft_denoised_final = unpadImage(ft_denoised, padPixels)
        fftRealDenoised = np.real(fft_denoised_final)

        # Create a 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image without LogNorm
        im1 = axes[0].imshow(imgDisplay, cmap='gray')
        axes[0].set_title('Original Image')

        for i in range(3):  # Loop over RGB channels
            im2 = axes[1].imshow(fftRealDenoised[:,:,i], cmap = 'gray')  # Use alpha for overlapping channels
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

        # builtInFft = np.fft.fft2(np.array(listForm))
        # builtNumpy = np.abs(builtInFft)
        # print(np.allclose(dftNumpy, fftNumpy, rtol=1e-5, atol=1e-8))
        # print(np.allclose(dftNumpy, builtNumpy, rtol=1e-5, atol=1e-8))
        # print(np.allclose(fftNumpy, builtNumpy, rtol=1e-5, atol=1e-8))
        
    








        


        



