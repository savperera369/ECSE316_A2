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

        #Absolute value of Fourier Transform for our function
        absFftImage = np.abs(fftImage)
        #Absolute value of Fourier Transform for built in Function
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

        keep_fraction = 0.3
        mod_fftImage = np.array(ftImage).copy()
        non_zero_countBefore = np.count_nonzero(mod_fftImage)
        height,width = mod_fftImage.shape
        mod_fftImage[int(height*keep_fraction):int(height*(1-keep_fraction))] = 0
        mod_fftImage[:, int(width*keep_fraction):int(width*(1-keep_fraction))] = 0
        non_zero_countAfter = np.count_nonzero(mod_fftImage)
        ft_modified = mod_fftImage.tolist()
        
        print("Number of Non Zeroes Before Denoising: ", non_zero_countBefore)
        print("Number of Non Zeroes After Denoising: ", non_zero_countAfter)
        print("Fraction of Non Zeroes: ", non_zero_countAfter/mod_fftImage.size)

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

        ftImage, builtInFtImage, imgDisplay, padPixelsRow, padPixelsCol = padImage(args.image)

        compressedImages = []
        
        compression_percentage = 20
        for i in range(5):
            mod_fft = np.array(ftImage).copy()
            threshold = np.percentile(np.abs(mod_fft), compression_percentage)
            # Set coefficients below the threshold to zero
            mod_fft[np.abs(mod_fft) < threshold] = 0
            numNonZeroes = np.count_nonzero(mod_fft)
            ft_modified = mod_fft.tolist()

            print(f"Number of Non Zeroes for Compression Percentage of {compression_percentage}: {numNonZeroes}")
            
            ft_denoised = fft_twoD_inverse(ft_modified)
            fft_denoised_final = unpadImage(ft_denoised, padPixelsRow, padPixelsCol)
            fftRealDenoised = np.real(fft_denoised_final)
            compressedImages.append(fftRealDenoised)

            if i==3:
                compression_percentage += 19.9
            else:
                compression_percentage += 20
        
        # Create a 2x3 subplot
        fig, axes = plt.subplots(2, 3, figsize=(10, 5))

        # Plot images in each subplot
        axes[0, 0].imshow(imgDisplay, cmap='gray')
        axes[0, 0].set_title('Original Image')

        axes[0, 1].imshow(compressedImages[0], cmap='gray')
        axes[0, 1].set_title('20% Compression')

        axes[0, 2].imshow(compressedImages[1], cmap='gray')
        axes[0, 2].set_title('40% Compression')

        axes[1, 0].imshow(compressedImages[2], cmap='gray')
        axes[1, 0].set_title('60% Compression')

        axes[1, 1].imshow(compressedImages[3], cmap='gray')
        axes[1, 1].set_title('80% Compression')

        axes[1, 2].imshow(compressedImages[4], cmap='gray')
        axes[1, 2].set_title('99.9% Compression')

        # Adjust layout to prevent clipping of titles
        plt.tight_layout()

        # Show the plot
        plt.show()
    
    elif args.mode == 4:

        runtimesDft = [[],[],[],[],[]]
        runtimesFft = [[],[],[],[],[]]
        sizes = [32*32, 64*64, 128*128, 256*256, 512*512]
        n=5

        for i in range(4):
            for j in range(5):
                random_array = np.random.rand(2**n, 2**n)
                listForm = random_array.tolist()

                start_dft = time.time()
                dftResult = dft_naive_twoD(listForm)
                end_dft = time.time()
                runtime_dft = end_dft - start_dft
                runtimesDft[j].append(runtime_dft)

                start_fft = time.time()
                fftResult = fft_twoD(listForm)
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
        dftError = (np.array(runTimeDftStd)) * 2
        
        fftMeans = np.array(runTimeFftAve)
        fftError = (np.array(runTimeFftStd)) * 2

        # Plotting with error bars
        plt.errorbar(sizeProblem, dftMeans, yerr=dftError, label='2D Naive', marker='o')
        plt.errorbar(sizeProblem, fftMeans, yerr=fftError, label='2D FFT', marker='s')

        # Adding labels and title
        plt.xlabel('Problem Size')
        plt.ylabel('Runtime in Seconds')
        plt.title('Average Runtimes of 2D Naive DFT vs 2D Fast Fourier Transform')

        # Adding legend
        plt.legend()

        # Display the plot
        plt.show()

        # runtimesDft1D = [[],[],[],[],[],[],[],[],[]]
        # runtimesFft1D = [[],[],[],[],[],[],[],[],[]]
        # sizes1D = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        # n=5

        # for i in range(10):
        #     for j in range(9):
        #         random_array = np.random.rand(2**n)
        #         listForm = random_array.tolist()

        #         start_dft = time.time()
        #         dftResult = dft_naive_oneD(listForm, False)
        #         end_dft = time.time()
        #         runtime_dft = end_dft - start_dft
        #         runtimesDft1D[j].append(runtime_dft)

        #         start_fft = time.time()
        #         fftResult = fast_fourier_transform(listForm, False)
        #         end_fft = time.time()
        #         runtime_fft = end_fft - start_fft
        #         runtimesFft1D[j].append(runtime_fft)

        #         n = n + 1

        #         if j==8:
        #             n = 5

        # runTimeDftAve1D = []
        # runTimeFftAve1D = []
        # runTimeDftStd1D = []
        # runTimeFftStd1D = []

        # for runTime in runtimesDft1D:
        #     npRuntime = np.array(runTime)
        #     dftMean = np.mean(npRuntime)
        #     runTimeDftAve1D.append(dftMean)
        #     dftStd = np.std(npRuntime)
        #     runTimeDftStd1D.append(dftStd)

        # for runTime in runtimesFft1D:
        #     npRuntime = np.array(runTime)
        #     fftMean = np.mean(npRuntime)
        #     runTimeFftAve1D.append(fftMean)
        #     fftStd = np.std(npRuntime)
        #     runTimeFftStd1D.append(fftStd)

        # sizeProblem1D = np.array(sizes1D)
        # dftMeans1D = np.array(runTimeDftAve1D)
        # dftError1D = (np.array(runTimeDftStd1D)) * 2
        # print("dft")
        # print(dftMeans1D)
        # print(dftError1D)
        # fftMeans1D = np.array(runTimeFftAve1D)
        # fftError1D = (np.array(runTimeFftStd1D)) * 2
        # print("\nfft")
        # print(fftMeans1D)
        # print(fftError1D)
        
        


        
    








        


        



