import argparse
import cv2
import numpy as np
from fourier_transform import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

    for row in modImage:
        ft_row = fft_twoD(row)
        modFftImage.append(ft_row)
        built_in_fft_row = np.fft.fft2(row)
        builtInFft.append(built_in_fft_row.tolist())

    return modFftImage, builtInFft, img, paddingPixels

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

        numZeroes = 0
        ft_denoised = []

        for twoDRow in ftImage:
            for row in twoDRow:
                for val in row:
                    if abs(abs(val) - (2*np.pi)) <= 0.2:
                        val = 0
                        numZeroes = numZeroes + 1
            
            inverse_row = fft_twoD_inverse(twoDRow)
            
            for i in range(padPixels):
                inverse_row.pop()
            
            ft_denoised.append(inverse_row)
        
        print(numZeroes)

        fftDenoised = np.array(ft_denoised)
        fftAbsDenoised = np.abs(fftDenoised)

        # Create a 1x2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the first image without LogNorm
        im1 = axes[0].imshow(imgDisplay, cmap='viridis')
        axes[0].set_title('Original Image')

        im2 = axes[1].imshow(fftAbsDenoised, cmap='viridis')
        axes[1].set_title('Denoised Image')

        # Add colorbars
        cb1 = plt.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
        cb2 = plt.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)

        # Adjust layout to prevent clipping of colorbars
        plt.tight_layout()

        # Show the plot
        plt.show()

        


        



