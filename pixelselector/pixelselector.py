import numpy as np
from PIL import Image
import random

class MapImage:
    def __init__(self, width_px, height_px, img_url):
        self.width = width_px
        self.height = height_px
        
        self.imgurl = img_url
        self.imgobj = Image.open(self.imgurl).convert('RGB')
        
        self.imgarray1D = np.array(self.imgobj.getdata())
        self.imgarray2D = np.array(self.imgobj)

    def show2DArray(self):
        print(self.imgarray2D)

    def show2DPixel(self, x, y):
        print(self.imgobj.getpixel((x,y)))

    def getXYFrom1Dpixel(self, i):
        x = i % self.width
        y = i // self.width
        return (x,y)
    
    def selectRandomPixels(self, numpixels):
        validpixels = []
        
        for i,pixel in enumerate(self.imgarray1D):
            isPixelWhite = (pixel == [255,255,255]).all()
            # check all list elements for if pixel is exactly [255,255,255]
            
            if not isPixelWhite:
                validpixels.append((pixel, self.getXYFrom1Dpixel(i)))

        selectedpixels = random.sample(validpixels, numpixels)

        for pixel in selectedpixels:
            print(pixel[1])

  
print('--- Metro Manila ---')
MetroManila = MapImage(19, 45, 'metromanila_pixelated.png')
# MetroManila.show2DArray()
MetroManila.selectRandomPixels(25)
print()


print('--- Metro Iloilo ---')
MetroIloilo = MapImage(11, 10, 'metroiloilo_pixelated.png')
# MetroIloilo.show2DArray()
MetroIloilo.selectRandomPixels(5)
