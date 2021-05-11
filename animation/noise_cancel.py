import numpy as np

class Noise_cancel:
    def __init__(self):
        print("ok noise cancel")

    def cancelling(self,array,increment,noise_margin):
        if array[noise_margin]>0:
            array[noise_margin:noise_margin+increment]=0

        return array