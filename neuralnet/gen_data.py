import numpy as np
import matplotlib.pyplot as plt
import pickle

def rect(x: np.ndarray) -> np.ndarray:
    """
    Rectangle function.
    """
    y = (5/2)*np.ones([len(x)])
    for i in range(1, 40, 2):
        y += np.sin(x*i)/(np.pi*i)
    return 2*(y - 5/2)

def sinc(x: np.ndarray) -> np.ndarray:
    """
    sinc function
    """
    return np.sinc(x)

def saw_tooth(x:np.ndarray)->np.ndarray:
    """
    saw_tooth function
    """
    return -1.5*np.arctan(1/np.tan(x))

def sin(x:np.ndarray):
    return np.sin(x)

def cos(x:np.ndarray):
    return np.cos(x)

class GEN_DATA:
 
    def __init__(self,wave_size=41,visualize=True):
        self.data=list()
        self.funcs=[sin, cos, sinc, saw_tooth, rect]
        self.wave_size= wave_size
        self.visualize=visualize

    def gen_data(self, data_num):
        t= np.linspace(2,3,self.wave_size)
        
        for idx in range(data_num):
            amp, phase, T=3*np.random.rand(1), 5*np.pi*np.random.rand(1), np.random.randint(-3,3,1)
            modified_t = 2*np.pi*(t+phase)/(T+0.0001)
            # print(amp, phase, period)

            if idx %int(data_num*0.1)==0:
                print("progress",idx*100/data_num,"(%)")

            idx = idx % len(self.funcs)
            y_t= amp*self.funcs[idx](modified_t)
            y_noist_t= y_t+np.random.randn(self.wave_size)
            
            wave_data={"pure":y_t,"noise":y_noist_t}
            self.data.append(wave_data)
            
            if self.visualize:
                print("ok")
                plt.plot(modified_t,y_t, color='red')
                plt.plot(modified_t,y_noist_t, color='blue')
                plt.show()
        
        with open('wave_data_val.pickle','wb') as f:
            pickle.dump(self.data,f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    gen= GEN_DATA(visualize=False)
    gen.gen_data(10000)
