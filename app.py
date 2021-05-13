"""
The main application
"""
# from animation import FourierAnimation
from functions import VariableNotFoundError
from matplotlib.backends import backend_tkagg
import tkinter as tk
import torch
import numpy as np
import matplotlib.pyplot as plt
from animator import Animator
from circles import Circles, VerticalRescaler
from sympy import abc
# from noise_cancel import Noise_cancel
from functions import FunctionRtoR
from noise_net import WaveRNN
class Noise_cancel:
    def __init__(self):
        print("ok noise cancel")
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        # else:
        device = torch.device('cpu')
        
        self.model= WaveRNN().to(device)
        self.model.load_state_dict(torch.load("neuralnet/weight.pth"))
        self.model.eval()        


    def cancelling(self,array,increment,noise_margin):
        print(noise_margin)
        if array[noise_margin]>0:
            array[noise_margin:noise_margin+increment]=0

        return array
class FourierAnimation(Animator):

    def __init__(self, function_name: str, dpi: int = 120) -> None:
        """
        The constructor
        """
        figsize = (8, 4)
        Animator.__init__(self, dpi, figsize, 15)
        self.counts = -2
        self.increment = 4
        ax = self.figure.add_subplot(
            1, 1, 1, aspect="equal")
        maxval = 2.0
        view = 2*maxval
        ax.set_xlim(-1.0*maxval - 0.1*view, 8)
        ax.set_ylim(-maxval-0.1*view, maxval+0.1*view)

        #이 부위를 뭔가 손대야할 것 같음
        ax.set_xticks([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        ax.set_yticks([])
        ax.set_xticklabels(["noise","noise\ncancelling",r"$s - 1/3\pi$", r"$s$", r"$s+1/3\pi$",
                            r"$s + 2/3\pi$", r"$s + \pi$"])
        ax.grid(linestyle="--")
        self.circles = Circles(ax)
        function = FunctionRtoR(function_name, abc.t)
        self.function = function
        # self.function_display = ax.text(-2.05, 1,
        #                                 r"$f(t) = %s$"
        #                                r" ,   $ t = s (mod(2 \pi)) - \pi $" %
        #                                 function.latex_repr)
        self.function_display = ax.text(-2.05, 1,
                                        r"$f(t)$"
                                        r" ,   $ t = s (mod(2 \pi)) - \pi $")
        self.circles.set_function(function)
        self.add_plots([self.circles.get_plot()])

        v_line, = ax.plot([0.0, 2.0], [0.0, 0.0],
                          linewidth=1.0, linestyle="--", color="red")
        self.v_line = v_line

        self.t = np.linspace(-np.pi, np.pi - 2*np.pi/256.0, 256)

        self.x = function(self.t) + np.random.randn(256)/20
        self.x = np.roll(self.x, self.counts)
        function_plot, = ax.plot(np.linspace(2.0, 8.0, 256),
                                 self.x, linewidth=1.0,
                                 # color="black"
                                 color="gray"
                                 )
        amps = self.circles.get_amplitudes()
        self.x2 = np.fft.irfft(amps)
        self.x2 = np.roll(self.x2, self.counts)
        circle_function_plot, = ax.plot(np.linspace(2.0, 8.0, 256),
                                        self.x, linewidth=1.0,
                                        color="red")
        self.function_plot = function_plot
        self.circle_function_plot = circle_function_plot
        # TODO: Have the vertical rescaler class defined in this file instead
        # and pass it to Circles. Try to only use one instance of this class.
        self._rescale = VerticalRescaler()
        self.y_limits = [-1.0, 1.0]
        self.noise_cancel= Noise_cancel()
        self.noise_cancel_toggle=False
        self.x_queue=np.zeros(256)
    def noise_cancelling(self):
        print("ok")
     
    def update(self, delta_t: float) -> None:
        """
        Overridden update method from the Animator class
        """
        # print("fps: %.0f" % (1/delta_t))
        self.counts += self.increment
        # t1 = perf_counter()
        self.circles.update_plots(self.counts)
        # t2 = perf_counter()
        end_point = self.circles.get_end_point()
        # print("%f" % ((t2 - t1) / delta_t))
        x1 = np.imag(end_point)
        y = np.real(end_point)
        self.v_line.set_xdata([x1, 2.0])
        self.v_line.set_ydata([y, y])
        self.x = np.roll(self.x, self.increment)
        
        noise_margin=41
        self.x_queue[self.increment:]=self.x_queue[:256-self.increment]
        self.x_queue[:self.increment]=self.x[:self.increment]+np.random.randn(1)/20
        
        #button 누를시 동작가능하게 만들어야함.
        if self.noise_cancel_toggle:
            self.x_show = self.noise_cancel.cancelling(self.x_queue,self.increment,noise_margin+2)
        else:
            self.x_show=self.x_queue.copy()
            
        self.x2 = np.roll(self.x2, self.increment)
        self.function_plot.set_ydata(self.x_show)
        self.circle_function_plot.set_ydata(self.x2)
        

    def set_speed(self, speed: int) -> None:
        """
        Set the speed of the animation.
        """
        speed = int(speed)  # Fix a bug
        self.increment = speed

    def get_speed(self) -> int:
        """
        get the speed of the animation
        """
        return self.increment

    def set_function(self, function_name: str) -> None:
        """
        Set function.
        """
        
        self.function = FunctionRtoR(function_name, abc.t)
        self.function_display.set_text(r"$f(t) = %s$"
                                       r" ,   $ t = s (mod(2 \pi)) - \pi $" %
                                       self.function.latex_repr)
        self.circles.set_function(self.function)
        self._set_x()

    def set_number_of_circles(self, resolution: int) -> None:
        """
        Set the number of circles.
        """
        self.circles.set_number_of_circles(resolution)
        self._set_x2()

    def _set_x(self, *params) -> None:
        """
        Set x data.
        """
        if params == ():
            self.x = self.function(self.t)
        else:
            self.x = self.function(self.t, *params)
        self._rescale.set_scale_values(self.x, self.y_limits)
        if not self._rescale.in_bounds():
            self.x = self._rescale(self.x)
        self.x = np.roll(self.x, self.counts)
        self._set_x2()

    def _set_x2(self) -> None:
        """
        Set the x2 data.
        """
        amps = self.circles.get_amplitudes()
        self.x2 = np.fft.irfft(amps)
        self.x2 = np.roll(self.x2, self.counts)

    def set_params(self, *args: float) -> None:
        """
        Set the parameters of the function.
        """
        self.circles.set_params(*args)
        self._set_x(*args)


class App(FourierAnimation):
    """
    App class
    """
    def __init__(self) -> None:
        """
        The constructor.
        """
        self.window = tk.Tk()
        self.window.title("Visualization of Fourier Series")
        width = self.window.winfo_screenwidth()
        dpi = int(150*width//1920)
        FourierAnimation.__init__(self, "3*rect(t)/2", dpi)

        # Thanks to StackOverflow user rudivonstaden for
        # giving a way to get the colour of the tkinter widgets:
        # https://stackoverflow.com/questions/11340765/
        # default-window-colour-tkinter-and-hex-colour-codes
        #
        #     https://stackoverflow.com/q/11340765
        #     [Question by user user2063:
        #      https://stackoverflow.com/users/982297/user2063]
        #
        #     https://stackoverflow.com/a/11342481
        #     [Answer by user rudivonstaden:
        #      https://stackoverflow.com/users/1453643/rudivonstaden]
        #
        colour = self.window.cget('bg')
        if colour == 'SystemButtonFace':
            colour = "#F0F0F0"
        # Thanks to StackOverflow user user1764386 for
        # giving a way to change the background colour of a plot.
        #
        #    https://stackoverflow.com/q/14088687
        #    [Question by user user1764386:
        #     https://stackoverflow.com/users/1764386/user1764386]
        #
        self.figure.patch.set_facecolor(colour)

        self.canvas = backend_tkagg.FigureCanvasTkAgg(
            self.figure,
            master=self.window
            )
        self.canvas.get_tk_widget().grid(
                row=1, column=0, rowspan=8, columnspan=3)
        self.function_dropdown_dict = {
            "sine": "sin(t)",
            "cosine": "cos(t)",
            "gaussian": "3*exp(-t**2/(2*sigma**2 ))/2 - 1/2",
            "sinc": "3*sinc(k*(6.5)*t)/2 - 1/2",
            "rectangle": "3*rect(t)/2",
            "sawtooth": "t/pi",
            "triangle": "abs(t)"
            }
        self.function_dropdown_string = tk.StringVar(self.window)
        self.function_dropdown_string.set("Preset Waveform f(t)")
        self.function_dropdown = tk.OptionMenu(
            self.window,
            self.function_dropdown_string,
            *tuple(key for key in self.function_dropdown_dict),
            command=self.set_function_dropdown
            )
        self.function_dropdown.grid(
                row=2, column=3, padx=(10, 10), pady=(0, 0))

        

        # self.enter_function_label = tk.Label(
        #         self.window,
        #         text="Enter waveform f(t)",
        #         )
        # self.enter_function_label.grid(row=3, column=3,
        #                                sticky=tk.S + tk.E + tk.W,
        #                                padx=(10, 10),
        #                                pady=(0, 0))
        # self.enter_function = tk.Entry(self.window)
        # self.enter_function.grid(row=4, column=3,
        #                          sticky=tk.N + tk.E + tk.W + tk.S,
        #                          padx=(10, 10))
        # self.enter_function.bind("<Return>", self.set_function_entry)
        # self.update_button = tk.Button(self.window, text='OK',
        #                                command=self.set_function_entry)
        # self.update_button.grid(row=5, column=3,
        #                         sticky=tk.N + tk.E + tk.W,
        #                         padx=(10, 10),
        #                         pady=(0, 0)
        #                         )
        self.sliderslist = []
        self.circles_slider = None
        self.slider_speed = None
        self._speed = 1
        self.quit_button = None
        self._number_of_circles = 80
        self._set_widgets_after_param_sliders()

        self.noise_cancel_button = tk.Button(
                self.window, 
                text='noise canceling',
                command=self.noise_cancelling
                )
        self.noise_cancel_button.grid(row=5, column=3, padx=(10, 20),pady=(0, 0))


    def _set_widgets_after_param_sliders(self, k: int = 5) -> None:
        """
        Set widgets after parameter sliders
        """
        self.circles_slider = tk.Scale(self.window, from_=1, to=80,
                                       label="Maximum Frequency: ",
                                       orient=tk.HORIZONTAL,
                                       # length=128,
                                       command=self.set_number_of_circles)
        self.circles_slider.grid(row=k+1, column=3,
                                 sticky=tk.N + tk.E + tk.W,
                                 padx=(10, 10))
        self.circles_slider.set(self._number_of_circles)
        self.slider_speed = tk.Scale(self.window, from_=0, to=8,
                                     label="Animation Speed: ",
                                     orient=tk.HORIZONTAL,
                                     length=200,
                                     command=self.set_animation_speed)
        self.slider_speed.grid(row=k+2, column=3,
                               sticky=tk.N + tk.E + tk.W,
                               padx=(10, 10))
        self.slider_speed.set(self._speed)
        self.quit_button = tk.Button(
                self.window, text='QUIT',
                command=lambda *args: [
                        self.window.quit(), self.window.destroy()]
                    )
        self.quit_button.grid(row=k+3, column=3, pady=(0, 0))

    def set_animation_speed(self, *arg: tk.Event):
        """
        Set the speed of the animation.
        """
        j = self.slider_speed.get()
        self._speed = j
        self.set_speed(j)

    def set_function_entry(self, *event: tk.Event):
        """
        Update the function using the text entry.
        """
        try:
            self.set_function(self.enter_function.get())
            self.set_widgets()
        except VariableNotFoundError:
            print("Input not recognized.\nInput function must:\n"
                  "- depend on at least t\n"
                  "- be a recognized function\n"
                  "- use '**' instead of '^' for powers\n")

    def set_function_dropdown(self, *event: tk.Event) -> None:
        """
        Update the function by the dropdown menu.
        """
        event = event[0]
        self.set_function(self.function_dropdown_dict[event])
        if event == "gaussian":
            self.function_display.set_text(r"$f(t; \sigma) = "
                                           r"exp(-t^2/2 \sigma^2)$"
                                           r" ,   $ t = s (mod(2 \pi)) - \pi $"
                                           )
        elif event == "sinc":
            self.function_display.set_text(r"$f(t; k) = "
                                           r"sinc(kt)$"
                                           r" ,   $ t = s (mod(2 \pi)) - \pi $"
                                           )
        else:
            self.function_display.set_text(r"$f(t)$"
                                           r" ,   $ t = s (mod(2 \pi)) - \pi $"
                                           )
        self.set_widgets()


    def set_number_of_circles(self, *event: tk.Event) -> None:
        """
        Set the number of circles.
        """
        resolution = self.circles_slider.get()
        self._number_of_circles = resolution
        FourierAnimation.set_number_of_circles(self, resolution+1)

    def noise_cancelling(self):
        if self.noise_cancel_toggle:
            self.noise_cancel_toggle=False
        else:
            self.noise_cancel_toggle=True

    def slider_update(self, *event: tk.Event) -> None:
        """
        Update the parameters using the sliders.
        """
        params = []
        for i in range(len(self.sliderslist)):
            params.append(self.sliderslist[i].get())
        self.set_params(*params)

    def set_widgets(self) -> None:
        """
        Set the widgets
        """
        rnge = 10.0
        for slider in self.sliderslist:
            slider.destroy()
        self.sliderslist = []
        self.circles_slider.destroy()
        self.slider_speed.destroy()
        self.quit_button.destroy()
        default_vals = self.function.get_default_values()
        k = 0
        for i, symbol in enumerate(self.function.parameters):
            self.sliderslist.append(tk.Scale(self.window,
                                             label="change "
                                             + str(symbol) + ":",
                                             from_=-rnge, to=rnge,
                                             resolution=0.01,
                                             orient=tk.HORIZONTAL,
                                             length=200,
                                             command=self.slider_update))
            self.sliderslist[i].grid(row=i + 6, column=3,
                                     sticky=tk.N + tk.E + tk.W,
                                     padx=(10, 10), pady=(0, 0))
            self.sliderslist[i].set(default_vals[symbol])
            k += 1
        self._set_widgets_after_param_sliders(k+5)


if __name__ == "__main__":
    app = App()
    app.animation_loop()
    tk.mainloop()
