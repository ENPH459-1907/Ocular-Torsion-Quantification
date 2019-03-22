from matplotlib import pyplot as plt
import numpy as np
import cv2

global_click_coord = {}

def click_coordinates(img, window_title='', block=False):
    '''
    Returns the coordinates of the first click on an image.
    Warning: Makes use of a global variable: global_click_coord

    INPUTS
    img - cv2 image (Numpy array of intensities)
    window_title - string that becomes the title of the window, typically displays
                   instructions of what to click on in the window.
    block - Boolean parameter which instructs the function whether or not the image
            should block code progression. Should be set to True for use outside of 
            GUI development.

    OUTPUTS
    click_coord - dictionary object containing row and column indicies of click
                  coordinates

                  click_coord = {'c': column index, 'r': row index}
    '''
    global_click_coord = {}
    fig, ax = plt.subplots()
    ax.set_title(window_title)
    implot = ax.imshow(img, 'gray')

    def onclick(event):

        if event.xdata != None and event.ydata != None:

            global global_click_coord
            global_click_coord['c'] = event.xdata
            global_click_coord['r'] = event.ydata
            plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=block)
    plt.show()

def get_click_coordinates():
    return global_click_coord.copy()
