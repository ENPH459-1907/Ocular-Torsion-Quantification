import numpy as np

def noise_replace(iris, upper_occlusion_theta, lower_occlusion_theta):
    '''
    Replaces portions of the iris with noise.

    Input:
        iris - numpy array of pixel intensities of transformed iris.
        upper_occlusion_theta - tuple that defines the angular bounds of the upper iris
                                that is to be replaced with noise.
        lower_occlustion_theta - tuple that defines the angular bounds of the lower iris
                                 that is to be replaced with noise.

    Output:
        iris - numpy array of pixel intensities of transformed iris with desired
               portions replaced with noise.
    '''
    height = iris.shape[0]

    top_lid_min = int(upper_occlusion_theta[0])
    top_lid_max = int(upper_occlusion_theta[1])

    lower_lid_min = int(lower_occlusion_theta[0])
    lower_lid_max = int(lower_occlusion_theta[1])

    width_upper = top_lid_max  - top_lid_min
    width_lower = lower_lid_max - lower_lid_min

    # find mean iris intensity and use it to construct noise with same mean.
    normalized_magnitude = (np.sum(iris)/iris.size)

    upper_noise = np.random.random((height,width_upper))*normalized_magnitude
    lower_noise = np.random.random((height,width_lower))*normalized_magnitude

    iris[:,top_lid_min:top_lid_max] = upper_noise
    iris[:,lower_lid_min:lower_lid_max] = lower_noise
    return iris

def iris_extension(iris, theta_resolution, lower_theta = 0, upper_theta = 0):
    '''
    Extends the iris by inserting portions of the iris before zero at the beginning and appending
    portions of th iris after zero to the end.

    Inputs:
        iris - numpy array of pixel intensities of transformed iris
        theta_resolution - double; degree of upsampling used in the transform along the theta axis
        lower_theta - int; degrees below zero that define the iris insertion bounds
        upper_theta - int; degrees above zero that define the iris extension bounds

    Outputs:
        iris - numpy array of pixel intensities that represents the extended iris
    '''
    upper_theta = int(upper_theta/theta_resolution)
    lower_theta = int(lower_theta/theta_resolution)
    UPPER_BOUND = int(360/theta_resolution)

    iris_extension = iris[:,0:upper_theta]
    iris_insertion = iris[:,(UPPER_BOUND+lower_theta):UPPER_BOUND]

    iris = np.concatenate((iris_insertion,iris,iris_extension),axis=1)
    return iris
