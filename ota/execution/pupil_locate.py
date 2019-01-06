from ota.video import video as vid
from ota.pupil import pupil
from ota.iris import iris
from ota.data import data as dat
from tqdm import tqdm

def construct_pupil_list(video, first_frame, last_frame, threshold=10):
    '''
    Construct a dictionary of pupil objects for a series of video frames.

    Inputs:
        video - video object
        first_frame - int
        last_frame - int

    Outputs:
        pupil_list: Dictionary of pupil objects where the key is the frame number and the value is the pupil object.
    '''

    pupil_list = {}

    for i,frame in tqdm(enumerate(video[first_frame:last_frame+1])):
        frame_loc = i + first_frame
        try:
            pupil_i = pupil.Pupil(frame, threshold)
        except pupil.EmptyAreas:
            print('Pupil not found in frame: %d \n None type object used inplace' % frame_loc)
            pupil_list[frame_loc] = None
        else:
            pupil_list[frame_loc] = pupil_i

    return pupil_list
