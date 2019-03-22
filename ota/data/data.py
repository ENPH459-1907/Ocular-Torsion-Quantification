import sys
import os
import pickle

from matplotlib import pyplot as plt
import numpy as np
import csv
import pdb
import ota.pupil as p

class Data():

    def __init__(self, name, path=None):
        """
        Initialize Data object.

        Parameters
        -----------------------------------
        name : String
            Name of data object. Used to as the filename when saving the data.
        path : String
            Path of desired directory where the data should be saved.
        """
        self.name = name
        self.file_name = str(self.name) + '.csv'
        self.metadata = None
        self.start_frame = None
        self.torsion = None
        self.pupil_list = None

        # If save path is not specified, set it to the current directory
        if path is not None:
            self.path = os.path.abspath(path)
        else:
            self.path = os.path.curdir

    def set(self, torsion, start_frame=0, pupil_list=None, metadata=None, frame_index_list=None):
        """
        Populate data fields with values.

        Parameters
        -------------------------------------
        torsion : list
            List of ocular torsion values.
        start_frame : int
            Video frame index corresponding to the start point of torsion analysis.
        pupil_list : list
            List of pupil objects.
        metadata : dict
            Any video/torsion metadata the user wishes to store.
        frame_index_list : list
            List of same length as torsion. Maps torsion values to corresponding video frame indeces.
            Specified when the video frames analyzed are not just subsequent frames.
        """
        self.metadata = metadata
        self.start_frame = start_frame
        self.torsion = torsion
        self.frame_index_list = frame_index_list
        self.pupil_list = pupil_list

    def save(self):
        """
        Save data at data object path.

        Parameters
        -------------------------------------
        file_name : String
            Desired save file name.
            e.g. "saved_data1"

        path : String
            Optional parameter specifying a save location.

        mode : String
            Optional parameter specifiying the desired type of save file.
            'csv' - Save data as a csv file.
            'pickle' - save data as a pickled python object.
        """
        # Check to make sure file_name is string.
        if not isinstance(self.file_name, str):
            # TODO throw exception
            print('Please enter a valid file name string.')
            return None

        # Check to make sure path is valid
        # TODO use os.path.isfile to check file path
        if os.path.isdir(self.path):
            save_loc = self.path
        else:
            print(self.path)
            save_loc = os.path.abspath(settings.SAVE_PATH)

        save_str = os.path.join(save_loc, self.file_name)

        with open(save_str, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # save metadata first
            csvwriter.writerow(['METADATA'])

            if self.metadata:
                for k, v in self.metadata.items():
                    csvwriter.writerow([k, v])

                # optional fps to calculate video time
                fps = self.metadata.get('VIDEO_FPS', None)

            csvwriter.writerow(['TORSION RESULTS'])
            csvwriter.writerow(['Frame Index',
                                'Frame Time',
                                'Torsion [deg]',
                                'Torsion Derivative [deg]',
                                'Pupil Center Column',
                                'Pupil Center Row',
                                'Pupil Radius [pixels]'])

            # default time is empty string
            time = ''

            # save to csv
            for i, deg in enumerate(self.torsion):
                # Check if a specific frame index list exists or not
                if self.frame_index_list is None:
                    if fps:
                        time = 1/fps * (self.start_frame + i)
                        frame = self.start_frame + i
                else:
                    if fps:
                        time = 1/fps * self.frame_index_list[i]
                        frame = self.frame_index_list[i]

                # Check to see if a pupil list exists
                if self.pupil_list is None:
                    pupil_center_col = ''
                    pupil_center_row = ''
                    pupil_radius = ''
                else:
                    temp_pupil = self.pupil_list[i+self.start_frame]
                    pupil_center_col = temp_pupil.center_col
                    pupil_center_row = temp_pupil.center_row
                    pupil_radius = temp_pupil.radius

                # Find the change in angle
                # ie cross correlation with respect to previous frame
                if i == 0:
                    delta_deg = 0
                else:
                    delta_deg = deg - self.torsion[i]
                print(delta_deg)

                # Write the results
                csvwriter.writerow([frame,
                                    time,
                                    repr(deg),
                                    repr(delta_deg),
                                    pupil_center_col,
                                    pupil_center_row,
                                    pupil_radius])

    def load(self):
        """
        Load data saved in .csv file saved at path specified in self.path.
        """
        torsion = []
        pupil_list = []
        frame_index_list = []
        start_frame = 0
        metadata = {}

        file_path = os.path.abspath(os.path.join(self.path, self.file_name))

        if os.path.isfile(file_path):
            with open(file_path, newline='') as f:
                line = next(f)

                metadata_flag = False
                if 'metadata' in line.lower():
                    metadata_flag = True

                i = 0

                for line in f:

                    if 'torsion' in line.lower():
                        metadata_flag = False
                        continue

                    line = line.replace('\n','').split(',')

                    if metadata_flag:
                        metadata[line[0]] = line[1]
                    else:
                        if i == 0:
                            start_frame = int(line[0])

                        frame_index_list.append(int(line[0]))
                        torsion.append(float(line[2]))
                        if line[3] == '' and line[4] == '' and line[5] == '':
                            pupil_list.append(None)
                        else:
                            temp_pupil = p.Pupil(None, None, skip_init=True)
                            temp_pupil.center_col = float(line[3])
                            temp_pupil.center_row = float(line[4])
                            temp_pupil.radius = float(line[5])
                            pupil_list.append(temp_pupil)

                        i += 1

            # set object from file
            self.set(torsion, start_frame=start_frame, pupil_list=pupil_list, metadata=metadata, frame_index_list=frame_index_list)

def load(file_str):
    """
    Load data from file into data object.
    Overwrites any currently existing data in that object.

    Parameters
    -------------------------------------
    file_str : String
        String pointing to file name to be loaded.
        e.g. "D:\data\saved_data1.csv"

    Returns
    -------------------------------------
    data : Data object
        Data object stored in file being loaded
    """

    if os.path.isfile(file_str):
        name = os.path.basename(file_str)
        path = os.path.dirname(file_str)

        # Initialize empty data object
        d = Data(name, path=path)

        # Load the data
        d.load()

        # return data object
        return d

    else:
        print('Please enter a valid file string.')
        return None
