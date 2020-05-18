"""
This routine is aimed at cropping the melspectrogram images retrieved from https://osf.io/jyfpg/
Sick sounds dataset that can be found in this url https://osf.io/tmkud/
"""
from skimage import io
import os


def crop_melspecs(path):

    # Cropping box
    x1 = 80
    x2 = 575
    y1 = 60
    y2 = 427

    os.chdir(path)
    melspecs = os.listdir(path)
    for melspec in melspecs:
    
        _im = io.imread(melspec)
        cropped_melspec = _im[y1:y2, x1:x2]
        io.imsave(melspec, cropped_melspec)

if __name__ == '__main__':

    path = '/home/ideaslocascdo/personal-projects/cough_covid19/melspectrograms/train/sick'
    crop_melspecs(path)
