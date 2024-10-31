"""
frame_resizer.py
Nick Flanders

Creates a new directory with all images in the given directory
resized to 800 x 600 for to save space when being displayed in
a digital picture frame
"""

import sys
import os
import math
from PIL import Image


def update_progress(completed, message=None, width=40):
    """
    Display a progress bar for a task that is the given percent completed
    :param completed:   the ratio of the task completed (con the closed interval [0, 1])
    :param message:     the preceding message to display in front of the progress bar
    :param width:       the width of the progress bar
    """
    if message is None:
        message_str = ""
    else:
        message_str = message
    done_width = int(math.ceil(completed * width))
    sys.stdout.write("\r" + message_str + " [{}]".format(" " * (width - 1)) + " " + str(int(completed * 100)) + "%")
    sys.stdout.write("\r" + message_str + " " + '\u2588' * (done_width + 1))

# constants for the max height and width to resize images to
WIDTH = 800
HEIGHT = 600
"""
# check command line syntax
if len(sys.argv) != 2:
    print("Syntax:   python frame_resizer.py <directory of images>")
    sys.exit(1)
"""
directory = "/home/cassio/Documents/Doutorado/Disciplinas/RNA/cats/yuru"

# ensure target directory is reachable
if not os.path.exists(directory):
    print("Unable to find directory: ", directory)
    sys.exit(1)

output_dir = directory + "/resized"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

print("\nResizing images in", directory, "\n")

# log all errors to be displayed at script termination
error_log = []
img_files = os.listdir(directory)
img_files.remove("resized")
for index, infile in enumerate(img_files):
    outfile = output_dir + "/" + infile
    if not os.path.exists(outfile):
        try:
            im = Image.open(directory + "/" + infile)
            ratio = min(WIDTH / im.width, HEIGHT / im.height)
            #im.thumbnail((im.size[0] * ratio, im.size[1] * ratio), Image.BICUBIC)
            im = im.resize((224, 224), Image.BICUBIC)
            im.save(outfile, "JPEG")
        except IOError:
            error_log.append("cannot create resized image for '%s'" % infile)
    # display progress bar
    update_progress((index + 1) / len(img_files), message="Resizing")

# check if any images failed to be resized
if len(error_log) == 0:
    print("\n\nAll images successfully resized!")
else:
    print("\n\nThe following errors occurred during resizing:")
    for error in error_log:
        print(error)