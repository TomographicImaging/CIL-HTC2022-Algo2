#   Copyright 2022 United Kingdom Research and Innovation
#   Copyright 2022 Technical University of Denmark
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
from PIL import Image
import util
import os, sys

def calculate_score(path_to_our_segmentation, path_to_reference_image_png):

    img_data = Image.open(path_to_our_segmentation)
    img_arr = np.array(img_data)

    ref = util.loadImg(path_to_reference_image_png)

    return util.calcScoreArray(img_arr[:,:,0], ref)  

if __name__ == "__main__":
    score = calculate_score(os.path.abspath(sys.argv[1]),
                            os.path.abspath(sys.argv[2]))
    print ("Score: {}".format(score))