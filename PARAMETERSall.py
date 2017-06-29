from __future__ import print_function
from helpers import *
from sliding_window_helpers import *
#from imdb_data import imdb_data
import fastRCNN, time, datetime
from fastRCNN.pascal_voc import pascal_voc
print(datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))


############################
# Adjust these parameters
# to make scripts run
############################
rootDir = os.path.dirname(os.path.realpath(sys.argv[0]))
datasetName = "drone"
#directories 
procDir_1 = rootDir + "/proc1/" + datasetName + "/"
modelDir_1 = procDir_1 + "models/"
procDir_2 = rootDir + "/proc2/" + datasetName + "/"
modelDir_2 = procDir_2 + "models/"

# cntk model
cntk_nrRois_1     = 3000 
cntk_nrRois_2     = 20000     # DNN input number of ROIs per image. Zero-padded/truncated if necessary

# nn and svm training
train_posOverlapThres = 0.5      # DNN and SVM threshold for marking ROIs with significant overlap with a GT object as positive

# postprocessing
nmsThreshold = 0.1                      # Non-Maxima suppression threshold (in range [0,1])
                                        # The lower the more ROIs will be combined. Used during evaluation and visualization (scripts 5_)
vis_decisionThresholds = {'svm' : 0.5,  # Reject detections with low confidence, used only in 5_visualizeResults
                          'nn' : None}

# evaluation
evalVocOverlapThreshold = 0.5 # voc-style intersection-over-union threshold used to determine if object was found

classes_1 = ('__background__', "mast")
classes_2 = ('__background__', "insulator")
nrClasses_1 = len(classes_1)
nrClasses_2 = len(classes_2)