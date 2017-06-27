# -*- coding: utf-8 -*-
import sys, os, importlib, random, json
from helpers_cntk import *
import PARAMETERSall
import SLIDINGWINDOWPARAMS
locals().update(importlib.import_module("PARAMETERSall").__dict__)
locals().update(importlib.import_module("SLIDINGWINDOWPARAMS").__dict__)

####################################
# Parameters
####################################
imgPath = rootDir + "/data/" + datasetName + "/testImages/" + "1000093223.jpg"

#choose which classifier to use
classifier = 'nn'

# no need to change these parameters
boAddSelectiveSearchROIs = True
boAddGridROIs = True
boFilterROIs = True
boUseNonMaximaSurpression = True


####################################
# Main
####################################
random.seed(0)

# load cntk model
print("Loading DNN..")
tstart = datetime.datetime.now()
model_path_1 = os.path.join(modelDir_1, "frcn_" + classifier + ".model")
model_1 = load_model(model_path_1)
model_path_2 = os.path.join(modelDir_2, "frcn_" + classifier + ".model")
model_2 = load_model(model_path_2)
print("Time loading DNN [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# read the image
img = imread(imgPath)

# compute ROIs
tstart = datetime.datetime.now()
gridRois_1 = get_grid_rois(imW, imH, stepsize/2, rectangles)
if len(gridRois_1) >= cntk_nrRois_1:
    gridRois_1 = gridRois_1[:cntk_nrRois_1]
else:
    gridRois_1 = gridRois_1 + [[0,0,1,1]]*(cntk_nrRois_1-len(gridRois_1))
print("Time roi computation [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# prepare DNN inputs
tstart = datetime.datetime.now()
_, _, roisCntk_1 = getCntkInputs(imgPath, gridRois_1, None, train_posOverlapThres, nrClasses_1, cntk_nrRois_1, imW, imH)
arguments_1 = {
    model_1.arguments[0]: [np.ascontiguousarray(np.array(img, dtype=np.float32).transpose(2, 0, 1))], # convert to CNTK's HWC format
    model_1.arguments[1]: [np.array(roisCntk_1, np.float32)]
}
print("Time cnkt input generation [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# run DNN model
print("Running model..")
tstart = datetime.datetime.now()
dnnOutputs_1 = model_1.eval(arguments_1)[0][0]
dnnOutputs_1 = dnnOutputs_1[:len(gridRois_1)] 
print("Time running model [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# score all ROIs
tstart = datetime.datetime.now()
labels_1, scores_1 = scoreRois(classifier, dnnOutputs_1, 1, 1, 1, len(classes_1),
                           decisionThreshold = vis_decisionThresholds[classifier])
print("Time making prediction [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# perform non-maxima surpression
tstart = datetime.datetime.now()
nmsKeepIndices_1 = []
if boUseNonMaximaSurpression:
    nmsKeepIndices_1 = applyNonMaximaSuppression(nmsThreshold, labels_1, scores_1, gridRois_1)
    print("Non-maxima surpression kept {:4} of {:4} rois (nmsThreshold={})".format(
        len(nmsKeepIndices_1), len(labels_1), nmsThreshold))
print("Time non-maxima surpression [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

print([scores_1[x] for x in nmsKeepIndices_1])
print([labels_1[x] for x in nmsKeepIndices_1])
print([gridRois_1[x] for x in nmsKeepIndices_1])

print(sum(labels_1))

#NMS transfer
transfer_index = [x for x in nmsKeepIndices_1 if labels_1[x]>0]
transfer_boxes = [gridRois_1[x] for x in transfer_index]

print(transfer_boxes)

# compute ROIs
tstart = datetime.datetime.now()
if len(transfer_boxes) > 0:
    gridRois_2=get_grid_rois_inbox(transfer_boxes, ratios, stepsize_ratio/2, imW, imH)
else:
    gridRois_2 = get_grid_rois(imW, imH, stepsize/2, small_rectangles)
if len(gridRois_2) >= cntk_nrRois_2:
    gridRois_2 = gridRois_2[:cntk_nrRois_2]
else:
    gridRois_2 = gridRois_2 + [[0,0,1,1]]*(cntk_nrRois_2-len(gridRois_2))
print("Time roi computation [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))


# prepare DNN inputs
tstart = datetime.datetime.now()
_, _, roisCntk_2 = getCntkInputs(imgPath, gridRois_2, None, train_posOverlapThres, nrClasses_2, cntk_nrRois_2, imW, imH)
arguments_2 = {
    model_2.arguments[0]: [np.ascontiguousarray(np.array(img, dtype=np.float32).transpose(2, 0, 1))], # convert to CNTK's HWC format
    model_2.arguments[1]: [np.array(roisCntk_2, np.float32)]
}
print("Time cnkt input generation [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# run DNN model
print("Running model..")
tstart = datetime.datetime.now()
dnnOutputs_2 = model_2.eval(arguments_2)[0][0]
dnnOutputs_2 = dnnOutputs_2[:len(gridRois_2)] 
print("Time running model [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# score all ROIs
tstart = datetime.datetime.now()
labels_2, scores_2 = scoreRois(classifier, dnnOutputs_2, 1, 1, 1, len(classes_2),
                           decisionThreshold = vis_decisionThresholds[classifier])
print("Time making prediction [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

# perform non-maxima surpression
tstart = datetime.datetime.now()
nmsKeepIndices_2 = []
if boUseNonMaximaSurpression:
    nmsKeepIndices_2 = applyNonMaximaSuppression(nmsThreshold, labels_2, scores_2, gridRois_2)
    print("Non-maxima surpression kept {:4} of {:4} rois (nmsThreshold={})".format(
        len(nmsKeepIndices_2), len(labels_2), nmsThreshold))
print("Time non-maxima surpression [ms]: " + str((datetime.datetime.now() - tstart).total_seconds() * 1000))

#print([scores_2[x] for x in nmsKeepIndices_2])
#print([labels_2[x] for x in nmsKeepIndices_2])
  

# visualize results
#imgDebug = visualizeResults(imgPath, labels_2, scores_2, gridRois_2, classes_2, nmsKeepIndices_2,
                            boDrawNegativeRois=False, boDrawNmsRejectedRois=False)
#imshow(imgDebug, waitDuration=0, maxDim=500)


#create json-encoded string of all detections
outDict = [{"label": str(l), "score": str(s), "nms": str(False), "left": str(r[0]), "top": str(r[1]), "right": str(r[2]), "bottom": str(r[3])} for l,s, r in zip(labels_2, scores_2, gridRois_2)]
for i in nmsKeepIndices_2:
    outDict[i]["nms"] = str(True)
outJsonString = json.dumps(outDict)
#print("Json-encoded detections: " + outJsonString[:500] + '...')

print("DONE.")


