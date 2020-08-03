"""
Use this script to evaluate your model. It stores metrics in the file
`scores.txt` and the more detailed `results.json` in the current directory.

Based on the official CityScapes evaluation script:
    https://github.com/mcordts/cityscapesScripts

- Assumes dataset and results to be in default location. Alternatively, specify
paths to predictions and minicity folder as optional arguments.
- Assumes predictions to have the same file names as the inputs.

Usage:
    evaluate.py --results <predictions> --cityscapes <dataset>
"""
# python imports
from __future__ import print_function, absolute_import, division
from PIL import Image
import os, sys
import platform
import fnmatch
import math
import numpy as np
import glob
import argparse

# Cityscapes imports
from helpers.csHelpers import printError, getColorEntry, \
getCsFileInfo, colors, ensurePath, writeDict2JSON, writeDict2Txt
from helpers.labels import labels, category2labels, id2label

parser = argparse.ArgumentParser(description='VIPriors Segmentation evaluation tool')
parser.add_argument('--results', metavar='path/to/predictions', default='results',
                    help='path to predictions')
parser.add_argument('--minicity', metavar='path/to/dataset', default='minicity',
                    help='path to dataset root (ends with /minicity)')

def getPrediction( args, groundTruthFile ):
    # determine the prediction path, if the method is first called
    if not pargs.results:
        rootPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'results')
        if not os.path.isdir(rootPath):
            printError('Could not find a result root folder.')
        pargs.results = rootPath

    # walk the prediction path, if not happened yet
    if not args.predictionWalk:
        walk = []
        for root, dirnames, filenames in os.walk(pargs.results):
            walk.append( (root,filenames) )
        args.predictionWalk = walk

    csFile = getCsFileInfo(groundTruthFile)
    filePattern = '{}_{}_{}*.png'.format( csFile.city , csFile.sequenceNb , csFile.frameNb )

    predictionFile = None
    for root, filenames in args.predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)
            else:
                printError('Found multiple predictions for ground truth {}'.format(groundTruthFile))

    if not predictionFile:
        printError('Found no prediction for ground truth {}'.format(groundTruthFile))

    return predictionFile


######################
# Parameters
######################


# A dummy class to collect all bunch of data
class CArgs(object):
    pass
# And a global object of that class
args = CArgs()

# Define directories
args.exportFile = 'results.json'

# Remaining params
args.evalInstLevelScore = True
args.evalPixelAccuracy  = True
args.evalLabels         = []
args.printRow           = 5
args.normalized         = True
args.colorized          = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty() and platform.system()=='Linux'
args.bold               = colors.BOLD if args.colorized else ''
args.nocol              = colors.ENDC if args.colorized else ''
args.JSONOutput         = True
args.quiet              = False

args.avgClassSize       = {
    'bicycle'    :  4672.3249222261 ,
    'caravan'    : 36771.8241758242 ,
    'motorcycle' :  6298.7200839748 ,
    'rider'      :  3930.4788056518 ,
    'bus'        : 35732.1511111111 ,
    'train'      : 67583.7075812274 ,
    'car'        : 12794.0202738185 ,
    'person'     :  3462.4756337644 ,
    'truck'      : 27855.1264367816 ,
    'trailer'    : 16926.9763313609 ,
}

# value is filled when the method getPrediction is first called
args.predictionWalk = None


#########################
# Methods
#########################


# Generate empty confusion matrix and create list of relevant labels
def generateMatrix(args):
    args.evalLabels = []
    for label in labels:
        if (label.id < 0):
            continue
        # we append all found labels, regardless of being ignored
        args.evalLabels.append(label.id)
    maxId = max(args.evalLabels)
    # We use longlong type to be sure that there are no overflows
    return np.zeros(shape=(maxId+1, maxId+1),dtype=np.ulonglong)

def generateInstanceStats(args):
    instanceStats = {}
    instanceStats['classes'   ] = {}
    instanceStats['categories'] = {}
    for label in labels:
        if label.hasInstances and not label.ignoreInEval:
            instanceStats['classes'][label.name] = {}
            instanceStats['classes'][label.name]['tp'] = 0.0
            instanceStats['classes'][label.name]['tpWeighted'] = 0.0
            instanceStats['classes'][label.name]['fn'] = 0.0
            instanceStats['classes'][label.name]['fnWeighted'] = 0.0
    for category in category2labels:
        labelIds = []
        allInstances = True
        for label in category2labels[category]:
            if label.id < 0:
                continue
            if not label.hasInstances:
                allInstances = False
                break
            labelIds.append(label.id)
        if not allInstances:
            continue

        instanceStats['categories'][category] = {}
        instanceStats['categories'][category]['tp'] = 0.0
        instanceStats['categories'][category]['tpWeighted'] = 0.0
        instanceStats['categories'][category]['fn'] = 0.0
        instanceStats['categories'][category]['fnWeighted'] = 0.0
        instanceStats['categories'][category]['labelIds'] = labelIds

    return instanceStats


# Get absolute or normalized value from field in confusion matrix.
def getMatrixFieldValue(confMatrix, i, j, args):
    if args.normalized:
        rowSum = confMatrix[i].sum()
        if (rowSum == 0):
            return float('nan')
        return float(confMatrix[i][j]) / rowSum
    else:
        return confMatrix[i][j]

# Calculate and return IOU score for a particular label
def getIouScoreForLabel(label, confMatrix, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    # the number of true positive pixels for this label
    # the entry on the diagonal of the confusion matrix
    tp = np.longlong(confMatrix[label,label])

    # the number of false negative pixels for this label
    # the row sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = np.longlong(confMatrix[label,:].sum()) - tp

    # the number of false positive pixels for this labels
    # Only pixels that are not on a pixel with ground truth label that is ignored
    # The column sum of the corresponding column in the confusion matrix
    # without the ignored rows and without the actual label of interest
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular label
def getInstanceIouScoreForLabel(label, confMatrix, instStats, args):
    if id2label[label].ignoreInEval:
        return float('nan')

    labelName = id2label[label].name
    if not labelName in instStats['classes']:
        return float('nan')

    tp = instStats['classes'][labelName]['tpWeighted']
    fn = instStats['classes'][labelName]['fnWeighted']
    # false postives computed as above
    notIgnored = [l for l in args.evalLabels if not id2label[l].ignoreInEval and not l==label]
    fp = np.longlong(confMatrix[notIgnored,label].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate prior for a particular class id.
def getPrior(label, confMatrix):
    return float(confMatrix[label,:].sum()) / confMatrix.sum()

# Get average of scores.
# Only computes the average over valid entries.
def getScoreAverage(scoreList, args):
    validScores = 0
    scoreSum    = 0.0
    for score in scoreList:
        if not math.isnan(scoreList[score]):
            validScores += 1
            scoreSum += scoreList[score]
    if validScores == 0:
        return float('nan')
    return scoreSum / validScores

# Calculate and return IOU score for a particular category
def getIouScoreForCategory(category, confMatrix, args):
    # All labels in this category
    labels = category2labels[category]
    # The IDs of all valid labels in this category
    labelIds = [label.id for label in labels if not label.ignoreInEval and label.id in args.evalLabels]
    # If there are no valid labels, then return NaN
    if not labelIds:
        return float('nan')

    # the number of true positive pixels for this category
    # this is the sum of all entries in the confusion matrix
    # where row and column belong to a label ID of this category
    tp = np.longlong(confMatrix[labelIds,:][:,labelIds].sum())

    # the number of false negative pixels for this category
    # that is the sum of all rows of labels within this category
    # minus the number of true positive pixels
    fn = np.longlong(confMatrix[labelIds,:].sum()) - tp

    # the number of false positive pixels for this category
    # we count the column sum of all labels within this category
    # while skipping the rows of ignored labels and of labels within this category
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom

# Calculate and return IOU score for a particular category
def getInstanceIouScoreForCategory(category, confMatrix, instStats, args):
    if not category in instStats['categories']:
        return float('nan')
    labelIds = instStats['categories'][category]['labelIds']

    tp = instStats['categories'][category]['tpWeighted']
    fn = instStats['categories'][category]['fnWeighted']

    # the number of false positive pixels for this category
    # same as above
    notIgnoredAndNotInCategory = [l for l in args.evalLabels if not id2label[l].ignoreInEval and id2label[l].category != category]
    fp = np.longlong(confMatrix[notIgnoredAndNotInCategory,:][:,labelIds].sum())

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')

    # return IOU
    return float(tp) / denom


# create a dictionary containing all relevant results
def createResultDict( confMatrix, classScores, classInstScores, categoryScores, categoryInstScores, perImageStats, args ):
    # write JSON result file
    wholeData = {}
    wholeData['confMatrix'] = confMatrix.tolist()
    wholeData['priors'] = {}
    wholeData['labels'] = {}
    for label in args.evalLabels:
        wholeData['priors'][id2label[label].name] = getPrior(label, confMatrix)
        wholeData['labels'][id2label[label].name] = label
    wholeData['classScores'] = classScores
    wholeData['classInstScores'] = classInstScores
    wholeData['categoryScores'] = categoryScores
    wholeData['categoryInstScores'] = categoryInstScores
    wholeData['averageScoreClasses'] = getScoreAverage(classScores, args)
    wholeData['averageScoreInstClasses'] = getScoreAverage(classInstScores, args)
    wholeData['averageScoreCategories'] = getScoreAverage(categoryScores, args)
    wholeData['averageScoreInstCategories'] = getScoreAverage(categoryInstScores, args)
    wholeData['accuracy'] = np.trace(confMatrix) / np.sum(confMatrix)

    if perImageStats:
        wholeData['perImageScores'] = perImageStats

    return wholeData

def writeJSONFile(wholeData, args):
    path = os.path.dirname(args.exportFile)
    ensurePath(path)
    writeDict2JSON(wholeData, args.exportFile)

# Print confusion matrix
def printConfMatrix(confMatrix, args):
    # print line
    print('\b{text:{fill}>{width}}'.format(width=15, fill='-', text=' '), end=' ')
    for label in args.evalLabels:
        print('\b{text:{fill}>{width}}'.format(width=args.printRow + 2, fill='-', text=' '), end=' ')
    print('\b{text:{fill}>{width}}'.format(width=args.printRow + 3, fill='-', text=' '))

    # print label names
    print('\b{text:>{width}} |'.format(width=13, text=''), end=' ')
    for label in args.evalLabels:
        print('\b{text:^{width}} |'.format(width=args.printRow, text=id2label[label].name[0]), end=' ')
    print('\b{text:>{width}} |'.format(width=6, text='Prior'))

    # print line
    print('\b{text:{fill}>{width}}'.format(width=15, fill='-', text=' '), end=' ')
    for label in args.evalLabels:
        print('\b{text:{fill}>{width}}'.format(width=args.printRow + 2, fill='-', text=' '), end=' ')
    print('\b{text:{fill}>{width}}'.format(width=args.printRow + 3, fill='-', text=' '))

    # print matrix
    for x in range(0, confMatrix.shape[0]):
        if (not x in args.evalLabels):
            continue
        # get prior of this label
        prior = getPrior(x, confMatrix)
        # skip if label does not exist in ground truth
        if prior < 1e-9:
            continue

        # print name
        name = id2label[x].name
        if len(name) > 13:
            name = name[:13]
        print('\b{text:>{width}} |'.format(width=13,text=name), end=' ')
        # print matrix content
        for y in range(0, len(confMatrix[x])):
            if (not y in args.evalLabels):
                continue
            matrixFieldValue = getMatrixFieldValue(confMatrix, x, y, args)
            print(getColorEntry(matrixFieldValue, args) + '\b{text:>{width}.2f}  '.format(width=args.printRow, text=matrixFieldValue) + args.nocol, end=' ')
        # print prior
        print(getColorEntry(prior, args) + '\b{text:>{width}.4f} '.format(width=6, text=prior) + args.nocol)
    # print line
    print('\b{text:{fill}>{width}}'.format(width=15, fill='-', text=' '), end=' ')
    for label in args.evalLabels:
        print('\b{text:{fill}>{width}}'.format(width=args.printRow + 2, fill='-', text=' '), end=' ')
    print('\b{text:{fill}>{width}}'.format(width=args.printRow + 3, fill='-', text=' '), end=' ')

# Print intersection-over-union scores for all classes.
def printClassScores(scoreList, instScoreList, args):
    if (args.quiet):
        return
    print(args.bold + 'classes          IoU      nIoU' + args.nocol)
    print('--------------------------------')
    for label in args.evalLabels:
        if (id2label[label].ignoreInEval):
            continue
        labelName = str(id2label[label].name)
        iouStr = getColorEntry(scoreList[labelName], args) + '{val:>5.3f}'.format(val=scoreList[labelName]) + args.nocol
        niouStr = getColorEntry(instScoreList[labelName], args) + '{val:>5.3f}'.format(val=instScoreList[labelName]) + args.nocol
        print('{:<14}: '.format(labelName) + iouStr + '    ' + niouStr)

# Print intersection-over-union scores for all categorys.
def printCategoryScores(scoreDict, instScoreDict, args):
    if (args.quiet):
        return
    print(args.bold + 'categories       IoU      nIoU' + args.nocol)
    print('--------------------------------')
    for categoryName in scoreDict:
        if all( label.ignoreInEval for label in category2labels[categoryName] ):
            continue
        iouStr  = getColorEntry(scoreDict[categoryName], args) + '{val:>5.3f}'.format(val=scoreDict[categoryName]) + args.nocol
        niouStr = getColorEntry(instScoreDict[categoryName], args) + '{val:>5.3f}'.format(val=instScoreDict[categoryName]) + args.nocol
        print('{:<14}: '.format(categoryName) + iouStr + '    ' + niouStr)

# Evaluate image lists pairwise.
def evaluateImgLists(predictionImgList, groundTruthImgList, args):
    if len(predictionImgList) != len(groundTruthImgList):
        printError('List of images for prediction and groundtruth are not of equal size.')
    confMatrix    = generateMatrix(args)
    instStats     = generateInstanceStats(args)
    perImageStats = {}
    nbPixels      = 0

    if not args.quiet:
        print('Evaluating {} pairs of images...'.format(len(predictionImgList)))

    # Evaluate all pairs of images and save them into a matrix
    for i in range(len(predictionImgList)):
        predictionImgFileName = predictionImgList[i]
        groundTruthImgFileName = groundTruthImgList[i]
        #print 'Evaluate ', predictionImgFileName, '<>', groundTruthImgFileName
        nbPixels += evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instStats, perImageStats, args)

        # sanity check
        if confMatrix.sum() != nbPixels:
            printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

        if not args.quiet:
            print('\rImages Processed: {}'.format(i+1), end=' ')
            sys.stdout.flush()
    if not args.quiet:
        print('\n')

    # sanity check
    if confMatrix.sum() != nbPixels:
        printError('Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(confMatrix.sum(),nbPixels))

    # print confusion matrix
    if (not args.quiet):
        printConfMatrix(confMatrix, args)

    # Calculate IOU scores on class level from matrix
    classScoreList = {}
    for label in args.evalLabels:
        labelName = id2label[label].name
        classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)

    # Calculate instance IOU scores on class level from matrix
    classInstScoreList = {}
    for label in args.evalLabels:
        labelName = id2label[label].name
        classInstScoreList[labelName] = getInstanceIouScoreForLabel(label, confMatrix, instStats, args)

    # Print IOU scores
    if (not args.quiet):
        print('')
        print('')
        printClassScores(classScoreList, classInstScoreList, args)
        iouAvgStr  = getColorEntry(getScoreAverage(classScoreList, args), args) + '{avg:5.3f}'.format(avg=getScoreAverage(classScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(classInstScoreList , args), args) + '{avg:5.3f}'.format(avg=getScoreAverage(classInstScoreList , args)) + args.nocol
        print('--------------------------------')
        print('Score Average : ' + iouAvgStr + '    ' + niouAvgStr)
        print('--------------------------------')
        print('')

    # Calculate IOU scores on category level from matrix
    categoryScoreList = {}
    for category in category2labels.keys():
        categoryScoreList[category] = getIouScoreForCategory(category,confMatrix,args)

    # Calculate instance IOU scores on category level from matrix
    categoryInstScoreList = {}
    for category in category2labels.keys():
        categoryInstScoreList[category] = getInstanceIouScoreForCategory(category,confMatrix,instStats,args)

    # Print IOU scores
    if (not args.quiet):
        print('')
        printCategoryScores(categoryScoreList, categoryInstScoreList, args)
        iouAvgStr = getColorEntry(getScoreAverage(categoryScoreList, args), args) + '{avg:5.3f}'.format(avg=getScoreAverage(categoryScoreList, args)) + args.nocol
        niouAvgStr = getColorEntry(getScoreAverage(categoryInstScoreList, args), args) + '{avg:5.3f}'.format(avg=getScoreAverage(categoryInstScoreList, args)) + args.nocol
        print('--------------------------------')
        print('Score Average : ' + iouAvgStr + '    ' + niouAvgStr)
        print('--------------------------------')
        print('')

    allResultsDict = createResultDict( confMatrix, classScoreList, classInstScoreList, categoryScoreList, categoryInstScoreList, perImageStats, args )
    # write result file
    if args.JSONOutput:
        writeJSONFile( allResultsDict, args)
        
    writeDict2Txt(allResultsDict, 'results.txt')

    # return confusion matrix
    return allResultsDict

# Main evaluation method. Evaluates pairs of prediction and ground truth
# images which are passed as arguments.
def evaluatePair(predictionImgFileName, groundTruthImgFileName, confMatrix, instanceStats, perImageStats, args):
    # Loading all resources for evaluation.
    try:
        predictionImg = Image.open(predictionImgFileName)
        predictionNp  = np.array(predictionImg)
    except:
        printError('Unable to load ' + predictionImgFileName)
    try:
        groundTruthImg = Image.open(groundTruthImgFileName)
        groundTruthNp = np.array(groundTruthImg)
    except:
        printError('Unable to load ' + groundTruthImgFileName)
    # load ground truth instances, if needed
    if args.evalInstLevelScore:
        groundTruthInstanceImgFileName = groundTruthImgFileName.replace('labelIds','instanceIds')
        try:
            instanceImg = Image.open(groundTruthInstanceImgFileName)
            instanceNp  = np.array(instanceImg)
        except:
            printError('Unable to load ' + groundTruthInstanceImgFileName)

    # Check for equal image sizes
    if (predictionImg.size[0] != groundTruthImg.size[0]):
        printError('Image widths of ' + predictionImgFileName + ' and ' + groundTruthImgFileName + ' are not equal.')
    if (predictionImg.size[1] != groundTruthImg.size[1]):
        printError('Image heights of ' + predictionImgFileName + ' and ' + groundTruthImgFileName + ' are not equal.')
    if ( len(predictionNp.shape) != 2 ):
        printError('Predicted image has multiple channels.')

    imgWidth  = predictionImg.size[0]
    imgHeight = predictionImg.size[1]
    nbPixels  = imgWidth*imgHeight

    # Evaluate images
    encoding_value = max(groundTruthNp.max(), predictionNp.max()).astype(np.int32) + 1
    encoded = (groundTruthNp.astype(np.int32) * encoding_value) + predictionNp

    values, cnt = np.unique(encoded, return_counts=True)

    for value, c in zip(values, cnt):
        pred_id = value % encoding_value
        gt_id = int((value - pred_id)/encoding_value)
        if not gt_id in args.evalLabels:
            printError('Unknown label with id {:}'.format(gt_id))
        confMatrix[gt_id][pred_id] += c
        

    if args.evalInstLevelScore:
        # Generate category masks
        categoryMasks = {}
        for category in instanceStats['categories']:
            categoryMasks[category] = np.in1d( predictionNp , instanceStats['categories'][category]['labelIds'] ).reshape(predictionNp.shape)

        instList = np.unique(instanceNp[instanceNp > 1000])
        for instId in instList:
            labelId = int(instId/1000)
            label = id2label[ labelId ]
            if label.ignoreInEval:
                continue

            mask = instanceNp==instId
            instSize = np.count_nonzero( mask )

            tp = np.count_nonzero( predictionNp[mask] == labelId )
            fn = instSize - tp

            weight = args.avgClassSize[label.name] / float(instSize)
            tpWeighted = float(tp) * weight
            fnWeighted = float(fn) * weight

            instanceStats['classes'][label.name]['tp']         += tp
            instanceStats['classes'][label.name]['fn']         += fn
            instanceStats['classes'][label.name]['tpWeighted'] += tpWeighted
            instanceStats['classes'][label.name]['fnWeighted'] += fnWeighted

            category = label.category
            if category in instanceStats['categories']:
                catTp = 0
                catTp = np.count_nonzero( np.logical_and( mask , categoryMasks[category] ) )
                catFn = instSize - catTp

                catTpWeighted = float(catTp) * weight
                catFnWeighted = float(catFn) * weight

                instanceStats['categories'][category]['tp']         += catTp
                instanceStats['categories'][category]['fn']         += catFn
                instanceStats['categories'][category]['tpWeighted'] += catTpWeighted
                instanceStats['categories'][category]['fnWeighted'] += catFnWeighted

    if args.evalPixelAccuracy:
        notIgnoredLabels = [l for l in args.evalLabels if not id2label[l].ignoreInEval]
        notIgnoredPixels = np.in1d( groundTruthNp , notIgnoredLabels , invert=True ).reshape(groundTruthNp.shape)
        erroneousPixels = np.logical_and( notIgnoredPixels , ( predictionNp != groundTruthNp ) )
        perImageStats[predictionImgFileName] = {}
        perImageStats[predictionImgFileName]['nbNotIgnoredPixels'] = np.count_nonzero(notIgnoredPixels)
        perImageStats[predictionImgFileName]['nbCorrectPixels']    = np.count_nonzero(erroneousPixels)

    return nbPixels

# The main method
def main():
    global args
    global pargs
    
    # Parse optional arguments
    pargs = parser.parse_args()
    # Parameters that should be modified by user
    args.groundTruthSearch  = os.path.join(pargs.minicity , 'gtFine' , 'val' , '*_gtFine_labelIds.png')

    predictionImgList = []
    groundTruthImgList = []

    # use the ground truth search string specified above
    groundTruthImgList = glob.glob(args.groundTruthSearch)
    if not groundTruthImgList:
        printError('Cannot find any ground truth images to use for evaluation. Searched for: {}'.format(args.groundTruthSearch))
    # get the corresponding prediction for each ground truth imag
    for gt in groundTruthImgList:
        predictionImgList.append( getPrediction(args,gt) )

    # evaluate
    evaluateImgLists(predictionImgList, groundTruthImgList, args)

    return

# call the main method
if __name__ == '__main__':
    main()
