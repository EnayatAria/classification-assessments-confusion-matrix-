# classification-assessments-confusion-matrix

This code assesses a classification map by usiung test datasets, and produces oevrall accuracy, confusion matrix and confusion matrix in precent in a text file.

This code is written  for 4 main classes of vineyards: soil, vine, shadow, and grass

Inputs:

- DataPath: of the classification map
- ROIPath: the test samples for evaluating the classification map (it is a ASCII file obtanied by saving a ROI to ASCII in envi)

Output:
 - Confusion matrix including the overall accuracy and confusion matrix in percent 


A sample output result is uploaded named:  subset_3_classification_CM_with0GrassClass

There is also an IDL code for classification assessment named classification_accuracy.pro.This code using ENVI functions in batch mode.  
