'''
This program is to assess the classification accuracy using ROI
Input image format: ENVI standard format
ROI (Region Of Interest): text format - using ROI tools of ENVI to extract training sites and convert it to ASCII format

'''
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt


def read_image(DataPath):
    img1 = envi.open(DataPath + '.hdr', DataPath + '.img')
    data = img1.load()
    return data


def read_ROI(ROIPath):
    # making ROI map
    f: TextIO = open(ROIPath, 'r')
    line_no = 0
    cls_no: int = 0

    for line in f:
        if 'File Dimension' in line:
            dim = line[18:]
            dim = dim.split(sep='x')
            x_dim = int(dim[0])
            y_dim = int(dim[1])
            ROI_arr = np.zeros([y_dim, x_dim])
        # line_no += 1
        # print(line_no)
        if line[0] != ';' and line[0] != '\n':
            # set_trace()
            line_no += 1
            line = " ".join(line.split())
            fields = line.split(' ')
            if int(fields[0]) == 1:
                cls_no += 1
                line_no = 1
            ROI_arr[int(fields[2]), int(fields[1])] = cls_no
    f.close()
    return ROI_arr


def confusion_matrix(map, roi):
    map[np.where(roi == 0)] = -1
    roi[np.where(roi == 0)] = -1
    no_cls_roi = int((np.unique(roi)).shape[0]) - 1
    no_cls_map = int((np.unique(map)).shape[0]) - 1
    cm = np.zeros((no_cls_roi + 1, no_cls_roi + 1), dtype=int)
    for i in range(no_cls_roi + 1):
        if i == no_cls_roi:
            cm[i, :] = np.sum(cm, axis=0)
            cm[:, i] = np.sum(cm, axis=1)
        else:
            diff = map[np.where(map == i + 1)] - roi[np.where(map == i + 1)]
            d = (np.unique(diff)).astype(int)
            for j in range(d.shape[0]):
                if d[j] == 0:
                    cm[i, i] = int((np.where(diff == d[j])[0]).shape[0])
                else:
                    cm[i, i - d[j]] = int((np.where(diff == d[j])[0]).shape[0])

    return cm


def write_CM(cm, DataPath):
    s = cm.shape
    cm_percent = np.zeros(s, dtype=float)
    ccv = 0  # correctly classified pixels
    for i in range(s[0]):
        cm_percent[i, :] = cm[i, :] * 100 / cm[s[0] - 1, :]
        if i != (int(s[0]) - 1):
            ccv = ccv + cm[i, i]
    OA = ccv * 100 / cm[s[0] - 1, s[0] - 1]  # overal accuracy
    CMfile = DataPath + '_CM_with0GrassClass.txt'
    f = open(CMfile, "w")
    f.write("Confusion Matrix: " + DataPath + '\n\n')
    f.write("Overal accuracy : ({0:d}/{1:d})    {2:,.4f}%\n\n".format(ccv, cm[s[0] - 1, s[0] - 1], OA))
    f.write("                  Ground Truth (Pixels) \n")
    if s[0] == 4:
        f.write(
            "{0:>10s} {1:>14s} {2:>14s} {3:>14s} {4:>14s} \n".format('Class', 'Soil', 'Vine', 'Shadow', 'Total'))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} \n".format('Soil', cm[0, 0], cm[0, 1], cm[0, 2],
                                                                         cm[0, 3]))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} \n".format('Vine', cm[1, 0], cm[1, 1], cm[1, 2],
                                                                         cm[1, 3]))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} \n".format('Shadow', cm[2, 0], cm[2, 1], cm[2, 2],
                                                                         cm[2, 3]))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} \n".format('Total', cm[3, 0], cm[3, 1], cm[3, 2],
                                                                         cm[3, 3]))

        f.write("\n                  Ground Truth (Percent) \n")
        f.write(
            "{0:>10s} {1:>14s} {2:>14s} {3:>14s} {4:>14s} \n".format('Class', 'Soil', 'Vine', 'Shadow', 'Total'))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} \n".format('Soil', cm_percent[0, 0],
                                                                                 cm_percent[0, 1],
                                                                                 cm_percent[0, 2],
                                                                                 cm_percent[0, 3]))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} \n".format('Vine', cm_percent[1, 0],
                                                                                 cm_percent[1, 1],
                                                                                 cm_percent[1, 2],
                                                                                 cm_percent[1, 3]))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} \n".format('Shadow', cm_percent[2, 0],
                                                                                 cm_percent[2, 1],
                                                                                 cm_percent[2, 2],
                                                                                 cm_percent[2, 3]))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} \n".format('Total', cm_percent[3, 0],
                                                                                 cm_percent[3, 1],
                                                                                 cm_percent[3, 2],
                                                                                 cm_percent[3, 3]))
    if s[0] == 5:
        f.write(
            "{0:>10s} {1:>14s} {2:>14s} {3:>14s} {4:>14s} {5:>14s} \n".format('Class', 'Soil', 'Vine', 'Shadow',
                                                                              'Grass', 'Total'))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} {5:>14d} \n".format('Soil', cm[0, 0], cm[0, 1], cm[0, 2],
                                                                                  cm[0, 3], cm[0, 4]))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} {5:>14d} \n".format('Vine', cm[1, 0], cm[1, 1], cm[1, 2],
                                                                                  cm[1, 3], cm[1, 4]))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} {5:>14d} \n".format('Shadow', cm[2, 0], cm[2, 1],
                                                                                  cm[2, 2], cm[2, 3], cm[2, 4]))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} {5:>14d} \n".format('Grass', cm[3, 0], cm[3, 1], cm[3, 2],
                                                                                  cm[3, 3], cm[3, 4]))
        f.write("{0:>10s} {1:>14d} {2:>14d} {3:>14d} {4:>14d} {5:>14d} \n".format('Total', cm[4, 0], cm[4, 1], cm[4, 2],
                                                                                  cm[4, 3], cm[4, 4]))

        f.write("\n                  Ground Truth (Percent) \n")
        f.write(
            "{0:>10s} {1:>14s} {2:>14s} {3:>14s} {4:>14s} {5:>14s}       \n".format('Class', 'Soil', 'Vine', 'Shadow',
                                                                                    'Grass', 'Total'))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} {5:>14.2f} \n".format('Soil', cm_percent[0, 0],
                                                                                            cm_percent[0, 1],
                                                                                            cm_percent[0, 2],
                                                                                            cm_percent[0, 3],
                                                                                            cm_percent[0, 4]))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} {5:>14.2f} \n".format('Vine', cm_percent[1, 0],
                                                                                            cm_percent[1, 1],
                                                                                            cm_percent[1, 2],
                                                                                            cm_percent[1, 3],
                                                                                            cm_percent[1, 4]))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} {5:>14.2f} \n".format('Shadow', cm_percent[2, 0],
                                                                                            cm_percent[2, 1],
                                                                                            cm_percent[2, 2],
                                                                                            cm_percent[2, 3],
                                                                                            cm_percent[2, 4]))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} {5:>14.2f} \n".format('Grass', cm_percent[3, 0],
                                                                                            cm_percent[3, 1],
                                                                                            cm_percent[3, 2],
                                                                                            cm_percent[3, 3],
                                                                                            cm_percent[3, 4]))
        f.write("{0:>10s} {1:>14.2f} {2:>14.2f} {3:>14.2f} {4:>14.2f} {5:>14.2f} \n".format('Total', cm_percent[4, 0],
                                                                                            cm_percent[4, 1],
                                                                                            cm_percent[4, 2],
                                                                                            cm_percent[4, 3],
                                                                                            cm_percent[4, 4]))
    f.close()


# name and address of the classification map and ROI
# OT_type = ['EMD', 'Sinkhorn', 'L1L2', 'Laplace']
OT_type = ['L1L2']
subscene = 'subset3'
ROI_name = 'class_samples_3.txt'  # Class.txt
dir_list = ['with Grass transformation using classification map']#'without Grass transformation using classification map', 'with Grass transformation using training site',
            # 'without Grass transformation using training site', ]
for direct in dir_list:
    '''
    if direct == 'without Grass transformation using training site':
        ov_t = 'ts'
    #    lambda_reg = 340000
    else:
        ov_t = 'cm'
    #    lambda_reg = 940000
    '''
    reg = 0.01000
    while reg <= 1000000:
        #for lambda_reg in range(40000, 1000000, 200000):
    # for i in OT_type:
        classification_map_name = 'L1L2_'+str(990000)+'_regCL_'+str(reg) + '_classification'
        print(classification_map_name)
        DataPath = 'Z:/Aria-data/Minervois_2016/' + subscene + '/' + direct + '/REG_CL (reg_e optimum)/' + classification_map_name
        ROIPath = 'Z:/Aria-data/Minervois_2016/' + subscene + '/' + ROI_name
        # reading the data
        map = read_image(DataPath)
        map = np.asarray(map)
        map = np.squeeze(map)
        roi = read_ROI(ROIPath)
        # Compute confusion matrix
        cm = confusion_matrix(map, roi)
        # Write in an external file
        write_CM(cm, DataPath)
        reg *= 10
plt.figure()
plt.imshow(map)
plt.show()
print('Hello world')
