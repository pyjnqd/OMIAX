import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import numpy as np
import sys

def kappa_validation(pred_path):
    label_path = "/home/xxxxxx/workspace/qingguangyan/glaucoma_grading.xlsx"
    label = pd.read_excel(label_path)
    try:
        pred = pd.read_csv(pred_path)
    except:
        raise ValueError("The CSV file of the prediction of glaucoma grading reads an exception!")

    try:
        merge = label.merge(pred, on='data', how='left')
    except:
        raise ValueError("The indexes of the CSV file of glaucoma grading "
                         "are not ‘data’, ‘non’, ‘early’, ‘mid_advanced’! OR "
                         "Please check the spelling of the image names "
                         "in the glaucoma grading results!")

    tmp = merge.iloc[:, 1:].values
    if np.isnan(tmp).any():
        raise ValueError("Please check the spelling of the image names "
                         "in the glaucoma grading results!")

    dense_label = merge.iloc[:, 1:4].values.argmax(1)
    dense_pred = merge.iloc[:, 4:].values.argmax(1)


    if len(dense_label) != len(pred):
        raise ValueError("The number of predicted samples in the glaucoma grading results "
                         "is not consistent with the number of test samples provided!")

    kappa = cohen_kappa_score(dense_label, dense_pred, weights='quadratic')
    acc = accuracy_score(dense_label, dense_pred, normalize=True) 
    matrix = confusion_matrix(dense_label, dense_pred)
    matrix_normalize = confusion_matrix(dense_label, dense_pred, normalize="true")
    acc_details = matrix_normalize.diagonal()
    return kappa, acc, matrix, acc_details

def evaluate_total(results_file):

    if os.path.exists(results_file):
        grading_kappa, grading_acc, grading_matrix, acc_details = kappa_validation(results_file)
    else:
        raise ValueError("The filename of the prediction of glaucoma grading is wrong!")
    
    print(f"--kappa    {round(grading_kappa, 5)}")
    print(f"--global acc    {round(grading_acc, 5)}")
    print(f"--detailed acc\n  -non: {round(acc_details[0], 5)}\n  -early: {round(acc_details[1], 5)}\n  -mid_advanced: {round(acc_details[2], 5)}")
    print(f"--confusion matrix")
    for row in grading_matrix:
        print(f"  {row[0]}  {row[1]}  {row[2]}")

if __name__ == '__main__':
    results_file = sys.argv[1]  # "/home/xxxxxx/workspace/qingguangyan/result/Results_0315_Med18_224.csv"
    evaluate_total(results_file)
