
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import OrderedDict
from tools.reliability_diagrams import *

import stats as sts


class Evaluatorv2(object):
    def __init__(self, num_class,out_folder='test_out'):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8
        self.out_folder = out_folder

    def get_tp_fp_tn_fn(self):
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - np.diag(self.confusion_matrix)
        fn = self.confusion_matrix.sum(axis=1) - np.diag(self.confusion_matrix)
        tn = np.diag(self.confusion_matrix).sum() - np.diag(self.confusion_matrix)
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def Pixel_Accuracy_Class(self):
        #         TP                                  TP+FP
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=0) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image,pre_logits,pre_score,title):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self._ece(gt_image,pre_image,pre_logits,pre_score,title)

    def _ece(self, gt_image, pre_image,pre_logits,pre_score,title):
        
        # Override matplotlib default styling.
        confidence = np.max(pre_logits,axis=0) 
        score = np.max(pre_score,axis=0)

        fig = reliability_diagram(gt_image.flatten(), pre_image.flatten(), confidence.flatten(), num_bins=5, draw_ece=True,
                          draw_bin_importance="alpha", draw_averages=True,
                          title=title, figsize=(6, 6), dpi=100, 
                          return_fig=True,show=False)
        bin_data = compute_calibration(gt_image.flatten(), pre_image.flatten(), confidence.flatten(), num_bins=5)
        
        self.uncertainty['name'].append(title)
        self.uncertainty['macc'].append(bin_data['avg_accuracy'])
        self.uncertainty['mconf'].append(bin_data['avg_confidence'])
        self.uncertainty['ece'].append(bin_data['expected_calibration_error'])
        self.uncertainty['mce'].append(bin_data['max_calibration_error'])

        pos = (gt_image==pre_image)
        neg = (gt_image!=pre_image)
        self.uncertainty['avg_score_pos'].append(np.mean(score[pos].flatten()))
        self.uncertainty['avg_score_neg'].append(np.mean(score[neg].flatten()))
        self.uncertainty['var_score_pos'].append(np.var(score[pos].flatten()))
        self.uncertainty['var_score_neg'].append(np.var(score[neg].flatten()))

        with open('{}/{}.txt'.format(self.out_folder,title),'w') as f:    #设置文件对象
            f.write('{}'.format(bin_data))  
        #plt.show()
        plt.savefig('{}/{}_ece.png'.format(self.out_folder,title))
        plt.close(fig)
        plt.clf()
        plt.cla()

        for i in range(self.num_class):
            idxs = (gt_image==i)
            confidence = np.max(pre_logits,axis=0) 
            fig = reliability_diagram(gt_image[idxs].flatten(), pre_image[idxs].flatten(), confidence[idxs].flatten(), num_bins=5, draw_ece=True,
                            draw_bin_importance="alpha", draw_averages=True,
                            title='{}_{}'.format(title,i), figsize=(6, 6), dpi=100, 
                            return_fig=True,show=False)
            bin_data_sub = compute_calibration(gt_image[idxs].flatten(), pre_image[idxs].flatten(), confidence[idxs].flatten(), num_bins=5)
            self.uncertainty['macc_{}'.format(i)].append(bin_data_sub['avg_accuracy'])
            self.uncertainty['mconf_{}'.format(i)].append(bin_data_sub['avg_confidence'])
            self.uncertainty['ece_{}'.format(i)].append(bin_data_sub['expected_calibration_error'])
            self.uncertainty['mce_{}'.format(i)].append(bin_data_sub['max_calibration_error'])

            self.uncertainty['avg_score_pos_{}'.format(i)].append(np.mean(score[(pos*idxs).astype(np.bool)].flatten()))
            self.uncertainty['avg_score_neg_{}'.format(i)].append(np.mean(score[(neg*idxs).astype(np.bool)].flatten()))
            self.uncertainty['var_score_pos_{}'.format(i)].append(np.var(score[(pos*idxs).astype(np.bool)].flatten()))
            self.uncertainty['var_score_neg_{}'.format(i)].append(np.var(score[(neg*idxs).astype(np.bool)].flatten()))

            slice_score = pre_score[i,:,:].flatten()
            self.uncertainty['avg_score_{}'.format(i)].append(np.mean(slice_score))
            self.uncertainty['var_score_{}'.format(i)].append(np.var(slice_score))
            self.uncertainty['max_score_{}'.format(i)].append(np.max(slice_score))
            self.uncertainty['min_score_{}'.format(i)].append(np.min(slice_score))
            self.uncertainty['upquantile_score_{}'.format(i)].append(sts.quantile(slice_score, p=0.75))
            self.uncertainty['downquantile_score_{}'.format(i)].append(sts.quantile(slice_score, p=0.25))

            #plt.show()
            plt.savefig('{}/{}_ece_{}.png'.format(self.out_folder,title,i))
            plt.close(fig)
            plt.clf()
            plt.cla()
            
    def uncertaintytocsv(self):
        df = pd.DataFrame.from_dict(self.uncertainty)
        df.to_csv('{}/uncertainty.csv'.format(self.out_folder))
        print('write to csv files')

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

        #uncertainty
        self.uncertainty = {}
        self.uncertainty['name']=[]
        self.uncertainty['ece']=[]
        self.uncertainty['mce']=[]
        self.uncertainty['macc']=[]
        self.uncertainty['mconf']=[]

        self.uncertainty['avg_score_pos']=[]
        self.uncertainty['avg_score_neg']=[]
        self.uncertainty['var_score_pos']=[]
        self.uncertainty['var_score_neg']=[]


        for i in range(self.num_class):
            self.uncertainty['ece_{}'.format(i)]=[]
            self.uncertainty['mce_{}'.format(i)]=[]
            self.uncertainty['macc_{}'.format(i)]=[]
            self.uncertainty['mconf_{}'.format(i)]=[]

            self.uncertainty['avg_score_pos_{}'.format(i)]=[]
            self.uncertainty['avg_score_neg_{}'.format(i)]=[]
            self.uncertainty['var_score_pos_{}'.format(i)]=[]
            self.uncertainty['var_score_neg_{}'.format(i)]=[]

            self.uncertainty['avg_score_{}'.format(i)]=[]
            self.uncertainty['var_score_{}'.format(i)]=[]
            self.uncertainty['max_score_{}'.format(i)]=[]
            self.uncertainty['min_score_{}'.format(i)]=[]
            self.uncertainty['upquantile_score_{}'.format(i)]=[]
            self.uncertainty['downquantile_score_{}'.format(i)]=[]

            #sts.quantile(mylist, p=0.25)



if __name__ == '__main__':

    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 1],
                   [2, 0, 1],
                   [1, 1, 1]])

    eval = Evaluator(num_class=3)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.Precision())
    print(eval.Recall())
    print(eval.Intersection_over_Union())
    print(eval.OA())
    print(eval.F1())
    print(eval.Frequency_Weighted_Intersection_over_Union())
