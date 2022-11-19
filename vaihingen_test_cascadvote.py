import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

is_gpu = False

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def img_writer(inp):
    (mask, gt, mask_id, rgb , confidmap_0, confidmap_1,prob_0,prob_1) = inp
    confid_name = mask_id + '_confid_0.png'
    cv2.imwrite(confid_name, ((confidmap_0)*255).astype(np.uint8))

    confid_name = mask_id + '_confid_1.png'
    cv2.imwrite(confid_name, ((confidmap_1)*255).astype(np.uint8))

    confid_name = mask_id + '_prob_0.png'
    cv2.imwrite(confid_name, ((prob_0)*255).astype(np.uint8))

    confid_name = mask_id + '_prob_1.png'
    cv2.imwrite(confid_name, ((prob_1)*255).astype(np.uint8))


    if rgb:
        mask_name_tif = mask_id + '_pred.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)

        mask_name_tif = mask_id + '_gt.png'
        mask_tif = label2rgb(gt)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '_pred.png'
        cv2.imwrite(mask_name_png, mask_png)

        mask_png = gt.astype(np.uint8)
        mask_name_png = mask_id + '_gt.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()

def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
        print (path+' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path+' 目录已存在')
        return False

def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    config.net.setCurrentTestStage(config.net.TestVote)
    
    args.output_path.mkdir(exist_ok=True, parents=True)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    if is_gpu:
        model.cuda(config.gpus[0])

    mkdir('{}/main/'.format(args.output_path.name))
    mkdir('{}/head_1/'.format(args.output_path.name))
    mkdir('{}/head_2/'.format(args.output_path.name))
    mkdir('{}/confid_vote/'.format(args.output_path.name))
    mkdir('{}/prob_vote/'.format(args.output_path.name))
    
    evaluator = Evaluatorv2(num_class=config.num_classes,out_folder='{}/confid_vote/'.format(args.output_path))
    evaluator.reset()

    evaluator_0 = Evaluatorv2(num_class=config.num_classes,out_folder='{}/head_1/'.format(args.output_path))
    evaluator_0.reset()

    evaluator_1 = Evaluatorv2(num_class=config.num_classes,out_folder='{}/head_2/'.format(args.output_path))
    evaluator_1.reset()

    evaluator_main = Evaluatorv2(num_class=config.num_classes,out_folder='{}/main/'.format(args.output_path))
    evaluator_main.reset()

    evaluator_prob_vote = Evaluatorv2(num_class=config.num_classes,out_folder='{}/prob_vote/'.format(args.output_path))
    evaluator_prob_vote.reset()
    
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            if is_gpu:
                vote_logits, main_logits, score_0,factor_0, score_1,factor_1,prob_logits,prob_factor0,prob_factor1= model(input['img'].cuda(config.gpus[0]))
            else:
                vote_logits, main_logits, score_0,factor_0, score_1,factor_1,prob_logits,prob_factor0,prob_factor1 = model(input['img'])

            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']

            softmax_vote_logits = nn.Softmax(dim=1)(vote_logits)
            predictions = softmax_vote_logits.argmax(dim=1)

            softmax_score_0 = nn.Softmax(dim=1)(score_0)
            predictions_0 = softmax_score_0.argmax(dim=1)

            softmax_score_1 = nn.Softmax(dim=1)(score_1)
            predictions_1 = softmax_score_1.argmax(dim=1)

            softmax_main_logits = nn.Softmax(dim=1)(main_logits)
            predictions_main = softmax_main_logits.argmax(dim=1)

            softmax_prob_logits = nn.Softmax(dim=1)(prob_logits)
            predictions_prob = softmax_prob_logits.argmax(dim=1)

            for i in range(softmax_vote_logits.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, 
                        gt_image=masks_true[i].cpu().numpy(),
                        pre_logits=softmax_vote_logits[i].cpu().numpy(),
                        pre_score=vote_logits[i].cpu().numpy(),
                        title=image_ids[i])
                
                mask = predictions_0[i].cpu().numpy()
                evaluator_0.add_batch(pre_image=mask, 
                        gt_image=masks_true[i].cpu().numpy(),
                        pre_logits=softmax_score_0[i].cpu().numpy(),
                        pre_score=score_0[i].cpu().numpy(),
                        title=image_ids[i])

                mask = predictions_1[i].cpu().numpy()
                evaluator_1.add_batch(pre_image=mask, 
                        gt_image=masks_true[i].cpu().numpy(),
                        pre_logits=softmax_score_1[i].cpu().numpy(),
                        pre_score=score_1[i].cpu().numpy(),
                        title=image_ids[i])
                
                mask = predictions_main[i].cpu().numpy()
                evaluator_main.add_batch(pre_image=mask, 
                        gt_image=masks_true[i].cpu().numpy(),
                        pre_logits=softmax_main_logits[i].cpu().numpy(),
                        pre_score=main_logits[i].cpu().numpy(),
                        title=image_ids[i])

                mask = predictions_prob[i].cpu().numpy()
                evaluator_prob_vote.add_batch(pre_image=mask, 
                        gt_image=masks_true[i].cpu().numpy(),
                        pre_logits=softmax_prob_logits[i].cpu().numpy(),
                        pre_score=prob_logits[i].cpu().numpy(),
                        title=image_ids[i])

                pre_mask = predictions[i].cpu().numpy()
                gt=masks_true[i].cpu().numpy()
                mask_name = image_ids[i]
                cond1 = factor_0[i].squeeze(0).cpu().numpy()
                cond2 = factor_1[i].squeeze(0).cpu().numpy()
                prob1 = prob_factor0[i].squeeze(0).cpu().numpy()
                prob2 = prob_factor1[i].squeeze(0).cpu().numpy()

                ex = cond1-cond2
                results.append((pre_mask,gt, str(args.output_path / mask_name), args.rgb,cond1,cond2,prob1,prob2))
        
        evaluator.uncertaintytocsv()
        evaluator_main.uncertaintytocsv()
        evaluator_0.uncertaintytocsv()
        evaluator_1.uncertaintytocsv()
        evaluator_prob_vote.uncertaintytocsv()
        record(evaluator,config,results,"evaluator_confid_vote")
        record(evaluator_main,config,results,"evaluator_main")
        record(evaluator_0,config,results,"evaluator_0")
        record(evaluator_1,config,results,"evaluator_1")
        record(evaluator_prob_vote,config,results,"evaluator_prob_vote")

def record(evaluator,config,results,title=""):
    print("================{}===================".format(title))
    
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))
    


if __name__ == "__main__":
    main()
