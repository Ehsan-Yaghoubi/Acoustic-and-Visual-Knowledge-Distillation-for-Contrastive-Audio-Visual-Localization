import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
import numpy as np
import argparse
from model_cnn import EZVSL
from datasets import get_test_dataset, inverse_normalize
import cv2
import random
from statistics import mean, stdev

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='vggsound_144k_run1', help='experiment name (experiment folder set to "args.model_dir/args.experiment_name)"')
    parser.add_argument('--save_visualizations', action='store_false', help='Set to store all VSL visualizations (saved in viz directory within experiment folder)')

    # Dataset
    #parser.add_argument('--testset', default='flickr', type=str, help='testset (flickr or vggss)')
    #parser.add_argument('--test_data_path', default='/data2/dataset/labeled_5k_flicker/Data/', type=str, help='Root directory path of data')
    #parser.add_argument('--test_gt_path', default='/data2/dataset/labeled_5k_flicker/Annotations/', type=str)
    parser.add_argument('--testset', default='vggss', type=str, help='testset (flickr or vggss)')
    parser.add_argument('--test_data_path', default='/data2/dataset/vggss/vggss_dataset_different_naming/', type=str, help='Root directory path of data')
    parser.add_argument('--test_gt_path', default='/data2/dataset/vggss/', type=str)
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')

    # Model
    parser.add_argument('--tau', default=0.03, type=float, help='tau')
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--alpha', default=0.4, type=float, help='alpha')

    # Distributed params
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')
    parser.add_argument("--seed", type=list, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], help="random seed")

    return parser.parse_args()


def main(args):

    AV_best_mAP_all = []
    AV_best_cIoU_all = []
    AV_best_Auc_all = []

    Resnet18_best_mAP_all = []
    Resnet18_best_cIoU_all = []
    Resnet18_best_Auc_all = []

    detr_best_mAP_all = []
    detr_best_cIoU_all = []
    detr_best_Auc_all = []

    AVO_best_mAP_all = []
    AVO_best_cIoU_all = []
    AVO_best_Auc_all = []

    for test_round, seed_num in enumerate(args.seed):
        """
        We use seed_num for reproducibility of the results. 
        Note: the up-sampling operation that is used in our codes
        is not reproducible,so you may not reach the exact 
        same results each time, even when using the same seeds.
        """
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(seed_num)
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        print(">>>>>> We will test the model for {} times with seeds {}. Then the average values for cIoU and AUC will be calculated.".format(len(args.seed), args.seed))
        print(">>>>>> Testing seed for this round of test is {}.".format(seed_num))

        # Model dir
        model_dir = os.path.join(args.model_dir, args.experiment_name)
        viz_dir = os.path.join(model_dir, "viz_seed_"+str(seed_num))
        os.makedirs(viz_dir, exist_ok=True)

        # Models
        audio_visual_model = EZVSL(args.tau, args.out_dim)

        from torchvision.models import resnet18
        object_saliency_model = resnet18(weights="ResNet18_Weights.DEFAULT")
        object_saliency_model.avgpool = nn.Identity()
        object_saliency_model.fc = nn.Sequential(
            nn.Unflatten(1, (512, 7, 7)),
            NormReducer(dim=1),
            Unsqueeze(1)
        )

        detr_model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
        elif args.multiprocessing_distributed:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                audio_visual_model.cuda(args.gpu)
                object_saliency_model.cuda(args.gpu)
                detr_model.cuda(args.gpu)
                audio_visual_model = torch.nn.parallel.DistributedDataParallel(audio_visual_model, device_ids=[args.gpu])
                object_saliency_model = torch.nn.parallel.DistributedDataParallel(object_saliency_model, device_ids=[args.gpu])

        # Load weights
        ckp_fn = os.path.join(model_dir, 'best_{}.pth'.format(seed_num))
        if os.path.exists(ckp_fn):
            ckp = torch.load(ckp_fn, map_location='cpu')
            audio_visual_model.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
            print(f'loaded from {os.path.join(model_dir, "best_{}.pth".format(seed_num))}')
        else:
            raise ValueError(print(f"Checkpoint not found: {ckp_fn}. Make sure the path is correct and 'args.experiment_name' and 'args.seed' are as same as your training phase."))

        # Dataloader
        testdataset = get_test_dataset(args)
        testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        print("Loaded dataloader.")

        outputs = validate(testdataloader, audio_visual_model, object_saliency_model, detr_model, viz_dir, args)

        AV_best_mAP_all.append(outputs[0])
        AV_best_cIoU_all.append(outputs[1])
        AV_best_Auc_all.append(outputs[2])

        Resnet18_best_mAP_all.append(outputs[3])
        Resnet18_best_cIoU_all.append(outputs[4])
        Resnet18_best_Auc_all.append(outputs[5])

        detr_best_mAP_all.append(outputs[6])
        detr_best_cIoU_all.append(outputs[7])
        detr_best_Auc_all.append(outputs[8])

        AVO_best_mAP_all.append(outputs[9])
        AVO_best_cIoU_all.append(outputs[10])
        AVO_best_Auc_all.append(outputs[11])

    mean_AV_mAP = mean(AV_best_mAP_all)
    stdev_AV_mAP = stdev(AV_best_mAP_all)
    try:
        mean_AV_cIoU = mean(AV_best_cIoU_all)
        stdev_AV_cIoU = stdev(AV_best_cIoU_all)
    except ValueError as e:
        print(e)
        mean_AV_cIoU, stdev_AV_cIoU = 0, 0
    mean_AV_auc = mean(AV_best_Auc_all)
    stdev_AV_auc = stdev(AV_best_Auc_all)
    print(">>>>>> AV model: mAP_averaged for {} runs is: {}±{}".format(len(args.seed), mean_AV_mAP, stdev_AV_mAP))
    print(">>>>>> AV model: cIoU_averaged for {} runs is: {}±{}".format(len(args.seed), mean_AV_cIoU, stdev_AV_cIoU))
    print(">>>>>> AV model: Auc_averaged for {} runs is: {}±{}".format(len(args.seed), mean_AV_auc, stdev_AV_auc))

    mean_res_mAP = mean(Resnet18_best_mAP_all)
    stdev_res_mAP = stdev(Resnet18_best_mAP_all)
    try:
        mean_res_cIoU = mean(Resnet18_best_cIoU_all)
        stdev_res_cIoU = stdev(Resnet18_best_cIoU_all)
    except ValueError as e:
        print(e)
        mean_res_cIoU, stdev_res_cIoU = 0, 0
    mean_res_auc = mean(Resnet18_best_Auc_all)
    stdev_res_auc = stdev(Resnet18_best_Auc_all)
    print(">>>>>> resnet18 model: mAP_averaged for {} runs is: {}±{}".format(len(args.seed), mean_res_mAP, stdev_res_mAP))
    print(">>>>>> resnet18 model: cIoU_averaged for {} runs is: {}±{}".format(len(args.seed), mean_res_cIoU, stdev_res_cIoU))
    print(">>>>>> resnet18 model: Auc_averaged for {} runs is: {}±{}".format(len(args.seed), mean_res_auc, stdev_res_auc))

    mean_detr_mAP = mean(detr_best_mAP_all)
    stdev_detr_mAP = stdev(detr_best_mAP_all)
    mean_detr_auc = mean(detr_best_Auc_all)
    stdev_detr_auc = stdev(detr_best_Auc_all)
    try:
        mean_detr_cIoU = mean(detr_best_cIoU_all)
        stdev_detr_cIoU = stdev(detr_best_cIoU_all)
    except ValueError as e:
        print(e)
        mean_detr_cIoU, stdev_detr_cIoU = 0, 0
    print(">>>>>> detr model: mAP_averaged for {} runs is: {}±{}".format(len(args.seed), mean_detr_mAP, stdev_detr_mAP))
    print(">>>>>> detr model: cIoU_averaged for {} runs is: {}±{}".format(len(args.seed), mean_detr_cIoU, stdev_detr_cIoU))
    print(">>>>>> detr model: Auc_averaged for {} runs is: {}±{}".format(len(args.seed), mean_detr_auc, stdev_detr_auc))

    mean_AVO_mAP = mean(AVO_best_mAP_all)
    stdev_AVO_mAP = stdev(AVO_best_mAP_all)
    try:
        mean_AVO_cIoU = mean(AVO_best_cIoU_all)
        stdev_AVO_cIoU = stdev(AVO_best_cIoU_all)
    except ValueError as e:
        print(e)
        mean_AVO_cIoU, stdev_AVO_cIoU = 0, 0
    mean_AVO_auc = mean(AVO_best_Auc_all)
    stdev_AVO_auc = stdev(AVO_best_Auc_all)
    print(">>>>>> AV+obj model: mAP_averaged for {} runs is: {}±{}".format(len(args.seed), mean_AVO_mAP, stdev_AVO_mAP))
    print(">>>>>> AV+obj model: cIoU_averaged for {} runs is: {}±{}".format(len(args.seed), mean_AVO_cIoU, stdev_AVO_cIoU))
    print(">>>>>> AV+obj model: Auc_averaged for {} runs is: {}±{}".format(len(args.seed), mean_AVO_auc, stdev_AVO_auc))



@torch.no_grad()
def validate(testdataloader, audio_visual_model, object_saliency_model, detr_model, viz_dir, args):
    audio_visual_model.train(False)
    object_saliency_model.train(False)

    evaluator_av = utils.Evaluator()
    evaluator_obj = utils.Evaluator()
    evaluator_detr_224 = utils.Evaluator()
    evaluator_detr_800 = utils.Evaluator()
    evaluator_av_obj = utils.Evaluator()
    for step, (image, detr_image, spec, bboxes, bboxes_detr, name) in enumerate(testdataloader):
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            detr_image = detr_image.cuda(args.gpu, non_blocking=True)

        measure_complexity = False
        if measure_complexity:
            from thop import profile
            #audio_visual model
            flops, params = profile(audio_visual_model, inputs=(image.float(),spec.float()))
            print(f"FLOPs of audio_visual_model: {flops/1000000000}")
            print(f"params of audio_visual_model: {params/1000000}")
            #Resnet18 OG model
            flops, params = profile(object_saliency_model, inputs=(image,))
            print(f"FLOPs of Resnet18 OG model: {flops/1000000000}")
            print(f"params Resnet18 OG model: {params/1000000}")
            #detr OG model
            flops, params = profile(detr_model, inputs=(detr_image,))
            print(f"FLOPs of detr model: {flops/1000000000}")
            print(f"params detr model: {params/1000000}")


        # Compute S_AVL
        img_f, aud_f = audio_visual_model(image.float(), spec.float())
        with torch.no_grad():
            Slogits = torch.einsum('nchw,mc->nmhw', img_f, aud_f) / args.tau
            Savl = Slogits[torch.arange(img_f.shape[0]), torch.arange(img_f.shape[0])]
            heatmap_av = Savl.unsqueeze(1)

        heatmap_av = F.interpolate(heatmap_av, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_av = heatmap_av.data.cpu().numpy()


        # Compute S_OBJ
        img_feat = object_saliency_model(image)
        heatmap_obj = F.interpolate(img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        heatmap_obj = heatmap_obj.data.cpu().numpy()

        # Compute detr_OBJ

        detr_img_feat = get_detr_features(detr_model, detr_image)
        detr_heatmap_800 = F.interpolate(detr_img_feat, size=(800, 800), mode='bilinear', align_corners=True)
        detr_heatmap_800 = detr_heatmap_800.data.cpu().numpy()
        detr_heatmap_224 = F.interpolate(detr_img_feat, size=(224, 224), mode='bilinear', align_corners=True)
        detr_heatmap_224 = detr_heatmap_224.data.cpu().numpy()

        # Compute eval metrics and save visualizations
        for i in range(spec.shape[0]):
            pred_av = utils.normalize_img(heatmap_av[i, 0])
            pred_obj = utils.normalize_img(heatmap_obj[i, 0])
            pred_detr_224 = utils.normalize_img(detr_heatmap_224[i, 0])
            pred_detr_800 = utils.normalize_img(detr_heatmap_800[i, 0])
            if args.testset == "flickr":
                pred_av_obj = utils.normalize_img(pred_av/3 + pred_obj/3 + pred_detr_224/3)
            if args.testset == "vggss":
                pred_av_obj = utils.normalize_img(pred_av + args.alpha * pred_obj)

            try:
                gt_map_224 = bboxes['gt_map'].data.cpu().numpy()
                gt_map_800 = bboxes_detr['gt_map'].data.cpu().numpy()

                thr_av = np.sort(pred_av.flatten())[int(pred_av.shape[0] * pred_av.shape[1] * 0.5)]
                evaluator_av.cal_CIOU(pred_av, gt_map_224, (224, 224), thr_av)

                thr_obj = np.sort(pred_obj.flatten())[int(pred_obj.shape[0] * pred_obj.shape[1] * 0.5)]
                evaluator_obj.cal_CIOU(pred_obj, gt_map_224, (224, 224), thr_obj)

                thr_detr_224 = np.sort(pred_detr_224.flatten())[int(pred_detr_224.shape[0] * pred_detr_224.shape[1] * 0.5)]
                evaluator_detr_224.cal_CIOU(pred_detr_224, gt_map_224, (224, 224), thr_detr_224)
                thr_detr_800 = np.sort(pred_detr_800.flatten())[int(pred_detr_800.shape[0] * pred_detr_800.shape[1] * 0.5)]
                evaluator_detr_800.cal_CIOU(pred_detr_800, gt_map_800, (800, 800), thr_detr_800)

                thr_av_obj = np.sort(pred_av_obj.flatten())[int(pred_av_obj.shape[0] * pred_av_obj.shape[1] * 0.5)]
                evaluator_av_obj.cal_CIOU(pred_av_obj, gt_map_224, (224, 224), thr_av_obj)

                if args.save_visualizations:
                    denorm_image = inverse_normalize(image).squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                    denorm_image = (denorm_image*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_image.jpg'), denorm_image)

                    # visualize bboxes on raw images
                    gt_boxes_img = utils.visualize(denorm_image, bboxes['bboxes'])
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_gt_boxes.jpg'), gt_boxes_img)

                    # visualize heatmaps
                    heatmap_img = np.uint8(pred_av*255)
                    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av.jpg'), fin)

                    heatmap_img = np.uint8(pred_obj*255)
                    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_obj.jpg'), fin)

                    heatmap_img_detr_800 = np.uint8(pred_detr_800 * 255)
                    heatmap_img_detr_800 = cv2.applyColorMap(heatmap_img_detr_800[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin_detr_800 = cv2.addWeighted(heatmap_img_detr_800, 0.8, np.uint8(heatmap_img_detr_800), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_detr_800.jpg'), fin_detr_800)

                    heatmap_img = np.uint8(pred_av_obj*255)
                    heatmap_img = cv2.applyColorMap(heatmap_img[:, :, np.newaxis], cv2.COLORMAP_JET)
                    fin = cv2.addWeighted(heatmap_img, 0.8, np.uint8(denorm_image), 0.2, 0)
                    cv2.imwrite(os.path.join(viz_dir, f'{name[i]}_pred_av_obj.jpg'), fin)
            except KeyError as e:
                print("ground truth bboxes for sample {} is not found.". format(name), e, bboxes)
        print(f'{step+1}/{len(testdataloader)}: '
              f'map_av={evaluator_av.finalize_AP50():.2f} '
              f'map_obj={evaluator_obj.finalize_AP50():.2f} '
              f'map_detr_800={evaluator_detr_800.finalize_AP50():.2f} '
              f'map_detr_224={evaluator_detr_224.finalize_AP50():.2f} '
              f'map_av_obj={evaluator_av_obj.finalize_AP50():.2f}')

    def compute_stats(eval):
        mAP = eval.finalize_AP50()
        ciou = eval.finalize_cIoU()
        auc = eval.finalize_AUC()
        return mAP, ciou, auc

    print('AV: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av)))
    print('Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_obj)))
    print('detr_224: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_detr_224)))
    print('detr_800: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_detr_800)))
    print('AV_Obj: AP50(cIoU)={}, Avg-cIoU={}, AUC={}'.format(*compute_stats(evaluator_av_obj)))

    utils.save_iou(evaluator_av.ciou, 'av', viz_dir)
    utils.save_iou(evaluator_obj.ciou, 'obj', viz_dir)
    utils.save_iou(evaluator_detr_224.ciou, 'detr', viz_dir)
    utils.save_iou(evaluator_detr_800.ciou, 'detr', viz_dir)
    utils.save_iou(evaluator_av_obj.ciou, 'av_obj', viz_dir)

    return *compute_stats(evaluator_av), *compute_stats(evaluator_obj), *compute_stats(evaluator_detr_224), *compute_stats(evaluator_av_obj)

class NormReducer(nn.Module):
    def __init__(self, dim):
        super(NormReducer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.abs().mean(self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


def get_detr_features(model, input_img):
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []
    hooks = [model.backbone[-2].register_forward_hook(lambda self, input, output: conv_features.append(output)),
             model.transformer.encoder.layers[-1].self_attn.register_forward_hook(lambda self, input, output: enc_attn_weights.append(output[1])),
             model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(lambda self, input, output: dec_attn_weights.append(output[1]))]
    outputs = model(input_img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.3

    for hook in hooks:
        hook.remove()
    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]
    dert_dec_feats = dec_attn_weights.view(dec_attn_weights.size()[1], h, w)
    if len(keep.nonzero()) == 1:
        feats = dert_dec_feats[keep.nonzero()[0][0]].unsqueeze(0).unsqueeze(0)
    else:  # len(keep.nonzero()[0]) > 1:
        strong_featues = torch.zeros((len(keep.nonzero())), dert_dec_feats.size()[1], dert_dec_feats.size()[2])
        for idx, index in enumerate(keep.nonzero()):
            one_feat = dert_dec_feats[index[0]]
            strong_featues[idx, :, :] = one_feat
        feats = strong_featues.abs().mean(dim=0).unsqueeze(0).unsqueeze(0)
    return feats

if __name__ == "__main__":
    main(get_arguments())

