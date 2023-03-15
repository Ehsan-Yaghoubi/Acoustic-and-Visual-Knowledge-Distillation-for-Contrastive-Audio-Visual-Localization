import os
import argparse
import builtins
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp
import torch.distributed as dist

import utils
from model import EZVSL
from datasets import get_train_dataset, get_test_dataset
import random
from statistics import mean, stdev

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./checkpoints', help='path to save trained model weights')
    parser.add_argument('--experiment_name', type=str, default='flickr_10k_10_seed_run3', help='experiment name (used for checkpointing and logging)')

    # Data params
    parser.add_argument('--trainset', default='flickr_10k_random', type=str, help='trainset (flickr or vggss)')
    parser.add_argument('--testset', default='flickr', type=str, help='testset,(flickr or vggss)')
    parser.add_argument('--train_data_path', default='/data2/datasets/small_subset_flicker/10k_flicker', type=str, help='Root directory path of train data')
    parser.add_argument('--test_data_path', default='/data2/datasets/labeled_5k_flicker/Data/', type=str, help='Root directory path of test data')
    parser.add_argument('--test_gt_path', default='/data2/datasets/labeled_5k_flicker/Annotations/', type=str)

    # ez-vsl hyper-params
    parser.add_argument('--out_dim', default=512, type=int)
    parser.add_argument('--tau', default=0.03, type=float, help='tau')

    # training/evaluation parameters
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--seed", type=list, default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100], help="random seed")

    # Distributed params
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--node', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:12345')
    parser.add_argument('--multiprocessing_distributed', action='store_true')

    return parser.parse_args()


def main(args):
    ## os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    # multiprocessing
    mp.set_start_method('spawn')
    args.dist_url = f'tcp://{args.node}:{args.port}'
    print('Using url {}'.format(args.dist_url))

    ngpus_per_node = torch.cuda.device_count()

    best_cIoU_all = []
    best_Auc_all = []
    for train_round, seed_num in enumerate(args.seed):
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
        print(">>>>>> We will train the model for {} times with seeds {}. Then the average values for cIoU and AUC will be calculated.".format(len(args.seed), args.seed))
        print(">>>>>> Training seed for this round of train is {}.".format(seed_num))

        if args.multiprocessing_distributed:
            args.world_size = ngpus_per_node
            mp.spawn(main_worker,
                     nprocs=ngpus_per_node,
                     args=(ngpus_per_node, args))

        else:
            best_cIoU, best_Auc = main_worker(seed_num, args.gpu, ngpus_per_node, args)

        best_cIoU_all.append(best_cIoU)
        best_Auc_all.append(best_Auc)

    mean_cIoU = mean(best_cIoU_all)
    stdev_cIoU = stdev(best_cIoU_all)
    mean_auc = mean(best_Auc_all)
    stdev_auc = stdev(best_Auc_all)
    print(">>>>>> cIoU_averaged for {} runs is: {}±{}".format(len(args.seed), mean_cIoU, stdev_cIoU))
    print(">>>>>> Auc_averaged for {} runs is: {}±{}".format(len(args.seed), mean_auc, stdev_auc))
    print("Note: Use the test_N_times.py for getting the evaluation results, when we use the AV_model alongside object detectors explicitly")


def main_worker(train_round, gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Setup distributed environment
    if args.multiprocessing_distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Create model dir
    model_dir = os.path.join(args.model_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    utils.save_json(vars(args), os.path.join(model_dir, 'configs_{}.json'.format(train_round)), sort_keys=True, save_pretty=True)

    # Create model
    model = EZVSL(args.tau, args.out_dim)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    #print(model)

    # Optimizer
    optimizer, scheduler = utils.build_optimizer_and_scheduler_adam(model, args)

    # Resume if possible
    start_epoch, best_cIoU, best_Auc = 0, 0., 0.
    if os.path.exists(os.path.join(model_dir, 'latest_{}.pth'.format(train_round))):
        ckp = torch.load(os.path.join(model_dir, 'latest_{}.pth'.format(train_round)), map_location='cpu')
        start_epoch, best_cIoU, best_Auc = ckp['epoch'], ckp['best_cIoU'], ckp['best_Auc']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        print(f'loaded from {os.path.join(model_dir, "latest_{}.pth".format(train_round))}')

    # Dataloaders
    traindataset = get_train_dataset(args)
    train_sampler = None
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)
    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
        persistent_workers=args.workers > 0)

    testdataset = get_test_dataset(args)
    test_loader = torch.utils.data.DataLoader(
        testdataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        persistent_workers=args.workers > 0)
    print("Loaded dataloader.")

    # =============================================================== #
    # Training loop
    cIoU, auc = validate(test_loader, model, args)
    print(f'cIoU (epoch {start_epoch}): {cIoU}')
    print(f'AUC (epoch {start_epoch}): {auc}')
    print(f'best_cIoU: {best_cIoU}')
    print(f'best_Auc: {best_Auc}')

    for epoch in range(start_epoch, args.epochs):
        if args.multiprocessing_distributed:
            train_loader.sampler.set_epoch(epoch)

        # Train
        train(train_loader, model, optimizer, epoch, args)

        # Evaluate
        cIoU, auc = validate(test_loader, model, args)
        print(f'cIoU (epoch {epoch+1}): {cIoU}')
        print(f'AUC (epoch {epoch+1}): {auc}')
        print(f'best_cIoU: {best_cIoU}')
        print(f'best_Auc: {best_Auc}')

        # Checkpoint
        if args.rank == 0:
            ckp = {'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'epoch': epoch+1,
                   'best_cIoU': best_cIoU,
                   'best_Auc': best_Auc}
            torch.save(ckp, os.path.join(model_dir, 'latest_{}.pth'.format(train_round)))
            print(f"Model saved to {model_dir}")
        if cIoU >= best_cIoU:
            best_cIoU, best_Auc = cIoU, auc
            if args.rank == 0:
                torch.save(ckp, os.path.join(model_dir, 'best_{}.pth'.format(train_round)))

    return best_cIoU, best_Auc

def max_xmil_loss(img, aud):
    B = img.shape[0]
    Slogits = torch.einsum('nchw,mc->nmhw', img, aud) / 0.03
    logits = Slogits.flatten(-2, -1).max(dim=-1)[0]
    labels = torch.arange(B).long().to(img.device)
    loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.permute(1, 0), labels)
    return loss, Slogits

def detr_panns_loss(img, aud, aud_embedding, conv_detr_feat, enc_detr_feat, dec_detr_feat):
    #mae_loss = torch.nn.L1Loss()
    mae_loss = torch.nn.MSELoss()
    #cos_loss = torch.nn.CosineSimilarity()
    aud_loss = mae_loss(aud, aud_embedding)

    img_f = F.interpolate(img, (25, 25), mode='bilinear', align_corners=False)

    # Flatten:
    shape = dec_detr_feat.shape
    tensor_reshaped = dec_detr_feat.reshape(shape[0], -1)
    # Drop all rows containing any nan:
    tensor_filtered = tensor_reshaped[~torch.any(tensor_reshaped.isnan(), dim=1)]
    # same samples are selected from img_f
    img_feat = torch.mean(img_f, dim=1)
    img_feat_reshaped = img_feat.reshape(shape[0], -1)
    img_feat_reshaped = img_feat_reshaped[~torch.any(tensor_reshaped.isnan(), dim=1)]
    img_loss = mae_loss(img_feat_reshaped, tensor_filtered)

    return (100000*img_loss) + (1000*aud_loss)


def train(train_loader, model, optimizer, epoch, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_mtr = AverageMeter('Loss', ':.3f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, loss_mtr],
        prefix="Epoch: [{}]".format(epoch),
    )

    end = time.time()
    for i, (image, _, spec, aud_embedding, conv_detr_feat, enc_detr_feat, dec_detr_feat, _, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)
            aud_embedding = aud_embedding.cuda(args.gpu, non_blocking=True)
            conv_detr_feat = conv_detr_feat.cuda(args.gpu, non_blocking=True)
            enc_detr_feat = enc_detr_feat.cuda(args.gpu, non_blocking=True)
            dec_detr_feat = dec_detr_feat.cuda(args.gpu, non_blocking=True)

        img_f, aud_f = model(image.float(), spec.float())
        # Compute loss
        av_loss, logits = max_xmil_loss(img_f, aud_f)
        # Learn from Detr (object detector) and the Panns (sound classifier) features
        pk_loss = detr_panns_loss(img_f, aud_f, aud_embedding, conv_detr_feat, enc_detr_feat, dec_detr_feat)

        loss = pk_loss + av_loss
        # loss = av_loss  # ablate teachers and only keep AVC
        # Compute avl maps
       # with torch.no_grad():
            #B = img_f.shape[0]
            #Savl = logits[torch.arange(B), torch.arange(B)]
        loss_mtr.update(loss.item(), image.shape[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0 or i == len(train_loader) - 1:
            progress.display(i)
        del loss


def validate(test_loader, model, args):
    model.train(False)
    evaluator = utils.Evaluator()
    assert len(test_loader) > 0
    for step, (image, _, spec, bboxes, _, _) in enumerate(test_loader):
        if torch.cuda.is_available():
            spec = spec.cuda(args.gpu, non_blocking=True)
            image = image.cuda(args.gpu, non_blocking=True)

        img_f, aud_f = model(image.float(), spec.float())
        with torch.no_grad():
            Slogits = torch.einsum('nchw,mc->nmhw', img_f, aud_f) / args.tau
            Savl = Slogits[torch.arange(img_f.shape[0]), torch.arange(img_f.shape[0])]

        avl_map = F.interpolate(Savl.unsqueeze(1), size=(224, 224), mode='bicubic', align_corners=False)
        avl_map = avl_map.data.cpu().numpy()

        for i in range(spec.shape[0]):
            pred = utils.normalize_img(avl_map[i, 0])
            gt_map = bboxes['gt_map'].data.cpu().numpy()
            thr = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]
            evaluator.cal_CIOU(pred, gt_map, (224, 224), thr)

    cIoU = evaluator.finalize_AP50()
    AUC = evaluator.finalize_AUC()
    return cIoU, AUC


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", fp=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.fp = fp

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg, flush=True)
        if self.fp is not None:
            self.fp.write(msg+'\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == "__main__":
    main(get_arguments())

