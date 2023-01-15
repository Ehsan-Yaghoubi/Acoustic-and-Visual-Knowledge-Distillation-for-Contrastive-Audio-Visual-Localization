import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av


def load_image(path):
    return Image.open(path).convert('RGB')


def load_spectrogram(path, dur=3.):
    # Load audio
    audio_ctr = open_audio_av(path)
    audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)

    # To Mono
    audio = np.clip(audio, -1., 1.).mean(0)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram


def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    gt_bboxes_dert = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            dert_bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    dert_bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text) / 256))
                            dert_bbox.append(int(800 * int(ch.text) / 256))
                    bboxes.append(bbox)
                    dert_bboxes.append(dert_bbox)
            gt_bboxes[file] = bboxes
            gt_bboxes_dert[file] = dert_bboxes

    elif format == 'vggss':
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            gt_bboxes[annotation['file']] = bboxes
            dert_bboxes = [(np.clip(np.array(dert_bbox), 0, 1) * 800).astype(int) for dert_bbox in annotation['bbox']]
            gt_bboxes_dert[annotation['file']] = dert_bboxes

    return gt_bboxes, gt_bboxes_dert


def bbox2gtmap(bboxes, dert_bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    gt_map_dert = np.zeros([800, 800])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    for xmin, ymin, xmax, ymax in dert_bboxes:
        temp = np.zeros([800, 800])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map_dert += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1
        gt_map_dert = gt_map_dert / 2
        gt_map_dert[gt_map_dert > 1] = 1

    elif format == 'vggss':
        # Single annotation
        gt_map[gt_map > 0] = 1
        gt_map_dert[gt_map_dert > 0] = 1

    return gt_map, gt_map_dert


class AudioVisualDatasetTest(Dataset):
    def __init__(self, paths, files, audio_dur=3., image_transform=None, detr_transform=None, audio_transform=None, all_bboxes=None, all_bboxes_detr=None, bbox_format='flickr'):
        super().__init__()
        self.image_files = files[0]
        self.audio_files = files[1]
        self.image_path = paths[0]
        self.audio_path = paths[1]

        self.audio_dur = audio_dur
        self.all_bboxes = all_bboxes
        self.all_bboxes_detr = all_bboxes_detr
        self.bbox_format = bbox_format

        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.detr_transform = detr_transform

    def getitem(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = self.image_transform(load_image(img_fn))
        if self.detr_transform is not None:
            detr_frame = self.detr_transform(load_image(img_fn))
        else:
            detr_frame = np.zeros((1, 1))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram = self.audio_transform(load_spectrogram(audio_fn))

        bboxes = {}
        bboxes_detr = {}
        if self.all_bboxes is not None:
            bboxes['bboxes'], bboxes_detr['bboxes'] = self.all_bboxes[file_id], self.all_bboxes_detr[file_id]
            bboxes['gt_map'], bboxes_detr['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.all_bboxes_detr[file_id], self.bbox_format)

        return frame, detr_frame, spectrogram, bboxes, bboxes_detr, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


class AudioVisualDatasetTrain(Dataset):
    def __init__(self, paths, files, audio_dur=3., image_transform=None, detr_transform=None, audio_transform=None, all_bboxes=None, all_bboxes_detr=None,
                 bbox_format='flickr', trans=None):
        super().__init__()
        self.image_files = files[0]
        self.audio_files = files[1]
        self.audio_feats = files[2]
        self.detr_conv_feats = files[3]
        self.detr_dec_feats = files[4]
        self.detr_enc_feats = files[5]
        self.image_path = paths[0]
        self.audio_path = paths[1]
        self.audio_feat_path = paths[2]
        self.detr_conv_feats_path = paths[3]
        self.detr_dec_feats_path = paths[4]
        self.detr_enc_feats_path = paths[5]

        self.audio_dur = audio_dur
        self.all_bboxes = all_bboxes
        self.all_bboxes_detr = all_bboxes_detr
        self.bbox_format = bbox_format

        self.image_transform = image_transform
        self.audio_transform = audio_transform
        self.detr_transform = detr_transform
        self.trans = trans

    def getitem(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = self.image_transform(load_image(img_fn))
        if self.detr_transform is not None:
            detr_frame = self.detr_transform(load_image(img_fn))
        else:
            detr_frame = np.zeros((1, 1))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram = self.audio_transform(load_spectrogram(audio_fn))

        aud_feat_path = os.path.join(self.audio_feat_path, self.audio_feats[idx])
        f_conv_detr_path = os.path.join(self.detr_conv_feats_path, self.detr_conv_feats[idx])
        f_enc_detr_path = os.path.join(self.detr_enc_feats_path, self.detr_enc_feats[idx])
        f_dec_detr__path = os.path.join(self.detr_dec_feats_path, self.detr_dec_feats[idx])
        aud_embedding = torch.tensor(np.load(aud_feat_path).squeeze())
        conv_detr_feat = torch.tensor(np.load(f_conv_detr_path).squeeze())
        enc_detr_feat = torch.tensor(np.load(f_enc_detr_path).squeeze())
        dec_detr_feat = torch.tensor(np.load(f_dec_detr__path).squeeze())
        assert conv_detr_feat.size() == enc_detr_feat.size() == dec_detr_feat.size() == (25, 25)
        assert aud_embedding.size() == (2048,)

        bboxes = {}
        bboxes_detr = {}
        if self.all_bboxes is not None:
            bboxes['bboxes'], bboxes_detr['bboxes'] = self.all_bboxes[file_id], self.all_bboxes_detr[file_id]
            bboxes['gt_map'], bboxes_detr['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.all_bboxes_detr[file_id], self.bbox_format)

        return frame, detr_frame, spectrogram, aud_embedding, conv_detr_feat, enc_detr_feat, dec_detr_feat, bboxes, bboxes_detr, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])


def check_available_samples(data_path, kind):
    _path = os.path.join(data_path, '{}'.format(kind))
    _files = {f.split('.')[0] for dirpath, dirnames, filenames in os.walk(_path) for f in filenames}
    print('Number of available {} files: {}'.format(kind, len(_files)))
    assert len(_files) != 0
    return _files, _path


def get_train_dataset(args):
    # List directory
    audio_files, audio_path = check_available_samples(args.train_data_path, 'audio')
    image_files, image_path = check_available_samples(args.train_data_path, 'frames')
    audio_feats, audio_feat_path = check_available_samples(args.train_data_path, 'audio_feats')
    detr_conv_feats, detr_conv_feats_path = check_available_samples(args.train_data_path, 'detr_conv_feats')
    detr_dec_feats, detr_dec_feats_path = check_available_samples(args.train_data_path, 'detr_dec_feats')
    detr_enc_feats, detr_enc_feats_path = check_available_samples(args.train_data_path, 'detr_enc_feats')
    # check available samples
    avail_files = audio_files.intersection(image_files).intersection(audio_feats).intersection(detr_conv_feats).intersection(detr_dec_feats).intersection(detr_enc_feats)
    print(f"{len(avail_files)} available samples")
    assert len(avail_files) != 0

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))
    audio_files = sorted([dt+'.flac' for dt in avail_files])
    audio_feats = sorted([dt + '.flac.npy' for dt in avail_files])
    image_files = sorted([dt + '.jpg' for dt in avail_files])
    detr_conv_feats = sorted([dt + '.jpg.npy' for dt in avail_files])
    detr_dec_feats = sorted([dt + '.jpg.npy' for dt in avail_files])
    detr_enc_feats = sorted([dt + '.jpg.npy' for dt in avail_files])

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    trans = transforms.Compose([transforms.ToTensor()])

    files = [image_files, audio_files, audio_feats, detr_conv_feats, detr_dec_feats, detr_enc_feats]
    paths = [image_path, audio_path, audio_feat_path, detr_conv_feats_path, detr_dec_feats_path, detr_enc_feats_path]

    return AudioVisualDatasetTrain(paths, files, audio_dur=3., image_transform=image_transform, audio_transform=audio_transform)


def get_test_dataset(args):
    audio_path = args.test_data_path + 'audio/'
    image_path = args.test_data_path + 'frames/'

    if args.testset == 'flickr':
        testcsv = 'metadata/flickr_test.csv'
    elif args.testset == 'vggss':
        testcsv = 'metadata/vggss_test.csv'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss'}[args.testset]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])
    assert len(testset) > 0

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)
    assert len(testset) > 0
    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    audio_files = [dt+'.wav' for dt in testset]
    assert len(image_files) > 0
    assert len(audio_files) > 0
    # Bounding boxes
    all_bboxes, all_bboxes_detr = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    detr_transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trans = transforms.Compose([transforms.ToTensor()])

    files = [image_files, audio_files]
    paths = [image_path, audio_path]
    return AudioVisualDatasetTest(paths, files,
                                  audio_dur=3.,
                                  image_transform=image_transform,
                                  detr_transform=detr_transform,
                                  audio_transform=audio_transform,
                                  all_bboxes=all_bboxes,
                                  all_bboxes_detr=all_bboxes_detr,
                                  bbox_format=bbox_format)


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor



