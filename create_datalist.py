import os
import random
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flicker_path', type=str, default="/data2/datasets/vggsound/")
    parser.add_argument('--data_split', type=str, default="./metadata/vggsound_144k_random.txt", help= "choices:[flickr_10k_random.txt, flickr_144k_random.txt, vggsound_10k_random.txt, vggsound_144k_random.txt]")
    return parser.parse_args()


def main(args):
    audio_feats = {f.split('.')[0] for dirpath, dirnames, filenames in os.walk(os.path.join(args.flicker_path, 'audio_feats')) for f in filenames}
    print('Number of available audio features: ', len(audio_feats))
    assert len(audio_feats) != 0
    img_feats = {f.split('.')[0] for dirpath, dirnames, filenames in os.walk(os.path.join(args.flicker_path, 'detr_dec_feats')) for f in filenames}
    print('Number of available image features: ', len(img_feats))
    assert len(img_feats) != 0
    audio_files = {f.split('.')[0] for dirpath, dirnames, filenames in os.walk(os.path.join(args.flicker_path, 'audio')) for f in filenames}
    print('Number of available audio files: ', len(audio_files))
    assert len(audio_files) != 0
    image_files = {f.split('.')[0] for dirpath, dirnames, filenames in os.walk(os.path.join(args.flicker_path, 'frames')) for f in filenames}
    print('Number of available images: ', len(image_files))
    assert len(image_files) != 0
    avail_files = audio_files.intersection(image_files).intersection(audio_feats).intersection(img_feats)
    print(f"{len(avail_files)} available samples")
    assert len(avail_files) != 0

    random.seed(1)
    if args.data_split in ["./metadata/flickr_10k_random.txt", "./metadata/vggsound_10k_random.txt"]:
        data_10k = random.sample(list(avail_files), 10000)
        with open(args.data_split, '+a') as file:
            file.write("\n".join(data_10k))
        print('datalist {} is created.'.format(args.data_split))

    if args.data_split in ["./metadata/flickr_144k_random.txt", "./metadata/vggsound_144k_random.txt"]:
        assert len(avail_files) >= 144000
        data_144k = random.sample(list(avail_files), 144000)
        with open(args.data_split, '+a') as file:
            file.write("\n".join(data_144k))
        print('datalist {} is created.'.format(args.data_split))


if __name__ == "__main__":
    main(get_arguments())
