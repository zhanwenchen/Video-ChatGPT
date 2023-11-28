'''
This script converts the json file containing the instructions to the training format of LOO for siq2.
python scripts/split_train_eval_loo.py \
    --input_json_file data/tomloc/qa/tomloc_train_removed_merged_n3_with_frames_idx_instruction.json \
    --qas_train_loo_fpath data/tomloc/qa/tomloc_train_loo_removed_merged_n3_with_frames_idx_instruction.json \
    --qas_eval_loo_fpath data/tomloc/qa/tomloc_eval_loo_removed_merged_n3_with_frames_idx_instruction.json
'''
from json import load as json_load, dump as json_dump
from collections import defaultdict
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Merging Videos")
    parser.add_argument("--input_json_file", type=str, required=True, help='')
    parser.add_argument("--qas_train_loo_fpath", type=str, required=True, help='')
    parser.add_argument("--qas_eval_loo_fpath", type=str, required=True, help='')
    return parser.parse_args()


def get_unique_ts(qas):
    qas_by_video = defaultdict(set)
    for qa in qas:
        qas_by_video[qa['vid_name']].add(qa['ts'])
    return qas_by_video


def split_train_eval_loo(json):
    with open(json, 'r') as f:
        data = json_load(f)
    qas_by_video = get_unique_ts(data)
    qas_train = []
    qas_eval_loo = []
    for video, qas in qas_by_video.items():
        qas_train_vid, qas_eval_vid = qas[:-1], qas[-1]
        qas_train.append(qas_train_vid)
        qas_eval_loo.append(qas_eval_vid)
    return qas_train, qas_eval_loo


def split_and_save(json, qas_train_loo_fpath, qas_eval_loo_fpath):
    qas_train_loo, qas_eval_loo = split_train_eval_loo(json)

    with open(qas_train_loo_fpath, 'w') as f:
        json_dump(qas_train_loo, f)
    with open(qas_eval_loo_fpath, 'w') as f:
        json_dump(qas_eval_loo, f)


def main():
    args = parse_args()
    split_and_save(args.input_json_file, args.qas_train_loo_fpath, args.qas_eval_loo_fpath)


if __name__ == "__main__":
    main()
