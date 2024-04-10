from json import dump as json_dump
from argparse import ArgumentParser
from utils_use_gpt4v import load_and_add_ts_to_gt


def parse_args():
    parser = ArgumentParser(description="Merging Videos")

    parser.add_argument("--gpt4v_result_dirpath", required=True, help="")
    parser.add_argument("--qa_file_in", required=True, help="")
    parser.add_argument("--qa_file_out", required=True, help="")

    return parser.parse_args()


def load_and_save(gpt4v_result_dirpath, qa_file_in, qa_file_out):
    dict_gt_with_ts = load_and_add_ts_to_gt(gpt4v_result_dirpath, qa_file_in)
    with open(qa_file_out, 'w') as f:
        json_dump(dict_gt_with_ts, f)


if __name__ == '__main__':
    args = parse_args()
    load_and_save(args.gpt4v_result_dirpath, args.qa_file_in, args.qa_file_out)
