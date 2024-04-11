from json import load as json_load, dump as json_dump
from argparse import ArgumentParser
from tqdm import tqdm


PROMPT_STRING = 'Please watch the video and output a video frame (by index, an integer between 1 and 100) most relevant to answering the question:\n'


def parse_args():
    parser = ArgumentParser(description="Convert Instruction to Training Format")

    parser.add_argument("--input_json_file", required=True, help="")
    parser.add_argument("--output_json_file", required=True, help="")
    parser.add_argument("--gt_ts_file", required=True, help="")

    return parser.parse_args()


def main():
    args = parse_args()
    input_json_file = args.input_json_file
    output_json_file = args.output_json_file
    gt_ts_file = args.gt_ts_file

    with open(input_json_file, 'r') as file_in:
        input_json_contents = json_load(file_in)
    with open(gt_ts_file, 'r') as file_in:
        gt_ts_dict = json_load(file_in)

    output_json_contents = []
    for i, content in enumerate(tqdm(input_json_contents)):
        video_id = content['vid_name']
        question = content['q']
        if video_id not in gt_ts_dict:
            print(f"Skipping video {video_id} as it is not in the ground truth timestamp file")
            continue
        max_frames_indices = gt_ts_dict[video_id][content['qid']]['gt_frame_idx_max'][0] # NOTE: the first index is selected in case of equality
        answer = str(max_frames_indices)
        prompt = f'{PROMPT_STRING}\n{question}\n'

        output_content = {'id': video_id, 'video': f"{video_id}.pkl", 'conversations': []}

        if i % 2 == 0:
            output_content['conversations'].append({'from': 'human', 'value': f"{prompt}\n<video>"})
        else:
            output_content['conversations'].append({'from': 'human', 'value': f"<video>\n{prompt}"})

        output_content['conversations'].append({'from': 'gpt', 'value': answer})
        output_json_contents.append(output_content)

    print(f"Total annotations retained: {len(output_json_contents)}")
    with open(output_json_file, 'w') as f:
        json_dump(output_json_contents, f)


if __name__ == '__main__':
    main()
