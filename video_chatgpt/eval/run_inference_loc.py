from os import makedirs as os_makedirs
from os.path import join as os_path_join, exists as os_path_exists
from argparse import ArgumentParser
from json import load as json_load, dump as json_dump
from glob import glob
from warnings import warn
from tqdm import tqdm
from torch import device as torch_device, no_grad as torch_no_grad
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_qa', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=False)

    return parser.parse_args()


@torch_no_grad()
def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    video_dir = args.video_dir
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # Load both ground truth file containing questions and answers
    with open(args.gt_file_qa) as file:
        gt_qa = json_load(file)

    # Create the output directory if it doesn't exist
    output_dir = args.output_dir
    if not os_path_exists(output_dir):
        os_makedirs(output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    device = torch_device('cuda')

    # Iterate over each sample in the ground truth file
    for question_dict in tqdm(gt_qa):
        conversations = question_dict['conversations']
        question = conversations[0]['value']
        question_id = question_dict['id']
        answer = conversations[1]['value']

        sample_set = {'id': question_id, 'question': question, 'answer': answer}

        videos_search_path = os_path_join(video_dir, question_id+'*')
        videos_match_list = glob(videos_search_path)
        if not videos_match_list:
            warn(f'No videos found for {videos_search_path}')
            continue
        video_fpath = videos_match_list[0]
        # Load the video file

        # Check if the video exists
        video_frames = load_video(video_fpath, device)

        try:
            # Run inference on the video and add the output to the list
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                            tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_fpath}': {e}")

    # Save the output list to a JSON file
    with open(os_path_join(output_dir, f"{args.output_name}.json"), 'w') as file:
        json_dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
