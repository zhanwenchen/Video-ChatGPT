from os import makedirs as os_makedirs
from os.path import join as os_path_join, exists as os_path_exists
from argparse import ArgumentParser
import json
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
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
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
    with open(args.gt_file_question) as file:
        gt_questions = json.load(file)
    with open(args.gt_file_answers) as file:
        gt_answers = json.load(file)

    # Create the output directory if it doesn't exist
    output_dir = args.output_dir
    if not os_path_exists(output_dir):
        os_makedirs(output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    nonexistent_videos = []
    device = torch_device('cuda:0')

    # Iterate over each sample in the ground truth file
    # index = 0
    len_gt_questions = len(gt_questions)
    assert len_gt_questions == len(gt_answers)
    for sample_q, sample_a in tqdm(zip(gt_questions, gt_answers), total=len_gt_questions):
        video_name = sample_q['video_name']
        question = sample_q['question']
        id = sample_q['question_id']
        # answer = gt_answers[index]['answer']
        answer = sample_a['answer']
        # index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        video_path = None
        for fmt in video_formats:  # Added this line
            video_path = os_path_join(video_dir, f"v_{video_name}{fmt}")
            if os_path_exists(video_path):
                break

        # Check if the video exists
        # if os_path_exists(video_path):
        if video_path:
            video_frames = load_video(video_path, device)
        else:
            nonexistent_videos.append(video_name)
            warn(f'video_path={video_path} doesn\'t exist')
            continue

        try:
            # Run inference on the video and add the output to the list
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output
            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # Save the output list to a JSON file
    with open(os_path_join(output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
