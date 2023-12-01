from os import makedirs as os_makedirs
from os.path import join as os_path_join, exists as os_path_exists
from argparse import ArgumentParser
from json import load as json_load, dump as json_dump
from glob import glob
from warnings import warn
from tqdm import tqdm
from torch import device as torch_device, no_grad as torch_no_grad, load as torch_load
from torch.nn import Parameter
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
    parser.add_argument("--loo_mm_projector_path", type=str, required=False)

    return parser.parse_args()

# @torch_no_grad()
# def evaluate_accuracy(model, device, dataloader_test) -> None:
#     '''
#     3. evalute on each video. Assume the dataloader batch size is 1. Model is not changed.
#     '''
#     model.eval()
#     accuracy_per_vid = {}
#     for batch in dataloader_test:
#         vid = batch.pop('id')
#         assert len(vid) == 1
#         vid = vid[0]
#         batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
#         _, logits = model(**batch, return_dict=False)
#         breakpoint()
#         # TODO
#         accuracy = None
#         labels = batch.pop('labels')
#         accuracy_per_vid[vid] = accuracy

#     return accuracy_per_vid

@torch_no_grad()
def load_nn_from_file(model, fname_weight_bias_frame_idx: str) -> None:
    dict_weights_biases = torch_load(fname_weight_bias_frame_idx)
    # model[0].weight = nn.Parameter(torch.ones_like(model[0].weight))
    model.model.mm_projector.weight =  Parameter(dict_weights_biases['weight'].to(dtype=model.dtype)).to(dtype=model.dtype)
    model.model.mm_projector.bias = Parameter(dict_weights_biases['bias'].to(dtype=model.dtype)).to(dtype=model.dtype)


@torch_no_grad()
def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    video_dir = args.video_dir
    fname_weight_bias_frame_idx = args.loo_mm_projector_path
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    load_nn_from_file(model, fname_weight_bias_frame_idx)
    # Load both ground truth file containing questions and answers
    with open(args.gt_file_qa) as file:
        gt_qa = json_load(file)

    # Create the output directory if it doesn't exist
    output_dir = args.output_dir
    if not os_path_exists(output_dir):
        os_makedirs(output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode


    device = torch_device('cuda:0')

    # Iterate over each sample in the ground truth file
    # index = 0
    len_gt_questions = len(gt_qa)
    for sample_qa in tqdm(gt_qa, total=len_gt_questions):
        conversations = sample_qa['conversations']
        question = conversations[0]['value']
        id = sample_qa['id']
        # answer = gt_answers[index]['answer']
        answer = conversations[1]['value']
        # index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        videos_search_path = os_path_join(video_dir, id+'*')
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
