from os import environ
from os.path import expanduser as os_path_expanduser, basename as os_path_basename
import numpy as np
from torch import (
    zeros as torch_zeros,
    as_tensor as torch_as_tensor,
    float16 as torch_float16,
    float32 as torch_float32,
    uint8 as torch_uint8,
    device as torch_device,
    load as torch_load,
)
from torch.nn.functional import interpolate as F_interpolate
from PIL.Image import fromarray as Image_fromarray
from torch import (
    zeros as torch_zeros,
    as_tensor as torch_as_tensor,
    float16 as torch_float16,
    float32 as torch_float32,
    uint8 as torch_uint8,
    device as torch_device,
    load as torch_load,
)
from torch.nn.functional import interpolate as F_interpolate
from PIL.Image import fromarray as Image_fromarray
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from decord import VideoReader, cpu, gpu
from decord.bridge import set_bridge
from accelerate import Accelerator
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.constants import DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN


set_bridge('torch')
environ['DECORD_EOF_RETRY_MAX'] = '20480'


def load_video(vis_path, device, num_frm=100, n_clips=1):
    vr = VideoReader(vis_path, ctx=cpu(), num_threads=0) # with 1, it's 6-8 hours
    total_frame_num = len(vr)
    # Currently, this function supports only 1 clip
    assert n_clips == 1
    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frm)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx) # (n_clips*num_frm, H, W, 3)
    del vr
    h, w = 224, 224
    # If image shape is not as target, resize it
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch_as_tensor(img_array, dtype=torch_float32, device=device).permute(0, 3, 1, 2)
        img_array = F_interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(device='cpu', dtype=torch_uint8, non_blocking=True).numpy()
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # Convert numpy arrays to PIL Image objects
    clip_imgs = [Image_fromarray(img_array[0, j]) for j in range(total_num_frm)]
    return clip_imgs



def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def initialize_model(model_name, projection_path=None):
    """
    Initializes the model with given parameters.

    Parameters:
    model_name (str): Name of the model to initialize.
    projection_path (str, optional): Path to the projection weights. Defaults to None.

    Returns:
    tuple: Model, vision tower, tokenizer, image processor, vision config, and video token length.
    """

    # Disable initial torch operations
    disable_torch_init()

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    device_map = 'auto'
    # Convert model name to user path
    model_name = os_path_expanduser(model_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map)

    # Load model
    model = VideoChatGPTLlamaForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch_float16,
                                                         use_cache=True, device_map=device_map)

    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch_float16, device_map=device_map)

    # Set to use start and end tokens for video
    mm_use_vid_start_end = True

    # Add tokens to tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # Resize token embeddings of the model
    model.resize_token_embeddings(len(tokenizer))

    # Load the weights from projection_path after resizing the token_embeddings
    if projection_path:
        print(f"Loading weights from {projection_path}")
        status = model.load_state_dict(torch_load(projection_path, map_location='cpu'), strict=False)
        if status.unexpected_keys:
            print(f"Unexpected Keys: {status.unexpected_keys}.\nThe Video-ChatGPT weights are not loaded correctly.")
        print(f"Weights loaded from {projection_path}")

    # Set model to evaluation mode and move to GPU
    model = model.eval()
    model = model.cuda()

    vision_tower_name = "openai/clip-vit-large-patch14"

    # Load vision tower and move to GPU
    vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name, torch_dtype=torch_float16,
                                                   low_cpu_mem_usage=True).cuda()
    vision_tower = vision_tower.eval()

    # Configure vision model
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    # Set video token length
    video_token_len = 356

    return model, vision_tower, tokenizer, image_processor, video_token_len
