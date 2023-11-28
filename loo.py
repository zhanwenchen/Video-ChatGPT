'''
Leave-one-out training.
python loo.py \
    --qas_train_loo data/tomloc/qa/tomloc_train_loo_removed_merged_n3_with_frames_idx_instruction.json \
    --qas_eval_loo data/tomloc/qa/tomloc_eval_loo_removed_merged_n3_with_frames_idx_instruction.json \
    --topk 5 \
    --lr 1e-5 \
    --model_name tomloc_checkpoints_1/checkpoint-400


    PYTHONPATH="./:$PYTHONPATH" python video_chatgpt/eval/run_inference_tomloc_qa.py \
    --model-name tomloc_checkpoints_1/checkpoint-400 \
    --video_dir data/tomloc/video_merged_n3 \
    --gt_file_qa data/tomloc/qa/tomloc_val_removed_merged_n3_with_frames_idx_instruction.json \
    --output_dir data/tomloc/output \
    --output_name video_chatgpt_tomloc_qa_preds_val
'''
from argparse import ArgumentParser
from collections import defaultdict
from json import load as json_load, dump as json_dump
from tqdm import tqdm, trange
from torch import (
    no_grad as torch_no_grad,
    zeros as torch_zeros,
    float32 as torch_float32,
    as_tensor as torch_as_tensor,
    zeros_like as torch_zeros_like,
    cat as torch_cat,
    save as torch_save,
    Tensor as torch_Tensor,
    where as torch_where,
    topk as torch_topk,
)
from torch.jit import script as torch_jit_script
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import Trainer, HfArgumentParser, TrainingArguments as HFTrainingArguments, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
from accelerate import Accelerator
from video_chatgpt.train.train import make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt import video_conversation as conversation_lib
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM, VideoChatGPTLlamaForCausalLMLoo
# 1. For each video, select a (prompt, answer) as the evaluation set, and save it to disk. How do I do that? Think about it later. Maybe in the dataloader? Maybe in the split?
# 2. Train the model on all non-eval (prompt, answer)s and all 100 frames.
# 3. Evaluate the trained model. Evaluate the loss of the model on the evaluation set. Save the model and the loss.
# 4. Initialize an empty score list S.
# 5. For each video, For each frame x_i in the training set D,
# 5.1 Remove x_i from the retraining set D and retrain the model f on D_{-i} to have model f_{-i}
# 5.2 Compute loss on the x_t, or l_{-i} = CE(f_{-i}(x_t), y_t)
# 5.3 Compute score_i = |l_{-i} - l_0|
# 5.4 Put score_i into S.
# 5.5 Put back x_i into the set of training frames D.
# 5.6 Save the model parameters of f_{-i}.
# 6. Select the top-k values from S and we have their corresponding training frames.
# 7. Only save the top-k model parameters.
# 8. Each time we discard a frame, we need to train model from scratch. call backward. Clone from non-removed state.
# 9. Only save the linear layer weights.
# The leave-one out is about each retraining. Motivation is that we want to know which frame is the most important.
def parse_args():
    parser = ArgumentParser(description="Merging Videos")
    parser.add_argument("--topk", type=int, required=True, help='')
    parser.add_argument("--qas_train_loo_fpath", type=str, required=True, help='')
    parser.add_argument("--qas_eval_loo_fpath", type=str, required=True, help='')
    parser.add_argument("--lr", type=str, required=True, help='')
    parser.add_argument("--model-name", type=str, required=True, help='')
    return parser.parse_args()


def setup():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        device_map=device_map,
    )

    model = VideoChatGPTLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        # torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float,
    )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)


    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    model_vision_dict = model.get_model().initialize_vision_modules(
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )
    vision_config = model_vision_dict['vision_config']

    data_args.video_token_len = model_vision_dict['video_token_len']
    data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_vid_start_end = data_args.mm_use_vid_start_end = model_args.mm_use_vid_start_end
    vision_config.use_vid_start_end = model_args.mm_use_vid_start_end
    model.config.sep_video_conv_front = data_args.sep_video_conv_front
    model.initialize_vision_tokenizer(mm_use_vid_start_end=model_args.mm_use_vid_start_end, tokenizer=tokenizer,
                                      device=training_args.device, tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
                    len(params_no_grad), params_no_grad))
            else:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                        len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    return model, tokenizer, data_module
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # trainer.train()
    # trainer.save_state()
    # return trainer


def main():
    args = parse_args()
    topk = args.topk
    lr = args.lr
    model = None
    model, tokenizer, data_module = setup()
    # trainer = setup()
    # model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name)
    # dataset_train_loo = data_module['dataset_train']
    dataset_eval_loo = data_module['dataset_eval']

    # model = train_f(model, dataloader_train)
    model.eval()
    loss_0_by_qid = evaluate_with_loss(model, dataset_eval_loo)
    # 4. Initialize an empty score list S.
    # losses_0 = initialize_empty_score_list(loss_per_batch)
    dict_weights_and_biases = get_model_weights_biases(model)
    dict_score_per_vid: dict = compute_scores_and_save_model(model, loss_0_by_qid, dataset_eval_loo, dict_weights_and_biases, topk)
    print('finished loo training')
    return dict_score_per_vid


def compute_scores_and_save_model(model, loss_0_by_qid, dataloader_eval, dict_weights_and_biases, topk: int):
    # Return: dict_score_per_vid
    dict_score_per_vid = {}
    for batch_eval in tqdm(dataloader_eval):
        qid = batch_eval['qid']
        loss_0 = loss_0_by_qid[qid]
        dict_score_qid = compute_scores_and_save_model_single_video(model, loss_0, batch_eval, dict_weights_and_biases, topk)
        dict_score_per_vid[qid] = dict_score_qid
    return dict_score_per_vid


@torch_jit_script
def ablate_frames(video_spatio_temporal_features: torch_Tensor, t: int):
    # [100, 1024]
    # video_spatio_temporal_features_clone = video_spatio_temporal_features.detach().clone()
    temporal_features = video_spatio_temporal_features[:, :100, :]
    # [256, 1024]
    spatial_features = video_spatio_temporal_features[:, 100:, :]
    temporal_features_t = temporal_features[:, t, :]  # [1024]
    temporal_features[:, t, :].zero_()
    spatial_features -= temporal_features_t * 256/100  # Approximation
    return torch_cat((temporal_features, spatial_features), dim=1)


def compute_scores_and_save_model_single_video(model, loss_0, batch: dict, dict_weights_and_biases: dict, topk: int, optimizer):
    scores_topk = torch_zeros(topk, dtype=torch_float32)
    scores_topk_indices = torch_zeros(topk, dtype=torch_float32)
    vid = batch['vid']
    qid_val = batch['qid']
    frames_original = batch['frames']
    # frames_ablated = ablate_frame_all(frames_original)
    # For each frame, construct a new batch
    for idx in trange(frames_original.size(0)):
        # Reset batch
        frames = ablate_frames(frames_original, idx)
        batch['frames'] = frames
        # D_minus_i = frames_ablated[i]
        # Reset model
        reset_model(model, dict_weights_and_biases)
        # loss = model(batch)
        optimizer = AdamW(model.parameters, lr=lr)
        model, loss = train_model_on_batch(model, optimizer, batch)
        score = abs(loss - loss_0)
        # if scores is good, Save parameters
        if score_is_good(scores_topk, score):
            frame_idx = f'frame_idx_{idx}'
            save_linear_layer_weights(model, vid, qid_val, frame_idx)
            maintain_topk(scores_topk, scores_topk_indices, score)
    dict_score_qid = {
        'qid': qid_val,
        'scores_topk': scores_topk,
        'scores_topk_indices': scores_topk_indices,
    }
    return dict_score_qid


def train_model_on_batch(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch)
    loss.backward()
    optimizer.step()
    return model, loss


def score_is_good(scores_topk: torch_Tensor, score: torch_Tensor) -> bool:
    # scores_topk is always sorted
    # Example 1: scores_topk = torch.as_tensor([0, 0, 0]); score = torch.as_tensor(2). biggest_where = [True, True, True] scores_topk = [2, 0, 0]
    return score > scores_topk[0]


def maintain_topk(scores_topk: torch_Tensor, scores_topk_indices: torch_Tensor, score: torch_Tensor):
    scores = torch_cat(scores_topk, score)
    scores_topk = torch_topk(scores, scores_topk.size(0), out=(scores_topk, scores_topk_indices))

# def evaluate_and_save_frames(model, frames: torch_Tensor):
#     losses_minus_i_all_cs = []
#     for c in range(frames.size(0)):
#         # For each
#         D_minus_i = frames[c]
#         losses_minus_i_per_c = evaluate_on_datapoint(model, D_minus_i, False)
#         losses_minus_i_all_cs.append(losses_minus_i_per_c)
#     return torch_cat(losses_minus_i_all_cs) # [100, C, 100, 256, 1024]


def reset_model(model, dict_weights_biases: dict) -> None:
    model.set_parameter("model.mm_projector.weight", dict_weights_biases['weight'])
    model.set_parameter("model.mm_projector.bias", dict_weights_biases['bias'])


def get_model_weights_biases(model) -> dict:
    layer_weights = model.get_parameter("model.mm_projector.weight")
    layer_bias = model.get_parameter("model.mm_projector.bias")
    return {'weight': layer_weights, 'bias': layer_bias}


def save_linear_layer_weights(model, vid: str, qid_val: str, frame_idx: str) -> None:
# Get the weight of the first layer
    dict_weights_biases = get_model_weights_biases(model)
    dict_weights_biases['vid'] = vid
    dict_weights_biases['qid_val'] = qid_val
    dict_weights_biases['frame_idx'] = frame_idx
    suffix = f'vid={vid}_qid_val={qid_val}_frame_idx={frame_idx}'

    # Save the weight to a file
    torch_save(dict_weights_biases, f'mm_projector_{suffix}.pt')


# def ablate_frame_all(frames):
#     batch_size = frames.size(1)
#     frames.unsqueeze(0).expand() # [100, C, 100, 256, 1024]
#     for c in range(batch_size):
#         frames_c = frames[c]
#         for i in range(100):
#             frames_c[i, c, i, :, :] = 0

#     return frames

# 1.
def select_evaluation(dataloader_train):
    # Decision: do this before training, cache to disk.
    # dataloader: (prompt, answer, frames)
    # dataloader_train: (prompt, answer, frames)
    # dataloader_eval: (prompt, answer, frames)
    # Naively, can just return the last one.

    return dataloader_train, dataloader_eval


# 2. Train the model on all non-eval (prompt, answer)s and all 100 frames.
def train_f(model, dataloader_train):
    # Note: must train on batch_size = 1. Actually, maybe not, if I don't do loss reduction (reduction='none')
    # Save each loss.
    for batch_train in dataloader_train:
        train_model_on_batch(model, batch_train)
    # Save the model and the loss
    model.save()
    return model

# 3. evalute_on_datapoint():
@torch_no_grad()
def evaluate_with_loss(model, dataloader_eval):
    loss_per_batch = {}
    # running_loss = 0.0
    for batch_eval in dataloader_eval:
        qid = batch_eval['qid']
        loss = model(batch_eval)
        loss_per_batch[qid] = loss
        # running_loss += loss.item()

    return loss_per_batch


# 4. Initialize an empty score list S.
def initialize_empty_score_list(loss_per_batch):
    losses_0 = torch_as_tensor(loss_per_batch.values(), dtype=torch_float32)
    # scores = torch_zeros_like(losses_0)
    return losses_0
    # return torch_zeros(len(loss_per_batch), dtype=torch_float32)


if __name__ == '__main__':
    main()
