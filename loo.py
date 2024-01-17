'''
Leave-one-out training.
python loo.py
'''
from dataclasses import dataclass, field
from typing import Tuple
from tqdm import tqdm, trange
from torch import (
    no_grad as torch_no_grad,
    zeros as torch_zeros,
    bfloat16 as torch_bfloat16,
    float32 as torch_float32,
    int64 as torch_int64,
    as_tensor as torch_as_tensor,
    zeros_like as torch_zeros_like,
    cat as torch_cat,
    save as torch_save,
    Tensor as torch_Tensor,
    FloatTensor as torch_FloatTensor,
    LongTensor as torch_LongTensor,
    BoolTensor as Torch_BoolTensor,
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
@dataclass
class LooArguments:
    topk: int = field()


def get_tokenizer_model(which: str, model_args, training_args, device_map):
    match which:
        case 'pretrained':
            model_name_or_path = model_args.model_name_or_path_pretrained
        case 'untrained':
            model_name_or_path = model_args.model_name_or_path_untrained
        case _:
            raise ValueError(f'Unknown which: {which}')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        device_map=device_map,
    )

    model = VideoChatGPTLlamaForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        torch_dtype=torch_bfloat16 if training_args.bf16 else torch_float32,
    )

    return tokenizer, model


def setup(which):
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LooArguments))
    model_args, data_args, training_args, loo_args = parser.parse_args_into_dataclasses()

    device_index = Accelerator().process_index
    device_map = {"": device_index}
    tokenizer, model = get_tokenizer_model(which, model_args, training_args, device_map)

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
    return model, tokenizer, data_module, model_args, loo_args
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # trainer.train()
    # trainer.save_state()
    # return trainer


class LOOTrainer:
    def __init__(self, dataloader_eval_loo, model_untrained, model_pretrained, lr: float, topk: int, num_frames: int):
        self.model_untrained = model_untrained
        self.model_pretrained = model_pretrained
        self.device = model_untrained.device
        self.lr = lr
        self.topk = topk
        self.dataloader_eval_loo = dataloader_eval_loo
        self.loss_per_vid = None
        self.model_hash_init = self.hash()
        # Cache linear layer for easy restoration
        self.layer_weights, self.layer_bias = self.get_model_weights_biases()
        self.num_frames = num_frames

    def init_model(self):
        pass

    def init_dataloader_eval_loo(self):
        pass

    @torch_no_grad()
    def evaluate_loss(self) -> None:
        '''
        3. evalute on each video. Assume the dataloader batch size is 1. Model is not changed.
        '''
        assert self.loss_per_vid is None
        model = self.model_untrained
        model.eval()
        loss_per_vid = {}
        device = self.device
        for batch in self.dataloader_eval_loo:
            vid = batch.pop('id')
            assert len(vid) == 1
            vid = vid[0]
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, _ = model(**batch, return_dict=False)
            loss_per_vid[vid] = loss

        self.loss_per_vid = loss_per_vid
        assert self.check_model_hash()

    def hash(self):
        return hash(str(self.model_untrained))

    def check_model_hash(self):
        return self.hash() == self.model_hash_init

    def reset_model(self, is_training: bool):
        model = self.model_untrained
        # named_parameters = dict(model.named_parameters())
        # parameter_name_weight = named_parameters['model.mm_projector.weight']
        # parameter_name_bias = named_parameters['model.mm_projector.bias']
        # model_dict = {
        #     parameter_name_weight: self.layer_weights,
        #     parameter_name_bias: self.layer_bias,
        # }
        with torch_no_grad():
            # model[0].weight = nn.Parameter(torch.ones_like(model[0].weight))
            model.model.mm_projector.weight = self.layer_weights
            model.model.mm_projector.bias = self.layer_bias

        # model_dict = {
        #     'model.mm_projector.weight': self.layer_weights,
        #     'model.mm_projector.bias': self.layer_bias,
        # }
        # breakpoint()
        # model.load_state_dict(model_dict)
        # set_parameter = model.set_parameter
        # set_parameter("model.mm_projector.weight", self.layer_weights)
        # set_parameter("model.mm_projector.bias", self.layer_bias)
        model.train(is_training)
        assert self.check_model_hash()

    def get_model_weights_biases(self) -> Tuple:
        get_parameter = self.model_untrained.get_parameter
        layer_weights = get_parameter("model.mm_projector.weight")
        layer_bias = get_parameter("model.mm_projector.bias")
        # return {'weight': layer_weights, 'bias': layer_bias}
        return layer_weights, layer_bias

    def run(self):
        '''
        For all videos
        '''
        loss_per_vid = self.loss_per_vid
        # Return: dict_score_per_vid
        dict_score_per_vid = {}
        run_batch = self.run_batch
        for batch in self.dataloader_eval_loo:
            # vid = batch.pop('id')
            vid = batch['id']
            assert len(vid) == 1
            vid = vid[0]
            loss_0 = loss_per_vid[vid]
            dict_score_qid = run_batch(batch, loss_0)
            dict_score_per_vid[vid] = dict_score_qid
        return dict_score_per_vid
    # TODO: import __getitem__?

    def run_batch(self, batch, loss_0):
        model = self.model_untrained
        topk = self.topk
        device  = loss_0.device
        # each vid (295) has 1 score for each top-k frames. [295, k] # [295, k], but now it's just k.
        scores_topk = torch_zeros(topk, device=device, dtype=torch_float32)
        # each vid (295) has 1 argmax (the frame index of each top-k frames)
        scores_topk_indices = torch_zeros(topk, device=device, dtype=torch_int64) # [295, k]. But now it's just k.
        vid = batch['id']
        # qid_val = batch['qid']
        qid_val = 'TODO'
        frames_original = batch['video_spatio_temporal_features'].squeeze()
        # frames_ablated = ablate_frame_all(frames_original)
        # For each frame, construct a new batch
        train_model_on_batch = self.train_model_on_batch
        for idx in trange(self.num_frames):
            # Reset batch
            frames = ablate_frames(frames_original, idx)
            batch['video_spatio_temporal_features'] = frames
            # D_minus_i = frames_ablated[i]
            # Reset model
            self.reset_model(True) # Retrain on batch? Where does the batch even go?
            loss = train_model_on_batch(batch)
            score = absdiff(loss, loss_0)
            # if scores is good, Save parameters
            if score_is_good(scores_topk, score):
                frame_idx = f'frame_idx_{idx}'
                save_linear_layer_weights(model, vid, qid_val, frame_idx)
                # breakpoint()
                # (Pdb) scores_topk.device
                # device(type='cpu')
                # (Pdb) scores_topk_indices.device
                # device(type='cpu')
                # (Pdb) score.device
                # device(type='cuda', index=0)
                # (Pdb)
                maintain_topk(scores_topk, scores_topk_indices, score)
        dict_score_qid = {
            'qid': qid_val,
            'scores_topk': scores_topk,
            'scores_topk_indices': scores_topk_indices,
        }
        return dict_score_qid

    def train_model_on_batch(self, batch: dict) -> torch_FloatTensor:
        '''
        Train each model only on a single data point.
        '''
        model = self.model_untrained
        optimizer = AdamW(model.parameters(), lr=self.lr)
        optimizer.zero_grad(set_to_none=True)
        device = self.device
        if 'id' in batch:
            batch.pop('id')
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss, _ = model(**batch, return_dict=False)
        loss.backward()
        optimizer.step()
        return loss


# @torch_jit_script
def ablate_frames(video_spatio_temporal_features: torch_Tensor, t: int):
    # [100, 1024]
    # video_spatio_temporal_features_clone = video_spatio_temporal_features.detach().clone()
    temporal_features = video_spatio_temporal_features[:100, :]
    # [256, 1024]
    spatial_features = video_spatio_temporal_features[100:, :]
    temporal_features_t = temporal_features[t, :]  # [1024]
    temporal_features[t, :].zero_()
    spatial_features -= temporal_features_t * 256/100  # Approximation
    return torch_cat((temporal_features, spatial_features), dim=0).unsqueeze(0)


# @torch_jit_script
def absdiff(loss: torch_FloatTensor, loss_0: torch_FloatTensor) -> torch_FloatTensor:
    return (loss-loss_0).abs()


# @torch_jit_script
def score_is_good(scores_topk: torch_FloatTensor, score: torch_FloatTensor) -> Torch_BoolTensor:
    # scores_topk is always sorted
    # Example 1: scores_topk = torch.as_tensor([0, 0, 0]); score = torch.as_tensor(2). biggest_where = [True, True, True] scores_topk = [2, 0, 0]
    return score > scores_topk[-1]


# @torch_jit_script
def maintain_topk(scores_topk: torch_FloatTensor, scores_topk_indices: torch_LongTensor, score: torch_Tensor):
    scores = torch_cat((scores_topk, score.unsqueeze(-1)))
    # scores_topk = torch_topk(scores, scores_topk.size(0), out=(scores_topk, scores_topk_indices))
    scores_topk, scores_topk_indices = torch_topk(scores, scores_topk.size(0))
    # scores_topk = torch_topk(scores, scores_topk.size(0))
    # return scores_topk
# def evaluate_and_save_frames(model, frames: torch_Tensor):
#     losses_minus_i_all_cs = []
#     for c in range(frames.size(0)):
#         # For each
#         D_minus_i = frames[c]
#         losses_minus_i_per_c = evaluate_on_datapoint(model, D_minus_i, False)
#         losses_minus_i_all_cs.append(losses_minus_i_per_c)
#     return torch_cat(losses_minus_i_all_cs) # [100, C, 100, 256, 1024]



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


# 4. Initialize an empty score list S.
def initialize_empty_score_list(loss_per_batch):
    losses_0 = torch_as_tensor(loss_per_batch.values(), dtype=torch_float32)
    # scores = torch_zeros_like(losses_0)
    return losses_0
    # return torch_zeros(len(loss_per_batch), dtype=torch_float32)


def main():
    # args = parse_args()
    # lr = args.lr
    lr = 1e-5
    model_untrained, _, data_module, model_args, loo_args = setup('untrained')
    model_pretrained, _, data_module, model_args, loo_args = setup('pretrained')
    topk = loo_args.topk
    # trainer = setup()
    # model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name)
    dataset_eval_loo = data_module['train_dataset']
    data_collator = data_module['data_collator']
    dataloader_eval_loo = DataLoader(dataset_eval_loo, collate_fn=data_collator)
    num_frames = model_args.num_frames

    loo_trainer = LOOTrainer(dataloader_eval_loo, model_untrained, model_pretrained, lr, topk, num_frames)

    loo_trainer.evaluate_loss()
    # 4. Initialize an empty score list S.
    dict_score_per_vid: dict = loo_trainer.run()
    print('finished loo training')
    return dict_score_per_vid


if __name__ == '__main__':
    main()
