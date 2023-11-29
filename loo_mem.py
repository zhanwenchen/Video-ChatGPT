# Need to call this before importing transformers.
from video_chatgpt.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from loo import main


replace_llama_attn_with_flash_attn()


if __name__ == "__main__":
    main()
