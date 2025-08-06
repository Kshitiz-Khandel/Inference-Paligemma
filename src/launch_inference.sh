
MODEL_PATH="/home/jupyter/Paligemma2/google/paligemma-3b-pt-896"  

PROMPT="the dog is ,the building is ,the mountain is "

# Comma-separated list of image file paths (must match number of prompts)
IMAGE_FILE_PATH="/home/jupyter/Paligemma2/images/dog.jpeg,/home/jupyter/Paligemma2/images/building.jpeg,/home/jupyter/Paligemma2/images/mountains.jpeg"

MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="False"
ONLY_CPU="False"
ATTENTION_TYPE="GLOBAL"      ## SWA or GLOBAL

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompts "$PROMPT" \
    --image_file_paths "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
    --attention_type "$ATTENTION_TYPE" \
