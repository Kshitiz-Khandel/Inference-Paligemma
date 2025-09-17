MODEL_PATH="/home/jupyter/Paligemma/google/paligemma-3b-pt-896"  
DRAFT_MODEL_PATH="/home/jupyter/Paligemma/draft/draft_model_10"

PROMPT="the building is "

# Pass multiple --image_file_path arguments
python inference.py \
    --target_model_path "$MODEL_PATH" \
    --draft_model_path "$DRAFT_MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path /home/jupyter/Inference-Paligemma/Images/building.jpeg \
    --max_tokens_to_generate 100 \
    --temperature 0.8 \
    --top_p 0.9 \
    --do_sample True \
    --only_cpu False \
    --speculate_k 4