

## Setup


### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Kshitiz-Khandel/Inference-Paligemma.git
   cd src
   ```

2. Install the required dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

   pip install -r requirements.txt
   sudo apt-get update
   sudo apt-get install --reinstall libc-bin
   export PATH=$PATH:/sbin:/usr/sbin
   ```

3. Download the model weights:
   Weights can be found at [HuggingFace](https://huggingface.co/google/paligemma-3b-pt-896/tree/main)

   For pretrained only model (made for further finetuning)
   ```bash
   pip install huggingface_hub && python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/paligemma-3b-pt-896',    local_dir='/home/jupyter/Paligemma/google/paligemma-3b-pt-896')"

   ```

## Inference

```bash
./launch_inference.sh
```

### Configuration

You can modify the `launch_inference.sh` script to customize the inference parameters:

```bash
#!/bin/bash

#Path to model weights
MODEL_PATH="/home/jupyter/Paligemma/google/paligemma-3b-pt-896"  

#Text prompt for the model
PROMPT="the dog is ,the building is ,the mountain is "   

#Comma-separated list of image file paths (must match number of prompts)
IMAGE_FILE_PATH="/home/jupyter/Paligemma2/images/dog.jpeg,/home/jupyter/Paligemma2/images/building.jpeg,/home/jupyter/Paligemma2/images/mountains.jpeg"  

# Maximum response length
MAX_TOKENS_TO_GENERATE=100 

# Temperature for sampling
TEMPERATURE=0.8

# Top-p sampling parameter
TOP_P=0.9

# Whether to use Greedy decoding or Top P
DO_SAMPLE="False"

# Whether to use CPU only
ONLY_CPU="False"




python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \
```

## Citation

This work builds upon the following repositories:

```
@misc{gemma_pytorch,
  author = {Google},
  title = {Gemma PyTorch},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/google/gemma_pytorch}
}

@misc{pytorch_paligemma,
  author = {hkproj},
  title = {PyTorch Paligemma},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/hkproj/pytorch-paligemma}
}
```



