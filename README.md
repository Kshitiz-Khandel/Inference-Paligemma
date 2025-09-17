

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

## Speculative decoding(Create draft model)

```bash
python draft_model.py \
  --target_model_path "/home/jupyter/Paligemma/google/paligemma-3b-pt-896" \
  --draft_save_path "/home/jupyter/Paligemma/draft/draft_model_10" \
  --target_num_layers 10 \
  --apply_fixes True \
  --diagnose_only False \
  --device "cuda"


```

## Launch inferencing

```bash
./launch_inference.sh
  
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



