

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

## Inference with Quantization

```bash
python optimized_quantized_inference.py   --target_model_path "/home/jupyter/Paligemma/google/paligemma-3b-pt-896"   --prompt "The building is "   --image_file_path "/home/jupyter/Inference-Paligemma/Images/building.jpeg"   --target_quantization "int8"   --group_size 64   --do_sample True   --max_tokens_to_generate 50   --temperature 0.8   --top_p 0.9   --save_quantized "/home/jupyter/Paligemma/quantized/int8"   --enable_performance_monitoring True   --benchmark_performance True   --use_torch_compile True   --enable_optimizations True
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



