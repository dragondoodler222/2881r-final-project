# Remote Experiment Setup & Execution Guide

This guide details how to set up the Mafia RL environment on a remote server (e.g., AWS EC2, Azure VM, or university cluster) and run long-training experiments.

## 1. Prerequisites

- **SSH Access**: You should have `ssh` access to the remote machine.
- **GPU**: NVIDIA GPU with at least 16GB VRAM (for 7B models in 4-bit) or 24GB+ (for 8-bit/16-bit).
- **Storage**: At least 50GB free space for model weights and checkpoints.

## 2. Initial Setup

### Connect to Server
```bash
ssh user@your-remote-ip
```

### Install Conda (if not present)
If `conda` is not installed, install Miniconda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Clone Repository
```bash
git clone https://github.com/dragondoodler222/2881r-final-project.git
cd 2881r-final-project
```

### Create Environment
Create a dedicated environment to avoid conflicts:
```bash
conda create -n mafia_rl python=3.10 -y
conda activate mafia_rl
```

### Install Dependencies
```bash
# Install PyTorch with CUDA support (adjust cuda version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install project requirements
pip install -r requirements.txt
```

## 3. Model Configuration

### HuggingFace Login
You need to authenticate to download models like Llama-2 or Mistral.
1. Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Run:
```bash
huggingface-cli login
# Paste your token when prompted
```

### Verify GPU
Check if your GPU is visible:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 4. Running Experiments

### Quick Test
Run a short test to verify everything works before starting a long job.
```bash
python example_train.py
```
*If this runs for 1-2 minutes without error, you are ready.*

### Long-Running Training (Background)
Use `nohup` to keep the process running even if you disconnect SSH.

```bash
# Run in background and redirect output to nohup.out (logs are also saved to logs/training.log)
nohup python example_train.py > nohup.out 2>&1 &

# Get the Process ID (PID)
echo $!
```

### Monitoring

**Check Logs:**
```bash
# View real-time output
tail -f logs/training.log

# Or view the nohup output
tail -f nohup.out
```

**Monitor GPU Usage:**
```bash
watch -n 1 nvidia-smi
```

## 5. Managing the Experiment

### Stopping the Training
If you need to stop the experiment:
```bash
# Find the PID
ps aux | grep example_train.py

# Kill the process
kill <PID>
```

### Syncing Results
To download checkpoints or logs to your local machine:
```bash
# Run this ON YOUR LOCAL MACHINE
scp -r user@your-remote-ip:~/2881r-final-project/logs ./local_logs
scp -r user@your-remote-ip:~/2881r-final-project/checkpoints ./local_checkpoints
```

## 6. Troubleshooting

**CUDA Out of Memory (OOM)**
- **Solution**: Enable 4-bit quantization in `example_train.py`:
  ```python
  "use_4bit": True
  ```
- **Solution**: Reduce batch size or `max_tokens` in `LLMAgent`.

**"Killed" message**
- Usually means System RAM OOM.
- **Solution**: Increase swap space or use a machine with more RAM.

**HuggingFace Download Error**
- **Solution**: Ensure you ran `huggingface-cli login` and have accepted the model license on the HuggingFace website (common for Llama-2).
