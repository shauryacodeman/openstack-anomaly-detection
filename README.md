# Distributed-system anomaly detection

This data mining project compares three neural detectors for unusual activity in logs and service calls. It follows the temporal and spatial approach described by [Xu et al.](https://doi.org/10.1016/j.eswa.2023.121675)

| Model | Input |
| --- | --- |
| Event order only | A bidirectional LSTM reads normalized log messages in time order. |
| Service transitions only | An attention CNN reads a matrix of service calls. |
| Both together | The two models' features are combined for one prediction. |

## Setup

Use Python 3.12 and `uv`. Run the commands from the repository root in a Linux or Ubuntu (WSL) terminal. Training and testing automatically use an NVIDIA GPU when CUDA is available to PyTorch and use the CPU otherwise.

### CPU laptop

Use the CPU build of PyTorch. This setup runs entirely on the laptop's processor.

```bash
uv venv --allow-existing --python 3.12 .venv
source .venv/bin/activate
uv pip install --index https://download.pytorch.org/whl/cpu torch
uv pip install -r requirements.txt
```

### NVIDIA GPU on WSL

Use the CUDA build of PyTorch with a compatible NVIDIA driver. The last command checks that PyTorch can see the GPU.

```bash
uv venv --allow-existing --python 3.12 .venv-cuda
source .venv-cuda/bin/activate
UV_HTTP_TIMEOUT=240 uv pip install --index https://download.pytorch.org/whl/cu128 torch==2.9.1
uv pip install -r requirements.txt
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
```

## Train and test with RCAEval

The repository includes the logs, traces, and fault injection times for all 30 Online Boutique runs in RCAEval RE3-OB. The bundled dataset is about 128 MiB. After installing the dependencies, run the experiment from the repository root. [Dataset source and license](data/rcaeval/README.md)

```bash
python -u train.py --dataset rcaeval
python -u evaluate.py --dataset rcaeval
```

The same training and testing commands work on both devices. Look for `Training device: CPU` or the NVIDIA GPU name at startup. Training prints the loss after each epoch and saves `rcaeval_re3ob_neural.pth` in the repository root.

For a quick demo on a CPU laptop, copy a trained `rcaeval_re3ob_neural.pth` into the repository root and run the evaluation command. A checkpoint trained on the GPU can be evaluated on CPU.

### How the comparison works

- Each run supplies a five-minute normal window and a five-minute faulty window. Both stay at least one minute away from the recorded injection time.
- For each scenario, the first two runs go into training and the third goes into testing. That gives 20 normal and 20 faulty training windows, plus 10 of each for testing.
- The service-call matrix records call counts, average durations, and error rates. Trace features are scaled using healthy training windows.
- Each alarm cutoff is chosen using validation predictions from the training runs. The cutoff allows up to 30% false alarms on those validation windows. The test false-alarm rate is measured separately.

### Results on the RTX 4060

| Model | Faults caught | Faults missed | False alarms | F1 |
| --- | ---: | ---: | ---: | ---: |
| Event order only | 10 | 0 | 7 | 0.741 |
| Service transitions only | 5 | 5 | 6 | 0.476 |
| Both together | 9 | 1 | 5 | **0.750** |

The combined model raised fewer false alarms than the logs-only model and missed one additional fault. These results are exploratory because the test split was examined during development. It contains 10 repeats of known injected fault scenarios and 10 normal windows.

Evaluating this GPU checkpoint on CPU reproduced the table above. A separate CPU-only training run caught nine faults and raised six false alarms with the combined model.

## OpenStack baseline

Extract the [Loghub OpenStack archive](https://zenodo.org/records/8196385/files/OpenStack.tar.gz?download=1) into `loghub/OpenStack/`. Run `python -u train.py --dataset openstack` and then `python -u evaluate.py --dataset openstack`. This dataset uses service transitions inferred from logs and contains four labeled affected VMs. All three neural models missed the three fault windows in its small test split.

## Code checks

```bash
python -m unittest -q
```

## Author

[Shaurya Pradhan](https://github.com/shauryacodeman)
