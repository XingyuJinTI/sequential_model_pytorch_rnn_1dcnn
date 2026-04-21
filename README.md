# Sequential Model Pytorch

The repo provides library supporting:
- RNN, LSTM, GRU, bi-GRU, 1d-CNN, RCNN, etc.
- Adaptable to variable length input sequence.

## Dataset Setup

This project uses the **UCI HAR Dataset** (Human Activity Recognition Using Smartphones).

### 1. Download the Dataset

- **UCI ML Repository:** https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- Direct download: https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

Extract the zip into a `data/` directory in the project root.

### 2. Generate TrainList & TestList

Run the dataset list generator to create the required list files:

```bash
python dataset_list.py --data_dir ./data/UCI_HAR_Dataset --output_dir ./data
```

This creates `TrainList.txt` and `TestList.txt` used by the data loader.

### 3. Update Paths

In `train.py` and `test.py`, ensure the paths point to your generated lists:

```python
TrainList = './data/TrainList.txt'
TestList  = './data/TestList.txt'
```

## Quick Start

Choose the type of neural network in `train.py` (RNN, LSTM, GRU, bi-GRU, 1d-CNN, RCNN).

To train:

```bash
python train.py
```

To test:

```bash
python test.py
```

## Models Supported

| Model | Description |
|-------|-------------|
| RNN | Vanilla Recurrent Neural Network |
| LSTM | Long Short-Term Memory |
| GRU | Gated Recurrent Unit |
| bi-GRU | Bidirectional GRU |
| 1d-CNN | 1D Convolutional Neural Network |
| RCNN | Recurrent Convolutional Neural Network |

## Requirements

```bash
pip install torch numpy
```
