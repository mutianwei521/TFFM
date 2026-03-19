<div align="center">
<h1>Bridging Physics and Data: A Hierarchical Graph Intelligence for Water Leak Localization</h1>

[**[Paper]**]() [**[Code]**]()

</div>

This is the official code for the paper **Physics-Informed Graph Intelligence Enables Interpretable Leak Localization in Water Distribution Networks**.

We introduce **Tri-Focus Feature Matching (TFFM)**, a transformative Physics-Informed Graph Intelligence framework that embeds rigorous hydraulic constraints directly into the neural computation graph. TFFM acts as a "glass-box", solving the notoriously difficult inverse problem of leakage localization in Water Distribution Networks (WDNs).

---

## 🌟 News
- **[2026.03.18]** Code and pre-trained models are officially released!

---

## 💡 Abstract
Leakage localization in Water Distribution Networks (WDNs) is a critical inverse problem hindered by data scarcity and hydraulic non-linearity. Existing methods struggle to balance physical rigor with computational flexibility, often resulting in "black-box" models lacking interpretability. In this study, we propose **Tri-Focus Feature Matching (TFFM)**. Our innovation lies in a hierarchical "Look-Three-Times" mechanism that progressively constrains the solution space: 
1. **Global** graph embeddings detect system-wide anomalies (Look 1); 
2. **Regional** variational attention maximizes mutual information to isolate hydraulically coherent zones (Look 2); 
3. A novel **Physics-Informed Node Localizer (PINL)** integrates the hydraulic sensitivity matrix as a hard inductive bias (Look 3). 

This architecture ensures predictions are not merely statistically likely but physically consistent with fluid mechanics. Validated on benchmark networks (e.g., L-Town), TFFM achieves near-perfect accuracy and superior resilience to sensor noise.

---

## 🚀 Architecture overview

TFFM progressively narrows the diagnostic scope from global leak detection (Look 1), to regional sub-partition isolation (Look 2), and finally to precise nodal pinpointing (Look 3), establishing a scale-invariant "glass-box" interpretability.
![Overall Network Architecture](figure/framework.png)


---

## 📈 Main Results

**Comparative Classification Performance on L-Town Network**

| Model Type | Accuracy (Look 1) | Recall (Look 2) | Top-1 Accuracy (Look 3) |
| :--- | :--- | :--- | :--- |
| Physics-Based (RPM) | N/A | 72.1% | 45.6% |
| Data-Driven (CNN) | 94.5% | 88.2% | 62.8% |
| Data-Driven (GCN) | 98.2% | 89.3% | 74.1% |
| **TFFM (Ours)** | **100%** | **100%** | **98.2%** |

TFFM exhibits remarkable resilience, maintaining 96.5% localization accuracy even under extreme high-noise regimes (sensor noise $\sigma=0.5m$), significantly outperforming purely data-driven baselines.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.8+
- PyTorch (>= 2.0.0)
- EPANET 2.2
- CUDA 11.8
- Other dependencies in `requirements.txt`

Clone the repo and install dependencies:
```bash
git clone https://github.com/yourusername/TFFM-WaterNetwork.git
cd TFFM-WaterNetwork
pip install -r requirements.txt
```

### 2. Data Preparation
For default execution, the network topologies (e.g., `Net1.inp` or `L-Town.inp`) and configurations are specified via `config.yaml`.
- Ensure your structural graph representations and hydraulic simulations are properly placed in `data/`.
- Perform the Phase I Offline Initialization by calculating the sensitivity matrix.

### 3. Training
To train the TFFM model using default configurations:
```bash
python main.py --mode train
```
Or use custom parameters (e.g., number of scenario data augmentations):
```bash
python main.py --mode train --n-scenarios 1000
```
This script handles the end-to-end composite loss optimization for the Global Adaptor, regional LTFM layers, VT Selector, and finally the Physics-Informed Node Localizer.

### 4. Evaluation & Inference
During inference, TFFM applies its "Look-Thrice" cascade seamlessly.
```bash
# Real-time monitoring mode
python main.py --mode inference

# Batch prediction with external test data
python main.py --mode inference --test-data data/test.csv
```

---

## 📂 Project Structure

```text
TFFM-WaterNetwork/
├── data/                  # WDN topological files (*.inp) and datasets
├── logs/                  # Training logs
├── output/                # Evaluation output and results
├── src/                   # Core TFFM source code
│   ├── data/              # Hydraulic data processors and EPANET wrapper
│   ├── models/            # TFFM models: Adaptor, LTFM Layers, VT Selector, PINL
│   ├── training/          # Composite loss functions and trainer logic
│   ├── inference/         # Prediction engine
│   └── utils/             # Graph2Vec encoders, FCM Partitioning
├── config.yaml            # Hyperparameters and running configuration
├── main.py                # Main executable program
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

---

## 🏷️ Citation

waiting to continue

---

## 📝 License
This project is released under the [MIT License](LICENSE). See the LICENSE file for more details.

---

## 👏 Acknowledgement
We extend our gratitude to the PyTorch team and the contributors of EPANET frameworks which greatly facilitated this research into intelligent WDNs.
