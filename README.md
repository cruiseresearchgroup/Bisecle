# Bisecle: Binding and Separation in Continual Learning for Video Language Understanding

This repository provides the official implementation of **Bisecle**,

> **Bisecle: Binding and Separation in Continual Learning for Video Language Understanding**  
> Yue Tan, Xiaoqian Hu, Hao Xue, Celso M de Melo, Flora D. Salim 
> <!-- > **NeurIPS**, 2025   -->
> [[Paper](https://arxiv.org/pdf/2507.00469)]

Bisecle is a neurobiology-inspired continual learning framework specifically designed for multimodal video-language understanding tasks. It addresses the critical issues of **catastrophic forgetting**, **update conflict**, and **semantic overlap** across sequential video question answering tasks. The model leverages a multi-directional supervision module alongside a contrastive prompt learning mechanism, significantly enhancing model stability and generalization capabilities.


## 🧠 Core Contributions
1. **Multi-Directional Supervision Module**

Inspired by the mechanism of hippocampus to achieve rapid binding over multiple modalities, linking multimodal information dynamically into cohesive memory.

2. **Contrastive Prompt Learning**

Addresses inter-task prompt conflicts using contrastive learning, promoting semantic separation and preventing forgetting.


## 🏗️ Overall Framework




## 📁 Project Structure
```bash
bisecle/
├── dataloader/ # Dataset configuration training and evaluation
├── llama/ # Tokenizer, implementation of model components (adapter layers, prompts, CL modules), model forward pass
├── util/ # Logger, scheduler, etc.
├── engine.py # Training engine including train_one_epoch and forward pass logic
├── lama_vqa.py # Define the baseline model for LLaMA-Adaptor
├── setup.sh # Environment setup file
├── train_dramaQA.py # Main training script for dramaQA
├── train_nextQA.py # Main training script for nextQA
└── train_starQA.py # Main training script for starQA
```

## 🔧 Installation

1. **Clone the repository:**

    ```
    git clone https://github.com/cruiseresearchgroup/bisecle.git
    cd bisecle
    ```

2. **Setup environment (recommended):**

    ```
    conda create -n bisecle python=3.10 -y
    conda activate bisecle
    ```

3. **Install dependencies:**

    ```
    bash setup.sh
    ```

    *Alternatively, you can manually install the modules in setup.sh:*
    
    
## 🔧 Running the experiments


The baseline experiment trains the model in the conventional way.

* To fine-tune the model with Bisecle on NExT-QA:
```
python train_nextQA.py --batch_size 32 --adapter_layer 32 --tqcp_weight 0.1 --weight_decay 0.14
```
* To fine-tune the model with Bisecle on DramaQA:
```
python train_dramaQA.py --batch_size 4 --adapter_layer 32 --tqcp_weight 1 --weight_decay 0.1
```
* To fine-tune the model with Bisecle on STAR:
```
python train_starQA.py --batch_size 16 --adapter_layer 32 --tqcp_weight 0.1 --weight_decay 0.1
```

## 📊 Experimental Results


## 🔍 Ablation Studies

## 🎨 Qualitative Results


## 📝 Citation
If you find this work helpful, please consider citing:
```bash
@article{tan2025bisecle,
  title={Bisecle: Binding and Separation in Continual Learning for Video Language Understanding},
  author={Tan, Yue and Hu, Xiaoqian and Xue, Hao and De Melo, Celso and Salim, Flora D},
  journal={arXiv preprint arXiv:2507.00469},
  year={2025}
}
```
