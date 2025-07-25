# Awesome Spiking Neural Networks[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

Collect some spiking neural network papers & codes.  (**Actively keep updating**)

If you own or find some overlooked SNN papers, you can add them to this document by pull request. 

## News

<details>

<summary> News 2025 </summary>

[2025.05.26] Update SNN-related papers in ICML 2025 (18 papers).

[2025.04.11] Update SNN-related papers in ICLR 2025 (11 papers), CVPR 2025 (14 papers).

[2025.02.06] Update SNN-related papers in AAAI 2025 (18 papers).

</details>

<details>

<summary> News 2024 </summary>

[2024.11.13] Update SNN-related papers in NeurIPS 2024 (18 papers).

[2024.10.31] Update SNN-related papers in ACM MM 2024 (5 papers).

[2024.10.15] Update SNN-related papers in ECCV 2024 (8 papers).

[2024.05.29] Update SNN-related papers in ICML 2024 (13 papers), IJCAI 2024 (5).

[2024.04.29] Update SNN-related papers in ICLR 2024 (17 papers), AAAI 2024 (8), CVPR 2024 (3).

</details>


<details>

<summary> News 2023 </summary>


[2023.12.31] Update SNN-related papers in TPAMI 2023, Frontiers in Neuroscience 2023.

[2023.10.31] Update SNN-related papers in CVPR 2023 (2 papers), ICML 2023 (2), IJCAI 2023 (3), and ICCV 2023 (10), NeurIPS 2023 (12).

[2023.06.25] Update SNN-related papers in ICLR 2023 (6 papers), AAAI 2023 (6 papers).

</details>



## Papers

### 2025
**AAAI, ICLR, CVPR, ICML**
- SpikeVideoFormer: An Efficient Spike-Driven Video Transformer with Hamming Attention and $\mathcal{O}(T)$ Complexity (**ICML 2025**). [[paper](https://arxiv.org/abs/2505.10352)]
- Efficient ANN-SNN Conversion with Error Compensation Learning (**ICML 2025**). [[paper](https://www.arxiv.org/abs/2506.01968)]
- Differential Coding for Training-Free ANN-to-SNN Conversion (**ICML 2025**). [[paper](https://arxiv.org/abs/2503.00301)] 
- Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/44825)] 
- ReverB-SNN: Reversing Bit of the Weight and Activation for Spiking Neural Networks (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/43640)] 
- TTFSFormer: A TTFS-based Lossless Conversion of Spiking Transformer (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/45087)] 
- BSO: Binary Spiking Online Optimization (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/45087)] 
- Delay-DSGN: A Dynamic Spiking Graph Neural Network with Delay Mechanisms for Evolving Graph (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/43816)] 
- TS-SNN: Temporal Shift Module for Spiking Neural Networks (**ICML 2025**). [[paper](https://arxiv.org/abs/2505.04165)] 
- SpikF: Spiking Fourier Network for Efficient Long-term Prediction (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/46411)] 
- Self-cross Feature based Spiking Neural Networks for Efficient Few-shot Learning (**ICML 2025**). [[paper](https://arxiv.org/pdf/2505.07921)] 
- Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/44986)] 
- Efficient Parallel Training Methods for Spiking Neural Networks with Constant Time Complexity (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/45776)] 
- Training High Performance Spiking Neural Networks by Temporal Model Calibration (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/44216)] 
- Temporal Misalignment in ANN-SNN Conversion and Its Mitigation via Probabilistic Spiking Neurons (**ICML 2025**). [[paper](https://icml.cc/virtual/2025/poster/45627)] 
- Time to Spike? Understanding the Representational Power of Spiking Neural Networks in Discrete Time (**ICML 2025**). [[paper](https://arxiv.org/abs/2505.18023)] 
- Hybrid Spiking Vision Transformer for Object Detection with Event Cameras (**ICML 2025**). [[paper](https://arxiv.org/abs/2505.07715)] 
- Sorbet: A Neuromorphic Hardware-Compatible Transformer-Based Spiking Language Model (**ICML 2025**). [[paper](https://arxiv.org/abs/2409.15298)] 
- EventGPT: Event Stream Understanding with Multimodal Large Language Models (**CVPR 2025**). [[paper](https://arxiv.org/pdf/2412.00832)]  [[code](https://github.com/XduSyL/EventGPT)]
- Spk2SRImgNet: Super-Resolve Dynamic Scene from Spike Stream via Motion Aligned Collaborative Filtering (**CVPR 2025**). [[paper](https://cvpr.thecvf.com/virtual/2025/poster/33079)]
- Decision SpikeFormer: Spike-Driven Transformer for Decision Making (**CVPR 2025**). [[paper](https://cvpr.thecvf.com/virtual/2025/poster/32864)]
- Self-Supervised Learning for Color Spike Camera Reconstruction (**CVPR 2025**). [[paper](https://cvpr.thecvf.com/virtual/2025/poster/34093)]
- USP-Gaussian: Unifying Spike-based Image Reconstruction, Pose Correction and Gaussian Splatting (**CVPR 2025**). [[paper](https://arxiv.org/abs/2411.10504)]
- VISTREAM: Improving Computation Efficiency of Visual Perception Streaming via Law-of-Charge-Conservation Inspired Spiking Neural Network (**CVPR 2025**). [[paper](https://cvpr.thecvf.com/virtual/2025/poster/34908)]
- Efficient ANN-Guided Distillation: Aligning Rate-based Features of Spiking Neural Networks through Hybrid Block-wise Replacement (**CVPR 2025**). [[paper](https://arxiv.org/abs/2503.16572)]
- Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer (**CVPR 2025**). [[paper](https://arxiv.org/abs/2503.00226)]
- Brain-Inspired Spiking Neural Networks for Energy-Efficient Object Detection (**CVPR 2025**). [[paper](https://cvpr.thecvf.com/virtual/2025/poster/33275)]
- Temporal Separation with Entropy Regularization for Knowledge Distillation in Spiking Neural Networks (**CVPR 2025**). [[paper](https://arxiv.org/abs/2503.03144)]
- STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks (**CVPR 2025**). [[paper](https://arxiv.org/abs/2503.02689)]
- Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients (**CVPR 2025**). [[paper](https://arxiv.org/abs/2503.03272)]
- Rethinking Spiking Self-Attention Mechanism: Implementing α-XNOR Similarity Calculation in Spiking Transformers (**CVPR 2025**). [[paper](https://cvpr.thecvf.com/virtual/2025/poster/33850)]
- Spiking Transformer with Spatial-Temporal Attention (**CVPR 2025**). [[paper](https://arxiv.org/abs/2409.19764)]
- Quantized Spike-driven Transformer (**ICLR 2025**). [[paper](https://openreview.net/forum?id=5J9B7Sb8rO)]
- SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training (**ICLR 2025**). [[paper](https://openreview.net/forum?id=L9eBxTCpQG)]
- Rethinking Spiking Neural Networks from an Ensemble Learning Perspective (**ICLR 2025**). [[paper](https://openreview.net/forum?id=ZyknpOQwkT)]
- DeepTAGE: Deep Temporal-Aligned Gradient Enhancement for Optimizing Spiking Neural Networks (**ICLR 2025**). [[paper](https://openreview.net/forum?id=drPDukdY3t)]
- QP-SNN: Quantized and Pruned Spiking Neural Networks (**ICLR 2025**). [[paper](https://openreview.net/forum?id=MiPyle6Jef)]
- Temporal Flexibility in Spiking Neural Networks: Towards Generalization Across Time Steps and Deployment Friendliness (**ICLR 2025**). [[paper](https://openreview.net/forum?id=9HsfTgflT7)]
- P-SpikeSSM: Harnessing Probabilistic Spiking State Space Models for Long-Range Dependency Tasks (**ICLR 2025**). [[paper](https://openreview.net/forum?id=Sf4ep9Udjf)]
- TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting (**ICLR 2025**). [[paper](https://openreview.net/forum?id=rDe9yQQYKt)]
- Improving the Sparse Structure Learning of Spiking Neural Networks from the View of Compression Efficiency (**ICLR 2025**). [[paper](https://openreview.net/forum?id=gcouwCx7dG)]
- SpikeLLM: Scaling up Spiking Neural Network to Large Language Models via Saliency-based Spiking (**ICLR 2025**). [[paper](https://openreview.net/forum?id=ZadnlOHsHv)]
- Spiking Vision Transformer with Saccadic Attention (**ICLR 2025**). [[paper](https://openreview.net/forum?id=qzZsz6MuEq)]
- SpikeGS: Reconstruct 3D scene captured by a fast moving bio-inspired camera (**AAAI 2025**). [[paper](https://arxiv.org/pdf/2407.03771v2)]
- Rethinking High-speed Image Reconstruction Framework with Spike Camera (**AAAI 2025**). [[paper](https://arxiv.org/pdf/2501.04477)] [[code](https://github.com/chenkang455/SpikeCLIP)]
- Spiking Point Transformer for Point Cloud Classification (**AAAI 2025**).
- Efficient 3D Recognition with Event-driven Spike Sparse Convolution (**AAAI 2025**).[[paper](https://arxiv.org/pdf/2412.07360)] [[code](https://github.com/bollossom/e-3dsnn)]
- GRSN: Gated Recurrent Spiking Neurons for POMDPs and MARL (**AAAI 2025**).[[paper](https://arxiv.org/pdf/2404.15597)]
- EventZoom: A Progressive Approach to Event-Based Data Augmentation for Enhanced Neuromorphic Vision (**AAAI 2025**).[[paper](https://arxiv.org/pdf/2405.18880)]
- Leveraging Asynchronous Spiking Neural Networks for Ultra Efficient Event-Based Visual Processing (**AAAI 2025**).
- CREST: An Efficient Conjointly-trained Spike-driven Framework for Event-based Object Detection Exploiting Spatiotemporal Dynamics  (**AAAI 2025**).[[paper](https://arxiv.org/pdf/2412.12525)] [[code](https://github.com/shen-aoyu/CREST/)]
- UCF-Crime-DVS: A Novel Event-Based Dataset for Video Anomaly Detection with Spiking Neural Networks (**AAAI 2025**).
- SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking State Space Models (**AAAI 2025**). [[paper](https://arxiv.org/pdf/2408.14909)][[code](https://github.com/shenshuaijie/SDN
)]
- Advancing Spiking Neural Networks towards Multiscale Spatiotemporal Interaction Learning (**AAAI 2025**). [[paper](https://arxiv.org/pdf/2405.13672)]
- SpikingYOLOX: Improved YOLOX Object Detection with Fast Fourier Convolution and Spiking Neural Networks(**AAAI 2025**).
- ALADE-SNN: Adaptive Logit Alignment in Dynamically Expandable Spiking Neural Networks for Class Incremental Learning (**AAAI 2025**).[[paper](https://arxiv.org/pdf/2412.12696)]
- Efficient Spike-driven Transformer For High-performance Image Segmentation (**AAAI 2025**).[[paper](https://arxiv.org/pdf/2412.14587)] [[code](https://github.com/BICLab/Spike2Former)]
- Towards Accurate Binary Spiking Neural Networks: Learning with Adaptive Gradient Modulation Mechanism (**AAAI 2025**).
- Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks (**AAAI 2025**).
- Towards More Discriminative Feature Learning in SNNs with Temporal-Self-Erasing Supervision (**AAAI 2025**).
- FSTA-SNN: Frequency-based Spatial-Temporal Attention Module for Spiking Neural Networks (**AAAI 2025**).[[paper](https://arxiv.org/pdf/2501.14744)] [[code](https://github.com/yukairong/FSTA-SNN)]

### 2024

**Review**
- Direct Training High-Performance Deep Spiking Neural Networks: A Review of Theories and Methods (**Frontiers in Neuroscience 2024**). [[paper]](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full) [[arxiv](https://arxiv.org/abs/2405.04289v2)] 

**NeurIPS, ACM MM, ECCV, AAAI, ICLR, Frontiers in Neuroscience, CVPR, ICML, IJCAI**
- SpikedAttention: Training-Free and Fully Spike-Driven Transformer-to-SNN Conversion with Winner-Oriented Spike Shift for Softmax Operation (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=fs28jccJj5)]
- Spiking Graph Neural Network on Riemannian Manifolds (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=VKt0K3iOmO)]
- Rethinking the Dynamics of Spiking Neural Networks (**NeurIPS 2024**). [[paper](https://neurips.cc/virtual/2024/poster/96543)]
- Long-Range Feedback Spiking Network Captures Dynamic and Static Representations of the Visual Cortex under Movie Stimuli (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=bxDok3uaK6)] [[code](https://github.com/Grasshlw/SNN-Neural-Similarity-Movie)]
- Take A Shortcut Back: Mitigating the Gradient Vanishing for Training Spiking Neural Networks (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=xjyU6zmZD7)]
- Advancing Training Efficiency of Deep Spiking Neural Networks through Rate-based Backpropagation (**NeurIPS 2024**). [[paper](https://arxiv.org/abs/2410.11488)] [[code](https://github.com/Tab-ct/rate-based-backpropagation)]
- Latent Diffusion for Neural Spiking Data (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=ZX6CEo1Wtv)]
- Autonomous Driving with Spiking Neural Networks (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=95VyH4VxN9)] [[code](https://github.com/ridgerchu/SAD)]
- Exact Gradients for Stochastic Spiking Neural Networks Driven by Rough Signals  (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=mCWZj7pa0M)]
- Spatio-Temporal Interactive Learning for Efficient Image Reconstruction of Spiking Cameras  (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=S4ZqnMywcM)]
- Slack-Free Spiking Neural Network Formulation for Hypergraph Minimum Vertex Cover (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=4A5IQEjG8c)]
- EnOF: Training Accurate Spiking Neural Networks via Enhancing the Output Feature Representation (**NeurIPS 2024**). [[paper](https://openreview.net/pdf/5a4dfaf8dc6861efa8e8356b3bd86743ab98838d.pdf)]
- Spiking Token Mixer: A event-driven friendly Former structure for spiking neural networks (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=iYcY7KAkSy)] [[code](https://github.com/brain-intelligence-lab/STMixer_demo)]
- SpGesture: Source-Free Domain-adaptive sEMG-based Gesture Recognition with Jaccard Attentive Spiking Neural Network (**NeurIPS 2024**). [[paper](https://arxiv.org/abs/2405.14398)] [[code](https://github.com/guoweiyu/SpGesture/)]
- Spiking Transformer with Experts Mixture (**NeurIPS 2024**). [[paper](https://openreview.net/pdf/35a5bc54de368426f66605d8e3f447638863888a.pdf)] 
- FEEL-SNN: Robust Spiking Neural Networks with Frequency Encoding and Evolutionary Leak Factor (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=TuCQdBo4NC)] [[code](https://github.com/zju-bmi-lab/FEEL_SNN)]
- Spiking Neural Network as Adaptive Event Stream Slicer (**NeurIPS 2024**). [[paper](https://arxiv.org/abs/2410.02249)] 
- Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators (**NeurIPS 2024**). [[paper](https://arxiv.org/abs/2405.14362)] [[code](https://github.com/microsoft/SeqSNN)]
- QKFormer: Hierarchical Spiking Transformer using Q-K Attention (**NeurIPS 2024**). [[paper](https://openreview.net/pdf?id=AVd7DpiooC)] [[code](https://github.com/zhouchenlin2096/QKFormer)]
- Q-SNNs: Quantized Spiking Neural Networks (**ACM MM 2024**). [[paper](https://dl.acm.org/doi/10.1145/3664647.3681186)]
- RSC-SNN: Exploring the Trade-off Between Adversarial Robustness and Accuracy in Spiking Neural Networks via Randomized Smoothing Coding (**ACM MM 2024**). [[paper](https://dl.acm.org/doi/10.1145/3664647.3680639)] [[code](https://github.com/KemingWu/RSC-SNN)]
- Reversing Structural Pattern Learning with Biologically Inspired Knowledge Distillation for Spiking Neural Networks (**ACM MM 2024**). [[paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3680655)]
- Towards High-performance Spiking Transformers from ANN to SNN Conversion (**ACM MM 2024**). [[paper](https://dl.acm.org/doi/10.1145/3664647.3680620)]  [[code](https://github.com/h-z-h-cell/Transformer-to-SNN-ECMT)]
- Towards Low-latency Event-based Visual Recognition with Hybrid Step-wise Distillation Spiking Neural Networks (**ACM MM 2024**). [[paper](https://dl.acm.org/doi/10.1145/3664647.3680832)]  [[code](https://github.com/hsw0929/HSD)]
- Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection (**ECCV 2024**). [[paper](https://arxiv.org/pdf/2407.20708)] [[code](https://github.com/BICLab/SpikeYOLO)]
- Spiking Wavelet Transformer (**ECCV 2024**). [[paper](https://arxiv.org/pdf/2403.11138)] [[code](https://github.com/bic-L/Spiking-Wavelet-Transformer)]
- Efficient Training of Spiking Neural Networks with Multi-Parallel Implicit Stream Architecture (**ECCV 2024**). [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05068.pdf)] [[code](https://github.com/kiritozc/MPIS-SNNs)]
- Asynchronous Bioplausible Neuron for Spiking Neural Networks for Event-Based Vision (**ECCV 2024**). [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08133.pdf)] 
- BKDSNN: Enhancing the Performance of Learning-based Spiking Neural Networks Training with Blurred Knowledge Distillation (**ECCV 2024**). [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06649.pdf)] [[code](https://github.com/Intelligent-Computing-Research-Group/BKDSNN)]
- Exploring Vulnerabilities in Spiking Neural Networks: Direct Adversarial Attacks on Raw Event Data (**ECCV 2024**). [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09164.pdf)]
- EAS-SNN: End-to-End Adaptive Sampling and Representation for Event-based Detection with Recurrent Spiking Neural Networks (**ECCV 2024**). [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07766.pdf)] [[code](https://github.com/Windere/EAS-SNN)]
- Spike-Temporal Latent Representation for Energy-Efficient Event-to-Video Reconstruction (**ECCV 2024**). [[paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05843.pdf)]
- EC-SNN: Splitting Deep Spiking Neural Networks on Edge Devices (**IJCAI 2024**). [[code](https://github.com/AmazingDD/EC-SNN)] 
- One-step Spiking Transformer with a Linear Complexity (**IJCAI 2024**).
- TIM: An Efficient Temporal Interaction Module for Spiking Transformer (**IJCAI 2024**). [[paper](https://arxiv.org/pdf/2401.11687)] [[code](https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/TIM)] 
- Learning a Spiking Neural Network for Efficient Image Deraining (**IJCAI 2024**). [[code](https://github.com/MingTian99/ESDNet)] 
- LitE-SNN: Designing Lightweight and Efficient Spiking Neural Network through Spatial-Temporal Compressive Network Search and Joint Optimization (**IJCAI 2024**). [[paper](https://arxiv.org/pdf/2401.14652)] 
- Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/35073)] 
- Towards efficient deep spiking neural networks construction with spiking activity based pruning (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/33505)] 
- Efficient and Effective Time-Series Forecasting with Spiking Neural Networks (**ICML 2024**). [[paper](https://arxiv.org/pdf/2402.01533)] 
- Autaptic Synaptic Circuit Enhances Spatio-temporal Predictive Learning of Spiking Neural Networks (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/33269)] 
- Robust Stable Spiking Neural Networks (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/33217)]
- CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks (**ICML 2024**). [[paper](https://arxiv.org/pdf/2402.04663)]
- NDOT: Neuronal Dynamics-based Online Training for Spiking Neural Networks  (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/33481)]
- High-Performance Temporal Reversible Spiking Neural Networks with $O(L)$ Training Memory and $O(1)$ Inference Cost (**ICML 2024**). [[paper](https://arxiv.org/pdf/2405.16466)]
- Towards Efficient Spiking Transformer: a Token Sparsification Framework for Training and Inference Acceleration (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/32674)]
- SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/35024)]
- Sign Gradient Descent-based Neuronal Dynamics: ANN-to-SNN Conversion Beyond ReLU Network (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/33242)]
- Enhancing Adversarial Robustness in SNNs with Sparse Gradients (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/34066)]
- SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN (**ICML 2024**). [[paper](https://icml.cc/virtual/2024/poster/34194)]
- Are Conventional SNNs Really Efficient? A Perspective from Network Quantization  (**CVPR 2024**). [[paper](https://arxiv.org/pdf/2311.10802)]
- SFOD: Spiking Fusion Object Detector (**CVPR 2024**). [[paper](https://arxiv.org/pdf/2403.15192)] [[code](https://github.com/yimeng-fan/SFOD)]
- SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks (**CVPR 2024**). [[paper](https://arxiv.org/abs/2403.14302)] [[code](https://github.com/xyshi2000/SpikingResformer)]
- SGLFormer: Spiking Global-Local-Fusion Transformer with high performance (**Frontiers in Neuroscience 2024**).[[paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1371290/full)] [[code](https://github.com/ZhangHanN1/SGLFormer)]
- Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework  (**ICLR 2024**). [[paper](https://openreview.net/forum?id=eoSeaK4QJo&referrer=%5Bthe%20profile%20of%20Zecheng%20Hao%5D(%2Fprofile%3Fid%3D~Zecheng_Hao1))]
- Online Stabilization of Spiking Neural Networks  (**ICLR 2024**). [[paper](https://openreview.net/forum?id=CIj1CVbkpr)]
- SpikePoint: An Efficient Point-based Spiking Neural Network for Event Cameras Action Recognition (**ICLR 2024**). [[paper](https://arxiv.org/pdf/2310.07189.pdf)]
- Spatio-Temporal Approximation: A Training-Free SNN Conversion for Transformers  (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=XrunSYwoLr)]
- Sparse Spiking Neural Network: Exploiting Heterogeneity in Timescales for Pruning Recurrent SNN (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=0jsfesDZDq)]
- Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=4r2ybzJnmN)] [[code](https://github.com/Thvnvtos/SNN-delays)]
- Threaten Spiking Neural Networks through Combining Rate and Temporal Information (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=xv8iGxENyI)] [[code](https://github.com/hzc1208/HART_Attack)]
- TAB: Temporal Accumulated Batch Normalization in Spiking Neural Networks (**ICLR 2024**). [[paper](https://openreview.net/forum?id=k1wlmtPGLq&noteId=p5M9gOLAOf)] 
- Certified Adversarial Robustness for Rate Encoded Spiking Neural Networks (**ICLR 2024**). [[paper](https://openreview.net/forum?id=5bNYf0CqxY)] 
- Bayesian Bi-clustering of Neural Spiking Activity with Latent Structures (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=ZYm1Ql6udy)] 
- Adaptive deep spiking neural network with global-local learning via balanced excitatory and inhibitory mechanism (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=wpnlc2ONu0)] 
- Hebbian Learning based Orthogonal Projection for Continual Learning of Spiking Neural Networks (**ICLR 2024**). [[paper](https://arxiv.org/pdf/2402.11984.pdf)] [[code](https://github.com/pkuxmq/HLOP-SNN)]
- A Progressive Training Framework for Spiking Neural Networks with Learnable Multi-hierarchical Model (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=g52tgL8jy6)] [[code](https://github.com/hzc1208/STBP_LMH)]
- LMUFormer: Low Complexity Yet Powerful Spiking Model With Legendre Memory Units (**ICLR 2024**). [[paper](https://arxiv.org/pdf/2402.04882.pdf)] [[code](https://github.com/zeyuliu1037/LMUFormer)]
- Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=1SIBN5Xyw7)] [[code](https://github.com/BICLab/Spike-Driven-Transformer-V2)]
- Can we get the best of both Binary Neural Networks and Spiking Neural Networks for Efficient Computer Vision? (**ICLR 2024**). [[paper](https://openreview.net/pdf?id=lGUyAuuTYZ)] [[code](https://github.com/godatta/Ultra-Low-Latency-SNN)]
- A Graph is Worth 1-bit Spikes: When Graph Contrastive Learning Meets Spiking Neural Networks (**ICLR 2024**).  [[paper](https://openreview.net/pdf?id=LnLySuf1vp)] [[code](https://github.com/EdisonLeeeee/SpikeGCL)]
- Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2312.06372.pdf)] [[code](https://github.com/yfguo91/Ternary-Spike)]
- Memory-Efficient Reversible Spiking Neural Networks (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2312.07922.pdf)] [[code](https://github.com/mi804/RevSNN)]
- Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2308.06582.pdf)]
- SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2308.10873.pdf)] [[code](https://github.com/NeuroCompLab-psu/SpikingBERT)]
- TC-LIF: A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2308.13250.pdf)] [[code](https://github.com/ZhangShimin1/TC-LIF)]
- Shrinking Your TimeStep: Towards Low-Latency Neuromorphic Object Recognition with Spiking Neural Networks (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2401.01912.pdf)]
- Dynamic Spiking Graph Neural Networks (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2401.05373.pdf)]
- An Efficient Knowledge Transfer Strategy for Spiking Neural Networks from Static to Event Domain (**AAAI 2024**).  [[paper](https://arxiv.org/pdf/2303.13077.pdf)] [[code](https://github.com/Brain-Cog-Lab/Transfer-for-DVS)]



**Arxiv**
- Q-SNNs: Quantized Spiking Neural Networks. [[paper](https://arxiv.org/pdf/2406.13672)]
- Scalable MatMul-free Language Modeling. [[paper](https://arxiv.org/pdf/2406.02528)] [[code](https://github.com/ridgerchu/matmulfreellm)]
- QKFormer: Hierarchical Spiking Transformer using Q-K Attention. [[paper](https://arxiv.org/pdf/2403.16552.pdf)] [[code](https://github.com/zhouchenlin2096/QKFormer)]
- Spikformer V2: Join the High Accuracy Club on ImageNet with an SNN Ticket. [[paper](https://arxiv.org/pdf/2401.02020.pdf)] [[code](https://github.com/ZK-Zhou/spikformer)]
- SpikeNAS: A Fast Memory-Aware Neural Architecture Search Framework for Spiking Neural Network Systems. [[paper](https://arxiv.org/pdf/2402.11322.pdf)]
- Astrocyte-Enabled Advancements in Spiking Neural Networks for Large Language Modeling. [[paper](https://arxiv.org/pdf/2312.07625v2.pdf)]


### 2023

**Review**
- Direct Learning-Based Deep Spiking Neural Networks: A Review (**Frontiers in Neuroscience 2023**). [[paper](https://arxiv.org/pdf/2305.19725.pdf)]

**AAAI, ICLR, CVPR, ICML, IJCAI, ICCV, NeurIPS, TPAMI, Science Advances**
- SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence (**Science Advances 2023**). [[paper](https://www.science.org/doi/10.1126/sciadv.adi1480)] [[code](https://github.com/fangwei123456/spikingjelly)]
- Spike-driven Transformer [[paper](https://arxiv.org/pdf/2307.01694.pdf)] [[code](https://github.com/BICLab/Spike-Driven-Transformer)]
- Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability (**NeurIPS 2023**). [[paper](https://arxiv.org/abs/2304.12760)] [[code](https://github.com/fangwei123456/Parallel-Spiking-Neuron)]
- Temporal Conditioning Spiking Latent Variable Models of the Neural Response to Natural Visual Scenes (**NeurIPS 2023**). [[paper](https://arxiv.org/pdf/2306.12045.pdf)]
- SEENN: Towards Temporal Spiking Early Exit Neural Networks (**NeurIPS 2023**). [[paper](https://openreview.net/pdf?id=mbaN0Y0QTw)]
- EICIL: Joint Excitatory Inhibitory Cycle Iteration Learning for Deep Spiking Neural Networks (**NeurIPS 2023**). [[paper](https://openreview.net/pdf?id=OMDgOjdqoZ)]
- Addressing the speed-accuracy simulation trade-off for adaptive spiking neurons (**NeurIPS 2023**). [[paper](https://openreview.net/pdf?id=Ht79ZTVMsn)]
- Enhancing Adaptive History Reserving by Spiking Convolutional Block Attention Module in Recurrent Neural Networks (**NeurIPS 2023**). [[paper](https://openreview.net/pdf?id=aGZp61S9Lj)]
- Trial matching: capturing variability with data-constrained spiking neural networks (**NeurIPS 2023**). [[paper](https://arxiv.org/abs/2306.03603)]
- Evolving Connectivity for Recurrent Spiking Neural Networks (**NeurIPS 2023**). [[paper](https://arxiv.org/pdf/2305.17650.pdf)]
- SparseProp: Efficient Event-Based Simulation and Training of Sparse Recurrent Spiking Neural Networks (**NeurIPS 2023**). [[paper](https://openreview.net/pdf?id=yzZbwQPkmP)] 
- Spiking PointNet: Spiking Neural Networks for Point Clouds (**NeurIPS 2023**). [[paper](https://arxiv.org/pdf/2310.06232v1.pdf)] [[code](https://github.com/dayongren/spiking-pointnet)]
- Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks (**NeurIPS 2023**). [[paper](https://openreview.net/pdf?id=8IvW2k5VeA)]
- Membrane Potential Batch Normalization for Spiking Neural Networks (**ICCV 2023**). [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Membrane_Potential_Batch_Normalization_for_Spiking_Neural_Networks_ICCV_2023_paper.pdf)]
- Unleashing the Potential of Spiking Neural Networks with Dynamic Confidence (**ICCV 2023**). [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Unleashing_the_Potential_of_Spiking_Neural_Networks_with_Dynamic_Confidence_ICCV_2023_paper.pdf)]
- RMP-Loss: Regularizing Membrane Potential Distribution for Spiking Neural Networks	(**ICCV 2023**). [[paper](https://arxiv.org/abs/2308.06787)]
- Inherent Redundancy in Spiking Neural Networks	(**ICCV 2023**). [[paper](https://arxiv.org/abs/2308.08227)]
- Temporal-Coded Spiking Neural Networks with Dynamic Firing Threshold: Learning with Event-Driven Backpropagation (**ICCV 2023**). [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wei_Temporal-Coded_Spiking_Neural_Networks_with_Dynamic_Firing_Threshold_Learning_with_ICCV_2023_paper.pdf)]
- Efficient Converted Spiking Neural Network for 3D and 2D Classification	(**ICCV 2023**). [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lan_Efficient_Converted_Spiking_Neural_Network_for_3D_and_2D_Classification_ICCV_2023_paper.pdf)]
- Deep Directly-Trained Spiking Neural Networks for Object Detection (**ICCV 2023**). [[paper](https://arxiv.org/abs/2307.11411)]
- Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks (**ICCV 2023**). [[paper](https://arxiv.org/abs/2302.14311)]
- SSF: Accelerating Training of Spiking Neural Networks with Stabilized Spiking Flow (**ICCV 2023**). [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_SSF_Accelerating_Training_of_Spiking_Neural_Networks_with_Stabilized_Spiking_ICCV_2023_paper.pdf)]
- Masked Spiking Transformer (**ICCV 2023**). [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Masked_Spiking_Transformer_ICCV_2023_paper.pdf)]
- Spatial-Temporal Self-Attention for Asynchronous Spiking Neural Networks (**IJCAI 2023**). [[paper](https://www.ijcai.org/proceedings/2023/0344.pdf)]
- Learnable Surrogate Gradient for Direct Training Spiking Neural Networks (**IJCAI 2023**). [[paper](https://www.ijcai.org/proceedings/2023/0335.pdf)]
- Enhancing Efficient Continual Learning with Dynamic Structure Development of Spiking Neural Networks (**IJCAI 2023**). [[paper](https://www.ijcai.org/proceedings/2023/0334.pdf)]
- Adaptive Smoothing Gradient Learning for Spiking Neural Networks (**ICML 2023**). [[paper](https://openreview.net/pdf?id=GdkwSGTpbC)]
- Surrogate Module Learning: Reduce the Gradient Error Accumulation in Training Spiking Neural Networks (**ICML 2023**). [[paper](https://openreview.net/pdf?id=zRkz4duLKp)] [[code](https://github.com/brain-intelligence-lab/surrogate_module_learning)]
- Rate Gradient Approximation Attack Threats Deep Spiking Neural Networks (**CVPR 2023**). [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Bu_Rate_Gradient_Approximation_Attack_Threats_Deep_Spiking_Neural_Networks_CVPR_2023_paper.pdf)]
- Constructing Deep Spiking Neural Networks from Artificial Neural Networks with Knowledge Distillation (**CVPR 2023**). [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Constructing_Deep_Spiking_Neural_Networks_From_Artificial_Neural_Networks_With_CVPR_2023_paper.pdf)]
- Attention Spiking Neural Networks  (**TPAMI 2023**) .[[paper](https://ieeexplore.ieee.org/abstract/document/10032591)] [[code](https://github.com/fangwei123456/spikingjelly/pull/329)]
- Heterogeneous neuronal and synaptic dynamics for spike-efficient unsupervised learning: Theory and design principles (**ICLR 2023**).[[paper](https://arxiv.org/pdf/2302.11618.pdf)]
- Spiking Convolutional Neural Networks for Text Classification (**ICLR 2023**) .[[paper](https://openreview.net/pdf?id=pgU3k7QXuz0)]
- Bridging the Gap between ANNs and SNNs by Calibrating Offset Spikes (**ICLR 2023**).[[paper](https://arxiv.org/pdf/2302.10685.pdf)] [[code](https://github.com/hzc1208/ANN2SNN_COS)]
- Spikformer: When Spiking Neural Network Meets Transformer (**ICLR 2023**) .[[paper](https://openreview.net/forum?id=frE4fUwz_h)] [[code](https://github.com/ZK-Zhou/spikformer)]
- A Unified Framework of Soft Threshold Pruning (**ICLR 2023**). [[paper](https://openreview.net/forum?id=cCFqcrq0d8)] [[code](https://github.com/Yanqi-Chen/LATS)]
- Bridging the Gap between ANNs and SNNs by Calibrating Offset Spikes (**ICLR 2023**). [[paper](https://openreview.net/forum?id=PFbzoWZyZRX)] [[code](https://github.com/hzc1208/ANN2SNN_COS)]
- Reducing ANN-SNN Conversion Error through Residual Membrane Potential (**AAAI 2023**). [[paper](https://arxiv.org/abs/2302.02091)] [[code](https://github.com/hzc1208/ANN2SNN_SRP)]
- Deep Spiking Neural Networks with High Representation Similarity Model Visual Pathways of Macaque and Mouse (**AAAI 2023**). [[paper](https://arxiv.org/abs/2303.06060)]
- ESL-SNNs: An Evolutionary Structure Learning Strategy for Spiking Neural Networks (**AAAI 2023**). [[paper](https://arxiv.org/pdf/2306.03693.pdf)]
- Exploring Temporal Information Dynamics in Spiking Neural Networks (**AAAI 2023**). [[paper](https://arxiv.org/pdf/2211.14406.pdf)] [[code](https://github.com/Intelligent-Computing-Lab-Yale/Exploring-Temporal-Information-Dynamics-in-Spiking-Neural-Networks)]
- Scaling Up Dynamic Graph Representation Learning via Spiking Neural Networks(**AAAI 2023**). [[paper](https://arxiv.org/pdf/2208.10364.pdf)] [[code](https://github.com/EdisonLeeeee/SpikeNet)]
- Complex Dynamic Neurons Improved Spiking Transformer Network for Efficient Automatic Speech Recognition(**AAAI 2023**). [[paper](https://arxiv.org/pdf/2302.01194.pdf)] 

**Arxiv**
- Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network [[paper](https://arxiv.org/abs/2304.11954)] [[code](https://github.com/zhouchenlin2096/Spikingformer)]
- Enhancing the Performance of Transformer-based Spiking Neural Networks by Improved Downsampling with Precise Gradient Backpropagation [[paper](https://arxiv.org/abs/2305.05954)] [[code](https://github.com/zhouchenlin2096/Spikingformer-CML)]
- Training Full Spike Neural Networks via Auxiliary Accumulation Pathway [[paper](https://arxiv.org/pdf/2301.11929.pdf)]
- MSS-DepthNet: Depth Prediction with Multi-Step Spiking Neural Network [[paper](https://arxiv.org/abs/2211.12156)]
- SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks [[paper](https://arxiv.org/abs/2302.13939)] [[code](https://github.com/ridgerchu/SpikeGPT)]
- Auto-Spikformer: Spikformer Architecture Search [[paper](https://arxiv.org/pdf/2306.00807.pdf)]
- Advancing Spiking Neural Networks Towards Deep Residual Learning [[paper](https://arxiv.org/pdf/2112.08954.pdf)]


### 2022

**NeurIPS, CVPR, ICLR, AAAI, ICML, Nature Communications**

- Event-based Video Reconstruction via Potential-assisted Spiking Neural Network [[paper](https://arxiv.org/abs/2201.10943)] [[code](https://github.com/LinZhu111/EVSNN)]
- Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks [[paper](https://openreview.net/forum?id=7B3IJMM1k_M)] [[code](https://github.com/putshua/SNN-conversion-QCFS)]
- Optimized Potential Initialization for Low-latency Spiking Neural Networks (**AAAI 2022**).  [[paper](https://arxiv.org/abs/2202.01440)]
- AutoSNN: Towards Energy-Efficient Spiking Neural Networks [[paper](https://arxiv.org/abs/2201.12738)]
- Neural Architecture Search for Spiking Neural Networks [[paper](https://arxiv.org/abs/2201.10355)] [[code](https://github.com/Intelligent-Computing-Lab-Yale/Neural-Architecture-Search-for-Spiking-Neural-Networks)]
- Neuromorphic Data Augmentation for Training Spiking Neural Networks [[paper](https://arxiv.org/abs/2203.06145)] [[code](https://github.com/Intelligent-Computing-Lab-Yale/NDA_SNN)]
- State Transition of Dendritic Spines Improves Learning of Sparse Spiking Neural Networks [[paper](https://proceedings.mlr.press/v162/chen22ac.html)] [[code](https://github.com/Yanqi-Chen/STDS)]
- Training High-Performance Low-Latency Spiking Neural Networks by Differentiation on Spike Representation [[paper](https://arxiv.org/abs/2205.00459)] [[code](https://github.com/qymeng94/DSR)]
- Exploring Lottery Ticket Hypothesis in Spiking Neural Networks [[paper](https://arxiv.org/abs/2207.01382)] [[code](https://github.com/Intelligent-Computing-Lab-Yale/Exploring-Lottery-Ticket-Hypothesis-in-SNNs)]
- Spiking Graph Convolutional Networks [[paper](https://arxiv.org/abs/2205.02767)] [[code](https://github.com/ZulunZhu/SpikingGCN)]
- A calibratable sensory neuron based on epitaxial VO2 for spike-based neuromorphic multisensory system [[paper](https://www.nature.com/articles/s41467-022-31747-w)] [[code](https://github.com/billyuanpku96/SNN-for-sensory-neuron)]
- Online Training Through Time for Spiking Neural Networks (**NeurIPS 2022**).  [[paper](https://arxiv.org/abs/2210.04195)] [[code](https://github.com/pkuxmq/OTTT-SNN)]
- Training Spiking Neural Networks with Event-driven Backpropagation [[paper](https://openreview.net/forum?id=d4JmP1T45WE)] [[code](https://github.com/zhuyaoyu/SNN-event-driven-learning)]
- GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks [[paper](https://openreview.net/forum?id=UmFSx2c4ubT)] [[code](https://github.com/Ikarosy/Gated-LIF)]
- Temporal Effective Batch Normalization in Spiking Neural Networks [[paper](https://openreview.net/forum?id=fLIgyyQiJqz)]
- Training Spiking Neural Networks with Local Tandem Learning (**NeurIPS 2022**). [[paper](https://arxiv.org/pdf/2210.04532.pdf)]
- IM-Loss: Information Maximization Loss for Spiking Neural Networks (**NeurIPS 2022**). [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/010c5ba0cafc743fece8be02e7adb8dd-Paper-Conference.pdf)]
- Temporal Effective Batch Normalization in Spiking Neural Networks (**NeurIPS 2022**). [[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/de2ad3ed44ee4e675b3be42aa0b615d0-Paper-Conference.pdf)]
- Biologically Inspired Dynamic Thresholds for Spiking Neural Networks (**NeurIPS 2022**). [[paper](https://arxiv.org/pdf/2206.04426.pdf)]
- Optimal Conversion of Conventional Artificial Neural Networks to Spiking Neural Networks (**ICLR 2022**).  [[paper](https://arxiv.org/pdf/2103.00476.pdf)] [[code](https://github.com/Jackn0/snn_optimal_conversion_pipeline)]
- Multi-Level Firing with Spiking DS-ResNet: Enabling Better and Deeper Directly-Trained Spiking Neural Networks (**IJCAI 2022**). [[paper](https://arxiv.org/pdf/2210.06386.pdf)]

### 2021

**NeurIPS, ICCV, IJCAI, ICML, AAAI**

- Deep Residual Learning in Spiking Neural Networks (**NeurIPS 2021**). [[paper](https://proceedings.neurips.cc/paper/2021/file/afe434653a898da20044041262b3ac74-Paper.pdf)] [[code](https://github.com/fangwei123456/Spike-Element-Wise-ResNet)]
- Spiking Deep Residual Network[[paper](https://arxiv.org/pdf/1805.01352.pdf)]
- Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks (**ECCV 2021**).  [[paper](https://arxiv.org/abs/2007.05785)]  [[code](https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron)]
- Pruning of Deep Spiking Neural Networks through Gradient Rewiring [[paper](https://arxiv.org/abs/2105.04916)] [[code](https://github.com/Yanqi-Chen/Gradient-Rewiring)]
- A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration  (**ICML 2021**).  [[paper](https://arxiv.org/pdf/2106.06984)] [[code](https://github.com/yhhhli/SNN_Calibration)]
- Optimal ANN-SNN Conversion for Fast and Accurate Inference in Deep Spiking Neural Networks [[paper](https://arxiv.org/pdf/2105.11654)] [[code](https://github.com/DingJianhao/OptSNNConvertion-RNL-RIL)]
- Sparse Spiking Gradient Descent (**NeurIPS 2021**). [[paper](https://proceedings.neurips.cc/paper/2021/file/61f2585b0ebcf1f532c4d1ec9a7d51aa-Paper.pdf)]
- Training Spiking Neural Networks with Accumulated Spiking Flow (**AAAI 2021**). [[paper](https://arxiv.org/pdf/2011.05280.pdf)]
- Temporal-wise Attention Spiking Neural Networks for Event Streams Classification. (**ECCV 2021**). [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.pdf)]


### Reference
If you find this repo useful, please consider citing:
```
@article{zhou2024direct,
  title={Direct training high-performance deep spiking neural networks: a review of theories and methods},
  author={Zhou, Chenlin and Zhang, Han and Yu, Liutao and Ye, Yumin and Zhou, Zhaokun and Huang, Liwei and Ma, Zhengyu and Fan, Xiaopeng and Zhou, Huihui and Tian, Yonghong},
  journal={Frontiers in Neuroscience},
  volume={18},
  pages={1383844},
  year={2024},
  publisher={Frontiers Media SA}
}
```
