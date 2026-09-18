# Yufeng Program — Complete Course Assignments

**Author: Zelin Shen** ([GitHub](https://github.com/Zelin-Shen) | [Google Scholar](https://scholar.google.com/citations?user=D6zIduEAAAAJ))

All five training camps of the **Yufeng Program (驭风计划)** on XuetangX (学堂在线), 2025 — from Python fundamentals to deep learning and NLP. **99 assignments/code files across 5 camps**, organized one directory per experiment, with completion certificates.

> ⚠️ **Note on copyright**: Course lecture slides and teaching materials are the program's property and are **not** included. Only my own assignment code and reports are published here.

## Repository Structure

```
.
├── dl-camp/                 # Deep learning camp — 7 experiments
│   ├── exp1-softmax-mnist/
│   ├── exp2-mlp-mnist/
│   ├── exp3-cifar10/
│   ├── exp4-brain-mri-segmentation/
│   ├── exp5-didi-traffic-detection/
│   ├── exp6-image-captioning/
│   └── exp7-image-super-resolution/
├── ml-camp/                 # Machine learning camp — 7 assignments
│   ├── exp1-decision-tree-lol/
│   ├── exp2-regression-scores/
│   ├── exp3-naive-bayes-spam/
│   ├── exp4-knn-license-plate/
│   ├── exp5-clustering-aaai/
│   ├── exp6-ensemble-amazon-reviews/
│   └── exp7-churn-prediction/
├── nlp-camp/                # NLP camp — 6 experiments
│   ├── exp1-word2vec-transE/
│   ├── exp2-seq2seq/
│   ├── exp3-sentiment/
│   ├── exp4-pretrained-lm/
│   ├── exp5-legal-qa/
│   └── exp6-covid-social/
├── algorithm-camp/          # Algorithm camp — 71 solutions
│   ├── weekly-test/  exercise/  final-exam/  pre-class/  practice/
├── python-camp/             # Python fundamentals — chapter assignments
├── certificates/            # Camp and course completion certificates
├── requirements.txt
└── LICENSE
```

## Camps Overview

| Camp | Content | Directory | Certificate |
|---|---|---|---|
| **Deep Learning** | 7 experiments: SoftMax/MLP MNIST, CIFAR-10, U-Net MRI segmentation, MMDetection traffic detection, image captioning, SRGAN super-resolution | [`dl-camp/`](dl-camp/) | [✓](certificates/deep-learning-camp.jpg) |
| **Machine Learning** | 7 assignments: decision tree, regression, Naive Bayes, k-NN, clustering, ensemble, churn prediction | [`ml-camp/`](ml-camp/) | [✓](certificates/machine-learning-camp.jpg) |
| **NLP** | 6 experiments: Word2Vec/TransE, Seq2Seq, sentiment analysis, pretrained LMs, legal QA, COVID social computing | [`nlp-camp/`](nlp-camp/) | [✓](certificates/nlp-camp.jpg) |
| **Algorithm** | 71 solutions: weekly tests, exercises, final exam (Python & C++) | [`algorithm-camp/`](algorithm-camp/) | [✓](certificates/algorithm-camp.jpg) |
| **Python Fundamentals** | Chapter assignments: sequences, modules, algorithms, matplotlib, pandas | [`python-camp/`](python-camp/) | [Program completion](certificates/yufeng-program-completion.pdf) |

**Additional course certificates** ([`certificates/`](certificates/)): Convex Optimization (NUDT), Big Data Machine Learning (Tsinghua), Machine Learning Basics & Advanced (Nanjing University), Discrete Mathematics (UESTC).

## Deep Learning Camp

| # | Topic | Key Techniques |
|---|---|---|
| 1 | [SoftMax MNIST](dl-camp/exp1-softmax-mnist/) | Softmax classifier from scratch (NumPy) |
| 2 | [MLP MNIST](dl-camp/exp2-mlp-mnist/) | Multi-layer perceptron in PyTorch |
| 3 | [CIFAR-10 Classification](dl-camp/exp3-cifar10/) | PyTorch CNN training pipeline |
| 4 | [Brain MRI Tumor Segmentation](dl-camp/exp4-brain-mri-segmentation/) | U-Net, Dice loss, 8 tuning iterations |
| 5 | [Didi Traffic Object Detection](dl-camp/exp5-didi-traffic-detection/) | Faster/Cascade R-CNN (MMDetection), 4 config variants |
| 6 | [Image Captioning](dl-camp/exp6-image-captioning/) | CNN+RNN with attention, adaptive attention |
| 7 | [Image Super-Resolution](dl-camp/exp7-image-super-resolution/) | SRGAN, spectral normalization, asymmetric LR |

## Machine Learning Camp

Decision tree (LoL match outcome) · linear regression (university scores) · Naive Bayes (spam) · k-NN (license plates) · clustering (AAAI papers) · ensemble learning (Amazon review quality) · churn prediction (mobile game, with sequence extension). See [`ml-camp/`](ml-camp/).

## NLP Camp

Word2Vec/TransE embeddings · Seq2Seq · sentiment analysis · pretrained LMs for relation extraction · legal QA with GNN variants · COVID-19 social computing. See [`nlp-camp/`](nlp-camp/).

## Environment

```
pip install -r requirements.txt
```

DL Camp Exp 5 additionally requires [MMDetection](https://github.com/open-mmlab/mmdetection) installed separately.

## Notes on Individual Camps

- **NLP Exp 5 (legal QA)**: builds on a course-provided baseline framework for the CAIL legal-QA task. The baseline source code is the program's material and is **not** redistributed here; this repo contains my own experiment notebooks, the GNN and GNN-GAT variants I implemented, and the resulting metrics.
- **Algorithm camp**: solutions organized by source — `weekly-test/`, `exercise/`, `final-exam/`, `pre-class/`, `practice/`. Problem statements are the program's material and are not included.
- **Python camp**: `ch10_modules_homework1.ipynb` reads `ch10_input_data.csv` (included).

## Third-Party Code

- `dl-camp/exp7-image-super-resolution/imresize.py` is adapted from [fatheral/matlab_imresize](https://github.com/fatheral/matlab_imresize) (MIT License), © 2018 Alex fatheral, with modifications.
- DL Camp Exp 5 uses [MMDetection](https://github.com/open-mmlab/mmdetection) (Apache-2.0); only my configuration variants and results are included.
- DL Camp Exp 4's U-Net follows Ronneberger et al., 2015. Exp 6 follows the show-and-tell + attention captioning line of work. Exp 7's SR network reproduces Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*, CVPR 2017.

## License

MIT (see [LICENSE](LICENSE)) — covers my own code only.
