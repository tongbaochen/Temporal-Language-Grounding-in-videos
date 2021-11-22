# Temporal Language Grounding

## Introduction

Task：

- given a query, find the corresponding moment in a given video. **(major focus of this repo)**

## Format

Markdown format:

```markdown
- [Paper Name](link) - Author 1 et al, `Conference Year`. [[code]](link)
```

## Change Log

* 2020/07/27 start the repo.
* Papers before 2020 are mainly collected by [muketong](https://github.com/iworldtong).

## Table of Contents

- to be updated ...

## Keywords used in searching

- grounding, retrieval, localization

## Papers

### Survey

- None.

### Before
- [Grounded Language Learning from Video Described with Sentences](https://www.aclweb.org/anthology/P13-1006/) - H. Yu et al, `ACL 2013`. 
- [Visual Semantic Search: Retrieving Videos via Complex Textual Queries](<https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Lin_Visual_Semantic_Search_2014_CVPR_paper.pdf>) - Dahua Lin et al, `CVPR 2014`.
- [Jointly Modeling Deep Video and Compositional Text to Bridge Vision and Language in a Unified Framework](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9734) - R. Xu et al, `AAAI 2015`.
- [Unsupervised Alignment of Actions in Video with Text Descriptions](https://pdfs.semanticscholar.org/5893/7d427ff36e1470b18120245148355047e4ea.pdf) - Y. C. Song et al, `IJCAI 2016`.

### 2017
- [Localizing Moments in Video with Natural Language](https://arxiv.org/abs/1708.01641) - Lisa Anne Hendricks et al, `ICCV 2017`. [[code]](<https://people.eecs.berkeley.edu/~lisa_anne/didemo.html>)
- [TALL: Temporal Activity Localization via Language Query](https://arxiv.org/abs/1705.02101) - Jiyang Gao et al, `ICCV 2017`. [[code]](<https://github.com/jiyanggao/TALL>). 
- !(Still on arxiv 20200609)[Where to Play: Retrieval of Video Segments using Natural-Language Queries](<https://arxiv.org/abs/1707.00251>) - S. Lee et al, `arxiv 2017`.

### 2018
- [Attentive Moment Retrieval in Videos](http://staff.ustc.edu.cn/~hexn/papers/sigir18-video-retrieval.pdf) - M. Liu et al, `SIGIR 2018`.
- [Temporal Modular Networks for Retrieving Complex Compositional Activities in Videos](<http://svl.stanford.edu/assets/papers/liu2018eccv.pdf>) - B. Liu et al, `ECCV 2018`.
- (Video Retrieval+Grounding)[Find and Focus: Retrieve and Localize Video Events with Natural Language Queries](<http://openaccess.thecvf.com/content_ECCV_2018/papers/Dian_SHAO_Find_and_Focus_ECCV_2018_paper.pdf>) - Dian Shao  et al, `ECCV 2018`.
- [Temporally Grounding Natural Sentence in Video](<https://aclweb.org/anthology/papers/D/D18/D18-1015/>) - J. Chen et al, `EMNLP 2018`.
- [Localizing Moments in Video with Temporal Language](<https://arxiv.org/abs/1809.01337>) - Lisa Anne Hendricks et al, `EMNLP 2018`.
- [Cross-modal Moment Localization in Videos](<https://doi.org/10.1145/3240508.3240549>) - Meng Liu et al, `MM 2018`.

### 2019
Supervised:
- [MAC: Mining Activity Concepts for Language-based Temporal Localization](https://arxiv.org/abs/1811.08925) - Runzhou Ge Ge et al, `WACV 2019`. [[code]](https://github.com/runzhouge/MAC)
- [Multilevel Language and Vision Integration for Text-to-Clip Retrieval](<https://arxiv.org/abs/1804.05113>) - H. Xu et al, `AAAI 2019`. [[code]](<https://github.com/VisionLearningGroup/Text-to-Clip_Retrieval>)
- [Read, Watch, and Move: Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos](https://arxiv.org/abs/1901.06829) - He, Dongliang et al, `AAAI 2019`.
- [To Find Where You Talk: Temporal Sentence Localization in Video with Attention Based Location Regression](http://arxiv.org/abs/1804.07014) - Y. Yuan et al, `AAAI 2019`. [[code]](https://github.com/yytzsy/ABLR_code)
- [Semantic Proposal for Activity Localization in Videos via Sentence Query](http://yugangjiang.info/publication/19AAAI-actionlocalization.pdf) - S. Chen et al, `AAAI 2019`.
- [Localizing natural language in videos](https://www.aaai.org/ojs/index.php/AAAI/article/view/4827/4700) - J. Chen et al, `AAAI 2019`.
- [ExCL: Extractive Clip Localization Using Natural Language Descriptions](https://arxiv.org/abs/1904.02755) - S. Ghosh et al, `NAACL 2019`.
- [Cross-Modal Video Moment Retrieval with Spatial and Language-Temporal Attention](https://dl.acm.org/citation.cfm?id=3325019) - B. Jiang et al, `ICMR 2019`. [[code]](https://github.com/BonnieHuangxin/SLTA)
- [Language-Driven Temporal Activity Localization_ A Semantic Matching Reinforcement Learning Model](<http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Language-Driven_Temporal_Activity_Localization_A_Semantic_Matching_Reinforcement_Learning_Model_CVPR_2019_paper.pdf>) - W. Wang et al, `CVPR 2019`. 
- [MAN: Moment Alignment Network for Natural Language Moment Retrieval via Iterative Graph Adjustment](https://arxiv.org/abs/1812.00087) - Da Zhang et al, `CVPR 2019`. 
- [Cross-Modal Interaction Networks for Query-Based Moment Retrieval in Videos](https://arxiv.org/abs/1906.02497) - Zhu Zhang et al, `SIGIR 2019`. [[code]](https://github.com/ikuinen/CMIN_moment_retrieval)
- [Semantic Conditioned Dynamic Modulation for Temporal Sentence Grounding in Videos](https://arxiv.org/pdf/1910.14303.pdf) - Yitian Yuan et al, `NeurIPS 2019`. [[code]](https://github.com/yytzsy/SCDM)
- [DEBUG: A Dense Bottom-Up Grounding Approach for Natural Language Video Localization](https://www.aclweb.org/anthology/D19-1518.pdf) - Chujie Lu et al, `EMNLP 2019`.
- !(still on arxiv 20200609)[Temporal Localization of Moments in Video Collections with Natural Language](https://arxiv.org/abs/1907.12763v1) - V. Escorcia et al, `arxiv 2019`. 

Weakly Supervised:
- [Weakly Supervised Video Moment Retrieval From Text Queries](<https://arxiv.org/abs/1904.03282>) - N. C. Mithun et al, `CVPR 2019`. 
- [Weakly-supervised spatio-temporally grounding natural sentence in video](https://www.aclweb.org/anthology/P19-1183.pdf) - Zhenfang Chen et al, `ACL 2019`. [[code]](https://github.com/JeffCHEN2017/WSSTG.git.)
- [WSLLN: Weakly Supervised Natural Language Localization Networks](https://arxiv.org/abs/1909.00239) - M. Gao et al, `EMNLP 2019`. 

### 2020

Supervised:
- [Moment Retrieval via Cross-Modal Interaction Networks With Query Reconstruction](https://ieeexplore.ieee.org/abstract/document/8962274) - Zhijie Lin et al, `TIP 2020`.
- [Rethinking the Bottom-Up Framework for Query-based Video Localization](https://aaai.org/ojs/index.php/AAAI/article/view/6627) - Long Chen et al, `AAAI 2020`.
- [Temporally Grounding Language Queries in Videos by Contextual Boundary-aware Prediction](https://arxiv.org/abs/1909.05010) - Jingwen Wang et al, `AAAI 2020`. [[code]](https://github.com/JaywongWang/CBP)
- [Learning 2D Temporal Adjacent Networks for Moment Localization with Natural Language](https://arxiv.org/pdf/1912.03590.pdf) - Songyang Zhang et al, `AAAI 2020`. [[code]](https://github.com/microsoft/2D-TAN)
- [Tree-Structured Policy based Progressive Reinforcement Learning for Temporally Language Grounding in Video](https://arxiv.org/pdf/2001.06680.pdf) - Jie Wu et al, `AAAI 2020`. [[code]](https://github.com/WuJie1010/TSP-PRL)
- [Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention](https://arxiv.org/abs/1908.07236) - C. R. Opazo et al, `WACV 2020`. [[code]](https://github.com/crodriguezo/TMLGA)
- [Local-Global Video-Text Interactions for Temporal Grounding](http://arxiv.org/abs/2004.07514) - Mun Jonghwan et al, `CVPR 2020`. [[code]](https://github.com/JonghwanMun/LGI4temporalgrounding)
- [Dense Regression Network for Video Grounding](http://arxiv.org/abs/2004.03545) - Zeng Runhao et al, `CVPR 2020`. [[code]](https://github.com/Alvin-Zeng/DRN)
- [Tripping through time: Efficient Localization of Activities in Videos](https://arxiv.org/abs/1904.09936) - Meera Hahn et al, `BMVC 2020`.
- [Span-based Localizing Network for Natural Language Video Localization](https://www.aclweb.org/anthology/2020.acl-main.585/) - Hao Zhang et al, `ACL 2020`. [[code]](https://github.com/IsaacChanghau/VSLNet)
- [Hierarchical Visual-Textual Graph for Temporal Activity Localization via Language](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650596.pdf) - Shaoxiang Chen et al, `ECCV 2020`. [[code]](https://github.com/forwchen/HVTG)
- [Learning Modality Interaction for Temporal Sentence Localization and Event Captioning in Videos](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490324.pdf) - Shaoxiang Chen et al, `ECCV 2020`.
- [Jointly Cross- and Self-Modal Graph Attention Network for Query-Based Moment Localization](http://arxiv.org/abs/2008.01403) - Daizong Liu et al, `MM 2020`. [[code]](https://github.com/liudaizong/CSMGAN)
- [Fine-grained Iterative Attention Network for Temporal Language Localization in Videos](http://arxiv.org/abs/2008.02448) - Xiaoye Qu et al, `MM 2020`.
- [Dual Path Interaction Network for Video Moment Localization](https://dl.acm.org/doi/10.1145/3394171.3413975) - Hao Wang et al, `MM 2020`.
- [Adversarial Video Moment Retrieval by Jointly Modeling Ranking and Localization](https://dl.acm.org/doi/10.1145/3394171.3413841) -  et al, `MM 2020`. [[code]](https://github.com/yawenzeng/AVMR)
- [STRONG: Spatio-Temporal Reinforcement Learning for Cross-Modal Video Moment Localization](https://dl.acm.org/doi/10.1145/3394171.3413840) - Da Cao et al, `MM 2020`. [[code]](https://github.com/yawenzeng/STRONG)
- [Reinforcement Learning for Weakly Supervised Temporal Grounding of Natural Language in Untrimmed Videos](http://arxiv.org/abs/2009.08614) - Jie Wu et al, `MM 2020`.
- [Language Guided Networks for Cross-modal Moment Retrieval](http://arxiv.org/abs/2006.10457) - Kun Liu et al, `arxiv`.

Weakly Supervised:
- [Weakly-Supervised Video Moment Retrieval via Semantic Completion Network](https://arxiv.org/pdf/1911.08199.pdf) - Zhijie Lin et al, `AAAI 2020`.
- [VLANet: Video-Language Alignment Network for Weakly-Supervised Video Moment Retrieval](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730154.pdf) - Minuk Ma et al, `ECCV 2020`.
- [Two-Stream Consensus Network for Weakly-Supervised Temporal Action Localization](https://arxiv.org/abs/2010.11594) - Yuanhao Zhai et al, `ECCV 2020`.
- [Regularized Two-Branch Proposal Networks for Weakly-Supervised Moment Retrieval in Videos](https://arxiv.org/abs/2008.08257) - Zhu Zhang et al, `MM 2020`. [[code]](https://github.com/ikuinen/regularized_two-branch_proposal_network)
- [Counterfactual Contrastive Learning for Weakly-Supervised Vision-Language Grounding](https://proceedings.neurips.cc/paper/2020/file/d27b95cac4c27feb850aaa4070cc4675-Paper.pdf) - Zhang Zhu et al, `NeruIPS 2020`.

### 2021
- [Interaction-Integrated Network for Natural Language Moment Localization](https://ieeexplore.ieee.org/abstract/document/9334438) - Ke Ning et al, 'TIP 2021'.
- [Boundary Proposal Network for Two-Stage Natural Language Video Localization](https://arxiv.org/abs/2103.08109) - Shaoning Xiao et al, `AAAI 2021`.
- [Context-Aware Biaffine Localizing Network for Temporal Sentence Grounding](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Context-Aware_Biaffine_Localizing_Network_for_Temporal_Sentence_Grounding_CVPR_2021_paper.html) - Liu et al, `CVPR 2021`. 
- [Multi-Modal Relational Graph for Cross-Modal Video Moment Retrieval](https://openaccess.thecvf.com/content/CVPR2021/html/Zeng_Multi-Modal_Relational_Graph_for_Cross-Modal_Video_Moment_Retrieval_CVPR_2021_paper.html) - Zeng et al, `CVPR 2021`. 
- [Thinking Fast and Slow: Efficient Text-to-Visual Retrieval With Transformers](https://openaccess.thecvf.com/content/CVPR2021/html/Miech_Thinking_Fast_and_Slow_Efficient_Text-to-Visual_Retrieval_With_Transformers_CVPR_2021_paper.html) - Miech et al, `CVPR 2021`. 
- [Fast Video Moment Retrieval](https://openaccess.thecvf.com/content/ICCV2021/html/Gao_Fast_Video_Moment_Retrieval_ICCV_2021_paper.html) - Gao et al, `ICCV 2021`.

Conferences to be update:
- None

## Dataset

- [ActivityNet Captions](http://cs.stanford.edu/people/ranjaykrishna/densevid/)
- [Charades-STA](<https://allenai.org/plato/charades/>)
- [DiDeMo](<https://github.com/LisaAnne/LocalizingMoments>)
- [TACoS](http://www.coli.uni-saarland.de/projects/smile/page.php?id=software)

## Benchmark Results

#### ActivityNet Captions

|                 | R@1 IoU@0.1 | R@1 IoU@0.3 | R@1 IoU@0.5 | R@1 IoU@0.7 | R@5 IoU@0.1 | R@5 IoU@0.3 | R@5 IoU@0.5 | R@5 IoU@0.7 | Method |
| :-------------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :----: |
|       MCN       |    42.80    |    21.37    |    9.58     |      -      |      -      |      -      |      -      |      -      |   PB   |
|      CTRL       |    49.09    |    28.70    |    14.0     |      -      |      -      |      -      |      -      |      -      |   PB   |
|      ACRN       |    50.37    |    31.29    |    16.17    |      -      |      -      |      -      |      -      |      -      |   PB   |
|      QSPN       |      -      |    45.3     |    27.7     |    13.6     |      -      |    75.7     |    59.2     |    38.3     |   PB   |
|       TGN       |    70.06    |    45.51    |    28.47    |      -      |    79.10    |    57.32    |    44.20    |      -      |   PB   |
|      SCDM       |      -      |    54.80    |    36.75    |    19.86    |      -      |    77.29    |    64.99    |    41.53    |   PB   |
|       CBP       |      -      |    54.30    |    35.76    |    17.80    |      -      |    77.63    |    65.89    |    46.20    |   PB   |
|     TripNet     |      -      |    48.42    |    32.19    |    13.93    |      -      |      -      |      -      |      -      |   RL   |
|      ABLR       |    73.30    |    55.67    |    36.79    |      -      |      -      |      -      |      -      |      -      |   RL   |
|      ExCL       |      -      |    63.30    |    43.6     |    24.1     |      -      |      -      |      -      |      -      |   PF   |
|      PFGA       |    75.25    |    51.28    |    33.04    |    19.26    |      -      |      -      |      -      |      -      |   PF   |
| WSDEC-X(Weakly) |    62.7     |    42.0     |    23.3     |      -      |      -      |      -      |      -      |      -      |        |
| WSLLN (Weakly)  |    75.4     |    42.8     |    22.7     |      -      |      -      |      -      |      -      |      -      |        |
|CMIN  |- |63.61| 43.40| 23.88| - |80.54| 67.95| 50.73|  |
| HVTG            |    -        |    57.60    |    40.15    |      18.27  |      -      |      -      |      -      |      -      |  graph based   |

#### Charades-STA

|         | R@1 IoU@0.1 | R@1 IoU@0.3 | R@1 IoU|@0.5 | R@1 IoU@0.7 | R@5 IoU@0.1 | R@5 IoU@0.3 | R@5 IoU@0.5 | R@5 IoU@0.7 | Method |
| :-----: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :----: |
|  CTRL   |      -      |      -      |    23.63    |    8.89     |      -      |      -      |    58.92    |    29.52    |   PB   |
|  ABLR   |      -      |      -      |    24.36    |    9.01     |      -      |      -      |      -      |      -      |   PB   |
|  SMRL   |      -      |      -      |    24.36    |    11.17    |      -      |      -      |    61.25    |    32.08    |   PB   |
|  ACL-K  |      -      |      -      |    30.48    |    12.20    |      -      |      -      |    64.84    |    35.13    |   PB   |
|   SAP   |      -      |      -      |    27.42    |    13.36    |      -      |      -      |    66.37    |    38.15    |   PB   |
|  QSPN   |      -      |    54.7     |    35.6     |    15.8     |      -      |    95.8     |    79.4     |    45.4     |   PB   |
|   MAN   |      -      |      -      |    46.53    |    22.72    |      -      |      -      |    86.23    |    53.72    |   PB   |
|  SCDM   |      -      |      -      |    54.44    |    33.43    |      -      |      -      |    74.43    |    58.08    |   PB   |
|   CBP   |      -      |      -      |    36.80    |    18.87    |      -      |      -      |    70.94    |    50.19    |   PB   |
| TripNet |      -      |    51.33    |    36.61    |    14.50    |      -      |      -      |      -      |      -      |   RL   |
|  ExCL   |      -      |    65.1     |    44.1     |    23.3     |      -      |      -      |      -      |      -      |   RL   |
|  PFGA   |      -      |    67.53    |    52.02    |    33.74    |      -      |      -      |      -      |      -      |   PF   |
|  HVTG   |      -      |    61.37    |    47.27    |    23.30    |      -      |      -      |      -      |      -      |   graph based   |
|  MMRG   |     88.27   | 71.60       |44.25        |    -        |      92.35  |      78.67  |      60.22  |      -      |   graph based   |


#### DiDeMo

|                | R@1 IoU@0.1 | R@1 IoU@0.3 | R@1 IoU@0.5 | R@1 IoU@0.7 | R@5 IoU@0.1 | R@5 IoU@0.3 | R@5 IoU@0.5 | R@5 IoU@0.7 |
| :------------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|      TMN       |    22.92    |      -      |      -      |      -      |    76.08    |      -      |      -      |      -      |
|      MCN       |    28.10    |      -      |      -      |      -      |    78.21    |      -      |      -      |      -      |
|      TGN       |    28.23    |      -      |      -      |      -      |    79.26    |      -      |      -      |      -      |
|      MAN       |    27.02    |      -      |      -      |      -      |    81.70    |      -      |      -      |      -      |
| WSLLN (Weakly) |    19.4     |      -      |      -      |      -      |    54.4     |      -      |      -      |      -      |

#### TACoS

|         | R@1 IoU@0.1 | R@1 IoU@0.3 | R@1 IoU@0.5 | R@1 IoU@0.7 | R@5 IoU@0.1 | R@5 IoU@0.3 | R@5 IoU@0.5 | R@5 IoU@0.7 | Method |
| :-----: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :----: |
|   MCN   |    2.62     |    1.64     |    1.25     |      -      |    2.88     |    1.82     |    1.01     |      -      |   PB   |
|  CTRL   |    24.32    |    18.32    |    13.30    |      -      |    48.73    |    36.69    |    25.42    |      -      |   PB   |
|   TGN   |    41.87    |    21.77    |    18.90    |      -      |    53.40    |    39.06    |    31.02    |      -      |   PB   |
|  ACRN   |    24.22    |    19.52    |    14.62    |      -      |    47.42    |    34.97    |    24.88    |      -      |   PB   |
|  ACL-K  |    31.64    |    24.17    |    20.01    |      -      |    57.85    |    42.15    |    30.66    |      -      |   PB   |
|  SCDM   |      -      |    26.11    |    21.17    |      -      |      -      |    40.16    |    32.18    |      -      |   PB   |
|   CBP   |      -      |    27.31    |    24.79    |    19.10    |      -      |    43.64    |    37.40    |    25.59    |   PB   |
| TripNet |      -      |    23.95    |    19.17    |    9.52     |      -      |      -      |      -      |      -      |   RL   |
|  SMRL   |    26.51    |    20.25    |    15.95    |      -      |    50.01    |    38.47    |    27.84    |      -      |   RL   |
|  ABLR   |    34.7     |    19.5     |     9.4     |      -      |      -      |      -      |      -      |      -      |   RL   |
|  ExCL   |      -      |    45.5     |    28.0     |    14.6     |      -      |      -      |      -      |      -      |   PF   |
| CMIN| 32.48| 24.64 | 18.05|-| 62.13| 38.46| 27.02|-| |
|  MMRG   |    85.34    |    57.83    |    39.28    |    -        |    84.37    |     78.38   |     56.34   |      -      |   graph based   |

## Popular Implementations

### PyTorch

- [ikuinen/CMIN_moment_retrieval](https://github.com/ikuinen/CMIN_moment_retrieval)

### TensorFlow

- [jiyanggao/TALL](<https://github.com/jiyanggao/TALL>)
- [runzhouge/MAC](https://github.com/runzhouge/MAC)
- [BonnieHuangxin/SLTA](https://github.com/BonnieHuangxin/SLTA)
- [yytzsy/ABLR_code](https://github.com/yytzsy/ABLR_code)
- [yytzsy/SCDM](https://github.com/yytzsy/SCDM)
- [JaywongWang/TGN](https://github.com/JaywongWang/TGN)
- [JaywongWang/CBP](https://github.com/JaywongWang/CBP)


## Licenses

[![CC0](http://i.creativecommons.org/p/zero/1.0/88x31.png)](http://creativecommons.org/publicdomain/zero/1.0/)
