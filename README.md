# A Heterogeneous Hypergraph-Transformer Hybrid Architecture for Business Process Next-Activity Prediction 

This repository contains the complete source code and datasets for the project titled "A Heterogeneous Hypergraph-Transformer Hybrid Architecture for Business Process Next-Activity Prediction".  

The `data` folder includes most datasets you may need, which have been pre-split for three-fold cross-validation. The `config.py` file configures basic hyperparameters. `get_causal_rule.py` is used for causal relationship extraction, allowing you to choose between binary or multivariate relationships. `HeteroHG_Batch.py` and `model.py` store the core model structure and encapsulation. `Read_all_log.py` handles data preprocessing and prefix extraction.  `test.py` and `train.py` are used for model testing and training, respectively.

## Usage
First, run `Read_all_log.py` to process the data. 
Then execute `get_causal_rule.py` to obtain the causal relationships. 
Finally, run `train.py` for model training. 
After the training is completed, execute `test.py` to perform testing.

## Environment
python == 3.11.11  

**Core Dependencies (Required for Project Execution)**
torch==2.1.2+cu121        
dgl==2.1.0+cu121          
dhg==0.9.4                
pandas==2.2.2             
numpy==1.26.4             
scikit-learn==1.5.1      
scipy==1.11.4         
**Causal Relationship Extraction (for get_causal_rule.py)**
causal-learn==0.1.4.3    
cdt==0.6.0               
**Business Process Data Processing (for Read_all_log.py)**
pm4py==2.7.11.12         
**Auxiliary Tools (Ensure Smooth Code Execution)**
tqdm==4.66.5             
pyyaml==6.0.2            
pydot==4.0.1             

## Baseline
### MiDA
**Reference**: V. Pasquadibisceglie, A. Appice, G. Castellano, and D. Malerba, “A
 multi-view deep learning approach for predictive business process mon
itoring,” IEEE Transactions on Services Computing, vol. 15, no. 4, pp.
 2382–2395, 2022.
### MHG-Predictor
**Reference**:  J. Wang, Y. Yu, N. Fang, B. Cao, J. Fan, and J. Zhang, “Mhg-predictor:
 A multi-layer heterogeneous graph-based predictor for next activity in
 complex business processes,” in Companion Proceedings of the ACM
 on Web Conference 2025, 2025, pp. 500–509.
### HiGNN
**Reference**:  J. Wang, C. Lu, Y. Yu, B. Cao, K. Fang, and J. Fan, “Higpp: A
 history-informed graph-based process predictor for next activity,” in
 International Conference on Service-Oriented Computing. Springer,
 2025, pp. 337–353.
### MiTFM
**Reference**: J. Wang, C. Lu, B. Cao, and J. Fan, “Mitfm: A multi-view information
 fusion method based on transformer for next activity prediction of
 business processes,” in Proceedings of the 14th Asia-Pacific Symposium
 on Internetware, 2023, pp. 281–291
### RLHGNN
**Reference**:Wang, Jiaxing, et al. "RLHGNN: Reinforcement Learning-driven Heterogeneous Graph Neural Network for Next Activity Prediction in Business Processes." arXiv preprint arXiv:2507.02690 (2025).
### JARVIS
**Reference**:  V. Pasquadibisceglie, A. Appice, G. Castellano, and D. Malerba, “Jarvis:
 Joining adversarial training with vision transformers in next-activity
 prediction,” IEEE Transactions on Services Computing, vol. 17, no. 4,
 pp. 1593–1606, 2024.
### SGAP
**Reference**:  Y. Deng, J. Wang, C. Wang, C. Zheng, M. Li, and B. Li, “Enhancing
 predictive process monitoring with sequential graphs and trace atten
tion,” in 2024 IEEE International Conference on Web Services (ICWS).
 IEEE, 2024, pp. 406–415.
