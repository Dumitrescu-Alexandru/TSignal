# TSignal

TSignal is a deep architecture for Signal Peptide (SP) and cleavage site (CS) prediction inspired by the popular sequence translator model introduced in "Attention is all you need" [1]. We introduce the first unstructured predictor model in this field and provide extensive analysis to visualize its advantages over the structured counterparts. Our model takes advantage of the powerful encoder-decoder architecture of the original transformer natural language translator, to which we incorporate various task-specific additions, to help the model learn in our small data setting. Our model uses transfer learning with pre-trained weights from [4]. For detailed explanations of the architecture, our preprint is now available at [*TSignal: A transformer model for signal peptide prediction*](https://www.biorxiv.org/content/10.1101/2022.06.02.493958v1) [2].

![TSignal architecture](architecture.jpg-1.jpg)

## 1 Installation

Please install Python 3.7.7. Other versions are not guaranteed to work.

### 1.1 Required packages

Use pip install -r requirements.txt

## 2 Data

We use data from [3]. In order to train the model, download from [SignalP 6.0](https://services.healthtech.dtu.dk/service.php?SignalP-6.0) 
the fasta file the training set "SignalP 6.0 Training set" and add it to sp_data/ folder. The benchmark data on which we
report the results is found in "SignalP 5.0 Benchmark set". Note that the latter is not used during training, 
but in the functions of misc/visualization.py, it will be used to report performance on this data if specified so.

The model should automatically set up binaries "sp6_partitioned_data_<test/train>_<fold_no>.bin" in sp_data folder based 
on the downloaded fasta file (which should also be in sp_data folder) the first time a model is trained.
  

## 3 Usage

ProtBERT will automatically be downloaded to sp_data/models folder when training with <tune_bert> parameter set to true.

The base fasta file needs to be in the correct folder for training (refer to Section 2).

### 3.1 Using our deployment model

Download our pre-trained model from [here](https://www.dropbox.com/s/lfuleg9470s7nqx/deployment_sep_pe_swa_extra_inpemb_on_gen_best_eval_only_dec.pth?dl=0) and add it in sp_data folder.


To extract sequence predictions:
- prepare binary file containing a dictionary of the form {sequence:[embedding, label_sequence, life_group, sp-type]}. Since
it is used for testing, all entries may be random, and the binaries are created this way for compatibility with training
dataloaders.

To extract saliency maps of some sequences:

- Follow the same procedures as in extracting predictions, and simply add "--compute_saliency" argument to your call.

### 3.2  Train a new model while tuning bert
First, follow the instructions on Section 2. If you wish to replicate the experiment results, refer to arguments --train_folds 
from main.py. If you wish to create another deployment model (trained on D<sub>1,train</sub>, D<sub>2,train</sub>, 
D<sub>3,train</sub>; validated on D<sub>1,test</sub>, D<sub>2,test</sub>, D<sub>3,test</sub>) refer to --deployment_model 
argument in main.py.

### 3.3  Train a new model while not tuning bert

Currently, a non-tuning-ProtBERT model only works with global label predictions. Therefore, use_glbl_lbls needs to be 
true and glbl_lbl_version to the desired SP type prediction approach. Additionally, for efficiency reasons, when not 
tuning the ProtBERT model, the embeddings for a dataset are pre-computed. When not calling main.py with --tune_bert, 
these will be automatically computed for SignaP 6.0 data and the load will starts from there.

### 3.4  Test on your own sequences

If you want to test the model on your own sequences, please refer to create_test_files.py script, method for the method
create_test_file(). If you want to use our pre-trained model, refer to Example 1 in example_runs.txt. To also create a 
saliency map for the predictions, add --compute_saliency to the example 1.

### 3.5  Test the performance of the model

- To re-create a similar experiment on SignalP 6.0 data, 3 different runs are needed (refer to example 2 from
example_runs.txt and its comments). At the end, extract_all_param_results method from misc/visualize_cs_pred_results.py 
may be used.
- If a different dataset needs to be tested, refer to the 3.4 to create your own sequence file. After predicting the 
sequences, use the correct and true predictions with misc/visualize_cs_pred_results.py methods get_pred_perf_sptype and get_cs_perf
to get the SP type/CS performance on your predictions.


## References 
[1] Vaswani, A. et al. (2017). Attention is All you Need. In I. Guyon,
U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan,
and R. Garnett, editors, Advances in Neural Information Processing
Systems, volume 30. Curran Associates, Inc.

[2] Dumitrescu, A. et al. (*submitted in* 2022). TSignal: A transformer model for signal peptide prediction.  

[3] Teufel, F. et al. (2022). SignalP 6.0 predicts all five types of signal peptides
using protein language models. Nature Biotechnology.

[4] Elnaggar, A. et al. (2021). ProtTrans: Towards Cracking the Language
of Lifes Code Through Self-Supervised Deep Learning and High
Performance Computing. IEEE Transactions on Pattern Analysis and
Machine Intelligence, pages 1–1.