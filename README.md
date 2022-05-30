# TSignal

TSignal is a deep architecture for Signal Peptide (SP) and cleavage site (CS) prediction inspired by the popular sequence translator model introduced in "Attention is all you need paper" [1]. We introduce the first unstructured predictor model in this field and provide extensive analysis to visualize its advantages over the structured counterparts. Our model takes advantage of the powerful encoder-decoder architecture of the original transformer natural language translator, to which we incorporate various task-specific additions, to help the model learn in our small data setting. Our model uses transfer learning with pre-trained weights from [4]. For detailed explanations of the architecture, see our paper *TSignal: A transformer model for signal peptide prediction* [2].

![TSignal architecture](architecture.jpg-1.jpg)

## Installation

Please install Python 3.7.7. Other versions are not guaranteed to work.

### Required packages

Use pip install -r requirements.txt

## Data

We use data from [3]. In order to train the model, download from [SignalP 6.0](https://services.healthtech.dtu.dk/service.php?SignalP-6.0).

In order to set up the same training procedure, do:
- run read_extract_sp6_data.py from sp_data/sp6_data/
- ...
  

## Usage

### Using our deployment model

Download our pre-trained model from [here](https://www.dropbox.com/s/lfuleg9470s7nqx/deployment_sep_pe_swa_extra_inpemb_on_gen_best_eval_only_dec.pth?dl=0) and add it in sp_data folder.

To extract sequence predictions:
- prepare binary sequence file with:...

To extract saliency maps of some sequences:

- To extract sequence predictions use python main.py --test_seqs ...
- to extract the saliency map of some prediction use ...; a binary file with the gradients wrt. the inputs will be saved as "blabla.bin"


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