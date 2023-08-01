# sampling-analysis 
## 1) Fine-tuning ProtGPT2 for Generation
To start from fine-tuning ProtGpt2 on a set of unuspervised sequences, you'll need these few commands: 
### 1.a Install Transformers from source: 
`pip install git+https://github.com/huggingface/transformers`
### 1.b Clone the Transformers repo: 
`git clone https://github.com/huggingface/transformers.git`
### 1.c Replace HuggingFace's run_clm.py 
Change your directory to: 
`cd transformers/examples/pytorch/language-modeling `
Replace the "run_clm.py"file in the following directory with the one in this repo. 
### 1.d Format your data to be compatible with ProtGPT2 required input 
Now, to meet the formating required by ProtGP2, the authors adivse the following if your starting point is fasta files: <br>
    (i) substitute the FASTA headers for each sequence with the expression "<|endoftext|>" and <br>
    (ii) split the originating dataset into training and validation .txt files (this is often done with the ratio 90/10, 80/20 or 95/5). The fine-tuning requires only training and validation sets, **please exclude your test set prior to fine-tuning.** <br>

### 1.e Upload your train, validation, and test.txt files to the same directory 
The repo contains examples for the desired
