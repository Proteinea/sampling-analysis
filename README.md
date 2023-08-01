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
