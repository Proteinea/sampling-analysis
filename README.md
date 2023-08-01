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
Replace the "run_clm.py" file in the following directory with the one in this repo. The modifications include automatic loading from the last checkpoint when fine-tuning as well as better control over changing several training parameters starting from a specific checkpoint (e.g., training epochs, batch size, learning rate, etc.) but most importantly it enables pre-training from scratch vs. fine-tuning using the same script with just a few argument changes.
### 1.d Format your data to be compatible with ProtGPT2 required input 
Now, to meet the formating required by ProtGP2, the authors adivse the following if your starting point is fasta files: <br>
    (i) substitute the FASTA headers for each sequence with the expression "<|endoftext|>" and <br>
    <br>
    (ii) split the originating dataset into training and validation .txt files (this is often done with the ratio 90/10, 80/20 or 95/5). The fine-tuning requires only training and validation sets, **please exclude your test set prior to fine-tuning.** <br>
    <br>
    (iii) you can do any problem-specific data pre-processing prior to this step (e.g., max_length truncation) <br>
    <br>
    (iv) I attached a my script, "ProtGPT2_Input_preprocessing.py" for converting a pre-processed fasta file into the desired format and do the splitting into 3 .txt files according to the desired split ratio<br>
    <br>
    (v) Also attached is an example .fasta file that can be used <br>
    <br>
### 1.e Upload your train, validation, and test.txt files to the same directory 
The repo contains examples for the desired
### 1.f Fulfill Dependencies 
Providing you don't want to use an the attached environment and that you're using the default machine, you'll need a few dependencies. Here're there installation commands: <br>
    (i) `pip install datasets` <br> 
    (ii) `pip install evaluate` <br>
    (iii) `pip install torch` <br> 
    (iv) `pip install transformers[torch]`
    (v) `pip install scikit-learn`
### 1.g Start the fine-tuning
Run the following command to start the fine-tuning with the specified training argument values. Feel free to change them according to your use case and also feel free to check the available arguments and their default values in the modified run_clm.py script: <br>
`python3 run_clm.py --model_name_or_path nferruz/ProtGPT2 --train_file train.txt --validation_file validation.txt --tokenizer_name nferruz/ProtGPT2 --do_train --do_eval --output_dir output --learning_rate 1e-05 --block_size 512 --per_device_train_batch_size 2 --num_train_epochs 4` <br> <br>
**It's advised to use a screen prior to fine-tuning start** <br>
<br>
<br>
## 2) Pre-training ProtGPT2 for Generation from Scratch
The existing ProtGPT2 model on HuggingFace is built on the base gpt2 architecture whose parameter account is ~760M. Providing the training data is large enough, you can use the same model loaded in the fine-tuning example. However, for my use case the trianing data was definitely inadequate to scale with such a model. Therefore, I loaded a smaller gpt2 version "gpt2 which is around 124M parameter, while loading the pre-trained tokenizer of the ProtGPT2 model. 
To do the same, do the same steps as the fine-tuning example but replace the fine-tuning command with the one below:<br> 
`python3 run_clm.py --model_type gpt2 --tokenizer_name nferruz/ProtGPT2 --output_dir output --do_train --train_file train.txt --validation_file validation.txt --learning_rate 1e-4 --num_train_epochs 10 --per_device_train_batch_size 2 --block_size 512 --config_overrides "num_hidden_layers=6,num_attention_heads=8"  --overwrite_output_dir` <br> <br>
**As the last example, it's advised to use a screen prior to training start** <br>

## 3) Generating Sequences Using the Fine-tuned/Pre-trained Model 
Providing you are through with fine-tuning/pre-training your model from scratch, you can now use it for generation. With this regard, run the "ProtGPT2_generate.py" attached script. 
Do not forget to specify the directories of both the model and tokenizer (e.g., do u already have the model on ur machine? do you need to load it from a bucket first? etc.). Furthermore, the directory to the input dataset should denote the excluded test set to be used for prompting.
