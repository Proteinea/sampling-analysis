import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from typing import Dict, List
from Bio import SeqIO
from stateval.src.identities import (
    Identities,
)
import os
from stateval.src.cath import Cath
import matplotlib.pylab as plt
from stateval.src.configs import ClustaloConfig, IdentitiesConfig, MSAConfig
from stateval.src.msa import Msa 
from stateval.src.sh_entropies import ShannonEntropies 
import numpy as np
import wandb

def fasta2dict(infile_path:str)->Dict[str, list]:
    """read fasta file 

    Args:
        infile_path (str): path to the file

    Returns:
        Dict[str, list]: dict containing IDs and sequences 
    """
    tmp = SeqIO.parse(infile_path,'fasta')
    fasta_dict = {'ID':[],'Seq':[]}
    for record in tmp:
        fasta_dict['ID'].append(record.id)
        fasta_dict['Seq'].append(str(record.seq))
    return fasta_dict 


def generate_variants(model:AutoModelForCausalLM, tokenizer:AutoTokenizer, sequences:List[str], config:Dict, output_file:str)-> List[str]:
    generated_sequences = []
    batch_size = config["generate_batch_size"]
    for batch_idx in range(0, len(sequences), batch_size):
            inputs = tokenizer.batch_encode_plus(
                sequences[batch_idx:batch_idx+batch_size],
                padding="longest",
                truncation=True,
                max_length=512,  # Set the maximum length to 512
                return_tensors="pt"
            ).to(device)

            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            # Generate multiple sequences
            outputs = model.generate(
                input_ids[:, :config["sequence_prompt_index"]],
                attention_mask=attention_mask[:, :config["sequence_prompt_index"]],
                **config["generate_kwargs"]
                # num_return_sequences=5
            )
            # Decode and store the generated sequences
            for generated_output in outputs:
                generated_text = tokenizer.decode(generated_output, skip_special_tokens=True)
                generated_text = generated_text.replace("|endoftext|>", "")  # Remove the header
                generated_sequences.append(generated_text)
    generated_sequences = list(set(generated_sequences))
    with open(output_file, "w") as outf:
        for i, seq  in enumerate(generated_sequences):    
            outf.write(f">gen_seq_{i}\n")
            outf.write(f"{seq}\n")


device = "cuda" if torch.cuda.is_available() else "cpu"
with open('generation_config.yml', 'r') as file:
    config  = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint_path"])
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token
model = AutoModelForCausalLM.from_pretrained(config["model_checkpoint_path"]).to(device)
model.config.max_length = 512  # Set the maximum length to 512

input_file = config["reference_file"]  # Replace with the path to your input file
# # exit()
identity = Identities(IdentitiesConfig(backend_config=ClustaloConfig(), closest=True))
msa = Msa(MSAConfig())
cath =  Cath()
se_calc = ShannonEntropies()
real_data_internal_identities = identity.compute_clustalo_identities(config["reference_file"])
real_data_internal_identities_plot = identity.plot_identities_displot(real_data_internal_identities, title="real data internal identities")
input_sequences = fasta2dict(input_file)["Seq"]
input_sequences = [v for v in input_sequences if v]
real_hits = cath.search_cath(config["reference_file"])
real_hits_df = cath.hits_to_df(real_hits)
unique_per_super_family = real_hits_df.groupby("superfamily")["domain"].aggregate(lambda x: len(np.unique(x)))

for key in config["change_list"]:
    for v in config["change_list"][key]:
        
        output_file = f"experiments_mdh/{config['experiment_prefix']}_{key}_{v}.fasta"
        config["generate_kwargs"][key] = v
        generate_variants(model, tokenizer, input_sequences, config, output_file)        
        # exit()
        wandb.init(
                config=config["generate_kwargs"],
                project=config["wandb_project"],
                name=f"{config['experiment_prefix']}_{key}_{v}")
        internal_identities = identity.compute_clustalo_identities(output_file)
        
        plt.figure(figsize=(12, 8))
        internal_identities_plot = identity.plot_identities_displot(internal_identities, title="generated internal identities")

        wandb.log({"generated_internal_identities": wandb.Image(internal_identities_plot.figure)})
        plt.clf()
        cross_identities = identity.compute_clustalo_identities(output_file, reference=config["reference_file"], )
        
        cross_identities_plot = identity.plot_identities_displot(cross_identities, title="generated cross identities")
        wandb.log({"generated_cross_identities": wandb.Image(cross_identities_plot.figure)})
        wandb.log({"real_data_internal_identities": wandb.Image(real_data_internal_identities_plot.figure)})
        plt.clf() 
        
        generated_seqs = fasta2dict(output_file)
        ref_seqs = fasta2dict(config["reference_file"])
        sequences_to_align = { generated_seqs["ID"][i]: seq for i, seq in enumerate(generated_seqs["Seq"])}
        sequences_to_align.update({ ref_seqs["ID"][i]: seq for i, seq in enumerate(ref_seqs["Seq"])})
        alignment = msa.align(sequences_to_align, )
        entropies = se_calc.calculate_entropies(aligned_seqs=alignment, original_splits={"reference":ref_seqs["ID"], "generated":generated_seqs["ID"]})
        plt.figure(figsize=(12, 8))
        entropies_plot = se_calc.plot_entropies(entropies_df=entropies)
        wandb.log({"shannon_entropies": wandb.Image(entropies_plot.figure)})
        plt.clf()

        generated_hits = cath.search_cath(output_file)
        generated_hits_df = cath.hits_to_df(generated_hits)

        generated_unique_per_super_family = generated_hits_df.groupby("superfamily")["domain"].aggregate(lambda x: len(np.unique(x)))
        wandb.log({
            "unique_generated_sequences" : len(set(generated_seqs["Seq"])),
            "gen_unique_cath_comains": generated_hits_df["domain"].nunique(),
            "gen_unique_superfamily": generated_hits_df["superfamily"].nunique(),
            "real_unique_cath_comains": real_hits_df["domain"].nunique(),
            "real_unique_superfamily": real_hits_df["superfamily"].nunique(),
            })
        wandb.save(output_file)
        
        fig, ax = plt.subplots()
        ax.set_title("unique cath domains per superfamily")
        ax.bar(generated_unique_per_super_family.index, height=generated_unique_per_super_family.values, alpha=0.8 , label="generated")
        ax.bar(unique_per_super_family.index, height=unique_per_super_family.values, alpha=0.5, label="reference")
        ax.legend()
        wandb.log({"unique_cath_domains_per_superfamily": wandb.Image(fig)})
        plt.clf()
        wandb.finish()