import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
from typing import Dict, List
from stateval.src.identities import (
    Identities,
)
from stateval.src.cath import Cath
import matplotlib.pylab as plt
from stateval.src.configs import MMseqsConfig, IdentitiesConfig, MSAConfig
from stateval.src.msa import Msa 
from stateval.src.sh_entropies import ShannonEntropies 
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
import wandb
from stat_eval_utils import fasta2dict
from stat_eval_utils import novelty
from stat_eval_utils import diversity
from stat_eval_utils import mse
from stat_eval_utils import get_position_property_frequency
from stat_eval_utils import get_property_precentages
from ot.sliced import sliced_wasserstein_distance



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
            print(input_ids.shape)
            with torch.no_grad():

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
    generated_sequences = [s for s in generated_sequences if s not in sequences]
    with open(output_file, "w") as outf:
        for i, seq  in enumerate(generated_sequences):    
            outf.write(f">gen_seq_{i}\n")
            outf.write(f"{seq}\n")
    return generated_sequences


device = "cuda" if torch.cuda.is_available() else "cpu"
with open('generation_config.yml', 'r') as file:
    config  = yaml.safe_load(file)

tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint_path"])
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add a padding token
model = AutoModelForCausalLM.from_pretrained(config["model_checkpoint_path"], device_map="auto", load_in_8bit=True)
model.to_bettertransformer()
model.config.max_length = 512  # Set the maximum length to 512

input_file = config["reference_file"]  # Replace with the path to your input file
# # exit()
identity = Identities(IdentitiesConfig(backend_config=MMseqsConfig(), closest=True))
msa = Msa(MSAConfig(max_gap_ratio=config["max_gap_ratio"]))
cath =  Cath()
se_calc = ShannonEntropies()
real_data_internal_identities = identity.compute_mmseqs2_identities(config["reference_file"], config["reference_file"])
real_data_internal_identities_plot = identity.plot_identities_displot(real_data_internal_identities, title="real data internal identities")
input_sequences = fasta2dict(input_file)["Seq"]
input_sequences = [v for v in input_sequences if v]
real_hits = cath.search_cath(config["reference_file"])
real_hits_df = cath.hits_to_df(real_hits)
normalized_real_internal_identities = real_data_internal_identities["identity"].to_numpy()/real_data_internal_identities["identity"].sum() 
unique_per_super_family = real_hits_df.groupby("superfamily")["domain"].aggregate(lambda x: len(np.unique(x)))

for key in config["change_list"]:
    for v in config["change_list"][key]:
        
        output_file = f"experiments_mdh/{config['experiment_prefix']}_{key}_{v}.fasta"
        config["generate_kwargs"][key] = v
        outputs = generate_variants(model, tokenizer, input_sequences, config, output_file)        
        wandb.init(
                config=config,
                entity=config["wandb_entity"],
                project=config["wandb_project"],
                name=f"{config['experiment_prefix']}_{key}_{v}")

        if len(outputs) <= 2: 
            wandb.log({"unique_generated_sequences" : len(set(generated_seqs["Seq"])), })
            wandb.finish()
            continue

        
        internal_identities = identity.compute_mmseqs2_identities(output_file, output_file)
        
        normalized_internal_identities = internal_identities["identity"].to_numpy()/internal_identities["identity"].sum() 
        wandb.log({"internal_identities_wasserstein_dist": sliced_wasserstein_distance(normalized_real_internal_identities.reshape(-1, 1), normalized_internal_identities.reshape(-1, 1))})

        plt.figure(figsize=(12, 8))
        internal_identities_plot = identity.plot_identities_displot(internal_identities, title="generated internal identities")
        internal_identities_plot.tight_layout()
        wandb.log({"generated_internal_identities": wandb.Image(internal_identities_plot.figure)})
        plt.clf()
        cross_identities = identity.compute_mmseqs2_identities(output_file, reference=config["reference_file"], )
        
        cross_identities_plot = identity.plot_identities_displot(cross_identities, title="generated cross identities")
        cross_identities_plot.tight_layout()
        real_data_internal_identities_plot.tight_layout()
        wandb.log({"generated_cross_identities": wandb.Image(cross_identities_plot.figure)})
        wandb.log({"real_data_internal_identities": wandb.Image(real_data_internal_identities_plot.figure)})
        plt.clf() 
        
        generated_seqs = fasta2dict(output_file)
        ref_seqs = fasta2dict(config["reference_file"])
        sequences_to_align = { generated_seqs["ID"][i]: seq for i, seq in enumerate(generated_seqs["Seq"])}
        sequences_to_align.update({ ref_seqs["ID"][i]: seq for i, seq in enumerate(ref_seqs["Seq"])})
        alignment = msa.align(sequences_to_align, )
        entropies = se_calc.calculate_entropies(aligned_seqs=alignment, original_splits={"reference":ref_seqs["ID"], "generated":generated_seqs["ID"]})
        wandb.log({"se_spearman": spearmanr(entropies[entropies.columns[0]], entropies[entropies.columns[1]])[0],
                   "se_mse": mse(entropies[entropies.columns[0]], entropies[entropies.columns[1]])})
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
        max_length = max([len(s) for s in generated_seqs["Seq"]+ref_seqs["Seq"]])
        wandb.log({"max_legnth": max_length})
        gen_novelity = novelty(generated_seqs["Seq"], ref_seqs["Seq"])
        gen_diversity = diversity(generated_seqs["Seq"])
        wandb.log({
            "novelty": gen_novelity,
            "diversity": gen_diversity})
        gen_sequence_property_positions = get_position_property_frequency(generated_seqs["Seq"], max_length)
        real_sequence_property_positions = get_position_property_frequency(ref_seqs["Seq"], max_length)
        for key in gen_sequence_property_positions.keys():
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(gen_sequence_property_positions[key], density=True, alpha=0.8, label="gen_"+key)
            # if cl_args.use_diff_position_set:
            #     for p in diff_set:
            #         for i in range(len(bins)-1):
            #             if p >= bins[i] and p <= bins[i+1] :
            #                 patches[i].set_fc("r")
            #                 break
            ax.hist(real_sequence_property_positions[key], density=True, alpha=0.5, label="real_"+key)
            normalized_property_gen = np.array(gen_sequence_property_positions[key])/sum(gen_sequence_property_positions[key])
            normalized_property_real = np.array(real_sequence_property_positions[key])/sum(real_sequence_property_positions[key])
            wandb.log({f"{key}_wasserstein_dist": sliced_wasserstein_distance(normalized_property_gen.reshape(-1, 1), normalized_property_real.reshape(-1, 1))})
            ax.legend()
            wandb.log({f"{key}_hist": wandb.Image(fig)})
            plt.clf()
        plt.figure(figsize=(14,5))
        gen_percents = get_property_precentages(generated_seqs["Seq"])
        real_percents = get_property_precentages(ref_seqs["Seq"], "real")
        gen_real_df = pd.concat([gen_percents, real_percents])
        boxplot = sns.boxplot(gen_real_df, x="property", y="percent", hue="seq_type")
        wandb.log({"composition comparison": wandb.Image(boxplot)})
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_title("unique cath domains per superfamily")
        ax.bar(generated_unique_per_super_family.index, height=generated_unique_per_super_family.values, alpha=0.8 , label="generated")
        ax.bar(unique_per_super_family.index, height=unique_per_super_family.values, alpha=0.5, label="reference")
        ax.legend()
        wandb.log({"unique_cath_domains_per_superfamily": wandb.Image(fig)})
        plt.clf()
        wandb.finish()