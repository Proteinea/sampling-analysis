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
from scipy.stats import pearsonr
import wandb
from transformers.generation import LogitsProcessorList
from stat_eval_utils import fasta2dict
from stat_eval_utils import novelty
from stat_eval_utils import diversity
from stat_eval_utils import mse
from stat_eval_utils import get_position_property_frequency
from stat_eval_utils import get_property_precentages
from ot.sliced import sliced_wasserstein_distance
from transformers.generation.logits_process import LogitsProcessor
from typing import List

class LogitEntropyProcessor(LogitsProcessor):
    
    def __init__(self, segments_indicies:List[int], temps:List[float], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert len(temps) >= 1
        self.temps = temps
        self.segments_indicies = segments_indicies
        self.temp_index = 0
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
         
        scores = scores/ self.temps[self.temp_index]

        if input_ids.shape[1]+1 > self.segments_indicies[self.temp_index] and self.temp_index +1 < len(self.temps):
            self.temp_index += 1
        return scores



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
with open('entropy_configs.yml', 'r') as file:
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
input_file = config["msa_seqs_file"] 
sequences = fasta2dict(input_file)
sequences = {id: seq for id, seq in zip(sequences["ID"], sequences["Seq"])}

msa_entropies = Msa(MSAConfig(max_gap_ratio=0.8))
aligned_seqs = msa_entropies.align(sequences)
se = ShannonEntropies()
entropies = se.calculate_entropies(aligned_seqs)
se_calc = ShannonEntropies()
real_data_internal_identities = identity.compute_mmseqs2_identities(config["reference_file"], config["reference_file"])
real_data_internal_identities_plot = identity.plot_identities_displot(real_data_internal_identities, title="real data internal identities")
input_sequences = fasta2dict(input_file)["Seq"]
input_sequences = [v for v in input_sequences if v]
real_hits = cath.search_cath(config["reference_file"])
real_hits_df = cath.hits_to_df(real_hits)
normalized_real_internal_identities = real_data_internal_identities["identity"].to_numpy()/real_data_internal_identities["identity"].sum() 
unique_per_super_family = real_hits_df.groupby("superfamily")["domain"].aggregate(lambda x: len(np.unique(x)))

for gen_kwargs in config["entropy_configs"]:

    n_segments = gen_kwargs["n_segments"]
    indicies = np.linspace(0, len(entropies), n_segments+1).astype(int)
    entropies_median = [np.median(entropies[indicies[i]: indicies[i+1]]) for i in range(n_segments)] # median instead of mean so it won't be affected by outliers
    temps = [gen_kwargs["base_temp"]+ v* gen_kwargs["entropy_factor"] for v in entropies_median] # a base temp in case the entropy is too low 
    name = "_".join([f"{k}_{v}" for k,v in gen_kwargs.items()])
    wandb.init(
            config=config,
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=name
            )
    logits_processor = LogitsProcessorList([LogitEntropyProcessor(segments_indicies=indicies[1:], temps=temps)])

    output_file = f"experiments_mdh_eval_entropy/{name}.fasta"
    gen_config = {}
    gen_config["generate_batch_size"] = config["generate_batch_size"]
    gen_config["sequence_prompt_index"] = config["sequence_prompt_index"]
    gen_config["generate_kwargs"] = {
         "eos_token_id" : config["eos_token_id"],
         "do_sample" :True,
         "logits_processor" : logits_processor,
         "use_cache":True,
         "max_length": 512,
         }
    outputs = generate_variants(model, tokenizer, input_sequences, gen_config, output_file)        
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
    wandb.log({"se_spearman": pearsonr(entropies[entropies.columns[0]], entropies[entropies.columns[1]])[0],
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
    for property_name in gen_sequence_property_positions.keys():
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(gen_sequence_property_positions[property_name], density=True, alpha=0.8, label="gen_"+property_name)
        ax.hist(real_sequence_property_positions[property_name], density=True, alpha=0.5, label="real_"+property_name)
        normalized_property_gen = np.array(gen_sequence_property_positions[property_name])/sum(gen_sequence_property_positions[property_name])
        normalized_property_real = np.array(real_sequence_property_positions[property_name])/sum(real_sequence_property_positions[property_name])
        wandb.log({f"{property_name}_wasserstein_dist": sliced_wasserstein_distance(normalized_property_gen.reshape(-1, 1), normalized_property_real.reshape(-1, 1))})
        ax.legend()
        wandb.log({f"{property_name}_hist": wandb.Image(fig)})
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