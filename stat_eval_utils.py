 
"""
Helper functions for data processing
"""
from typing import Dict, Union, List, Set
import json 
import numpy as np
from Bio import SeqIO
import argparse
import pandas as pd
import os
from tqdm import tqdm
from pandas import DataFrame
from Bio import pairwise2

def mse(x, y) : return np.mean(np.square(x-y)) 


# Define sets of amino acids for each property
HYDROPHOBIC_AMINO_ACIDS = {"A", "C", "I", "L", "M", "F", "W", "V"}
HYDROPHILIC_AMINO_ACIDS = {"R", "N", "D", "Q", "E", "K"}
AROMATIC_AMINO_ACIDS = {"F", "W", "Y"}
SMALL_AMINO_ACIDS = {"A", "G", "S", "T"}
POSITIVE_AMINO_ACIDS = {"R", "K", "H"}
NEGATIVE_AMINO_ACIDS = {"D", "E"}
ALIPHATIC_AMINO_ACIDS = {"G", "A", "V", "L", "I", "M"}
HYDROXY_AMINO_ACIDS = {"S", "T", "Y"}
POLAR_UNCHARGED_AMINO_ACIDS = {"N", "Q", "S", "T", "Y"}


def diversity(seqs:list):
    norm = len(seqs) * (len(seqs) - 1)
    ret = 0
    for i, veci in tqdm(enumerate(seqs)):
        for j, vecj in enumerate(seqs):
            if i != j:
                ret += pairwise2.align.globalxx(veci, vecj, score_only=True) #np.linalg.norm(veci - vecj)#np.dot(veci.T, vecj)
    
    return ret / norm

def novelty(gseq:list, dseq:list):
    norm = len(gseq)
    ret = 0
    for i, vec in tqdm(enumerate(gseq)):
        mind = 1000000000
        for j, dvec in enumerate(dseq):
            d = pairwise2.align.globalxx(vec, dvec, score_only=True) #np.linalg.norm(vec - dvec)#np.dot(vec.T , dvec)
            if d < mind:
                mind = d
        ret += mind    
    return ret / norm 

def get_mutation_positions(seqs: List[str], wt: str=None)-> Set[int]:
    """gets positions where the sequences differ from each other 

    Args:
        seqs (List[str]): list of sequences

    Returns:
        Set[int]: list of positions
    """
    min_length = min([len(s) for s in seqs])
    # print(min_length)
    seqs = [np.array(list(s[:min_length])) for s in seqs]
    # print(seqs[:2])
    all_positions = []
    if wt is not None:
        for s in seqs:
            all_positions.extend(np.where((np.array(list(s)) != np.array(list(wt))))[0].tolist())
        return set(all_positions)
    for i1 in range(len(seqs)):
        for i2  in range(i1+1, len(seqs)): 
            all_positions.extend(np.where(seqs[i1] == seqs[i2])[0].tolist())
    return set(all_positions)

def get_property_precentages(seqs: List[str], seq_type="gen")->DataFrame:
    """_summary_

    Args:
        seqs (List[str]): _description_
        seq_type (str, optional): _description_. Defaults to "gen".

    Returns:
        DataFrame: 
    """
    percents = {"percent":[], "property": [], "seq_type":[]}
    for s in seqs:
        seq_properties = classify_amino_acids_properties(s)
        property_names = seq_properties[0].keys()
        for key in property_names:
             precent = sum([v[key] for v in seq_properties])/len(s)
             percents["percent"].append(precent)
             percents["property"].append(key)
             percents["seq_type"].append(seq_type)
    return pd.DataFrame(percents) 

def classify_amino_acids_properties(protein_sequence):
    """Tests an amino acid for all of the given properties.

    Args:
    protein_sequence: A string representing the protein sequence.
    amino_acid: A string representing the amino acid to test.

    Returns:
    A dictionary mapping property names to boolean values, indicating whether the amino acid has the given property.
    """
    sequence_properties= []
    for amino_acid in protein_sequence:
        properties = {}

        # Hydrophobicity
        properties["hydrophobic"] = amino_acid in HYDROPHOBIC_AMINO_ACIDS

        # Hydrophilicity
        properties["hydrophilic"] = amino_acid in HYDROPHILIC_AMINO_ACIDS

        # Aromaticity
        properties["aromatic"] = amino_acid in AROMATIC_AMINO_ACIDS

        # Size
        properties["small"] = amino_acid in SMALL_AMINO_ACIDS

        # Charge
        properties["positive"] = amino_acid in POSITIVE_AMINO_ACIDS
        properties["negative"] = amino_acid in NEGATIVE_AMINO_ACIDS

        # Aliphaticity
        properties["aliphatic"] = amino_acid in ALIPHATIC_AMINO_ACIDS

        # Hydroxylation
        properties["hydroxy"] = amino_acid in HYDROXY_AMINO_ACIDS

        # Polarity

        properties["polar_uncharged"] = (
            amino_acid in POLAR_UNCHARGED_AMINO_ACIDS and not properties["positive"] and not properties["negative"])
        properties["charged"] = properties["positive"] or properties["negative"]
        sequence_properties.append(properties)

    return sequence_properties

def get_position_property_frequency(sequences:List[str], max_length:int):
    positions = {
        "hydrophobic": [],
        "hydrophilic": [],
        "aromatic": [],
        "small": [],
        "positive": [],
        "negative": [],
        "aliphatic": [],
        "hydroxy": [],
        "polar_uncharged": [],
        "charged": [],
        }
    for g in sequences:
        properties = classify_amino_acids_properties(g)
        for i, v in enumerate(properties):
            for key in v.keys():
                if v[key]:
                    positions[key].append(i+1) 
    return positions


def str2bool(v:Union[str,bool])->bool:
    """str to bool

    Args:
        v (Union[str,bool]): 

    Raises:
        argparse.ArgumentTypeError:

    Returns:
        bool
    """
    
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# def get_tokenizer(vocab:Union[Dict[str,int], str])-> Tokenizer:
#     """character based tokenizer
#     Args:
#         vocab (Union[Dict[str,int], str]): path to a json file or a dictionary that contains { amino_acid : index }, the vocab dictionary is expected to have <sos> <pad> <end> tokens 

#     Returns:
#         tokenizers.Tokenizer: a tokenizer configured for sequence tasks
#     """
#     if isinstance(vocab, str):
#         with open(vocab, "r") as f:
#             vocab= json.load(f)
#     tokenizer = Tokenizer(models.Unigram(vocab=list(vocab.items())))
#     tokenizer.add_special_tokens(["<pad>", "<eos>", "<sos>"])
#     tokenizer.decoder = decoders.ByteLevel()
#     tokenizer.post_processor = processors.TemplateProcessing(
#         single=f"<sos>:0 $A:0 <eos>:0",
#         special_tokens=[("<sos>", vocab["<sos>"]), ("<eos>", vocab["<eos>"]), ("<pad>", vocab["<pad>"])],
#     )
#     tokenizer.enable_padding(pad_id=vocab["<pad>"])
#     return tokenizer


def fasta2dict(infile_path:str)->Dict[str, list]:
    """_summary_

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


def save_sequences(file_path:str, sequences_to_save:List[List[str]], existing_sequences:List[str], sequence_prefix, save_unique=True)-> None:
    """

    Args:
        file_path (str): path to save the file
        sequences_to_save (List[List[str]]):
        existing_sequences (List[str]): sequences to check unique 
        sequence_prefix (_type_): prefix to sequence IDs
        save_unique (bool, optional): only save unique_sequences. Defaults to True.
    """
    unique_seqs = []
    with open(file_path, 'w') as f:
        for seed_index in range(len(sequences_to_save)):
            if isinstance(sequences_to_save[seed_index], str) :  
                if sequences_to_save[seed_index] in existing_sequences and save_unique: continue
                if sequences_to_save[seed_index] in unique_seqs and save_unique: continue
                unique_seqs.append(sequences_to_save[seed_index])
                f.write(f'>{sequence_prefix}_{seed_index}_seqntG\n')
                f.write(f'{sequences_to_save[seed_index]}\n')
                continue
            
            for seed_step in range(len(sequences_to_save[seed_index])):
                if sequences_to_save[seed_index][seed_step] in existing_sequences and save_unique: continue
                if sequences_to_save[seed_index][seed_step] in unique_seqs and save_unique: continue
                unique_seqs.append(sequences_to_save[seed_index][seed_step])
                f.write(f'>{sequence_prefix}_{seed_index}_seqntG_{seed_step}\n')
                f.write(f'{sequences_to_save[seed_index][seed_step]}\n')
