import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

import pyrosetta
pyrosetta.init('-mute all')

def get_pos_sasa(pose, position):
    sas = pyrosetta.rosetta.core.scoring.sasa.SasaCalc()
    sas.calculate(pose)
    residue_sasa_list = sas.get_residue_sasa()
    return residue_sasa_list[position] # sasa method returns 1 index vector

def three_from_one(one):
    """ Returns a 3 AA letter code from 1 letter"""
    three = pyrosetta.rosetta.core.chemical.name_from_aa(pyrosetta.rosetta.core.chemical.aa_from_oneletter_code(one))
    return three

def calc_feat(pdb_code_l):
    plus1_list  = []
    sasa_list = []
    secstruct_list = []
    phi_list = []
    psi_list = []
    pdb_list = []
    position_list = []
    disulfide_list = []
    trunc_seq_list = []
    label_list = []
    last_pdb = ''
    pdb_list = []
    avg_plDDT_list = []
    local_plDDT_list = []

    pose = pyrosetta.pose_from_file('./models/AF-' + pdb_code_l + '-F1-model_v3.pdb')
    # checking quality of AF models
    plDDT_list = []
    for position in range(1, len(pose.sequence())):
        plDDT_list.append(pose.pdb_info().bfactor(position,1))


    for index in df_filt_glyc[df_filt_glyc.ID == pdb_code_l].index:
        index_l = int(df_filt_glyc.loc[index]['position'])
        trunc_seq = df_filt_glyc.loc[index]['sequence']
        label = df_filt_glyc.loc[index]['label']

        avg_plDDT_list.append(np.mean(plDDT_list))
        local_plDDT_list.append(np.mean(plDDT_list[index_l-11:index_l+10]))
        asn_pos = pose.sequence().find(trunc_seq[10:15])+1

        sub_list_phi = []
        sub_list_psi = []
        for i in range(-5, 6):
            sub_list_phi.append(pose.phi(asn_pos+i))
            sub_list_psi.append(pose.psi(asn_pos+i))

        phi_list.append(sub_list_phi)
        psi_list.append(sub_list_psi)

        asn_sasa = get_pos_sasa(pose, asn_pos)
        #Secstruct
        DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()
        DSSP.apply(pose)
        asn_sec = pose.secstruct(asn_pos-1)

        # SASA
        sasa_list.append(asn_sasa)
        secstruct_list.append(asn_sec)

        # append pdb, pos, chain
        position_list.append(asn_pos)

        # append trunc_seq
        trunc_seq_list.append(trunc_seq)

        label_list.append(label)

        pdb_list.append(pdb_code_l)

    df = pd.DataFrame()
    df['label'] = label_list
    df['entry'] = pdb_list
    df['pos'] = position_list
    df['trunc_seq'] = trunc_seq_list
    df['sasa'] = sasa_list
    df['phi'] = phi_list
    df['psi'] = psi_list
    df['secstruct'] = secstruct_list
    df['local_plDDT'] = local_plDDT_list
    df['avg_plDDT'] = avg_plDDT_list

    return df
