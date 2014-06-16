from rootpy.tree import Tree, TreeModel, FloatCol, IntCol
from rootpy.io import root_open

import pandas as pd
import numpy as np

# define the Htautau model
class Event(TreeModel):
    # derived variables
    mass_MMC = FloatCol()
    mass_transverse_met_lep = FloatCol()
    mass_vis = FloatCol()
    pt_h = FloatCol()
    deltaeta_jet_jet = FloatCol()
    mass_jet_jet = FloatCol()
    prodeta_jet_jet = FloatCol()
    deltar_tau_lep = FloatCol()
    pt_tot = FloatCol()
    sum_pt = FloatCol()
    pt_ratio_lep_tau = FloatCol()
    met_phi_centrality = FloatCol()
    lep_eta_centrality = FloatCol()

    # tau variables
    tau_pt = FloatCol()
    tau_eta = FloatCol()
    tau_phi = FloatCol()
    
    # lepton variables
    lep_pt = FloatCol()
    lep_eta = FloatCol()
    lep_phi = FloatCol()

    # met variables
    met = FloatCol()
    met_phi = FloatCol()
    met_sumet = FloatCol()

    # other
    jet_num = IntCol()

    jet_leading_pt = FloatCol()
    jet_leading_eta = FloatCol()
    jet_leading_phi = FloatCol()

    jet_subleading_pt = FloatCol()
    jet_subleading_eta = FloatCol()
    jet_subleading_phi = FloatCol()

    jet_all_pt = FloatCol()

    Htautau = IntCol()

def csv_to_d3pd():
    df = pd.read_csv('./data/training.csv', header=0)
    
    # drop the Eventid and Weight columns
    df = df.drop(['EventId', 'Weight'], axis=1)

    # convert the labels columin into bits (1=singal, 0=background)
    conv_dic = {'s' : 1, 'b' : 0}
    df['Label'] = df['Label'].map( lambda x: conv_dic[x])

    # create and fill the training tree
    f = root_open('./data/training.root', 'recreate')
    training_tree = Tree('training', model=Event)
    
    # fill the tree
    for index, event in df.iterrows():
        # derived variables
        training_tree.mass_MMC = event['DER_mass_MMC']
        training_tree.mass_transverse_met_lep = event['DER_mass_transverse_met_lep']
        training_tree.mass_vis = event['DER_mass_vis']
        training_tree.pt_h = event['DER_pt_h']
        training_tree.deltaeta_jet_jet = event['DER_deltaeta_jet_jet']
        training_tree.mass_jet_jet = event['DER_mass_jet_jet']
        training_tree.prodeta_jet_jet = event['DER_prodeta_jet_jet']
        training_tree.deltar_tau_lep = event['DER_deltar_tau_lep']
        training_tree.pt_tot = event['DER_pt_tot']
        training_tree.sum_pt = event['DER_sum_pt']
        training_tree.pt_ratio_lep_tau = event['DER_pt_ratio_lep_tau']
        training_tree.met_phi_centrality = event['DER_met_phi_centrality']
        training_tree.lep_eta_centrality = event['DER_lep_eta_centrality']

        # tau variables
        training_tree.tau_pt = event['PRI_tau_pt']
        training_tree.tau_eta = event['PRI_tau_eta']
        training_tree.tau_phi = event['PRI_tau_phi']

        # lepton variables
        training_tree.lep_pt = event['PRI_lep_pt']
        training_tree.lep_eta = event['PRI_lep_eta']
        training_tree.lep_phi= event['PRI_lep_phi']

        # met variables
        training_tree.met = event['PRI_met']
        training_tree.met_phi = event['PRI_met_phi']
        training_tree.met_sumet = event['PRI_met_sumet'] 

        # other
        training_tree.jet_num = event['PRI_jet_num']

        training_tree.jet_leading_pt = event['PRI_jet_leading_pt']
        training_tree.jet_leading_eta = event['PRI_jet_leading_eta']
        training_tree.jet_leading_phi = event['PRI_jet_leading_phi']

        training_tree.jet_subleading_pt = event['PRI_jet_subleading_pt']
        training_tree.jet_subleading_eta = event['PRI_jet_subleading_eta']
        training_tree.jet_subleading_phi= event['PRI_jet_subleading_phi']

        training_tree.jet_all_pt = event['PRI_jet_all_pt']

        training_tree.Htautau = event['Label']

        training_tree.fill()

    training_tree.write()

csv_to_d3pd()
