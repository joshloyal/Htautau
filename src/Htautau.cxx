#include "AGILEPack"
#include <iostream>

int main(int arc, char const *argv[])
{

// Load data
//=============================================================================
    agile::root::tree_reader reader;    // declare a tree_reader instance
    reader.add_file("./data/training.root", "training"); // Load the file and TTree

    // Set all the branches
    reader.set_branch("mass_MMC", agile::root::single_precision);
    reader.set_branch("mass_transverse_met_lep", agile::root::single_precision);
    reader.set_branch("mass_vis", agile::root::single_precision);
    reader.set_branch("pt_h", agile::root::single_precision);
    reader.set_branch("deltaeta_jet_jet", agile::root::single_precision);
    reader.set_branch("mass_jet_jet", agile::root::single_precision);
    reader.set_branch("prodeta_jet_jet", agile::root::single_precision);
    reader.set_branch("deltar_tau_lep", agile::root::single_precision);
    reader.set_branch("pt_tot", agile::root::single_precision);
    reader.set_branch("sum_pt", agile::root::single_precision);
    reader.set_branch("pt_ratio_lep_tau", agile::root::single_precision);
    reader.set_branch("met_phi_centrality", agile::root::single_precision);
    reader.set_branch("lep_eta_centrality", agile::root::single_precision);
    reader.set_branch("tau_pt", agile::root::single_precision);
    reader.set_branch("tau_eta", agile::root::single_precision);
    reader.set_branch("tau_phi", agile::root::single_precision);
    reader.set_branch("lep_pt", agile::root::single_precision);
    reader.set_branch("lep_eta", agile::root::single_precision);
    reader.set_branch("lep_phi", agile::root::single_precision);
    reader.set_branch("met", agile::root::single_precision);
    reader.set_branch("met_phi", agile::root::single_precision);
    reader.set_branch("met_sumet", agile::root::single_precision);
    reader.set_branch("jet_num", agile::root::integer);
    reader.set_branch("jet_leading_pt", agile::root::single_precision);
    reader.set_branch("jet_leading_eta", agile::root::single_precision);
    reader.set_branch("jet_leading_phi", agile::root::single_precision);
    reader.set_branch("jet_subleading_pt", agile::root::single_precision);
    reader.set_branch("jet_subleading_eta", agile::root::single_precision);
    reader.set_branch("jet_subleading_phi", agile::root::single_precision);
    reader.set_branch("jet_all_pt", agile::root::single_precision);
    reader.set_branch("Htautau", agile::root::integer);

    agile::dataframe D = reader.get_dataframe();

// Estimating the neural network
//=============================================================================
    agile::neural_net my_net;
    my_net.add_data(std::move(D));

    // model using 4vector information
    my_net.model_formula("Htautau ~ tau_pt + tau_eta + tau_phi + lep_pt + lep_eta + lep_phi + met + met_phi + jet_leading_pt + jet_leading_eta + jet_leading_phi + jet_subleading_pt + jet_subleading_phi + jet_subleading_eta");

    // "stacks" the layers (stacked autoencoders)
    // NOTE: mention that the first input is the number of previous layers
    my_net.emplace_back(new autoencoder(14, 17, sigmoid, linear));
    my_net.emplace_back(new autoencoder(17, 12, sigmoid, sigmoid));
    my_net.emplace_back(new autoencoder(12, 5, sigmoid, sigmoid));
    my_net.emplace_back(new autoencoder(5, 3, sigmoid, sigmoid));
    my_net.emplace_back(new layer(3, 1, sigmoid));

    // parameter setting
    my_net.set_learning(0.05);
    my_net.set_regularizer(0.001);
    my_net.set_batch_size(1);
    my_net.check();

    // The unsupervised training trains each layer seperatly
    my_net.train_unsupervised(10);

    my_net.set_learning(0.01);

    my_net.train_supervised(10);
    my_net.to_yaml("Htautau.yaml", reader.get_var_types());

// Prediction time
//=============================================================================
    
    auto input_variables = my_net.get_inputs();
    for (int higgs = 0; higgs < 10; ++higgs)
    {
        // This pulls a map out
        auto predictions = my_net.predict_map(reader(higgs, input_variables));
        std::cout << "Prob[higg #" << higgs << " == Htautau] = ";
        std::cout << predictions["Htautau"] << std::endl;
    }

    return 0;
}
