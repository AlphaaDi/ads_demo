config = {
    "maximum_number_of_frames": 150,
    "resx": 768,
    "resy": 432,
    "iters_num": 400000,
    "samples_batch": 20000,
    "optical_flow_coeff": 5.0,
    "evaluate_every": 50000,
    "derivative_amount": 1,
    "rgb_coeff": 5000,
    "rigidity_coeff": 1.0,
    "uv_mapping_scale": 0.8,
    "pretrain_mapping1": True,
    "pretrain_mapping2": True,
    "alpha_bootstrapping_factor": 2000.0,
    "alpha_flow_factor": 49.0,
    "positional_encoding_num_alpha": 5,
    "number_of_channels_atlas": 256,
    "number_of_layers_atlas": 8,
    "number_of_channels_alpha": 256,
    "number_of_layers_alpha": 8,
    "stop_bootstrapping_iteration": 10000,
    "number_of_channels_mapping1": 256,
    "number_of_layers_mapping1": 6,
    "number_of_channels_mapping2": 256,
    "number_of_layers_mapping2": 4,
    "gradient_loss_coeff": 1000,
    "use_gradient_loss": True,
    "sparsity_coeff": 1000.0,
    "positional_encoding_num_atlas": 10,
    "use_positional_encoding_mapping1": False,
    "number_of_positional_encoding_mapping1": 4,
    "use_positional_encoding_mapping2": False,
    "number_of_positional_encoding_mapping2": 2,
    "pretrain_iter_number": 100,
    "load_checkpoint": False,
    "include_global_rigidity_loss": True,
    "global_rigidity_derivative_amount_fg": 100,
    "global_rigidity_derivative_amount_bg": 100,
    "global_rigidity_coeff_fg": 5.0,
    "global_rigidity_coeff_bg": 50.0,
    "stop_global_rigidity": 5000,
    "add_to_experiment_folder_name": "_",
    "checkpoint_path": "pretrained_models/checkpoints/blackswan/checkpoint",
    "data_folder": "data/blackswan",
    "results_folder_name": "results",
}