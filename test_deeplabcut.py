import deeplabcut 
config_path= path_config_file = 'C:/Users/sophie/Desktop/laval/PHD/CDPQ/deeplabcut/CDPQ_test-CDPQ_experiment-2022-02-22/config.yaml'
#deeplabcut.check_labels(path_config_file)
"""deeplabcut.create_multianimaltraining_dataset(
    path_config_file,
    num_shuffles=1,
    net_type="dlcrnet_ms5",
)"""
#deeplabcut.train_network(path_config_file, shuffle=1, max_snapshots_to_keep=5, displayiters=100, saveiters=200, maxiters=500)
deeplabcut.train_network(
    config_path,
    saveiters=1,
    maxiters=5,
    allow_growth=True,
)