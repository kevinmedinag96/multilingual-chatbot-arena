

## Run ###
dataset_generator_comet:
	python datasets_creator\src\generate_dataset.py --rel_input_path "datasets_creator/data/original/train.parquet" --train_size ${train_size} --num_val_sets ${num_val_sets} --batch_size ${batch_size}

dataset_generator_local:
	python datasets_creator\src\generate_dataset.py --rel_input_path "datasets_creator/data/original/train.parquet" --rel_output_path ${output_path} --train_size ${train_size} --num_val_sets ${num_val_sets}

