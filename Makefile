

## Run ###
dataset_generator_comet_win:
	python datasets_creator\src\generate_dataset.py --rel_input_path "datasets_creator/data/original/train.parquet" --train_size ${train_size} --num_val_sets ${num_val_sets} --batch_size ${batch_size}

dataset_generator_local_win:
	python datasets_creator\src\generate_dataset.py --rel_input_path "datasets_creator/data/original/train.parquet" --rel_output_path ${output_path} --train_size ${train_size} --num_val_sets ${num_val_sets}

dataset_generator_comet_linux:
	python datasets_creator/src/generate_dataset.py --rel_input_path ${rel_input_path} --train_size ${train_size} --num_val_sets ${num_val_sets} --batch_size ${batch_size} --comet_ml_dataset_name ${comet_dataset_name} --comet_ml_dataset_description ${comet_dataset_description}

dataset_generator_local_linux:
	python datasets_creator/src/generate_dataset.py --rel_input_path ${rel_input_path} --rel_output_path ${output_path} --train_size ${train_size} --num_val_sets ${num_val_sets}

run_trainer:
	python multilingual_chatbot_arena/train/main.py 