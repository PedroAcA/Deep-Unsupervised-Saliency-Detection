# Model testing
To replicate our results, please follow the steps below 
## Software setup
	1. Install Python 3.7.13 (a python virtual enviroment can be used)
	2. Install Pytorch and Torchvision
		- For the GPU version, install the CUDA Toolkit 11.3 available at https://developer.nvidia.com/cuda-11.3.0-download-archive (last access on 20th, June 2023) and run python -m pip install -r requirements.txt
		- For the CPU version, edit the requirements.txt changing '+cu113' to '+cpu' and run python -m pip install -r requirements.txt	 

## Data download
	1. Install Python 3.7.13, Pytorch 1.12.1 + cu113 (or its cpu counterpart) and Torchvision 0.13.1 + cu113 (or its cpu counterpart)
	2. Download JSRT data from https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt (last access on 19th of June, 2023) and uncompress it into datasets/JSRT/ folder
	3. Download SCR masks for JSRT from https://drive.google.com/drive/folders/1FqT2XsBQ_cPwGkq8g1QnSg2PUmh9E9fq?usp=sharing (last access in 19th of June, 2023) and uncompress it into datasets/JSRT/ folder
	4. Download weights from https://drive.google.com/drive/folders/1TdAAP6pdZGI7GFfRBdz3_GbaZWIiFP9U?usp=sharing  (last access on 19th of June, 2023) and uncompress it into experiments/ folder. The experiments described are associated to the following folders:
		- S1: jsrt_lung_segmentation_input_pre_processed_pseudo_labels_post_processed/
		- S2: jsrt_lung_segmentation_input_pre_processed_pseudo_labels_not_post_processed/
		- S3: jsrt_lung_segmentation_input_not_pre_processed_pseudo_labels_not_post_processed/ 
		- S4: jsrt_lung_segmentation_input_not_pre_processed_pseudo_labels_post_processed/
	5. For results on Montgomery County (MC) dataset, download both the X-rays and inputs from https://openi.nlm.nih.gov/faq#faq-tb-coll (last access on 20th of June, 2023) and uncompress them into datasets/MC/ folder
	
## Model running
	- If running S1 or S2, pre-process the X-Ray input using python preprocess_input.py --data_root ../datasets/JSRT/archive/images/images/ --out_dir ../datasets/JSRT/pre_processed_inputs/ to pre-process images stored on datasets/JSRT/archive/images/images/ and save the results into datasets/JSRT/pre_processed_inputs/
	- To run the model, use the following command python deep_unsup_sd.py --config_file ../experiments/jsrt_lung_segmentation_input_not_pre_processed_pseudo_labels_not_post_processed/config.txt --mode test --chkpt_file ../experiments/jsrt_lung_segmentation_input_not_pre_processed_pseudo_labels_not_post_processed/chkpt/checkpoint_epoch_jsrt_lung_segmentation_input_not_pre_processed_pseudo_labels_not_post_processed_19.pth --test_batch_size 1 --test_type quantitative. This examples run the S3 experiment for quantitative results. To change the experiments, change the arguments --config_file and --chkpt_file to point to the config and .pth files respectively. To perform qualitative results, change --test_type to qualitative or all (note: qualitative tests need --test_batch_size to be 1 in order to work properly).

# Model training
	1.  Download GS and RBD Matlab implementations from  (last access on 20th, June 2023)
