
############### Dataset processing commands ##################

createXrayDataset:
	 python dataset_tool.py create_from_images datasets/xray_frontal_train /research/boundaryCrossing/chestxray_raz/square_frontal_train

#last argument contains a folder with all pngs in the same power-of-2 resolution.
createBrainDataset:
	 python dataset_tool.py create_from_images datasets/brains_test_mono  /razspace/brain_datasets/pngs_test
	

######### Model training commands #################

trainBrains:
	python run_training.py --num-gpus=4 --data-dir=datasets --config=config-e --dataset=brains_train_mono --mirror-augment=true


######### Image generation (post-training) #################


genFFHQ:
	python run_generator.py generate-images --seeds=0-200 --truncation-psi=0.5 --network=results/00344-stylegan2-ffhq_train-4gpu-config-e/network-snapshot-007127.pkl

genBrains:
	python run_generator.py generate-images --seeds=0-200 --truncation-psi=0.5 --network=results/00296-stylegan2-brains_train_mono-4gpu-config-e/network-snapshot-001451.pkl

genXray:
	python run_generator.py generate-images --seeds=0-200 --truncation-psi=0.5 --network=results/00341-stylegan2-xray_frontal_train-4gpu-config-e/network-snapshot-007946.pkl

project: 
	python run_projector.py project-real-images --dataset=xray_ood --data-dir=datasets --network=results/00037-stylegan2-xray_1k_v2-8gpu-config-e/network-snapshot-005488.pkl


######### Image reconstruction #################

# recontype =  "super-resolution", "inpaint"


super-resolutionFFHQ:
	python recon.py recon-real-images --input=datasets/ffhq --tag=ffhq --network=dropbox:ffhq.pkl --recontype=super-resolution


# this toy dataset contains chst x-ray images from wikipedia
inpaintXray:
	python recon.py recon-real-images --input=datasets/xray --masks=masks/1024x1024 --tag=xray --network=dropbox:xray.pkl --recontype=inpaint

# brains contains test images from the OASIS dataset only
inpaintBrains:
	python recon.py recon-real-images --input=datasets/brains --masks=masks/256x256 --tag=brains --network=dropbox:brains.pkl --recontype=inpaint



