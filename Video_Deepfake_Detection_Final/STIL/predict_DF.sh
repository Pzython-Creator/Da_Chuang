#One of the ['Deepfakes','Face2face','FaceSwap','NerualTextures']
METHOD=Deepfakes
#One of the ['c23','c40']
COMPRESSION=c23
#Model_Path
MODEL_PATH=data_Deepfakes_c23/ckpt/Epoch-13-Step-3693-ACC-0.9630-RealACC-0.9851-FakeACC-0.9412-Loss-0.06776-LR-0.0001.tar
torchrun predict.py -c configs/ffpp.yaml --method ${METHOD} --compression ${COMPRESSION} --model.resume ${MODEL_PATH}