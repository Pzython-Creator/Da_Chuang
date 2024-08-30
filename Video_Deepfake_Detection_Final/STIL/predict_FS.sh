 #One of the ['Deepfakes','Face2face','FaceSwap','NerualTextures']
METHOD=FaceSwap
#One of the ['c23','c40']
COMPRESSION=c23
#Model_Path
MODEL_PATH=data_FaceSwap_c23/ckpt/Epoch-8-Step-2273-ACC-0.9964-RealACC-1.0000-FakeACC-0.9928-Loss-0.02352-LR-0.0001.tar
torchrun predict.py -c configs/ffpp.yaml --method ${METHOD} --compression ${COMPRESSION} --model.resume ${MODEL_PATH}