#One of the ['Deepfakes','Face2face','FaceSwap','NerualTextures']
METHOD=Face2Face
#One of the ['c23','c40']
COMPRESSION=c23
#Model_Path
MODEL_PATH=data_Face2Face_c23/ckpt/Epoch-24-Step-6817-ACC-0.9928-RealACC-1.0000-FakeACC-0.9855-Loss-0.03513-LR-5e-05.tar
torchrun predict.py -c configs/ffpp.yaml --method ${METHOD} --compression ${COMPRESSION} --model.resume ${MODEL_PATH}
#python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 test.py -c configs/ffpp.yaml  --method ${METHOD} --compression ${COMPRESSION} --model.resume ${MODEL_PATH}