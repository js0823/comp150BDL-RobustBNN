Source codes for RobustBNN

THEANO_FLAGS='mode=FAST_RUN,device=cuda,floatX=float32,optimizer_including=cudnn'
- This makes the Keras with Theano backend use the gpu instead of cpu.