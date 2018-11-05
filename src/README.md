Source codes for RobustBNN

Testing:
THEANO_FLAGS='mode=FAST_RUN,device=cuda,floatX=float32,optimizer_including=cudnn'
- This makes the Keras with Theano backend use the gpu instead of cpu.

- Create new conda environment with python 3 and installing tensorflow(or tensorflow-gpu if you have gpu).