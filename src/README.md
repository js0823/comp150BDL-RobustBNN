Source codes for RobustBNN

Jong's Environment
- Ubuntu 18.04
- Python 3.6.6 on Anaconda
- Cuda 9.2 and Cudnn 7.3 installed for Geforce GTX 1060 6GB
- pymc3, tensorflow-gpu, cleverhans, theano installed

Testing:
THEANO_FLAGS='mode=FAST_RUN,device=cuda,floatX=float64,optimizer_including=cudnn'
- This makes Theano use the gpu instead of cpu
- Create new conda environment with python 3 and installing tensorflow(or tensorflow-gpu if you have gpu)

NUTS sampler on pymc3
- Using gpu, max iteration is around 47s/it, making it 12 hours to run.