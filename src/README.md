Source codes for RobustBNN

Jong's Environment
- Ubuntu 18.04
- Python 3.6.6 on Anaconda
- Cuda 9.2 and Cudnn 7.3 installed for Geforce GTX 1060 6GB
- pymc3, tensorflow-gpu, cleverhans, theano installed

Testing:
THEANO_FLAGS='mode=FAST_RUN,device=cuda,floatX=float32,optimizer_including=cudnn' for gpu
THEANO_FLAGS='mode=FAST_RUN,floatX=float32' for cpu
- This makes Theano use the gpu instead of cpu
- Create new conda environment with python 3 and installing tensorflow(or tensorflow-gpu if you have gpu)
- You can export this to your bashrc if you are using linux. For example in your bashrc add
    export THEANO_FLAGS='mode=FAST_RUN,device=cuda,floatX=float32,optimizer_including=cudnn'

Using Tufts cluster:
- srun -p gpu -t 0-06:00:00 --mem 8000 --pty bash
- module load cuda

Using AWS
- Guide is here https://www.cs.tufts.edu/comp/150BDL/2018f/tufts_aws_setup.html
- our aws server = aws-gpu-3.eecs.tufts.edu

Current Data
- advi-bnn-MNIST.pkl
  - Test data Accuracy = 94.38%
- advi-bnn-CIFAR10.pkl
  - Test data Accuracy = 19.15%
  - Test data Accuracy(Grayscale) = 31.34% (Tried 300 neurons but that didn't help)
- nuts-bnn-MNIST.pkl
  - Test data Accuracy = 46.32%
- nuts-bnn-CIFAR10.pkl
  - Test data Accuracy = 41.44%

- advi-bcnn-MNIST.pkl
  - Test data Accuracy = 
- advi-bcnn-CIFAR10.pkl
  - Test data Accuracy = 
- nuts-bcnn-MNIST.pkl
  - Test data Accuracy = 
- nuts-bcnn-CIFAR10.pkl
  - Test data Accuracy = 