pip install numpy
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX
pip install cython
git clone https://github.com/jorge-pessoa/pytorch-msssim.git
cd pytorch-msssim
python setup.py install
cd ../
@RD /S /Q pytorch-msssim
git clone https://github.com/STVIR/pysot.git
xcopy /s pysot Source\PySOT
@RD /S /Q pysot
cd Source\PySOT
python setup.py build_ext --inplace
cd ../
cd DBLLNet\bilateral_slicing_op
python setup.py install


