# Towards Fingerprint Presentation Attack Generation using Generative Adversarial Networks

## About
Implementation of the paper [_"Towards Fingerprint Presentation Attack Generation using Generative Adversarial Networks"_](paper.pdf) by Leonardo Capozzi, Tiago Gonçalves, Jaime S. Cardoso and Ana Rebelo.



## Abstract 
Most of the available systems rely on fingerprint recognition and have shown to be reliable in terms of accuracy, speed and purported security. However, they also present several vulnerabilities against spoof attacks. To overcome this flaw, several automated spoofing detection models have been developed, but they end up assuming that spoof detection is a binary closed-set problem, which is not realistic. Recent works have proposed the use of adversarial methodologies to improve the model’s generalisation capacity to unseen spoof attacks. Following this research line, we performed a study on the application of generative adversarial neural networks (GANs) for the generation of synthetic data to be employed during the model’s training. We hypothesise that by using GANs, one could learn a distribution that could contain all possible spoofing attacks, thus opening the possibility to learn classifiers that could be more robust. In this work, we optimised a GAN and conditional GAN (cGAN) to generate synthetic images of real and fake fingerprints and used this data in the training of classifiers for the detection of single spoofing attacks.



## Clone this repository
To clone this repository, open a Terminal window and type:
```bash
$ git clone git@github.com:leonardogomesc/Image-to-image-Generative-Adversarial-Network.git
```
Then go to the repository's main directory:
```bash
$ cd Image-to-image-Generative-Adversarial-Network
```



## Dependencies
### Install the necessary Python packages
We advise you to create a virtual Python environment first (Python 3.7). To install the necessary Python packages run:
```bash
$ pip install -r requirements.txt
```



## Data
To know more about the data used in this paper, please send an e-mail to  [**tiago.f.goncalves@inesctec.pt**](mailto:tiago.f.goncalves@inesctec.pt) or to [**leonardo.g.capozzi@inesctec.pt**](mailto:leonardo.g.capozzi@inesctec.pt).



## Usage
### Data Augmentation
To reproduce the experiments:
```bash
$ python code/gan/augment_data.py
$ python code/conditional-gan/augment_data.py
```

### Train Models
To reproduce the experiments:
```bash
$ python code/gan/gan_train.py
$ python code/conditional-gan/gan_train.py
$ python code/classifier/classifier_train.py
$ python code/classifier/classifier_train_daugm.py
```

### Test Models
To reproduce the experiments:
```bash
$ python code/gan/gan_gen_test.py
$ python code/conditional-gan/gan_gen_test.py
$ python code/classifier/classifier_test.py
```

### Generate the Results
To plot the results:
```bash
$ python code/gan/gan_create_imgs_eg.py
$ python code/conditional-gan/gan_create_imgs_eg.py
```



## Citation
If you use this repository in your research work, please cite this paper:
```bibtex
@inproceedings{capozzigoncalves2024recpad,
	author = {Leonardo Capozzi, Tiago Gonçalves, Jaime S. Cardoso and Ana Rebelo},
	title = {{Towards Fingerprint Presentation Attack Generation using Generative Adversarial Networks}},
	booktitle = {{30th Portuguese Conference in Pattern Recognition (RECPAD)}},
	year = {2024},
	address = {{Covilhã, Portugal}}
}
```