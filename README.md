# Single image reflection removal utilizing multi-scale spatial attention and feature enhancement network

This code is based on tensorflow. The test results on the real dataset are as follows:

## ![real images](https://github.com/knight936/Single-image-reflection-removal/blob/main/test_images/real/test%20images.png)


## Setup

* Clone/Download this repo
* `$ cd perceptual-reflection-removal`
* `$ mkdir VGG_Model`
* Download [VGG-19](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Search `imagenet-vgg-verydeep-19` in this page and download `imagenet-vgg-verydeep-19.mat`. We need the pre-trained VGG-19 model for our hypercolumn input and feature loss
* move the downloaded vgg model to folder `VGG_Model`

### Core Environment Requirements

* tensorflow<=2.10

* tf-slim==1.1.0 

#### Training dataset

* 7,643 pairs of images from[Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) 

* 90 real-world training images from [Berkeley real dataset](https://github.com/ceciliavision/perceptual-reflection-removal) 

Try with your own dataset

You can also try with your own dataset. For example, to generate your own synthetic dataset, prepare for two folders of images, one used for transmission layer and the other used for reflection layer. In order to use our data loader, please follow the instructions below to organize your dataset directories. Overall, we name the directory of input image with reflection `blended`, the ground truth transmission layer `transmission_layer` and the ground truth reflection layer `reflection_layer`.

For synthetic data, since we generate images on the fly, there is no need to have the `blended` subdirectory.
>+-- `root_training_synthetic_data`<br>
>>+-- `reflection_layer`<br>
>>+-- `transmission_layer`<br>

For real data, since the ground truth reflection layer is not available, there is no need to have the `reflection_layer` subdirectory.
>+-- `root_training_real_data`<br>
>>+-- `blended`<br>
>>+-- `transmission_layer`<br>


### Training and continuous training

`python main.py `（Adjust the training strategy based on the parameter settings section）

###  Testing

* Download pre-trained model [here](https://drive.google.com/file/d/1660b2B_0a5lS7fxRfusMXqqobO2nNJfx/view?usp=drivesdk)
* this should extract the models into a newly created folder called `pre-trained`
* Change `test_path`  to your test image folder. If you want to test on the provided test images (`./test_images/real/`), keep it as it is.
* test results can be found in `./test_results/`

Then, run

`$ python main.py --task pre-trained --is_training 0`


###  Contact

Please contact me if there is any question (ct2268662256@163.com)
