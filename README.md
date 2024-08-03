# Multi-granularity Graph Prompt Distillation for Robust Test-Time Adaptation

Official implementation of the paper: "Multi-granularity Graph Prompt Distillation for Robust Test-Time Adaptation"

## Dataset
We consider 5 datasets for out-of-distrubition generalization:

* [ImageNet](https://image-net.org/index.php) 
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)
* [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

Also, we consider 9 datasets for cross-domain generalization:
* [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
* [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
* [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
* [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
* [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

For multi-task setting, we consider 2 large datasets:
* [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/)
* MiniDomainNet ([DomainNet](http://ai.bu.edu/M3SDA/), [split file](https://drive.google.com/file/d/15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7/view))

For cross-dataset generalization, we adopt the same train/val/test splits as CoOp. Please refer to [this page](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets), and look for download links of `split_zhou_${dataset_name}.json`, and put the json files under `./data/data_splits/`. 


## Contact
If you have any questions, feel free to create an issue in this repository or contact us via email at zq_126@mail.ustc.edu.cn or zmsxxd@mail.ustc.edu.cn.

## Acknowledgements
Our gratitude goes to the authors of [TPT](https://github.com/azshue/TPT) and [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) for sharing their work through open-source implementation and for providing detailed instructions on data preparation.
