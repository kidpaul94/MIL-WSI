# MIL_WSI

**What are MIL and WSI?**

Multi-Instance Learning (MIL) is one type of semi-supervised learning technique. In a nutshell, MIL is a semi-supervised learning in a sense that labels are provided for each bag which include multiple instances. There is a great introduction about MIL in this [blog](https://nilg.ai/blog/202105/an-introduction-to-multiple-instance-learning/) if you need more details about this technique. 

![Example 0](./images/tissue.jpg)

As shown above, Whole Slide Images (WSI) refers to a single high-resolution digital file of a complete microscope slide. This is commonly achieved by capturing many small high-resolution image tiles or strips and then montaging them to create a full image of a histological section. In pathology, professionals deal with WSI of tissues to diagnosing cancer to managing chronic diseases through accurate laboratory testing. Just like every other fields, deel learning is getting implemented in pathology. However, especially in WSI classification, it is challenging to utilize such technique due to their ultra high resolution (~60K). Hence, researchers have tried to use MIL to subdivide WSI into multiple ROIs to reduce the overall dimension of the WSI while maintaining detail features and patterns of tissues. 

This repository contains a simple MIL model, training pipeline, and etc. Note that PyTorch does not have a built-in classes to handle bag-wise dataloader. Therefore, necessary features and functions (e.g., epoch and dataset shuffle) is custom built to train and test the MIL model.

**ToDo Lists**
- [ ] Add train.py, eval.py, and ohter necessary files for data preprocessing.
