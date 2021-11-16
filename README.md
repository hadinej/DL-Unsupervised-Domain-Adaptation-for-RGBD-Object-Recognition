# Unsupervised Domain Adaptation through Inter-modal Rotation and Jigsaw Puzzle assembly for RGB-D Object Recognition



## Project's Abstract
In general, domain adaptation uses labeled data in one or more source domains to solve new tasks in a target domain existing DA methods are not designed to cope with the multi-modal nature of RGB-D data, which are widely used in robotic vision. We propose a novel RGB-D DA method that reduces the synthetic-to-real domain shift by exploiting the intermodal relation between the RGB and depth image. Our method consists of training a convolutional neural network to solve, in addition to the main recognition task, the pretext task of predicting the relative rotation between the RGB and depth image. Also proposing a variation to the
pretext task by changing the self-supervision method from rotation to Jigsaw puzzle and comparing their results; Which resulted in 2% improvement in accuracy scores

### Dataset
We make our own dataset by collecting a new synthetic dataset called synROD. the objects are carefully selected from the 51 categories of ROD dataset, most popular RGB-D object categorization repository. ”Next challenge is to create the 2.5D scenes to account for 3D nature of objects. This is done by using a ray-tracing engine in Blender to simulate photo-realistic lighting”. Multiple techniques were performed to obtain usual and realistic object posture. In the following figure we can observe the different data distribution among domains.
![example](/images/apple.png)
<p align = "center">
Sample images from the synROD dataset for the class 'apple' one for each domain.
</p>

## Implementation 
(Further details about discussion and results are available in the [project report](./report.pdf).)

▶ Our goal is to train a neural network to predict the object class of the target data, using only labeled source data and unlabelled target data. As it can be seen from figure, the
model is made of three principal sections which are the Feature Extractor (E), the Main Task(M) and the Pretext Task (P)
![Network architecure](/images/structure.png)
<p align = "center">
Network architecure
</p>

▶ The self-supervised task is applied to reduce the domain shift between the two data modalities. we define a sample of the source images, where xc s is the RGB image
and xd is the Depth image.

![self](/images/puz.png)
<p align = "center">
Self-Supervision Task
</p>

▶ Regarding the results, the proposed method was able to promote the baseline by 14% in accuracy by integrating the pretext task which shows improvement in the reduction of domain gap by aligning the distributions of features between the two domains. With the purposed variation, domain adaptation with Jigsaw Puzzle, the accuracy is further improved by nearly 2% which is a considerable.

![res](/images/res.png)
<p align = "center">
Results
</p>


  
  
---

### References

[1] J. Deng, W. Dong, R. Socher, L. Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database, 2009.

[2] A. Eitel, J. T. Springenberg, L. Spinello, M. Riedmiller, and W. Burgard. Multimodal deep learning for robust rgb-d object recognition. In 2015 IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS), pages 681–687. IEEE, 2015.

[3] S. Gidaris, P. Singh, and N. Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728, 2018.

[4] K. Lai, L. Bo, and D. Fox. Unsupervised feature learning for 3d scene labeling. In 2014 IEEE International Conference on Robotics and Automation (ICRA), pages 3050–3057. IEEE, 2014.

[5] M. R. Loghmani, L. Robbiano, M. Planamente, K. Park, B. Caputo, and M. Vincze. Unsupervised domain adaptation through inter-modal rotation for rgb-d object recognition, 2020.

[6] M. Noroozi and P. Favaro. Unsupervised learning of visual
representations by solving jigsaw puzzles. In European Conference on Computer Vision, pages 69–84. Springer, 2016.
