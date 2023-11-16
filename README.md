# AutoCaCoNet
## Abstract
* Colorization is one of the attractive topics in computer vision areas. Traditional approaches generally depend on object-based approaches which require establishing a large-scale image dataset. Recent deep neural network approaches shed light on the potentiality and practicality of automatic image colorization tasks.
* In this work, we propose a novel fully automatic cartoon colorization network called AutoCaCoNet using  conditional generative adversarial network with self-attention, segmentation, and color correction techniques.
* Then, the experimental results, which are organized by both qualitative and quantitative evaluations, show the significance of AutoCaCoNet using the real-world cartoon dataset with greater evaluation metrics than prior research.
* Moreover, a user survey with both user and expert groups examines the results of AutoCaCoNet as the highest scores.
## Methods
### cGAN
AutoCaCoNet employs a conditional generative adversial network (cGAN) as its main framework.
### Self-Attention
We applied the self-attention module to the generator of cGAN in order not to lose the detailed information and catch the long-range dependency in the cartoon colorization work.
### Segmentation and Color Correction
We performed segmentation and color correction as postprocessing to improve the quality of the output image of cGAN with self-attention.
## Overall architecture of AutoCaCoNet
![model_architecture_3](https://user-images.githubusercontent.com/42277033/200303721-45c63572-1270-4f83-895e-a561666362fe.jpg)
## Experiments
### Datasets
We employed cartoon images of South Korean children’s history books, which were commercially sold, with permission from the copyright holder, Sahoipyoungnon, one of the well-known publishers in South Korea. The book title is “Yong teacher’s cartoon Korean history". We put 10 original images in the data folder as samples.
### Compared Models
* Iizuka et al. [1] (CNN-based)
* Zhang et al. [2] (CNN-based)
* CycleGAN [3] (GAN-based)
* Pix2Pix [4] (GAN-based)
### Quantitative Evaluation
To examine the performance, we employ three evaluation metrics: FID, SSIM, PSNR. Our network 2 shows the greatest FID level (44.853). In case of SSIM and PSNR, our network 1 outperforms with the highest scores, 0.683 (SSIM) and 14.343 (PSNR).

<img width="700" alt="1" src="https://user-images.githubusercontent.com/42277033/201195538-c97685a2-cbce-4235-ac55-ffd3bfb9cba1.png">

### Qualitative Evaluation
We present the results of cartoon colorization tasks in figure below. In qualitative evaluation, artifacts were significantly reduced in the coloring result of our network 2 compared to other models.

![result_samples_modify_3](https://user-images.githubusercontent.com/42277033/201199881-833745e8-b397-41db-9611-6ade3392f46a.jpg)

### User Study and Additional Interview Analysis
* As a result of satisfaction evaluation, our network 2 shows the most satisfactory responses (7.52 (1.91) in the general user group, and 6.66 (1.56) in the expert group).
* As a result of practicality evaluation, the average was 0.45 (SD=0.96). It means
that the results of our network 2 is positively evaluated by the practical perspectives of the coloring experts.

<img width="400" alt="2" src="https://user-images.githubusercontent.com/42277033/201205185-108eb45e-7336-4129-b062-43dde4548e2c.png">

## References
* [1] Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Let there be color! joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification. _ACM Transactions on Graphics (ToG)_, 35(4):1–11, 2016.
* [2] Richard Zhang, Jun-Yan Zhu, Phillip Isola, Xinyang Geng, Angela S Lin, Tianhe Yu, and Alexei A Efros. Real-time user-guided image colorization with learned deep priors. _arXiv preprint arXiv:1705.02999_, 2017.
* [3] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In _Proceedings of the IEEE international conference on computer vision_, pages 2223–2232, 2017.
* [4] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 1125–1134, 2017.
