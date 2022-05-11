# Awesome Face Generation [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

## Contents
- [Awesome Face Generation ![Awesome](https://awesome.re)](#awesome-face-generation-)
  - [Contents](#contents)
    - [Basic Nets - Image2Image](#basic-nets---image2image)
    - [Basic Nets - Vector2Image](#basic-nets---vector2image)
    - [Face Swap](#face-swap)
    - [Face Reenactment](#face-reenactment)
    - [Face Stylizing](#face-stylizing)
    - [Audio-driven talking face](#audio-driven-talking-face)
    - [Tools](#tools)
    - [Datasets](#datasets)
    - [Great Repos](#great-repos)

---
### Basic Nets - Image2Image
- [arXiv'21.06] [[**AttnFlow**](https://arxiv.org/pdf/2106.03959.pdf)] Generative Flows with Invertible Attentions
- [arXiv'21.04] [[**StyleGAN-Inversion**](https://arxiv.org/pdf/2104.07661.pdf)] A Simple Baseline for StyleGAN Inversion [[web](https://wty-ustc.github.io/inversion/)] [[pytorch](https://github.com/bes-dev/MobileStyleGAN.pytorch)]
- [arXiv'21.02] [[**TransUNet**](https://arxiv.org/pdf/2102.04306.pdf)] TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation [[pytorch](https://github.com/bes-dev/MobileStyleGAN.pytorch)]
- [CVPR'21] [[**GH-Feat**](https://openaccess.thecvf.com/content/CVPR2021/papers/Xu_Generative_Hierarchical_Features_From_Synthesizing_Images_CVPR_2021_paper.pdf)] Generative Hierarchical Features from Synthesizing Images
[[pytorch](https://github.com/genforce/ghfeat)]
- [CVPR'21] [[**pSp**](https://openaccess.thecvf.com/content/CVPR2021/papers/Richardson_Encoding_in_Style_A_StyleGAN_Encoder_for_Image-to-Image_Translation_CVPR_2021_paper.pdf)] Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation [[pytorch](https://github.com/eladrich/pixel2style2pixel)]
- [CVPR'21] [[**DivCo**](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_DivCo_Diverse_Conditional_Image_Synthesis_via_Contrastive_Generative_Adversarial_Network_CVPR_2021_paper.pdf)] DivCo: Diverse Conditional Image Synthesis via Contrastive Generative Adversarial Network
[[pytorch](https://github.com/ruiliu-ai/DivCo)]
- [CVPR'21] [[**GPEN**](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_GAN_Prior_Embedded_Network_for_Blind_Face_Restoration_in_the_CVPR_2021_paper.pdf)] GAN Prior Embedded Network for Blind Face Restoration in the Wild [[pytorch](https://github.com/yangxy/GPEN)]
- [ICLR'21] [[**Score-SDE**](https://openreview.net/pdf?id=PxTIG12RRHS)] Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation [[pytorch](https://github.com/yang-song/score_sde_pytorch)] [[tensorflow](https://github.com/yang-song/score_sde)]
- [ICLR'21] [[**DINO**](https://openreview.net/pdf?id=WAISmwsqDsb)] DINO: A Conditional Energy-Based GAN for Domain Translation [[pytorch](https://github.com/DinoMan/DINO)]
- [ICML'20] [[**AGD**](https://arxiv.org/pdf/2006.08198.pdf)] AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks [[pytorch](https://github.com/VITA-Group/AGD)]
- [NIPS'20] [[**ContraGAN**](https://proceedings.neurips.cc//paper/2020/file/f490c742cd8318b8ee6dca10af2a163f-Paper.pdf)] ContraGAN: Contrastive Learning for Conditional Image Generation  [[tensorflow](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN)]
- [NIPS'20] [[**CircleGAN**](https://papers.nips.cc/paper/2020/file/f14bc21be7eaeed046fed206a492e652-Paper.pdf)] CircleGAN: Generative Adversarial Learning across Spherical Circles  [[tensorflow](https://github.com/POSTECH-CVLab/circlegan)]
- [NIPS'20] [[**DeepI2I**](https://proceedings.neurips.cc/paper/2020/file/88855547570f7ff053fff7c54e5148cc-Paper.pdf)] DeepI2I: Enabling Deep Hierarchical Image-to-Image Translation by Transferring from GANs  [[pytorch](https://github.com/yaxingwang/DeepI2I)]
- [NIPS'20] [[**NVAE**](https://proceedings.neurips.cc/paper/2020/file/e3b21256183cf7c2c7a66be163579d37-Paper.pdf)] NVAE: A Deep Hierarchical Variational Autoencoder  [[pytorch](https://github.com/NVlabs/NVAE)]
- [NIPS'20] [[**Swapping-Autoencoder**](https://proceedings.neurips.cc/paper/2020/file/50905d7b2216bfeccb5b41016357176b-Paper.pdf)] Swapping Autoencoder for Deep Image Manipulation  [[web](https://taesung.me/SwappingAutoencoder/)] [[pytorch](https://github.com/taesungp/swapping-autoencoder-pytorch)]
- [ECCV'20] [[**COCO-FUNIT**](https://nvlabs.github.io/COCO-FUNIT/paper.pdf)] COCO-FUNIT: Few-Shot Unsupervised Image Translation with a Content Conditioned Style Encoder  [[web](https://nvlabs.github.io/COCO-FUNIT/)] [[pytorch](https://github.com/NVlabs/imaginaire/blob/master/projects/coco_funit/README.md)]
- [ECCV'20] [[**TopologyGAN**](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480120.pdf)] TopologyGAN: Topology Optimization Using Generative Adversarial Networks Based on Physical Fields Over the Initial Domain  [[pytorch](https://github.com/basiralab/topoGAN)]
- [ECCV'20] [[**wc-Vid2Vid**](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530358.pdf)] World-Consistent Video-to-Video Synthesis  [[web](https://nvlabs.github.io/wc-vid2vid/)] [[pytorch](https://github.com/NVlabs/imaginaire/blob/master/projects/wc_vid2vid/README.md)]
- [CVPR'20] [[**StarGAN2**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.pdf)] StarGAN v2: Diverse Image Synthesis for Multiple Domains  [[pytorch](https://github.com/clovaai/stargan-v2)]
- [CVPR'20] [[**Spectral-Regularization**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.pdf)] Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions  [[pytorch](https://github.com/cc-hpc-itwm/UpConv)]
- [NeurIPS'19] [[**fs-Vid2Vid**](https://openreview.net/pdf?id=rkluKVrl8H)] Few-shot Video-to-Video Synthesis  [[web](https://nvlabs.github.io/few-shot-vid2vid/)] [[pytorch](https://github.com/NVlabs/few-shot-vid2vid)]
- [ICCV'19] [[**FUNIT**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Few-Shot_Unsupervised_Image-to-Image_Translation_ICCV_2019_paper.pdf)] Few-Shot Unsupervised Image-to-Image Translation  [[web](https://nvlabs.github.io/FUNIT/)] [[pytorch](https://github.com/NVlabs/FUNIT)]
- [CVPR'19] [[**SPADE**](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.pdf)] Semantic Image Synthesis with Spatially-Adaptive Normalization  [[web](https://nvlabs.github.io/SPADE/)] [[pytorch](https://github.com/nvlabs/spade/)]
- [NeurIPS'18] [[**Glow**](https://papers.nips.cc/paper/2018/file/d139db6a236200b21cc7f752979132d0-Paper.pdf)] Glow: Generative Flow with Invertible 1x1 Convolutions  [[pytorch](https://github.com/chaiyujin/glow-pytorch)] [[tensorflow](https://github.com/openai/glow)]
- [NeurIPS'18] [[**Vid2Vid**](https://papers.nips.cc/paper/2018/file/d86ea612dec96096c5e0fcc8dd42ab6d-Paper.pdf)] Video-to-Video Synthesis  [[web](https://tcwang0509.github.io/vid2vid/)] [[pytorch](https://github.com/NVIDIA/vid2vid)]
- [CVPR'18] [[**StarGAN**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf)] StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation  [[pytorch](https://github.com/yunjey/stargan)]
- [CVPR'18] [[**Pix2PixHD**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.pdf)] High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs  [[web](https://tcwang0509.github.io/pix2pixHD/)] [[pytorch](https://github.com/NVIDIA/pix2pixHD)]
- [ECCV'18] [[**MUNIT**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf)] Multimodal Unsupervised Image-to-Image Translation  [[pytorch](https://github.com/NVlabs/MUNIT)]
- [NeurIPS'17] [[**UNIT**](https://papers.nips.cc/paper/2017/file/dc6a6489640ca02b0d42dabeb8e46bb7-Paper.pdf)] Unsupervised Image-to-Image Translation Networks  [[pytorch](https://github.com/mingyuliutw/UNIT)]
- [ICCV'17] [[**CycleGAN**](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks  [[web](https://junyanz.github.io/CycleGAN/)] [[pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]
- [CVPR'17] [[**Pix2Pix**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)] Image-to-Image Translation with Conditional Adversarial Networks  [[web](https://phillipi.github.io/pix2pix/)] [[pytorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)]

---
### Basic Nets - Vector2Image
- [arXiv'21.04] [[**ViTGAN**](https://arxiv.org/pdf/2107.04589.pdf)] ViTGAN: Training GANs with Vision Transformers
- [arXiv'21.06] [[**Alias-Free-GAN**](https://arxiv.org/pdf/2106.12423.pdf)] Alias-Free Generative Adversarial Networks 
- [arXiv'21.04] [[**MobileStyleGAN**](https://arxiv.org/pdf/2104.04767.pdf)] MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis  [[pytorch](https://github.com/bes-dev/MobileStyleGAN.pytorch)]
- [arXiv'21.03] [[**GANformer**](https://arxiv.org/pdf/2103.01209.pdf)] GANformer: Generative Adversarial Transformers  [[tensorflow](https://github.com/dorarad/gansformer)]
- [arXiv'21.02] [[**TransGAN**](https://arxiv.org/pdf/2102.07074.pdf)] TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up  [[pytorch](https://github.com/VITA-Group/TransGAN)]
- [arXiv'21.02] [[**SWAGAN**](https://arxiv.org/pdf/2102.06108.pdf)] SWAGAN: A Style-based Wavelet-driven Generative Model  [[pytorch](https://github.com/rosinality/stylegan2-pytorch)]
- [arXiv'21.02] [[**MedT**](https://arxiv.org/pdf/2102.10662.pdf)] Medical Transformer: Gated Axial-Attention for Medical Image Segmentation  [[pytorch](https://github.com/jeya-maria-jose/Medical-Transformer)]
- [arXiv'20.06] [[**StyleGAN2-ADA**](https://arxiv.org/pdf/2006.06676.pdf)] Training Generative Adversarial Networks with Limited Data  [[pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)]
- [CVPR'21] [[**BNCR-GAN**](https://openaccess.thecvf.com/content/CVPR2021/papers/Kaneko_Blur_Noise_and_Compression_Robust_Generative_Adversarial_Networks_CVPR_2021_paper.pdf)] Blur, Noise, and Compression Robust Generative Adversarial Networks  [[web](https://takuhirok.github.io/BNCR-GAN/)]
- [CVPR'21] [[**P2GAN**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Posterior_Promoted_GAN_With_Distribution_Discriminator_for_Unsupervised_Image_Synthesis_CVPR_2021_paper.pdf)] Posterior Promoted GAN with Distribution Discriminator for Unsupervised Image Synthesis 
- [CVPR'21] [[**OSGAN**](https://openaccess.thecvf.com/content/CVPR2021/papers/Shen_Training_Generative_Adversarial_Networks_in_One_Stage_CVPR_2021_paper.pdf)] Training Generative Adversarial Networks in One Stage  [[pytorch](https://github.com/zju-vipa/OSGAN)]
- [CVPR'21] [[**LeCam-GAN**](https://openaccess.thecvf.com/content/CVPR2021/papers/Tseng_Regularizing_Generative_Adversarial_Networks_Under_Limited_Data_CVPR_2021_paper.pdf)] Regularizing Generative Adversarial Networks under Limited Data  [[tensorflow](https://github.com/google/lecam-gan)]
- [CVPR'21] [[**EFTs**](https://openaccess.thecvf.com/content/CVPR2021/papers/Verma_Efficient_Feature_Transformations_for_Discriminative_and_Generative_Continual_Learning_CVPR_2021_paper.pdf)] Efficient Feature Transformations for Discriminative and Generative Continual Learning 
- [CVPR'21] [[**CIPS**](https://openaccess.thecvf.com/content/CVPR2021/papers/Anokhin_Image_Generators_With_Conditionally-Independent_Pixel_Synthesis_CVPR_2021_paper.pdf)] Image Generators with Conditionally-Independent Pixel Synthesis  [[pytorch](https://github.com/saic-mdal/CIPS)]
- [ICLR'21] [[**ContraD**](https://openreview.net/pdf?id=eo6U4CAwVmg)] Training GANs with Stronger Augmentations via Contrastive Discriminator  [[pytorch](https://github.com/jh-jeong/ContraD)]
- [ICLR'21] [[**CcGAN**](https://openreview.net/pdf?id=PxTIG12RRHS)] CcGAN: Continuous Conditional Generative Adversarial Networks for Image Generation  [[pytorch](https://github.com/UBCDingXin/improved_CcGAN)]
- [CVPR'20] [[**StyleGAN2**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.pdf)] Analyzing and Improving the Image Quality of StyleGAN  [[pytorch](https://github.com/rosinality/stylegan2-pytorch)] [[tensorflow](https://github.com/NVlabs/stylegan2)]
- [CVPR'20] [[**MSG-GAN**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Karnewar_MSG-GAN_Multi-Scale_Gradients_for_Generative_Adversarial_Networks_CVPR_2020_paper.pdf)] MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks  [[pytorch](https://github.com/akanimax/BMSG-GAN)]
- [CVPR'20] [[**UNet-GAN**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schonfeld_A_U-Net_Based_Discriminator_for_Generative_Adversarial_Networks_CVPR_2020_paper.pdf)] A U-Net Based Discriminator for Generative Adversarial Networks  [[pytorch](https://github.com/boschresearch/unetgan)]
- [CVPR'20] [[**AdversarialNAS**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gao_AdversarialNAS_Adversarial_Neural_Architecture_Search_for_GANs_CVPR_2020_paper.pdf)] AdversarialNAS: Adversarial Neural Architecture Search for GANs  [[pytorch](https://github.com/chengaopro/AdversarialNAS)]
- [CVPR'19] [[**StyleGAN**](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf)] A Style-Based Generator Architecture for Generative Adversarial Networks  [[pytorch](https://github.com/rosinality/style-based-gan-pytorch)] [[tensorflow](https://github.com/NVlabs/stylegan)]
- [ICLR'19] [[**BigGAN**](https://openreview.net/pdf?id=B1xsqj09Fm)] Large Scale GAN Training for High Fidelity Natural Image Synthesis  [[pytorch](https://github.com/sxhxliang/BigGAN-pytorch)] [[tensorflow](https://github.com/taki0112/BigGAN-Tensorflow)]
- [ICLR'19] [[**COCO-GAN**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_COCO-GAN_Generation_by_Parts_via_Conditional_Coordinating_ICCV_2019_paper.pdf)] COCO-GAN: Generation by Parts via Conditional Coordinating  [[pytorch](https://github.com/shaanrockz/COCO-GAN)] [[tensorflow](https://github.com/hubert0527/COCO-GAN)]
- [ICLR'18] [[**PGGAN**](https://openreview.net/pdf?id=Hk99zCeAb)] Progressive Growing of GANs for Improved Quality, Stability, and Variation  [[tensorflow](https://github.com/tkarras/progressive_growing_of_gans)]
- [ICLR'16] [[**DCGAN**](https://arxiv.org/pdf/1511.06434.pdf)] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks  [[pytorch](https://github.com/eriklindernoren/PyTorch-GAN#deep-convolutional-gan)]
- [NIPS'16] [[**CPixelCNN**](https://papers.nips.cc/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf)] Conditional Image Generation with PixelCNN Decoders  [[tensorflow](https://github.com/anantzoid/Conditional-PixelCNN-decoder)]
- [PMLR'16] [[**PixelRNN**](https://arxiv.org/pdf/1601.06759.pdf)] Pixel Recurrent Neural Networks  [[pytorch](https://github.com/j-min/PixelCNN)] [[tensorflow](https://github.com/carpedm20/pixel-rnn-tensorflow)]
- [arXiv'14.11] [[**CGAN**](https://arxiv.org/pdf/1411.1784.pdf)] Conditional Generative Adversarial Nets  [[pytorch](https://github.com/eriklindernoren/PyTorch-GAN#conditional-gan)]


---
### Face Swap
- [IJCAI'21] [[**HifiFace**](https://arxiv.org/pdf/2106.09965.pdf)] HifiFace: 3D Shape and Semantic Prior Guided High Fidelity Face Swapping  [[web](https://johann.wang/HifiFace/)]
- [CVPR'21] [[**HFaceInpainter**](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_FaceInpainter_High_Fidelity_Face_Adaptation_to_Heterogeneous_Domains_CVPR_2021_paper.pdf)] HFaceInpainter: High Fidelity Face Adaptation to Heterogeneous Domains 
- [CVPR'21] [[**MegaFS**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_One_Shot_Face_Swapping_on_Megapixels_CVPR_2021_paper.pdf)] One Shot Face Swapping on Megapixels  [[pytorch](https://github.com/zyainfal/One-Shot-Face-Swapping-on-Megapixels)]
- [CVPR'20] [[**Faceshifter**](https://arxiv.org/pdf/1912.13457.pdf)] FaceShifter: Towards High Fidelity And Occlusion Aware Face Swapping  [[web](https://lingzhili.com/FaceShifterPage/)]
- [SIGGRAPH'20] [[**IDDis**](https://arxiv.org/pdf/2005.07728.pdf)] Face Identity Disentanglement via Latent Space Mapping  [[pytorch](https://github.com/YotamNitzan/ID-disentanglement)]
- [AAAI'20] [[**Facecontroller**](https://arxiv.org/pdf/2102.11464.pdf)] FaceController: Controllable Attribute Editing for Face in the Wild 
- [ACCV'20] [[**UnifiedSR**](https://openaccess.thecvf.com/content/ACCV2020/papers/Le_Minh_Ngo_Unified_Application_of_Style_Transfer_for_Face_Swapping_and_Reenactment_ACCV_2020_paper.pdf)] Unified Application of Style Transfer for Face Swapping and Reenactment 
- [ACM'20] [[**SimSwap**](https://dl.acm.org/doi/pdf/10.1145/3394171.3413630)] SimSwap: An Efficient Framework For High Fidelity Face Swapping  [[pytorch](https://github.com/neuralchen/SimSwap)]
- [arXiv'20.05] [[**DeepFaceLab**](https://arxiv.org/pdf/2005.05535.pdf)] DeepFaceLab: Integrated, flexible and extensible face-swapping framework 
- [ICCV'19] [[**FSGAN**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Nirkin_FSGAN_Subject_Agnostic_Face_Swapping_and_Reenactment_ICCV_2019_paper.pdf)] FSGAN: Subject Agnostic Face Swapping and Reenactment  [[pytorch](https://github.com/YuvalNirkin/fsgan)]
- [arXiv'18.04] [[**RSGAN**](https://arxiv.org/pdf/1804.03447.pdf)] RSGAN: Face Swapping and Editing using Face and Hair Representation in Latent Spaces 
- [CVPR'18] [[**OIPFS**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bao_Towards_Open-Set_Identity_CVPR_2018_paper.pdf)] Towards Open-Set Identity Preserving Face Synthesis 
- [arXiv'18.04] [[**RSGAN**](https://arxiv.org/pdf/1804.03447.pdf)] RSGAN: Face Swapping and Editing using Face and Hair Representation in Latent Spaces 
- [CVPR'18] [[**IPNet**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bao_Towards_Open-Set_Identity_CVPR_2018_paper.pdf)] Towards Open-Set Identity Preserving Face Synthesis 
- [ACCV'18] [[**FSNet**](https://arxiv.org/pdf/1811.12666.pdf)] FSNet: An Identity-Aware Generative Model for Image-based Face Swapping  [[web](https://tatsy.github.io/projects/fsnet/)]

---
### Face Reenactment
- [CVPR'21] [[**face-vid2vid**](https://nvlabs.github.io/face-vid2vid/main.pdf)] One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing  [[web](https://nvlabs.github.io/face-vid2vid/)]
- [CVPR'20] [[**FReeNet**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_FReeNet_Multi-Identity_Face_Reenactment_CVPR_2020_paper.pdf)] FReeNet: Multi-Identity Face Reenactment  [[pytorch](https://github.com/zhangzjn/FReeNet)]
- [ICASSP'20] [[**APB2Face**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9052977&tag=1)] APB2Face: Audio-guided face reenactment with auxiliary pose and blink signals  [[pytorch](https://github.com/zhangzjn/APB2Face)]
- [ECCV'20] [[**Bi-layer**](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570511.pdf)] Fast Bi-layer Neural Synthesis of One-Shot Realistic Head Avatars  [[pytorch](https://github.com/saic-violet/bilayer-model)]
- [arXiv'20.12] [[**HeadGAN**](https://arxiv.org/pdf/2012.08261.pdf)] HeadGAN: One-shot Neural Head Synthesis and Editing
- [arXiv'20.05] [[**FaR-GAN**](https://arxiv.org/pdf/2005.06402.pdf)] FaR-GAN for One-Shot Face Reenactment 
- [AAAI'20] [[**DAE-GAN**](https://ojs.aaai.org/index.php/AAAI/article/view/6970/6824)] Realistic Face Reenactment via Self-Supervised Disentangling of Identity and Pose 
- [NIPS'19] [[**FOMM**](https://proceedings.neurips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf)] First Order Motion Model for Image Animation  [[web](https://aliaksandrsiarohin.github.io/first-order-model-website/)] [[pytorch](https://github.com/AliaksandrSiarohin/first-order-model)]
- [ECCV'18] [[**OSFR**](https://arxiv.org/pdf/1908.03251.pdf)] One-shot Face Reenactment  [[pytorch](https://github.com/bj80heyue/One_Shot_Face_Reenactment)]
- [ECCV'18] [[**X2Face**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Olivia_Wiles_X2Face_A_network_ECCV_2018_paper.pdf)] X2Face: A network for controlling face generation by using images, audio, and pose codes  [[pytorch](https://github.com/oawiles/X2Face)]
- [ECCV'18] [[**ReenactGAN**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Wayne_Wu_Learning_to_Reenact_ECCV_2018_paper.pdf)] ReenactGAN: Learning to Reenact Faces via Boundary Transfer  [[pytorch](https://github.com/wywu/ReenactGAN)]

---
### Face Stylizing
- [arXiv'21.06] [[**Cartoon-StyleGan2**](https://arxiv.org/pdf/2106.12445.pdf)] Fine-Tuning StyleGAN2 For Cartoon Face Generation  [[pytorch](https://github.com/happy-jihye/Cartoon-StyleGan2)]
- [Siggraph'21] [[**AgileGAN**](https://guoxiansong.github.io/homepage/paper/AgileGAN.pdf)] AgileGAN: Stylizing Portraits by Inversion-Consistent Transfer Learning  [[web](https://guoxiansong.github.io/homepage/agilegan_cn.html)]
- [ACM TOG'21] [[**StyleFlow**](https://dl.acm.org/doi/pdf/10.1145/3447648)] StyleFlow: Attribute-conditioned Exploration of StyleGAN-Generated Images using Conditional Continuous Normalizing Flows  [[pytorch](https://github.com/RameenAbdal/StyleFlow)]
- [NeurIPSW'20] [[**Toonify**](https://arxiv.org/pdf/2010.05334.pdf)] Resolution Dependent GAN Interpolation for Controllable Image Synthesis Between Domains  [[web](https://rameenabdal.github.io/StyleFlow/)] [[tensorflow](https://github.com/justinpinkney/toonifyl)]

---
### Audio-driven talking face

- [TIFS'22] [[**EBT**](https://arxiv.org/abs/2001.05201)] Everybody’s talkin’: Let me talk as you want  [[pytorch](https://wywu.github.io/projects/EBT/EBT.html)]
- [CVPR'21] [[**EVP**](https://jixinya.github.io/projects/evp/resources/evp.pdf)] Audio-Driven Emotional Video Portraits   [[pytorch](https://github.com/jixinya/EVP/)]
- [ACMMM'21] [[**style_avatar**](https://arxiv.org/abs/2111.00203)] Imitating Arbitrary Talking Style for Realistic Audio-Driven Talking Face Synthesis   [[pytorch](https://github.com/wuhaozhe/style_avatar)]
- [ICCV'21] [[**AD-NeRF**](https://arxiv.org/abs/2103.11078)] Audio Driven Neural Radiance Fields for Talking Head Synthesis   [[pytorch](https://github.com/YudongGuo/AD-NeRF)]
- [CVPR'21] [[**Talking-Face_PC-AVS**](https://arxiv.org/abs/2104.11116)] Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation   [[pytorch](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS)]
- [arXiv'20] [[**Audio-driven-TalkingFace-HeadPose**](http://arxiv.org/abs/2002.10137)] Audio-driven Talking Face Video Generation with Learning-based Personalized Head Pose   [[pytorch](https://github.com/yiranran/Audio-driven-TalkingFace-HeadPose)]
- [ECCV'20] [[**Neural voice puppetry**](https://arxiv.org/abs/1912.05566)] Neural voice puppetry: Audio-driven facial reenactment  [[pytorch](https://github.com/miu200521358/NeuralVoicePuppetryMMD)]
- [CVPR'19] [[**ATVGnet**](https://arxiv.org/abs/1905.03820)] Hierarchical cross-modal talking face generation with dynamic pixel-wise loss  [[pytorch](https://github.com/lelechen63/ATVGnet)]
- [AAAI'19] [[**Talking-Face-Generation-DAVS**](https://arxiv.org/abs/1807.07860)] Talking Face Generation by Adversarially Disentangled Audio-Visual Representation    [[pytorch](https://github.com/Hangz-nju-cuhk/Talking-Face-Generation-DAVS)]
- [ECCV'18] [[**X2Face**](https://www.robots.ox.ac.uk/~vgg/publications/2018/Wiles18/wiles18.pdf)] X2Face: A network for controlling face generation using images, audio, and pose codes    [[pytorch](https://github.com/oawiles/X2Face)]
- [SIGGRAPH'17] [[**obama**](http://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf)] Audio-Driven Emotional Video Portraits   [[tensorflow](https://github.com/supasorn/synthesizing_obama_network_training)]
  
---
### Tools
- [CVPR'19] [Face Recognition] [[**ArcFace**](https://arxiv.org/pdf/1801.07698.pdf)] ArcFace: Additive Angular Margin Loss for Deep Face Recognition [[code](https://github.com/deepinsight/insightface)]
- [ECCV'18] [Face Segmentation] [[**BiSeNet**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper.pdf)] BiSeNet: Bilateral segmentation network for real-time semantic segmentation [[code](https://github.com/zllrunning/face-parsing.PyTorch)]
- [Face Segmentation] [[**CeleAMask-HQ**]()] [[code](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing)]
- [CVPRW'19] [[**3DMM**](https://openaccess.thecvf.com/content_CVPRW_2019/papers/AMFG/Deng_Accurate_3D_Face_Reconstruction_With_Weakly-Supervised_Learning_From_Single_Image_CVPRW_2019_paper.pdf)] Accurate 3D face reconstruction with weakly-supervised learning: From single image to image set [[code](hhttps://github.com/microsoft/Deep3DFaceReconstruction)]
- [IEEE SPL'16] [Face Alignment / Detection] [[**MTCNN**](https://arxiv.org/pdf/1604.02878.pdf)] Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks [[code](https://github.com/ipazc/mtcnn)]
- [IJCB'17] [Face Alignment / Detection] [[**Faceboxes**](https://arxiv.org/pdf/1708.05234.pdf)] FaceBoxes: A CPU Real-time Face Detector with High Accuracy [[code](https://github.com/zisianw/FaceBoxes.PyTorch)]
- [CVPR'20] [Face Alignment / Detection] [[**RetinaFace**](https://arxiv.org/pdf/1905.00641.pdf)] RetinaFace: Single-stage Dense Face Localisation in the Wild [[code](https://github.com/biubug6/Pytorch_Retinaface)]
- [Face Alignment / Detection] [[**DLIB**]()] [[code](https://github.com/davisking/dlib)]

---
### Datasets
- [[**CelebA**](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)]
- [[**CelebA-HQ**](https://github.com/tkarras/progressive_growing_of_gans)]
- [[**CelebAMask-HQ**](https://github.com/switchablenorms/CelebAMask-HQ)]
- [[**CelebA-Spoof**](https://github.com/Davidzhangyuanhan/CelebA-Spoof)]
- [[**FFHQ**](https://github.com/NVlabs/ffhq-dataset)]
- [[**VoxCeleb1**](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)]
- [[**VoxCeleb2**](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)]
- [[**VGGFace**](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/)]
- [[**VGGFace2**](https://github.com/ox-vgg/vgg_face2)]
- [[**RaFD**](http://www.socsci.ru.nl:8180/RaFD2/RaFD)]
- [[**Multi-PIE**](https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)]
- [[**FaceForensics++**](https://github.com/ondyari/FaceForensics)]
- [[**SCUT-FBP**](https://link.zhihu.com/?target=https%3A//github.com/HCIILAB/SCUT-FBP5500-Database-Release)]
- [[**MakeUp**](http://www.antitza.com/makeup-datasets.html)]

---
### Great Repos
- [[**awesome-gan-inversion**](https://github.com/weihaox/awesome-gan-inversion)]
