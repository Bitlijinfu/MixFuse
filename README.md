# MixFuse: An iterative mix-attention transformer for multi-modal image fusion (Expert Systems with Applications 2024)

This is the official implementation of the MixFuse model proposed in the paper ([MixFuse: An iterative mix-attention transformer for multi-modal image fusion](https://authors.elsevier.com/c/1jtYo3PiGTTUsU)) with Pytorch.

## Abstract

Multi-modal image fusion plays a crucial role in various visual systems. However, existing methods typically involve a multi-stage pipeline, i.e., feature extraction, integration, and reconstruction, which limits the effectiveness and efficiency of feature interaction and aggregation. In this paper, we propose MixFuse, a compact multi-modal image fusion framework based on Transformers. It smoothly unifies the process of feature extraction and integration, As its core, the Mix Attention Transformer Block (MATB) integrates the Cross-Attention Transformer Module (CATM) and the Self-Attention Transformer Module (SATM). The CATM introduces a symmetrical cross-attention mechanism to identify modality-specific and general features, filtering out irrelevant and redundant information. Meanwhile, the SATM is designed to refine the combined features via a self-attention mechanism, enhancing the internal consistency and proper preservation of the features. This successive cross and self-attention modules work together to enhance the generation of more accurate and refined feature maps, which are essential for later reconstruction. Extensive evaluation of MixFuse on five public datasets shows its superior performance and adaptability over state-of-the-art methods. The code and model will be released at <https://github.com/Bitlijinfu/MixFuse>.

<figure>
    <img src=img\pipeline.jpg />
    <figcaption>
        <p> Fig.1. A comparison of multi-modal image fusion pipeline. (a) The mainstream image fusion framework consists of three parts: a CNN/Transformer-based backbone network for feature extraction, predefined rules for feature integration, and a convolutional reconstruction module. The typical fusion results of SwinFusion (Ma et al., 2022) overemphasizes textures of VIS images while reducing the intensity of IR images, resulting in unclear visual elements, e.g., buildings and persons in the red and blue boxes, respectively. (b) MixFuse involves two parts: a mix-attention Transformer for simultaneous feature extraction and integration and a convolutional reconstruction module. It effectively mines the relationships of different modalities, which not only generates a visually attractive output but also bridges the gap between them.</p>
    </figcaption>
</figure>

<figure>
    <img src=img\scene.jpg />
    <figcaption>
        <p> Fig. 2. The typical fusion results of SwinFusion (Ma et al., 2022) and our MixFuse in five popular scenarios. The top row represents the pairs of source images, and the bottom row displays the corresponding fusion results. SwinFusion often weakens the features of IR and MRI sources, leading to an excessive enhancement of features from VIS, PET, and SPECT images. However, MixFuse consistently produces images with improved contrast while preserving appropriate modality-specific and modality-general information.</p>
    </figcaption>
</figure>

<figure>
    <img src=img\mainframework.jpg />
    <figcaption>
        <p> Fig. 3. MixFuse is constructed by a backbone based on the MATB and a reconstruction module. The MATB seamlessly merges feature extraction and integration by sequentially stacking the CATM and the SATM.</p>
    </figcaption>
</figure>

## Citation

If this work is helpful to you, please cite it as:

@article{LI2025125427,  
title = {MixFuse: An iterative mix-attention transformer for multi-modal image fusion},
journal = {Expert Systems with Applications},  
volume = {261},  
pages = {125427},  
year = {2025},  
issn = {0957-4174},  
doi = {<https://doi.org/10.1016/j.eswa.2024.125427>},  
url = {<https://www.sciencedirect.com/science/article/pii/S0957417424022942>},  
author = {Jinfu Li and Hong Song and Lei Liu and Yanan Li and Jianghan Xia and Yuqi Huang and Jingfan Fan and Yucong Lin and Jian Yang}
}
