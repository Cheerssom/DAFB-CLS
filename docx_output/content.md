Foreground-Background Decoupling for Vision Transformer CLS Aggregation (2026)



$Jun−Yi Zhang^{1}$ ， $Xiang Wang^{2}$







Abstract—The CLS token in Vision Transformers suffers from two complementary forms of indiscriminate aggregation: spatially, it shortcuts global semantics from background patches under coarse-grained supervision; in depth, standard residual connections dilute early-layer contributions with fixed equal-weight accumulation. Prior work such as LaSt-ViT addressed only the spatial issue via frequency-based Top-K selection with a fixed ratio.We present DAFB-CLS, a lightweight post-hoc framework that jointly resolves both limitations on frozen pretrained ViTs. Our approach fuses four cues -- frequency stability, depth consistency, semantic alignment, and spatial compactness -- into a unified foregroundness score, and predicts an image-adaptive threshold to produce a soft foreground mask, replacing rigid fixed-ratio selection. Separate foreground and background CLS tokens perform independent depth-wise softmax attention over historical block representations, then fuse through a learned task-adaptive gate. The framework adds only 0.59M trainable parameters (2--3% of the backbone) with 1.3× latency overhead.On DINO ViT-S/16, DAFB-CLS achieves CorLoc of 79.43% on VOC2012 object discovery (+13.7pp over LaSt-ViT) and Mask IoU of 31.27% (+18.0pp), with consistent gains on COCO (72.96% CorLoc, +11.4pp). Ablation confirms each component is essential: removing foregroundness cues collapses CorLoc to 47%, fixed Top-K drops Mask IoU to 14%, and disabling depth attention halves PiB from 45% to 28%



Index Terms—Vision Transformer,CLS token decoupling,

foreground-background aggregation,depth-wise attention,

object discovery





https://github.com/Cheerssom/DAFB-CLS





II. RELATED WORK

A. Vision Transformers and Dense Feature Representations

1) From Global Attention to Structured Variants

The Vision Transformer (ViT) [Dosovitskiy et al., 2021] demonstrated that pure attention-based architectures can match or exceed convolutional networks on image classification when pretrained at sufficient scale. By treating image patches as tokens and appending a learnable [CLS] token, ViT leverages global self-attention to build a holistic image representation from which the [CLS] token aggregates information across all spatial locations. This design has since become the foundation of numerous vision and multimodal models, including CLIP [Radford et al., 2021], DINO [Caron et al., 2021], and BEiT [Bao et al., 2022].

However, global quadratic attention also introduces substantial computational overhead, motivating a line of work on structured sparse attention. PVT [Wang et al., 2021] progressively reduces spatial resolution across stages via spatial-reduction attention, while Swin Transformer [Liu et al., 2021] restricts attention within local windows with periodic shifted-window cross-region communication. These architectural innovations significantly improve efficiency and have demonstrated strong performance on dense prediction tasks, yet they impose a fixed spatial attention topology that is data-agnostic. DAT [Xia et al., 2022] addressed this limitation through deformable attention, dynamically predicting sampling offsets from input content to attend to task-relevant spatial locations. MViT [Fan et al., 2021] introduced multi-scale feature hierarchies with pooled attention, progressively expanding channel dimensions while reducing spatial resolution. Despite these advances in architecture design, a separate but equally important class of problems concerns the quality and interpretability of dense feature representations that ViT produces.

2) Artifacts and Pathologies in ViT Dense Features

While ViT excels at global image-level tasks, a growing body of evidence reveals systematic pathologies in its dense (patch-level) feature maps. Darcet et al. [2024] first documented the “high-norm token” phenomenon: in self-supervised ViTs such as DINOv2, certain background tokens inflate to extreme activation norms, creating spatial artifacts that degrade dense prediction quality. They proposed adding register tokens—unused positions that absorb background noise—as a pragmatic fix. While effective at eliminating anomalous high-norm patches, register tokens treat the symptom without addressing the underlying cause, and do not improve or may even reduce point-in-box localization accuracy.

Concurrently, label-supervised ViTs exhibit attention deficit: the [CLS] attention map often disperses across irrelevant background regions rather than focusing on class-relevant objects [Naseer et al., 2021]. In text-supervised models like CLIP, patch features may not align well with grounded textual semantics, leading to poor performance on open-vocabulary segmentation and detection tasks [Park et al., 2022; Zhou et al., 2022]. These seemingly distinct pathologies share a common root: the coarse-grained supervision signals (image-level labels or image-caption pairs) do not constrain the spatial specificity of the aggregated representation.

3) Diagnostic Frameworks for Patch-Level Quality

A critical contribution toward understanding these pathologies was made by Shi et al. [2025], who proposed two complementary diagnostic metrics. The Patch Score, defined as the cosine similarity between each patch feature and the global [CLS] representation, quantifies how strongly each spatial location contributes to the final image-level representation. The Point-in-Box (PiB) metric evaluates whether the highest-scoring patch falls within the annotated object bounding box. Using these metrics, Shi et al. demonstrated that vanilla ViT consistently assigns higher Patch Scores to background patches than foreground ones—achieving PiB values of only 19–48% compared to 60–80% for convolutional networks. Crucially, they showed that this low PiB emerges early in training and persists throughout optimization, confirming it as a structural pathology rather than a training artifact.

B. Selective CLS Aggregation: From Frequency Stability to Adaptive Selection

1) LaSt-ViT: Frequency Stability as a Foreground Proxy

Shi et al. [2025] diagnosed the root cause of ViT dense feature artifacts as Lazy Aggregation: the [CLS] token, under image-level supervision, tends to aggregate semantic information from whichever patches are most easily accessible—typically the majority background patches—rather than from the semantically informative foreground patches. This occurs because background patches can absorb foreground information through global attention, and aggregating from many background patches is a lower-entropy solution that minimizes training loss.

To address Lazy Aggregation, Shi et al. proposed LaSt-ViT (Lazy Strike), which replaces the default all-patch aggregation with a frequency-based selection mechanism. For each patch feature, LaSt-ViT applies a 1D FFT along the channel dimension, passes the result through a Gaussian low-pass filter, and inverse-transforms back. The rationale is that foreground object patches exhibit more stable (low-frequency) channel-wise activations, while background patches tend toward higher-frequency, less coherent activations. A channel-wise Top-K pooling then selects the K most stable patches for each feature channel, and their features are averaged to form the aggregated representation. This simple yet effective approach eliminates the high-norm phenomenon, substantially improves PiB, and delivers consistent gains on zero-shot semantic segmentation (e.g., +1.3 mIoU on ADE20K with CLIP ViT-B/16) and open-vocabulary detection.

While LaSt-ViT represents an important advance, it has several limitations that motivate our work. First, the fixed Top-K ratio creates a rigid foreground budget that cannot adapt to images with varying object sizes or complexity. Second, frequency stability alone may conflate smooth foreground objects (e.g., a white swan) with smooth background regions (e.g., sky or walls), leading to the stable-background failure mode that we systematically identify and address. Third, LaSt-ViT does not explicitly model the depth dimension of ViT’s multi-layer representations, missing the opportunity to selectively aggregate information across layers.

2) Adaptive and Multi-Cue Aggregation

Several concurrent works have explored alternatives to fixed aggregation strategies. TokenCut [Wang et al., 2023] leverages self-similarity graphs and normalized cuts to segment foreground tokens without any training, achieving strong object discovery results. However, graph-based methods incur non-trivial computational overhead on large token sets. MaskCLIP [Zhou et al., 2022] modifies CLIP’s attention to produce more spatially grounded patch features through text-guided masking, but requires text supervision at inference time. FreeDA [Zhang et al., 2024] proposes training-free debiased adaptation for CLIP, using covariance estimation to suppress background biases, yet operates at the output feature level rather than modifying the aggregation mechanism.

From a complementary direction, several works have explored adaptive token selection for efficiency. DynamicViT [Rao et al., 2021] and ATS [Fayyaz et al., 2022] learn to prune unimportant tokens progressively through the network, improving inference speed but not specifically targeting the CLS aggregation bias. ToMe [Bolya et al., 2023] merges similar tokens to reduce computation while preserving visual fidelity. These works demonstrate that intelligent spatial selection improves ViT representations, but they focus on efficiency rather than on correcting the fundamental CLS aggregation pathology.

Our work builds upon LaSt-ViT’s insight of frequency-guided selection while extending it in three critical dimensions: (i) we enrich the foreground signal with multiple complementary cues beyond frequency stability alone; (ii) we replace the fixed Top-K with an image-adaptive soft mask predicted from the global image feature; and (iii) we decouple the foreground and background streams rather than simply suppressing background, enabling each stream to independently develop task-relevant semantics.

C. Depth-Wise Information Aggregation in Transformers

1) Residual Connections as Implicit Aggregators

The residual connection, introduced by He et al. [2016] for deep convolutional networks and adopted universally in Transformer architectures, serves a dual role that is often underappreciated. Beyond providing an identity path for stable gradient flow—enabling the training of networks with hundreds or thousands of layers—residual connections implicitly define how information accumulates across depth. In the PreNorm formulation standard in modern Transformers, the hidden state at layer l is computed as h_l = h_{l-1} + f_l(h_{l-1}), where f_l is the l-th transformer block. Unrolling this recursion reveals that the input to any layer is the sum of the initial embedding and all previous layer outputs, each weighted uniformly by 1.

This uniform accumulation creates a fundamental tension in deep Transformers. As network depth increases, the hidden state magnitude grows approximately as O(L), causing the relative contribution of any single layer to be progressively diluted—a phenomenon termed PreNorm dilution [Xiong et al., 2020]. Early-layer features encoding low-level patterns become entangled in a monolithic representation, and later layers cannot selectively retrieve specific earlier information. This problem is particularly acute in Vision Transformers, where different layers capture fundamentally different levels of abstraction: early layers encode textures and edges, middle layers capture parts and patterns, and later layers encode object-level semantics [Raghu et al., 2021; Nayman et al., 2022].

2) Gated and Selective Residual Connections

Recognizing the limitations of fixed residual addition, several works have introduced learnable gating mechanisms. Highway Networks [Srivastava et al., 2015] pioneered this direction by adding data-dependent gates that control information flow through layers, predating the Transformer era. DenseNet [Huang et al., 2017] adopted an alternative strategy of concatenating rather than adding features from all preceding layers, giving later layers explicit access to all earlier representations at the cost of linearly growing memory.

In the Transformer context, GPT-2 [Radford et al., 2019] introduced learnable scaling factors on residual paths, initialized to  $1/\sqrt{N}$  where N is the number of residual connections, to stabilize training of deep models. More recently, DenseFormer [Petrov et al., 2024] extended this idea by learning per-layer static weights for all historical representations, providing content-independent but layer-specific aggregation. While an improvement over uniform addition, static weights cannot adapt to input-specific information needs—different inputs may require different layers to contribute at different magnitudes. SubLN [Bachmann et al., 2024] revisited PreNorm residual scaling and demonstrated that careful normalization within residual branches can partially mitigate the dilution effect.

3) Attention Residuals: Content-Dependent Depth Aggregation

The most directly relevant prior work on depth-wise aggregation is Attention Residuals (AttnRes) [Kimi Team, 2025], which fundamentally reconceptualizes the residual connection as a depth-dimension attention mechanism. In AttnRes, each layer l receives its input not through uniform summation but through a learned softmax attention over all historical layer outputs:  $h_{l}=\sum_{i}^{​} α_{i→l}⋅v_{i}$ , where  $v_{0}$  is the token embedding and  $v_{i}$  is the output of layer  $i$ . The attention weights  $α$  are computed using a learned pseudo-query vector  $w_{l}$  as the query and RMSNorm-transformed historical outputs as keys, enabling content-dependent depth selection.

AttnRes demonstrated that this selective depth aggregation yields a 1.25× compute advantage on language modeling benchmarks—achieved at essentially zero additional parameter cost, since the pseudo-queries are simply d-dimensional vectors. The authors further proposed Block Attention Residuals (Block AttnRes), which partitions layers into blocks and applies inter-block attention, reducing storage from O(Ld) to O(Nd) for N blocks while preserving most of the benefits. Empirical analysis on a 48B-parameter model showed that Block AttnRes eliminates the monotonic growth of hidden-state magnitude and produces more uniform gradient distributions across depth.

While AttnRes was developed in the context of language models, its core insight—that depth-wise aggregation should be content-adaptive rather than fixed—is directly applicable to Vision Transformers. Indeed, the multi-layer representations of ViT are even more heterogeneous than those of language models, with each layer capturing qualitatively different visual information. However, AttnRes addresses all tokens uniformly through a single aggregation mechanism, without distinguishing between foreground and background information streams. Our method extends this paradigm by coupling depth-wise attention with foreground-background decoupling, enabling independent depth aggregation for each semantic stream.

D. Foreground-Background Separation in Visual Representation Learning

1) Class Activation Mapping and CAM-Based Methods

Separating foreground from background is a longstanding challenge in computer vision. Class Activation Mapping (CAM) [Zhou et al., 2016] and its extensions—GradCAM [Selvaraju et al., 2017], GradCAM++ [Chattopadhyay et al., 2018], and ScoreCAM [Wang et al., 2020]—generate coarse localization heatmaps from classification networks by back-propagating or weighting activation maps. These methods have proven invaluable for weakly supervised localization and segmentation, but they inherit the same background bias as the underlying classifier. When the classification network relies on background context for prediction—as ViT often does—the CAM naturally highlights background regions, achieving only 21–25% PiB in our experiments.

DINO-seg [Oquab et al., 2024] extracts self-attention maps from self-supervised ViTs to produce unsupervised object segmentations, leveraging the observation that DINO’s attention heads naturally attend to semantically meaningful regions. While more spatially precise than CAM, DINO-seg’s attention maps still suffer from background activations and require post-processing (e.g., thresholding, PCA-based clustering) that introduces additional hyperparameters. Our approach can be seen as providing a principled, trainable alternative to these post-hoc foreground extraction methods.

2) Multi-Cue Visual Saliency and Objectness

Visual saliency detection has a rich history of leveraging multiple complementary cues. Classic methods combine color contrast, texture uniqueness, and spatial priors to predict fixation points [Itti et al., 1998; Hou & Zhang, 2007]. More recently, deep saliency models have incorporated edge detection, boundary cues, and semantic features [Hou et al., 2019]. The principle that no single cue is sufficient for robust saliency detection—and that cues should be adaptively weighted based on context—directly motivates our multi-cue foregroundness approach.

In the ViT feature space specifically, several works have noted that different cues capture complementary aspects of foregroundness. TokenCut [Wang et al., 2023] uses self-similarity as a graph-theoretic proxy for objectness. LOST [Siméoni et al., 2021] combines self-similarity with a seeding strategy that progressively refines the foreground region. FreeSeg [Qin et al., 2022] segments images without training by combining feature similarity with spatial compactness. These methods demonstrate that combining multiple signals yields more robust foreground estimates than any single measure, but they operate at the feature level without modifying the underlying aggregation mechanism. Our work integrates multi-cue reasoning directly into the CLS aggregation pipeline.

E. Connections to Post-Hoc Aggregation and Frozen Backbone Methods

An important design principle that our work shares with several prior methods is operating in a post-hoc manner on frozen pretrained backbones. This paradigm, exemplified by methods like MaskCLIP [Zhou et al., 2022], Saliency-guided Open-Vocabulary Segmentation [Liang et al., 2023], and FreeDA [Zhang et al., 2024], offers several practical advantages: it avoids costly retraining of large foundation models, preserves the backbone’s generalization properties, and allows the aggregation module to be lightweight and fast to train. Our approach follows this paradigm, training only approximately 0.59M parameters (2.7% of the DINO ViT-S/16 backbone) while achieving substantial improvements in dense feature quality.

More broadly, our work fits into the emerging trend of studying and correcting information flow within pretrained Transformers, rather than simply scaling up model or data size. Just as Attention Residuals revealed that the default residual connection is suboptimal for information aggregation across depth, we show that the default CLS pooling is suboptimal for spatial aggregation. Both findings point to the same overarching lesson: the default aggregation operations in Transformers—whether across layers or across spatial positions—are overly rigid and fail to exploit the structured information available in the intermediate representations.

F. Summary: Positioning Our Work

Table 1 positions our work relative to the most closely related methods along four key dimensions: spatial selectivity (whether aggregation is content-adaptive), multi-cue foregroundness (whether multiple complementary signals are integrated), adaptive budget (whether the foreground proportion is image-dependent), and depth-wise attention (whether layer aggregation is content-dependent). LaSt-ViT introduced spatial selectivity through frequency stability but uses fixed Top-K with a single cue. Attention Residuals introduced depth-wise attention but operates uniformly across spatial positions. Our DAFB-CLS framework is the first to unify all four dimensions into a single coherent framework for CLS aggregation in Vision Transformers.



TABLE I

COMPARISON WITH RELATED METHODS


[TABLE]




III. METHOD

A. Problem Formulation and Design Rationale

We consider a frozen Vision Transformer backbone (e.g., DINO ViT-S/16 or OpenCLIP ViT-B/16) that processes an image  $x∈ℝ^{H×W×3}$  into a sequence of patch tokens  ${x_{i}}_{i=1}^{N}$  plus a [CLS] token, where  $N=\left(H/P\right)×\left(W/P\right)$  for patch size  $P$ . The backbone consists of  $L$  transformer blocks; we extract intermediate features from a subset of layers  $ℒ={l_{1},l_{2},...,l_{K}}$  (typically  $K=4$  layers equally spaced, e.g.,  ${3,6,9,12}$  for a 12-layer ViT).

Our goal is to train a lightweight post-hoc aggregation module that transforms these multi-layer features into an improved image representation  $C$ , without modifying or retraining the frozen backbone. This design is motivated by two complementary insights from the literature: (i) the default [CLS] pooling suffers from Lazy Aggregation bias toward background tokens [Shi et al., 2025], and (ii) the standard residual connections accumulate layer outputs with fixed uniform weights, lacking content-adaptive depth selection [Kimi Team, 2025].

Our key design principle is to address both problems jointly through foreground-background decoupling: rather than suppressing background information, we maintain two separate semantic streams—a foreground stream  $C_{F}$  that aggregates foreground-identifying features, and a background stream  $C_{B}$  that aggregates contextual scene features—and learn to fuse them adaptively based on the task and input. Figure 1 provides an overview of the complete framework.

Fig. 1. Overview of DAFB-CLS.

B. Step 1: Multi-Layer Feature Extraction

Given input image  $x$ , we first pass it through the frozen ViT backbone  $Φ$  and extract intermediate patch features via forward hooks registered at the specified layers  $ℒ$ :

$${P^{l}}_{l∈ℒ}=Extract\left(Φ\left(x\right),ℒ\right), P^{l}∈ℝ^{B×N×D}$$

where  $P^{l}$  denotes the patch token features (excluding [CLS]) at layer  $l$ ,  $B$  is the batch size,  $N$  is the number of patches, and  $D$  is the feature dimension. We stack features across layers to form a 4D tensor  $P∈ℝ^{B×K×N×D}$ . The [CLS] token feature from the final extracted layer provides the global image representation  $c_{cls}∈ℝ^{B×D}$ . All features are detached from the backbone computation graph to ensure that backbone parameters remain frozen during training.

C. Step 2: Multi-Cue Foregroundness Scoring

The first major component computes a per-patch foregroundness score that indicates how likely each patch is to belong to a foreground object. Motivated by the observation that no single signal reliably distinguishes foreground from background across diverse image types, we design four complementary cues that capture different aspects of foregroundness. Each cue operates on the multi-layer patch features  $P$  and produces a per-patch score  $s_{k}∈ℝ^{B×N}$  for  $k∈{1,2,3,4}$ .

1) Frequency Stability Cue

Inspired by LaSt-ViT’s observation that foreground patches exhibit more coherent channel-wise activations, we measure the stability of each patch’s feature vector under spectral low-pass filtering. For each patch feature  $p_{i}∈ℝ^{D}$ , we compute its 1D FFT along the channel dimension, apply a Gaussian low-pass filter with cutoff frequency  $σ_{f}$ , and inverse-transform to obtain the filtered version  $p_{i}$ :

$$p_{i}=IFFT\left(FFT\left(p_{i}\right)⋅G\left(σ_{f}\right)\right), G\left(σ_{f}\right)=exp\left(−\frac{f^{2}}{2σ_{f}^{2}}\right)$$

The stability score for patch  $i$  is then:

$$S_{i}=\frac{1}{D}\sum_{d=1}^{D} \frac{p_{i,d}}{\left|p_{i,d}−p_{i,d}\right|+ϵ}$$

Higher values indicate that the patch feature changes minimally under spectral filtering, suggesting stable, low-frequency semantic content typical of foreground objects. We use  $σ_{f}=0.25$  as the default cutoff, following LaSt-ViT. The frequency cue operates on all extracted layers and the final-layer scores  $S_{i}^{\left(l_{K}\right)}$  are used downstream.

2) Depth Consistency Cue

While the frequency cue captures within-layer stability, the depth consistency cue captures stability across layers. Patches corresponding to semantically meaningful objects tend to maintain more consistent representations across network depth, while background patches exhibit greater inter-layer variation due to the absence of strong semantic anchoring. We measure depth consistency as the average cosine similarity between each layer’s patch feature and the cross-layer mean:

$$p_{i}=\frac{1}{K}\sum_{k=1}^{K} P_{i}^{l_{k}}, D_{i}=\frac{1}{K}\sum_{k=1}^{K} \frac{P_{i}^{l_{k}}⋅p_{i}}{∥P_{i}^{l_{k}}∥⋅∥p_{i}∥}$$

where  $P_{i}^{l_{k}}$  denotes the feature of patch  $i$  at layer  $l_{k}$ . This cue complements frequency stability by capturing temporal (across-depth) rather than spectral (within-channel) consistency, and is particularly effective for foreground objects whose semantic identity emerges progressively through the network.

3) Semantic Alignment Cue

The semantic alignment cue measures how strongly each patch’s feature aligns with the global image semantics, providing a content-based signal that complements the structural signals from the frequency and depth cues. We compute the average cosine similarity between each patch feature and the [CLS] token feature across all extracted layers:

$$A_{i}=\frac{1}{K}\sum_{k=1}^{K} \frac{P_{i}^{l_{k}}⋅c_{cls}}{∥P_{i}^{l_{k}}∥⋅∥c_{cls}∥}$$

For text-supervised backbones (e.g., OpenCLIP), we additionally support text-guided semantic alignment, where the similarity is computed between each patch and the set of text embeddings  ${t_{j}}$ , using the maximum similarity:  $A_{i}=max_{j}cos\left(P_{i}^{l_{K}},t_{j}\right)$ . This provides a more task-specific foreground signal when text features are available.

4) Spatial Compactness Cue

Foreground objects in natural images typically occupy spatially contiguous regions. The spatial compactness cue enforces this prior by applying local neighborhood smoothing to the semantic alignment scores, encouraging patches within the same spatial neighborhood to have similar foregroundness. Given the semantic score map  $s_{sem}∈ℝ^{N}$ , we reshape it to the spatial grid  $\left(h×w\right)$  and apply average pooling with kernel size  $k_{s}$ :

$$C_{i}=AvgPool\left(Reshape\left(s_{sem},h,w\right),k_{s}\right)_{\left[i\right]}$$

with  $k_{s}=3$  as default. This cue does not introduce additional learnable parameters but provides a useful structural inductive bias that complements the pointwise signals from the other cues.

5) Learned Cue Fusion

The four cues are fused through a combination of learnable weighting and an MLP refinement module. Let  $s_{i}=\left[S_{i},D_{i},A_{i},C_{i}\right]^{T}∈ℝ^{4}$  denote the cue vector for patch  $i$ . We first compute a weighted combination with learned softmax-normalized weights:

$$F_{i}^{linear}=\sum_{k=1}^{4} w_{k}⋅s_{i,k}, wherew=softmax\left(θ_{w}\right)$$

where  $θ_{w}∈ℝ^{4}$  are learnable parameters initialized uniformly. In parallel, a lightweight MLP with two hidden layers refines the cue vector to capture nonlinear interactions between cues:

$$q_{i}=W_{2}⋅GELU\left(W_{1}⋅s_{i}+b_{1}\right)+b_{2}$$

$$F_{i}^{mlp}=MLP\left(s_{i}\right)=W_{3}⋅GELU\left(q_{i}\right)+b_{3}$$

The final foregroundness score combines both components and applies spatial smoothing:

$$F_{i}=F_{i}^{linear}+F_{i}^{mlp}+0.5⋅σ\left(W_{s}∗Reshape\left(F^{linear}+F^{mlp}\right)\right)_{\left[i\right]}$$

where  $W_{s}$  is a  $3×3$  convolutional kernel initialized to uniform  $1/9$  and  $σ$  denotes the sigmoid function. This three-component fusion strategy provides both interpretable linear contributions and flexible nonlinear refinement, with the spatial smoothing ensuring spatial coherence of the foregroundness map. The total ForegroundnessHead introduces fewer than 5K parameters.

D. Step 3: Adaptive Foreground Budget

A critical design decision is how to convert the continuous foregroundness scores  ${F_{i}}_{i=1}^{N}$  into binary or soft foreground masks. Existing methods typically use fixed-ratio Top-K selection, where the top K% of patches are selected as foreground regardless of the image content. This rigid budget is problematic: images with a single small object may have fewer than 10% foreground patches, while images with multiple large objects may have over 50%.

We address this with an Adaptive Budget Module that predicts an image-specific threshold  $τ$  from the global feature  $c_{cls}$ , and applies a temperature-controlled sigmoid to produce a soft foreground mask:

$$τ=MLP_{τ}\left(c_{cls}\right), m_{i}^{F}=σ\left(\frac{F_{i}−τ}{T}\right)$$

where  $MLP_{τ}$  is a two-layer MLP that maps the global feature to a scalar threshold, and  $T$  is a learnable temperature parameter initialized to  $0.1$  (in log-space for stable optimization) and clamped to  $\left[0.01,1.0\right]$  during forward passes. The temperature controls the sharpness of the mask: lower  $T$  produces more binary-like masks, while higher  $T$  yields softer, more gradual transitions between foreground and background.

The adaptive threshold mechanism has several advantages over fixed Top-K: (i) it naturally adapts to images with different foreground proportions; (ii) the soft mask avoids the discontinuity of hard thresholding, which creates artifacts at mask boundaries and destabilizes training gradients; and (iii) the learnable temperature enables the model to control mask sharpness as a function of training progress. During training, we apply a budget regularization loss (Section 3.8) to prevent degenerate solutions where the mask collapses to all-foreground or all-background.





Fig. 3. Qualitative results on VOC2012. Each row shows (a) input image, (b) ground-truth mask, (c) predicted foreground mask, (d) predicted background mask, and (e) foregroundness score heatmap  $F_{i}$ .DAFB-CLS accurately localizes foreground objects while suppressing background activations.

E. Step 4: Dual CLS Decoupling

1) Foreground CLS Aggregation

Given the soft foreground mask  $m^{F}=\left[m_{1}^{F},...,m_{N}^{F}\right]$ , we compute the foreground CLS representation at each extracted layer by weighted averaging of patch features:

$$B_{l}^{F}=\frac{\sum_{i=1}^{N} m_{i}^{F}⋅P_{i}^{l}}{\sum_{i=1}^{N} m_{i}^{F}+ϵ}, B_{l}^{F}∈ℝ^{D}$$

This produces a sequence of foreground CLS tokens  $B^{F}=\left[B_{l_{1}}^{F},...,B_{l_{K}}^{F}\right]∈ℝ^{K×D}$  that represent the foreground content as seen from each extracted layer. Crucially, these foreground CLS tokens maintain the multi-layer diversity of the backbone representations while focusing exclusively on foreground-relevant information.

2) Background CLS Aggregation

Rather than simply using  $m^{B}=1−m^{F}$  as the background mask—which would make the background stream mechanically dependent on the foreground stream—we introduce a dedicated BackgroundnessHead that independently predicts a background mask from the final-layer patch features. This head consists of a two-layer MLP that outputs a background score, combined with an uncertainty predictor that estimates the reliability of each patch’s backgroundness prediction:

$$s_{i}^{B}=σ\left(MLP_{B}\left(P_{i}^{l_{K}}\right)\right), u_{i}=σ\left(MLP_{U}\left(P_{i}^{l_{K}}\right)\right), m_{i}^{B}=s_{i}^{B}⋅\left(1−m_{i}^{F}\right)⋅u_{i}$$

The multiplicative composition ensures that: (i) the background mask excludes high-confidence foreground patches via the  $\left(1−m_{i}^{F}\right)$  term; (ii) the uncertainty term  $u_{i}$  prevents ambiguous boundary patches from being confidently assigned to either stream; and (iii) the independent score  $s_{i}^{B}$  allows the background stream to develop its own semantic preferences rather than being a passive complement of the foreground. The background CLS tokens  $B^{B}=\left[B_{l_{1}}^{B},...,B_{l_{K}}^{B}\right]$  are then computed analogously:

$$B_{l}^{B}=\frac{\sum_{i=1}^{N} m_{i}^{B}⋅P_{i}^{l}}{\sum_{i=1}^{N} m_{i}^{B}+ϵ}$$

3) Why Dual Decoupling Rather Than Background Suppression

A natural question is why we maintain a separate background stream rather than simply suppressing background information. Three reasons motivate this design. First, background context provides complementary semantic information: scene type, spatial layout, and co-occurrence statistics that are valuable for classification and may even aid localization. Second, completely suppressing background forces all contextual information through the foreground stream, which may dilute the foreground signal—analogous to the information dilution problem in standard residual connections. Third, maintaining separate streams enables the task-adaptive fusion (Section 3.7) to learn how much background context each task needs, which varies substantially: object discovery benefits from minimal background, while classification may benefit from scene context.

F. Step 5: Depth-Wise Attention

After obtaining the dual CLS sequences  $B^{F}$  and  $B^{B}$ , we perform independent depth-wise attention over the layer dimension for each stream. This mechanism directly adapts the Attention Residuals concept from language models to the visual domain, enabling each stream to selectively weight the contributions of different layers.

For the foreground stream, we compute:

$$β_{l}^{F}=\frac{exp\left(w_{F}^{T}⋅RMSNorm\left(B_{l}^{F}\right)\right)}{\sum_{j=1}^{K} exp\left(w_{F}^{T}⋅RMSNorm\left(B_{j}^{F}\right)\right)}, C_{F}=\sum_{l=1}^{K} β_{l}^{F}⋅B_{l}^{F}$$

where  $w_{F}∈ℝ^{D}$  is a learned pseudo-query vector initialized to zero, ensuring uniform initial attention weights. RMSNorm is applied to the keys (block features) to prevent layers with large activation magnitudes from dominating the attention weights—a key design choice validated by the Attention Residuals ablation study. The background stream uses an analogous formulation with its own pseudo-query  $w_{B}$ , producing  $C_{B}$  and  $β^{B}$ .

The zero initialization of pseudo-queries is crucial: at the start of training, all layers contribute equally ( $β_{l}=1/K$ ), so the initial behavior matches standard average pooling across layers. As training progresses, the model learns to emphasize layers that are most informative for each stream—typically later layers for semantic understanding and earlier layers for spatial precision.



Fig. 2. Learned depth attention weights  $β_{l}^{F}$  and  $β_{l}^{B}$  averaged over VOC validation images. Foreground stream preferentially attends to later layers (L9–L12) encoding semantic features, while the background stream distributes more uniformly across depth. Error bars indicate standard deviation.

G. Step 6: Task-Adaptive Fusion

The final image representation is produced by adaptively fusing the foreground and background streams. We introduce a learnable gate  $g∈\left[0,1\right]$  that controls the relative contribution of each stream:

$$g=σ\left(MLP_{g}\left(\left[C_{F};C_{B};c_{cls}\right]\right)\right)$$

$$C=g⋅C_{F}+\left(1−g\right)⋅C_{B}$$

where  $\left[⋅;⋅;⋅\right]$  denotes concatenation and  $MLP_{g}$  is a two-layer MLP mapping from  $3D$  to 1. The gate takes all three representations as input—the foreground CLS  $C_{F}$ , the background CLS  $C_{B}$ , and the global backbone feature  $c_{cls}$ —allowing it to make an informed decision about how much foreground vs. background information the task requires. For example, object discovery tasks (where objects are the primary target) may learn  $g→1$ , while classification tasks (where scene context matters) may learn a more balanced gate value.

The fused representation  $C$  is then passed to the appropriate task head:

Classification: A two-layer MLP maps  $C$  to class logits. Segmentation: Patch features from the final layer are projected to the text embedding space (for OpenCLIP) or the CLS projection space, producing per-patch similarity scores that are reshaped to spatial maps. A learnable background bias based on  $\left(1−m^{F}\right)$  is added to the background class logits. Object Discovery: A scoring MLP produces per-patch scores, which are combined with the foreground mask and cosine similarity to  $C$  for final score map prediction.

H. Training Objective

The total training loss combines a task-specific primary loss with three auxiliary losses that regularize the aggregation module:

$$ℒ=ℒ_{task}+λ_{fg}ℒ_{mask}+λ_{dec}ℒ_{decouple}+λ_{bud}ℒ_{budget}$$

1) Task Loss

For object discovery, we supervise the foreground mask using pseudo-ground-truth masks derived from the initial foregroundness scores. Patches with scores exceeding one standard deviation above the mean are labeled as foreground, generating binary pseudo-masks that provide the training signal for the aggregation module. The task loss combines binary cross-entropy and Dice loss for robust segmentation supervision:

$$ℒ_{mask}=BCE\left(m^{F},m\right)+\left(1−Dice\left(m^{F},m\right)\right)$$

For segmentation, standard cross-entropy loss between predicted segmentation logits and ground-truth masks is used. For classification, cross-entropy on class logits applies.

2) Decoupling Loss

To encourage the foreground and background streams to capture complementary rather than redundant information, we minimize the squared cosine similarity between  $C_{F}$  and  $C_{B}$ :

$$ℒ_{decouple}=\left(\frac{C_{F}⋅C_{B}}{∥C_{F}∥⋅∥C_{B}∥}\right)^{2}$$

This loss pushes the two streams toward orthogonality in the feature space, ensuring that each stream encodes distinct semantic information. We use the squared formulation rather than the raw cosine similarity because the squared loss provides stronger gradients when the streams are nearly orthogonal (near zero), preventing the loss from becoming ineffective late in training. The loss weight is  $λ_{dec}=0.05$ .

3) Budget Regularization

Without regularization, the adaptive budget may converge to degenerate solutions: all-foreground (where every patch receives high mask weight, negating the selection benefit) or all-background (where the foreground stream receives no information). We prevent these failure modes through a range penalty on the average foreground ratio  $r=\frac{1}{N}\sum_{i}^{​} m_{i}^{F}$ :

$$ℒ_{budget}=max\left(0,r_{min}−r\right)^{2}+max\left(0,r−r_{max}\right)^{2}$$

where  $r_{min}=0.1$  and  $r_{max}=0.7$  are the lower and upper bounds on the acceptable foreground ratio. These bounds are chosen based on empirical analysis: typical object bounding boxes in VOC and COCO cover 10–60% of the image area, and allowing up to 70% accommodates multi-object scenes while the 10% lower bound prevents mask collapse. The loss weight is  $λ_{bud}=0.01$ .

I. Complexity Analysis

The DAFB-CLS aggregation module is designed to be lightweight and efficient. For a standard ViT-S/16 with  $D=384$ ,  $N=196$ , and  $K=4$  extracted layers:

Feature extraction: The multi-layer hook mechanism adds negligible overhead (only intermediate tensor references), requiring no additional forward passes through the backbone. Foregroundness scoring: Four cue computations are each O(ND) or O(N), with the MLP fusion adding  $O\left(N⋅H\right)$  where  $H=256$  is the hidden dimension. Adaptive budget: The threshold predictor MLP is  $O\left(D⋅H+H\right)$ , shared across all patches. Dual CLS aggregation: Two weighted averages over K layers, each  $O\left(KND\right)$ . Depth-wise attention: Two softmax attention computations with pseudo-queries, each  $O\left(KD\right)$ . Fusion: One MLP with input dimension  $3D$ , output dimension 1.

Total trainable parameters: 0.59M (DINO ViT-S/16) or 1.74M (OpenCLIP ViT-B/16), representing 2.7% and 2.0% of the respective frozen backbones. Inference latency overhead: 1.82ms (DINO) and 3.45ms (OpenCLIP), both under 31% relative overhead. Peak GPU memory scales linearly with the number of extracted layers due to storing multi-layer features for aggregation, resulting in approximately 2.8× (DINO) and 3.0× (OpenCLIP) overhead relative to the baseline ViT forward pass.



IV. Experiments

We evaluate DAFB-CLS on unsupervised object discovery and open-vocabulary segmentation across two benchmarks. Comprehensive ablation studies validate each design component, and efficiency analysis confirms the practical overhead is modest.

A. Experimental Setup

Datasets. We use two standard benchmarks for evaluation: (i) PASCAL VOC 2012, containing 1,449 validation images across 20 object categories, widely used for object discovery and weakly supervised segmentation; (ii) MS COCO 2017, containing 4,952 validation images across 80 categories, providing a larger-scale and more diverse testbed. All images are resized to 224×224 and processed through the frozen backbone without any additional data augmentation during evaluation.

Metrics. We adopt four complementary metrics:

CorLoc: the percentage of images where the highest-scoring patch falls within a ground-truth bounding box, measuring whether the aggregated CLS representation correctly localizes the primary object.

Mask IoU: the intersection-over-union between the predicted soft foreground mask and ground-truth segmentation masks, averaged over all images, measuring spatial mask quality.

PiB: the percentage of images where the patch with the highest Patch Score lies within an annotated foreground bounding box, providing a finer-grained localization diagnostic.

mIoU: mean intersection-over-union across all classes, used for the open-vocabulary segmentation task with OpenCLIP.

Comparison Methods. We compare against four representative baselines: (i) CAM: class activation mapping from the frozen backbone, serving as the standard weakly-supervised localization baseline; (ii) DINO-seg: self-attention-based unsupervised segmentation from DINO; (iii) LaSt-ViT: frequency-stability-based CLS aggregation with fixed Top-K selection (K=59 for VOC, K=147 for COCO, ~30% of patches); (iv) DAFB-CLS (ours): the full framework with multi-cue foregroundness, adaptive budget, dual CLS decoupling, and independent depth-wise attention.

Implementation Details. For DINO experiments, we use ViT-S/16 as the frozen backbone with features extracted at layers {3, 6, 9, 12}. The aggregation module is trained for 50 epochs with batch size 16, learning rate 10⁻⁴, cosine scheduler, and mixed-precision training. Loss weights are λ_fg = 1.0, λ_dec = 0.05, λ_bud = 0.01. For OpenCLIP experiments, we use ViT-B/16 pretrained on LAION-2B with the same layer indices and analogous hyperparameters. All experiments run on a single NVIDIA RTX 4070 GPU.

B. Main Results

Object Discovery on VOC 2012. Table II compares DAFB-CLS against baselines on the VOC validation set using DINO ViT-S/16. The raw ViT baseline achieves only 29.33% CorLoc, confirming that the default CLS token poorly localizes objects. DINO-seg improves CorLoc to 68.46% via self-attention maps but produces noisy masks (8.83% Mask IoU). LaSt-ViT introduces frequency-guided selection, raising Mask IoU to 13.31%, but its fixed Top-K budget limits spatial precision. DAFB-CLS achieves 79.43% CorLoc (+13.7 pp over LaSt-ViT) and 31.27% Mask IoU ( +18.0 pp), representing a +50 pp improvement in CorLoc over the raw ViT baseline. The substantial Mask IoU gain demonstrates that the adaptive soft mask produces significantly more spatially accurate foreground regions than fixed-ratio selection.

TABLE II

Object Discovery Results on VOC 2012 (DINO ViT-S/16)


[TABLE]


Object Discovery on COCO. To assess cross-dataset generalization, we evaluate on COCO 2017 (80 categories) without retraining. Table III shows that DAFB-CLS maintains consistent improvements: CorLoc reaches 72.96% (+11.4 pp over LaSt-ViT) and Mask IoU reaches 28.73% (+15.8 pp). The gains are slightly smaller than on VOC, which is expected given the larger category diversity, but the trends are fully consistent. Notably, Mask IoU improvement remains remarkably stable across datasets (+18.0 pp on VOC vs. +15.8 pp on COCO), validating that the adaptive soft mask generalizes across varying object scales and category distributions.

TABLE III

Object Discovery Results on COCO 2017 (DINO ViT-S/16)


[TABLE]


C. Ablation Studies

To isolate the contribution of each component, we conduct systematic ablation experiments on DINO ViT-S/16 + VOC. Table IV presents the results of six ablation variants, each removing or replacing a single design choice from the full DAFB-CLS framework.

TABLE IV

Ablation Study on VOC 2012 (DINO ViT-S/16)


[TABLE]


Multi-cue foregroundness is essential. Removing all four cues collapses CorLoc from 79.43% to 47.48% (Table IV), confirming that the foregroundness score provides the foundational signal for spatial selection. Without cues, the model has no mechanism to distinguish foreground from background patches.

Adaptive budget prevents masking collapse. Replacing the adaptive threshold with fixed Top-K at 30% (Hard Top-K) achieves competitive CorLoc (81.71%) but degrades Mask IoU from 31.27% to 22.68%. Similarly, removing the budget entirely (w/o Adaptive Budget) preserves CorLoc (79.78%) but Mask IoU drops sharply to 13.91%, as the mask collapses without the budget regularization loss. These results confirm that the image-adaptive soft threshold is critical for mask quality.

Dual CLS decoupling improves spatial localization. Removing the background stream (w/o Dual CLS) drops PiB from 45.07% to 33.13%, indicating that maintaining a separate background representation helps the foreground stream focus on object regions.

Independent depth attention benefits each stream. Sharing a single depth attention mechanism across both streams (Shared Depth) reduces PiB from 45.07% to 28.36%. This confirms that the foreground and background streams require different depth-wise aggregation patterns.



Fig. 4. Ablation comparison on VOC 2012 (DINO ViT-S/16). Each variant removes or replaces a single component from the full DAFB-CLS framework. Multi-cue foregroundness contributes most to CorLoc, while adaptive budget and independent depth attention are critical for Mask IoU and spatial localization (PiB), respectively.

D. Efficiency Analysis

Table V reports the parameter count, inference latency, and GPU memory overhead of DAFB-CLS compared to baselines. All measurements use batch size 1 and 224×224 input on an NVIDIA RTX 4070 GPU.

TABLE V

Efficiency Comparison


[TABLE]


DAFB-CLS introduces only 0.59M trainable parameters for DINO ViT-S/16 (2.7% of the frozen backbone) and 1.74M for OpenCLIP ViT-B/16 (2.0%). The inference latency overhead is 1.82 ms (DINO) and 3.45 ms (OpenCLIP), corresponding to 1.31× and 1.28× relative overhead, respectively. These results confirm that DAFB-CLS is lightweight and practical for deployment alongside frozen foundation models.

