# Evidence-Weighted Architecture Strategy for RSNA Knee Abnormality Detection

## Bottom line

Your error pattern is much more consistent with an **anatomical attribution bottleneck** than with a generic representation-capacity bottleneck. The strongest evidence is internal: medial meniscus versus lateral meniscus differs by 0.089 AUC, medial versus lateral OA by 0.172, and simply widening slice coverage gave lateral meniscus +0.039 despite no additional in-plane information. That makes “better global backbone” a lower-priority direction than forcing the network to represent **where a feature came from**.

My highest-confidence recommendation is therefore not another sequence model. It is a **canonicalize → extract named anatomical regions → classify with explicitly compartment-conditioned readouts** pipeline, while keeping your existing ImageNet-pretrained ConvNeXt/CoAtNet members. The localization need not be segmentation-quality: a handful of stable landmarks or coarse boxes is sufficient. This closely matches the architectural pattern of successful RSNA 2024 lumbar-spine solutions, where keypoints were used to define local 3D ROIs before classification rather than asking an end-to-end study model to discover both anatomy and disease simultaneously. One first-party 7th-place solution used sagittal 2D/3D keypoint models, axial 3D keypoints/level classifiers, and then cropped local ROIs around predicted XYZ positions before multi-view classification. citeturn7view1 Its repository is MIT-licensed, so the code structure is permissible as a reference, though your constraint correctly rules out its trained weights. fileciteturn3file0L1-L10

The second important conclusion is that **symmetry is unusually well matched to your particular failure mode**. Bilateral mammography work provides much more direct experimental support than generic “anatomy-aware attention”: symmetry-based Siamese pretraining raised multiclass AUC from 0.760 with BYOL to 0.784 on VinDr-256 and from 0.679 to 0.744 on OPTIMAM-256, while a 2026 breast-MRI preprint reports a 7.19-point Macro-AUC advantage over its strongest baseline in-distribution and 8.20 points out-of-distribution; ablation of its bilateral branch reduced QWK by 14.56 and 13.40 points, respectively. The latter is not yet peer-reviewed and is breast rather than knee imaging, so I would treat the magnitude as evidence for the mechanism, not as a knee-AUC forecast. citeturn21view0turn21view1turn21view2turn21view3

The third conclusion is that the best-supported small-lesion strategy is **coarse-to-fine spatial sampling inside the image**, not selecting entire slices. GMIC's weakly supervised mammography architecture took a full-resolution global image, generated a saliency map, selected six 256×256 patches, and classified the local evidence. Its localization branch reached malignant AUC 0.885 versus 0.827 for a full-image ResNet-22, a **+0.058 AUC** difference; benign AUC rose from 0.731 to 0.777, **+0.046**. A local-MIL variant reached 0.878 malignant AUC, while random patch selection collapsed to 0.757. citeturn12view0turn12view1 That experiment is importantly different from your failed “useful-slice” selector: the proposed selection happens **within already anatomically constrained slices**, where the lesion-to-field-of-view ratio is the actual problem.

I would not spend the next compute budget on another cross-slice transformer, generic SSL, a large 3D model, unconstrained deformable attention, or a segmentation-first pipeline. The literature does not offset your own negative evidence strongly enough, and several attractive published localization systems have licensing or hardware problems under your rules. In particular, MedLSAM/MedLAM is methodologically relevant but its repository explicitly states **CC-BY-NC 4.0**, making the released implementation/weights unusable for you. citeturn23search7turn23search12

## Ranked shortlist and expected return

The “expected macro gain” column below is **my estimate for your task**, not a published effect size. I am deliberately shrinking literature effects heavily because your baseline is already 0.929 public LB and several findings are near saturation. The ranking is approximately `(expected macro-AUC gain × confidence) / incremental T4 cost`.

| Rank | Experiment | Expected macro-AUC gain | Main targets | Extra T4 training | Added inference | Confidence |
|---|---|---:|---|---:|---:|---|
| **1** | Few-shot landmarks → canonical compartment crops | **+0.008 to +0.018** | lateral meniscus, lateral OA, PF OA, MCL | ~2–6 GPU-h for locator; classifier similar | ~3–10% | High |
| **2** | Shared encoder + explicit medial/lateral spatial heads + anti-swap auxiliary | **+0.005 to +0.012** | lateral OA, lateral meniscus, MCL | ~0–1 extra GPU-h equivalent | <2% | High-medium |
| **3** | Mirrored medial↔lateral difference branch | **+0.003 to +0.009** | lateral OA, menisci, MCL | ~0–2 extra GPU-h equivalent | 1–5% | Medium-high |
| **4** | Anatomically gated GMIC-style high-resolution patch branch | **+0.004 to +0.010** | fracture, contusion, MCL, menisci, synovitis | ~4–8 GPU-h/member experiment | 15–35% | Medium-high |
| **5** | DICOM-coordinate cross-plane ROI fusion | **+0.003 to +0.008** | menisci, OA, ACL/MCL, PF OA | ~2–5 incremental GPU-h | 5–25% depending patch budget | Medium |
| **6** | Registration/template propagation to bootstrap the locator | **+0.002 to +0.006 indirectly** | same as rank 1 | mostly CPU + ~1–3 GPU-h locator | **0% after locator trained** | Medium-low |
| **7** | OA ordinal/severity auxiliary head | **+0.001 to +0.005 conditional** | lateral OA, PF OA, medial OA | negligible | <1% | Low unless grades exist |

The key point is that the top three can be tested **without increasing the number of study images much at all**. They change the semantics of your pooling rather than brute-force the input resolution.

### Few-shot landmarks feeding canonical compartment crops

**What it is.** Hand-label approximately 50–100 training studies with a very small landmark vocabulary rather than masks: for example tibial-spine/notch center, medial and lateral tibiofemoral joint centers, patellofemoral center, and optionally meniscal-horn anchors in one or two planes. Train a lightweight heatmap/keypoint detector, convert its outputs through DICOM patient coordinates, and make every pathology head consume one or more **named physical ROIs** rather than an unlabeled global stack.

This is the most direct translation of recent RSNA practice into your problem. The first-party RSNA 2024 lumbar-spine solution cited above explicitly localized anatomy with sagittal and axial keypoint systems and then constructed local 3D ROIs around those coordinates before classification. citeturn7view1 Another first-party solution documented an especially relevant implementation hazard: horizontal-flip augmentation initially failed to reorder left/right points correctly, exactly the class of side-attribution bug you are trying to eliminate. citeturn7view0

There is also knee-specific evidence that compartment localization supports strong downstream MRI classification, although it does **not** satisfy your annotation regime directly. Astuto et al. used knee-MRI cartilage compartment segmentation to derive volumetric boxes, followed by 17 3D CNN classifiers for cartilage, bone-marrow edema, meniscus and ACL findings; binary lesion AUCs were roughly **0.83–0.93**. That study demonstrates that compartment-specific knee MRI classification is viable, but because its localization depended on segmentation labels, it is architectural evidence rather than a directly usable recipe. citeturn1search16

For annotation-light localization, SAM—Self-supervised Anatomical eMbedding, not Meta's Segment Anything—provides unusually strong correspondence evidence. With one labeled template it surpassed supervised landmark systems trained on 50 labeled X-rays; on 19 chest-CT landmarks it outperformed conventional registration while requiring about **0.23 seconds** for matching, and it obtained **91% accuracy** on longitudinal lesion matching. Crucially, SAM learns *pixel-wise anatomical correspondence*, not a generic image embedding, so your negative DINO/DINOv2 result does not directly invalidate it. citeturn22academia37 [Paper](https://arxiv.org/abs/2012.02383)

There is, however, **no direct downstream classification-AUC ablation in SAM** showing “localization raises pathology AUC by X.” I found no credible 2021–2026 study matching all of your conditions—knee MRI, no masks, few landmarks, downstream multi-label classification—and reporting that exact ablation. That absence is why I would use ≤100 genuine keypoint annotations rather than betting the project on correspondence SSL.

**How I would map it onto your model.** Keep your present 140-mm global stack as a context path. Add, for each series, a locator operating at 192–256 px and predict heatmaps for perhaps 4–6 landmarks. Transform landmarks into one canonical right-knee coordinate convention; construct 60–90-mm physical ROIs for medial TF, lateral TF and PF compartments. Feed the same ConvNeXt weights over global and ROI images, or cheaper, ROI-align intermediate feature maps.

Do **not** train twelve independent localizers. Anatomy is independent of pathology; one locator should be shared by all findings.

The expected largest improvements are lateral meniscus and lateral OA because those are your strongest evidence of “lesion exists but side is wrong.” PF OA should also benefit because patellofemoral versus tibiofemoral attribution is another spatial decision. MCL may gain from explicit medial-edge coverage. Effusion and Baker's cyst are already strong and should mostly use the global route.

**Cost.** A small ResNet-18/U-Net-like heatmap locator on ~500 series from 100 annotated studies should be comfortably within roughly **2–6 T4 GPU-hours total** for several experiments, not per final ensemble member. At inference, a 256-px locator should add only low-single-digit percentage compute relative to processing 44–64 ConvNeXt/CoAtNet images. These are engineering estimates, not literature timings.

**Failure mode.** The locator can become pathology-sensitive: advanced lateral OA changes joint-space geometry, and a detector trained mostly on normal/mild knees can systematically move the lateral anchor on the very positives you need. Measure landmark error **stratified by finding**, especially lateral OA positives, rather than reporting only overall millimeters.

**Implementation sketch.** Annotate points, not boxes. Use Gaussian heatmap targets; convert coordinates to normalized physical XYZ; aggressively augment rotation/translation/scale but always remap medial/lateral correctly. At inference compute the joint-center geometry once and generate all compartment boxes deterministically.

### Shared encoder with compartment-conditioned readouts and an anti-swap loss

Your present architecture already has twelve finding-specific attention queries, but those queries have no hard reason to associate “lateral OA” with the lateral tibiofemoral compartment. I would change the inductive bias: **finding identity should select pathology features; compartment identity should select spatial support**.

The simplest version requires essentially no extra image computation. From the backbone feature map, create fixed or softly predicted masks \(M_\text{medial}\), \(M_\text{lateral}\), \(M_\text{PF}\), \(M_\text{central}\), etc., derived from the canonical landmarks. Pool lateral-meniscus and lateral-OA representations predominantly from \(M_\text{lateral}\); medial findings use \(M_\text{medial}\). Let a low-weight global residual pathway handle unusual anatomy.

Recent RSNA lumbar-spine solutions supply practical precedent for this “shared representation, anatomy-specific ROI” pattern: the first-party 2024 7th-place implementation localized levels/coordinates first and then extracted center/left/right ROIs for stage-two classification instead of training a separate full image encoder for every level/side. citeturn7view1 Its code is MIT licensed. fileciteturn3file0L1-L10 [Repository](https://github.com/lhwcv/solution-rsna-2024-lumbar-spine)

The piece I would add specifically for your labels is a **joint side-state auxiliary task**. For the menisci, derive from existing labels one of four states:

`none / medial-only / lateral-only / both`

Do the same for tibiofemoral OA. Predict this from the concatenated medial and lateral ROI features. Keep the original independent BCE outputs as the official heads, but add perhaps 0.1–0.3× cross-entropy on the four-state auxiliary target. This makes a medial↔lateral swap a *structured error* rather than two unrelated binary errors.

That auxiliary is my adaptation rather than a published knee ablation, so **no published measured AUC delta exists for exactly this loss**. The evidence supporting it is your own highly asymmetric paired-task performance plus the recurring use of explicitly side/level-specific ROIs in spine competition systems. I would nevertheless rank it second because its GPU cost is almost zero.

The targets are lateral meniscus, lateral OA and MCL. Medial scores should stay approximately neutral; monitor them carefully because an overly hard compartment mask could remove contextual evidence that currently helps the strong medial heads.

**Cost:** effectively the same training time, roughly <5% extra wall time and <2% inference. **Failure mode:** bilateral disease. A mutually exclusive medial/lateral formulation would be wrong because both compartments can be positive; that is why the four-state target contains `both`.

### Mirrored medial-versus-lateral feature comparison

This is the cleanest answer to the symmetry question.

After canonicalization, extract features \(f_M\) and \(f_L\) from corresponding medial/lateral boxes, reflect one side into the other's orientation, and form:

\[
h_M=[f_M,\ f_L,\ f_M-f_L,\ |f_M-f_L|]
\]

\[
h_L=[f_L,\ f_M,\ f_L-f_M,\ |f_L-f_M|].
\]

The classifiers remain side-specific, but the encoder is shared. There is no GAN, no second backbone, and no need for the two compartments to be exactly symmetric voxel-for-voxel.

The best quantitative analogy I found is mammography. Van Vorst and Shen aligned contralateral mammograms, sampled corresponding patches, and trained a Siamese representation to distinguish normal pairs from asymmetric abnormal pairs. For downstream VinDr BI-RADS classification on 256×256 patches, symmetry-based pretraining reached **0.784 multiclass AUC versus 0.760 for BYOL and 0.594 for SimCLR**. On OPTIMAM-256 background/benign/malignant classification, it reached **0.744 versus 0.679 for BYOL and 0.643 for SimCLR**. citeturn20view0turn21view0turn21view1 [Paper](https://arxiv.org/html/2401.05570)

That result is especially relevant to your DINO experience: task-aligned symmetry pretraining beat generic SSL even though both were self-supervised. The authors also explicitly report an important failure mechanism: normal paired regions can differ visually enough to produce false positives, which is why I recommend feeding both absolute features and differences rather than making difference magnitude itself the diagnosis. citeturn20view0

A newer and substantially stronger but **preprint-only** result is PRISM-Net for bilateral breast DCE-MRI. It reported Macro AUC **84.11 ± 2.33%** in-distribution and **68.51 ± 4.54%** out-of-distribution, gains of **+7.19 and +8.20 percentage points** over its strongest baseline; removing the bilateral asymmetry component reduced QWK by **14.56 and 13.40 points**, respectively. citeturn21view3turn22view0 [Paper](https://arxiv.org/html/2607.26799)

The older bilateral counterfactual mammography work was subsequently published in IEEE Transactions on Image Processing and includes explicit bilateral-ablation experiments, further supporting the premise that contralateral information is useful, although I would not import its GAN machinery into your setting. citeturn16view2turn23view0

For knees, your symmetry is **within one knee across the intercondylar region**, not left-knee versus right-knee. That is actually computationally easier: both structures are already present in the same acquisition with the same scanner parameters.

**Expected targets:** lateral OA first, then lateral meniscus, potentially MCL. I would expect little value for effusion, Baker's cyst or ACL.

**Cost:** if you reuse the same backbone feature tensor, virtually zero additional convolutional compute; only feature crops and a small MLP are added. **Failure mode:** advanced bilateral/whole-joint OA makes “other side = normal reference” false. Do not train a loss that forces one side to be normal. Treat the opposite side as a patient-specific reference feature only.

### Anatomically gated coarse-to-fine patch branch

This is the strongest literature-backed answer to the “tiny lesion” portion of the problem.

GMIC processes a downsampled whole mammogram to produce a saliency map, picks **six 256×256 high-resolution patches**, encodes them locally, and aggregates them. The reported ResNet-22 full-image baseline had malignant/benign AUC **0.827/0.731**. The localization branch reached **0.885/0.777**, i.e. **+0.058/+0.046**, while its local MIL branch reached **0.878/0.766**. Random patch selection was much worse at **0.757/0.692**, showing that the gain was not merely extra local pixels. citeturn12view0turn12view1

The key adaptation is to avoid repeating your failed slice selector. I would first pick the anatomical compartment deterministically and only then perform **spatial top-k within that compartment**. For a lateral-meniscus head, for example:

1. use the landmark system to obtain lateral TF ROIs on the relevant sagittal/coronal slices;
2. use the existing coarse classifier feature map to score an 8×8 or 12×12 spatial grid;
3. take the top 2–4 non-overlapping native-resolution locations;
4. crop perhaps 40–60 mm around those locations from the original DICOM pixels;
5. run a shared lightweight local encoder;
6. concatenate global, compartment and local features.

This gives you genuinely higher effective detail without the “336→512 interpolation” mistake. You are reducing the physical field of view while staying near native pixel spacing.

I would target **fracture, bone contusion, MCL, both menisci and synovitis**. Fracture is only 0.874 despite a strong global system and is exactly the kind of small focal abnormality for which high-resolution local evidence can matter. Synovitis is less focal and may not improve.

**Cost estimate:** K=2 spatial patches and a ConvNeXt-Tiny/local shared encoder should add roughly **15–25% inference**; K=4 could add 25–40%. A single development run should fit in approximately **4–8 T4 GPU-hours**, particularly if the global encoder is initialized from your current member. Use K=2 first.

**Failure mode:** label noise can cause the saliency mechanism to chase correlated anatomy rather than pathology. This is why I would never let CAM determine *which compartment* is being classified; CAM/top-k should only decide *where inside the already correct compartment* to zoom.

### DICOM-coordinate multi-plane fusion without a transformer

This is the idea with the strongest mechanistic fit and weakest directly comparable published AUC evidence.

You already have the information required to solve correspondence exactly. For pixel coordinates \((r,c)\), DICOM Image Position Patient, Image Orientation Patient and Pixel Spacing map the pixel into patient-space XYZ. The inverse mapping for every other series tells you which sagittal/coronal/axial slice and in-plane location observes approximately the same tissue. There is no reason to make a transformer rediscover that geometry.

The 2024 RSNA lumbar-spine precedent again matters: a first-party top solution localized coordinates in sagittal/axial series, constructed local 3D ROIs from XYZ points, and fused the three view types at feature level. citeturn7view1 However, its write-up does **not** isolate a “known-geometry fusion versus learned attention” classification delta. I therefore do not claim one.

The most relevant recent 3D-classification experiment I found is the 2025 TomoGraphView preprint, which explicitly generated geometrically defined 2D cross-sections from 3D volumes and aggregated their spatial relationships; averaged across six oncology classification datasets, its omnidirectional slicing raised AUROC from **0.7701 to 0.8154** relative to conventional canonical slicing. It is useful evidence that choosing views from known volume geometry can materially affect downstream classification, but it is not an inter-series knee-MRI correspondence ablation and uses graph aggregation that I would omit in your first experiment. citeturn21search0

So for your exact Q4—**cross-series feature sampling at corresponding physical coordinates with a downstream classification AUC ablation on comparable MRI**—I found **no strong direct evidence** that clears the bar you requested. The literature is dominated by registration, segmentation and generic multi-view fusion rather than your unusually clean DICOM-correspondence setting.

That should actually push the implementation toward simplicity:

\[
F(x)=\operatorname{concat}
\left(
F_\text{sag}(T_\text{sag}(x)),
F_\text{cor}(T_\text{cor}(x)),
F_\text{ax}(T_\text{ax}(x))
\right)
\]

where \(x\) is a physical-space landmark or small lattice of points and each \(T_s\) converts patient-space XYZ into series coordinates. Bilinear/trilinear sampling is enough. Follow it with a two-layer MLP or weighted average, **not another cross-slice transformer**.

For menisci and OA, I would sample a small 3D lattice spanning each tibiofemoral compartment rather than a single point. For ACL use the notch trajectory. For PF OA use the patellofemoral center. Each pathology gets a known physical support volume.

**Cost:** if the aligned patches replace some of your current 44/64 inputs rather than being appended, inference can remain near-neutral. If added, expect roughly 5–25%, depending on how many physical points you sample.

**Failure mode:** series are not guaranteed to be perfectly motion-consistent. Treat DICOM geometry as an initial rigid correspondence and use a ±5–10 mm local pooling neighborhood rather than a single exact voxel.

### Registration or anatomical correspondence only as a bootstrap mechanism

I would not run deformable registration in the Kaggle inference loop. I would use it only to eliminate annotation labor.

SAM's one-template result is the strongest proof of concept for self-supervised anatomical correspondences. citeturn22academia37 SAME++, a later SAM-based registration method, reports **+4.2% to +8.2% Dice** over leading registration approaches across three inter-subject registration tasks involving more than 50 labeled organs, but again reports registration/segmentation quality rather than downstream classification. citeturn22academia39

A practical conservative approach is:

- manually mark 6–10 anatomical landmarks in perhaps 10 pristine studies;
- register those templates to several hundred training studies with rigid/affine plus a mild deformable stage;
- transfer the points;
- manually review/correct 50–100 studies;
- train a cheap landmark CNN from the corrected+pseudo-labeled data;
- throw registration away at inference.

SimpleITK is an appropriate implementation base and its current public repository declares **Apache-2.0**, which meets your whitelist. fileciteturn2file0L1-L10 [Repository](https://github.com/SimpleITK/SimpleITK)

I would **not** use MedLAM/MedLSAM despite the methodological fit. MedLAM localizes arbitrary 3D anatomy from a few annotated templates and was pretrained with self-supervised tasks on 14,012 CT scans, but the project's own README states that it is **CC-BY-NC 4.0**. That violates your permissive-only condition. citeturn22academia40turn23search7 [Repository](https://github.com/openmedlab/MedLSAM)

Also note that SAM/SAME++ were demonstrated primarily on CT/X-ray correspondence rather than heterogeneous knee-MRI sequences. Fluid-sensitive versus non-fluid-sensitive MRI changes local appearance much more than same-modality CT registration. That is why this ranks sixth rather than first.

### OA severity modeling only if you can recover real ordinal targets

There is evidence that OA is better treated as ordered severity than as unrelated categorical classes. In a knee radiograph study explicitly comparing classification with regression on Kellgren–Lawrence grades, the authors report that regression outperformed classification; their two-stage detector-plus-regressor was trained on OAI, and after domain-specific fine-tuning the external-set MAE dropped from **1.09 to 0.28**. citeturn23view5turn23view6 [Paper](https://arxiv.org/html/2004.08572)

More recent work continues this direction. The 2026 **ClinNet preprint** combines bilateral-asymmetry features with evidential ordinal regression and reports **QWK 0.892 and accuracy 0.768** for KL grading, with a stated significant advantage over its comparison baselines. It is a preprint rather than peer-reviewed evidence and the accessible abstract does not give an isolated “ordinal head versus cross-entropy = X” effect, so it should not drive your next experiment by itself. citeturn21academia13

The important constraint is label availability. If your training target literally contains only `OA = 0/1`, you **cannot train a genuine ordinal model merely because the disease is biologically continuous**. An ordinal loss requires ordered supervision.

If your original report-extraction workflow can recover phrases such as no/mild/moderate/severe OA, joint-space narrowing or compartment-specific chondrosis from the **training** reports, then an auxiliary cumulative target is worth trying:

\[
P(y \ge 1),\ P(y \ge 2),\ P(y \ge 3)
\]

with a monotonic/cumulative-link loss, while retaining the competition's binary OA BCE head. At inference the ordinal scalar becomes another feature for the binary logit; no reports are required.

If those severity labels cannot be recovered, I would move this out of the active queue. Synthetic grades based on the current model's confidence would mostly distill its existing errors.

The OAI data used in much of this literature are also irrelevant as a dependency because you have already ruled OAI out. The architecture/loss concept can be implemented entirely on your own data; **do not import OAI images or OAI-trained weights**.

## What the evidence says for each research question

| Question | Evidence-weighted answer |
|---|---|
| **Localize then classify without masks** | **Yes, but favor points/boxes over masks.** SAM shows one-template anatomical correspondence can exceed supervised landmark models trained with 50 images; RSNA 2024 solutions show keypoint→ROI pipelines are competition-proven. GMIC gives the clearest *downstream* classification ablation for weak localization: +0.058 malignant AUC over its full-image baseline. SAM/registration papers generally do **not** report downstream classification gains. citeturn22academia37turn7view1turn12view1 |
| **Side / compartment attribution** | The most credible strategy for you is **one shared encoder plus explicit compartment-conditioned spatial support**, not separate complete networks. Recent spine competition pipelines localize levels/sides and classify ROIs; no clean paper was found showing that duplicating the entire encoder per side beats a shared anatomy-conditioned encoder under ~4k-study conditions. citeturn7view1 |
| **Symmetry** | **Yes; this has real downstream evidence.** Bilateral Siamese pretraining beat BYOL by +0.024 AUC on VinDr-256 BI-RADS and +0.065 on OPTIMAM-256; PRISM-Net's full bilateral model beat its strongest baseline by +7.19/+8.20 Macro-AUC points ID/OOD, though PRISM is a 2026 preprint. citeturn21view0turn21view1turn21view3 |
| **Multi-plane geometric fusion** | Mechanistically compelling, but **direct evidence for your exact formulation is sparse**. RSNA spine solutions use XYZ-localized multi-view ROIs, and geometric slicing improves 3D classification elsewhere, but I found no convincing knee-MRI paper isolating DICOM-coordinate cross-view sampling with a clean AUC delta. citeturn7view1turn21search0 |
| **Fine-grained / small lesion** | **Coarse-to-fine high-resolution local patches have the best direct evidence.** GMIC showed +0.058 malignant and +0.046 benign AUC for its localization pathway versus a full-image baseline. I found no comparably convincing reason to put HRNet, deformable attention or unconstrained token pruning ahead of it for your T4/data regime. citeturn12view0turn12view1 |
| **Graded severity** | Ordinal/regression formulations make sense only with actual severity supervision. Knee-OA regression has outperformed classification in published experiments, and recent ordinal systems report high QWK, but your binary labels make this a conditional rather than primary recommendation. citeturn23view5turn21academia13 |
| **What not to bother with** | Generic SSL, another cross-slice transformer, heavy segmentation-first systems, giant 3D models, separate full encoders per compartment, and unlabeled “severity” regression all have a poor evidence/cost match here. Your own experiments already eliminate several of these more strongly than external literature could. |

One nuance on **equivariant keypoints**: I found papers demonstrating equivariant/self-supervised keypoint discovery and registration accuracy, but **no convincing recent medical-imaging study with a downstream pathology-classification ablation comparable to yours**. I would therefore not allocate a precious training week to unsupervised keypoint discovery before testing 50–100 manually supervised landmark studies. At your annotation budget, simple supervised heatmaps are a much lower-risk experiment.

Likewise, I found **no good measured case for a completely separate medial and lateral CNN**. That approach throws away your strongest statistical asset—the fact that medial and lateral structures share appearance and pathology semantics—and approximately doubles convolutional cost. Your problem is not that lateral OA needs an unrelated visual vocabulary; it is that a shared vocabulary needs an unambiguous spatial address.

## Licensing, code and data audit

“Verified” below means I found the license explicitly in a current first-party repository/source during this research. For anything ambiguous, I have treated it as unusable rather than assuming permissiveness.

| Resource | Role | License/status found | Verified? | Usable under your rules? |
|---|---|---|---:|---:|
| **SimpleITK** | Rigid/affine/deformable registration; geometry utilities | **Apache-2.0** in current GitHub metadata. fileciteturn2file0L1-L10 | **Yes** | **Yes** |
| **lhwcv RSNA 2024 lumbar-spine solution** | Reference keypoint→ROI→multi-view competition architecture | **MIT** in repository metadata. fileciteturn3file0L1-L10 | **Yes** | **Code yes; competitor weights no** |
| **hengck23 RSNA 2024 solution** | Reference for shape alignment, level/side handling and augmentation | MIT reported by the first-party repository; solution also documents left/right flip-point remapping issues. citeturn7view0 | Yes from source inspected | **Code concepts yes; weights no** |
| **MedLSAM / MedLAM** | Few-template 3D localization | README explicitly says **CC-BY-NC 4.0**. citeturn23search7turn23search12 | **Yes** | **No — hard violation** |
| **SAM anatomical embedding method** | Self-supervised pixel-wise anatomical correspondence | Paper/method available; I did **not** verify a permissively licensed official implementation/weights suitable for your use during this search. citeturn22academia37 | **No for software/weights** | **Reimplement method only** |
| **SAME++** | SAM-enhanced registration/correspondence | Public code is referenced by the paper, but I did not verify its repository license in the retrieved primary material. citeturn22academia39 | **No** | **Do not copy code/weights until verified; concept usable** |
| **GMIC** | Weakly supervised global→top-k local patches | Primary implementation exists, but I did not obtain a sufficiently reliable license verification in this pass. The paper itself supplies enough detail for a clean reimplementation. citeturn11search0turn12view0 | **No for code license** | **Reimplement architecture idea; don't copy unverified code** |
| **PRISM-Net** | Bilateral/asymmetry-aware feature matching | 2026 preprint; no required weights/code for the simple difference-branch adaptation proposed here. citeturn20view1 | No software dependency | **Method idea yes; external weights unnecessary** |
| **VinDr-Mammo** | Evidence dataset for symmetry study | Used only as published evidence here. citeturn20view0 | Not audited against your whitelist | **Not needed** |
| **OPTIMAM** | Evidence dataset for symmetry study | Restricted-access mammography resource in the cited study. citeturn20view0 | Not permissive dependency | **No / not needed** |
| **OAI** | OA grading evidence dataset | Used in the regression-vs-classification study; you have already explicitly rejected it. citeturn23view5 | N/A | **No — user constraint** |
| **Astuto knee-MRI segmentation/classification pipeline** | Architectural precedent | Requires segmentation supervision rather than being a reusable dependency. citeturn1search16 | N/A | **Concept only; annotation recipe violates constraint** |

A special warning on MedLSAM is warranted. GitHub's page metadata can misleadingly surface an Apache license indicator in some views, while the project's own README explicitly states **CC-BY-NC 4.0**. Under your “permissive only” policy, the conservative decision is unequivocally **do not use its code or weights**. citeturn23search7turn23search12

## Do not pursue

**Another generic self-supervised pretraining project.** Your own DINO-on-knee experiment already lost to scratch on the leaderboard, and frozen DINOv2 organized slices by scanner protocol rather than pathology. The most interesting external result actually reinforces that conclusion: on 256-pixel mammography patches, anatomy-specific bilateral Siamese learning produced multiclass AUC **0.784 versus BYOL 0.760**, and **0.744 versus BYOL 0.679** on a second dataset. The lesson is not “try a different SSL acronym”; it is that the pretext has to encode the anatomical relation you care about. citeturn21view0turn21view1

**A third attempt at learned cross-slice attention.** You already have two strong negative experiments—cross-slice transformer 0.801 versus 0.845 and attention-selected slices 0.827 versus 0.845. Known DICOM geometry gives you a cheaper, lower-variance supervisory signal than asking ~4,000 noisy studies to learn 3D correspondence. The appropriate experiment is deterministic XYZ correspondence plus simple pooling/concatenation, not a more elaborate transformer.

**Full cartilage/meniscus segmentation before classification.** Knee-specific work demonstrates that this can work, with downstream lesion AUCs around 0.83–0.93, but that pipeline used explicit compartment segmentation and therefore solves a different supervision problem. citeturn1search16 At ≤100 annotated studies, the useful target is 4–8 points or coarse boxes, not pixel-perfect cartilage masks.

**MedLAM/MedLSAM despite its attractive few-shot localization results.** It is nearly tailor-made for Q1 conceptually—few template scans, anatomical localization, self-supervised training—but its released implementation is CC-BY-NC, outside your whitelist. citeturn22academia40turn23search7 Do not let a good paper create a licensing exception.

**A large 3D CNN/ViT over the complete 178-slice study.** The memory/compute scaling is poor relative to the information you need, and the small/noisy dataset makes a large end-to-end 3D representation statistically unattractive. The 2022 RSNA cervical-spine competition ecosystem includes high-ranking implementations built around substantially larger-memory hardware; one first-party third-place repository states a requirement of at least roughly **40 GB GPU memory**, already outside your 16-GB T4 constraint before ensembling. citeturn6search1 More importantly, your 2.5D system already extracts good pathology features; localization is the missing variable.

**Separate full CNNs for medial and lateral compartments.** I found no convincing evidence that the doubled convolutional cost is justified here. Share the image encoder and separate the *spatial support and prediction heads*. That preserves statistical sharing between homologous structures while preventing the rare lateral labels from having to learn their visual vocabulary from scratch.

**Unconstrained CAM pseudo-box self-training.** GMIC proves that weakly supervised saliency can be useful for selecting local patches, but it also shows random/localization design choices matter greatly; random patch selection dropped malignant AUC to **0.757** compared with **0.885** for its localization route. citeturn12view1 In your setting the image-level labels themselves are noisy and the central concern is side attribution, so a CAM is not trustworthy enough to decide “medial or lateral.” Use anatomy to choose the compartment, then CAM/top-k only within it.

**Pure top-k token pruning or deformable attention as the first small-lesion experiment.** These mechanisms can save computation in large transformers, but tiny lesions are exactly the tokens an early scorer can discard. You have already observed the analogous failure at slice level. The GMIC-style alternative retains a global route and uses top-k only to add high-resolution local evidence, which is safer. citeturn12view0turn12view1

**HRNet solely for preserving feature-map resolution.** Nothing about your evidence suggests that another high-resolution backbone is the highest-value intervention; the input is already approximately native-resolution and widening *coverage*, not interpolating resolution, moved lateral meniscus. Local physical crops can increase effective lesion scale without carrying high-resolution features for the entire 140-mm knee through every layer.

**Ordinal OA without ordinal labels.** The clinical premise is right and regression/ordinal OA work is encouraging, but a scalar severity latent trained only against binary labels is not magically ordinal. citeturn23view5turn21academia13 Unless you can recover severity from the training-report extraction, spend the compute on localization first.

## Recommended experimental sequence

The most informative first week is a **cheap attribution experiment before building any sophisticated locator**. On your existing fixed stacks, manually define approximate medial/lateral/PF feature-map regions using normalized coordinates after standardizing all knees to the same orientation. Replace unrestricted attention pooling for the five spatially specific weak labels with restricted or strongly biased compartment pooling; add the four-state meniscus/OA auxiliary target. This should cost essentially one normal training run. If lateral OA or lateral meniscus moves materially while medial performance is preserved, you have experimentally confirmed the attribution hypothesis without spending annotation effort.

The next experiment should be the **50–100-study landmark set**. It is worth manually labeling more than one landmark because the desired output is a coordinate system, not merely a knee center. I would annotate approximately 6 landmarks on 60 studies first, deliberately stratified across normal, medial OA, lateral OA, meniscal tears and severe deformation. A preliminary locator from 60 studies should be enough to tell whether you are in the ~3–5 mm or ~15–20 mm error regime. The literature shows that one/few-template correspondence can work surprisingly well, but given your permitted annotation budget, explicit supervision is cheaper than debugging an elaborate SSL localizer. citeturn22academia37

Then test **symmetry as a head-only ablation**. Do not pretrain anything. Extract medial and lateral features from the new ROIs and add the difference/absolute-difference features to the two meniscus and two OA heads. This costs almost nothing and has unusually good external mechanistic support. citeturn21view0turn21view1turn21view3

Only after those three should you spend material GPU time on **local high-resolution patches**. Start with lateral meniscus and fracture as two opposite tests: one finding where compartment attribution appears dominant and one where lesion size/local visibility is more plausible. K=2 local patches is enough to establish whether GMIC-style zoom contributes complementary signal. The mammography ablation suggests the architecture can generate large classification effects when lesions occupy a tiny fraction of a large image. citeturn12view1

Finally, add **cross-plane patient-coordinate fusion around the exact same anatomical ROIs**. This experiment should be deliberately transformer-free. Compare three conditions under identical slice/image budgets: best single plane; naive rank/logit fusion of planes; physical-coordinate matched ROI-feature concatenation. A win in the third condition would answer Q4 much more cleanly than another learned sequence module.

Given your 30 GPU-hour weekly budget, a realistic allocation is roughly **one normal classifier ablation + one landmark-localizer experiment + one symmetry/head experiment per week**, rather than one large architectural gamble. The biggest likely payoff is not improving every one of the twelve classes: it is lifting the three pathological outliers—**lateral OA 0.796, synovitis 0.795, PF OA 0.831**, plus lateral meniscus 0.856—without disturbing the six already near 0.95. On an equal-weight 12-label macro metric, raising lateral OA by 0.06, lateral meniscus by 0.035 and PF OA by 0.025 alone is worth roughly **+0.010 macro AUC** before any gain on MCL or synovitis. That is large enough to explain a meaningful fraction of the present 0.929-to-0.957 leaderboard gap, and it attacks the failure mode for which your own measurements provide the strongest causal evidence.