# FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models
![main](https://github.com/FairMedFM/FairMedFM/blob/v1/figs/main.png)

# Abstract
The advent of foundation models (FMs) in healthcare offers unprecedented opportunities to enhance medical diagnostics through automated classification and segmentation tasks. However, these models also raise significant concerns about their fairness, especially when applied to diverse and underrepresented populations in healthcare applications. Currently, there is a lack of comprehensive benchmarks, standardized pipelines, and easily adaptable libraries to evaluate and understand the fairness performance of FMs in medical imaging, leading to considerable challenges in formulating and implementing solutions that ensure equitable outcomes across diverse patient populations. To fill this gap, we introduce FairMedFM, a fairness benchmark for FM research in medical imaging. FairMedFM integrates with 17 popular medical imaging datasets, encompassing different modalities, dimensionalities, and sensitive attributes. It explores 20 widely used FMs, with various usages such as zero-shot learning, linear probing, parameter-efficient fine-tuning, and prompting in various downstream tasks -- classification and segmentation. Our exhaustive analysis evaluates the fairness performance over different evaluation metrics from multiple perspectives, revealing the existence of bias, varied utility-fairness trade-offs on different FMs, consistent disparities on the same datasets regardless FMs, and limited effectiveness of existing unfairness mitigation methods. Furthermore, FairMedFM provides an open-sourced codebase at ~\url{https://github.com/FairMedFM/FairMedFM}, supporting extendible functionalities and applications and inclusive for studies on FMs in medical imaging over the long term.


# Checklist
[] Realease of the official code.

[] Release of the dataset.

# Citation
TO Appear.

## Structure

- models: forward return the visual embeddings
  - CLIP
  - MedCLIP
  - DINOv2
  - etc
- usages: warppers over models for different usages
  - LPWarpper: forward() returns the logits (to be activated using softmax)
  - CLIPWarpper: forward() returns the cos similarity over image and text features
  - LoRAWarpper: similar to LPWarpper, but the model is warpped using LoRA
  - TBD: segmentation usages
- trainers: for training, evaluation, model saving, etc. Different debias methods to be implemented here. I haven't fully prepared this part, so just have a rough look at the structure, don't need to implement new based on the current version.
  - BaseTrainer: a base trainer for CLS and SEG
  - CLSTrainer: a base trainer for classification, equivalent to ERM on classification
  - TBD: different debias method, segmentation training logic 

main: I haven't fully prepared this part, so just have a rough look at the structure, don't need to implement new based on the current version.

## Notes

- I'm not an expert on the system design/organization or something like that. I'm just organizing the codes Ruinan and I used, with some modifications for the integration of segmentation. So please feel free to edit or propose something new if required.
- I haven't test the codes, and I'm still working on refining the code. So there must be some bugs. This should be kind of enough for appendix writing before this deadline, but of course not enough for releasing into real applications.
- I'm still work on it (especially the trainers and main), if you want add things in, you can edit datasets, models, usages parts, since there are more ready.
