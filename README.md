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
