# counterfeit-signatures
An ML model that can detect counterfeit  signatures.

<b>Task</b>: binary verification — given pair (signature A, signature B) decide if same-person genuine or a forgery

<b>Approach</b>: Siamese network with a CNN embedding backbone (lightweight ResNetish or small custom CNN). Learn embedding with contrastive loss (or triplet loss). At inference, compute distance between embeddings and threshold. 

<b>Why Siamese</b>: works with many signers and limited samples per signer; generalizes to unseen signers.

<b>Evaluation</b>: ROC curve and AUC, plus false acceptance rate (FAR) and false rejection rate (FRR) at chosen thresholds. 

# Finding Datasets

I will need to look for public signature datasets commonly used in research such as GPDS, MCYT, CEDAR, SigComp. Data will need to consist of genuine and (skilled) forgeries per person for best results. 

If I cannot get data, I can train a Siamese model by treating different people's genuine signatures as negatives (less ideal vs skillful forged negatives).

## Data collection tips

* Scan at consistent DPI (300 dpi) -> convert to greyscale
* Crop tightly around the signature to preserve stroke thickness. Make sure signatures are on a consistent background. 
* Collect multiple smaples per signer (>= 5 genuine signatures) and forgeries (if possible).

# Preprocessing and Augmentation

* Convert data to greyscale and resize to a fixed size such as 155x220 or 140x220. Maintain an aspect ration with padding. 
* Contast ecnhancement (CLAHE) can be done, but is optional. 
* Binarize/greyscale: start data with grayscale since binarization can remove useful pressure info.
* Data Augmentations: random rotations with small angels (+- 5 degrees), random affine/shear small, random erasing (to simulate ink breaks), elastic transform (but done carefully), Gaussian noise, brightness/contrast jitter. Avoid huge distortions - signatures are delicate.  



