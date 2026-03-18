STEP 1:
Extract positive (smok) and negative (no smoke) samples from sequences.
Start on all sequences:

augmented_yolo_set.py -i <input_dir> -o <output_dir> [-r -l -h]

Use separate <output_dir> for training, validation and test sequences.

STEP 2:
(start python)
import set_utils
set_utils.processDir('<output_dir_from_prvious_step>', '<sample_dir>')

This step balances the number of positive and negative samples in such a way as to discard excess negative samples. If there are more positive samples than negative, all samples are retained. 
Additionally, the order of the samples generated in the previous step is randomized.
