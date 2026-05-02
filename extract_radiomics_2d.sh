python extract_radiomics_2d.py \
  --meta_image_dir /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net/dataset_processed/new_train/train_Meta \
  --meta_mask_dir /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net/dataset_processed/new_train_Meta_predictions \
  --nonmeta_image_dir /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net/dataset_processed/new_train/train_NonMeta \
  --nonmeta_mask_dir /mnt/wangbd8/workspace/ThyroidAgent/LLNM-Net/dataset_processed/new_train_NonMeta_predictions \
  --output_csv ./output/radiomics_features.csv