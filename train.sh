CUDA_VISIBLE_DEVICES="0" \
python train.py \
--sam3_path "<set your sam3.pt path here>" \
--train_image_path "<set your training image dir here>" \
--train_mask_path "<set your training mask dir here>" \
--save_path "<set your checkpoint saving dir here>" \
--epoch 20 \
--lr 0.0002 \
--batch_size 12