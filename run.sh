# CUDA_VISIBLE_DEVICES=2, python3 train_iq.py \
# --hidden_dim 256 \
# --latent_dim 256 \
# --pwffn_dim 1024 \
# --enc_num_layers 12 \
# --dec_num_layers 1 \
# --num_heads 8 \
# --batch_size 64 \
# --num_pretraining_steps 8000 \
# --enable_t_space True \
# --val_every 500 \
# --use_gpu True \
# --input_mode "ans" \
# --print_note "posterior is Cat, Q"


CUDA_VISIBLE_DEVICES=1, python3 train_iq.py \
--hidden_dim 1024 \
--latent_dim 1024 \
--pwffn_dim 2048 \
--enc_num_layers 6 \
--dec_num_layers 6 \
--num_heads 8 \
--batch_size 64 \
--num_pretraining_steps 10000 \
--enable_t_space True \
--use_gpu True \
--input_mode "ans" \
--print_note "Figure 2 from Marek."
