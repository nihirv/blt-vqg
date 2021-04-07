CUDA_VISIBLE_DEVICES=1, python3 train_iq.py \
--hidden_dim 1024 \
--latent_dim 1024 \
--pwffn_dim 2048 \
--num_layers 6 \
--num_heads 8 \
--batch_size 64 \
--num_pretraining_steps 6000 \
--input_mode "cat" \
--print_note "posterior is Cat, Q. Adaptive LR"
