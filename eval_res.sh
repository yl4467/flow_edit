CUDA_VISIBLE_DEVICES=3 python evaluation/evaluation_dir.py \
    --res_image_folder /home/yanli/h-edit/text-guided/results/h_edit_w_guide \
    --result_path ./results/results_guide.csv

CUDA_VISIBLE_DEVICES=3 python evaluation/evaluation_dir.py \
    --res_image_folder /home/yanli/h-edit/text-guided/results/nmg \
    --result_path ./results/results_nmg.csv

CUDA_VISIBLE_DEVICES=3 python evaluation/evaluation_dir.py \
    --res_image_folder /home/yanli/h-edit/text-guided/results/ef \
    --result_path ./results/results_ef.csv

CUDA_VISIBLE_DEVICES=3 python evaluation/evaluation_dir.py \
    --res_image_folder /home/yanli/h-edit/text-guided/results/ef_p2p \
    --result_path ./results/results_ef_p2p.csv

CUDA_VISIBLE_DEVICES=3 python evaluation/evaluation_dir.py \
    --res_image_folder /home/yanli/h-edit/text-guided/results/pnp \
    --result_path ./results/results_pnp.csv