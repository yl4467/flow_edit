CUDA_VISIBLE_DEVICES='7' python main_p2p_pie.py --mode ef --output_path /home/yanli/h-edit/text-guided/pie_results/ef/ > log1.log &
CUDA_VISIBLE_DEVICES='5' python main_p2p_pie.py --mode nmg --output_path /home/yanli/h-edit/text-guided/pie_results/nmg_p2p/ > log3.log &
CUDA_VISIBLE_DEVICES='6' python main_p2p_pie.py --mode pnp_inv_p2p --output_path /home/yanli/h-edit/text-guided/pie_results/pnp/ > log4.log &



  