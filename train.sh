python run_classifier.py --train_data_dir "../data_version3/" --backbone "Attention_GCN_backbone" \
--checkpoints_folder checkpoints/version3_baseline_weighted_smallclass --preprocessing "False" 
# --checkpoints_dir checkpoints/version2_baseline_weighted_smallclass/model_-1.pth --lr 1e-3 --epochs 100
# --extend_dataset "../data_version2_output/debugs/"