python plot_thesis.py --model_path mnist/ann/standardFalse/modelmnist.pt --output_dir mnist/ann/standardFalse/ --type standard --model_type linear --dataset mnist --batch_size 1 --tau -4.5
python plot_thesis.py --model_path mnist/ann/standardFalse/modelEuroSAT.pt --output_dir mnist/ann/standardFalse/ --type standard --model_type linear --dataset EuroSAT --batch_size 1 --tau -5.5
python plot_thesis.py --model_path mnist/ann/convFalse/modelmnist.pt --output_dir mnist/ann/convFalse/ --type standard --model_type conv --dataset mnist --batch_size 1 --tau 1.5
python plot_thesis.py --model_path mnist/ann/convFalse/modelEuroSAT.pt --output_dir mnist/ann/convFalse/ --type standard --model_type conv --dataset EuroSAT --batch_size 1 --tau -17.0

python plot_thesis.py --model_path mnist/snn/modelstandard_datasetMNIST/checkpoint_max.pt --output_dir mnist/snn/modelstandard_datasetMNIST/ --type spike --model_type snn --dataset mnist --batch_size 1 --tau 4.999999
python plot_thesis.py --model_path mnist/snn/modelstandard_datasetMNIST/checkpoint_max.pt --output_dir mnist/snn/modelstandard_datasetMNIST/ --type spike --model_type snn --dataset mnist --batch_size 1 --tau 5.0
python plot_thesis.py --model_path mnist/snn/modelstandard_datasetEuroSAT_case3False/checkpoint_max.pt --output_dir mnist/snn/modelstandard_datasetEuroSAT_case3False/ --type spike --model_type snn --dataset EuroSAT --batch_size 1 --tau 4.1
python plot_thesis.py --model_path mnist/snn/modelconv_datasetMNIST/checkpoint_max.pt --output_dir mnist/snn/modelconv_datasetMNIST/ --type spike --model_type csnn --dataset mnist --batch_size 1 --tau 4.3
python plot_thesis.py --model_path mnist/snn/modelconv_datasetEuroSAT_case3False/checkpoint_max.pt --output_dir mnist/snn/modelconv_datasetEuroSAT_case3False/ --type spike --model_type csnn --dataset EuroSAT --batch_size 1 --tau 3.0

