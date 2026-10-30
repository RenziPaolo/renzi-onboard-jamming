python testing/test_thesis.py --model_path mnist/ann/standardFalse/modelmnist.pt --output_dir mnist/ann/standardFalse/ --type standard --model linear -dataset mnist --batch 1 --tau -4.5
python testing/test_thesis.py --model_path mnist/ann/standardFalse/modelEuroSAT.pt --output_dir mnist/ann/standardFalse/ --type standard --model linear -dataset EuroSAT --batch 1 --tau -5.5
python testing/test_thesis.py --model_path mnist/ann/convFalse/modelmnist.pt --output_dir mnist/ann/convFalse/ --type standard --model conv -dataset mnist --batch 1 --tau 1.5
python testing/test_thesis.py --model_path mnist/ann/convFalse/modelEuroSAT.pt --output_dir mnist/ann/convFalse/ --type standard --model conv -dataset EuroSAT --batch 1 --tau -17.0

python testing/test_thesis.py --model_path mnist/snn/modelstandard_datasetMNIST/checkpoint_max.pt --output_dir mnist/snn/modelstandard_datasetMNIST/ --type spike --model snn -dataset mnist --batch 1 --tau 4.999999
python testing/test_thesis.py --model_path mnist/snn/modelstandard_datasetMNIST/checkpoint_max.pt --output_dir mnist/snn/modelstandard_datasetMNIST/ --type spike --model snn -dataset mnist --batch 1 --tau 5.0
python testing/test_thesis.py --model_path mnist/snn/modelstandard_datasetEuroSAT_case3False/checkpoint_max.pt --output_dir mnist/snn/modelstandard_datasetEuroSAT_case3False/ --type spike --model snn -dataset EuroSAT --batch 1 --tau 4.1
python testing/test_thesis.py --model_path mnist/snn/modelconv_datasetMNIST/checkpoint_max.pt --output_dir mnist/snn/modelconv_datasetMNIST/ --type spike --model csnn -dataset mnist --batch 1 --tau 4.3
python testing/test_thesis.py --model_path mnist/snn/modelconv_datasetEuroSAT_case3False/checkpoint_max.pt --output_dir mnist/snn/modelconv_datasetEuroSAT_case3False/ --type spike --model csnn -dataset EuroSAT --batch 1 --tau 3.0

