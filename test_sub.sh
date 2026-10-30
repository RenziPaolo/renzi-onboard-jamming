#!/bin/bash

# Loop through all subdirectories in the current directory
for dir in */; do
    dir=${dir%/}
    if [ -f "$dir/checkpoint_min.pt" ]; then
        if [[ "$dir" == *"image"* ]]; then
            type="image"
        else
            type="line"
        fi

        if [[ "$dir" == *"T100"* ]]; then
            T="100"
        else
            T="10"
        fi

        if [[ "$dir" == *"lifTrue"* ]]; then
                    if [[ "$dir" == *"vaeTrue"* ]]; then
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            fi
        else
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            fi
        fi
        else
                    if [[ "$dir" == *"vaeTrue"* ]]; then
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ 
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/
                echo "$dir"
            fi
        else
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ 
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_min.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ 
                echo "$dir"
            fi
        fi
        fi
    else
        if [[ "$dir" == *"image"* ]]; then
            type="image"
        else
            type="line"
        fi

        if [[ "$dir" == *"T100"* ]]; then
            T="100"
        else
            T="10"
        fi

        if [[ "$dir" == *"lifTrue"* ]]; then
                    if [[ "$dir" == *"vaeTrue"* ]]; then
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            fi
        else
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ --lif $lif
                echo "$dir"
            fi
        fi
        else
                    if [[ "$dir" == *"vaeTrue"* ]]; then
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ 
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --vae True --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/
                echo "$dir"
            fi
        else
            if [[ "$dir" == *"only_magTrue"* ]]; then
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 --only_mag True -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ 
                echo "$dir"
            else
                python /home/ricky/uni/tesi/RFInject/code/renzi-onboard-jamming/testing/test_anomaly.py --model_path "$dir/checkpoint_latest.pt" --output_dir "$dir" --model $type --batch 1 --tau 0 --T $T --latent_dim 1 -data_dir /home/ricky/uni/tesi/RFInject/data/v2/RFInject/ 
                echo "$dir"
            fi
        fi
        fi


    fi

done
