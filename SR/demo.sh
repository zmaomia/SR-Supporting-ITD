
python SRMains.py train --dataset_name 'MS' --n_blocks 9 --epochs 40 --batch_size 8 --n_feats 128 --n_scale 4 --gpus "0"
python NAFtest.py test --dataset_name 'MS' --n_blocks 9 --epochs 40 --batch_size 4 --n_feats 128 --n_scale 4 --gpus "0"
python NAFPRED2.py test --dataset_name 'MS' --n_blocks 9 --epochs 40 --batch_size 1 --n_feats 128 --n_scale 4 --gpus "0"
python NAFtest_PRED.py test --dataset_name 'MS' --n_blocks 9 --epochs 40 --batch_size 4 --n_feats 128 --n_scale 4 --gpus "0"


n_scale #是超分尺度

