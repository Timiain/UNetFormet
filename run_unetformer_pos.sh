# python ./train_supervision.py --config ./config/potsdam/unetformer.py > unetformer_train_potsdam_RERFRRN_512.txt
python ./potsdam_test.py -c ./config/potsdam/unetformer.py -o ./fig_results/potsdam/unetformer-res18 --rgb  > unetformer_test_potsdam_RERFRRN_512.txt
python ./potsdam_test.py -c ./config/potsdam/unetformer.py -o ./fig_results/potsdam/unetformer-res18_lr --rgb -t 'lr' > unetformer_test_potsdam_RERFRRN_512_lr.txt
python ./potsdam_test.py -c ./config/potsdam/unetformer.py -o ./fig_results/potsdam/unetformer-res18_d4 --rgb -t 'd4' > unetformer_test_potsdam_RERFRRN_512_d4.txt