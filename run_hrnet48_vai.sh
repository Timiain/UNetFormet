#python ./train_hrnet.py --config ./config/vaihingen/hrnet.py > hrnet_trainv2_vaihingen_RERFRRN_512.txt
python ./vaihingen_test.py -c ./config/vaihingen/hrnet.py -o ./fig_results/vaihingen/hrnet-res18 --rgb  > hrnet_testv2_vaihingen_RERFRRN_512.txt
python ./vaihingen_test.py -c ./config/vaihingen/hrnet.py -o ./fig_results/vaihingen/hrnet-res18_lr --rgb -t 'lr' > hrnet_testv2_vaihingen_RERFRRN_512_lr.txt
python ./vaihingen_test.py -c ./config/vaihingen/hrnet.py -o ./fig_results/vaihingen/hrnet-res18_d4 --rgb -t 'd4' > hrnet_testv2_vaihingen_RERFRRN_512_d4.txt

#python ./train_proto.py --config ./config/vaihingen/hrnet_proto.py > hrnet_proto_trainv2_vaihingen_RERFRRN_512.txt
python ./vaihingen_test.py -c ./config/vaihingen/hrnet_proto.py -o ./fig_results/vaihingen/hrnet_proto-res18 --rgb  > hrnet_proto_testv2_vaihingen_RERFRRN_512.txt
python ./vaihingen_test.py -c ./config/vaihingen/hrnet_proto.py -o ./fig_results/vaihingen/hrnet_proto-res18_lr --rgb -t 'lr' > hrnet_proto_testv2_vaihingen_RERFRRN_512_lr.txt
python ./vaihingen_test.py -c ./config/vaihingen/hrnet_proto.py -o ./fig_results/vaihingen/hrnet_proto-res18_d4 --rgb -t 'd4' > hrnet_proto_testv2_vaihingen_RERFRRN_512_d4.txt