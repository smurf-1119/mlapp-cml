
nohup python CML.py --load_para='true' --epochs=20 --lr=0.001 --model='2021_12_13_00_56_07epochs:20_lr0.001__Lambda.pt' --data_type="Heud2000" > Heud2000_load.log

nohup python CML.py --load_para='false' --epochs=30 --lr=0.001 --data_type="Reuters2000" > Reuters2000.log

nohup python CML.py --load_para='false' --epochs=30 --lr=0.001 --data_type="Heud2000" > Heud2000.log

nohup python CML.py --load_para='true' --epochs=20 --lr=0.001 --data_type="Reuters2000" > Reuters2000_load.log
