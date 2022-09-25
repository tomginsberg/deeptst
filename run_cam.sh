echo 'staring run 1'
nohup python scratch/detectron_camelyon.py --run_name cam_harm --harmful --gpu 0 --splits p --samples 10 --resume >p10.log &
echo 'staring run 2'
nohup python scratch/detectron_camelyon.py --run_name cam_harm --harmful --gpu 1 --splits p --samples 20 --resume >p20.log &
echo 'staring run 3'
nohup python scratch/detectron_camelyon.py --run_name cam_harm --harmful --gpu 2 --splits p --samples 50 --resume >p50.log &
echo 'staring run 4'
nohup python scratch/detectron_camelyon.py --run_name cam_harm --harmful --gpu 3 --splits q --samples 10 --resume >q10.log &
echo 'staring run 5'
nohup python scratch/detectron_camelyon.py --run_name cam_harm --harmful --gpu 0 --splits q --samples 20 --resume >q20.log &
echo 'staring run 6'
nohup python scratch/detectron_camelyon.py --run_name cam_harm --harmful --gpu 1 --splits q --samples 50 --resume >q50.log &
