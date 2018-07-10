ssh -X nielsrolf@master.ml.tu-berlin.de
screen -S "train_model_$1"
qlogin
source irs2/bin/activate
cd irs/Predictive-Analysis
git co "origin/$2"
screen -dm bash -c 'python src/model_training/task.py'