AWS_HOSTNAME=brrn@dgx2.cloudlab.zhaw.ch

rsync -e "ssh" --exclude 'experiments' --exclude '.git' --exclude '.idea' --exclude 'pretrained_models' --exclude 'wandb' --exclude 'data/spider/testsuite_databases' --exclude 'data/spider/original/database' -avzh * $AWS_HOSTNAME:/cluster/home/brrn/PA3/valuenet