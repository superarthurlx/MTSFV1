The environment and data required to reproduce the experimental results in the paper can be referenced from BasicTS https://github.com/GestaltCogTeam/BasicTS
Once ready, execute the following command:

```bash
python experiments/train.py -c baselines/STHG/PEMS03.py -g 0 
python experiments/train.py -c baselines/STHG/PEMS04.py -g 0 
python experiments/train.py -c baselines/STHG/PEMS07.py -g 0 
python experiments/train.py -c baselines/STHG/PEMS08.py -g 0
python experiments/train.py -c baselines/STHG/SD.py -g 0
python experiments/train.py -c baselines/STHG/GBA.py -g 0 
python experiments/train.py -c baselines/STHG/GLA.py -g 0  
python experiments/train.py -c baselines/STHG/CA.py -g 0  
```
