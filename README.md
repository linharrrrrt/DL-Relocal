# DL-Relocal

this version of code is of original ,w1 ,w2 and w1_w2.

usage:
···
`mkdir build`
`cd build`
`cmake ..`
`make`
···

when you are training or testing, this optional will help you restore from break:
···
`-sscript` scoring script
`-omodel` useing model name
`-oscript` training and test script 
`-continue` if you want use optinal blow , you need to set it 1
`-testResultsFile` the information of test will write in a file which start with this properties,like a prefix,default ""
`-testFolder` the test image,depth and put poeses in it,default "./test/"
`-roundsStart` the start number of file last.it dafult 0,if you want to use this optional for test ,you should set it to the number which you want start.
`-testStep` the step to next test file suffix.
`-rounds` the iterate end number, training test dosen't stop until the roundStart+n*testStep equals to rounds.
`-GPUNo` set use GPU ,default 1.
`-storeCounter` set the start number of files stored.
···

datasets:
first copy the scenes example to the same folder,
rename the folder name,
replace the test and training folder to your dataset.

for example: 
copy chess and paste,you will get a folder named chess(copy),
then rename the chess(copy) folder to fire.
cd fire,delete folder test and training.
copy your sorted test and training data which has the same folder structure to fire folder.

how to sort your datasets:
```
unzip DSAC-master,
cd DSAC-master,
unzip 7scenes.tar.gz,
copy your 'datasets'.zip into this folder.
unzip the 'datasets'.zip,
cd 'datasets'
unzip all *.zip .
cd ..
```
edit link_all.sh:
```
$1:the absolute path to DSAC-master eg./home/iccd/Documents/datasets/DSAC-master
$2:the absolute path to DSAC-master/7scenes eg./home/iccd/Documents/datasets/DSAC-master/7scenes
```
only the used dataset be left.
for example:
````
-------------------
#!/bin/bash

DATA_DIR="/home/iccd/Documents/datasets/DSAC-master"
DEST_DIR="/home/iccd/Documents/datasets/DSAC-master/7scenes"

for DATASET in "fire"
do
    /usr/bin/env python3 link_7scenes.py \
		 --data_dir="${DATA_DIR}/${DATASET}" \
		 --dest_dir="${DEST_DIR}/7scenes_${DATASET}" \
		 --dry_run=False
done
-------------------
```
```
sh link_all.sh

cp 7scenes/7scenes_fire/test $(your fire path)
cp 7scenes/7scenes_fire/training $(your fire path)
```
make sure has the same folder structure to example folder i.e. chess.

if you make datasets and envriment all ok, it's your time for training and testing:

training:
```
`../../train_reproDemi -oscript train_objVGG.lua`
or
`../../train_reproAngle -oscript train_objPyramid.lua`

train_reproDemi train_reproAngle 

train_objVGG train_objPyramid train_objVGGAttention train_objPyramidAttention 

```


test:
```
`./../test_ransac -oscript train_objVGG.lua -omodel (***.net) -sscript score_incount_ec6.lua -rdraw 0 -continue 1 -testResultsFile "(w1)4w_10w" -testFolder "./test/" -roundsStart 40000 -testStep 1000 -rounds 100001 -GPUNo 1`

`./../test_ransac -oscript train_objPyramid.lua -omodel  (***.net) -sscript score_incount_ec6.lua -rdraw 0 -continue 1 -testResultsFile "(w1)4w_10w" -testFolder "./test/" -roundsStart 40000 -testStep 1000 -rounds 100001 -GPUNo 1`

```




training:
```
`../../train_obj -oscript train_obj.lua`
`../../train_reprow1 -oscript train_obj.lua -omodel obj_model_fcn_init.net+i`
`../../train_ransac -oscript train_obj_e2e.lua -omodel obj_model_fcn_repro.net+i -sscript score_incount_ec6.lua`
```
testing:
```
`../../test_ransac -oscript train_obj.lua -omodel obj_model_fcn_repro.net -sscript score_incount_ec6.lua -rdraw 0 -continue 1 -testResultsFile "1k_10w" -testFolder "./test/" -roundsStart 1000 -testStep 1000 -rounds 100001 -GPUNo 1`
```

step:
```
1. `../../train_obj -oscript train_obj.lua`
2. `../../test_ransac -oscript test_obj.lua -omodel obj_model_fcn_init.net -sscript score_incount_ec6.lua -rdraw 0 -continue 1 -testResultsFile "(init)8w_10w" -testFolder "./test/" -roundsStart 80000 -testStep 1000 -rounds 100001 -GPUNo 1`
3. `cp mydensenet-201_1-29_init.t7 mydensenet-201_1-29_repro.t7`
4. `../../train_reprow1 -oscript train_repro.lua -omodel obj_model_fcn_init.net+i`
5. `../../test_ransac -oscript test_obj.lua -omodel obj_model_fcn_repro.net -sscript score_incount_ec6.lua -rdraw 0 -continue 1 -testResultsFile "(w1)4w_10w" -testFolder "./test/" -roundsStart 40000 -testStep 1000 -rounds 100001 -GPUNo 1`
```
