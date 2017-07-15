# DrawWithAI
CNN classifier for hand drawn sketchs written with TensorFlow

You can prepare your own dataset placing them in a folder named "data". The structure should look like this.

./data/class1/1.png
./data/class2/1.png
...

run prepare_data.py to produce dataset.csv

cnn.py parameters:

-s: save model as chekpoint file
-r: restore checkpoint file
-f: run on floydhub



