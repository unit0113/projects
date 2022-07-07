So, admittedly, I didn't feel like changing around my python setup (upgraded to 3.10 eary, won't do that again) and tensorflow hadn't been updated to support 3.10. So I spend the time doing other AI classes and projects, one of the more interesting things I stumbled accross was a video by computerfile on a CNN for the MNIST database. That video impressed upon me how awesome convolution/pooling layers are and that led into some further reading on sites like kaggle and places like that.

With the load function, other than the standard issues involved in figuring out a new library or two, it became apparent that initial loading takes a while. So I added in status checks at the beginning, end, and once each segment is loaded.

For the model, I decided to start with three layers of convulation + pooling + normalization + dropout, then one normal dense hidden layer, and the standard softmax output layer. While it was running I was thinking about what to improve on for the next attempt, but it ended up with an accuracy of 0.9906, and I figured that was pretty good.

Epoch 1/10
500/500 [==============================] - 7s 12ms/step - loss: 1.6321 - accuracy: 0.5424
Epoch 2/10
500/500 [==============================] - 6s 12ms/step - loss: 0.3850 - accuracy: 0.8803
Epoch 3/10
500/500 [==============================] - 6s 12ms/step - loss: 0.2184 - accuracy: 0.9331
Epoch 4/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1455 - accuracy: 0.9547
Epoch 5/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1248 - accuracy: 0.9611
Epoch 6/10
500/500 [==============================] - 6s 12ms/step - loss: 0.1092 - accuracy: 0.9632
Epoch 7/10
500/500 [==============================] - 6s 12ms/step - loss: 0.0936 - accuracy: 0.9697
Epoch 8/10
500/500 [==============================] - 6s 12ms/step - loss: 0.0839 - accuracy: 0.9730
Epoch 9/10
500/500 [==============================] - 6s 12ms/step - loss: 0.0882 - accuracy: 0.9712
Epoch 10/10
500/500 [==============================] - 6s 12ms/step - loss: 0.0725 - accuracy: 0.9760
333/333 - 1s - loss: 0.0298 - accuracy: 0.9906 - 1s/epoch - 4ms/step

Decided just to mess around.

First decided to see what would happen if all dropout layers were removed. The accuracy of this was 0.9939. An improvement, though during training the accuracy reached up to 0.9996 and I worry that this induced some overfitting.

Epoch 1/10
500/500 [==============================] - 6s 11ms/step - loss: 0.6711 - accuracy: 0.8197 
Epoch 2/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0680 - accuracy: 0.9817
Epoch 3/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0450 - accuracy: 0.9873
Epoch 4/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0251 - accuracy: 0.9931
Epoch 5/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0335 - accuracy: 0.9905
Epoch 6/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0401 - accuracy: 0.9873
Epoch 7/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0187 - accuracy: 0.9949
Epoch 8/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0293 - accuracy: 0.9915
Epoch 9/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0230 - accuracy: 0.9937
Epoch 10/10
500/500 [==============================] - 5s 11ms/step - loss: 0.0018 - accuracy: 0.9996
333/333 - 1s - loss: 0.0265 - accuracy: 0.9939 - 1s/epoch - 3ms/step

Next, I doubled the size of the dense hidden layer. The accuracy for this was functionally the same as the first run, though this model would require more computational power to train.

Epoch 1/10
500/500 [==============================] - 7s 13ms/step - loss: 1.4027 - accuracy: 0.6081
Epoch 2/10
500/500 [==============================] - 7s 13ms/step - loss: 0.3664 - accuracy: 0.8819
Epoch 3/10
500/500 [==============================] - 6s 13ms/step - loss: 0.2010 - accuracy: 0.9352
Epoch 4/10
500/500 [==============================] - 6s 13ms/step - loss: 0.1564 - accuracy: 0.9494
Epoch 5/10
500/500 [==============================] - 7s 13ms/step - loss: 0.1172 - accuracy: 0.9605
Epoch 6/10
500/500 [==============================] - 7s 14ms/step - loss: 0.0993 - accuracy: 0.9683
Epoch 7/10
500/500 [==============================] - 7s 13ms/step - loss: 0.0900 - accuracy: 0.9713
Epoch 8/10
500/500 [==============================] - 7s 13ms/step - loss: 0.0862 - accuracy: 0.9737
Epoch 9/10
500/500 [==============================] - 7s 13ms/step - loss: 0.0855 - accuracy: 0.9725
Epoch 10/10
500/500 [==============================] - 7s 14ms/step - loss: 0.0796 - accuracy: 0.9740
333/333 - 1s - loss: 0.0334 - accuracy: 0.9909 - 1s/epoch - 4ms/step

Next I ran it without the normalization layers. Accuracy here was a solid .9688, though curiously, the training accuracy never achieved above a 0.9. Which seems like a large step up for going from the training data to the testing data, a small part of this is that outliers are less likely to appear in smaller batches of data, but that much of difference seems to be unlikely to be explained entirely by that (at least to me). It also trained much slower than other models (no other model's initial epoch ended with an accuracy below 0.5, this one is half of that.)

Epoch 1/10
500/500 [==============================] - 6s 11ms/step - loss: 3.8601 - accuracy: 0.2382   
Epoch 2/10
500/500 [==============================] - 5s 11ms/step - loss: 1.5558 - accuracy: 0.5372
Epoch 3/10
500/500 [==============================] - 6s 11ms/step - loss: 1.0095 - accuracy: 0.6918
Epoch 4/10
500/500 [==============================] - 5s 11ms/step - loss: 0.7359 - accuracy: 0.7762
Epoch 5/10
500/500 [==============================] - 5s 11ms/step - loss: 0.6065 - accuracy: 0.8169
Epoch 6/10
500/500 [==============================] - 6s 11ms/step - loss: 0.4966 - accuracy: 0.8532
Epoch 7/10
500/500 [==============================] - 6s 11ms/step - loss: 0.4614 - accuracy: 0.8612
Epoch 8/10
500/500 [==============================] - 6s 11ms/step - loss: 0.4130 - accuracy: 0.8800
Epoch 9/10
500/500 [==============================] - 6s 11ms/step - loss: 0.3856 - accuracy: 0.8888
Epoch 10/10
500/500 [==============================] - 6s 11ms/step - loss: 0.3745 - accuracy: 0.8940
333/333 - 1s - loss: 0.1101 - accuracy: 0.9688 - 1s/epoch - 3ms/step

And finally, I added an additional convolution layer prior to the first max pooling layer. This model ended with the same accuracy on the testing data, but it's loss decreased faster during training and acheived a consistently higher training accuracy per epoch.

Epoch 1/10
500/500 [==============================] - 20s 39ms/step - loss: 1.1525 - accuracy: 0.6777
Epoch 2/10
500/500 [==============================] - 19s 38ms/step - loss: 0.1802 - accuracy: 0.9443
Epoch 3/10
500/500 [==============================] - 19s 38ms/step - loss: 0.0937 - accuracy: 0.9710
Epoch 4/10
500/500 [==============================] - 20s 40ms/step - loss: 0.0818 - accuracy: 0.9749
Epoch 5/10
500/500 [==============================] - 19s 37ms/step - loss: 0.0618 - accuracy: 0.9812
Epoch 6/10
500/500 [==============================] - 20s 40ms/step - loss: 0.0633 - accuracy: 0.9795
Epoch 7/10
500/500 [==============================] - 18s 37ms/step - loss: 0.0584 - accuracy: 0.9812
Epoch 8/10
500/500 [==============================] - 18s 36ms/step - loss: 0.0593 - accuracy: 0.9828
Epoch 9/10
500/500 [==============================] - 18s 36ms/step - loss: 0.0434 - accuracy: 0.9865
Epoch 10/10
500/500 [==============================] - 19s 37ms/step - loss: 0.0413 - accuracy: 0.9872
333/333 - 2s - loss: 0.0341 - accuracy: 0.9902 - 2s/epoch - 7ms/step

And after all this, I finally got tensorflow to work with my GPU (getting that to work was a pain...). This unsurprsingly had no affect on accuracy, put it did run about ~7x faster per epoch.

Epoch 1/10
500/500 [==============================] - 9s 7ms/step - loss: 1.1100 - accuracy: 0.6915   
Epoch 2/10
500/500 [==============================] - 3s 7ms/step - loss: 0.1748 - accuracy: 0.9470
Epoch 3/10
500/500 [==============================] - 3s 7ms/step - loss: 0.1019 - accuracy: 0.9695
Epoch 4/10
500/500 [==============================] - 3s 7ms/step - loss: 0.0848 - accuracy: 0.9745
Epoch 5/10
500/500 [==============================] - 3s 7ms/step - loss: 0.0687 - accuracy: 0.9779
Epoch 6/10
500/500 [==============================] - 3s 7ms/step - loss: 0.0665 - accuracy: 0.9795
Epoch 7/10
500/500 [==============================] - 3s 6ms/step - loss: 0.0585 - accuracy: 0.9821
Epoch 8/10
500/500 [==============================] - 3s 7ms/step - loss: 0.0582 - accuracy: 0.9830
Epoch 9/10
500/500 [==============================] - 3s 7ms/step - loss: 0.0601 - accuracy: 0.9820
Epoch 10/10
500/500 [==============================] - 3s 7ms/step - loss: 0.0459 - accuracy: 0.9856
333/333 - 1s - loss: 0.0351 - accuracy: 0.9901 - 936ms/epoch - 3ms/step