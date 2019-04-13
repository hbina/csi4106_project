# <=13th November 2018

Just setting up the program. 
Around this time, I finally able to set up the parsing and splitting of the data correctly.
This includes turning some of the columns into one-hot encoded columns.

Other than that, not much....most of the code is taken from the internet...I hope I referenced them in the code.
The one that we have now should be the baseline.
It's basically a bunch of dense layers.

Considering how simple the dataset, relatively speaking, current neuron densities are probably not good.
The prediction function is probably too complex, and thus over-fitting the dataset (reference: Introduction to Statistical Learning, earlier chapters).

Next thing to do is to reduce the neuron densities...and consider some other form of networks...

# 14th November 2018

Now only have 1 layer.
We think that we overestimated the complexity of the dataset.
A single dataset is apparently enough to capture the dataset without overfitting during testing.
Thanks to Nick Broadbent for mentioning this.

There is also a problem with the given test dataset during the encoding process.
This is caused by the fact that the training dataset does not have all the possible values for" 'Parch' column.
Therefore, the generated encoding columns are less than what is possible in the testing dataset.

The current solution involves pruning that additional column.
We think that there are better ways of doing but we haven't even started looking at anything yet.

# 14th November 2018 - Evening

There appear to be a problem with the network...
Occasionally, it will contain weights that are completely zero, causing the outputs to be all zeros
Not sure what to make of this.
Perhaps its because of the pruning of additional encoding bit?

Also added Tensorboard support...

# 2nd December 2018

Really...what else can I do....lel