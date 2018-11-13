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