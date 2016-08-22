# SwiftLearner hotel recommendations example

<img align="right" src="img/swiftlearner.jpg" alt="Swift Learner"/>

This is based on [Expedia hotel recommendations competition on Kaggle](https://www.kaggle.com/c/expedia-hotel-recommendations)

I have extracted a subset of fields and data rows to test with NN/Backprop.
This is only a small technical demo.

The full competition dataset is 500MB compressed, and is quite complex.
It also turned out it contained solution leakage, which this demo also picks to some degree.
However, there were analytical methods to solve the competition more precisely than with ML,
that was the winning approach on Kaggle.

This demo shows that SwiftLearner backprop classifier scales fine to thousands
of inputs and millions of examples. The accuracy achieved so far is 0.058, which
is nothing spectacular, but certainly an evidence of some learning, compared
to a random guess at 0.01.