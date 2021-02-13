# Binary-Classification-Sigmoid-vs-Softmax
- Sample code for the illustration of Binary Classification with Sigmoid and Softmax activation

- This repo serves as a code illustration to confirm that for binary classification, Softmax activation function can be represented by a Sigmoid activation function with little modification.

- So, for binary classification, softmax activation function is equivalent to sigmoid activation function.

- Reference derivation of the same can be found here: https://blog.nex3z.com/2017/05/02/sigmoid-%E5%87%BD%E6%95%B0%E5%92%8C-softmax-%E5%87%BD%E6%95%B0%E7%9A%84%E5%8C%BA%E5%88%AB%E5%92%8C%E5%85%B3%E7%B3%BB/

- Once this is proved, the cross entropy loss can be applied without debating over which one activation function to use as the loss value for both will be the same.

- Equation for sigmoid cross entropy
 
    -  = (-1) * y_label * log(y_pred) + (-1) * (1 - y_label) * log(1 - y_pred)
 
    - Here, y_pred can be sigmoid(z) in case of sigmoid cross entropy and z is the logits of the last layers.

- Equation for sigmoid cross entropy
 
    - = sum_over_all_classes( (-1) * y_label * log(y_pred) )
 
    - Here, y_pred can be softmax(z) in case of sigmoid cross entropy and z is the logits of the last layers.


- In the code, it is actually proved that the Softmax is equivalent to Sigmoid with modification to weights of the last layer.
- Also, it is proved that the loss value of sigmoid cross entropy and softmax cross entropy in that setting is the same.