# Import things


```python
import pca # explain, apply, visualize
import decision_tree # DecisionTree
import neural_network # NeuralNetwork
import adapter # Adapter
import pandas
```

    2023-12-01 14:55:10.895931: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2, in other operations, rebuild TensorFlow with the appropriate compiler flags.


# Open dataset


```python
dataset = pandas.read_csv("dataset.csv")
```

# Explain with PCA


```python
scaled_data = adapter.Adapter(dataset).skip(["class"]).with_categoricals().only_numericals().scale().ok()
pca.explain(scaled_data)
```


    
![png](output_5_0.png)
    


# Select PCA Components


```python
components, data, explained_variance, most_important_features = pca.apply(scaled_data)
print("explained variance = %s%%" % (explained_variance.round(3) * 100))
print("number of selected features = %s" % len(most_important_features))
print("most important features = %s" % most_important_features)
```

    explained variance = 83.6%
    number of selected features = 9
    most important features = ['gill-attachment', 'cap-surface', 'stalk-color-above-ring', 'ring-type', 'stalk-surface-below-ring', 'habitat', 'cap-shape', 'ring-number', 'stalk-surface-above-ring']


# Visualize PCA Components


```python
pca.visualize(components, dataset)
```


    
![png](output_9_0.png)
    


# Using a Decision Tree on all Dataset


```python
trainset, testset = adapter.Adapter(dataset).split('class')
tree = decision_tree.DecisionTree()
tree.fit(trainset)
tree.plot()
cm, acc = tree.evaluate(testset)
print("accuracy = %s" % (acc.round(3) * 100))
```

    accuracy = 100.0



    
![png](output_11_1.png)
    


# Selecting features with PCA


```python
trainset, testset = adapter.Adapter(dataset).only(most_important_features + ['class']).split('class')
tree = decision_tree.DecisionTree()
tree.fit(trainset)
tree.plot()
cm, acc = tree.evaluate(testset)
print("accuracy = %s" % (acc.round(3) * 100))
```

    accuracy = 96.8



    
![png](output_13_1.png)
    


# Using a Neural Network instead


```python
trainset, testset = adapter.Adapter(dataset).split('class')
nn = neural_network.NeuralNetwork((len(trainset['x'].columns),), (3,3,3,3), (len(trainset['y'].columns),))
nn.plot()
```




    
![png](output_15_0.png)
    




```python
nn.fit(trainset)
loss, acc = nn.evaluate(testset)
print("accuracy = %s" % (acc.round(3) * 100))
```

    Epoch 1/10
    1138/1138 [==============================] - 2s 2ms/step - loss: 0.9835 - accuracy: 0.7320
    Epoch 2/10
    1138/1138 [==============================] - 2s 2ms/step - loss: 0.3940 - accuracy: 0.8711
    Epoch 3/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.3134 - accuracy: 0.8936
    Epoch 4/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.2943 - accuracy: 0.9068
    Epoch 5/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.2416 - accuracy: 0.9202
    Epoch 6/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.2248 - accuracy: 0.9300
    Epoch 7/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.2457 - accuracy: 0.9335
    Epoch 8/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.2023 - accuracy: 0.9409
    Epoch 9/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.1983 - accuracy: 0.9434
    Epoch 10/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.1952 - accuracy: 0.9435
    accuracy = 94.0


# Again exploiting PCA 


```python
trainset, testset = adapter.Adapter(dataset).only(most_important_features + ['class']).split('class')
nn = neural_network.NeuralNetwork((len(trainset['x'].columns),), (9,), (len(trainset['y'].columns),))
nn.plot()
```




    
![png](output_18_0.png)
    




```python
nn.fit(trainset)
loss, acc = nn.evaluate(testset)
print("accuracy = %s" % (acc.round(3) * 100))
```

    Epoch 1/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 1.0875 - accuracy: 0.6622
    Epoch 2/10
    1138/1138 [==============================] - 1s 1ms/step - loss: 0.5313 - accuracy: 0.7677
    Epoch 3/10
    1138/1138 [==============================] - 1s 1ms/step - loss: 0.4652 - accuracy: 0.7983
    Epoch 4/10
    1138/1138 [==============================] - 1s 1ms/step - loss: 0.4263 - accuracy: 0.8227
    Epoch 5/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.3694 - accuracy: 0.8514
    Epoch 6/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.3540 - accuracy: 0.8834
    Epoch 7/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.3306 - accuracy: 0.8910
    Epoch 8/10
    1138/1138 [==============================] - 2s 1ms/step - loss: 0.3152 - accuracy: 0.9020
    Epoch 9/10
    1138/1138 [==============================] - 1s 1ms/step - loss: 0.2854 - accuracy: 0.9040
    Epoch 10/10
    1138/1138 [==============================] - 1s 1ms/step - loss: 0.2792 - accuracy: 0.9063
    accuracy = 91.3

