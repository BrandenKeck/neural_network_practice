import numpy as np
import tensorflow as tf

# INTRODUCTION ----

def run_introduction():
    # Typing in Tensorflow:
    string = tf.Variable("this is a string", tf.string)
    number = tf.Variable(324, tf.int16)
    floating = tf.Variable(3.567, tf.float64)

    # We can make things multidimensional:
    rank1_tensor = tf.Variable(["Test"], tf.string)
    rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)
    print(tf.rank(rank2_tensor))
    print(rank2_tensor.shape)

    # We can re-shape the multidimensional
    tensor1 = tf.ones([1,2,3])
    tensor2 = tf.reshape(tensor1, [2,3,1])
    tensor3 = tf.reshape(tensor2, [3, -1])

# LINEAR REGRESSION  ----

def run_linear_regression():
    # Get a dataset
    import pandas as pd
    X_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    X_test = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
    Y_train = X_train.pop('survived')
    Y_test = X_test.pop('survived')

    # Data massaging...
    categorical = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
    numeric = ['age', 'fare']
    feature_columns = []
    for feature_name in categorical:
      vocabulary = X_train[feature_name].unique()  # gets a list of all unique values from given feature column
      feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    for feature_name in numeric:
      feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    # The TensorFlow model we are going to use requires that the data we pass it comes in as a tf.data.Dataset object.
    # This means we must create a input function that can convert our current pandas dataframe into that object.
    def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
      def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
          ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
      return input_function
    training_func = make_input_fn(X_train, Y_train)
    testing_func = make_input_fn(X_test, Y_test, num_epochs=1, shuffle=False)

    # Use a linear estimator to utilize the linear regression algorithm.
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

    # Make some predictions
    pred_dicts = list(linear_est.predict(testing_func))

    # Training call
    linear_est.train(training_func)
    result = linear_est.evaluate(testing_func)
    print(result['accuracy'])  # Print

# CLASSIFICATION ----

def run_classification_example():

    # Data stuff
    import pandas as pd
    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
    test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
    X_train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    X_test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    Y_train = X_train.pop('Species')
    Y_test = X_test.pop('Species')

    # Data massaging...
    # Feature columns describe how to use the input.
    feature_columns = []
    for key in X_train.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # The TensorFlow model we are going to use requires that the data we pass it comes in as a tf.data.Dataset object.
    # This means we must create a input function that can convert our current pandas dataframe into that object.
    def input_fn(features, labels, training=True, batch_size=256):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        if training:
            dataset = dataset.shuffle(1000).repeat()
        return dataset.batch(batch_size)

    # Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[30, 10],
        n_classes=3
    )

    # Training call
    # We include a lambda to avoid creating an inner function previously
    classifier.train(
        input_fn=lambda: input_fn(X_train, Y_train, training=True),
        steps=5000
    )

    # Make some predictions
    eval_result = classifier.evaluate(
        input_fn=lambda: input_fn(X_test, Y_test, training=False)
    )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Convert the inputs to a Dataset without labels.
    def predict_input_fn(features, batch_size=256):
        return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # Make a prediction
    predict = X_test.sample()
    predictions = classifier.predict(input_fn=lambda: predict_input_fn(predict))
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{}" ({:.1f}%)'.format(
            SPECIES[class_id], 100 * probability))

#run_introduction()
#run_linear_regression()
#run_classification_example()
