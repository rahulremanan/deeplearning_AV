import pandas as pd
import numpy as np
import sklearn.ensemble as ske
from sklearn import cross_validation
from sklearn.feature_selection import SelectFromModel
from keras.models import Sequential
from keras.layers import Dense
import h5py

# fix random seed for reproducibility
np.random.seed(7)

data = pd.read_csv('C:/Users/Rahul/Desktop/antivirus_demo-master/data.csv', sep='|')
X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values

print('Researching important feature based on %i total features\n' % X.shape[1])

# Feature selection using Trees Classifier
fsel = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y ,test_size=0.2)

features = []

print('%i features identified as important:' % nb_features)

indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

# XXX : take care of the feature order
for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2+f])

# Deep learning:
# create model
model = Sequential()
model.add(Dense(12, input_dim=54, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=10, batch_size=10)

# evaluate the model
scores = model.evaluate(X, y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save model
model.save('C:/Users/Rahul/Desktop/antivirus_demo-master/deep_calssifier/deep_classifier.h5')
