# A simple ML exercise #

 # Load libraries
from file_utility import write_to_file
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# to print the relevant data
is_debug = True

# Load dataset
def load_dataset():
	url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
	names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
	global dataset
	dataset = pandas.read_csv(url, names=names)


# Summarize the dataset
def summarize_dataset():
	# - check dataset size
	print(dataset.shape)

	# - check last few data pieces
	print(dataset.head(20))

	# - check dataset STATISTICAL summary
	print(dataset.describe())

	# - check dataset CLASS DISTRIBUTION summary
	print(dataset.groupby('class').size())

	# Visualization

	# univariate plots
	# - box & whisker plots
	dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
	plt.show()

	# histogram plots
	dataset.hist()
	plt.show()

	# multivariate plots
	# scatter plot matrix
	scatter_matrix(dataset)
	plt.show()


# split-out validation dataset
def split_dataset():
	array = dataset.values
	X = array[:,0:4]
	Y = array[:,4]
	validation_size = 0.20
	seed = 7
	global X_train, X_validation, Y_train, Y_validation
	X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
	if is_debug:
		print_split_data()


# utility function
def print_split_data():
	print("Training data set: \n")
	print(X_train)
	print(Y_train)
	print("\n\n***\n\nValidation data set: \n")
	print(X_validation)
	print(Y_validation)


# test options and evaluation metric
def add_test_harness_data():
	global seed, scoring
	seed = 7
	# Accuracy: percentage ratio of the number of correctly predicted instances divided by the total number of instances in the dataset 
	scoring = 'accuracy'


# get results derived from linear and non-linear models
def evaluate_models():
	# spot-check Algorithms
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))

	# evaluate each model turn by turn
	global results, names
	results = []
	names = []
	for name, model in models:
		# each model is evaluated 10 times
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		if debug:
			print(msg)


# plot model evaluation results to compare spread and mean accuracy of each model
def plot_models_results():
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.show()


# get accuracy of most accurate model on validation set
def make_prediction():
	# Validating KNN classifier
	knn = KNeighborsClassifier()
	knn.fit(X_train, Y_train)
	predictions = knn.predict(X_validation)
	print(accuracy_score(Y_validation, predictions))
	print(confusion_matrix(Y_validation, predictions))
	print(classification_report(Y_validation, predictions))


# experiment with a few models
def evaluate_algorithms():
	split_dataset()
	add_test_harness_data()
	evaluate_models()
	plot_models_results()


# top-level caller
def main() :
	load_dataset()
	if is_debug:
		summarize_dataset()
	evaluate_algorithms()
	make_prediction()


# main function
if __name__ == '__main__':
    main()

# EOF