from treewalk.treewalk_extraction import extract_features
from treewalk.treewalk_classification import trainning_classifier, testing_classifier

def main():
	dirpath = '../BaseDatos/Treewalk/Seg_Trainning/flower/'

	desc_SVM_input, classInput = extract_features(dirpath)
	trainned_classifier_list = trainning_classifier(desc_SVM_input, classInput)
	
	dirpath_test = '../BaseDatos/Treewalk/Seg_Test/flower/'
	desc_SVM_test, classTest  = extract_features(dirpath_test)
	testing_classifier(trainned_classifier_list, desc_SVM_test, classTest)

if __name__ == '__main__':
	main()