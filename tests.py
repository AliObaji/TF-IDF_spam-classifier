from sklearn.model_selection import train_test_split
from helpers import review_messages, remove_stopWords, stem_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score


def plain(data):
    data['text'] = data['text'].apply(review_messages).apply(remove_stopWords).apply(stem_words)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.1, random_state = 1)
    
    vectorizer = TfidfVectorizer()
    model = build_svm(X_train, y_train, vectorizer)
    
    #Predict test results
   
    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
    
    #return accuracy_score(y_test, y_pred);
    return confusion_matrix(y_test, y_pred)
    
    
def medium_cleanup(data):
    data['text'] = data['text'].apply(review_messages).apply(remove_stopWords)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.1, random_state = 1)
    
    vectorizer = TfidfVectorizer()
    model = build_svm(X_train, y_train, vectorizer)
    
    #Predict test results
    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
        
    #return accuracy_score(y_test, y_pred);
    return confusion_matrix(y_test, y_pred)
    
    
def high_cleanup(data):
    data['text'] = data['text'].apply(review_messages).apply(remove_stopWords).apply(stem_words)
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.1, random_state = 1)
    
    vectorizer = TfidfVectorizer()
    model = build_svm(X_train, y_train, vectorizer)
    
    #Predict test results
    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
        
    #return accuracy_score(y_test, y_pred);
    return confusion_matrix(y_test, y_pred)
    
    
    
def build_svm(x_train, y_train, vectorizer):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn import svm

    x_train = vectorizer.fit_transform(x_train)

    svm = svm.SVC(C=1000)
    svm.fit(x_train, y_train)
    
    return svm;