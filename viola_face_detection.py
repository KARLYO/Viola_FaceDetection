import cv2
from sklearn.feature_selection import SelectPercentile, chi2, f_classif

import glob as gb

import numpy as np 

alphas=[]
clfs=[]

def Integral(image):
    integral_matrix = np.zeros(image.shape)
    sum_matrix = np.zeros(image.shape)
    
    for y in range(len(image)):
        for x in range(len(image[y])):
            sum_matrix[y,x] = sum_matrix[y-1,x] + image[y,x] if y-1 >= 0 else image[y,x]
            integral_matrix[y,x] = integral_matrix[y,x-1]+sum_matrix[y,x] if x-1 >= 0 else sum_matrix[y,x]
    return integral_matrix


class rect:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    def rect_integral(self, image):
     ii=Integral(image)
     A=ii[self.y,self.x] 
     ABCD=ii[self.y+self.height,self.x+self.width]
     AC=ii[self.y+self.height,self.x]
     AB=ii[self.y,self.x+self.width]
     return ABCD-AB-AC+A



def build_features(Height,Width):
    height=Height
    width=Width
    number=[0]*5
    features = []
    for w in range(1, 9):
        for h in range(1, 9):
            i = 0
            while i + w < width:
                j = 0
                while j + h < height:
                    
                    immediate = rect(i, j, w, h)
                    right = rect(i+w, j, w, h)
                    if i + 2 * w < width:  
                        features.append(([immediate], [right]))#two vertical
                        number[0]=number[0]+1

                    bottom = rect(i, j+h, w, h)
                    if j + 2 * h < height: 
                        features.append(([bottom], [immediate])) #two horizontal
                        number[1]=number[1]+1
                    
                    
                    
                    bottom_2 = rect(i, j+2*h, w, h) #threee horizontal
                    if j + 3 * h < height: 
                        features.append(([bottom_2, immediate],[bottom]))
                        number[2]=number[2]+1   
                        
                        
                        
                        
                    right_2 = rect(i+2*w, j, w, h) #three vertical
                    if i + 3 * w < width: 
                        features.append(( [right_2, immediate],[right]))
                        number[3]=number[3]+1
                    
                    
                   
                    bottom_right = rect(i+w, j+h, w, h) #four
                    if i + 2 * w < width and j + 2 * h < height:
                        features.append(( [immediate, bottom_right],[right, bottom]))
                        number[4]=number[4]+1
                    j += 1
                i += 1
    return features,number

def apply_features(features, data):
    feature_data_table = np.zeros((len(data),len(features)))
   
    
    for i in range(len(features)):
        positive_regions=features[i][0]
        negative_regions=features[i][1]
        for j in range(len(data)): 
         pos_sum=0
         neg_sum=0
         for pos in positive_regions:
            pos_sum=pos_sum+pos.rect_integral(data[j])
         for neg in negative_regions:
            neg_sum=neg_sum+neg.rect_integral(data[j])
         feature=pos_sum-neg_sum
         feature_data_table[j,i] = feature
        
    return feature_data_table

 



class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    def classify(self, image):
        pos_sum=0
        neg_sum=0
        for pos in self.positive_regions:
            pos_sum=pos_sum+pos.rect_integral(image)
        for neg in self.negative_regions:
            neg_sum=neg_sum+neg.rect_integral(image)
        feature=pos_sum-neg_sum
        if self.polarity * feature < self.polarity * self.threshold:
          return 1 
        else:
          return 0



def train_weak(features,X,label,weights):
    
    pos_weights, neg_weights = 0, 0
    for i in range(len(label)):
     if label[i] == 1:
        pos_weights += weight[i]
     else:
        neg_weights += weight[i]
    classifiers = []
    total_features = X.shape[1]
    for i in range(total_features):
        feature=X[:,i]
        sorted_feature = sorted(zip(weights, feature, label), key=lambda x: x[1])
        pos_number, neg_number = 0, 0
        pos_weights_now, neg_weight_now = 0, 0
        min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
        for w, f, l in sorted_feature:
            error = min(neg_weight_now + pos_weights - pos_weights_now, pos_weights_now + neg_weights - neg_weight_now)
            if error < min_error:
                min_error = error
                best_feature = features[i]
                best_threshold = f
                best_polarity = 1 if pos_number > neg_number else -1
            if l == 1:
                pos_number += 1
                pos_weights_now += w
            else:
                neg_number += 1
                neg_weight_now += w
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
    return classifiers

def get_lowest_error_classsifier(classifiers,label, weights,data):
    best_clf, best_error, best_accuracy = None, float('inf'), None
    for clf in classifiers:
        error, accuracy = 0, []
        for i in range(len(data)):
            correctness = abs(clf.classify(data[i]) - label[i])
            accuracy.append(correctness)
            error += weights[i] * correctness
        error = error / len(data)
        if error < best_error:
            best_clf, best_error, best_accuracy = clf, error, accuracy
    return best_clf, best_error, best_accuracy


def train( data,label):
    T=1
    alphas=[]
    clfs=[]
    p_n,n_n=0,0
    for i in range(len(label)):
     if label[i] == 1:
        p_n += 1
     else:
        n_n += 1
    weights = np.zeros(len(data))
    
    for x in range(len(data)):
        
        if label[x] == 1:
            weights[x] = 1.0 / (2 * p_n)
        else:
            weights[x] = 1.0 / (2 * n_n)
    features,number =build_features(data[0].shape[0],data[0].shape[1])
    
    X = apply_features(features, data)
    indices = SelectPercentile(f_classif, percentile=2).fit(X, label).get_support(indices=True)
    X = X[:,indices]
    features = features[indices]
    for t in range(T):
        weights = weights / np.linalg.norm(weights)
        weak_clfs = train_weak( features,X,label, weights)
        clf, error, accuracy = get_lowest_error_classifier(weak_clfs,label,weights,data)
        beta = error / (1.0 - error)
        for i in range(len(accuracy)):
            weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
        alpha = math.log(1.0/beta)
        alphas.append(alpha)
        clfs.append(clf)


def strong_classify(image):
    total = 0
    
    for alpha, clf in zip(alphas, clfs):
        total += alpha * clf.classify(image)
    return 1 if total >= 0.5 * sum(alphas) else 0


def show_strong_classifier(data,label):
    acc=0
    false_p=0
    for i in range(len(data)):
        if strong_classify(data[i])==label[i]:
           
            acc=acc+1
        elif label[i]==1:
            false_p=false_p+1
    print("the total is: ",len(data))
    print("the accurancy is: ",acc)
    print("the  false postive is: ",false_p)
    print("the flase negative is:",len(data)-acc-flase_p)
    for clf in clfs:
     for pos in clf[0]:
        print(pos.x,",",pos.y,",",pos.width,",",pos.height)
     for neg in clf[1]:
       print(neg.x,",",neg.y,",",neg.width,",",neg.height)


def main():
    train_image=[]
    test_image=[]
    label=[]
    label2=[]
    face_path = gb.glob("D:\\dataset\\trainset\\faces\\*.png") 
    noface_path=gb.glob("D:\\dataset\\trainset\\non-faces\\*.png")
    for path in face_path:
     img  = cv2.imread(path) 
     img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
     train_image.append(img_gray)
     label.append(1)
    for path in noface_path:
     img=cv2.imread(path)
     img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
     train_image.append(img_gray)
     label.append(0)
    train(train_image,label)

    face_path = gb.glob("D:\\dataset\\testset\\faces\\*.png") 
    noface_path=gb.glob("D:\\dataset\\testset\\non-faces\\*.png")
    for path in face_path:
     img  = cv2.imread(path) 
     img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
     test_image.append(img_gray)
     label2.append(1)
    for path in noface_path:
     img=cv2.imread(path)
     img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
     test_image.append(img_gray)
     label2.append(0)
    
    show_strong_classifier(test_image,label2)




if __name__ == "__main__":
    main() 

