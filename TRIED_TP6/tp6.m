load('iris_cls.mat','-ascii')
load('iris_don.mat','-ascii')
% load('iris_cls_wrong.mat','-ascii')

varnames = {'sepal height','sepal width','petal height','petal width'}
clsnames = {'setosa','versicolor','virginica'}
style = {'b*','g*','r*'}
affby2(iris_don,iris_cls,varnames,style,iris_cls_wrong)
legend(clsnames)