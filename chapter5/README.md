逻辑回归(Logistic regression)
===

[逻辑回归](http://52opencourse.com/125/coursera%E5%85%AC%E5%BC%80%E8%AF%BE%E7%AC%94%E8%AE%B0-%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AC%AC%E5%85%AD%E8%AF%BE-%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92-logistic-regression)的主要思想是：根据现有数据对分类边界线建立回归公式，以此进行分类。

此处，目标函数采用对数似然函数，优化算法采用[梯度上升算法](http://www.cnblogs.com/hitwhhw09/p/4715030.html)

> 当然如果你的目标函数也可以采用[交叉熵](http://blog.csdn.net/u012162613/article/details/44239919)，那么优化算法就可采用梯度下降算法。


    [[ 4.12414349]
     [ 0.48007329]
     [-0.6168482 ]]


![]('./logRegression.png')


