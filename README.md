# Machine Learning Course from Coursera
This repository consist of assignments of Machine learning course from [Coursera](https://www.coursera.org/learn/machine-learning/). Programming language used for these assignments are implemented in [Octave](https://www.gnu.org/software/octave/). However source codes can be executed on Matlab. Synatx for matlab and Octave are almost same.

## Programming Assignment - 1
This assignment consist of implementation of Linear regression on data. 


```Source code
ex1.m file contain the implementation of assignment evaluation script. 
It calls other script and displays implementation of different modules.
```

**1. warmUpExercise.m**
	- Generates identity matrix of size 5. <br/>
	- 	```	1   0   0   0   0  
			0   1   0   0   0  
			0   0   1   0   0   
			0   0   0   1   0   
			0   0   0   0   1   
		```

**2. plotData.m**
	
	- Plot input data to visualize the structure that might help in modelling the parameters.
	  This scripts read data from file and plot input feature against output value.

![Input Data](machine-learning-ex1/img/inputData.jpg)

**3. computeCost.m**
	
	- While performing gradient descent algorithm, visualizing cost function gives better understanding of data convergence.
	- Linear regression algorithm computes cost by taking square of difference of predicted value and actual value
		(h(theta) - y)^2
	- **computeCost.m** script implements calculation of cost function for linear regression.

**4. gradientDescent.m**
	
	- Gradient descent algorithm tries to minimize the cost function calculated in above steps,
	  by optimizing the unknown parameters.
	- This function implemets gradient descent algorithm.

**5. Visualization**
	
	- With every iteration of Gradient descent algorithm, cost function decreases with specified learning rate.
	- Learning rate can be fixed or variable, mostly depends on the kind of data.
	- learning rate of gradient descent can be set by playing around with it.
	- Below shows the expected phenomenon occured while training data.
![Input Data](machine-learning-ex1/img/surfPlot.jpg)
![Input Data](machine-learning-ex1/img/contourPlot.jpg)

```Source code
ex1_multi.m file contain the implementation of assignment evaluation script for multi feature data.
It calls other script and displays implementation of different modules.
```
**1. featureNormalize.m**
	
	- For training data with mutiple variable, feature nomalization brings all input feature in some ranges.
	- Gradient descent to work better for normalized data and convergence becomes faster.
	- This scripts performs normalization of data using (x-mu)/sigma formula. mu = average, sigma = (max val - min val)

**2. computeCostMulti.m**
	
	- Computes cost for multiple features in training set.
	- Vectorized implementation.

**3. gradientDescentMulti.m**
	
	- Vectorized Implemntation of gradient descent algorithm for multiple feature training data.

**4. Visualization**
	
	- With every iteration of Gradient descent algorithm, cost function decreases with specified learning rate.
	- Below shows the expected phenomenon occured while training data with multiple feature.
![Input Data](machine-learning-ex1/img/gradDestMulti.jpg)

## Programming Assignment - 2
This assignment consist of implementation of Logistic regression on data. Logistic regression is an algorithm to classify data into two or multiple classes. 


```Source code
ex2.m file contain the implementation of assignment evaluation script. 
It calls other script and displays implementation of different modules.
```

**1. plotData.m**
	
	- Plot input data to visualize the structure that might help in modelling the parameters.
	  This scripts read data from file and plot input feature against output value.

![Input Data](machine-learning-ex2/img/dataVisualize.jpg)

**2. sigmoid.m**
	
	- Given a input in the form of vector, matrix, scalar .. this scipt computes sigmoid value.
	- Sigmoid function can ve calculated using the formula sig(z) = 1/1+exp(-z).
	- Sigmoid value is used to classify data into two classes. As properties of sigmoid function, its value is close to zero for
	  negative input paramters and close to one for positive input value.
	- Classification done by rule, if z >=0 , sig(z) >= 0.5, in this case input feature is classified as positive class.
	- Classification done by rule, if z < 0 , sig(z) < 0.5, in this case input feature is classified as Negative class.

**3. costFunction.m**
	
	- Cost function implements the cost calculation based on formula: 
![Input Data](machine-learning-ex2/img/costfunc.png)
	- gradient has to be calculated, which is used to minimize cost function and optimize logistic regression paramaters during
	  gradient descent.
![Input Data](machine-learning-ex2/img/gradient.png)

**4. plotDecisionBoundary.m**
	
	- fminc function is used to train the logistic regression classifier.
	- It calls cost function and computes cost and gradient.
	- Optimize theta value to get the better decision boundary.
	- Below is decision boundary after gradient descent completes computing theta,
![Input Data](machine-learning-ex2/img/decisionBoundary.jpg)

```Source code
ex2_reg.m file contain the implementation of assignment evaluation script for multi feature data.
It calls other script and displays implementation of different modules. It evaluates regularized
logistic regression algorithm.
```
**1. Data Visualization**
	
	- Visualize input data.
![Input Data](machine-learning-ex2/img/dataVisualizeReg.jpg)

**2. Feature mapping**
	
	- From the above figure we can see that data is not linearly separable. So the decision boundary doesn't follow a line property.
	- To create a decision boundary which is not linear, input feature has to be mapped to higher dimesion.
	- mapFeature.m script takes input and returns higher dimesional data.

**3. costFunctionReg.m**
	
	- Taking higher dimensional data and running logistic regression algorithm prone to overfitting.
	- To Avoid overfitting, regularization term has to be introduced.
	- By controlling regularization parameters, overfitting problems can be avoided.
	- Following formula has been implemented to compute cost and gradient.
![Input Data](machine-learning-ex2/img/costFuncReg.png)
![Input Data](machine-learning-ex2/img/gradientReg.png)

**4. Decision Boundary Visualization**
	
	- After computing theta values from gradient descent, it gives the idea about decision boundary.
	- Below shows decision boundary
![Input Data](machine-learning-ex2/img/decisionBoundaryReg.jpg)


