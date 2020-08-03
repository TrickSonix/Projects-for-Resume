import numpy as np

class LogisticRegression():

    def __init__(self, penalty='l2', alpha=0.0001, C = 1.0, max_iter=100, random_state=None):
        self.penalty = penalty
        self.alpha = alpha
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_ = None
        self.iterations = None
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        np.random.seed(self.random_state)
        W = np.reshape(np.random.randn(X.shape[1]), (X.shape[1], -1))
        n = X.shape[0]
        if self.penalty == 'l2':
            for i in range(1, self.max_iter+1):
                y_pred = self.sigmoid(np.dot(X, W))
                new_W = W - 1/i*(1/n*np.dot(X.T, y_pred-y) + 1/self.C*W)
                norm = np.linalg.norm(W-new_W, ord=2)
                if norm < self.alpha:
                    self.iterations = i
                    break
                W = new_W
            if not self.iterations:
                self.iterations = self.max_iter
            self.coef_ = W
        elif self.penalty == 'l1':
            for i in range(1, self.max_iter+1):
                y_pred = self.sigmoid(np.dot(X, W))
                new_W = W - 1/i*(1/n*np.dot(X.T, y_pred-y) + 1/self.C*np.sign(W))
                norm = np.linalg.norm(W-new_W, ord=2)
                if norm < self.alpha:
                    self.iterations = i
                    break
                W = new_W
            if not self.iterations:
                self.iterations = self.max_iter
            self.coef_ = W
        else:
            for i in range(1, self.max_iter+1):
                y_pred = self.sigmoid(np.dot(X, W))
                new_W = W - 1/i*(1/n*np.dot(X.T, y_pred-y))
                norm = np.linalg.norm(W-new_W, ord=2)
                if norm < self.alpha:
                    self.iterations = i
                    break
                W = new_W
            if not self.iterations:
                self.iterations = self.max_iter
            self.coef_ = W

    def predict(self, X):
        if isinstance(self.coef_, np.ndarray):
            return self.sigmoid(np.dot(X, self.coef_))
        else:
            print('Fit model first!')

class PolynomialFeatures():

    def __init__(self, degree=2):
        self.degree = degree
        self.n_input_features = None
        self.n_output_features = None

    def combinations_with_replacement(self, iterable, r):
        # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
        pool = tuple(iterable)
        n = len(pool)
        if not n and r:
            return
        indices = [0] * r
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != n - 1:
                    break
            else:
                return
            indices[i:] = [indices[i] + 1] * (r - i)
            yield tuple(pool[i] for i in indices)
            

    def fit(self, X):
        self.n_input_features = X.shape[1]
        self.n_output_features = np.sum([len(list(self.combinations_with_replacement(range(self.n_input_features), d))) for d in range(self.degree+1)])

    def fit_transform(self, X):
        samples, features = X.shape
        self.n_input_features = features
        self.n_output_features = np.sum([len(list(self.combinations_with_replacement(range(self.n_input_features), d))) for d in range(self.degree+1)])
        result = np.ones((samples, 1))
        result = np.hstack((result, X))
        if self.degree == 1:
            return result
        for d in range(2, self.degree+1):
            for comb in self.combinations_with_replacement(range(features), d):
                out_col = 1
                for index in comb:
                    out_col = np.multiply(X[:, index], out_col)
                out_col = out_col.reshape(-1, 1)
                result = np.hstack((result, out_col))
        return result

class StandardScaler():
    def __init__(self):
        self.mean_list = None
        self.std_list = None
        self.n_features = None

    def fit(self, X):
        self.n_features = X.shape[1]
        self.mean_list = np.mean(X, axis=0)
        self.std_list = np.std(X, axis=0)

        for index, std in enumerate(self.std_list):
            if std == 0:
                self.std_list[index] = 1
                self.mean_list[index] = 0


    def transform(self, X):
        if X.shape[1] != self.n_features:
            raise ValueError('Number of features does not match the training data.')
        
        to_stack = []

        for row in X:
            to_stack.append((row-self.mean_list)/self.std_list)
        
        result = np.vstack(to_stack)
        return result

    def fit_transform(self, X):
        self.n_features = X.shape[1]
        self.mean_list = np.mean(X, axis=0)
        self.std_list = np.std(X, axis=0)

        for index, std in enumerate(self.std_list):
            if std == 0:
                self.std_list[index] = 1
                self.mean_list[index] = 0
        
        to_stack = []

        for row in X:
            to_stack.append((row-self.mean_list)/self.std_list)
        
        result = np.vstack(to_stack)
        return result

class GridSearch():

    def __init__(self, estimator, metric, param_grid):
        self.estimator = estimator
        self.metric = metric
        self.param_grid = param_grid

    def product(self, *args, repeat=1):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = [tuple(pool) for pool in args] * repeat
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def confusion_matrix(self, actual, predicted):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        actual = actual.flatten()
        predicted = predicted.flatten()
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                if actual[i] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if actual[i] == 1:
                    fn += 1
                else:
                    fp += 1
        return tp, fp, fn, tn


    def f1_score(self, actual, predicted):
        tp, fp, fn, tn = self.confusion_matrix(actual, predicted)
        if tp != 0 and fp != 0:
            precision = tp/(tp + fp)
        else:
            precision = None
        if tp != 0 and fn != 0:
            recall = tp/(tp + fn)
        else:
            recall = None

        if recall and precision:
            if recall != 0 and precision != 0:
                return 2*recall*precision/(precision+recall)
            else:
                return 0
        else:
            return 0

    def roc_auc_score(self, actual, predicted):
        actual = actual.flatten()
        predicted = predicted.flatten()
        roc_auc_curve_x = [0]
        roc_auc_curve_y = [0]
        dx = 1/len(np.where(actual==0)[0])
        dy = 1/len(np.where(actual==1)[0])
        prev_value = None
        prev_label = None
        coef_x = 0
        coef_y = 0
        for label, value in sorted(zip(actual, predicted), key=(lambda x: x[1]), reverse=True):
            if prev_value:
                if value == prev_value:
                    if label == 1:
                        coef_y += 1
                    else:
                        coef_x += 1
                    continue
                else:
                    if prev_label == 1:
                        coef_y += 1
                    else:
                        coef_x += 1
                roc_auc_curve_x.append(roc_auc_curve_x[-1] + coef_x*dx)
                roc_auc_curve_y.append(roc_auc_curve_y[-1] + coef_y*dy)
            prev_value = value
            prev_label = label
            coef_x = 0
            coef_y = 0
        roc_auc_curve_x.append(1)
        roc_auc_curve_y.append(1)
        return np.trapz(roc_auc_curve_y, roc_auc_curve_x)

    def search(self, X_train, X_test, y_train, y_test):
        param_lists = []
        param_names = []
        for param in self.param_grid:
            param_lists.append(self.param_grid[param])
            param_names.append(param)

        param_grid_list = []
        for prod in self.product(*param_lists):
            param_dict = {}
            for name, value in zip(param_names, prod):
                param_dict[name] = value
            param_grid_list.append(param_dict)

        best_params = None
        best_metric = 0
        metric_func = getattr(self, self.metric)
        for element in param_grid_list:
            if self.estimator == 'LogisticRegression':
                model = LogisticRegression(**element)
            elif self.estimator == 'RandomForestClassifier':
                model = RandomForestClassifier(**element)
            model.fit(X_train, y_train)
            if self.metric != 'roc_auc_score':
                y_pred = np.round(model.predict(X_test))
            else:
                y_pred = model.predict(X_test)
            curr_metric = metric_func(y_test, y_pred)
            if curr_metric > best_metric:
                best_metric = curr_metric
                best_params = element
        
        return best_metric, best_params

class Node():
    def __init__(self, index, t, true_branch, false_branch):
        self.index = index
        self.t = t
        self.true_branch = true_branch
        self.false_branch = false_branch

class Leaf():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.prediction = self.predict()

    def predict(self):
        classes = {}
        sum_classes = 0
        for label in self.labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
            sum_classes += 1
        prediction = classes[max(set(self.labels))]/sum_classes
        return prediction

class RandomForestClassifier():

    def __init__(self, n_estimators=100, max_depth=None, max_features=None, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.forest = None

    def get_bootstrap(self, data, labels):
        np.random.seed(self.random_state)

        n_samples = data.shape[0]
        bootstrap = []

        for _ in range(self.n_estimators):
            b_data = np.zeros(data.shape)
            b_labels = np.zeros(labels.shape)

            for i in range(n_samples):
                index = np.random.randint(0, n_samples)
                b_data[i] = data[index]
                b_labels[i] = labels[index]
            
            bootstrap.append((b_data, b_labels))
        
        return bootstrap

    def get_subsample(self, len_sample):
        np.random.seed(self.random_state)
        sample_indexes = [i for i in range(len_sample)]

        np.random.shuffle(sample_indexes)
        if not self.max_features:
            len_subsample = int(np.sqrt(len_sample))
        else:
            len_subsample = self.max_features
        subsample_indexes = []
        for _ in range(len_subsample):
            subsample_indexes.append(sample_indexes.pop())

        return subsample_indexes
    
    def gini(self, labels):
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        
        impurity = 1
        for label in classes:
            p = classes[label]/len(labels)
            impurity -= p**2
        
        return impurity

    def quality(self, left_labels, right_labels, current_gini):
        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])

        return current_gini - p*self.gini(left_labels) - (1-p)*self.gini(right_labels)

    def split(self, data, labels, index, t):
    
        left = np.where(data[:, index] <= t)
        right = np.where(data[:, index] > t)
            
        true_data = data[left]
        false_data = data[right]
        true_labels = labels[left]
        false_labels = labels[right]
            
        return true_data, false_data, true_labels, false_labels

    def find_best_split(self, data, labels):

        current_gini = self.gini(labels)

        best_quality = 0
        best_t = None
        best_index = None

        subsample = self.get_subsample(data.shape[1])

        for index in subsample:
            t_values = np.unique([row[index] for row in data])

            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)
                if len(true_data) < self.min_samples_leaf or len(false_data) < self.min_samples_leaf:
                    continue
                current_quality = self.quality(true_labels, false_labels, current_gini)

                if current_quality > best_quality:
                    best_quality, best_t, best_index = current_quality, t, index

        return best_quality, best_t, best_index
    
    def build_tree(self, data, labels, current_depth=1):

        if self.max_depth:
            if current_depth >= self.max_depth:
                return Leaf(data, labels)

        quality, t, index = self.find_best_split(data, labels)

        if quality == 0:
            return Leaf(data, labels)     

        true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)

        true_branch = self.build_tree(true_data, true_labels, current_depth+1)
        false_branch = self.build_tree(false_data, false_labels, current_depth+1)

        return Node(index, t, true_branch, false_branch)
        
    def fit(self, X, y):
        self.forest = []
        bootstrap = self.get_bootstrap(X, y)

        for b_data, b_labels in bootstrap:
            self.forest.append(self.build_tree(b_data, b_labels))

    def classify_obj(self, obj, node):

        if isinstance(node, Leaf):
            return node.prediction

        if obj[node.index] <= node.t:
            return self.classify_obj(obj, node.true_branch)
        else:
            return self.classify_obj(obj, node.false_branch)

    def predict(self, X):

        if not isinstance(self.forest, list):
            print("Fit model first")
            return None

        predictions = []

        for tree in self.forest:
            classes = []
            for row in X:
                classes.append(self.classify_obj(row, tree))
            predictions.append(classes)

        voted_predictions = []
        for obj in zip(*predictions):
            voted_predictions.append(np.mean(obj))
        
        return np.array(voted_predictions)




        
