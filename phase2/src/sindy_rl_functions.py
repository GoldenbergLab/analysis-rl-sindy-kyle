import numpy as np
import pysindy as ps
from pysindy.feature_library import GeneralizedLibrary, FourierLibrary, PolynomialLibrary, CustomLibrary, TensoredLibrary
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

def SINDYlibrary(bandits, rw=False):

    # Get feature names for SINDYc model
    qvalnames = []; choicenames = []
    for x in range(bandits):
        qvalnames.append('qvals[%s]' % x) # we want feature names for each qval (only one for single bandit)
        choicenames.append('choice[%s]' % x) # we want feature names for each possible choice (only one for single bandit)
    feat_names = qvalnames + ['rewards', 'time'] + choicenames # these are all of the features we will have SINDy consider
    print(feat_names)

    # Define all of the functions you want to consider (Don't worry about what features pair with which
    # functions, we'll handle that later, but list all of the needed functions)
    library_functions = [
        lambda x: x,
        lambda x: np.power(x, 2),
        lambda x,y: x*y,
        lambda x: np.exp(-x/30),
        lambda x: np.exp(-x/20),
        lambda x: np.exp(-x/10)
    ]

    library_functions_rw = [
        lambda y,x: x-y,
    ]

    # List all of the names that pair with each function
    library_names = [
        lambda x: x,
        lambda x: x+'^2',
        lambda x,y: x+'*'+y,
        lambda x: "exp(-" + x + "/30)",
        lambda x: "exp(-" + x + "/20)",
        lambda x: "exp(-" + x + "/10)"
    ]

    library_names_rw = [
        lambda y,x: '('+x+'-'+y+')',
    ]

    # Define the features that pair with each function (IMPORTANT: this needs to be modified if you include more bandits, features, functions, etc.)
    # There is one row for each library/function, and there is one column for each possible variable
    # Note that with single bandit, we always drop variable 3, choice, because it is always 1. Choice becomes important for multi-bandit problems
    library_features = np.array([
        [0, 1, 2, 2], # this means that we're considering the first function for variables 0, 1, and 2
        [0, 0, 2, 2], # this means we're considering the second function for only variables 0 and 2
        [0, 1, 2, 2], # same variables as first row but for the third function
        [2, 2, 2, 2], # consider only variable 2 for the fourth function
        [2, 2, 2, 2], # same as above, but for fifth function
        [2, 2, 2, 2]
    ])
    library_features_rw = np.array([
        [0, 1, 1, 1],
    ])

    # Here is an example array for a 2-armed bandit; SINDy will not use this if you are using a single bandit
    library_features_2bandit = np.array([
        [0, 1, 2, 3, 3],
        [0, 0, 1, 1, 3],
        [0, 1, 2, 3, 4],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
    ])
    library_features_2bandit_rw = np.array([
        [0, 1, 1, 1, 1],
    ])

    # Here is an example array for a 3-armed bandit; SINDy will not use this if you are using a single bandit
    library_features_3bandit = np.array([
        [0, 1, 2, 3, 4, 4, 4],
        [0, 0, 1, 1, 2, 2, 4],
        [0, 1, 2, 3, 4, 5, 6],
        [4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4, 4, 4],
    ])
    library_features_3bandit_rw = np.array([
        [0, 1, 1, 1, 1, 1, 1],
    ])

    # Select the inputs_per_library array conditional on the number of bandits
    if rw == False:
        if bandits == 1:
            inputs_per_library = library_features
        elif bandits == 2:
            inputs_per_library = library_features_2bandit
        elif bandits == 3:
            inputs_per_library = library_features_3bandit
    else:
        if bandits == 1:
            inputs_per_library = library_features_rw
        elif bandits == 2:
            inputs_per_library = library_features_2bandit_rw
        elif bandits == 3:
            inputs_per_library = library_features_3bandit_rw

    # Put all of the libraries into one list, and then use that list of libraries and their feature pairings for a single generalized library
    libraries = []
    for i, x in enumerate(library_functions):
        libraries.append(ps.CustomLibrary(library_functions=[x], function_names=[library_names[i]]))
    libraries_rw = []
    for i, x in enumerate(library_functions_rw):
        libraries_rw.append(ps.CustomLibrary(library_functions=[x], function_names=[library_names_rw[i]]))


    # Code below is commented out, but uncomment if you want to tensor libraries
    tensor_array = [
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1]
    ]

    if rw == False:
        generalized_library = ps.GeneralizedLibrary(
            libraries,
            tensor_array=tensor_array,
            inputs_per_library=inputs_per_library,
        )
    else:
        generalized_library = ps.GeneralizedLibrary(
            libraries_rw,
            inputs_per_library=inputs_per_library,
        )

    return generalized_library, feat_names 

def testSINDy(qvalslist, control_inputs_training, generalized_library, feat_names, sub=None, testing=None, r2min=None, terms=None, half=[], rw=False, runs=1):
    for loop in range(runs): # run tests for x number of times, according to the runs keyword
        test = [i for i, x in enumerate(qvalslist)]
        # First we have to remove any missing data from our qvals and control variables. This includes qvals recorded during
        # attention checks. Ideally, we could include controls that precede these checks, but doing so creates an unequal 
        # number of controls compared to qvals, and SINDy returns errors when this happens (it would be great to find a
        # solution to this)
        qvalslist_nonan = [x[~np.isnan(x)] for i, x in enumerate(qvalslist) if i in test]
        control_inputs_training_nonan = [np.array([x for x in control_inputs_training[y] if ~np.isnan(x[0])]) for y in test]

        if len(half) == 1:
            if half[0] == 0:
                for x in range(0, len(qvalslist_nonan)):
                    qvalslist_nonan[x] = qvalslist_nonan[x][:-int(len(qvalslist_nonan[x])/2)]
                    control_inputs_training_nonan[x] = control_inputs_training_nonan[x][:-int(len(control_inputs_training_nonan[x])/2)]
            elif half[0] == 1:
                for x in range(0, len(qvalslist_nonan)):
                    qvalslist_nonan[x] = qvalslist_nonan[x][int(len(qvalslist_nonan[x])/2):]
                    control_inputs_training_nonan[x] = control_inputs_training_nonan[x][int(len(control_inputs_training_nonan[x])/2):]
        
        # Second, we need to choose a random subject for testing, if one hasn't already been specifed with the sub keyword
        if sub == None:
            leaveone = np.random.randint(0, len(qvalslist_nonan))
        else:
            leaveone = test.index(sub)

        # Third, we separate our training and testing data using the chosen subject as an index
        X_test = qvalslist_nonan[leaveone] # testing qvals
        U_test = control_inputs_training_nonan[leaveone] # testing control variables
        if testing == True:
            X_train = [x for i,x in enumerate(qvalslist_nonan) if i!=leaveone] # training qvals
            U_train = [x for i,x in enumerate(control_inputs_training_nonan) if i!=leaveone] # training control variables
        else:
            X_train = [x for i,x in enumerate(qvalslist_nonan)]
            U_train = [x for i,x in enumerate(control_inputs_training_nonan)]
            
        # Fourth, we specify the optimizer that we want to use. We could alternatively use STLSQ and specify thresholds for
        # the L2 regularized coefficients, but I've found that SSR tends to give better fitting models
        optimizer = ps.SSR(max_iter=1000, criteria="model_residual", normalize_columns=True, 
                           fit_intercept=False, verbose=False, alpha=0.2) # using a greedy algorithm here to hopefully find the most parsimonious model

        # Sixth, we bring together the generalized library, our optimizer, a differentiation method and the names of the
        # features in our library to form the SINDy model
        model = ps.SINDy(feature_library = generalized_library,
                         optimizer=optimizer,
                         differentiation_method = FiniteDifference(order=1),
                         feature_names=feat_names)

        # Seventh, we fit the model to our training qvals and control variables
        model.fit(X_train,
                  u=U_train,
                  t=1,
                  multiple_trajectories=True,
                 )
        
        r2=0; idx=0; r2cap=r2min
        t = np.linspace(0, len(X_train[0]), len(X_train[0]))
        
        # Eigth, we make some choices as to what model we want from SINDy. If a specific number of terms were specified, 
        # we choose the model with the x most important terms. Otherwise, we choose a model that minimizes complexity while
        # striding for some minimum fit. The r2min keyword specifies this minimum fit. What happens is that we iterate
        # backwards over the most to least parsimonious model according to the optimizer, and we choose the first model that
        # meets the minimum fit criterion. If no model meets this criterion, we slightly decrease the criterion and restart
        # the loop at the most parsimonious model. What we get as a result of this is the most parsimonious model that meets
        # some threshold for fit, and if no such model exists, we get the most parsimonious model that maximizes fit.
        if terms != None:
            optimizer.coef_ = np.asarray(optimizer.history_)[len(optimizer.history_)-terms, :, :]
            print('MSE of model coefficients: %.4f' % np.asarray(optimizer.err_history_)[len(optimizer.err_history_)-terms])
            if testing == True:
                r2 = model.score(X_test, u=U_test, t=t[1]-t[0])
        elif r2min != None and testing == True:
            while r2 < r2cap:
                try:
                    idx+=1
                    optimizer.coef_ = np.asarray(optimizer.history_)[len(optimizer.history_)-idx, :, :]
                    print('MSE of model coefficients: %.4f' % np.asarray(optimizer.err_history_)[len(optimizer.err_history_)-idx])
                    r2 = model.score(X_test, u=U_test, t=t[1]-t[0], metric=metrics.r2_score)
                except:
                    r2cap-=0.01
                    idx=0
        else:
            idx = 0
            changemse = [x-optimizer.err_history_[i-1] if i!=0 else 0 for i, x in enumerate(optimizer.err_history_)]
            changemse.reverse()
            for i, x in enumerate(changemse):
                idx += 1
                if x < 0.05*optimizer.err_history_[-1-i]:
                    break
            optimizer.coef_ = np.asarray(optimizer.history_)[len(optimizer.history_)-idx, :, :]
            print('MSE of model coefficients: %.4f' % np.asarray(optimizer.err_history_)[len(optimizer.err_history_)-idx])
            if testing == True:
                r2 = model.score(X_test, u=U_test, t=t[1]-t[0])
                    
        # Ninth, we print out some information about each test run
        if testing == True:
            print("\nTest subject = %s; R^2 = %s" % (test[leaveone], r2))
        else:
            print("\nAll subjects, N=%s" % (len(test)))
        model.print()
        print(model.score(X_train, u=U_train, t=t[1]-t[0], metric=metrics.r2_score, multiple_trajectories=True))
    return model