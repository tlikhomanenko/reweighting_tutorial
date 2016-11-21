import matplotlib.pyplot as plt
import numpy

from carl.distributions import Join
from carl.distributions import Mixture
from carl.distributions import Normal
from carl.distributions import Exponential
from carl.distributions import LinearTransform
from carl.ratios import ClassifierRatio
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.base import clone 
from sklearn.utils import check_random_state

import corner

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


hist_settings = {'bins': 50, 'normed': True, 'alpha': 0.3}

# a helper function for plotting the reweighted data
def draw_distributions(original, target, original_weights, target_weights=None):
    """
    Plot weighted distributions for original and target samples.

    :param numpy.ndarray original: original samples
    :param numpy.ndarray target: target samples
    :param numpy.array original_weights: weights for original samples
    :param numpy.array target_weights: weights for target samples
    """
    if target_weights is None:
        target_weights = numpy.ones(target.shape[0])
    columns = range(original.shape[1])
    
    if len(columns) > 3:
        plt.figure(figsize=(16, 8))
    else:
        plt.figure(figsize=(16, 4))
    
    for ind, column in enumerate(columns, 1):
        xlim1 = numpy.percentile(numpy.hstack(target[:, column]), [0.01, 99.99])
        xlim2 = numpy.percentile(numpy.hstack(original[:, column]), [0.01, 99.99])
        xlim = (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
        if len(columns) > 3:
            plt.subplot(2, numpy.ceil(len(columns) / 2.), ind)
        else:
            plt.subplot(1, len(columns), ind)
        plt.hist(original[:, column], weights=original_weights, label='original',
                 range=xlim, color='b', **hist_settings)
        plt.hist(target[:, column], range=xlim, weights=target_weights, color='orange', label='target',
                 **hist_settings)
        plt.xlabel('X%d' %(ind))
        plt.legend(loc='best')
    plt.show()
        
        
def generate_samples(with_linear_transformation=False, add_variation=False, n_samples=50000, verbose=True):
    """
    Generate 5 independent variables: two Gaussian, mixture of Gaussian, two exponents. 
    Two Gaussian have different means for original and target distributions.
    
    if with_linear_transformation is True then add linear transformation of generated 5 variables.
    
    if add_variation is True then add random values in variance to obtain gaussian pdf 
    for orignal and target samples not only with different mean but also with different variance.
    
    :param bool with_linear_transformation: apply or not linear transformation for samples features
    :param bool add_variation: make or not different variance for Gaussian distribution for original and target samples.
    :param int n_samples: number of generated samples for original/target distributions. For test samples 2*n_samples will be generated
    :param bool verbose: print and plot additional info during generation.
    
    :return: train original, train target, exact weights for train original, test original, test target, exact weights for test original
    """
    # define linear transformation matrix
    R = make_sparse_spd_matrix(5, alpha=0.5, random_state=7)

    variation_origin, variation_target = (0, 0)
    if add_variation:
        r = check_random_state(42)
        variation_origin, variation_target = r.uniform() / 3., r.uniform() / 3.
        
    p0 = Join(components=[
            Normal(mu=.5, sigma=1 + variation_origin),
            Normal(mu=-.5, sigma=3 + variation_origin),
            Mixture(components=[Normal(mu=-2, sigma=1),
                                Normal(mu=2, sigma=0.5)]),
            Exponential(inverse_scale=3.0),
            Exponential(inverse_scale=0.5)])

    p1 = Join(components=[
            Normal(mu=0, sigma=1 + variation_target),
            Normal(mu=0, sigma=3 + variation_target),
            Mixture(components=[Normal(mu=-2, sigma=1),
                                Normal(mu=2, sigma=0.5)]),
            Exponential(inverse_scale=3.0),
            Exponential(inverse_scale=0.5)])
    
    if with_linear_transformation:
        p0 = LinearTransform(p0, R)
        p1 = LinearTransform(p1, R)
        
    X0 = p0.rvs(n_samples, random_state=777)
    X1 = p1.rvs(n_samples, random_state=777)
    exact_weights = numpy.exp(p0.nll(X0) - p1.nll(X0))
    exact_weights[numpy.isinf(exact_weights)] = 0.
    
    # generate samples to test reweighting rule (to avoid overfitting)
    X0_roc = p0.rvs(2 * n_samples, random_state=777 * 2)
    X1_roc = p1.rvs(2 * n_samples, random_state=777 * 2)
    # Weighted with true ratios
    exact_weights_roc = numpy.exp(p0.nll(X0_roc) - p1.nll(X0_roc))
    exact_weights_roc[numpy.isinf(exact_weights_roc)] = 0.

    if verbose:
        print "Original distribution"
        fig = corner.corner(X0, bins=20, smooth=0.85, labels=["X0", "X1", "X2", "X3", "X4"])
        plt.show()
        print "Target distribution"
        fig = corner.corner(X1, bins=20, smooth=0.85, labels=["X0", "X1", "X2", "X3", "X4"])
        plt.show()
        print "Exact reweighting"
        # In this example, we know p0(x) and p1(x) exactly, 
        #so we can compare the other can compare the approximate reweighting approaches with the exact weights.
        draw_distributions(X0, X1, exact_weights)
    
    return X0, X1, exact_weights, X0_roc, X1_roc, exact_weights_roc


def generate_samples_for_blow_up_demo(n_samples=50000):
    """
    Generate 3 independent Gaussian variables and apply linear transformation to them.
    These Gaussian have different means and different sigmas for target and original distribution.
    
    This is example of samples with regions with high target samples number zero original samples. In this case exact reweighting rule blow up and the same happens for algorithms.
    :param int n_samples: number of generated samples for original/target distributions. For test samples 2*n_samples will be generated
    
    :return: train original, train target, exact weights for train original, test original, test target, exact weights for test original
    """
    p0 = Join(components=[
            Normal(mu=1, sigma=0.7),
            Normal(mu=-1, sigma=0.7),
            Normal(mu=1, sigma=1.5)])

    p1 = Join(components=[
            Normal(mu=0, sigma=0.7),
            Normal(mu=0, sigma=0.7),
            Normal(mu=0, sigma=1.5)])
    
    R = make_sparse_spd_matrix(3, alpha=0.5, random_state=7)
    p0 = LinearTransform(p0, R)
    p1 = LinearTransform(p1, R)
    
    X0 = p0.rvs(n_samples, random_state=777)
    X1 = p1.rvs(n_samples, random_state=777)
    exact_weights = numpy.exp(p0.nll(X0) - p1.nll(X0))
    exact_weights[numpy.isinf(exact_weights)] = 1.
    
    # generate samples to test reweighting rule (to avoid overfitting)
    X0_roc = p0.rvs(2 * n_samples, random_state=777 * 2)
    X1_roc = p1.rvs(2 * n_samples, random_state=777 * 2)
    # Weighted with true ratios
    exact_weights_roc = numpy.exp(p0.nll(X0_roc) - p1.nll(X0_roc))
    exact_weights_roc[numpy.isinf(exact_weights_roc)] = 1.

    draw_distributions(X0, X1, numpy.ones(len(X0)))
    print "Exact weights are used (inf weights are set to 1)"
    draw_distributions(X0, X1, exact_weights)
    
    return X0, X1, exact_weights, X0_roc, X1_roc, exact_weights_roc


def check_reweighting_by_ML_gb(original, target, original_weights, target_weights=None, n_iterations=1):
    """
    Compare multidimentional distributions after reweighting by Gradient Boosting classifier. 
    If reweighting is perfect then classifier cannot distinguish samples and roc auc score will be 0.5
    
    For splot target, when target weights can be negative, we move smaples with negative weights into original samples with positive weights.
    
    :param numpy.ndarray original: original samples
    :param numpy.ndarray target: target samples
    :param numpy.array original_weights: weights for original samples
    :param numpy.array target_weights: weights for target samples
    :param int n_iterations: number of bootstrap iterations (randomly divide into train/test; on test compute auc; repeat this procedure n_iterations times).
    
    :return: fpr, tpr, mean auc, std auc
    """
    if target_weights is None:
        target_weights = numpy.ones(target.shape[0])
        
    aucs = []
    
    data = numpy.concatenate([original, target])
    labels = numpy.array([0] * original.shape[0] + [1] * target.shape[0])
    W = numpy.concatenate([original_weights, target_weights])
    
    for _ in range(n_iterations):
        Xtr, Xts, Ytr, Yts, Wtr, Wts = train_test_split(data, labels, W, train_size=0.51)
        
        original_w = Wtr[Ytr == 0] 
        target_w = Wtr[Ytr == 1] 

        original_w /= numpy.sum(original_w)
        target_w /= numpy.sum(target_w)

        original_tr = Xtr[Ytr == 0]
        target_tr = Xtr[Ytr == 1]
        
        # put target events with negative weights into original samples with -weights
        data_neg = target_tr[target_w < 0]
        weights_neg = -target_w[target_w < 0]
        original_tr = numpy.concatenate((original_tr, data_neg))
        original_w = numpy.concatenate((original_w, weights_neg))
        target_tr = target_tr[target_w >= 0]
        target_w = target_w[target_w >= 0]
        
        Xtr = numpy.concatenate([original_tr, target_tr])
        Ytr = numpy.array([0] * original_tr.shape[0] + [1] * target_tr.shape[0])
        Wtr = numpy.concatenate([original_w, target_w])
    
        clf = GradientBoostingClassifier(n_estimators=50, subsample=0.5).fit(Xtr, Ytr, sample_weight=Wtr)
        proba = clf.predict_proba(Xts)[:, 1]
        aucs.append(roc_auc_score(Yts, proba, sample_weight=Wts))
        
    fpr, tpr, _  = roc_curve(Yts, proba, sample_weight=Wts)
    return fpr, tpr, numpy.mean(aucs), numpy.std(aucs)


def plot_roc(fpr, tpr, auc, auc_std, name=""):
    """
    Plot roc curve.
    """
    plt.plot(fpr, tpr, label=name + "\nAUC=%.3f$\pm$%.3f" % (auc, auc_std))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Discriminator trained with weights')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    
def plot_scatter_weights(exact_weights, weights, title=""):
    """
    Plot scatter between exact weights and reconstructed weights.
    """
    w1 = exact_weights *1. / numpy.mean(exact_weights)
    w2 = weights * 1. / numpy.mean(weights)
    plt.scatter(w1, w2, alpha=0.1)
    plt.plot([0, 10], [0, 10], lw=2, c='r')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('exact weights')
    plt.ylabel('estimated weights')
    plt.title(title)


def run_verbose_info(original_test, target_test, weights_test, exact_weights_test, cv_val=3):
    draw_distributions(original_test, target_test, weights_test)
    plt.show()

    plt.subplot(1, 3, 3)
    h = plt.hist(weights_test, bins=numpy.exp(numpy.linspace(-4, 3, 50)), alpha=0.2, label="estimated")
    h = plt.hist(exact_weights_test, bins=numpy.exp(numpy.linspace(-4, 3, 50)), alpha=0.2, label="exact")
    plt.semilogx()
    plt.legend()
    plt.xlabel("weight")

    plt.subplot(1, 3, 3)
    fpr, tpr, auc, auc_std = check_reweighting_by_ML_gb(original_test, target_test,
                                                        weights_test, n_iterations=cv_val)
    plot_roc(fpr, tpr, auc, auc_std, 'with weights')

    fpr, tpr, auc, auc_std = check_reweighting_by_ML_gb(original_test, target_test,
                                                        weights_test, n_iterations=cv_val)
    plot_roc(fpr, tpr, auc, auc_std, 'with exact weights')

    plt.show()

    
def reconstruct_ratio_using_estimated_pdfs(classifier, classifier_parameters, cv_val=3, with_linear_transformation=False,
                                           add_variation=False, n_samples=50000, verbose=True, inverse_weights=False,
                                           test_by_ML_GB=False):
    """
    Reconstruct weights by discriinative classifiers (calibrated and non-calibrated) 
    from `carl` on the generated samples defined by function `generate_samples`
    """
    original, target, exact_weights, original_test, target_test, exact_weights_test = \
        generate_samples(with_linear_transformation=with_linear_transformation, 
                         add_variation=add_variation, n_samples=n_samples, verbose=verbose)
 
    predicted_weights = []
    for params in classifier_parameters:
        if verbose:
            print "Used parameters ", params
        classifier_clone = clone(classifier)
        classifier_clone.set_params(**params)
        ratio = ClassifierRatio(base_estimator=classifier_clone, random_state=42)

        #reformat X0 and X1 into training data
        X = numpy.vstack((original, target))
        y = numpy.array([1] * original.shape[0] + [0] * target.shape[0])

        # fit the ration
        ratio.fit(X, y)
    
        carl_weights_test = ratio.predict(original_test, log=False)
        carl_weights_test[numpy.isinf(carl_weights_test)] = 0.
        predicted_weights.append(carl_weights_test)
        
        # plot 1d distribution for test sample
        if verbose:
            run_verbose_info(original_test, target_test, carl_weights_test, exact_weights_test, cv_val=cv_val)
    regime = [False]
    if inverse_weights:
        regime = [True, False]
        
    for inverse in regime:
        plt.figure(figsize=(len(classifier_parameters) * 5, 4))
        m = len(predicted_weights)
        for n, (weights, params) in enumerate(zip(predicted_weights, classifier_parameters)):
            plt.subplot(1, m, n + 1)
            if inverse:
                plot_scatter_weights(1. / exact_weights_test, 1. / weights, title="Inverse weights for\n" + str(params))
            else:
                plot_scatter_weights(exact_weights_test, weights, title="Weights for\n" + str(params))
    
    
def GBreweighter_fit(reweighter, reweighter_parameters, original, target, 
                     exact_weights, original_test, target_test, exact_weights_test, verbose=False, print_weights=False):
    predicted_weights = []
    for params in reweighter_parameters:
        if verbose:
            print "Used parameters ", params
        reweighter_clone = clone(reweighter)
        reweighter_clone.set_params(**params)
        reweighter_clone.fit(original, target, numpy.ones(len(original)))
    
        gb_weights_test = reweighter_clone.predict_weights(original_test)
        predicted_weights.append(gb_weights_test)
        
        # plot 1d distribution for test sample
        if verbose:
            run_verbose_info(original_test, target_test, gb_weights_test, exact_weights_test, cv_val=cv_val) 
        if print_weights:
            print "weights ", numpy.sort(gb_weights_test)
            # plot 1d distribution for test sample
            draw_distributions(original_test, target_test, gb_weights_test)
    plt.figure(figsize=(len(reweighter_parameters) * 5, 4))
    m = len(predicted_weights)
    for n, (weights, params) in enumerate(zip(predicted_weights, reweighter_parameters)):
        plt.subplot(1, m, n + 1)
        plot_scatter_weights(exact_weights_test, weights, title="Weights for\n" + str(params))
    
    
def reconstruct_weights_by_GBreweighter(reweighter, reweighter_parameters, with_linear_transformation=False,
                                        add_variation=False, n_samples=50000, verbose=True):
    """
    Reconstruct weights by BDT reweighter from `hep_ml` on the generated 
    samples defined by function `generate_samples`
    """
    original, target, exact_weights, original_test, target_test, exact_weights_test = \
        generate_samples(with_linear_transformation=with_linear_transformation, 
                         add_variation=add_variation, n_samples=n_samples, verbose=verbose)
 
    GBreweighter_fit(reweighter, reweighter_parameters, original, target, 
                     exact_weights, original_test, target_test, exact_weights_test, 
                     verbose=verbose, print_weights=False)
        

def reconstruct_weights_by_GBreweighter_with_blow_up(reweighter, reweighter_parameters, n_samples=50000):
    """
    Reconstruct weights by BDT reweighter from `hep_ml` on the generated 
    samples defined by function `generate_samples_for_blow_up_demo`
    """
    original, target, exact_weights, original_test, target_test, exact_weights_test = \
        generate_samples_for_blow_up_demo()
 
    GBreweighter_fit(reweighter, reweighter_parameters, original, target, 
                     exact_weights, original_test, target_test, exact_weights_test, 
                     verbose=False, print_weights=True)