from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # load file
    X = np.load(filename)
    # center data
    return (X - np.mean(X, axis=0))

def get_covariance(dataset):
    S = (1/(len(dataset)-1)) * np.dot(np.transpose(dataset), dataset)
    return S

def get_eig(S, m):
    # get eigenvectors and values
    n = len(S)
    eigval, eigvec = eigh(S, subset_by_index=[n-m, n-1])
    
    # sort the eigvec and vals
    idx = np.argsort(a=eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
    
    # put eigvals into a diagonal array
    diag_eigval = np.zeros((m,m), float)
    np.fill_diagonal(diag_eigval, [eigval])
    
    return diag_eigval, eigvec
    

def get_eig_prop(S, prop):
    eigval, eigvec = eigh(S, subset_by_value=[np.trace(S)*prop, np.inf])
    
    # sort from high to low
    idx = np.argsort(a=eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:,idx]
        
    # put eigvals into a diagonal array
    diag_eigval = np.zeros((len(eigval), len(eigval)), float)
    np.fill_diagonal(diag_eigval, [eigval])
    
    return diag_eigval, eigvec

def project_image(image, U):
    return np.dot(U, np.dot(image, U))
    

def display_image(orig, proj):
    # reshape matricies
    orig = np.reshape(orig, (32, 32))
    proj = np.reshape(proj, (32, 32))
    
    # create figure with two axis
    fig, (axis1, axis2) = plt.subplots(figsize = (9,3), ncols=2)
    
    # set titles of ax
    axis1.set_title("Original")
    axis2.set_title("Projection")
    
    # color bars
    cb1 = axis1.imshow(np.transpose(orig), aspect='equal')
    cb2 = axis2.imshow(np.transpose(proj), aspect='equal')
    
    # add colors bars
    fig.colorbar(cb1, ax = axis1)
    fig.colorbar(cb1, ax = axis2)
    
    # show
    plt.show()
