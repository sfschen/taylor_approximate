import numpy as np

def make_param_grid(x0s, dxs, order=4):


    Nparams = len(x0s)
    template = np.arange(-order,order+1,1)
    Npoints = 2*order + 1
    grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

    Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
    Inds = [ind.flatten() for ind in Inds]
    center_ii = (order,)*Nparams
    Coords = np.meshgrid( *grid_axes, indexing='ij')
    
    return Coords, Inds, center_ii


def make_predictions_grid(func, Coords, Inds, output_shape=None):
    
    Nparams = len(Coords)
    Npoints = Coords[0].shape[0]
    
    if output_shape is None:
        coord = [Coords[i][(0,)*Nparams] for i in range(Nparams)]
        output_shape = func(*coord).shape
        
    Fii = np.zeros( (Npoints,)*Nparams+ output_shape)
        
    for nn, iis in enumerate(zip(*Inds)):
        coord = [Coords[i][iis] for i in range(Nparams)]
        Fii[iis] =  func(*coord)
        
    return Fii