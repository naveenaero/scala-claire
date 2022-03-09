import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm
import file_io as fio
from mpi4py import MPI
from netCDF4 import Dataset
import os,sys

def getSpherical_np(xyz):
    ptsnew = np.zeros_like(xyz)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    ptsnew[:,1] = np.arctan2(np.sqrt(xy), xyz[:,2]) # polar/elevation angle defined from Z-axis down (scipy phi)
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0]) # azimuth (scipy theta)
    ptsnew[:,2] = np.mod(ptsnew[:,2], 2*np.pi)
    return ptsnew

def normalize(img):
    fmax, fmin = img.max(), img.min()
    return (img - fmin)/(fmax - fmin)

def get_mask(m,l,XYZ_sph,nl,R,alpha):
  fcolors_mask = sph_harm(m,l, XYZ_sph[:,2], XYZ_sph[:,1])
  fcolors_mask = np.abs(np.sqrt(2) * (-1)**m * fcolors_mask.real)
  fcolors_mask = np.reshape(fcolors_mask, (nl[0],nl[1],nl[2]))
  threshold = alpha*fcolors_mask
  mask = np.where(R < threshold)
  return mask

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # The process ID (integer 0-3 for 4-process run)
nprocs = comm.Get_size() # Get total processes

m, l = int(sys.argv[1]),int(sys.argv[2])

N = sys.argv[3]
[N0,N1,N2] = [int(x) for x in N.split("x")]
if rank==0:
    print("N={}, m={}, l={}".format(N,m,l))
domain = 2*np.pi
ng = (N0,N1,N2)
nl = (ng[0]//nprocs, ng[1], ng[2])
hx = (domain/ng[0], domain/ng[1], domain/ng[2])
grid0 = np.linspace(-np.pi,np.pi,N0)
grid1 = np.linspace(-np.pi,np.pi,N1)
grid2 = np.linspace(-np.pi,np.pi,N2)

ix0 = nl[0]*rank
ix1 = ix0 + nl[0]
X,Y,Z = np.meshgrid(grid0[ix0:ix1],grid1,grid2, indexing='ij')

scale_factor = 7
nlabels = int(sys.argv[4])
np.random.seed(seed=nlabels)
shift = 0.40*2*np.pi
centers = -shift/2. + shift * np.random.rand(nlabels,3)
V = np.zeros_like(X)

for i in range(nlabels):
    # generate random translation centers
    center = centers[i,:]
    Xp,Yp,Zp = X-center[0], Y-center[1], Z-center[2]
    Xp,Yp,Zp = np.random.permutation([Xp,Yp,Zp])
    XYZ = np.stack((Xp.flatten(), Yp.flatten(), Zp.flatten()), axis=1)
    R = (Xp)**2 + (Yp)**2 + (Zp)**2
    XYZ_sph = getSpherical_np(XYZ)
    mask = get_mask(m,l,XYZ_sph,nl,R,scale_factor)
    V[mask] += 1;

nx_str = "x".join([str(x) for x in [N0,N1,N2]])
nc = Dataset('/scratch1/04716/naveen15/spherical_harmonics/Y_m{}_l{}_nx{}_nlabels{}.nc'.format(m,l,nx_str,nlabels), 'w', parallel=True, comm=MPI.COMM_WORLD,
        info=MPI.Info(),format='NETCDF4_CLASSIC')
x = nc.createDimension("x",ng[0])
y = nc.createDimension("y",ng[1])
z = nc.createDimension("z",ng[2])
v = nc.createVariable('data', "f4", ("x","y","z",))
# switch to collective mode
v.set_collective(True)
# write data to v
v[ix0:ix1,:,:] = V
nc.close()

