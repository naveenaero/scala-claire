import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.special import sph_harm
import file_io as fio
from mpi4py import MPI
from netCDF4 import Dataset
import os,sys


def mpi_minmax_scalar(x):
    comm = MPI.COMM_WORLD
    comm.Barrier()
    xmin = comm.allreduce(x.flatten().min(), op=MPI.MIN)
    xmax = comm.allreduce(x.flatten().max(), op=MPI.MAX)
    comm.Barrier()
    return xmin, xmax

def normalize(x, xmin, xmax, a, b):
    return (b-a)*(x-xmin)/(xmax-xmin) + a

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # The process ID (integer 0-3 for 4-process run)
nprocs = comm.Get_size() # Get total processes

N = sys.argv[1]
[N0,N1,N2] = [int(x) for x in N.split("x")]
if rank==0:
    print("N = {}x{}x{}".format(N0,N1,N2))
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
M = int(sys.argv[2]) 

v_minmax = 0.15
case = 4

nx_str = "x".join([str(x) for x in [N0,N1,N2]])

output_dir = '/scratch1/04716/naveen15/spherical_harmonics/v_minmax_{}/case_{}'.format(v_minmax,case)
os.makedirs(output_dir, exist_ok=True)
if rank==0:
    print(output_dir)

nc1 = Dataset(output_dir + '/syn_velocity-x1_freq{}_nx{}.nc'.format(M,nx_str), 'w', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info(),format='NETCDF4_CLASSIC')
nc2 = Dataset(output_dir + '/syn_velocity-x2_freq{}_nx{}.nc'.format(M,nx_str), 'w', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info(),format='NETCDF4_CLASSIC')
nc3 = Dataset(output_dir + '/syn_velocity-x3_freq{}_nx{}.nc'.format(M,nx_str), 'w', parallel=True, comm=MPI.COMM_WORLD, info=MPI.Info(),format='NETCDF4_CLASSIC')
# below should work also - MPI_COMM_WORLD and MPI_INFO_NULL will be used.
x = nc1.createDimension("x",ng[0])
y = nc1.createDimension("y",ng[1])
z = nc1.createDimension("z",ng[2])
v1 = nc1.createVariable('data', "f4", ("x","y","z",))
x = nc2.createDimension("x",ng[0])
y = nc2.createDimension("y",ng[1])
z = nc2.createDimension("z",ng[2])
v2 = nc2.createVariable('data', "f4", ("x","y","z",))
x = nc3.createDimension("x",ng[0])
y = nc3.createDimension("y",ng[1])
z = nc3.createDimension("z",ng[2])
v3 = nc3.createVariable('data', "f4", ("x","y","z",))

v1.set_collective(True)
v2.set_collective(True)
v3.set_collective(True)

v1[ix0:ix1,:,:] = 0
v2[ix0:ix1,:,:] = 0
v3[ix0:ix1,:,:] = 0
Ak = lambda k : k**(-0.5)
for k in range(1,M):
    if case == 0:
        v1[ix0:ix1,:,:] += Ak(k)*np.cos(k*Y)*np.cos(k*Z)
        v2[ix0:ix1,:,:] += Ak(k)*np.sin(k*Z)*np.sin(k*X)
        v3[ix0:ix1,:,:] += Ak(k)*np.cos(k*X)*np.cos(k*Y)
    if case == 1:
        v1[ix0:ix1,:,:] += Ak(k)*(np.sin(k*Y) + np.sin(k*Z))
        v2[ix0:ix1,:,:] += Ak(k)*(np.sin(k*X) + np.sin(k*Z))
        v3[ix0:ix1,:,:] += Ak(k)*(np.sin(k*Y) + np.sin(k*X))
    if case == 2:
        v1[ix0:ix1,:,:] += Ak(k)*np.sin(k*Z)*np.cos(k*Y)*np.sin(k*Y)
        v2[ix0:ix1,:,:] += Ak(k)*np.sin(k*X)*np.cos(k*Z)*np.sin(k*Z)
        v3[ix0:ix1,:,:] += Ak(k)*np.sin(k*Y)*np.cos(k*X)*np.sin(k*X)
    if case == 3:
        v1[ix0:ix1,:,:] += Ak(k)*np.sin(k*(Z+Y)/2)
        v2[ix0:ix1,:,:] += Ak(k)*np.sin(k*(Z+X)/2)
        v3[ix0:ix1,:,:] += Ak(k)*np.sin(k*(X+Y)/2)
    if case == 4:
        v1[ix0:ix1,:,:] += Ak(k)*np.cos(k*Y)*np.cos(k*X)
        v2[ix0:ix1,:,:] += Ak(k)*np.sin(k*Z)*np.sin(k*Y)
        v3[ix0:ix1,:,:] += Ak(k)*np.cos(k*X)*np.cos(k*Z)
    if case == 5:
        v1[ix0:ix1,:,:] += Ak(k)*np.cos(k*Y) + np.sin(k*X)
        v2[ix0:ix1,:,:] += Ak(k)*np.sin(k*Z) + np.cos(k*Y)
        v3[ix0:ix1,:,:] += Ak(k)*np.cos(k*X) + np.sin(k*Z)



# get minmax
v1min, v1max = mpi_minmax_scalar(v1[ix0:ix1,:,:])
v2min, v2max = mpi_minmax_scalar(v2[ix0:ix1,:,:])
v3min, v3max = mpi_minmax_scalar(v3[ix0:ix1,:,:])

## normalize
a,b = -v_minmax, v_minmax
v1[ix0:ix1,:,:] = normalize(v1[ix0:ix1,:,:], v1min, v1max, a, b)
v2[ix0:ix1,:,:] = normalize(v2[ix0:ix1,:,:], v2min, v2max, a, b)
v3[ix0:ix1,:,:] = normalize(v3[ix0:ix1,:,:], v3min, v3max, a, b)

nc1.close()
nc2.close()
nc3.close()
