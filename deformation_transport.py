import nibabel as nib
import numpy as np
from scipy.interpolate import interpn
import os, sys
import argparse

###
### ------------------------------------------------------------------------ ###
def writeNII(img, filename, affine=None, ref_image=None):
    '''
    function to write a nifti image, creates a new nifti object
    '''
    if ref_image is not None:
        data = nib.Nifti1Image(img, affine=ref_image.affine, header=ref_image.header);
        data.header['datatype'] = 64
        data.header['glmax'] = np.max(img)
        data.header['glmin'] = np.min(img)
    elif affine is not None:
        data = nib.Nifti1Image(img, affine=affine);
    else:
        data = nib.Nifti1Image(img, np.eye(4))

    nib.save(data, filename);

def readImage(filename):
    return nib.load(filename).get_fdata(), nib.load(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='deform moving image using CLAIRE deformation-maps')
    parser.add_argument ('-odir', type=str, help = 'output directory where CLAIRE deformation-map-x*.nii.gz fields are stored'); 
    parser.add_argument ('-ifile', type=str, help = 'moving image name');
    parser.add_argument ('-xfile', type=str, help = 'output deformed moving image name');
    args = parser.parse_args();
    
    ext = ".nii.gz"

    print("loading images")
    odir = args.odir
    def1,_ = readImage(os.path.join(odir, "deformation-map-x1{}".format(ext)))
    def2,_ = readImage(os.path.join(odir, "deformation-map-x2{}".format(ext)))
    def3,_ = readImage(os.path.join(odir, "deformation-map-x3{}".format(ext)))
    template, ref_image = readImage(args.ifile) 

    nx = template.shape
    pi = np.pi
    h = 2*pi/np.asarray(nx);
    x = np.linspace(0, 2*pi-h[0], nx[0]);
    y = np.linspace(0, 2*pi-h[1], nx[1]);
    z = np.linspace(0, 2*pi-h[2], nx[2]);

    points = (x,y,z)
    
    if args.format == "nifti":
      query = np.stack([def3.flatten(), def2.flatten(), def1.flatten()], axis=1)


    print("evaluating interpolation")
    iporder = 1
    if iporder == 0:
        method = "nearest"
    if iporder == 1:
        method = "linear"
    output = interpn(points, template, query, method=method, bounds_error=False, fill_value=0)
    output = np.reshape(output, nx)

    print("writing output")
    if args.format == "nifti":
        writeNII(output, args.xfile, ref_image=ref_image)

