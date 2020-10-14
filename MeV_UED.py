"""
Code example for the simulation of electron diffraction patterns based on molecular geometries 
within the independent atom model. The simulations yield results compatible with the electron 
beam parameters of the MeV Ultrafast Electron Diffraction (UED) facility at SLAC National 
Accelerator Laboratory (https://lcls.slac.stanford.edu/instruments/mev-ued). 
Created by Thomas Wolf, 02/26/2020
"""

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import glob, os
from timer import Timer

############################################################################################################
## Classes and functions ###################################################################################
############################################################################################################

class mol_geom():
    """
    Creates a molecular geometry object.
    Arguments: 
    filename: Path to a molecular geometry (*.xyz) file. See below for the file format expected
    by the code.
    """
    def __init__(self,filename):
        """
        Function to initialize the geometry object by loading geometry data.
        """
        self.loadxyz(filename)
        
        
    def loadxyz(self,filename):
        """
        Function to load geometry data from an *.xyz file. The code assumes the file synthax 
        as read and written by programs like Molden (http://cheminf.cmbi.ru.nl/molden/). It 
        ignores the first two lines of the *.xyz file. The first line usually contains the 
        number of atoms in the molecular geometry, the second line contains comments. The code
        expects a line for each atom of the molecular geometry in the remainder of the file.
        Each line contains the following information in the exact order: Element letter, x, y, and z 
        coordinates. The different items are separated by spaces.
        """
        # Load geometry file as strings
        with open(filename,'r') as geofile:
            geostr = geofile.readlines()

        # Extract element information (elements) and coordinates (geom)
        geostr2 = geostr[2:]
        self.coordinates = np.zeros((len(geostr2),3))
        self.elements = []
        for i in np.arange(len(geostr2)):
            arr = geostr2[i].split()
            self.elements.append(arr[0])
            self.coordinates[i,0] = float(arr[1])
            self.coordinates[i,1] = float(arr[2])
            self.coordinates[i,2] = float(arr[3])

############################################################################################################
############################################################################################################

class mol_trajs():
    """Process many trajectories"""
            
    def __init__(self,searchStr="*.xyz"):
        self.workDir = '/Users/bwt/working/Diffraction_simulation/Traj_SI_113'
        os.chdir(self.workDir)
        print(f"Setting current directory to: {self.workDir}, searching for files that match {searchStr}")
        self.fnames = self.files(searchStr)
        
    def files(self,searchStr):
        """Function to add files to object."""
        filenames = [fname for fname in glob.glob(searchStr)]
        return sorted(filenames)

    
    def proc_trajs(self):
        """Function to process *.xyz trajectories."""
        self.alltraj_mol = []
        self.alltraj_sM = []
        AtScatXSect = Atomic_Scattering_Cross_Sections()
        
        for fname in self.fnames:
            traj_i = mol_traj(fname,AtScatXSect)
            traj_i.proc_traj()
            self.alltraj_mol.append(np.array(traj_i.traj_I_mol_1D))
            self.alltraj_sM.append (np.array(traj_i.traj_sM_1D))       
    
             
############################################################################################################

class mol_traj():
    """Creates a molecular trajectory object."""
    def __init__(self,filename,AtScatXSect):
        """Function to initialize the geometry object by loading geometry data."""
        self.deltat = 0.194
        self.load_traj(filename)
        self.AtScatXSect = AtScatXSect
        
    
    def load_traj(self,filename):
        """Function to load geometry data from an *.xyz trajectory. Took about 0.0094 seconds per file"""
        with open(filename,'r') as geofile:
            geostr = geofile.readlines()
            
        nAtoms = int(geostr[0]) # number of atoms
        nSteps = int(len(geostr) / (nAtoms + 2))
        print(f"{filename}:\tnAtoms = {nAtoms} & nSteps = {nSteps}.")
        self.fname = filename
        self.nAtoms = nAtoms
        self.nSteps = nSteps
        self.tAxis = np.linspace(0,nSteps*self.deltat,nSteps) 
        self.geoarray = []
        
        for step in range(nSteps):
            # The indicies for 5 atoms at steps 0,1,2 are 2:7, 1: 8:13, 14:19
            xyz = geostr[(2+nAtoms)*step + 2 : (2+nAtoms)*(step+1)]
            self.geoarray.append(xyz)
                  
        try:
            self.coordinates = []
            for i, geo in enumerate(self.geoarray):
                tmp = self.geom_parser(geo)
                #print("step: i, "\n", tmp)
                self.coordinates.append(tmp)
        except: 
            print("Problem loading or parsing trajectory")    
            
        
    def geom_parser(self,geostr):
        """Should have removed header, just N X Y Z entries"""
        coordinates = np.zeros((len(geostr),3))
        self.elements = []
        for i in np.arange(len(geostr)):
            arr = geostr[i].split()
            self.elements.append(arr[0])
            coordinates[i,0] = float(arr[1])
            coordinates[i,1] = float(arr[2])
            coordinates[i,2] = float(arr[3])
            
        return coordinates    
    
    
    def proc_traj(self):
        """Function to process *.xyz trajectory."""
        #t = Timer()
        #t.start()
        self.traj_I_at_1D = [] #  Python list append uses 2.3 seconds per traj. Try numpy...
                                # Not quite true, seems to be from parent mol_trajs class. Just running mol_traj() only takes 0.62 seconds
        self.traj_I_mol_1D = []
        self.traj_sM_1D = []
        #self.traj_I_at_1D = np.zeros((self.nSteps,120))
        #self.traj_I_mol_1D = np.zeros_like(self.traj_I_at_1D)
        #self.traj_sM_1D = np.zeros_like(self.traj_I_at_1D)     
        
        params = Parameters()
        params.elements = self.elements # may go wrong if the atoms get reordered in XYZ file?
        Diff = Diffraction(params,self.AtScatXSect)
        
        for step in range(self.nSteps):
            params.coord_step = self.coordinates[step]
            Diff.make_1D_diffraction(params)
            self.traj_I_at_1D.append(Diff.I_at_1D)
            self.traj_I_mol_1D.append(Diff.I_mol_1D)
            self.traj_sM_1D.append(Diff.sM_1D)
            #self.traj_I_at_1D[step]  = Diff.I_at_1D
            #self.traj_I_mol_1D[step] = Diff.I_mol_1D
            #self.traj_sM_1D[step]    = Diff.sM_1D
        
        self.s = Diff.s
        #t.stop()

############################################################################################################

class Parameters(): 
    pass
            
############################################################################################################
            
class Atomic_Scattering_Cross_Sections():
    """
    Creates an object containing form factors for different elements. This class currently 
    supports the following elements: H, He, C, N, O, F, S, Fe, Br, I. The form factors are
    calculated with the ELSEPA program (https://github.com/eScatter/elsepa) assuming the 
    standard electron kinetic energy of 3.7 MeV used at the SLAC UED facility. This class must
    be modified to add unsupported elements.
    """
    def __init__(self):
        """
        Function to initialize the form factor object by loading form factors.
        """
        self.ELSEPAdir = '/Users/bwt/working/Diffraction_simulation/ELSEPA'
        # This line must be edited to add elements:
        self.supported_elements = ['H', 'He', 'C', 'N', 'O', 'F', 'S', 'Fe', 'Br', 'I']
        for element in self.supported_elements:
            exec('self.' + element +", self.thetadeg = self.load_form_fact('" + element + "')")
    
    
    def load_form_fact(self,Element):
        """
        Function to load the scattering form factor for a specific element from an ELSEPA
        output file. 
        Arguments:
        Element:  Element symbol as string
        Returns:
        FF:       Angle-dependent scattering intensity in units of a0^2/sr
        thetadeg: Scattering angle in degrees
        """
        if len(Element)<2:
            Element = Element + ' '
            
        with open(self.ELSEPAdir + '/' + Element + '3p7MeV.dat') as f:
            lines = f.readlines()
        
        for i,line in enumerate(lines):
            if line.find('#')!=-1:
                continue
            else:
                break
                
        lines = lines[i:]
        thetadeg = np.zeros((len(lines),))
        FF = np.zeros_like(thetadeg)
        for i in np.arange(len(lines)):
            thetadeg[i] = (float(lines[i].split()[0]))
            FF[i] = (float(lines[i].split()[3]))
        return FF, thetadeg
      
        
from scipy.interpolate import interp1d     

############################################################################################################

class Diffraction():
    """
    Creates a diffraction object.
    Arguments:
    geom:   mol_geom object
    AtScatXSect: Scattering cross-section object
    Npixel: Length of Q-array
    Max_Q:  Maximum Q in inverse Angstroms
    """
    def __init__(self,params,AtScatXSect,Npixel=120,Max_s=12):
        """
        Function to initialize Diffraction object.
        """
        self.elements = params.elements
        self.natom = len(self.elements)
        self.AtScatXSect = AtScatXSect
        self.U = 3.7 # Electron kinetic energy
        self.Max_s = Max_s
        self.Npixel = Npixel
        self.s = np.linspace(0,np.float(self.Max_s),self.Npixel)
        
        E=self.U*1e6*1.6022*1e-19
        m=9.1094e-31
        h=6.6261e-34
        c=299792458

        lambdaEl=h/np.sqrt(2*m*E)/np.sqrt(1+E/(2*m*c**2)) # Electron wavelength
        k=2*np.pi/lambdaEl # Electron wave vector

        thetarad = self.AtScatXSect.thetadeg/360*2*np.pi
        self.a = 4*np.pi/lambdaEl*np.sin(thetarad/2)/1E10
        self.make_at_diffraction()
        
        
    def make_at_diffraction(self):
        """Invariant to atom position, we can compute this just one time and store away"""
        
        # 1D
        self.I_at_1D = np.zeros((len(self.s),)) # Atomic scattering contribution to diffraction signal
        self.fmap1D = []
        for element in self.elements:
            namespace = {'interp1d'}
            f = eval('interp1d(self.a,np.sqrt(self.AtScatXSect.' + element + '))')
            self.fmap1D.append(f(self.s))
            self.I_at_1D += np.square(abs(f(self.s)))      
                  
        # 2D    
        self.sy,self.sz = np.meshgrid(np.arange(-1*self.Max_s,self.Max_s,2*self.Max_s/self.Npixel), \
                            np.arange(-1*self.Max_s,self.Max_s,2*self.Max_s/self.Npixel))
        self.sr = np.sqrt(np.square(self.sy)+np.square(self.sz))
        self.I_at_2D = np.zeros_like(self.sr) # Atomic scattering contribution to diffraction signal  
        self.fmap2D = []
        for element in self.elements:
            f = eval('interp1d(self.a,np.sqrt(self.AtScatXSect.' + element + '))')
            self.fmap2D.append(f(self.sr))
            self.I_at_2D += np.square(abs(f(self.sr)))
            
        
    def make_1D_diffraction(self,params):
        """Function to create a 1D diffraction pattern assuming an ensemble of randomly oriented molecules."""
        # Contribution from interference between atoms to diffaction signal:
        self.coord_step = params.coord_step
        self.I_mol_1D = np.zeros_like(self.I_at_1D) 
        for i in np.arange(self.natom):
            for j in np.arange(self.natom):
                if i!=j:
                    dist = np.sqrt(np.square(self.coord_step[i,:]-self.coord_step[j,:]).sum())
                    # Error from divide by zero, can either add eps = 1e-7 to dist*self.s or np.seterr(divide='ignore', invalid='ignore')
                    #print(f"Division involves zeros: \n{dist*self.s}, \n{np.sin(dist*self.s)}")
                    self.I_mol_1D += abs(self.fmap1D[i])*abs(self.fmap1D[j])*np.sin(dist*self.s)/(dist*self.s)

        self.sM_1D = self.s*self.I_mol_1D/self.I_at_1D # Modified molecular diffraction


    def make_2D_diffraction(self):
        """Function to create a 2D diffraction pattern assuming an ensemble of randomly oriented molecules."""
        # Contribution from interference between atoms to diffaction signal:
        self.I_mol_2D = np.zeros_like(self.I_at_2D) 
        for i in np.arange(self.natom):
            for j in np.arange(self.natom):
                if i!=j:
                    dist = np.sqrt(np.square(self.coord_step[i,:]-self.coord_step[j,:]).sum())
                    self.I_mol_2D += abs(self.fmap2D[i])*abs(self.fmap2D[j])*np.sin(dist*self.sr)/(dist*self.sr) 

        self.sM_2D = self.sr*self.I_mol_2D/self.I_at_2D # Modified molecular diffraction        
        
        
############################################################################################################        
        