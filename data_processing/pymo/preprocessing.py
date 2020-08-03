'''
Preprocessing Tranformers Based on sci-kit's API

By Omid Alemi
Created on June 12, 2017

Modified by Simon Alexanderson, 2020-06-24
'''
import copy
import pandas as pd
import numpy as np
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R

from sklearn.base import BaseEstimator, TransformerMixin

class MocapParameterizer(BaseEstimator, TransformerMixin):
    def __init__(self, param_type = 'euler'):
        '''
        
        param_type = {'euler', 'quat', 'expmap', 'position', 'expmap2pos'}
        '''
        self.param_type = param_type

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("MocapParameterizer: " + self.param_type)
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return self._to_expmap(X)
        elif self.param_type == 'quat':
            return X
        elif self.param_type == 'position':
            return self._to_pos(X)
        elif self.param_type == 'expmap2pos':
            return self._expmap_to_pos(X)
        else:
            raise 'param types: euler, quat, expmap, position, expmap2pos'

#        return X
    
    def inverse_transform(self, X, copy=None): 
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            return self._expmap_to_euler(X)
        elif self.param_type == 'quat':
            raise 'quat2euler is not supported'
        elif self.param_type == 'position':
            raise 'positions 2 eulers is not supported'
            return X
        else:
            raise 'param types: euler, quat, expmap, position'

    def fix_rotvec(self, rots):
        '''fix problems with discontinuous rotation vectors'''
        new_rots = rots.copy()
        
        # Compute angles and alternative rotation angles
        angs = np.linalg.norm(rots, axis=1)
        alt_angs=2*np.pi-angs

        #find discontinuities by checking if the alternative representation is closer
        d_angs = np.diff(angs, axis=0)
        d_angs2 = alt_angs[1:]-angs[:-1]        
        swps = np.where(np.abs(d_angs2)<np.abs(d_angs))[0]

        #reshape into intervals where we should flip rotation axis
        isodd = swps.shape[0] % 2 == 1
        if isodd:
            swps=swps[:-1]
        intv = 1+swps.reshape((swps.shape[0]//2, 2))
        
        #flip rotations in selected intervals
        for ii in range(intv.shape[0]):
            new_ax = -rots[intv[ii,0]:intv[ii,1],:]/np.tile(angs[intv[ii,0]:intv[ii,1], None], (1,3))
            new_angs = alt_angs[intv[ii,0]:intv[ii,1]]
            new_rots[intv[ii,0]:intv[ii,1],:] = new_ax*np.tile(new_angs[:, None], (1,3))

        return new_rots
        
    def _to_pos(self, X):
        '''Converts joints rotations in Euler angles to joint positions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c)]

            # List the columns that contain position channels
            pos_cols = [c for c in euler_df.columns if ('position' in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)
            
            tree_data = {}

            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                rot_order = track.skeleton[joint]['order']

                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                # Get the position columns that belong to this joint
                pc = euler_df[[c for c in pos_cols if joint in c]]

                # Make sure the columns are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = [[0,0,0] for f in rc.iterrows()]
                    rot_order = "XYZ"
                else:
                    euler_values = [[f[1]['%s_%srotation'%(joint, rot_order[0])], 
                                     f[1]['%s_%srotation'%(joint, rot_order[1])], 
                                     f[1]['%s_%srotation'%(joint, rot_order[2])]] for f in rc.iterrows()]

                if pc.shape[1] < 3:
                    pos_values = [[0,0,0] for f in pc.iterrows()]
                else:
                    pos_values =[[f[1]['%s_Xposition'%joint], 
                                  f[1]['%s_Yposition'%joint], 
                                  f[1]['%s_Zposition'%joint]] for f in pc.iterrows()]
                
                # Convert the eulers to rotation matrices
                rotmats = R.from_euler(rot_order, euler_values, degrees=True).inv()                  
                tree_data[joint]=[
                                    [], # to store the rotation matrix
                                    []  # to store the calculated position
                                 ] 

                if track.root_name == joint:
                    tree_data[joint][0] = rotmats
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = rotmats*tree_data[parent][0]

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + np.asarray(track.skeleton[joint]['offsets'])

                    # multiply k to the rotmat of the parent for every frame i
                    q = tree_data[parent][0].inv().apply(k)

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = tree_data[parent][1] + q


                # Create the corresponding columns in the new DataFrame
                pos_df['%s_Xposition'%joint] = pd.Series(data=[e[0] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Yposition'%joint] = pd.Series(data=[e[1] for e in tree_data[joint][1]], index=pos_df.index)
                pos_df['%s_Zposition'%joint] = pd.Series(data=[e[2] for e in tree_data[joint][1]], index=pos_df.index)


            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q

    def _to_expmap(self, X):
        '''Converts Euler angles to Exponential Maps'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            exp_df = euler_df.copy()

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                
                exp_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                euler = [[f[1][r1_col], f[1][r2_col], f[1][r3_col]] for f in r.iterrows()]
                exps = np.array(self.fix_rotvec(R.from_euler(rot_order.lower(), euler, degrees=True).as_rotvec()))

                # Create the corresponding columns in the new DataFrame    
                exp_df.insert(loc=0, column='%s_gamma'%joint, value=pd.Series(data=[e[2] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_beta'%joint, value=pd.Series(data=[e[1] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_alpha'%joint, value=pd.Series(data=[e[0] for e in exps], index=exp_df.index))

            #print(exp_df.columns)
            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q

    def _expmap_to_euler(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            euler_df = exp_df.copy()
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                
                euler_df.drop(['%s_alpha'%joint, '%s_beta'%joint, '%s_gamma'%joint], axis=1, inplace=True)
                expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                rot_order = track.skeleton[joint]['order']
                euler_rots = np.array(R.from_rotvec(expmap).as_euler(rot_order.lower(), degrees=True))
                
                # Create the corresponding columns in the new DataFrame    
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q

class Mirror(BaseEstimator, TransformerMixin):
    def __init__(self, axis="X", append=True):
        """
        Mirrors the data 
        """
        self.axis = axis
        self.append = append
        
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print("Mirror: " + self.axis)
        Q = []
        
        if self.append:
            for track in X:
                Q.append(track)
            
        for track in X:
            channels = []
            titles = []
            
            if self.axis == "X":
                signs = np.array([1,-1,-1])
            if self.axis == "Y":
                signs = np.array([-1,1,-1])
            if self.axis == "Z":
                signs = np.array([-1,-1,1])

            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=-signs[0]*euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=-signs[1]*euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=-signs[2]*euler_df[rzp], index=new_df.index)
            
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]
            #lft_rots = [c for c in euler_df.columns if ('Left' in c and 'rotation' in c and 'Nub' not in c)]
            #rgt_rots = [c for c in euler_df.columns if ('Right' in c and 'rotation' in c and 'Nub' not in c)]
            lft_joints = (joint for joint in track.skeleton if 'Left' in joint and 'Nub' not in joint)
            rgt_joints = (joint for joint in track.skeleton if 'Right' in joint and 'Nub' not in joint)
                        
            new_track = track.clone()

            for lft_joint in lft_joints:                
                rgt_joint = lft_joint.replace('Left', 'Right')
                
                # Create the corresponding columns in the new DataFrame                
                new_df['%s_Xrotation'%lft_joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%rgt_joint], index=new_df.index)
                new_df['%s_Yrotation'%lft_joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%rgt_joint], index=new_df.index)
                new_df['%s_Zrotation'%lft_joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%rgt_joint], index=new_df.index)
                
                new_df['%s_Xrotation'%rgt_joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%lft_joint], index=new_df.index)
                new_df['%s_Yrotation'%rgt_joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%lft_joint], index=new_df.index)
                new_df['%s_Zrotation'%rgt_joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%lft_joint], index=new_df.index)
    
            # List the joints that are not left or right, i.e. are on the trunk
            joints = (joint for joint in track.skeleton if 'Nub' not in joint and 'Left' not in joint and 'Right' not in joint)

            for joint in joints:
                # Create the corresponding columns in the new DataFrame
                new_df['%s_Xrotation'%joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%joint], index=new_df.index)
                new_df['%s_Yrotation'%joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%joint], index=new_df.index)
                new_df['%s_Zrotation'%joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%joint], index=new_df.index)

            new_track.values = new_df
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        return X

class JointSelector(BaseEstimator, TransformerMixin):
    '''
    Allows for filtering the mocap data to include only the selected joints
    '''
    def __init__(self, joints, include_root=False):
        self.joints = joints
        self.include_root = include_root

    def fit(self, X, y=None):
        selected_joints = []
        selected_channels = []

        if self.include_root:
            selected_joints.append(X[0].root_name)
        
        selected_joints.extend(self.joints)

        for joint_name in selected_joints:
            selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o and 'Nub' not in o])
        
        self.selected_joints = selected_joints
        self.selected_channels = selected_channels
        self.not_selected = X[0].values.columns.difference(selected_channels)
        self.not_selected_values = {c:X[0].values[c].values[0] for c in self.not_selected}

        self.orig_skeleton = X[0].skeleton
        return self

    def transform(self, X, y=None):
        print("JointSelector")
        Q = []

        for track in X:
            t2 = track.clone()
            
            for key in track.skeleton.keys():
                if key not in self.selected_joints:
                    t2.skeleton.pop(key)
            t2.values = track.values[self.selected_channels]

            Q.append(t2)
      

        return Q
    
    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            t2 = track.clone()
            t2.skeleton = self.orig_skeleton
            for d in self.not_selected:
                t2.values[d] = self.not_selected_values[d]
            Q.append(t2)

        return Q


class Numpyfier(BaseEstimator, TransformerMixin):
    '''
    Just converts the values in a MocapData object into a numpy array
    Useful for the final stage of a pipeline before training
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        print("Numpyfier")
        Q = []
        
        for track in X:
            Q.append(track.values.values)
            #print("Numpyfier:" + str(track.values.columns))
            
        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df
            

            Q.append(new_mocap)

        return Q
    
class Slicer(BaseEstimator, TransformerMixin):
    '''
    Slice the data into intervals of equal size 
    '''
    def __init__(self, window_size, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        print("Slicer")
        Q = []
        
        for track in X:
            vals = track.values.values
            nframes = vals.shape[0]
            overlap_frames = (int)(self.overlap*self.window_size)
            
            n_sequences = (nframes-overlap_frames)//(self.window_size-overlap_frames)
            
            if n_sequences>0:
                y = np.zeros((n_sequences, self.window_size, vals.shape[1]))

                # extract sequences from the input data
                for i in range(0,n_sequences):
                    frameIdx = (self.window_size-overlap_frames) * i
                    Q.append(vals[frameIdx:frameIdx+self.window_size,:])

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df
            

            Q.append(new_mocap)

        return Q

class RootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method, position_smoothing=0, rotation_smoothing=0):
        """
        Accepted methods:
            abdolute_translation_deltas
            pos_rot_deltas
        """
        self.method = method
        self.position_smoothing=position_smoothing
        self.rotation_smoothing=rotation_smoothing
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print("RootTransformer")
        Q = []

        for track in X:
            if self.method == 'abdolute_translation_deltas':
                new_df = track.values.copy()
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name
                
                x=track.values[xpcol].copy()
                z=track.values[zpcol].copy()
                
                if self.position_smoothing>0:
                    x_sm = filters.gaussian_filter1d(x, self.position_smoothing, axis=0, mode='nearest')    
                    z_sm = filters.gaussian_filter1d(z, self.position_smoothing, axis=0, mode='nearest')                    
                    dx = pd.Series(data=x_sm, index=new_df.index).diff()
                    dz = pd.Series(data=z_sm, index=new_df.index).diff()
                    new_df[xpcol] = x-x_sm
                    new_df[zpcol] = z-z_sm
                else:
                    dx = x.diff()
                    dz = z.diff()
                    new_df.drop([xpcol, zpcol], axis=1, inplace=True)
                    
                dx[0] = dx[1]
                dz[0] = dz[1]
                
                new_df[dxpcol] = dx
                new_df[dzpcol] = dz
                
                new_track = track.clone()
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'hip_centric':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                
                new_df = track.values.copy()

                all_zeros = np.zeros(track.values[xp_col].values.shape)

                new_df[xp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zp_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_df[xr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zr_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_track.values = new_df

            #print(new_track.values.columns)
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        #TODO: simplify this implementation

        startx = 0
        startz = 0

        if start_pos is not None:
            startx, startz = start_pos

        for track in X:
            new_track = track.clone()
            if self.method == 'abdolute_translation_deltas':
                new_df = new_track.values
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name

                dx = track.values[dxpcol].values
                dz = track.values[dzpcol].values

                recx = [startx]
                recz = [startz]

                for i in range(dx.shape[0]-1):
                    recx.append(recx[i]+dx[i+1])
                    recz.append(recz[i]+dz[i+1])

                # recx = [recx[i]+dx[i+1] for i in range(dx.shape[0]-1)]
                # recz = [recz[i]+dz[i+1] for i in range(dz.shape[0]-1)]
                # recx = dx[:-1] + dx[1:]
                # recz = dz[:-1] + dz[1:]
                if self.position_smoothing > 0:                    
                    new_df[xpcol] = pd.Series(data=new_df[xpcol]+recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=new_df[zpcol]+recz, index=new_df.index)
                else:
                    new_df[xpcol] = pd.Series(data=recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=recz, index=new_df.index)

                new_df.drop([dxpcol, dzpcol], axis=1, inplace=True)
                
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            Q.append(new_track)

        return Q


class RootCentricPositionNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            all_but_root = [joint for joint in track.skeleton if track.root_name not in joint]
            # all_but_root = [joint for joint in track.skeleton]
            for joint in all_but_root:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]-projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]-projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]-projected_root_pos[rzp], index=new_df.index)
            

            # keep the root as it is now
            new_df[rxp] = track.values[rxp]
            new_df[ryp] = track.values[ryp]
            new_df[rzp] = track.values[rzp]

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            for joint in track.skeleton:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]+projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]+projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]+projected_root_pos[rzp], index=new_df.index)
                

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

class Flattener(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X, axis=0)

class ConstantsRemover(BaseEstimator, TransformerMixin):
    '''
    For now it just looks at the first track
    '''

    def __init__(self, eps = 1e-6):
        self.eps = eps
        

    def fit(self, X, y=None):
        stds = X[0].values.std()
        cols = X[0].values.columns.values
        self.const_dims_ = [c for c in cols if (stds[c] < self.eps).any()]
        self.const_values_ = {c:X[0].values[c].values[0] for c in cols if (stds[c] < self.eps).any()}
        return self

    def transform(self, X, y=None):
        Q = []
        

        for track in X:
            t2 = track.clone()
            #for key in t2.skeleton.keys():
            #    if key in self.ConstDims_:
            #        t2.skeleton.pop(key)
            #print(track.values.columns.difference(self.const_dims_))
            t2.values.drop(self.const_dims_, axis=1, inplace=True)
            #t2.values = track.values[track.values.columns.difference(self.const_dims_)]
            Q.append(t2)
        
        return Q
    
    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            t2 = track.clone()
            for d in self.const_dims_:
                t2.values[d] = self.const_values_[d]
#                t2.values.assign(d=pd.Series(data=self.const_values_[d], index = t2.values.index))
            Q.append(t2)

        return Q

class ListStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_mean_ = np.mean(X_train_flat, axis=0)
        self.data_std_ = np.std(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_mean_) / self.data_std_
            else:
                normalized_track = (track - self.data_mean_) / self.data_std_

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * self.data_std_) + self.data_mean_
            else:
                unnormalized_track = (track * self.data_std_) + self.data_mean_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

class ListMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_max_ = np.max(X_train_flat, axis=0)
        self.data_min_ = np.min(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_min_) / (self.data_max_ - self.data_min_) 
            else:
                normalized_track = (track - self.data_min_) / (self.data_max_ - self.data_min_)

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * (self.data_max_ - self.data_min_)) + self.data_min_
            else:
                unnormalized_track = (track * (self.data_max_ - self.data_min_)) + self.data_min_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)
        
class DownSampler(BaseEstimator, TransformerMixin):
    def __init__(self, tgt_fps, keep_all=False):
        self.tgt_fps = tgt_fps
        self.keep_all = keep_all
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            orig_fps=round(1.0/track.framerate)
            rate = orig_fps//self.tgt_fps
            if orig_fps%self.tgt_fps!=0:
                print("error orig_fps (" + str(orig_fps) + ") is not dividable with tgt_fps (" + str(self.tgt_fps) + ")")
            else:
                print("downsampling with rate: " + str(rate))
                
            for ii in range(0,rate):
                new_track = track.clone()
                new_track.values = track.values[ii:-1:rate].copy()            
                new_track.framerate = 1.0/self.tgt_fps
                Q.append(new_track)
                if not self.keep_all:
                    break
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X

class ReverseTime(BaseEstimator, TransformerMixin):
    def __init__(self, append=True):
        self.append = append
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        Q = []
        if self.append:
            for track in X:
                Q.append(track)
        for track in X:
            new_track = track.clone()                            
            new_track.values = track.values[-1::-1]
            Q.append(new_track)
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X

#TODO: JointsSelector (x)
#TODO: SegmentMaker
#TODO: DynamicFeaturesAdder
#TODO: ShapeFeaturesAdder
#TODO: DataFrameNumpier (x)

class TemplateTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

