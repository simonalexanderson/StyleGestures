'''
Tools for Manipulating and Converting 3D Rotations

By Omid Alemi
Created: June 12, 2017

Adapted from that matlab file...
'''

import math
import numpy as np
import transforms3d as t3d
from pymo.Quaternions import Quaternions

def deg2rad(x):
    return x/180*math.pi


def rad2deg(x):
    return x/math.pi*180

def unroll(rots):
    return unroll_1(rots)

def unroll_1(rots):

    new_rots = rots.copy()
    
    # Compute angles and alternative rotation angles
    angs = np.linalg.norm(rots, axis=1)
    alt_angs=2*np.pi-angs

    #find discontinuities
    d_angs = np.diff(angs, axis=0)
    d_angs2 = alt_angs[1:]-angs[:-1]
    
    swps = np.where(np.abs(d_angs2)<np.abs(d_angs))[0]

    #reshape into intervals where we should unroll the rotations
    isodd = swps.shape[0] % 2 == 1
    if isodd:
        swps = np.append(swps, rots.shape[0]-1)
    intv = 1+swps.reshape((swps.shape[0]//2, 2))
    for ii in range(intv.shape[0]):
        new_ax = -rots[intv[ii,0]:intv[ii,1],:]/np.tile(angs[intv[ii,0]:intv[ii,1], None], (1,3))
        new_angs = alt_angs[intv[ii,0]:intv[ii,1]]
        new_rots[intv[ii,0]:intv[ii,1],:] = new_ax*np.tile(new_angs[:, None], (1,3))

    return new_rots

def unroll_2(rots):

    new_rots = rots.copy()
    
    # Compute angles and alternative rotation angles
    angs = np.linalg.norm(rots, axis=1)
    dotprod = np.einsum('ij,ij->i', rots[:-1,:], rots[1:,:])
    #ax = rots/np.tile(angs[:, None], (1,3))
    #d_ax = np.linalg.norm(np.diff(ax, axis=0), axis=1)
    alt_angs=2*np.pi-angs

    #find discontinuities
    d_angs = np.diff(angs, axis=0)
    d_angs2 = alt_angs[1:]-angs[:-1]
    
    # FIXME should check if dot product is <0 not norm d_ax
    swps = np.where((dotprod<-1))[0]
    #swps = np.where((np.abs(d_ax)>0.5))[0]
    #swps = np.where(np.abs(d_angs2)<np.abs(d_angs))[0]

    #reshape into intervals where we should unroll the rotations
    isodd = swps.shape[0] % 2 == 1
    if isodd:
        swps=swps[:-1]
        #swps = np.append(swps, rots.shape[0]-1)
    intv = 1+swps.reshape((swps.shape[0]//2, 2))
    for ii in range(intv.shape[0]):
        new_ax = -rots[intv[ii,0]:intv[ii,1],:]/np.tile(angs[intv[ii,0]:intv[ii,1], None], (1,3))
        new_angs = alt_angs[intv[ii,0]:intv[ii,1]]
        new_rots[intv[ii,0]:intv[ii,1],:] = new_ax*np.tile(new_angs[:, None], (1,3))

    return new_rots

def euler_reorder2(rots, order='XYZ', new_order='XYZ',use_deg=False):
    if order==new_order:
        return rots
    
    if use_deg:
        rots = np.deg2rad(rots)

    quats = Quaternions.from_euler(rots, order=order.lower())
    eul = quats.euler(order=new_order.lower())
    
    if use_deg:
        eul = np.rad2deg(eul)

    return eul

def euler_reorder(rot, order='XYZ', new_order='XYZ',use_deg=False):
    if order==new_order:
        return rot
    
    if use_deg:
        rot = np.deg2rad(rot)
    print("order:" + order)
    print("new_order:" + new_order)
    
    rotmat = t3d.euler.euler2mat(rot[0], rot[1], rot[2], 'r' + order.lower())
    eul = t3d.euler.mat2euler(rotmat, 'r' + new_order.lower())

    #quat = t3d.euler.euler2quat(rot[0], rot[1], rot[2], 'r' + order.lower())
    #eul = t3d.euler.quat2euler(quat, 'r' + new_order.lower())
    
    if use_deg:
        eul = np.rad2deg(eul)

    return eul

def offsets_inv(offset, rots, order='XYZ',use_deg=False):
    if use_deg:
        offset = np.deg2rad(offset)
        rots = np.deg2rad(rots)

    q0 = t3d.euler.euler2quat(rots[0], rots[1], rots[2], 'r' + order.lower())
    q_off = t3d.euler.euler2quat(offset[0], offset[1], offset[2], 'r' + order.lower())
    q2=t3d.euler.quat2euler(t3d.quaternions.qmult(q0,t3d.quaternions.qinverse(q_off)), 'r' + order.lower())
    #q0=Quaternions.from_euler(rots, order=order.lower())   
    #q_off=Quaternions.from_euler(offset, order=order.lower())
    #q2=((-q_off)*q0).euler(order=order.lower())
    
    if use_deg:
        q2 = np.rad2deg(q2)

    return q2

def offsets(offset, rots, order='XYZ',use_deg=False):
    if use_deg:
        offset = np.deg2rad(offset)
        rots = np.deg2rad(rots)

    q0 = t3d.euler.euler2quat(rots[0], rots[1], rots[2], 'r' + order.lower())
    q_off = t3d.euler.euler2quat(offset[0], offset[1], offset[2], 'r' + order.lower())
    q2=t3d.euler.quat2euler(t3d.quaternions.qmult(q0,q_off), 'r' + order.lower())
    #q0=Quaternions.from_euler(rots, order=order.lower())   
    #q_off=Quaternions.from_euler(offset, order=order.lower())
    #q2=(q_off*q0).euler(order=order.lower())
    
    if use_deg:
        q2 = np.rad2deg(q2)

    return q2

def euler2expmap2(rots, order='XYZ',use_deg=False):
    if use_deg:
        rots = np.deg2rad(rots)
    #print("rot:" + str(rot))
    quats=Quaternions.from_euler(rots, order=order.lower())    
    theta, vec = quats.angle_axis()
    return unroll(vec*np.tile(theta[:,None],(1,3)))

def euler2expmap(rot, order='XYZ',use_deg=False):
    if use_deg:
        rot = np.deg2rad(rot)
    #print("rot:" + str(rot))
    vec, theta = t3d.euler.euler2axangle(rot[0], rot[1], rot[2], 'r' + order.lower())
    return vec*theta

def expmap2euler(rot, order='XYZ',use_deg=False):
    theta = np.linalg.norm(rot)
    if theta > 1.0e-10:
        vector = rot / theta
    else:
        vector = np.array([1.,0.,0.])
        theta=0.0
    eul = t3d.euler.axangle2euler(vector, theta, 'r' + order.lower())
    if use_deg:
        return np.rad2deg(eul)
    else:
        return eul

class Rotation():
    def __init__(self,rot, param_type, **params):
        self.rotmat = []
        if param_type == 'euler':
            self._from_euler(rot[0],rot[1],rot[2], params)
        elif param_type == 'expmap':
            self._from_expmap(rot[0], rot[1], rot[2], params)

    def _from_euler(self, alpha, beta, gamma, params):
        '''Expecting degress'''
        
        if params['from_deg']==True:
            alpha = deg2rad(alpha)
            beta = deg2rad(beta)
            gamma = deg2rad(gamma)
            
        order = "s" + ((params['order']).lower())[::-1]
#        Quaternions.from_euler()        
        self.rotmat = np.transpose(t3d.euler.euler2mat(gamma, beta , alpha, axes=order))

#        ca = math.cos(alpha)
#        cb = math.cos(beta)
#        cg = math.cos(gamma)
#        sa = math.sin(alpha)
#        sb = math.sin(beta)
#        sg = math.sin(gamma)        
#
#        Rx = np.asarray([[1, 0, 0], 
#              [0, ca, sa], 
#              [0, -sa, ca]
#              ])
#
#        Ry = np.asarray([[cb, 0, -sb], 
#              [0, 1, 0],
#              [sb, 0, cb]])
#
#        Rz = np.asarray([[cg, sg, 0],
#              [-sg, cg, 0],
#              [0, 0, 1]])
#
#        self.rotmat = np.eye(3)
#        
#        order = params['order']
#        for i in range(0,len(order)):
#            if order[i]=='X':
#                self.rotmat = np.matmul(Rx, self.rotmat)
#            elif order[i]=='Y':
#                self.rotmat = np.matmul(Ry, self.rotmat)
#            elif order[i]=='Z':
#                self.rotmat = np.matmul(Rz, self.rotmat)
#            else:
#                print('unknown rotation axis: ' + order[i])
#                
#        # self.rotmat = np.matmul(np.matmul(Rz, Ry), Rx)
#        print ("------" + "TRUE")
#        print (self.rotmat)
   
    def _from_expmap(self, alpha, beta, gamma, params):
        if (alpha == 0 and beta == 0 and gamma == 0):
            self.rotmat = np.eye(3)
            return

        #TODO: Check exp map params

        theta = np.linalg.norm([alpha, beta, gamma])

        expmap = [alpha, beta, gamma] / theta

        x = expmap[0]
        y = expmap[1]
        z = expmap[2]

        s = math.sin(theta/2)
        c = math.cos(theta/2)

        self.rotmat = np.asarray([
            [2*(x**2-1)*s**2+1,  2*x*y*s**2-2*z*c*s,  2*x*z*s**2+2*y*c*s],
            [2*x*y*s**2+2*z*c*s,  2*(y**2-1)*s**2+1,  2*y*z*s**2-2*x*c*s],
            [2*x*z*s**2-2*y*c*s, 2*y*z*s**2+2*x*c*s , 2*(z**2-1)*s**2+1]
        ])
        


    def get_euler_axis(self):
        R = self.rotmat
        theta = math.acos((self.rotmat.trace() - 1) / 2)
        axis = np.asarray([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        axis = axis/(2*math.sin(theta))
        return theta, axis

    def to_expmap(self):
        axis, theta = t3d.axangles.mat2axangle(self.rotmat, unit_thresh=1e-05)
#        theta, axis = self.get_euler_axis()
        rot_arr = theta * axis
        if np.isnan(rot_arr).any():
            rot_arr = [0, 0, 0]
        return rot_arr
    
    def to_euler(self, use_deg=False, order='xyz'):
        order = "s" + order.lower()
        eulers = t3d.euler.mat2euler(np.transpose(self.rotmat), axes=order)
        return eulers[::-1]
        
#        eulers = np.zeros((2, 3))
#
#        if np.absolute(np.absolute(self.rotmat[2, 0]) - 1) < 1e-12:
#            #GIMBAL LOCK!
#            print('Gimbal')
#            if np.absolute(self.rotmat[2, 0]) - 1 < 1e-12:
#                eulers[:,0] = math.atan2(-self.rotmat[0,1], -self.rotmat[0,2])
#                eulers[:,1] = -math.pi/2
#            else:
#                eulers[:,0] = math.atan2(self.rotmat[0,1], -elf.rotmat[0,2])
#                eulers[:,1] = math.pi/2
#            
#            return eulers
#
#        theta = - math.asin(self.rotmat[2,0])
#        theta2 = math.pi - theta
#
#        # psi1, psi2
#        eulers[0,0] = math.atan2(self.rotmat[2,1]/math.cos(theta), self.rotmat[2,2]/math.cos(theta))
#        eulers[1,0] = math.atan2(self.rotmat[2,1]/math.cos(theta2), self.rotmat[2,2]/math.cos(theta2))
#
#        # theta1, theta2
#        eulers[0,1] = theta
#        eulers[1,1] = theta2
#
#        # phi1, phi2
#        eulers[0,2] = math.atan2(self.rotmat[1,0]/math.cos(theta), self.rotmat[0,0]/math.cos(theta))
#        eulers[1,2] = math.atan2(self.rotmat[1,0]/math.cos(theta2), self.rotmat[0,0]/math.cos(theta2))
#
        if use_deg:
            eulers = rad2deg(eulers)

        return eulers
    
    def to_quat(self):
        #TODO
        pass
    
    def __str__(self):
        return "Rotation Matrix: \n " + self.rotmat.__str__()
    



