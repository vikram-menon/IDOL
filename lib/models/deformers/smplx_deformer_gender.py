# Modified from Deformer of AG3D

from .fast_snarf.lib.model.deformer_smplx import ForwardDeformer, skinning
from .smplx import SMPLX
import torch
from pytorch3d import ops
import numpy as np
import pickle
import json

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
class SMPLXDeformer_gender(torch.nn.Module):
    
    def __init__(self, gender, is_sub2=False) -> None: 
        super().__init__()
        self.body_model = SMPLX('lib/models/deformers/smplx/SMPLX', gender=gender, \
                                create_body_pose=False, \
                                create_betas=False, \
                                create_global_orient=False, \
                                create_transl=False,
                                create_expression=False,
                                create_jaw_pose=False,
                                create_leye_pose=False,
                                create_reye_pose=False,
                                create_right_hand_pose=False,
                                create_left_hand_pose=False,
                                use_pca=True,
                                num_pca_comps=12,
                                num_betas=10,
                                flat_hand_mean=False,ext='pkl')
        self.deformer = ForwardDeformer()
        
        self.threshold = 0.12

                
        base_cache_dir = 'work_dirs/cache'   
        if is_sub2:
            base_cache_dir = 'work_dirs/cache_sub2'

        if gender in ('neutral', 'female'):
            init_spdir_neutral = torch.as_tensor(np.load(base_cache_dir+'/init_spdir_smplx_thu_newNeutral.npy'))
            self.register_buffer('init_spdir', init_spdir_neutral, persistent=False)
                    
            init_podir_neutral = torch.as_tensor(np.load(base_cache_dir+'/init_podir_smplx_thu_newNeutral.npy'))
            self.register_buffer('init_podir', init_podir_neutral, persistent=False)

            init_lbs_weights = torch.as_tensor(np.load(base_cache_dir+'/init_lbsw_smplx_thu_newNeutral.npy'))
            self.register_buffer('init_lbsw', init_lbs_weights.unsqueeze(0), persistent=False)
            init_faces = torch.as_tensor(np.load(base_cache_dir+'/init_faces_smplx_newNeutral.npy'))
            self.register_buffer('init_faces', init_faces.unsqueeze(0), persistent=False)
            if gender == 'female':
                print("[WARN] Female SMPL-X uses neutral deformation cache templates in this repo.")

        elif gender == 'male':
            init_spdir_male = torch.as_tensor(np.load(base_cache_dir+'/init_spdir_smplx_thu_newMale.npy'))
            self.register_buffer('init_spdir', init_spdir_male, persistent=False)

            init_podir_male = torch.as_tensor(np.load(base_cache_dir+'/init_podir_smplx_thu_newMale.npy'))
            self.register_buffer('init_podir', init_podir_male, persistent=False)
            init_lbs_weights = torch.as_tensor(np.load(base_cache_dir+'/init_lbsw_smplx_thu_newMale.npy'))
            self.register_buffer('init_lbsw', init_lbs_weights.unsqueeze(0), persistent=False)

        
            init_faces = torch.as_tensor(np.load(base_cache_dir+'/init_faces_smplx_neuMale.npy'))
            self.register_buffer('init_faces', init_faces.unsqueeze(0), persistent=False)
        else:
            raise ValueError(f"Unsupported gender: {gender}")

        self.initialize()
        self.initialized = True

    def initialize(self):
        '''
         Will only be called once, used to initialize lbs volume
        '''
        batch_size = 1
        device = self.body_model.posedirs.device
        # canonical space is defined in t-pose / star-pose
        body_pose_t = torch.zeros((batch_size, 63)).to(device)

        jaw_pose_t = torch.zeros((batch_size, 3)).to(device)

        ##flat_hand_mean = False
        left_hand_pose_t = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
         -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device)
        right_hand_pose_t = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
         -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device)
         
        ## flat_hand_mean = True
        leye_pose_t = torch.zeros((batch_size, 3)).to(device)
        reye_pose_t = torch.zeros((batch_size, 3)).to(device)
        expression_t = torch.zeros((batch_size, 10)).to(device)

        global_orient = torch.zeros((batch_size, 3)).to(device)
        
        betas = torch.zeros((batch_size, 10)).to(device)
        smpl_outputs = self.body_model(betas=betas, body_pose=body_pose_t, jaw_pose=jaw_pose_t, 
                                        left_hand_pose=left_hand_pose_t, right_hand_pose=right_hand_pose_t,
                                        leye_pose=leye_pose_t, reye_pose=reye_pose_t, expression=expression_t,
                                        transl=None, global_orient=global_orient) 
        
        tfs_inv_t = torch.inverse(smpl_outputs.A.float().detach()) # from template to posed space
        vs_template = smpl_outputs.vertices
        smpl_faces = torch.as_tensor(self.body_model.faces.astype(np.int64))
        pose_offset_cano = torch.matmul(smpl_outputs.pose_feature, self.init_podir).reshape(1, -1, 3)
        pose_offset_cano = torch.cat([pose_offset_cano[:, self.init_faces[..., i]] for i in range(3)], dim=1).mean(1)
        self.register_buffer('tfs_inv_t', tfs_inv_t, persistent=False)
        self.register_buffer('vs_template', vs_template, persistent=False)
        self.register_buffer('smpl_faces', smpl_faces, persistent=False)
        self.register_buffer('pose_offset_cano', pose_offset_cano, persistent=False)

        # initialize SNARF
        smpl_verts = smpl_outputs.vertices.float().detach().clone()

        self.deformer.switch_to_explicit(resolution=64,
                                         smpl_verts=smpl_verts,
                                         smpl_faces=self.smpl_faces,
                                         smpl_weights=self.body_model.lbs_weights.clone()[None].detach(),
                                         use_smpl=True)

        self.dtype = torch.float32
        self.deformer.lbs_voxel_final = self.deformer.lbs_voxel_final.type(self.dtype)
        self.deformer.grid_denorm = self.deformer.grid_denorm.type(self.dtype)
        self.deformer.scale = self.deformer.scale.type(self.dtype)
        self.deformer.offset = self.deformer.offset.type(self.dtype)
        self.deformer.scale_kernel = self.deformer.scale_kernel.type(self.dtype)
        self.deformer.offset_kernel = self.deformer.offset_kernel.type(self.dtype)

    def forword_body_model(self, smpl_params, point_pool=4):
        batchsize = smpl_params.shape[0]
        if_use_pca=True
        if smpl_params.shape[1] == 123:
            scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 12, 12, 3, 3, 3, 10], dim=1)
        else: # not use pca 12 , 189
            scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
            if_use_pca = False
        smpl_params = {
            'betas': betas.reshape(-1, 10),
            'expression': expression.reshape(-1, 10),
            'body_pose': pose.reshape(-1, 63),
            'left_hand_pose': left_hand_pose.reshape(batchsize, -1),
            'right_hand_pose': right_hand_pose.reshape(batchsize, -1),
            'jaw_pose': jaw_pose.reshape(-1, 3),
            'leye_pose': leye_pose.reshape(-1, 3),
            'reye_pose': reye_pose.reshape(-1, 3),
            'global_orient': global_orient.reshape(-1, 3),
            'transl': transl.reshape(-1, 3),
            'scale': scale.reshape(-1, 1)
        }
        
        device = smpl_params["betas"].device
        smpl_outputs = self.body_model(**smpl_params, use_pca=if_use_pca)
        return smpl_outputs
    def prepare_deformer(self, smpl_params=None, num_scenes=1, device=None):
        if smpl_params is None:
            smpl_params = torch.zeros((num_scenes, 120)).to(device)
            scale, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 63, 10, 12, 12, 3, 3, 3, 10], dim=1)
            left_hand_pose = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
                -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device).repeat(num_scenes, 1)
            right_hand_pose = torch.tensor([1.4624, -0.1615,  0.1361,  1.3851, -0.2597,  0.0247, -0.0683, -0.4478,
                -0.6652, -0.7290,  0.0084, -0.4818]).unsqueeze(0).to(device).repeat(num_scenes, 1)
         
            smpl_params = {
                'betas': betas,
                'expression': expression,
                'body_pose': pose,
                'left_hand_pose': left_hand_pose,
                'right_hand_pose': right_hand_pose,
                'jaw_pose': jaw_pose,
                'leye_pose': leye_pose,
                'reye_pose': reye_pose,
                'global_orient': global_orient,
                'transl': None,
                'scale': None,
            }
            
        else:
            batchsize = smpl_params.shape[0]
            if_use_pca=True
            if smpl_params.shape[1] == 123:
                scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 12, 12, 3, 3, 3, 10], dim=1)
            else: # not use pca 12 , 165
                scale, transl, global_orient, pose, betas, left_hand_pose, right_hand_pose, jaw_pose, leye_pose, reye_pose, expression = torch.split(smpl_params, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
                if_use_pca = False
            smpl_params = {
                'betas': betas.reshape(-1, 10),
                'expression': expression.reshape(-1, 10),
                'body_pose': pose.reshape(-1, 63),
                'left_hand_pose': left_hand_pose.reshape(batchsize, -1),
                'right_hand_pose': right_hand_pose.reshape(batchsize, -1),
                'jaw_pose': jaw_pose.reshape(-1, 3),
                'leye_pose': leye_pose.reshape(-1, 3),
                'reye_pose': reye_pose.reshape(-1, 3),
                'global_orient': global_orient.reshape(-1, 3),
                'transl': transl.reshape(-1, 3),
                'scale': scale.reshape(-1, 1)
            }
        
        device = smpl_params["betas"].device

        if not self.initialized:
            self.initialize(smpl_params["betas"])
            self.initialized = True
    
        smpl_outputs = self.body_model(**smpl_params, use_pca=if_use_pca)


        self.smpl_outputs = smpl_outputs

        tfs = (smpl_outputs.A) @ self.tfs_inv_t.expand(smpl_outputs.A.shape[0],-1,-1,-1)

        self.tfs = tfs # self.tfs_A @ self.tfs_inv_t
        self.tfs_A = smpl_outputs.A 
        # X_posed = smpl_outputs.A @ X_template, and (self.tfs_inv_t) @ X_tposed = X_template; 
        # so X_posed = (smpl_outputs.A @ self.tfs_inv_t) @ X_tposed == equal to ==> self.tfs_A @ self.tfs_inv_t @ X_tposed
        self.shape_offset = torch.einsum('bl,mkl->bmk', [smpl_outputs.betas, self.init_spdir]) #  betas-torch.Size([1, 20]) ; init_spdir-([25254, 3, 20])
        self.pose_offset = torch.matmul(smpl_outputs.pose_feature, self.init_podir).reshape(self.shape_offset.shape) # batch_size, ([1, 25254, 3])

    def __call__(self, pts_in, rot_in, mask=None, cano=True, offset_gs=None, if_rotate_gaussian=False):
        '''
            to calculate the skinning results
            pts_in (tensor, [bs, N, 3]): the canonical space points + offset_gs, represented a batch of clothed human
            rot_in (tensor, [bs, N, 3]): the canonical space gaussians points' rotation
            mask (tensor, [bs, N]): the mask of the vertices (face, hands), 1 for the vertices that use the skinning weights from template directly
            cono (bool): if True, return the input pts directly
            offset_gs (tensor, [bs, N, 3]): the estimated offset of the vertices in the canonical space

            use some of the attributes from the "prepare_deformer" to calculate the skinning, including:
            pose_offset[bs_pose, N, 3]
            shape_offset[bs_pose, N, 3]

        '''
        pts = pts_in.clone()
        rot = rot_in.clone()

        if cano:
            return pts, None
        else:
            init_faces = self.init_faces
           
        b, n, _ = pts.shape

        smpl_nn = False

        if smpl_nn:
            # deformer based on SMPL nearest neighbor search
            
            k = 1
            dist_sq, idx, neighbors = ops.knn_points(pts, self.smpl_outputs.vertices.float().expand(b, -1, -1), K=k, return_nn=True)

            
            dist = dist_sq.sqrt().clamp_(0.00003, 0.1)
            weights = self.body_model.lbs_weights.clone()[idx]


            ws=1./dist
            ws=ws/ws.sum(-1,keepdim=True)
            weights = (ws[..., None]*weights).sum(2).detach()

            shape_offset = torch.cat([self.shape_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            pts += shape_offset
            pts_cano_all, w_tf = skinning(pts, weights, self.tfs, inverse=False)
            pts_cano_all = pts_cano_all.unsqueeze(2)
            
        else:
            # defromer based on fast-SNARF
            shape_offset = torch.cat([self.shape_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            pose_offset = torch.cat([self.pose_offset[:, init_faces[..., i]] for i in range(3)], dim=1).mean(1)
            
            pts_query_lbs = pts.detach() # T_pose + gs_offset

            pts_cano_all, w_tf = self.deformer.forward_skinning(pts, shape_offset, pose_offset, cond=None, tfs=self.tfs_A, tfs_inv=self.tfs_inv_t, \
                                                                poseoff_ori=self.pose_offset_cano, lbsw=self.init_lbsw, mask=mask)

        pts_cano_all = pts_cano_all.reshape(b, n, -1, 3)

        if if_rotate_gaussian:
            # rotate the gaussian points
            # pts_cano_all =  rot
            # rot_mats = quaternion_to_matrix(rot)
            # rot_mats = torch.einsum('nxy,nyz->nxz', w_tf[..., :3, :3], rot_mats)
            # rot_res = matrix_to_quaternion(rot_mats)
            # return pts_cano_all, w_tf.clone(), rot_res
            raise NotImplementedError("Code is not correct!")
           
        
        assert pts_in.dim() != 2
        
        return pts_cano_all, w_tf.clone()
