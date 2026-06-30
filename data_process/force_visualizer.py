import numpy as np
import cv2
import math

class ForceVisualizer:
    """
    Visualizes a 3D force vector on a 2D image by projecting it from the robot's end-effector.
    """

    def __init__(self, camera_matrix, dist_coeffs, T_base2cam, force_scale=0.01):
        """
        Initializes the ForceVisualizer.

        :param camera_matrix: 3x3 camera intrinsic matrix.
        :param dist_coeffs: Camera distortion coefficients.
        :param T_base2cam: 4x4 transformation matrix from robot base to camera frame.
        :param force_scale: Scaling factor for visualizing the force vector.
        """
        self.camera_matrix = np.array(camera_matrix)
        self.dist_coeffs = np.array(dist_coeffs)
        self.T_base2cam = np.array(T_base2cam)
        self.force_scale = force_scale

        # Standard DH parameters for Doosan M1013 robot (in meters)
        self.d = np.array([0.2125, 0, 0, 0.579, 0.1215, 0.110])
        self.a = np.array([0, 0.550, 0.150, 0, 0, 0])
        self.alpha = np.array([-math.pi/2, 0, math.pi/2, -math.pi/2, math.pi/2, 0])
        self.theta_offsets = np.array([0, -math.pi/2, 0, 0, 0, 0])

    def _dh_transformation_matrix(self, theta, d, a, alpha):
        """
        Computes the transformation matrix for a single joint using DH parameters.
        """
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0,             np.sin(alpha),                 np.cos(alpha),                 d],
            [0,             0,                             0,                             1]
        ])

    def forward_kinematics(self, joint_angles):
        """
        Calculates the forward kinematics for the robot to find the end-effector pose.

        :param joint_angles: A list or array of 6 joint angles in radians.
        :return: A 4x4 transformation matrix from the robot base to the end-effector.
        """
        if len(joint_angles) != 6:
            raise ValueError("Expected 6 joint angles.")

        T = np.eye(4)
        thetas = np.array(joint_angles) + self.theta_offsets

        for i in range(6):
            T_i = self._dh_transformation_matrix(thetas[i], self.d[i], self.a[i], self.alpha[i])
            T = T @ T_i
        
        return T

    def visualize_force_on_image(self, image, joint_angles, force_vector, show_magnitude=False):
        """
        Projects the force vector onto the image and draws it as an arrow.

        :param image: The image on which to draw.
        :param joint_angles: The current joint angles of the robot arm.
        :param force_vector: The 3D force vector [fx, fy, fz] or 6D wrench [fx,fy,fz,tx,ty,tz].
        :param show_magnitude: If True, displays the force magnitude as text.
        :return: The image with the force vector visualized.
        """
        try:
            # 1. Get end-effector pose using Forward Kinematics
            T_base2ee = self.forward_kinematics(joint_angles)
            ee_pos_base = T_base2ee[:3, 3]
            
            # 2. Define force start and end points in the base frame
            force_vec_base = np.array(force_vector[:3])
            force_start_pos_base = ee_pos_base
            force_end_pos_base = ee_pos_base + force_vec_base * self.force_scale

            # 3. Transform points to homogeneous coordinates for matrix multiplication
            force_start_h = np.append(force_start_pos_base, 1)
            force_end_h = np.append(force_end_pos_base, 1)

            # 4. Transform points to the camera frame
            force_start_cam_h = self.T_base2cam @ force_start_h
            force_end_cam_h = self.T_base2cam @ force_end_h
            
            # Convert back to 3D Cartesian coordinates
            force_start_cam = force_start_cam_h[:3] / force_start_cam_h[3]
            force_end_cam = force_end_cam_h[:3] / force_end_cam_h[3]

            # 5. Project 3D points from camera frame to 2D image plane
            # We need to project relative to the camera, so rvec and tvec are zero.
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            points_3d = np.array([force_start_cam, force_end_cam], dtype=np.float32)
            points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs)
            
            if points_2d is None:
                return image

            pt1 = tuple(points_2d[0].ravel().astype(int))
            pt2 = tuple(points_2d[1].ravel().astype(int))

            # 6. Draw the arrow on the image
            cv2.arrowedLine(image, pt1, pt2, (0, 255, 0), 2, tipLength=0.3)

            # 7. Optionally, display the force magnitude
            if show_magnitude:
                magnitude = np.linalg.norm(force_vec_base)
                cv2.putText(image, f"Force: {magnitude:.2f} N", (pt1[0] + 5, pt1[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Error in force visualization: {e}")

        return image
